// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Layer-separated blocks whose per-layer byte widths are not uniform.

use anyhow::{Result, anyhow, bail};
use validator::Validate;

use super::serialize::{LayoutTypeDetails, RaggedLayerSeparateDetails};
use super::view::LayoutView;
use super::{Buffer, KvBlockLayout, Layout, LayoutConfig, MemoryRegion};

/// A logical block assembled from independently sized contiguous segments.
#[derive(Debug)]
pub struct RaggedLayerSeparateLayout {
    config: LayoutConfig,
    bytes_per_layer_block: Vec<usize>,
    bytes_per_block: usize,
    memory_regions: Vec<Buffer>,
}

impl RaggedLayerSeparateLayout {
    pub(crate) fn new(
        config: LayoutConfig,
        memory_regions: Vec<Buffer>,
        bytes_per_layer_block: Vec<usize>,
        kv_block_layout: KvBlockLayout,
    ) -> Result<Self> {
        config.validate()?;
        if config.outer_dim != 1 {
            bail!(
                "RaggedLayerSeparateLayout requires outer_dim=1, got {}",
                config.outer_dim
            );
        }
        if !matches!(kv_block_layout, KvBlockLayout::Unknown) {
            bail!(
                "RaggedLayerSeparateLayout currently requires KvBlockLayout::Unknown, got {kv_block_layout:?}"
            );
        }
        if bytes_per_layer_block.len() != config.num_layers {
            bail!(
                "ragged layer byte count ({}) must match num_layers ({})",
                bytes_per_layer_block.len(),
                config.num_layers
            );
        }
        if memory_regions.len() != config.num_layers {
            bail!(
                "ragged memory region count ({}) must match num_layers ({})",
                memory_regions.len(),
                config.num_layers
            );
        }
        if let Some(layer) = bytes_per_layer_block.iter().position(|&bytes| bytes == 0) {
            bail!("ragged layer {layer} has zero bytes per block");
        }

        let mut bytes_per_block = 0usize;
        for (layer, (&segment_bytes, region)) in bytes_per_layer_block
            .iter()
            .zip(&memory_regions)
            .enumerate()
        {
            let required = config
                .num_blocks
                .checked_mul(segment_bytes)
                .ok_or_else(|| anyhow!("ragged layer {layer} allocation size overflow"))?;
            if region.size() < required {
                bail!(
                    "ragged memory region {layer} too small: requires {required} bytes, got {}",
                    region.size()
                );
            }
            bytes_per_block = bytes_per_block
                .checked_add(segment_bytes)
                .ok_or_else(|| anyhow!("ragged block byte size overflow"))?;
        }

        Ok(Self {
            config,
            bytes_per_layer_block,
            bytes_per_block,
            memory_regions,
        })
    }
}

impl Layout for RaggedLayerSeparateLayout {
    fn layout_view(&self) -> Result<LayoutView> {
        bail!("ragged layer-separated layouts do not support planner projections")
    }

    fn config(&self) -> &LayoutConfig {
        &self.config
    }

    fn bytes_per_block(&self) -> usize {
        self.bytes_per_block
    }

    fn block_region_sizes(&self) -> Vec<usize> {
        self.bytes_per_layer_block.clone()
    }

    fn memory_regions(&self) -> &[Buffer] {
        &self.memory_regions
    }

    fn memory_region(
        &self,
        block_id: usize,
        layer_id: usize,
        outer_id: usize,
    ) -> Result<MemoryRegion> {
        if block_id >= self.config.num_blocks {
            bail!(
                "Block ID {block_id} out of range (max: {})",
                self.config.num_blocks
            );
        }
        if layer_id >= self.config.num_layers {
            bail!(
                "Layer ID {layer_id} out of range (max: {})",
                self.config.num_layers
            );
        }
        if outer_id != 0 {
            bail!("Outer ID {outer_id} out of range for ragged outer_dim=1");
        }
        let segment_bytes = self.bytes_per_layer_block[layer_id];
        let offset = block_id
            .checked_mul(segment_bytes)
            .ok_or_else(|| anyhow!("ragged block address overflow"))?;
        let addr = self.memory_regions[layer_id]
            .addr()
            .checked_add(offset)
            .ok_or_else(|| anyhow!("ragged block address overflow"))?;
        Ok(MemoryRegion::new(addr, segment_bytes))
    }

    /// The reserved pool footprint. Round each pool's per-block bytes up to
    /// the configured alignment before applying the page count, so the
    /// configured pages fit even when the exact tensor stride is unaligned.
    /// The exact stride stays in `block_region_sizes` and `memory_region`.
    fn required_allocations(&self) -> Vec<usize> {
        let alignment = self.config.alignment.max(1);
        self.bytes_per_layer_block
            .iter()
            .map(|bytes| {
                let aligned_bytes = bytes.div_ceil(alignment).saturating_mul(alignment);
                self.config.num_blocks.saturating_mul(aligned_bytes)
            })
            .collect()
    }

    fn is_fully_contiguous(&self) -> bool {
        false
    }

    fn num_blocks(&self) -> usize {
        self.config.num_blocks
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn outer_dim(&self) -> usize {
        self.config.outer_dim
    }

    fn page_size(&self) -> usize {
        self.config.page_size
    }

    fn inner_dim(&self) -> usize {
        self.config.inner_dim
    }

    fn dtype_width_bytes(&self) -> usize {
        self.config.dtype_width_bytes
    }

    fn serialization_details(&self) -> LayoutTypeDetails {
        LayoutTypeDetails::RaggedLayerSeparate(RaggedLayerSeparateDetails {
            bytes_per_layer_block: self.bytes_per_layer_block.clone(),
            kv_block_layout: KvBlockLayout::Unknown,
        })
    }

    fn block_layout(&self) -> KvBlockLayout {
        KvBlockLayout::Unknown
    }
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::*;
    use crate::layout::tests::MockMemory;

    const BLOCKS: usize = 3;
    const ALIGNMENT: usize = 64;
    const POOLS: [usize; 2] = [16, 24];

    fn two_pool_layout() -> RaggedLayerSeparateLayout {
        let config = LayoutConfig::builder()
            .num_blocks(BLOCKS)
            .num_layers(POOLS.len())
            .outer_dim(1)
            .page_size(8)
            .inner_dim(1)
            .alignment(ALIGNMENT)
            .dtype_width_bytes(1)
            .build()
            .expect("two-pool layout config");
        let regions = POOLS
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                Buffer::from_arc(MockMemory::new(0x1000 + index * 0x1000, BLOCKS * bytes))
            })
            .collect();
        RaggedLayerSeparateLayout::new(config, regions, POOLS.to_vec(), KvBlockLayout::Unknown)
            .expect("two-pool ragged layout")
    }

    fn align_up(bytes: usize, alignment: usize) -> usize {
        bytes.div_ceil(alignment) * alignment
    }

    #[test]
    fn base_capacity_two_pool_footprint_includes_aligned_pools() {
        let layout = two_pool_layout();

        assert_eq!(
            layout.bytes_per_block(),
            POOLS.iter().sum::<usize>(),
            "the exact block bytes include both backing pools"
        );
        assert_eq!(
            layout.required_allocations(),
            vec![
                align_up(POOLS[0], ALIGNMENT) * BLOCKS,
                align_up(POOLS[1], ALIGNMENT) * BLOCKS,
            ],
            "the reserved footprint aligns each pool before multiplying by live pages"
        );
    }
}
