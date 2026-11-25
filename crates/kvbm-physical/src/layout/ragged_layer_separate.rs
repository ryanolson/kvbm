// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Layer-separated blocks whose per-layer byte widths are not uniform.

use anyhow::{Result, anyhow, bail};
use validator::Validate;

use super::serialize::{LayoutTypeDetails, RaggedLayerSeparateDetails};
use super::view::LayoutView;
use super::{Buffer, KvBlockLayout, Layout, LayoutConfig, MemoryRegion};

/// The reserved bytes for one pool page: round the exact per-block bytes up
/// to `alignment` before multiplying by the live page count.
fn aligned_pool_bytes(bytes: usize, alignment: usize) -> usize {
    let alignment = alignment.max(1);
    bytes.div_ceil(alignment).saturating_mul(alignment)
}

/// A logical block assembled from independently sized contiguous segments.
#[derive(Debug)]
pub struct RaggedLayerSeparateLayout {
    config: LayoutConfig,
    bytes_per_layer_block: Vec<usize>,
    bytes_per_block: usize,
    memory_regions: Vec<Buffer>,
    reserved_allocations: Vec<usize>,
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

        let reserved_allocations = bytes_per_layer_block
            .iter()
            .map(|bytes| {
                config
                    .num_blocks
                    .saturating_mul(aligned_pool_bytes(*bytes, config.alignment))
            })
            .collect();

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
            reserved_allocations,
        })
    }

    /// Reconstructed descriptors report the actual regions they carry, not the
    /// local aligned accounting reservation.
    pub(crate) fn with_reserved_allocations(mut self, reserved: Vec<usize>) -> Self {
        debug_assert_eq!(reserved.len(), self.bytes_per_layer_block.len());
        self.reserved_allocations = reserved;
        self
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

    /// The reserved accounting footprint. This is larger than the exact
    /// region sizes when a pool's per-block bytes are not aligned. The builder
    /// still accepts the exact supplied sizes; only this reservation rounds up.
    fn required_allocations(&self) -> Vec<usize> {
        self.reserved_allocations.clone()
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
    use crate::layout::builder::PhysicalLayoutBuilderDefault;
    use crate::layout::tests::MockMemory;
    use crate::layout::{
        LayoutDescriptor, LayoutTypeDetails, NixlMetadata, PhysicalLayout,
        RaggedLayerSeparateDetails,
    };
    use kvbm_memory::{MemoryRegion, StorageKind, SystemStorage, nixl::MemType};

    const BLOCKS: usize = 3;
    const ALIGNMENT: usize = 64;
    const POOLS: [usize; 2] = [16, 24];

    fn two_pool_config() -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(BLOCKS)
            .num_layers(POOLS.len())
            .outer_dim(1)
            .page_size(8)
            .inner_dim(1)
            .alignment(ALIGNMENT)
            .dtype_width_bytes(1)
            .build()
            .expect("two-pool layout config")
    }

    fn two_pool_layout() -> RaggedLayerSeparateLayout {
        let config = two_pool_config();
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

    #[test]
    fn base_capacity_builder_accepts_exact_regions_and_accounts_aligned() {
        let regions = vec![
            SystemStorage::new(BLOCKS * POOLS[0]).expect("pool 0 region"),
            SystemStorage::new(BLOCKS * POOLS[1]).expect("pool 1 region"),
        ];
        let layout = PhysicalLayoutBuilderDefault::new_local("base-capacity-builder")
            .with_config(two_pool_config())
            .ragged_layer_separate(POOLS.to_vec())
            .with_memory_regions(regions)
            .expect("exact supplied regions are accepted")
            .build()
            .expect("two-pool ragged layout through the builder");

        assert_eq!(
            layout.layout().bytes_per_block(),
            POOLS.iter().sum::<usize>(),
            "the exact block bytes include both backing pools"
        );
        let region_sizes = layout
            .layout()
            .memory_regions()
            .iter()
            .map(|region| region.size())
            .collect::<Vec<_>>();
        assert_eq!(
            region_sizes,
            vec![BLOCKS * POOLS[0], BLOCKS * POOLS[1]],
            "the builder accepts the exact supplied regions"
        );
        let reserved = layout.layout().required_allocations();
        assert_eq!(
            reserved,
            vec![
                align_up(POOLS[0], ALIGNMENT) * BLOCKS,
                align_up(POOLS[1], ALIGNMENT) * BLOCKS,
            ],
            "the accounting uses the aligned reservation"
        );
        let reserved: usize = reserved.iter().sum();
        let exact: usize = region_sizes.iter().sum();
        assert_eq!(reserved, 384);
        assert_eq!(exact, 120);
        assert_eq!(
            reserved - exact,
            264,
            "264 of 384 reserved bytes are unused because the stride is exact"
        );
    }

    #[test]
    fn base_capacity_descriptor_reports_exact_regions() {
        let descriptor = LayoutDescriptor {
            version: LayoutDescriptor::CURRENT_VERSION,
            layout_config: two_pool_config(),
            location: StorageKind::System,
            nixl_metadata: NixlMetadata::new("base-capacity-remote".to_string(), MemType::Dram, 0),
            memory_descriptors: vec![
                MemoryRegion {
                    addr: 0x1000,
                    size: BLOCKS * POOLS[0],
                },
                MemoryRegion {
                    addr: 0x2000,
                    size: BLOCKS * POOLS[1],
                },
            ],
            layout_type_details: LayoutTypeDetails::RaggedLayerSeparate(
                RaggedLayerSeparateDetails {
                    bytes_per_layer_block: POOLS.to_vec(),
                    kv_block_layout: KvBlockLayout::Unknown,
                },
            ),
        };

        let layout = PhysicalLayout::from_descriptor(descriptor).expect("reconstruct descriptor");

        assert_eq!(
            layout.layout().required_allocations(),
            vec![BLOCKS * POOLS[0], BLOCKS * POOLS[1]],
            "a descriptor reports the actual supplied regions"
        );
        assert_eq!(
            layout
                .layout()
                .memory_regions()
                .iter()
                .map(|region| region.size())
                .collect::<Vec<_>>(),
            vec![BLOCKS * POOLS[0], BLOCKS * POOLS[1]]
        );
    }
}
