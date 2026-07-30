// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stress tests for transfers containing many independently completed regions.

use super::gate::gpu_serial;
use super::local_transfers::build_agent_for_kinds;
use super::*;
use crate::transfer::executor::{TransferOptionsInternal, execute_transfer};
use anyhow::Result;

const REGION_COUNT: usize = 232;

fn many_heterogeneous_region_sizes() -> Vec<usize> {
    const SIZE_CLASSES: [usize; 8] = [
        16 * 1024,
        32 * 1024,
        64 * 1024,
        96 * 1024,
        128 * 1024,
        192 * 1024,
        256 * 1024,
        384 * 1024,
    ];
    (0..REGION_COUNT)
        .map(|region| SIZE_CLASSES[(region * 5) % SIZE_CLASSES.len()])
        .collect()
}

fn build_many_region_layout(
    agent: NixlAgent,
    storage_kind: StorageKind,
    num_blocks: usize,
) -> PhysicalLayout {
    let region_sizes = many_heterogeneous_region_sizes();
    let config = LayoutConfig::builder()
        .num_blocks(num_blocks)
        .num_layers(region_sizes.len())
        .outer_dim(1)
        .page_size(1)
        .inner_dim(1)
        .dtype_width_bytes(1)
        .num_heads(Some(1))
        .build()
        .unwrap();
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .ragged_layer_separate(region_sizes);
    match storage_kind {
        StorageKind::System => builder.allocate_system().build().unwrap(),
        StorageKind::Pinned => builder.allocate_pinned(Some(0)).build().unwrap(),
        StorageKind::Device(device_id) => builder.allocate_device(device_id).build().unwrap(),
        StorageKind::Disk(_) => builder.allocate_disk(None).build().unwrap(),
    }
}

#[tokio::test]
async fn many_heterogeneous_ragged_regions_survive_repeated_round_trips() -> Result<()> {
    skip_if_stubs_and_device!(StorageKind::Pinned, StorageKind::Device(0));
    if kvbm_memory::nixl::is_stub() {
        eprintln!("skipping ragged transfer stress test: NIXL is in stub mode");
        return Ok(());
    }
    gpu_serial!();
    let agent = build_agent_for_kinds(&[StorageKind::Pinned, StorageKind::Device(0)])?;
    let seed = build_many_region_layout(agent.clone(), StorageKind::Pinned, 1);
    let device = build_many_region_layout(agent.clone(), StorageKind::Device(0), 2);
    let snapshots = build_many_region_layout(agent.clone(), StorageKind::Pinned, 4);
    let restored = build_many_region_layout(agent.clone(), StorageKind::Pinned, 1);
    let checksums = fill_and_checksum(&seed, &[0], FillPattern::Sequential)?;
    let ctx = create_transfer_context(agent, None)?;

    execute_transfer(
        &seed,
        &device,
        &[0],
        &[0],
        TransferOptionsInternal::default(),
        ctx.context(),
    )?
    .await?;

    for snapshot in 0..4 {
        execute_transfer(
            &device,
            &snapshots,
            &[0],
            &[snapshot],
            TransferOptionsInternal::default(),
            ctx.context(),
        )?
        .await?;
    }

    execute_transfer(
        &snapshots,
        &device,
        &[2],
        &[1],
        TransferOptionsInternal::default(),
        ctx.context(),
    )?
    .await?;
    execute_transfer(
        &device,
        &restored,
        &[1],
        &[0],
        TransferOptionsInternal::default(),
        ctx.context(),
    )?
    .await?;

    verify_checksums_by_position(&checksums, &[0], &restored, &[0])?;
    Ok(())
}
