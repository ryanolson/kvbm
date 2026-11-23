// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{any::Any, sync::Arc};

use anyhow::{Context, Result};
use kvbm_common::{LogicalLayoutHandle, LogicalResourceId};
use kvbm_config::KvbmConfig;
use kvbm_engine::worker::{DirectWorker, LeaderLayoutConfig, ResourceTierConfig};
use kvbm_memory::{
    DeviceStorage, MemPolicyMode, MemoryDescriptor, StorageKind, TensorDescriptor,
    nixl::{NixlAgent, NixlDescriptor},
    numa::get_device_numa_node,
    page_placement,
};
use kvbm_physical::{layout::LayoutConfig, manager::LayoutHandle};

use super::{PendingLayoutMode, PendingWorkerResources, PendingWorkerState};
use crate::KvbmRuntime;

const DEVICE_ID: u32 = 0;
const HOST_BLOCK_COUNT: usize = 4;

#[derive(Debug)]
struct DeviceTensor(DeviceStorage);

impl MemoryDescriptor for DeviceTensor {
    fn addr(&self) -> usize {
        self.0.addr()
    }

    fn size(&self) -> usize {
        self.0.size()
    }

    fn storage_kind(&self) -> StorageKind {
        self.0.storage_kind()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        self.0.nixl_descriptor()
    }
}

impl DeviceTensor {
    const SHAPE: [usize; 5] = [4, 1, 1, 16, 8];
    const STRIDE: [usize; 5] = [128, 128, 128, 8, 1];
    const ELEMENT_SIZE: usize = 2;
}

impl TensorDescriptor for DeviceTensor {
    fn shape(&self) -> &[usize] {
        &Self::SHAPE
    }

    fn stride(&self) -> &[usize] {
        &Self::STRIDE
    }

    fn element_size(&self) -> usize {
        Self::ELEMENT_SIZE
    }
}

fn pending_worker() -> Result<PendingWorkerState> {
    let [blocks, layers, outer, page, inner] = DeviceTensor::SHAPE;
    let layout = LayoutConfig::builder()
        .num_blocks(blocks)
        .num_layers(layers)
        .outer_dim(outer)
        .page_size(page)
        .inner_dim(inner)
        .dtype_width_bytes(DeviceTensor::ELEMENT_SIZE)
        .build()?;
    let tensor = DeviceTensor(DeviceStorage::new(layout.required_bytes(), DEVICE_ID)?);
    PendingWorkerState::builder()
        .tensors(vec![Arc::new(tensor)])
        .num_device_blocks(blocks)
        .dtype_width_bytes(DeviceTensor::ELEMENT_SIZE)
        .layout_config(layout)
        .mode(PendingLayoutMode::FullyContiguous {
            block_layout: kvbm_common::KvBlockLayout::OperationalNHD,
        })
        .build()
}

async fn worker_runtime(registered_mmap: bool) -> Result<KvbmRuntime> {
    let mut config = KvbmConfig::default();
    let mut host = serde_json::json!({"num_blocks": HOST_BLOCK_COUNT});
    if registered_mmap {
        host["allocator"] = serde_json::json!("registered_mmap");
    }
    config.cache.host = serde_json::from_value(host)?;
    config.messenger.backend.tcp_addr = Some("127.0.0.1".into());
    config.messenger.backend.uds_enabled = false;

    let agent =
        NixlAgent::with_backends(&format!("g2-allocator-{}", uuid::Uuid::new_v4()), &["UCX"])?;
    KvbmRuntime::builder(config)
        .with_nixl_agent(agent)
        .with_runtime_handle(tokio::runtime::Handle::current())
        .build_worker()
        .await
}

fn leader_config() -> LeaderLayoutConfig {
    LeaderLayoutConfig {
        rank: 0,
        worker_count: 1,
        host_block_count: HOST_BLOCK_COUNT,
        disk_block_count: None,
        resource_tiers: Default::default(),
        resource_parallelism: Default::default(),
        object: None,
        parallelism: Default::default(),
        collective: None,
    }
}

fn check_g2_policy(
    worker: &DirectWorker,
    handle: LayoutHandle,
    registered_mmap: bool,
) -> Result<()> {
    let registry = worker.transfer_manager().registry().read().unwrap();
    let layout = registry
        .get_layout(handle)
        .context("G2 layout was not registered")?;
    let [region] = layout.layout().memory_regions() else {
        anyhow::bail!("expected one G2 region");
    };
    assert!(
        region.nixl_descriptor().is_some(),
        "G2 region is missing a NIXL descriptor"
    );
    let placement = page_placement(region.addr()).context("query G2 region's kernel policy")?;

    // The live policy distinguishes the allocators. CUDA can rewrite it
    // during registration, so this cannot verify the earlier bind.
    if registered_mmap {
        // Allow mbind and leave DYN_MEMORY_DISABLE_NUMA unset. A default policy fails by design.
        let node = get_device_numa_node(DEVICE_ID).context("test GPU has no known NUMA node")?;
        assert!(
            matches!(
                placement.mode,
                MemPolicyMode::Bind | MemPolicyMode::PreferredMany
            ),
            "expected a node policy on the registered mmap region, got {placement:?}"
        );
        assert_eq!(placement.nodes, [node]);
        assert_eq!(placement.node, node);
    } else {
        assert_eq!(placement.mode, MemPolicyMode::Default);
        assert!(placement.nodes.is_empty(), "{placement:?}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn primary_g2_follows_the_allocator_setting() -> Result<()> {
    for registered_mmap in [false, true] {
        let runtime = worker_runtime(registered_mmap).await?;
        let (worker, _) = pending_worker()?.complete_initialization(&runtime, leader_config())?;
        let handle = worker.g2_handle().context("primary G2 handle missing")?;
        check_g2_policy(&worker, handle, registered_mmap)?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial_test::serial]
async fn resource_g2_follows_the_allocator_setting() -> Result<()> {
    let primary = LogicalResourceId(0);
    let additional = LogicalResourceId(1);
    for registered_mmap in [false, true] {
        let runtime = worker_runtime(registered_mmap).await?;
        let mut resources = PendingWorkerResources::new(primary);
        resources.insert(primary, pending_worker()?)?;
        resources.insert(additional, pending_worker()?)?;
        let mut config = leader_config();
        config.resource_tiers.insert(
            additional,
            ResourceTierConfig {
                host_block_count: HOST_BLOCK_COUNT,
                disk_block_count: None,
            },
        );

        let (worker, _) = resources.complete_initialization(&runtime, config)?;
        let handle = worker
            .resource_handles()
            .and_then(|handles| handles.handle(additional, LogicalLayoutHandle::G2))
            .context("additional resource G2 handle missing")?;
        assert_ne!(Some(handle), worker.g2_handle());
        check_g2_policy(&worker, handle, registered_mmap)?;
    }
    Ok(())
}
