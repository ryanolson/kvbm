// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifest-resource registration over one shared transfer manager.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use kvbm_common::{LogicalLayoutHandle, LogicalResourceId};
use kvbm_engine::worker::{
    DirectWorker, LeaderLayoutConfig, ResourceTierConfig, WorkerCacheConfig, WorkerLayoutResponse,
};
use kvbm_physical::TransferManager;
use kvbm_physical::layout::PhysicalLayoutBuilder;
use kvbm_physical::manager::{ResourceLayoutHandles, TierLayoutHandles};

use crate::KvbmRuntime;

use super::{PendingLayoutMode, PendingWorkerState, pinned_allocator, select_g2_block_layout};

/// Deferred registrations for every manifest logical resource.
pub(in crate::connector::worker) struct PendingWorkerResources {
    primary: LogicalResourceId,
    resources: BTreeMap<LogicalResourceId, PendingWorkerState>,
}

impl PendingWorkerResources {
    pub(in crate::connector::worker) fn new(primary: LogicalResourceId) -> Self {
        Self {
            primary,
            resources: BTreeMap::new(),
        }
    }

    pub(in crate::connector::worker) fn insert(
        &mut self,
        resource: LogicalResourceId,
        pending: PendingWorkerState,
    ) -> Result<()> {
        anyhow::ensure!(
            self.resources.insert(resource, pending).is_none(),
            "logical resource {resource:?} is already registered"
        );
        Ok(())
    }

    pub(in crate::connector::worker) fn primary(&self) -> LogicalResourceId {
        self.primary
    }

    pub(in crate::connector::worker) fn config(&self) -> Result<WorkerCacheConfig> {
        let config = WorkerCacheConfig {
            manifest: None,
            primary: self.primary,
            resources: self
                .resources
                .iter()
                .map(|(&resource, pending)| (resource, pending.layout_config.clone()))
                .collect(),
        };
        config.validate()?;
        Ok(config)
    }

    pub(in crate::connector::worker) fn num_layers(&self) -> usize {
        self.resources
            .values()
            .map(|pending| pending.layout_config.num_layers)
            .sum()
    }

    pub(in crate::connector::worker) fn complete_initialization(
        mut self,
        runtime: &KvbmRuntime,
        mut config: LeaderLayoutConfig,
    ) -> Result<(Arc<DirectWorker>, WorkerLayoutResponse)> {
        let primary_state = self.resources.remove(&self.primary).ok_or_else(|| {
            anyhow::anyhow!("primary resource {:?} is not registered", self.primary)
        })?;
        if let Some(primary_tier) = config.resource_tiers.get(&self.primary).copied() {
            config.host_block_count = primary_tier.host_block_count;
            config.disk_block_count = primary_tier.disk_block_count;
        }
        let (primary_worker, primary_response) =
            primary_state.complete_initialization(runtime, config.clone())?;
        if self.resources.is_empty() {
            return Ok((primary_worker, primary_response));
        }

        let manager = primary_worker.transfer_manager().clone();
        let mut handles = vec![(
            self.primary,
            TierLayoutHandles::new(
                primary_worker.g1_handle(),
                primary_worker.g2_handle(),
                primary_worker.g3_handle(),
            ),
        )];
        let mut created_layouts = primary_response.created_layouts;
        for (resource, pending) in self.resources {
            let tier = config
                .resource_tiers
                .get(&resource)
                .copied()
                .ok_or_else(|| {
                    anyhow::anyhow!("leader provided no tier capacity for resource {resource:?}")
                })?;
            let (resource_handles, resource_layouts) =
                pending.register_additional_layouts(runtime, &manager, tier, resource)?;
            handles.push((resource, resource_handles));
            for layout in resource_layouts {
                if !created_layouts.contains(&layout) {
                    created_layouts.push(layout);
                }
            }
        }

        let mut builder = DirectWorker::builder()
            .manager(manager)
            .resource_handles(ResourceLayoutHandles::new(self.primary, handles)?)
            .rank(config.rank);
        if let Some(client) = primary_worker.object_client() {
            builder = builder.object_client(Arc::clone(client));
        }
        let worker = Arc::new(builder.build()?);
        let response = WorkerLayoutResponse {
            metadata: worker.export_metadata()?,
            created_layouts,
        };
        Ok((worker, response))
    }
}

impl PendingWorkerState {
    fn register_additional_layouts(
        self,
        runtime: &KvbmRuntime,
        manager: &TransferManager,
        tier: ResourceTierConfig,
        resource: LogicalResourceId,
    ) -> Result<(TierLayoutHandles, Vec<LogicalLayoutHandle>)> {
        let nixl_agent = runtime
            .nixl_agent()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("NIXL agent not found"))?;
        let g1_layout = match self.mode {
            PendingLayoutMode::LayerSeparate {
                block_dim,
                block_layout,
            } => PhysicalLayoutBuilder::new(nixl_agent.clone())
                .with_config(self.layout_config.clone())
                .layer_separate(block_dim)
                .with_block_layout(block_layout)
                .with_external_device_regions(self.tensors)?
                .build()?,
            PendingLayoutMode::FullyContiguous { block_layout } => {
                PhysicalLayoutBuilder::new(nixl_agent.clone())
                    .with_config(self.layout_config.clone())
                    .fully_contiguous()
                    .with_block_layout(block_layout)
                    .with_external_device_regions(self.tensors)?
                    .build()?
            }
        };
        let g1 = manager.register_layout(g1_layout)?;
        let mut created = vec![LogicalLayoutHandle::G1];
        let lower_layout =
            select_g2_block_layout(self.mode.block_layout(), runtime.config().block_layout);
        let bypass_host = runtime.config().cache.bypass_host_cache();
        let g2 = if bypass_host {
            None
        } else {
            let mut config = self.layout_config.clone();
            config.num_blocks = tier.host_block_count;
            let allocator = runtime.config().cache.host.allocator;
            tracing::info!(
                ?resource,
                ?allocator,
                "Allocating pinned host memory for G2 layout"
            );
            let layout = PhysicalLayoutBuilder::new(nixl_agent.clone())
                .with_config(config)
                .fully_contiguous()
                .with_block_layout(lower_layout)
                .allocate_pinned_with(
                    Some(self.cuda_device_id as u32),
                    pinned_allocator(allocator),
                )
                .build()?;
            created.push(LogicalLayoutHandle::G2);
            Some(manager.register_layout(layout)?)
        };
        let g3 = match tier.disk_block_count {
            Some(blocks) => {
                let mut config = self.layout_config;
                config.num_blocks = blocks;
                let path = PathBuf::from(format!(
                    "/tmp/kvbm_g3_{}_{}.bin",
                    runtime.messenger().instance_id(),
                    resource.0
                ));
                crate::connector::disk_cleanup::register(path.clone());
                let layout = PhysicalLayoutBuilder::new(nixl_agent)
                    .with_config(config)
                    .fully_contiguous()
                    .with_block_layout(lower_layout)
                    .allocate_disk(Some(path.clone()))
                    .build()?;
                let handle = manager.register_layout(layout)?;
                created.push(LogicalLayoutHandle::G3);
                match std::fs::remove_file(&path) {
                    Ok(()) => crate::connector::disk_cleanup::deregister(&path),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => tracing::warn!(
                        path = %path.display(),
                        %error,
                        "failed to unlink resource G3 cache file"
                    ),
                }
                Some(handle)
            }
            None => None,
        };
        Ok((TierLayoutHandles::new(Some(g1), g2, g3), created))
    }
}
