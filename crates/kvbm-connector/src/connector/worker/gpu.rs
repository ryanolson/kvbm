// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side GPU runtime slots + the leader-driven initialize path.
//!
//! [`GpuState`] is the connector analogue of the legacy `WorkerState` minus the
//! pieces that moved: the per-layer CUDA events and the forward-pass state
//! machine live in [`kvbm_engine::WorkerEngine`] now, and completion tracking
//! lives in the worker's `WorkerCompletionState`. What remains here are the
//! deferred-init slots (REFACTOR.md §7 worker column):
//!
//! 1. `register_kv_caches` caches a [`PendingWorkerState`] (no NIXL yet).
//! 2. The leader's `initialize` RPC drives [`GpuState::initialize`]:
//!    `complete_initialization` builds the NIXL layouts + `DirectWorker`,
//!    then `WorkerEngine::build` receives the connector's completion sink
//!    (delegate-lifetime inversion) and the engine slot is set — alongside the
//!    `VeloWorkerService` that serves the leader-driven inter-pass transfers
//!    over the same `DirectWorker`.

use std::sync::{Arc, OnceLock};

#[cfg(feature = "nccl")]
use anyhow::Context;
use anyhow::{Result, bail};
use parking_lot::Mutex;

use kvbm_common::LogicalResourceId;
use kvbm_engine::WorkerEngine;
#[cfg(feature = "nccl")]
use kvbm_engine::collectives::{NcclBootstrap, NcclCollectives};
#[cfg(feature = "nccl")]
use kvbm_engine::leader::parallelism::worker_data_placement;
use kvbm_engine::worker::{
    CollectiveBootstrap, LeaderLayoutConfig, VeloWorkerService, WorkerCacheConfig,
    WorkerLayoutResponse, WorkerTransfers,
};
#[cfg(feature = "nccl")]
use kvbm_engine::worker::{ReplicatedDataWorker, ResourceDispatchWorker};
use kvbm_protocols::connector::EngineWorkerSink;

use crate::KvbmRuntime;

use super::init::{PendingWorkerResources, PendingWorkerState};

/// Deferred-init slots for the connector worker's GPU runtime. `OnceLock` for the
/// one-shot transitions, `Mutex` for the pending hand-off — shareable as
/// `Arc<GpuState>` into the velo initialize handler without an outer lock.
#[derive(Default)]
pub(crate) struct GpuState {
    /// Manifest digest bound before resource registration.
    manifest: Mutex<Option<kvbm_protocols::cache_manifest::CacheManifestId>>,
    /// Set at KV-cache registration (the per-layer tensor count).
    num_layers: OnceLock<usize>,
    /// Set at KV-cache registration; served to the leader by the
    /// `get_layout_config` handler.
    layout_config: Mutex<Option<WorkerCacheConfig>>,
    /// Cached registration state, consumed by [`Self::initialize`].
    pending: Mutex<Option<PendingWorkerResources>>,
    /// The leader-driven transfer service over the built `DirectWorker`.
    /// `Some` == initialized (mirrors the legacy `is_initialized` probe).
    service: OnceLock<VeloWorkerService>,
    /// The worker-side GPU pass engine, built by [`Self::initialize`].
    engine: OnceLock<Arc<WorkerEngine>>,
}

impl GpuState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Complete the deferred initialization (the leader's `initialize` RPC):
    /// build `DirectWorker` from the pending registration, then the
    /// [`WorkerEngine`] (injecting `completion` — the connector's live sink)
    /// and the [`VeloWorkerService`].
    pub(crate) fn initialize(
        &self,
        runtime: &Arc<KvbmRuntime>,
        config: LeaderLayoutConfig,
        completion: Arc<dyn EngineWorkerSink>,
    ) -> Result<WorkerLayoutResponse> {
        if self.service.get().is_some() {
            bail!("Worker already initialized");
        }

        let pending =
            self.pending.lock().take().ok_or_else(|| {
                anyhow::anyhow!("No pending state - call register_kv_caches first")
            })?;
        let pending_num_layers = pending.num_layers();
        let _ = self.num_layers.set(pending_num_layers);

        tracing::info!(
            host_block_count = config.host_block_count,
            disk_block_count = ?config.disk_block_count,
            "Completing deferred NIXL initialization (connector)"
        );

        let parallelism = config.parallelism;
        let worker_count = config.worker_count;
        let collective_bootstrap = config.collective.clone();
        let resource_parallelism = config.resource_parallelism.clone();

        let (worker, response) = pending
            .complete_initialization(runtime, config)
            .map_err(|e| {
                tracing::error!(error = %e, "Worker complete_initialization failed");
                e
            })?;

        let num_layers = *self
            .num_layers
            .get()
            .ok_or_else(|| anyhow::anyhow!("Worker details not set"))?;

        let transfers = if resource_parallelism.len() > 1 {
            build_resource_transfers(
                runtime,
                worker.clone(),
                worker_count,
                collective_bootstrap,
                resource_parallelism,
            )?
        } else {
            build_replicated_transfers(
                runtime,
                worker.clone(),
                parallelism,
                worker_count,
                collective_bootstrap,
            )?
        };

        let engine = if let Some(transfers) = transfers.clone() {
            WorkerEngine::build_replicated(
                worker.clone(),
                transfers,
                num_layers,
                completion,
                runtime.messenger().clone(),
                runtime.tokio(),
            )?
        } else {
            WorkerEngine::build(
                worker.clone(),
                num_layers,
                completion,
                runtime.messenger().clone(),
                runtime.tokio(),
            )?
        };
        self.engine
            .set(engine)
            .map_err(|_| anyhow::anyhow!("worker engine already set (race condition)"))?;

        let service = if let Some(transfers) = transfers {
            VeloWorkerService::new_with_transfers(runtime.messenger().clone(), worker, transfers)?
        } else {
            VeloWorkerService::new(runtime.messenger().clone(), worker)?
        };
        self.service
            .set(service)
            .map_err(|_| anyhow::anyhow!("service already initialized (race condition)"))?;

        tracing::info!(
            created_layouts = ?response.created_layouts,
            "Deferred initialization complete - NIXL registered (connector)"
        );

        Ok(response)
    }

    /// Stash the registration state. Sets `layout_config` (one-shot) and the
    /// pending hand-off; `num_layers` is set separately by the caller.
    pub(crate) fn set_pending(&self, pending: PendingWorkerState) -> Result<()> {
        self.set_pending_resource(LogicalResourceId::default(), true, pending)
    }

    pub(crate) fn set_pending_resource(
        &self,
        resource: LogicalResourceId,
        primary: bool,
        pending: PendingWorkerState,
    ) -> Result<()> {
        let mut guard = self.pending.lock();
        if guard.is_none() {
            anyhow::ensure!(primary, "the first registered resource must be primary");
            *guard = Some(PendingWorkerResources::new(resource));
        }
        let resources = guard.as_mut().expect("initialized above");
        anyhow::ensure!(
            !primary || resources.primary() == resource,
            "primary resource is already {:?}",
            resources.primary()
        );
        resources.insert(resource, pending)?;
        let mut config = resources.config()?;
        config.manifest = *self.manifest.lock();
        *self.layout_config.lock() = Some(config);
        Ok(())
    }

    pub(crate) fn set_manifest(
        &self,
        manifest: kvbm_protocols::cache_manifest::CacheManifestId,
    ) -> Result<()> {
        let mut installed = self.manifest.lock();
        anyhow::ensure!(
            installed.is_none_or(|current| current == manifest),
            "worker cache manifest is already registered with a different digest"
        );
        *installed = Some(manifest);
        if let Some(config) = self.layout_config.lock().as_mut() {
            config.manifest = Some(manifest);
        }
        Ok(())
    }

    pub(crate) fn has_pending(&self) -> bool {
        self.pending.lock().is_some()
    }

    /// One-shot `num_layers` record (set at KV-cache registration).
    pub(crate) fn set_num_layers(&self, num_layers: usize) -> Result<()> {
        self.num_layers
            .set(num_layers)
            .map_err(|_| anyhow::anyhow!("num_layers already set"))
    }

    pub(crate) fn num_layers_set(&self) -> bool {
        self.num_layers.get().is_some()
    }

    /// The pending registration's layout config, serialized for the worker's
    /// handshake metadata.
    pub(crate) fn pending_layout_config(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(&self.layout_config()?)?)
    }

    /// The registered layout config (served to the leader's
    /// `get_layout_config`).
    pub(crate) fn layout_config(&self) -> Result<WorkerCacheConfig> {
        self.layout_config
            .lock()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("layout config not set"))
    }

    pub(crate) fn is_initialized(&self) -> bool {
        self.service.get().is_some()
    }

    /// The GPU pass engine, `Some` once [`Self::initialize`] has run.
    pub(crate) fn engine(&self) -> Option<&Arc<WorkerEngine>> {
        self.engine.get()
    }
}

fn build_replicated_transfers(
    runtime: &Arc<KvbmRuntime>,
    worker: Arc<kvbm_engine::worker::DirectWorker>,
    parallelism: kvbm_config::ParallelismMode,
    worker_count: usize,
    bootstrap: Option<CollectiveBootstrap>,
) -> Result<Option<Arc<dyn WorkerTransfers>>> {
    let collective_required =
        parallelism == kvbm_config::ParallelismMode::ReplicatedData && worker_count > 1;
    if !collective_required {
        anyhow::ensure!(
            bootstrap.is_none(),
            "collective bootstrap supplied for non-replicated or single-worker cache"
        );
        return Ok(None);
    }

    let bootstrap = bootstrap.ok_or_else(|| {
        anyhow::anyhow!(
            "replicated cache data with {worker_count} workers requires a collective bootstrap"
        )
    })?;

    #[cfg(feature = "nccl")]
    {
        let CollectiveBootstrap::Nccl { serialized } = bootstrap;
        let bootstrap = NcclBootstrap::deserialize(&serialized)
            .context("decoding KVBM NCCL collective bootstrap")?;
        anyhow::ensure!(
            bootstrap.world_size() == worker_count,
            "NCCL bootstrap world size {} does not match worker count {worker_count}",
            bootstrap.world_size()
        );
        let collective = Arc::new(NcclCollectives::from_worker_bootstrap(
            &bootstrap,
            worker.clone(),
            runtime,
        )?);
        let transfers = Arc::new(ReplicatedDataWorker::new(
            worker,
            runtime.clone(),
            collective,
        )?);
        Ok(Some(transfers))
    }

    #[cfg(not(feature = "nccl"))]
    {
        let _ = (runtime, worker, bootstrap);
        anyhow::bail!(
            "replicated cache data with {worker_count} workers requires the kvbm-connector `nccl` feature"
        )
    }
}

fn build_resource_transfers(
    runtime: &Arc<KvbmRuntime>,
    worker: Arc<kvbm_engine::worker::DirectWorker>,
    worker_count: usize,
    bootstrap: Option<CollectiveBootstrap>,
    modes: std::collections::BTreeMap<LogicalResourceId, kvbm_config::ParallelismMode>,
) -> Result<Option<Arc<dyn WorkerTransfers>>> {
    let needs_collective = worker_count > 1
        && modes
            .values()
            .any(|mode| *mode == kvbm_config::ParallelismMode::ReplicatedData);
    if !needs_collective {
        anyhow::ensure!(
            bootstrap.is_none(),
            "collective bootstrap supplied without a replicated multi-worker resource"
        );
        return Ok(None);
    }
    let bootstrap = bootstrap.ok_or_else(|| {
        anyhow::anyhow!("replicated resource data requires a collective bootstrap")
    })?;

    #[cfg(feature = "nccl")]
    {
        let CollectiveBootstrap::Nccl { serialized } = bootstrap;
        let bootstrap = NcclBootstrap::deserialize(&serialized)
            .context("decoding resource NCCL collective bootstrap")?;
        anyhow::ensure!(
            bootstrap.world_size() == worker_count,
            "NCCL bootstrap world size does not match worker count"
        );
        let collective = Arc::new(NcclCollectives::from_worker_bootstrap(
            &bootstrap,
            worker.clone(),
            runtime,
        )?);
        let placements = modes
            .into_iter()
            .map(|(resource, mode)| (resource, worker_data_placement(mode)))
            .collect();
        let transfers =
            ResourceDispatchWorker::new(worker, Arc::clone(runtime), Some(collective), placements)?;
        Ok(Some(Arc::new(transfers)))
    }

    #[cfg(not(feature = "nccl"))]
    {
        let _ = (runtime, worker, bootstrap, modes);
        anyhow::bail!("replicated resource data requires the `nccl` feature")
    }
}
