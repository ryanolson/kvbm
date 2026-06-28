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

use kvbm_engine::WorkerEngine;
#[cfg(feature = "nccl")]
use kvbm_engine::collectives::{NcclBootstrap, NcclCollectives};
#[cfg(feature = "nccl")]
use kvbm_engine::worker::ReplicatedDataWorker;
use kvbm_engine::worker::{
    CollectiveBootstrap, LeaderLayoutConfig, VeloWorkerService, WorkerLayoutResponse,
    WorkerTransfers,
};
use kvbm_physical::layout::LayoutConfig;
use kvbm_protocols::connector::EngineWorkerSink;

use crate::KvbmRuntime;

use super::init::PendingWorkerState;

/// Deferred-init slots for the connector worker's GPU runtime. `OnceLock` for the
/// one-shot transitions, `Mutex` for the pending hand-off — shareable as
/// `Arc<GpuState>` into the velo initialize handler without an outer lock.
#[derive(Default)]
pub(crate) struct GpuState {
    /// Set at KV-cache registration (the per-layer tensor count).
    num_layers: OnceLock<usize>,
    /// Set at KV-cache registration; served to the leader by the
    /// `get_layout_config` handler.
    layout_config: OnceLock<LayoutConfig>,
    /// Cached registration state, consumed by [`Self::initialize`].
    pending: Mutex<Option<PendingWorkerState>>,
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

        tracing::info!(
            cuda_device = pending.cuda_device_id,
            host_block_count = config.host_block_count,
            disk_block_count = ?config.disk_block_count,
            "Completing deferred NIXL initialization (connector)"
        );

        let parallelism = config.parallelism;
        let worker_count = config.worker_count;
        let collective_bootstrap = config.collective.clone();

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

        let transfers = build_replicated_transfers(
            runtime,
            worker.clone(),
            parallelism,
            worker_count,
            collective_bootstrap,
        )?;

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
        self.layout_config
            .set(pending.layout_config.clone())
            .map_err(|_| anyhow::anyhow!("layout config already set"))?;
        *self.pending.lock() = Some(pending);
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
        let guard = self.pending.lock();
        match guard.as_ref() {
            Some(pending) => Ok(serde_json::to_vec(&pending.layout_config)?),
            None => bail!("No pending state - call register_kv_caches first"),
        }
    }

    /// The registered layout config (served to the leader's
    /// `get_layout_config`).
    pub(crate) fn layout_config(&self) -> Result<LayoutConfig> {
        Ok(self
            .layout_config
            .get()
            .ok_or_else(|| anyhow::anyhow!("layout config not set"))?
            .clone())
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
