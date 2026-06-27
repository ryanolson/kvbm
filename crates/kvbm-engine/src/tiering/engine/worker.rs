// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side connector engine: the per-TP-rank GPU forward-pass runtime.
//!
//! [`WorkerEngine`] is the worker-process counterpart of the leader's
//! `LocalConnectorEngine` (REFACTOR.md §7): it wraps an already-built
//! [`DirectWorker`] plus the pre-allocated per-layer CUDA events, and owns the
//! per-iteration **pass state machine** the connector worker's vLLM hooks
//! drive. The connector binds an engine-typed [`WorkerPassPlan`] each step
//! (translated from its own wire metadata — no connector types cross into this
//! crate), then forwards the raw `(layer_index, stream_handle)` hook calls.
//!
//! The faithful-port source is the legacy connector worker
//! (`kvbm-connector/src/connector/worker/{mod,state}.rs`): intra-pass G2→G1
//! layer-wise onboarding with `cuStreamWaitEvent` layer gates, intra-pass
//! G1→G2 layer-wise offload legs on a dedicated D2H stream, and the
//! last-layer forward-pass completion trigger (CUDA-event poll → velo event).
//!
//! Delegate-lifetime inversion (REFACTOR.md §7): `build` **receives** the
//! connector's completion sink — the same `Arc` backing the worker's velo
//! completion handlers and `get_finished` drain. The engine must never mint
//! its own sink.

use std::sync::Arc;

use anyhow::{Result, bail};
use cudarc::driver::sys::{
    CUevent, CUresult, CUstream, cuEventQuery, cuEventRecord, cuStreamWaitEvent, cudaError_enum,
};
use cudarc::driver::{CudaEvent, CudaStream};
use parking_lot::Mutex;
use velo::{EventHandle, Messenger};

use kvbm_common::{BlockId, LogicalLayoutHandle};
use kvbm_physical::TransferOptions;
use kvbm_protocols::connector::{EngineWorkerSink, FenceToken, WorkerEngineDriver};

use crate::worker::{DirectWorker, WorkerTransfers};

/// Per-iteration transfer plan the connector worker binds before the forward
/// pass. Engine-typed: the connector translates its wire metadata into this.
#[derive(Debug, Default)]
pub struct WorkerPassPlan {
    /// Intra-pass G2→G1 layer-wise onboard, executed at `start_load`.
    pub onboard: Option<PassOnboard>,
    /// Intra-pass G1→G2 layer-wise offload, executed leg-by-leg in
    /// `save_kv_layer`.
    pub offload: Option<PassOffload>,
    /// Leader-minted velo event to trigger once the last layer's compute event
    /// completes (the forward-pass completion notification).
    pub completion_event: Option<EventHandle>,
}

/// Intra-pass onboard legs: `g2_src_block_ids[i] → g1_dst_block_ids[i]`.
#[derive(Debug)]
pub struct PassOnboard {
    pub g2_src_block_ids: Vec<BlockId>,
    pub g1_dst_block_ids: Vec<BlockId>,
}

/// Intra-pass offload legs: `g1_src_block_ids[i] → g2_dst_block_ids[i]`.
#[derive(Debug)]
pub struct PassOffload {
    pub g1_src_block_ids: Vec<BlockId>,
    pub g2_dst_block_ids: Vec<BlockId>,
}

/// Offload armed for the current pass: the D2H stream is acquired at bind and
/// held for the whole pass (legacy `IntraPassOffloadState`).
struct ActiveOffload {
    stream: Arc<CudaStream>,
    g1_src_block_ids: Arc<[BlockId]>,
    g2_dst_block_ids: Arc<[BlockId]>,
}

/// Mutable per-pass state, reset by `clear_pass`.
#[derive(Default)]
struct PassState {
    /// Armed at `bind_pass`, consumed by `start_load`.
    pending_onboard: Option<PassOnboard>,
    /// True once `start_load` executed the onboard — `wait_for_layer_load`
    /// inserts layer gates only then.
    onboard_active: bool,
    offload: Option<ActiveOffload>,
    /// Consumed by the last layer's `save_kv_layer`.
    completion_event: Option<EventHandle>,
}

/// The per-rank worker-side engine (REFACTOR.md §7 right column). Not a
/// `LeaderEngine` — it implements [`WorkerEngineDriver`] for the forward-pass
/// boundaries and exposes the layer-granular hooks inherently.
pub struct WorkerEngine {
    worker: Arc<DirectWorker>,
    num_layers: usize,
    /// One per layer; recorded on the engine's H2D transfer stream during the
    /// intra-pass onboard, waited on by the torch stream per layer.
    onboard_layer_events: Vec<Arc<CudaEvent>>,
    /// One per layer; recorded on the torch stream in `save_kv_layer` — the
    /// moment the layer is computed and safe to offload.
    compute_layer_events: Vec<Arc<CudaEvent>>,
    /// Recorded on the offload stream after the last layer's leg; awaited
    /// synchronously in `wait_for_save`.
    offload_complete_event: Arc<CudaEvent>,
    /// The connector's completion delegate (same `Arc` as the worker's velo
    /// handlers + `get_finished` drain). Held per the §7 build contract; the
    /// engine-internal completion sources that push through it (eviction-drain
    /// fence terminals, direct layer-wise offload completions) land in P-E.
    #[expect(dead_code)]
    completion: Arc<dyn EngineWorkerSink>,
    messenger: Arc<Messenger>,
    tokio: tokio::runtime::Handle,
    pass: Mutex<PassState>,
}

impl WorkerEngine {
    /// Build the engine over an already-built [`DirectWorker`]. Pre-allocates
    /// the per-layer CUDA events on the transfer streams (faithful port of the
    /// legacy `WorkerState::initialize` event bracket).
    ///
    /// `completion` MUST be the connector worker's live sink (delegate-lifetime
    /// inversion) — the engine stores it, never creates one.
    pub fn build(
        worker: Arc<DirectWorker>,
        num_layers: usize,
        completion: Arc<dyn EngineWorkerSink>,
        messenger: Arc<Messenger>,
        tokio: tokio::runtime::Handle,
    ) -> Result<Arc<Self>> {
        let transfer_manager = worker.transfer_manager();

        // Pre-allocate onboard events (H2D stream).
        let h2d_stream = transfer_manager.context().acquire_h2d_stream();
        let mut onboard_layer_events = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            onboard_layer_events.push(Arc::new(h2d_stream.record_event(None)?));
        }

        // Pre-allocate save/offload events (D2H stream for consistency).
        let d2h_stream = transfer_manager.context().acquire_d2h_stream();
        let mut compute_layer_events = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            compute_layer_events.push(Arc::new(d2h_stream.record_event(None)?));
        }

        let offload_complete_event = Arc::new(d2h_stream.record_event(None)?);

        tracing::debug!(
            num_layers,
            "WorkerEngine built — per-layer events pre-allocated"
        );

        Ok(Arc::new(Self {
            worker,
            num_layers,
            onboard_layer_events,
            compute_layer_events,
            offload_complete_event,
            completion,
            messenger,
            tokio,
            pass: Mutex::new(PassState::default()),
        }))
    }

    /// The wrapped [`DirectWorker`] (the connector's velo transfer service
    /// wraps the same instance).
    pub fn worker(&self) -> &Arc<DirectWorker> {
        &self.worker
    }

    /// Arm the per-iteration plan. The offload's D2H stream is acquired here
    /// and held for the whole pass (legacy `bind_connector_metadata`).
    pub fn bind_pass(&self, plan: WorkerPassPlan) -> Result<()> {
        let offload = match plan.offload {
            Some(offload) => {
                let stream = self
                    .worker
                    .transfer_manager()
                    .context()
                    .acquire_d2h_stream();
                Some(ActiveOffload {
                    stream,
                    g1_src_block_ids: Arc::from(offload.g1_src_block_ids),
                    g2_dst_block_ids: Arc::from(offload.g2_dst_block_ids),
                })
            }
            None => None,
        };

        let mut pass = self.pass.lock();
        pass.pending_onboard = plan.onboard;
        pass.onboard_active = false;
        pass.offload = offload;
        pass.completion_event = plan.completion_event;
        Ok(())
    }

    /// Reset the pass state (legacy `clear_connector_metadata`).
    pub fn clear_pass(&self) {
        let mut pass = self.pass.lock();
        if pass.completion_event.take().is_some() {
            // Could happen if the pass errored or processed no layers; the
            // leader's completion await will lag, but nothing is corrupted.
            tracing::trace!(
                "forward-pass completion event not consumed — save_kv_layer may not have run on the last layer"
            );
        }
        *pass = PassState::default();
    }

    /// Execute the armed intra-pass G2→G1 layer-wise onboard (legacy
    /// `start_load_kv`). Pure-CUDA: a dedicated H2D stream with one event
    /// recorded per layer, consumed by [`Self::wait_for_layer_load`].
    pub fn start_load(&self) -> Result<()> {
        let onboard = self.pass.lock().pending_onboard.take();
        let Some(onboard) = onboard else {
            return Ok(());
        };

        tracing::debug!(
            g2_blocks = onboard.g2_src_block_ids.len(),
            g1_blocks = onboard.g1_dst_block_ids.len(),
            "Starting intra-pass layer-wise onboard from G2 to G1"
        );

        self.worker.execute_local_layerwise_onboard(
            &onboard.g2_src_block_ids,
            &onboard.g1_dst_block_ids,
            &self.onboard_layer_events,
        )?;

        self.pass.lock().onboard_active = true;
        tracing::debug!("Intra-pass onboard initiated — events recorded on transfer stream");
        Ok(())
    }

    /// Gate the torch stream on this layer's onboard event (legacy
    /// `wait_for_layer_load`). No-op unless an intra-pass onboard is active.
    pub fn wait_for_layer_load(&self, layer_index: usize, stream_handle: u64) -> Result<()> {
        if !self.pass.lock().onboard_active {
            return Ok(());
        }

        let event = self.onboard_layer_events.get(layer_index).ok_or_else(|| {
            anyhow::anyhow!(
                "layer_index {layer_index} out of range (num_layers={})",
                self.num_layers
            )
        })?;

        // Insert cuStreamWaitEvent so the torch stream waits for this layer's
        // onboard before its attention reads the KV slots.
        unsafe {
            let status = cuStreamWaitEvent(stream_handle as CUstream, event.cu_event(), 0);
            if status != cudaError_enum::CUDA_SUCCESS {
                bail!("cuStreamWaitEvent failed with status: {status:?}");
            }
        }

        tracing::trace!(layer_index, "Inserted cuStreamWaitEvent for layer onboard");
        Ok(())
    }

    /// Per-layer save hook (legacy `save_kv_layer`): record this layer's
    /// compute event on the torch stream, run the layer's intra-pass offload
    /// leg when armed, and on the last layer fire the forward-pass completion
    /// trigger when armed. Returns immediately when no action applies.
    pub fn save_kv_layer(&self, layer_index: usize, stream_handle: u64) -> Result<()> {
        let is_last_layer = layer_index == self.num_layers - 1;

        let pass = self.pass.lock();
        let offload_armed = pass.offload.is_some();
        let completion_on_last = is_last_layer && pass.completion_event.is_some();
        if !offload_armed && !completion_on_last {
            return Ok(());
        }

        let event = self
            .compute_layer_events
            .get(layer_index)
            .ok_or_else(|| anyhow::anyhow!("layer_index {layer_index} out of range"))?
            .clone();

        // Record this layer's compute event on the torch stream.
        unsafe {
            let status = cuEventRecord(event.cu_event(), stream_handle as CUstream);
            if status != cudaError_enum::CUDA_SUCCESS {
                bail!("cuEventRecord failed with status: {status:?}");
            }
        }
        tracing::trace!(layer_index, "Recorded save layer CUDA event");

        if let Some(offload) = pass.offload.as_ref() {
            // Make the offload stream wait for this layer's compute, then run
            // the layer's G1→G2 leg on it.
            unsafe {
                let status = cuStreamWaitEvent(offload.stream.cu_stream(), event.cu_event(), 0);
                if status != cudaError_enum::CUDA_SUCCESS {
                    bail!("cuStreamWaitEvent failed with status: {status:?}");
                }
            }

            let options = TransferOptions::builder()
                .layer_range(layer_index..layer_index + 1)
                .cuda_stream(offload.stream.clone())
                .build()?;

            self.worker.execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                offload.g1_src_block_ids.clone(),
                offload.g2_dst_block_ids.clone(),
                options,
            )?;

            if is_last_layer {
                self.offload_complete_event
                    .record(offload.stream.as_ref())?;
            }
        }

        // Last layer with a completion event armed: consume it and spawn the
        // CUDA-poll → velo-trigger task.
        let completion_event = if completion_on_last {
            let mut pass = pass;
            pass.completion_event.take()
        } else {
            None
        };
        if let Some(velo_event) = completion_event {
            self.trigger_forward_pass_completion(event, velo_event);
        }

        Ok(())
    }

    /// Block until the intra-pass offload's last leg completes (legacy
    /// `wait_for_save`). No-op unless an offload is armed.
    pub fn wait_for_save(&self) -> Result<()> {
        let offload_armed = self.pass.lock().offload.is_some();
        if offload_armed {
            self.offload_complete_event.synchronize()?;
        }
        Ok(())
    }

    /// Spawn the async task that polls the layer's CUDA event to completion,
    /// then triggers the leader-minted velo event (legacy
    /// `trigger_forward_pass_completion`).
    fn trigger_forward_pass_completion(&self, cuda_event: Arc<CudaEvent>, velo_event: EventHandle) {
        let messenger = self.messenger.clone();
        let cuda_event_handle = cuda_event.cu_event() as u64;

        tracing::debug!(
            ?velo_event,
            cuda_event = cuda_event_handle,
            "Spawning forward pass completion task"
        );

        self.messenger.tracker().spawn_on(
            async move {
                // Keep the event Arc alive for the poll's duration.
                let _cuda_event = cuda_event;
                loop {
                    let status = unsafe { cuEventQuery(cuda_event_handle as CUevent) };
                    match status {
                        CUresult::CUDA_SUCCESS => break,
                        CUresult::CUDA_ERROR_NOT_READY => {
                            tokio::task::yield_now().await;
                        }
                        _ => {
                            tracing::error!("CUDA event query failed: {status:?}");
                            break;
                        }
                    }
                }

                tracing::debug!(?velo_event, "CUDA event complete, triggering velo event");
                if let Err(e) = messenger.events().trigger(velo_event).await {
                    tracing::error!("Failed to trigger forward pass event: {e}");
                }
            },
            &self.tokio,
        );
    }
}

impl WorkerEngineDriver for WorkerEngine {
    fn begin_forward_pass(&self, iteration: usize) {
        if let Err(e) = self.start_load() {
            tracing::error!(error = %e, iteration, "begin_forward_pass: intra-pass onboard failed");
        }
    }

    fn finish_forward_pass(&self, iteration: usize) {
        if let Err(e) = self.wait_for_save() {
            tracing::error!(error = %e, iteration, "finish_forward_pass: offload wait failed");
        }
    }

    fn await_fence(&self, _token: FenceToken) {
        // The connector worker awaits fences on its own WorkerCompletionState;
        // the engine-side eviction-drain terminal that completes them is P-E.
    }

    fn shutdown(&self) {
        self.clear_pass();
    }
}
