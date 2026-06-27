// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side entry point for connector.
//!
//! [`Worker`] reproduces the binding's [`ConnectorWorkerInterface`] surface:
//! the deferred KV-cache registration + leader-driven NIXL init ([`gpu`] /
//! [`init`]), the GPU forward-pass hooks (forwarded to the
//! [`kvbm_engine::WorkerEngine`] built at initialize), the live worker→engine
//! forward-pass boundaries, and the engine→worker completion + eviction-fence
//! machinery. The engine OWNS completion and PUSHES it through the injected
//! [`kvbm_protocols::connector::EngineWorkerSink`] (here [`WorkerSink`], backed
//! by the worker-owned [`WorkerCompletionState`]); the worker DRAINS the
//! finished/failed sets in `get_finished`/`get_failed_onboarding` and BLOCKS its
//! next forward pass on the eviction fence (`bind_metadata` → `await_fences`).
//!
//! The same [`WorkerCompletionState`] `Arc` backs the velo completion handlers,
//! the `get_finished` drain, AND the sink injected into the `WorkerEngine` at
//! initialize (the REFACTOR.md §7 delegate-lifetime inversion).

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, bail};
use derive_getters::Dissolve;
use parking_lot::Mutex;

use kvbm_common::{KvBlockLayout, KvDimLayout};
use kvbm_engine::{PassOffload, PassOnboard, WorkerPassPlan};
use kvbm_protocols::connector::WorkerRank;
use kvbm_protocols::connector::{EngineWorkerSink, FenceToken, WorkerEngineDriver};

use crate::TensorDescriptor;
use crate::common::KvConnectorMetadata;
use crate::vllm::layout::{determine_cross_layer_kv_layout, determine_kv_layout};

use super::metadata::{ConnectorMetadata, WireControl, WireMetadata};

mod gpu;
mod init;
mod state;
pub(crate) mod velo;

// Leader-driven worker velo control client + its wire contract (handler keys +
// message types). The legacy worker velo service speaks the same `protocol`.
pub(crate) mod client;
pub(crate) mod protocol;

use gpu::GpuState;
use init::{PendingLayoutMode, PendingWorkerState};

pub use client::ConnectorWorkerClient;
pub use state::{WorkerCompletionState, WorkerSink};

/// vLLM worker-hook protocol surface — the method set the connector's worker
/// exposes to the framework.
pub trait ConnectorWorkerInterface: Send + Sync {
    fn register_kv_caches(
        &self,
        tensors: Vec<Arc<dyn TensorDescriptor>>,
        num_device_blocks: usize,
        dtype_width_bytes: usize,
        dim_layout: KvDimLayout,
        block_layout: KvBlockLayout,
    ) -> Result<()>;

    fn register_cross_layers_kv_cache(
        &self,
        tensor: Arc<dyn TensorDescriptor>,
        num_device_blocks: usize,
        dtype_width_bytes: usize,
        dim_layout: KvDimLayout,
        block_layout: KvBlockLayout,
    ) -> Result<()>;

    fn bind_connector_metadata(&self, metadata: KvConnectorMetadata) -> Result<()>;
    fn clear_connector_metadata(&self) -> Result<()>;
    fn start_load_kv(&self) -> Result<()>;
    fn wait_for_layer_load(&self, layer_index: usize, stream_handle: u64) -> Result<()>;
    fn save_kv_layer(&self, layer_index: usize, stream_handle: u64) -> Result<()>;
    fn wait_for_save(&self) -> Result<()>;
    fn is_initialized(&self) -> bool;
    fn shutdown(&self) -> Result<()>;
    fn get_finished(&self) -> FinishedRequests;
    fn get_failed_onboarding(&self) -> HashSet<usize>;
}

/// Actions the worker takes on the leader; the only one today is teardown.
pub trait WorkerLeaderActions: Send + Sync + 'static {
    fn shutdown(&self);
}

/// Drained finished-request view. `dissolve()` yields `(offloading, onboarding)`
/// — field order is load-bearing for the binding.
#[derive(Default, Debug, Clone, Dissolve)]
pub struct FinishedRequests {
    pub offloading: HashSet<String>,
    pub onboarding: HashSet<String>,
}

/// Worker-side entry point.
pub struct Worker {
    /// Forward-pass boundaries + shutdown on the engine. `await_fence` is NOT
    /// driven through here — see [`WorkerCompletionState`]'s "`await_fence` home"
    /// note: the worker waits on its OWN completion state.
    engine_driver: Arc<dyn WorkerEngineDriver>,
    leader_actions: Arc<dyn WorkerLeaderActions>,
    /// Engine→worker completion + eviction-fence state. The worker drains/awaits
    /// it; the engine pushes into it through the [`WorkerSink`] from
    /// [`Worker::worker_sink`].
    completion: Arc<WorkerCompletionState>,
    /// This worker's TP rank — selects its [`FenceToken`] out of each
    /// `EvictionFence::per_worker`.
    rank: WorkerRank,
    metadata: Mutex<Option<ConnectorMetadata>>,
    /// Deferred-init GPU slots; the velo `initialize` handler shares this
    /// `Arc` and fills the `DirectWorker` + [`kvbm_engine::WorkerEngine`] in.
    gpu: Arc<GpuState>,
    /// `Some` for the binding constructor; the test constructors run without a
    /// runtime (no velo, no GPU paths).
    runtime: Option<Arc<crate::KvbmRuntime>>,
}

impl Worker {
    /// Binding constructor. Registers the engine→worker completion velo handlers
    /// AND the leader-driven init handlers (`initialize`/`get_layout_config`),
    /// all capturing this worker's [`WorkerCompletionState`] — the SAME `Arc`
    /// that `get_finished` drains, `bind_metadata`'s `await_fences` waits on,
    /// and the initialize handler injects into the [`WorkerEngine`] (the
    /// delegate-lifetime inversion). `engine_driver` stays a `NullEngineSink`:
    /// the GPU pass machinery is driven through the engine slot the initialize
    /// handler fills.
    pub fn new(runtime: Arc<crate::KvbmRuntime>) -> Self {
        let mut worker = Self::with_actions(
            Arc::new(super::shim::NullEngineSink),
            Arc::new(super::shim::NullLeaderSink),
        );
        worker.runtime = Some(Arc::clone(&runtime));
        velo::register_completion_handlers(runtime.messenger(), Arc::clone(&worker.completion));
        velo::register_init_handlers(
            runtime.messenger(),
            Arc::clone(&runtime),
            Arc::clone(&worker.gpu),
            worker.worker_sink(),
        );
        worker
    }

    /// Test/wiring constructor with explicit action sinks. Defaults to rank 0;
    /// use [`Worker::with_actions_and_rank`] for a non-zero TP rank.
    pub fn with_actions(
        engine_driver: Arc<dyn WorkerEngineDriver>,
        leader_actions: Arc<dyn WorkerLeaderActions>,
    ) -> Self {
        Self::with_actions_and_rank(engine_driver, leader_actions, 0)
    }

    /// Test/wiring constructor with explicit action sinks and TP rank.
    pub fn with_actions_and_rank(
        engine_driver: Arc<dyn WorkerEngineDriver>,
        leader_actions: Arc<dyn WorkerLeaderActions>,
        rank: WorkerRank,
    ) -> Self {
        Self {
            engine_driver,
            leader_actions,
            completion: Arc::new(WorkerCompletionState::new()),
            rank,
            metadata: Mutex::new(None),
            gpu: Arc::new(GpuState::new()),
            runtime: None,
        }
    }

    /// The engine→worker completion sink. The worker owns the backing
    /// [`WorkerCompletionState`]; this hands the engine its push face (as
    /// `Arc<dyn EngineWorkerSink>`) at construction (P-D1b). The engine drives
    /// it OFF the forward-pass thread.
    pub fn worker_sink(&self) -> Arc<dyn EngineWorkerSink> {
        Arc::new(WorkerSink::new(Arc::clone(&self.completion)))
    }

    /// Stash the per-step control payload, then GATE the next forward pass on the
    /// eviction fence: read this rank's [`FenceToken`]s out of `metadata.fences`
    /// and block until the engine has drained those evicted actions
    /// (REFACTOR.md §5). Runs on the model-runner (forward-pass) thread, which it
    /// records first so the off-thread completion guard is armed.
    ///
    /// The wait routes through the iteration-idempotent funnel shared with
    /// [`Self::handle_preemptions`]: whichever of the two hooks runs first for
    /// a step awaits (and takes) its fences; the other no-ops.
    pub fn bind_metadata(&self, metadata: ConnectorMetadata) {
        self.completion.mark_forward_pass_thread();
        let tokens = Self::fence_tokens_for_rank(&metadata, self.rank);
        let iteration = metadata.iteration;
        *self.metadata.lock() = Some(metadata);
        self.completion.ensure_fences_awaited(iteration, &tokens);
    }

    /// vLLM's per-step preemption hook, fed the SAME sealed envelope bytes the
    /// metadata bind receives. Parses ONLY the control payload (the transfer
    /// plan is skipped, not materialized), selects this rank's fence tokens,
    /// and routes them through the same iteration-idempotent funnel as
    /// [`Self::bind_metadata`] — so the step's fences gate the forward pass
    /// regardless of whether the runner fires this hook before or after the
    /// bind. Blocks on the forward-pass thread, which it records first so the
    /// off-thread completion guard is armed.
    pub fn handle_preemptions(&self, bytes: &[u8]) -> Result<()> {
        self.completion.mark_forward_pass_thread();
        let wire: WireControl = serde_json::from_slice(bytes)?;
        let tokens = Self::fence_tokens_for_rank(&wire.control, self.rank);
        self.completion
            .ensure_fences_awaited(wire.control.iteration, &tokens);
        Ok(())
    }

    /// Unseal and bind the per-step connector wire envelope (the binding's byte
    /// path): the transfer plan arms the GPU engine pass; the control payload
    /// gates this rank's next pass on its eviction fences. Returns whether the
    /// payload was bound (mirrors the legacy binding's `should_bind` contract;
    /// a control payload carrying fences or evictions always binds — skipping
    /// it would skip the G1-reuse gate).
    pub fn bind_serialized_metadata(&self, bytes: &[u8]) -> Result<bool> {
        let wire: WireMetadata = serde_json::from_slice(bytes)?;
        let must_bind = wire.plan.should_bind()
            || !wire.control.fences.is_empty()
            || !wire.control.evicted_requests.is_empty();
        if !must_bind {
            return Ok(false);
        }
        self.bind_plan(&wire.plan)?;
        self.bind_metadata(wire.control);
        Ok(true)
    }

    /// Translate the wire transfer plan into the engine-typed pass plan and arm
    /// the GPU engine (no-op until the leader's `initialize` fills the engine
    /// slot).
    fn bind_plan(&self, plan: &KvConnectorMetadata) -> Result<()> {
        let Some(engine) = self.gpu.engine() else {
            return Ok(());
        };
        let completion_event = match (&plan.foward_pass_completion_events, &self.runtime) {
            (Some(event_map), Some(runtime)) => {
                let my_instance_id = runtime.messenger().instance_id();
                event_map.get(&my_instance_id).copied()
            }
            _ => None,
        };
        let pass_plan = WorkerPassPlan {
            onboard: plan.intra_pass_load.as_ref().map(|load| PassOnboard {
                g2_src_block_ids: load.g2_src_block_ids.clone(),
                g1_dst_block_ids: load.g1_dst_block_ids.clone(),
            }),
            offload: plan.intra_pass_store.as_ref().map(|store| PassOffload {
                g1_src_block_ids: store.g1_src_block_ids.clone(),
                g2_dst_block_ids: store.g2_dst_block_ids.clone(),
            }),
            completion_event,
        };
        engine.bind_pass(pass_plan)
    }

    /// This rank's fence tokens across every [`EvictionFence`] in `metadata`.
    fn fence_tokens_for_rank(metadata: &ConnectorMetadata, rank: WorkerRank) -> Vec<FenceToken> {
        metadata
            .fences
            .iter()
            .flat_map(|fence| fence.per_worker.iter().copied())
            .filter(|token| token.rank == rank)
            .collect()
    }

    /// Handshake metadata for the leader: the pending registration's layout
    /// config (JSON), matching what `get_layout_config` later serves.
    pub fn handshake_metadata(&self) -> Result<Vec<u8>> {
        self.gpu.pending_layout_config()
    }

    fn iteration(&self) -> usize {
        self.metadata
            .lock()
            .as_ref()
            .map_or(0, |m| m.iteration as usize)
    }
}

impl ConnectorWorkerInterface for Worker {
    fn register_kv_caches(
        &self,
        tensors: Vec<Arc<dyn TensorDescriptor>>,
        num_device_blocks: usize,
        dtype_width_bytes: usize,
        dim_layout: KvDimLayout,
        block_layout: KvBlockLayout,
    ) -> Result<()> {
        // Prevent double registration.
        if self.gpu.is_initialized() {
            bail!("KV caches already registered");
        }
        if self.gpu.has_pending() {
            bail!("KV caches already pending registration");
        }
        if self.gpu.num_layers_set() {
            bail!("Worker details already set");
        }

        // Resolve LayoutConfig + BlockDimension from the labeled layout (the
        // relabeler inspects tensor strides; `block_layout` is cross-checked
        // against the stride-derived view).
        let (layout_config, block_dim) = determine_kv_layout(
            num_device_blocks,
            dtype_width_bytes,
            &tensors,
            &dim_layout,
            block_layout,
        )?;

        tracing::debug!(
            ?layout_config,
            ?block_dim,
            ?block_layout,
            "Determined KV layout configuration"
        );

        let num_layers = tensors.len();

        let pending = PendingWorkerState::builder()
            .tensors(tensors)
            .num_device_blocks(num_device_blocks)
            .dtype_width_bytes(dtype_width_bytes)
            .layout_config(layout_config)
            .mode(PendingLayoutMode::LayerSeparate {
                block_dim,
                block_layout,
            })
            .build()?;

        tracing::info!(
            cuda_device = pending.cuda_device_id,
            num_tensors = pending.tensors.len(),
            num_device_blocks,
            dtype_width_bytes,
            mode = ?pending.mode,
            "KV caches registered (deferred mode - waiting for leader RPC)"
        );

        self.gpu.set_pending(pending)?;
        self.gpu.set_num_layers(num_layers)?;
        Ok(())
    }

    fn register_cross_layers_kv_cache(
        &self,
        tensor: Arc<dyn TensorDescriptor>,
        num_device_blocks: usize,
        dtype_width_bytes: usize,
        dim_layout: KvDimLayout,
        block_layout: KvBlockLayout,
    ) -> Result<()> {
        if self.gpu.is_initialized() {
            bail!("KV caches already registered");
        }
        if self.gpu.has_pending() {
            bail!("KV caches already pending registration");
        }
        if self.gpu.num_layers_set() {
            bail!("Worker details already set");
        }

        let layout_config = determine_cross_layer_kv_layout(
            num_device_blocks,
            dtype_width_bytes,
            &tensor,
            &dim_layout,
            block_layout,
        )?;
        let num_layers = layout_config.num_layers;

        tracing::info!(
            num_device_blocks,
            num_layers,
            ?block_layout,
            "Registering cross-layer KV cache (fully-contiguous, deferred mode)"
        );

        let pending = PendingWorkerState::builder()
            .tensors(vec![tensor])
            .num_device_blocks(num_device_blocks)
            .dtype_width_bytes(dtype_width_bytes)
            .layout_config(layout_config)
            .mode(PendingLayoutMode::FullyContiguous { block_layout })
            .build()?;

        self.gpu.set_pending(pending)?;
        self.gpu.set_num_layers(num_layers)?;
        Ok(())
    }

    fn bind_connector_metadata(&self, metadata: KvConnectorMetadata) -> Result<()> {
        tracing::debug!(iteration = metadata.iteration, "Binding connector metadata");

        self.bind_plan(&metadata)?;

        // Legacy-shaped surface: the bare plan carries no control payload, so
        // synthesize an iteration-only one. The connector binding path
        // ([`Self::bind_serialized_metadata`]) carries the real fences.
        self.bind_metadata(ConnectorMetadata {
            iteration: metadata.iteration as u64,
            evicted_requests: Vec::new(),
            fences: Vec::new(),
        });
        Ok(())
    }

    fn clear_connector_metadata(&self) -> Result<()> {
        *self.metadata.lock() = None;
        if let Some(engine) = self.gpu.engine() {
            engine.clear_pass();
        }
        Ok(())
    }

    fn start_load_kv(&self) -> Result<()> {
        self.engine_driver.begin_forward_pass(self.iteration());
        // Drive the GPU engine inherently (not via the driver boundary) so the
        // intra-pass onboard error propagates to vLLM instead of being logged.
        if let Some(engine) = self.gpu.engine() {
            engine.start_load()?;
        }
        Ok(())
    }

    fn wait_for_layer_load(&self, layer_index: usize, stream_handle: u64) -> Result<()> {
        if let Some(engine) = self.gpu.engine() {
            engine.wait_for_layer_load(layer_index, stream_handle)?;
        }
        Ok(())
    }

    fn save_kv_layer(&self, layer_index: usize, stream_handle: u64) -> Result<()> {
        if let Some(engine) = self.gpu.engine() {
            engine.save_kv_layer(layer_index, stream_handle)?;
        }
        Ok(())
    }

    fn wait_for_save(&self) -> Result<()> {
        if let Some(engine) = self.gpu.engine() {
            engine.wait_for_save()?;
        }
        self.engine_driver.finish_forward_pass(self.iteration());
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.gpu.is_initialized()
    }

    fn shutdown(&self) -> Result<()> {
        self.engine_driver.shutdown();
        self.leader_actions.shutdown();
        Ok(())
    }

    fn get_finished(&self) -> FinishedRequests {
        // Drain whatever the engine has pushed since the last tick.
        self.completion.drain_finished()
    }

    fn get_failed_onboarding(&self) -> HashSet<usize> {
        // Drain the failed-load G1 block ids the engine has pushed.
        self.completion.drain_failed()
    }
}

#[cfg(test)]
mod tests {
    use super::{ConnectorMetadata, ConnectorWorkerInterface, Worker, WorkerLeaderActions};
    use kvbm_protocols::connector::{ActionId, SearchId};
    use kvbm_protocols::connector::{
        ActionStatus, BlockId, EvictionFence, EvictionOutcome, FenceToken, FindBlocksHandle,
        FindBlocksOutcome, FindBlocksRequest, LeaderEngine, LeaderEngineError, LoadOutcome,
        OffloadHandle, OnboardHandle, RequestId, RequestOffloadDrain, SaveOutcome, SequenceHash,
        WorkerEngineDriver,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct CountingEngineActions {
        begin: AtomicUsize,
        finish: AtomicUsize,
        shutdown: AtomicUsize,
    }
    impl WorkerEngineDriver for CountingEngineActions {
        fn begin_forward_pass(&self, _: usize) {
            self.begin.fetch_add(1, Ordering::SeqCst);
        }
        fn finish_forward_pass(&self, _: usize) {
            self.finish.fetch_add(1, Ordering::SeqCst);
        }
        fn await_fence(&self, _: FenceToken) {
            // The connector worker waits on its own WorkerCompletionState, not
            // the driver; this must never be called.
            unreachable!("connector worker does not route await_fence through the driver");
        }
        fn shutdown(&self) {
            self.shutdown.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[derive(Default)]
    struct CountingLeaderActions {
        shutdown: AtomicUsize,
    }
    impl WorkerLeaderActions for CountingLeaderActions {
        fn shutdown(&self) {
            self.shutdown.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn fresh_worker() -> (
        Worker,
        Arc<CountingEngineActions>,
        Arc<CountingLeaderActions>,
    ) {
        let eng = Arc::new(CountingEngineActions::default());
        let led = Arc::new(CountingLeaderActions::default());
        let worker = Worker::with_actions(
            Arc::clone(&eng) as Arc<dyn WorkerEngineDriver>,
            Arc::clone(&led) as Arc<dyn WorkerLeaderActions>,
        );
        (worker, eng, led)
    }

    fn meta(iteration: u64, evicted: &[&str]) -> ConnectorMetadata {
        ConnectorMetadata {
            iteration,
            evicted_requests: evicted.iter().map(|s| s.to_string()).collect(),
            fences: Vec::new(),
        }
    }

    // ---- forward-pass / shutdown boundaries ----

    #[test]
    fn shutdown_forwards_to_engine_and_leader_actions() {
        let (worker, eng, led) = fresh_worker();
        worker.shutdown().unwrap();
        assert_eq!(eng.shutdown.load(Ordering::SeqCst), 1);
        assert_eq!(led.shutdown.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn start_load_and_wait_for_save_call_engine_pass_boundaries() {
        let (worker, eng, _) = fresh_worker();
        worker.bind_metadata(meta(42, &[]));
        worker.start_load_kv().unwrap();
        worker.wait_for_save().unwrap();
        assert_eq!(eng.begin.load(Ordering::SeqCst), 1);
        assert_eq!(eng.finish.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn clear_metadata_drops_cached_iteration() {
        let (worker, eng, _) = fresh_worker();
        worker.bind_metadata(meta(7, &[]));
        worker.clear_connector_metadata().unwrap();
        // With no metadata, iteration defaults to 0 inside the worker.
        worker.start_load_kv().unwrap();
        assert_eq!(eng.begin.load(Ordering::SeqCst), 1);
    }

    // ---- engine→worker completion push (via the injected sink) ----

    /// (b) The engine pushes through `worker_sink()`; `get_finished` drains the
    /// right request into each side (save→offloading, load→onboarding) and a
    /// second drain is empty (consume-once).
    #[test]
    fn sink_marks_drain_into_get_finished_once() {
        let (worker, _, _) = fresh_worker();
        let sink = worker.worker_sink();

        sink.mark_save_finished(&"req-send".to_string(), SaveOutcome::Done);
        sink.mark_load_finished(&"req-recv".to_string(), LoadOutcome::Done);

        let finished = worker.get_finished();
        assert_eq!(
            finished.offloading,
            ["req-send".to_string()].into_iter().collect()
        );
        assert_eq!(
            finished.onboarding,
            ["req-recv".to_string()].into_iter().collect()
        );

        // Drained — a second poll is empty.
        let empty = worker.get_finished();
        assert!(empty.offloading.is_empty());
        assert!(empty.onboarding.is_empty());
    }

    /// (c) A failed partial load surfaces the request in `get_finished().onboarding`
    /// (so vLLM leaves WAITING_FOR_REMOTE_KVS) AND its block ids in
    /// `get_failed_onboarding`, paired in the same tick.
    #[test]
    fn failed_load_surfaces_request_and_block_ids() {
        let (worker, _, _) = fresh_worker();
        let sink = worker.worker_sink();

        sink.mark_load_finished(
            &"req-bad".to_string(),
            LoadOutcome::FailedPartial {
                block_ids: vec![7, 9],
            },
        );

        let finished = worker.get_finished();
        assert!(finished.onboarding.contains("req-bad"));
        assert_eq!(worker.get_failed_onboarding(), [7, 9].into_iter().collect());
        // Consume-once.
        assert!(worker.get_failed_onboarding().is_empty());
    }

    /// `fence_tokens_for_rank` selects ONLY this rank's tokens (clean-fail
    /// discriminator for the rank filter, so the blocking test below isn't the
    /// only thing guarding it).
    #[test]
    fn fence_tokens_for_rank_filters_by_rank() {
        let r0 = FenceToken::new(0);
        let r1 = FenceToken::new(1);
        let metadata = ConnectorMetadata {
            iteration: 1,
            evicted_requests: vec!["a".to_string(), "b".to_string()],
            fences: vec![
                EvictionFence {
                    request_id: "a".to_string(),
                    per_worker: vec![r0, r1],
                },
                EvictionFence {
                    request_id: "b".to_string(),
                    per_worker: vec![r1],
                },
            ],
        };

        assert_eq!(Worker::fence_tokens_for_rank(&metadata, 0), vec![r0]);
        assert_eq!(Worker::fence_tokens_for_rank(&metadata, 1), vec![r1, r1]);
    }

    /// Minimal leader engine whose `evict` mints a one-token (rank 0) fence so
    /// the wire tests have real fences to carry; every other verb is inert.
    struct FencingEngine;
    impl LeaderEngine for FencingEngine {
        fn offload(
            self: Arc<Self>,
            _req: &RequestId,
            _pairs: Vec<(SequenceHash, BlockId)>,
        ) -> Result<OffloadHandle, LeaderEngineError> {
            Err(LeaderEngineError::Shutdown)
        }
        fn evict(&self, req: &RequestId) -> EvictionOutcome {
            EvictionOutcome {
                fence: EvictionFence {
                    request_id: req.clone(),
                    per_worker: vec![FenceToken::new(0)],
                },
                handle: None,
            }
        }
        fn find_blocks(
            self: Arc<Self>,
            _req: &FindBlocksRequest,
            _live: Option<&FindBlocksHandle>,
        ) -> Result<FindBlocksOutcome, LeaderEngineError> {
            Ok(FindBlocksOutcome::Resolved {
                matched_tokens: 0,
                minted: None,
                release_parked: false,
            })
        }
        fn onboard_blocks(
            self: Arc<Self>,
            _handle: &FindBlocksHandle,
            _dest: &[BlockId],
            _num_external_tokens: usize,
        ) -> Result<OnboardHandle, LeaderEngineError> {
            Err(LeaderEngineError::Shutdown)
        }
        fn take_offload_drain(&self, _req: &RequestId) -> Option<RequestOffloadDrain> {
            None
        }
        fn poll_action(&self, _id: &ActionId) -> ActionStatus {
            ActionStatus::Complete
        }
        fn release_search(&self, _id: &SearchId) {}
        fn release_action(&self, _action: &ActionId) {}
    }

    /// The wire envelope: a leader-staged eviction's fences + evicted ids ride
    /// `serialize_metadata` → `bind_serialized_metadata` end-to-end. The worker
    /// is rank 1 and the fence carries only a rank-0 token, so the bind never
    /// blocks; the assertion is that the CONTROL payload landed.
    #[test]
    fn wire_envelope_carries_fences_leader_to_worker() {
        use crate::common::KvConnectorMetadata;
        use crate::connector::leader::Leader;

        let leader = Leader::with_engine(Arc::new(FencingEngine), 4);
        // The slot the eviction keys on — created from the Request, as the
        // binding layer does before every GNMT poll.
        leader
            .create_slot(crate::common::Request::with_token_limits(
                "r1",
                (0..8u32).collect::<Vec<u32>>(),
                None,
                None,
                None,
                None,
                None,
            ))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.on_evicted("r1").unwrap();

        let bytes = leader
            .serialize_metadata(KvConnectorMetadata::new(7))
            .unwrap();

        let eng = Arc::new(CountingEngineActions::default());
        let led = Arc::new(CountingLeaderActions::default());
        let worker = Worker::with_actions_and_rank(eng, led, 1);
        let bound = worker.bind_serialized_metadata(&bytes).unwrap();
        assert!(bound);

        let stashed = worker.metadata.lock();
        let control = stashed.as_ref().expect("control payload bound");
        assert_eq!(control.evicted_requests, vec!["r1".to_string()]);
        assert_eq!(control.fences.len(), 1, "the eviction fence rode the wire");
        assert_eq!(control.fences[0].request_id, "r1");
    }

    /// A fence for THIS worker's rank blocks `bind_metadata` until the engine
    /// reports the token complete; a fence for a different rank never blocks.
    /// The completer is spawned BEFORE `bind_metadata` so the bounded forward
    /// pass cannot hang.
    #[test]
    fn bind_metadata_gates_on_own_rank_fence() {
        use std::time::{Duration, Instant};

        let (worker, _, _) = fresh_worker(); // rank 0

        let my_token = FenceToken::new(0); // this worker's rank
        let other_token = FenceToken::new(1); // a different rank — must be ignored
        let metadata = ConnectorMetadata {
            iteration: 5,
            evicted_requests: vec!["evicted".to_string()],
            fences: vec![EvictionFence {
                request_id: "evicted".to_string(),
                per_worker: vec![my_token, other_token],
            }],
        };

        // Complete the rank-0 token after a bounded delay, off the bind thread.
        let sink = worker.worker_sink();
        let start = Instant::now();
        let completer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(25));
            sink.mark_fence_complete(my_token);
        });

        worker.bind_metadata(metadata); // blocks until my_token completes
        assert!(
            start.elapsed() >= Duration::from_millis(20),
            "bind_metadata returned before its rank's fence completed",
        );
        completer.join().unwrap();
    }

    /// One step's envelope (control + plan) with a rank-0 fence token, as the
    /// leader seals it, plus the bare control payload for the bind side.
    fn fenced_step(iteration: u64, token: FenceToken) -> (Vec<u8>, ConnectorMetadata) {
        use crate::common::KvConnectorMetadata;
        use crate::connector::metadata::WireMetadata;

        let control = ConnectorMetadata {
            iteration,
            evicted_requests: vec!["evicted".to_string()],
            fences: vec![EvictionFence {
                request_id: "evicted".to_string(),
                per_worker: vec![token],
            }],
        };
        let bytes = serde_json::to_vec(&WireMetadata {
            plan: KvConnectorMetadata::new(iteration as usize),
            control: control.clone(),
        })
        .unwrap();
        (bytes, control)
    }

    /// V1 runner ordering: `handle_preemptions` fires BEFORE the metadata bind.
    /// The hook awaits + takes the step's tokens; the bind for the same
    /// iteration must no-op on the funnel — re-awaiting the taken tokens would
    /// hang the forward pass (bounded here so a regression fails, not hangs).
    #[test]
    fn handle_preemptions_then_bind_awaits_once_per_iteration() {
        use std::time::Duration;

        let (worker, _, _) = fresh_worker(); // rank 0
        let worker = Arc::new(worker);
        let token = FenceToken::new(0);
        let (bytes, control) = fenced_step(9, token);

        // The engine drained before the hook ran.
        worker.worker_sink().mark_fence_complete(token);

        worker.handle_preemptions(&bytes).unwrap();
        assert!(
            !worker.completion.fence_recorded(&token),
            "the first caller awaited AND took the token"
        );

        let w = Arc::clone(&worker);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            w.bind_metadata(control);
            tx.send(()).ok();
        });
        assert!(
            rx.recv_timeout(Duration::from_secs(2)).is_ok(),
            "the bind after handle_preemptions must no-op for the same iteration"
        );
    }

    /// connector runner ordering: the metadata bind fires BEFORE `handle_preemptions`.
    /// The bind awaits + takes; the hook for the same iteration must no-op.
    #[test]
    fn bind_then_handle_preemptions_awaits_once_per_iteration() {
        use std::time::Duration;

        let (worker, _, _) = fresh_worker(); // rank 0
        let worker = Arc::new(worker);
        let token = FenceToken::new(0);
        let (bytes, control) = fenced_step(11, token);

        worker.worker_sink().mark_fence_complete(token);

        worker.bind_metadata(control);
        assert!(
            !worker.completion.fence_recorded(&token),
            "the bind awaited AND took the token"
        );

        let w = Arc::clone(&worker);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            w.handle_preemptions(&bytes).unwrap();
            tx.send(()).ok();
        });
        assert!(
            rx.recv_timeout(Duration::from_secs(2)).is_ok(),
            "handle_preemptions after the bind must no-op for the same iteration"
        );
    }

    /// `handle_preemptions` parses the control payload out of the FULL wire
    /// envelope (plan field present and skipped) and blocks on this rank's
    /// tokens exactly like the bind path.
    #[test]
    fn handle_preemptions_gates_on_own_rank_fence() {
        use std::time::{Duration, Instant};

        let (worker, _, _) = fresh_worker(); // rank 0
        let token = FenceToken::new(0);
        let (bytes, _control) = fenced_step(13, token);

        let sink = worker.worker_sink();
        let start = Instant::now();
        let completer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(25));
            sink.mark_fence_complete(token);
        });

        worker.handle_preemptions(&bytes).unwrap();
        assert!(
            start.elapsed() >= Duration::from_millis(20),
            "handle_preemptions returned before its rank's fence completed",
        );
        completer.join().unwrap();
    }
}
