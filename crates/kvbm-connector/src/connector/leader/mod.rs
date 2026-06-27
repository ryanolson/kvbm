// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-side entry point for connector.
//!
//! [`Leader`] wraps a private [`state::LeaderState`] behind a `Mutex` and an
//! `Arc<dyn LeaderEngine>`. It implements the byte-for-byte C4
//! [`ConnectorLeaderApi`] (on `Arc<Leader>`, never on `Leader`) and reproduces
//! the inherent methods the PyO3 bindings call. The full request lifecycle is
//! live: `get_num_new_matched_tokens` translates slot facts into the engine's
//! unified `find_blocks` and maps its outcome, `update_state_after_alloc`
//! commits the external load (`allocate` → `onboard_blocks`),
//! `build_connector_meta` walks the scheduler output (save cursor → offload)
//! and arms the forward-pass flush trigger, and `request_finished` /
//! `update_connector_output` / `on_evicted` drive the finish, reap, and
//! eviction-fence machinery. The leader is a pure vLLM adapter: match routing
//! (window math, fresh-vs-refresh, prefill-vs-local, the deferral guard) lives
//! engine-side.

mod cd;
mod construct;
mod sink;
mod slot;
mod state;

// Leader-side hub/P2P infrastructure consumed by `construct` and the CD wiring:
// the hub handshake (GET /v1/config feature resolution), the KV-index ZMQ
// publisher, the hub peer resolver, and the hub-client builder.
pub(crate) mod hub_handshake;
pub(crate) mod hub_indexer;
pub(crate) mod peer_resolver;

mod hub_client;
pub use hub_client::build_hub_client;

pub use slot::RequestSlot;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow, bail};
use parking_lot::Mutex;
use velo::PeerInfo;

use kvbm_common::BlockId;
use kvbm_engine::worker::{SerializedLayout, VeloWorkerClient};
use kvbm_hub::HubClient;
use kvbm_logical::events::KvbmCacheEventsPublisher;
use kvbm_protocols::connector::{
    FinishedStatus as EngineFinishedStatus, LeaderEngineError, WorkerEngineDriver,
};

use crate::common::{
    ConsolidatorEndpoints, FinishedStatus, KvConnectorMetadata, Request, SchedulerOutput,
};
use crate::connector::worker::ConnectorWorkerClient;
use crate::{InstanceId, KvbmRuntime, WorkerAddress};

use super::engine::noop_leader_engine;
use super::metadata::{ConnectorMetadata, WireMetadata};
use state::LeaderState;

/// Leader-side errors surfaced by the lifecycle methods.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("slot not found for request id: {0}")]
    SlotNotFound(String),
    #[error("slot already exists for request id: {0}")]
    SlotAlreadyExists(String),
    /// vLLM committed to an external load (`num_external_tokens > 0`) for a
    /// slot parking no lifecycle handle — a broken GNMT↔USAA contract. Failing
    /// loud beats leaving the request parked in `WAITING_FOR_REMOTE_KVS`
    /// waiting on a load nobody started.
    #[error("external load for request id {0}: no parked lifecycle to onboard from")]
    ExternalLoadWithoutSearch(String),
    /// The engine refused `find_blocks` (prefill misconfiguration, digest
    /// divergence, lifecycle desync). Failing loud beats silently recomputing
    /// the whole prompt from token zero.
    #[error("find_blocks for request id {request_id}: engine refused")]
    FindBlocksRejected {
        request_id: String,
        #[source]
        source: LeaderEngineError,
    },
    /// The engine refused the onboard for a committed external load (e.g. the
    /// pin went `Lost` between GNMT and USAA, or the committed external-token
    /// count diverged from the engine-stored promise).
    #[error("external load for request id {request_id}: engine rejected onboard")]
    OnboardRejected {
        request_id: String,
        #[source]
        source: LeaderEngineError,
    },
    /// A committed external load arrived while the slot already parks an
    /// onboard handle. Re-entering would replace `slot.onboard`, detaching
    /// the leader's view of the live load (vLLM issues at most one committed
    /// external load per GNMT promise, so this is always a contract anomaly).
    #[error("external load for request id {0}: onboard already in flight")]
    OnboardAlreadyInFlight(String),
    /// The token sequence extension was rejected by the underlying
    /// `TokenBlockSequence::extend` (e.g. multimodal runs present, or an
    /// internal commit failure).
    #[error("token extension failed for request id {request_id}: {message}")]
    TokenExtend { request_id: String, message: String },
}

/// Scheduler-side connector leader. Drop-in for the legacy `ConnectorLeader`.
///
/// The [`ConnectorLeaderApi`] impl lives on `Arc<Leader>`, never on the bare
/// struct — mirroring the legacy leader so the binding's `ApiRoute::Direct`
/// arm must hold `&Arc<Leader>` and call inherent methods. The bound holds for
/// the `Arc`:
///
/// ```
/// use kvbm_connector::connector::Leader;
/// use kvbm_connector::connector::leader::ConnectorLeaderApi;
/// use std::sync::Arc;
/// fn carries_api<T: ConnectorLeaderApi>() {}
/// carries_api::<Arc<Leader>>();
/// ```
///
/// …but deliberately not for the bare struct:
///
/// ```compile_fail
/// use kvbm_connector::connector::Leader;
/// use kvbm_connector::connector::leader::ConnectorLeaderApi;
/// fn carries_api<T: ConnectorLeaderApi>() {}
/// carries_api::<Leader>();
/// ```
pub struct Leader {
    state: Mutex<LeaderState>,
    block_size: usize,
    /// Single-shot guard for [`Self::install_engine`]. The binding constructors
    /// (`new`/`new_with_consolidator`) start the state on a placeholder
    /// `NoopEngine` so a pre-init `propose` is benign (search → no match), then
    /// swap in the real engine exactly once after deferred construction builds
    /// it. `with_engine` injects directly and never installs.
    engine_installed: OnceLock<()>,
    /// Deferred-construction inputs, `Some` on the binding path and `None` on
    /// the `with_engine` test path (engine injected directly, no handshake).
    /// `register_worker` accumulates the worker peers here; the engine-stack
    /// build in `initialize` consumes them.
    construction: Option<Construction>,
    /// KV-index publisher held alive for the leader's life (dropping it aborts
    /// the publish task). Installed by `initialize` when the hub indexer is
    /// effective; empty otherwise.
    indexer_publisher: OnceLock<KvbmCacheEventsPublisher>,
    /// KV-index-only hub registration held alive (RAII `DELETE` on drop).
    /// Installed by `initialize` when Indexer is the sole effective hub feature.
    indexer_hub_client: OnceLock<Arc<HubClient>>,
    /// CD/P2P hub registration held alive (RAII `DELETE` on drop). Installed
    /// by `initialize` when the conditional-disagg wiring registers with the
    /// hub; empty otherwise.
    cd_hub_client: OnceLock<Arc<HubClient>>,
    /// Forward-pass flush trigger, armed by `initialize` (empty on the
    /// `with_engine` test path — the glue is velo-coupled). See [`FlushGlue`].
    flush: OnceLock<FlushGlue>,
}

/// Post-`initialize` handles for the cross-seam offload-flush trigger
/// (REFACTOR.md §3 "Decision D, part 2"). When a step's walk schedules
/// offloads, the leader mints one velo event per worker — the handles ride
/// the wire plan's `foward_pass_completion_events` and each worker's engine
/// triggers its event once the last layer's compute completes — then a
/// spawned task awaits the merge and calls the engine driver's
/// `finish_forward_pass(iteration)`, which submits that pass's buffered
/// offloads (the engine flush is iteration-scoped, so a late merge can never
/// submit a later pass's mid-pass buffer).
struct FlushGlue {
    /// The leader engine's driver face (same object as the installed
    /// `LeaderEngine`).
    driver: Arc<dyn WorkerEngineDriver>,
    runtime: Arc<KvbmRuntime>,
    /// Rank-indexed worker instance ids, as accumulated by `register_worker`.
    worker_ids: Vec<InstanceId>,
    /// Missed-flush high-water mark (see [`Self::spawn_flush`]'s failure
    /// policy). While a strand is outstanding, every PASS-RUNNING step arms
    /// the flush trigger — not only offload-scheduling ones — so the stranded
    /// buffer is swept by the next confirmed pass instead of waiting for a
    /// later offload to happen.
    stranded: Arc<StrandMark>,
}

/// Missed-flush high-water mark. Encodes "stranded through iteration `n`" as
/// `n + 1`, so iteration 0 is representable (`0` = no strand outstanding —
/// nothing guarantees scheduler iterations start at 1).
#[derive(Default)]
struct StrandMark(std::sync::atomic::AtomicUsize);

impl StrandMark {
    /// Record a missed flush for `iteration` (monotone: keeps the highest).
    fn record(&self, iteration: usize) {
        self.0.fetch_max(
            iteration.saturating_add(1),
            std::sync::atomic::Ordering::AcqRel,
        );
    }

    /// Whether any missed flush awaits recovery.
    fn outstanding(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Acquire) != 0
    }

    /// Clear the mark iff every recorded strand is at or below `iteration` (a
    /// confirmed flush swept entries stamped `<= iteration`). An out-of-order
    /// OLDER confirmation must not clear a newer strand its sweep did not
    /// cover.
    fn clear_through(&self, iteration: usize) {
        let _ = self.0.fetch_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |mark| (mark != 0 && mark <= iteration.saturating_add(1)).then_some(0),
        );
    }
}

impl FlushGlue {
    /// One velo event per worker. The leader keeps the awaitable `Event`
    /// objects; the wire carries only the handles.
    fn mint_events(&self) -> Result<HashMap<InstanceId, Arc<velo::Event>>> {
        self.worker_ids
            .iter()
            .map(|id| {
                let event = self.runtime.messenger().events().new_event()?;
                Ok((*id, Arc::new(event)))
            })
            .collect()
    }

    /// Await the merge of every worker's pass-completion event, then submit
    /// the pass's buffered offloads via `finish_forward_pass`. The events move
    /// into the task and drop only after the await — dropping an untriggered
    /// `velo::Event` poisons its waiters, the right failure shape for a worker
    /// that died mid-pass.
    ///
    /// **Failure policy: never flush on an UNCONFIRMED pass completion.** A
    /// flush without evidence the pass finished reads G1 while the GPU still
    /// writes it. If the awaiter/merge cannot be built or the await fails, the
    /// iteration's offloads stay buffered and the [`StrandMark`] records it —
    /// every later pass-running step then arms the flush trigger, so the next
    /// CONFIRMED pass sweeps the stragglers (`finish_forward_pass(n)` drains
    /// entries stamped `<= n`; completed blocks are immutable, so the late
    /// submit is safe). Stranded handles stay `Pending` until then; teardown
    /// (`shutdown`) clears the buffer.
    fn spawn_flush(&self, events: HashMap<InstanceId, Arc<velo::Event>>, iteration: usize) {
        let event_manager = self.runtime.messenger().event_manager();
        let driver = Arc::clone(&self.driver);
        let stranded = Arc::clone(&self.stranded);
        self.runtime.tokio().spawn(async move {
            let handles: Vec<velo::EventHandle> =
                events.values().map(|event| event.handle()).collect();
            let await_completion = async |handle: velo::EventHandle| -> bool {
                match event_manager.awaiter(handle) {
                    Ok(awaiter) => match awaiter.await {
                        Ok(()) => true,
                        Err(e) => {
                            tracing::error!(error = %e, "pass-completion await failed");
                            false
                        }
                    },
                    Err(e) => {
                        tracing::error!(error = %e, "flush awaiter creation failed");
                        false
                    }
                }
            };
            let confirmed = if handles.len() == 1 {
                await_completion(handles[0]).await
            } else {
                match event_manager.merge_events(handles) {
                    Ok(merged) => await_completion(merged).await,
                    Err(e) => {
                        tracing::error!(error = %e, "flush merge creation failed");
                        false
                    }
                }
            };
            if confirmed {
                driver.finish_forward_pass(iteration);
                stranded.clear_through(iteration);
            } else {
                stranded.record(iteration);
                tracing::error!(
                    iteration,
                    "pass completion unconfirmed; offloads stay buffered — every \
                     later pass-running step arms the flush until a confirmed pass \
                     sweeps them"
                );
            }
            drop(events);
        });
    }
}

/// Leader-side construction state — the connector analogue of the legacy
/// `ConnectorLeader`'s init fields, kept local so the legacy leader stays
/// untouched. `register_worker` (pre-`initialize`) accumulates the per-worker
/// velo peers; `initialize` drains them to build the engine stack.
struct Construction {
    runtime: Arc<KvbmRuntime>,
    // Consumed by the engine-stack build (`InstanceLeader::with_consolidator`).
    consolidator_endpoints: Option<ConsolidatorEndpoints>,
    workers: Mutex<WorkerAccum>,
}

/// Per-worker velo peers accumulated across `register_worker` calls. The four
/// vectors are rank-indexed and grow in lockstep (one push per worker);
/// [`construct::build_engine_stack`] drains them to build the engine stack.
#[derive(Default)]
struct WorkerAccum {
    instance_ids: Vec<InstanceId>,
    connector_clients: Vec<ConnectorWorkerClient>,
    transfer_clients: Vec<VeloWorkerClient>,
    /// Filled by `initialize` from each worker's init response; rank-aligned
    /// with the client vectors above.
    metadata: Vec<SerializedLayout>,
}

impl Leader {
    // ---- constructors ----

    /// Build a leader over an explicitly injected engine (tests/wiring).
    pub fn with_engine(
        engine: Arc<dyn kvbm_protocols::connector::LeaderEngine>,
        block_size: usize,
    ) -> Self {
        Self {
            state: Mutex::new(LeaderState::new(engine, block_size, None)),
            block_size,
            engine_installed: OnceLock::new(),
            construction: None,
            indexer_publisher: OnceLock::new(),
            indexer_hub_client: OnceLock::new(),
            cd_hub_client: OnceLock::new(),
            flush: OnceLock::new(),
        }
    }

    /// Binding constructor. The skeleton runs over a standalone Noop engine;
    /// `runtime` is reserved for the real engine wiring.
    pub fn new(runtime: Arc<KvbmRuntime>, block_size: usize) -> Self {
        Self::new_with_consolidator(runtime, block_size, None)
    }

    /// Binding constructor with consolidator endpoints (unused in the skeleton).
    pub fn new_with_consolidator(
        runtime: Arc<KvbmRuntime>,
        block_size: usize,
        consolidator_endpoints: Option<ConsolidatorEndpoints>,
    ) -> Self {
        let matched_tokens = runtime
            .observability()
            .compat_metrics()
            .matched_tokens
            .clone();
        Self {
            state: Mutex::new(LeaderState::new(
                noop_leader_engine(),
                block_size,
                Some(matched_tokens),
            )),
            block_size,
            engine_installed: OnceLock::new(),
            construction: Some(Construction {
                runtime,
                consolidator_endpoints,
                workers: Mutex::new(WorkerAccum::default()),
            }),
            indexer_publisher: OnceLock::new(),
            indexer_hub_client: OnceLock::new(),
            cd_hub_client: OnceLock::new(),
            flush: OnceLock::new(),
        }
    }

    /// Install the real [`LeaderEngine`](kvbm_protocols::connector::LeaderEngine)
    /// built by deferred construction, replacing the placeholder `NoopEngine` the
    /// binding constructors start on (REFACTOR.md construction step 8). Install
    /// runs at the end of `initialize`, before vLLM's first `propose`.
    ///
    /// Two invariants are enforced rather than merely documented, so a misuse
    /// surfaces loudly instead of silently orphaning handles:
    /// - **no live handles**: refuse if any slot holds an engine-minted handle
    ///   (`search`/`onboard`/`offload`/drain-holder) — swapping the engine out
    ///   from under them would orphan their RAII release. A benign pre-init
    ///   propose on the placeholder `NoopEngine` mints nothing (REFACTOR.md:525),
    ///   so its handle-free slot does NOT block install — the documented
    ///   pre-init path stays recoverable. Checked first and with no side effect.
    /// - **single-shot**: the `OnceLock` guard makes a second install error
    ///   rather than replace a live engine mid-flight.
    pub fn install_engine(
        &self,
        engine: Arc<dyn kvbm_protocols::connector::LeaderEngine>,
    ) -> Result<()> {
        let mut state = self.state.lock();
        if state.has_live_handles() {
            bail!(
                "cannot install the leader engine while a slot holds a live handle \
                 (search/onboard/offload); the swap would orphan its RAII release"
            );
        }
        self.engine_installed
            .set(())
            .map_err(|_| anyhow!("leader engine already installed"))?;
        state.install_engine(engine);
        Ok(())
    }

    // ---- inherent binding methods ----

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn has_slot(&self, request_id: &str) -> bool {
        self.state.lock().contains(request_id)
    }

    pub fn create_slot(&self, request: Request) -> Result<()> {
        self.state.lock().create_slot(request)?;
        Ok(())
    }

    pub fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        self.state.lock().extend_tokens(request_id, tokens)?;
        Ok(())
    }

    pub fn get_slot_total_tokens(&self, request_id: &str) -> Result<usize> {
        Ok(self.state.lock().total_tokens(request_id)?)
    }

    /// GNMT. Translates slot facts (full hash chain + counts + optional
    /// decode-minted prefill params) into the engine's unified `find_blocks`
    /// — the engine derives the window and routes — and reports
    /// `(matched_tokens, load_async)`: `(None, false)` while the match
    /// resolves (vLLM re-polls), `(Some(0), false)` on no match,
    /// `(Some(n), true)` on a hit — the async external load the later
    /// `update_state_after_alloc` commits to.
    pub fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        Ok(self.state.lock().gnmt(request_id, num_computed_tokens)?)
    }

    /// USAA. Records the allocated block ids; a nonzero `num_external_tokens`
    /// is vLLM committing to the async external load promised at GNMT time —
    /// the engine onboard (`onboard_blocks`) is issued in-call with the FULL
    /// allocated list as dest (the engine routes off the parked lifecycle's
    /// kind and slices the external window itself).
    pub fn update_state_after_alloc(
        self: &Arc<Self>,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        self.state
            .lock()
            .allocate(request_id, block_ids, num_external_tokens)?;
        Ok(())
    }

    /// Build the per-step connector metadata. The per-step walk syncs block
    /// ids, advances each slot's `evaluated_tokens` save cursor (capped at
    /// hash-complete × allocated blocks), and hands completed novel blocks to
    /// `engine.offload` as `(SequenceHash, BlockId)` pairs. The offload
    /// pipeline's presence filter deduplicates already-cached blocks, so the
    /// cursor starts at 0 on a fresh slot. Intra-pass fields stay `None`
    /// (engine-owned inter-pass onboard).
    ///
    /// When the flush glue is armed ([`Self::initialize`] ran), this also
    /// mints one velo event per worker — the handles ride
    /// `foward_pass_completion_events`, each worker's engine triggers its
    /// event after the last layer's compute — and spawns the merge-await that
    /// calls the engine's `finish_forward_pass(iteration)`, submitting this
    /// pass's buffered offloads.
    ///
    /// The mint happens BEFORE the walk: a mint failure (broken velo runtime)
    /// aborts the step before any offload is buffered, so the error path
    /// strands nothing. Unused events for a no-offload step drop harmlessly
    /// (no waiter ever sees their handles).
    pub fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata> {
        let iteration = output.iteration;
        // Only a frame that actually runs a forward pass can trigger
        // completion events — arming an empty frame would await an event no
        // worker ever fires (the worker binds nothing for it), leaking the
        // task and the events.
        let runs_pass = output.total_num_scheduled_tokens > 0;
        let events = match self.flush.get() {
            Some(glue) if runs_pass => {
                // Stamp the engine's iteration BEFORE the walk buffers
                // offloads under it (the engine flush is iteration-scoped).
                glue.driver.begin_forward_pass(iteration);
                Some(glue.mint_events()?)
            }
            _ => None,
        };

        let walk = self.state.lock().build_connector_meta(&output);
        let mut metadata = walk.metadata;

        if let Some(glue) = self.flush.get()
            && runs_pass
        {
            // Arm on this step's own offloads OR while a missed flush is
            // outstanding — recovery must not wait for a later
            // offload-scheduling pass (a pure-decode tail would otherwise
            // never sweep the stranded buffer).
            if walk.scheduled_offloads || glue.stranded.outstanding() {
                let events =
                    events.expect("minted above for every pass-running frame with the glue armed");
                if events.is_empty() {
                    // Degenerate zero-worker wiring: no pass can ever trigger
                    // a flush, but none can write G1 either. Leave the buffer
                    // for teardown rather than flush into an undefined pass
                    // model.
                    tracing::warn!(
                        iteration,
                        "offloads scheduled with no registered workers; leaving them buffered"
                    );
                } else {
                    metadata = metadata.with_events(
                        events
                            .iter()
                            .map(|(id, event)| (*id, event.handle()))
                            .collect(),
                    );
                    glue.spawn_flush(events, iteration);
                }
            }
        }
        Ok(metadata)
    }

    /// Seal the per-step worker payload: the routed transfer `plan` plus this
    /// step's control payload (draining the staged eviction fences via
    /// [`Self::build_metadata`]) as one [`WireMetadata`] blob. The binding
    /// ships the bytes; the connector worker unseals them in
    /// `Worker::bind_serialized_metadata`.
    pub fn serialize_metadata(&self, plan: KvConnectorMetadata) -> Result<Vec<u8>> {
        let wire = WireMetadata {
            plan,
            control: self.build_metadata(),
        };
        Ok(serde_json::to_vec(&wire)?)
    }

    pub fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        self.state
            .lock()
            .update_connector_output(finished_sending, finished_recving);
        Ok(())
    }

    pub fn request_finished(&self, request_id: &str) -> FinishedStatus {
        self.state.lock().request_finished(request_id)
    }

    /// Register a worker peer (scheduler-side `set_xfer_handshake_metadata`).
    /// Mirrors the legacy `ConnectorLeader::register_worker`: registers the
    /// velo peer and accumulates its rank-indexed RPC clients for `initialize`
    /// to consume. Ranks must arrive in order (`rank == count so far`).
    pub fn register_worker(
        &self,
        rank: usize,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> Result<()> {
        let construction = self
            .construction
            .as_ref()
            .ok_or_else(|| anyhow!("register_worker requires a runtime-backed leader"))?;
        let messenger = construction.runtime.messenger();
        let mut workers = construction.workers.lock();

        if rank != workers.instance_ids.len() {
            bail!(
                "Rank mismatch: got rank {rank}, expected {}",
                workers.instance_ids.len()
            );
        }

        messenger.register_peer(PeerInfo::new(instance_id, worker_address))?;
        workers.instance_ids.push(instance_id);
        workers
            .connector_clients
            .push(ConnectorWorkerClient::new(messenger.clone(), instance_id));
        workers
            .transfer_clients
            .push(VeloWorkerClient::new(messenger.clone(), instance_id));
        Ok(())
    }

    /// Initialize the registered workers and install the real engine.
    ///
    /// Drives [`construct::build_engine_stack`] on the runtime (it awaits the
    /// per-worker velo round-trips), then mints the in-process `LeaderEngine` via
    /// the engine-crate factory and swaps it in with [`Self::install_engine`].
    ///
    /// The engine is built over a real [`VeloWorkerSink`](sink::VeloWorkerSink)
    /// (the leader half of the delegate loop). Still not runnable end-to-end:
    /// the worker's transfer/forward-pass runtime (`WorkerEngine`, `DirectWorker`,
    /// CUDA) and the cross-seam offload-flush trigger are later steps, and the
    /// binding still drives the legacy worker until P-D2.
    pub fn initialize(self: &Arc<Self>) -> Result<()> {
        let runtime = self
            .construction
            .as_ref()
            .ok_or_else(|| anyhow!("initialize requires a runtime-backed leader"))?
            .runtime
            .clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let this = self.clone();
        runtime.tokio().spawn(async move {
            let _ = tx.send(this.initialize_async().await);
        });
        rx.recv()
            .map_err(|_| anyhow!("initialize task dropped without sending a result"))?
    }

    /// Async body of [`Self::initialize`]: build the engine stack, then install
    /// the engine. Separated so `initialize` can await it on the runtime.
    async fn initialize_async(self: Arc<Self>) -> Result<()> {
        let construction = self
            .construction
            .as_ref()
            .ok_or_else(|| anyhow!("initialize requires a runtime-backed leader"))?;
        let mut stack = construct::build_engine_stack(construction).await?;

        // Hold the KV-index RAII guards alive for the leader's life (dropping
        // them aborts the publish task / fires a premature hub DELETE).
        if let Some(publisher) = stack.indexer_publisher.take() {
            let _ = self.indexer_publisher.set(publisher);
        }
        if let Some(hub) = stack.indexer_hub_client.take() {
            let _ = self.indexer_hub_client.set(hub);
        }

        // Leader half of the delegate loop: the engine pushes completions
        // through this sink to the worker velo peers (each worker registers the
        // receiving handlers at `Worker::new`). Built from the rank-indexed
        // worker velo ids `register_worker` accumulated.
        let worker_ids = construction.workers.lock().instance_ids.clone();
        let sink = sink::VeloWorkerSink::new(
            construction.runtime.messenger().clone(),
            worker_ids.clone(),
            construction.runtime.tokio(),
        );

        let block_size = stack.reference_config.page_size;
        // Conditional-disagg transports: wired only when BOTH the parsed
        // `disagg` config and a hub handshake carrying ConditionalDisagg are
        // present (the wiring clones what it needs from the stack BEFORE the
        // factory below consumes it). Absent, the engine keeps
        // `RemoteOps::default()` — fully local, byte-equivalent to the
        // pre-CD build. No remote-search discovery is wired for the connector yet;
        // `search` stays `None` either way, which is behavior-identical to
        // the previous hardcoded search_remote=true (proof: the engine's
        // search_remote_without_discovery_is_ready_local test).
        let disagg_cfg = construction.runtime.config().disagg.clone();
        let remote = if cd::wiring_enabled(disagg_cfg.as_ref(), stack.handshake.as_ref()) {
            let disagg_cfg = disagg_cfg
                .as_ref()
                .expect("CD wiring gate requires a disagg config");
            let handshake = stack
                .handshake
                .as_ref()
                .expect("CD wiring gate requires a hub handshake");
            cd::wire_disagg(&self, construction, &stack, disagg_cfg, handshake).await?
        } else {
            kvbm_engine::RemoteOps::default()
        };
        let (engine, driver) = kvbm_engine::build_local_connector_engine(
            stack.instance_leader,
            sink,
            kvbm_engine::ConnectorEngineConfig { block_size, remote },
            stack.offload,
        );
        self.install_engine(engine)?;
        // Arm the forward-pass flush trigger — the driver is the other face of
        // the engine the install just swapped in; the glue only fires for
        // steps whose walk scheduled offloads.
        let _ = self.flush.set(FlushGlue {
            driver,
            runtime: construction.runtime.clone(),
            worker_ids,
            stranded: Arc::new(StrandMark::default()),
        });
        Ok(())
    }

    // ---- engine-seam lifecycle (durable core) ----

    /// USAA core: record the allocated G1 block ids and issue the engine
    /// onboard when vLLM commits to the promised external load.
    pub fn allocate(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> std::result::Result<(), Error> {
        self.state
            .lock()
            .allocate(request_id, block_ids, num_external_tokens)
    }

    /// Preemption hook: reset the slot's G1 ids and stage its fence.
    pub fn on_evicted(&self, request_id: &str) -> std::result::Result<EngineFinishedStatus, Error> {
        self.state.lock().on_evicted(request_id)
    }

    /// Drain pending evictions into the internal per-step payload.
    pub fn build_metadata(&self) -> ConnectorMetadata {
        self.state.lock().build_metadata()
    }
}

/// Scheduler-facing connector leader API used by the bindings and by
/// wrapper/composition leaders. The impl rides the `Arc<Leader>` handle (never
/// the bare struct) so a wrapper can hold a base leader behind this trait and
/// intercept only the methods it needs, such as GNMT and USAA.
pub trait ConnectorLeaderApi: Send + Sync {
    fn create_slot(&self, request: Request) -> Result<()>;

    fn has_slot(&self, request_id: &str) -> bool;

    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()>;

    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)>;

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()>;

    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata>;

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()>;

    fn request_finished(&self, request_id: &str) -> FinishedStatus;
}

impl ConnectorLeaderApi for Arc<Leader> {
    fn create_slot(&self, request: Request) -> Result<()> {
        self.as_ref().create_slot(request)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.as_ref().has_slot(request_id)
    }

    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        self.as_ref().extend_slot_tokens(request_id, tokens)
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        self.as_ref()
            .get_num_new_matched_tokens(request_id, num_computed_tokens)
    }

    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        Leader::update_state_after_alloc(self, request_id, block_ids, num_external_tokens)
    }

    fn build_connector_meta(&self, output: SchedulerOutput) -> Result<KvConnectorMetadata> {
        self.as_ref().build_connector_meta(output)
    }

    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        self.as_ref()
            .update_connector_output(finished_sending, finished_recving)
    }

    fn request_finished(&self, request_id: &str) -> FinishedStatus {
        self.as_ref().request_finished(request_id)
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, Leader};
    use crate::common::{CachedRequestData, NewRequestData, RequestMetadata, SchedulerOutput};
    use crate::common::{FinishedStatus, Request};
    use crate::connector::engine::noop_leader_engine;
    use kvbm_common::{BlockId, SequenceHash};
    use kvbm_protocols::connector::{ActionId, SearchId};
    use kvbm_protocols::connector::{
        ActionStatus, EvictionFence, EvictionOutcome, FenceHandle, FenceToken, FindBlocksHandle,
        FindBlocksOutcome, FindBlocksRequest, LeaderEngine, LeaderEngineError, OffloadHandle,
        OnboardHandle, RequestId, RequestOffloadDrain,
    };
    use kvbm_protocols::disagg::{RemotePrefillParams, SessionEndpoint, TransferParams};
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, Weak};

    /// Block size used in all connector leader tests.
    const BS: usize = 4;

    fn fresh_leader() -> Leader {
        Leader::with_engine(noop_leader_engine(), BS)
    }

    // --- engine install (deferred construction) --------------------------

    /// The binding path starts on the placeholder engine and swaps in the real
    /// one via `install_engine` on the still-empty leader. Verifies the swap
    /// re-routes GNMT, and that the install is single-shot (a second call on
    /// the still-empty leader errors and does not replace the live engine).
    #[test]
    fn install_engine_swaps_placeholder_and_is_single_shot() {
        let leader = Leader::with_engine(noop_leader_engine(), BS);

        // Install the real engine on the empty leader; GNMT then routes
        // through it (the recording engine resolves a 2-block hit).
        let engine = recording_engine(Refresh::Lost);
        leader.install_engine(engine.dyn_clone()).unwrap();

        // Single-shot: a second install (still pre-GNMT, so the empty-state
        // gate passes) is rejected by the OnceLock guard.
        let other = recording_engine(Refresh::Lost);
        assert!(leader.install_engine(other.dyn_clone()).is_err());

        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader
            .create_slot(make_request("r2", (0..16u32).collect()))
            .unwrap();
        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (Some(2 * BS), true)
        );
        assert_eq!(
            leader.get_num_new_matched_tokens("r2", 0).unwrap(),
            (Some(2 * BS), true)
        );
        assert_eq!(
            other.minted.lock().unwrap().len(),
            0,
            "the rejected install never took effect"
        );
        assert_eq!(
            engine.minted.lock().unwrap().len(),
            2,
            "the originally-installed engine serves both polls"
        );
    }

    /// Reproducer for the documented pre-init path (REFACTOR.md:525): a GNMT
    /// on the placeholder `NoopEngine` is benign (zero `Resolved` → no handle
    /// minted), leaving a handle-FREE slot. `install_engine` must still
    /// SUCCEED — that slot orphans nothing. (A guard keyed on slot existence
    /// rather than live handles would wedge the leader on Noop forever.)
    ///
    /// The pre-init poll runs on an EXISTING slot: the binding layer creates
    /// the slot from the vLLM Request at the top of every
    /// `get_num_new_matched_tokens` call, so a pre-init poll can never hit a
    /// missing slot — and if that ordering ever broke, the missing-slot
    /// error (`gnmt_missing_slot_errors`) is the correct loud answer on the
    /// placeholder too, leaving install unaffected (no handle either way).
    #[test]
    fn install_engine_succeeds_after_benign_pre_init_noop_gnmt() {
        let leader = Leader::with_engine(noop_leader_engine(), BS);
        // Pre-init GNMT on the placeholder: benign, no handle minted.
        leader
            .create_slot(make_request("r0", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r0", 0).unwrap();
        assert!(leader.has_slot("r0"));
        assert!(
            leader.state.lock().get("r0").unwrap().proposal.is_none(),
            "the placeholder minted no handle"
        );

        // Install must still succeed; the real engine then serves.
        let engine = recording_engine(Refresh::Lost);
        leader.install_engine(engine.dyn_clone()).unwrap();
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (Some(2 * BS), true)
        );
        assert_eq!(engine.minted.lock().unwrap().len(), 1);
    }

    /// Reproducer for the no-live-handles invariant: swapping the engine while a
    /// slot holds a live engine-minted handle would orphan that handle's RAII
    /// release, so `install_engine` must REFUSE. (This is the genuine orphan
    /// case; it is reachable only via the `with_engine` wiring path, since the
    /// binding path holds a non-minting placeholder until install.) The refusal
    /// has no side effect — the serving engine stays, the rejected one is unused.
    #[test]
    fn install_engine_rejected_while_a_slot_holds_a_live_handle() {
        // A leader already serving a real engine that parks a live lifecycle.
        let serving = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(serving.dyn_clone(), BS);
        leader
            .create_slot(make_request("r0", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r0", 0).unwrap();
        assert!(
            leader.state.lock().get("r0").unwrap().proposal.is_some(),
            "the serving engine minted a live handle"
        );

        let other = recording_engine(Refresh::Lost);
        let err = leader.install_engine(other.dyn_clone()).unwrap_err();
        assert!(
            err.to_string().contains("live handle"),
            "rejection names the live-handle violation, got: {err}"
        );

        // No side effect: the serving engine still routes; `other` never used.
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        assert_eq!(
            other.minted.lock().unwrap().len(),
            0,
            "the rejected engine was never installed"
        );
    }

    // ---- GNMT outcome mapping (the adapter contract) ---------------------

    /// Build a create-slot Request with the given tokens and no salt.
    fn make_request(req_id: &str, tokens: Vec<u32>) -> Request {
        Request::with_token_limits(req_id, tokens, None, None, None, None, None)
    }

    /// The first GNMT must not blow away the created slot's tokens: the slot
    /// map is entry-or-insert, never replace.
    #[test]
    fn gnmt_preserves_slot_tokens_from_create_slot() {
        let leader = fresh_leader();
        leader
            .create_slot(make_request("r1", vec![1u32, 2, 3]))
            .unwrap();
        assert_eq!(leader.get_slot_total_tokens("r1").unwrap(), 3);

        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        assert_eq!(
            leader.get_slot_total_tokens("r1").unwrap(),
            3,
            "the first GNMT must not blow away the created slot's tokens"
        );
    }

    /// The slot's chain cache: consecutive polls hand the engine the SAME
    /// `Arc` (pointer equality — a re-poll is a refcount bump, never a hash
    /// copy); a token extension invalidates the cache, so the next poll
    /// carries a REBUILT (longer) chain.
    #[test]
    fn gnmt_chain_arc_stable_across_polls_and_rebuilt_on_extend() {
        let engine = recording_engine(Refresh::Refined(2 * BS));
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..12u32).collect()))
            .unwrap();

        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        {
            let calls = engine.find_calls.lock().unwrap();
            assert_eq!(calls.len(), 2);
            assert!(
                std::sync::Arc::ptr_eq(&calls[0].0.sequence_hashes, &calls[1].0.sequence_hashes),
                "a pure re-poll shares the cached chain Arc"
            );
        }

        // Extending the sequence (a new complete block) invalidates the cache.
        leader
            .extend_slot_tokens("r1", (12..16u32).collect())
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        let calls = engine.find_calls.lock().unwrap();
        assert_eq!(calls.len(), 3);
        assert!(
            !std::sync::Arc::ptr_eq(&calls[1].0.sequence_hashes, &calls[2].0.sequence_hashes),
            "the extension rebuilt the chain"
        );
        assert_eq!(
            calls[2].0.sequence_hashes.len(),
            4,
            "the rebuilt chain carries the newly completed block"
        );
    }

    /// `Deferred` maps to `(None, false)` — vLLM parks the request in
    /// skipped-waiting and re-polls — with NOTHING parked and no slot side
    /// effect beyond the tracking mint.
    #[test]
    fn gnmt_deferred_maps_to_none_false_without_parking() {
        let engine = recording_engine(Refresh::Refined(2 * BS));
        *engine.defer_fresh.lock().unwrap() = true;
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..12u32).collect()))
            .unwrap();

        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (None, false),
            "a deferral parks the request for a re-poll"
        );
        assert!(
            leader.state.lock().get("r1").unwrap().proposal.is_none(),
            "a deferral parks no lifecycle"
        );
        assert!(
            engine.minted.lock().unwrap().is_empty(),
            "the engine minted nothing for a deferred poll"
        );
    }

    /// `Searching { minted: Some }` parks the fresh lifecycle and reports
    /// `(None, false)` so vLLM re-polls next step.
    #[test]
    fn gnmt_searching_parks_minted_handle_and_reports_none() {
        let engine = pending_search_engine();
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();

        let (matched, load_async) = leader.get_num_new_matched_tokens("r1", 0).unwrap();

        assert_eq!(matched, None, "still resolving — vLLM must re-poll");
        assert!(!load_async);
        assert!(
            leader.state.lock().get("r1").unwrap().proposal.is_some(),
            "the pending lifecycle stays parked for the re-poll"
        );
    }

    /// A `Searching` re-poll (live handle passed, no mint) keeps the parked
    /// handle — never a second park, never a release.
    #[test]
    fn gnmt_searching_repoll_keeps_parked_handle_without_second_mint() {
        let engine = recording_engine_with(Refresh::Pending, true);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();

        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (None, false)
        );
        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (None, false)
        );

        let calls = engine.find_calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert!(!calls[0].1, "first poll is a fresh mint");
        assert!(calls[1].1, "the re-poll hands the engine the live handle");
        assert_eq!(
            engine.minted.lock().unwrap().len(),
            1,
            "the re-poll never mints a second lifecycle"
        );
        assert!(
            leader.state.lock().get("r1").unwrap().proposal.is_some(),
            "the parked handle survives the re-poll"
        );
        assert!(engine.releases.lock().unwrap().is_empty());
    }

    /// A resolved miss (the Noop engine) reports a synchronous zero.
    #[test]
    fn gnmt_no_match_reports_zero_sync() {
        let leader = fresh_leader();
        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();

        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (Some(0), false)
        );
    }

    /// GNMT on a request that never saw `create_slot` is a runtime logic
    /// error, not a silent zero: the binding layer creates the slot from the
    /// vLLM Request before every poll, so a missing slot means that ordering
    /// broke and the answer would be derived from tokens never seen. The
    /// ghost tracking-slot mint is dead — the error parks nothing and mints
    /// nothing.
    #[test]
    fn gnmt_missing_slot_errors() {
        let leader = fresh_leader();
        let err = leader.get_num_new_matched_tokens("ghost", 0).unwrap_err();
        assert!(
            matches!(err.downcast_ref::<Error>(), Some(Error::SlotNotFound(_))),
            "expected SlotNotFound, got: {err}"
        );
        assert!(!leader.has_slot("ghost"), "no ghost slot is minted");
    }

    /// A resolved hit parks the minted lifecycle, maps token-granular
    /// `matched_tokens` to `(Some(n), true)`, and the request the engine saw
    /// carries the FULL chain + raw counts (no connector-side slicing — the
    /// engine derives the window).
    #[test]
    fn gnmt_hit_reports_external_tokens_async() {
        let engine = recording_engine(Refresh::Refined(2 * BS));
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..12u32).collect()))
            .unwrap();

        let (matched, load_async) = leader.get_num_new_matched_tokens("r1", 0).unwrap();

        assert_eq!(matched, Some(2 * BS), "token-granular engine answer");
        assert!(load_async, "a hit commits vLLM to the async external load");
        {
            let state = leader.state.lock();
            let slot = state.get("r1").unwrap();
            assert!(slot.proposal.is_some(), "the minted lifecycle parks");

            let calls = engine.find_calls.lock().unwrap();
            assert_eq!(calls.len(), 1);
            let (req, live) = &calls[0];
            assert!(!live, "first poll carries no live handle");
            assert_eq!(req.num_computed_tokens, 0, "raw count, not blocks");
            assert_eq!(req.total_tokens, 12, "raw count, not a derived window");
            assert_eq!(
                req.sequence_hashes.to_vec(),
                slot.all_sequence_hashes(),
                "FULL chain in absolute order, NOT pre-sliced to a window"
            );
            assert!(req.transfer_params.is_none());
        }
    }

    /// `release_parked` (zero-refine / Lost / empty window) drops the parked
    /// handle EXACTLY once; the next poll fresh-mints instead of refreshing a
    /// dead latch.
    #[test]
    fn gnmt_release_parked_drops_parked_handle_exactly_once() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..12u32).collect()))
            .unwrap();

        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (Some(2 * BS), true)
        );
        let minted = engine.minted.lock().unwrap().clone();
        assert!(engine.releases.lock().unwrap().is_empty());

        // Re-poll: the engine resolves zero with `release_parked` — the
        // connector executes the drop (RAII fires the engine release).
        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (Some(0), false)
        );
        assert_eq!(
            *engine.releases.lock().unwrap(),
            minted,
            "release_parked must drop the parked handle now"
        );
        assert!(
            leader.state.lock().get("r1").unwrap().proposal.is_none(),
            "slot cleared so a later poll fresh-mints"
        );

        // Next poll fresh-mints (live = None again).
        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (Some(2 * BS), true)
        );
        assert_eq!(engine.minted.lock().unwrap().len(), 2);

        drop(leader);
        assert_eq!(
            engine.releases.lock().unwrap().len(),
            2,
            "no double release on final drop"
        );
    }

    /// A refresh re-poll hands the engine the LIVE handle, resolves in place,
    /// and never re-parks or releases the handle it just refined.
    #[test]
    fn gnmt_refresh_passes_live_handle_and_never_reparks() {
        let engine = recording_engine(Refresh::Refined(3 * BS));
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();

        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (Some(2 * BS), true)
        );
        assert_eq!(
            leader.get_num_new_matched_tokens("r1", BS).unwrap(),
            (Some(3 * BS), true),
            "the refresh's refined answer serves vLLM"
        );

        let calls = engine.find_calls.lock().unwrap();
        assert!(calls[1].1, "the re-poll handed the engine the live handle");
        assert_eq!(
            engine.minted.lock().unwrap().len(),
            1,
            "a refresh never mints"
        );
        assert!(
            engine.releases.lock().unwrap().is_empty(),
            "the refresh must not release the lifecycle it just refined"
        );

        let minted = engine.minted.lock().unwrap().clone();
        drop(calls);
        drop(leader);
        assert_eq!(
            *engine.releases.lock().unwrap(),
            minted,
            "release fires once on final drop"
        );
    }

    /// The `release_parked` instruction is Issue-A gated like every other
    /// release site: with an onboard in flight (still reading the
    /// lifecycle-pinned source), the handle must stay parked — only the
    /// recv-side release (or reap) frees it once the load is terminal.
    #[test]
    fn gnmt_release_parked_gated_on_inflight_onboard() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..12u32).collect()))
            .unwrap();
        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (Some(2 * BS), true)
        );
        // USAA commits the load; the onboard parks (Pending).
        leader.allocate("r1", vec![10, 11, 12], 2 * BS).unwrap();

        // A re-poll resolving `release_parked` must not release the handle
        // the in-flight load is reading.
        assert_eq!(
            leader.get_num_new_matched_tokens("r1", 0).unwrap(),
            (Some(0), false)
        );
        {
            let state = leader.state.lock();
            let slot = state.get("r1").unwrap();
            assert!(
                slot.proposal.is_some(),
                "handle stays while the load reads it"
            );
            assert!(slot.onboard.is_some());
        }
        assert!(
            engine.releases.lock().unwrap().is_empty(),
            "no release while the onboard drains"
        );

        // Load terminal → the recv-side release frees both as usual.
        *engine.onboard_cells.lock().unwrap()[0].lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::from(["r1".to_string()]))
            .unwrap();
        let state = leader.state.lock();
        let slot = state.get("r1").unwrap();
        assert!(slot.proposal.is_none() && slot.onboard.is_none());
    }

    /// An engine refusal propagates loudly as `FindBlocksRejected` (prefill
    /// misconfiguration / digest divergence / lifecycle desync are engine
    /// errors now — the connector adds nothing).
    #[test]
    fn gnmt_error_propagates_as_find_blocks_rejected() {
        let engine = recording_engine(Refresh::Lost);
        *engine.reject_find.lock().unwrap() = true;
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..12u32).collect()))
            .unwrap();

        let err = leader.get_num_new_matched_tokens("r1", 0).unwrap_err();
        assert!(
            matches!(
                err.downcast_ref::<Error>(),
                Some(Error::FindBlocksRejected { .. })
            ),
            "expected FindBlocksRejected, got: {err}"
        );
    }

    // ---- remote prefill translation (request construction only) ----------

    /// Decode-minted params with a session endpoint, sized to `num_provided`
    /// committed tokens.
    fn prefill_params(num_provided_tokens: usize) -> RemotePrefillParams {
        let mut params =
            RemotePrefillParams::new(uuid::Uuid::new_v4(), uuid::Uuid::new_v4().into());
        params.num_provided_tokens = num_provided_tokens;
        params.decode_endpoint = Some(SessionEndpoint {
            kind: "velo".to_string(),
            payload: serde_json::Value::Null,
        });
        params
    }

    /// Build a create-slot Request whose `kv_transfer_params` carry the
    /// dispatcher wire shape (`serde_json::to_value(TransferParams)`).
    fn make_prefill_request(
        req_id: &str,
        tokens: Vec<u32>,
        params: &RemotePrefillParams,
    ) -> Request {
        let raw = serde_json::to_value(TransferParams::remote_prefill(params.clone())).unwrap();
        Request::with_token_limits(
            req_id,
            tokens,
            None,
            None,
            None,
            None,
            Some(RequestMetadata::with_kv_transfer_params(raw)),
        )
    }

    /// Slot creation parses the dispatcher wire shape into typed params.
    #[test]
    fn create_slot_parses_remote_prefill_transfer_params() {
        let leader = fresh_leader();
        let params = prefill_params(2 * BS);
        leader
            .create_slot(make_prefill_request("p1", (0..12u32).collect(), &params))
            .unwrap();

        let state = leader.state.lock();
        let slot = state.get("p1").unwrap();
        assert_eq!(slot.remote_prefill_params(), Some(&params));
        assert!(slot.proposal.is_none(), "no poll ran yet — nothing parked");
    }

    /// Malformed `kv_transfer_params` must not fail slot creation: the parse
    /// warns, yields `None`, and the request degrades to the plain local path
    /// (the engine sees `remote_prefill: None`).
    #[test]
    fn create_slot_tolerates_malformed_kv_transfer_params() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        let request = Request::with_token_limits(
            "p1",
            (0..12u32).collect::<Vec<u32>>(),
            None,
            None,
            None,
            None,
            Some(RequestMetadata::with_kv_transfer_params(
                serde_json::json!({
                    "remote_prefill": { "protocol_version": "not-a-number" }
                }),
            )),
        );

        leader.create_slot(request).unwrap();
        assert!(
            leader
                .state
                .lock()
                .get("p1")
                .unwrap()
                .transfer_params
                .is_none(),
            "malformed params parse to None"
        );

        let answer = leader.get_num_new_matched_tokens("p1", 0).unwrap();
        assert_eq!(answer, (Some(2 * BS), true), "plain local path serves it");
        let calls = engine.find_calls.lock().unwrap();
        assert!(
            calls[0].0.transfer_params.is_none(),
            "the engine sees a plain local request"
        );
    }

    /// GNMT forwards the slot's parsed `TransferParams` WHOLE as a request
    /// field — the connector never branches on them and never extracts the
    /// inner `remote_prefill` (the engine does both) — alongside the same
    /// full chain + raw counts the local path sends.
    #[test]
    fn gnmt_request_carries_whole_transfer_params() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        let params = prefill_params(2 * BS);
        leader
            .create_slot(make_prefill_request("p1", (0..12u32).collect(), &params))
            .unwrap();
        leader.get_num_new_matched_tokens("p1", BS).unwrap();

        let calls = engine.find_calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        let (req, _) = &calls[0];
        assert_eq!(
            req.transfer_params,
            Some(TransferParams::remote_prefill(params.clone())),
            "the WHOLE wire struct passes through verbatim"
        );
        let sent = req
            .transfer_params
            .as_ref()
            .and_then(|t| t.remote_prefill.as_ref())
            .expect("decode-minted params ride inside");
        assert_eq!(sent.session_id, params.session_id);
        assert_eq!(sent.decode_endpoint, params.decode_endpoint);
        assert_eq!(sent.num_provided_tokens, 2 * BS);
        assert_eq!(req.num_computed_tokens, BS, "this poll's computed count");
        assert_eq!(req.total_tokens, 12);
        assert_eq!(
            req.sequence_hashes.len(),
            3,
            "FULL unsliced chain even for a dispatched prefill"
        );
    }

    // --- allocate (USAA) ---------------------------------------------------

    #[test]
    fn allocate_unknown_request_errors() {
        let leader = fresh_leader();
        let err = leader.allocate("nope", vec![1], 0).unwrap_err();
        assert!(matches!(err, Error::SlotNotFound(_)));
    }

    /// USAA with committed external tokens issues `onboard_blocks` against
    /// the slot's parked lifecycle, passing the FULL allocated list as dest
    /// plus the committed count (the engine routes, slices, and validates),
    /// and parks the minted handle on the slot.
    #[test]
    fn usaa_external_tokens_issue_onboard_with_full_dest_set() {
        let engine = recording_engine(Refresh::Refined(2 * BS));
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        let (matched, load_async) = leader.get_num_new_matched_tokens("r1", 0).unwrap();
        assert_eq!((matched, load_async), (Some(2 * BS), true));

        leader
            .allocate("r1", vec![10, 11, 12, 13], matched.unwrap())
            .unwrap();

        {
            let onboards = engine.onboards.lock().unwrap();
            let minted = engine.minted.lock().unwrap();
            assert_eq!(onboards.len(), 1);
            assert_eq!(
                onboards[0].0, minted[0],
                "the onboard rode the lifecycle GNMT minted"
            );
            assert_eq!(
                onboards[0].1,
                vec![10, 11, 12, 13],
                "dest is the FULL allocated list, not a pre-sliced window"
            );
            assert_eq!(
                onboards[0].2,
                2 * BS,
                "the committed count crosses the seam for engine validation"
            );
        }
        let state = leader.state.lock();
        let slot = state.get("r1").unwrap();
        assert!(slot.onboard.is_some(), "slot holds the in-flight load");
        assert!(
            slot.proposal.is_some(),
            "the lifecycle stays parked while the load reads it"
        );
    }

    #[test]
    fn usaa_without_external_tokens_records_ids_only() {
        let engine = recording_engine(Refresh::Refined(2 * BS));
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();

        leader.allocate("r1", vec![10, 11], 0).unwrap();

        assert!(engine.onboards.lock().unwrap().is_empty());
        let state = leader.state.lock();
        let slot = state.get("r1").unwrap();
        assert!(slot.onboard.is_none());
        assert_eq!(slot.block_ids, vec![10, 11]);
    }

    /// A second committed external load while one is already in flight must be
    /// REJECTED at runtime: re-entering onboard would replace `slot.onboard`,
    /// detaching the leader's view of the live load. (The engine refuses
    /// re-entry per latched generation; this is the per-slot double-park
    /// guard.)
    #[test]
    fn usaa_second_external_load_while_in_flight_errors() {
        let engine = recording_engine(Refresh::Refined(2 * BS));
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.allocate("r1", vec![10, 11, 12, 13], 2 * BS).unwrap();

        let err = leader
            .allocate("r1", vec![10, 11, 12, 13], 2 * BS)
            .unwrap_err();
        assert!(matches!(err, Error::OnboardAlreadyInFlight(_)));
        assert_eq!(
            engine.onboards.lock().unwrap().len(),
            1,
            "no second engine onboard is issued"
        );
        let state = leader.state.lock();
        let slot = state.get("r1").unwrap();
        assert!(
            slot.onboard.is_some(),
            "the original in-flight handle stays parked"
        );
    }

    /// External tokens committed for a slot parking no lifecycle (the Noop
    /// poll resolved zero without minting) is a broken GNMT↔USAA contract —
    /// USAA must fail loud, not leave the request waiting on a load nobody
    /// started.
    #[test]
    fn usaa_external_tokens_without_parked_lifecycle_errors() {
        let leader = fresh_leader();
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();

        let err = leader.allocate("r1", vec![10, 11], 2 * BS).unwrap_err();
        assert!(matches!(err, Error::ExternalLoadWithoutSearch(_)));
    }

    /// An engine refusal at onboard time (stale pin, external-token mismatch —
    /// all engine-validated now) propagates as `OnboardRejected` and parks
    /// nothing.
    #[test]
    fn usaa_engine_rejection_propagates_as_onboard_rejected() {
        let engine = recording_engine(Refresh::Refined(2 * BS));
        *engine.reject_onboard.lock().unwrap() = true;
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..12u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();

        let err = leader.allocate("r1", vec![10, 11], BS).unwrap_err();
        assert!(matches!(
            err,
            Error::OnboardRejected {
                source: LeaderEngineError::ExternalTokensMismatch { .. },
                ..
            }
        ));
        let state = leader.state.lock();
        let slot = state.get("r1").unwrap();
        assert!(slot.onboard.is_none(), "a refused onboard parks nothing");
        assert!(
            slot.proposal.is_some(),
            "the parked lifecycle survives the refusal (vLLM may retry)"
        );
    }

    /// `finished_recving` is cadence, the onboard handle is authority: the
    /// report alone (load still pending) releases nothing; once the handle is
    /// terminal the report drops both the onboard and the parked lifecycle —
    /// the engine consumed the pinned state at onboard time and the terminal
    /// load no longer reads the source — while the slot itself lives on (the
    /// request is still decoding).
    #[test]
    fn finished_recving_releases_terminal_onboard_and_lifecycle() {
        let engine = recording_engine(Refresh::Refined(2 * BS));
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.allocate("r1", vec![10, 11, 12, 13], 2 * BS).unwrap();

        // Cadence without authority: the load is still pending.
        leader
            .update_connector_output(HashSet::new(), HashSet::from(["r1".to_string()]))
            .unwrap();
        {
            let state = leader.state.lock();
            let slot = state.get("r1").unwrap();
            assert!(
                slot.onboard.is_some(),
                "a pending load must not be detached by the report alone"
            );
            assert!(slot.proposal.is_some());
        }
        assert!(engine.releases.lock().unwrap().is_empty());

        // Load terminal + cadence: both handles drop, slot survives.
        *engine.onboard_cells.lock().unwrap()[0].lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::from(["r1".to_string()]))
            .unwrap();
        {
            let state = leader.state.lock();
            let slot = state.get("r1").unwrap();
            assert!(slot.onboard.is_none());
            assert!(slot.proposal.is_none(), "lifecycle released with the load");
        }
        assert_eq!(
            *engine.releases.lock().unwrap(),
            vec![engine.minted.lock().unwrap()[0]],
            "exactly the consumed lifecycle is released"
        );
        assert!(
            leader.has_slot("r1"),
            "request still decoding — only the load bookkeeping clears"
        );
    }

    // --- eviction --------------------------------------------------------

    #[test]
    fn on_evicted_clears_block_ids_and_save_cursor() {
        let leader = fresh_leader();
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.allocate("r1", vec![10, 11, 12], 0).unwrap();
        // Force a non-zero cursor to verify reset.
        {
            let mut state = leader.state.lock();
            state.get_mut("r1").unwrap().evaluated_tokens = 2;
        }

        leader.on_evicted("r1").unwrap();

        let state = leader.state.lock();
        let slot = state.get("r1").unwrap();
        assert!(slot.block_ids.is_empty());
        assert_eq!(slot.evaluated_tokens, 0);
    }

    /// Eviction is non-terminal: the slot survives and is not in any
    /// `finished_sending` set, so `update_connector_output` does not reap it.
    #[test]
    fn on_evicted_does_not_reap_slot() {
        let leader = fresh_leader();
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.on_evicted("r1").unwrap();

        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(leader.has_slot("r1"));
    }

    #[test]
    fn evict_then_restore_then_finish_lifecycle() {
        // Recording engine so the restore exercises the visible-reset → fresh-GNMT
        // transition: a fresh poll MINTS a handle, so the second poll minting again
        // (rather than refreshing the old lifecycle) is observable.
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.allocate("r1", vec![10, 11, 12], 0).unwrap();
        assert_eq!(engine.minted.lock().unwrap().len(), 1);

        leader.on_evicted("r1").unwrap();
        assert!(leader.has_slot("r1"));

        // Restore reuses the same req id. Because eviction reset the slot's
        // *visible* lifecycle to `None`, this poll takes the fresh-mint arm
        // (a second mint), NOT the refresh arm — the structural fresh-GNMT claim.
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.allocate("r1", vec![20, 21, 22], 0).unwrap();
        assert_eq!(
            leader.state.lock().get("r1").unwrap().block_ids,
            vec![20, 21, 22]
        );
        assert_eq!(
            engine.minted.lock().unwrap().len(),
            2,
            "restore must re-enter GNMT fresh (mint), not refresh the evicted lifecycle"
        );

        assert_eq!(leader.request_finished("r1"), FinishedStatus::Finished);
        assert!(!leader.has_slot("r1"));
    }

    // --- metadata --------------------------------------------------------

    #[test]
    fn build_metadata_advances_iteration_and_drains_evictions() {
        let leader = fresh_leader();
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.on_evicted("r1").unwrap();

        let m1 = leader.build_metadata();
        assert_eq!(m1.iteration, 1);
        assert_eq!(m1.evicted_requests, vec!["r1".to_string()]);

        let m2 = leader.build_metadata();
        assert_eq!(m2.iteration, 2);
        assert!(m2.evicted_requests.is_empty());
    }

    #[test]
    fn duplicate_on_evicted_collapses_to_one_entry_in_metadata() {
        let leader = fresh_leader();
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.on_evicted("r1").unwrap();
        leader.on_evicted("r1").unwrap();

        let m = leader.build_metadata();
        assert_eq!(m.evicted_requests, vec!["r1".to_string()]);
    }

    #[test]
    fn duplicate_on_evicted_calls_engine_evict_at_most_once_per_cycle() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();

        // First cycle: two on_evicted calls ⇒ one engine publish.
        leader.on_evicted("r1").unwrap();
        leader.on_evicted("r1").unwrap();
        assert_eq!(
            engine.evict_calls.lock().unwrap().len(),
            1,
            "duplicate on_evicted within one cycle must publish at most once"
        );

        // Drain into metadata — next cycle starts.
        let _ = leader.build_metadata();

        // Second cycle: another two ⇒ one more publish.
        leader.on_evicted("r1").unwrap();
        leader.on_evicted("r1").unwrap();
        assert_eq!(
            engine.evict_calls.lock().unwrap().len(),
            2,
            "next cycle re-enables one engine publish"
        );
    }

    /// Same-cycle evict → restore → re-evict: the restored generation re-arms
    /// live handles (fresh lifecycle + in-flight onboard), and the second
    /// eviction must mint its OWN fence to capture them — the per-cycle dedup
    /// alone would detach the new views with no fence, letting their in-flight
    /// G1 access escape the worker barrier. The wire's `evicted_requests`
    /// list still dedups to one entry.
    #[test]
    fn same_cycle_re_eviction_with_live_handles_mints_second_fence() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();

        // Gen 1: parked lifecycle, then evict (fence #1; lifecycle → holder).
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.on_evicted("r1").unwrap();
        assert_eq!(engine.evict_calls.lock().unwrap().len(), 1);

        // Same-cycle restore: fresh mint + committed external load park
        // live handles back on the slot.
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.allocate("r1", vec![10, 11, 12, 13], 2 * BS).unwrap();
        {
            let state = leader.state.lock();
            let slot = state.get("r1").unwrap();
            assert!(slot.proposal.is_some() && slot.onboard.is_some());
        }

        // Same-cycle re-eviction: the new live handles need their own capture.
        leader.on_evicted("r1").unwrap();
        assert_eq!(
            engine.evict_calls.lock().unwrap().len(),
            2,
            "re-eviction with live handles must fence the restored generation"
        );

        let meta = leader.build_metadata();
        assert_eq!(
            meta.fences.len(),
            2,
            "both generations' fences ride the wire"
        );
        assert_eq!(
            meta.evicted_requests,
            vec!["r1".to_string()],
            "the evicted-request list still dedups per cycle"
        );
    }

    /// Reproducer (P-C3, the A×E source-pin hazard): evicting a slot with a live
    /// onboard MOVES `proposal`+`onboard` into the drain-holder instead of
    /// dropping the lifecycle mid-drain. The visible slot resets to fresh-GNMT,
    /// `engine.evict` fires once and its `FenceToken` lands in `build_metadata`,
    /// and the held lifecycle is released ONLY after the deferred sweep observes
    /// the onboard terminal — never synchronously on eviction.
    #[test]
    fn evict_with_live_onboard_moves_to_drain_holder_and_defers_release() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        // Fresh poll parks a lifecycle; record the minted generation.
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        let minted = engine.minted.lock().unwrap().clone();
        assert_eq!(minted.len(), 1, "the fresh poll minted exactly one pin");

        // Stage an in-flight (Pending) onboard reading the pinned source.
        let (onboard, cell) = engine.arm_onboard();
        leader.state.lock().get_mut("r1").unwrap().onboard = Some(onboard);

        // Evict: proposal + onboard MOVE into the drain-holder; the visible
        // slot resets so the next poll runs GNMT fresh.
        leader.on_evicted("r1").unwrap();
        {
            let state = leader.state.lock();
            let slot = state.get("r1").unwrap();
            assert_eq!(
                slot.drain_holders.len(),
                1,
                "eviction moves the parked lifecycle into the drain-holder"
            );
            assert!(
                slot.drain_holder_draining(),
                "the in-flight onboard rides along to drain"
            );
            assert!(
                slot.proposal.is_none(),
                "visible lifecycle reset for fresh GNMT"
            );
            assert!(
                slot.onboard.is_none(),
                "visible onboard reset for fresh GNMT"
            );
        }

        // engine.evict fired exactly once; the pin must NOT release yet.
        assert_eq!(engine.evict_calls.lock().unwrap().len(), 1);
        assert!(
            engine.releases.lock().unwrap().is_empty(),
            "the held lifecycle must not release while the onboard drains"
        );

        // The fence (with its per-worker FenceToken) rides this cycle's metadata.
        let meta = leader.build_metadata();
        assert_eq!(
            meta.fences.len(),
            1,
            "the eviction fence rides the metadata"
        );
        assert_eq!(meta.fences[0].request_id, "r1");
        assert_eq!(
            meta.fences[0].per_worker.len(),
            1,
            "the fence carries one per-(generation, worker) FenceToken"
        );

        // Worker load terminal: flip the held onboard cell, then run the deferred
        // sweep. The drain-holder drops → the RAII release frees the pin.
        *cell.lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();

        assert_eq!(
            *engine.releases.lock().unwrap(),
            minted,
            "the release fires once the onboard drains and the holder drops"
        );
        assert!(
            leader
                .state
                .lock()
                .get("r1")
                .unwrap()
                .drain_holders
                .is_empty(),
            "drain-holder cleared after the onboard reached terminal"
        );
    }

    /// The uniform, kind-blind holder push through the PRODUCTION paths: GNMT
    /// parks, USAA commits the load, eviction pushes (proposal, onboard) into
    /// the holder; a restored generation evicted again ACCUMULATES a second
    /// holder (one push per eviction generation — the A×E same-request shape),
    /// and each entry releases only when its own onboard drains.
    #[test]
    fn on_evicted_uniform_proposal_holder_push_accumulates_per_generation() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();

        // Gen 1: hit + committed load, then evict.
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.allocate("r1", vec![10, 11, 12, 13], 2 * BS).unwrap();
        leader.on_evicted("r1").unwrap();
        {
            let state = leader.state.lock();
            let slot = state.get("r1").unwrap();
            assert_eq!(slot.drain_holders.len(), 1, "one push per eviction");
            assert!(slot.drain_holder_draining(), "the holder holds the load");
            assert!(slot.proposal.is_none() && slot.onboard.is_none());
        }
        assert!(
            engine.releases.lock().unwrap().is_empty(),
            "the pin drains, it never drops at evict"
        );

        // Gen 2 restore: fresh mint, then a second eviction while gen 1 still
        // drains — the holders ACCUMULATE (overwrite would release gen 1
        // mid-read).
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.on_evicted("r1").unwrap();
        {
            let state = leader.state.lock();
            let slot = state.get("r1").unwrap();
            assert_eq!(
                slot.drain_holders.len(),
                2,
                "the second eviction pushes its own holder"
            );
        }
        assert!(engine.releases.lock().unwrap().is_empty());

        // Gen 1's load drains: the sweep drops BOTH safe holders (gen 2's has
        // no onboard) and both generations release.
        *engine.onboard_cells.lock().unwrap()[0].lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        let minted = engine.minted.lock().unwrap().clone();
        assert_eq!(
            engine.releases.lock().unwrap().len(),
            2,
            "each holder released its own generation"
        );
        for id in minted {
            assert!(engine.releases.lock().unwrap().contains(&id));
        }
        assert!(
            leader
                .state
                .lock()
                .get("r1")
                .unwrap()
                .drain_holders
                .is_empty()
        );
    }

    /// Reproducer (fence-gated holder release): a slot evicted with a parked
    /// proposal AND a still-pending offload. The eviction's view-detach drops
    /// the offload handle, so the holder carries no onboard — the old
    /// onboard-only rule released it at the FIRST sweep, freeing the held pin
    /// while the engine was still draining the fenced offload. The new rule
    /// keys the release on the eviction's leader-side fence handle: held until
    /// `is_complete()`, released after.
    #[test]
    fn evicted_holder_with_pending_offload_releases_on_fence_not_first_sweep() {
        let engine = recording_engine(Refresh::Lost);
        *engine.mint_fence_handles.lock().unwrap() = true;
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        // A still-Pending offload reading G1 at eviction time.
        let (offload, _offload_cell) = engine.arm_offload(&"r1".to_string());
        leader
            .state
            .lock()
            .get_mut("r1")
            .unwrap()
            .offloads
            .push(offload);

        leader.on_evicted("r1").unwrap();
        {
            let state = leader.state.lock();
            let slot = state.get("r1").unwrap();
            assert_eq!(slot.drain_holders.len(), 1);
            assert!(
                slot.drain_holder_draining(),
                "the fence is pending — the holder must report draining"
            );
        }

        // First sweep with the engine fence still pending: the holder (and
        // its lifecycle pin) must survive — this is the pre-fix release point.
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(
            engine.releases.lock().unwrap().is_empty(),
            "holder must hold until the engine fence completes, not the first sweep"
        );
        assert_eq!(
            leader.state.lock().get("r1").unwrap().drain_holders.len(),
            1
        );

        // Engine barrier drops (the fenced offload drained): the next sweep
        // releases the holder and the pin.
        engine.fence_cells.lock().unwrap()[0].store(true, Ordering::Release);
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert_eq!(
            engine.releases.lock().unwrap().len(),
            1,
            "fence completion releases the held lifecycle"
        );
        assert!(
            leader
                .state
                .lock()
                .get("r1")
                .unwrap()
                .drain_holders
                .is_empty()
        );
    }

    /// Preempt → restore → re-evict while the first generation still drains:
    /// TWO drain-holder entries, each carrying its OWN engine fence. The
    /// release is keyed per holder — gen 1's barrier dropping frees gen 1
    /// alone; gen 2 holds until its own fence completes. (The fence-less
    /// accumulation double covers the fallback arm; this is the production
    /// shape, where a real engine always returns a handle when work is armed.)
    #[test]
    fn restored_re_eviction_holders_release_each_on_their_own_fence() {
        let engine = recording_engine(Refresh::Lost);
        *engine.mint_fence_handles.lock().unwrap() = true;
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();

        // Gen 1: hit + committed load, then evict — holder 1 rides fence 1.
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.allocate("r1", vec![10, 11, 12, 13], 2 * BS).unwrap();
        leader.on_evicted("r1").unwrap();

        // Gen 2 restore: fresh mint, re-evicted while gen 1 still drains —
        // holder 2 rides fence 2.
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.on_evicted("r1").unwrap();
        assert_eq!(
            leader.state.lock().get("r1").unwrap().drain_holders.len(),
            2
        );
        assert_eq!(engine.fence_cells.lock().unwrap().len(), 2);

        // Both fences pending: the sweep releases nothing.
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(engine.releases.lock().unwrap().is_empty());

        // Gen 1's barrier drops: ONLY gen 1's holder releases.
        engine.fence_cells.lock().unwrap()[0].store(true, Ordering::Release);
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        {
            let minted = engine.minted.lock().unwrap().clone();
            let releases = engine.releases.lock().unwrap().clone();
            assert_eq!(
                releases.len(),
                1,
                "gen 1 frees on its own fence; gen 2 must stay held"
            );
            assert_eq!(releases[0], minted[0]);
        }
        assert_eq!(
            leader.state.lock().get("r1").unwrap().drain_holders.len(),
            1
        );

        // Gen 2's barrier drops: the remaining holder releases.
        engine.fence_cells.lock().unwrap()[1].store(true, Ordering::Release);
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert_eq!(engine.releases.lock().unwrap().len(), 2);
        assert!(
            leader
                .state
                .lock()
                .get("r1")
                .unwrap()
                .drain_holders
                .is_empty()
        );
    }

    /// An eviction whose engine fence is EMPTY (nothing was in flight) stays
    /// off the wire — the workers would await zero tokens — while
    /// `evicted_requests` still records the rid. The noop engine is exactly
    /// that shape.
    #[test]
    fn empty_fence_is_not_embedded_in_metadata() {
        let leader = fresh_leader();
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        leader.on_evicted("r1").unwrap();

        let meta = leader.build_metadata();
        assert_eq!(
            meta.evicted_requests,
            vec!["r1".to_string()],
            "the rid still rides the wire"
        );
        assert!(
            meta.fences.is_empty(),
            "an empty fence is wire noise and must not be embedded"
        );
    }

    /// Holderless eviction (in-flight offload, NO parked proposal): there is no
    /// drain-holder entry to carry the leader's fence handle, so it lands in
    /// the slot-level `fence_holders` and is swept once the fence completes.
    #[test]
    fn holderless_eviction_parks_fence_handle_until_complete() {
        let engine = recording_engine(Refresh::Lost);
        *engine.mint_fence_handles.lock().unwrap() = true;
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        // No GNMT poll — nothing parked; only an in-flight offload.
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        let (offload, _cell) = engine.arm_offload(&"r1".to_string());
        leader
            .state
            .lock()
            .get_mut("r1")
            .unwrap()
            .offloads
            .push(offload);

        leader.on_evicted("r1").unwrap();
        {
            let state = leader.state.lock();
            let slot = state.get("r1").unwrap();
            assert!(slot.drain_holders.is_empty(), "no proposal — no holder");
            assert_eq!(
                slot.fence_holders.len(),
                1,
                "the leader keeps its observational view of the drain"
            );
        }

        // Pending fence survives the sweep; a completed one is dropped.
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert_eq!(
            leader.state.lock().get("r1").unwrap().fence_holders.len(),
            1
        );

        engine.fence_cells.lock().unwrap()[0].store(true, Ordering::Release);
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(
            leader
                .state
                .lock()
                .get("r1")
                .unwrap()
                .fence_holders
                .is_empty(),
            "completed fence swept"
        );
    }

    // --- preemption (scheduler-output fan-in) -----------------------------

    /// A preempted rid on the scheduler output is evicted BEFORE the walk, so
    /// its fence lands in the SAME step's metadata payload.
    #[test]
    fn preempted_req_ids_evict_and_stage_fences_for_this_step() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();

        let mut out = SchedulerOutput::new(7);
        out.preempted_req_ids = vec!["r1".to_string()];
        // An all-preemption frame schedules nothing; the preemption must be
        // processed anyway.
        leader.build_connector_meta(out).unwrap();
        assert_eq!(
            engine.evict_calls.lock().unwrap().as_slice(),
            ["r1".to_string()],
            "the preempted rid reached engine.evict"
        );

        let meta = leader.build_metadata();
        assert_eq!(meta.evicted_requests, vec!["r1".to_string()]);
        assert_eq!(
            meta.fences.len(),
            1,
            "the preemption fence rides this step's payload"
        );
        assert_eq!(meta.fences[0].request_id, "r1");
    }

    /// vLLM can preempt a request the connector finished tracking; an unknown
    /// rid warns and is skipped — the step (and its known preemptions) proceed.
    #[test]
    fn preempted_unknown_rid_warns_and_continues() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();

        let mut out = SchedulerOutput::new(7);
        out.preempted_req_ids = vec!["ghost".to_string(), "r1".to_string()];
        leader.build_connector_meta(out).unwrap();

        let meta = leader.build_metadata();
        assert_eq!(
            meta.evicted_requests,
            vec!["r1".to_string()],
            "the unknown rid is skipped, the known one proceeds"
        );
        assert_eq!(
            engine.evict_calls.lock().unwrap().as_slice(),
            ["r1".to_string()],
            "no engine publish for the unknown rid"
        );
    }

    // --- finish / reap ---------------------------------------------------

    /// No in-flight offload, so `request_finished` reaps the slot inline — no
    /// poll, no `update_connector_output` needed.
    #[test]
    fn request_finished_finished_reaps_slot_inline() {
        let leader = fresh_leader();
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();

        assert_eq!(leader.request_finished("r1"), FinishedStatus::Finished);
        assert!(!leader.has_slot("r1"));
    }

    /// An unslotted finish is benign: `UntrackedRequest` (the PyO3 layer maps
    /// it to the same non-raising delay flag as `Finished`). Returning an
    /// error here would crash EngineCore on a routine WAITING-abort.
    #[test]
    fn request_finished_unknown_request_is_untracked() {
        let leader = fresh_leader();
        assert_eq!(
            leader.request_finished("never-polled"),
            FinishedStatus::UntrackedRequest
        );
        assert!(!leader.has_slot("never-polled"));
    }

    /// The inline `Finished` reap drops the slot — and with it the parked
    /// lifecycle handle, whose RAII drop fires the generation-bound engine
    /// release.
    #[test]
    fn request_finished_inline_reap_drops_proposal() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..12u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        let minted = engine.minted.lock().unwrap().clone();
        assert_eq!(minted.len(), 1);

        assert_eq!(leader.request_finished("r1"), FinishedStatus::Finished);
        assert!(!leader.has_slot("r1"));
        assert_eq!(
            *engine.releases.lock().unwrap(),
            minted,
            "the inline reap fired the parked lifecycle's RAII release"
        );
    }

    /// The deferred (handle-gated) sweep reap drops the parked handle too: a
    /// finish with the load still in flight keeps the slot until the terminal,
    /// then releases the lifecycle.
    #[test]
    fn finishing_sweep_reap_drops_proposal() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..12u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        let minted = engine.minted.lock().unwrap().clone();
        leader.allocate("r1", vec![10, 11], 2 * BS).unwrap();

        assert_eq!(leader.request_finished("r1"), FinishedStatus::Pending);
        assert!(
            engine.releases.lock().unwrap().is_empty(),
            "the lifecycle holds while the load drains"
        );

        *engine.onboard_cells.lock().unwrap()[0].lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(!leader.has_slot("r1"));
        assert_eq!(*engine.releases.lock().unwrap(), minted);
    }

    /// Offload-in-flight finish (REFACTOR.md §4 active-offloading case, D
    /// semantics): the slot holds a non-terminal `offload`, so
    /// `request_finished` returns `Pending`, keeps the slot, and hands the
    /// engine the coordination IN-CALL — `take_offload_drain` + `commit`
    /// (commit arms the engine's emit-on-last-terminal; the connector never
    /// re-polls to decide when to emit). The deferred sweep in
    /// `update_connector_output` then reaps the slot once its handles are
    /// terminal — gated on the HANDLES, not on the vLLM sets (which are only
    /// the wake-up cadence).
    #[test]
    fn offload_in_flight_finish_commits_drain_in_call_and_reaps_on_terminal() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();

        // Stage an in-flight (Pending) offload on the slot.
        let (handle, cell) = engine.arm_offload(&"r1".to_string());
        leader
            .state
            .lock()
            .get_mut("r1")
            .unwrap()
            .offloads
            .push(handle);

        // A non-terminal offload defers the finish: slot kept, status Pending,
        // and the drain is committed IN-CALL (D: the engine owns the terminal
        // coordination from here; commit = arm, not emit).
        assert_eq!(leader.request_finished("r1"), FinishedStatus::Pending);
        assert!(leader.has_slot("r1"));
        assert_eq!(
            engine.commits.lock().unwrap().as_slice(),
            ["r1".to_string()],
            "request_finished(Pending) must commit the drain in-call"
        );

        // The sweep does NOT reap while the offload is still pending — even if
        // vLLM's sets name the request (handles are the gate).
        leader
            .update_connector_output(HashSet::from(["r1".to_string()]), HashSet::new())
            .unwrap();
        assert!(leader.has_slot("r1"), "non-terminal handles block the reap");

        // Worker save terminal: flip the offload cell; the next sweep reaps —
        // with EMPTY vLLM sets (the sets are cadence, not the gate).
        *cell.lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(!leader.has_slot("r1"), "the Pending slot is reaped");

        // The drain committed exactly once (consume-once); later sweeps are
        // no-ops.
        leader
            .update_connector_output(HashSet::from(["r1".to_string()]), HashSet::new())
            .unwrap();
        assert_eq!(
            engine.commits.lock().unwrap().len(),
            1,
            "take_offload_drain is consume-once: exactly one commit total"
        );
    }

    /// The mixed {onboard=Pending, offload=Pending} cell: a single slot carries
    /// BOTH a visible in-flight onboard AND a visible in-flight offload at finish
    /// time. The reap gate is an AND over onboard-terminal && all-offloads-terminal;
    /// flipping ONE handle terminal must NOT reap while the other is still pending.
    /// The drain is committed once, in-call, and consumed once across the mixed
    /// terminal ordering (onboard terminal first, offload last).
    #[test]
    fn both_onboard_and_offload_in_flight_finish_commits_drain_once_and_reaps_only_when_both_terminal()
     {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();

        // Stage BOTH an in-flight onboard and an in-flight offload on the slot.
        let (onboard, onboard_cell) = engine.arm_onboard();
        let (offload, offload_cell) = engine.arm_offload(&"r1".to_string());
        {
            let mut state = leader.state.lock();
            let slot = state.get_mut("r1").unwrap();
            slot.onboard = Some(onboard);
            slot.offloads.push(offload);
        }

        // A non-terminal handle defers the finish: slot kept, status Pending,
        // drain committed IN-CALL exactly once.
        assert_eq!(leader.request_finished("r1"), FinishedStatus::Pending);
        assert_eq!(
            engine.commits.lock().unwrap().as_slice(),
            ["r1".to_string()],
            "request_finished(Pending) must commit the drain in-call"
        );
        assert!(leader.has_slot("r1"));

        // Both handles pending: the sweep keeps the slot.
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(
            leader.has_slot("r1"),
            "both handles pending blocks the reap"
        );

        // Flip ONLY the onboard terminal: the offload is still pending, so the
        // AND clause holds the slot.
        *onboard_cell.lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(
            leader.has_slot("r1"),
            "onboard terminal but offload pending must NOT reap"
        );

        // Flip the offload terminal too: now both handles are terminal and the
        // sweep reaps.
        *offload_cell.lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(
            !leader.has_slot("r1"),
            "reap once BOTH handles are terminal"
        );

        // The drain committed exactly once across the mixed terminal ordering.
        leader
            .update_connector_output(HashSet::from(["r1".to_string()]), HashSet::new())
            .unwrap();
        assert_eq!(
            engine.commits.lock().unwrap().len(),
            1,
            "take_offload_drain is consume-once: exactly one commit total"
        );
        assert!(
            engine.take_offload_drain(&"r1".to_string()).is_none(),
            "the drain was already consumed by the finish path"
        );
    }

    /// Codex finding (the finish-during-drain A×E twin): finishing an EVICTED
    /// request whose drain-holder still holds a non-terminal onboard must NOT
    /// reap inline — the visible handles are empty (`pending == false`), but
    /// dropping the slot drops the holder, releasing the lifecycle pin the old
    /// onboard is still READING. The directive answer is still `Finished`
    /// (nothing pending for vLLM); the reap defers to the handle-gated sweep.
    #[test]
    fn finish_after_evict_keeps_slot_until_drain_holder_drains() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        let minted = engine.minted.lock().unwrap().clone();
        let (onboard, cell) = engine.arm_onboard();
        leader.state.lock().get_mut("r1").unwrap().onboard = Some(onboard);

        // Evict: proposal + in-flight onboard move into the drain-holder.
        leader.on_evicted("r1").unwrap();

        // Finish while the evicted onboard still drains.
        assert_eq!(leader.request_finished("r1"), FinishedStatus::Finished);
        assert!(
            leader.has_slot("r1"),
            "slot must survive while the drain-holder's onboard drains"
        );
        assert!(
            engine.releases.lock().unwrap().is_empty(),
            "the held lifecycle must not release while the old onboard reads it"
        );

        // Sweep with the onboard still pending: no reap, no release.
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(leader.has_slot("r1"));
        assert!(engine.releases.lock().unwrap().is_empty());

        // The old onboard drains: the sweep reaps and the pin releases.
        *cell.lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(!leader.has_slot("r1"));
        assert_eq!(*engine.releases.lock().unwrap(), minted);
    }

    /// The sweep-side twin: an evicted request is RESTORED, then finishes with
    /// new-generation work pending. Once the new work drains, the sweep must
    /// still wait for the OLD generation's drain-holder before reaping.
    #[test]
    fn finishing_sweep_waits_for_drain_holder_of_restored_request() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        // Old generation: parked lifecycle + in-flight onboard, then evict.
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        let old_pin = engine.minted.lock().unwrap()[0];
        let (onboard, onboard_cell) = engine.arm_onboard();
        leader.state.lock().get_mut("r1").unwrap().onboard = Some(onboard);
        leader.on_evicted("r1").unwrap();

        // Restore (fresh GNMT mint), then finish with a pending offload.
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        let (offload, offload_cell) = engine.arm_offload(&"r1".to_string());
        leader
            .state
            .lock()
            .get_mut("r1")
            .unwrap()
            .offloads
            .push(offload);
        assert_eq!(leader.request_finished("r1"), FinishedStatus::Pending);

        // New-generation offload drains; the OLD onboard has not. The sweep
        // must keep the slot — reaping would drop the holder and release the
        // old pin mid-drain.
        *offload_cell.lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(
            leader.has_slot("r1"),
            "sweep must wait for the old generation's drain-holder"
        );
        assert!(
            !engine.releases.lock().unwrap().contains(&old_pin),
            "the old pin must not release while its onboard drains"
        );

        // Old onboard drains too: now the sweep reaps and both pins release.
        *onboard_cell.lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(!leader.has_slot("r1"));
        assert!(engine.releases.lock().unwrap().contains(&old_pin));
    }

    /// Codex finding #2 (double-eviction): a restored request evicted AGAIN
    /// while the older generation's drain-holder still drains must PRESERVE
    /// that holder — overwriting it drops the old pin mid-read. The holders
    /// accumulate per eviction generation and each releases only when its own
    /// onboard drains.
    #[test]
    fn second_eviction_preserves_older_draining_holder() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        // Gen 1: parked lifecycle + in-flight onboard, evict.
        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        let gen1_pin = engine.minted.lock().unwrap()[0];
        let (onboard1, cell1) = engine.arm_onboard();
        leader.state.lock().get_mut("r1").unwrap().onboard = Some(onboard1);
        leader.on_evicted("r1").unwrap();

        // Restore (gen 2 mints a new pin) + another in-flight onboard.
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        let gen2_pin = engine.minted.lock().unwrap()[1];
        let (onboard2, cell2) = engine.arm_onboard();
        leader.state.lock().get_mut("r1").unwrap().onboard = Some(onboard2);

        // Second eviction while gen 1's onboard still drains.
        leader.on_evicted("r1").unwrap();
        assert!(
            engine.releases.lock().unwrap().is_empty(),
            "the second eviction must not drop the gen-1 holder mid-drain"
        );

        // Gen 2 drains first: gen 1's pin must STILL hold.
        *cell2.lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(
            !engine.releases.lock().unwrap().contains(&gen1_pin),
            "gen-1 pin must hold until its own onboard drains"
        );
        assert!(
            engine.releases.lock().unwrap().contains(&gen2_pin),
            "gen-2's drained holder releases independently"
        );

        // Gen 1 drains: its pin finally releases.
        *cell1.lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(engine.releases.lock().unwrap().contains(&gen1_pin));
    }

    /// A `Finished` answer scrubs an unconsumed drain registration WITHOUT
    /// committing it: vLLM frees the blocks on `Finished` immediately, so a
    /// later `finished_sending` for that request would assert in the
    /// scheduler; and never taking it would leak the registration.
    #[test]
    fn request_finished_finished_scrubs_unconsumed_drain() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();
        // Offload registered but already terminal at finish time.
        let (offload, cell) = engine.arm_offload(&"r1".to_string());
        *cell.lock().unwrap() = ActionStatus::Complete;
        leader
            .state
            .lock()
            .get_mut("r1")
            .unwrap()
            .offloads
            .push(offload);

        assert_eq!(leader.request_finished("r1"), FinishedStatus::Finished);
        assert!(!leader.has_slot("r1"));
        assert!(
            engine.commits.lock().unwrap().is_empty(),
            "Finished must not commit (vLLM already freed the blocks)"
        );
        assert!(
            !engine.drains_armed.lock().unwrap().contains("r1"),
            "the unconsumed drain registration is scrubbed"
        );
    }

    /// Onboard-only-pending finish (no offloads ever registered): `Pending`
    /// with NO drain to commit — the engine's load terminal will surface the
    /// request via `finished_recving` (vLLM frees a finished request's blocks
    /// on that path too). The connector's job is only the handle-gated reap:
    /// the sweep must reap the slot once the onboard is terminal, with empty
    /// vLLM sets.
    #[test]
    fn onboard_in_flight_finish_reaps_on_terminal_without_drain() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        leader
            .create_slot(make_request("r1", (0..16u32).collect()))
            .unwrap();
        leader.get_num_new_matched_tokens("r1", 0).unwrap();

        // Stage an in-flight (Pending) onboard on the slot.
        let (onboard, cell) = engine.arm_onboard();
        leader.state.lock().get_mut("r1").unwrap().onboard = Some(onboard);

        assert_eq!(leader.request_finished("r1"), FinishedStatus::Pending);
        assert!(leader.has_slot("r1"));
        assert!(
            engine.commits.lock().unwrap().is_empty(),
            "no offloads were registered, so there is no drain to commit"
        );

        // Still loading: the sweep must not reap.
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(leader.has_slot("r1"));

        // Load terminal: the handle-gated sweep reaps with empty vLLM sets.
        *cell.lock().unwrap() = ActionStatus::Complete;
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(!leader.has_slot("r1"), "reap once the onboard is terminal");
    }

    // ---- build_connector_meta / offload walk --------------------------------

    /// Construct a SchedulerOutput for a single new request.
    fn sched_new(
        iteration: usize,
        req_id: &str,
        block_ids: Vec<BlockId>,
        num_computed: usize,
        num_scheduled: usize,
    ) -> SchedulerOutput {
        let mut out = SchedulerOutput::new(iteration);
        out.scheduled_new_reqs.push(NewRequestData {
            req_id: req_id.to_string(),
            prompt_token_ids: Vec::new(),
            block_ids,
            num_computed_tokens: num_computed,
        });
        out.num_scheduled_tokens
            .insert(req_id.to_string(), num_scheduled);
        out.total_num_scheduled_tokens = num_scheduled;
        out
    }

    /// Construct a SchedulerOutput for a single cached (non-resumed) request.
    fn sched_cached(
        iteration: usize,
        req_id: &str,
        new_block_ids: Vec<BlockId>,
        num_computed: usize,
        num_scheduled: usize,
    ) -> SchedulerOutput {
        let mut out = SchedulerOutput::new(iteration);
        out.scheduled_cached_reqs.push(CachedRequestData {
            req_id: req_id.to_string(),
            resumed: false,
            new_token_ids: Vec::new(),
            all_token_ids: None,
            new_block_ids,
            num_computed_tokens: num_computed,
            num_output_tokens: 0,
        });
        out.num_scheduled_tokens
            .insert(req_id.to_string(), num_scheduled);
        out.total_num_scheduled_tokens = num_scheduled;
        out
    }

    /// Walk a full 8-token prefill: two complete blocks are handed to the
    /// engine as a single offload call with the correct (hash, id) pairs.
    #[test]
    fn walk_full_prefill_offloads_completed_blocks() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        // create_slot with 8 tokens; BS=4 → 2 complete blocks.
        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();
        // USAA: allocate both blocks, no external load.
        leader.allocate("r1", vec![10, 11], 0).unwrap();

        let out = sched_new(42, "r1", vec![10, 11], 0, 8);
        let meta = leader.build_connector_meta(out).unwrap();

        assert_eq!(meta.iteration, 42);
        let calls = engine.offload_calls.lock().unwrap();
        assert_eq!(calls.len(), 1, "one offload call for both complete blocks");
        assert_eq!(calls[0].0, "r1");
        assert_eq!(calls[0].1.len(), 2, "both blocks in one call");
        // Block ids match what we allocated.
        assert_eq!(calls[0].1[0].1, 10);
        assert_eq!(calls[0].1[1].1, 11);
        // Hashes match what the slot computed.
        let state = leader.state.lock();
        let slot = state.get("r1").unwrap();
        assert_eq!(calls[0].1[0].0, slot.sequence_hash(0));
        assert_eq!(calls[0].1[1].0, slot.sequence_hash(1));
        assert_eq!(slot.offloads.len(), 1);
        assert_eq!(slot.evaluated_tokens, 8);
    }

    /// Chunked prefill: step 1 schedules 6 tokens (completes block 0 only);
    /// step 2 schedules the remaining 2 (completes block 1). Two separate
    /// offload calls in total.
    #[test]
    fn walk_chunked_prefill_advances_cursor_across_steps() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();
        leader.allocate("r1", vec![10, 11], 0).unwrap();

        // Step 1: 6 tokens → only block 0 complete.
        let out1 = sched_new(1, "r1", vec![10, 11], 0, 6);
        leader.build_connector_meta(out1).unwrap();
        {
            let calls = engine.offload_calls.lock().unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].1.len(), 1, "only block 0 complete after 6 tokens");
            assert_eq!(calls[0].1[0].1, 10);
        }
        {
            let state = leader.state.lock();
            assert_eq!(state.get("r1").unwrap().evaluated_tokens, 6);
        }

        // Step 2: cached non-resumed, 2 more tokens → block 1 complete.
        let out2 = sched_cached(2, "r1", vec![], 0, 2);
        leader.build_connector_meta(out2).unwrap();
        {
            let calls = engine.offload_calls.lock().unwrap();
            assert_eq!(calls.len(), 2, "second step adds one more offload call");
            assert_eq!(calls[1].1[0].1, 11);
        }
        {
            let state = leader.state.lock();
            assert_eq!(state.get("r1").unwrap().evaluated_tokens, 8);
        }
    }

    /// Cursor is capped at assigned_blocks × block_size. When only one block
    /// is allocated, scheduling 8 tokens advances the cursor to 4 only; the
    /// second block unlocks once it is allocated.
    #[test]
    fn walk_caps_cursor_at_allocated_blocks() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();
        // Allocate only one block first.
        leader.allocate("r1", vec![10], 0).unwrap();

        let out1 = sched_new(1, "r1", vec![10], 0, 8);
        leader.build_connector_meta(out1).unwrap();
        {
            let calls = engine.offload_calls.lock().unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].1.len(), 1, "only block 0 complete and allocated");
            assert_eq!(calls[0].1[0].1, 10);
        }
        {
            let state = leader.state.lock();
            // cursor is capped at 1 * BS = 4, not 8
            assert_eq!(state.get("r1").unwrap().evaluated_tokens, 4);
        }

        // Second step: block 11 allocated via delta, 4 more scheduled → block 1 done.
        let out2 = sched_cached(2, "r1", vec![11], 0, 4);
        leader.build_connector_meta(out2).unwrap();
        {
            let calls = engine.offload_calls.lock().unwrap();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[1].1[0].1, 11);
        }
        {
            let state = leader.state.lock();
            assert_eq!(state.get("r1").unwrap().evaluated_tokens, 8);
        }
    }

    /// Decode growth: after a full prefill, extending tokens by 4 and
    /// scheduling them offloads the new block on the boundary.
    #[test]
    fn walk_decode_growth_offloads_block_on_boundary() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        // Full prefill: 8 tokens, 2 blocks.
        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();
        leader.allocate("r1", vec![10, 11], 0).unwrap();
        let out1 = sched_new(1, "r1", vec![10, 11], 0, 8);
        leader.build_connector_meta(out1).unwrap();
        assert_eq!(engine.offload_calls.lock().unwrap().len(), 1);

        // Decode: extend with 4 more tokens, allocate block 12.
        leader
            .extend_slot_tokens("r1", (8..12u32).collect())
            .unwrap();
        let out2 = sched_cached(2, "r1", vec![12], 0, 4);
        leader.build_connector_meta(out2).unwrap();

        let calls = engine.offload_calls.lock().unwrap();
        assert_eq!(calls.len(), 2, "one new offload call for block 2");
        assert_eq!(calls[1].1[0].1, 12);
        // Hash is from block index 2.
        let state = leader.state.lock();
        let slot = state.get("r1").unwrap();
        assert_eq!(calls[1].1[0].0, slot.sequence_hash(2));
        assert_eq!(slot.evaluated_tokens, 12);
    }

    /// Mid-block decode: scheduling 1 token does not cross a block boundary
    /// and therefore produces no offload call.
    #[test]
    fn walk_mid_block_decode_advances_without_offload() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        // Full prefill: 8 tokens, 2 blocks.
        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();
        leader.allocate("r1", vec![10, 11], 0).unwrap();
        let out1 = sched_new(1, "r1", vec![10, 11], 0, 8);
        leader.build_connector_meta(out1).unwrap();
        let initial_calls = engine.offload_calls.lock().unwrap().len();

        // Decode one token, allocate block 12 (not yet full).
        leader.extend_slot_tokens("r1", vec![99]).unwrap();
        let out2 = sched_cached(2, "r1", vec![12], 0, 1);
        leader.build_connector_meta(out2).unwrap();

        let calls = engine.offload_calls.lock().unwrap();
        assert_eq!(
            calls.len(),
            initial_calls,
            "no new offload: block 2 boundary not crossed"
        );
        let state = leader.state.lock();
        assert_eq!(state.get("r1").unwrap().evaluated_tokens, 9);
    }

    /// A zero-scheduled-tokens output skips the walk entirely and returns
    /// metadata with only the iteration number set.
    #[test]
    fn walk_zero_scheduled_tokens_is_empty_passthrough() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();
        leader.allocate("r1", vec![10, 11], 0).unwrap();

        let mut out = SchedulerOutput::new(7);
        // total_num_scheduled_tokens stays 0
        out.scheduled_new_reqs.push(NewRequestData {
            req_id: "r1".to_string(),
            prompt_token_ids: Vec::new(),
            block_ids: vec![10, 11],
            num_computed_tokens: 0,
        });
        // total stays 0
        let meta = leader.build_connector_meta(out).unwrap();

        assert_eq!(meta.iteration, 7);
        assert!(meta.intra_pass_load.is_none());
        assert!(meta.foward_pass_completion_events.is_none());
        assert!(
            engine.offload_calls.lock().unwrap().is_empty(),
            "zero-token output must not trigger any offload"
        );
    }

    /// An output that names a request with no slot is silently skipped — no
    /// panic, no offload calls.
    #[test]
    fn walk_unknown_request_skipped() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        let out = sched_new(1, "ghost", vec![99], 0, 4);
        leader.build_connector_meta(out).unwrap();

        assert!(engine.offload_calls.lock().unwrap().is_empty());
    }

    /// A resumed request re-syncs the full token list (extending the
    /// sequence by any suffix not already present), re-syncs block ids from
    /// new_block_ids, and re-offloads from evaluated_tokens=0 using the new
    /// allocation but the same hashes (the chain is unchanged).
    #[test]
    fn walk_resumed_request_resyncs_and_reoffloads_from_zero() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        // Full prefill: 8 tokens, blocks [10,11].
        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();
        leader.allocate("r1", vec![10, 11], 0).unwrap();
        let out1 = sched_new(1, "r1", vec![10, 11], 0, 8);
        leader.build_connector_meta(out1).unwrap();
        let (h0_orig, h1_orig) = {
            let state = leader.state.lock();
            let slot = state.get("r1").unwrap();
            (slot.sequence_hash(0), slot.sequence_hash(1))
        };
        assert_eq!(engine.offload_calls.lock().unwrap().len(), 1);

        // Eviction: cursor resets to 0, block ids cleared.
        leader.on_evicted("r1").unwrap();

        // Restore: fresh USAA with new block ids [20, 21].
        leader.allocate("r1", vec![20, 21], 0).unwrap();

        // Resumed step: carries all_token_ids with 2 extra tokens (8+2=10).
        // Only 2 blocks have complete hashes (10 tokens → 2×4 blocks + 2 tail),
        // and we have 2 allocated → assigned_blocks = 2 → capped to 8.
        let all_tokens: Vec<u32> = (0..10u32).collect();
        let mut out2 = SchedulerOutput::new(2);
        out2.scheduled_cached_reqs.push(CachedRequestData {
            req_id: "r1".to_string(),
            resumed: true,
            new_token_ids: Vec::new(),
            all_token_ids: Some(all_tokens),
            new_block_ids: vec![20, 21],
            num_computed_tokens: 0,
            num_output_tokens: 0,
        });
        out2.num_scheduled_tokens.insert("r1".to_string(), 10);
        out2.total_num_scheduled_tokens = 10;
        leader.build_connector_meta(out2).unwrap();

        let calls = engine.offload_calls.lock().unwrap();
        assert_eq!(calls.len(), 2, "resumed slot re-offloads from zero");
        let resumed_call = &calls[1];
        assert_eq!(resumed_call.0, "r1");
        // New block ids are used.
        assert_eq!(resumed_call.1[0].1, 20);
        assert_eq!(resumed_call.1[1].1, 21);
        // Hashes are the SAME — the token chain is unchanged.
        assert_eq!(resumed_call.1[0].0, h0_orig);
        assert_eq!(resumed_call.1[1].0, h1_orig);
    }

    /// Eviction must mint the engine fence BEFORE detaching any handle view.
    /// Dropping a walk-minted offload handle first fires `release_action`,
    /// which prunes the action record the fence capture reads — the in-flight
    /// G1 read escapes the fence and vLLM recycles the blocks mid-copy.
    #[test]
    fn on_evicted_fences_before_dropping_offload_handles() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        // Walk-produced in-flight offload (the production shape).
        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();
        leader.allocate("r1", vec![10, 11], 0).unwrap();
        leader
            .build_connector_meta(sched_new(1, "r1", vec![10, 11], 0, 8))
            .unwrap();
        assert_eq!(leader.state.lock().get("r1").unwrap().offloads.len(), 1);

        leader.on_evicted("r1").unwrap();

        let events = engine.event_order.lock().unwrap();
        let evict_pos = events
            .iter()
            .position(|e| *e == "evict")
            .expect("evict must be called");
        let release_pos = events
            .iter()
            .position(|e| *e == "release_action")
            .expect("the dropped offload handle fires release_action");
        assert!(
            evict_pos < release_pos,
            "fence must be minted before any handle view detaches: {events:?}"
        );
    }

    /// End-to-end D machinery: a walk-produced offload handle drives the
    /// existing drain-commit / request_finished / update_connector_output flow.
    #[test]
    fn walk_offload_lifecycle_reaches_request_finished() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);

        // Full prefill producing one offload handle.
        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();
        leader.allocate("r1", vec![10, 11], 0).unwrap();
        let out = sched_new(1, "r1", vec![10, 11], 0, 8);
        leader.build_connector_meta(out).unwrap();
        assert_eq!(leader.state.lock().get("r1").unwrap().offloads.len(), 1);

        // request_finished with in-flight offload → Pending, drain committed.
        assert_eq!(leader.request_finished("r1"), FinishedStatus::Pending);
        assert!(leader.has_slot("r1"));
        assert_eq!(
            engine.commits.lock().unwrap().as_slice(),
            ["r1".to_string()],
            "drain committed in-call"
        );

        // Flip offload_cells[0] to terminal.
        *engine.offload_cells.lock().unwrap()[0].lock().unwrap() = ActionStatus::Complete;

        // Sweep reaps.
        leader
            .update_connector_output(HashSet::new(), HashSet::new())
            .unwrap();
        assert!(!leader.has_slot("r1"), "slot reaped after offload terminal");
    }

    /// The walk reports whether it scheduled offloads — the signal that arms
    /// the forward-pass flush trigger. True when a block boundary handed pairs
    /// to the engine; false for cursor-only advances.
    #[test]
    fn walk_reports_offload_scheduling_for_flush_arming() {
        let engine = recording_engine(Refresh::Lost);
        let leader = Leader::with_engine(engine.dyn_clone(), BS);
        leader
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();
        leader.allocate("r1", vec![10, 11], 0).unwrap();

        let out = sched_new(1, "r1", vec![10, 11], 0, 8);
        let walk = leader.state.lock().build_connector_meta(&out);
        assert!(
            walk.scheduled_offloads,
            "full prefill hands pairs to the engine"
        );

        // Mid-block decode: cursor advances, no boundary crossed.
        leader.extend_slot_tokens("r1", vec![99]).unwrap();
        let out2 = sched_cached(2, "r1", vec![12], 0, 1);
        let walk2 = leader.state.lock().build_connector_meta(&out2);
        assert!(
            !walk2.scheduled_offloads,
            "no boundary crossed → nothing to flush"
        );
    }

    // ---- matched_tokens counter (kvbm_matched_tokens) -------------------

    /// A GNMT hit increments the counter by the engine-reported token count
    /// exactly once; a re-poll of the same lifecycle (refresh resolves a hit
    /// again, no mint) must not double-count.
    #[test]
    fn gnmt_hit_increments_counter_exactly_once_across_repoll() {
        use super::state::LeaderState;
        use prometheus::{IntCounter, Opts};

        let counter = IntCounter::with_opts(Opts::new("mt_hit", "test")).unwrap();
        let engine = recording_engine(Refresh::Refined(2 * BS));
        let mut state = LeaderState::new(engine.dyn_clone(), BS, Some(counter.clone()));

        // Slot must exist before GNMT can build its chain.
        state
            .create_slot(make_request("r1", (0..12u32).collect()))
            .unwrap();

        // First GNMT: fresh mint resolves a hit → counter += 2 * BS.
        let (matched, load_async) = state.gnmt("r1", 0).unwrap();
        assert_eq!(matched, Some(2 * BS));
        assert!(load_async);
        assert_eq!(
            counter.get(),
            (2 * BS) as u64,
            "first hit must increment the counter"
        );

        // Re-poll: refresh resolves a hit again — flag already set, no second
        // increment.
        let (matched2, _) = state.gnmt("r1", 0).unwrap();
        assert_eq!(matched2, Some(2 * BS));
        assert_eq!(
            counter.get(),
            (2 * BS) as u64,
            "re-poll of the same lifecycle must not double-count"
        );
    }

    /// A GNMT miss (zero hit) never increments the counter.
    #[test]
    fn gnmt_miss_does_not_increment_counter() {
        use super::state::LeaderState;
        use prometheus::{IntCounter, Opts};

        let counter = IntCounter::with_opts(Opts::new("mt_miss", "test")).unwrap();
        let engine = noop_leader_engine();
        let mut state = LeaderState::new(engine, BS, Some(counter.clone()));

        state
            .create_slot(make_request("r1", (0..8u32).collect()))
            .unwrap();
        let (matched, _) = state.gnmt("r1", 0).unwrap();
        assert_eq!(matched, Some(0));
        assert_eq!(counter.get(), 0, "a miss must not touch the counter");
    }

    /// After the parked lifecycle is released (`release_parked` on a Lost
    /// refresh), the next GNMT mints fresh — which re-arms the once-flag so
    /// the new hit is counted independently.
    #[test]
    fn gnmt_fresh_mint_after_release_re_arms_counter() {
        use super::state::LeaderState;
        use prometheus::{IntCounter, Opts};

        let counter = IntCounter::with_opts(Opts::new("mt_rearm", "test")).unwrap();
        let engine = recording_engine(Refresh::Lost);
        let mut state = LeaderState::new(engine.dyn_clone(), BS, Some(counter.clone()));

        state
            .create_slot(make_request("r1", (0..12u32).collect()))
            .unwrap();

        // First poll: fresh-mint hit → counter = 2 * BS, flag set.
        let (matched1, _) = state.gnmt("r1", 0).unwrap();
        assert_eq!(matched1, Some(2 * BS));
        assert_eq!(counter.get(), (2 * BS) as u64);

        // Second poll: refresh resolves Lost → handle released, zero reported,
        // no increment.
        let (matched2, _) = state.gnmt("r1", 0).unwrap();
        assert_eq!(matched2, Some(0));
        assert_eq!(
            counter.get(),
            (2 * BS) as u64,
            "a Lost result reports zero tokens — no increment"
        );

        // Third poll: no parked lifecycle → fresh mint re-arms the flag → new
        // hit counted.
        let (matched3, _) = state.gnmt("r1", 0).unwrap();
        assert_eq!(matched3, Some(2 * BS));
        assert_eq!(
            counter.get(),
            (4 * BS) as u64,
            "a fresh mint after the release must count the new hit"
        );
    }

    // ---- StrandMark (missed-flush recovery marker) -----------------------

    /// Iteration 0 is a valid scheduler iteration and must be representable —
    /// the previous raw-`usize` marker used `0` as its none-sentinel, so a
    /// strand at iteration 0 was silently lost (never armed recovery).
    #[test]
    fn strand_mark_represents_iteration_zero() {
        let mark = super::StrandMark::default();
        assert!(!mark.outstanding());

        mark.record(0);
        assert!(
            mark.outstanding(),
            "a strand at iteration 0 must arm recovery"
        );

        mark.clear_through(0);
        assert!(!mark.outstanding(), "the iteration-0 sweep clears it");
    }

    /// An out-of-order OLDER confirmation must not clear a newer strand its
    /// `<=` sweep did not cover; the at-or-past confirmation does.
    #[test]
    fn strand_mark_out_of_order_confirmation_keeps_newer_strand() {
        let mark = super::StrandMark::default();
        mark.record(5);

        mark.clear_through(3);
        assert!(
            mark.outstanding(),
            "a pass-3 sweep does not cover the pass-5 strand"
        );

        mark.clear_through(5);
        assert!(!mark.outstanding());

        // Monotone record: an older strand cannot lower the mark.
        mark.record(7);
        mark.record(2);
        mark.clear_through(6);
        assert!(
            mark.outstanding(),
            "the mark must stay at the HIGHEST stranded iteration"
        );
        mark.clear_through(7);
        assert!(!mark.outstanding());
    }

    // ---- test doubles ----

    /// Scripted `find_blocks` outcome for a re-poll carrying a live handle.
    #[derive(Clone, Copy)]
    enum Refresh {
        /// `Resolved` with this many matched TOKENS, no mint. Zero maps to the
        /// zero-refine shape: `Resolved { 0, None, release_parked: true }`.
        Refined(usize),
        /// The pin vanished: `Resolved { 0, None, release_parked: true }`.
        Lost,
        /// Still resolving: `Searching { minted: None }`.
        Pending,
    }

    /// connector [`LeaderEngine`] double for the unified seam. `find_blocks` records
    /// the FULL request (chain + counts + remote-prefill params) plus whether
    /// a live handle was passed; a fresh call mints a lifecycle (recording the
    /// minted [`SearchId`]) and resolves a `2 * BS`-token hit (or `Searching`
    /// when `pending_search`); a live-handle call answers the scripted
    /// [`Refresh`]. `onboard_blocks` records `(generation, dest, committed)`
    /// and mints a `Pending` handle whose status cell lands in `onboard_cells`
    /// (flip it to simulate the load terminal). `offload` records
    /// `(request_id, pairs)`, mints a `Pending` handle, and arms the drain;
    /// `evict` / `release_search` / `release_action` record their calls.
    struct RecordingLeaderEngine {
        refresh: Refresh,
        /// Fresh `find_blocks` answers `Searching { minted: Some }` (status
        /// still resolving) instead of the default resolved hit.
        pending_search: bool,
        /// Fresh `find_blocks` answers `Deferred` (no mint, no side effect).
        defer_fresh: Mutex<bool>,
        /// `find_blocks` refuses outright (engine-side validation failure).
        reject_find: Mutex<bool>,
        /// `onboard_blocks` refuses outright (engine-side validation failure).
        reject_onboard: Mutex<bool>,
        /// One `(request, live-handle present)` entry per `find_blocks` call —
        /// asserts the adapter's translation fidelity (full chain, raw counts,
        /// verbatim remote-prefill params) and the fresh-vs-refresh routing
        /// input.
        find_calls: Mutex<Vec<(FindBlocksRequest, bool)>>,
        /// The generation of every lifecycle this engine minted, in order.
        minted: Mutex<Vec<SearchId>>,
        /// Every generation released back (RAII drops), in order.
        releases: Mutex<Vec<SearchId>>,
        evict_calls: Mutex<Vec<RequestId>>,
        /// When set, `evict` also mints a leader-side [`FenceHandle`] whose
        /// backing cell lands in `fence_cells` — flip it to simulate the
        /// engine barrier's drop. Off by default so the fence-less fallback
        /// suites keep exercising the onboard-terminal release rule.
        mint_fence_handles: Mutex<bool>,
        /// Backing cell of each minted fence handle, in `evict_calls` order.
        fence_cells: Mutex<Vec<Arc<AtomicBool>>>,
        self_weak: Mutex<Option<Weak<dyn LeaderEngine>>>,
        /// One `(generation, dest, committed external tokens)` entry per
        /// `onboard_blocks` call — asserts USAA hands the engine the parked
        /// lifecycle, the FULL allocated set, and the committed count.
        onboards: Mutex<Vec<(SearchId, Vec<BlockId>, usize)>>,
        /// Status cell of each minted onboard handle, in `onboards` order.
        onboard_cells: Mutex<Vec<Arc<Mutex<ActionStatus>>>>,
        /// One `(request_id, pairs)` entry per `offload` call.
        #[allow(clippy::type_complexity)]
        offload_calls: Mutex<Vec<(RequestId, Vec<(SequenceHash, BlockId)>)>>,
        /// Status cell of each minted offload handle, in `offload_calls` order.
        offload_cells: Mutex<Vec<Arc<Mutex<ActionStatus>>>>,
        /// Requests whose offload drain is armed; consumed (removed) by
        /// `take_offload_drain` so a second take returns `None` (consume-once).
        drains_armed: Mutex<HashSet<RequestId>>,
        /// One push per `RequestOffloadDrain::commit` — the once-per-request
        /// `finished_sending` emission the leader's deferred sweep drives.
        commits: Arc<Mutex<Vec<RequestId>>>,
        /// Ordered log of `evict` / `release_action` calls. The order is
        /// load-bearing: a handle dropped BEFORE `evict` prunes its action
        /// record from the engine, so the fence capture never sees it.
        event_order: Mutex<Vec<&'static str>>,
    }

    impl RecordingLeaderEngine {
        fn dyn_clone(self: &Arc<Self>) -> Arc<dyn LeaderEngine> {
            Arc::clone(self) as Arc<dyn LeaderEngine>
        }

        fn weak(&self) -> Weak<dyn LeaderEngine> {
            self.self_weak
                .lock()
                .unwrap()
                .clone()
                .expect("self_weak set")
        }

        /// Mint a `Pending` [`OffloadHandle`] and arm `req`'s offload drain. The
        /// caller pushes the handle into `slot.offloads`; flipping the returned
        /// status cell to `Complete` simulates the worker save terminal.
        fn arm_offload(&self, req: &RequestId) -> (OffloadHandle, Arc<Mutex<ActionStatus>>) {
            let cell = Arc::new(Mutex::new(ActionStatus::Pending));
            let handle = OffloadHandle::new(ActionId::new(), self.weak(), Arc::clone(&cell));
            self.drains_armed.lock().unwrap().insert(req.clone());
            (handle, cell)
        }

        /// Mint a `Pending` [`OnboardHandle`]; the caller moves it into
        /// `slot.onboard`. Flipping the returned status cell to `Complete`
        /// simulates the worker load terminal that lets the drain-holder release
        /// its lifecycle pin. The handle's RAII drop fires `release_action` (a
        /// no-op here), so the only release recorded on holder-drop is the
        /// lifecycle handle's `release_search`.
        fn arm_onboard(&self) -> (OnboardHandle, Arc<Mutex<ActionStatus>>) {
            let cell = Arc::new(Mutex::new(ActionStatus::Pending));
            let handle =
                OnboardHandle::new(ActionId::new(), self.weak(), Arc::clone(&cell), Vec::new());
            (handle, cell)
        }
    }

    fn recording_engine(refresh: Refresh) -> Arc<RecordingLeaderEngine> {
        recording_engine_with(refresh, false)
    }

    /// Variant whose fresh `find_blocks` reports an in-flight `Searching` mint.
    fn pending_search_engine() -> Arc<RecordingLeaderEngine> {
        recording_engine_with(Refresh::Pending, true)
    }

    fn recording_engine_with(refresh: Refresh, pending_search: bool) -> Arc<RecordingLeaderEngine> {
        let engine = Arc::new(RecordingLeaderEngine {
            refresh,
            pending_search,
            defer_fresh: Mutex::new(false),
            reject_find: Mutex::new(false),
            reject_onboard: Mutex::new(false),
            find_calls: Mutex::new(Vec::new()),
            minted: Mutex::new(Vec::new()),
            releases: Mutex::new(Vec::new()),
            evict_calls: Mutex::new(Vec::new()),
            mint_fence_handles: Mutex::new(false),
            fence_cells: Mutex::new(Vec::new()),
            self_weak: Mutex::new(None),
            onboards: Mutex::new(Vec::new()),
            onboard_cells: Mutex::new(Vec::new()),
            offload_calls: Mutex::new(Vec::new()),
            offload_cells: Mutex::new(Vec::new()),
            drains_armed: Mutex::new(HashSet::new()),
            commits: Arc::new(Mutex::new(Vec::new())),
            event_order: Mutex::new(Vec::new()),
        });
        *engine.self_weak.lock().unwrap() = Some(Arc::downgrade(
            &(Arc::clone(&engine) as Arc<dyn LeaderEngine>),
        ));
        engine
    }

    impl LeaderEngine for RecordingLeaderEngine {
        fn find_blocks(
            self: Arc<Self>,
            req: &FindBlocksRequest,
            live: Option<&FindBlocksHandle>,
        ) -> Result<FindBlocksOutcome, LeaderEngineError> {
            self.find_calls
                .lock()
                .unwrap()
                .push((req.clone(), live.is_some()));
            if *self.reject_find.lock().unwrap() {
                return Err(LeaderEngineError::FindBlocksDesync);
            }
            // REFRESH arm: a live handle is reconciled in place, never a
            // second mint.
            if live.is_some() {
                return Ok(match self.refresh {
                    Refresh::Refined(tokens) if tokens > 0 => FindBlocksOutcome::Resolved {
                        matched_tokens: tokens,
                        minted: None,
                        release_parked: false,
                    },
                    Refresh::Refined(_) | Refresh::Lost => FindBlocksOutcome::Resolved {
                        matched_tokens: 0,
                        minted: None,
                        release_parked: true,
                    },
                    Refresh::Pending => FindBlocksOutcome::Searching { minted: None },
                });
            }
            // FRESH arm.
            if *self.defer_fresh.lock().unwrap() {
                return Ok(FindBlocksOutcome::Deferred);
            }
            let id = SearchId::new();
            self.minted.lock().unwrap().push(id);
            let handle = FindBlocksHandle::search(req.request_id.clone(), id, self.weak());
            Ok(if self.pending_search {
                FindBlocksOutcome::Searching {
                    minted: Some(handle),
                }
            } else {
                FindBlocksOutcome::Resolved {
                    matched_tokens: 2 * BS,
                    minted: Some(handle),
                    release_parked: false,
                }
            })
        }

        fn onboard_blocks(
            self: Arc<Self>,
            handle: &FindBlocksHandle,
            dest: &[BlockId],
            num_external_tokens: usize,
        ) -> Result<OnboardHandle, LeaderEngineError> {
            if *self.reject_onboard.lock().unwrap() {
                return Err(LeaderEngineError::ExternalTokensMismatch {
                    expected: 2 * BS,
                    got: num_external_tokens,
                });
            }
            let generation = handle
                .search_id()
                .expect("this double mints only search-kind lifecycles");
            self.onboards
                .lock()
                .unwrap()
                .push((generation, dest.to_vec(), num_external_tokens));
            let cell = Arc::new(Mutex::new(ActionStatus::Pending));
            self.onboard_cells.lock().unwrap().push(Arc::clone(&cell));
            Ok(OnboardHandle::new(
                ActionId::new(),
                self.weak(),
                cell,
                dest.to_vec(),
            ))
        }

        fn offload(
            self: Arc<Self>,
            req: &RequestId,
            pairs: Vec<(SequenceHash, BlockId)>,
        ) -> Result<OffloadHandle, LeaderEngineError> {
            self.offload_calls
                .lock()
                .unwrap()
                .push((req.clone(), pairs.clone()));
            let cell = Arc::new(Mutex::new(ActionStatus::Pending));
            self.offload_cells.lock().unwrap().push(Arc::clone(&cell));
            self.drains_armed.lock().unwrap().insert(req.clone());
            Ok(OffloadHandle::new(ActionId::new(), self.weak(), cell))
        }

        fn evict(&self, req: &RequestId) -> EvictionOutcome {
            self.evict_calls.lock().unwrap().push(req.clone());
            self.event_order.lock().unwrap().push("evict");
            // The leader-side handle is opt-in (`mint_fence_handles`); a test
            // simulates the engine barrier's drop by flipping the recorded cell.
            let handle = self.mint_fence_handles.lock().unwrap().then(|| {
                let cell = Arc::new(AtomicBool::new(false));
                self.fence_cells.lock().unwrap().push(Arc::clone(&cell));
                FenceHandle::new(cell)
            });
            // Mint one per-(generation, worker) token so the fence carries real
            // work into metadata (rank 0; the double models a single worker).
            EvictionOutcome {
                fence: EvictionFence {
                    request_id: req.clone(),
                    per_worker: vec![FenceToken::new(0)],
                },
                handle,
            }
        }

        fn take_offload_drain(&self, req: &RequestId) -> Option<RequestOffloadDrain> {
            // Consume-once: an armed req yields its drain exactly once; a second
            // take (or an unarmed req) returns `None`.
            if self.drains_armed.lock().unwrap().remove(req) {
                let commits = Arc::clone(&self.commits);
                let req = req.clone();
                Some(RequestOffloadDrain::new(move || {
                    commits.lock().unwrap().push(req);
                }))
            } else {
                None
            }
        }

        fn poll_action(&self, _id: &ActionId) -> ActionStatus {
            ActionStatus::Complete
        }

        fn release_search(&self, id: &SearchId) {
            self.releases.lock().unwrap().push(*id);
        }

        fn release_action(&self, _action: &ActionId) {
            self.event_order.lock().unwrap().push("release_action");
        }
    }
}
