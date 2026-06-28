// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `LocalConnectorEngine` — the in-process [`LeaderEngine`] over an
//! `Arc<InstanceLeader>`.
//!
//! The connector holds an `Arc<dyn LeaderEngine>` and drives match/onboard
//! through the frozen connector seam ([`kvbm_protocols::connector`]); this is the
//! local impl that backs it. It is held as `Arc<InstanceLeader>` **concretely**
//! (not `Arc<dyn Leader>`) because `release_session` is an inherent method, not
//! a trait method.
//!
//! # Coordinate rebase (suffix → absolute)
//!
//! A [`SearchRequest`] carries only `computed_blocks` and `block_plhs`
//! (= the per-block hashes of the suffix after the computed prefix); it carries
//! **no** `total_tokens` and **no** prefix hashes. The reconcile core, however,
//! works in **absolute** block-index coordinates over the full sequence-hash
//! vector. The bridge:
//!
//! * Each live search owns an absolute-indexed `buffer`: index `i` holds the
//!   hash of absolute block `i`. The `[0 .. base)` head (blocks never searched —
//!   `base` is the lowest `computed_blocks` ever seen) is padding that is never
//!   sliced; `[base .. high)` is filled from each poll's `block_plhs` at offset
//!   `computed_blocks`. A `computed_blocks` drop (Case C) front-extends real
//!   hashes; a `total_tokens` grow (Case D) back-extends them.
//! * `total_tokens` is **synthesized** as `last_block_index * block_size + 1`
//!   (a one-token in-progress trailing block) so the reconcile core's
//!   `compute_last_block_index` reproduces exactly `last_block_index =
//!   computed_blocks + block_plhs.len()` — the same upper bound the initial
//!   shard used. This keeps the reconcile core (and its 21 ported tests)
//!   byte-for-byte.
//!
//! # `block_plhs` contract
//!
//! The match cores treat `block_plhs` as the **exact** set of blocks eligible
//! for external matching and search `[computed_blocks .. computed_blocks +
//! block_plhs.len())` with **no further trimming**. The `find_blocks` router
//! owns the trimming: it derives the window from the request's full hash chain
//! plus `total_tokens` (vLLM's "last-must-recompute" exclusion — see
//! `find::derive_window`) and hands the cores the already-trimmed view.

// The unit tests below exercise entry points the lib target reads as dead
// (`cfg(test)` is stripped), and a handful of seam types keep test-only
// consumers; one module-level allow covers that gap.
#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use anyhow::Result;
use dashmap::DashMap;

use kvbm_protocols::connector::{
    AcceptId, ActionFailure, ActionId, ActionStatus, EngineWorkerSink, EvictionFence,
    EvictionOutcome, FenceToken, FindBlocksHandle, FindBlocksOutcome, FindBlocksRequest,
    LeaderEngine, LeaderEngineError, OffloadHandle, OnboardHandle, RequestOffloadDrain,
    ResourceOnboard, SearchId, WorkerEngineDriver,
};
use kvbm_protocols::connector::{BlockId, RequestId, SequenceHash};

use kvbm_logical::blocks::ImmutableBlock;

use super::driver::{ActionRecord, FenceBarrier};
use super::inflight::{InflightKey, InflightOnboards};
use super::offload::{self, BufferedOffload, DisabledOffloadSubmit, OffloadSubmit};
use super::onboard;
use super::reconcile::{
    MatchCheckOutcome, OnboardingState, compute_outcome, issue_shard, reconcile_state,
};
use crate::G2;
use crate::leader::{InstanceLeader, Leader};
use crate::p2p::session::{PeerResolver, SessionFactory};
use crate::remote::cd::DisaggConfig;
use crate::remote::cd::budget::{InflightBudget, TierCell};
use crate::remote::cd::commit::{RemoteCommitPlan, open_and_commit};
use crate::remote::cd::decode::{self, LocalReason, PlanInputs, PlanOutcome};
use crate::remote::cd::output::PrefillOutputObserver;
use crate::remote::cd::prefill::PrefillRequests;
use crate::remote::cd::state::{CdRequestState, CdRequests};
use crate::remote::cd::wire::{PrefillDispatch, PrefillPlane};

/// Per-search engine state, keyed by [`SearchId`] in `searches`.
pub(super) struct SearchState {
    /// The request this search belongs to (for `by_request` upkeep on onboard).
    pub(super) request_id: RequestId,
    /// The match-status cell the engine advances (via `local_refresh`) and the
    /// find router projects into the source-agnostic outcome; the engine is the
    /// only writer, preserving the "no public advance" invariant.
    pub(super) status: Arc<Mutex<MatchStatus>>,
    /// The reconciled match state (shards + `num_computed_tokens`).
    pub(super) onboarding: OnboardingState,
    /// Absolute-indexed sequence-hash buffer (see module docs).
    pub(super) buffer: Vec<SequenceHash>,
}

/// The local, in-process [`LeaderEngine`].
pub(crate) struct LocalConnectorEngine {
    pub(super) leader: Arc<InstanceLeader>,
    pub(super) sink: Arc<dyn EngineWorkerSink>,
    pub(super) block_size: usize,
    /// Whether shard finds request the leader's remote-search path. Set from
    /// the [`RemoteOps`](super::RemoteOps) selection at construction; threaded
    /// into every `FindMatchesOptions` the engine issues.
    pub(super) search_remote: bool,
    pub(super) searches: DashMap<SearchId, SearchState>,
    pub(super) actions: DashMap<ActionId, ActionRecord>,
    /// `request_id → in-flight onboard/offload action ids` (read by `evict`).
    pub(super) by_request: DashMap<RequestId, Vec<ActionId>>,
    /// The offload-submission seam (real: [`super::offload::OffloadEngineSubmit`];
    /// onboard-only tests default to [`DisabledOffloadSubmit`]). Behind a trait
    /// so the GPU/velo-bound `OffloadEngine` stays mockable.
    pub(super) offload_submit: Arc<dyn OffloadSubmit>,
    /// Pairs buffered by `offload`, flushed by `finish_forward_pass` (Decision A:
    /// never enqueue a G1 read mid-forward-pass).
    pub(super) offload_buffer: Mutex<Vec<BufferedOffload>>,
    /// Requests with at least one offload — the once-only source for
    /// `take_offload_drain` (removal *is* the consume-once guard).
    pub(super) offload_drains: DashMap<RequestId, ()>,
    /// The current forward-pass iteration (recorded by `begin_forward_pass`).
    pub(super) current_iteration: AtomicUsize,
    /// Self-`Weak` so `finish_forward_pass(&self)` can mint an `Arc<Self>` to
    /// move into the per-offload completion driver (the `WorkerEngineDriver`
    /// receiver is `&self`, unlike onboard's `self: Arc<Self>`).
    weak_self: Weak<LocalConnectorEngine>,
    /// Conditional-disaggregation runtime, present iff CD is configured (built
    /// from [`super::DisaggOps`] at construction). The search path interposes
    /// CD when this is `Some`; `None` is a plain local-tiering engine.
    pub(super) cd: Option<CdRuntime>,
    /// In-flight onboard hash guard (see [`super::inflight`]). Recorded ONCE
    /// per lifecycle at the three onboard mint sites (keyed by the lifecycle
    /// generation), cleared at the lifecycle release funnels
    /// (`release_search` / `prefill_release`), and consulted by the
    /// `find_blocks` router on every non-exempt arm (see [`super::find`]).
    pub(super) inflight: Mutex<InflightOnboards>,
}

/// The conditional-disaggregation runtime: the resource-free decision config +
/// breaker tier, the inflight-token admission budget (sized from the config),
/// the session-factory + prefill-plane transports, and the per-request CD
/// state container every CD lifecycle bubbles through.
///
/// A Remote commit latches a UNIFIED hit count (`fbet / block_size`) that can
/// exceed the local match; `onboard()` interposes the local/remote fan-out
/// (`cd_onboard`) so a unified-committed hit onboards the local matched span via
/// the in-process G2→G1 collect AND pulls the remote slice over the parked CD
/// session. The load terminal (`complete_load`, fired by the CD producers AFTER
/// `finish_load_action` returns, against the originating lifecycle's Arc) is the
/// eager budget-release + session-finalize path alongside decline (handle drop)
/// and evict.
///
/// A vLLM-computed prefix is served by the SAME commit when its `[0, computed)`
/// blocks are G2-resident (the offload save cursor walks from 0, so steady-state
/// traffic backfills them naturally); a not-fully-resident prefix downgrades the
/// request to a local prefill before the budget is touched.
///
/// PARKED (still unbuilt, test-only reachability): the remote-search-pending
/// ledger source — a deferred-availability tail that would leave the ledger
/// undrained at the load terminal. Until it lands the production commit always
/// passes an empty `pending_hashes`, and the load terminal always observes a
/// drained ledger.
pub(super) struct CdRuntime {
    cfg: DisaggConfig,
    tier: Arc<TierCell>,
    /// The inflight remote-prefill token budget — the waiting consumer of
    /// `cfg.max_inflight_remote_prefill_tokens`, built once at construction.
    budget: InflightBudget,
    pub(super) sessions: Arc<dyn SessionFactory>,
    plane: Arc<dyn PrefillPlane>,
    requests: CdRequests,
    /// Resolves + registers the decode peer before the prefill pipeline
    /// attaches (velo's streaming-transport registry is lazily populated).
    /// `None` only works when the peer is already registered.
    pub(super) peer_resolver: Option<Arc<dyn PeerResolver>>,
    /// Prefill-side per-request lifecycles — the inbound counterpart of
    /// `requests`. No budget coupling: the inflight budget is a decode-side
    /// admission concept.
    pub(super) prefill: PrefillRequests,
    /// The prefill OUTPUT capture: the engine-owned G2 register observer
    /// (registered on the offload pipeline's G1→G2 register step at
    /// construction) that publishes freshly-offloaded computed blocks into
    /// each accepted lifecycle's parked session.
    pub(super) output: Arc<PrefillOutputObserver>,
}

impl CdRuntime {
    /// Build the runtime, sizing the inflight budget from the config.
    pub(super) fn new(
        cfg: DisaggConfig,
        tier: Arc<TierCell>,
        sessions: Arc<dyn SessionFactory>,
        plane: Arc<dyn PrefillPlane>,
        peer_resolver: Option<Arc<dyn PeerResolver>>,
    ) -> Self {
        Self {
            budget: InflightBudget::new(cfg.max_inflight_remote_prefill_tokens),
            cfg,
            tier,
            sessions,
            plane,
            requests: CdRequests::new(),
            peer_resolver,
            prefill: PrefillRequests::new(),
            output: Arc::new(PrefillOutputObserver::new()),
        }
    }

    /// Poll interval of the prefill release's deferred-finalize drain.
    pub(super) fn output_drain_poll(&self) -> std::time::Duration {
        self.cfg.output_drain_poll
    }

    /// Watchdog bound on that drain (force-finalize past it).
    pub(super) fn output_drain_watchdog(&self) -> std::time::Duration {
        self.cfg.output_drain_watchdog
    }

    /// Idempotent CD cleanup for a request that the connector declined, that was
    /// evicted, or that finished: close the parked session with `reason` and
    /// release its budget reservation back to the container. Identity-checked,
    /// so a stale release from a prior lifecycle of the same request id is a
    /// no-op. Must NOT be called under a `searches` guard.
    ///
    /// Current-lifecycle: tears down whatever is latched at the re-fetch. Used
    /// by `evict` (which refers to NOW — the latched lifecycle is the one being
    /// evicted).
    fn cleanup(&self, request_id: &RequestId, reason: &str) {
        self.cleanup_guarded(request_id, reason, None);
    }

    /// Generation-bound twin of [`Self::cleanup`]. When `expect` is `Some(sid)`
    /// the latched state is torn down ONLY if its originating `search_id`
    /// matches `sid`; otherwise the re-fetch hit a DIFFERENT generation (an
    /// evict + re-latch of the same request id installed a fresh lifecycle) and
    /// this no-ops. `None` is current-lifecycle (the [`Self::cleanup`] / evict
    /// behaviour). Used by `release_search`, where a stale OLD-generation
    /// search-kind `FindBlocksHandle` parked in a drain-holder can drop AFTER the rid re-latched
    /// — without the guard its `release_search` would close the fresh session +
    /// release the fresh budget. Must NOT be called under a `searches` guard.
    fn cleanup_guarded(&self, request_id: &RequestId, reason: &str, expect: Option<SearchId>) {
        if let Some(state) = self.requests.get(request_id) {
            if let Some(sid) = expect
                && state.search_id() != sid
            {
                // Re-fetched a fresh lifecycle (evict + re-latch). The stale
                // release must not touch it.
                return;
            }
            if let Some(session) = state.take_session() {
                session.close(Some(reason.to_string()));
            }
            self.requests
                .release_if_matches(request_id, &state, &self.budget);
        }
    }

    /// Load-terminal CD bubble: release the inflight budget and finalize/close
    /// the parked session once the onboard fan-out reaches a terminal outcome.
    /// Identity-checked + idempotent (same shape as [`Self::cleanup`]), so it is
    /// a no-op when an evict/decline cleanup already released this lifecycle —
    /// the races are all safe. Must NOT be called under a `searches`/`actions`
    /// guard (the `finish_load_action` post-guard zone guarantees this).
    ///
    /// `state` is the ORIGINATING lifecycle's `Arc<CdRequestState>` (the CD
    /// producer — the onboard driver task or `mint_failed_onboard` — captured it
    /// at fan-out time). Teardown reads `take_session` / `release_if_matches`
    /// off THAT `Arc`, never a fresh `requests` re-fetch: a stale terminal that
    /// fires after an evict + re-latch of the SAME rid would otherwise tear down
    /// the FRESH lifecycle's session/reservation. `take_session` on the stale
    /// `Arc` yields `None` (the evict already took it) and `release_if_matches`
    /// no-ops on the `Arc::ptr_eq` mismatch, so the fresh lifecycle is untouched.
    ///
    /// On `Complete` the session is finalized cooperatively, but ONLY when the
    /// deferred-availability ledger has drained. In the current regime the
    /// ledger is always drained at the load terminal (no deferred
    /// remote-search source is wired, and the commit's `pending_hashes` is
    /// always empty). A `Complete` with an undrained ledger is unreachable
    /// today; were one to arrive, the unconditional
    /// `release_if_matches` below has already removed this entry from `requests`,
    /// so no ledger owner could finalize through the map — the session is closed
    /// deterministically (naming the abandoned ledger) rather than leaked. On
    /// failure the session is closed with the reason.
    pub(super) fn complete_load(
        &self,
        request_id: &RequestId,
        state: &Arc<CdRequestState>,
        outcome: &ActionStatus,
    ) {
        match outcome {
            ActionStatus::Complete => {
                if state.ledger_is_drained() {
                    if let Some(session) = state.take_session() {
                        session.finalize(None);
                    }
                } else {
                    debug_assert!(
                        false,
                        "cd complete_load: Complete with undrained ledger for {request_id} \
                         — the deferred-availability finalize path is unwired"
                    );
                    if let Some(session) = state.take_session() {
                        session.close(Some(format!(
                            "cd load complete but availability ledger undrained \
                             (abandoned) for {request_id}"
                        )));
                    }
                }
            }
            ActionStatus::Failed(_) | ActionStatus::Pending => {
                if let Some(session) = state.take_session() {
                    session.close(Some("cd onboard failed".to_string()));
                }
            }
        }
        self.requests
            .release_if_matches(request_id, state, &self.budget);
    }
}

impl LocalConnectorEngine {
    /// Build the engine over a concrete leader, a worker sink, and the layout
    /// block size (carried here because `InstanceLeader` exposes no block-size
    /// accessor). Returns `Arc<Self>` — callers coerce to `Arc<dyn LeaderEngine>`.
    ///
    /// Offload submission defaults to [`DisabledOffloadSubmit`]; the wired engine
    /// stack constructs the real [`OffloadEngine`] via [`Self::with_offload_submit`].
    pub(crate) fn new(
        leader: Arc<InstanceLeader>,
        sink: Arc<dyn EngineWorkerSink>,
        block_size: usize,
        search_remote: bool,
    ) -> Arc<Self> {
        Self::with_offload_submit(
            leader,
            sink,
            block_size,
            search_remote,
            Arc::new(DisabledOffloadSubmit),
            None,
        )
    }

    /// Build the engine with an explicit offload-submission seam (the
    /// `tiering::engine`
    /// [`build_local_connector_engine`](super::build_local_connector_engine)
    /// factory injects the real [`OffloadEngine`]; offload tests inject a
    /// double) and an optional conditional-disagg runtime (the factory builds
    /// it from [`super::DisaggOps`]; offload/onboard tests pass `None`). Kept
    /// `pub(super)` so the offload + CD seams stay encapsulated.
    pub(super) fn with_offload_submit(
        leader: Arc<InstanceLeader>,
        sink: Arc<dyn EngineWorkerSink>,
        block_size: usize,
        search_remote: bool,
        offload_submit: Arc<dyn OffloadSubmit>,
        cd: Option<CdRuntime>,
    ) -> Arc<Self> {
        // Source the in-flight-onboard gauge from the leader's observability
        // BEFORE `leader` moves into the cyclic closure. Bare test leaders have
        // no observability → `None` → an inert guard.
        let inflight_gauge = leader
            .observability()
            .map(|o| o.compat_metrics().inflight_onboard_hashes.clone());
        Arc::new_cyclic(|weak| Self {
            leader,
            sink,
            block_size,
            search_remote,
            searches: DashMap::new(),
            actions: DashMap::new(),
            by_request: DashMap::new(),
            offload_submit,
            offload_buffer: Mutex::new(Vec::new()),
            offload_drains: DashMap::new(),
            current_iteration: AtomicUsize::new(0),
            weak_self: weak.clone(),
            cd,
            inflight: Mutex::new(InflightOnboards::with_gauge(inflight_gauge)),
        })
    }
}

// ============================================================================
// Match cores (over `&dyn Leader`, for testability)
// ============================================================================

/// Engine-internal match status (relocated from the protocols seam, which no
/// longer carries it). The status cell a live search shares with the engine
/// reads `Matched { hit_blocks }` / `Pending` / `Lost`; `find_blocks` /
/// `onboard_blocks` project it into the source-agnostic outcome the connector
/// reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MatchStatus {
    /// The match resolved (or refined). `hit_blocks` may be `0` (search done,
    /// no external prefix).
    Matched { hit_blocks: u32 },
    /// Some shard is still non-terminal — re-poll.
    Pending,
    /// The underlying source was lost between polls (eviction/timeout).
    Lost,
}

/// Engine-internal owned match-window input (relocated from the protocols
/// seam). The `find_blocks` router slices the request's shared hash chain in
/// place via [`MatchWindow`]; this owned shape is the test harness's input that
/// [`MatchWindow::of`] borrows.
#[derive(Debug, Clone)]
pub(super) struct SearchRequest {
    pub(super) request_id: RequestId,
    /// The computed prefix's per-block hashes `[0, computed)`, in
    /// absolute-position order (empty when nothing is computed).
    pub(super) prefix_plhs: Vec<SequenceHash>,
    /// Per-block hashes after the computed prefix, with vLLM's
    /// last-must-recompute exclusion already applied.
    pub(super) block_plhs: Vec<SequenceHash>,
    /// Blocks already resident in G1 (vLLM's `num_computed` in blocks).
    pub(super) computed_blocks: u32,
}

/// Borrowed view of one poll's eligible match window — the input shape every
/// engine-internal match core takes. The `find_blocks` router builds it as a
/// zero-copy index-range slice of the request's shared hash chain (no
/// per-poll window `Vec`); test harnesses adapt a [`SearchRequest`] via
/// [`MatchWindow::of`].
pub(super) struct MatchWindow<'a> {
    pub(super) request_id: &'a RequestId,
    /// The computed prefix's per-block hashes `[0, computed)`, in
    /// absolute-position order — what a CD remote commit must serve in
    /// addition to the window. Clamped to the hashed chain by the router, so
    /// it may run shorter than `computed_blocks` on a malformed poll (the
    /// residency gate then keeps prefill local).
    pub(super) prefix_plhs: &'a [SequenceHash],
    /// Per-block hashes after the computed prefix, with vLLM's
    /// last-must-recompute exclusion already applied.
    pub(super) block_plhs: &'a [SequenceHash],
    /// Blocks already resident in G1 (vLLM's `num_computed` in blocks).
    pub(super) computed_blocks: u32,
}

impl<'a> MatchWindow<'a> {
    /// View over a legacy [`SearchRequest`] (the old seam verbs' input).
    pub(super) fn of(req: &'a SearchRequest) -> Self {
        Self {
            request_id: &req.request_id,
            prefix_plhs: &req.prefix_plhs,
            block_plhs: &req.block_plhs,
            computed_blocks: req.computed_blocks,
        }
    }
}

/// The outcome of an initial [`perform_search`] before a handle is minted.
pub(super) enum SearchInit {
    /// Nothing to search, or a terminal zero-block hit — no handle is minted.
    NoMatch,
    /// A live search: the reconciled state + buffer to store, the initial
    /// status to share with the handle, and the projection (`pending` → the
    /// `Pending` outcome, else `Hit { hit_blocks }`).
    Active {
        state: OnboardingState,
        buffer: Vec<SequenceHash>,
        status: MatchStatus,
        pending: bool,
        hit_blocks: u32,
    },
}

/// Build the absolute-indexed buffer for an initial search: `[0 .. cb)` padded
/// with `plhs[0]` (never sliced), `[cb .. cb + plhs.len())` the real suffix.
fn build_buffer(computed_blocks: usize, plhs: &[SequenceHash]) -> Vec<SequenceHash> {
    debug_assert!(!plhs.is_empty());
    let last = computed_blocks + plhs.len();
    let mut buffer = vec![plhs[0]; last];
    buffer[computed_blocks..last].copy_from_slice(plhs);
    buffer
}

/// Merge a refresh poll's `block_plhs` into the persistent absolute buffer at
/// offset `computed_blocks`, growing it as needed. A `computed_blocks` drop
/// (Case C) overwrites previously-padded head slots with real hashes; a grow
/// (Case D) extends the tail.
fn merge_buffer(buffer: &mut Vec<SequenceHash>, computed_blocks: usize, plhs: &[SequenceHash]) {
    if plhs.is_empty() {
        return;
    }
    // A `computed_blocks` above the current high-water mark would leave a gap of
    // never-seen blocks between them (the pathological B+D case the legacy code
    // also cannot serve); surface it loudly in debug rather than slicing
    // padding into a find.
    debug_assert!(
        computed_blocks <= buffer.len(),
        "refresh gap: computed_blocks {} exceeds buffer high-water {}",
        computed_blocks,
        buffer.len()
    );
    let last = computed_blocks + plhs.len();
    if buffer.len() < last {
        buffer.resize(last, plhs[0]);
    }
    buffer[computed_blocks..last].copy_from_slice(plhs);
}

/// The refresh set-equality heuristic: cheap facts of the incoming poll vs the
/// stored search state. Same `num_computed_tokens`-derived offset, same
/// eligible end (the buffer high-water mark), and first + last window hashes
/// unchanged => identical (a pure re-poll). Deliberately NOT a full compare —
/// a content change at identical positions cannot happen by the per-block
/// chain invariants (each hash folds its full prefix lineage), so the
/// endpoint spot-check is sufficient to guard it.
fn refresh_window_unchanged(state: &SearchState, req: &MatchWindow<'_>, block_size: usize) -> bool {
    let computed_blocks = req.computed_blocks as usize;
    let plhs = req.block_plhs;
    if plhs.is_empty() {
        // A zero-width refresh has nothing to spot-check; let the merge path
        // (a no-op merge + reconcile) answer it.
        return false;
    }
    state.onboarding.num_computed_tokens == computed_blocks * block_size
        && state.buffer.len() == computed_blocks + plhs.len()
        && state.buffer[computed_blocks] == plhs[0]
        && state.buffer[state.buffer.len() - 1] == plhs[plhs.len() - 1]
}

/// Issue the initial shard and project the first match outcome. Operates over
/// `&dyn Leader` so it is unit-testable without an `InstanceLeader`.
pub(super) fn perform_search(
    leader: &dyn Leader,
    block_size: usize,
    search_remote: bool,
    cd_enabled: bool,
    req: &MatchWindow<'_>,
) -> Result<SearchInit> {
    let computed_blocks = req.computed_blocks as usize;
    let plhs = req.block_plhs;
    // No full block left to match → no handle.
    if plhs.is_empty() {
        return Ok(SearchInit::NoMatch);
    }

    let last_block_index = computed_blocks + plhs.len();
    let buffer = build_buffer(computed_blocks, plhs);
    let shard = issue_shard(
        leader,
        &buffer,
        computed_blocks,
        last_block_index,
        search_remote,
    )?;
    let total_tokens = last_block_index * block_size + 1;
    let state = OnboardingState::new(computed_blocks * block_size, total_tokens, shard);

    match compute_outcome(&state, block_size) {
        MatchCheckOutcome::InProgress => Ok(SearchInit::Active {
            state,
            buffer,
            status: MatchStatus::Pending,
            pending: true,
            hit_blocks: 0,
        }),
        MatchCheckOutcome::Found { matched_tokens } if matched_tokens > 0 => {
            let hit_blocks = (matched_tokens / block_size) as u32;
            Ok(SearchInit::Active {
                state,
                buffer,
                status: MatchStatus::Matched { hit_blocks },
                pending: false,
                hit_blocks,
            })
        }
        // Terminal zero-match on the *initial* poll. With CD enabled a
        // zero-local-match request can still go Remote, so mint a latched
        // zero-block `Active` and let `cd_interpose` decide (the cold-prompt
        // case — the primary disagg offload shape). Without CD there is no
        // consumer for an empty handle: mint nothing, dropping `state` (and any
        // RAII pins) here — a terminal zero-match `Ready` shard holds no
        // session, so there is nothing to release explicitly.
        MatchCheckOutcome::Found { .. } | MatchCheckOutcome::NoMatch => {
            if cd_enabled {
                Ok(SearchInit::Active {
                    state,
                    buffer,
                    status: MatchStatus::Matched { hit_blocks: 0 },
                    pending: false,
                    hit_blocks: 0,
                })
            } else {
                Ok(SearchInit::NoMatch)
            }
        }
    }
}

/// Reconcile a live search against the poll's current `computed_blocks` and
/// project the refined [`MatchStatus`]. Operates over `&dyn Leader`.
pub(super) fn perform_refresh(
    leader: &dyn Leader,
    block_size: usize,
    search_remote: bool,
    state: &mut OnboardingState,
    buffer: &mut Vec<SequenceHash>,
    req: &MatchWindow<'_>,
) -> Result<MatchStatus> {
    let computed_blocks = req.computed_blocks as usize;
    let plhs = req.block_plhs;
    merge_buffer(buffer, computed_blocks, plhs);

    let last_block_index = computed_blocks + plhs.len();
    let total_tokens = last_block_index * block_size + 1;
    reconcile_state(
        state,
        computed_blocks * block_size,
        total_tokens,
        block_size,
        buffer,
        leader,
        search_remote,
    )?;

    Ok(match compute_outcome(state, block_size) {
        MatchCheckOutcome::InProgress => MatchStatus::Pending,
        MatchCheckOutcome::Found { matched_tokens } => MatchStatus::Matched {
            hit_blocks: (matched_tokens / block_size) as u32,
        },
        MatchCheckOutcome::NoMatch => MatchStatus::Matched { hit_blocks: 0 },
    })
}

// ============================================================================
// LeaderEngine impl
// ============================================================================

impl LocalConnectorEngine {
    fn buffer_offload(
        self: Arc<Self>,
        resource: Option<kvbm_common::LogicalResourceId>,
        req: &RequestId,
        pairs: Vec<(SequenceHash, BlockId)>,
    ) -> Result<OffloadHandle, LeaderEngineError> {
        if let Some(resource) = resource
            && !self.offload_submit.supports_resource(resource)
        {
            return Err(LeaderEngineError::ResourceOffloadNotConfigured { resource });
        }
        let action_id = ActionId::new();
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        self.actions.insert(
            action_id,
            ActionRecord::new(req.clone(), Arc::downgrade(&cell)),
        );
        self.by_request
            .entry(req.clone())
            .or_default()
            .push(action_id);
        self.offload_drains.insert(req.clone(), ());

        self.offload_buffer
            .lock()
            .expect("offload-buffer mutex poisoned")
            .push(BufferedOffload {
                action_id,
                request_id: req.clone(),
                resource,
                pairs,
                iteration: self.current_iteration.load(Ordering::Relaxed),
            });

        let me: Arc<dyn LeaderEngine> = self;
        Ok(OffloadHandle::new(action_id, Arc::downgrade(&me), cell))
    }
}

impl LeaderEngine for LocalConnectorEngine {
    fn offload(
        self: Arc<Self>,
        req: &RequestId,
        pairs: Vec<(SequenceHash, BlockId)>,
    ) -> Result<OffloadHandle, LeaderEngineError> {
        self.buffer_offload(None, req, pairs)
    }

    fn offload_for_resource(
        self: Arc<Self>,
        resource: kvbm_common::LogicalResourceId,
        req: &RequestId,
        pairs: Vec<(SequenceHash, BlockId)>,
    ) -> Result<OffloadHandle, LeaderEngineError> {
        self.buffer_offload(Some(resource), req, pairs)
    }

    fn evict(&self, req: &RequestId) -> EvictionOutcome {
        // Find in-flight onboard actions for this request and, if any, flag them
        // cancelled-for-emission: their terminal fires `mark_fence_complete`
        // (not `mark_load_finished`) and mints one barrier token per worker.
        let action_ids: Vec<ActionId> = self
            .by_request
            .get(req)
            .map(|ids| ids.clone())
            .unwrap_or_default();

        // ONE critical section per action: re-check pending AND arm the fence under
        // the SAME `get_mut` write guard, serialized against `finish_*_action`'s
        // `get_mut` on that action — so a terminal cannot land between the check and
        // the arm. The shared `FenceBarrier` is minted lazily on the first arm and
        // each armed record holds a clone; `evict` keeps one clone (`barrier`) as an
        // ARMING GUARD for the whole loop, so the barrier cannot complete mid-loop
        // even if an armed action drains before the loop ends. The fence completes
        // only when the LAST clone drops (the last armed action's terminal), so a
        // worker reuses G1 blocks only after every in-flight-at-eviction action has
        // drained — never on the first.
        let mut barrier: Option<Arc<FenceBarrier>> = None;
        for id in &action_ids {
            if let Some(mut record) = self.actions.get_mut(id) {
                let still_pending = record.cell.upgrade().is_some_and(|cell| {
                    matches!(
                        *cell.lock().expect("action-status mutex poisoned"),
                        ActionStatus::Pending
                    )
                });
                // Arm only an unfenced action. The drain-holder/fresh-GNMT design
                // precludes re-evicting the same in-flight action, but guarding
                // `fence.is_none()` is a cheap strict improvement: it stops a second
                // evict from reassigning a live barrier (which would drop the prior
                // clone and complete that fence one drain early).
                if still_pending && record.fence.is_none() {
                    let shared = barrier.get_or_insert_with(|| {
                        let worker_count = self.leader.worker_count().max(1);
                        let tokens = (0..worker_count as u32).map(FenceToken::new).collect();
                        Arc::new(FenceBarrier::new(tokens, self.sink.clone()))
                    });
                    record.fence = Some(Arc::clone(shared));
                }
            }
        }

        // Tokens (and the leader's observational handle) are returned IFF at least
        // one action was armed (`barrier` is Some), so an empty arm set yields no
        // orphan fence and no handle. The arming-guard clone drops at end of scope:
        // if every armed action already drained during the loop, that drop is the
        // last clone and completes the fence immediately (the worker's await — and
        // the leader's `is_complete` poll — then see an already-complete fence).
        let (per_worker, handle) = match barrier.as_ref() {
            Some(b) => (b.tokens().to_vec(), Some(b.leader_handle())),
            None => (Vec::new(), None),
        };

        // Bubble eviction into CD: close the parked session and release the
        // budget reservation (idempotent + identity-checked). Not under any
        // searches guard; the cd map is independent of `actions`/`by_request`.
        if let Some(cd) = &self.cd {
            cd.cleanup(req, "evicted");
            // Prefill-side teardown is engine-internal too (the connector's
            // eviction path needs no prefill knowledge): release the rid's
            // CURRENT latched generation. Runs AFTER the fence arming above so
            // an in-flight USAA kick action is fenced before this release's
            // session close can drive its terminal. A stale connector handle
            // dropping later double-releases safely: the generation is gone
            // from the map (or re-accepted under a fresh `AcceptId`), so
            // `release_prefill_session` no-ops on its guards.
            if let Some(state) = cd.prefill.get(req) {
                self.prefill_release(req, state.accept_id());
            }
        }

        EvictionOutcome {
            fence: EvictionFence {
                request_id: req.clone(),
                per_worker,
            },
            handle,
        }
    }

    fn take_offload_drain(&self, req: &RequestId) -> Option<RequestOffloadDrain> {
        // Consume-once: removal IS the guard. A second call finds nothing → None;
        // a request that never offloaded was never registered → None. (`remove`
        // over a taken-flag is self-cleaning; under `request_id` reuse a fresh
        // `offload` re-registers and a new drain is legitimately mintable —
        // consistent with the `FenceToken.generation` reuse handling.)
        self.offload_drains.remove(req)?;
        // D semantics: `commit` ARMS the engine's emit-on-last-terminal (it
        // never emits synchronously from the caller's context unless nothing is
        // pending). The connector commits at `request_finished(Pending)` time —
        // possibly with actions still in flight — and the engine fires the
        // single `mark_save_finished` when the last pending-at-commit action
        // drains (immediately, if none are). See `arm_drain_emission`.
        let weak = self.weak_self.clone();
        let req = req.clone();
        Some(RequestOffloadDrain::new(move || {
            match weak.upgrade() {
                Some(engine) => engine.arm_drain_emission(&req),
                // Engine teardown: nothing to coordinate; the workers are gone too.
                None => tracing::debug!(%req, "drain commit after engine teardown; dropped"),
            }
        }))
    }

    fn poll_action(&self, id: &ActionId) -> ActionStatus {
        // Sync map read — NO await on this path (the driver is async; this is a
        // plain lookup + cell read). A live cell (handle still alive) carries the
        // real status; the map guard is released before the cell lock.
        let live = self.actions.get(id).and_then(|r| r.cell.upgrade());
        if let Some(cell) = live {
            return cell.lock().expect("action-status mutex poisoned").clone();
        }
        // No entry, or the handle dropped (dead `Weak`). Self-clean any dead entry
        // on access so the map cannot grow without bound, then report the
        // stateless default: a vanished action has no observer to mislead, and a
        // never-minted id has nothing in flight — matching the noop answer.
        self.actions
            .remove_if(id, |_, r| r.cell.strong_count() == 0);
        ActionStatus::Complete
    }

    fn release_search(&self, id: &SearchId) {
        // Clear this lifecycle's in-flight onboard deferral record FIRST,
        // unconditionally: an onboarded lifecycle's `searches` entry was
        // consumed at onboard time, so the map check below must not gate the
        // clear. Idempotent and generation-bound by the key (a stale release
        // can only touch its own entry, never a fresh re-latch's).
        self.inflight
            .lock()
            .expect("inflight-guard mutex poisoned")
            .clear(&InflightKey::Search(*id));
        if let Some((_id, search_state)) = self.searches.remove(id) {
            // Best-effort: release each shard's server-side session. Ready pins
            // are no-ops (RAII drop). If onboard already consumed the search,
            // the entry is gone and this is a no-op (no double release).
            search_state.onboarding.release_all(&self.leader);
            // Bubble into CD: reaching here means the handle dropped WITHOUT
            // onboarding (a decline of the committed hit) — `onboard` removes the
            // searches entry itself before minting its action and never calls
            // `release_search`, so a removed entry here is never a post-onboard
            // drop. Close the parked session + release the budget (idempotent).
            //
            // Generation-bound by the released `SearchId`: a stale OLD-generation
            // handle parked in a drain-holder can drop AFTER an evict + re-latch
            // of the same rid installed a fresh lifecycle; the guard then no-ops
            // instead of tearing down that fresh session + budget.
            if let Some(cd) = &self.cd {
                cd.cleanup_guarded(&search_state.request_id, "declined", Some(*id));
            }
        }
    }

    fn release_action(&self, id: &ActionId) {
        // RAII prune (fired by `OnboardHandle`/`OffloadHandle` drop): drop the
        // by-id `actions` entry and scrub the `by_request` index for it. Idempotent:
        // an unknown or already-released id finds no entry and does nothing (the noop
        // offload path and a second drop are both no-ops).
        //
        // DEFER if a fence is armed: the action was evicted and its driver has not
        // reached terminal (`finish_*_action` takes the fence). Removing the record
        // now would drop the live fence clone and could complete the eviction fence
        // BEFORE the transfer drains — freeing G1 blocks mid-transfer. Instead flag
        // `dropped_by_handle` under the per-action guard (serialized against
        // `finish_*_action`); the driver's terminal then removes the record. Takes
        // only DashMap guards — lock order dashmap→cell is preserved.
        let defer = {
            if let Some(mut record) = self.actions.get_mut(id) {
                let armed = record.fence.is_some() || record.drain.is_some();
                if armed {
                    record.dropped_by_handle = true;
                }
                armed
            } else {
                false
            }
        };
        if !defer {
            self.remove_action_record(id);
        }
    }

    fn find_blocks(
        self: Arc<Self>,
        req: &FindBlocksRequest,
        live: Option<&FindBlocksHandle>,
    ) -> Result<FindBlocksOutcome, LeaderEngineError> {
        // Body in `super::find` (the unified source-agnostic router).
        self.route_find_blocks(req, live)
    }

    fn onboard_blocks(
        self: Arc<Self>,
        handle: &FindBlocksHandle,
        dest: &[BlockId],
        num_external_tokens: usize,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        self.route_onboard_blocks(handle, dest, num_external_tokens)
    }

    fn onboard_resources(
        self: Arc<Self>,
        req: &RequestId,
        resources: Vec<ResourceOnboard>,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        if resources.is_empty() {
            return Err(LeaderEngineError::InvalidResourceOnboard {
                reason: "at least one resource is required".to_owned(),
            });
        }
        let mut seen = std::collections::HashSet::new();
        for transfer in &resources {
            if !seen.insert(transfer.resource) {
                return Err(LeaderEngineError::InvalidResourceOnboard {
                    reason: format!("duplicate logical resource {:?}", transfer.resource),
                });
            }
            if transfer.source_block_ids.is_empty()
                || transfer.source_block_ids.len() != transfer.destination_block_ids.len()
            {
                return Err(LeaderEngineError::InvalidResourceOnboard {
                    reason: format!(
                        "resource {:?} has {} G2 sources and {} G1 destinations",
                        transfer.resource,
                        transfer.source_block_ids.len(),
                        transfer.destination_block_ids.len()
                    ),
                });
            }
            if self.leader.g2_manager_for(transfer.resource).is_none() {
                return Err(LeaderEngineError::ResourceOnboardNotConfigured {
                    resource: transfer.resource,
                });
            }
        }

        let action_id = ActionId::new();
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        self.actions.insert(
            action_id,
            ActionRecord::new(req.clone(), Arc::downgrade(&cell)),
        );
        self.by_request
            .entry(req.clone())
            .or_default()
            .push(action_id);
        let handle_dest_ids = resources
            .iter()
            .flat_map(|transfer| transfer.destination_block_ids.iter().copied())
            .collect::<Vec<_>>();
        let request_id = req.clone();
        let driver = Arc::clone(&self);
        let terminal_dest_ids = handle_dest_ids.clone();
        self.leader.runtime().spawn(async move {
            let mut outcome = ActionStatus::Complete;
            for transfer in resources {
                let notification = match driver.leader.execute_local_transfer_for_resource(
                    transfer.resource,
                    kvbm_common::LogicalLayoutHandle::G2,
                    kvbm_common::LogicalLayoutHandle::G1,
                    transfer.source_block_ids,
                    transfer.destination_block_ids,
                    kvbm_physical::TransferOptions::default(),
                ) {
                    Ok(notification) => notification,
                    Err(error) => {
                        tracing::error!(
                            error = %error,
                            resource = ?transfer.resource,
                            "resource onboard dispatch failed"
                        );
                        outcome = ActionStatus::Failed(ActionFailure::AllBlocks);
                        break;
                    }
                };
                if let Err(error) = notification.await {
                    tracing::error!(
                        error = %error,
                        resource = ?transfer.resource,
                        "resource onboard transfer failed"
                    );
                    outcome = ActionStatus::Failed(ActionFailure::AllBlocks);
                    break;
                }
            }
            driver.finish_load_action(action_id, &request_id, outcome, terminal_dest_ids);
        });

        let engine: Arc<dyn LeaderEngine> = self;
        Ok(OnboardHandle::new(
            action_id,
            Arc::downgrade(&engine),
            cell,
            handle_dest_ids,
        ))
    }

    fn release_prefill_session(&self, request_id: &RequestId, accept_id: AcceptId) {
        self.prefill_release(request_id, accept_id);
    }
}

// ============================================================================
// Local match cores (driven by the find router)
// ============================================================================

/// Outcome of [`LocalConnectorEngine::local_search`] before any handle is
/// minted: the latched generation plus its (possibly CD-interposed) status.
/// The `find_blocks` router mints a pure-RAII [`FindBlocksHandle`] over the
/// latch and lets the outcome carry the facts.
#[derive(Debug)]
pub(super) enum LocalSearchOutcome {
    /// A live search was latched into `searches` under `search_id`.
    Latched {
        search_id: SearchId,
        status: MatchStatus,
    },
    /// Nothing to search, or a terminal zero-block hit — nothing latched.
    NoMatch,
}

impl LocalConnectorEngine {
    /// The initial-search core: issue the find, latch the reconciled state into
    /// `searches`, and (on a resolved hit) run the conditional-disagg
    /// interposition. Search failures collapse to `NoMatch` (loud log, never an
    /// error across the seam).
    pub(super) fn local_search(&self, req: &MatchWindow<'_>) -> LocalSearchOutcome {
        match perform_search(
            self.leader.as_ref(),
            self.block_size,
            self.search_remote,
            self.cd.is_some(),
            req,
        ) {
            Ok(SearchInit::Active {
                state,
                buffer,
                status,
                pending,
                ..
            }) => {
                let search_id = SearchId::new();
                let cell = Arc::new(Mutex::new(status));
                // Pre-clone the prefix + matched local-G2 blocks for a possible
                // CD commit BEFORE `state` moves into the searches map: CD reads
                // no searches entry (the one-way searches→cd lock order). Both
                // empty unless CD is configured and this is a resolved hit.
                let (cd_prefix, cd_local) = if pending {
                    (Vec::new(), Vec::new())
                } else {
                    self.cd_clone_local_blocks(req, status, &state)
                };
                self.searches.insert(
                    search_id,
                    SearchState {
                        request_id: req.request_id.clone(),
                        status: cell.clone(),
                        onboarding: state,
                        buffer,
                    },
                );
                if pending {
                    LocalSearchOutcome::Latched {
                        search_id,
                        status: MatchStatus::Pending,
                    }
                } else {
                    // CD interposition runs post-insert with NO searches guard
                    // held; it writes the (possibly Remote-committed) status
                    // through the shared cell. A Reject parks the request on
                    // Pending (the caller re-polls until the budget frees).
                    let status =
                        self.cd_interpose(search_id, req, status, &cell, cd_prefix, cd_local);
                    // A zero-hit that CD did not promote to Remote must not latch
                    // (mirrors the terminal-zero collapse in `perform_search`,
                    // deferred until CD had its commit chance): vLLM never
                    // onboards an empty match, so the handle and the `searches`
                    // entry would linger forever. A fresh zero-block mint inserted
                    // nothing into `cd.requests`/`inflight`, so a DIRECT remove
                    // (not `release_search`, whose decline hook would be
                    // misleading) tears down exactly the one map entry; the
                    // terminal-Ready shard releases nothing on drop.
                    if matches!(status, MatchStatus::Matched { hit_blocks: 0 }) {
                        self.searches.remove(&search_id);
                        LocalSearchOutcome::NoMatch
                    } else {
                        LocalSearchOutcome::Latched { search_id, status }
                    }
                }
            }
            Ok(SearchInit::NoMatch) => LocalSearchOutcome::NoMatch,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    request_id = %req.request_id,
                    "search find_matches failed; reporting NoMatch"
                );
                LocalSearchOutcome::NoMatch
            }
        }
    }

    /// The refresh core: reconcile a live search against the poll's current
    /// window and project the refined status. A missing latch (already released
    /// or onboarded) reports `Lost`.
    ///
    /// A PURE re-poll — same computed offset, same eligible end, first and
    /// last window hash unchanged (the cheap set-equality heuristic; content
    /// changes at identical positions cannot happen by the chain invariants,
    /// so the spot check suffices) — skips the buffer merge and the
    /// reconcile/reissue walk entirely and only re-projects the outcome (shard
    /// completions still advance the status). Any mismatch warns loudly and
    /// takes the full merge/reconcile path.
    pub(super) fn local_refresh(&self, search_id: SearchId, req: &MatchWindow<'_>) -> MatchStatus {
        // Reconcile and write the LOCAL status under the searches guard, then
        // clone out ONLY what CD needs (the shared status cell + the prefix and
        // matched local-G2 blocks). CD must run AFTER the guard drops and must
        // never touch `self.searches` in the same call (the one-way searches→cd
        // order).
        let (cell, status, cd_prefix, cd_local) = {
            let Some(mut entry) = self.searches.get_mut(&search_id) else {
                // The search was already released (or onboarded); nothing to refine.
                return MatchStatus::Lost;
            };
            let state = &mut *entry;
            let status = if refresh_window_unchanged(state, req, self.block_size) {
                // Pure re-poll: project the current outcome without touching
                // the buffer or the shard set.
                match compute_outcome(&state.onboarding, self.block_size) {
                    MatchCheckOutcome::InProgress => MatchStatus::Pending,
                    MatchCheckOutcome::Found { matched_tokens } => MatchStatus::Matched {
                        hit_blocks: (matched_tokens / self.block_size) as u32,
                    },
                    MatchCheckOutcome::NoMatch => MatchStatus::Matched { hit_blocks: 0 },
                }
            } else {
                tracing::warn!(
                    request_id = %req.request_id,
                    computed_blocks = req.computed_blocks,
                    window = req.block_plhs.len(),
                    stored_computed_tokens = state.onboarding.num_computed_tokens,
                    stored_high_water = state.buffer.len(),
                    "refresh window changed between polls; merging \
                     (a content change here would violate the chain invariants)"
                );
                match perform_refresh(
                    self.leader.as_ref(),
                    self.block_size,
                    self.search_remote,
                    &mut state.onboarding,
                    &mut state.buffer,
                    req,
                ) {
                    Ok(status) => status,
                    Err(e) => {
                        tracing::warn!(error = %e, "local_refresh reconcile failed; reporting Lost");
                        MatchStatus::Lost
                    }
                }
            };
            // Advance the shared cell the handle reads (the only writer).
            *state.status.lock().expect("search-status mutex poisoned") = status;
            let cell = state.status.clone();
            let (cd_prefix, cd_local) = self.cd_clone_local_blocks(req, status, &state.onboarding);
            (cell, status, cd_prefix, cd_local)
        };
        // Post-guard: CD may re-resolve the answer (latch / commit / reject),
        // writing the new status through the cloned cell. No searches access here.
        self.cd_interpose(search_id, req, status, &cell, cd_prefix, cd_local)
    }

    /// The onboard core, keyed by [`SearchId`]. The latch must still be *live*
    /// (`Matched`) to onboard from it; the matched-status read comes off the
    /// engine-side latch (the same shared cell the find router projects from).
    pub(super) fn local_onboard(
        self: Arc<Self>,
        search_id: SearchId,
        dest: &[BlockId],
    ) -> Result<OnboardHandle, LeaderEngineError> {
        // Sufficient precondition: a latch exists and its pin is still live
        // (Matched). The guard drops before the take below.
        let hit_blocks = {
            let Some(entry) = self.searches.get(&search_id) else {
                return Err(LeaderEngineError::SearchNotMatched);
            };
            let status = *entry.status.lock().expect("search-status mutex poisoned");
            match status {
                MatchStatus::Matched { hit_blocks } => hit_blocks,
                MatchStatus::Pending | MatchStatus::Lost => {
                    return Err(LeaderEngineError::SearchNotMatched);
                }
            }
        };

        // Take ownership of the pinned state: the onboard now owns the sources'
        // lifecycle. The blocks stay resident inside the driver task until the
        // transfer completes (drain-not-drop — eviction handles in-flight
        // onboards via `evict`, not via search-handle drop).
        let (_id, search_state) = self
            .searches
            .remove(&search_id)
            .ok_or(LeaderEngineError::SearchNotMatched)?;
        let request_id = search_state.request_id;
        let mut onboarding = search_state.onboarding;
        // The absolute-indexed hash buffer, snapshotted before the local mint so
        // the in-flight guard records the matched window even though `onboarding`
        // moves into the CD fan-out or the driver below. Unused on the CD arm
        // (it records its own unified window).
        let buffer = search_state.buffer;

        let block_size = self.block_size;
        let num_computed_tokens = onboarding.num_computed_tokens;
        let num_external_tokens = hit_blocks as usize * block_size;
        let g1_block_ids = onboard::select_onboard_block_ids(
            dest,
            num_computed_tokens,
            num_external_tokens,
            block_size,
        );

        // CD fan-out interposition: a latched conditional-disagg request onboards
        // through the local/remote split (the local matched span via the
        // in-process G2→G1 collect, the remote slice pulled over the parked CD
        // session) rather than the plain local collect below. `g1_block_ids` is
        // the external dest slice (the computed prefix already excluded); the
        // fan-out splits it at `local_hit_blocks`. The lookup clones an owned
        // `Arc`, so no `self.cd`/`requests` borrow is held across the move.
        // `search_id` rides along as the deferral-guard key: the connector's
        // parked handle carries THIS generation, so the record binds to the
        // release that will actually fire (a re-latched CD state may carry an
        // older originating id).
        let cd_state = self.cd.as_ref().and_then(|cd| cd.requests.get(&request_id));
        if let Some(state) = cd_state {
            return Ok(self.cd_onboard(
                state,
                search_id,
                request_id,
                onboarding,
                g1_block_ids,
                hit_blocks,
            ));
        }

        let staging_futs: Vec<_> = onboarding
            .shards
            .iter()
            .map(|shard| shard.find_session.wait_for_completion())
            .collect();

        let action_id = ActionId::new();
        // The completion cell: the engine keeps a `Weak` in `actions` (the by-id
        // `poll_action` path), the handle owns the strong — so the cell frees by
        // RAII on handle drop, with no terminal-category-dependent prune. Insert
        // the in-flight record BEFORE returning the handle so `evict` and the
        // by-id path observe a live `Pending` action, not the missing-key default.
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        self.actions.insert(
            action_id,
            ActionRecord::new(request_id.clone(), Arc::downgrade(&cell)),
        );
        self.by_request
            .entry(request_id.clone())
            .or_default()
            .push(action_id);

        // Record the in-flight onboard's hit window for the deferral guard,
        // keyed by the lifecycle generation so the clear lands at the
        // connector-visible release (`release_search`). The window is the
        // matched span `buffer[computed .. computed + hit]` —
        // absolute-indexed, copied once per lifecycle (never per poll). `min`
        // is defensive: a well-behaved match never reaches past the buffer.
        let computed_blocks = num_computed_tokens / block_size;
        let avail = buffer.len().saturating_sub(computed_blocks);
        let hit = (hit_blocks as usize).min(avail);
        let window = buffer[computed_blocks..computed_blocks + hit].to_vec();
        self.inflight
            .lock()
            .expect("inflight-guard mutex poisoned")
            .record(InflightKey::Search(search_id), window);

        // The dest set travels twice: into the driver terminal (which resolves
        // a total failure to these concrete ids — see `finish_load_action`) and
        // into the handle (whose `outcome()` projection names them likewise).
        let handle_dest_ids = g1_block_ids.clone();
        let this = self.clone();
        self.leader.runtime().spawn(async move {
            let dest_ids = g1_block_ids.clone();
            let outcome = onboard::run_onboard(
                &this.leader,
                &mut onboarding,
                g1_block_ids,
                staging_futs,
                block_size,
            )
            .await;
            // Release every shard's server-side session (Ready pins are no-ops).
            onboarding.release_all(&this.leader);
            // Write terminal into the cell + notify with no engine lock held.
            this.finish_load_action(action_id, &request_id, outcome, dest_ids);
        });

        // The handle owns the strong cell and a `Weak` back-ref; its RAII drop
        // fires `release_action` to prune both the by-id `actions` map and the
        // `by_request` index for this action (the entry lives until then).
        let me: Arc<dyn LeaderEngine> = self;
        Ok(OnboardHandle::new(
            action_id,
            Arc::downgrade(&me),
            cell,
            handle_dest_ids,
        ))
    }
}

// ============================================================================
// Conditional-disagg interposition (search-path, both terminal sites)
// ============================================================================

impl LocalConnectorEngine {
    /// Clone the G2 blocks a possible CD search-time commit must serve,
    /// WITHOUT consuming them (the onboard's later `take_g2_blocks` must still
    /// work): the matched local-window span out of the search shards, plus —
    /// when a vLLM-computed prefix exists — the absolute `[0, computed)`
    /// prefix pins out of this leader's G2 cache (`match_blocks` over the
    /// request's prefix hashes; the session-serve contract pins the WHOLE
    /// `[0, DNPT)` window, so a remote commit must name and serve the prefix
    /// too). A prefix shorter than `computed_blocks` (not fully G2-resident)
    /// is returned as-is; `cd_interpose`'s residency gate downgrades on it.
    /// Both vecs are empty unless CD is configured and `status` is a resolved
    /// match.
    fn cd_clone_local_blocks(
        &self,
        req: &MatchWindow<'_>,
        status: MatchStatus,
        onboarding: &OnboardingState,
    ) -> (Vec<ImmutableBlock<G2>>, Vec<ImmutableBlock<G2>>) {
        if self.cd.is_none() {
            return (Vec::new(), Vec::new());
        }
        let local_hit = match status {
            MatchStatus::Matched { hit_blocks } => hit_blocks as usize,
            MatchStatus::Pending | MatchStatus::Lost => return (Vec::new(), Vec::new()),
        };
        // A latched re-poll answers at the latch and drops these clones
        // unused — skip the O(computed) prefix walk for it on the GNMT hot
        // path. (A stale latch downgrades to local on this poll; the next
        // poll, latch-free, re-captures.)
        let prefix_blocks = if req.computed_blocks > 0
            && self
                .cd
                .as_ref()
                .is_some_and(|cd| cd.requests.get(req.request_id).is_none())
        {
            self.leader.g2_manager().match_blocks(req.prefix_plhs)
        } else {
            Vec::new()
        };
        let local_blocks = clone_leading_g2_blocks(onboarding, local_hit).unwrap_or_default();
        (prefix_blocks, local_blocks)
    }

    /// Conditional-disagg interposition, shared by both terminal sites
    /// (`local_search`'s Matched arm and `local_refresh`). Returns the status to
    /// surface, writing any change THROUGH the shared `cell`. Runs with NO
    /// searches guard held and never touches `self.searches` (the one-way
    /// searches→cd order); it may read/write `self.cd`'s own maps.
    ///
    /// `search_id` is the originating search's id; a Remote commit stamps it
    /// onto the latched `CdRequestState` so the decline release hook can bind
    /// teardown to this generation.
    fn cd_interpose(
        &self,
        search_id: SearchId,
        req: &MatchWindow<'_>,
        local_status: MatchStatus,
        cell: &Arc<Mutex<MatchStatus>>,
        prefix_blocks: Vec<ImmutableBlock<G2>>,
        local_blocks: Vec<ImmutableBlock<G2>>,
    ) -> MatchStatus {
        // 1. CD off, or not a resolved match → passthrough. CD evaluates every
        //    resolved count (including a zero-block match — a zero-local-match
        //    request can still go Remote); Pending/Lost are not resolved.
        let Some(cd) = self.cd.as_ref() else {
            return local_status;
        };
        let local_hit = match local_status {
            MatchStatus::Matched { hit_blocks } => hit_blocks as usize,
            MatchStatus::Pending | MatchStatus::Lost => return local_status,
        };
        let rid = req.request_id;
        let bs = self.block_size;
        let computed_blocks = req.computed_blocks as usize;
        let num_computed = computed_blocks * bs;

        // 2. Latch: an already-committed request returns its unified count with
        //    NO re-plan and NO budget touch (the idempotent-retry guard) —
        //    PROVIDED the poll's computed prefix still matches the offset the
        //    commit was made at. The unified count and its window hashes are
        //    absolute facts of THAT offset; answering them against a moved
        //    prefix would misalign `matched_tokens` with vLLM's
        //    beyond-num_computed contract. A moved prefix releases the stale
        //    lifecycle (session closed, budget back) and downgrades to the
        //    freshly-reconciled local answer; the next poll re-plans fresh.
        if let Some(state) = cd.requests.get(rid) {
            if state.base_offset() != num_computed {
                tracing::warn!(
                    %rid,
                    committed_offset = state.base_offset(),
                    poll_offset = num_computed,
                    "cd: latched lifecycle is stale (computed prefix moved); \
                     releasing it and keeping prefill local"
                );
                if let Some(session) = state.take_session() {
                    session.close(Some("stale cd latch: computed prefix moved".to_string()));
                }
                cd.requests.release_if_matches(rid, &state, &cd.budget);
                return local_status;
            }
            let status = MatchStatus::Matched {
                hit_blocks: state.unified_hit_blocks(),
            };
            *cell.lock().expect("search-status mutex poisoned") = status;
            return status;
        }

        // 3. Prefix-residency gate: a remote commit pins the WHOLE
        //    `[0, DNPT)` window on the wire (DNPT is a length, not a set), so
        //    the `[0, computed)` blocks must be servable from this G2 — a hole
        //    is not expressible to the prefill peer. Not fully resident →
        //    keep prefill local. Runs BEFORE the plan so the budget is never
        //    touched on this downgrade; the offload save cursor starts at 0,
        //    so steady-state traffic backfills the prefix into G2 naturally
        //    and the next request with this prefix can go Remote.
        if computed_blocks > 0 && prefix_blocks.len() != computed_blocks {
            tracing::info!(
                %rid,
                computed_blocks,
                resident = prefix_blocks.len(),
                "cd: computed prefix not fully G2-resident; keeping prefill local"
            );
            return local_status;
        }

        // 4. Plan over block-quantized token counts.
        let window_blocks = req.block_plhs.len();
        let inputs = PlanInputs {
            total_tokens: (computed_blocks + window_blocks) * bs,
            num_computed_tokens: num_computed,
            matched_tokens: local_hit * bs,
            block_size: bs,
        };

        match decode::plan(&cd.cfg, &cd.tier, &cd.budget, &inputs) {
            // 5. Local downgrade — emit the legacy label, surface the local hit.
            PlanOutcome::Local { reason } => {
                tracing::info!(
                    %rid,
                    reason = local_reason_label(reason),
                    "cd: prefill stays local"
                );
                local_status
            }
            // 6. Reject — park on Pending (no latch). The next poll re-runs the
            //    plan and commits Remote once the budget frees (the legacy
            //    re-poll-until-budget semantics).
            PlanOutcome::Reject => {
                tracing::info!(%rid, "cd: remote prefill rejected (budget); parking pending");
                *cell.lock().expect("search-status mutex poisoned") = MatchStatus::Pending;
                MatchStatus::Pending
            }
            // 7. Remote — the search-time commit (the budget reservation is HELD).
            PlanOutcome::Remote {
                full_block_external_tokens: fbet,
            } => self.cd_commit_remote(
                cd,
                search_id,
                req,
                local_hit,
                num_computed,
                fbet,
                prefix_blocks,
                local_blocks,
                cell,
                local_status,
            ),
        }
    }

    /// The Remote search-time commit: open + commit + make_available + finish the
    /// holder session, latch the per-request CD state, dispatch the
    /// remote-prefill request on the leader runtime, and surface the unified
    /// count. Any pre-latch failure releases the HELD budget and downgrades to
    /// `local_status`.
    #[allow(clippy::too_many_arguments)]
    fn cd_commit_remote(
        &self,
        cd: &CdRuntime,
        search_id: SearchId,
        req: &MatchWindow<'_>,
        local_hit: usize,
        num_computed: usize,
        fbet: usize,
        prefix_blocks: Vec<ImmutableBlock<G2>>,
        local_blocks: Vec<ImmutableBlock<G2>>,
        cell: &Arc<Mutex<MatchStatus>>,
        local_status: MatchStatus,
    ) -> MatchStatus {
        let rid = req.request_id;
        let bs = self.block_size;

        // 7a. The pre-cloned blocks must cover what the commit promises: the
        //     local-match blocks the matched span, the prefix blocks the whole
        //     `[0, computed)` window (re-checked post-reserve — the residency
        //     gate ran pre-plan, this arm guards the invariant at the commit).
        //     A mismatch is the safe-downgrade — release the budget, stay local.
        if local_blocks.len() != local_hit || prefix_blocks.len() != num_computed / bs {
            cd.budget.release(fbet);
            tracing::error!(
                %rid,
                local_got = local_blocks.len(),
                local_want = local_hit,
                prefix_got = prefix_blocks.len(),
                prefix_want = num_computed / bs,
                "cd: commit block count mismatch; downgrading to local"
            );
            return local_status;
        }
        let local_match_hashes: Vec<SequenceHash> = req.block_plhs[..local_hit].to_vec();

        // 7b. The commit set in absolute-position order: the vLLM-computed
        //     prefix `[0, computed)` (already G2-resident — the residency gate
        //     proved it), then the matched window span. Remote-search results
        //     have already landed at the matched terminal (the composer
        //     finalizes before Complete), so they count as part of the local
        //     match; nothing is deferred (`pending_hashes` empty — the ledger
        //     starts drained).
        let plan = RemoteCommitPlan {
            prefix_hashes: req.prefix_plhs.to_vec(),
            pending_hashes: Vec::new(),
            local_match_hashes,
        };

        // 7c. Open the holder session and publish the commit set up front. The
        //     initial availability set is prefix ++ local-match — the same
        //     absolute order as the commit set.
        let initial_blocks: Vec<ImmutableBlock<G2>> =
            prefix_blocks.into_iter().chain(local_blocks).collect();
        let session_id = uuid::Uuid::new_v4();
        let (session, ledger) = match open_and_commit(
            &cd.sessions,
            session_id,
            &plan,
            initial_blocks,
        ) {
            Ok(pair) => pair,
            Err(e) => {
                cd.budget.release(fbet);
                tracing::error!(%rid, error = %e, "cd: open_and_commit failed; downgrading to local");
                return local_status;
            }
        };

        // 7d. Latch the committed state — budget ownership transfers to the
        //     container on Ok. A duplicate cannot happen given the step-2 latch,
        //     but the pre-insert discipline still releases directly + closes.
        //     The full committed window `[0, unified)` is parked so the onboard
        //     fan-out can derive the remote slice `[local_hit, unified)`; the
        //     slice clamps to `block_plhs` so a short window can never panic.
        let unified = (fbet / bs) as u32;
        let window_end = (unified as usize).min(req.block_plhs.len());
        let window_hashes: Vec<SequenceHash> = req.block_plhs[..window_end].to_vec();
        let state = Arc::new(CdRequestState::for_remote_commit(
            fbet,
            num_computed,
            unified,
            local_hit as u32,
            window_hashes,
            search_id,
            session.clone(),
            ledger,
        ));
        if let Err(e) = cd.requests.insert(rid.clone(), Arc::clone(&state)) {
            cd.budget.release(fbet);
            session.close(Some("duplicate cd request".to_string()));
            tracing::error!(%rid, error = %e, "cd: duplicate request state; downgrading to local");
            return local_status;
        }

        // 7e. Dispatch the remote-prefill request on the leader runtime. On Err
        //     the budget stays HELD (released by the decline/evict/finish hooks);
        //     the failure is stashed for onboard to surface and the session closed.
        let dispatch = PrefillDispatch {
            request_id: rid.clone(),
            session_id,
            decode_endpoint: session.endpoint(),
            num_provided_tokens: num_computed + local_hit * bs,
            num_window_tokens: num_computed + fbet,
        };
        let plane = cd.plane.clone();
        let dispatch_state = Arc::clone(&state);
        let dispatch_session = session.clone();
        let dispatch_rid = rid.clone();
        self.leader.runtime().spawn(async move {
            if let Err(e) = plane.dispatch(dispatch).await {
                tracing::error!(rid = %dispatch_rid, error = %e, "cd: prefill dispatch failed");
                dispatch_state.stash_failure(format!("prefill dispatch failed: {e}"));
                dispatch_session.close(Some("prefill dispatch failed".to_string()));
            }
        });

        // 7f. Latch the unified count through the cell.
        let status = MatchStatus::Matched {
            hit_blocks: unified,
        };
        *cell.lock().expect("search-status mutex poisoned") = status;
        status
    }
}

// ============================================================================
// Conditional-disagg onboard fan-out (the USAA local/remote split)
// ============================================================================

impl LocalConnectorEngine {
    /// Onboard a latched conditional-disagg request through the local/remote
    /// fan-out. `external_ids` is the external dest slice (the computed prefix
    /// already excluded by `select_onboard_block_ids`); `unified` is the latched
    /// hit count the handle reported.
    ///
    /// Three guards bail to an immediately-failed action carrying the EXTERNAL
    /// slice only (reporting the computed prefix would force vLLM to recompute
    /// from token 0): a stashed pre-onboard failure (the dispatch-failed path), a
    /// dest count short of `unified` (the `select_onboard_block_ids` clamp — the
    /// legacy USAA-1 count bail), and a missing session. Otherwise it splits
    /// `external_ids` at `local_hit_blocks`, builds the remote pairs from
    /// `window_hashes[local_hit..unified]`, and spawns the driver: the local kick
    /// (`run_onboard` over `local_g1` only — the matched span counts now agree)
    /// then the remote pull pipeline. The single load terminal carries the merged
    /// outcome; `finish_load_action`'s CD hook releases the budget + tears the
    /// session down.
    fn cd_onboard(
        self: Arc<Self>,
        state: Arc<CdRequestState>,
        search_id: SearchId,
        request_id: RequestId,
        mut onboarding: OnboardingState,
        external_ids: Vec<BlockId>,
        unified: u32,
    ) -> OnboardHandle {
        let block_size = self.block_size;
        let unified = unified as usize;
        let local_hit = state.local_hit_blocks() as usize;
        let window = state.window_hashes();

        // a. Stash replay: a failure observed before onboard (e.g. the prefill
        //    dispatch resolved Err) surfaces here with the now-known dest ids.
        if state.pending_failure().is_some() {
            tracing::warn!(%request_id, "cd onboard: replaying pre-onboard failure stash");
            return self.mint_failed_onboard(request_id, external_ids, &state);
        }

        // b. Count / split guards (the legacy USAA-1 bails, never a panic).
        if external_ids.len() != unified || local_hit > unified || window.len() < unified {
            tracing::error!(
                %request_id,
                external = external_ids.len(),
                unified,
                local_hit,
                window = window.len(),
                "cd onboard: external/window split mismatch; failing the load"
            );
            return self.mint_failed_onboard(request_id, external_ids, &state);
        }

        let local_g1: Vec<BlockId> = external_ids[..local_hit].to_vec();
        let remote_g1: Vec<BlockId> = external_ids[local_hit..unified].to_vec();
        let remote_pairs: Vec<(SequenceHash, BlockId)> = window[local_hit..unified]
            .iter()
            .copied()
            .zip(remote_g1.iter().copied())
            .collect();

        // Clone the session Arc (the terminal hooks own teardown via
        // take_session — the driver must not consume it).
        let Some(session) = state.clone_session() else {
            tracing::error!(%request_id, "cd onboard: session already gone; failing the load");
            return self.mint_failed_onboard(request_id, external_ids, &state);
        };

        let staging_futs: Vec<_> = onboarding
            .shards
            .iter()
            .map(|shard| shard.find_session.wait_for_completion())
            .collect();

        let action_id = ActionId::new();
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        self.actions.insert(
            action_id,
            ActionRecord::new(request_id.clone(), Arc::downgrade(&cell)),
        );
        self.by_request
            .entry(request_id.clone())
            .or_default()
            .push(action_id);

        // Record the in-flight onboard's UNIFIED window (the local+remote
        // committed span) for the deferral guard, keyed by the DRIVING search
        // generation so the clear lands at the connector's release
        // (`release_search` on the parked handle's id). The bail paths above
        // route through `mint_failed_onboard`, which records nothing (born
        // terminal) — only this live fan-out records. `window.len() >=
        // unified` was proven by the split guard above.
        self.inflight
            .lock()
            .expect("inflight-guard mutex poisoned")
            .record(InflightKey::Search(search_id), window[..unified].to_vec());

        let handle_dest_ids = external_ids.clone();
        // The originating lifecycle's Arc, carried into the terminal so the CD
        // load-terminal hook tears down THIS lifecycle (never a stale map
        // re-fetch). See `CdRuntime::complete_load`.
        let terminal_state = Arc::clone(&state);
        let this = self.clone();
        self.leader.runtime().spawn(async move {
            let leader = this.leader.clone();

            // LOCAL kick: the matched span sources `local_g1` only. An empty
            // `local_g1` (zero local match) short-circuits to Complete in
            // run_onboard without awaiting the staging futures.
            let local_status =
                onboard::run_onboard(&leader, &mut onboarding, local_g1, staging_futs, block_size)
                    .await;
            onboarding.release_all(&leader);

            let outcome = if !matches!(local_status, ActionStatus::Complete) {
                // Local failed before any remote pull landed → nothing onboarded,
                // report the whole external slice failed.
                ActionStatus::Failed(ActionFailure::Partial {
                    block_ids: external_ids.clone(),
                })
            } else {
                // REMOTE pull pipeline over the parked CD session.
                let mut filled = std::collections::HashSet::new();
                match onboard::run_remote_onboard(
                    &leader,
                    &session,
                    &remote_pairs,
                    block_size,
                    &mut filled,
                )
                .await
                {
                    Ok(()) => ActionStatus::Complete,
                    Err(e) => {
                        tracing::error!(error = %e, %request_id, "cd remote onboard pipeline failed");
                        // The local span landed; only the remote dest ids whose
                        // hash never filled are reported failed.
                        let unfilled: Vec<BlockId> = remote_pairs
                            .iter()
                            .filter(|(hash, _)| !filled.contains(hash))
                            .map(|(_, g1)| *g1)
                            .collect();
                        ActionStatus::Failed(ActionFailure::Partial { block_ids: unfilled })
                    }
                }
            };

            this.finish_load_action(action_id, &request_id, outcome.clone(), external_ids);
            // CD load terminal, fired AFTER finish_load_action returns (lock-free)
            // against the originating lifecycle's Arc.
            if let Some(cd) = &this.cd {
                cd.complete_load(&request_id, &terminal_state, &outcome);
            }
        });

        let me: Arc<dyn LeaderEngine> = self;
        OnboardHandle::new(action_id, Arc::downgrade(&me), cell, handle_dest_ids)
    }

    /// Mint an in-flight action and resolve it IMMEDIATELY to
    /// `Failed(Partial { external_ids })`. The shared `finish_load_action` writes
    /// the cell and fires the worker sink with the failed dest ids; this then
    /// runs the CD load-terminal hook (budget release + session close) against
    /// the originating lifecycle's `state` Arc. Used by the onboard fan-out's
    /// pre-pull bail paths (stash replay, count mismatch, missing session).
    fn mint_failed_onboard(
        self: Arc<Self>,
        request_id: RequestId,
        external_ids: Vec<BlockId>,
        state: &Arc<CdRequestState>,
    ) -> OnboardHandle {
        let action_id = ActionId::new();
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        self.actions.insert(
            action_id,
            ActionRecord::new(request_id.clone(), Arc::downgrade(&cell)),
        );
        self.by_request
            .entry(request_id.clone())
            .or_default()
            .push(action_id);

        let outcome = ActionStatus::Failed(ActionFailure::Partial {
            block_ids: external_ids.clone(),
        });
        self.finish_load_action(
            action_id,
            &request_id,
            outcome.clone(),
            external_ids.clone(),
        );
        // CD load terminal against the originating lifecycle's Arc (lock-free,
        // post-guard). See `CdRuntime::complete_load`.
        if let Some(cd) = &self.cd {
            cd.complete_load(&request_id, state, &outcome);
        }

        let me: Arc<dyn LeaderEngine> = self;
        OnboardHandle::new(action_id, Arc::downgrade(&me), cell, external_ids)
    }
}

/// Clone the leading `want` matched G2 blocks across the terminal shards WITHOUT
/// consuming them. Returns `None` if a covering shard's blocks are not yet
/// readable or fewer than `want` are available. The shards cover only the
/// match WINDOW (they start at the computed prefix), so the leading `want`
/// blocks ARE the matched span regardless of the prefix; the prefix blocks a
/// CD commit additionally serves come from the G2 cache, not the shards.
fn clone_leading_g2_blocks(
    onboarding: &OnboardingState,
    want: usize,
) -> Option<Vec<ImmutableBlock<G2>>> {
    if want == 0 {
        return Some(Vec::new());
    }
    let mut out: Vec<ImmutableBlock<G2>> = Vec::with_capacity(want);
    for shard in &onboarding.shards {
        if out.len() >= want {
            break;
        }
        out.extend(shard.find_session.clone_g2_blocks()?);
    }
    if out.len() < want {
        return None;
    }
    out.truncate(want);
    Some(out)
}

/// Map a [`LocalReason`] to the legacy decode-CD metric label (the authoritative
/// table on `LocalReason`), keeping the wiring layer's tracing strings identical
/// to the legacy path.
fn local_reason_label(reason: LocalReason) -> &'static str {
    match reason {
        LocalReason::Policy => "local",
        LocalReason::BreakerHot => "remote_downgraded_breaker_hot",
        LocalReason::ZeroBlock => "remote_downgraded_zero_block",
        LocalReason::OverloadFallback => "remote_downgraded_overload",
    }
}

// ============================================================================
// Offload flush (driven by the worker forward-pass boundary)
// ============================================================================

impl LocalConnectorEngine {
    /// Flush every offload buffered under iteration `<= iteration`.
    ///
    /// Iteration-scoped: the leader's merge-await for pass `n` may land after
    /// pass `n+1`'s scheduler walk has already buffered new offloads, and
    /// those mid-pass G1 sources are still being written — entries stamped
    /// `> iteration` stay buffered for their own pass's flush. The `<=` sweep
    /// also drains stragglers from a pass whose flush never fired, gated on a
    /// LATER pass completing (safe: completed blocks are immutable).
    ///
    /// Mints the per-iteration forward-pass precondition event (the seam pins
    /// the mint to *finish*), submits each drained offload through the
    /// offload-submit seam gated on that precondition, spawns a per-handle
    /// completion driver (mirrors onboard's spawn), then triggers the event.
    ///
    /// P-B2 stand-in: the trigger is synchronous so a real `OffloadEngine`'s
    /// queued copy can proceed. P-D hands the event's ownership to the worker's
    /// real GPU forward-pass completion (`velo::Event::into_handle`) so the
    /// G1→G2 copy waits on the stream — Decision A's buffer→flush already
    /// prevents the eager mid-forward-pass enqueue; this is the second gate.
    fn flush_offloads(&self, iteration: usize) {
        let buffered: Vec<BufferedOffload> = {
            let mut buf = self
                .offload_buffer
                .lock()
                .expect("offload-buffer mutex poisoned");
            let (drain, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut *buf)
                .into_iter()
                .partition(|b| b.iteration <= iteration);
            *buf = keep;
            drain
        };
        if buffered.is_empty() {
            return;
        }

        // `Arc<Self>` for the spawned drivers (the `&self` receiver can't move).
        let Some(this) = self.weak_self.upgrade() else {
            return; // engine is tearing down
        };

        // Mint the precondition event for this iteration (see the P-B2 note).
        let event = match self.leader.messenger().events().new_event() {
            Ok(ev) => Some(ev),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    iteration,
                    "finish_forward_pass: forward-pass event mint failed; flushing without precondition"
                );
                None
            }
        };
        let precondition = event.as_ref().map(|ev| ev.handle());

        for b in buffered {
            let blocks = offload::build_external_blocks(&b.pairs);
            let action_id = b.action_id;
            let request_id = b.request_id;
            match self
                .offload_submit
                .submit_g1_to_g2(b.resource, blocks, precondition)
            {
                Ok(transfer) => {
                    let driver = this.clone();
                    self.leader.runtime().spawn(async move {
                        let outcome = offload::run_offload(transfer).await;
                        // Write terminal into the cell + notify with no engine
                        // lock held (per-action terminal flips the cell only —
                        // `mark_save_finished` is drain-driven).
                        driver.finish_save_action(action_id, &request_id, outcome);
                    });
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        %request_id,
                        "offload submit failed; marking action Failed(AllBlocks)"
                    );
                    this.finish_save_action(
                        action_id,
                        &request_id,
                        ActionStatus::Failed(ActionFailure::AllBlocks),
                    );
                }
            }
        }

        // Trigger the precondition (P-B2 stand-in — see the method docs). `Event`
        // is an RAII drop-guard that poisons waiters if neither triggered nor
        // handed off, so this consume is also what keeps a real queued copy from
        // failing on a poisoned precondition.
        if let Some(event) = event
            && let Err(e) = event.trigger()
        {
            tracing::warn!(
                error = %e,
                iteration,
                "finish_forward_pass: forward-pass event trigger failed"
            );
        }
    }
}

impl WorkerEngineDriver for LocalConnectorEngine {
    fn begin_forward_pass(&self, iteration: usize) {
        // Record the iteration; the precondition event is minted at finish (the
        // seam pins the mint there), so begin needs no event state.
        self.current_iteration.store(iteration, Ordering::Relaxed);
    }

    fn finish_forward_pass(&self, iteration: usize) {
        self.flush_offloads(iteration);
    }

    fn await_fence(&self, _token: FenceToken) {
        // The worker-side eviction-drain barrier is wired in P-E; the engine-side
        // fence terminal already fires `mark_fence_complete` via
        // `finish_{load,save}_action`. No P-B2 path drives this.
    }

    fn shutdown(&self) {
        // Drop buffered-but-unflushed offloads; their handles' cells stay
        // `Pending` and free by RAII on drop. Orderly teardown sequencing is P-D.
        self.offload_buffer
            .lock()
            .expect("offload-buffer mutex poisoned")
            .clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::offload::OffloadTransfer;
    use super::super::prefill::PrefillAcceptCore;
    use super::super::reconcile::OnboardingShard;
    use super::*;
    use crate::leader::{
        AsyncSessionResult, FindMatchesOptions, FindMatchesResult, MatchBreakdown,
        OnboardingStatus, ReadyResult, SessionId,
    };
    use crate::offload::{ExternalBlock, TransferStatus};
    use kvbm_protocols::connector::NoopWorkerSink;
    use kvbm_protocols::connector::{LoadOutcome, ResourceOnboard, SaveOutcome};
    use std::sync::Mutex as StdMutex;
    use tokio::sync::{Mutex as TokioMutex, watch};
    use uuid::Uuid;

    // ----- offload-submission double (OffloadEngine needs GPU/velo) -----

    /// Records each `submit_g1_to_g2` call's external blocks (as `(block_id,
    /// sequence_hash)` — the post-reversal mapping) and whether a precondition
    /// was provided, then hands back a transfer pre-set to a configured terminal.
    struct MockOffloadSubmit {
        submits: StdMutex<Vec<Vec<(BlockId, SequenceHash)>>>,
        resources: StdMutex<Vec<Option<kvbm_common::LogicalResourceId>>>,
        preconditions: StdMutex<Vec<bool>>,
        terminal: TransferStatus,
        failed: Vec<BlockId>,
    }
    impl MockOffloadSubmit {
        fn new(terminal: TransferStatus, failed: Vec<BlockId>) -> Arc<Self> {
            Arc::new(Self {
                submits: StdMutex::new(Vec::new()),
                resources: StdMutex::new(Vec::new()),
                preconditions: StdMutex::new(Vec::new()),
                terminal,
                failed,
            })
        }
        fn submit_count(&self) -> usize {
            self.submits.lock().unwrap().len()
        }
        fn last_submit(&self) -> Vec<(BlockId, SequenceHash)> {
            self.submits
                .lock()
                .unwrap()
                .last()
                .cloned()
                .unwrap_or_default()
        }
        fn last_resource(&self) -> Option<kvbm_common::LogicalResourceId> {
            self.resources.lock().unwrap().last().copied().flatten()
        }
        fn last_precondition_present(&self) -> bool {
            self.preconditions
                .lock()
                .unwrap()
                .last()
                .copied()
                .unwrap_or(false)
        }
    }
    impl OffloadSubmit for MockOffloadSubmit {
        fn supports_resource(&self, _resource: kvbm_common::LogicalResourceId) -> bool {
            true
        }

        fn submit_g1_to_g2(
            &self,
            resource: Option<kvbm_common::LogicalResourceId>,
            blocks: Vec<ExternalBlock<crate::G1>>,
            precondition: Option<velo::EventHandle>,
        ) -> Result<Box<dyn OffloadTransfer>> {
            self.resources.lock().unwrap().push(resource);
            self.submits.lock().unwrap().push(
                blocks
                    .iter()
                    .map(|b| (b.block_id, b.sequence_hash))
                    .collect(),
            );
            self.preconditions
                .lock()
                .unwrap()
                .push(precondition.is_some());
            Ok(Box::new(MockTransfer {
                status: self.terminal,
                failed: self.failed.clone(),
            }))
        }
    }

    /// A pre-terminal [`OffloadTransfer`] double.
    struct MockTransfer {
        status: TransferStatus,
        failed: Vec<BlockId>,
    }
    impl OffloadTransfer for MockTransfer {
        fn status(&self) -> TransferStatus {
            self.status
        }
        fn completed_blocks(&self) -> Vec<BlockId> {
            Vec::new()
        }
        fn failed_blocks(&self) -> Vec<BlockId> {
            self.failed.clone()
        }
        fn wait_terminal(&self) -> futures::future::BoxFuture<'static, ()> {
            // Already terminal — resolve immediately.
            Box::pin(async {})
        }
    }

    async fn wait_offload_complete(handle: &OffloadHandle) {
        for _ in 0..200 {
            if handle.is_complete() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("offload did not reach a terminal state");
    }

    const BS: usize = 4;

    // ----- `&dyn Leader` stub + builders (mirrors reconcile_tests) -----

    struct TestLeader {
        responses: StdMutex<Vec<FindMatchesResult>>,
        /// Records the hashes each `find_matches_with_options` call was given,
        /// so the rebase test can assert the *absolute* slice content.
        calls: StdMutex<Vec<Vec<SequenceHash>>>,
    }

    impl TestLeader {
        fn new(responses: Vec<FindMatchesResult>) -> Self {
            Self {
                responses: StdMutex::new(responses),
                calls: StdMutex::new(Vec::new()),
            }
        }

        fn call_hashes(&self) -> Vec<Vec<SequenceHash>> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl Leader for TestLeader {
        fn find_matches_with_options(
            &self,
            sequence_hashes: &[SequenceHash],
            _options: FindMatchesOptions,
        ) -> Result<FindMatchesResult> {
            self.calls.lock().unwrap().push(sequence_hashes.to_vec());
            let mut q = self.responses.lock().unwrap();
            assert!(
                !q.is_empty(),
                "TestLeader: unexpected find_matches_with_options call ({} hashes)",
                sequence_hashes.len()
            );
            Ok(q.remove(0))
        }
    }

    /// Distinct, identifiable hash for absolute block `i` (current = position = i).
    fn h(i: u64) -> SequenceHash {
        SequenceHash::new(i, None, i)
    }

    /// `block_plhs` for absolute blocks `[start .. start + n)`.
    fn plhs_for(start: u64, n: u64) -> Vec<SequenceHash> {
        (start..start + n).map(h).collect()
    }

    fn ready_zero() -> FindMatchesResult {
        FindMatchesResult::Ready(ReadyResult::new(vec![], MatchBreakdown::default()))
    }

    fn complete_async(matched: usize) -> FindMatchesResult {
        let (tx, rx) = watch::channel(OnboardingStatus::Complete {
            matched_blocks: matched,
        });
        drop(tx);
        FindMatchesResult::AsyncSession(AsyncSessionResult::new(
            SessionId::from(Uuid::nil()),
            rx,
            Arc::new(TokioMutex::new(Some(Vec::new()))),
            Arc::new(TokioMutex::new(MatchBreakdown::default())),
        ))
    }

    fn pending_async() -> (FindMatchesResult, watch::Sender<OnboardingStatus>) {
        let (tx, rx) = watch::channel(OnboardingStatus::Searching);
        let session = AsyncSessionResult::new(
            SessionId::from(Uuid::nil()),
            rx,
            Arc::new(TokioMutex::new(None)),
            Arc::new(TokioMutex::new(MatchBreakdown::default())),
        );
        (FindMatchesResult::AsyncSession(session), tx)
    }

    fn async_with_session_id(sid: Uuid) -> FindMatchesResult {
        let (tx, rx) = watch::channel(OnboardingStatus::Complete { matched_blocks: 1 });
        drop(tx);
        FindMatchesResult::AsyncSession(AsyncSessionResult::new(
            SessionId::from(sid),
            rx,
            Arc::new(TokioMutex::new(Some(Vec::new()))),
            Arc::new(TokioMutex::new(MatchBreakdown::default())),
        ))
    }

    fn req(request_id: &str, computed_blocks: u32, plhs: Vec<SequenceHash>) -> SearchRequest {
        SearchRequest {
            request_id: request_id.to_string(),
            prefix_plhs: Vec::new(),
            block_plhs: plhs,
            computed_blocks,
        }
    }

    /// [`req`] carrying the computed prefix's hashes (the CD prefix-capture
    /// input shape); `computed_blocks` is implied by the prefix length.
    fn req_with_prefix(
        request_id: &str,
        prefix_plhs: Vec<SequenceHash>,
        plhs: Vec<SequenceHash>,
    ) -> SearchRequest {
        SearchRequest {
            request_id: request_id.to_string(),
            computed_blocks: prefix_plhs.len() as u32,
            prefix_plhs,
            block_plhs: plhs,
        }
    }

    // ----- search / refresh core tests (no InstanceLeader needed) -----

    #[test]
    fn search_empty_plhs_is_no_match() {
        // The empty-window early return runs BEFORE the cd_enabled branch, so it
        // is NoMatch for both flag values (no shard issued either way).
        let leader = TestLeader::new(vec![]);
        let r = req("rq", 0, vec![]);
        for cd_enabled in [false, true] {
            assert!(matches!(
                perform_search(&leader, BS, true, cd_enabled, &MatchWindow::of(&r)).unwrap(),
                SearchInit::NoMatch
            ));
        }
        assert!(leader.call_hashes().is_empty());
    }

    #[test]
    fn search_terminal_zero_match_is_no_match() {
        // A terminal Ready shard with no matched blocks. With CD off no handle is
        // minted (NoMatch); with CD on a zero-block `Active` is minted so
        // `cd_interpose` gets its chance to send the cold prompt Remote.
        let no_cd = TestLeader::new(vec![ready_zero()]);
        let r = req("rq", 0, plhs_for(0, 5));
        assert!(matches!(
            perform_search(&no_cd, BS, true, false, &MatchWindow::of(&r)).unwrap(),
            SearchInit::NoMatch
        ));

        let with_cd = TestLeader::new(vec![ready_zero()]);
        match perform_search(&with_cd, BS, true, true, &MatchWindow::of(&r)).unwrap() {
            SearchInit::Active {
                status,
                pending,
                hit_blocks,
                ..
            } => {
                assert!(!pending);
                assert_eq!(hit_blocks, 0);
                assert_eq!(status, MatchStatus::Matched { hit_blocks: 0 });
            }
            SearchInit::NoMatch => panic!("cd_enabled must mint a zero-block latch"),
        }
    }

    #[test]
    fn search_immediate_hit_reports_matched() {
        // Terminal match of 3 of 5 queried blocks → Hit { 3 } (ImmutableBlock<G2>
        // is not constructible in tests, so an already-`Complete` async session
        // stands in for an immediate Ready hit — same `compute_outcome` path).
        let leader = TestLeader::new(vec![complete_async(3)]);
        let r = req("rq", 0, plhs_for(0, 5));
        match perform_search(&leader, BS, true, false, &MatchWindow::of(&r)).unwrap() {
            SearchInit::Active {
                status,
                pending,
                hit_blocks,
                ..
            } => {
                assert!(!pending);
                assert_eq!(hit_blocks, 3);
                assert_eq!(status, MatchStatus::Matched { hit_blocks: 3 });
            }
            SearchInit::NoMatch => panic!("expected a hit"),
        }
    }

    #[test]
    fn search_async_pending_then_refresh_becomes_terminal() {
        // Initial poll: pending async → Pending. After the watch flips to
        // terminal, a refresh (Case A) projects the match → Matched.
        let (pending, tx) = pending_async();
        let leader = TestLeader::new(vec![pending]);
        let r = req("rq", 0, plhs_for(0, 5));

        let (mut state, mut buffer) =
            match perform_search(&leader, BS, true, false, &MatchWindow::of(&r)).unwrap() {
                SearchInit::Active {
                    state,
                    buffer,
                    status,
                    pending,
                    ..
                } => {
                    assert!(pending);
                    assert_eq!(status, MatchStatus::Pending);
                    (state, buffer)
                }
                SearchInit::NoMatch => panic!("expected pending"),
            };

        // The driver: the underlying find session reaches terminal.
        tx.send(OnboardingStatus::Complete { matched_blocks: 4 })
            .unwrap();

        // Case A refresh issues no new shard.
        let refresh_leader = TestLeader::new(vec![]);
        let status = perform_refresh(
            &refresh_leader,
            BS,
            true,
            &mut state,
            &mut buffer,
            &MatchWindow::of(&r),
        )
        .unwrap();
        assert_eq!(status, MatchStatus::Matched { hit_blocks: 4 });
    }

    #[test]
    fn rebase_case_c_prepend_uses_absolute_coordinates() {
        // Reproducer-first: this FAILS on a naive impl that treats `block_plhs`
        // as a 0-based vector. First poll cb=10 over absolute [10..15); a refresh
        // drops cb to 8, whose suffix-only `block_plhs` covers absolute [8..15).
        // The correct (absolute) impl must (a) issue the prepend over the real
        // PREFIX hashes h(8),h(9) — not the suffix front — and (b) report a
        // (15-8)*BS match. A no-rebase impl computes new_start_block = 8 and
        // slices a length-7 `block_plhs` at [8..10) → out-of-bounds panic.
        let first = req("rq", 10, plhs_for(10, 5)); // abs [10..15)
        let search_leader = TestLeader::new(vec![complete_async(5)]);
        let (mut state, mut buffer) = match perform_search(
            &search_leader,
            BS,
            true,
            false,
            &MatchWindow::of(&first),
        )
        .unwrap()
        {
            SearchInit::Active {
                state,
                buffer,
                hit_blocks,
                ..
            } => {
                assert_eq!(hit_blocks, 5);
                (state, buffer)
            }
            SearchInit::NoMatch => panic!("expected a hit"),
        };
        // Initial shard was issued over absolute [10..15).
        assert_eq!(search_leader.call_hashes(), vec![plhs_for(10, 5)]);

        // Refresh: cb drops to 8; suffix now covers abs [8..15).
        let refresh = req("rq", 8, plhs_for(8, 7));
        let refresh_leader = TestLeader::new(vec![complete_async(2)]); // prepend [8..10)
        let status = perform_refresh(
            &refresh_leader,
            BS,
            true,
            &mut state,
            &mut buffer,
            &MatchWindow::of(&refresh),
        )
        .unwrap();

        // Teeth: the prepend find received exactly the PREFIX hashes h(8),h(9)
        // (proves the absolute buffer + merge, not a 0-based suffix slice).
        assert_eq!(refresh_leader.call_hashes(), vec![vec![h(8), h(9)]]);
        // Full contiguous match across [8..15) → 7 blocks.
        assert_eq!(status, MatchStatus::Matched { hit_blocks: 7 });
        assert_eq!(state.shards.len(), 2);
        assert_eq!(state.shards[0].start_block, 8);
    }

    // ----- engine-level tests (real InstanceLeader; default features) -----

    async fn build_test_leader() -> Result<InstanceLeader> {
        use crate::testing::{managers::TestManagerBuilder, messenger::create_messenger_tcp};
        use kvbm_logical::blocks::BlockRegistry;

        let messenger = create_messenger_tcp().await?;
        let registry = BlockRegistry::builder().build();
        let g2 = Arc::new(
            TestManagerBuilder::<crate::G2>::new()
                .block_count(2)
                .block_size(BS)
                .registry(registry.clone())
                .build(),
        );
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry)
            .g2_manager(g2)
            .build()
    }

    /// Recording worker sink — captures `mark_load_finished` /
    /// `mark_save_finished` / `mark_fence_complete` calls so the offload tests
    /// can prove exactly one save_finished (from `commit`) and zero from any
    /// per-action terminal.
    struct RecordingSink {
        loads: StdMutex<Vec<(RequestId, LoadOutcome)>>,
        saves: StdMutex<Vec<(RequestId, SaveOutcome)>>,
        fences: StdMutex<Vec<FenceToken>>,
    }
    impl RecordingSink {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                loads: StdMutex::new(Vec::new()),
                saves: StdMutex::new(Vec::new()),
                fences: StdMutex::new(Vec::new()),
            })
        }
        fn loads(&self) -> Vec<(RequestId, LoadOutcome)> {
            self.loads.lock().unwrap().clone()
        }
        fn saves(&self) -> Vec<(RequestId, SaveOutcome)> {
            self.saves.lock().unwrap().clone()
        }
        fn fences(&self) -> Vec<FenceToken> {
            self.fences.lock().unwrap().clone()
        }
    }
    impl EngineWorkerSink for RecordingSink {
        fn mark_load_finished(&self, req: &RequestId, outcome: LoadOutcome) {
            self.loads.lock().unwrap().push((req.clone(), outcome));
        }
        fn mark_save_finished(&self, req: &RequestId, outcome: SaveOutcome) {
            self.saves.lock().unwrap().push((req.clone(), outcome));
        }
        fn mark_fence_complete(&self, token: FenceToken) {
            self.fences.lock().unwrap().push(token);
        }
    }

    async fn wait_complete(handle: &OnboardHandle) {
        for _ in 0..200 {
            if handle.is_complete() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("onboard did not reach a terminal state");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn release_search_releases_session() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sid = Uuid::new_v4();
        leader.insert_test_session_marker(sid);
        assert!(leader.has_session(sid));

        let engine = LocalConnectorEngine::new(leader.clone(), NoopWorkerSink::new(), BS, true);

        // Inject a search whose single shard holds an async session id == sid.
        let search_id = SearchId::new();
        let shard = OnboardingShard {
            start_block: 0,
            num_queried_blocks: 1,
            find_session: async_with_session_id(sid),
        };
        engine.searches.insert(
            search_id,
            SearchState {
                request_id: "rq".into(),
                status: Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 })),
                onboarding: OnboardingState::new(0, BS + 1, shard),
                buffer: Vec::new(),
            },
        );

        engine.release_search(&search_id);
        assert!(
            !leader.has_session(sid),
            "release_search must release_session"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn onboard_zero_blocks_completes() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        // A `Matched { 0 }` search → onboard moves no blocks → immediate Done.
        let search_id = SearchId::new();
        let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 0 }));
        let shard = OnboardingShard {
            start_block: 0,
            num_queried_blocks: 0,
            find_session: ready_zero(),
        };
        engine.searches.insert(
            search_id,
            SearchState {
                request_id: "rq".into(),
                status: cell.clone(),
                onboarding: OnboardingState::new(0, 1, shard),
                buffer: Vec::new(),
            },
        );
        let onboard = engine
            .clone()
            .local_onboard(search_id, &[10, 11, 12])
            .unwrap();
        wait_complete(&onboard).await;

        assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
        assert_eq!(sink.loads(), vec![("rq".into(), LoadOutcome::Done)]);
        // The search entry was consumed by onboard.
        assert!(engine.searches.get(&search_id).is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn onboard_undeliverable_blocks_fails_and_retains() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        // Matched 1 block, but the terminal shard carries no G2 payload, so the
        // G2 collect bails → Failed(AllBlocks). (No parallel worker needed: the
        // failure precedes the transfer dispatch.)
        let search_id = SearchId::new();
        let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 }));
        let shard = OnboardingShard {
            start_block: 0,
            num_queried_blocks: 1,
            find_session: complete_async(1),
        };
        engine.searches.insert(
            search_id,
            SearchState {
                request_id: "rq".into(),
                status: cell.clone(),
                onboarding: OnboardingState::new(0, BS + 1, shard),
                buffer: Vec::new(),
            },
        );
        let onboard = engine.clone().local_onboard(search_id, &[10]).unwrap();
        wait_complete(&onboard).await;

        // A total load failure resolves to the CONCRETE dest ids (the external
        // G1 blocks handed to `onboard`): vLLM's invalid-block reporting is by
        // block_id, and only the engine still knows the dest set. An empty
        // failed set would finish the request's recv with nothing invalidated
        // (`LoadOutcome` has no id-less failure for exactly this reason).
        let failed = LoadOutcome::FailedPartial {
            block_ids: vec![10],
        };
        assert_eq!(onboard.outcome(), Some(failed.clone()));
        assert_eq!(sink.loads(), vec![("rq".into(), failed)]);
        // While the handle is live, its completion cell carries the failure, so
        // the by-id poll path resolves it through the engine's `Weak`.
        assert_eq!(
            engine.poll_action(onboard.id()),
            ActionStatus::Failed(kvbm_protocols::connector::ActionFailure::Partial {
                block_ids: vec![10]
            })
        );
        Ok(())
    }

    /// Production-path boundedness for terminal onboards. Pre-fix, a terminal
    /// onboard left a strong `ActionRecord` in `actions` forever on the in-process
    /// path: the connector reads `handle.outcome()` (a local cell read) and never
    /// calls `poll_action` (the only pruner then), so the by-id key — and its
    /// `by_request` link — leaked once a real caller was wired. With RAII
    /// `OnboardHandle::drop -> release_action` (the action analogue of
    /// `release_search`), dropping each terminal handle prunes BOTH maps.
    ///
    /// This proves the fix **without calling `poll_action`**: `finish_load_action`
    /// never removes from `actions`, so `actions.len() == 0` can only be reached by
    /// the handle Drop -> `release_action` path. If these assertions held only
    /// after a `poll_action` self-prune, the RAII fix would be wrong.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_then_dropped_onboards_retain_no_strong_cell() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        const N: usize = 8;
        for _ in 0..N {
            // Same shape as `onboard_undeliverable_blocks_fails_and_retains`: a
            // Matched(1) search whose terminal shard carries no G2 payload → the
            // onboard fails with `Failed(AllBlocks)` (resolved to dest ids at
            // the terminal).
            let search_id = SearchId::new();
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 }));
            let shard = OnboardingShard {
                start_block: 0,
                num_queried_blocks: 1,
                find_session: complete_async(1),
            };
            engine.searches.insert(
                search_id,
                SearchState {
                    request_id: "rq".into(),
                    status: cell.clone(),
                    onboarding: OnboardingState::new(0, BS + 1, shard),
                    buffer: Vec::new(),
                },
            );
            let onboard = engine.clone().local_onboard(search_id, &[10]).unwrap();
            wait_complete(&onboard).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![10]
                })
            );

            // Drop the terminal handle. Its RAII `Drop -> release_action` is the
            // ONLY pruner exercised here — no `poll_action` call anywhere.
            drop(onboard);
        }

        // Teeth (no `poll_action` involved): handle Drop alone kept both maps
        // bounded. `actions` can only reach 0 via `release_action`, since
        // `finish_load_action` never removes from it — so this isolates the RAII fix.
        assert_eq!(
            engine.actions.len(),
            0,
            "release_action (handle Drop) must prune the actions map"
        );
        assert!(
            engine.by_request.is_empty(),
            "release_action (handle Drop) must scrub the by_request index"
        );
        let live_cells = engine
            .actions
            .iter()
            .filter(|r| r.cell.strong_count() > 0)
            .count();
        assert_eq!(live_cells, 0, "no strong completion cell may remain");
        Ok(())
    }

    /// Regression (stop-hook): a request evicted with TWO in-flight actions must
    /// not complete its fence until BOTH drain — the old code fired
    /// `mark_fence_complete` on the FIRST drain, letting the worker reuse G1 blocks
    /// while the second transfer was still running.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eviction_fence_completes_only_after_all_armed_actions_drain() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        // Two in-flight actions for one request. Hold the cells alive (as live
        // handles would) so `evict`'s pending check sees them `Pending`.
        let req: RequestId = "rq".into();
        let (a1, a2) = (ActionId::new(), ActionId::new());
        let cell1 = Arc::new(Mutex::new(ActionStatus::Pending));
        let cell2 = Arc::new(Mutex::new(ActionStatus::Pending));
        engine
            .actions
            .insert(a1, ActionRecord::new(req.clone(), Arc::downgrade(&cell1)));
        engine
            .actions
            .insert(a2, ActionRecord::new(req.clone(), Arc::downgrade(&cell2)));
        engine.by_request.insert(req.clone(), vec![a1, a2]);

        let fence = engine.evict(&req).fence;
        assert!(
            !fence.per_worker.is_empty(),
            "evict must return fence tokens for armed actions"
        );
        assert!(
            sink.fences().is_empty(),
            "no fence completes at eviction time"
        );

        // First action drains: the fence must NOT complete yet (the shared barrier
        // still holds the second action's clone). This is the regression point.
        engine.finish_load_action(a1, &req, ActionStatus::Complete, Vec::new());
        assert!(
            sink.fences().is_empty(),
            "fence must NOT complete after only the FIRST armed action drains"
        );
        assert!(
            sink.loads().is_empty(),
            "an armed (cancel-for-emission) action fires no mark_load_finished"
        );

        // Last action drains: the barrier's final clone drops → fence completes
        // once, one token per worker.
        engine.finish_load_action(a2, &req, ActionStatus::Complete, Vec::new());
        assert_eq!(
            sink.fences().len(),
            fence.per_worker.len(),
            "fence completes exactly once after the LAST armed action drains"
        );
        Ok(())
    }

    /// A fenced OFFLOAD (save) action resolving `Cancelled` drives fence
    /// completion through the save terminal — symmetric with the fenced-LOAD
    /// path. `evict` arms the fence over an in-flight save exactly as over a
    /// load, and `finish_save_action` is status-agnostic, so the Cancelled fold
    /// (`project_offload_status(Cancelled, ..) == Complete`) reaches the fence
    /// via the same last-clone-drop. A fenced save fires NO `mark_save_finished`,
    /// mirroring the fenced-load suppression of `mark_load_finished`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evicted_offload_with_cancelled_status_completes_fence_and_folds_to_complete()
    -> Result<()> {
        // The fold arm itself: a cancelled save is a drained best-effort save.
        assert_eq!(
            offload::project_offload_status(TransferStatus::Cancelled, vec![]),
            ActionStatus::Complete,
            "Cancelled folds to Complete (the landed blocks live on in G2)"
        );

        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        // One in-flight save action for one request. Hold the cell alive so
        // `evict`'s pending check sees it `Pending` and arms the fence.
        let req: RequestId = "rq".into();
        let a1 = ActionId::new();
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        engine
            .actions
            .insert(a1, ActionRecord::new(req.clone(), Arc::downgrade(&cell)));
        engine.by_request.insert(req.clone(), vec![a1]);

        let fence = engine.evict(&req).fence;
        assert!(
            !fence.per_worker.is_empty(),
            "evict must arm a fence over the in-flight save"
        );
        assert!(
            sink.fences().is_empty(),
            "no fence completes at eviction time"
        );

        // The lone save action drains via a Cancelled-folded terminal: the
        // fence's last barrier clone drops → the fence completes once.
        engine.finish_save_action(
            a1,
            &req,
            offload::project_offload_status(TransferStatus::Cancelled, vec![]),
        );
        assert_eq!(
            sink.fences().len(),
            fence.per_worker.len(),
            "the cancelled save terminal completes the fence exactly once"
        );
        assert!(
            sink.saves().is_empty(),
            "a fenced save action fires NO mark_save_finished"
        );
        Ok(())
    }

    /// The leader-side fence handle observes the SAME barrier the worker tokens
    /// ride: it stays incomplete while any armed action is still draining and
    /// flips exactly when the last one drains (alongside the worker pushes).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eviction_leader_handle_completes_on_last_armed_drain() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let req: RequestId = "rq".into();
        let (a1, a2) = (ActionId::new(), ActionId::new());
        let cell1 = Arc::new(Mutex::new(ActionStatus::Pending));
        let cell2 = Arc::new(Mutex::new(ActionStatus::Pending));
        engine
            .actions
            .insert(a1, ActionRecord::new(req.clone(), Arc::downgrade(&cell1)));
        engine
            .actions
            .insert(a2, ActionRecord::new(req.clone(), Arc::downgrade(&cell2)));
        engine.by_request.insert(req.clone(), vec![a1, a2]);

        let outcome = engine.evict(&req);
        let handle = outcome
            .handle
            .expect("a barrier was minted, so the leader gets its handle");
        assert!(
            !handle.is_complete(),
            "the handle must not complete at eviction time"
        );

        engine.finish_load_action(a1, &req, ActionStatus::Complete, Vec::new());
        assert!(
            !handle.is_complete(),
            "the handle must NOT complete after only the FIRST armed action drains"
        );

        engine.finish_load_action(a2, &req, ActionStatus::Complete, Vec::new());
        assert!(
            handle.is_complete(),
            "the handle completes when the LAST armed action drains"
        );
        assert_eq!(
            sink.fences().len(),
            outcome.fence.per_worker.len(),
            "the worker tokens completed alongside the leader cell"
        );
        Ok(())
    }

    /// Evicting a request with nothing in flight mints no barrier: the fence
    /// carries no tokens and the leader gets no handle to poll.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evict_with_nothing_pending_returns_empty_fence_and_no_handle() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let outcome = engine.evict(&"idle".to_string());
        assert_eq!(outcome.fence.request_id, "idle");
        assert!(
            outcome.fence.per_worker.is_empty(),
            "no armed action — no tokens"
        );
        assert!(outcome.handle.is_none(), "no barrier — no leader handle");
        assert!(sink.fences().is_empty());
        Ok(())
    }

    /// `finish_load_action` itself resolves a total failure to the dest set the
    /// signature demands — the resolution is part of the terminal, not one
    /// caller's discipline — so the cell (handle/by-id reads) and the sink
    /// projection both name the concrete dest ids, never an id-less failure.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn load_terminal_resolves_total_failure_to_dest_ids() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let req: RequestId = "rq".into();
        let a1 = ActionId::new();
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        engine
            .actions
            .insert(a1, ActionRecord::new(req.clone(), Arc::downgrade(&cell)));
        engine.by_request.insert(req.clone(), vec![a1]);

        engine.finish_load_action(
            a1,
            &req,
            ActionStatus::Failed(ActionFailure::AllBlocks),
            vec![3, 5],
        );

        let resolved = LoadOutcome::FailedPartial {
            block_ids: vec![3, 5],
        };
        assert_eq!(sink.loads(), vec![(req, resolved)]);
        assert_eq!(
            *cell.lock().expect("cell"),
            ActionStatus::Failed(ActionFailure::Partial {
                block_ids: vec![3, 5]
            })
        );
        Ok(())
    }

    /// Regression (Codex): if a fenced action's handle drops BEFORE its driver's
    /// terminal, `release_action` must DEFER record removal — else dropping the live
    /// fence clone early orphans the fence (worker hangs) or completes it before the
    /// transfer drains. The driver's terminal both completes the fence and removes
    /// the deferred record.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eviction_fence_survives_handle_drop_before_terminal() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let req: RequestId = "rq".into();
        let a1 = ActionId::new();
        let cell1 = Arc::new(Mutex::new(ActionStatus::Pending));
        engine
            .actions
            .insert(a1, ActionRecord::new(req.clone(), Arc::downgrade(&cell1)));
        engine.by_request.insert(req.clone(), vec![a1]);

        let fence = engine.evict(&req).fence;
        assert!(!fence.per_worker.is_empty());

        // Handle drops before the driver terminal: removal must be deferred so the
        // live fence clone survives, and the fence must NOT complete yet.
        engine.release_action(&a1);
        assert!(
            engine.actions.contains_key(&a1),
            "release_action must defer removal of a fence-armed action"
        );
        assert!(
            sink.fences().is_empty(),
            "handle drop must not complete the fence early"
        );

        // Driver finally reaches terminal: completes the fence AND removes the
        // deferred record.
        engine.finish_load_action(a1, &req, ActionStatus::Complete, Vec::new());
        assert_eq!(
            sink.fences().len(),
            fence.per_worker.len(),
            "fence completes on the driver terminal even though the handle already dropped"
        );
        assert!(
            !engine.actions.contains_key(&a1),
            "the deferred record is removed by its terminal"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn onboard_against_pending_search_errors() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let engine = LocalConnectorEngine::new(leader.clone(), NoopWorkerSink::new(), BS, true);

        let search_id = SearchId::new();
        let cell = Arc::new(Mutex::new(MatchStatus::Pending));
        engine.searches.insert(
            search_id,
            SearchState {
                request_id: "rq".into(),
                status: cell.clone(),
                onboarding: OnboardingState::new(
                    0,
                    BS + 1,
                    OnboardingShard {
                        start_block: 0,
                        num_queried_blocks: 1,
                        find_session: ready_zero(),
                    },
                ),
                buffer: Vec::new(),
            },
        );
        assert_eq!(
            engine.clone().local_onboard(search_id, &[10]).unwrap_err(),
            LeaderEngineError::SearchNotMatched
        );
        Ok(())
    }

    // ----- in-flight onboard guard (record-at-mint / clear-at-release) -----

    /// Plain-local `FindBlocksRequest` over `hashes` (fresh poll shape).
    fn fb(request_id: &str, hashes: Vec<SequenceHash>, total_tokens: usize) -> FindBlocksRequest {
        FindBlocksRequest {
            request_id: request_id.to_string(),
            sequence_hashes: Arc::from(hashes),
            num_computed_tokens: 0,
            total_tokens,
            transfer_params: None,
        }
    }

    /// The core clear mutant-kill, inverted from the earlier terminal-clear
    /// scheme: the engine action terminal does NOT clear the guard — the
    /// lifecycle RELEASE does. Removing the `release_search` clear leaks this
    /// entry forever (the leak tests fail); restoring a terminal clear would
    /// fail the still-recorded assertion.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inflight_release_search_clears_what_terminal_does_not() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let req: RequestId = "rq".into();
        let sid = SearchId::new();
        engine
            .inflight
            .lock()
            .unwrap()
            .record(InflightKey::Search(sid), vec![h(0), h(1)]);
        {
            let g = engine.inflight.lock().unwrap();
            assert!(g.overlaps(&[h(1)]), "recorded window overlaps");
            assert_eq!(g.len(), 2);
        }

        engine.finish_load_action(ActionId::new(), &req, ActionStatus::Complete, vec![10]);
        assert!(
            engine.inflight.lock().unwrap().overlaps(&[h(0), h(1)]),
            "the action terminal must NOT clear the lifecycle's record"
        );

        engine.release_search(&sid);
        let g = engine.inflight.lock().unwrap();
        assert!(g.is_empty(), "release_search clears the lifecycle's record");
        assert!(!g.overlaps(&[h(0), h(1)]));
        Ok(())
    }

    /// THE deferral-outlives-terminal discriminator. Mint kind 1 (local
    /// onboard): the USAA-time mint records the matched window
    /// `buffer[computed .. computed + hit]` keyed by the search generation.
    /// After the load's `finish_load_action` terminal but BEFORE the handle
    /// release, an overlapping fresh `find_blocks` STILL defers (the loaded
    /// blocks are not yet vLLM-registered); after `release_search` it
    /// proceeds.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inflight_local_onboard_defers_past_terminal_until_release() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let (find_session, _tx) = pending_async(); // held: the driver parks here
        let search_id = SearchId::new();
        let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 }));
        engine.searches.insert(
            search_id,
            SearchState {
                request_id: "rq".into(),
                status: cell.clone(),
                onboarding: OnboardingState::new(
                    0,
                    BS + 1,
                    OnboardingShard {
                        start_block: 0,
                        num_queried_blocks: 1,
                        find_session,
                    },
                ),
                buffer: vec![h(7)],
            },
        );
        let onboard = engine.clone().local_onboard(search_id, &[10]).unwrap();
        // Record-at-mint: the synchronous mint recorded the hit window.
        {
            let g = engine.inflight.lock().unwrap();
            assert!(g.overlaps(&[h(7)]), "the matched block hash is recorded");
            assert!(!g.overlaps(&[h(8)]), "an unmatched hash is not");
            assert_eq!(g.len(), 1);
        }

        // Drive the terminal the parked driver would eventually reach; the
        // entry SURVIVES it.
        let req: RequestId = "rq".into();
        engine.finish_load_action(*onboard.id(), &req, ActionStatus::Complete, vec![10]);
        let overlapping = fb("other", vec![h(7)], BS + 1);
        assert!(
            matches!(
                engine.clone().find_blocks(&overlapping, None)?,
                FindBlocksOutcome::Deferred
            ),
            "post-terminal, pre-release: an overlapping fresh mint still defers"
        );

        // The connector-visible release lifts the deferral.
        engine.release_search(&search_id);
        assert!(engine.inflight.lock().unwrap().is_empty());
        assert!(
            !matches!(
                engine.clone().find_blocks(&overlapping, None)?,
                FindBlocksOutcome::Deferred
            ),
            "after the release the fresh mint proceeds"
        );
        Ok(())
    }

    /// A FAILED local onboard's record also outlives its `Failed` terminal and
    /// clears at the lifecycle release — failure does not re-admit early.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inflight_failed_local_onboard_clears_at_release() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let search_id = SearchId::new();
        let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 }));
        engine.searches.insert(
            search_id,
            SearchState {
                request_id: "rq".into(),
                status: cell.clone(),
                onboarding: OnboardingState::new(
                    0,
                    BS + 1,
                    OnboardingShard {
                        start_block: 0,
                        num_queried_blocks: 1,
                        find_session: complete_async(1), // no G2 payload → Failed
                    },
                ),
                buffer: vec![h(3)],
            },
        );
        let onboard = engine.clone().local_onboard(search_id, &[10]).unwrap();
        wait_complete(&onboard).await;
        assert_eq!(
            onboard.outcome(),
            Some(LoadOutcome::FailedPartial {
                block_ids: vec![10]
            })
        );
        assert!(
            engine.inflight.lock().unwrap().overlaps(&[h(3)]),
            "a failed load's record survives its terminal"
        );
        engine.release_search(&search_id);
        assert!(
            engine.inflight.lock().unwrap().is_empty(),
            "the lifecycle release clears the failed load's record"
        );
        Ok(())
    }

    /// An evicted onboard's record survives BOTH the evict and the fence-armed
    /// driver terminal; it clears only when the drain-holder's eventual handle
    /// drop fires `release_search` — the deferral spans the entire drain.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inflight_evicted_onboard_clears_at_holder_release() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let (find_session, _tx) = pending_async(); // park the driver pre-terminal
        let search_id = SearchId::new();
        let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 }));
        engine.searches.insert(
            search_id,
            SearchState {
                request_id: "rq".into(),
                status: cell.clone(),
                onboarding: OnboardingState::new(
                    0,
                    BS + 1,
                    OnboardingShard {
                        start_block: 0,
                        num_queried_blocks: 1,
                        find_session,
                    },
                ),
                buffer: vec![h(5)],
            },
        );
        let req: RequestId = "rq".into();
        let onboard = engine.clone().local_onboard(search_id, &[10]).unwrap();
        assert!(engine.inflight.lock().unwrap().overlaps(&[h(5)]));

        // Evict arms the fence over the in-flight onboard but must NOT clear
        // the guard — the transfer is still live.
        let fence = engine.evict(&req).fence;
        assert!(
            !fence.per_worker.is_empty(),
            "evict arms a fence over the live onboard"
        );
        assert!(
            engine.inflight.lock().unwrap().overlaps(&[h(5)]),
            "evict does not clear; the window stays deferred while live"
        );

        // The fence-armed driver terminal does not clear either …
        engine.finish_load_action(*onboard.id(), &req, ActionStatus::Complete, vec![10]);
        assert!(
            engine.inflight.lock().unwrap().overlaps(&[h(5)]),
            "the terminal leaves the deferral in place until the holder drops"
        );
        // … only the drain-holder's release does.
        engine.release_search(&search_id);
        assert!(engine.inflight.lock().unwrap().is_empty());
        Ok(())
    }

    /// Integration-level refcount independence: two lifecycles record
    /// overlapping windows. Releasing the FIRST keeps the shared hash deferred
    /// for the still-live second; only the second release empties the guard.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inflight_two_overlapping_onboards_refcount_at_releases() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        // Park both drivers so neither terminal interferes.
        let mint = |engine: &Arc<LocalConnectorEngine>, window: Vec<SequenceHash>| {
            let (find_session, tx) = pending_async();
            let search_id = SearchId::new();
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 2 }));
            engine.searches.insert(
                search_id,
                SearchState {
                    request_id: "rq".into(),
                    status: cell.clone(),
                    onboarding: OnboardingState::new(
                        0,
                        2 * BS + 1,
                        OnboardingShard {
                            start_block: 0,
                            num_queried_blocks: 2,
                            find_session,
                        },
                    ),
                    buffer: window,
                },
            );
            let onboard = engine.clone().local_onboard(search_id, &[10, 11]).unwrap();
            // The onboard stays parked (as the connector parks it); the explicit
            // release_search below models the connector's eventual handle drop.
            (search_id, onboard, tx)
        };

        let (sid_a, _onboard_a, _tx_a) = mint(&engine, vec![h(0), h(1)]);
        let (sid_b, _onboard_b, _tx_b) = mint(&engine, vec![h(1), h(2)]); // shares h(1)
        {
            let g = engine.inflight.lock().unwrap();
            assert!(g.overlaps(&[h(1)]));
            assert_eq!(g.len(), 3, "h(0), h(1), h(2)");
        }

        engine.release_search(&sid_a);
        {
            let g = engine.inflight.lock().unwrap();
            assert!(!g.overlaps(&[h(0)]), "A's exclusive hash cleared");
            assert!(
                g.overlaps(&[h(1)]),
                "shared hash stays deferred while B is live"
            );
            assert_eq!(g.len(), 2);
        }

        engine.release_search(&sid_b);
        assert!(
            engine.inflight.lock().unwrap().is_empty(),
            "the last covering release empties the guard"
        );
        Ok(())
    }

    /// An offload (save) action records NOTHING; neither its terminal
    /// (`finish_save_action`) nor its handle's `release_action` touches the
    /// guard — record and clear are lifecycle-scoped, not action-scoped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inflight_offload_action_neither_records_nor_clears() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let req: RequestId = "rq".into();
        let offload = engine
            .clone()
            .offload(&req, vec![(h(0), 10)])
            .expect("offload mints a handle");
        assert!(
            engine.inflight.lock().unwrap().is_empty(),
            "offload mints record nothing in the onboard guard"
        );

        // A foreign recorded lifecycle is untouched by the offload terminal
        // and by the offload handle's RAII action release.
        let other = InflightKey::Search(SearchId::new());
        engine.inflight.lock().unwrap().record(other, vec![h(0)]);
        engine.finish_save_action(*offload.id(), &req, ActionStatus::Complete);
        assert!(
            engine.inflight.lock().unwrap().overlaps(&[h(0)]),
            "the save terminal does not clear an unrelated lifecycle"
        );
        drop(offload);
        assert!(
            engine.inflight.lock().unwrap().overlaps(&[h(0)]),
            "release_action does not clear an unrelated lifecycle"
        );
        Ok(())
    }

    // ----- offload tests -----

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn resource_batched_onboard_mints_one_request_action() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let engine = LocalConnectorEngine::new(leader, RecordingSink::new(), BS, true);

        let onboard = engine
            .clone()
            .onboard_resources(
                &"resource-restore".into(),
                vec![ResourceOnboard {
                    resource: kvbm_common::LogicalResourceId::default(),
                    source_block_ids: vec![0],
                    destination_block_ids: vec![5],
                }],
            )
            .unwrap();

        wait_complete(&onboard).await;
        assert_eq!(
            onboard.outcome(),
            Some(LoadOutcome::FailedPartial { block_ids: vec![5] }),
            "the test leader has no physical worker, but the resource action must run to terminal"
        );
        Ok(())
    }

    /// The highest-value test: `(SequenceHash, BlockId)` pairs map to
    /// `ExternalBlock::new(block_id, sequence_hash)` — REVERSED. A naive splat
    /// offloads each block under the wrong hash. Pure fold, no leader needed.
    #[test]
    fn build_external_blocks_reverses_pair_to_block_id_first() {
        let h7 = h(7);
        let h9 = h(9);
        let pairs = vec![(h7, 100usize), (h9, 200usize)];
        let blocks = offload::build_external_blocks(&pairs);
        assert_eq!(blocks.len(), 2);
        // Teeth: assert the per-field mapping, not just the count.
        assert_eq!(blocks[0].block_id, 100);
        assert_eq!(blocks[0].sequence_hash, h7);
        assert_eq!(blocks[1].block_id, 200);
        assert_eq!(blocks[1].sequence_hash, h9);
    }

    /// `offload` buffers and does NOT enqueue until `finish_forward_pass`; the
    /// flush submits with the reversed arg order and a real precondition.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn offload_buffers_until_finish_forward_pass() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let submit = MockOffloadSubmit::new(TransferStatus::Complete, vec![]);
        let engine = LocalConnectorEngine::with_offload_submit(
            leader.clone(),
            sink.clone(),
            BS,
            true,
            submit.clone(),
            None,
        );

        let pairs = vec![(h(1), 10usize), (h(2), 11usize)];
        let handle = engine.clone().offload(&"rq".into(), pairs).unwrap();

        // Buffered, not enqueued, not yet terminal.
        assert_eq!(submit.submit_count(), 0, "offload must buffer, not enqueue");
        assert!(!handle.is_complete());

        // Flush at the forward-pass boundary.
        engine.begin_forward_pass(7);
        engine.finish_forward_pass(7);

        assert_eq!(
            submit.submit_count(),
            1,
            "finish_forward_pass must flush exactly one submit"
        );
        // Arg order survives the full flush path: (block_id, seq_hash).
        assert_eq!(submit.last_submit(), vec![(10, h(1)), (11, h(2))]);
        // A precondition (the iteration's forward-pass event) was provided.
        assert!(
            submit.last_precondition_present(),
            "flush must gate the offload on the forward-pass precondition"
        );

        // The completion driver drives the handle to terminal.
        wait_offload_complete(&handle).await;
        assert_eq!(handle.outcome(), Some(SaveOutcome::Done));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn resource_offload_buffers_and_flushes_through_the_connector_engine() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let submit = MockOffloadSubmit::new(TransferStatus::Complete, vec![]);
        let engine =
            LocalConnectorEngine::with_offload_submit(leader, sink, BS, true, submit.clone(), None);

        let handle = engine
            .clone()
            .offload_for_resource(
                kvbm_common::LogicalResourceId(7),
                &"resource-rq".into(),
                vec![(h(4), 13usize)],
            )
            .unwrap();
        assert_eq!(submit.submit_count(), 0);

        engine.finish_forward_pass(0);

        assert_eq!(submit.submit_count(), 1);
        assert_eq!(submit.last_submit(), vec![(13, h(4))]);
        assert_eq!(
            submit.last_resource(),
            Some(kvbm_common::LogicalResourceId(7))
        );
        wait_offload_complete(&handle).await;
        assert_eq!(handle.outcome(), Some(SaveOutcome::Done));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn resource_offload_without_a_route_fails_before_buffering() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let engine = LocalConnectorEngine::new(leader, RecordingSink::new(), BS, true);
        let resource = kvbm_common::LogicalResourceId(9);

        let error = engine
            .clone()
            .offload_for_resource(resource, &"missing-route".into(), vec![(h(4), 13usize)])
            .unwrap_err();

        assert_eq!(
            error,
            LeaderEngineError::ResourceOffloadNotConfigured { resource }
        );
        assert!(engine.offload_buffer.lock().unwrap().is_empty());
        Ok(())
    }

    /// `finish_forward_pass(n)` submits only offloads buffered under iteration
    /// `<= n`: the leader's merge-await for pass `n` can land after pass
    /// `n+1`'s scheduler walk has buffered new offloads, and those mid-pass G1
    /// sources are still being written — the later pass's own flush drains
    /// them.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flush_scopes_to_buffered_iteration() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let submit = MockOffloadSubmit::new(TransferStatus::Complete, vec![]);
        let engine = LocalConnectorEngine::with_offload_submit(
            leader.clone(),
            sink.clone(),
            BS,
            true,
            submit.clone(),
            None,
        );

        // Pass 1 buffers one offload…
        engine.begin_forward_pass(1);
        let h1 = engine
            .clone()
            .offload(&"rq".into(), vec![(h(1), 10usize)])
            .unwrap();
        // …then pass 2's walk buffers another BEFORE pass 1's flush lands.
        engine.begin_forward_pass(2);
        let h2 = engine
            .clone()
            .offload(&"rq".into(), vec![(h(2), 11usize)])
            .unwrap();

        // The late pass-1 flush submits ONLY the pass-1 entry.
        engine.finish_forward_pass(1);
        assert_eq!(
            submit.submit_count(),
            1,
            "pass-1 flush must not take pass-2's mid-pass buffer"
        );
        assert_eq!(submit.last_submit(), vec![(10, h(1))]);
        wait_offload_complete(&h1).await;
        assert!(!h2.is_complete(), "pass-2 entry stays buffered");

        // Pass 2's own flush drains its entry.
        engine.finish_forward_pass(2);
        assert_eq!(submit.submit_count(), 2);
        assert_eq!(submit.last_submit(), vec![(11, h(2))]);
        wait_offload_complete(&h2).await;
        Ok(())
    }

    /// A pass whose flush never fired (the leader's merge-await failed, so it
    /// strands rather than unsafe-flushes) is RECOVERED by the next confirmed
    /// flush: `finish_forward_pass(n)`'s `<= n` sweep drains the stragglers,
    /// gated on that later pass completing (safe — completed blocks are
    /// immutable).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flush_sweeps_stragglers_from_missed_pass() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let submit = MockOffloadSubmit::new(TransferStatus::Complete, vec![]);
        let engine = LocalConnectorEngine::with_offload_submit(
            leader.clone(),
            sink.clone(),
            BS,
            true,
            submit.clone(),
            None,
        );

        // Pass 1 buffers an offload but its flush never fires.
        engine.begin_forward_pass(1);
        let h1 = engine
            .clone()
            .offload(&"rq".into(), vec![(h(1), 10usize)])
            .unwrap();
        // Pass 2 buffers another; ITS flush is the recovery point.
        engine.begin_forward_pass(2);
        let h2 = engine
            .clone()
            .offload(&"rq".into(), vec![(h(2), 11usize)])
            .unwrap();

        engine.finish_forward_pass(2);
        assert_eq!(
            submit.submit_count(),
            2,
            "the pass-2 flush must sweep the stranded pass-1 entry too"
        );
        wait_offload_complete(&h1).await;
        wait_offload_complete(&h2).await;
        Ok(())
    }

    /// `take_offload_drain` is Some once then None; `commit` fires
    /// `mark_save_finished` EXACTLY once, and the per-action terminal fires it
    /// zero times.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn offload_drain_commits_save_finished_once_never_per_action() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let submit = MockOffloadSubmit::new(TransferStatus::Complete, vec![]);
        let engine = LocalConnectorEngine::with_offload_submit(
            leader.clone(),
            sink.clone(),
            BS,
            true,
            submit.clone(),
            None,
        );

        let handle = engine
            .clone()
            .offload(&"rq".into(), vec![(h(1), 10usize)])
            .unwrap();
        engine.begin_forward_pass(1);
        engine.finish_forward_pass(1);
        wait_offload_complete(&handle).await;

        // The per-action terminal fired NO save_finished (and no fence — this
        // request was not evicted).
        assert!(
            sink.saves().is_empty(),
            "per-offload-action terminal must not fire mark_save_finished"
        );
        assert!(sink.fences().is_empty());

        // First take → Some; commit → exactly one save_finished.
        let drain = engine
            .take_offload_drain(&"rq".into())
            .expect("first take yields a drain");
        drain.commit();
        assert_eq!(sink.saves(), vec![("rq".to_string(), SaveOutcome::Done)]);

        // Second take → None; still exactly one save_finished total.
        assert!(
            engine.take_offload_drain(&"rq".into()).is_none(),
            "second take must be None"
        );
        assert_eq!(sink.saves().len(), 1, "exactly one save_finished total");
        Ok(())
    }

    /// D semantics: `commit` on a drain while the request still has PENDING
    /// actions must NOT emit immediately — it arms emit-on-last-terminal. The
    /// emission fires exactly once, when the LAST pending action drains.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_commit_defers_emission_until_last_pending_action_drains() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        // Two in-flight actions for one finishing request (cells held live, as
        // live handles would).
        let req: RequestId = "rq".into();
        let (a1, a2) = (ActionId::new(), ActionId::new());
        let cell1 = Arc::new(Mutex::new(ActionStatus::Pending));
        let cell2 = Arc::new(Mutex::new(ActionStatus::Pending));
        engine
            .actions
            .insert(a1, ActionRecord::new(req.clone(), Arc::downgrade(&cell1)));
        engine
            .actions
            .insert(a2, ActionRecord::new(req.clone(), Arc::downgrade(&cell2)));
        engine.by_request.insert(req.clone(), vec![a1, a2]);
        engine.offload_drains.insert(req.clone(), ());

        // request_finished(Pending) arm point: take + commit with work pending.
        let drain = engine.take_offload_drain(&req).expect("drain registered");
        drain.commit();
        assert!(
            sink.saves().is_empty(),
            "commit with pending actions must DEFER the emission, not fire it"
        );

        // First action drains: still no emission (the second clone is live).
        engine.finish_save_action(a1, &req, ActionStatus::Complete);
        assert!(
            sink.saves().is_empty(),
            "emission must NOT fire after only the FIRST pending action drains"
        );

        // Last action drains: the single finished_sending emission.
        engine.finish_save_action(a2, &req, ActionStatus::Complete);
        assert_eq!(sink.saves(), vec![("rq".to_string(), SaveOutcome::Done)]);
        Ok(())
    }

    /// D semantics: arming covers PENDING LOAD actions too — vLLM frees the
    /// request's G1 blocks on `finished_sending`, and an in-flight onboard is
    /// still writing into them. A drain-armed load terminal must SUPPRESS its
    /// own `mark_load_finished`: vLLM's `_free_blocks` deletes the request, so
    /// a finished request surfacing in BOTH `finished_recving` and
    /// `finished_sending` asserts in the scheduler. The load's completion folds
    /// into the single `finished_sending`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_armed_load_folds_into_save_emission() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let req: RequestId = "rq".into();
        let a1 = ActionId::new();
        let cell1 = Arc::new(Mutex::new(ActionStatus::Pending));
        engine
            .actions
            .insert(a1, ActionRecord::new(req.clone(), Arc::downgrade(&cell1)));
        engine.by_request.insert(req.clone(), vec![a1]);
        engine.offload_drains.insert(req.clone(), ());

        engine.take_offload_drain(&req).expect("drain").commit();
        assert!(
            sink.saves().is_empty(),
            "deferred while the load is pending"
        );

        engine.finish_load_action(a1, &req, ActionStatus::Complete, Vec::new());
        // The drain-armed load terminal fires NO mark_load_finished (it would
        // surface finished_recving alongside the request's finished_sending —
        // a double-free assert in vLLM's scheduler)…
        assert!(
            sink.loads().is_empty(),
            "drain-armed load must not fire mark_load_finished"
        );
        // …and the drain's last clone dropping fires the single save emission.
        assert_eq!(sink.saves(), vec![("rq".to_string(), SaveOutcome::Done)]);
        Ok(())
    }

    /// D semantics: commit with NOTHING pending (already-terminal or no actions
    /// at all) emits immediately — Ryan's "it might already be done; doesn't
    /// matter".
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_commit_emits_immediately_when_nothing_pending() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let req: RequestId = "rq".into();
        engine.offload_drains.insert(req.clone(), ());
        engine.take_offload_drain(&req).expect("drain").commit();
        assert_eq!(sink.saves(), vec![("rq".to_string(), SaveOutcome::Done)]);
        Ok(())
    }

    /// Regression analog of the fence handle-drop hazard: a drain-armed action
    /// whose handle drops BEFORE its driver terminal must keep its record (and
    /// the live drain clone) until the terminal — else the emission fires while
    /// the transfer is still draining.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_armed_action_survives_handle_drop_before_terminal() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(leader.clone(), sink.clone(), BS, true);

        let req: RequestId = "rq".into();
        let a1 = ActionId::new();
        let cell1 = Arc::new(Mutex::new(ActionStatus::Pending));
        engine
            .actions
            .insert(a1, ActionRecord::new(req.clone(), Arc::downgrade(&cell1)));
        engine.by_request.insert(req.clone(), vec![a1]);
        engine.offload_drains.insert(req.clone(), ());

        engine.take_offload_drain(&req).expect("drain").commit();

        // Handle drops first: removal must be deferred, emission must not fire.
        engine.release_action(&a1);
        assert!(
            engine.actions.contains_key(&a1),
            "release_action must defer removal of a drain-armed action"
        );
        assert!(
            sink.saves().is_empty(),
            "handle drop must not fire the emission early"
        );

        // Driver terminal: fires the emission AND removes the deferred record.
        engine.finish_save_action(a1, &req, ActionStatus::Complete);
        assert_eq!(sink.saves(), vec![("rq".to_string(), SaveOutcome::Done)]);
        assert!(!engine.actions.contains_key(&a1));
        Ok(())
    }

    /// A request that never offloaded has no drain to commit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn take_offload_drain_none_without_offload() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let engine = LocalConnectorEngine::new(leader.clone(), NoopWorkerSink::new(), BS, true);
        assert!(engine.take_offload_drain(&"rq".into()).is_none());
        Ok(())
    }

    /// A failed transfer reaches terminal (drained) and projects to a
    /// `SaveOutcome::Failed*` — never hangs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn offload_failed_transfer_reaches_terminal() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let submit = MockOffloadSubmit::new(TransferStatus::Failed, vec![10]);
        let engine = LocalConnectorEngine::with_offload_submit(
            leader.clone(),
            sink.clone(),
            BS,
            true,
            submit.clone(),
            None,
        );

        let handle = engine
            .clone()
            .offload(&"rq".into(), vec![(h(1), 10usize)])
            .unwrap();
        engine.begin_forward_pass(1);
        engine.finish_forward_pass(1);
        wait_offload_complete(&handle).await;

        assert!(handle.is_complete());
        assert_eq!(
            handle.outcome(),
            Some(SaveOutcome::FailedPartial {
                block_ids: vec![10]
            })
        );
        // Still no save_finished from the per-action terminal.
        assert!(sink.saves().is_empty());
        Ok(())
    }

    /// A submit error (e.g. the disabled seam) folds the action to a terminal
    /// failure rather than leaving the handle pending forever.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn offload_submit_error_marks_action_failed() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        // Default `new` => DisabledOffloadSubmit (submit returns Err).
        let engine = LocalConnectorEngine::new(leader.clone(), NoopWorkerSink::new(), BS, true);

        let handle = engine
            .clone()
            .offload(&"rq".into(), vec![(h(1), 10usize)])
            .unwrap();
        engine.begin_forward_pass(1);
        engine.finish_forward_pass(1);
        wait_offload_complete(&handle).await;

        assert_eq!(handle.outcome(), Some(SaveOutcome::FailedAllBlocks));
        Ok(())
    }

    /// RAII: dropping each terminal offload handle prunes BOTH the `actions` map
    /// and the `by_request` index (the action analogue of the onboard RAII test;
    /// `finish_save_action` never removes from `actions`, so reaching 0 isolates
    /// the handle-Drop → `release_action` path).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn offload_handle_drop_prunes_actions_and_by_request() -> Result<()> {
        let leader = Arc::new(build_test_leader().await?);
        let sink = RecordingSink::new();
        let submit = MockOffloadSubmit::new(TransferStatus::Complete, vec![]);
        let engine = LocalConnectorEngine::with_offload_submit(
            leader.clone(),
            sink.clone(),
            BS,
            true,
            submit.clone(),
            None,
        );

        const N: usize = 8;
        for i in 0..N {
            let handle = engine
                .clone()
                .offload(&"rq".into(), vec![(h(i as u64), i)])
                .unwrap();
            engine.begin_forward_pass(i);
            engine.finish_forward_pass(i);
            wait_offload_complete(&handle).await;
            assert_eq!(handle.outcome(), Some(SaveOutcome::Done));
            // RAII Drop -> release_action is the ONLY pruner exercised here.
            drop(handle);
        }

        assert_eq!(
            engine.actions.len(),
            0,
            "offload handle Drop must prune the actions map"
        );
        assert!(
            engine.by_request.is_empty(),
            "offload handle Drop must scrub the by_request index"
        );
        Ok(())
    }

    /// Construction wiring (the single discovery-install path):
    /// `RemoteOps { search: Some(...) }` installs the discovery on the leader
    /// during construction, so a *subsequent* `set_remote_discovery` finds the
    /// cell already filled and returns `false`. `RemoteOps::default()` installs
    /// nothing, so a subsequent `set_remote_discovery` succeeds (`true`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn search_remoteops_installs_discovery_disabled_does_not() -> Result<()> {
        use super::super::{ConnectorEngineConfig, RemoteOps, build_local_connector_engine};
        use crate::remote::search::discovery::{
            RemoteBlockDiscovery, RemoteCandidates, RemoteDiscoveryHandle,
        };
        use futures::future::BoxFuture;

        struct StubDiscovery;
        impl RemoteBlockDiscovery for StubDiscovery {
            fn discover(
                &self,
                _hashes: Vec<SequenceHash>,
            ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
                Box::pin(async { Ok(None) })
            }
        }
        fn stub() -> RemoteDiscoveryHandle {
            Arc::new(StubDiscovery)
        }

        // search: Some(...) installs the discovery during construction.
        let search_leader = Arc::new(build_test_leader().await?);
        let _engine = build_local_connector_engine(
            search_leader.clone(),
            NoopWorkerSink::new(),
            ConnectorEngineConfig {
                block_size: BS,
                remote: RemoteOps::with_search(stub()),
            },
            None,
        );
        assert!(
            !search_leader.set_remote_discovery(stub()),
            "search: Some must install the discovery (the cell is now full)"
        );

        // search: None installs nothing.
        let disabled_leader = Arc::new(build_test_leader().await?);
        let _engine = build_local_connector_engine(
            disabled_leader.clone(),
            NoopWorkerSink::new(),
            ConnectorEngineConfig {
                block_size: BS,
                remote: RemoteOps::default(),
            },
            None,
        );
        assert!(
            disabled_leader.set_remote_discovery(stub()),
            "search: None must install no discovery (the cell is still empty)"
        );
        Ok(())
    }

    /// `RemoteOps::default()` has both siblings `None` — the safe-inert default
    /// that installs nothing and keeps the engine fully local.
    #[test]
    fn remote_ops_default_is_fully_local() {
        use super::super::RemoteOps;
        let ops = RemoteOps::default();
        assert!(
            ops.search.is_none(),
            "default RemoteOps must have search: None"
        );
        assert!(
            ops.disagg.is_none(),
            "default RemoteOps must have disagg: None"
        );
    }

    // ========================================================================
    // Conditional-disagg interposition tests
    // ========================================================================

    mod cd_tests {
        use super::*;

        use futures::FutureExt;
        use futures::future::BoxFuture;

        use kvbm_logical::manager::BlockManager;
        use kvbm_protocols::disagg::{RemotePrefillParams, SessionEndpoint};

        use crate::p2p::session::{
            AvailabilityStream, CommitStream, LifecycleEvent, LifecycleStream, MockSession,
            MockSessionFactory, PeerAvailable, PeerCommitted, Session, SessionId,
        };
        use crate::remote::cd::policy::SelectionPolicy;
        use crate::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
        use crate::testing::token_blocks::create_token_sequence;
        use crate::{ConnectorEngineConfig, RemoteOps, build_local_connector_engine};

        // ----- fixtures: real G2 immutable blocks + recording prefill plane -----

        fn cd_g2_manager(count: usize) -> Arc<BlockManager<G2>> {
            let registry = TestRegistryBuilder::new().build();
            Arc::new(
                TestManagerBuilder::<G2>::new()
                    .block_count(count)
                    .block_size(BS)
                    .registry(registry)
                    .build(),
            )
        }

        /// Allocate `count` registered immutable G2 blocks whose hash chain
        /// starts at `start_token` (distinct `start_token`s ⇒ disjoint hashes).
        fn cd_immutables(
            manager: &Arc<BlockManager<G2>>,
            count: usize,
            start_token: u32,
        ) -> Vec<ImmutableBlock<G2>> {
            let seq = create_token_sequence(count, BS, start_token);
            let mutable = manager.allocate_blocks(count).expect("alloc failed");
            let complete: Vec<_> = mutable
                .into_iter()
                .zip(seq.blocks().iter())
                .map(|(b, tb)| b.complete(tb).expect("complete failed"))
                .collect();
            manager.register_blocks(complete)
        }

        fn cd_block_hashes(blocks: &[ImmutableBlock<G2>]) -> Vec<SequenceHash> {
            blocks.iter().map(|b| b.sequence_hash()).collect()
        }

        /// Records every dispatched [`PrefillDispatch`]; resolves `Ok` by
        /// default, `Err` in failing mode (still recording the dispatch).
        struct RecordingPrefillPlane {
            dispatches: StdMutex<Vec<RecordedDispatch>>,
            fail: bool,
        }
        struct RecordedDispatch {
            request_id: String,
            session_id: Uuid,
            decode_endpoint: Option<SessionEndpoint>,
            num_provided_tokens: usize,
            num_window_tokens: usize,
        }
        impl RecordingPrefillPlane {
            fn ok() -> Arc<Self> {
                Arc::new(Self {
                    dispatches: StdMutex::new(Vec::new()),
                    fail: false,
                })
            }
            fn failing() -> Arc<Self> {
                Arc::new(Self {
                    dispatches: StdMutex::new(Vec::new()),
                    fail: true,
                })
            }
            fn count(&self) -> usize {
                self.dispatches.lock().unwrap().len()
            }
            fn last_request_id(&self) -> Option<String> {
                self.dispatches
                    .lock()
                    .unwrap()
                    .last()
                    .map(|d| d.request_id.clone())
            }
            fn last_session_id(&self) -> Option<Uuid> {
                self.dispatches.lock().unwrap().last().map(|d| d.session_id)
            }
            fn last_num_provided(&self) -> Option<usize> {
                self.dispatches
                    .lock()
                    .unwrap()
                    .last()
                    .map(|d| d.num_provided_tokens)
            }
            fn last_num_window(&self) -> Option<usize> {
                self.dispatches
                    .lock()
                    .unwrap()
                    .last()
                    .map(|d| d.num_window_tokens)
            }
            fn last_endpoint_present(&self) -> Option<bool> {
                self.dispatches
                    .lock()
                    .unwrap()
                    .last()
                    .map(|d| d.decode_endpoint.is_some())
            }
        }
        impl PrefillPlane for RecordingPrefillPlane {
            fn dispatch(&self, req: PrefillDispatch) -> BoxFuture<'static, anyhow::Result<()>> {
                self.dispatches.lock().unwrap().push(RecordedDispatch {
                    request_id: req.request_id,
                    session_id: req.session_id,
                    decode_endpoint: req.decode_endpoint,
                    num_provided_tokens: req.num_provided_tokens,
                    num_window_tokens: req.num_window_tokens,
                });
                let fail = self.fail;
                async move {
                    if fail {
                        Err(anyhow::anyhow!("dispatch boom"))
                    } else {
                        Ok(())
                    }
                }
                .boxed()
            }
        }

        fn cd_runtime(
            selection: SelectionPolicy,
            capacity: usize,
            fallback: bool,
            sessions: Arc<dyn SessionFactory>,
            plane: Arc<dyn PrefillPlane>,
        ) -> CdRuntime {
            CdRuntime::new(
                DisaggConfig {
                    selection,
                    max_inflight_remote_prefill_tokens: capacity,
                    local_fallback_on_overload: fallback,
                    ..DisaggConfig::default()
                },
                Arc::new(TierCell::default()),
                sessions,
                plane,
                None,
            )
        }

        fn cd_engine(leader: Arc<InstanceLeader>, cd: CdRuntime) -> Arc<LocalConnectorEngine> {
            LocalConnectorEngine::with_offload_submit(
                leader,
                NoopWorkerSink::new(),
                BS,
                false,
                Arc::new(DisabledOffloadSubmit),
                Some(cd),
            )
        }

        /// Poll a sync predicate to true (sleeping 5ms), panicking after ~1s.
        async fn wait_for(pred: impl Fn() -> bool) {
            for _ in 0..200 {
                if pred() {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            panic!("cd condition not met within timeout");
        }

        /// A real `InstanceLeader` whose G2 holds `count` registered blocks, so a
        /// local-only `find_matches` over their hashes covers them all (the
        /// `local_covers_all` Ready short-circuit) and `search` resolves
        /// synchronously to a `Matched` hit. The returned blocks MUST be held
        /// alive by the caller to keep them resident.
        async fn leader_with_resident_blocks(
            count: usize,
            start: u32,
        ) -> Result<(InstanceLeader, Vec<SequenceHash>, Vec<ImmutableBlock<G2>>)> {
            use crate::testing::messenger::create_messenger_tcp;
            use kvbm_logical::blocks::BlockRegistry;

            let messenger = create_messenger_tcp().await?;
            let registry = BlockRegistry::builder().build();
            let g2 = Arc::new(
                TestManagerBuilder::<G2>::new()
                    .block_count(count + 4)
                    .block_size(BS)
                    .registry(registry.clone())
                    .build(),
            );
            let blocks = cd_immutables(&g2, count, start);
            let plhs = cd_block_hashes(&blocks);
            let leader = InstanceLeader::builder()
                .messenger(messenger)
                .registry(registry)
                .g2_manager(g2)
                .build()?;
            Ok((leader, plhs, blocks))
        }

        // ------------------------------------------------------------------
        // w1: Never policy returns the local hit, commits nothing.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_never_returns_local_hit() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Never,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);

            let mgr = cd_g2_manager(8);
            let blocks = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&blocks);
            let req = req("rq", 0, plhs);
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));

            let out = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                Vec::new(),
                blocks,
            );
            assert_eq!(out, MatchStatus::Matched { hit_blocks: 3 });

            let cdr = engine.cd.as_ref().unwrap();
            assert_eq!(factory.open_count(), 0, "Never opens no session");
            assert!(cdr.requests.is_empty(), "Never sets no latch");
            assert_eq!(cdr.budget.available(), 256, "budget untouched");
            assert_eq!(plane.count(), 0, "no dispatch");
            Ok(())
        }

        // ------------------------------------------------------------------
        // w2: Always + computed=0 commits Remote via the real local_search path.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_search_immediate_hit_commits_remote() -> Result<()> {
            let (leader, plhs, _held) = leader_with_resident_blocks(3, 100).await?;
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();

            // Factory path: exercises with_disagg + mod.rs CdRuntime construction.
            let config = ConnectorEngineConfig {
                block_size: BS,
                remote: RemoteOps::default().with_disagg(
                    factory.clone(),
                    plane.clone(),
                    Arc::new(TierCell::default()),
                    DisaggConfig {
                        selection: SelectionPolicy::Always,
                        max_inflight_remote_prefill_tokens: 256,
                        local_fallback_on_overload: true,
                        ..DisaggConfig::default()
                    },
                ),
            };
            let (engine, _driver) =
                build_local_connector_engine(Arc::new(leader), NoopWorkerSink::new(), config, None);

            // Factory-built engine: drive the surviving trait verb (`find_blocks`)
            // over the dyn handle. A pure-local request (no transfer params) with
            // a 3-block eligible window resolves to a 3-block (= `3 * BS`-token)
            // local hit and commits Remote through the same CD interpose the
            // legacy local-search path exercised.
            match Arc::clone(&engine).find_blocks(&fb("rq", plhs.clone(), 4 * BS), None)? {
                FindBlocksOutcome::Resolved { matched_tokens, .. } => {
                    assert_eq!(
                        matched_tokens,
                        3 * BS,
                        "unified == window_blocks * block_size"
                    )
                }
                other => panic!("expected Resolved, got {other:?}"),
            }

            wait_for(|| plane.count() >= 1).await;
            assert_eq!(plane.last_request_id(), Some("rq".to_string()));
            assert_eq!(
                plane.last_num_provided(),
                Some(3 * BS),
                "num_provided == local_hit * bs"
            );
            assert_eq!(
                plane.last_num_window(),
                Some(3 * BS),
                "num_window == computed + fbet (the whole eligible window here)"
            );
            assert_eq!(
                plane.last_endpoint_present(),
                Some(true),
                "carries the decode session endpoint"
            );

            let session = factory.last_opened().expect("session opened once");
            assert_eq!(factory.open_count(), 1);
            assert_eq!(
                session.commit_calls(),
                vec![plhs.clone()],
                "commit == window"
            );
            assert_eq!(
                session.make_available_calls(),
                vec![plhs],
                "made the local blocks available"
            );
            assert!(session.finish_commits_called());
            assert!(session.finish_availability_called(), "no pending → sealed");
            Ok(())
        }

        // ------------------------------------------------------------------
        // Cold prompt (zero local match) over the remote window commits Remote.
        // Reproducer for the terminal-zero skip: a fully-cold prompt used to
        // collapse to NoMatch in perform_search and never reach cd_interpose, so
        // it always prefilled local. POST-FIX a zero-block latch is minted and CD
        // promotes it to a Remote commit over the whole window.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_cold_prompt_commits_remote() -> Result<()> {
            // Empty G2 + synthetic hashes ⇒ a local find resolves to a SYNCHRONOUS
            // terminal-zero (Found{0}) with search_remote=false (the colocated
            // smoke shape).
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            // 3 cold blocks; total carries vLLM's last-block exclusion ⇒
            // window_blocks == 3, num_computed == 0. Hold the minted handle for
            // the duration: dropping it would release the search and tear the CD
            // lifecycle back down before the latch assertions run.
            let cold = plhs_for(500, 3);
            let minted = match Arc::clone(&engine).find_blocks(&fb("rq", cold, 4 * BS), None)? {
                FindBlocksOutcome::Resolved {
                    matched_tokens,
                    minted,
                    ..
                } => {
                    assert_eq!(matched_tokens, 3 * BS, "unified == window_blocks * bs");
                    minted.expect("cold-remote parks a search handle")
                }
                other => panic!("expected Resolved, got {other:?}"),
            };

            wait_for(|| plane.count() >= 1).await;
            assert_eq!(plane.last_request_id(), Some("rq".to_string()));
            assert_eq!(
                plane.last_num_provided(),
                Some(0),
                "num_provided == num_computed + local_hit*bs == 0 (nothing local)"
            );
            assert_eq!(
                plane.last_num_window(),
                Some(3 * BS),
                "num_window == num_computed + fbet == the whole cold window"
            );
            assert_eq!(plane.last_endpoint_present(), Some(true));
            assert_eq!(factory.open_count(), 1);

            // Empty-commit wire shape: cold ⇒ prefix+pending+local_match all
            // empty, so the holder sees exactly one empty commit + one empty
            // make_available, then is sealed (no pending tail).
            let session = factory.last_opened().expect("session opened once");
            assert_eq!(session.commit_calls(), vec![Vec::<SequenceHash>::new()]);
            assert_eq!(
                session.make_available_calls(),
                vec![Vec::<SequenceHash>::new()]
            );
            assert!(session.finish_commits_called());
            assert!(session.finish_availability_called(), "no pending → sealed");

            // The Remote commit latched a live CD request holding the budget.
            assert!(!cdr.requests.is_empty(), "cold-remote latched the request");
            assert_eq!(cdr.budget.available(), 256 - 3 * BS, "fbet reserved");
            drop(minted);
            Ok(())
        }

        // ------------------------------------------------------------------
        // A cold prompt the CD policy keeps Local (Never) must NOT linger: the
        // zero-block mint collapses to NoMatch so nothing latches and no handle
        // is reported (vLLM would never onboard an empty match).
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_cold_policy_local_does_not_linger() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Never,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            let cold = plhs_for(500, 3);
            match Arc::clone(&engine).find_blocks(&fb("rq", cold, 4 * BS), None)? {
                FindBlocksOutcome::Resolved {
                    matched_tokens,
                    minted,
                    release_parked,
                } => {
                    assert_eq!(matched_tokens, 0, "policy Never keeps prefill local");
                    assert!(minted.is_none(), "collapsed mint hands back no handle");
                    assert!(!release_parked);
                }
                other => panic!("expected Resolved, got {other:?}"),
            }

            assert!(engine.searches.is_empty(), "no lingering latch");
            assert!(cdr.requests.is_empty(), "no CD lifecycle");
            assert_eq!(cdr.budget.available(), 256, "budget untouched");
            assert_eq!(plane.count(), 0, "no dispatch");
            assert_eq!(factory.open_count(), 0, "no session opened");
            Ok(())
        }

        // ------------------------------------------------------------------
        // A committed cold-remote re-poll is idempotent: the step-2 latch
        // re-answers the same unified count with no re-dispatch and no budget
        // touch.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_cold_remote_repoll_is_idempotent() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            let cold = plhs_for(500, 3);
            // Poll 1: fresh cold-remote commit, parks the live handle.
            let minted =
                match Arc::clone(&engine).find_blocks(&fb("rq", cold.clone(), 4 * BS), None)? {
                    FindBlocksOutcome::Resolved {
                        matched_tokens,
                        minted: Some(handle),
                        ..
                    } => {
                        assert_eq!(matched_tokens, 3 * BS);
                        handle
                    }
                    other => panic!("expected Resolved+minted, got {other:?}"),
                };
            wait_for(|| plane.count() >= 1).await;
            let avail = cdr.budget.available();
            assert_eq!(avail, 256 - 3 * BS);

            // Poll 2: re-pass the live handle ⇒ the REFRESH arm hits the latch.
            match Arc::clone(&engine).find_blocks(&fb("rq", cold, 4 * BS), Some(&minted))? {
                FindBlocksOutcome::Resolved {
                    matched_tokens,
                    minted,
                    ..
                } => {
                    assert_eq!(matched_tokens, 3 * BS, "same unified count");
                    assert!(minted.is_none(), "refresh never re-mints");
                }
                other => panic!("expected Resolved, got {other:?}"),
            }
            assert_eq!(plane.count(), 1, "no second dispatch");
            assert_eq!(factory.open_count(), 1, "no second open");
            assert_eq!(cdr.budget.available(), avail, "budget unchanged on re-poll");
            Ok(())
        }

        // ------------------------------------------------------------------
        // w3: idempotent re-poll returns the same unified count, no re-commit.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_repoll_is_idempotent() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let blocks = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&blocks);
            let req = req("rq", 0, plhs);
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));

            let out1 = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                Vec::new(),
                blocks.clone(),
            );
            assert_eq!(out1, MatchStatus::Matched { hit_blocks: 3 });
            wait_for(|| plane.count() >= 1).await;
            let avail = cdr.budget.available();
            assert_eq!(avail, 256 - 3 * BS, "first commit reserved fbet");

            let out2 = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                Vec::new(),
                blocks,
            );
            assert_eq!(
                out2,
                MatchStatus::Matched { hit_blocks: 3 },
                "same unified count"
            );
            assert_eq!(factory.open_count(), 1, "no second open");
            assert_eq!(cdr.budget.available(), avail, "budget unchanged on re-poll");
            assert_eq!(plane.count(), 1, "no second dispatch");
            Ok(())
        }

        // ------------------------------------------------------------------
        // w4: budget exhausted + fallback=true downgrades to the local hit.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_budget_exhausted_fallback_downgrades_local() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            // capacity BS < fbet (3*BS) ⇒ try_reserve fails.
            let cd = cd_runtime(
                SelectionPolicy::Always,
                BS,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let blocks = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&blocks);
            let req = req("rq", 0, plhs);
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));

            let out = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                Vec::new(),
                blocks,
            );
            assert_eq!(
                out,
                MatchStatus::Matched { hit_blocks: 3 },
                "downgrade → local hit"
            );
            assert_eq!(factory.open_count(), 0, "no session");
            assert_eq!(
                cdr.budget.available(),
                BS,
                "failed reserve leaves budget untouched"
            );
            assert!(cdr.requests.is_empty());
            Ok(())
        }

        // ------------------------------------------------------------------
        // w5: budget exhausted + fallback=false parks Pending; freeing it commits.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_budget_exhausted_reject_then_repoll_commits() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                3 * BS,
                false,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            // Exhaust the budget so try_reserve(fbet) fails.
            assert!(cdr.budget.try_reserve(3 * BS));
            assert_eq!(cdr.budget.available(), 0);

            let mgr = cd_g2_manager(8);
            let blocks = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&blocks);
            let req = req("rq", 0, plhs);
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));

            let out = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                Vec::new(),
                blocks.clone(),
            );
            assert_eq!(out, MatchStatus::Pending, "reject parks pending");
            assert_eq!(
                *cell.lock().unwrap(),
                MatchStatus::Pending,
                "the handle cell reads Pending"
            );
            assert!(cdr.requests.is_empty(), "reject sets no latch");
            assert_eq!(factory.open_count(), 0);

            // Free the budget; the re-poll commits Remote.
            cdr.budget.release(3 * BS);
            let out2 = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                Vec::new(),
                blocks,
            );
            assert_eq!(out2, MatchStatus::Matched { hit_blocks: 3 });
            wait_for(|| plane.count() >= 1).await;
            assert_eq!(factory.open_count(), 1, "session opens once budget frees");
            Ok(())
        }

        // ------------------------------------------------------------------
        // w6: a vLLM-computed prefix that is NOT fully G2-resident keeps
        // prefill local (the residency gate) — and the downgrade runs BEFORE
        // the plan, so the budget is never touched and no session opens.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_computed_prefix_not_resident_keeps_local() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let prefix = cd_immutables(&mgr, 2, 0);
            let blocks = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&blocks);
            // computed_blocks = 2 but only ONE prefix block resolved resident:
            // the partial-residency downgrade.
            let req = req_with_prefix("rq", cd_block_hashes(&prefix), plhs);
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));

            let out = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                prefix[..1].to_vec(),
                blocks,
            );
            assert_eq!(out, MatchStatus::Matched { hit_blocks: 3 }, "stays local");
            assert_eq!(factory.open_count(), 0, "no session opened");
            assert!(cdr.requests.is_empty(), "no latch");
            assert_eq!(
                cdr.budget.available(),
                256,
                "the downgrade precedes the plan — budget never touched"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // A fully G2-resident computed prefix commits Remote: the session
        // commit set is prefix ++ local-match in absolute order, the prefix
        // pins ride the initial availability set, the budget reserves the
        // block-floored SUFFIX only, and the surfaced count stays
        // window-relative (vLLM's beyond-num_computed contract). A re-poll at
        // the same offset answers from the latch (no re-plan, no re-open).
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_computed_prefix_resident_commits_remote() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let prefix = cd_immutables(&mgr, 2, 0);
            let window = cd_immutables(&mgr, 3, 100);
            let prefix_h = cd_block_hashes(&prefix);
            let window_h = cd_block_hashes(&window);
            let req = req_with_prefix("rq", prefix_h.clone(), window_h.clone());
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 }));

            let out = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 1 },
                &cell,
                prefix.clone(),
                window[..1].to_vec(),
            );
            assert_eq!(
                out,
                MatchStatus::Matched { hit_blocks: 3 },
                "unified count is WINDOW-relative — the prefix never inflates it"
            );

            wait_for(|| plane.count() >= 1).await;
            assert_eq!(
                plane.last_num_provided(),
                Some(2 * BS + BS),
                "num_provided = computed + local_hit*bs"
            );
            assert_eq!(
                plane.last_num_window(),
                Some(2 * BS + 3 * BS),
                "num_window = computed + fbet"
            );
            assert_eq!(
                cdr.budget.available(),
                256 - 3 * BS,
                "fbet reserves the suffix window only — never the prefix"
            );

            let session = factory.last_opened().expect("session opened");
            let expected: Vec<SequenceHash> = prefix_h
                .iter()
                .chain(window_h[..1].iter())
                .copied()
                .collect();
            assert_eq!(
                session.commit_calls(),
                vec![expected.clone()],
                "commit set = prefix ++ local-match, absolute order"
            );
            assert_eq!(
                session.make_available_calls(),
                vec![expected],
                "the prefix pins ride the initial availability set"
            );
            assert!(session.finish_commits_called());
            assert!(session.finish_availability_called(), "nothing deferred");

            let state = cdr.requests.get("rq").expect("latched");
            assert_eq!(state.base_offset(), 2 * BS, "committed at this offset");
            assert_eq!(state.window_hashes(), &window_h[..3], "window-relative");

            // Idempotent re-poll at the SAME computed offset: the latch
            // answers; no second plan/open/dispatch and the budget is untouched.
            let out2 = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 1 },
                &cell,
                prefix,
                window[..1].to_vec(),
            );
            assert_eq!(out2, MatchStatus::Matched { hit_blocks: 3 });
            assert_eq!(factory.open_count(), 1, "no second open");
            assert_eq!(plane.count(), 1, "no second dispatch");
            assert_eq!(cdr.budget.available(), 256 - 3 * BS);
            Ok(())
        }

        // ------------------------------------------------------------------
        // The post-reserve commit guard: a prefix block count short of the
        // committed offset releases exactly the reserved fbet and downgrades
        // (no session, no latch) — the budget-leak hazard on the new arm.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_commit_prefix_count_mismatch_releases_budget() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let prefix = cd_immutables(&mgr, 2, 0);
            let window = cd_immutables(&mgr, 3, 100);
            let req = req_with_prefix("rq", cd_block_hashes(&prefix), cd_block_hashes(&window));
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 }));
            let local_status = MatchStatus::Matched { hit_blocks: 1 };

            // Drive the commit arm directly with the reservation already HELD
            // (as `plan` leaves it) but a prefix vec short of computed_blocks.
            assert!(cdr.budget.try_reserve(3 * BS));
            let out = engine.cd_commit_remote(
                cdr,
                SearchId::new(),
                &MatchWindow::of(&req),
                1,
                2 * BS,
                3 * BS,
                prefix[..1].to_vec(),
                window[..1].to_vec(),
                &cell,
                local_status,
            );
            assert_eq!(out, local_status, "safe-downgrade to the local answer");
            assert_eq!(
                cdr.budget.available(),
                256,
                "the held fbet is released exactly once"
            );
            assert_eq!(factory.open_count(), 0, "no session opened");
            assert!(cdr.requests.is_empty(), "no latch");
            Ok(())
        }

        // ------------------------------------------------------------------
        // Stale-latch staleness guard: a re-poll whose computed prefix moved
        // off the committed offset must NOT answer the old unified count
        // (matched_tokens would misalign with vLLM's beyond-num_computed
        // contract). The stale lifecycle is released — session closed, budget
        // back — the poll answers local, and the NEXT poll re-plans fresh.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_stale_latch_on_moved_computed_releases_and_replans() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            // Absolute chain a0..a3; first poll computed=0, window = a0..a2.
            let mgr = cd_g2_manager(8);
            let chain = cd_immutables(&mgr, 4, 100);
            let plhs = cd_block_hashes(&chain);
            let req0 = req("rq", 0, plhs[..3].to_vec());
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));
            let out = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req0),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                Vec::new(),
                chain[..3].to_vec(),
            );
            assert_eq!(out, MatchStatus::Matched { hit_blocks: 3 });
            wait_for(|| plane.count() >= 1).await;
            assert_eq!(cdr.budget.available(), 256 - 3 * BS);
            let first_session = factory.last_opened().expect("first session");

            // Re-poll with the prefix moved to 1 block: the latch (committed
            // at offset 0) is stale. The guard releases it and answers the
            // freshly-reconciled LOCAL count — never the old unified 3.
            let req1 = req_with_prefix("rq", plhs[..1].to_vec(), plhs[1..3].to_vec());
            let cell1 = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 2 }));
            let out1 = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req1),
                MatchStatus::Matched { hit_blocks: 2 },
                &cell1,
                chain[..1].to_vec(),
                chain[1..3].to_vec(),
            );
            assert_eq!(
                out1,
                MatchStatus::Matched { hit_blocks: 2 },
                "the stale unified count is never surfaced against the new offset"
            );
            assert_eq!(
                first_session.closed_reason(),
                Some(Some("stale cd latch: computed prefix moved".to_string())),
                "the stale lifecycle's session is closed"
            );
            assert_eq!(cdr.budget.available(), 256, "stale reservation released");
            assert!(cdr.requests.is_empty(), "stale latch removed");

            // The NEXT poll re-plans fresh at the new offset and commits
            // Remote again (fbet = the new 2-block suffix window).
            let out2 = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req1),
                MatchStatus::Matched { hit_blocks: 2 },
                &cell1,
                chain[..1].to_vec(),
                chain[1..3].to_vec(),
            );
            assert_eq!(out2, MatchStatus::Matched { hit_blocks: 2 });
            wait_for(|| plane.count() >= 2).await;
            assert_eq!(factory.open_count(), 2, "fresh lifecycle opened");
            assert_eq!(cdr.budget.available(), 256 - 2 * BS, "fresh suffix fbet");
            assert_eq!(
                cdr.requests.get("rq").map(|s| s.base_offset()),
                Some(BS),
                "fresh latch committed at the new offset"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // w7: declining (release_search after commit) closes + releases.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_decline_releases_via_release_search() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let blocks = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&blocks);
            let req = req("rq", 0, plhs);

            // A searches entry must exist for release_search to find the rid.
            let search_id = SearchId::new();
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));
            engine.searches.insert(
                search_id,
                SearchState {
                    request_id: "rq".into(),
                    status: cell.clone(),
                    onboarding: OnboardingState::new(
                        0,
                        3 * BS + 1,
                        OnboardingShard {
                            start_block: 0,
                            num_queried_blocks: 3,
                            find_session: ready_zero(),
                        },
                    ),
                    buffer: Vec::new(),
                },
            );

            // The committed state must carry the SAME search_id the handle is
            // released under, so the generation-bound decline cleanup matches.
            engine.cd_interpose(
                search_id,
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                Vec::new(),
                blocks,
            );
            wait_for(|| plane.count() >= 1).await;
            assert_eq!(cdr.budget.available(), 256 - 3 * BS);
            assert!(cdr.requests.get("rq").is_some(), "latched");

            engine.release_search(&search_id);
            let session = factory.last_opened().unwrap();
            assert_eq!(
                session.closed_reason(),
                Some(Some("declined".to_string())),
                "decline closes the session"
            );
            assert_eq!(cdr.budget.available(), 256, "budget restored");
            assert!(cdr.requests.is_empty(), "latch gone");
            Ok(())
        }

        // ------------------------------------------------------------------
        // A stale OLD-generation decline must not tear down a re-latched fresh
        // lifecycle: commit gen1 (search S1), evict it (releases gen1), commit
        // gen2 (search S2) for the SAME rid, then drop S1's handle
        // (release_search(S1)). The generation guard makes that a no-op — gen2's
        // session + reservation are untouched. (Without the search_id binding,
        // release_search re-fetches the rid and tears down gen2.)
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_decline_of_stale_generation_does_not_touch_fresh_lifecycle() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);

            // --- gen1: commit Remote under search S1 (handle H1 stays alive). ---
            // S1's searches entry must persist so release_search(S1) reaches the
            // CD cleanup later (the drain-holder keeps H1 — and the entry — alive).
            let s1 = SearchId::new();
            let cell1 = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));
            engine.searches.insert(
                s1,
                SearchState {
                    request_id: "rq".into(),
                    status: cell1.clone(),
                    onboarding: OnboardingState::new(
                        0,
                        3 * BS + 1,
                        OnboardingShard {
                            start_block: 0,
                            num_queried_blocks: 3,
                            find_session: ready_zero(),
                        },
                    ),
                    buffer: Vec::new(),
                },
            );
            let blocks1 = cd_immutables(&mgr, 3, 100);
            let plhs1 = cd_block_hashes(&blocks1);
            let req1 = req("rq", 0, plhs1);
            engine.cd_interpose(
                s1,
                &MatchWindow::of(&req1),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell1,
                Vec::new(),
                blocks1,
            );
            wait_for(|| plane.count() >= 1).await;
            assert_eq!(cdr.budget.available(), 256 - 3 * BS);

            // --- evict gen1: releases gen1's session + budget (current-lifecycle). ---
            engine.evict(&"rq".into());
            assert_eq!(cdr.budget.available(), 256, "gen1 evict restored budget");
            assert!(cdr.requests.is_empty(), "gen1 latch gone after evict");

            // --- gen2: re-latch the SAME rid under a fresh search S2. ---
            let s2 = SearchId::new();
            let cell2 = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));
            let blocks2 = cd_immutables(&mgr, 3, 200);
            let plhs2 = cd_block_hashes(&blocks2);
            let req2 = req("rq", 0, plhs2);
            engine.cd_interpose(
                s2,
                &MatchWindow::of(&req2),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell2,
                Vec::new(),
                blocks2,
            );
            wait_for(|| plane.count() >= 2).await;
            assert_eq!(cdr.budget.available(), 256 - 3 * BS, "gen2 reserved");
            let gen2_session = factory.last_opened().expect("gen2 session opened");

            // --- drop H1: release_search(S1) re-fetches the rid (now gen2). The
            //     generation guard (S1 != S2) makes the CD cleanup a no-op. ---
            engine.release_search(&s1);

            assert!(
                gen2_session.closed_reason().is_none(),
                "fresh (gen2) session must NOT be closed by the stale decline"
            );
            assert_eq!(
                cdr.budget.available(),
                256 - 3 * BS,
                "fresh (gen2) reservation must survive the stale decline"
            );
            assert!(
                cdr.requests.get("rq").is_some(),
                "fresh (gen2) lifecycle still latched"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // w8: eviction after commit closes + releases with "evicted".
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_evict_releases() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let blocks = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&blocks);
            let req = req("rq", 0, plhs);
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));

            engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                Vec::new(),
                blocks,
            );
            wait_for(|| plane.count() >= 1).await;
            assert_eq!(cdr.budget.available(), 256 - 3 * BS);

            engine.evict(&"rq".into());
            let session = factory.last_opened().unwrap();
            assert_eq!(
                session.closed_reason(),
                Some(Some("evicted".to_string())),
                "evict closes the session"
            );
            assert_eq!(cdr.budget.available(), 256, "budget restored");
            assert!(cdr.requests.is_empty());
            Ok(())
        }

        // ------------------------------------------------------------------
        // w9: a failed dispatch stashes the failure + closes; budget stays held.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_dispatch_failure_stashes_and_holds_budget() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::failing();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let blocks = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&blocks);
            let req = req("rq", 0, plhs);
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));

            let out = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                Vec::new(),
                blocks,
            );
            assert_eq!(
                out,
                MatchStatus::Matched { hit_blocks: 3 },
                "still latches the hit"
            );

            // The spawned dispatch resolves Err → stash + close, budget HELD.
            let state = cdr.requests.get("rq").expect("latched");
            wait_for(|| state.pending_failure().is_some()).await;
            assert!(
                state.pending_failure().is_some(),
                "pre-onboard failure stashed"
            );
            let session = factory.last_opened().unwrap();
            wait_for(|| session.closed_reason().is_some()).await;
            assert!(session.closed_reason().is_some(), "session closed");
            assert_eq!(
                cdr.budget.available(),
                256 - 3 * BS,
                "budget STILL held after dispatch failure"
            );
            assert!(cdr.requests.get("rq").is_some(), "latch persists");

            // The release hook frees it.
            engine.evict(&"rq".into());
            assert_eq!(cdr.budget.available(), 256, "evict frees the held budget");
            Ok(())
        }

        // ------------------------------------------------------------------
        // w10: zero-local-match Remote — pure remote prefill (empty commit set).
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_zero_local_match_commits_empty_remote() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);
            let cdr = engine.cd.as_ref().unwrap();

            // A window of 3 blocks, but ZERO matched locally.
            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            drop(window);
            let req = req("rq", 0, plhs);
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 0 }));

            let out = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 0 },
                &cell,
                Vec::new(),
                Vec::new(),
            );
            assert_eq!(
                out,
                MatchStatus::Matched { hit_blocks: 3 },
                "unified == window even with zero local match"
            );
            wait_for(|| plane.count() >= 1).await;

            let session = factory.last_opened().expect("session opened");
            let empty: Vec<SequenceHash> = Vec::new();
            // MockSession tolerates an empty commit set: pure remote prefill.
            assert_eq!(
                session.commit_calls(),
                vec![empty.clone()],
                "empty commit set"
            );
            assert_eq!(session.make_available_calls(), vec![empty]);
            assert!(session.finish_commits_called());
            assert!(session.finish_availability_called());
            assert_eq!(
                plane.last_num_provided(),
                Some(0),
                "nothing provided locally"
            );
            assert_eq!(
                plane.last_num_window(),
                Some(3 * BS),
                "window end still spans the full reserved window"
            );
            assert_eq!(
                cdr.budget.available(),
                256 - 3 * BS,
                "reserved the whole window"
            );
            assert_eq!(
                cdr.requests.get("rq").map(|s| s.unified_hit_blocks()),
                Some(3)
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // local_refresh (site 2) wiring: terminal-with-count commits Remote,
        // writing the unified answer THROUGH the shared cell post-guard.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_refresh_terminal_commits_remote() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(leader, cd);

            let mgr = cd_g2_manager(8);
            let blocks = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&blocks);

            // Inject a search whose single Ready shard holds the real blocks.
            let search_id = SearchId::new();
            let cell = Arc::new(Mutex::new(MatchStatus::Pending));
            let shard = OnboardingShard {
                start_block: 0,
                num_queried_blocks: 3,
                find_session: FindMatchesResult::Ready(ReadyResult::new(
                    blocks,
                    MatchBreakdown {
                        host_blocks: 3,
                        disk_blocks: 0,
                        object_blocks: 0,
                    },
                )),
            };
            engine.searches.insert(
                search_id,
                SearchState {
                    request_id: "rq".into(),
                    status: cell.clone(),
                    onboarding: OnboardingState::new(0, 3 * BS + 1, shard),
                    buffer: plhs.clone(),
                },
            );
            let req = req("rq", 0, plhs.clone());
            let status = engine.local_refresh(search_id, &MatchWindow::of(&req));
            assert_eq!(
                status,
                MatchStatus::Matched { hit_blocks: 3 },
                "refresh commits Remote → unified"
            );
            assert_eq!(
                *cell.lock().unwrap(),
                MatchStatus::Matched { hit_blocks: 3 },
                "the cell carries the CD answer (written post-guard)"
            );
            wait_for(|| plane.count() >= 1).await;
            assert_eq!(factory.open_count(), 1);
            assert_eq!(factory.last_opened().unwrap().commit_calls(), vec![plhs]);
            Ok(())
        }

        // ------------------------------------------------------------------
        // A non-CD engine never interposes (the default path is untouched).
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_absent_engine_passes_through() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, false);

            let mgr = cd_g2_manager(8);
            let blocks = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&blocks);
            let req = req("rq", 0, plhs);
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 }));

            let out = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req),
                MatchStatus::Matched { hit_blocks: 3 },
                &cell,
                Vec::new(),
                blocks,
            );
            assert_eq!(out, MatchStatus::Matched { hit_blocks: 3 }, "passthrough");
            Ok(())
        }

        // The factory wires CD from RemoteOps::with_disagg (no panic, returns a
        // live engine) — the dead-code keeper for `with_disagg` beyond w2.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_factory_wires_disagg() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let config = ConnectorEngineConfig {
                block_size: BS,
                remote: RemoteOps::default().with_disagg(
                    factory,
                    plane,
                    Arc::new(TierCell::default()),
                    DisaggConfig::default(),
                ),
            };
            let (_engine, _driver) =
                build_local_connector_engine(leader, NoopWorkerSink::new(), config, None);
            Ok(())
        }

        // ==================================================================
        // USAA onboard fan-out tests (local kick + remote pull pipeline)
        // ==================================================================

        use crate::InstanceId;
        use crate::object::ObjectBlockOps;
        use crate::p2p::session::CommittedBlock;
        use crate::worker::group::ParallelWorkers;
        use crate::worker::{ConnectRemoteResponse, RemoteDescriptor, Worker, WorkerTransfers};
        use kvbm_common::LogicalLayoutHandle;
        use kvbm_physical::manager::SerializedLayout;
        use kvbm_physical::transfer::{TransferCompleteNotification, TransferOptions};

        /// A `ParallelWorkers` double whose only useful behaviour is recording
        /// each `execute_local_transfer`'s G1 DESTINATION ids and returning a
        /// pre-completed notification (or, in failing mode, a dispatch `Err` —
        /// still recording the call) — enough for the fan-out's local kick +
        /// each remote run to "land" while the test asserts the local/remote
        /// dest split. Every other transfer/object surface is inert.
        struct RecordingWorkers {
            transfers: StdMutex<Vec<Vec<BlockId>>>,
            fail_local: std::sync::atomic::AtomicBool,
        }
        impl RecordingWorkers {
            fn new() -> Arc<Self> {
                Arc::new(Self {
                    transfers: StdMutex::new(Vec::new()),
                    fail_local: std::sync::atomic::AtomicBool::new(false),
                })
            }
            /// Make every subsequent `execute_local_transfer` fail at dispatch.
            fn fail_local_transfers(&self) {
                self.fail_local.store(true, Ordering::SeqCst);
            }
            /// The G1 dest id vec of each `execute_local_transfer`, in call order.
            fn transfer_calls(&self) -> Vec<Vec<BlockId>> {
                self.transfers.lock().unwrap().clone()
            }
        }
        impl WorkerTransfers for RecordingWorkers {
            fn execute_local_transfer(
                &self,
                _src: LogicalLayoutHandle,
                _dst: LogicalLayoutHandle,
                _src_block_ids: Arc<[BlockId]>,
                dst_block_ids: Arc<[BlockId]>,
                _options: TransferOptions,
            ) -> Result<TransferCompleteNotification> {
                self.transfers.lock().unwrap().push(dst_block_ids.to_vec());
                if self.fail_local.load(Ordering::SeqCst) {
                    anyhow::bail!("recording stub: injected local-transfer failure");
                }
                Ok(TransferCompleteNotification::completed())
            }
            fn execute_remote_onboard(
                &self,
                _src: RemoteDescriptor,
                _dst: LogicalLayoutHandle,
                _dst_block_ids: Arc<[BlockId]>,
                _options: TransferOptions,
            ) -> Result<TransferCompleteNotification> {
                anyhow::bail!("recording stub: execute_remote_onboard unused")
            }
            fn execute_remote_offload(
                &self,
                _src: LogicalLayoutHandle,
                _src_block_ids: Arc<[BlockId]>,
                _dst: RemoteDescriptor,
                _options: TransferOptions,
            ) -> Result<TransferCompleteNotification> {
                anyhow::bail!("recording stub: execute_remote_offload unused")
            }
            fn connect_remote(
                &self,
                _instance_id: InstanceId,
                _metadata: Vec<SerializedLayout>,
            ) -> Result<ConnectRemoteResponse> {
                Ok(ConnectRemoteResponse::ready())
            }
            fn has_remote_metadata(&self, _instance_id: InstanceId) -> bool {
                false
            }
            fn execute_remote_onboard_for_instance(
                &self,
                _instance_id: InstanceId,
                _remote_logical_type: LogicalLayoutHandle,
                _src_block_ids: Vec<BlockId>,
                _dst: LogicalLayoutHandle,
                _dst_block_ids: Arc<[BlockId]>,
                _options: TransferOptions,
            ) -> Result<TransferCompleteNotification> {
                anyhow::bail!("recording stub: execute_remote_onboard_for_instance unused")
            }
        }
        impl ObjectBlockOps for RecordingWorkers {
            fn has_blocks(
                &self,
                keys: Vec<SequenceHash>,
            ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
                Box::pin(async move { keys.into_iter().map(|k| (k, None)).collect() })
            }
            fn put_blocks(
                &self,
                keys: Vec<SequenceHash>,
                _layout: LogicalLayoutHandle,
                _block_ids: Vec<BlockId>,
            ) -> BoxFuture<'static, Vec<std::result::Result<SequenceHash, SequenceHash>>>
            {
                Box::pin(async move { keys.into_iter().map(Err).collect() })
            }
            fn get_blocks(
                &self,
                keys: Vec<SequenceHash>,
                _layout: LogicalLayoutHandle,
                _block_ids: Vec<BlockId>,
            ) -> BoxFuture<'static, Vec<std::result::Result<SequenceHash, SequenceHash>>>
            {
                Box::pin(async move { keys.into_iter().map(Err).collect() })
            }
        }
        impl ParallelWorkers for RecordingWorkers {
            fn export_metadata(&self) -> Result<Vec<crate::worker::SerializedLayoutResponse>> {
                Ok(Vec::new())
            }
            fn import_metadata(
                &self,
                _metadata: Vec<SerializedLayout>,
            ) -> Result<Vec<crate::worker::ImportMetadataResponse>> {
                Ok(Vec::new())
            }
            fn worker_count(&self) -> usize {
                0
            }
            fn workers(&self) -> &[Arc<dyn Worker>] {
                &[]
            }
        }

        /// Build a CD-configured engine over a real `InstanceLeader` whose
        /// `execute_local_transfer` is served by a [`RecordingWorkers`] stub and
        /// whose G2 manager owns enough blocks for the remote-pull allocations.
        async fn cd_onboard_engine(
            capacity: usize,
            factory: Arc<dyn SessionFactory>,
            plane: Arc<dyn PrefillPlane>,
            workers: Arc<dyn ParallelWorkers>,
        ) -> Result<Arc<LocalConnectorEngine>> {
            cd_onboard_engine_with_sink(capacity, factory, plane, workers, NoopWorkerSink::new())
                .await
        }

        /// [`cd_onboard_engine`] with an explicit worker sink (the round-trip
        /// test asserts what the decode sink saw).
        async fn cd_onboard_engine_with_sink(
            capacity: usize,
            factory: Arc<dyn SessionFactory>,
            plane: Arc<dyn PrefillPlane>,
            workers: Arc<dyn ParallelWorkers>,
            sink: Arc<dyn EngineWorkerSink>,
        ) -> Result<Arc<LocalConnectorEngine>> {
            use crate::testing::messenger::create_messenger_tcp;
            use kvbm_logical::blocks::BlockRegistry;

            let messenger = create_messenger_tcp().await?;
            let registry = BlockRegistry::builder().build();
            let g2 = Arc::new(
                TestManagerBuilder::<G2>::new()
                    .block_count(16)
                    .block_size(BS)
                    .registry(registry.clone())
                    .build(),
            );
            let leader = InstanceLeader::builder()
                .messenger(messenger)
                .registry(registry)
                .g2_manager(g2)
                .parallel_worker(workers)
                .build()?;
            let cd = cd_runtime(SelectionPolicy::Always, capacity, true, factory, plane);
            Ok(LocalConnectorEngine::with_offload_submit(
                Arc::new(leader),
                sink,
                BS,
                false,
                Arc::new(DisabledOffloadSubmit),
                Some(cd),
            ))
        }

        /// Latch a Remote commit for request `"rq"` (local hit = `local_hit`,
        /// window = the `window` blocks' hashes) and install a search whose
        /// onboarding holds the local-match span, returning the opened holder
        /// session and an onboard-ready handle reading the unified hit.
        fn latch_and_prepare_onboard(
            engine: &Arc<LocalConnectorEngine>,
            factory: &Arc<MockSessionFactory>,
            window: &[ImmutableBlock<G2>],
            local_hit: usize,
        ) -> (Arc<MockSession>, SearchId) {
            latch_and_prepare_onboard_with_prefix(engine, factory, &[], window, local_hit)
        }

        /// [`latch_and_prepare_onboard`] with a vLLM-computed prefix: the
        /// `prefix` blocks (which the caller must have registered in the
        /// ENGINE LEADER's own G2 — the residency gate matches there) precede
        /// the window at absolute `[0, prefix.len())`, and the installed
        /// search's onboarding carries the matching computed offset.
        fn latch_and_prepare_onboard_with_prefix(
            engine: &Arc<LocalConnectorEngine>,
            factory: &Arc<MockSessionFactory>,
            prefix: &[ImmutableBlock<G2>],
            window: &[ImmutableBlock<G2>],
            local_hit: usize,
        ) -> (Arc<MockSession>, SearchId) {
            let prefix_plhs = cd_block_hashes(prefix);
            let plhs = cd_block_hashes(window);
            let computed = prefix.len();
            let unified = window.len() as u32;

            // Search-time commit (the local-match blocks decode provides).
            let local_blocks: Vec<ImmutableBlock<G2>> = window[..local_hit].to_vec();
            let cell0 = Arc::new(Mutex::new(MatchStatus::Matched {
                hit_blocks: local_hit as u32,
            }));
            let req0 = req_with_prefix("rq", prefix_plhs.clone(), plhs.clone());
            let out = engine.cd_interpose(
                SearchId::new(),
                &MatchWindow::of(&req0),
                MatchStatus::Matched {
                    hit_blocks: local_hit as u32,
                },
                &cell0,
                prefix.to_vec(),
                local_blocks,
            );
            assert_eq!(
                out,
                MatchStatus::Matched {
                    hit_blocks: unified
                },
                "Remote commit latches the unified window"
            );

            // The search the connector later onboards: a terminal shard holding
            // the local-match span (window-relative — it starts AT the computed
            // offset), with the handle reading the unified count.
            let (shard, total) = if local_hit == 0 {
                (
                    OnboardingShard {
                        start_block: computed,
                        num_queried_blocks: 0,
                        find_session: ready_zero(),
                    },
                    computed * BS + 1,
                )
            } else {
                let span: Vec<ImmutableBlock<G2>> = window[..local_hit].to_vec();
                (
                    OnboardingShard {
                        start_block: computed,
                        num_queried_blocks: local_hit,
                        find_session: FindMatchesResult::Ready(ReadyResult::new(
                            span,
                            MatchBreakdown {
                                host_blocks: local_hit,
                                disk_blocks: 0,
                                object_blocks: 0,
                            },
                        )),
                    },
                    (computed + local_hit) * BS + 1,
                )
            };
            let search_id = SearchId::new();
            let cell = Arc::new(Mutex::new(MatchStatus::Matched {
                hit_blocks: unified,
            }));
            // The absolute-indexed buffer: `[0, computed)` then the window.
            let buffer: Vec<SequenceHash> = prefix_plhs.into_iter().chain(plhs).collect();
            engine.searches.insert(
                search_id,
                SearchState {
                    request_id: "rq".into(),
                    status: cell.clone(),
                    onboarding: OnboardingState::new(computed * BS, total, shard),
                    buffer,
                },
            );
            let session = factory.last_opened().expect("session opened at commit");
            (session, search_id)
        }

        // ------------------------------------------------------------------
        // u1: happy path — window 3, local 1, remote 2 -> Complete.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_happy_path_local_and_remote() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            let (session, handle) = latch_and_prepare_onboard(&engine, &factory, &window, 1);
            wait_for(|| plane.count() >= 1).await;
            assert_eq!(cdr.budget.available(), 256 - 3 * BS, "reserved the window");

            let dest = vec![50usize, 51, 52];
            let onboard = engine.clone().local_onboard(handle, &dest).unwrap();

            // Drive the puller side: the prefill peer commits + makes available
            // the 2 remote hashes; resolve the single pull.
            session.inject_peer_commit(vec![plhs[1], plhs[2]]);
            session.inject_peer_available(vec![
                CommittedBlock {
                    hash: plhs[1],
                    peer_block_id: 900,
                },
                CommittedBlock {
                    hash: plhs[2],
                    peer_block_id: 901,
                },
            ]);
            wait_for(|| !session.pull_calls().is_empty()).await;
            session.resolve_pull(0, Ok(()));

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));

            // The pull named exactly the remote hashes against allocated G2 dst.
            let pulls = session.pull_calls();
            assert_eq!(pulls.len(), 1);
            assert_eq!(pulls[0].0, vec![plhs[1], plhs[2]], "remote hashes pulled");
            assert_eq!(pulls[0].1.len(), 2, "two G2 dst blocks allocated");

            // G1 dest split: local kick to the first id, remote run to the rest.
            let transfers = workers.transfer_calls();
            assert_eq!(
                transfers,
                vec![vec![50], vec![51, 52]],
                "local kick targets local_g1, remote run targets remote_g1"
            );

            // Load terminal released the budget + finalized the session.
            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(cdr.budget.available(), 256, "budget released at terminal");
            assert!(session.finished_reason().is_some(), "session finalized");
            Ok(())
        }

        // ==================================================================
        // Seam-E2E (find_blocks -> onboard_blocks over the real dyn handle):
        // sibling failure isolation + sparse/out-of-order availability split.
        // Both drive the unified seam, not the internal `local_onboard` verb.
        // ==================================================================

        /// A [`PrefillPlane`] that fails dispatch ONLY for one request id and
        /// succeeds for every other. `RecordingPrefillPlane::failing()` fails
        /// ALL dispatches globally, so per-request failure isolation needs this
        /// selective double. Records the dispatched request ids so the test can
        /// gate on dispatch arrival.
        struct SelectivePrefillPlane {
            fail_request_id: String,
            dispatched: StdMutex<Vec<String>>,
        }
        impl SelectivePrefillPlane {
            fn new(fail_request_id: impl Into<String>) -> Arc<Self> {
                Arc::new(Self {
                    fail_request_id: fail_request_id.into(),
                    dispatched: StdMutex::new(Vec::new()),
                })
            }
            fn count(&self) -> usize {
                self.dispatched.lock().unwrap().len()
            }
        }
        impl PrefillPlane for SelectivePrefillPlane {
            fn dispatch(&self, req: PrefillDispatch) -> BoxFuture<'static, anyhow::Result<()>> {
                let rid = req.request_id;
                let fail = rid == self.fail_request_id;
                self.dispatched.lock().unwrap().push(rid.clone());
                async move {
                    if fail {
                        Err(anyhow::anyhow!("selective dispatch failure for {rid}"))
                    } else {
                        Ok(())
                    }
                }
                .boxed()
            }
        }

        /// Seam-E2E sibling isolation: two DISTINCT requests (r1, r2) latch CD
        /// lifecycles concurrently over `find_blocks`. r1's dispatch fails
        /// (selective plane); r2 proceeds. r1's terminal removes ONLY r1 from
        /// `cd.requests` and releases ONLY r1's budget; r2's reservation and CD
        /// state are untouched, and r2's `onboard_blocks` runs a REAL
        /// external-suffix transfer to a `Done` load terminal. This is the
        /// per-rid isolation invariant the state-level `release_if_matches` /
        /// `concurrent_release` unit tests guard, exercised through two distinct
        /// requests over the real seam.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_seam_sibling_dispatch_failure_isolates_to_r1() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = SelectivePrefillPlane::new("r1");
            let workers = RecordingWorkers::new();
            let sink = RecordingSink::new();
            let engine = cd_onboard_engine_with_sink(
                256,
                factory.clone(),
                plane.clone(),
                workers.clone(),
                sink.clone(),
            )
            .await?;
            let cdr = engine.cd.as_ref().unwrap();

            // Two DISJOINT 3-block hash chains (distinct start_token) so the two
            // requests never overlap-defer in the find router. Seed block[0] of
            // each into the engine leader's own G2 for a partial (local_hit=1)
            // hit — the residency the real find path needs to commit Remote.
            let mgr1 = cd_g2_manager(8);
            let w1 = cd_immutables(&mgr1, 3, 100);
            let h1 = cd_block_hashes(&w1);
            let mgr2 = cd_g2_manager(8);
            let w2 = cd_immutables(&mgr2, 3, 500);
            let h2 = cd_block_hashes(&w2);
            let _res1 = cd_immutables(engine.leader.g2_manager(), 1, 100);
            let _res2 = cd_immutables(engine.leader.g2_manager(), 1, 500);
            assert_eq!(cd_block_hashes(&_res1), h1[..1].to_vec());
            assert_eq!(cd_block_hashes(&_res2), h2[..1].to_vec());

            // Both lifecycles latch over the seam; both reserve their window.
            let r1_handle = expect_resolved(
                Arc::clone(&engine).find_blocks(&fb("r1", h1.clone(), 3 * BS + 1), None)?,
            )
            .1
            .expect("r1 latched");
            let r2_handle = expect_resolved(
                Arc::clone(&engine).find_blocks(&fb("r2", h2.clone(), 3 * BS + 1), None)?,
            )
            .1
            .expect("r2 latched");
            // r2's session opened at its commit; grab it before r1's async
            // failure handling can churn the factory.
            let r2_session = factory.last_opened().expect("r2 session opened at commit");
            wait_for(|| plane.count() >= 2).await;

            // Checkpoint A: both reserved their windows, both in cd.requests.
            assert_eq!(
                cdr.budget.available(),
                256 - 2 * (3 * BS),
                "both siblings reserved their windows"
            );
            assert!(cdr.requests.get("r1").is_some(), "r1 latched");
            let r2_state = cdr.requests.get("r2").expect("r2 latched");

            // r1's dispatch fails async → stash pending_failure; budget HELD
            // until r1's own terminal (no premature release).
            let r1_state = cdr.requests.get("r1").expect("r1 latched");
            wait_for(|| r1_state.pending_failure().is_some()).await;
            assert_eq!(
                cdr.budget.available(),
                256 - 2 * (3 * BS),
                "r1's dispatch failure holds its budget until its terminal"
            );
            assert!(
                r2_state.pending_failure().is_none(),
                "r2's lifecycle is unaffected by r1's dispatch failure"
            );

            // r1 terminal over the seam: onboard_blocks replays the stash →
            // FailedPartial (no pull, no transfer).
            let r1_onboard =
                Arc::clone(&engine).onboard_blocks(&r1_handle, &[60usize, 61, 62], 2 * BS)?;
            wait_for(|| r1_onboard.is_complete()).await;
            assert!(
                matches!(
                    r1_onboard.outcome(),
                    Some(LoadOutcome::FailedPartial { .. })
                ),
                "r1 failed (stash replay)"
            );

            // Checkpoint B: ONLY r1 released + removed; r2 reservation + state
            // survive r1's teardown.
            wait_for(|| cdr.requests.get("r1").is_none()).await;
            assert_eq!(
                cdr.budget.available(),
                256 - 3 * BS,
                "only r1's window released; r2's reservation untouched"
            );
            assert!(
                cdr.requests.get("r2").is_some(),
                "r2 CD state survives r1's teardown"
            );

            // r2 proceeds normally over the seam to a Done terminal.
            let r2_onboard =
                Arc::clone(&engine).onboard_blocks(&r2_handle, &[50usize, 51, 52], 2 * BS)?;
            r2_session.inject_peer_commit(vec![h2[1], h2[2]]);
            r2_session.inject_peer_available(vec![
                CommittedBlock {
                    hash: h2[1],
                    peer_block_id: 900,
                },
                CommittedBlock {
                    hash: h2[2],
                    peer_block_id: 901,
                },
            ]);
            wait_for(|| !r2_session.pull_calls().is_empty()).await;
            r2_session.resolve_pull(0, Ok(()));
            wait_for(|| r2_onboard.is_complete()).await;
            assert_eq!(
                r2_onboard.outcome(),
                Some(LoadOutcome::Done),
                "r2 completed despite r1's sibling failure"
            );

            // Non-vacuity: r2 ran a REAL external-suffix transfer (local kick +
            // remote run); r1's stash replay contributed none.
            assert_eq!(
                workers.transfer_calls(),
                vec![vec![50], vec![51, 52]],
                "only r2's local kick + remote run ran"
            );

            // Checkpoint C: r2's terminal released the rest; both gone; the sink
            // saw r2 Done and r1 as a failure (never Done).
            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(cdr.budget.available(), 256, "all budget released");
            wait_for(|| sink.loads().len() >= 2).await;
            let loads = sink.loads();
            assert!(
                loads.contains(&("r2".to_string(), LoadOutcome::Done)),
                "r2 reached the Done load terminal"
            );
            assert!(
                loads
                    .iter()
                    .any(|(rid, o)| rid == "r1" && matches!(o, LoadOutcome::FailedPartial { .. })),
                "r1's load terminal is a failure"
            );
            assert!(
                !loads
                    .iter()
                    .any(|(rid, o)| rid == "r1" && matches!(o, LoadOutcome::Done)),
                "r1 never reached Done"
            );
            Ok(())
        }

        /// Seam-E2E sparse availability: a remote/CD request whose external
        /// window spans 5 blocks receives peer availability SPARSELY and
        /// OUT-OF-ORDER across two `make_available` deltas. The pull pipeline
        /// regroups each delta into maximal contiguous runs — proving
        /// non-contiguous arrival is split into the correct pull runs (NOT a
        /// single contiguous pull). Drives the real `find_blocks` ->
        /// `onboard_blocks` seam and reaches `Done` with one
        /// `mark_load_finished(Done)`. The sparse/out-of-order counterpart to
        /// `cd_usaa_duplicate_hash_within_delta_pulls_once`.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_seam_sparse_availability_splits_into_contiguous_pull_runs() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let sink = RecordingSink::new();
            let engine = cd_onboard_engine_with_sink(
                256,
                factory.clone(),
                plane.clone(),
                workers.clone(),
                sink.clone(),
            )
            .await?;
            let cdr = engine.cd.as_ref().unwrap();

            // A 6-block window with block[0] resident (local_hit=1) → a 5-block
            // REMOTE slice (slots 0..5 of the remote pull). Seed block[0] into
            // the engine leader's own G2 for the partial hit.
            let mgr = cd_g2_manager(16);
            let window = cd_immutables(&mgr, 6, 100);
            let plhs = cd_block_hashes(&window);
            let _resident = cd_immutables(engine.leader.g2_manager(), 1, 100);
            assert_eq!(cd_block_hashes(&_resident), plhs[..1].to_vec());

            let handle = expect_resolved(
                Arc::clone(&engine).find_blocks(&fb("rq", plhs.clone(), 6 * BS + 1), None)?,
            )
            .1
            .expect("latched");
            wait_for(|| plane.count() >= 1).await;
            let session = factory.last_opened().expect("session opened at commit");

            // Remote slice = window[1..6]; remote slot i ↔ plhs[1 + i].
            let onboard = Arc::clone(&engine).onboard_blocks(
                &handle,
                &[50usize, 51, 52, 53, 54, 55],
                5 * BS,
            )?;

            // Commit the whole remote slice so the commit barrier clears and the
            // availability drain subscribes.
            session.inject_peer_commit(vec![plhs[1], plhs[2], plhs[3], plhs[4], plhs[5]]);

            // Delta 1: slots [0, 1, 4] delivered OUT OF ORDER. After the drain
            // sorts + regroups, this is the contiguous run [0,1] then the
            // isolated run [4] → a 2-block pull then a 1-block pull.
            session.inject_peer_available(vec![
                CommittedBlock {
                    hash: plhs[5], // slot 4
                    peer_block_id: 904,
                },
                CommittedBlock {
                    hash: plhs[1], // slot 0
                    peer_block_id: 900,
                },
                CommittedBlock {
                    hash: plhs[2], // slot 1
                    peer_block_id: 901,
                },
            ]);
            wait_for(|| !session.pull_calls().is_empty()).await;
            session.resolve_pull(0, Ok(()));
            wait_for(|| session.pull_calls().len() >= 2).await;
            session.resolve_pull(1, Ok(()));

            // Delta 2: slots [2, 3] — a LATER delta, injected only after
            // delta-1's runs landed so the availability stream cannot coalesce
            // them. This fills the gap between the earlier runs → one 2-block
            // run that does NOT merge with the already-pulled [0,1].
            session.inject_peer_available(vec![
                CommittedBlock {
                    hash: plhs[3], // slot 2
                    peer_block_id: 902,
                },
                CommittedBlock {
                    hash: plhs[4], // slot 3
                    peer_block_id: 903,
                },
            ]);
            wait_for(|| session.pull_calls().len() >= 3).await;
            session.resolve_pull(2, Ok(()));

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));

            // Non-vacuity: the sparse/out-of-order availability split into THREE
            // contiguous pull runs — NOT one contiguous 5-block pull.
            let pulls = session.pull_calls();
            let pulled_hashes: Vec<Vec<SequenceHash>> = pulls.iter().map(|p| p.0.clone()).collect();
            assert_eq!(
                pulled_hashes,
                vec![
                    vec![plhs[1], plhs[2]], // delta-1 run [slot 0, slot 1]
                    vec![plhs[5]],          // delta-1 run [slot 4]
                    vec![plhs[3], plhs[4]], // delta-2 run [slot 2, slot 3]
                ],
                "sparse arrival regrouped into contiguous runs, one pull per run"
            );
            assert_eq!(
                pulls.len(),
                3,
                "three runs from non-contiguous arrival, not a single 5-block pull"
            );
            assert_eq!(pulls[0].1.len(), 2, "first run: 2 G2 dst blocks");
            assert_eq!(pulls[1].1.len(), 1, "second run: 1 G2 dst block");
            assert_eq!(pulls[2].1.len(), 2, "third run: 2 G2 dst blocks");

            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(cdr.budget.available(), 256, "budget released at terminal");
            wait_for(|| !sink.loads().is_empty()).await;
            assert_eq!(sink.loads(), vec![("rq".to_string(), LoadOutcome::Done)]);
            Ok(())
        }

        // ------------------------------------------------------------------
        // Mint kind 2 (cd onboard): the mint records the UNIFIED window keyed
        // by the DRIVING search generation; the merged load terminal does NOT
        // clear it — the lifecycle release (the parked handle's drop) does.
        // The driver parks awaiting the remote pull, so record-at-mint is
        // observed deterministically before the terminal.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn inflight_cd_onboard_records_unified_window_clears_at_release() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            let (session, handle) = latch_and_prepare_onboard(&engine, &factory, &window, 1);
            wait_for(|| plane.count() >= 1).await;

            let dest = vec![50usize, 51, 52];
            let onboard = engine.clone().local_onboard(handle, &dest).unwrap();

            // Record-at-mint: the local kick lands then the driver parks on the
            // (uninjected) remote pull, so the unified window is observable here.
            {
                let g = engine.inflight.lock().unwrap();
                assert!(g.overlaps(&[plhs[0]]), "local span hash recorded");
                assert!(g.overlaps(&[plhs[1]]), "remote span hash recorded");
                assert!(g.overlaps(&[plhs[2]]), "remote span hash recorded");
                assert_eq!(g.len(), 3, "the full unified window is recorded");
            }

            // Drive the remote pull to completion → the merged load terminal.
            session.inject_peer_commit(vec![plhs[1], plhs[2]]);
            session.inject_peer_available(vec![
                CommittedBlock {
                    hash: plhs[1],
                    peer_block_id: 900,
                },
                CommittedBlock {
                    hash: plhs[2],
                    peer_block_id: 901,
                },
            ]);
            wait_for(|| !session.pull_calls().is_empty()).await;
            session.resolve_pull(0, Ok(()));
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));

            assert!(
                engine.inflight.lock().unwrap().overlaps(&[plhs[1]]),
                "the merged load terminal leaves the unified window recorded"
            );
            // The connector's recv-side handle drop is the release that clears.
            engine.release_search(&handle);
            assert!(
                engine.inflight.lock().unwrap().is_empty(),
                "the lifecycle release clears the unified window"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // u2: stash replay — a pre-onboard dispatch failure fails immediately
        // with ONLY the external slice ids.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_stash_replay_fails_external_slice_only() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::failing();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let (_session, handle) = latch_and_prepare_onboard(&engine, &factory, &window, 1);

            // The spawned dispatch resolves Err → stash + close, budget HELD.
            let state = cdr.requests.get("rq").expect("latched");
            wait_for(|| state.pending_failure().is_some()).await;
            assert_eq!(
                cdr.budget.available(),
                256 - 3 * BS,
                "budget held on failure"
            );

            // A trailing "new" dest block proves the external slice excludes it.
            let dest = vec![50usize, 51, 52, 53];
            let onboard = engine.clone().local_onboard(handle, &dest).unwrap();
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![50, 51, 52]
                }),
                "external slice only (the trailing new block is excluded)"
            );
            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(
                cdr.budget.available(),
                256,
                "budget released via failure flavor"
            );
            assert!(
                workers.transfer_calls().is_empty(),
                "stash replay issues no transfer"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // u3: commits closed short — fewer commits than expected fails with the
        // remote unfilled ids.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_commits_closed_short_fails_remote_unfilled() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            let (session, handle) = latch_and_prepare_onboard(&engine, &factory, &window, 1);

            let onboard = engine
                .clone()
                .local_onboard(handle, &[50usize, 51, 52])
                .unwrap();
            // Commit only 1 of the 2 expected remote hashes, then close short.
            session.inject_peer_commit(vec![plhs[1]]);
            session.inject_peer_finish_commits();

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![51, 52]
                }),
                "commit under-delivery fails the remote slice"
            );
            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(cdr.budget.available(), 256, "budget released");
            assert!(
                session.closed_reason().is_some(),
                "session closed on failure"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // u4: availability carrying an unexpected hash is SKIPPED (the peer
        // may publish a superset of this onboard's slice); the expected
        // hashes never arrive, so the Drained-with-shortfall guard fails the
        // remote unfilled ids — bounded, no wedge.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_availability_unexpected_hash_fails() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            let (session, handle) = latch_and_prepare_onboard(&engine, &factory, &window, 1);

            let onboard = engine
                .clone()
                .local_onboard(handle, &[50usize, 51, 52])
                .unwrap();
            // Commit the full remote slice so the pipeline reaches availability…
            session.inject_peer_commit(vec![plhs[1], plhs[2]]);
            // …then deliver a block whose hash was never in the remote slice
            // (skipped with a warn, not a bail) and drain with the expected
            // hashes still unfilled (the under-delivery guard).
            session.inject_peer_available(vec![CommittedBlock {
                hash: h(9999),
                peer_block_id: 700,
            }]);
            session.inject_peer_drained();

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![51, 52]
                }),
                "drained with the expected hashes unfilled fails the remote slice"
            );
            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(cdr.budget.available(), 256, "budget released");
            assert!(
                session.closed_reason().is_some(),
                "session closed on failure"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // u5: eviction mid-pipeline — the drains observe the session close, the
        // action fails, the fence is minted, and the budget is released exactly
        // once (identity-checked, no double release).
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_eviction_mid_pipeline_no_double_release() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let (session, handle) = latch_and_prepare_onboard(&engine, &factory, &window, 1);
            wait_for(|| plane.count() >= 1).await;
            assert_eq!(cdr.budget.available(), 256 - 3 * BS);

            // No commits injected: the remote pipeline parks on commits. Evict
            // arms the in-flight onboard, then closes the session — the commit
            // drain observes Closed and the driver bails.
            let onboard = engine
                .clone()
                .local_onboard(handle, &[50usize, 51, 52])
                .unwrap();
            let fence = engine.evict(&"rq".into()).fence;
            assert!(
                !fence.per_worker.is_empty(),
                "evict armed the in-flight onboard (fence minted)"
            );

            wait_for(|| onboard.is_complete()).await;
            assert!(
                matches!(onboard.outcome(), Some(LoadOutcome::FailedPartial { .. })),
                "an evicted in-flight onboard resolves Failed"
            );
            assert_eq!(
                session.closed_reason(),
                Some(Some("evicted".to_string())),
                "evict closed the session"
            );
            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(
                cdr.budget.available(),
                256,
                "budget released exactly once (load terminal no-ops after evict)"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // u6: zero-local-match — every external id is remote; the local kick is
        // skipped (only the remote run transfers).
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_zero_local_match_all_remote() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            // local_hit == 0: pure remote prefill, empty decode-side commit set.
            let (session, handle) = latch_and_prepare_onboard(&engine, &factory, &window, 0);

            let onboard = engine
                .clone()
                .local_onboard(handle, &[50usize, 51, 52])
                .unwrap();
            session.inject_peer_commit(vec![plhs[0], plhs[1], plhs[2]]);
            session.inject_peer_available(vec![
                CommittedBlock {
                    hash: plhs[0],
                    peer_block_id: 800,
                },
                CommittedBlock {
                    hash: plhs[1],
                    peer_block_id: 801,
                },
                CommittedBlock {
                    hash: plhs[2],
                    peer_block_id: 802,
                },
            ]);
            wait_for(|| !session.pull_calls().is_empty()).await;
            session.resolve_pull(0, Ok(()));

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));

            // No local kick — exactly one transfer (the remote run) covering all
            // three external dest ids.
            assert_eq!(
                workers.transfer_calls(),
                vec![vec![50, 51, 52]],
                "all external ids are remote; local kick skipped"
            );
            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(cdr.budget.available(), 256, "budget released at terminal");
            assert!(session.finished_reason().is_some(), "session finalized");
            Ok(())
        }

        // ------------------------------------------------------------------
        // USAA fan-out with a computed prefix: the external dest slice starts
        // AFTER the computed dest block, the local kick covers the matched
        // window span, and the remote pairs are the window's absolute
        // [computed+local_hit, computed+unified) hashes.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_computed_prefix_splits_external_window() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            // The prefix lives in the ENGINE LEADER's own G2 (the natural-
            // backfill precondition); the window blocks in a side manager.
            let prefix = cd_immutables(engine.leader.g2_manager(), 1, 5000);
            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            let (session, handle) =
                latch_and_prepare_onboard_with_prefix(&engine, &factory, &prefix, &window, 1);
            wait_for(|| plane.count() >= 1).await;
            assert_eq!(
                plane.last_num_provided(),
                Some(BS + BS),
                "num_provided = computed + local_hit*bs"
            );
            assert_eq!(cdr.budget.available(), 256 - 3 * BS, "suffix fbet only");

            // dest = [computed | external]: id 40 is vLLM's computed block.
            let dest = vec![40usize, 50, 51, 52];
            let onboard = engine.clone().local_onboard(handle, &dest).unwrap();

            session.inject_peer_commit(vec![plhs[1], plhs[2]]);
            session.inject_peer_available(vec![
                CommittedBlock {
                    hash: plhs[1],
                    peer_block_id: 900,
                },
                CommittedBlock {
                    hash: plhs[2],
                    peer_block_id: 901,
                },
            ]);
            wait_for(|| !session.pull_calls().is_empty()).await;
            session.resolve_pull(0, Ok(()));

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));

            // The pull names the window-relative remote hashes; the dest split
            // never touches the computed dest block.
            let pulls = session.pull_calls();
            assert_eq!(pulls[0].0, vec![plhs[1], plhs[2]], "remote slice hashes");
            assert_eq!(
                workers.transfer_calls(),
                vec![vec![50], vec![51, 52]],
                "local kick + remote run target the EXTERNAL slice only"
            );

            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(cdr.budget.available(), 256, "budget released at terminal");
            assert!(session.finished_reason().is_some(), "session finalized");
            Ok(())
        }

        // ------------------------------------------------------------------
        // USAA count guard with a computed prefix and a SHORT vLLM dest: the
        // clamped external slice trips the split guard and the failed action
        // carries the EXTERNAL slice only — never the computed dest block
        // (reporting it would force a recompute from token zero).
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_computed_prefix_short_dest_fails_external_only() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            let prefix = cd_immutables(engine.leader.g2_manager(), 1, 5000);
            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let (session, handle) =
                latch_and_prepare_onboard_with_prefix(&engine, &factory, &prefix, &window, 1);
            wait_for(|| plane.count() >= 1).await;

            // computed(1) + unified(3) needs 4 dest ids; 3 silently clamp the
            // external slice to [50, 51] — the count guard must fail the load.
            let dest = vec![40usize, 50, 51];
            let onboard = engine.clone().local_onboard(handle, &dest).unwrap();
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![50, 51]
                }),
                "the failed set is the clamped EXTERNAL slice — never id 40"
            );
            assert!(workers.transfer_calls().is_empty(), "no transfer issued");
            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(cdr.budget.available(), 256, "budget released");
            assert!(session.closed_reason().is_some(), "session closed");
            Ok(())
        }

        // ------------------------------------------------------------------
        // u7: eviction mid-PULL — the remote slice is committed + available and
        // the pull is IN FLIGHT (not resolved). Evict closes the session, which
        // fails the parked pull; the driver bails, the action resolves
        // Failed(Partial { remote unfilled }) with no hang, and the budget is
        // released exactly once. (Finding 1: a close mid-pull must not strand
        // the puller — exercised here through MockSession::close parity.)
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_eviction_mid_pull_resolves() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            let (session, handle) = latch_and_prepare_onboard(&engine, &factory, &window, 1);
            wait_for(|| plane.count() >= 1).await;
            assert_eq!(cdr.budget.available(), 256 - 3 * BS);

            let onboard = engine
                .clone()
                .local_onboard(handle, &[50usize, 51, 52])
                .unwrap();

            // Commit + make available the 2 remote hashes so the driver completes
            // the local kick and issues the remote pull — but DO NOT resolve it.
            // The pull is now parked in flight (its oneshot installed).
            session.inject_peer_commit(vec![plhs[1], plhs[2]]);
            session.inject_peer_available(vec![
                CommittedBlock {
                    hash: plhs[1],
                    peer_block_id: 900,
                },
                CommittedBlock {
                    hash: plhs[2],
                    peer_block_id: 901,
                },
            ]);
            wait_for(|| !session.pull_calls().is_empty()).await;
            assert!(
                session.finished_reason().is_none() && session.closed_reason().is_none(),
                "session still live with the pull parked"
            );

            // Evict mid-pull: cleanup closes the session, the close fails the
            // parked pull, the driver task bails (bounded — no hang).
            let fence = engine.evict(&"rq".into()).fence;
            assert!(
                !fence.per_worker.is_empty(),
                "evict armed the in-flight onboard"
            );

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![51, 52]
                }),
                "the parked pull failed; only the unfilled remote ids are reported"
            );
            assert_eq!(
                session.closed_reason(),
                Some(Some("evicted".to_string())),
                "evict closed the session"
            );
            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(
                cdr.budget.available(),
                256,
                "budget released exactly once (load terminal no-ops after evict)"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // u8: a STALE load terminal must not touch a FRESH lifecycle of the same
        // rid. Latch lifecycle 1, evict it (releases L1 + closes its session),
        // re-latch the SAME rid (lifecycle 2, fresh reservation + session), then
        // fire L1's terminal via complete_load with the STALE Arc. L2's budget
        // reservation and session must be untouched. (Finding 2: complete_load
        // tears down off the originating Arc, never a map re-fetch.)
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_stale_terminal_does_not_touch_fresh_lifecycle() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);

            // Lifecycle 1: latch + capture its Arc and session.
            let (session1, handle1) = latch_and_prepare_onboard(&engine, &factory, &window, 1);
            wait_for(|| plane.count() >= 1).await;
            let life1 = cdr.requests.get("rq").expect("lifecycle 1 tracked");
            assert_eq!(cdr.budget.available(), 256 - 3 * BS);

            // Evict lifecycle 1: closes its session, releases its budget, removes
            // it from the map. Drop its handle while the map is empty so the
            // The search-kind handle's release-on-drop is a clean no-op (and cannot tear
            // down lifecycle 2 once it is latched).
            engine.evict(&"rq".into());
            assert_eq!(
                session1.closed_reason(),
                Some(Some("evicted".to_string())),
                "lifecycle 1 session closed by evict"
            );
            assert_eq!(cdr.budget.available(), 256, "lifecycle 1 budget released");
            wait_for(|| cdr.requests.is_empty()).await;
            engine.release_search(&handle1);

            // Lifecycle 2: re-latch the SAME rid — fresh reservation + session.
            let (session2, _handle2) = latch_and_prepare_onboard(&engine, &factory, &window, 1);
            wait_for(|| plane.count() >= 2).await;
            let life2 = cdr.requests.get("rq").expect("lifecycle 2 tracked");
            assert!(
                !Arc::ptr_eq(&life1, &life2),
                "re-latch installed a distinct lifecycle"
            );
            assert_eq!(cdr.budget.available(), 256 - 3 * BS, "lifecycle 2 reserved");

            // Fire lifecycle 1's STALE terminal. With the fix it tears down off the
            // L1 Arc only: take_session(L1) is None (evict took it) and
            // release_if_matches(L1) no-ops on the ptr mismatch.
            cdr.complete_load(&"rq".into(), &life1, &ActionStatus::Complete);

            // Lifecycle 2 untouched: still tracked (same Arc), budget still
            // reserved, session neither finalized nor closed.
            let still = cdr.requests.get("rq").expect("lifecycle 2 still tracked");
            assert!(
                Arc::ptr_eq(&still, &life2),
                "stale terminal must not evict the fresh lifecycle"
            );
            assert_eq!(
                cdr.budget.available(),
                256 - 3 * BS,
                "stale terminal must not release the fresh reservation"
            );
            assert!(
                session2.finished_reason().is_none() && session2.closed_reason().is_none(),
                "stale terminal must not finalize/close the fresh session"
            );
            Ok(())
        }

        // ==================================================================
        // Prefill side: accept → pull pipeline → USAA kick
        // ==================================================================

        /// Records every peer resolve the prefill pipeline performs. In failing
        /// mode it records the call then errors, so the pull pipeline bails
        /// before it can attach — the "prefill failed to attach" path — while
        /// the recorded count still reports how many pipelines were spawned.
        struct RecordingResolver {
            calls: StdMutex<Vec<InstanceId>>,
            fail: bool,
        }
        impl RecordingResolver {
            fn new() -> Arc<Self> {
                Arc::new(Self {
                    calls: StdMutex::new(Vec::new()),
                    fail: false,
                })
            }
            fn failing() -> Arc<Self> {
                Arc::new(Self {
                    calls: StdMutex::new(Vec::new()),
                    fail: true,
                })
            }
            fn calls(&self) -> Vec<InstanceId> {
                self.calls.lock().unwrap().clone()
            }
        }
        impl PeerResolver for RecordingResolver {
            fn resolve_and_register(
                &self,
                instance_id: InstanceId,
            ) -> BoxFuture<'_, anyhow::Result<()>> {
                self.calls.lock().unwrap().push(instance_id);
                let fail = self.fail;
                async move {
                    if fail {
                        Err(anyhow::anyhow!("resolve boom"))
                    } else {
                        Ok(())
                    }
                }
                .boxed()
            }
        }

        /// The paired-session prefill harness: a CD engine over the PULLER
        /// half of a `MockSessionFactory::make_paired` pair (so the
        /// decode-side holder is a real paired session each test drives), a
        /// recording sink/workers/resolver, and a 3-block provided window.
        struct PrefillRig {
            holder_f: Arc<MockSessionFactory>,
            puller_f: Arc<MockSessionFactory>,
            resolver: Arc<RecordingResolver>,
            sink: Arc<RecordingSink>,
            workers: Arc<RecordingWorkers>,
            engine: Arc<LocalConnectorEngine>,
            window: Vec<ImmutableBlock<G2>>,
            plhs: Vec<SequenceHash>,
            /// Keeps the window blocks' manager (and so the blocks) alive.
            _mgr: Arc<BlockManager<G2>>,
        }

        /// The standard prefill-side CD config (matches the rig defaults).
        fn prefill_cfg() -> DisaggConfig {
            DisaggConfig {
                selection: SelectionPolicy::Always,
                max_inflight_remote_prefill_tokens: 256,
                local_fallback_on_overload: true,
                ..DisaggConfig::default()
            }
        }

        /// [`prefill_cfg`] with shrunk output-drain knobs for the deferred-
        /// finalize tests.
        fn prefill_cfg_with_drain(
            poll: std::time::Duration,
            watchdog: std::time::Duration,
        ) -> DisaggConfig {
            DisaggConfig {
                output_drain_poll: poll,
                output_drain_watchdog: watchdog,
                ..prefill_cfg()
            }
        }

        /// Build the prefill-side CD engine over an explicit session factory
        /// and config, with caller-supplied sink/workers doubles.
        async fn prefill_engine_with_cfg(
            sink: Arc<dyn EngineWorkerSink>,
            workers: Arc<RecordingWorkers>,
            sessions: Arc<dyn SessionFactory>,
            resolver: Arc<dyn PeerResolver>,
            cfg: DisaggConfig,
        ) -> Result<Arc<LocalConnectorEngine>> {
            use crate::testing::messenger::create_messenger_tcp;
            use kvbm_logical::blocks::BlockRegistry;

            let messenger = create_messenger_tcp().await?;
            let registry = BlockRegistry::builder().build();
            let g2 = Arc::new(
                TestManagerBuilder::<G2>::new()
                    .block_count(16)
                    .block_size(BS)
                    .registry(registry.clone())
                    .build(),
            );
            let leader = InstanceLeader::builder()
                .messenger(messenger)
                .registry(registry)
                .g2_manager(g2)
                .parallel_worker(workers as Arc<dyn ParallelWorkers>)
                .build()?;
            let cd = CdRuntime::new(
                cfg,
                Arc::new(TierCell::default()),
                sessions,
                RecordingPrefillPlane::ok(),
                Some(resolver),
            );
            Ok(LocalConnectorEngine::with_offload_submit(
                Arc::new(leader),
                sink,
                BS,
                false,
                Arc::new(DisabledOffloadSubmit),
                Some(cd),
            ))
        }

        /// Build the prefill-side CD engine over the PULLER half of a paired
        /// mock-session factory, with caller-supplied sink/workers doubles.
        async fn prefill_engine_with(
            sink: Arc<dyn EngineWorkerSink>,
            workers: Arc<RecordingWorkers>,
            puller_f: Arc<MockSessionFactory>,
            resolver: Arc<RecordingResolver>,
        ) -> Result<Arc<LocalConnectorEngine>> {
            prefill_engine_with_cfg(sink, workers, puller_f, resolver, prefill_cfg()).await
        }

        async fn prefill_rig() -> Result<PrefillRig> {
            let (holder_f, puller_f) = MockSessionFactory::make_paired();
            let resolver = RecordingResolver::new();
            let sink = RecordingSink::new();
            let workers = RecordingWorkers::new();
            let engine = prefill_engine_with(
                sink.clone(),
                workers.clone(),
                puller_f.clone(),
                resolver.clone(),
            )
            .await?;

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            Ok(PrefillRig {
                holder_f,
                puller_f,
                resolver,
                sink,
                workers,
                engine,
                window,
                plhs,
                _mgr: mgr,
            })
        }

        fn open_holder(rig: &PrefillRig, session_id: Uuid) -> Arc<MockSession> {
            rig.holder_f.open(session_id).expect("holder open");
            rig.holder_f.last_opened().expect("holder session recorded")
        }

        /// Test-local RAII stand-in for the connector's parked prefill lifecycle
        /// handle (the unified seam parks one opaque `FindBlocksHandle`); this
        /// lease fires `prefill_release` on drop so the existing drop-site release
        /// assertions stay verbatim.
        struct PrefillLease {
            engine: Arc<LocalConnectorEngine>,
            request_id: RequestId,
            accept_id: AcceptId,
        }
        impl PrefillLease {
            fn accept_id(&self) -> AcceptId {
                self.accept_id
            }
        }
        impl Drop for PrefillLease {
            fn drop(&mut self) {
                self.engine
                    .prefill_release(&self.request_id, self.accept_id);
            }
        }

        fn prefill_accept_args(
            plhs: &[SequenceHash],
            session_id: Uuid,
            initiator: InstanceId,
            endpoint: Option<SessionEndpoint>,
            provided_tokens: usize,
            computed_tokens: usize,
        ) -> (RemotePrefillParams, Vec<SequenceHash>, usize) {
            let mut params = RemotePrefillParams::new(session_id, initiator);
            params.decode_endpoint = endpoint;
            params.num_provided_tokens = provided_tokens;
            (params, plhs.to_vec(), computed_tokens)
        }

        /// Accept "rq" against the rig's window and unwrap the first-latch
        /// handle. `provided`/`computed` are in tokens.
        fn accept_rq(
            rig: &PrefillRig,
            session_id: Uuid,
            initiator: InstanceId,
            endpoint: Option<SessionEndpoint>,
            provided: usize,
            computed: usize,
        ) -> Result<(PrefillLease, usize)> {
            let (params, hashes, computed) = prefill_accept_args(
                &rig.plhs, session_id, initiator, endpoint, provided, computed,
            );
            match rig.engine.clone().prefill_accept_core(
                &"rq".to_string(),
                &params,
                &hashes,
                computed,
            )? {
                PrefillAcceptCore::Accepted {
                    accept_id,
                    external_tokens,
                    ..
                } => Ok((
                    PrefillLease {
                        engine: rig.engine.clone(),
                        request_id: "rq".to_string(),
                        accept_id,
                    },
                    external_tokens,
                )),
                PrefillAcceptCore::Refreshed { .. } => {
                    panic!("first accept must latch, not refresh")
                }
            }
        }

        // ------------------------------------------------------------------
        // p1: happy path, USAA after pulls — attach recorded, holder
        // commits + avails, kick copies the external G2 suffix onto the
        // external G1 slice, terminal Complete reaches the sink; the session
        // is NOT finalized and our planes NOT sealed until the release.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_happy_path_usaa_after_pulls() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let (handle, external_tokens) =
                accept_rq(&rig, session_id, initiator, holder.endpoint(), 3 * BS, BS)?;
            assert_eq!(external_tokens, 2 * BS);

            // Pipeline resolved the peer then attached with the dispatch's
            // exact session/peer/endpoint.
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            let attaches = rig.puller_f.attach_calls();
            assert_eq!(attaches[0].0, session_id);
            assert_eq!(attaches[0].1, initiator);
            assert_eq!(attaches[0].2.kind, "mock");
            assert_eq!(
                rig.resolver.calls(),
                vec![initiator],
                "peer resolved before attach"
            );

            // Decode-side holder publishes the full provided window.
            holder.commit(rig.plhs.clone())?;
            holder.make_available(rig.window.clone())?;
            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pulls_complete()).await;

            let puller = rig.puller_f.last_attached().expect("attached");
            assert_eq!(puller.pull_calls().len(), 1, "one contiguous-run pull");
            assert_eq!(puller.pull_calls()[0].0, rig.plhs, "whole window pulled");

            // USAA with the FULL allocation [computed prefix | external].
            let onboard = rig
                .engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            assert_eq!(
                rig.workers.transfer_calls(),
                vec![vec![50, 51]],
                "kick targets exactly the external G1 suffix"
            );
            wait_for(|| !rig.sink.loads().is_empty()).await;
            assert_eq!(
                rig.sink.loads(),
                vec![("rq".to_string(), LoadOutcome::Done)]
            );

            // Output direction stays open: no finalize, no sealed planes, the
            // state stays latched.
            assert!(puller.finished_reason().is_none() && puller.closed_reason().is_none());
            assert!(!puller.finish_commits_called() && !puller.finish_availability_called());
            assert!(
                cdr.prefill.get("rq").is_some(),
                "state latched until release"
            );

            // RAII release: pipeline complete + unfailed → finalize (deferred
            // through the output drain, which is empty here — no output owed).
            drop(onboard);
            drop(handle);
            assert!(cdr.prefill.get("rq").is_none(), "release removed the state");
            wait_for(|| puller.finished_reason().is_some()).await;
            Ok(())
        }

        // ------------------------------------------------------------------
        // p2: USAA BEFORE pulls complete — the kick parks, the pipeline
        // fires it on completion, and exactly one transfer ever runs
        // (with p1's reverse interleave this pins the two-phase handshake).
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_usaa_before_pulls_fires_exactly_one_kick() -> Result<()> {
            let rig = prefill_rig().await?;
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let (handle, _) =
                accept_rq(&rig, session_id, initiator, holder.endpoint(), 3 * BS, BS)?;
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;

            // Nothing committed yet: pulls are incomplete, so the kick parks.
            let onboard = rig
                .engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();
            assert!(!onboard.is_complete(), "kick parked until pulls complete");
            assert!(rig.workers.transfer_calls().is_empty());

            holder.commit(rig.plhs.clone())?;
            holder.make_available(rig.window.clone())?;

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            assert_eq!(
                rig.workers.transfer_calls(),
                vec![vec![50, 51]],
                "exactly one kick fired across both handshake sides"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // p3: pre-USAA failure stash + replay — the pipeline fails before
        // USAA (no decode endpoint), the state stays LATCHED, and the replay
        // mints an immediately-Failed action naming the EXTERNAL SLICE ONLY
        // (a naive impl reporting the full allocation, or dropping the state
        // at failure time, fails these asserts).
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_pre_usaa_failure_stash_replays_external_slice_only() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let initiator: InstanceId = Uuid::new_v4().into();

            // No decode endpoint → the pipeline fails before attach.
            let (handle, _) = accept_rq(&rig, session_id, initiator, None, 3 * BS, BS)?;
            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pending_failure().is_some()).await;
            assert!(
                cdr.prefill.get("rq").is_some(),
                "pre-USAA failure keeps the state latched for the replay"
            );

            let onboard = rig
                .engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![50, 51]
                }),
                "replay covers the external slice only — never the computed prefix"
            );
            assert!(cdr.prefill.get("rq").is_none(), "state removed at replay");
            assert!(rig.workers.transfer_calls().is_empty(), "no transfer ran");
            Ok(())
        }

        // ------------------------------------------------------------------
        // The pre-USAA replay path: the suffix is recorded at the prefill
        // mint, the replay resolves the action Failed AND removes the
        // engine-side state — but the record survives until the connector's
        // handle release, whose `release_prefill_session` must clear by the
        // handle's OWN accept generation even though the map entry is gone.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn inflight_prefill_pre_usaa_replay_record_clears_at_release() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let initiator: InstanceId = Uuid::new_v4().into();

            // No decode endpoint → the pipeline fails before attach (pre-USAA).
            let (handle, _) = accept_rq(&rig, session_id, initiator, None, 3 * BS, BS)?;
            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pending_failure().is_some()).await;

            let onboard = rig
                .engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();
            wait_for(|| onboard.is_complete()).await;
            assert!(
                matches!(onboard.outcome(), Some(LoadOutcome::FailedPartial { .. })),
                "the replay resolves Failed"
            );
            assert!(cdr.prefill.get("rq").is_none(), "state removed at replay");
            assert!(
                !rig.engine.inflight.lock().unwrap().is_empty(),
                "the replay terminal leaves the recorded suffix in place"
            );
            // The handle release clears it — past the state removal, keyed by
            // the handle's own generation.
            drop(handle);
            assert!(
                rig.engine.inflight.lock().unwrap().is_empty(),
                "the lifecycle release clears the recorded suffix"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // The failure-path re-issue bound: when the pull pipeline fails to
        // attach (a failing peer-resolve, before any attach), the state stays
        // LATCHED with the stashed failure (pre-USAA). vLLM then re-polls GNMT
        // every scheduler step while the request waits in WAITING_FOR_REMOTE_KVS
        // — the idempotent arm. Each re-poll must REFRESH the shared cell and
        // NEVER re-spawn the pipeline: the spawn count is bounded to one per
        // generation, no matter how many times the failed lifecycle is polled.
        // A re-spawn-per-poll would hammer resolve/attach until the decode
        // watchdog (the legacy hot-retry shape). The stashed failure then
        // surfaces promptly at the next onboard — fail-fast over the external
        // slice, not a stall to the watchdog.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_failed_attach_repoll_storm_spawns_pipeline_once() -> Result<()> {
            let (holder_f, puller_f) = MockSessionFactory::make_paired();
            let resolver = RecordingResolver::failing();
            let sink = RecordingSink::new();
            let workers = RecordingWorkers::new();
            let engine = prefill_engine_with(
                sink.clone(),
                workers.clone(),
                puller_f.clone(),
                resolver.clone(),
            )
            .await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);

            let session_id = Uuid::new_v4();
            holder_f.open(session_id).expect("holder open");
            let holder = holder_f.last_opened().expect("holder session recorded");
            let initiator: InstanceId = Uuid::new_v4().into();

            // First accept latches the lifecycle and spawns the pull pipeline.
            let handle = {
                let (params, hashes, computed) = prefill_accept_args(
                    &plhs,
                    session_id,
                    initiator,
                    holder.endpoint(),
                    3 * BS,
                    BS,
                );
                match engine.clone().prefill_accept_core(
                    &"rq".to_string(),
                    &params,
                    &hashes,
                    computed,
                )? {
                    PrefillAcceptCore::Accepted { accept_id, .. } => PrefillLease {
                        engine: engine.clone(),
                        request_id: "rq".to_string(),
                        accept_id,
                    },
                    PrefillAcceptCore::Refreshed { .. } => {
                        panic!("first accept must latch, not refresh")
                    }
                }
            };

            // The pipeline resolves the peer once; the resolve fails, so it
            // bails before any attach and stashes the failure pre-USAA.
            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pending_failure().is_some()).await;
            assert_eq!(
                resolver.calls().len(),
                1,
                "the pipeline spawned and resolved exactly once"
            );
            assert!(
                puller_f.attach_calls().is_empty(),
                "the resolve failed before any attach"
            );

            // The hot-retry scenario: many GNMT re-polls against the latched-
            // but-failed lifecycle. Each is the idempotent re-poll.
            for i in 0..64 {
                let (params, hashes, computed) = prefill_accept_args(
                    &plhs,
                    session_id,
                    initiator,
                    holder.endpoint(),
                    3 * BS,
                    BS,
                );
                let outcome = engine.clone().prefill_accept_core(
                    &"rq".to_string(),
                    &params,
                    &hashes,
                    computed,
                )?;
                assert!(
                    matches!(outcome, PrefillAcceptCore::Refreshed { .. }),
                    "re-poll {i} must refresh, never re-spawn the pipeline"
                );
            }

            // Give a straggler spawn time to surface, then assert the bound held
            // across the whole storm: one spawn, one resolve, zero attaches.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            assert_eq!(
                resolver.calls().len(),
                1,
                "the re-poll storm spawned no second pipeline"
            );
            assert!(
                puller_f.attach_calls().is_empty(),
                "still no attach after the re-poll storm"
            );

            // The stashed failure surfaces promptly at the next onboard — a
            // terminal Failed over the external slice, not a stall to the
            // decode watchdog.
            let onboard = engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![50, 51]
                }),
                "fail-fast over the external slice"
            );
            assert!(cdr.prefill.get("rq").is_none(), "state removed at replay");
            assert!(workers.transfer_calls().is_empty(), "no transfer ran");
            drop(handle);
            Ok(())
        }

        // ------------------------------------------------------------------
        // p4: decode closes the session mid-pull (post-USAA) — the pipeline
        // fails bounded, the latched action resolves Failed over the external
        // slice, the session is closed, the state removed.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_decode_close_mid_pull_fails_external_slice() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let (handle, _) =
                accept_rq(&rig, session_id, initiator, holder.endpoint(), 3 * BS, BS)?;
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;

            // Commits land, availability never does; USAA latches the kick;
            // then decode dies.
            holder.commit(rig.plhs.clone())?;
            let onboard = rig
                .engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();
            holder.close(Some("decode died".to_string()));

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![50, 51]
                })
            );
            wait_for(|| cdr.prefill.get("rq").is_none()).await;
            let puller = rig.puller_f.last_attached().expect("attached");
            assert!(puller.closed_reason().is_some(), "engine closed its side");
            assert!(rig.workers.transfer_calls().is_empty(), "no transfer ran");
            Ok(())
        }

        // ------------------------------------------------------------------
        // p5: stale-generation release — after an evict-equivalent release +
        // re-accept of the SAME rid, dropping the OLD handle must not touch
        // the fresh lifecycle (the same-key-different-Arc shape; a release
        // keyed only by rid fails this).
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_stale_generation_release_no_ops_on_fresh_lifecycle() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let initiator: InstanceId = Uuid::new_v4().into();

            // Generation 1.
            let session1 = Uuid::new_v4();
            let holder1 = open_holder(&rig, session1);
            let (handle1, _) =
                accept_rq(&rig, session1, initiator, holder1.endpoint(), 3 * BS, BS)?;
            let state1 = cdr.prefill.get("rq").expect("gen 1 latched");

            // Evict-equivalent teardown of generation 1 (the connector's
            // eviction path releases through the same engine target the RAII
            // drop uses).
            rig.engine
                .release_prefill_session(&"rq".to_string(), handle1.accept_id());
            assert!(cdr.prefill.get("rq").is_none(), "gen 1 released");

            // Generation 2: same rid, fresh session + state.
            let session2 = Uuid::new_v4();
            let holder2 = open_holder(&rig, session2);
            let (_handle2, _) =
                accept_rq(&rig, session2, initiator, holder2.endpoint(), 3 * BS, BS)?;
            let state2 = cdr.prefill.get("rq").expect("gen 2 latched");
            assert!(!Arc::ptr_eq(&state1, &state2), "fresh lifecycle installed");
            wait_for(|| state2.has_session()).await;

            // The OLD handle drops AFTER the re-accept: the accept_id guard
            // must no-op on the fresh lifecycle.
            drop(handle1);
            let still = cdr.prefill.get("rq").expect("fresh lifecycle untouched");
            assert!(Arc::ptr_eq(&still, &state2));
            assert!(
                state2.has_session(),
                "stale release must not take/close the fresh session"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // p6: zero-external (prefill already computed everything) — accept
        // reports 0 external tokens, the pipeline still attaches and pulls
        // (cache-warming parity with the legacy run_setup), and no kick runs.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_zero_external_warms_cache_without_kick() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            // computed >= provided ⇒ external saturates to 0.
            let (handle, external_tokens) = accept_rq(
                &rig,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                4 * BS,
            )?;
            assert_eq!(external_tokens, 0);

            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            holder.commit(rig.plhs.clone())?;
            holder.make_available(rig.window.clone())?;
            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pulls_complete()).await;

            let puller = rig.puller_f.last_attached().expect("attached");
            assert_eq!(puller.pull_calls().len(), 1, "cache-warming pull still ran");
            assert_eq!(puller.pull_calls()[0].0, rig.plhs);
            assert!(rig.workers.transfer_calls().is_empty(), "no kick");

            drop(handle);
            assert!(cdr.prefill.get("rq").is_none());
            // The finalize defers through the (empty) output drain.
            wait_for(|| puller.finished_reason().is_some()).await;
            Ok(())
        }

        // ------------------------------------------------------------------
        // p7: idempotent re-accept — the second poll refreshes the external
        // count through the shared cell and spawns no second pipeline.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_re_accept_refreshes_without_second_attach() -> Result<()> {
            let rig = prefill_rig().await?;
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let (_handle, external_tokens) =
                accept_rq(&rig, session_id, initiator, holder.endpoint(), 3 * BS, BS)?;
            assert_eq!(external_tokens, 2 * BS);
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;

            // Re-poll with a larger computed prefix.
            let (params, hashes, computed) = prefill_accept_args(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                2 * BS,
            );
            let outcome = rig.engine.clone().prefill_accept_core(
                &"rq".to_string(),
                &params,
                &hashes,
                computed,
            )?;
            let PrefillAcceptCore::Refreshed { external_tokens } = outcome else {
                panic!("re-poll must refresh, never mint a second handle");
            };
            assert_eq!(external_tokens, BS, "recomputed against the new prefix");

            // No second pipeline: give a straggler spawn time to surface.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            assert_eq!(rig.puller_f.attach_calls().len(), 1, "no second attach");
            Ok(())
        }

        // ------------------------------------------------------------------
        // p7b: stale-latch re-dispatch — the decode side recompute-
        // reschedules the same rid onto a FRESH session, so the next accept
        // arrives with the SAME rid but a DIFFERENT session_id. It must
        // REPLACE the abandoned lifecycle (latch + attach the fresh session),
        // not answer as an idempotent re-poll against the dead one — keying
        // by rid alone would attach nothing for the fresh session and wedge
        // decode until its watchdog.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_session_mismatch_replaces_lifecycle() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let initiator: InstanceId = Uuid::new_v4().into();

            // Generation 1: latch + attach the soon-to-be-abandoned session.
            let session1 = Uuid::new_v4();
            let holder1 = open_holder(&rig, session1);
            let (handle1, _) =
                accept_rq(&rig, session1, initiator, holder1.endpoint(), 3 * BS, BS)?;
            let accept1 = handle1.accept_id();
            let state1 = cdr.prefill.get("rq").expect("gen 1 latched");
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            wait_for(|| state1.has_session()).await;
            let puller1 = rig.puller_f.last_attached().expect("gen 1 attached");

            // Generation 2 re-dispatch: SAME rid, DIFFERENT session_id. Drive
            // accept_core directly — `accept_rq` panics on a Refreshed outcome.
            let session2 = Uuid::new_v4();
            let holder2 = open_holder(&rig, session2);
            let (params2, hashes2, computed2) = prefill_accept_args(
                &rig.plhs,
                session2,
                initiator,
                holder2.endpoint(),
                3 * BS,
                BS,
            );
            let outcome = rig.engine.clone().prefill_accept_core(
                &"rq".to_string(),
                &params2,
                &hashes2,
                computed2,
            )?;
            let PrefillAcceptCore::Accepted {
                accept_id: accept2, ..
            } = outcome
            else {
                panic!("session_id mismatch must replace the lifecycle, not refresh");
            };
            assert_ne!(accept2, accept1, "a fresh generation was latched");

            // The fresh pipeline attaches the NEW session...
            wait_for(|| rig.puller_f.attach_calls().len() == 2).await;
            assert_eq!(rig.puller_f.attach_calls()[1].0, session2);

            // ...and the abandoned generation-1 session is torn down (closed,
            // not finalized: gen 1 never reached pulls-complete).
            wait_for(|| puller1.closed_reason().is_some()).await;
            let state2 = cdr.prefill.get("rq").expect("gen 2 latched");
            assert!(!Arc::ptr_eq(&state1, &state2), "fresh lifecycle installed");

            // The stale gen-1 handle drop must no-op on the fresh lifecycle.
            drop(handle1);
            let still = cdr.prefill.get("rq").expect("fresh lifecycle untouched");
            assert!(Arc::ptr_eq(&still, &state2));
            Ok(())
        }

        // ------------------------------------------------------------------
        // p7c: same-session re-poll is the idempotent arm — it must REFRESH
        // the shared cell and leave the live session attached, never trip the
        // mismatch teardown that p7b drives.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_same_session_repoll_keeps_lifecycle() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let (_handle, _) =
                accept_rq(&rig, session_id, initiator, holder.endpoint(), 3 * BS, BS)?;
            let state1 = cdr.prefill.get("rq").expect("gen 1 latched");
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            wait_for(|| state1.has_session()).await;
            let puller = rig.puller_f.last_attached().expect("attached");

            // Re-poll the SAME session with a larger computed prefix.
            let (params, hashes, computed) = prefill_accept_args(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                2 * BS,
            );
            let outcome = rig.engine.clone().prefill_accept_core(
                &"rq".to_string(),
                &params,
                &hashes,
                computed,
            )?;
            let PrefillAcceptCore::Refreshed { external_tokens } = outcome else {
                panic!("same-session re-poll must refresh, not replace the lifecycle");
            };
            assert_eq!(external_tokens, BS, "recomputed against the new prefix");

            // The lifecycle is untouched: same Arc, same attached session, no
            // teardown, no second attach.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let still = cdr.prefill.get("rq").expect("lifecycle survives");
            assert!(Arc::ptr_eq(&still, &state1));
            assert_eq!(rig.puller_f.attach_calls().len(), 1, "no second attach");
            assert!(puller.closed_reason().is_none(), "live session not closed");
            Ok(())
        }

        // ------------------------------------------------------------------
        // p8: commits under-delivery — the holder seals its commit plane
        // before the expected count; the pipeline fails (bounded, no hang)
        // and the latched action resolves Failed over the external slice.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_commits_under_delivery_fails() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let (handle, _) =
                accept_rq(&rig, session_id, initiator, holder.endpoint(), 3 * BS, BS)?;
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            let onboard = rig
                .engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();

            // 1 of 3 expected hashes, then Closed.
            holder.commit(vec![rig.plhs[0]])?;
            holder.finish_commits()?;

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![50, 51]
                })
            );
            wait_for(|| cdr.prefill.get("rq").is_none()).await;
            Ok(())
        }

        // ------------------------------------------------------------------
        // The computed>0 under-delivery wedge shape: a decode that commits
        // only its window SUFFIX while DNPT promised the whole absolute
        // [0, computed + local) window leaves the prefill's commit barrier
        // unsatisfiable. It must fail FAST on the close terminator — never
        // park until session teardown.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_suffix_only_commit_fails_fast() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            // DNPT = 3 blocks: the decode promised absolute [0, 3).
            let (handle, _) =
                accept_rq(&rig, session_id, initiator, holder.endpoint(), 3 * BS, BS)?;
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            let onboard = rig
                .engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();

            // The naive suffix-only commit: window hashes WITHOUT the prefix.
            holder.commit(rig.plhs[1..].to_vec())?;
            holder.finish_commits()?;

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![50, 51]
                }),
                "fails fast over the external slice — no wedge"
            );
            wait_for(|| cdr.prefill.get("rq").is_none()).await;
            Ok(())
        }

        // ------------------------------------------------------------------
        // p9: committed-set mismatch — a hash outside the expected window
        // means the two sides disagree on the provided slice; failure, not
        // a pull against unknown keys.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_commit_outside_window_fails() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let (handle, _) =
                accept_rq(&rig, session_id, initiator, holder.endpoint(), 3 * BS, BS)?;
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            let onboard = rig
                .engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();

            holder.commit(vec![h(9999)])?;

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![50, 51]
                })
            );
            wait_for(|| cdr.prefill.get("rq").is_none()).await;
            assert!(rig.workers.transfer_calls().is_empty(), "no pull, no kick");
            Ok(())
        }

        /// A sink that parks the caller INSIDE `mark_load_finished` until the
        /// test opens the gate — pinning the engine task at the exact point
        /// where a load terminal is already observable (the status cell is
        /// written) but the code after the terminal has not yet run.
        struct GatedSink {
            entered: std::sync::atomic::AtomicBool,
            gate: StdMutex<bool>,
            unblock: std::sync::Condvar,
        }
        impl GatedSink {
            fn new() -> Arc<Self> {
                Arc::new(Self {
                    entered: std::sync::atomic::AtomicBool::new(false),
                    gate: StdMutex::new(false),
                    unblock: std::sync::Condvar::new(),
                })
            }
            fn entered(&self) -> bool {
                self.entered.load(Ordering::SeqCst)
            }
            fn open(&self) {
                *self.gate.lock().unwrap() = true;
                self.unblock.notify_all();
            }
        }
        impl EngineWorkerSink for GatedSink {
            fn mark_load_finished(&self, _req: &RequestId, _outcome: LoadOutcome) {
                self.entered.store(true, Ordering::SeqCst);
                let mut open = self.gate.lock().unwrap();
                while !*open {
                    open = self.unblock.wait(open).unwrap();
                }
            }
            fn mark_save_finished(&self, _req: &RequestId, _outcome: SaveOutcome) {}
            fn mark_fence_complete(&self, _token: FenceToken) {}
        }

        // ------------------------------------------------------------------
        // p10: a FAILED kick must never reach the decode peer as a clean
        // cooperative Finished. The Failed terminal is what lets the
        // connector reap the slot and drop the session handle, so the RAII
        // release can run its finalize-vs-close decision while the kick task
        // is still inside its failure tail — and that decision must already
        // see the failure stash. The gated sink parks the kick task inside
        // the terminal notification (terminal observable, post-terminal code
        // not yet run) while the test drops the handles.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_kick_failure_racing_release_closes_not_finalizes() -> Result<()> {
            let (holder_f, puller_f) = MockSessionFactory::make_paired();
            let resolver = RecordingResolver::new();
            let sink = GatedSink::new();
            let workers = RecordingWorkers::new();
            workers.fail_local_transfers();
            let engine =
                prefill_engine_with(sink.clone(), workers.clone(), puller_f.clone(), resolver)
                    .await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);

            let session_id = Uuid::new_v4();
            holder_f.open(session_id).expect("holder open");
            let holder = holder_f.last_opened().expect("holder session recorded");
            let initiator: InstanceId = Uuid::new_v4().into();

            let handle = {
                let (params, hashes, computed) = prefill_accept_args(
                    &plhs,
                    session_id,
                    initiator,
                    holder.endpoint(),
                    3 * BS,
                    BS,
                );
                match engine.clone().prefill_accept_core(
                    &"rq".to_string(),
                    &params,
                    &hashes,
                    computed,
                )? {
                    PrefillAcceptCore::Accepted { accept_id, .. } => PrefillLease {
                        engine: engine.clone(),
                        request_id: "rq".to_string(),
                        accept_id,
                    },
                    PrefillAcceptCore::Refreshed { .. } => {
                        panic!("first accept must latch, not refresh")
                    }
                }
            };
            wait_for(|| puller_f.attach_calls().len() == 1).await;

            holder.commit(plhs.clone())?;
            holder.make_available(window.clone())?;
            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pulls_complete()).await;

            // USAA fires the kick; its G2→G1 dispatch fails, the Failed
            // terminal lands, and the sink gate parks the kick task there.
            let onboard = engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();
            wait_for(|| sink.entered()).await;
            assert!(
                onboard.is_complete(),
                "terminal observable while the kick task is parked"
            );

            // The reap that terminal triggers: handles drop, the RAII release
            // runs inside the window.
            drop(onboard);
            drop(handle);

            let puller = puller_f.last_attached().expect("attached");
            let finished = puller.finished_reason();
            let closed = puller.closed_reason();

            // Unpark the kick task BEFORE asserting so a failing assert can't
            // strand a worker thread on the gate.
            sink.open();

            assert!(
                closed.is_some(),
                "racing release must CLOSE the failed lifecycle's session, got finalize={finished:?}"
            );
            assert!(
                finished.is_none(),
                "failed kick must not reach the peer as a clean cooperative Finished"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // Mint kind 3 (prefill onboard): the mint records the EXTERNAL suffix
        // of the decode-provided window, keyed by the accept generation; the
        // USAA kick terminal leaves it recorded and the lifecycle release
        // (`release_prefill_session` via the handle drop) clears it.
        // the USAA onboard runs BEFORE the pull completes, so the kick
        // latches (not fired) and the record is observed before the terminal.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn inflight_prefill_onboard_records_suffix_clears_at_release() -> Result<()> {
            let rig = prefill_rig().await?;
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            // provided = 3 blocks, computed = 1 block → external = 2 blocks.
            let (handle, external) =
                accept_rq(&rig, session_id, initiator, holder.endpoint(), 3 * BS, BS)?;
            assert_eq!(external, 2 * BS);
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;

            // USAA before the pull completes: the kick latches, the action parks.
            let onboard = rig
                .engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();
            {
                let g = rig.engine.inflight.lock().unwrap();
                assert!(g.overlaps(&[rig.plhs[1]]), "external suffix hash recorded");
                assert!(g.overlaps(&[rig.plhs[2]]), "external suffix hash recorded");
                assert!(
                    !g.overlaps(&[rig.plhs[0]]),
                    "the provided-window prefix is computed, not external"
                );
                assert_eq!(g.len(), 2, "only the external suffix is recorded");
            }

            // Drive the pull to completion → fires the latched kick → terminal.
            holder.commit(rig.plhs.clone())?;
            holder.make_available(rig.window.clone())?;
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            assert!(
                rig.engine.inflight.lock().unwrap().overlaps(&[rig.plhs[1]]),
                "the kick terminal leaves the external suffix recorded"
            );
            // The connector's eventual handle drop is the release that clears.
            drop(handle);
            assert!(
                rig.engine.inflight.lock().unwrap().is_empty(),
                "the lifecycle release clears the external suffix"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // The only prefill record site is the onboard mint: an accepted
        // lifecycle that is never USAA'd (accept → attach → release, no
        // onboard call) must leave the guard empty end to end — there is no
        // entry to leak because nothing records before the mint.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn inflight_prefill_accept_without_usaa_records_nothing() -> Result<()> {
            let rig = prefill_rig().await?;
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let (handle, external) =
                accept_rq(&rig, session_id, initiator, holder.endpoint(), 3 * BS, BS)?;
            assert_eq!(external, 2 * BS);
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            assert!(
                rig.engine.inflight.lock().unwrap().is_empty(),
                "accept alone must not record into the guard"
            );

            drop(handle);
            wait_for(|| {
                rig.puller_f
                    .last_attached()
                    .unwrap()
                    .closed_reason()
                    .is_some()
            })
            .await;
            assert!(
                rig.engine.inflight.lock().unwrap().is_empty(),
                "release of a never-onboarded lifecycle leaves the guard empty"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // Release wins the pre-attach race AFTER a USAA kick latched: the
        // release's `take_session` finds nothing parked (attach in flight)
        // and the pipeline's Refused park is then the ONLY surviving owner
        // of the latched kick — it must resolve the load action through the
        // driver terminal (Failed over the external slice) and complete the
        // eviction fence. A Refused arm that exits without consuming the
        // kick leaves the action Pending forever and the fence never
        // completes. The guard suffix clears earlier, at the evict's
        // engine-internal `prefill_release` (the evict-teardown funnel).
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_release_before_attach_resolves_latched_kick() -> Result<()> {
            let (holder_f, puller_f) = MockSessionFactory::make_paired();
            let gate = Arc::new(GatedResolver::default());
            let sink = RecordingSink::new();
            let engine = prefill_engine_with_cfg(
                sink.clone(),
                RecordingWorkers::new(),
                puller_f.clone(),
                gate.clone(),
                prefill_cfg(),
            )
            .await?;
            let rid = "rq".to_string();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);

            let session_id = Uuid::new_v4();
            holder_f.open(session_id).expect("holder open");
            let holder = holder_f.last_opened().expect("holder recorded");
            let initiator: InstanceId = Uuid::new_v4().into();

            // provided = 3 blocks, computed = 1 block → external = 2 blocks.
            // The pipeline parks at the resolver: no attach, nothing parked.
            let handle = {
                let (params, hashes, computed) = prefill_accept_args(
                    &plhs,
                    session_id,
                    initiator,
                    holder.endpoint(),
                    3 * BS,
                    BS,
                );
                match engine
                    .clone()
                    .prefill_accept_core(&rid, &params, &hashes, computed)?
                {
                    PrefillAcceptCore::Accepted { accept_id, .. } => PrefillLease {
                        engine: engine.clone(),
                        request_id: rid.clone(),
                        accept_id,
                    },
                    PrefillAcceptCore::Refreshed { .. } => panic!("first accept must latch"),
                }
            };
            assert!(
                puller_f.attach_calls().is_empty(),
                "pipeline parked pre-attach"
            );

            // USAA while the attach is in flight: the kick latches (pulls
            // incomplete) and the guard records the external suffix.
            let onboard = engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();
            assert!(!onboard.is_complete(), "kick parked until pulls complete");
            assert!(
                engine
                    .inflight
                    .lock()
                    .unwrap()
                    .overlaps(&[plhs[1], plhs[2]]),
                "external suffix recorded at mint"
            );

            // Evict while the pull is in flight (the WAITING_FOR_REMOTE_KVS
            // eviction): the fence arms over the Pending action, then the
            // connector's view-detach drops the session handle — the release
            // claims cleanup and finds no session to close.
            let fence = engine.evict(&rid).fence;
            assert!(
                !fence.per_worker.is_empty(),
                "evict arms a fence over the live onboard"
            );
            assert!(
                engine.inflight.lock().unwrap().is_empty(),
                "evict's internal prefill teardown is a release funnel — it \
                 clears the recorded suffix (the fence covers the worker side)"
            );
            drop(handle);

            // The attach lands AFTER the release: the park is Refused and the
            // pipeline exits cleanly — it owns resolving the orphaned kick.
            gate.open();
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial {
                    block_ids: vec![50, 51]
                }),
                "the orphaned kick resolves Failed over the external slice"
            );
            // The fence-armed terminal completes the eviction fence instead
            // of firing `mark_load_finished`.
            wait_for(|| sink.fences().len() == fence.per_worker.len()).await;
            assert!(
                sink.loads().is_empty(),
                "a fenced terminal fires no load_finished"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // p11: a window hash duplicated WITHIN one availability delta (the
        // wire replay coalesces deltas, so a double publish can merge into
        // one) must not break the prefill window drain: the duplicate maps
        // to the same slot twice, which violated `group_contiguous_runs`'
        // strictly-increasing contract (debug panic → dead pipeline, pulls
        // never complete) and split a second single-block pull of an
        // already-pulled hash in release. The per-delta dedup pulls each
        // window hash exactly once and the kick still completes.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_duplicate_hash_within_delta_pulls_once() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let (handle, _) =
                accept_rq(&rig, session_id, initiator, holder.endpoint(), 3 * BS, BS)?;
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;

            holder.commit(rig.plhs.clone())?;
            // ONE make_available call → ONE delta carrying window[0] twice.
            holder.make_available(vec![
                rig.window[0].clone(),
                rig.window[0].clone(),
                rig.window[1].clone(),
                rig.window[2].clone(),
            ])?;

            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pulls_complete()).await;
            let puller = rig.puller_f.last_attached().expect("attached");
            let pulls = puller.pull_calls();
            assert_eq!(pulls.len(), 1, "the duplicate must not split a second pull");
            assert_eq!(pulls[0].0, rig.plhs, "each window hash pulled exactly once");

            let onboard = rig
                .engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), handle.accept_id(), &[40usize, 50, 51])
                .unwrap();
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            assert_eq!(
                rig.workers.transfer_calls(),
                vec![vec![50, 51]],
                "kick targets exactly the external G1 suffix"
            );
            Ok(())
        }

        // ==================================================================
        // Prefill OUTPUT direction: register-observer publish + deferred
        // finalize (the decode side of the wire is the paired holder).
        // ==================================================================

        /// Register blocks staged by EXPLICIT hashes in `manager`. The output
        /// fixtures need mid-chain hashes without their parent blocks, so this
        /// stages by `(hash, block_size)` — the same path the pull pipelines
        /// use — instead of completing a token chain from block zero.
        fn blocks_for_hashes(
            manager: &Arc<BlockManager<G2>>,
            hashes: &[SequenceHash],
        ) -> Vec<ImmutableBlock<G2>> {
            let mutables = manager.allocate_blocks(hashes.len()).expect("alloc");
            let completes: Vec<_> = mutables
                .into_iter()
                .zip(hashes)
                .map(|(b, hash)| b.stage(*hash, BS).expect("stage"))
                .collect();
            manager.register_blocks(completes)
        }

        fn peer_committed_hashes(session: &MockSession) -> std::collections::HashSet<SequenceHash> {
            session
                .peer_committed()
                .as_slice()
                .iter()
                .copied()
                .collect()
        }

        fn peer_available_hashes(session: &MockSession) -> std::collections::HashSet<SequenceHash> {
            session
                .peer_available()
                .as_slice()
                .iter()
                .map(|b| b.hash)
                .collect()
        }

        /// Accept `"rq"` directly against `engine` (rig-free variant of
        /// [`accept_rq`] for tests that build their own engine), with
        /// `num_computed_tokens = 0`.
        fn accept_direct(
            engine: &Arc<LocalConnectorEngine>,
            plhs: &[SequenceHash],
            session_id: Uuid,
            initiator: InstanceId,
            endpoint: Option<SessionEndpoint>,
            provided: usize,
        ) -> Result<PrefillLease> {
            let (params, hashes, computed) =
                prefill_accept_args(plhs, session_id, initiator, endpoint, provided, 0);
            match engine.clone().prefill_accept_core(
                &"rq".to_string(),
                &params,
                &hashes,
                computed,
            )? {
                PrefillAcceptCore::Accepted { accept_id, .. } => Ok(PrefillLease {
                    engine: engine.clone(),
                    request_id: "rq".to_string(),
                    accept_id,
                }),
                PrefillAcceptCore::Refreshed { .. } => {
                    panic!("first accept must latch, not refresh")
                }
            }
        }

        /// [`prefill_rig`] with caller-chosen drain knobs.
        async fn prefill_rig_with_cfg(cfg: DisaggConfig) -> Result<PrefillRig> {
            let (holder_f, puller_f) = MockSessionFactory::make_paired();
            let resolver = RecordingResolver::new();
            let sink = RecordingSink::new();
            let workers = RecordingWorkers::new();
            let engine = prefill_engine_with_cfg(
                sink.clone(),
                workers.clone(),
                puller_f.clone(),
                resolver.clone(),
                cfg,
            )
            .await?;

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            Ok(PrefillRig {
                holder_f,
                puller_f,
                resolver,
                sink,
                workers,
                engine,
                window,
                plhs,
                _mgr: mgr,
            })
        }

        // ------------------------------------------------------------------
        // o1: output happy path — observed registered blocks matching the
        // expected-output set publish into the parked session (commit +
        // make_available with exactly the matching hashes), unmatched blocks
        // are ignored, and the residual shrinks per delivery.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_output_happy_path_publishes_matches_only() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();
            let rid = "rq".to_string();

            // Provided window = 1 block; the remaining 2 are owed as output.
            let (handle, _) = accept_rq(&rig, session_id, initiator, holder.endpoint(), BS, 0)?;
            let aid = handle.accept_id();
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            holder.commit(rig.plhs[..1].to_vec())?;
            holder.make_available(rig.window[..1].to_vec())?;
            let state = cdr.prefill.get(&rid).expect("latched");
            wait_for(|| state.pulls_complete()).await;
            let puller = rig.puller_f.last_attached().expect("attached");

            // First delivery: one matching output block + one unrelated block.
            let g2 = rig.engine.leader.g2_manager();
            let out1 = blocks_for_hashes(g2, &rig.plhs[1..2]);
            let unrelated = blocks_for_hashes(g2, &[h(9999)]);
            cdr.output.observe(&[out1[0].clone(), unrelated[0].clone()]);

            assert_eq!(
                puller.commit_calls(),
                vec![vec![rig.plhs[1]]],
                "exactly the matching hash committed; the unrelated block ignored"
            );
            assert_eq!(puller.make_available_calls(), vec![vec![rig.plhs[1]]]);
            assert_eq!(cdr.output.residual(&rid, aid), 1, "residual shrank by one");
            assert!(cdr.output.has_pending(&rid, aid));

            // Second delivery drains the residual.
            let out2 = blocks_for_hashes(g2, &rig.plhs[2..3]);
            cdr.output.observe(&out2);
            assert_eq!(
                puller.commit_calls(),
                vec![vec![rig.plhs[1]], vec![rig.plhs[2]]]
            );
            assert!(!cdr.output.has_pending(&rid, aid), "residual drained");

            // The holder (decode) side observed both publishes.
            assert_eq!(
                peer_committed_hashes(&holder),
                rig.plhs[1..3].iter().copied().collect()
            );
            assert_eq!(
                peer_available_hashes(&holder),
                rig.plhs[1..3].iter().copied().collect()
            );

            drop(handle);
            wait_for(|| puller.finished_reason().is_some()).await;
            Ok(())
        }

        /// A resolver that parks the pipeline BEFORE attach until released —
        /// holds the pre-park window open deterministically.
        #[derive(Default)]
        struct GatedResolver {
            release: tokio::sync::Notify,
        }
        impl GatedResolver {
            fn open(&self) {
                self.release.notify_one();
            }
        }
        impl PeerResolver for GatedResolver {
            fn resolve_and_register(
                &self,
                _instance_id: InstanceId,
            ) -> BoxFuture<'_, anyhow::Result<()>> {
                async move {
                    self.release.notified().await;
                    Ok(())
                }
                .boxed()
            }
        }

        // ------------------------------------------------------------------
        // o2: pre-park buffering — output observed before the session parks
        // is buffered (nothing on the session), and the park drains it under
        // ONE guard: published exactly once, nothing lost.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_output_pre_park_buffer_drains_once_at_attach() -> Result<()> {
            let (holder_f, puller_f) = MockSessionFactory::make_paired();
            let gate = Arc::new(GatedResolver::default());
            let engine = prefill_engine_with_cfg(
                RecordingSink::new(),
                RecordingWorkers::new(),
                puller_f.clone(),
                gate.clone(),
                prefill_cfg(),
            )
            .await?;
            let cdr = engine.cd.as_ref().unwrap();
            let rid = "rq".to_string();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);

            let session_id = Uuid::new_v4();
            holder_f.open(session_id).expect("holder open");
            let holder = holder_f.last_opened().expect("holder recorded");
            let initiator: InstanceId = Uuid::new_v4().into();

            let handle =
                accept_direct(&engine, &plhs, session_id, initiator, holder.endpoint(), BS)?;
            let aid = handle.accept_id();

            // The pipeline is parked at the resolver: no attach, no park.
            assert!(puller_f.attach_calls().is_empty());

            // Output lands BEFORE the park → buffered; nothing on any session.
            let out = blocks_for_hashes(engine.leader.g2_manager(), &plhs[1..3]);
            cdr.output.observe(&out);
            assert!(
                !cdr.output.has_pending(&rid, aid),
                "residual consumed into the buffer"
            );
            assert!(
                peer_committed_hashes(&holder).is_empty(),
                "nothing published before the park"
            );

            // Release the pipeline: attach → park drains the buffer under the
            // slot guard — exactly one publish, nothing lost.
            gate.open();
            wait_for(|| puller_f.attach_calls().len() == 1).await;
            let puller = puller_f.last_attached().expect("attached");
            wait_for(|| !puller.commit_calls().is_empty()).await;
            assert_eq!(
                puller.commit_calls(),
                vec![plhs[1..3].to_vec()],
                "the buffer drained in one publish, never twice"
            );
            assert_eq!(puller.make_available_calls(), vec![plhs[1..3].to_vec()]);
            assert_eq!(
                peer_committed_hashes(&holder),
                plhs[1..3].iter().copied().collect()
            );
            assert_eq!(
                peer_available_hashes(&holder),
                plhs[1..3].iter().copied().collect()
            );
            drop(handle);
            Ok(())
        }

        // ------------------------------------------------------------------
        // o3: already-in-G2 (the legacy hole's reproducer) — expected-output
        // blocks G2-resident BEFORE accept never re-register, so the
        // observer alone would never deliver them and the decode side would
        // wedge until the watchdog. The accept-time sweep must publish them:
        // they reach the holder with NO observe() call, the residual is
        // empty, and the release finalize does not hang.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_output_already_in_g2_publishes_without_observe() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();
            let rid = "rq".to_string();

            // Resident BEFORE accept (held so they stay pinned).
            let resident = blocks_for_hashes(rig.engine.leader.g2_manager(), &rig.plhs[1..3]);

            let (handle, _) = accept_rq(&rig, session_id, initiator, holder.endpoint(), BS, 0)?;
            let aid = handle.accept_id();

            // The sweep consumed the residual at accept — no observe() ever runs.
            assert!(!cdr.output.has_pending(&rid, aid));

            // The swept blocks reach the holder once the session parks.
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            let puller = rig.puller_f.last_attached().expect("attached");
            wait_for(|| !puller.commit_calls().is_empty()).await;
            assert_eq!(puller.commit_calls(), vec![rig.plhs[1..3].to_vec()]);
            assert_eq!(
                peer_committed_hashes(&holder),
                rig.plhs[1..3].iter().copied().collect()
            );

            // Drive the pull side so the release takes the finalize path.
            holder.commit(rig.plhs[..1].to_vec())?;
            holder.make_available(rig.window[..1].to_vec())?;
            let state = cdr.prefill.get(&rid).expect("latched");
            wait_for(|| state.pulls_complete()).await;

            // Release does NOT hang: nothing is left pending.
            drop(handle);
            wait_for(|| puller.finished_reason().is_some()).await;
            drop(resident);
            Ok(())
        }

        // ------------------------------------------------------------------
        // o4: deferred finalize — releasing a parked, pulls-complete
        // lifecycle while output is still owed must NOT finalize; the
        // finalize fires only once the observer delivers the rest.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_release_defers_finalize_until_output_drains() -> Result<()> {
            let rig = prefill_rig_with_cfg(prefill_cfg_with_drain(
                std::time::Duration::from_millis(5),
                std::time::Duration::from_secs(5),
            ))
            .await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();
            let rid = "rq".to_string();

            let (handle, _) = accept_rq(&rig, session_id, initiator, holder.endpoint(), BS, 0)?;
            let aid = handle.accept_id();
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            holder.commit(rig.plhs[..1].to_vec())?;
            holder.make_available(rig.window[..1].to_vec())?;
            let state = cdr.prefill.get(&rid).expect("latched");
            wait_for(|| state.pulls_complete()).await;
            let puller = rig.puller_f.last_attached().expect("attached");

            // Release with 2 output blocks still owed: the state leaves the
            // map inline, but the finalize defers behind the drain.
            drop(handle);
            assert!(
                cdr.prefill.get(&rid).is_none(),
                "release removes the state inline (a re-accept must not collide)"
            );
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            assert!(
                puller.finished_reason().is_none(),
                "finalize must wait while output is owed"
            );

            // The observer delivers the rest → publish, then finalize.
            let out = blocks_for_hashes(rig.engine.leader.g2_manager(), &rig.plhs[1..3]);
            cdr.output.observe(&out);
            wait_for(|| puller.finished_reason().is_some()).await;
            assert_eq!(
                puller.commit_calls(),
                vec![rig.plhs[1..3].to_vec()],
                "the late output published before the finalize"
            );
            assert!(
                !cdr.output.is_tracked(&rid, aid),
                "drain untracked the entry"
            );
            Ok(())
        }

        /// Gate parking the caller INSIDE the wrapped session's `commit` —
        /// pins the output dispatch in its in-flight window (observer
        /// residual already consumed, session sends not yet landed).
        struct CommitGate {
            open: StdMutex<bool>,
            unblock: std::sync::Condvar,
            entered: std::sync::atomic::AtomicBool,
        }
        impl CommitGate {
            fn new() -> Arc<Self> {
                Arc::new(Self {
                    open: StdMutex::new(false),
                    unblock: std::sync::Condvar::new(),
                    entered: std::sync::atomic::AtomicBool::new(false),
                })
            }
            fn entered(&self) -> bool {
                self.entered.load(Ordering::SeqCst)
            }
            fn open(&self) {
                *self.open.lock().unwrap() = true;
                self.unblock.notify_all();
            }
            fn wait(&self) {
                let mut open = self.open.lock().unwrap();
                while !*open {
                    open = self.unblock.wait(open).unwrap();
                }
            }
        }

        /// Wraps the puller factory so attached sessions gate their `commit`.
        struct GatedCommitFactory {
            inner: Arc<MockSessionFactory>,
            gate: Arc<CommitGate>,
        }
        impl SessionFactory for GatedCommitFactory {
            fn open(&self, session_id: SessionId) -> Result<Arc<dyn Session>> {
                self.inner.open(session_id)
            }
            fn attach(
                &self,
                session_id: SessionId,
                peer_instance_id: InstanceId,
                peer_endpoint: SessionEndpoint,
            ) -> BoxFuture<'static, Result<Arc<dyn Session>>> {
                let fut = self
                    .inner
                    .attach(session_id, peer_instance_id, peer_endpoint);
                let gate = Arc::clone(&self.gate);
                async move {
                    let inner = fut.await?;
                    Ok(Arc::new(GatedCommitSession { inner, gate }) as Arc<dyn Session>)
                }
                .boxed()
            }
            fn active_session_count(&self) -> usize {
                self.inner.active_session_count()
            }
        }

        struct GatedCommitSession {
            inner: Arc<dyn Session>,
            gate: Arc<CommitGate>,
        }
        impl Session for GatedCommitSession {
            fn session_id(&self) -> SessionId {
                self.inner.session_id()
            }
            fn endpoint(&self) -> Option<SessionEndpoint> {
                self.inner.endpoint()
            }
            fn commit(&self, hashes: Vec<SequenceHash>) -> Result<()> {
                self.gate.entered.store(true, Ordering::SeqCst);
                self.gate.wait();
                self.inner.commit(hashes)
            }
            fn finish_commits(&self) -> Result<()> {
                self.inner.finish_commits()
            }
            fn make_available(&self, blocks: Vec<ImmutableBlock<G2>>) -> Result<()> {
                self.inner.make_available(blocks)
            }
            fn finish_availability(&self) -> Result<()> {
                self.inner.finish_availability()
            }
            fn commits(&self) -> CommitStream {
                self.inner.commits()
            }
            fn availability(&self) -> AvailabilityStream {
                self.inner.availability()
            }
            fn peer_committed(&self) -> PeerCommitted {
                self.inner.peer_committed()
            }
            fn peer_available(&self) -> PeerAvailable {
                self.inner.peer_available()
            }
            fn pull(
                &self,
                hashes: Vec<SequenceHash>,
                dst: Vec<kvbm_logical::blocks::MutableBlock<G2>>,
            ) -> BoxFuture<'static, Result<Vec<kvbm_logical::blocks::MutableBlock<G2>>>>
            {
                self.inner.pull(hashes, dst)
            }
            fn lifecycle(&self) -> LifecycleStream {
                self.inner.lifecycle()
            }
            fn finalize(&self, reason: Option<String>) {
                self.inner.finalize(reason)
            }
            fn close(&self, reason: Option<String>) {
                self.inner.close(reason)
            }
        }

        // ------------------------------------------------------------------
        // o5: the inflight window — observe() empties the residual and is
        // parked INSIDE the session commit. The discriminating asserts are
        // the direct has_pending/inflight_count reads taken while the
        // dispatch is parked (they fail if the bump is missing, the
        // decrement runs early, or has_pending ignores inflight). The
        // finalize-ordering observation is additionally protected by the
        // slot lock (the drain's take_session blocks on it), so it alone
        // would not catch a broken inflight discipline.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_output_inflight_dispatch_defers_finalize() -> Result<()> {
            let (holder_f, puller_f) = MockSessionFactory::make_paired();
            let gate = CommitGate::new();
            let gated_f = Arc::new(GatedCommitFactory {
                inner: puller_f.clone(),
                gate: gate.clone(),
            });
            let engine = prefill_engine_with_cfg(
                RecordingSink::new(),
                RecordingWorkers::new(),
                gated_f,
                RecordingResolver::new(),
                prefill_cfg_with_drain(
                    std::time::Duration::from_millis(5),
                    std::time::Duration::from_secs(5),
                ),
            )
            .await?;
            let cdr = engine.cd.as_ref().unwrap();
            let rid = "rq".to_string();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);

            let session_id = Uuid::new_v4();
            holder_f.open(session_id).expect("holder open");
            let holder = holder_f.last_opened().expect("holder recorded");
            let initiator: InstanceId = Uuid::new_v4().into();

            let handle =
                accept_direct(&engine, &plhs, session_id, initiator, holder.endpoint(), BS)?;
            let aid = handle.accept_id();
            wait_for(|| puller_f.attach_calls().len() == 1).await;
            holder.commit(plhs[..1].to_vec())?;
            holder.make_available(window[..1].to_vec())?;
            let state = cdr.prefill.get(&rid).expect("latched");
            wait_for(|| state.pulls_complete()).await;
            let puller = puller_f.last_attached().expect("attached");

            // Observe on a dedicated OS thread: it parks inside commit().
            let out = blocks_for_hashes(engine.leader.g2_manager(), &plhs[1..3]);
            let observer = Arc::clone(&cdr.output);
            let dispatcher = std::thread::spawn(move || observer.observe(&out));
            wait_for(|| gate.entered()).await;

            // The residual is spent but the dispatch is in flight: pending.
            assert!(
                cdr.output.has_pending(&rid, aid),
                "the in-flight dispatch must keep has_pending true"
            );
            assert_eq!(cdr.output.inflight_count(&rid, aid), 1);

            // Release while the dispatch is parked: no finalize yet.
            drop(handle);
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            assert!(
                puller.finished_reason().is_none(),
                "finalize must wait for the in-flight dispatch"
            );

            // Open the gate: the dispatch lands, THEN the drain finalizes.
            gate.open();
            dispatcher.join().expect("observe thread");
            wait_for(|| puller.finished_reason().is_some()).await;
            assert_eq!(
                puller.commit_calls(),
                vec![plhs[1..3].to_vec()],
                "the parked commit landed before the finalize"
            );
            assert_eq!(cdr.output.inflight_count(&rid, aid), 0, "inflight balanced");
            Ok(())
        }

        // ------------------------------------------------------------------
        // o6: watchdog — a residual that never drains force-finalizes after
        // the (shrunk) watchdog; the entry is untracked, late output is
        // dropped silently, and nothing hangs.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_output_watchdog_force_finalizes() -> Result<()> {
            let rig = prefill_rig_with_cfg(prefill_cfg_with_drain(
                std::time::Duration::from_millis(5),
                std::time::Duration::from_millis(200),
            ))
            .await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();
            let rid = "rq".to_string();

            let (handle, _) = accept_rq(&rig, session_id, initiator, holder.endpoint(), BS, 0)?;
            let aid = handle.accept_id();
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            holder.commit(rig.plhs[..1].to_vec())?;
            holder.make_available(rig.window[..1].to_vec())?;
            let state = cdr.prefill.get(&rid).expect("latched");
            wait_for(|| state.pulls_complete()).await;
            let puller = rig.puller_f.last_attached().expect("attached");

            // Release with the residual never draining: the watchdog forces
            // the finalize (bounded, no hang) and untracks the entry.
            drop(handle);
            assert!(cdr.prefill.get(&rid).is_none(), "state removed at release");
            wait_for(|| puller.finished_reason().is_some()).await;
            assert!(
                !cdr.output.is_tracked(&rid, aid),
                "untracked after the watchdog"
            );

            // Late output after the forced finalize: dropped silently.
            let late = blocks_for_hashes(rig.engine.leader.g2_manager(), &rig.plhs[1..3]);
            cdr.output.observe(&late);
            assert!(
                puller.commit_calls().is_empty(),
                "late output must not publish after the finalize"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // o7: run_remote_onboard superset tolerance (the decode side of the
        // round-trip). The prefill publishes its whole computed suffix — a
        // SUPERSET of the decode's expected slice (it cannot know the decode
        // unified bound): a commit batch whose COUNT covers the expectation
        // while an expected hash is still missing, and an availability batch
        // carrying an extra hash. The pre-relaxation strict drain bailed on
        // the extra availability hash (and broke the commit barrier early on
        // the superset count); the relaxed drain completes.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_superset_commit_and_availability_completes() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            let (session, handle) = latch_and_prepare_onboard(&engine, &factory, &window, 1);

            let onboard = engine
                .clone()
                .local_onboard(handle, &[50usize, 51, 52])
                .unwrap();

            // Superset commit batch: one expected hash + the recompute-tail
            // extra. A count-based barrier would treat this as "all expected
            // committed" while plhs[2] is still missing.
            session.inject_peer_commit(vec![plhs[1], h(7777)]);
            // The second expected hash arrives in a later delta.
            session.inject_peer_commit(vec![plhs[2]]);
            // Availability superset: the extra rides along; pre-relaxation
            // this bailed the whole pipeline.
            session.inject_peer_available(vec![
                CommittedBlock {
                    hash: h(7777),
                    peer_block_id: 899,
                },
                CommittedBlock {
                    hash: plhs[1],
                    peer_block_id: 900,
                },
                CommittedBlock {
                    hash: plhs[2],
                    peer_block_id: 901,
                },
            ]);
            wait_for(|| !session.pull_calls().is_empty()).await;
            session.resolve_pull(0, Ok(()));

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(
                onboard.outcome(),
                Some(LoadOutcome::Done),
                "a superset publish completes the decode drain"
            );
            let pulls = session.pull_calls();
            assert_eq!(
                pulls[0].0,
                vec![plhs[1], plhs[2]],
                "only the expected slice is pulled — the extra hash is skipped"
            );
            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(cdr.budget.available(), 256, "budget released at terminal");
            assert!(session.finished_reason().is_some(), "completed → finalized");
            Ok(())
        }

        /// Routes each decode-side dispatch INTO the prefill engine's accept
        /// core — the round-trip stand-in for the hub queue + vLLM request hop.
        /// Carries the full PLH chain out-of-band (the wire form carries token
        /// ids, which never cross the engine seam) and custodies the minted
        /// lease for the test's USAA step.
        struct RoutingPrefillPlane {
            prefill: Arc<LocalConnectorEngine>,
            chain: Vec<SequenceHash>,
            initiator: InstanceId,
            lease: StdMutex<Option<PrefillLease>>,
        }
        impl RoutingPrefillPlane {
            fn new(
                prefill: Arc<LocalConnectorEngine>,
                chain: Vec<SequenceHash>,
                initiator: InstanceId,
            ) -> Arc<Self> {
                Arc::new(Self {
                    prefill,
                    chain,
                    initiator,
                    lease: StdMutex::new(None),
                })
            }
            fn has_lease(&self) -> bool {
                self.lease.lock().unwrap().is_some()
            }
            fn take_lease(&self) -> PrefillLease {
                self.lease.lock().unwrap().take().expect("lease custodied")
            }
        }
        impl PrefillPlane for RoutingPrefillPlane {
            fn dispatch(&self, req: PrefillDispatch) -> BoxFuture<'static, anyhow::Result<()>> {
                let mut params = RemotePrefillParams::new(req.session_id, self.initiator);
                params.decode_endpoint = req.decode_endpoint;
                params.num_provided_tokens = req.num_provided_tokens;
                let result = match self.prefill.clone().prefill_accept_core(
                    &req.request_id,
                    &params,
                    &self.chain,
                    0,
                ) {
                    Ok(PrefillAcceptCore::Accepted { accept_id, .. }) => {
                        *self.lease.lock().unwrap() = Some(PrefillLease {
                            engine: self.prefill.clone(),
                            request_id: req.request_id.clone(),
                            accept_id,
                        });
                        Ok(())
                    }
                    Ok(PrefillAcceptCore::Refreshed { .. }) => Ok(()),
                    Err(e) => Err(anyhow::anyhow!("prefill accept failed: {e}")),
                };
                async move { result }.boxed()
            }
        }

        // ------------------------------------------------------------------
        // o8: the full end-to-end two-engine round-trip. Decode commits a
        // Remote search over a 3-block window (local hit 1) and dispatches;
        // the routing plane lands the request on the prefill engine, whose
        // chain has a 4th (recompute-tail) block the decode excluded. The
        // prefill pulls the provided window, kicks the external suffix at
        // USAA, and its computed output — a SUPERSET of the decode slice —
        // arrives via the register observer and drains the decode onboard.
        // Decode finalizes at its load terminal; the prefill release drains
        // and finalizes; both sides rendezvous Detached.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn cd_e2e_round_trip_decode_prefill_output() -> Result<()> {
            use futures::StreamExt;

            // Shared 4-block chain; the decode window is the first 3 blocks.
            let chain_mgr = cd_g2_manager(8);
            let chain = cd_immutables(&chain_mgr, 4, 100);
            let all_hashes = cd_block_hashes(&chain);
            let decode_window: Vec<ImmutableBlock<G2>> = chain[..3].to_vec();
            let rid = "rq".to_string();

            let (holder_f, puller_f) = MockSessionFactory::make_paired();

            // PREFILL engine over the puller half.
            let prefill_sink = RecordingSink::new();
            let prefill_workers = RecordingWorkers::new();
            let prefill_engine = prefill_engine_with(
                prefill_sink.clone(),
                prefill_workers.clone(),
                puller_f.clone(),
                RecordingResolver::new(),
            )
            .await?;
            let prefill_cdr = prefill_engine.cd.as_ref().unwrap();

            // DECODE engine over the holder half, dispatching through the
            // routing plane into the prefill engine.
            let initiator: InstanceId = Uuid::new_v4().into();
            let plane =
                RoutingPrefillPlane::new(prefill_engine.clone(), all_hashes.clone(), initiator);
            let decode_sink = RecordingSink::new();
            let decode_workers = RecordingWorkers::new();
            let decode_engine = cd_onboard_engine_with_sink(
                256,
                holder_f.clone(),
                plane.clone(),
                decode_workers.clone(),
                decode_sink.clone(),
            )
            .await?;
            let decode_cdr = decode_engine.cd.as_ref().unwrap();

            // 1. Decode search-time Remote commit (local hit 1 of 3) + dispatch.
            let (holder_session, search_handle) =
                latch_and_prepare_onboard(&decode_engine, &holder_f, &decode_window, 1);

            // 2. The plane routed the dispatch into the prefill accept.
            wait_for(|| plane.has_lease()).await;
            let prefill_handle = plane.take_lease();
            let aid = prefill_handle.accept_id();

            // 3. The inbound pull pipeline drains the decode-committed window
            //    (paired sessions auto-resolve the pull).
            let prefill_state = prefill_cdr.prefill.get(&rid).expect("prefill latched");
            wait_for(|| prefill_state.pulls_complete()).await;

            // 4. USAA on the prefill: the external suffix (1 block) copies
            //    G2→G1 and the load terminal reaches the prefill sink.
            let prefill_onboard = prefill_engine
                .clone()
                .prefill_onboard_by_id(&"rq".to_string(), prefill_handle.accept_id(), &[70usize])
                .unwrap();
            wait_for(|| prefill_onboard.is_complete()).await;
            assert_eq!(prefill_onboard.outcome(), Some(LoadOutcome::Done));
            assert_eq!(
                prefill_workers.transfer_calls(),
                vec![vec![70]],
                "prefill kick targets the external G1 suffix"
            );

            // 5. Decode onboards its unified hit: local span + remote slice.
            let decode_onboard = decode_engine
                .clone()
                .local_onboard(search_handle, &[50usize, 51, 52])
                .unwrap();

            // 6. The prefill computes the rest: its output blocks (the full
            //    suffix INCLUDING the recompute-tail block the decode
            //    excluded) register in the prefill G2 and the register
            //    observer publishes them into the parked session.
            let out_blocks =
                blocks_for_hashes(prefill_engine.leader.g2_manager(), &all_hashes[1..]);
            prefill_cdr.output.observe(&out_blocks);

            // 7. The decode drain completes despite the superset publish.
            wait_for(|| decode_onboard.is_complete()).await;
            assert_eq!(decode_onboard.outcome(), Some(LoadOutcome::Done));
            assert_eq!(
                decode_workers.transfer_calls(),
                vec![vec![50], vec![51, 52]],
                "decode local kick + remote pull onboard"
            );

            // 8. The decode load terminal finalized ITS side (it does not
            //    wait for the prefill's Finished) and released the budget.
            wait_for(|| holder_session.finished_reason().is_some()).await;
            wait_for(|| decode_cdr.requests.is_empty()).await;
            assert_eq!(decode_cdr.budget.available(), 256, "no leaked budget");
            wait_for(|| !decode_sink.loads().is_empty()).await;
            assert_eq!(decode_sink.loads(), vec![(rid.clone(), LoadOutcome::Done)]);
            assert_eq!(prefill_sink.loads(), vec![(rid.clone(), LoadOutcome::Done)]);

            // 9. Prefill release: the output drained, so the deferred
            //    finalize fires and both sides rendezvous Detached.
            drop(prefill_onboard);
            drop(prefill_handle);
            let puller_session = puller_f.last_attached().expect("attached");
            wait_for(|| puller_session.finished_reason().is_some()).await;
            assert!(
                prefill_cdr.prefill.get(&rid).is_none(),
                "prefill state freed"
            );
            assert!(!prefill_cdr.output.has_pending(&rid, aid));
            assert!(
                !prefill_cdr.output.is_tracked(&rid, aid),
                "observer untracked"
            );
            assert!(
                holder_session.closed_reason().is_none()
                    && puller_session.closed_reason().is_none(),
                "cooperative shutdown — no aborts anywhere"
            );

            // Both lifecycle streams carry the rendezvous Detached terminal.
            let mut holder_life = holder_session.lifecycle();
            assert!(matches!(
                holder_life.next().await,
                Some(LifecycleEvent::Attached { .. })
            ));
            assert!(matches!(
                holder_life.next().await,
                Some(LifecycleEvent::Detached { .. })
            ));
            let mut puller_life = puller_session.lifecycle();
            assert!(matches!(
                puller_life.next().await,
                Some(LifecycleEvent::Detached { .. })
            ));

            decode_engine.release_search(&search_handle);
            Ok(())
        }

        // ------------------------------------------------------------------
        // The o8 round-trip with a vLLM-computed prefix on decode. The
        // prefix is G2-resident on decode up front (the natural-backfill
        // steady state — the offload save cursor walks from 0), so the
        // computed>0 request goes Remote: the prefill pulls the FULL
        // [0, DNPT) absolute window (prefix + local match), kicks its
        // external suffix, and decode pulls back exactly its window-relative
        // remote slice — one load terminal Complete on each side.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn cd_e2e_round_trip_with_computed_prefix() -> Result<()> {
            // Shared 5-block chain: c0 is decode's computed prefix, the
            // decode window is c1..c3, c4 is the recompute tail it excludes.
            let chain_mgr = cd_g2_manager(8);
            let chain = cd_immutables(&chain_mgr, 5, 100);
            let all_hashes = cd_block_hashes(&chain);
            let window: Vec<ImmutableBlock<G2>> = chain[1..4].to_vec();
            let rid = "rq".to_string();

            let (holder_f, puller_f) = MockSessionFactory::make_paired();

            // PREFILL engine over the puller half.
            let prefill_sink = RecordingSink::new();
            let prefill_workers = RecordingWorkers::new();
            let prefill_engine = prefill_engine_with(
                prefill_sink.clone(),
                prefill_workers.clone(),
                puller_f.clone(),
                RecordingResolver::new(),
            )
            .await?;
            let prefill_cdr = prefill_engine.cd.as_ref().unwrap();

            // DECODE engine over the holder half.
            let initiator: InstanceId = Uuid::new_v4().into();
            let plane =
                RoutingPrefillPlane::new(prefill_engine.clone(), all_hashes.clone(), initiator);
            let decode_sink = RecordingSink::new();
            let decode_workers = RecordingWorkers::new();
            let decode_engine = cd_onboard_engine_with_sink(
                256,
                holder_f.clone(),
                plane.clone(),
                decode_workers.clone(),
                decode_sink.clone(),
            )
            .await?;
            let decode_cdr = decode_engine.cd.as_ref().unwrap();

            // Natural-backfill precondition: the computed prefix already
            // landed in the DECODE leader's own G2 (a prior request's
            // offload). Same token chain ⇒ same hash.
            let prefix = cd_immutables(decode_engine.leader.g2_manager(), 1, 100);
            assert_eq!(cd_block_hashes(&prefix), all_hashes[..1].to_vec());

            // 1. Decode Remote commit with computed=1, local hit 1 of 3.
            let (holder_session, search_handle) = latch_and_prepare_onboard_with_prefix(
                &decode_engine,
                &holder_f,
                &prefix,
                &window,
                1,
            );

            // 2. The dispatch landed on the prefill accept with DNPT = 2*BS:
            //    the FULL [0, 2) absolute window — decode's computed prefix
            //    is inside the served set, never a hole.
            wait_for(|| plane.has_lease()).await;
            let prefill_handle = plane.take_lease();
            let prefill_state = prefill_cdr.prefill.get(&rid).expect("prefill latched");
            assert_eq!(
                prefill_state.expected_hashes(),
                &all_hashes[..2],
                "prefill pulls the whole [0, DNPT) window incl. the prefix"
            );
            wait_for(|| prefill_state.pulls_complete()).await;

            // 3. USAA on the prefill: both pulled blocks are external there.
            let prefill_onboard = prefill_engine
                .clone()
                .prefill_onboard_by_id(&rid, prefill_handle.accept_id(), &[70usize, 71])
                .unwrap();
            wait_for(|| prefill_onboard.is_complete()).await;
            assert_eq!(prefill_onboard.outcome(), Some(LoadOutcome::Done));
            assert_eq!(prefill_workers.transfer_calls(), vec![vec![70, 71]]);

            // 4. Decode onboards behind its computed dest block.
            let decode_onboard = decode_engine
                .clone()
                .local_onboard(search_handle, &[40usize, 50, 51, 52])
                .unwrap();

            // 5. The prefill's computed output (c2..c4 — a superset of the
            //    decode slice) publishes via the register observer.
            let out_blocks =
                blocks_for_hashes(prefill_engine.leader.g2_manager(), &all_hashes[2..]);
            prefill_cdr.output.observe(&out_blocks);

            // 6. Decode pulls back exactly its window-relative remote slice.
            wait_for(|| decode_onboard.is_complete()).await;
            assert_eq!(decode_onboard.outcome(), Some(LoadOutcome::Done));
            assert_eq!(
                decode_workers.transfer_calls(),
                vec![vec![50], vec![51, 52]],
                "local kick + remote pull never touch the computed dest"
            );

            // 7. One load terminal Complete per side; decode budget restored.
            wait_for(|| holder_session.finished_reason().is_some()).await;
            wait_for(|| decode_cdr.requests.is_empty()).await;
            assert_eq!(decode_cdr.budget.available(), 256, "no leaked budget");
            wait_for(|| !decode_sink.loads().is_empty()).await;
            assert_eq!(decode_sink.loads(), vec![(rid.clone(), LoadOutcome::Done)]);
            assert_eq!(prefill_sink.loads(), vec![(rid.clone(), LoadOutcome::Done)]);

            // 8. Cooperative teardown on the prefill release.
            drop(prefill_onboard);
            drop(prefill_handle);
            let puller_session = puller_f.last_attached().expect("attached");
            wait_for(|| puller_session.finished_reason().is_some()).await;
            decode_engine.release_search(&search_handle);
            Ok(())
        }

        // ------------------------------------------------------------------
        // o9: an expected hash duplicated WITHIN one availability delta must
        // not break the decode drain. The prefill side can publish a hash
        // twice (its accept-time already-in-G2 sweep racing the register
        // observer), and the wire replay coalesces every pre-subscribe delta
        // into one — so both copies land in a single delta. Pre-dedup the
        // equal slots violated `group_contiguous_runs`' strictly-increasing
        // contract (debug panic → wedged load) and split a second pull of a
        // hash whose holder pin the first PullAck already dropped (release).
        // The dedup'd drain pulls each expected hash exactly once.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_usaa_duplicate_hash_within_delta_pulls_once() -> Result<()> {
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let workers = RecordingWorkers::new();
            let engine =
                cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
            let cdr = engine.cd.as_ref().unwrap();

            let mgr = cd_g2_manager(8);
            let window = cd_immutables(&mgr, 3, 100);
            let plhs = cd_block_hashes(&window);
            let (session, handle) = latch_and_prepare_onboard(&engine, &factory, &window, 1);

            let onboard = engine
                .clone()
                .local_onboard(handle, &[50usize, 51, 52])
                .unwrap();

            session.inject_peer_commit(vec![plhs[1], plhs[2]]);
            // ONE delta carrying plhs[1] twice.
            session.inject_peer_available(vec![
                CommittedBlock {
                    hash: plhs[1],
                    peer_block_id: 900,
                },
                CommittedBlock {
                    hash: plhs[1],
                    peer_block_id: 900,
                },
                CommittedBlock {
                    hash: plhs[2],
                    peer_block_id: 901,
                },
            ]);
            wait_for(|| !session.pull_calls().is_empty()).await;
            session.resolve_pull(0, Ok(()));

            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            let pulls = session.pull_calls();
            assert_eq!(pulls.len(), 1, "the duplicate must not split a second pull");
            assert_eq!(
                pulls[0].0,
                vec![plhs[1], plhs[2]],
                "each expected hash pulled exactly once"
            );
            wait_for(|| cdr.requests.is_empty()).await;
            assert_eq!(cdr.budget.available(), 256, "budget released at terminal");
            assert!(session.finished_reason().is_some(), "completed → finalized");
            Ok(())
        }

        // ==================================================================
        // find_blocks / onboard_blocks router (the unified seam face)
        // ==================================================================

        fn fb_req(
            request_id: &str,
            sequence_hashes: Vec<SequenceHash>,
            num_computed_tokens: usize,
            total_tokens: usize,
        ) -> FindBlocksRequest {
            FindBlocksRequest {
                request_id: request_id.to_string(),
                sequence_hashes: Arc::from(sequence_hashes),
                num_computed_tokens,
                total_tokens,
                transfer_params: None,
            }
        }

        fn fb_prefill_req(
            plhs: &[SequenceHash],
            session_id: Uuid,
            initiator: InstanceId,
            endpoint: Option<SessionEndpoint>,
            provided_tokens: usize,
            computed_tokens: usize,
            total_tokens: usize,
        ) -> FindBlocksRequest {
            let mut params = RemotePrefillParams::new(session_id, initiator);
            params.decode_endpoint = endpoint;
            params.num_provided_tokens = provided_tokens;
            FindBlocksRequest {
                request_id: "rq".to_string(),
                sequence_hashes: Arc::from(plhs.to_vec()),
                num_computed_tokens: computed_tokens,
                total_tokens,
                transfer_params: Some(kvbm_protocols::disagg::TransferParams::remote_prefill(
                    params,
                )),
            }
        }

        fn expect_resolved(out: FindBlocksOutcome) -> (usize, Option<FindBlocksHandle>, bool) {
            match out {
                FindBlocksOutcome::Resolved {
                    matched_tokens,
                    minted,
                    release_parked,
                } => (matched_tokens, minted, release_parked),
                other => panic!("expected Resolved, got {other:?}"),
            }
        }

        // ------------------------------------------------------------------
        // Full router path with a RAGGED computed prefix: derive_window floors
        // the offset to whole blocks, the prefix capture matches the leader's
        // own G2, and every count the commit publishes is the QUANTIZED
        // offset — never the raw token count (digest/DNPT divergence hazard).
        // matched_tokens stays relative-to-computed.
        // ------------------------------------------------------------------
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn cd_find_blocks_ragged_computed_prefix_commits_remote() -> Result<()> {
            // The leader's own G2 holds the whole 4-block chain — prefix
            // residency and the window match both resolve against it.
            let (leader, plhs, _held) = leader_with_resident_blocks(4, 100).await?;
            let factory = MockSessionFactory::new();
            let plane = RecordingPrefillPlane::ok();
            let cd = cd_runtime(
                SelectionPolicy::Always,
                256,
                true,
                factory.clone(),
                plane.clone(),
            );
            let engine = cd_engine(Arc::new(leader), cd);
            let cdr = engine.cd.as_ref().unwrap();

            // Ragged computed (BS + 2 tokens) floors to ONE computed block;
            // total = 4*BS + 1 keeps all 4 chain blocks eligible → window
            // [1, 4).
            let req = fb_req("rq", plhs.clone(), BS + 2, 4 * BS + 1);
            let out = Arc::clone(&engine).find_blocks(&req, None)?;
            let (matched, minted, _) = expect_resolved(out);
            assert_eq!(
                matched,
                3 * BS,
                "matched_tokens is relative-to-computed (the window only)"
            );
            let _handle = minted.expect("search handle minted");

            wait_for(|| plane.count() >= 1).await;
            assert_eq!(
                plane.last_num_provided(),
                Some(4 * BS),
                "BS (quantized — never the raw BS+2) + 3 local blocks"
            );
            assert_eq!(plane.last_num_window(), Some(4 * BS), "computed + fbet");

            let session = factory.last_opened().expect("session opened");
            assert_eq!(
                session.commit_calls(),
                vec![plhs.clone()],
                "commit set = prefix [0,1) ++ window match [1,4) — the whole \
                 absolute chain"
            );
            let state = cdr.requests.get("rq").expect("latched");
            assert_eq!(state.base_offset(), BS, "quantized offset");
            assert_eq!(
                cdr.budget.available(),
                256 - 3 * BS,
                "fbet = the block-floored suffix"
            );
            Ok(())
        }

        // ------------------------------------------------------------------
        // Suffix-only G2 registration (the pulled-run shape): a window run
        // staged by hash + registered while its [0, computed) parents are
        // ABSENT from G2. Registration is flat (no parent-presence
        // requirement) and a window-relative match resolves the run, while a
        // from-zero linear match stops on the missing prefix — exactly the
        // two shapes the prefix-residency gate distinguishes.
        // ------------------------------------------------------------------
        #[test]
        fn suffix_only_g2_registration_matches_window_relative() {
            use crate::testing::token_blocks::generate_sequence_hashes;
            let mgr = cd_g2_manager(8);
            let seq = create_token_sequence(4, BS, 4242);
            let chain = generate_sequence_hashes(&seq);
            // Stage + register ONLY the suffix run [2, 4) — the
            // `pull_run_into_g2` stage(hash, block_size) shape.
            let mutables = mgr.allocate_blocks(2).expect("alloc");
            let completes: Vec<_> = mutables
                .into_iter()
                .zip(chain[2..].iter())
                .map(|(b, hash)| b.stage(*hash, BS).expect("stage"))
                .collect();
            let registered = mgr.register_blocks(completes);
            assert_eq!(registered.len(), 2, "registration needs no parents");
            assert_eq!(
                mgr.match_blocks(&chain[2..]).len(),
                2,
                "the window-relative match resolves the suffix run"
            );
            assert_eq!(
                mgr.match_blocks(&chain).len(),
                0,
                "a from-zero linear match stops on the absent prefix"
            );
        }

        /// A plain local-tiering engine (no CD) over the given leader.
        fn plain_engine(leader: InstanceLeader) -> Arc<LocalConnectorEngine> {
            LocalConnectorEngine::new(Arc::new(leader), NoopWorkerSink::new(), BS, false)
        }

        /// CD prefill engine whose LEADER'S OWN G2 holds `blocks` resident
        /// registered blocks — the zero-external fall-through's local search
        /// resolves against them (`PrefillRig`'s window lives in a separate
        /// manager, invisible to its leader). Recording workers capture the
        /// delegation onboard's G2→G1 transfers.
        struct FallthroughRig {
            workers: Arc<RecordingWorkers>,
            engine: Arc<LocalConnectorEngine>,
            plhs: Vec<SequenceHash>,
            _held: Vec<ImmutableBlock<G2>>,
        }

        async fn fallthrough_rig(blocks: usize) -> Result<FallthroughRig> {
            use crate::testing::messenger::create_messenger_tcp;
            use kvbm_logical::blocks::BlockRegistry;

            let (_holder_f, puller_f) = MockSessionFactory::make_paired();
            let resolver = RecordingResolver::new();
            let workers = RecordingWorkers::new();
            let messenger = create_messenger_tcp().await?;
            let registry = BlockRegistry::builder().build();
            let g2 = Arc::new(
                TestManagerBuilder::<G2>::new()
                    .block_count(blocks + 4)
                    .block_size(BS)
                    .registry(registry.clone())
                    .build(),
            );
            let held = cd_immutables(&g2, blocks, 700);
            let plhs = cd_block_hashes(&held);
            let leader = InstanceLeader::builder()
                .messenger(messenger)
                .registry(registry)
                .g2_manager(g2)
                .parallel_worker(workers.clone() as Arc<dyn ParallelWorkers>)
                .build()?;
            let cd = CdRuntime::new(
                prefill_cfg(),
                Arc::new(TierCell::default()),
                puller_f,
                RecordingPrefillPlane::ok(),
                Some(resolver),
            );
            let engine = LocalConnectorEngine::with_offload_submit(
                Arc::new(leader),
                NoopWorkerSink::new(),
                BS,
                false,
                Arc::new(DisabledOffloadSubmit),
                Some(cd),
            );
            Ok(FallthroughRig {
                workers,
                engine,
                plhs,
                _held: held,
            })
        }

        /// Fresh local hit: token-granular `hit_blocks × block_size`, a minted
        /// Search-kind handle, no release instruction.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_fresh_local_hit_resolves_tokens_and_mints_search_kind() -> Result<()> {
            let (leader, plhs, _held) = leader_with_resident_blocks(3, 400).await?;
            let engine = plain_engine(leader);

            // Ragged total: all 3 complete blocks stay eligible.
            let req = fb_req("rq", plhs, 0, 3 * BS + 1);
            let (matched, minted, release_parked) =
                expect_resolved(engine.clone().find_blocks(&req, None)?);
            assert_eq!(matched, 3 * BS, "token-granular: hit_blocks × block_size");
            let handle = minted.expect("a fresh hit mints the one handle");
            assert!(handle.search_id().is_some(), "local hit → Search-kind");
            assert_eq!(handle.request_id(), "rq");
            assert!(!release_parked);
            assert_eq!(engine.searches.len(), 1, "one latch parked engine-side");
            Ok(())
        }

        /// Engine-side window derivation applies vLLM's last-must-recompute
        /// exclusion: a block-aligned total leaves only the first two of three
        /// resident blocks eligible (`total / BS` instead of `(total-1) / BS`
        /// would match all three).
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_block_aligned_window_excludes_final_block() -> Result<()> {
            let (leader, plhs, _held) = leader_with_resident_blocks(3, 410).await?;
            let engine = plain_engine(leader);

            let req = fb_req("rq", plhs, 0, 3 * BS);
            let (matched, minted, _) = expect_resolved(engine.clone().find_blocks(&req, None)?);
            assert_eq!(matched, 2 * BS, "the final full block is excluded");
            assert!(minted.is_some());
            Ok(())
        }

        /// The computed prefix offsets the engine-derived window.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_computed_prefix_offsets_window() -> Result<()> {
            let (leader, plhs, _held) = leader_with_resident_blocks(3, 420).await?;
            let engine = plain_engine(leader);

            let req = fb_req("rq", plhs, BS, 3 * BS + 1);
            let (matched, minted, _) = expect_resolved(engine.clone().find_blocks(&req, None)?);
            assert_eq!(matched, 2 * BS, "window starts past the computed block");
            assert!(minted.is_some());
            Ok(())
        }

        /// No match resolves zero with nothing minted and nothing latched.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_no_match_resolves_zero_without_mint() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, false);

            let req = fb_req("rq", vec![h(900), h(901)], 0, 2 * BS + 1);
            let (matched, minted, release_parked) =
                expect_resolved(engine.clone().find_blocks(&req, None)?);
            assert_eq!(matched, 0);
            assert!(minted.is_none());
            assert!(!release_parked);
            assert!(engine.searches.is_empty());
            Ok(())
        }

        /// An empty window short-circuits to a synchronous zero carrying the
        /// release instruction, WITHOUT touching a live latch (the connector
        /// owns the Issue-A-gated drop).
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_empty_window_short_circuits_without_touching_live_search() -> Result<()>
        {
            let (leader, plhs, _held) = leader_with_resident_blocks(3, 430).await?;
            let engine = plain_engine(leader);

            let total = 3 * BS + 1;
            let first = fb_req("rq", plhs.clone(), 0, total);
            let (_, minted, _) = expect_resolved(engine.clone().find_blocks(&first, None)?);
            let live = minted.expect("hit minted");

            // The computed prefix now covers everything eligible.
            let repoll = fb_req("rq", plhs, 3 * BS, total);
            let (matched, minted, release_parked) =
                expect_resolved(engine.clone().find_blocks(&repoll, Some(&live))?);
            assert_eq!(matched, 0);
            assert!(minted.is_none());
            assert!(release_parked, "the parked pin has no further use");
            assert_eq!(
                engine.searches.len(),
                1,
                "the live latch was not refreshed or released engine-side"
            );
            Ok(())
        }

        /// A re-poll with a live handle refreshes in place: same answer, no
        /// second mint, no release instruction.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_refresh_resolves_in_place_never_mints() -> Result<()> {
            let (leader, plhs, _held) = leader_with_resident_blocks(3, 440).await?;
            let engine = plain_engine(leader);

            let req = fb_req("rq", plhs, 0, 3 * BS + 1);
            let (_, minted, _) = expect_resolved(engine.clone().find_blocks(&req, None)?);
            let live = minted.expect("hit minted");

            let (matched, minted, release_parked) =
                expect_resolved(engine.clone().find_blocks(&req, Some(&live))?);
            assert_eq!(matched, 3 * BS);
            assert!(minted.is_none(), "a refresh never mints a second handle");
            assert!(!release_parked);
            assert_eq!(engine.searches.len(), 1);
            Ok(())
        }

        /// A lost latch (already released or onboarded) resolves zero with the
        /// release instruction.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_refresh_lost_resolves_zero_release_parked() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, false);
            let dyn_engine: Arc<dyn LeaderEngine> = engine.clone();

            // A handle whose latch no longer exists engine-side.
            let live = FindBlocksHandle::search(
                "rq".to_string(),
                SearchId::new(),
                Arc::downgrade(&dyn_engine),
            );
            let req = fb_req("rq", vec![h(0)], 0, BS + 1);
            let (matched, minted, release_parked) =
                expect_resolved(engine.clone().find_blocks(&req, Some(&live))?);
            assert_eq!(matched, 0);
            assert!(minted.is_none());
            assert!(release_parked, "Lost → drop the dead pin");
            Ok(())
        }

        /// A still-pending refresh maps to `Searching`, never to a stale
        /// `Resolved` hit.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_refresh_pending_maps_to_searching() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, false);

            let (find_session, _tx) = pending_async(); // held: stays Pending
            let search_id = SearchId::new();
            let cell = Arc::new(Mutex::new(MatchStatus::Pending));
            engine.searches.insert(
                search_id,
                SearchState {
                    request_id: "rq".into(),
                    status: cell,
                    onboarding: OnboardingState::new(
                        0,
                        BS + 1,
                        OnboardingShard {
                            start_block: 0,
                            num_queried_blocks: 1,
                            find_session,
                        },
                    ),
                    buffer: vec![h(7)],
                },
            );
            let dyn_engine: Arc<dyn LeaderEngine> = engine.clone();
            let live =
                FindBlocksHandle::search("rq".to_string(), search_id, Arc::downgrade(&dyn_engine));

            let req = fb_req("rq", vec![h(7)], 0, BS + 1);
            match engine.clone().find_blocks(&req, Some(&live))? {
                FindBlocksOutcome::Searching { minted } => {
                    assert!(minted.is_none(), "a refresh never mints")
                }
                other => panic!("pending refresh must map to Searching, got {other:?}"),
            }
            Ok(())
        }

        /// Kind-mismatch desync: a Prefill-kind live handle on a request with
        /// no decode params fails loud.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_live_prefill_kind_without_params_desyncs() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, false);
            let dyn_engine: Arc<dyn LeaderEngine> = engine.clone();

            let live = FindBlocksHandle::prefill(
                "rq".to_string(),
                AcceptId::new(),
                Arc::downgrade(&dyn_engine),
            );
            let req = fb_req("rq", vec![h(0)], 0, BS + 1);
            assert!(matches!(
                engine.clone().find_blocks(&req, Some(&live)),
                Err(LeaderEngineError::FindBlocksDesync)
            ));
            Ok(())
        }

        /// The fresh-mint deferral guard: a window overlapping an in-flight
        /// onboard defers with no engine side effect; a disjoint window
        /// proceeds.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_fresh_overlap_defers_disjoint_proceeds() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, false);

            engine
                .inflight
                .lock()
                .unwrap()
                .record(InflightKey::Search(SearchId::new()), vec![h(0)]);

            let overlapping = fb_req("a", vec![h(0)], 0, BS + 1);
            assert!(matches!(
                engine.clone().find_blocks(&overlapping, None)?,
                FindBlocksOutcome::Deferred
            ));
            assert!(
                engine.searches.is_empty(),
                "a deferred poll mints and latches nothing"
            );

            let disjoint = fb_req("b", vec![h(9)], 0, BS + 1);
            assert!(
                !matches!(
                    engine.clone().find_blocks(&disjoint, None)?,
                    FindBlocksOutcome::Deferred
                ),
                "a disjoint window must not defer"
            );
            Ok(())
        }

        /// Fresh-mint cross-request deferral that RESOLVES on release: request
        /// A's real `local_onboard` records its matched window under A's
        /// generation; request B's FRESH poll over an overlapping window defers
        /// (minting nothing). Once A's `release_search` clears the guard, B's
        /// fresh poll MINTS a new Search-kind handle and positively matches the
        /// formerly-in-flight span — the fresh-arm twin of the refresh-arm
        /// reproducer, distinguished by the positive `minted: Some` resolution.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_fresh_mint_defers_until_other_onboard_releases_then_matches()
        -> Result<()> {
            // Two resident blocks in the leader's own G2: B's fresh poll resolves
            // SYNCHRONOUSLY to a local hit over their hashes.
            let (leader, plhs, _held) = leader_with_resident_blocks(2, 500).await?;
            let engine = plain_engine(leader);

            // Lifecycle A: a manually-built Matched latch whose buffer covers the
            // first resident hash. Park the find_session (no real transfer — the
            // leader has no parallel worker) so A's onboard records its window and
            // stays in flight.
            let (find_session_a, _tx_a) = pending_async();
            let sid_a = SearchId::new();
            let cell_a = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 }));
            engine.searches.insert(
                sid_a,
                SearchState {
                    request_id: "a".into(),
                    status: cell_a,
                    onboarding: OnboardingState::new(
                        0,
                        BS + 1,
                        OnboardingShard {
                            start_block: 0,
                            num_queried_blocks: 1,
                            find_session: find_session_a,
                        },
                    ),
                    buffer: vec![plhs[0]],
                },
            );
            // A's onboard records window [plhs[0]] into the in-flight guard under
            // sid_a; this also removes A from `engine.searches`.
            let _onboard_a = engine.clone().local_onboard(sid_a, &[10]).unwrap();

            // B's fresh poll over BOTH resident hashes overlaps A's recorded
            // window at plhs[0] → Deferred, minting/latching nothing.
            let req_b = fb_req("b", plhs.clone(), 0, 2 * BS + 1);
            assert!(
                matches!(
                    engine.clone().find_blocks(&req_b, None)?,
                    FindBlocksOutcome::Deferred
                ),
                "B's fresh poll defers on A's in-flight onboard"
            );
            assert!(
                engine.searches.is_empty(),
                "a deferred fresh poll mints and latches nothing"
            );

            // A's lifecycle releases (the connector drain-holder's handle drop):
            // the guard clears and B's fresh poll now MINTS and positively
            // matches the formerly-in-flight span.
            engine.release_search(&sid_a);
            let (matched, minted, release_parked) =
                expect_resolved(engine.clone().find_blocks(&req_b, None)?);
            assert_eq!(
                matched,
                2 * BS,
                "B resolves the full resident span once A released"
            );
            let handle = minted.expect("the fresh poll mints its own handle");
            assert!(
                handle.search_id().is_some(),
                "the fresh mint is Search-kind, not a refresh reconcile"
            );
            assert!(!release_parked);
            assert_eq!(engine.searches.len(), 1, "B latched its own fresh search");
            Ok(())
        }

        /// The two-request refresh-arm reproducer: A USAA-onboards (recording
        /// its window under A's generation) while B already holds a live
        /// overlapping search. B's refresh re-poll must now DEFER — the
        /// engine search keeps running, nothing is torn down — until A's
        /// lifecycle releases (the recv-side handle drop), after which B's
        /// refresh proceeds. Removing the refresh-arm guard check fails the
        /// Deferred assertion (B would race a duplicate load of h(0)).
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_refresh_overlap_defers_until_other_release() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, false);

            // Lifecycle A: its committed onboard covers h(0).
            let sid_a = SearchId::new();
            engine
                .inflight
                .lock()
                .unwrap()
                .record(InflightKey::Search(sid_a), vec![h(0)]);

            // Request B holds a live latch whose window overlaps h(0).
            let sid_b = SearchId::new();
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 }));
            engine.searches.insert(
                sid_b,
                SearchState {
                    request_id: "b".into(),
                    status: cell,
                    onboarding: OnboardingState::new(
                        0,
                        BS + 1,
                        OnboardingShard {
                            start_block: 0,
                            num_queried_blocks: 1,
                            find_session: complete_async(1),
                        },
                    ),
                    buffer: vec![h(0)],
                },
            );
            let dyn_engine: Arc<dyn LeaderEngine> = engine.clone();
            let live =
                FindBlocksHandle::search("b".to_string(), sid_b, Arc::downgrade(&dyn_engine));

            let req = fb_req("b", vec![h(0)], 0, BS + 1);
            assert!(
                matches!(
                    engine.clone().find_blocks(&req, Some(&live))?,
                    FindBlocksOutcome::Deferred
                ),
                "B's refresh defers on A's in-flight onboard"
            );
            assert!(
                engine.searches.contains_key(&sid_b),
                "the deferral leaves B's live search running untouched"
            );

            // A's lifecycle releases (recv-side handle drop) → B proceeds.
            engine.release_search(&sid_a);
            let (matched, minted, _) =
                expect_resolved(engine.clone().find_blocks(&req, Some(&live))?);
            assert_eq!(matched, BS, "B's refresh resolves once A released");
            assert!(minted.is_none());
            Ok(())
        }

        /// Refresh equality heuristic, IDENTICAL branch: a pure re-poll (same
        /// computed offset, same eligible end, first + last hash unchanged)
        /// skips the buffer merge entirely — pinned by a sentinel hash at an
        /// interior buffer position that a merge would overwrite — while
        /// shard completions still advance the projected status.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_refresh_identical_repoll_skips_merge_and_advances() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, false);

            let (find_session, tx) = pending_async();
            let sid = SearchId::new();
            let cell = Arc::new(Mutex::new(MatchStatus::Pending));
            let sentinel = h(9);
            engine.searches.insert(
                sid,
                SearchState {
                    request_id: "rq".into(),
                    status: cell,
                    onboarding: OnboardingState::new(
                        0,
                        3 * BS + 1,
                        OnboardingShard {
                            start_block: 0,
                            num_queried_blocks: 3,
                            find_session,
                        },
                    ),
                    // Interior sentinel differs from the incoming window; the
                    // endpoints match, so the spot-check reads "identical".
                    buffer: vec![h(0), sentinel, h(2)],
                },
            );
            let dyn_engine: Arc<dyn LeaderEngine> = engine.clone();
            let live = FindBlocksHandle::search("rq".to_string(), sid, Arc::downgrade(&dyn_engine));
            let req = fb_req("rq", vec![h(0), h(1), h(2)], 0, 3 * BS + 1);

            // Pending re-poll: identical → no merge ran (sentinel intact).
            assert!(matches!(
                engine.clone().find_blocks(&req, Some(&live))?,
                FindBlocksOutcome::Searching { minted: None }
            ));
            assert_eq!(
                engine.searches.get(&sid).unwrap().buffer[1],
                sentinel,
                "the pure re-poll skipped the buffer merge"
            );

            // The shard completes; the next identical re-poll advances the
            // status WITHOUT merging.
            tx.send(OnboardingStatus::Complete { matched_blocks: 3 })
                .unwrap();
            let (matched, minted, _) =
                expect_resolved(engine.clone().find_blocks(&req, Some(&live))?);
            assert_eq!(matched, 3 * BS, "shard completion advances the status");
            assert!(minted.is_none());
            assert_eq!(
                engine.searches.get(&sid).unwrap().buffer[1],
                sentinel,
                "still no merge on the resolving re-poll"
            );
            Ok(())
        }

        /// Refresh equality heuristic, MISMATCH branch: a changed window (here
        /// the eligible end grew — the Case-D shape) warns and takes the full
        /// merge/reconcile path: the buffer extends with the incoming hashes
        /// and a suffix shard is issued for the new span.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_refresh_changed_window_takes_merge_path() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, false);

            let sid = SearchId::new();
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 2 }));
            engine.searches.insert(
                sid,
                SearchState {
                    request_id: "rq".into(),
                    status: cell,
                    onboarding: OnboardingState::new(
                        0,
                        2 * BS + 1,
                        OnboardingShard {
                            start_block: 0,
                            num_queried_blocks: 2,
                            find_session: complete_async(2),
                        },
                    ),
                    buffer: vec![h(0), h(1)],
                },
            );
            let dyn_engine: Arc<dyn LeaderEngine> = engine.clone();
            let live = FindBlocksHandle::search("rq".to_string(), sid, Arc::downgrade(&dyn_engine));

            // The chain grew by one eligible block → length mismatch → merge.
            let req = fb_req("rq", vec![h(0), h(1), h(2)], 0, 3 * BS + 1);
            let (matched, minted, _) =
                expect_resolved(engine.clone().find_blocks(&req, Some(&live))?);
            assert_eq!(
                matched,
                2 * BS,
                "the matched prefix survives the reconcile (the new tail misses)"
            );
            assert!(minted.is_none());
            let entry = engine.searches.get(&sid).unwrap();
            assert_eq!(
                entry.buffer,
                vec![h(0), h(1), h(2)],
                "the merge path extended the buffer with the incoming window"
            );
            assert_eq!(
                entry.onboarding.shards.len(),
                2,
                "reconcile issued the Case-D suffix shard"
            );
            Ok(())
        }

        /// Preemption-restore self-deferral is INTENDED behavior, and it
        /// terminates at the drain-holder's RELEASE — not at evict, not at
        /// the fence-armed driver terminal: the restored request's fresh poll
        /// defers on its own prior generation's still-draining load until the
        /// connector's drain-holder drops the old handle (release_search).
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_evict_restore_self_deferral_holds_until_release() -> Result<()> {
            let leader = Arc::new(build_test_leader().await?);
            let sink = RecordingSink::new();
            let engine = LocalConnectorEngine::new(leader, sink, BS, false);

            let (find_session, _tx) = pending_async(); // park the driver pre-terminal
            let search_id = SearchId::new();
            let cell = Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 }));
            engine.searches.insert(
                search_id,
                SearchState {
                    request_id: "rq".into(),
                    status: cell.clone(),
                    onboarding: OnboardingState::new(
                        0,
                        BS + 1,
                        OnboardingShard {
                            start_block: 0,
                            num_queried_blocks: 1,
                            find_session,
                        },
                    ),
                    buffer: vec![h(5)],
                },
            );
            let req_id: RequestId = "rq".into();
            let onboard = engine.clone().local_onboard(search_id, &[10]).unwrap();

            let fence = engine.evict(&req_id).fence;
            assert!(!fence.per_worker.is_empty(), "the live onboard is fenced");

            // The restored request's fresh poll defers while the transfer is live.
            let repoll = fb_req("rq", vec![h(5)], 0, BS + 1);
            assert!(matches!(
                engine.clone().find_blocks(&repoll, None)?,
                FindBlocksOutcome::Deferred
            ));

            // The fence-armed driver terminal alone does NOT lift it — the
            // old generation's loaded blocks are not connector-visible yet.
            engine.finish_load_action(*onboard.id(), &req_id, ActionStatus::Complete, vec![10]);
            assert!(
                matches!(
                    engine.clone().find_blocks(&repoll, None)?,
                    FindBlocksOutcome::Deferred
                ),
                "the restored poll stays deferred past the terminal"
            );

            // The drain-holder's eventual handle drop releases the old
            // generation and lifts the self-deferral.
            engine.release_search(&search_id);
            assert!(
                !matches!(
                    engine.clone().find_blocks(&repoll, None)?,
                    FindBlocksOutcome::Deferred
                ),
                "the deferral terminates at the old generation's release"
            );
            Ok(())
        }

        /// Dispatched-prefill latch: the external count passes through
        /// token-granular and the one minted handle is Prefill-kind.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_prefill_latch_passes_external_through() -> Result<()> {
            let rig = prefill_rig().await?;
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let req = fb_prefill_req(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                BS,
                3 * BS + 1,
            );
            let (matched, minted, release_parked) =
                expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            assert_eq!(matched, 2 * BS, "external tokens pass through");
            let handle = minted.expect("first poll latches and mints");
            assert!(handle.prefill_accept_id().is_some(), "Prefill-kind");
            assert!(!release_parked, "an accepted prefill stays parked");

            let cdr = rig.engine.cd.as_ref().unwrap();
            assert!(cdr.prefill.get("rq").is_some(), "lifecycle latched");
            Ok(())
        }

        /// A re-poll refreshes the stored external count in place (no second
        /// mint) and the engine cell tracks the recompute.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_prefill_repoll_refreshes_stored_count() -> Result<()> {
            let rig = prefill_rig().await?;
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let first = fb_prefill_req(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                BS,
                3 * BS + 1,
            );
            let (_, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&first, None)?);
            let live = minted.expect("latched");

            // vLLM re-polls with more computed tokens.
            let repoll = fb_prefill_req(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                2 * BS,
                3 * BS + 1,
            );
            let (matched, minted, _) =
                expect_resolved(rig.engine.clone().find_blocks(&repoll, Some(&live))?);
            assert_eq!(matched, BS, "recomputed external count");
            assert!(minted.is_none(), "a refresh never mints a second handle");

            let cdr = rig.engine.cd.as_ref().unwrap();
            let state = cdr.prefill.get("rq").expect("latched");
            assert_eq!(state.external_tokens(), BS, "the stored cell tracked it");
            Ok(())
        }

        /// `find_blocks(live = None)` over an already-latched prefill
        /// lifecycle fails loud — the connector lost the handle that owns the
        /// RAII release.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_prefill_desync_without_live_handle() -> Result<()> {
            let rig = prefill_rig().await?;
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let req = fb_prefill_req(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                BS,
                3 * BS + 1,
            );
            let (_, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            let _live = minted.expect("latched");

            assert!(matches!(
                rig.engine.clone().find_blocks(&req, None),
                Err(LeaderEngineError::FindBlocksDesync)
            ));
            Ok(())
        }

        /// A Search-kind live handle on a dispatched-prefill request is a kind
        /// disagreement, refused BEFORE any lifecycle is latched.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_prefill_search_kind_live_desyncs() -> Result<()> {
            let rig = fallthrough_rig(3).await?;
            let dyn_engine: Arc<dyn LeaderEngine> = rig.engine.clone();
            let live = FindBlocksHandle::search(
                "rq".to_string(),
                SearchId::new(),
                Arc::downgrade(&dyn_engine),
            );
            let req = fb_prefill_req(
                &rig.plhs,
                Uuid::new_v4(),
                Uuid::new_v4().into(),
                None,
                BS,
                0,
                3 * BS + 1,
            );
            assert!(matches!(
                rig.engine.clone().find_blocks(&req, Some(&live)),
                Err(LeaderEngineError::FindBlocksDesync)
            ));
            let cdr = rig.engine.cd.as_ref().unwrap();
            assert!(cdr.prefill.is_empty(), "nothing latched on the refusal");
            Ok(())
        }

        /// `onboard_blocks` validates the committed external count against the
        /// engine-stored promise; a corrected retry still works (the
        /// validation precedes the one-onboard claim).
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn onboard_blocks_prefill_external_mismatch_then_corrected_retry() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let req = fb_prefill_req(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                BS,
                3 * BS + 1,
            );
            let (_, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            let handle = minted.expect("latched");
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            holder.commit(rig.plhs.clone())?;
            holder.make_available(rig.window.clone())?;
            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pulls_complete()).await;

            assert!(matches!(
                rig.engine
                    .clone()
                    .onboard_blocks(&handle, &[40usize, 50, 51], BS),
                Err(LeaderEngineError::ExternalTokensMismatch {
                    expected,
                    got,
                }) if expected == 2 * BS && got == BS
            ));

            // The mismatch did not consume the generation's one onboard.
            let onboard = rig
                .engine
                .clone()
                .onboard_blocks(&handle, &[40usize, 50, 51], 2 * BS)?;
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            Ok(())
        }

        /// A short `block_ids` refusal (`InvalidPrefillRequest`) must not
        /// consume the generation's one onboard claim — a corrected retry
        /// succeeds. Fails if the claim is taken before the length check.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn onboard_blocks_prefill_short_dest_then_corrected_retry() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let req = fb_prefill_req(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                BS,
                3 * BS + 1,
            );
            let (_, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            let handle = minted.expect("latched");
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            holder.commit(rig.plhs.clone())?;
            holder.make_available(rig.window.clone())?;
            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pulls_complete()).await;

            // external = 2 blocks but only 1 dest id — refused loudly.
            assert!(matches!(
                rig.engine
                    .clone()
                    .onboard_blocks(&handle, &[40usize], 2 * BS),
                Err(LeaderEngineError::InvalidPrefillRequest { .. })
            ));

            // The refusal did not consume the generation's one onboard.
            let onboard = rig
                .engine
                .clone()
                .onboard_blocks(&handle, &[40usize, 50, 51], 2 * BS)?;
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            Ok(())
        }

        /// Happy-path port of the USAA suffix contract through the unified
        /// verbs: the kick copies exactly the external G1 suffix.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn onboard_blocks_prefill_kick_targets_external_suffix() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let req = fb_prefill_req(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                BS,
                3 * BS + 1,
            );
            let (matched, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            assert_eq!(matched, 2 * BS);
            let handle = minted.expect("latched");

            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            holder.commit(rig.plhs.clone())?;
            holder.make_available(rig.window.clone())?;
            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pulls_complete()).await;

            // Full vLLM allocation `[computed prefix | external]`.
            let onboard = rig
                .engine
                .clone()
                .onboard_blocks(&handle, &[40usize, 50, 51], 2 * BS)?;
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            assert_eq!(
                rig.workers.transfer_calls(),
                vec![vec![50, 51]],
                "the kick targets exactly the external G1 suffix"
            );

            // The lifecycle stays parked until the handle's RAII release.
            assert!(cdr.prefill.get("rq").is_some());
            drop(onboard);
            drop(handle);
            assert!(cdr.prefill.get("rq").is_none(), "release removed the state");
            Ok(())
        }

        /// FULL decode lifecycle over the real `Arc<dyn LeaderEngine>` seam,
        /// asserting the engine PUSHED the load terminal to the worker sink (not
        /// merely that the `OnboardHandle` cell flipped). Drives the same verbs
        /// the connector does — `find_blocks` (a dispatched remote prefill) →
        /// peer commit + availability (the prefill peer's side, so the decode
        /// pull resolves) → `onboard_blocks` → transfer completion — and proves
        /// `mark_load_finished(LoadOutcome::Done)` reaches `rig.sink`.
        ///
        /// Non-vacuity is explicit: the sink is empty through find/commit/pull
        /// and a real external-suffix transfer runs, so the `Done` is the actual
        /// onboard terminal, not a zero-block no-op.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn onboard_blocks_prefill_full_lifecycle_pushes_mark_load_finished() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            // find_blocks: a dispatched remote prefill latches the lifecycle.
            let req = fb_prefill_req(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                BS,
                3 * BS + 1,
            );
            let (matched, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            assert_eq!(matched, 2 * BS);
            let handle = minted.expect("latched");

            // Inject the prefill peer's commit + availability so the decode pull
            // resolves; wait for the pulls to land.
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            holder.commit(rig.plhs.clone())?;
            holder.make_available(rig.window.clone())?;
            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pulls_complete()).await;

            // Nothing in the lifecycle so far fires the load terminal: find,
            // commit, availability, and the pull are all upstream of onboard.
            assert!(
                rig.sink.loads().is_empty(),
                "no load terminal before onboard_blocks"
            );

            // onboard_blocks: the kick copies the external G2 suffix onto the
            // external G1 slice of the full `[computed prefix | external]` alloc.
            let onboard = rig
                .engine
                .clone()
                .onboard_blocks(&handle, &[40usize, 50, 51], 2 * BS)?;
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            assert_eq!(
                rig.workers.transfer_calls(),
                vec![vec![50, 51]],
                "a real external-suffix transfer ran (Done is not a no-op)"
            );

            // THE assertion this canary adds: the engine pushed the per-request
            // load terminal to the worker sink (driver fires it lock-free, just
            // after the handle cell flips).
            wait_for(|| !rig.sink.loads().is_empty()).await;
            assert_eq!(
                rig.sink.loads(),
                vec![("rq".to_string(), LoadOutcome::Done)],
                "exactly one mark_load_finished(Done) for the request"
            );

            // Hold the handle until after the sink assert (its drop releases the
            // session); the state stays parked until then.
            assert!(cdr.prefill.get("rq").is_some());
            drop(onboard);
            drop(handle);
            assert!(cdr.prefill.get("rq").is_none(), "release removed the state");
            Ok(())
        }

        /// Re-entrancy, pre-pulls-complete: a second `onboard_blocks` on the
        /// live generation is refused while the first kick is parked; the
        /// parked kick still completes once the pulls land, and exactly one
        /// transfer ever runs.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn onboard_blocks_prefill_reentry_pre_pulls_complete_refused() -> Result<()> {
            let rig = prefill_rig().await?;
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let req = fb_prefill_req(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                BS,
                3 * BS + 1,
            );
            let (_, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            let handle = minted.expect("latched");
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;

            // Pulls incomplete: the first onboard parks its kick…
            let onboard = rig
                .engine
                .clone()
                .onboard_blocks(&handle, &[40usize, 50, 51], 2 * BS)?;
            assert!(!onboard.is_complete(), "kick parked until pulls complete");

            // …and a re-entrant onboard is refused instead of replacing it.
            assert!(matches!(
                rig.engine
                    .clone()
                    .onboard_blocks(&handle, &[40usize, 50, 51], 2 * BS),
                Err(LeaderEngineError::OnboardAlreadyInFlight)
            ));

            holder.commit(rig.plhs.clone())?;
            holder.make_available(rig.window.clone())?;
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            assert_eq!(
                rig.workers.transfer_calls(),
                vec![vec![50, 51]],
                "exactly one transfer despite the re-entrant attempt"
            );
            Ok(())
        }

        /// Re-entrancy, post-complete: after the first onboard's kick fired
        /// and completed, a second `onboard_blocks` on the same generation is
        /// refused (it would lose the exactly-once kick race and never reach a
        /// terminal).
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn onboard_blocks_prefill_reentry_post_complete_refused() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let req = fb_prefill_req(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                BS,
                3 * BS + 1,
            );
            let (_, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            let handle = minted.expect("latched");
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            holder.commit(rig.plhs.clone())?;
            holder.make_available(rig.window.clone())?;
            let state = cdr.prefill.get("rq").expect("latched");
            wait_for(|| state.pulls_complete()).await;

            let onboard = rig
                .engine
                .clone()
                .onboard_blocks(&handle, &[40usize, 50, 51], 2 * BS)?;
            wait_for(|| onboard.is_complete()).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));

            assert!(matches!(
                rig.engine
                    .clone()
                    .onboard_blocks(&handle, &[40usize, 50, 51], 2 * BS),
                Err(LeaderEngineError::OnboardAlreadyInFlight)
            ));
            assert_eq!(
                rig.workers.transfer_calls(),
                vec![vec![50, 51]],
                "the refused re-entry ran no second transfer"
            );
            Ok(())
        }

        /// Zero-external fall-through: the dispatched prefill's local search
        /// runs BOUND INSIDE the lifecycle — the local hit resolves
        /// token-granular, the one minted handle is Prefill-kind, and the
        /// internal latch is bound to the generation (never handed out).
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_prefill_fallthrough_local_hit_binds_internal_search() -> Result<()> {
            let rig = fallthrough_rig(4).await?;
            let cdr = rig.engine.cd.as_ref().unwrap();

            // provided == computed → zero external; chain[1..4] resident.
            let req = fb_prefill_req(
                &rig.plhs,
                Uuid::new_v4(),
                Uuid::new_v4().into(),
                None,
                BS,
                BS,
                4 * BS + 1,
            );
            let (matched, minted, release_parked) =
                expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            assert_eq!(matched, 3 * BS, "the fall-through local hit, in tokens");
            let handle = minted.expect("the first poll still mints");
            assert!(
                handle.prefill_accept_id().is_some(),
                "the connector-visible handle stays Prefill-kind"
            );
            assert!(
                !release_parked,
                "the prefill lifecycle must stay parked (cache-warm pull alive)"
            );

            let state = cdr.prefill.get("rq").expect("lifecycle latched");
            assert!(
                state.local_search_id().is_some(),
                "the local search is bound INSIDE the lifecycle"
            );
            assert_eq!(rig.engine.searches.len(), 1, "one internal latch");
            Ok(())
        }

        /// The fall-through's internal mint is EXEMPT from the deferral guard
        /// (the disagg dispatcher placed the request; it must never park) —
        /// while the same window defers a pure-local fresh mint.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_prefill_fallthrough_internal_mint_exempt_from_deferral() -> Result<()>
        {
            let rig = fallthrough_rig(4).await?;

            // An in-flight onboard covers part of the eligible window.
            rig.engine
                .inflight
                .lock()
                .unwrap()
                .record(InflightKey::Search(SearchId::new()), vec![rig.plhs[1]]);

            // Control: a pure-local fresh mint over the window defers.
            let local = fb_req("other", rig.plhs[1..].to_vec(), 0, 3 * BS + 1);
            assert!(matches!(
                rig.engine.clone().find_blocks(&local, None)?,
                FindBlocksOutcome::Deferred
            ));

            // The dispatched prefill's fall-through mint proceeds.
            let req = fb_prefill_req(
                &rig.plhs,
                Uuid::new_v4(),
                Uuid::new_v4().into(),
                None,
                BS,
                BS,
                4 * BS + 1,
            );
            let (matched, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            assert_eq!(matched, 3 * BS, "exempt: the internal mint ran");
            assert!(minted.is_some());
            Ok(())
        }

        /// Zero-stored delegation: `onboard_blocks` on a zero-external prefill
        /// routes through the internally-bound local search; a second call is
        /// naturally refused (the delegation consumed the latch).
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn onboard_blocks_zero_stored_delegates_to_bound_search() -> Result<()> {
            let rig = fallthrough_rig(4).await?;

            let req = fb_prefill_req(
                &rig.plhs,
                Uuid::new_v4(),
                Uuid::new_v4().into(),
                None,
                BS,
                BS,
                4 * BS + 1,
            );
            let (matched, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            assert_eq!(matched, 3 * BS);
            let handle = minted.expect("latched");

            // Full vLLM allocation `[computed prefix | external]`; the local
            // onboard slices the matched span past the computed block.
            let onboard = rig
                .engine
                .clone()
                .onboard_blocks(&handle, &[1usize, 2, 3, 4], 3 * BS)?;
            wait_complete(&onboard).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            assert_eq!(
                rig.workers.transfer_calls(),
                vec![vec![2, 3, 4]],
                "the delegation onboards the matched span onto the dest suffix"
            );
            // The delegated onboard recorded under the INTERNAL search
            // generation; the terminal above did not clear it.
            assert!(
                rig.engine.inflight.lock().unwrap().overlaps(&[rig.plhs[1]]),
                "the delegated onboard's window stays recorded past its terminal"
            );

            assert!(matches!(
                rig.engine
                    .clone()
                    .onboard_blocks(&handle, &[1usize, 2, 3, 4], 3 * BS),
                Err(LeaderEngineError::SearchNotMatched)
            ));

            // The fall-through funnel: dropping the PREFILL handle releases
            // the generation, whose `take_local_search` → `release_search`
            // clears the delegated onboard's record.
            drop(handle);
            assert!(
                rig.engine.inflight.lock().unwrap().is_empty(),
                "the prefill release clears the bound search's record"
            );
            Ok(())
        }

        /// The RAII release tears BOTH down: dropping the prefill handle
        /// releases the lifecycle AND its internally-bound local search.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_handle_drop_releases_prefill_and_bound_search() -> Result<()> {
            let rig = fallthrough_rig(4).await?;
            let cdr = rig.engine.cd.as_ref().unwrap();

            let req = fb_prefill_req(
                &rig.plhs,
                Uuid::new_v4(),
                Uuid::new_v4().into(),
                None,
                BS,
                BS,
                4 * BS + 1,
            );
            let (_, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            let handle = minted.expect("latched");
            assert_eq!(rig.engine.searches.len(), 1, "internal latch bound");
            assert!(cdr.prefill.get("rq").is_some());

            drop(handle);
            assert!(cdr.prefill.get("rq").is_none(), "lifecycle released");
            assert!(
                rig.engine.searches.is_empty(),
                "the bound internal search released with it"
            );
            Ok(())
        }

        /// `evict` releases the prefill generation engine-internally; a stale
        /// connector handle dropping later no-ops on the generation guard, and
        /// a fresh re-accept latches independently.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn evict_releases_prefill_generation_and_stale_drop_is_safe() -> Result<()> {
            let rig = prefill_rig().await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let session_id = Uuid::new_v4();
            let holder = open_holder(&rig, session_id);
            let initiator: InstanceId = Uuid::new_v4().into();

            let req = fb_prefill_req(
                &rig.plhs,
                session_id,
                initiator,
                holder.endpoint(),
                3 * BS,
                BS,
                3 * BS + 1,
            );
            let (_, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            let stale = minted.expect("generation 1 latched");
            wait_for(|| rig.puller_f.attach_calls().len() == 1).await;
            let puller = rig.puller_f.last_attached().expect("attached");

            // Evict tears the generation down engine-internally — the caller
            // needs no prefill knowledge.
            let _fence = rig.engine.evict(&"rq".to_string());
            assert!(
                cdr.prefill.get("rq").is_none(),
                "evict released the latched generation"
            );
            wait_for(|| puller.closed_reason().is_some()).await;

            // A fresh accept latches generation 2 independently.
            let (_, minted, _) = expect_resolved(rig.engine.clone().find_blocks(&req, None)?);
            let fresh = minted.expect("generation 2 latched");
            assert!(cdr.prefill.get("rq").is_some());

            // The stale generation-1 handle drops AFTER the re-accept: its
            // release no-ops on the AcceptId guard instead of tearing down
            // the fresh lifecycle.
            drop(stale);
            assert!(
                cdr.prefill.get("rq").is_some(),
                "the stale drop must not touch generation 2"
            );

            drop(fresh);
            assert!(cdr.prefill.get("rq").is_none(), "generation 2 released");
            Ok(())
        }

        /// Synthetic Pending-bound fall-through lifecycle for the two
        /// teardown-order pins below: a latched zero-external generation whose
        /// internally-bound local search is still `Pending`.
        fn pending_bound_lifecycle(
            engine: &Arc<LocalConnectorEngine>,
        ) -> (
            kvbm_protocols::connector::AcceptId,
            watch::Sender<OnboardingStatus>,
        ) {
            let (find_session, tx) = pending_async();
            let search_id = kvbm_protocols::connector::SearchId::new();
            engine.searches.insert(
                search_id,
                SearchState {
                    request_id: "rq".into(),
                    status: Arc::new(Mutex::new(MatchStatus::Pending)),
                    onboarding: OnboardingState::new(
                        0,
                        BS + 1,
                        OnboardingShard {
                            start_block: 0,
                            num_queried_blocks: 1,
                            find_session,
                        },
                    ),
                    buffer: vec![],
                },
            );
            let cdr = engine.cd.as_ref().unwrap();
            let accept_id = kvbm_protocols::connector::AcceptId::new();
            let state = Arc::new(crate::remote::cd::prefill::PrefillRequestState::new(
                accept_id,
                vec![],
                Arc::new(AtomicUsize::new(0)),
                uuid::Uuid::new_v4(),
            ));
            state.bind_local_search(search_id);
            cdr.prefill
                .insert("rq".to_string(), state)
                .expect("fresh latch");
            (accept_id, tx)
        }

        /// Teardown order pin: the RAII drop fires while the internally-bound
        /// fall-through search is still `Pending` — both lifecycles release.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn find_blocks_handle_drop_releases_pending_bound_search() -> Result<()> {
            let rig = fallthrough_rig(4).await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let (accept_id, _tx) = pending_bound_lifecycle(&rig.engine);

            let dyn_engine: Arc<dyn LeaderEngine> = rig.engine.clone();
            let handle =
                FindBlocksHandle::prefill("rq".to_string(), accept_id, Arc::downgrade(&dyn_engine));
            drop(handle);
            assert!(cdr.prefill.get("rq").is_none(), "lifecycle released");
            assert!(
                rig.engine.searches.is_empty(),
                "the Pending bound search released with it"
            );
            Ok(())
        }

        /// Teardown order pin: `evict` (not the RAII drop) releases the
        /// generation AND its internally-bound `Pending` search; the stale
        /// handle drop afterwards no-ops.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn evict_releases_pending_bound_search_with_generation() -> Result<()> {
            let rig = fallthrough_rig(4).await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let (accept_id, _tx) = pending_bound_lifecycle(&rig.engine);

            let dyn_engine: Arc<dyn LeaderEngine> = rig.engine.clone();
            let stale =
                FindBlocksHandle::prefill("rq".to_string(), accept_id, Arc::downgrade(&dyn_engine));

            let _fence = rig.engine.evict(&"rq".to_string());
            assert!(
                cdr.prefill.get("rq").is_none(),
                "evict released the generation"
            );
            assert!(
                rig.engine.searches.is_empty(),
                "evict released the Pending bound search"
            );

            drop(stale);
            assert!(cdr.prefill.get("rq").is_none(), "stale drop stays a no-op");
            Ok(())
        }

        /// The pre-USAA failure replay must release a fall-through binding
        /// acquired before the external count flipped positive — otherwise
        /// the bound internal search strands in `searches` forever.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn prefill_failure_replay_releases_bound_search() -> Result<()> {
            let rig = fallthrough_rig(4).await?;
            let cdr = rig.engine.cd.as_ref().unwrap();
            let (accept_id, _tx) = pending_bound_lifecycle(&rig.engine);

            let state = cdr.prefill.get("rq").expect("latched");
            state.store_external_tokens(2 * BS);
            state.stash_failure("pipeline failed pre-USAA".to_string());

            let err_or_handle = rig.engine.clone().prefill_onboard_by_id(
                &"rq".to_string(),
                accept_id,
                &[1usize, 2, 3],
            );
            let onboard = err_or_handle.expect("replay mints the immediately-Failed action");
            assert!(matches!(
                onboard.outcome(),
                Some(LoadOutcome::FailedPartial { .. })
            ));
            assert!(cdr.prefill.get("rq").is_none(), "state removed at replay");
            assert!(
                rig.engine.searches.is_empty(),
                "the replay teardown releases the bound internal search"
            );
            Ok(())
        }
    }
}
