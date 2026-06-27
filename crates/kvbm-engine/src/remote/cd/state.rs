// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request decode-side conditional-disaggregation bookkeeping.
//!
//! [`CdRequestState`] is the resource-free derivation core of the legacy
//! decode wrapper's per-request state: token accounting, the local/remote
//! destination block ids, the completion flags for the two onboard pipelines,
//! and the single-winner cleanup guard. The runtime resources (the open
//! session, the pinned G2 blocks) are owned elsewhere.
//!
//! [`CdRequests`] is the engine-owned container of those per-request states:
//! the `DashMap<request_id, Arc<CdRequestState>>` every CD lifecycle bubbles
//! through. It couples each state's budget reservation to its map entry so that
//! the eviction / action-terminal / request-finished release paths are
//! idempotent and identity-checked — a stale release from a recompute-
//! rescheduled prior lifecycle can never wipe a freshly-installed state for the
//! same request id.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use dashmap::DashMap;
use parking_lot::Mutex;

use kvbm_protocols::connector::SearchId;

use crate::p2p::session::Session;
use crate::{BlockId, SequenceHash};

use super::budget::InflightBudget;
use super::commit::AvailabilityLedger;

/// Position-indexed metadata for one block in the remote-prefill slice. No G2
/// mutable is held here — those are allocated in the pull task on availability
/// and consumed by `session.pull`.
struct RemoteSlotMeta {
    sequence_index: usize,
    g1_dst_block_id: BlockId,
}

/// Per-request decode-side CD bookkeeping.
pub(crate) struct CdRequestState {
    reserved_tokens: usize,

    /// The unified matched-block count latched at the search-time commit:
    /// `full_block_external_tokens / block_size`. Every subsequent search /
    /// refresh for this request returns this exact count (the idempotent-retry
    /// answer), so it never re-plans nor re-touches the budget.
    unified_hit_blocks: u32,

    /// The committed window hashes `block_plhs[..unified]`, in absolute-position
    /// order. The onboard fan-out derives the remote slice from this at USAA
    /// time: `window_hashes[local_hit_blocks..unified]` are the hashes the remote
    /// prefill peer commits + makes available for decode to pull.
    window_hashes: Vec<SequenceHash>,

    /// Number of leading window blocks the decode side matched locally — the
    /// split point between the local kick (`[0, local_hit)`) and the remote pull
    /// (`[local_hit, unified)`) at onboard time.
    local_hit_blocks: u32,

    /// The [`SearchId`] of the search that interposed this commit — the
    /// originating GENERATION of this CD lifecycle. The decline release hook
    /// ([`super::super::tiering::engine`]'s `release_search`) compares this
    /// against the `SearchId` it is releasing and tears down ONLY a matching
    /// lifecycle: a stale OLD-generation handle dropping AFTER an evict +
    /// re-latch of the same request id then no-ops on the FRESH lifecycle
    /// instead of closing its session + releasing its budget.
    search_id: SearchId,

    /// The holder-side CD session opened at the search-time commit, parked here
    /// and handed forward — never re-queried by name (the legacy USAA-1 race).
    /// The release hooks (decline / evict / finish) [`take`](Self::take_session)
    /// it to close the session.
    session: Mutex<Option<Arc<dyn Session>>>,

    /// The deferred-availability ledger from the search-time commit, parked
    /// alongside the session for the USAA fan-out to drive (incremental
    /// `make_available` + the single deferred `finish_availability`).
    ledger: Mutex<Option<AvailabilityLedger>>,

    /// GNMT-time `num_computed_tokens` (the vLLM-computed prefix length).
    /// Authoritative `num_computed` for the USAA-1 split: a vLLM prefix-cache hit
    /// with zero G2 match leaves the slot's onboarding state `None`, so the
    /// onboarding-derived num_computed would be 0 and the split would under-count
    /// `computed_blocks` (over-counting the remote blocks → USAA-1 mismatch crash).
    base_offset: usize,

    /// G1 destinations vLLM allocated for the local-match slice
    /// `[computed, computed + local_match)`.
    local_match_g1_block_ids: Vec<BlockId>,
    local_onboard_complete: AtomicBool,

    /// Per-position remote-slice metadata, in expected order. Built at USAA-1 and
    /// read-only afterward.
    remote_slots: Vec<RemoteSlotMeta>,
    /// `expected_hash → index in remote_slots` lookup; built once.
    remote_slot_index: HashMap<SequenceHash, usize>,

    remote_pipeline_complete: AtomicBool,
    completed: AtomicBool,

    /// Pre-onboard failure stash. A failure observed before the engine's onboard
    /// can surface it is stashed here so onboard can immediately return a failed
    /// action carrying the now-known G1 destination ids; if the request is torn
    /// down before onboard arrives, nothing is emitted.
    pending_failure: Mutex<Option<String>>,

    /// Single-winner cleanup guard. Many concurrent failure paths can race to
    /// clean up the same request; the CAS winner (false → true) runs the cleanup
    /// and notifies, the losers early-return so the failure is surfaced exactly
    /// once. See [`CdRequestState::claim_cleanup`].
    cleanup_claimed: AtomicBool,
}

impl CdRequestState {
    /// Construct the search-time committed state for a remote-prefill request.
    ///
    /// The open session + its availability ledger are parked here and handed
    /// forward; the USAA bookkeeping (block ids, remote slots) is empty until
    /// the onboard fan-out fills it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_remote_commit(
        reserved_tokens: usize,
        base_offset: usize,
        unified_hit_blocks: u32,
        local_hit_blocks: u32,
        window_hashes: Vec<SequenceHash>,
        search_id: SearchId,
        session: Arc<dyn Session>,
        ledger: AvailabilityLedger,
    ) -> Self {
        Self {
            reserved_tokens,
            unified_hit_blocks,
            window_hashes,
            local_hit_blocks,
            search_id,
            session: Mutex::new(Some(session)),
            ledger: Mutex::new(Some(ledger)),
            base_offset,
            local_match_g1_block_ids: Vec::new(),
            local_onboard_complete: AtomicBool::new(false),
            remote_slots: Vec::new(),
            remote_slot_index: HashMap::new(),
            remote_pipeline_complete: AtomicBool::new(false),
            completed: AtomicBool::new(false),
            pending_failure: Mutex::new(None),
            cleanup_claimed: AtomicBool::new(false),
        }
    }

    /// The latched unified matched-block count (the idempotent-retry answer).
    pub(crate) fn unified_hit_blocks(&self) -> u32 {
        self.unified_hit_blocks
    }

    /// The GNMT-time computed-prefix offset (in tokens) this lifecycle was
    /// committed at. The idempotent-retry latch compares this against each
    /// re-poll's offset: the latched unified count and window hashes are
    /// absolute facts of THIS offset, so a moved prefix marks the lifecycle
    /// stale.
    pub(crate) fn base_offset(&self) -> usize {
        self.base_offset
    }

    /// Number of leading window blocks matched locally (the local/remote split
    /// point used by the onboard fan-out).
    pub(crate) fn local_hit_blocks(&self) -> u32 {
        self.local_hit_blocks
    }

    /// The committed window hashes `[0, unified)` in absolute-position order.
    /// The onboard fan-out slices `[local_hit_blocks..unified]` to get the
    /// remote-pull hashes.
    pub(crate) fn window_hashes(&self) -> &[SequenceHash] {
        &self.window_hashes
    }

    /// The originating search generation of this CD lifecycle. The decline
    /// release hook compares this against the `SearchId` being released so a
    /// stale handle never tears down a re-latched fresh lifecycle.
    pub(crate) fn search_id(&self) -> SearchId {
        self.search_id
    }

    /// Clone the parked session WITHOUT taking it. The onboard fan-out drives the
    /// pull over this clone; the terminal release hooks own teardown via
    /// [`Self::take_session`].
    pub(crate) fn clone_session(&self) -> Option<Arc<dyn Session>> {
        self.session.lock().clone()
    }

    /// Whether the deferred-availability ledger has drained (every committed
    /// source landed and `finish_availability` fired). A request with no pending
    /// sources starts drained; the load terminal finalizes the session only once
    /// this is true. Reads `true` if no ledger is parked.
    pub(crate) fn ledger_is_drained(&self) -> bool {
        self.ledger
            .lock()
            .as_ref()
            .map(|l| l.is_drained())
            .unwrap_or(true)
    }

    /// Take the parked session, leaving `None`. The release hooks close it
    /// exactly once; a second take yields `None`.
    pub(crate) fn take_session(&self) -> Option<Arc<dyn Session>> {
        self.session.lock().take()
    }

    /// Stash a pre-onboard failure (e.g. the prefill dispatch resolved `Err`)
    /// so the engine's onboard can surface it as a failed action when the
    /// fan-out lands; if the request is torn down first, nothing is emitted.
    pub(crate) fn stash_failure(&self, reason: String) {
        *self.pending_failure.lock() = Some(reason);
    }

    /// The stashed pre-onboard failure, if any.
    pub(crate) fn pending_failure(&self) -> Option<String> {
        self.pending_failure.lock().clone()
    }

    /// G1 destination ids that still need filling: the local-match ids when the
    /// local onboard has not completed, followed by the remote-slice destination
    /// ids when the remote pipeline has not completed.
    fn unfilled_g1_block_ids(&self) -> Vec<BlockId> {
        let mut out = Vec::new();
        if !self.local_onboard_complete.load(Ordering::Acquire) {
            out.extend(self.local_match_g1_block_ids.iter().copied());
        }
        if !self.remote_pipeline_complete.load(Ordering::Acquire) {
            out.extend(self.remote_slots.iter().map(|s| s.g1_dst_block_id));
        }
        out
    }

    /// Claim the right to run cleanup for this request. Returns `true` for the
    /// single CAS winner (false → true) and `false` for every subsequent caller,
    /// so the failure is notified exactly once across the concurrent paths.
    fn claim_cleanup(&self) -> bool {
        self.cleanup_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }
}

/// Engine-owned container of per-request CD state — the map every CD lifecycle
/// bubbles through. Mirrors the legacy decode wrapper's `cd_request_state:
/// DashMap<String, Arc<CdRequestState>>` field plus its `release_request` /
/// `release_request_if_matches` methods.
///
/// Ownership invariant: each [`CdRequestState`] records a budget reservation in
/// its `reserved_tokens`. While a state is held here, THIS CONTAINER owns that
/// reservation, and every removal path ([`release`](Self::release) /
/// [`release_if_matches`](Self::release_if_matches)) hands exactly those tokens
/// back to the [`InflightBudget`]. Removals are idempotent (a second release, or
/// a release of an absent rid, is a no-op that returns `false`) and the
/// identity-checked variant guards the cross-lifecycle stale-release window, so
/// the budget can never be leaked nor double-released, and a recompute-
/// rescheduled lifecycle of the same rid is never wiped by a stale release.
pub(crate) struct CdRequests {
    inner: DashMap<String, Arc<CdRequestState>>,
}

impl Default for CdRequests {
    fn default() -> Self {
        Self::new()
    }
}

impl CdRequests {
    pub(crate) fn new() -> Self {
        Self {
            inner: DashMap::new(),
        }
    }

    /// Track `state` under `request_id`, taking ownership of the budget
    /// reservation recorded in `state.reserved_tokens`.
    ///
    /// Bails with [`CdInsertError::AlreadyTracked`] when `request_id` is already
    /// tracked — the legacy "begin_remote_prefill called twice" guard. The map
    /// is left untouched on the bail; the existing entry (and the reservation it
    /// owns) is preserved.
    ///
    /// Reservation-ownership transfer: BEFORE a successful insert the caller
    /// still owns the reservation and must release it DIRECTLY back to the
    /// budget on any pre-insert failure. AFTER `Ok(())` the reservation is owned
    /// by this container, so every post-insert failure (transfer/onboard fail,
    /// action terminal, eviction, request-finished) must route through
    /// [`release`](Self::release) / [`release_if_matches`](Self::release_if_matches)
    /// — handing the budget back exactly once and never leaking it.
    pub(crate) fn insert(
        &self,
        request_id: String,
        state: Arc<CdRequestState>,
    ) -> Result<(), CdInsertError> {
        use dashmap::mapref::entry::Entry;
        match self.inner.entry(request_id) {
            // Atomic check-and-set: the occupied branch holds the shard lock, so
            // a concurrent vacant insert of the same rid cannot slip in between a
            // separate `contains_key` and `insert`.
            Entry::Occupied(occupied) => Err(CdInsertError::AlreadyTracked {
                request_id: occupied.key().clone(),
            }),
            Entry::Vacant(vacant) => {
                vacant.insert(state);
                Ok(())
            }
        }
    }

    /// Clone out the tracked state for `request_id`, if any. The returned `Arc`
    /// is the identity a later [`release_if_matches`](Self::release_if_matches)
    /// caller compares against to stay bound to one specific lifecycle.
    pub(crate) fn get(&self, request_id: &str) -> Option<Arc<CdRequestState>> {
        self.inner.get(request_id).map(|e| Arc::clone(e.value()))
    }

    /// Unconditional release: remove `request_id` and hand its reservation back
    /// to `budget`. The terminal bubble for eviction / action-terminal /
    /// request-finished when the caller has not captured a specific `Arc`
    /// snapshot to guard against.
    ///
    /// Idempotent: a second release — or a release of a never-tracked rid — is a
    /// no-op that returns `false` and leaves the budget untouched, so the budget
    /// can never be double-released. Returns `true` only when an entry was
    /// removed and its budget released.
    pub(crate) fn release(&self, request_id: &str, budget: &InflightBudget) -> bool {
        match self.inner.remove(request_id) {
            Some((_, state)) => {
                budget.release(state.reserved_tokens);
                true
            }
            None => false,
        }
    }

    /// Identity-checked release: remove `request_id` ONLY when the currently
    /// tracked `Arc` is pointer-equal to `expected`, releasing the budget solely
    /// on that true removal. This is the cross-lifecycle stale-release guard.
    ///
    /// A spawn-replayed cleanup task parked from a PRIOR lifecycle of the SAME
    /// rid (under `kv_load_failure_policy=recompute`, a fresh [`CdRequestState`]
    /// is re-inserted under the same rid after the first was released) must NOT
    /// wipe the budget reservation or evict the freshly-installed state. The
    /// [`Arc::ptr_eq`] check inside the [`DashMap`] `remove_if` closes that
    /// window: a stale `expected` no longer matches what the map holds, so
    /// nothing is removed and no budget is released.
    ///
    /// Returns `true` only when the matching `Arc` was removed and exactly its
    /// `reserved_tokens` released.
    pub(crate) fn release_if_matches(
        &self,
        request_id: &str,
        expected: &Arc<CdRequestState>,
        budget: &InflightBudget,
    ) -> bool {
        match self
            .inner
            .remove_if(request_id, |_, v| Arc::ptr_eq(expected, v))
        {
            Some((_, state)) => {
                budget.release(state.reserved_tokens);
                true
            }
            None => false,
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Failure modes for [`CdRequests::insert`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum CdInsertError {
    /// `request_id` is already tracked — the legacy "begin_remote_prefill called
    /// twice" guard. The caller still owns the reservation it was about to hand
    /// over and must release it directly back to the budget.
    #[error("CD request already tracked: request_id={request_id}")]
    AlreadyTracked { request_id: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with(local_g1: Vec<BlockId>, remote_dst: Vec<BlockId>) -> CdRequestState {
        let remote_slots = remote_dst
            .into_iter()
            .enumerate()
            .map(|(sequence_index, g1_dst_block_id)| RemoteSlotMeta {
                sequence_index,
                g1_dst_block_id,
            })
            .collect();
        CdRequestState {
            reserved_tokens: 0,
            unified_hit_blocks: 0,
            window_hashes: Vec::new(),
            local_hit_blocks: 0,
            search_id: SearchId::new(),
            session: Mutex::new(None),
            ledger: Mutex::new(None),
            base_offset: 0,
            local_match_g1_block_ids: local_g1,
            local_onboard_complete: AtomicBool::new(false),
            remote_slots,
            remote_slot_index: HashMap::new(),
            remote_pipeline_complete: AtomicBool::new(false),
            completed: AtomicBool::new(false),
            pending_failure: Mutex::new(None),
            cleanup_claimed: AtomicBool::new(false),
        }
    }

    #[test]
    fn unfilled_g1_block_ids_both_incomplete_yields_local_then_remote() {
        let state = state_with(vec![1, 2], vec![10, 11]);
        assert_eq!(state.unfilled_g1_block_ids(), vec![1, 2, 10, 11]);
    }

    #[test]
    fn unfilled_g1_block_ids_local_complete_yields_remote_only() {
        let state = state_with(vec![1, 2], vec![10, 11]);
        state.local_onboard_complete.store(true, Ordering::Release);
        assert_eq!(state.unfilled_g1_block_ids(), vec![10, 11]);
    }

    #[test]
    fn unfilled_g1_block_ids_remote_complete_yields_local_only() {
        let state = state_with(vec![1, 2], vec![10, 11]);
        state
            .remote_pipeline_complete
            .store(true, Ordering::Release);
        assert_eq!(state.unfilled_g1_block_ids(), vec![1, 2]);
    }

    #[test]
    fn unfilled_g1_block_ids_both_complete_is_empty() {
        let state = state_with(vec![1, 2], vec![10, 11]);
        state.local_onboard_complete.store(true, Ordering::Release);
        state
            .remote_pipeline_complete
            .store(true, Ordering::Release);
        assert!(state.unfilled_g1_block_ids().is_empty());
    }

    #[test]
    fn claim_cleanup_single_winner() {
        let state = state_with(vec![1], vec![10]);
        assert!(state.claim_cleanup(), "first claim wins");
        assert!(!state.claim_cleanup(), "second claim loses");
        assert!(!state.claim_cleanup(), "subsequent claims keep losing");
    }

    // ---- CdRequests container -------------------------------------------

    /// `state_with()`-style builder for an `Arc<CdRequestState>` carrying a
    /// chosen `reserved_tokens` — the only field the container's budget coupling
    /// reads.
    fn state_reserving(reserved_tokens: usize) -> Arc<CdRequestState> {
        let mut state = state_with(vec![], vec![]);
        state.reserved_tokens = reserved_tokens;
        Arc::new(state)
    }

    #[test]
    fn insert_then_release_returns_budget_and_second_release_is_noop() {
        let reqs = CdRequests::new();
        let budget = InflightBudget::new(256);
        let reserved = 64;
        assert!(budget.try_reserve(reserved));
        assert_eq!(budget.available(), 256 - reserved);

        reqs.insert("r1".to_string(), state_reserving(reserved))
            .unwrap();
        assert_eq!(reqs.len(), 1);

        // First release removes the entry and returns the reservation to
        // capacity.
        assert!(reqs.release("r1", &budget));
        assert_eq!(budget.available(), 256);
        assert!(reqs.is_empty());

        // Idempotent: second release is a no-op, returns false, budget unchanged.
        assert!(!reqs.release("r1", &budget));
        assert_eq!(budget.available(), 256);
    }

    #[test]
    fn release_on_never_inserted_rid_is_false_and_budget_untouched() {
        let reqs = CdRequests::new();
        let budget = InflightBudget::new(128);
        assert!(!reqs.release("ghost", &budget));
        assert_eq!(budget.available(), 128);
    }

    #[test]
    fn duplicate_insert_bails_already_tracked_and_preserves_original() {
        let reqs = CdRequests::new();
        let original = state_reserving(32);
        reqs.insert("r1".to_string(), Arc::clone(&original))
            .unwrap();

        let err = reqs
            .insert("r1".to_string(), state_reserving(99))
            .unwrap_err();
        assert_eq!(
            err,
            CdInsertError::AlreadyTracked {
                request_id: "r1".to_string()
            }
        );

        // The original entry is undisturbed: same Arc identity, original
        // reserved_tokens, single entry.
        let held = reqs.get("r1").expect("original entry still present");
        assert!(Arc::ptr_eq(&held, &original));
        assert_eq!(held.reserved_tokens, 32);
        assert_eq!(reqs.len(), 1);
    }

    #[test]
    fn release_if_matches_stale_arc_does_not_remove_or_release() {
        let reqs = CdRequests::new();
        let budget = InflightBudget::new(256);

        // Lifecycle 1: reserve + insert, then release (request finished). Capture
        // the Arc a stale cleanup task from this lifecycle would still hold.
        assert!(budget.try_reserve(64));
        let first = state_reserving(64);
        reqs.insert("r1".to_string(), Arc::clone(&first)).unwrap();
        assert!(reqs.release("r1", &budget));
        assert_eq!(budget.available(), 256);

        // Lifecycle 2 (recompute-rescheduling): the SAME rid is re-tracked with a
        // FRESH state and a fresh reservation.
        assert!(budget.try_reserve(48));
        let second = state_reserving(48);
        reqs.insert("r1".to_string(), Arc::clone(&second)).unwrap();
        assert_eq!(budget.available(), 256 - 48);

        // The stale lifecycle-1 cleanup fires with its captured `first` Arc. The
        // identity check must reject it: lifecycle 2's entry is NOT removed and
        // its budget is NOT released.
        assert!(!reqs.release_if_matches("r1", &first, &budget));
        assert_eq!(
            budget.available(),
            256 - 48,
            "stale release must not touch the live reservation"
        );
        let held = reqs.get("r1").expect("lifecycle-2 entry survives");
        assert!(Arc::ptr_eq(&held, &second));
    }

    #[test]
    fn release_if_matches_matching_arc_removes_and_releases_reserved() {
        let reqs = CdRequests::new();
        let budget = InflightBudget::new(256);
        assert!(budget.try_reserve(80));
        let state = state_reserving(80);
        reqs.insert("r1".to_string(), Arc::clone(&state)).unwrap();
        assert_eq!(budget.available(), 256 - 80);

        // Matching Arc ⇒ removes the entry and releases exactly reserved_tokens.
        assert!(reqs.release_if_matches("r1", &state, &budget));
        assert_eq!(budget.available(), 256);
        assert!(reqs.is_empty());
    }

    #[test]
    fn concurrent_release_and_release_if_matches_never_over_release() {
        // Hammer the same rid with competing release / release_if_matches across
        // threads, repeated to shake out interleavings. Exactly one remover may
        // win; the budget must end precisely at capacity (the debug_assert in
        // InflightBudget::release would fire on any over-release in debug builds).
        for _ in 0..200 {
            let reqs = Arc::new(CdRequests::new());
            let budget = Arc::new(InflightBudget::new(256));
            let reserved = 64;
            assert!(budget.try_reserve(reserved));
            let state = state_reserving(reserved);
            reqs.insert("r1".to_string(), Arc::clone(&state)).unwrap();

            let mut handles = Vec::new();
            for _ in 0..4 {
                let reqs = Arc::clone(&reqs);
                let budget = Arc::clone(&budget);
                handles.push(std::thread::spawn(move || {
                    reqs.release("r1", &budget);
                }));
            }
            for _ in 0..4 {
                let reqs = Arc::clone(&reqs);
                let budget = Arc::clone(&budget);
                let expected = Arc::clone(&state);
                handles.push(std::thread::spawn(move || {
                    reqs.release_if_matches("r1", &expected, &budget);
                }));
            }
            for h in handles {
                h.join().unwrap();
            }

            assert_eq!(budget.available(), 256, "budget must not over-release");
            assert!(reqs.is_empty());
        }
    }
}
