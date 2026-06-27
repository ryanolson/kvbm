// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request prefill-side conditional-disaggregation bookkeeping.
//!
//! [`PrefillRequestState`] is the prefill analogue of
//! [`super::state::CdRequestState`]: the resource-free derivation core for one
//! accepted remote-prefill lifecycle — the expected provided-window hashes, the
//! shared external-token cell the engine refreshes (read off the
//! `find_blocks` outcome the connector sees), the parked puller-side session,
//! the pulled-and-registered G2 pins, and the pulls/USAA two-phase handshake
//! state. Ported from the legacy prefill coordinator's `PrefillBits`
//! (`ensure_started`/`run_setup`/`on_usaa`).
//!
//! [`PrefillRequests`] is the engine-owned container of those states: the
//! `DashMap<request_id, Arc<PrefillRequestState>>` every prefill lifecycle
//! bubbles through, with the same atomic insert latch and `Arc::ptr_eq`-guarded
//! release discipline as [`super::state::CdRequests`] (no budget coupling —
//! the inflight budget is a decode-side concept).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Context, Result};
use dashmap::DashMap;
use parking_lot::Mutex;

use kvbm_logical::blocks::ImmutableBlock;
use kvbm_protocols::connector::{AcceptId, ActionId, SearchId};

use crate::p2p::session::{Session, SessionId};
use crate::{BlockId, G2, SequenceHash};

/// Per-request prefill-side CD bookkeeping for one accept generation.
pub(crate) struct PrefillRequestState {
    /// The generation this lifecycle was accepted under. The RAII release hook
    /// compares this against the [`AcceptId`] it is releasing and tears down
    /// ONLY a matching lifecycle: a stale handle drop from a prior generation
    /// (after evict + re-accept of the same request id) no-ops on the FRESH
    /// lifecycle instead of closing its session.
    accept_id: AcceptId,

    /// The decode-held session this lifecycle was dispatched against. The
    /// accept path compares it against each re-dispatch's `session_id`: a
    /// MATCH is the idempotent re-poll; a MISMATCH means the decode side
    /// recompute-rescheduled this request id onto a fresh session, so the
    /// latched lifecycle is abandoned and must be torn down + replaced (keying
    /// by request id alone would answer the fresh dispatch as a re-poll
    /// against the dead session, and the fresh session would never attach).
    session_id: SessionId,

    /// The provided window `sequence_hashes[0 .. num_provided_tokens /
    /// block_size]` in absolute-position order — the hashes the pull pipeline
    /// drains commits/availability against and pulls into local G2.
    expected_hashes: Vec<SequenceHash>,

    /// The engine-stored external-token count; the engine is the only writer
    /// (each idempotent accept re-poll recomputes and stores `Release`). The
    /// `find_blocks` router reads it `Acquire` and passes it through the
    /// source-agnostic outcome the connector sees.
    external_tokens: Arc<AtomicUsize>,

    /// ONE mutex over the parked session AND the pre-park output buffer.
    /// [`Self::commit_output`] holds it across the full buffer-or-publish
    /// decision and the session dispatch; [`Self::park_session`] stores the
    /// session, takes the buffer, and publishes it under the SAME guard —
    /// making the two legacy buffer/finalize races structurally
    /// unrepresentable (a publish appended to a buffer the attach already
    /// drained; a finalize landing on the wire between the buffer take and
    /// its dispatch). Teardown `take`s the session exactly once (Mutex-take
    /// is the exactly-once close/finalize guard) and closes the slot to
    /// further output.
    session: Mutex<SessionSlot>,

    /// Pulled-and-registered G2 pins, each tagged with its ABSOLUTE
    /// expected-hash index. Availability deltas may arrive sparse or
    /// coalesced, so arrival order is not positional order — the USAA kick
    /// sorts by index before taking the external-count suffix.
    registered_g2: Mutex<Vec<(usize, ImmutableBlock<G2>)>>,

    /// Set (`Release`) by the pipeline once every expected hash is registered;
    /// read (`Acquire`) by the USAA path. One half of the two-phase kick
    /// handshake — see [`Self::latch_kick`].
    pulls_complete: AtomicBool,

    /// The pending-kick / pending-failure pair lives under ONE mutex on
    /// purpose: the pipeline-failure path (take kick, else stash) and the
    /// USAA path (replay stash, else latch kick) would otherwise be a
    /// two-lock Dekker where both sides can miss each other — a failure that
    /// never fires the latched action and a kick that never runs.
    gate: Mutex<UsaaGate>,

    /// CAS backstop ensuring at most one G2→G1 kick ever runs, whichever side
    /// of the two-phase handshake fires it.
    onboarding_scheduled: AtomicBool,

    /// Single-winner cleanup guard: the CAS winner among the concurrent
    /// teardown callers (pipeline failure, kick failure, RAII release) owns
    /// the session close/finalize decision; losers skip it. State REMOVAL
    /// stays idempotent and ptr-guarded outside this claim.
    cleanup_claimed: AtomicBool,

    /// The zero-external fall-through's INTERNAL local-search binding: a
    /// dispatched prefill with no external work still runs the local search,
    /// bound INSIDE this lifecycle (the connector parks only the one prefill
    /// handle). Bound by the router's internal mint, consumed take-once by the
    /// zero-stored onboard delegation's release path, and released with the
    /// generation (RAII release / evict) so one teardown covers both.
    local_search: Mutex<Option<SearchId>>,

    /// One onboard per accept generation: the USAA intercept claims this
    /// before minting its load action. A second onboard on a live generation
    /// would either replace a parked pre-pulls-complete kick (its action
    /// forever `Pending`) or lose the exactly-once kick CAS without a
    /// terminal — the claim refuses it instead.
    usaa_claimed: AtomicBool,
}

/// USAA kick arguments latched by `prefill_onboard_by_id`, consumed exactly once by
/// whichever side of the two-phase handshake fires the kick (or by the
/// pipeline-failure path, which resolves the latched action Failed instead).
pub(crate) struct PrefillKick {
    pub(crate) action_id: ActionId,
    /// The external G1 suffix vLLM allocated (`block_ids[len - external ..]`).
    pub(crate) external_g1: Vec<BlockId>,
}

/// See [`PrefillRequestState::gate`].
struct UsaaGate {
    pending_kick: Option<PrefillKick>,
    pending_failure: Option<String>,
}

/// See [`PrefillRequestState::session`].
struct SessionSlot {
    /// The puller-side session attached to the decode holder, parked here by
    /// the pipeline and handed forward — never re-queried by name.
    session: Option<Arc<dyn Session>>,
    /// Output blocks that arrived before the session was parked; drained and
    /// published by [`PrefillRequestState::park_session`] under this slot's
    /// guard.
    buffered: Vec<ImmutableBlock<G2>>,
    /// Set by [`PrefillRequestState::take_session`] (every teardown and the
    /// finalize both route through it): the lifecycle no longer accepts
    /// output, so late [`PrefillRequestState::commit_output`] calls drop
    /// their blocks instead of buffering them forever.
    output_closed: bool,
}

/// Outcome of [`PrefillRequestState::park_session`].
pub(crate) enum ParkOutcome {
    /// Stored; any buffered output was published.
    Parked,
    /// Cleanup already claimed this lifecycle — NOT stored; the caller owns
    /// closing the session it tried to park.
    Refused,
    /// Stored, but publishing the buffered output failed. The caller should
    /// fail the pipeline; the parked session is closed by that failure path.
    PublishFailed(anyhow::Error),
}

impl PrefillRequestState {
    pub(crate) fn new(
        accept_id: AcceptId,
        expected_hashes: Vec<SequenceHash>,
        external_tokens: Arc<AtomicUsize>,
        session_id: SessionId,
    ) -> Self {
        Self {
            accept_id,
            session_id,
            expected_hashes,
            external_tokens,
            session: Mutex::new(SessionSlot {
                session: None,
                buffered: Vec::new(),
                output_closed: false,
            }),
            registered_g2: Mutex::new(Vec::new()),
            pulls_complete: AtomicBool::new(false),
            gate: Mutex::new(UsaaGate {
                pending_kick: None,
                pending_failure: None,
            }),
            onboarding_scheduled: AtomicBool::new(false),
            cleanup_claimed: AtomicBool::new(false),
            local_search: Mutex::new(None),
            usaa_claimed: AtomicBool::new(false),
        }
    }

    /// The accept generation this lifecycle belongs to.
    pub(crate) fn accept_id(&self) -> AcceptId {
        self.accept_id
    }

    /// The decode-held session this lifecycle was dispatched against. The
    /// accept path tears down + replaces the lifecycle when a re-dispatch
    /// carries a different one.
    pub(crate) fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// The provided-window hashes in absolute-position order.
    pub(crate) fn expected_hashes(&self) -> &[SequenceHash] {
        &self.expected_hashes
    }

    /// The latest stored external-token count.
    pub(crate) fn external_tokens(&self) -> usize {
        self.external_tokens.load(Ordering::Acquire)
    }

    /// Idempotent-re-poll refresh: store the recomputed external-token count
    /// into the cell the held handle reads.
    pub(crate) fn store_external_tokens(&self, tokens: usize) {
        self.external_tokens.store(tokens, Ordering::Release);
    }

    /// Park the attached session, unless cleanup already claimed this
    /// lifecycle (a release that fired before attach completed). Returns
    /// [`ParkOutcome::Refused`] WITHOUT storing in that case — the caller
    /// must close the session itself, since the release's `take_session`
    /// found nothing to close. The claimed-check happens under the slot lock
    /// so it cannot interleave with a concurrent release's claim-then-take.
    ///
    /// Storing the session, taking the pre-park output buffer, and publishing
    /// it all happen under ONE guard — see [`Self::session`] for the two
    /// races this closes. The session ops are sync bounded channel sends, so
    /// holding a parking-lot guard across them is bounded (same discipline as
    /// the legacy attach drain).
    pub(crate) fn park_session(&self, session: Arc<dyn Session>) -> ParkOutcome {
        let mut slot = self.session.lock();
        if self.cleanup_claimed.load(Ordering::Acquire) {
            return ParkOutcome::Refused;
        }
        let buffered = std::mem::take(&mut slot.buffered);
        slot.session = Some(Arc::clone(&session));
        if buffered.is_empty() {
            return ParkOutcome::Parked;
        }
        match publish_output(session.as_ref(), buffered)
            .context("draining buffered prefill output at park")
        {
            Ok(()) => ParkOutcome::Parked,
            Err(e) => ParkOutcome::PublishFailed(e),
        }
    }

    /// Publish computed-output blocks into the parked session, buffering them
    /// when the attach has not landed yet. The single slot guard spans the
    /// whole buffer-or-publish decision AND the dispatch. A closed slot (any
    /// teardown or the finalize already took the session) drops the blocks —
    /// the decode peer is no longer listening.
    pub(crate) fn commit_output(&self, blocks: Vec<ImmutableBlock<G2>>) {
        if blocks.is_empty() {
            return;
        }
        let mut slot = self.session.lock();
        if slot.output_closed {
            tracing::debug!(
                count = blocks.len(),
                "prefill output after session teardown; dropping blocks"
            );
            return;
        }
        match &slot.session {
            Some(session) => {
                if let Err(e) = publish_output(session.as_ref(), blocks) {
                    // The session is closing under us (e.g. the decode peer
                    // died) — the pipeline/teardown paths own surfacing that;
                    // output publishing is best-effort past this point.
                    tracing::warn!(error = %e, "prefill output publish failed; dropping blocks");
                }
            }
            None => slot.buffered.extend(blocks),
        }
    }

    /// Take the parked session, leaving the slot empty AND closed to further
    /// output. Teardown closes/finalizes it exactly once; a second take
    /// yields `None`.
    pub(crate) fn take_session(&self) -> Option<Arc<dyn Session>> {
        let mut slot = self.session.lock();
        slot.output_closed = true;
        slot.session.take()
    }

    /// Whether a session is currently parked (read-only).
    pub(crate) fn has_session(&self) -> bool {
        self.session.lock().session.is_some()
    }

    /// Append a pulled-and-registered G2 pin tagged with its absolute
    /// expected-hash index.
    pub(crate) fn push_registered(&self, index: usize, block: ImmutableBlock<G2>) {
        self.registered_g2.lock().push((index, block));
    }

    /// The external-count SUFFIX of the registered G2 pins in absolute
    /// position order (clones the pins — they stay parked here too). `None`
    /// if fewer than `external_blocks` have registered, which the kick treats
    /// as a failure rather than a panic.
    pub(crate) fn registered_suffix(
        &self,
        external_blocks: usize,
    ) -> Option<Vec<ImmutableBlock<G2>>> {
        let registered = self.registered_g2.lock();
        if external_blocks > registered.len() {
            return None;
        }
        let mut by_position: Vec<(usize, ImmutableBlock<G2>)> =
            registered.iter().map(|(i, b)| (*i, b.clone())).collect();
        by_position.sort_by_key(|(i, _)| *i);
        let suffix_start = by_position.len() - external_blocks;
        Some(
            by_position
                .into_iter()
                .skip(suffix_start)
                .map(|(_, b)| b)
                .collect(),
        )
    }

    /// Pipeline half of the two-phase kick handshake: mark every expected pull
    /// registered. The pipeline calls this BEFORE its `take_pending_kick`; the
    /// USAA path latches its kick BEFORE loading this flag — so whichever
    /// ordering the two race into, at least one side observes the other and
    /// exactly one kick fires (the `onboarding_scheduled` CAS backstops).
    pub(crate) fn mark_pulls_complete(&self) {
        self.pulls_complete.store(true, Ordering::Release);
    }

    pub(crate) fn pulls_complete(&self) -> bool {
        self.pulls_complete.load(Ordering::Acquire)
    }

    /// USAA half of the gate: latch the kick — unless a pipeline failure
    /// already stashed, in which case the kick is refused and the stashed
    /// reason returned for the immediate-Failed replay. Atomic with the
    /// failure stash (one mutex), so the latch and a concurrent
    /// [`Self::fail_pipeline`] can never both miss each other.
    pub(crate) fn latch_kick(&self, kick: PrefillKick) -> Result<(), String> {
        let mut gate = self.gate.lock();
        if let Some(reason) = &gate.pending_failure {
            return Err(reason.clone());
        }
        gate.pending_kick = Some(kick);
        Ok(())
    }

    /// Consume the latched kick, if any. Exactly-once across the pipeline
    /// tail, the USAA pulls-already-complete path, and the failure path.
    pub(crate) fn take_pending_kick(&self) -> Option<PrefillKick> {
        self.gate.lock().pending_kick.take()
    }

    /// Failure half of the gate: a latched kick means USAA already armed the
    /// load action — return it so the caller resolves that action Failed
    /// (post-USAA semantics). No kick means pre-USAA: stash the reason
    /// (first failure wins) for `prefill_onboard_by_id` to replay and return `None`.
    pub(crate) fn fail_pipeline(&self, reason: String) -> Option<PrefillKick> {
        let mut gate = self.gate.lock();
        if let Some(kick) = gate.pending_kick.take() {
            return Some(kick);
        }
        gate.pending_failure.get_or_insert(reason);
        None
    }

    /// Stash a failure reason without touching the kick (first failure wins).
    pub(crate) fn stash_failure(&self, reason: String) {
        self.gate.lock().pending_failure.get_or_insert(reason);
    }

    /// The stashed failure, if any.
    pub(crate) fn pending_failure(&self) -> Option<String> {
        self.gate.lock().pending_failure.clone()
    }

    /// Claim the right to run the session close/finalize for this lifecycle.
    /// Returns `true` for the single CAS winner; every later caller loses.
    pub(crate) fn claim_cleanup(&self) -> bool {
        self.cleanup_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }

    /// CAS backstop for the single G2→G1 kick: `true` for the first caller.
    pub(crate) fn claim_kick(&self) -> bool {
        !self.onboarding_scheduled.swap(true, Ordering::AcqRel)
    }

    /// Bind the zero-external fall-through's internal local search to this
    /// lifecycle (see [`Self::local_search`]).
    pub(crate) fn bind_local_search(&self, search_id: SearchId) {
        *self.local_search.lock() = Some(search_id);
    }

    /// The bound internal local search, if any (the zero-stored onboard
    /// delegation routes through it).
    pub(crate) fn local_search_id(&self) -> Option<SearchId> {
        *self.local_search.lock()
    }

    /// Take the bound internal local search for release. Take-once, so the
    /// teardown paths cannot double-release it.
    pub(crate) fn take_local_search(&self) -> Option<SearchId> {
        self.local_search.lock().take()
    }

    /// Claim the single onboard for this accept generation: `true` for the
    /// first caller; every re-entrant onboard loses (see [`Self::usaa_claimed`]).
    pub(crate) fn claim_usaa(&self) -> bool {
        !self.usaa_claimed.swap(true, Ordering::AcqRel)
    }
}

/// Publish one batch of computed-output blocks: commit the hashes, then make
/// the pins available. The session derives each peer block id from
/// `block.block_id()` internally — the publisher only commits and makes
/// available.
fn publish_output(session: &dyn Session, blocks: Vec<ImmutableBlock<G2>>) -> Result<()> {
    let hashes: Vec<SequenceHash> = blocks.iter().map(|b| b.sequence_hash()).collect();
    session.commit(hashes)?;
    session.make_available(blocks)?;
    Ok(())
}

/// Engine-owned container of per-request prefill CD state. Mirrors
/// [`super::state::CdRequests`]'s atomic insert latch and identity-checked
/// release, minus the budget coupling (no decode-side reservation exists on
/// the prefill side).
pub(crate) struct PrefillRequests {
    inner: DashMap<String, Arc<PrefillRequestState>>,
}

impl Default for PrefillRequests {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefillRequests {
    pub(crate) fn new() -> Self {
        Self {
            inner: DashMap::new(),
        }
    }

    /// Track `state` under `request_id`. Bails when the rid is already tracked
    /// (the accept latch — callers treat the bail as an idempotent re-poll).
    /// Atomic check-and-set: the occupied branch holds the shard lock, so a
    /// concurrent vacant insert of the same rid cannot slip in between.
    pub(crate) fn insert(
        &self,
        request_id: String,
        state: Arc<PrefillRequestState>,
    ) -> Result<(), PrefillInsertError> {
        use dashmap::mapref::entry::Entry;
        match self.inner.entry(request_id) {
            Entry::Occupied(occupied) => Err(PrefillInsertError::AlreadyTracked {
                request_id: occupied.key().clone(),
            }),
            Entry::Vacant(vacant) => {
                vacant.insert(state);
                Ok(())
            }
        }
    }

    /// Clone out the tracked state, if any. The returned `Arc` is the identity
    /// a later [`Self::release_if_matches`] compares against to stay bound to
    /// one specific lifecycle.
    pub(crate) fn get(&self, request_id: &str) -> Option<Arc<PrefillRequestState>> {
        self.inner.get(request_id).map(|e| Arc::clone(e.value()))
    }

    /// Identity-checked release: remove `request_id` ONLY when the currently
    /// tracked `Arc` is pointer-equal to `expected` — the cross-lifecycle
    /// stale-release guard. A stale teardown parked from a PRIOR generation of
    /// the SAME rid (evict + re-accept installs a fresh state under the same
    /// key) no-ops instead of wiping the fresh lifecycle. Idempotent: a second
    /// release, or a release of an absent rid, returns `false`.
    pub(crate) fn release_if_matches(
        &self,
        request_id: &str,
        expected: &Arc<PrefillRequestState>,
    ) -> bool {
        self.inner
            .remove_if(request_id, |_, v| Arc::ptr_eq(expected, v))
            .is_some()
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

/// Failure modes for [`PrefillRequests::insert`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum PrefillInsertError {
    /// `request_id` is already tracked — the accept latch; the caller answers
    /// the poll from the existing lifecycle instead.
    #[error("prefill CD request already tracked: request_id={request_id}")]
    AlreadyTracked { request_id: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> PrefillRequestState {
        PrefillRequestState::new(
            AcceptId::new(),
            Vec::new(),
            Arc::new(AtomicUsize::new(0)),
            uuid::Uuid::new_v4(),
        )
    }

    /// Two lifecycles of the same shape carry distinct session ids; the accessor
    /// reports the originating session so the accept path can detect a
    /// recompute-rescheduled re-dispatch.
    #[test]
    fn session_id_is_stored_per_lifecycle() {
        let s1 = uuid::Uuid::new_v4();
        let s2 = uuid::Uuid::new_v4();
        let a = PrefillRequestState::new(
            AcceptId::new(),
            Vec::new(),
            Arc::new(AtomicUsize::new(0)),
            s1,
        );
        let b = PrefillRequestState::new(
            AcceptId::new(),
            Vec::new(),
            Arc::new(AtomicUsize::new(0)),
            s2,
        );
        assert_eq!(a.session_id(), s1);
        assert_eq!(b.session_id(), s2);
        assert_ne!(a.session_id(), b.session_id());
    }

    fn kick() -> PrefillKick {
        PrefillKick {
            action_id: ActionId::new(),
            external_g1: vec![50, 51],
        }
    }

    /// The gate's two sides share ONE mutex: a failure that lands BEFORE the
    /// USAA latch is observed by the latch (refused with the stashed reason),
    /// and a latch that lands BEFORE the failure is observed by the failure
    /// (the kick is surfaced for the Failed terminal). Two independent locks
    /// would admit an interleaving where both sides miss each other.
    #[test]
    fn gate_failure_then_latch_refuses_with_reason() {
        let state = state();
        assert!(
            state.fail_pipeline("boom".to_string()).is_none(),
            "pre-USAA failure stashes"
        );
        assert_eq!(state.latch_kick(kick()), Err("boom".to_string()));
        assert!(
            state.take_pending_kick().is_none(),
            "refused latch must not park a kick"
        );
    }

    #[test]
    fn gate_latch_then_failure_surfaces_the_kick() {
        let state = state();
        assert!(state.latch_kick(kick()).is_ok());
        let surfaced = state
            .fail_pipeline("boom".to_string())
            .expect("post-USAA failure takes the latched kick");
        assert_eq!(surfaced.external_g1, vec![50, 51]);
        assert!(state.take_pending_kick().is_none(), "kick consumed once");
    }

    #[test]
    fn first_failure_reason_wins() {
        let state = state();
        assert!(state.fail_pipeline("first".to_string()).is_none());
        assert!(state.fail_pipeline("second".to_string()).is_none());
        assert_eq!(state.pending_failure(), Some("first".to_string()));
    }

    #[test]
    fn claim_cleanup_single_winner() {
        let state = state();
        assert!(state.claim_cleanup());
        assert!(!state.claim_cleanup());
    }

    #[test]
    fn claim_kick_single_winner() {
        let state = state();
        assert!(state.claim_kick());
        assert!(!state.claim_kick());
    }

    #[test]
    fn claim_usaa_single_winner() {
        let state = state();
        assert!(state.claim_usaa());
        assert!(!state.claim_usaa(), "a re-entrant onboard loses the claim");
    }

    #[test]
    fn local_search_binding_is_take_once() {
        let state = state();
        assert!(state.local_search_id().is_none());
        assert!(state.take_local_search().is_none());

        let sid = SearchId::new();
        state.bind_local_search(sid);
        assert_eq!(state.local_search_id(), Some(sid));

        assert_eq!(state.take_local_search(), Some(sid));
        assert!(
            state.take_local_search().is_none(),
            "take-once: a second teardown path finds nothing to release"
        );
        assert!(state.local_search_id().is_none());
    }

    #[test]
    fn park_session_refused_after_cleanup_claimed() {
        use crate::p2p::session::{MockSessionFactory, SessionFactory};
        let factory = MockSessionFactory::new();
        let session = factory.open(uuid::Uuid::new_v4()).unwrap();

        let state = state();
        assert!(state.claim_cleanup(), "release claims before attach lands");
        assert!(
            matches!(state.park_session(session), ParkOutcome::Refused),
            "park after claim must refuse so the pipeline closes the session itself"
        );
        assert!(state.take_session().is_none(), "nothing was parked");
    }

    #[test]
    fn insert_latch_preserves_original() {
        let reqs = PrefillRequests::new();
        let original = Arc::new(state());
        reqs.insert("r1".to_string(), Arc::clone(&original))
            .unwrap();

        let err = reqs
            .insert("r1".to_string(), Arc::new(state()))
            .unwrap_err();
        assert_eq!(
            err,
            PrefillInsertError::AlreadyTracked {
                request_id: "r1".to_string()
            }
        );
        let held = reqs.get("r1").expect("original entry still present");
        assert!(Arc::ptr_eq(&held, &original));
        assert_eq!(reqs.len(), 1);
    }

    /// Same-key-different-Arc: a stale generation-1 teardown must not remove
    /// the freshly re-accepted generation-2 state under the same rid.
    #[test]
    fn release_if_matches_stale_arc_does_not_remove() {
        let reqs = PrefillRequests::new();
        let first = Arc::new(state());
        reqs.insert("r1".to_string(), Arc::clone(&first)).unwrap();
        assert!(reqs.release_if_matches("r1", &first));
        assert!(reqs.is_empty());

        let second = Arc::new(state());
        reqs.insert("r1".to_string(), Arc::clone(&second)).unwrap();

        assert!(
            !reqs.release_if_matches("r1", &first),
            "stale release must not wipe the fresh lifecycle"
        );
        let held = reqs.get("r1").expect("fresh lifecycle survives");
        assert!(Arc::ptr_eq(&held, &second));

        assert!(reqs.release_if_matches("r1", &second));
        assert!(reqs.is_empty());
    }
}
