// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process action completion map + the lock-release-then-notify terminal.
//!
//! Onboard/offload actions live in the engine's `actions` map keyed by
//! [`kvbm_protocols::connector::ActionId`]. The map holds only a **`Weak`**
//! to each action's completion cell — the **strong** cell is owned by the live
//! [`kvbm_protocols::connector::OnboardHandle`] /
//! [`kvbm_protocols::connector::OffloadHandle`], which reads its own cell for
//! `is_complete`/`outcome` (no engine round-trip). The engine's by-id
//! [`super::local::LocalConnectorEngine::poll_action`] upgrades the `Weak` for
//! the (M3) remote path. When a driver task reaches a terminal state it calls
//! [`LocalConnectorEngine::finish_load_action`], which writes the terminal into
//! the cell, **drops the map guard**, and only then fires the worker sink (the
//! REFACTOR.md §3 "no engine lock held" contract).
//!
//! Retention: there is **no** status-based prune. The action's `actions` entry
//! and its `by_request` link both live until the handle's RAII drop fires
//! [`LocalConnectorEngine::release_action`] (the action analogue of
//! `release_search`), which removes the entry from both maps — so a terminal
//! action's bookkeeping frees on handle drop rather than leaking a key, and the
//! strong completion cell (RAII-owned by the handle) frees with it regardless of
//! terminal category (success, failure, or evicted). `poll_action`'s dead-`Weak`
//! self-prune remains only as a backstop for the (M3) remote path, where no
//! local handle drop occurs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use kvbm_protocols::connector::{
    ActionFailure, ActionId, ActionStatus, EngineWorkerSink, FenceHandle, FenceToken,
};
use kvbm_protocols::connector::{LoadOutcome, RequestId, SaveOutcome};

use super::local::LocalConnectorEngine;

/// A reference-counted eviction-fence barrier shared by every action armed in a
/// single [`LocalConnectorEngine::evict`].
///
/// Each armed action's [`ActionRecord`] holds one `Arc` clone, and `evict` holds
/// one as an arming guard for the duration of its loop. The barrier's RAII `Drop`
/// fires `mark_fence_complete` for every token, so the fence completes **exactly
/// once, when the last clone drops** — i.e. when the last armed action's transfer
/// has drained (not the first). That is what lets a worker safely reuse the G1
/// blocks only after *every* in-flight-at-eviction action is done; completing on
/// the first drain would free blocks the engine is still transferring.
pub(super) struct FenceBarrier {
    tokens: Vec<FenceToken>,
    sink: Arc<dyn EngineWorkerSink>,
    /// Leader-side observational cell, shared with every
    /// [`FenceHandle`] minted off this barrier. The drop below performs its
    /// single Pending→Complete transition (fences cannot fail), so the leader
    /// observes completion on the SAME drain the workers gate G1 reuse on.
    leader_cell: Arc<AtomicBool>,
}

impl FenceBarrier {
    pub(super) fn new(tokens: Vec<FenceToken>, sink: Arc<dyn EngineWorkerSink>) -> Self {
        Self {
            tokens,
            sink,
            leader_cell: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The barrier's per-worker tokens — carried into the worker metadata by
    /// `evict` so each worker awaits its own token.
    pub(super) fn tokens(&self) -> &[FenceToken] {
        &self.tokens
    }

    /// Mint the leader's observational handle over this barrier's completion
    /// cell. Poll-only on the leader side: the cell flips exactly once, in the
    /// barrier's drop.
    pub(super) fn leader_handle(&self) -> FenceHandle {
        FenceHandle::new(Arc::clone(&self.leader_cell))
    }
}

impl Drop for FenceBarrier {
    fn drop(&mut self) {
        // No engine lock is held at any barrier-clone drop site (finish_*_action
        // drops the guard before dropping the fence; `evict` drops its guard at
        // function exit; `release_action` defers while a clone is live), so firing
        // the sink here honors the REFACTOR.md §3 "no engine lock held" contract.
        for token in &self.tokens {
            self.sink.mark_fence_complete(*token);
        }
        // Leader cell last: by the time the leader observes completion, every
        // worker notification has already been pushed.
        self.leader_cell.store(true, Ordering::Release);
    }
}

/// A reference-counted finished-request drain, armed by
/// [`LocalConnectorEngine::arm_drain_emission`] when the connector commits a
/// `RequestOffloadDrain` for a *finishing* request (D semantics: commit =
/// "engine, emit when drained", never "emit now").
///
/// Every action still pending at commit time holds one `Arc` clone (plus the
/// arming loop's guard clone), so the RAII `Drop` fires the request's single
/// `mark_save_finished` — the worker-side `finished_sending` — **exactly once,
/// when the last pending-at-commit action drains**. If nothing was pending,
/// the guard clone's drop at the end of the arming loop IS the emission
/// ("might already be done — doesn't matter"). vLLM frees the request's G1
/// blocks on `finished_sending`, so the emission must wait on pending LOADS
/// too (an in-flight onboard writes into those blocks), not just saves.
pub(super) struct DrainBarrier {
    request_id: RequestId,
    sink: Arc<dyn EngineWorkerSink>,
}

impl DrainBarrier {
    pub(super) fn new(request_id: RequestId, sink: Arc<dyn EngineWorkerSink>) -> Self {
        Self { request_id, sink }
    }
}

impl Drop for DrainBarrier {
    fn drop(&mut self) {
        // Same no-lock-held discipline as `FenceBarrier::drop` (all drop sites
        // release the per-action guard first). Failures collapse to `Done` —
        // there is no failed-offload wire path; per-action failures are logged
        // at their terminals.
        self.sink
            .mark_save_finished(&self.request_id, SaveOutcome::Done);
    }
}

/// One in-flight (or terminal-but-handle-alive) action's engine-side state.
pub(super) struct ActionRecord {
    /// The request this action belongs to — the `by_request` key, so
    /// [`LocalConnectorEngine::release_action`] can scrub the index given only an
    /// [`ActionId`] (the handle's RAII drop carries no `RequestId`).
    pub(super) request_id: RequestId,
    /// `Weak` to the completion cell owned by the live `OnboardHandle` /
    /// `OffloadHandle`. The engine writes the terminal through this `Weak`;
    /// `poll_action` reads it for the by-id path (and self-prunes a dead `Weak`
    /// as a backstop). The strong cell lives in the handle, so the map never pins
    /// completion state alive.
    pub(super) cell: Weak<Mutex<ActionStatus>>,
    /// A clone of the shared [`FenceBarrier`], set by [`LocalConnectorEngine::evict`]
    /// when this action is armed cancelled-for-emission. While held, the action's
    /// terminal fires no `mark_load_finished`/`mark_save_finished`; instead
    /// `finish_*_action` *takes* this clone and drops it, completing the shared
    /// fence only once it is the last armed action to drain. The terminal status is
    /// still written to the cell, so the by-id path stays correct.
    pub(super) fence: Option<Arc<FenceBarrier>>,
    /// A clone of the shared [`DrainBarrier`], set by
    /// [`LocalConnectorEngine::arm_drain_emission`] when the connector committed
    /// the finishing request's drain while this action was still pending.
    /// `finish_*_action` *takes* and drops it after its sink notify; the last
    /// armed action's drop fires the request's single `finished_sending`.
    /// Independent of `fence` — an action can be both (evicted, then the
    /// restored request finishes while the old drain is still in flight).
    pub(super) drain: Option<Arc<DrainBarrier>>,
    /// Set `true` if the handle's RAII drop fired [`LocalConnectorEngine::release_action`]
    /// while `fence` or `drain` was still armed (the driver had not reached terminal).
    /// Removal of the `actions` entry is then DEFERRED to the driver's terminal:
    /// dropping the record — and with it the live barrier clone(s) — now would
    /// complete the fence / fire the emission before the transfer drained. The
    /// terminal removes the record once it observes this flag.
    pub(super) dropped_by_handle: bool,
}

impl ActionRecord {
    /// A freshly-minted in-flight action over the handle's completion cell.
    pub(super) fn new(request_id: RequestId, cell: Weak<Mutex<ActionStatus>>) -> Self {
        Self {
            request_id,
            cell,
            fence: None,
            drain: None,
            dropped_by_handle: false,
        }
    }
}

/// Project a terminal [`ActionStatus`] onto the load-completion the worker sink
/// expects. `Pending` cannot legitimately reach here; it degrades to `Done`.
fn load_outcome_of(status: &ActionStatus) -> LoadOutcome {
    match status {
        ActionStatus::Pending | ActionStatus::Complete => LoadOutcome::Done,
        ActionStatus::Failed(ActionFailure::AllBlocks) => {
            // Unreachable for loads: `finish_load_action` resolves a total
            // failure to the concrete dest set before the cell write and this
            // projection ([`LoadOutcome`] has no id-less failure to map to).
            debug_assert!(
                false,
                "total load failure must be resolved to dest ids before projection"
            );
            LoadOutcome::FailedPartial {
                block_ids: Vec::new(),
            }
        }
        ActionStatus::Failed(ActionFailure::Partial { block_ids }) => LoadOutcome::FailedPartial {
            block_ids: block_ids.clone(),
        },
    }
}

impl LocalConnectorEngine {
    /// Terminal for a load (onboard) action.
    ///
    /// Writes the terminal status into the handle's completion cell (via the
    /// map's `Weak`) and reads any cancel-for-emission fences under the map
    /// guard, **drops the guard**, and only then notifies the worker sink —
    /// either the per-eviction `mark_fence_complete` tokens or the single
    /// `mark_load_finished`.
    ///
    /// No prune here: the action's `actions` entry and its `by_request` link both
    /// live until the handle's RAII drop fires
    /// [`LocalConnectorEngine::release_action`] (the action analogue of
    /// `release_search`). A dropped handle (dead `Weak`) has no observer — the
    /// cell write is skipped but the worker is still notified.
    ///
    /// `dest_ids` is the load's G1 dest set, demanded by the signature because
    /// the terminal needs it to resolve a total failure: vLLM invalidates
    /// failed loads by block id (the worker's `get_failed_onboarding`), and
    /// [`LoadOutcome`] has no id-less failure — an unresolved `AllBlocks` would
    /// otherwise cross the wire as an EMPTY failed set, finishing the request's
    /// recv with nothing invalidated.
    pub(super) fn finish_load_action(
        &self,
        action_id: ActionId,
        request_id: &RequestId,
        outcome: ActionStatus,
        dest_ids: Vec<usize>,
    ) {
        // The in-flight onboard deferral guard is NOT cleared here: it is
        // lifecycle-keyed and clears at the lifecycle's RAII release
        // (`release_search` / `release_prefill_session`), which the connector
        // fires only once the loaded blocks are connector-visible — see
        // `super::inflight`.

        // Resolve a total-failure terminal to the CONCRETE dest ids before the
        // cell write, so every read path (handle outcome, by-id poll, sink
        // projection) agrees on the named blocks.
        let outcome = match outcome {
            ActionStatus::Failed(ActionFailure::AllBlocks) => {
                ActionStatus::Failed(ActionFailure::Partial {
                    block_ids: dest_ids,
                })
            }
            other => other,
        };

        // Under the per-action guard: write the terminal into the handle's cell,
        // TAKE any armed fence/drain clones, and learn whether the handle already
        // dropped (so this terminal must remove the record). Release the guard
        // before any sink call or barrier-clone drop.
        let (fence, drain, remove_now) = {
            let mut guard = self.actions.get_mut(&action_id);
            match guard.as_deref_mut() {
                Some(record) => {
                    if let Some(cell) = record.cell.upgrade() {
                        *cell.lock().expect("action-status mutex poisoned") = outcome.clone();
                    }
                    (
                        record.fence.take(),
                        record.drain.take(),
                        record.dropped_by_handle,
                    )
                }
                None => (None, None, false),
            }
        };

        // Notify with NO engine lock held. A fenced (cancel-for-emission) action
        // fires no `mark_load_finished`; dropping its fence clone below completes
        // the shared barrier iff this is the last armed action to drain. A
        // DRAIN-armed load is suppressed too: its request is finishing, and
        // vLLM's `_free_blocks` deletes the request on the first finished-set
        // hit — surfacing `finished_recving` alongside the request's eventual
        // `finished_sending` would assert in the scheduler. The load's
        // completion folds into the drain emission instead.
        if fence.is_none() && drain.is_none() {
            self.sink
                .mark_load_finished(request_id, load_outcome_of(&outcome));
        }
        drop(fence);
        drop(drain);

        // The conditional-disagg load terminal (budget release + session
        // finalize/close) is NOT fired here: it must run against the ORIGINATING
        // lifecycle's `Arc<CdRequestState>`, which only the CD producers (the
        // onboard driver task, `mint_failed_onboard`) hold. They call
        // `CdRuntime::complete_load(rid, &state, outcome)` AFTER this returns, so
        // a stale terminal racing an evict + re-latch of the same rid can never
        // tear down the fresh lifecycle. `finish_load_action` stays CD-free.

        // If the handle already dropped, this terminal is the record's last owner.
        if remove_now {
            self.remove_action_record(&action_id);
        }
    }

    /// Terminal for a save (offload) action.
    ///
    /// Writes the terminal status into the handle's completion cell (via the
    /// map's `Weak`) and reads any cancel-for-emission fences under the map
    /// guard, **drops the guard**, then notifies. Unlike [`Self::finish_load_action`],
    /// the *non-evicted* path fires **nothing** on the sink: the
    /// once-per-request `finished_sending` is emitted only by consuming the
    /// request's [`kvbm_protocols::connector::RequestOffloadDrain`] (the vLLM
    /// `finished_sending`-subset contract), never per offload action. The
    /// cancel-for-emission (eviction) path still fires `mark_fence_complete`,
    /// exactly as the load terminal does.
    ///
    /// No prune here (same as the load terminal): the `actions` entry and its
    /// `by_request` link live until the handle's RAII drop fires
    /// [`LocalConnectorEngine::release_action`].
    pub(super) fn finish_save_action(
        &self,
        action_id: ActionId,
        request_id: &RequestId,
        outcome: ActionStatus,
    ) {
        // Under the per-action guard: write the terminal into the handle's cell,
        // TAKE any armed fence/drain clones, and learn whether the handle already
        // dropped. Release the guard before any sink call or barrier-clone drop.
        let (fence, drain, remove_now) = {
            let mut guard = self.actions.get_mut(&action_id);
            match guard.as_deref_mut() {
                Some(record) => {
                    if let Some(cell) = record.cell.upgrade() {
                        *cell.lock().expect("action-status mutex poisoned") = outcome;
                    }
                    (
                        record.fence.take(),
                        record.drain.take(),
                        record.dropped_by_handle,
                    )
                }
                None => (None, None, false),
            }
        };

        // Notify with NO engine lock held. The non-fenced offload terminal fires
        // NOTHING on the sink itself: the once-per-request `finished_sending` is
        // emitted only via the request's committed drain — dropping the `drain`
        // clone below fires it iff this is the last pending-at-commit action to
        // drain. A fenced action instead completes the shared fence barrier when
        // its clone drops.
        if fence.is_none() {
            tracing::trace!(
                %request_id,
                "offload action terminal (save_finished is drain-driven, not fired per-action)"
            );
        }
        drop(fence);
        drop(drain);

        // If the handle already dropped, this terminal is the record's last owner.
        if remove_now {
            self.remove_action_record(&action_id);
        }
    }

    /// Arm the finished-request drain emission (the `RequestOffloadDrain::commit`
    /// target — D semantics). Mints one shared [`DrainBarrier`] and, for every
    /// action of `req` still `Pending`, stores a clone under the SAME per-action
    /// `get_mut` guard that `finish_*_action` serializes on — so a terminal
    /// cannot land between the pending check and the arm. The local guard clone
    /// drops at return: if nothing was armed (every action already terminal, or
    /// none exist), that drop IS the immediate emission.
    pub(super) fn arm_drain_emission(&self, req: &RequestId) {
        let emission = Arc::new(DrainBarrier::new(req.clone(), self.sink.clone()));
        let action_ids: Vec<ActionId> = self
            .by_request
            .get(req)
            .map(|ids| ids.clone())
            .unwrap_or_default();
        for id in &action_ids {
            if let Some(mut record) = self.actions.get_mut(id) {
                let still_pending = record.cell.upgrade().is_some_and(|cell| {
                    matches!(
                        *cell.lock().expect("action-status mutex poisoned"),
                        ActionStatus::Pending
                    )
                });
                // `drain.is_none()` mirrors the evict arming guard: the drain is
                // consume-once so a second commit can't reach here for the same
                // registration, but a strict guard is cheap.
                if still_pending && record.drain.is_none() {
                    record.drain = Some(Arc::clone(&emission));
                }
            }
        }
        // `emission` (the arming-guard clone) drops here with no lock held.
    }

    /// Sever a finished action's `request_id → action_ids` link, dropping the
    /// per-request entry once it is empty.
    pub(super) fn untrack_action(&self, request_id: &RequestId, action_id: ActionId) {
        if let Some(mut ids) = self.by_request.get_mut(request_id) {
            ids.retain(|a| *a != action_id);
        }
        self.by_request
            .remove_if(request_id, |_, ids| ids.is_empty());
    }

    /// Remove an action's `actions` entry and scrub its `by_request` link. The
    /// single remover, called from the LATER of (handle drop via `release_action`,
    /// driver terminal via `finish_*_action` when `dropped_by_handle`), so a
    /// fence-armed action's record outlives a premature handle drop.
    pub(super) fn remove_action_record(&self, id: &ActionId) {
        if let Some((_id, record)) = self.actions.remove(id) {
            self.untrack_action(&record.request_id, *id);
        }
    }
}
