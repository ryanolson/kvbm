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
//! the cell and **drops the map guard** before any worker notification. A
//! physically draining bundle defers that notification to
//! [`LocalConnectorEngine::finish_physical_load_action`]; both phases preserve
//! the REFACTOR.md §3 "no engine lock held" contract.
//!
//! Retention follows handle and physical-work ownership. Handle drop calls
//! [`LocalConnectorEngine::release_action`]. Pending physical work keeps the
//! record until its driver terminal. That terminal removes a dropped record
//! and scrubs both indexes. `poll_action` never owns record removal.

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
    /// `poll_action` reads it for the by-id path. The strong cell lives in the
    /// handle, so the map never pins completion state alive.
    pub(super) cell: Weak<Mutex<ActionStatus>>,
    /// A clone of the shared [`FenceBarrier`], set by [`LocalConnectorEngine::evict`]
    /// when this action is armed cancelled-for-emission. While held, the action's
    /// terminal fires no `mark_load_finished`/`mark_save_finished`; instead the
    /// action's drain terminal (logical for ordinary actions, physical for a
    /// bundle onboard) *takes* this clone and drops it, completing the shared
    /// fence only once it is the last armed action to drain. The
    /// logical terminal status is still written to the cell promptly, so the
    /// by-id path stays correct.
    pub(super) fence: Option<Arc<FenceBarrier>>,
    /// A clone of the shared [`DrainBarrier`], set by
    /// [`LocalConnectorEngine::arm_drain_emission`] when the connector committed
    /// the finishing request's drain while this action was still pending.
    /// The action's drain terminal *takes* and drops it; the last armed action's
    /// drop fires the request's single `finished_sending`.
    /// Independent of `fence` — an action can be both (evicted, then the
    /// restored request finishes while the old drain is still in flight).
    pub(super) drain: Option<Arc<DrainBarrier>>,
    /// Set `true` if the handle's RAII drop fired [`LocalConnectorEngine::release_action`]
    /// while `fence`, `drain`, or physical work was still pending.
    /// Removal of the `actions` entry is then DEFERRED to the driver's terminal:
    /// dropping the record — and with it the live barrier clone(s) — now would
    /// complete the fence / fire the emission before the transfer drained. The
    /// terminal removes the record once it observes this flag.
    pub(super) dropped_by_handle: bool,
    /// Optional in-flight onboard generation cleared with this action record.
    /// While present, an early handle drop defers removal until terminal.
    pub(super) inflight: Option<super::inflight::InflightKey>,
    /// True while a bundle onboard owns launched transfer notifications.
    /// Logical completion can occur before this physical state clears.
    pub(super) physical_pending: bool,
    /// The action type and its terminal state.
    kind: ActionKind,
    /// Logical cancellation source for a physically draining bundle onboard.
    /// Cancelling never aborts submitted DMA; it only lets the action leave
    /// `Pending` while the physical task and its fence keep G1 quarantined.
    pub(super) cancel: Option<tokio_util::sync::CancellationToken>,
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
            inflight: None,
            physical_pending: false,
            kind: ActionKind::Load(LoadTerminalState::Pending),
            cancel: None,
        }
    }

    /// Create a save record before its physical work enters the buffer.
    pub(super) fn new_save(request_id: RequestId, cell: Weak<Mutex<ActionStatus>>) -> Self {
        Self {
            request_id,
            cell,
            fence: None,
            drain: None,
            dropped_by_handle: false,
            inflight: None,
            physical_pending: false,
            kind: ActionKind::Save(SaveTerminalState::Pending),
            cancel: None,
        }
    }

    pub(super) fn with_inflight(mut self, key: super::inflight::InflightKey) -> Self {
        self.inflight = Some(key);
        self
    }

    pub(super) fn with_physical_drain(
        mut self,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Self {
        assert!(
            matches!(self.kind, ActionKind::Load(_)),
            "physical load state requires a load action"
        );
        self.physical_pending = true;
        self.cancel = Some(cancel);
        self
    }

    /// Report physical work that can outlive the action handle.
    pub(super) fn has_pending_physical_work(&self) -> bool {
        self.physical_pending || matches!(self.kind, ActionKind::Save(SaveTerminalState::Pending))
    }

    /// Report work that must hold an eviction or request-drain barrier.
    pub(super) fn has_pending_work(&self) -> bool {
        self.cell.upgrade().is_some_and(|cell| {
            matches!(
                *cell.lock().expect("action-status mutex poisoned"),
                ActionStatus::Pending
            )
        }) || self.has_pending_physical_work()
    }

    /// Report whether handle drop must leave this record for its terminal.
    pub(super) fn must_retain_after_handle_drop(&self) -> bool {
        self.fence.is_some()
            || self.drain.is_some()
            || self.has_pending_physical_work()
            || (self.inflight.is_some() && self.has_pending_work())
    }

    /// Return the cancellation source for a physical load.
    pub(super) fn physical_load_cancel(&self) -> Option<tokio_util::sync::CancellationToken> {
        (self.physical_pending && matches!(self.kind, ActionKind::Load(_)))
            .then(|| self.cancel.clone())
            .flatten()
    }

    /// Settle the first save terminal and preserve all later terminals.
    fn settle_save_once(&mut self, outcome: ActionStatus) -> bool {
        let ActionKind::Save(terminal) = &mut self.kind else {
            debug_assert!(false, "save terminal requires a save action");
            return false;
        };
        if matches!(terminal, SaveTerminalState::Settled) {
            return false;
        }
        if let Some(cell) = self.cell.upgrade() {
            *cell.lock().expect("action-status mutex poisoned") = outcome;
        }
        *terminal = SaveTerminalState::Settled;
        true
    }
}

/// The action type owns only its valid terminal state.
enum ActionKind {
    Load(LoadTerminalState),
    Save(SaveTerminalState),
}

/// Two-phase load-terminal state owned by one action record.
enum LoadTerminalState {
    /// No logical load terminal has landed.
    Pending,
    /// The handle is terminal, but the worker must not reuse the named G1
    /// destinations until the physical phase drains.
    Deferred(LoadOutcome),
    /// Worker notification was emitted or deliberately suppressed by an
    /// eviction/request-drain barrier.
    Settled,
}

/// Save-terminal state owned by one action record.
enum SaveTerminalState {
    /// The physical save has not reported a terminal state.
    Pending,
    /// The first save terminal settled the cell and any barriers.
    Settled,
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
        ActionStatus::Failed(ActionFailure::Resource { block_ids, .. }) => {
            LoadOutcome::FailedPartial {
                block_ids: block_ids.clone().unwrap_or_default(),
            }
        }
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
    /// A live handle retains the record after this terminal. A prior handle
    /// drop defers removal to this terminal when physical work is pending.
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
            ActionStatus::Failed(ActionFailure::Resource {
                resource,
                block_ids: None,
            }) => ActionStatus::Failed(ActionFailure::Resource {
                resource,
                block_ids: Some(dest_ids),
            }),
            other => other,
        };

        // Under the per-action guard: write the terminal into the handle's cell,
        // TAKE any armed fence/drain clones, and learn whether the handle already
        // dropped (so this terminal must remove the record). Release the guard
        // before any sink call or barrier-clone drop.
        let (fence, drain, worker_terminal, remove_now) = {
            let mut guard = self.actions.get_mut(&action_id);
            match guard.as_deref_mut() {
                Some(record) => {
                    if !matches!(record.kind, ActionKind::Load(LoadTerminalState::Pending)) {
                        tracing::debug!(
                            ?action_id,
                            %request_id,
                            "ignoring duplicate load terminal"
                        );
                        return;
                    }
                    if let Some(cell) = record.cell.upgrade() {
                        *cell.lock().expect("action-status mutex poisoned") = outcome.clone();
                    }
                    let suppress_terminal = record.fence.is_some() || record.drain.is_some();
                    if record.physical_pending {
                        let terminal = if suppress_terminal {
                            LoadTerminalState::Settled
                        } else {
                            LoadTerminalState::Deferred(load_outcome_of(&outcome))
                        };
                        record.kind = ActionKind::Load(terminal);
                        (None, None, None, false)
                    } else {
                        record.kind = ActionKind::Load(LoadTerminalState::Settled);
                        (
                            record.fence.take(),
                            record.drain.take(),
                            (!suppress_terminal).then(|| load_outcome_of(&outcome)),
                            record.dropped_by_handle,
                        )
                    }
                }
                None => (None, None, Some(load_outcome_of(&outcome)), false),
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
        if let Some(worker_terminal) = worker_terminal {
            self.sink.mark_load_finished(request_id, worker_terminal);
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

    /// Mark only the physical-drain half of a bundle onboard complete.
    ///
    /// The logical terminal is deliberately owned by [`Self::finish_load_action`].
    /// This method never writes the handle cell. It releases eviction/request-
    /// drain barriers, publishes the observational physical fence, and emits
    /// an unfenced worker terminal that the logical phase deferred. Keeping
    /// the handle write in the logical phase preserves prompt failure while
    /// keeping failed destination ids quarantined until DMA has settled; the
    /// stored outcome also prevents a late completion from overwriting it.
    pub(super) fn finish_physical_load_action(
        &self,
        action_id: ActionId,
        physical_drain: &Arc<AtomicBool>,
    ) {
        let (request_id, worker_terminal, fence, drain, remove_now) = {
            let mut guard = self.actions.get_mut(&action_id);
            match guard.as_deref_mut() {
                Some(record) => {
                    record.physical_pending = false;
                    record.cancel = None;
                    let suppress_terminal = record.fence.is_some() || record.drain.is_some();
                    let prior_terminal = match &mut record.kind {
                        ActionKind::Load(terminal) => {
                            std::mem::replace(terminal, LoadTerminalState::Settled)
                        }
                        ActionKind::Save(_) => {
                            debug_assert!(false, "physical load terminal requires a load action");
                            return;
                        }
                    };
                    let worker_terminal = match prior_terminal {
                        LoadTerminalState::Deferred(outcome) if !suppress_terminal => Some(outcome),
                        LoadTerminalState::Pending => {
                            tracing::error!(
                                ?action_id,
                                "physical load settled before its logical terminal"
                            );
                            record.kind = ActionKind::Load(LoadTerminalState::Pending);
                            None
                        }
                        LoadTerminalState::Deferred(_) | LoadTerminalState::Settled => None,
                    };
                    (
                        Some(record.request_id.clone()),
                        worker_terminal,
                        record.fence.take(),
                        record.drain.take(),
                        record.dropped_by_handle,
                    )
                }
                None => (None, None, None, None, false),
            }
        };

        // Worker-side barriers first; once the leader-side physical fence is
        // observed complete, every other release signal for this drain has
        // already been emitted.
        drop(fence);
        drop(drain);
        physical_drain.store(true, Ordering::Release);
        if let (Some(request_id), Some(worker_terminal)) = (request_id, worker_terminal) {
            self.sink.mark_load_finished(&request_id, worker_terminal);
        }

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
    /// A live handle retains the record after this terminal. A prior handle
    /// drop makes this terminal remove the record and both index entries.
    pub(super) fn finish_save_action(
        &self,
        action_id: ActionId,
        request_id: &RequestId,
        outcome: ActionStatus,
    ) {
        // Under the per-action guard: write the terminal into the handle's cell,
        // TAKE any armed fence/drain clones, and learn whether the handle already
        // dropped. Release the guard before any sink call or barrier-clone drop.
        let settled = {
            let mut guard = self.actions.get_mut(&action_id);
            match guard.as_deref_mut() {
                Some(record) => {
                    if record.settle_save_once(outcome) {
                        Some((
                            record.fence.take(),
                            record.drain.take(),
                            record.dropped_by_handle,
                        ))
                    } else {
                        None
                    }
                }
                None => None,
            }
        };
        let Some((fence, drain, remove_now)) = settled else {
            tracing::debug!(
                ?action_id,
                %request_id,
                "ignoring missing or duplicate save terminal"
            );
            return;
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
    /// target — D semantics). Mints one shared [`DrainBarrier`] for each action
    /// with logical or physical work. It stores each clone under the same
    /// per-action guard that serializes terminal updates. The local guard clone
    /// drops at return. That drop emits immediately when no action was armed.
    pub(super) fn arm_drain_emission(&self, req: &RequestId) {
        let emission = Arc::new(DrainBarrier::new(req.clone(), self.sink.clone()));
        let action_ids: Vec<ActionId> = self
            .by_request
            .get(req)
            .map(|ids| ids.clone())
            .unwrap_or_default();
        for id in &action_ids {
            if let Some(mut record) = self.actions.get_mut(id) {
                // `drain.is_none()` mirrors the evict arming guard: the drain is
                // consume-once so a second commit can't reach here for the same
                // registration, but a strict guard is cheap.
                if record.has_pending_work() && record.drain.is_none() {
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
            if let Some(key) = record.inflight {
                self.inflight
                    .lock()
                    .expect("inflight-guard mutex poisoned")
                    .clear(&key);
            }
            self.untrack_action(&record.request_id, *id);
        }
    }
}
