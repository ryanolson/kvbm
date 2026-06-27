// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opaque, leader-local, poll-only handles + the consume-once offload drain.
//!
//! Each handle is a thin wrapper over an opaque id ([`super::protocol::SearchId`]
//! / [`super::protocol::ActionId`]) plus a `Weak<dyn LeaderEngine>` back-ref.
//! They hold no tokio/`watch` type and are deliberately **not** `Serialize` —
//! they never cross the wire, which is the property that keeps the engine free
//! to become an out-of-process peer later (only the trait *arguments and
//! returns* are wire types; the handles stay leader-local).
//!
//! ## Honest "engine-minted" convention (not a type seal)
//!
//! The engine impl lives in a sibling crate ([`kvbm-engine`]), so the
//! constructors below must be `pub` — the connector crate *could* call them
//! too. There is therefore **no** type-system unforgeability guarantee;
//! "engine-minted" is a curated-surface convention: the minting constructors are
//! `#[doc(hidden)]` and the connector never imports them (they are kept out of
//! the connector prelude, reachable only via the explicit
//! `kvbm_protocols::connector::…` path the sibling engine uses), so minting
//! one outside the engine is a deliberate, reviewable act. The load-bearing
//! safety property is **RAII drop**, not unforgeability.
//!
//! [`kvbm-engine`]: https://docs.rs/kvbm-engine

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::actions::{LoadOutcome, SaveOutcome};
use super::engine::LeaderEngine;
use super::protocol::RequestId;
use super::protocol::{AcceptId, ActionFailure, ActionId, ActionStatus, SearchId};

/// ONBOARD handle. Leader-local: an [`ActionId`] + a `Weak<dyn LeaderEngine>`
/// back-ref (for the RAII drop) + a shared completion cell the engine driver
/// advances. The engine holds a `Weak` to the same cell in its by-id map (for
/// the `poll_action` path), while the handle owns the **strong** ref — so the
/// cell frees by RAII on handle drop and the engine never pins completion state
/// alive (no map leak regardless of terminal category).
///
/// Leader-local: the cell carries a non-`Serialize` [`ActionStatus`], so the
/// handle is deliberately **not** `Serialize` and never crosses the wire:
///
/// ```compile_fail
/// fn needs_ser<T: serde::Serialize>() {}
/// needs_ser::<kvbm_protocols::connector::OnboardHandle>();
/// ```
///
/// RAII `Drop` is the **action analogue of the lifecycle-handle release**: it
/// fires [`LeaderEngine::release_action`] to prune the engine's per-action tracking
/// (the by-id action map + the `request_id → action_ids` index) so a terminal
/// onboard's bookkeeping frees on handle drop rather than leaking a key. Drop
/// does **not** cancel a healthy in-flight transfer (the engine drives it to
/// completion regardless; eviction *drains* via `evict`, not via handle drop).
pub struct OnboardHandle {
    id: ActionId,
    engine: Weak<dyn LeaderEngine>,
    /// Shared with the engine's by-id map (the engine holds a `Weak`); the
    /// engine writes the terminal, the handle reads it. No public mutator.
    status: Arc<Mutex<ActionStatus>>,
    /// The G1 dest block ids this load fills. [`LoadOutcome`] has no id-less
    /// failure (vLLM invalidates by block id), so `outcome()` uses this set to
    /// project an unresolved `Failed(AllBlocks)` cell onto the full dest set.
    dest_block_ids: Vec<usize>,
}

impl OnboardHandle {
    /// Engine-minted (honest convention — see the module docs). `status` is the
    /// cell the engine retains a `Weak` of and writes the terminal into; `engine`
    /// is the `Weak` back-ref the drop upgrades to fire `release_action`;
    /// `dest_block_ids` is the load's G1 dest set, kept so `outcome()` can name
    /// the failed blocks even for a total failure.
    #[doc(hidden)]
    pub fn new(
        id: ActionId,
        engine: Weak<dyn LeaderEngine>,
        status: Arc<Mutex<ActionStatus>>,
        dest_block_ids: Vec<usize>,
    ) -> Self {
        Self {
            id,
            engine,
            status,
            dest_block_ids,
        }
    }

    /// The opaque action id.
    pub fn id(&self) -> &ActionId {
        &self.id
    }

    /// `true` once the onboard reached a terminal state (a local cell read; no
    /// engine round-trip).
    pub fn is_complete(&self) -> bool {
        !matches!(
            *self.status.lock().expect("action-status mutex poisoned"),
            ActionStatus::Pending
        )
    }

    /// The terminal load outcome, or `None` while still pending. A load failure
    /// always names its blocks: a well-behaved engine writes the cell already
    /// resolved to concrete dest ids, and an unresolved `Failed(AllBlocks)`
    /// projects onto the handle's full dest set — never an empty failed set
    /// (which would let vLLM finish the recv with nothing invalidated).
    pub fn outcome(&self) -> Option<LoadOutcome> {
        match &*self.status.lock().expect("action-status mutex poisoned") {
            ActionStatus::Pending => None,
            ActionStatus::Complete => Some(LoadOutcome::Done),
            ActionStatus::Failed(ActionFailure::AllBlocks) => Some(LoadOutcome::FailedPartial {
                block_ids: self.dest_block_ids.clone(),
            }),
            ActionStatus::Failed(ActionFailure::Partial { block_ids }) => {
                Some(LoadOutcome::FailedPartial {
                    block_ids: block_ids.clone(),
                })
            }
        }
    }
}

impl Drop for OnboardHandle {
    /// RAII prune of the engine's per-action tracking. Best-effort: if the
    /// engine is already gone, the `Weak` upgrade fails and the prune is
    /// skipped. MUST NOT abort an in-flight transfer — the engine drives it to
    /// completion regardless.
    fn drop(&mut self) {
        if let Some(engine) = self.engine.upgrade() {
            engine.release_action(&self.id);
        }
    }
}

impl std::fmt::Debug for OnboardHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnboardHandle")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

/// THE one parked match-lifecycle handle. Pure opaque RAII + identity:
/// **no** status cell and **no** token accessor, because every fact the
/// connector needs rides [`super::protocol::FindBlocksOutcome`] (the connector
/// never reads engine state off a handle). Leader-local: a [`RequestId`] + a
/// private kind tag carrying the [`SearchId`] / [`AcceptId`] generation + a
/// `Weak<dyn LeaderEngine>` back-ref. Not `Serialize`; never crosses the wire.
///
/// Exactly ONE handle exists per latched lifecycle, so the kind-routed RAII
/// drop below fires at most once: a `Search` handle releases the match pin
/// (generation-bound by [`SearchId`]); a `Prefill` handle releases the accepted
/// prefill lifecycle (generation-guarded by [`AcceptId`], so a stale drop after
/// an evict + re-accept of the same `request_id` no-ops engine-side).
pub struct FindBlocksHandle {
    request_id: RequestId,
    kind: FindBlocksKind,
    engine: Weak<dyn LeaderEngine>,
}

/// Private latched-lifecycle discriminator embedded in a [`FindBlocksHandle`].
/// Never crosses the seam: the connector holds the handle opaquely and only the
/// engine routes [`super::engine::LeaderEngine::onboard_blocks`] and the RAII
/// drop off it.
enum FindBlocksKind {
    Search(SearchId),
    Prefill(AcceptId),
}

impl FindBlocksHandle {
    /// Engine-minted local-search handle (honest convention — see the module
    /// docs). Minted when a `find_blocks` call latches a fresh local search.
    #[doc(hidden)]
    pub fn search(request_id: RequestId, id: SearchId, engine: Weak<dyn LeaderEngine>) -> Self {
        Self {
            request_id,
            kind: FindBlocksKind::Search(id),
            engine,
        }
    }

    /// Engine-minted accepted-prefill handle (honest convention — see the module
    /// docs). Minted when a `find_blocks` call first accepts a dispatched remote
    /// prefill.
    #[doc(hidden)]
    pub fn prefill(
        request_id: RequestId,
        accept: AcceptId,
        engine: Weak<dyn LeaderEngine>,
    ) -> Self {
        Self {
            request_id,
            kind: FindBlocksKind::Prefill(accept),
            engine,
        }
    }

    /// The request this latched lifecycle belongs to.
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    /// Engine-side routing read: the local-search generation, or `None` for a
    /// prefill-kind handle. Doc-hidden — the connector never inspects the kind.
    #[doc(hidden)]
    pub fn search_id(&self) -> Option<SearchId> {
        match self.kind {
            FindBlocksKind::Search(id) => Some(id),
            FindBlocksKind::Prefill(_) => None,
        }
    }

    /// Engine-side routing read: the accepted-prefill generation, or `None` for a
    /// search-kind handle. Doc-hidden — the connector never inspects the kind.
    #[doc(hidden)]
    pub fn prefill_accept_id(&self) -> Option<AcceptId> {
        match self.kind {
            FindBlocksKind::Prefill(accept) => Some(accept),
            FindBlocksKind::Search(_) => None,
        }
    }
}

impl Drop for FindBlocksHandle {
    /// Kind-routed RAII, best-effort on a dead `Weak`:
    /// `Search(id)`   → [`LeaderEngine::release_search`] (generation-bound by `SearchId`);
    /// `Prefill(acc)` → [`LeaderEngine::release_prefill_session`] (generation-guarded by `AcceptId`).
    fn drop(&mut self) {
        let Some(engine) = self.engine.upgrade() else {
            return;
        };
        match self.kind {
            FindBlocksKind::Search(id) => engine.release_search(&id),
            FindBlocksKind::Prefill(accept) => {
                engine.release_prefill_session(&self.request_id, accept)
            }
        }
    }
}

impl std::fmt::Debug for FindBlocksHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("FindBlocksHandle");
        dbg.field("request_id", &self.request_id);
        match &self.kind {
            FindBlocksKind::Search(id) => {
                dbg.field("search", id);
            }
            FindBlocksKind::Prefill(accept) => {
                dbg.field("prefill", accept);
            }
        }
        dbg.finish_non_exhaustive()
    }
}

/// OFFLOAD handle. Same shape as [`OnboardHandle`] (an [`ActionId`] + a
/// `Weak<dyn LeaderEngine>` back-ref + a shared `Arc<Mutex<ActionStatus>>`
/// completion cell the engine holds a `Weak` of); its terminal projects to a
/// [`SaveOutcome`]. Per-action terminal flips **only** this handle — the
/// once-per-request `finished_sending` is emitted by consuming a
/// [`RequestOffloadDrain`] (REFACTOR.md §3/§4). RAII `Drop` fires
/// [`LeaderEngine::release_action`] to prune the engine's per-action tracking
/// (mirrors [`OnboardHandle`]); the cell itself frees by RAII on handle drop and
/// the offloaded blocks live on in the G2 pool.
pub struct OffloadHandle {
    id: ActionId,
    engine: Weak<dyn LeaderEngine>,
    /// Shared with the engine's by-id map (the engine holds a `Weak`); the
    /// engine writes the terminal, the handle reads it. No public mutator.
    status: Arc<Mutex<ActionStatus>>,
}

impl OffloadHandle {
    /// Engine-minted (honest convention — see the module docs). `status` is the
    /// cell the engine retains a `Weak` of and writes the terminal into; `engine`
    /// is the `Weak` back-ref the drop upgrades to fire `release_action`.
    #[doc(hidden)]
    pub fn new(
        id: ActionId,
        engine: Weak<dyn LeaderEngine>,
        status: Arc<Mutex<ActionStatus>>,
    ) -> Self {
        Self { id, engine, status }
    }

    /// The opaque action id.
    pub fn id(&self) -> &ActionId {
        &self.id
    }

    /// `true` once the offload reached a terminal state (a local cell read; no
    /// engine round-trip).
    pub fn is_complete(&self) -> bool {
        !matches!(
            *self.status.lock().expect("action-status mutex poisoned"),
            ActionStatus::Pending
        )
    }

    /// The terminal save outcome, or `None` while still pending.
    pub fn outcome(&self) -> Option<SaveOutcome> {
        match &*self.status.lock().expect("action-status mutex poisoned") {
            ActionStatus::Pending => None,
            ActionStatus::Complete => Some(SaveOutcome::Done),
            ActionStatus::Failed(ActionFailure::AllBlocks) => Some(SaveOutcome::FailedAllBlocks),
            ActionStatus::Failed(ActionFailure::Partial { block_ids }) => {
                Some(SaveOutcome::FailedPartial {
                    block_ids: block_ids.clone(),
                })
            }
        }
    }
}

impl Drop for OffloadHandle {
    /// RAII prune of the engine's per-action tracking (mirrors
    /// [`OnboardHandle::drop`]). Best-effort: a dead `Weak` skips the prune. MUST
    /// NOT abort an in-flight offload — the engine drives it to completion.
    fn drop(&mut self) {
        if let Some(engine) = self.engine.upgrade() {
            engine.release_action(&self.id);
        }
    }
}

impl std::fmt::Debug for OffloadHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OffloadHandle")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

/// EVICTION-FENCE handle. A leader-side **observational** completion cell over
/// one eviction fence: a shared `Arc<AtomicBool>` the engine flips exactly once,
/// Pending → Complete, when the last action armed at eviction time drains
/// (fences cannot fail, so a single boolean is the whole status space).
///
/// Unlike the action handles above, this handle has **no RAII drop behavior**:
/// it observes engine state, it does not own any. Dropping it merely abandons
/// the observation — the engine's fence still completes and the workers are
/// still notified through their own [`super::protocol::FenceToken`]s. It is the
/// leader's poll-only view of the same barrier the workers block on, so the
/// leader can gate its own bookkeeping (drain-holder release) on the SAME drain
/// the workers gate G1 reuse on.
///
/// Leader-local: not `Serialize`, never crosses the wire (the wire carries the
/// per-worker tokens instead).
pub struct FenceHandle {
    /// Shared with the engine's fence barrier; the barrier's drop writes `true`
    /// exactly once, the handle only reads. No public mutator.
    complete: Arc<AtomicBool>,
}

impl FenceHandle {
    /// Engine-minted (honest convention — see the module docs). `complete` is
    /// the cell the engine's fence barrier flips `true` when the last armed
    /// action drains.
    #[doc(hidden)]
    pub fn new(complete: Arc<AtomicBool>) -> Self {
        Self { complete }
    }

    /// `true` once the fence completed — every action captured at eviction time
    /// has drained (a local cell read; no engine round-trip).
    pub fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

impl std::fmt::Debug for FenceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FenceHandle")
            .field("complete", &self.is_complete())
            .finish()
    }
}

/// Consume-once token for the single `finished_sending(req)` emission.
///
/// The engine hands the leader one of these for a *finishing* request whose
/// offloads have started ([`super::engine::LeaderEngine::take_offload_drain`]);
/// the leader **consumes it by value** at `request_finished(Pending)` time.
/// `commit` ARMS the engine's emit-on-last-terminal: the engine fires the
/// single `finished_sending` when the last action still pending at commit
/// drains — immediately, if nothing is pending. The leader never has to wait
/// for the request's actions to reach terminal before committing.
/// This replaces a publicly re-callable `commit_request_offloads`: because
/// `commit` takes `self`, a second `commit` is a **compile** error, not a
/// runtime check — the load-bearing once-per-terminal vLLM gate is made
/// impossible to misuse. The emission closure is engine-supplied (it arms the
/// engine-side drain barrier that fires the injected worker sink's
/// `mark_save_finished`), so there is no re-callable method anywhere to invoke
/// twice.
///
/// A single `commit` resolves and compiles (this anchors the `compile_fail`
/// below to the *second* `commit` alone, not to a path/name error):
///
/// ```
/// let drain = kvbm_protocols::connector::RequestOffloadDrain::noop();
/// drain.commit();
/// ```
///
/// ```compile_fail
/// let drain = kvbm_protocols::connector::RequestOffloadDrain::noop();
/// drain.commit();
/// drain.commit(); // ERROR: use of moved value `drain`
/// ```
pub struct RequestOffloadDrain {
    emit: Box<dyn FnOnce() + Send>,
}

impl RequestOffloadDrain {
    /// Engine-minted (honest convention). `emit` arms the engine's
    /// emit-on-last-terminal for the single `finished_sending` (the engine
    /// fires `sink.mark_save_finished(req, outcome)` once the request's
    /// pending-at-commit actions drain; immediately if none are).
    #[doc(hidden)]
    pub fn new(emit: impl FnOnce() + Send + 'static) -> Self {
        Self {
            emit: Box::new(emit),
        }
    }

    /// A drain that emits nothing — for engines with no offloads to commit
    /// (e.g. [`super::noop::NoopBlockEngine`]) and for tests.
    #[doc(hidden)]
    pub fn noop() -> Self {
        Self::new(|| {})
    }

    /// Consume the drain, emitting the single `finished_sending`. Callable at
    /// most once (consume-self).
    pub fn commit(self) {
        (self.emit)();
    }
}

impl std::fmt::Debug for RequestOffloadDrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestOffloadDrain")
            .finish_non_exhaustive()
    }
}
