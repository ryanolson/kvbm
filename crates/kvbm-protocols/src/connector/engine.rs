// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The connector leader-facing engine trait, [`LeaderEngine`].
//!
//! The connector holds an `Arc<dyn LeaderEngine>` and is blind to whether the
//! impl is local (today) or an out-of-process proxy (later). Only the trait's
//! *arguments and returns* are wire types; the handles it returns are
//! leader-local (see [`super::handles`]).
//!
//! `find_blocks` / `onboard_blocks` / `offload` take `self: Arc<Self>` so the
//! handle they mint can embed a `Weak<dyn LeaderEngine>` for poll/RAII-release.
//! The remaining methods take `&self`.
//! Mixing the two receivers is object-safe — `Arc<dyn LeaderEngine>` works.

use std::sync::Arc;

use kvbm_common::LogicalResourceId;

use super::handles::{FindBlocksHandle, OffloadHandle, OnboardHandle, RequestOffloadDrain};
use super::protocol::{
    AcceptId, ActionId, ActionStatus, EvictionOutcome, FindBlocksOutcome, FindBlocksRequest,
    LeaderEngineError, SearchId,
};
use super::protocol::{BlockId, RequestId, ResourceOnboard, SequenceHash};

/// Leader-side block-engine contract. The connector drives unified match and
/// onboard, legacy or resource-explicit offload, eviction, and the request
/// offload drain. The engine routes everything else internally (local search
/// vs dispatched remote prefill, fresh vs refresh, window derivation, the
/// inflight-onboard deferral guard, and the selected resource pipeline).
/// The `poll_action` / `release_*` tail is **engine-internal** — completion and
/// RAII sources that the handles call on the connector's behalf, not part of
/// the connector's used surface (the connector only ever reads
/// [`FindBlocksOutcome`] and `handle.is_complete()` / `handle.outcome()`).
pub trait LeaderEngine: Send + Sync + 'static {
    /// OFFLOAD. Request-scoped (a cold no-match request still caches its novel
    /// blocks). The returned handle's per-action terminal flips **only** that
    /// handle — it does **not** push `mark_save_finished`. The one-per-request
    /// `finished_sending` is emitted by consuming the [`RequestOffloadDrain`]
    /// from [`Self::take_offload_drain`].
    fn offload(
        self: Arc<Self>,
        req: &RequestId,
        pairs: Vec<(SequenceHash, BlockId)>,
    ) -> Result<OffloadHandle, LeaderEngineError>;

    /// OFFLOAD one logical model resource.
    ///
    /// Legacy single-resource engines accept resource zero through
    /// [`Self::offload`] and fail closed for every other resource. Engines that
    /// own multiple logical resources override this method and route the
    /// action to the matching G1-to-G2 pipeline.
    fn offload_for_resource(
        self: Arc<Self>,
        resource: LogicalResourceId,
        req: &RequestId,
        pairs: Vec<(SequenceHash, BlockId)>,
    ) -> Result<OffloadHandle, LeaderEngineError> {
        if resource != LogicalResourceId::default() {
            return Err(LeaderEngineError::ResourceOffloadNotConfigured { resource });
        }
        self.offload(req, pairs)
    }

    /// EVICTION (non-terminal). DRAINS (does not cancel — submitted CUDA copies
    /// still complete) in-flight onboards for `req`, flags each
    /// cancelled-for-emission, arms offload drains, and mints one
    /// [`super::protocol::FenceToken`] per worker with outstanding work. The
    /// connector embeds the returned [`super::protocol::EvictionFence`] in
    /// metadata; the paired observational [`super::handles::FenceHandle`]
    /// (`Some` iff something was armed) lets the leader gate its own
    /// bookkeeping on the same drain. Not a `Finished` decision — that stays
    /// connector-local.
    fn evict(&self, req: &RequestId) -> EvictionOutcome;

    /// FIND BLOCKS — the unified, source-agnostic match-poll verb (the connector's
    /// only match poll). The engine routes internally: dispatched remote prefill
    /// (`req.transfer_params` carry `remote_prefill`) vs local search; fresh
    /// mint (`live` is `None`) vs refresh (`live` is `Some`); empty window; and
    /// the inflight-onboard deferral guard on every non-exempt arm (fresh AND
    /// refresh — only the dispatched-prefill path is exempt). Every fact the
    /// connector needs rides [`FindBlocksOutcome`] — the connector never reads
    /// handle state, so it cannot tell a local hit from an accepted prefill.
    /// Errors are loud (prefill misconfig / digest divergence /
    /// [`LeaderEngineError::FindBlocksDesync`]); local search failures collapse to
    /// a zero `Resolved` as today.
    fn find_blocks(
        self: Arc<Self>,
        req: &FindBlocksRequest,
        live: Option<&FindBlocksHandle>,
    ) -> Result<FindBlocksOutcome, LeaderEngineError>;

    /// ONBOARD BLOCKS — the unified onboard verb (the connector's only onboard).
    /// Routes by the handle's latched kind: a local hit copies the matched
    /// sources; an accepted prefill copies the pulled external suffix; a
    /// zero-stored prefill delegates to its internally-bound local search.
    /// `dest` is the FULL vLLM allocation `[computed prefix… | external…]`; the
    /// engine slices. Validates `num_external_tokens` against the engine's
    /// stored promise ([`LeaderEngineError::ExternalTokensMismatch`]).
    fn onboard_blocks(
        self: Arc<Self>,
        handle: &FindBlocksHandle,
        dest: &[BlockId],
        num_external_tokens: usize,
    ) -> Result<OnboardHandle, LeaderEngineError>;

    /// ONBOARD exact G2 blocks for multiple logical model resources under one
    /// request-scoped completion handle.
    fn onboard_resources(
        self: Arc<Self>,
        req: &RequestId,
        resources: Vec<ResourceOnboard>,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        let Some(first) = resources.first() else {
            return Err(LeaderEngineError::InvalidResourceOnboard {
                reason: "at least one resource is required".to_owned(),
            });
        };
        let _ = req;
        Err(LeaderEngineError::ResourceOnboardNotConfigured {
            resource: first.resource,
        })
    }

    /// Hand the leader the consume-once [`RequestOffloadDrain`] for a
    /// *finishing* request whose offloads have started. The leader commits it
    /// **once**, at `request_finished(Pending)` time; `commit` arms the
    /// engine's emit-on-last-terminal, which fires the single
    /// `finished_sending` when the request's last pending-at-commit action
    /// drains (immediately, if none are pending). Returns `None` when there is
    /// nothing to commit.
    fn take_offload_drain(&self, req: &RequestId) -> Option<RequestOffloadDrain>;

    /// ENGINE-INTERNAL completion source for `OnboardHandle`/`OffloadHandle`.
    /// In-process this is an in-process map read (DashMap-style); a future M3
    /// cross-process impl would read a leader-side completion cache fed by an
    /// engine→leader stream (batched per tick) — never a per-request RPC. The
    /// connector never calls this directly; the handles do.
    fn poll_action(&self, id: &ActionId) -> ActionStatus;

    /// ENGINE-INTERNAL RAII target for a search-kind [`FindBlocksHandle`] drop
    /// only. Best-effort; impls log internally. The connector never calls this
    /// directly; the handle does.
    fn release_search(&self, id: &SearchId);

    /// ENGINE-INTERNAL RAII target for [`OnboardHandle`]/[`OffloadHandle`] drop
    /// only — the action analogue of [`Self::release_search`]. Idempotent and
    /// best-effort: prunes the engine's per-action tracking (its by-id action
    /// map and the `request_id → action_ids` index) for `action`, so a terminal
    /// action's bookkeeping frees on handle drop rather than leaking a key.
    /// Releasing an unknown or already-released id is a no-op. The connector
    /// never calls this directly; the handles do.
    fn release_action(&self, action: &ActionId);

    /// ENGINE-INTERNAL RAII target for a prefill-kind [`FindBlocksHandle`] drop
    /// only — the prefill analogue of [`Self::release_search`]. Generation-
    /// guarded: a stale release from a prior lifecycle of the same request id
    /// (handle drop after evict + re-accept) no-ops on the [`AcceptId`]
    /// mismatch. Idempotent and best-effort; the connector never calls this
    /// directly; the handle does.
    fn release_prefill_session(&self, request_id: &RequestId, accept_id: AcceptId) {
        let _ = (request_id, accept_id);
    }
}
