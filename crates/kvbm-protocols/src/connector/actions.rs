// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Terminal load/save outcomes carried across the engine ↔ worker seam.
//!
//! [`LoadOutcome`] / [`SaveOutcome`] are the results an async onboard (load) or
//! offload (save) reports for a request once it reaches a terminal state. They
//! are plain, transport-agnostic types: the engine publishes them to the worker
//! sink (see [`EngineWorkerSink`]) as IO completes.

use super::protocol::{FenceToken, RequestId};

/// Terminal outcome of an async load (onboard) for a request.
///
/// A load failure must NAME the failed G1 dest block ids: vLLM invalidates
/// failed loads by block id (the worker's `get_failed_onboarding`), so an
/// id-less "total failure" is unrepresentable by construction — a producer
/// resolves it to the load's concrete dest set (the engine driver is the last
/// layer that knows it). The failure still reports the request id so vLLM can
/// leave `WAITING_FOR_REMOTE_KVS`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadOutcome {
    Done,
    FailedPartial { block_ids: Vec<usize> },
}

/// Terminal outcome of an async save (offload) for a request.
///
/// Unlike [`LoadOutcome`], this keeps an id-less `FailedAllBlocks`: there is no
/// worker-side invalidation consumer for save failures (the KV stays resident
/// in GPU and vLLM does not gate on save), so nothing needs the block ids
/// resolved. `FailedPartial` names the G1 block ids that failed to offload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaveOutcome {
    Done,
    FailedAllBlocks,
    FailedPartial { block_ids: Vec<usize> },
}

/// Engine → worker completion push (worker-side sink only; there is no leader
/// reap-map). The engine publishes per-request status as IO completes; the
/// worker records it so vLLM's worker-side hooks return it on the next tick.
/// All methods are fire-and-forget from the engine's point of view, and are
/// invoked with no engine lock held.
pub trait EngineWorkerSink: Send + Sync + 'static {
    /// Onboard (load) reached a terminal state — success or a failure in
    /// [`LoadOutcome`]. Fired on the engine's aggregated onboard terminal.
    fn mark_load_finished(&self, req: &RequestId, outcome: LoadOutcome);

    /// Offload (save) reached its single per-request terminal — fired once when
    /// the leader consumes the request's
    /// [`super::handles::RequestOffloadDrain`], never by the per-action driver.
    fn mark_save_finished(&self, req: &RequestId, outcome: SaveOutcome);

    /// The eviction drain is complete for the worker named by the token's
    /// `rank`/`generation`. Wakes the matching `await_fence` waiter.
    fn mark_fence_complete(&self, token: FenceToken);
}

/// Worker → engine boundary held by the connector worker.
pub trait WorkerEngineDriver: Send + Sync + 'static {
    fn begin_forward_pass(&self, iteration: usize);
    /// Mints the offload precondition event for this iteration.
    fn finish_forward_pass(&self, iteration: usize);
    /// Block this worker's next forward pass until the drain for `token`
    /// completes.
    fn await_fence(&self, token: FenceToken);
    fn shutdown(&self);
}
