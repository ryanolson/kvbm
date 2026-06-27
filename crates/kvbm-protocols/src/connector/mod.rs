// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine seam for the KVBM connector.
//!
//! This module carries only plain types, so the engine impl can be local today
//! and an RPC client tomorrow without the connector noticing.
//!
//! - [`LeaderEngine`] — the handle-first leader ↔ engine seam, returning RAII
//!   [`FindBlocksHandle`] / [`OnboardHandle`] / [`OffloadHandle`] and
//!   [`RequestOffloadDrain`].
//! - [`EngineWorkerSink`] / [`WorkerEngineDriver`] — the engine ↔ worker
//!   delegates, fence-keyed by [`FenceToken`].
//! - [`LoadOutcome`] / [`SaveOutcome`] — the terminal onboard/offload outcomes
//!   the engine publishes to the worker sink.
//! - [`FinishedStatus`] — the leader's request-finish / eviction gate.
//! - [`NoopBlockEngine`] — a caches-nothing engine used as a stand-in and test
//!   double.

mod actions;
mod engine;
mod handles;
mod noop;
mod protocol;

#[cfg(test)]
mod tests;

pub use actions::{EngineWorkerSink, LoadOutcome, SaveOutcome, WorkerEngineDriver};
pub use engine::LeaderEngine;
pub use handles::{
    FenceHandle, FindBlocksHandle, OffloadHandle, OnboardHandle, RequestOffloadDrain,
};
pub use noop::{NoopBlockEngine, NoopWorkerSink};
pub use protocol::{
    AcceptId, ActionFailure, ActionId, ActionStatus, BlockId, EvictionFence, EvictionOutcome,
    FenceToken, FindBlocksOutcome, FindBlocksRequest, FinishedStatus, LeaderEngineError, RequestId,
    SearchId, SequenceHash, WorkerRank,
};
