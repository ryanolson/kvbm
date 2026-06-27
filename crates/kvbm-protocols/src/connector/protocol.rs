// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plain wire types shared across the connector ↔ engine seam.
//!
//! Every type here is plain: [`RequestId`] is a `String`, and [`BlockId`] /
//! [`SequenceHash`] come from `kvbm-common`. No engine-internal type ever
//! appears on this surface — that is what lets the engine impl live in its own
//! crate or process.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::disagg::TransferParams;

use super::handles::FindBlocksHandle;

pub use kvbm_common::{BlockId, SequenceHash};

/// Request identifier, as the vLLM scheduler hands it to the connector.
///
/// vLLM identifies requests by string id; the connector never parses it.
pub type RequestId = String;

// ---------------------------------------------------------------------------
// Lifecycle gates
// ---------------------------------------------------------------------------

/// Returned by the leader's request-finish / eviction path.
///
/// `Finished` means the engine had nothing in flight and the connector may reap
/// the slot now. `Pending` means the engine accepted ownership of an in-flight
/// drain; the connector keeps the slot until the worker reports the matching
/// completion (offload → `finished_sending`) on a later step. Completions are
/// delivered to the worker by push (see [`super::EngineWorkerSink`]), never
/// pulled from the engine here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishedStatus {
    Finished,
    Pending,
}

/// Identifies a worker (TP rank) in an [`EvictionFence`].
pub type WorkerRank = u32;

// ---------------------------------------------------------------------------
// Opaque engine-side identifiers
// ---------------------------------------------------------------------------

/// Opaque key for an engine match session. Embedded in the search-kind variant
/// of a [`FindBlocksHandle`]; the connector never parses it.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchId(pub Uuid);

impl SearchId {
    /// Mint a fresh random id. Engine-side; the connector only ever receives
    /// one inside a minted handle.
    #[allow(clippy::new_without_default)] // random id; a `Default` would be misleading.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Opaque key for an in-flight onboard *or* offload action. Embedded in an
/// [`super::handles::OnboardHandle`] / [`super::handles::OffloadHandle`].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionId(pub Uuid);

impl ActionId {
    /// Mint a fresh random id. Engine-side.
    #[allow(clippy::new_without_default)] // random id; a `Default` would be misleading.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Opaque generation id for an accepted remote-prefill lifecycle. Embedded in
/// the prefill-kind variant of a [`FindBlocksHandle`]; the connector never
/// inspects it.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptId(pub u64);

impl AcceptId {
    /// Mint a fresh generation id. Engine-side; the connector only ever
    /// receives one inside a minted handle.
    #[allow(clippy::new_without_default)] // minted id; a `Default` would be misleading.
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// Per-(eviction-generation, worker) eviction-fence barrier token.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct FenceToken {
    /// Per-eviction-generation discriminator.
    pub generation: Uuid,
    /// The worker (TP rank) this barrier gates.
    pub rank: WorkerRank,
}

impl FenceToken {
    /// Mint a fresh per-generation token for `rank`. Engine-side.
    pub fn new(rank: WorkerRank) -> Self {
        Self {
            generation: Uuid::new_v4(),
            rank,
        }
    }
}

// ---------------------------------------------------------------------------
// Unified find / onboard seam
// ---------------------------------------------------------------------------

/// Input to [`super::engine::LeaderEngine::find_blocks`]. Hashes and counts
/// only — token ids never cross the seam.
#[derive(Debug, Clone)]
pub struct FindBlocksRequest {
    pub request_id: RequestId,
    /// Full per-block hash chain in absolute-position order.
    pub sequence_hashes: Arc<[SequenceHash]>,
    /// vLLM's `num_computed_tokens` at this poll.
    pub num_computed_tokens: usize,
    /// Total tokens in the sequence.
    pub total_tokens: usize,
    /// The slot's parsed `kv_transfer_params`, passed through whole.
    pub transfer_params: Option<TransferParams>,
}

/// Outcome of [`super::engine::LeaderEngine::find_blocks`]. Encodes everything
/// the connector needs without exposing handle internals.
#[derive(Debug)]
pub enum FindBlocksOutcome {
    /// The poll's window overlaps another lifecycle's in-flight onboard.
    Deferred,
    /// Still resolving — re-poll next step.
    Searching { minted: Option<FindBlocksHandle> },
    /// Resolved to a token-granular matched count.
    Resolved {
        matched_tokens: usize,
        /// `Some` iff this call latched a fresh lifecycle.
        minted: Option<FindBlocksHandle>,
        /// Engine instruction to drop the parked handle.
        release_parked: bool,
    },
}

// ---------------------------------------------------------------------------
// Action completion
// ---------------------------------------------------------------------------

/// Result of [`super::engine::LeaderEngine::poll_action`] — the engine-internal
/// completion source read by onboard/offload handles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionStatus {
    /// Still in flight.
    Pending,
    /// Terminal, all blocks transferred.
    Complete,
    /// Terminal, with a failure.
    Failed(ActionFailure),
}

/// The failure shape inside a terminal [`ActionStatus`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionFailure {
    /// Whole-request failure.
    AllBlocks,
    /// Named G1 block ids failed.
    Partial { block_ids: Vec<usize> },
}

// ---------------------------------------------------------------------------
// Eviction fence
// ---------------------------------------------------------------------------

/// Minted by [`super::engine::LeaderEngine::evict`]. The connector embeds it in
/// per-iteration metadata; each worker reads the token for its rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvictionFence {
    pub request_id: RequestId,
    pub per_worker: Vec<FenceToken>,
}

/// What [`super::engine::LeaderEngine::evict`] hands back: the wire fence paired
/// with the leader's own observational view of the same barrier.
#[derive(Debug)]
pub struct EvictionOutcome {
    /// The per-worker fence.
    pub fence: EvictionFence,
    /// Leader-side completion cell over the fence barrier.
    pub handle: Option<super::handles::FenceHandle>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Synchronous error from the leader seam.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum LeaderEngineError {
    /// `onboard_blocks` routed to a local search whose pin is no longer live.
    #[error("search not matched (pin lost or still pending)")]
    SearchNotMatched,
    /// A prefill-side verb reached an engine with no conditional-disagg plane.
    #[error("conditional disaggregation is not configured on this engine")]
    DisaggNotConfigured,
    /// `onboard_blocks` routed to a stale or unknown prefill lifecycle.
    #[error("prefill session unknown or stale for this request")]
    PrefillSessionStale,
    /// Prefill-verb arguments are inconsistent with the latched lifecycle.
    #[error("invalid prefill request: {reason}")]
    InvalidPrefillRequest { reason: String },
    /// The engine and connector disagree about who holds the RAII release.
    #[error("find_blocks desync: latched lifecycle with no caller handle")]
    FindBlocksDesync,
    /// `onboard_blocks`'s committed external count diverges from the engine's
    /// stored promise.
    #[error("external tokens mismatch: vLLM committed {got}, engine stored {expected}")]
    ExternalTokensMismatch { expected: usize, got: usize },
    /// A second onboard was driven against a lifecycle whose onboard is already
    /// in flight.
    #[error("onboard already in flight for this request's latched lifecycle")]
    OnboardAlreadyInFlight,
    /// The engine has shut down.
    #[error("engine has shut down")]
    Shutdown,
}
