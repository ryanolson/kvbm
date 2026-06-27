// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `NoopBlockEngine` — a block engine that caches nothing.
//!
//! It impls the `LeaderEngine` seam: every find reports no
//! match, every onboard/offload hands back an already-terminal handle, and
//! eviction observes no barrier. Used as the engine stand-in for the leader's
//! binding-facing constructors that have no injected engine, and as a test
//! double. The type is stateless — it holds no per-action state.

use std::sync::{Arc, Mutex};

use super::actions::{EngineWorkerSink, LoadOutcome, SaveOutcome};
use super::engine::LeaderEngine;
use super::handles::{FindBlocksHandle, OffloadHandle, OnboardHandle, RequestOffloadDrain};
use super::protocol::{
    ActionId, ActionStatus, BlockId, EvictionFence, EvictionOutcome, FenceToken, FindBlocksOutcome,
    FindBlocksRequest, LeaderEngineError, RequestId, SearchId, SequenceHash,
};

/// Engine that caches nothing.
pub struct NoopBlockEngine;

impl NoopBlockEngine {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl LeaderEngine for NoopBlockEngine {
    fn offload(
        self: Arc<Self>,
        _req: &RequestId,
        _pairs: Vec<(SequenceHash, BlockId)>,
    ) -> Result<OffloadHandle, LeaderEngineError> {
        let me: Arc<dyn LeaderEngine> = self;
        Ok(OffloadHandle::new(
            ActionId::new(),
            Arc::downgrade(&me),
            Arc::new(Mutex::new(ActionStatus::Complete)),
        ))
    }

    fn evict(&self, req: &RequestId) -> EvictionOutcome {
        EvictionOutcome {
            fence: EvictionFence {
                request_id: req.clone(),
                per_worker: Vec::new(),
            },
            handle: None,
        }
    }

    fn take_offload_drain(&self, _req: &RequestId) -> Option<RequestOffloadDrain> {
        Some(RequestOffloadDrain::noop())
    }

    fn poll_action(&self, _id: &ActionId) -> ActionStatus {
        ActionStatus::Complete
    }

    fn release_search(&self, _id: &SearchId) {}

    fn release_action(&self, _action: &ActionId) {}

    fn find_blocks(
        self: Arc<Self>,
        _req: &FindBlocksRequest,
        _live: Option<&FindBlocksHandle>,
    ) -> Result<FindBlocksOutcome, LeaderEngineError> {
        Ok(FindBlocksOutcome::Resolved {
            matched_tokens: 0,
            minted: None,
            release_parked: false,
        })
    }

    fn onboard_blocks(
        self: Arc<Self>,
        _handle: &FindBlocksHandle,
        dest: &[BlockId],
        _num_external_tokens: usize,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        let me: Arc<dyn LeaderEngine> = self;
        Ok(OnboardHandle::new(
            ActionId::new(),
            Arc::downgrade(&me),
            Arc::new(Mutex::new(ActionStatus::Complete)),
            dest.to_vec(),
        ))
    }
}

/// No-op [`EngineWorkerSink`] — a stand-in worker sink used in construction and
/// as a test double. Every completion publish is discarded.
pub struct NoopWorkerSink;

impl NoopWorkerSink {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl EngineWorkerSink for NoopWorkerSink {
    fn mark_load_finished(&self, _req: &RequestId, _outcome: LoadOutcome) {}
    fn mark_save_finished(&self, _req: &RequestId, _outcome: SaveOutcome) {}
    fn mark_fence_complete(&self, _token: FenceToken) {}
}
