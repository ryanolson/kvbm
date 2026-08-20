// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Container state that moves through the offload pipeline.

use std::sync::Arc;

use velo::EventHandle;

use crate::{BlockId, SequenceHash};
use kvbm_logical::blocks::{BlockMetadata, ImmutableBlock};

use super::cancel::CancellationUnit;
use super::handle::{TransferId, TransferState, fail_transfer_unit, settle_transfer_unit};
use super::pending::PendingGuard;
use super::source::{SourceBlock, SourceBlocks};

const ABANDONED_PRECOMMIT_CONTAINER_ERROR: &str =
    "precommit offload container dropped before settlement";

/// One source block that passed policy evaluation.
pub(crate) struct EvaluatedBlock<T: BlockMetadata> {
    pub(crate) block_id: Option<BlockId>,
    pub(crate) sequence_hash: SequenceHash,
    pub(crate) source: SourceBlock<T>,
    pub(crate) pending_guard: Option<PendingGuard>,
}

impl<T: BlockMetadata> EvaluatedBlock<T> {
    pub(crate) fn new(source: SourceBlock<T>, pending_guard: Option<PendingGuard>) -> Self {
        let block_id = source.block_id();
        let sequence_hash = source
            .sequence_hash()
            .expect("each source block has a sequence hash");
        Self {
            block_id,
            sequence_hash,
            source,
            pending_guard,
        }
    }
}

/// A physical block after the commitment boundary.
pub(crate) struct ResolvedBlock<T: BlockMetadata> {
    /// Transfer ID that owns this block.
    pub(crate) transfer_id: TransferId,
    /// Source-tier block ID.
    pub(crate) block_id: BlockId,
    /// Sequence hash for this block.
    pub(crate) sequence_hash: SequenceHash,
    /// Strong source guard. External blocks do not need one.
    #[allow(dead_code)]
    pub(crate) guard: Option<ImmutableBlock<T>>,
    /// Pending guard that stays alive through physical completion.
    #[allow(dead_code)]
    pub(crate) pending_guard: Option<PendingGuard>,
    /// Transfer state for progress tracking.
    pub(crate) state: Arc<std::sync::Mutex<TransferState>>,
}

/// One container after all of its source blocks cross the upgrade boundary.
pub(crate) struct UpgradedContainer<T: BlockMetadata> {
    pub(crate) transfer_id: TransferId,
    pub(crate) blocks: Vec<ResolvedBlock<T>>,
    pub(crate) evicted: Vec<SequenceHash>,
    pub(crate) state: Arc<std::sync::Mutex<TransferState>>,
    pub(crate) cancellation: CancellationUnit,
}

struct ContainerPayload<T: BlockMetadata> {
    transfer_id: TransferId,
    source: Option<SourceBlocks<T>>,
    evaluated_blocks: Vec<EvaluatedBlock<T>>,
    filtered_ids: Vec<BlockId>,
    state: Arc<std::sync::Mutex<TransferState>>,
    precondition: Option<EventHandle>,
    input_len: usize,
}

/// The cancellation unit for one enqueue request.
///
/// The container keeps the request identity, source data, policy result,
/// precondition, state, and handle token until the upgrade boundary.
pub(crate) struct OffloadContainer<T: BlockMetadata> {
    payload: Option<ContainerPayload<T>>,
    cancellation: Option<CancellationUnit>,
}

impl<T: BlockMetadata> OffloadContainer<T> {
    pub(crate) fn new(
        transfer_id: TransferId,
        source: SourceBlocks<T>,
        state: Arc<std::sync::Mutex<TransferState>>,
        precondition: Option<EventHandle>,
    ) -> Self {
        let cancel_token = state.lock().unwrap().cancellation_token();
        let cancellation = cancel_token
            .root_unit()
            .expect("a transfer owns exactly one root cancellation unit");
        Self::with_cancellation(transfer_id, source, state, precondition, cancellation)
    }

    pub(crate) fn with_cancellation(
        transfer_id: TransferId,
        source: SourceBlocks<T>,
        state: Arc<std::sync::Mutex<TransferState>>,
        precondition: Option<EventHandle>,
        cancellation: CancellationUnit,
    ) -> Self {
        let input_len = source.len();
        let cancellation = cancellation.bind_state(Arc::clone(&state));
        Self {
            payload: Some(ContainerPayload {
                transfer_id,
                source: Some(source),
                evaluated_blocks: Vec::new(),
                filtered_ids: Vec::new(),
                state,
                precondition,
                input_len,
            }),
            cancellation: Some(cancellation),
        }
    }

    pub(crate) fn transfer_id(&self) -> TransferId {
        self.payload().transfer_id
    }

    pub(crate) fn state(&self) -> Arc<std::sync::Mutex<TransferState>> {
        Arc::clone(&self.payload().state)
    }

    pub(crate) fn source_len(&self) -> usize {
        self.payload().input_len
    }

    pub(crate) fn take_source(&mut self) -> SourceBlocks<T> {
        self.payload_mut()
            .source
            .take()
            .expect("policy evaluation consumes source once")
    }

    pub(crate) fn finish_evaluation(
        &mut self,
        evaluated_blocks: Vec<EvaluatedBlock<T>>,
        filtered_ids: Vec<BlockId>,
    ) {
        let payload = self.payload_mut();
        payload.evaluated_blocks = evaluated_blocks;
        payload.filtered_ids = filtered_ids;
    }

    pub(crate) fn evaluated_len(&self) -> usize {
        self.payload().evaluated_blocks.len()
    }

    pub(crate) fn precondition(&self) -> Option<EventHandle> {
        self.payload().precondition
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancellation_token().is_precommit_cancelled()
    }

    pub(crate) fn cancellation_token(&self) -> super::cancel::CancellationToken {
        self.cancellation
            .as_ref()
            .expect("container cancellation unit exists before upgrade")
            .token()
    }

    /// Release all retained guards before this container publishes failure.
    pub(crate) fn fail(mut self, error: String) {
        let (state, cancellation) = self.release_payload();
        let committed = cancellation.is_committed();

        if committed {
            fail_transfer_unit(cancellation, state, error);
        } else {
            let precommit_cancelled = cancellation.token().is_precommit_cancelled();
            let mut state = state.lock().unwrap();
            if precommit_cancelled {
                state.set_cancelled();
            } else {
                state.set_error(error);
            }
            drop(state);
            drop(cancellation);
        }
    }

    /// Release all retained guards before this container publishes success.
    pub(crate) fn finish(mut self) {
        let (state, cancellation) = self.release_payload();
        let committed = cancellation.is_committed();

        if committed {
            settle_transfer_unit(cancellation, state);
        } else {
            let precommit_cancelled = cancellation.token().is_precommit_cancelled();
            let mut state = state.lock().unwrap();
            if precommit_cancelled {
                state.set_cancelled();
            } else {
                state.set_complete();
            }
            drop(state);
            drop(cancellation);
        }
    }

    /// Claim commitment for this complete container and make its physical vector.
    ///
    /// The caller performs the final cancellation sweep before this call.
    /// The claim selects one winner with `CancellationToken::request`. A
    /// request that wins drops this container without a physical vector.
    pub(crate) fn upgrade(mut self) -> Option<UpgradedContainer<T>> {
        if !self
            .cancellation
            .as_ref()
            .expect("container cancellation unit exists before upgrade")
            .claim_commitment()
        {
            return None;
        }

        let payload = self
            .payload
            .take()
            .expect("container payload exists before upgrade");
        let cancellation = self
            .cancellation
            .take()
            .expect("container cancellation unit exists before upgrade");
        debug_assert!(
            payload.source.is_none(),
            "upgrade requires policy evaluation"
        );

        {
            let mut state = payload.state.lock().unwrap();
            state.mark_committed();
        }

        let mut blocks = Vec::with_capacity(payload.evaluated_blocks.len());
        let mut evicted = Vec::new();

        for evaluated in payload.evaluated_blocks {
            let EvaluatedBlock {
                sequence_hash,
                source,
                pending_guard,
                ..
            } = evaluated;
            match source {
                SourceBlock::Strong(block) => blocks.push(ResolvedBlock {
                    transfer_id: payload.transfer_id,
                    block_id: block.block_id(),
                    sequence_hash,
                    guard: Some(block),
                    pending_guard,
                    state: Arc::clone(&payload.state),
                }),
                SourceBlock::External(block) => blocks.push(ResolvedBlock {
                    transfer_id: payload.transfer_id,
                    block_id: block.block_id,
                    sequence_hash,
                    guard: None,
                    pending_guard,
                    state: Arc::clone(&payload.state),
                }),
                SourceBlock::Weak(weak) => match weak.upgrade() {
                    Some(block) => blocks.push(ResolvedBlock {
                        transfer_id: payload.transfer_id,
                        block_id: block.block_id(),
                        sequence_hash,
                        guard: Some(block),
                        pending_guard,
                        state: Arc::clone(&payload.state),
                    }),
                    None => {
                        tracing::debug!(?sequence_hash, "Weak block evicted before transfer");
                        evicted.push(sequence_hash);
                    }
                },
            }
        }

        let block_ids: Vec<BlockId> = blocks.iter().map(|block| block.block_id).collect();
        let mut state = payload.state.lock().unwrap();
        // Weak inputs do not expose a block ID during policy evaluation. Add
        // their IDs only after a successful upgrade. Strong and external
        // inputs already exist in `passed_blocks`, so this does not duplicate
        // their progress entries.
        let newly_resolved: Vec<BlockId> = block_ids
            .iter()
            .copied()
            .filter(|block_id| !state.passed_blocks.contains(block_id))
            .collect();
        state.add_passed(newly_resolved);
        if !block_ids.is_empty() {
            state.mark_committed_blocks(block_ids);
        }
        drop(state);

        Some(UpgradedContainer {
            transfer_id: payload.transfer_id,
            blocks,
            evicted,
            state: Arc::clone(&payload.state),
            cancellation,
        })
    }

    fn payload(&self) -> &ContainerPayload<T> {
        self.payload
            .as_ref()
            .expect("container payload is unavailable after upgrade")
    }

    fn payload_mut(&mut self) -> &mut ContainerPayload<T> {
        self.payload
            .as_mut()
            .expect("container payload is unavailable after upgrade")
    }

    fn release_payload(&mut self) -> (Arc<std::sync::Mutex<TransferState>>, CancellationUnit) {
        let payload = self
            .payload
            .take()
            .expect("container payload exists before settlement");
        let cancellation = self
            .cancellation
            .take()
            .expect("container cancellation unit exists before settlement");
        let state = Arc::clone(&payload.state);

        // The payload owns source blocks and pending guards. Release it before
        // any terminal state or cancellation confirmation becomes visible.
        drop(payload);
        (state, cancellation)
    }
}

impl<T: BlockMetadata> Drop for OffloadContainer<T> {
    fn drop(&mut self) {
        let Some(payload) = self.payload.take() else {
            return;
        };
        let cancellation = self
            .cancellation
            .take()
            .expect("container payload owns a cancellation unit");
        let precommit_cancelled = cancellation.token().is_precommit_cancelled();
        let committed = cancellation.is_committed();
        let state = Arc::clone(&payload.state);

        // Drop source blocks and policy guards before any terminal state or
        // cancellation confirmation becomes visible.
        drop(payload);

        if !committed {
            let mut state = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if precommit_cancelled {
                state.set_cancelled();
            } else {
                state.set_error(ABANDONED_PRECOMMIT_CONTAINER_ERROR.to_string());
            }
        }

        drop(cancellation);
    }
}
