// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The local G1→G2 offload (save) submission seam + completion fold.
//!
//! This is the offload analogue of [`super::onboard`]. The engine buffers
//! `(SequenceHash, BlockId)` pairs in [`super::local::LocalConnectorEngine::offload`]
//! and flushes them at `finish_forward_pass` (Decision A: never enqueue a G1
//! read mid-forward-pass). The flush goes through [`OffloadSubmit`], a small
//! trait seam over the GPU/velo-bound [`OffloadEngine`]: the real
//! [`OffloadEngineSubmit`] forwards to
//! [`OffloadEngine::enqueue_g1_to_g2_with_precondition`], while a test double
//! impls it without a GPU (the concrete [`TransferHandle`] is not constructible
//! outside `offload/`, so the seam returns a [`OffloadTransfer`] — a
//! "`TransferHandle`-like" object — instead). The per-offload completion driver
//! ([`run_offload`]) awaits the transfer to terminal and projects it onto an
//! [`ActionStatus`]; the engine records that into the handle's cell with no
//! engine lock held (see [`super::driver::LocalConnectorEngine::finish_save_action`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use futures::future::BoxFuture;
use kvbm_common::LogicalResourceId;
use velo::EventHandle;

use kvbm_protocols::connector::{ActionFailure, ActionId, ActionStatus};
use kvbm_protocols::connector::{BlockId, RequestId, SequenceHash};

use crate::G1;
use crate::offload::{ExternalBlock, OffloadEngine, TransferHandle, TransferStatus};

/// One offload buffered by `offload`, flushed at `finish_forward_pass`.
///
/// Carries only what the flush needs: the action id (for the completion cell +
/// `by_request` upkeep), the owning request, the raw `(SequenceHash,
/// BlockId)` pairs the seam handed in, and the forward-pass iteration the
/// offload was buffered under — `finish_forward_pass(n)` submits only entries
/// stamped `<= n`, so a late pass-`n` flush can never submit pass-`n+1`'s
/// mid-pass buffer (those G1 sources are still being written). The G1
/// `ExternalBlock`s are built at flush time via [`build_external_blocks`]
/// (the arg order is reversed there).
pub(super) struct BufferedOffload {
    pub(super) action_id: ActionId,
    pub(super) request_id: RequestId,
    pub(super) resource: Option<LogicalResourceId>,
    pub(super) pairs: Vec<(SequenceHash, BlockId)>,
    pub(super) iteration: usize,
}

/// The offload-submission seam over [`OffloadEngine`].
///
/// Returns a [`OffloadTransfer`] (a "`TransferHandle`-like" object) rather than
/// the concrete [`TransferHandle`], because that type is only constructible
/// inside `offload/` — abstracting it keeps the GPU/velo-bound engine mockable
/// (and serves the future swappable-`BlockManager` goal).
pub(super) trait OffloadSubmit: Send + Sync {
    /// Whether an explicit logical resource has a configured submission route.
    fn supports_resource(&self, resource: LogicalResourceId) -> bool;

    /// Enqueue a G1→G2 offload, gated on `precondition` (the forward-pass
    /// completion event minted at flush). Mirrors
    /// [`OffloadEngine::enqueue_g1_to_g2_with_precondition`].
    fn submit_g1_to_g2(
        &self,
        resource: Option<LogicalResourceId>,
        blocks: Vec<ExternalBlock<G1>>,
        precondition: Option<EventHandle>,
    ) -> Result<Box<dyn OffloadTransfer>>;
}

/// A poll/await view over an in-flight offload transfer — the abstract surface
/// of [`TransferHandle`] the completion driver needs.
pub(super) trait OffloadTransfer: Send + Sync {
    /// The current (possibly non-terminal) transfer status.
    fn status(&self) -> TransferStatus;
    /// Blocks transferred successfully so far. Part of the poll surface for
    /// symmetry with [`Self::failed_blocks`]; the offload completion fold only
    /// needs the failed set (`project_offload_status`), so this is unused today.
    #[allow(dead_code)]
    fn completed_blocks(&self) -> Vec<BlockId>;
    /// Blocks that failed transfer.
    fn failed_blocks(&self) -> Vec<BlockId>;
    /// A future that resolves once the transfer reaches a terminal status
    /// (`Complete` | `Cancelled` | `Failed`). Owns its wait state so the
    /// returned future does not borrow `self`.
    fn wait_terminal(&self) -> BoxFuture<'static, ()>;
}

impl OffloadTransfer for TransferHandle {
    fn status(&self) -> TransferStatus {
        TransferHandle::status(self)
    }

    fn completed_blocks(&self) -> Vec<BlockId> {
        TransferHandle::completed_blocks(self)
    }

    fn failed_blocks(&self) -> Vec<BlockId> {
        TransferHandle::failed_blocks(self)
    }

    fn wait_terminal(&self) -> BoxFuture<'static, ()> {
        // Clone the status watch into a `'static` future — no borrow of `self`.
        let mut rx = self.subscribe_status();
        Box::pin(async move {
            loop {
                if rx.borrow().is_terminal() {
                    break;
                }
                if rx.changed().await.is_err() {
                    // Sender dropped without a terminal flip — treat as drained.
                    break;
                }
            }
        })
    }
}

/// The production [`OffloadSubmit`], forwarding to a real [`OffloadEngine`].
///
/// Constructed by the `tiering::engine` factory `build_local_connector_engine`
/// when the connector wires the real [`OffloadEngine`], via [`Self::new`].
pub(super) struct OffloadEngineSubmit {
    primary: LogicalResourceId,
    engines: BTreeMap<LogicalResourceId, Arc<OffloadEngine>>,
}

impl OffloadEngineSubmit {
    pub(super) fn new(engine: Arc<OffloadEngine>) -> Self {
        Self {
            primary: LogicalResourceId::default(),
            engines: BTreeMap::from([(LogicalResourceId::default(), engine)]),
        }
    }

    pub(super) fn from_resources(
        primary: LogicalResourceId,
        engines: Vec<(LogicalResourceId, Arc<OffloadEngine>)>,
    ) -> Result<Self> {
        let expected_len = engines.len();
        let engines = engines.into_iter().collect::<BTreeMap<_, _>>();
        anyhow::ensure!(
            engines.len() == expected_len,
            "duplicate resource offload engine"
        );
        anyhow::ensure!(
            engines.contains_key(&primary),
            "primary logical resource {primary:?} has no offload engine"
        );
        Ok(Self { primary, engines })
    }

    pub(super) fn primary_engine(&self) -> &Arc<OffloadEngine> {
        self.engines
            .get(&self.primary)
            .expect("resource offload routes validate their primary")
    }
}

impl OffloadSubmit for OffloadEngineSubmit {
    fn supports_resource(&self, resource: LogicalResourceId) -> bool {
        self.engines.contains_key(&resource)
    }

    fn submit_g1_to_g2(
        &self,
        resource: Option<LogicalResourceId>,
        blocks: Vec<ExternalBlock<G1>>,
        precondition: Option<EventHandle>,
    ) -> Result<Box<dyn OffloadTransfer>> {
        let resource = resource.unwrap_or(self.primary);
        let engine = self.engines.get(&resource).ok_or_else(|| {
            anyhow::anyhow!("offload submit has no route for logical resource {resource:?}")
        })?;
        let handle = engine.enqueue_g1_to_g2_with_precondition(blocks, precondition)?;
        Ok(Box::new(handle))
    }
}

/// A [`OffloadSubmit`] that refuses to submit — the fallback for an engine built
/// without an [`OffloadEngine`]: onboard-only tests and the offload-less
/// `LocalConnectorEngine::new` default (the wired engine uses the real
/// [`OffloadEngineSubmit`]). A flush against it folds each action to
/// `Failed(AllBlocks)`.
pub(super) struct DisabledOffloadSubmit;

impl OffloadSubmit for DisabledOffloadSubmit {
    fn supports_resource(&self, _resource: LogicalResourceId) -> bool {
        false
    }

    fn submit_g1_to_g2(
        &self,
        _resource: Option<LogicalResourceId>,
        _blocks: Vec<ExternalBlock<G1>>,
        _precondition: Option<EventHandle>,
    ) -> Result<Box<dyn OffloadTransfer>> {
        anyhow::bail!("offload submit not configured (engine built without an OffloadEngine)")
    }
}

/// Build the G1 `ExternalBlock`s for a flush from the seam's pairs.
///
/// **Arg order is reversed**: the seam hands `(SequenceHash, BlockId)` pairs,
/// but [`ExternalBlock::new`] is `(block_id, sequence_hash)`. A naive splat
/// would offload each block under the *wrong* hash, so the mapping is the
/// load-bearing detail this fold owns (and the highest-value test asserts).
pub(super) fn build_external_blocks(pairs: &[(SequenceHash, BlockId)]) -> Vec<ExternalBlock<G1>> {
    pairs
        .iter()
        .map(|(sequence_hash, block_id)| ExternalBlock::<G1>::new(*block_id, *sequence_hash))
        .collect()
}

/// Project a terminal transfer onto the engine-internal [`ActionStatus`] the
/// offload handle reads (which the handle then maps to a `SaveOutcome`).
///
/// `Complete`/`Cancelled` → `Complete` (a cancelled save is a *drained*,
/// best-effort save — the blocks that landed live on in G2). `Failed` →
/// `Failed`, naming the failed G1 ids when the transfer reports them, else the
/// whole-request `AllBlocks`.
pub(super) fn project_offload_status(
    status: TransferStatus,
    failed_blocks: Vec<BlockId>,
) -> ActionStatus {
    match status {
        TransferStatus::Complete | TransferStatus::Cancelled => ActionStatus::Complete,
        TransferStatus::Failed => {
            if failed_blocks.is_empty() {
                ActionStatus::Failed(ActionFailure::AllBlocks)
            } else {
                ActionStatus::Failed(ActionFailure::Partial {
                    block_ids: failed_blocks,
                })
            }
        }
        // Non-terminal cannot legitimately reach here (`run_offload` awaits
        // terminal first); degrade to `Complete` rather than reporting a false
        // failure.
        TransferStatus::Evaluating | TransferStatus::Queued | TransferStatus::Transferring => {
            ActionStatus::Complete
        }
    }
}

/// Drive one offload to a terminal [`ActionStatus`].
///
/// Runs on the leader's runtime, off the forward-pass thread (mirrors
/// [`super::onboard::run_onboard`]): awaits the transfer to terminal, then
/// projects its status + failed ids. The caller records the result into the
/// handle's cell and fires the worker sink only on the eviction-fence path —
/// never `mark_save_finished` (that is the once-per-request drain's job).
pub(super) async fn run_offload(transfer: Box<dyn OffloadTransfer>) -> ActionStatus {
    transfer.wait_terminal().await;
    project_offload_status(transfer.status(), transfer.failed_blocks())
}
