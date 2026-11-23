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
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::future::BoxFuture;
use kvbm_common::LogicalResourceId;
use velo::EventHandle;

use kvbm_protocols::connector::{ActionFailure, ActionId, ActionStatus};
use kvbm_protocols::connector::{BlockId, RequestId, SequenceHash};

use crate::offload::{ExternalBlock, OffloadEngine, TransferHandle, TransferStatus};
use crate::{G1, G2};
use kvbm_logical::blocks::ImmutableBlock;

use super::bundle::{BundleOffload, OffloadTransition};
use super::local::LocalConnectorEngine;
use crate::tiering::policy::ResourceLineage;

pub(super) type LocalBundleOffload =
    BundleOffload<Vec<(SequenceHash, BlockId)>, Vec<ImmutableBlock<G2>>>;

pub(super) enum BufferedOffloadCompletion {
    Single,
    Bundle(Arc<BundleOffloadRuntime>),
}

pub(super) struct BundleOffloadRuntime {
    transaction: Mutex<LocalBundleOffload>,
    drain: BundleChildDrain,
    lineages: Mutex<Option<Vec<ResourceLineage>>>,
}

impl BundleOffloadRuntime {
    pub(super) fn new(
        transaction: LocalBundleOffload,
        child_count: NonZeroUsize,
        lineages: Vec<ResourceLineage>,
    ) -> Self {
        Self {
            transaction: Mutex::new(transaction),
            drain: BundleChildDrain::new(child_count),
            lineages: Mutex::new(Some(lineages)),
        }
    }

    fn transaction(&self) -> std::sync::MutexGuard<'_, LocalBundleOffload> {
        self.transaction
            .lock()
            .expect("bundle-offload mutex poisoned")
    }

    fn finish_child(&self, terminal: Option<ActionStatus>) -> Option<ActionStatus> {
        self.drain.finish_child(terminal)
    }

    fn take_lineages(&self) -> Option<Vec<ResourceLineage>> {
        self.lineages
            .lock()
            .expect("bundle-lineages mutex poisoned")
            .take()
    }
}

struct BundleChildDrain {
    remaining: AtomicUsize,
    terminal: Mutex<Option<ActionStatus>>,
}

impl BundleChildDrain {
    fn new(child_count: NonZeroUsize) -> Self {
        Self {
            remaining: AtomicUsize::new(child_count.get()),
            terminal: Mutex::new(None),
        }
    }

    fn finish_child(&self, candidate: Option<ActionStatus>) -> Option<ActionStatus> {
        if let Some(candidate) = candidate {
            let mut terminal = self
                .terminal
                .lock()
                .expect("bundle-child-drain mutex poisoned");
            let replace = terminal.is_none()
                || matches!(
                    (&*terminal, &candidate),
                    (Some(ActionStatus::Complete), ActionStatus::Failed(_))
                );
            if replace {
                *terminal = Some(candidate);
            }
        }
        let previous = self
            .remaining
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            })
            .ok()?;
        if previous != 1 {
            return None;
        }
        Some(
            self.terminal
                .lock()
                .expect("bundle-child-drain mutex poisoned")
                .take()
                .unwrap_or(ActionStatus::Failed(ActionFailure::AllBlocks)),
        )
    }
}

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
    pub(super) planned_bytes: Option<u64>,
    pub(super) iteration: usize,
    pub(super) completion: BufferedOffloadCompletion,
}

#[derive(Clone, Copy)]
pub(super) struct BundleTransferObservation {
    resource: LogicalResourceId,
    planned_bytes: u64,
    started_at: Instant,
}

impl BundleTransferObservation {
    pub(super) fn start(resource: LogicalResourceId, planned_bytes: u64) -> Self {
        Self {
            resource,
            planned_bytes,
            started_at: Instant::now(),
        }
    }

    fn duration(self) -> Duration {
        self.started_at.elapsed()
    }
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
///
/// A non-terminal status cannot legitimately reach here (`run_offload` awaits
/// terminal first), so it is a logic error: it fails closed to
/// `Failed(AllBlocks)` and logs at error level, because no transfer has proven
/// the save is done.
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
        // terminal first). Fail closed rather than report a false success: no
        // transfer has proven that the save is done.
        TransferStatus::Evaluating | TransferStatus::Queued | TransferStatus::Transferring => {
            tracing::error!(
                ?status,
                "offload projected a non-terminal transfer; failing closed to Failed(AllBlocks)"
            );
            ActionStatus::Failed(ActionFailure::AllBlocks)
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

impl LocalConnectorEngine {
    pub(super) fn finish_offload_child(
        &self,
        action_id: ActionId,
        request_id: &RequestId,
        resource: Option<LogicalResourceId>,
        pairs: Vec<(SequenceHash, BlockId)>,
        observation: Option<BundleTransferObservation>,
        completion: BufferedOffloadCompletion,
        outcome: ActionStatus,
    ) {
        let BufferedOffloadCompletion::Bundle(runtime) = completion else {
            self.finish_save_action(action_id, request_id, outcome);
            return;
        };
        let Some(resource) = resource else {
            if let Some(outcome) =
                runtime.finish_child(Some(ActionStatus::Failed(ActionFailure::AllBlocks)))
            {
                self.finish_save_action(action_id, request_id, outcome);
            }
            return;
        };

        let completion = match &outcome {
            ActionStatus::Complete => {
                let hashes = pairs.iter().map(|(hash, _)| *hash).collect::<Vec<_>>();
                let pins = self
                    .leader
                    .g2_manager_for(resource)
                    .map(|manager| manager.match_blocks(&hashes))
                    .unwrap_or_default();
                if pins.len() == hashes.len() {
                    Ok(pins)
                } else {
                    Err(Some(pairs.iter().map(|(_, block_id)| *block_id).collect()))
                }
            }
            ActionStatus::Failed(ActionFailure::Partial { block_ids }) => {
                Err(Some(block_ids.clone()))
            }
            ActionStatus::Failed(ActionFailure::Resource { block_ids, .. }) => {
                Err(block_ids.clone())
            }
            ActionStatus::Failed(ActionFailure::AllBlocks) | ActionStatus::Pending => Err(None),
        };
        self.record_bundle_transfer(observation, &outcome, completion.is_ok());
        let transition = {
            let mut transaction = runtime.transaction();
            match completion {
                Ok(pins) => transaction.complete(resource, Ok(pins)),
                Err(failed_blocks) => transaction.fail(resource, failed_blocks),
            }
        };

        let terminal = match transition {
            Ok(OffloadTransition::Pending | OffloadTransition::Settled(_)) => None,
            Ok(OffloadTransition::Abort(abort)) => {
                self.record_bundle_transaction("abort");
                Some(ActionStatus::Failed(ActionFailure::Resource {
                    resource: abort.failure().resource(),
                    block_ids: abort.failure().failed_blocks().map(<[usize]>::to_vec),
                }))
            }
            Ok(OffloadTransition::Commit(commit)) => {
                let (publication, _retained_sources) = commit.into_publication();
                match runtime.take_lineages() {
                    Some(lineages) => {
                        let (identity, key, generation, resources) = publication.into_parts();
                        match self.commit_bundle(identity, key, generation, resources, lineages) {
                            Ok(()) => {
                                self.record_bundle_transaction("commit");
                                Some(ActionStatus::Complete)
                            }
                            Err(error) => {
                                tracing::error!(%error, "bundle catalog publication failed");
                                self.record_bundle_transaction("abort");
                                Some(ActionStatus::Failed(ActionFailure::AllBlocks))
                            }
                        }
                    }
                    None => {
                        tracing::error!("bundle lineage was already consumed before publication");
                        self.record_bundle_transaction("abort");
                        Some(ActionStatus::Failed(ActionFailure::AllBlocks))
                    }
                }
            }
            Err(error) => {
                tracing::error!(%error, ?resource, "bundle offload completion fold failed");
                self.record_bundle_transaction("abort");
                Some(ActionStatus::Failed(ActionFailure::Resource {
                    resource,
                    block_ids: None,
                }))
            }
        };
        if let Some(outcome) = runtime.finish_child(terminal) {
            self.finish_save_action(action_id, request_id, outcome);
        }
    }

    fn record_bundle_transaction(&self, outcome: &'static str) {
        if let Some(observability) = self.leader.observability() {
            observability
                .bundle_metrics()
                .record_transaction("offload", outcome);
        }
    }

    pub(super) fn record_bundle_transfer_start(
        &self,
        observation: Option<BundleTransferObservation>,
    ) {
        let Some(observation) = observation else {
            return;
        };
        if let Some(observability) = self.leader.observability() {
            observability
                .bundle_metrics()
                .record_resource_planned_bytes(
                    "offload_transfer",
                    observation.resource,
                    observation.planned_bytes,
                );
        }
    }

    fn record_bundle_transfer(
        &self,
        observation: Option<BundleTransferObservation>,
        status: &ActionStatus,
        resource_committed: bool,
    ) {
        let Some(observation) = observation else {
            return;
        };
        let Some(observability) = self.leader.observability() else {
            return;
        };
        let (outcome, reason, actual_safe_bytes) =
            transfer_metric_outcome(status, observation, resource_committed);
        let metrics = observability.bundle_metrics();
        let resource = observation.resource.0.to_string();
        let duration = observation.duration();
        metrics.record_transfer_bytes(&resource, "g1", "g2", actual_safe_bytes);
        metrics.observe_transfer(&resource, "offload", duration);
        metrics.record_resource_outcome(
            "offload_transfer",
            observation.resource,
            outcome,
            reason,
            actual_safe_bytes,
        );
        metrics.observe_resource_duration(
            "offload_transfer",
            observation.resource,
            outcome,
            duration,
        );
    }
}

fn transfer_metric_outcome(
    status: &ActionStatus,
    observation: BundleTransferObservation,
    resource_committed: bool,
) -> (&'static str, &'static str, u64) {
    if !resource_committed && matches!(status, ActionStatus::Complete) {
        return ("failed", "incomplete_registration", 0);
    }
    match status {
        ActionStatus::Complete => ("complete", "complete", observation.planned_bytes),
        ActionStatus::Failed(ActionFailure::Partial { .. }) => ("failed", "partial_failure", 0),
        ActionStatus::Failed(ActionFailure::Resource { .. }) => ("failed", "resource_failure", 0),
        ActionStatus::Failed(ActionFailure::AllBlocks) => ("failed", "all_blocks", 0),
        ActionStatus::Pending => ("failed", "nonterminal", 0),
    }
}

#[cfg(test)]
mod bundle_drain_tests {
    use super::BundleChildDrain;
    use kvbm_common::LogicalResourceId;
    use kvbm_protocols::connector::{ActionFailure, ActionStatus};
    use std::num::NonZeroUsize;

    #[test]
    fn failed_child_waits_for_every_sibling_before_parent_terminal() {
        let drain = BundleChildDrain::new(NonZeroUsize::new(3).unwrap());
        let failure = ActionStatus::Failed(ActionFailure::Resource {
            resource: LogicalResourceId(7),
            block_ids: None,
        });

        assert_eq!(drain.finish_child(Some(failure.clone())), None);
        assert_eq!(drain.finish_child(None), None);
        assert_eq!(drain.finish_child(None), Some(failure));
    }
}

#[cfg(test)]
mod status_projection_tests {
    use super::*;

    /// A non-terminal transfer proves nothing about the save: it must fail
    /// closed instead of projecting `Complete` for work that may still be
    /// running (or may never have started).
    #[test]
    fn nonterminal_transfer_status_fails_closed() {
        for status in [
            TransferStatus::Evaluating,
            TransferStatus::Queued,
            TransferStatus::Transferring,
        ] {
            assert_eq!(
                project_offload_status(status, vec![]),
                ActionStatus::Failed(ActionFailure::AllBlocks),
                "non-terminal {status:?} must not project a successful save"
            );
        }
    }
}
