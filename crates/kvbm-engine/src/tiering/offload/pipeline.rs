// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline coordination for offload transfers.
//!
//! A pipeline connects these stages:
//! 1. **PolicyEvaluator**: Evaluates blocks against policies, filters out non-passing blocks
//! 2. **PreconditionAwaiter**: Awaits each container precondition before batching
//! 3. **BatchCollector**: Accumulates complete containers into batches
//! 4. **BlockUpgrader**: Upgrades `WeakBlock` → `ImmutableBlock` (via `upgrade_batch`)
//! 5. **Transfer Executor**: Executes the actual data transfer
//!    - `BlockTransferExecutor`: For BlockManager destinations (G2, G3)
//!    - `ObjectTransferExecutor`: For object storage destinations (G4)
//!
//! # Cancellation Architecture
//!
//! Unlike mpsc-based pipelines where cancellation only happens at dequeue boundaries,
//! this implementation uses [`CancellableQueue`] which enables a dedicated sweeper task
//! to actively remove items from cancelled transfers. This ensures that `ImmutableBlock`
//! guards are dropped promptly when a transfer is cancelled.
//!
//! ```text
//! enqueue() ─┬─► [CancellableQueue A] ──► PolicyEvaluator ──┬─► [PreconditionAwaiter]
//!            │                                              │
//!            │                                      [CancellableQueue B]
//!            │                                              │
//!            └──────────────► [CancelSweeper] ◄─────────────┴──► BatchCollector ──► Executor
//!                                    │
//!                              (iterates queues,
//!                               removes by TransferId,
//!                               drops ImmutableBlock guards)
//! ```

mod block_executor;
mod config;
mod ingress;
mod local_ownership;
mod object_executor;
mod runtime;
pub(super) mod shutdown;

#[cfg(test)]
pub use object_executor::ObjectTransferExecutor;
#[cfg(test)]
use object_executor::SharedObjectExecutorState;

pub use config::{ObjectPipelineBuilder, ObjectPipelineConfig, PipelineBuilder, PipelineConfig};
pub(crate) use ingress::PipelineIngress;
pub use runtime::RegisterObserver;
pub(crate) use runtime::{ChainOutput, ChainOutputRx, ObjectPipeline, Pipeline};

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

#[cfg(test)]
use dashmap::DashMap;
use futures::future::Either;
#[cfg(test)]
use parking_lot::Mutex as ParkingMutex;
use tokio::sync::watch;
use tokio::task::{JoinError, JoinSet};

use crate::leader::InstanceLeader;
use crate::{BlockId, SequenceHash};
#[cfg(test)]
use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::blocks::BlockMetadata;

use super::batch::{TimingTrace, TransferBatch};
use super::cancel::CancellationUnit;
use super::container::{EvaluatedBlock, OffloadContainer, ResolvedBlock};
use super::handle::{TransferId, TransferState, TransferStatus, settle_transfer_unit};
use super::pending::PendingTracker;
use super::policy::{EvalContext, OffloadPolicy};
use super::queue::CancellableQueue;
use super::source::{SourceBlock, SourceBlocks};
use shutdown::PRECOMMIT_SHUTDOWN_ERROR;
/// Sweeper task that removes cancelled items from queues.
async fn cancel_sweeper<Src: BlockMetadata>(
    input_queues: Vec<Arc<CancellableQueue<OffloadContainer<Src>>>>,
    mut cancel_rx: watch::Receiver<HashSet<TransferId>>,
    mut shutdown_rx: watch::Receiver<bool>,
    interval: Duration,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                // Sweep all queues
                for queue in &input_queues {
                    let removed = queue.sweep();
                    if removed > 0 {
                        tracing::debug!("Sweeper removed {} cancelled input items", removed);
                    }
                }

            }
            result = cancel_rx.changed() => {
                if result.is_err() {
                    // Channel closed, shutdown
                    break;
                }
                // New cancellation added, sweep immediately
                for queue in &input_queues {
                    queue.sweep();
                }
            }
            result = shutdown_rx.changed() => {
                if result.is_err() || *shutdown_rx.borrow() {
                    break;
                }
            }
        }
    }
}

/// Policy evaluator stage.
struct PolicyEvaluator<T: BlockMetadata> {
    policies: Vec<Arc<dyn OffloadPolicy<T>>>,
    timeout: Duration,
    input_queue: Arc<CancellableQueue<OffloadContainer<T>>>,
    output_queue: Arc<CancellableQueue<OffloadContainer<T>>>,
    cancel_rx: watch::Receiver<HashSet<TransferId>>,
    /// Tracker for pending transfers - guards are created when blocks pass policy
    pending_tracker: Arc<PendingTracker>,
}

impl<T: BlockMetadata> PolicyEvaluator<T> {
    async fn run(mut self) {
        loop {
            if self.input_queue.is_closed() {
                while let Some(item) = self.input_queue.pop() {
                    item.data.fail(PRECOMMIT_SHUTDOWN_ERROR.to_string());
                }
                break;
            }

            while let Some(item) = self.input_queue.pop_valid() {
                self.evaluate(item.data).await;
            }

            if self.input_queue.is_closed() {
                break;
            }

            tokio::select! {
                _ = self.input_queue.notified() => {}
                result = self.cancel_rx.changed() => {
                    if result.is_err() {
                        break;
                    }
                }
            }
        }
    }

    async fn evaluate(&self, mut container: OffloadContainer<T>) {
        nvtx_range!("offload::policy");
        let transfer_id = container.transfer_id();
        let state = container.state();

        // Set total_expected_blocks for per-transfer sentinel flush
        let total_blocks = container.source_len();
        {
            let mut state = state.lock().unwrap();
            state.total_expected_blocks = total_blocks;
        }

        if container.is_cancelled() {
            tracing::debug!(%transfer_id, "Transfer cancelled before evaluation");
            return;
        }

        let mut passed = Vec::new();
        let mut filtered = Vec::new();

        // The source stays in the container until this stage consumes it. The
        // evaluated data then stays in that same container until upgrade.
        match container.take_source() {
            SourceBlocks::External(external_blocks) => {
                for ext in external_blocks {
                    if container.is_cancelled() {
                        return;
                    }

                    let ctx = EvalContext::from_external(ext.block_id, ext.sequence_hash);
                    let pass = self.evaluate_policies(&ctx).await;

                    if container.is_cancelled() {
                        return;
                    }

                    if pass {
                        if let Some(pending_guard) =
                            self.pending_tracker.try_claim(ext.sequence_hash)
                        {
                            passed.push(EvaluatedBlock::new(
                                SourceBlock::External(ext),
                                Some(pending_guard),
                            ));
                        } else {
                            filtered.push(ext.block_id);
                        }
                    } else {
                        filtered.push(ext.block_id);
                    }
                }
                tracing::debug!(%transfer_id, passed = passed.len(), filtered = filtered.len(), "External blocks evaluated");
            }
            SourceBlocks::Strong(strong_blocks) => {
                for block in strong_blocks {
                    if container.is_cancelled() {
                        return;
                    }

                    let ctx = EvalContext::new(block);
                    let pass = self.evaluate_policies(&ctx).await;

                    if container.is_cancelled() {
                        return;
                    }

                    if pass {
                        if let Some(pending_guard) =
                            self.pending_tracker.try_claim(ctx.sequence_hash)
                        {
                            let block = ctx.block.expect("Strong block context always has block");
                            passed.push(EvaluatedBlock::new(
                                SourceBlock::Strong(block),
                                Some(pending_guard),
                            ));
                        } else {
                            filtered.push(ctx.block_id);
                        }
                    } else {
                        filtered.push(ctx.block_id);
                    }
                }
            }
            SourceBlocks::Weak(weak_blocks) => {
                for weak in weak_blocks {
                    if container.is_cancelled() {
                        return;
                    }

                    let sequence_hash = weak.sequence_hash();
                    let ctx = EvalContext::from_weak(BlockId::default(), sequence_hash);
                    let pass = self.evaluate_policies(&ctx).await;

                    if container.is_cancelled() {
                        return;
                    }

                    if pass {
                        if let Some(pending_guard) = self.pending_tracker.try_claim(sequence_hash) {
                            passed.push(EvaluatedBlock::new(
                                SourceBlock::Weak(weak),
                                Some(pending_guard),
                            ));
                        } else {
                            tracing::debug!(
                                %transfer_id,
                                ?sequence_hash,
                                "Weak block filtered because another transfer owns its hash"
                            );
                        }
                    } else {
                        tracing::debug!(%transfer_id, ?sequence_hash, "Weak block filtered by policy");
                    }
                }
            }
        }

        if container.is_cancelled() {
            tracing::debug!(%transfer_id, "Transfer cancelled after evaluation");
            return;
        }

        tracing::debug!(%transfer_id, passed = passed.len(), filtered = filtered.len(), "Policy evaluation complete");

        // Update state with evaluation results
        {
            let mut state = state.lock().unwrap();
            state.add_passed(passed.iter().filter_map(|b| b.block_id));
            state.add_filtered(filtered.iter().copied());
            state.set_status(TransferStatus::Queued);
        }

        // Check if all blocks were filtered (transfer complete with no transfers)
        if passed.is_empty() {
            tracing::debug!(%transfer_id, "All blocks filtered, completing transfer");
            container.finish();
            return;
        }

        container.finish_evaluation(passed, filtered);
        if let Err(container) = self.output_queue.push_or_return(transfer_id, container) {
            let error = if self.output_queue.is_closed() {
                PRECOMMIT_SHUTDOWN_ERROR.to_string()
            } else {
                format!("policy output queue rejected {transfer_id}")
            };
            container.fail(error);
            tracing::debug!(%transfer_id, "Push to output queue failed (cancelled)");
        }
    }

    async fn evaluate_policies(&self, ctx: &EvalContext<T>) -> bool {
        for policy in &self.policies {
            let eval_future = policy.evaluate(ctx);
            let timed_result = tokio::time::timeout(self.timeout, async {
                match eval_future {
                    Either::Left(ready) => ready.await,
                    Either::Right(boxed) => boxed.await,
                }
            })
            .await;

            match timed_result {
                Ok(Ok(true)) => continue,
                Ok(Ok(false)) => return false,
                Ok(Err(e)) => {
                    tracing::warn!("Policy {} error: {}", policy.name(), e);
                    return false;
                }
                Err(_) => {
                    tracing::warn!("Policy {} timed out", policy.name());
                    return false;
                }
            }
        }
        true
    }
}

/// A batch of resolved blocks ready for transfer.
///
/// This is the output of the block upgrade stage and input to transfer executors.
pub(crate) struct ResolvedBatch<T: BlockMetadata> {
    /// Resolved blocks ready for transfer
    pub(crate) blocks: Vec<ResolvedBlock<T>>,
    /// Sequence hashes of blocks that were evicted during upgrade
    #[allow(dead_code)]
    pub evicted: Vec<SequenceHash>,
    /// Timing trace from the original batch (batch-level, not per-block)
    pub(crate) timing: TimingTrace,
    /// Opaque logical work units that stay live through physical completion.
    cancellation_units: Vec<ResolvedCancellationUnit>,
}

struct ResolvedCancellationUnit {
    transfer_id: TransferId,
    state: Arc<std::sync::Mutex<TransferState>>,
    cancellation: CancellationUnit,
}

impl ResolvedCancellationUnit {
    fn settle(self) {
        settle_transfer_unit(self.cancellation, self.state);
    }
}

impl<T: BlockMetadata> ResolvedBatch<T> {
    /// Check if the batch has any resolved blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Get the number of resolved blocks.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    fn take_cancellation_unit(&mut self, transfer_id: TransferId) -> Option<CancellationUnit> {
        let index = self
            .cancellation_units
            .iter()
            .position(|unit| unit.transfer_id == transfer_id)?;
        Some(self.cancellation_units.swap_remove(index).cancellation)
    }

    fn has_cancellation_unit(&self, transfer_id: TransferId) -> bool {
        self.cancellation_units
            .iter()
            .any(|unit| unit.transfer_id == transfer_id)
    }

    fn release_source_guards(&mut self) {
        self.blocks.clear();
    }

    fn settle_cancellation_units(&mut self) {
        for unit in self.cancellation_units.drain(..) {
            unit.settle();
        }
    }
}

/// Upgrade whole containers into physical blocks.
///
/// The final sweep runs directly before the commitment claims. Each claim
/// selects cancellation or commitment. Only committed containers flatten into
/// physical vectors after every selected container crosses the boundary.
pub(crate) fn upgrade_batch<T: BlockMetadata>(mut batch: TransferBatch<T>) -> ResolvedBatch<T> {
    let mut timing = std::mem::take(&mut batch.timing);
    timing.mark_transfer_start();

    let removed = batch.sweep_cancelled();
    if removed > 0 {
        tracing::debug!(removed, "Dropped cancelled containers before upgrade");
    }

    let mut upgraded_containers = Vec::with_capacity(batch.container_len());
    let mut evicted = Vec::new();
    for container in batch.containers {
        let Some(upgraded) = container.upgrade() else {
            tracing::debug!("Cancellation won the commitment claim");
            continue;
        };
        evicted.extend(upgraded.evicted.iter().copied());
        upgraded_containers.push(upgraded);
    }

    let mut blocks = Vec::new();
    let mut cancellation_units = Vec::with_capacity(upgraded_containers.len());
    for container in upgraded_containers {
        blocks.extend(container.blocks);
        cancellation_units.push(ResolvedCancellationUnit {
            transfer_id: container.transfer_id,
            state: container.state,
            cancellation: container.cancellation,
        });
    }

    ResolvedBatch {
        blocks,
        evicted,
        timing,
        cancellation_units,
    }
}

// ============================================================================
// Precondition Awaiter
// ============================================================================

/// Precondition awaiter stage.
///
/// It owns each complete container while it waits for a precondition. The
/// wait selects the container token, so cancellation drops the container.
pub(crate) struct PreconditionAwaiter<T: BlockMetadata> {
    input_queue: Arc<CancellableQueue<OffloadContainer<T>>>,
    output_queue: Arc<CancellableQueue<OffloadContainer<T>>>,
    leader: Arc<InstanceLeader>,
    max_concurrent_awaits: usize,
}

impl<T: BlockMetadata> PreconditionAwaiter<T> {
    pub(crate) fn new(
        input_queue: Arc<CancellableQueue<OffloadContainer<T>>>,
        output_queue: Arc<CancellableQueue<OffloadContainer<T>>>,
        leader: Arc<InstanceLeader>,
        max_concurrent_awaits: usize,
    ) -> Self {
        Self {
            input_queue,
            output_queue,
            leader,
            max_concurrent_awaits: max_concurrent_awaits.max(1),
        }
    }

    pub(crate) async fn run(self) {
        let input_queue = self.input_queue;
        let output_queue = self.output_queue;
        let leader = self.leader;
        let mut tasks = JoinSet::new();
        loop {
            if input_queue.is_closed() {
                while let Some(item) = input_queue.pop() {
                    item.data.fail(PRECOMMIT_SHUTDOWN_ERROR.to_string());
                }
                tasks.abort_all();
                while tasks.join_next().await.is_some() {}
                break;
            }

            while tasks.len() < self.max_concurrent_awaits
                && let Some(item) = input_queue.pop_valid()
            {
                let output_queue = output_queue.clone();
                let leader = leader.clone();
                tasks.spawn(async move {
                    Self::await_container(output_queue, leader, item.data).await;
                });
            }

            if tasks.is_empty() {
                input_queue.notified().await;
            } else {
                tokio::select! {
                    _ = input_queue.notified() => {}
                    result = tasks.join_next() => {
                        if let Some(result) = result {
                            Self::record_task_exit(result);
                        }
                    }
                }
            }
        }
    }

    fn record_task_exit(result: Result<(), JoinError>) {
        if let Err(error) = result {
            tracing::error!(%error, "Precondition await task stopped before completion");
        }
    }

    async fn await_container(
        output_queue: Arc<CancellableQueue<OffloadContainer<T>>>,
        leader: Arc<InstanceLeader>,
        container: OffloadContainer<T>,
    ) {
        nvtx_range!("offload::precondition");
        let transfer_id = container.transfer_id();
        if container.is_cancelled() {
            return;
        }

        if let Some(event_handle) = container.precondition() {
            tracing::debug!(%transfer_id, ?event_handle, "Awaiting container precondition");
            let velo = leader.messenger().clone();
            let awaiter = match velo.events().awaiter(event_handle) {
                Ok(awaiter) => awaiter,
                Err(error) => {
                    container.fail(format!("failed to create precondition awaiter: {error}"));
                    return;
                }
            };
            let cancel_token = container.cancellation_token();
            let outcome = tokio::select! {
                result = tokio::time::timeout(Duration::from_secs(300), awaiter) => Some(result),
                _ = cancel_token.wait_precommit_cancelled() => None,
            };
            match outcome {
                None => return,
                Some(Ok(Ok(()))) => {
                    tracing::debug!(%transfer_id, ?event_handle, "Precondition satisfied");
                }
                Some(Ok(Err(poison))) => {
                    container.fail(format!("precondition poisoned: {poison:?}"));
                    return;
                }
                Some(Err(_)) => {
                    container.fail("precondition timeout".to_string());
                    return;
                }
            }
        }

        if container.is_cancelled() {
            return;
        }
        if let Err(container) = output_queue.push_or_return(transfer_id, container) {
            let error = if output_queue.is_closed() {
                PRECOMMIT_SHUTDOWN_ERROR.to_string()
            } else {
                format!("precondition output queue rejected {transfer_id}")
            };
            container.fail(error);
            tracing::debug!(%transfer_id, "Precondition output rejected a cancelled container");
        }
    }
}

/// Record failure after a committed batch cannot finish, then settle its routes.
fn fail_resolved_batch<T: BlockMetadata>(batch: &mut ResolvedBatch<T>, error: &str) {
    let mut transfer_states: std::collections::HashMap<
        TransferId,
        (Arc<std::sync::Mutex<TransferState>>, Vec<BlockId>),
    > = std::collections::HashMap::new();
    for block in &batch.blocks {
        transfer_states
            .entry(block.transfer_id)
            .or_insert_with(|| (Arc::clone(&block.state), Vec::new()))
            .1
            .push(block.block_id);
    }

    batch.release_source_guards();

    for (state, block_ids) in transfer_states.into_values() {
        let mut state = state.lock().unwrap();
        state.mark_failed(block_ids);
        state.record_error(error.to_string());
    }
    batch.settle_cancellation_units();
}

#[cfg(test)]
mod tests;
