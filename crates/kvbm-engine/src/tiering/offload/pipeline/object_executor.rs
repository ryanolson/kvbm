// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use tokio::sync::{Semaphore, watch};

use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::blocks::BlockMetadata;

use crate::object::{ObjectBlockOps, ObjectLockManager};
use crate::{BlockId, SequenceHash};

use super::super::batch::BatchOutputRx;
use super::super::handle::{TransferId, TransferState, TransferStatus};
use super::shutdown::{
    CommittedPermit, ExecutorInput, ExecutorShutdown, PRECOMMIT_SHUTDOWN_ERROR,
    PipelineCommitmentGate, spawn_precommit_drainer,
};
use super::{ResolvedBatch, fail_resolved_batch, local_ownership};

// ============================================================================
// Object Transfer Executor (for G4 / object storage destinations)
// ============================================================================

/// Object transfer executor stage for object storage destinations.
///
/// Executes transfers to object storage (G4) via `ObjectBlockOps::put_blocks()`.
/// Unlike `BlockTransferExecutor`, this does not require a destination `BlockManager`.
///
/// # Source Requirements
///
/// The source blocks must be `ImmutableBlock<Src>` (post-upgrade). The executor:
/// 1. Receives `ResolvedBlock<Src>` from the upgrade stage
/// 2. Extracts `SequenceHash` as the object key
/// 3. Calls `ObjectBlockOps::put_blocks()` with the source layout
///
/// # Lock Management
///
/// When a `lock_manager` is provided, after successful transfers:
/// 1. Creates `.meta` file to mark block as offloaded
/// 2. Releases `.lock` file to allow other instances to proceed
///
/// # No Destination Registration
///
/// Object storage is external - there's no local `BlockManager<G4>` to register with.
/// The object is stored at the key derived from `SequenceHash`.
pub struct ObjectTransferExecutor<Src: BlockMetadata> {
    /// Input channel from the batch/precondition stage
    input_rx: BatchOutputRx<Src>,
    /// Object storage operations
    object_ops: Arc<dyn ObjectBlockOps>,
    /// Source logical layout handle for reading block data
    /// The ObjectBlockOps implementation resolves this to a physical layout
    src_layout: LogicalLayoutHandle,
    /// Skip actual transfers (for testing)
    skip_transfers: bool,
    /// Maximum concurrent transfer batches
    max_concurrent_transfers: usize,
    /// Optional lock manager for creating meta files and releasing locks
    lock_manager: Option<Arc<dyn ObjectLockManager>>,
    /// Releases queued precommit work after the pipeline owner drops.
    shutdown: ExecutorShutdown,
    /// Serializes this executor's upgrade with pipeline shutdown.
    commitment_gate: Arc<PipelineCommitmentGate>,
}

/// Shared state for ObjectTransferExecutor that can be cloned across concurrent tasks.
pub(super) struct SharedObjectExecutorState {
    pub(super) object_ops: Arc<dyn ObjectBlockOps>,
    pub(super) src_layout: LogicalLayoutHandle,
    pub(super) skip_transfers: bool,
    pub(super) lock_manager: Option<Arc<dyn ObjectLockManager>>,
}

impl<Src: BlockMetadata> ObjectTransferExecutor<Src> {
    /// Create a new object transfer executor.
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn new(
        input_rx: BatchOutputRx<Src>,
        object_ops: Arc<dyn ObjectBlockOps>,
        src_layout: LogicalLayoutHandle,
        skip_transfers: bool,
        max_concurrent_transfers: usize,
        lock_manager: Option<Arc<dyn ObjectLockManager>>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        Self::new_with_commitment_gate(
            input_rx,
            object_ops,
            src_layout,
            skip_transfers,
            max_concurrent_transfers,
            lock_manager,
            shutdown_rx,
            Arc::new(PipelineCommitmentGate::new()),
        )
    }

    pub(super) fn new_with_commitment_gate(
        input_rx: BatchOutputRx<Src>,
        object_ops: Arc<dyn ObjectBlockOps>,
        src_layout: LogicalLayoutHandle,
        skip_transfers: bool,
        max_concurrent_transfers: usize,
        lock_manager: Option<Arc<dyn ObjectLockManager>>,
        shutdown_rx: watch::Receiver<bool>,
        commitment_gate: Arc<PipelineCommitmentGate>,
    ) -> Self {
        Self {
            input_rx,
            object_ops,
            src_layout,
            skip_transfers,
            max_concurrent_transfers,
            lock_manager,
            shutdown: ExecutorShutdown::new(shutdown_rx),
            commitment_gate,
        }
    }

    /// Run the executor loop.
    pub async fn run(mut self) {
        let mut input_rx = Some(self.input_rx);
        let mut shutdown_drainer = None;
        // N slots for active transfers
        let transfer_semaphore = Arc::new(Semaphore::new(self.max_concurrent_transfers));
        // 1 slot for preparation (upgrade) work - on-deck
        let prepare_semaphore = Arc::new(Semaphore::new(1));

        // Extract shared state for concurrent tasks
        let shared = Arc::new(SharedObjectExecutorState {
            object_ops: self.object_ops.clone(),
            src_layout: self.src_layout,
            skip_transfers: self.skip_transfers,
            lock_manager: self.lock_manager.clone(),
        });

        loop {
            let batch = match self
                .shutdown
                .receive_batch(input_rx.as_mut().expect("executor input remains available"))
                .await
            {
                ExecutorInput::Batch(batch) => batch,
                ExecutorInput::Closed => break,
                ExecutorInput::Shutdown => {
                    shutdown_drainer = Some(spawn_precommit_drainer(
                        input_rx.take().expect("shutdown owns the executor input"),
                    ));
                    break;
                }
            };
            if batch.is_empty() {
                continue;
            }

            // Wait for prepare slot (only 1 batch preparing at a time)
            let prepare_permit = prepare_semaphore.clone().acquire_owned().await;
            if prepare_permit.is_err() {
                batch.fail(PRECOMMIT_SHUTDOWN_ERROR);
                break; // Semaphore closed
            }
            let prepare_permit = prepare_permit.unwrap();

            // Prepare stage: resolve/upgrade blocks (weak→strong).
            // The shared gate selects shutdown or commitment for this batch.
            let mut upgraded = match self.shutdown.upgrade_batch(&self.commitment_gate, batch) {
                Ok(upgraded) => upgraded,
                Err(batch) => {
                    batch.fail(PRECOMMIT_SHUTDOWN_ERROR);
                    shutdown_drainer = Some(spawn_precommit_drainer(
                        input_rx.take().expect("shutdown owns the executor input"),
                    ));
                    break;
                }
            };

            // Done preparing, release prepare slot for next batch
            drop(prepare_permit);

            if upgraded.is_empty() {
                tracing::debug!("All blocks in batch evicted, skipping object transfer");
                upgraded.settle_cancellation_units();
                continue;
            }

            // Now wait for transfer slot
            let transfer_permit = match self
                .shutdown
                .acquire_committed_permit(Arc::clone(&transfer_semaphore))
                .await
            {
                CommittedPermit::Acquired(permit) => permit,
                CommittedPermit::Shutdown => {
                    shutdown_drainer = Some(spawn_precommit_drainer(
                        input_rx.take().expect("shutdown owns the executor input"),
                    ));
                    match Arc::clone(&transfer_semaphore).acquire_owned().await {
                        Ok(permit) => permit,
                        Err(_) => {
                            fail_resolved_batch(
                                &mut upgraded,
                                "object executor stopped after commitment",
                            );
                            break;
                        }
                    }
                }
            };

            // Spawn transfer task
            let shared_clone = shared.clone();
            tokio::spawn(async move {
                let _permit = transfer_permit; // Hold permit until task completes
                if let Err(e) = Self::execute_transfer(&shared_clone, &mut upgraded).await {
                    tracing::error!("ObjectTransferExecutor: transfer failed: {}", e);
                    fail_resolved_batch(&mut upgraded, &e.to_string());
                }
            });

            if shutdown_drainer.is_some() {
                break;
            }
        }

        // Wait for all in-flight transfers to complete by acquiring all permits
        let _ = transfer_semaphore
            .acquire_many(self.max_concurrent_transfers as u32)
            .await;
        if let Some(drainer) = shutdown_drainer {
            let _ = drainer.await;
        }
    }

    /// Execute the actual transfer for resolved blocks to object storage.
    pub(super) async fn execute_transfer(
        shared: &SharedObjectExecutorState,
        batch: &mut ResolvedBatch<Src>,
    ) -> anyhow::Result<()> {
        nvtx_range!("offload::transfer");
        if batch.is_empty() {
            return Ok(());
        }

        // Collect keys (sequence hashes) and block_ids from resolved blocks
        let keys: Vec<SequenceHash> = batch.blocks.iter().map(|b| b.sequence_hash).collect();
        let block_ids: Vec<BlockId> = batch.blocks.iter().map(|b| b.block_id).collect();

        // Collect states for completion tracking (group by transfer_id)
        let mut transfer_states: std::collections::HashMap<
            TransferId,
            (Arc<std::sync::Mutex<TransferState>>, Vec<BlockId>),
        > = std::collections::HashMap::new();
        for block in &batch.blocks {
            transfer_states
                .entry(block.transfer_id)
                .or_insert_with(|| (block.state.clone(), Vec::new()))
                .1
                .push(block.block_id);
        }

        // Track successfully transferred sequence hashes for lock management
        let mut successful_hashes: Vec<SequenceHash> = Vec::new();

        // Skip actual transfers when in test mode
        if !shared.skip_transfers {
            for (state, _) in transfer_states.values() {
                state
                    .lock()
                    .unwrap()
                    .set_status(TransferStatus::Transferring);
            }
            // Arm ownership before the first object-upload future poll.
            let ownership = local_ownership::LocalPhysicalOwnership::new(batch, ());
            let results = shared
                .object_ops
                .put_blocks(keys.clone(), shared.src_layout, block_ids)
                .await;
            ownership.release(batch);

            // Guard: put_blocks must return exactly one result per input block.
            // If the counts differ, mark all blocks as failed. The results cannot map to the inputs.
            if results.len() != keys.len() {
                tracing::error!(
                    expected = keys.len(),
                    actual = results.len(),
                    "put_blocks returned mismatched result count"
                );
                fail_resolved_batch(batch, "put_blocks returned mismatched result count");
                return Ok(());
            }

            // Log results and track successful transfers
            let mut success_count = 0;
            let mut fail_count = 0;

            for result in results {
                match result {
                    Ok(hash) => {
                        success_count += 1;
                        successful_hashes.push(hash);
                    }
                    Err(hash) => {
                        fail_count += 1;
                        tracing::warn!(?hash, "Failed to transfer block to object storage");
                    }
                }
            }

            if fail_count > 0 {
                tracing::warn!(
                    success = success_count,
                    failed = fail_count,
                    "Object transfer partially failed"
                );
            } else {
                tracing::debug!(
                    num_blocks = success_count,
                    "Successfully transferred blocks to object storage"
                );
            }

            // TODO: Merge the other branch of this condition. Add the event tap for successful transfers.
            // Block transfer registration emits an event. G4 block registration does not emit an event.
            // The system needs a convention to announce object creation.

            // Create meta files and release locks for successful transfers
            if let Some(lock_manager) = &shared.lock_manager {
                for hash in &successful_hashes {
                    // Create meta file to mark block as offloaded
                    if let Err(e) = lock_manager.create_meta(*hash).await {
                        tracing::error!(?hash, error = %e, "Failed to create meta file");
                    }

                    // Release lock
                    if let Err(e) = lock_manager.release_lock(*hash).await {
                        tracing::error!(?hash, error = %e, "Failed to release lock");
                    }
                }
                tracing::debug!(
                    num_blocks = successful_hashes.len(),
                    "Created meta files and released locks"
                );
            }
        } else {
            for (state, _) in transfer_states.values() {
                state
                    .lock()
                    .unwrap()
                    .set_status(TransferStatus::Transferring);
            }
            // In skip mode, still do lock management if configured
            if let Some(lock_manager) = &shared.lock_manager {
                for hash in &keys {
                    if let Err(e) = lock_manager.create_meta(*hash).await {
                        tracing::error!(?hash, error = %e, "Failed to create meta file");
                    }
                    if let Err(e) = lock_manager.release_lock(*hash).await {
                        tracing::error!(?hash, error = %e, "Failed to release lock");
                    }
                }
            }
        }

        let resolved = &batch.blocks;

        // Mark physical transfer completion for batch timing.
        batch.timing.mark_transfer_complete();

        // Compute timing statistics from batch timing
        let unique_transfer_ids: std::collections::HashSet<_> =
            resolved.iter().map(|b| b.transfer_id).collect();

        let policy_us = batch
            .timing
            .policy_duration()
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        let precondition_us = batch
            .timing
            .precondition_duration()
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        let transfer_us = batch
            .timing
            .transfer_duration()
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        let total_us = batch
            .timing
            .total_duration()
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        tracing::info!(
            blocks = resolved.len(),
            containers = unique_transfer_ids.len(),
            policy_us,
            precondition_us,
            transfer_us,
            total_us,
            src = std::any::type_name::<Src>(),
            dst = "G4-object",
            "Object batch transfer complete"
        );

        // Build success lookup for filtering completion tracking.
        //
        // INVARIANT: SequenceHash values within a batch are unique. PolicyEvaluator
        // calls PendingTracker::try_claim after every policy pass. DashSet returns one
        // winner, and only that winner creates an EvaluatedBlock. Its PendingGuard stays
        // live through this executor. A losing claim is filtered before batching. This
        // makes hash-based object result correlation unambiguous.
        let block_to_hash: std::collections::HashMap<BlockId, SequenceHash> = resolved
            .iter()
            .map(|b| (b.block_id, b.sequence_hash))
            .collect();
        let success_set: std::collections::HashSet<SequenceHash> =
            successful_hashes.into_iter().collect();

        debug_assert_eq!(
            block_to_hash.len(),
            resolved.len(),
            "duplicate BlockId in batch — block_to_hash would lose entries"
        );
        debug_assert_eq!(
            resolved
                .iter()
                .map(|b| b.sequence_hash)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            resolved.len(),
            "duplicate SequenceHash in batch — hash-based success correlation is ambiguous"
        );

        // Release source and pending guards before any terminal status update.
        batch.release_source_guards();

        // Record route progress before logical settlement.
        for (transfer_id, (state, block_ids)) in transfer_states {
            let mut state_guard = state.lock().unwrap();

            if shared.skip_transfers {
                // In test/skip mode, all blocks are considered successful
                state_guard.mark_completed(block_ids);
            } else {
                let (succeeded, failed): (Vec<_>, Vec<_>) = block_ids.into_iter().partition(|id| {
                    block_to_hash
                        .get(id)
                        .is_some_and(|h| success_set.contains(h))
                });
                state_guard.mark_completed(succeeded);
                if !failed.is_empty() {
                    let failed_count = failed.len();
                    tracing::warn!(
                        %transfer_id,
                        failed_count,
                        "Marking blocks as failed in transfer state"
                    );
                    state_guard.mark_failed(failed);
                    state_guard.record_error(format!(
                        "{failed_count} blocks failed to transfer to object storage",
                    ));
                }
            }

            let total = state_guard.passed_blocks.len() + state_guard.filtered_out.len();
            let done = state_guard.completed.len()
                + state_guard.failed.len()
                + state_guard.filtered_out.len();
            tracing::debug!(
                %transfer_id,
                total,
                done,
                passed = state_guard.passed_blocks.len(),
                filtered = state_guard.filtered_out.len(),
                completed = state_guard.completed.len(),
                failed = state_guard.failed.len(),
                "Object transfer batch progress"
            );
        }
        batch.settle_cancellation_units();

        Ok(())
    }
}
