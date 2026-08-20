// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{Semaphore, mpsc};

use kvbm_common::{LogicalLayoutHandle, LogicalResourceId};
use kvbm_logical::blocks::{BlockMetadata, ImmutableBlock};
use kvbm_physical::transfer::TransferOptions;

use crate::leader::InstanceLeader;
use crate::{BlockId, SequenceHash};

use super::super::batch::BatchOutputRx;
use super::super::destination::BlockDestination;
use super::super::handle::{TransferId, TransferState, TransferStatus, fail_transfer_unit};
use super::runtime::{ChainOutput, RegisterObserver, RegisterObservers};
use super::shutdown::{
    CommittedPermit, ExecutorInput, ExecutorShutdown, PRECOMMIT_SHUTDOWN_ERROR,
    PipelineCommitmentGate, spawn_precommit_drainer,
};
use super::{ResolvedBatch, fail_resolved_batch, local_ownership};

// ============================================================================
// Block Transfer Executor (for G2, G3 destinations)
// ============================================================================

/// Block transfer executor stage for BlockManager-based destinations.
///
/// Executes transfers to destinations with a `BlockManager` (G2, G3).
/// Uses `leader.execute_local_transfer()` to copy block data between layouts.
///
/// For object storage destinations (G4), use `ObjectTransferExecutor` instead.
pub(super) struct BlockTransferExecutor<Src: BlockMetadata, Dst: BlockMetadata> {
    pub(super) input_rx: BatchOutputRx<Src>,
    pub(super) leader: Arc<InstanceLeader>,
    pub(super) destination: Arc<dyn BlockDestination<Dst>>,
    pub(super) resource: Option<LogicalResourceId>,
    pub(super) src_layout: LogicalLayoutHandle,
    pub(super) dst_layout: LogicalLayoutHandle,
    /// Skip actual transfers (for testing)
    pub(super) skip_transfers: bool,
    /// Maximum concurrent transfers
    pub(super) max_concurrent_transfers: usize,
    /// Channel to send registered blocks for chaining to downstream pipeline
    pub(super) chain_tx: Option<mpsc::Sender<ChainOutput<Dst>>>,
    /// Multicast observers invoked after each batch's register step.
    pub(super) register_observers: Arc<RegisterObservers<Dst>>,
    /// Releases queued precommit work after the pipeline owner drops.
    pub(super) shutdown: ExecutorShutdown,
    /// Serializes this executor's upgrade with pipeline shutdown.
    pub(super) commitment_gate: Arc<PipelineCommitmentGate>,
    pub(super) _src_marker: PhantomData<Src>,
}

/// Shared state for BlockTransferExecutor that can be cloned across concurrent tasks.
struct SharedBlockExecutorState<Dst: BlockMetadata> {
    leader: Arc<InstanceLeader>,
    destination: Arc<dyn BlockDestination<Dst>>,
    resource: Option<LogicalResourceId>,
    src_layout: LogicalLayoutHandle,
    dst_layout: LogicalLayoutHandle,
    skip_transfers: bool,
    chain_tx: Option<mpsc::Sender<ChainOutput<Dst>>>,
    register_observers: Arc<RegisterObservers<Dst>>,
}

impl<Src: BlockMetadata, Dst: BlockMetadata> BlockTransferExecutor<Src, Dst> {
    pub(super) async fn run(mut self) {
        let mut input_rx = Some(self.input_rx);
        let mut shutdown_drainer = None;
        // N slots for active transfers
        let transfer_semaphore = Arc::new(Semaphore::new(self.max_concurrent_transfers));
        // 1 slot for preparation (upgrade) work - on-deck
        let prepare_semaphore = Arc::new(Semaphore::new(1));

        // Extract shared state for concurrent tasks
        let shared = Arc::new(SharedBlockExecutorState {
            leader: self.leader.clone(),
            destination: self.destination.clone(),
            resource: self.resource,
            src_layout: self.src_layout,
            dst_layout: self.dst_layout,
            skip_transfers: self.skip_transfers,
            chain_tx: self.chain_tx.take(),
            register_observers: Arc::clone(&self.register_observers),
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
            // This is the "on-deck" slot for preparing while transfers run
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
                tracing::debug!("All blocks in batch evicted, skipping transfer");
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
                                "transfer executor stopped after commitment",
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
                    tracing::error!("BlockTransferExecutor: transfer failed: {}", e);
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

    /// Execute the actual transfer for resolved blocks.
    ///
    /// This is async I/O work that runs concurrently with other transfers.
    async fn execute_transfer(
        shared: &SharedBlockExecutorState<Dst>,
        batch: &mut ResolvedBatch<Src>,
    ) -> anyhow::Result<()> {
        nvtx_range!("offload::transfer");
        if batch.is_empty() {
            return Ok(());
        }

        // Collect block_ids and sequence_hashes from resolved blocks
        let src_block_ids: Vec<BlockId> = batch.blocks.iter().map(|b| b.block_id).collect();
        let sequence_hashes: Vec<SequenceHash> =
            batch.blocks.iter().map(|b| b.sequence_hash).collect();

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
        let mut chain_outputs_to_send = Vec::new();

        // Skip actual transfers when in test mode
        if !shared.skip_transfers {
            // Allocate destination blocks
            let dst_allocation = shared
                .destination
                .allocate(batch.blocks.len())?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Failed to allocate {} destination blocks",
                        batch.blocks.len()
                    )
                })?;

            let dst_block_ids = dst_allocation.block_ids();

            // Own all physical resources before dispatch. A synchronous
            // dispatch error restores them for normal failure handling.
            let ownership = local_ownership::LocalPhysicalOwnership::new(batch, dst_allocation);

            // Execute transfer via leader
            let start_xfer = Instant::now();
            let dispatch = match shared.resource {
                Some(resource) => shared.leader.execute_local_transfer_for_resource(
                    resource,
                    shared.src_layout,
                    shared.dst_layout,
                    src_block_ids.clone(),
                    dst_block_ids.clone(),
                    TransferOptions::default(),
                ),
                None => shared.leader.execute_local_transfer(
                    shared.src_layout,
                    shared.dst_layout,
                    src_block_ids.clone(),
                    dst_block_ids.clone(),
                    TransferOptions::default(),
                ),
            };
            let notification = match dispatch {
                Ok(notification) => notification,
                Err(error) => {
                    drop(ownership.release(batch));
                    return Err(error);
                }
            };

            // `execute_local_transfer` reserves the physical/simulated
            // transfer synchronously and returns an awaitable completion
            // notification. Only now is the handle a safe marker for
            // virtual-time accounting.
            for (state, _) in transfer_states.values() {
                let mut state_guard = state.lock().unwrap();
                state_guard.set_status(TransferStatus::Transferring);
            }

            // The guard retains source, destination, and cancellation state if
            // the receipt cannot prove drain or this task stops.
            let dst_allocation =
                local_ownership::await_local_drain(batch, ownership, notification).await?;
            let end_xfer = Instant::now();

            let resolved = &batch.blocks;

            // Register each transferred block in the destination tier
            let registered_blocks: Vec<ImmutableBlock<Dst>> =
                dst_allocation.register(&sequence_hashes)?;

            // Fan out to register observers (e.g. CD prefill capture).
            // Snapshot the observer list under the lock, then invoke
            // outside the lock so observers cannot block the pipeline.
            let observers: Vec<RegisterObserver<Dst>> = {
                let guard = shared.register_observers.lock();
                guard.iter().cloned().collect()
            };
            for observer in &observers {
                observer(&registered_blocks);
            }

            // Audit emit: blocks just landed in the destination tier. Lets
            // smokes / external observers discover the sequence hashes
            // that were just made available (e.g. P2P smoke uses these to
            // drive an open_session / pull_from_session pair).
            //
            // Hashes are emitted as 32-hex-char big-endian u128 values
            // (the on-wire serde shape is 16 bytes BE, matching
            // `u128::to_be_bytes`). Pythons / scripts can `bytes.fromhex`
            // each comma-separated entry to rebuild the JSON byte array
            // the hub's `SequenceHash` deserializer accepts. The human
            // Display form (`pos:b58:lineage_b58`) is lossless but more
            // work to parse — hex is the machine surface.
            if !sequence_hashes.is_empty() {
                let hashes_hex = sequence_hashes
                    .iter()
                    .map(|h| format!("{:032x}", h.as_u128()))
                    .collect::<Vec<_>>()
                    .join(",");
                crate::engine_audit!(
                    "offload_register_complete",
                    src = std::any::type_name::<Src>(),
                    dst = std::any::type_name::<Dst>(),
                    num_blocks = registered_blocks.len(),
                    sequence_hashes_hex = hashes_hex
                );
            }

            let registration_timepoint = Instant::now();

            // Compute timing statistics from batch timing (O(1), not per-block)
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
                xfer_us = end_xfer.duration_since(start_xfer).as_micros() as u64,
                registration_us =
                    registration_timepoint.duration_since(end_xfer).as_micros() as u64,
                total_us,
                src = std::any::type_name::<Src>(),
                dst = std::any::type_name::<Dst>(),
                "Batch transfer complete"
            );

            // Send registered blocks to downstream pipeline if chaining is enabled
            if shared.chain_tx.is_some() {
                #[allow(clippy::type_complexity)]
                let mut chain_outputs: std::collections::HashMap<
                    TransferId,
                    (
                        Arc<std::sync::Mutex<TransferState>>,
                        Vec<ImmutableBlock<Dst>>,
                    ),
                > = std::collections::HashMap::new();

                for (registered, resolved_block) in
                    registered_blocks.into_iter().zip(resolved.iter())
                {
                    chain_outputs
                        .entry(resolved_block.transfer_id)
                        .or_insert_with(|| (resolved_block.state.clone(), Vec::new()))
                        .1
                        .push(registered);
                }

                if let Some(transfer_id) = chain_outputs
                    .keys()
                    .find(|transfer_id| !batch.has_cancellation_unit(**transfer_id))
                {
                    return Err(anyhow::anyhow!(
                        "missing cancellation unit for chained transfer {transfer_id}",
                    ));
                }

                for (transfer_id, (state, blocks)) in chain_outputs {
                    let cancellation = batch
                        .take_cancellation_unit(transfer_id)
                        .expect("the preflight check found each chain cancellation unit");
                    let output = ChainOutput {
                        transfer_id,
                        blocks,
                        state,
                        cancellation,
                    };
                    chain_outputs_to_send.push(output);
                }
            }
        } else {
            for (state, _) in transfer_states.values() {
                let mut state_guard = state.lock().unwrap();
                state_guard.set_status(TransferStatus::Transferring);
            }
        }

        // Terminal status never becomes visible while source or pending guards remain.
        batch.release_source_guards();
        // Mark physical transfer completion for batch timing.
        batch.timing.mark_transfer_complete();

        // Record source-route progress before downstream work can finish.
        for (transfer_id, (state, block_ids)) in transfer_states {
            let mut state_guard = state.lock().unwrap();
            state_guard.mark_completed(block_ids);

            let total = state_guard.passed_blocks.len() + state_guard.filtered_out.len();
            let done = state_guard.completed.len() + state_guard.filtered_out.len();
            tracing::debug!(
                %transfer_id,
                total,
                done,
                passed = state_guard.passed_blocks.len(),
                filtered = state_guard.filtered_out.len(),
                completed = state_guard.completed.len(),
                "Transfer batch progress"
            );
        }

        if let Some(chain_tx) = &shared.chain_tx {
            for output in chain_outputs_to_send {
                let transfer_id = output.transfer_id;
                match chain_tx.send(output).await {
                    Ok(()) => {
                        tracing::debug!(
                            %transfer_id,
                            "Sent blocks to chain output for downstream processing"
                        );
                    }
                    Err(error) => {
                        let ChainOutput {
                            blocks,
                            state,
                            cancellation,
                            ..
                        } = error.0;
                        drop(blocks);
                        fail_transfer_unit(
                            cancellation,
                            state,
                            format!("chain channel closed for transfer {transfer_id}"),
                        );
                        tracing::warn!(
                            %transfer_id,
                            "Chain channel closed, downstream pipeline unavailable"
                        );
                    }
                }
            }
        }
        batch.settle_cancellation_units();

        Ok(())
    }
}
