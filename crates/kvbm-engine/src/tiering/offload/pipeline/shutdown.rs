// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared shutdown control for block and object transfer executors.

use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};
use tokio::task::JoinHandle;

use kvbm_logical::blocks::BlockMetadata;

use super::super::batch::{BatchOutputRx, TransferBatch};

pub(crate) const PRECOMMIT_SHUTDOWN_ERROR: &str = "pipeline shut down before commitment";

/// Serialize pipeline shutdown with the weak-to-strong commitment boundary.
pub(super) struct PipelineCommitmentGate {
    inner: Mutex<()>,
}

impl PipelineCommitmentGate {
    pub(super) fn new() -> Self {
        Self {
            inner: Mutex::new(()),
        }
    }

    /// Publish durable shutdown before closing any stage queue.
    pub(super) fn publish_shutdown<T>(
        &self,
        shutdown_tx: &watch::Sender<bool>,
        close_queues: impl FnOnce() -> T,
    ) -> T {
        let _commitment = self.inner.lock();
        shutdown_tx.send_replace(true);
        close_queues()
    }
}

/// Own the executor-side shutdown notification.
pub(super) struct ExecutorShutdown {
    receiver: watch::Receiver<bool>,
}

/// Result of receiving executor input while shutdown remains observable.
pub(super) enum ExecutorInput<Src: BlockMetadata> {
    Batch(TransferBatch<Src>),
    Closed,
    Shutdown,
}

/// Result of waiting for a permit after a batch commits.
pub(super) enum CommittedPermit {
    Acquired(OwnedSemaphorePermit),
    Shutdown,
}

impl ExecutorShutdown {
    pub(super) fn new(receiver: watch::Receiver<bool>) -> Self {
        Self { receiver }
    }

    /// Receive a batch unless shutdown starts first.
    pub(super) async fn receive_batch<Src: BlockMetadata>(
        &mut self,
        input_rx: &mut BatchOutputRx<Src>,
    ) -> ExecutorInput<Src> {
        if self.requested() {
            return ExecutorInput::Shutdown;
        }

        let batch = tokio::select! {
            batch = input_rx.recv() => batch,
            changed = self.receiver.changed() => {
                let _ = changed;
                return ExecutorInput::Shutdown;
            }
        };
        if self.requested() {
            if let Some(batch) = batch {
                batch.fail(PRECOMMIT_SHUTDOWN_ERROR);
            }
            return ExecutorInput::Shutdown;
        }
        match batch {
            Some(batch) => ExecutorInput::Batch(batch),
            None => ExecutorInput::Closed,
        }
    }

    /// Commit a batch only while pipeline shutdown remains unrequested.
    ///
    /// This holds the shared gate through the complete synchronous upgrade.
    /// If shutdown already won, it returns the original batch for failure after
    /// the gate releases.
    pub(super) fn upgrade_batch<Src: BlockMetadata>(
        &self,
        commitment_gate: &PipelineCommitmentGate,
        batch: TransferBatch<Src>,
    ) -> Result<super::ResolvedBatch<Src>, TransferBatch<Src>> {
        let _commitment = commitment_gate.inner.lock();
        if self.requested() {
            return Err(batch);
        }
        Ok(super::upgrade_batch(batch))
    }

    /// Wait for a committed batch permit while shutdown remains observable.
    pub(super) async fn acquire_committed_permit(
        &mut self,
        semaphore: Arc<Semaphore>,
    ) -> CommittedPermit {
        if self.requested() {
            return CommittedPermit::Shutdown;
        }

        tokio::select! {
            permit = Arc::clone(&semaphore).acquire_owned() => {
                match permit {
                    Ok(permit) => CommittedPermit::Acquired(permit),
                    Err(_) => CommittedPermit::Shutdown,
                }
            }
            changed = self.receiver.changed() => {
                let _ = changed;
                CommittedPermit::Shutdown
            }
        }
    }

    fn requested(&self) -> bool {
        *self.receiver.borrow()
    }
}

/// Close the receiver and fail every precommit batch until every sender closes.
pub(super) fn spawn_precommit_drainer<Src: BlockMetadata>(
    mut input_rx: BatchOutputRx<Src>,
) -> JoinHandle<()> {
    input_rx.close();
    tokio::spawn(async move {
        while let Some(batch) = input_rx.recv().await {
            batch.fail(PRECOMMIT_SHUTDOWN_ERROR);
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::super::super::batch::TransferBatch;
    use super::super::super::container::{EvaluatedBlock, OffloadContainer};
    use super::super::super::source::{ExternalBlock, SourceBlock, SourceBlocks};
    use super::*;
    use crate::offload::handle::{TransferId, TransferState, TransferStatus};
    use crate::{BlockId, G2, SequenceHash};
    use tokio::sync::mpsc;

    fn evaluated_batch(
        transfer_id: TransferId,
        block_id: BlockId,
        state: Arc<std::sync::Mutex<TransferState>>,
    ) -> TransferBatch<G2> {
        let sequence_hash = SequenceHash::new(block_id as u64, None, 0);
        let block = ExternalBlock::new(block_id, sequence_hash);
        let mut container = OffloadContainer::new(
            transfer_id,
            SourceBlocks::External(vec![block]),
            state,
            None,
        );
        let SourceBlocks::External(mut blocks) = container.take_source() else {
            unreachable!("the test container has external source blocks");
        };
        let block = blocks
            .pop()
            .expect("the test container has one source block");
        container.finish_evaluation(
            vec![EvaluatedBlock::new(SourceBlock::External(block), None)],
            Vec::new(),
        );
        TransferBatch::from_containers(vec![container])
    }

    #[tokio::test]
    async fn drainer_fails_a_batch_sent_through_a_reserved_permit() {
        let transfer_id = TransferId::new();
        let (state, mut handle) = TransferState::new(transfer_id, vec![1]);
        let state = Arc::new(std::sync::Mutex::new(state));
        let batch = TransferBatch::from_containers(vec![OffloadContainer::new(
            transfer_id,
            SourceBlocks::External(vec![ExternalBlock::<G2>::new(
                1,
                SequenceHash::new(91, None, 0),
            )]),
            state,
            None,
        )]);
        let (sender, receiver) = mpsc::channel(1);
        let permit = sender
            .clone()
            .reserve_owned()
            .await
            .expect("reserve before receiver shutdown");
        let drainer = spawn_precommit_drainer(receiver);

        permit.send(batch);

        let result = tokio::time::timeout(Duration::from_millis(250), handle.wait())
            .await
            .expect("reserved late batch must settle")
            .expect("transfer result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(result.error.as_deref(), Some(PRECOMMIT_SHUTDOWN_ERROR));
        drop(sender);
        drainer.await.expect("drainer exits after sender close");
    }

    #[tokio::test]
    async fn shutdown_wins_before_executor_commitment() {
        let commitment_gate = PipelineCommitmentGate::new();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let executor_shutdown = ExecutorShutdown::new(shutdown_rx);
        let transfer_id = TransferId::new();
        let (state, mut handle) = TransferState::new(transfer_id, vec![17]);
        let state = Arc::new(std::sync::Mutex::new(state));
        let batch = evaluated_batch(transfer_id, 17, Arc::clone(&state));

        commitment_gate.publish_shutdown(&shutdown_tx, || {});
        let batch = match executor_shutdown.upgrade_batch(&commitment_gate, batch) {
            Ok(_) => panic!("shutdown must reject a precommit batch"),
            Err(batch) => batch,
        };
        assert!(
            !state.lock().unwrap().committed,
            "shutdown rejection leaves the batch uncommitted"
        );

        batch.fail(PRECOMMIT_SHUTDOWN_ERROR);
        let result = handle
            .wait()
            .await
            .expect("shutdown rejection publishes a transfer result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(result.error.as_deref(), Some(PRECOMMIT_SHUTDOWN_ERROR));
    }

    #[tokio::test]
    async fn executor_commitment_wins_before_shutdown() {
        let commitment_gate = PipelineCommitmentGate::new();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let executor_shutdown = ExecutorShutdown::new(shutdown_rx);
        let transfer_id = TransferId::new();
        let (state, mut handle) = TransferState::new(transfer_id, vec![18]);
        let state = Arc::new(std::sync::Mutex::new(state));
        let batch = evaluated_batch(transfer_id, 18, Arc::clone(&state));

        let mut resolved = match executor_shutdown.upgrade_batch(&commitment_gate, batch) {
            Ok(resolved) => resolved,
            Err(_) => panic!("the executor must commit before shutdown starts"),
        };
        assert!(
            state.lock().unwrap().committed,
            "the winner crosses the commitment boundary"
        );

        commitment_gate.publish_shutdown(&shutdown_tx, || {});
        assert!(
            state.lock().unwrap().committed,
            "shutdown cannot revoke an existing commitment"
        );

        resolved.release_source_guards();
        state.lock().unwrap().mark_completed([18]);
        resolved.settle_cancellation_units();
        let result = handle
            .wait()
            .await
            .expect("committed work settles after shutdown");
        assert_eq!(result.status, TransferStatus::Complete);
    }

    #[tokio::test]
    async fn shutdown_rejection_settles_a_committed_chained_route() {
        let commitment_gate = PipelineCommitmentGate::new();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let executor_shutdown = ExecutorShutdown::new(shutdown_rx);
        let transfer_id = TransferId::new();
        let (mut state, mut handle) = TransferState::new(transfer_id, vec![19]);
        let cancellation_token = state.cancellation_token();
        let upstream = cancellation_token
            .root_unit()
            .expect("the upstream route owns its root unit");
        assert!(upstream.claim_commitment());
        state.mark_committed();
        let state = Arc::new(std::sync::Mutex::new(state));
        let mut children = upstream.fan_out(1);
        let batch = TransferBatch::from_containers(vec![OffloadContainer::with_cancellation(
            transfer_id,
            SourceBlocks::External(vec![ExternalBlock::<G2>::new(
                19,
                SequenceHash::new(19, None, 0),
            )]),
            Arc::clone(&state),
            None,
            children
                .pop()
                .expect("the downstream route owns one child unit"),
        )]);

        commitment_gate.publish_shutdown(&shutdown_tx, || {});
        let batch = match executor_shutdown.upgrade_batch(&commitment_gate, batch) {
            Ok(_) => panic!("shutdown must reject the downstream precommit batch"),
            Err(batch) => batch,
        };
        let confirmation = handle.cancel();
        batch.fail(PRECOMMIT_SHUTDOWN_ERROR);

        tokio::time::timeout(Duration::from_millis(250), confirmation.wait())
            .await
            .expect("failed downstream route settles its child unit");
        let result = handle
            .wait()
            .await
            .expect("failed downstream route publishes a transfer result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(result.error.as_deref(), Some(PRECOMMIT_SHUTDOWN_ERROR));
    }
}
