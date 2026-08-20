// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ownership for local physical work after dispatch.

use kvbm_logical::blocks::BlockMetadata;
use kvbm_physical::transfer::{TransferCompleteNotification, TransferDrainOutcome};

use super::super::container::ResolvedBlock;
use super::ResolvedBatch;

const ABANDONED_LOCAL_PHYSICAL_ERROR: &str =
    "local physical ownership ended without proven physical drain";

/// Owns every resource that local physical work can still access.
///
/// Construction moves committed route ownership out of `ResolvedBatch`.
/// A proven drain restores the route. An ambiguous exit retains all resources
/// for the process lifetime.
pub(super) struct LocalPhysicalOwnership<T: BlockMetadata, R> {
    blocks: Vec<ResolvedBlock<T>>,
    cancellation_units: Vec<super::ResolvedCancellationUnit>,
    resource: Option<R>,
}

impl<T: BlockMetadata, R> LocalPhysicalOwnership<T, R> {
    pub(super) fn new(batch: &mut ResolvedBatch<T>, resource: R) -> Self {
        Self {
            blocks: std::mem::take(&mut batch.blocks),
            cancellation_units: std::mem::take(&mut batch.cancellation_units),
            resource: Some(resource),
        }
    }

    pub(super) fn release(mut self, batch: &mut ResolvedBatch<T>) -> R {
        debug_assert!(batch.blocks.is_empty());
        debug_assert!(batch.cancellation_units.is_empty());
        batch.blocks = std::mem::take(&mut self.blocks);
        batch.cancellation_units = std::mem::take(&mut self.cancellation_units);
        self.resource
            .take()
            .expect("local physical ownership holds its destination resource")
    }

    pub(super) fn retain_unproven(mut self, error: String) {
        self.record_error(error);
        self.retain_for_process_lifetime();
    }

    fn record_error(&self, error: String) {
        for unit in &self.cancellation_units {
            unit.state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .record_error(error.clone());
        }
    }

    fn retain_for_process_lifetime(&mut self) {
        std::mem::forget(std::mem::take(&mut self.blocks));
        std::mem::forget(std::mem::take(&mut self.cancellation_units));
        if let Some(resource) = self.resource.take() {
            std::mem::forget(resource);
        }
    }
}

impl<T: BlockMetadata, R> Drop for LocalPhysicalOwnership<T, R> {
    fn drop(&mut self) {
        if self.resource.is_none() {
            return;
        }
        self.record_error(ABANDONED_LOCAL_PHYSICAL_ERROR.to_string());
        self.retain_for_process_lifetime();
    }
}

/// Await one local block-transfer receipt under complete physical ownership.
pub(super) async fn await_local_drain<T: BlockMetadata, R>(
    batch: &mut ResolvedBatch<T>,
    ownership: LocalPhysicalOwnership<T, R>,
    notification: TransferCompleteNotification,
) -> anyhow::Result<R> {
    match notification.await_drain().await {
        TransferDrainOutcome::Completed => Ok(ownership.release(batch)),
        TransferDrainOutcome::DrainedWithError(error) => {
            drop(ownership.release(batch));
            Err(anyhow::anyhow!(
                "local physical transfer drained with failure: {error}"
            ))
        }
        TransferDrainOutcome::Unproven(error) => {
            let error = format!("local physical completion is ambiguous: {error}");
            ownership.retain_unproven(error.clone());
            Err(anyhow::anyhow!(error))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use kvbm_logical::blocks::ImmutableBlock;
    use kvbm_physical::transfer::TransferCompleteNotification;

    use super::{LocalPhysicalOwnership, await_local_drain};
    use crate::G2;
    use crate::testing::{
        TestManagerBuilder, create_sequential_block, populate_manager_with_blocks,
    };

    use super::super::{ResolvedBatch, ResolvedCancellationUnit, fail_resolved_batch};
    use crate::tiering::offload::container::ResolvedBlock;
    use crate::tiering::offload::handle::{
        TransferHandle, TransferId, TransferState, TransferStatus,
    };

    #[derive(Debug)]
    struct RetainedResource {
        drops: Arc<AtomicUsize>,
    }

    impl Drop for RetainedResource {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn owned_batch() -> (
        ResolvedBatch<G2>,
        ImmutableBlock<G2>,
        Arc<Mutex<TransferState>>,
        TransferHandle,
    ) {
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(1)
                .block_size(4)
                .build(),
        );
        let token_block = create_sequential_block(0, manager.block_size());
        let sequence_hash =
            populate_manager_with_blocks(manager.as_ref(), std::slice::from_ref(&token_block))
                .expect("populate source manager")[0];
        let source = manager
            .match_blocks(&[sequence_hash])
            .pop()
            .expect("match source block");
        let observer = source.clone();
        let transfer_id = TransferId::new();
        let block_id = source.block_id();
        let (mut state, handle) = TransferState::new(transfer_id, vec![block_id]);
        let cancellation = state
            .cancellation_token()
            .root_unit()
            .expect("create root cancellation unit");
        assert!(cancellation.claim_commitment());
        state.add_passed([block_id]);
        state.mark_committed_blocks([block_id]);
        state.set_status(TransferStatus::Transferring);
        let state = Arc::new(Mutex::new(state));

        (
            ResolvedBatch {
                blocks: vec![ResolvedBlock {
                    transfer_id,
                    block_id,
                    sequence_hash,
                    guard: Some(source),
                    pending_guard: None,
                    state: Arc::clone(&state),
                }],
                evicted: Vec::new(),
                timing: super::super::super::batch::TimingTrace::new(),
                cancellation_units: vec![ResolvedCancellationUnit {
                    transfer_id,
                    state: Arc::clone(&state),
                    cancellation,
                }],
            },
            observer,
            state,
            handle,
        )
    }

    #[tokio::test]
    async fn unproven_notification_retains_all_local_physical_ownership() {
        let (mut batch, observer, state, mut handle) = owned_batch();
        let resource_drops = Arc::new(AtomicUsize::new(0));
        let events = Arc::new(velo::EventManager::local());
        let event = events.new_event().expect("create completion event");
        let notification = TransferCompleteNotification::from_awaiter(
            events
                .awaiter(event.handle())
                .expect("create completion awaiter"),
        );
        event
            .poison("physical completion did not prove drain")
            .expect("poison completion event");

        let ownership = LocalPhysicalOwnership::new(
            &mut batch,
            RetainedResource {
                drops: Arc::clone(&resource_drops),
            },
        );
        let error = await_local_drain(&mut batch, ownership, notification)
            .await
            .expect_err("an unproven receipt must fail without releasing ownership");

        assert!(error.to_string().contains("completion is ambiguous"));
        assert!(batch.is_empty());
        assert_eq!(observer.use_count(), 2);
        assert_eq!(resource_drops.load(Ordering::SeqCst), 0);
        assert_eq!(handle.status(), TransferStatus::Transferring);
        assert!(
            state
                .lock()
                .unwrap()
                .error
                .as_deref()
                .is_some_and(|error| error.contains("did not prove drain"))
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(25), handle.wait())
                .await
                .is_err(),
            "unproven local work must remain nonterminal",
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(25), handle.cancel().wait())
                .await
                .is_err(),
            "unproven local work must retain its cancellation unit",
        );
    }

    #[tokio::test]
    async fn drained_block_error_releases_ownership_and_fails_terminal() {
        let (mut batch, observer, _state, mut handle) = owned_batch();
        let resource_drops = Arc::new(AtomicUsize::new(0));
        let events = Arc::new(velo::EventManager::local());
        let notification = TransferCompleteNotification::aggregate_results(
            vec![
                Ok(TransferCompleteNotification::completed()),
                Err(anyhow::anyhow!("injected dispatch failure after launch")),
            ],
            &events,
            &tokio::runtime::Handle::current(),
        )
        .expect("a launched child returns a drain receipt");
        let ownership = LocalPhysicalOwnership::new(
            &mut batch,
            RetainedResource {
                drops: Arc::clone(&resource_drops),
            },
        );

        let error = await_local_drain(&mut batch, ownership, notification)
            .await
            .expect_err("the drained dispatch error must fail the route");

        assert!(error.to_string().contains("drained with failure"));
        assert_eq!(batch.blocks.len(), 1);
        assert_eq!(batch.cancellation_units.len(), 1);
        assert_eq!(resource_drops.load(Ordering::SeqCst), 1);

        fail_resolved_batch(&mut batch, &error.to_string());
        let result = handle
            .wait()
            .await
            .expect("a drained block error publishes a terminal result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(observer.use_count(), 1);
    }

    #[tokio::test]
    async fn completed_block_receipt_releases_ownership_for_registration() {
        let (mut batch, observer, state, handle) = owned_batch();
        let resource_drops = Arc::new(AtomicUsize::new(0));
        let ownership = LocalPhysicalOwnership::new(
            &mut batch,
            RetainedResource {
                drops: Arc::clone(&resource_drops),
            },
        );

        let resource = await_local_drain(
            &mut batch,
            ownership,
            TransferCompleteNotification::completed(),
        )
        .await
        .expect("a completed receipt restores physical ownership");

        assert_eq!(batch.blocks.len(), 1);
        assert_eq!(batch.cancellation_units.len(), 1);
        assert_eq!(observer.use_count(), 2);
        assert_eq!(resource_drops.load(Ordering::SeqCst), 0);
        drop(resource);
        assert_eq!(resource_drops.load(Ordering::SeqCst), 1);

        let block_id = batch.blocks[0].block_id;
        batch.release_source_guards();
        state.lock().unwrap().mark_completed([block_id]);
        batch.settle_cancellation_units();
        assert_eq!(handle.status(), TransferStatus::Complete);
        assert_eq!(observer.use_count(), 1);
    }

    #[tokio::test]
    async fn abort_after_local_dispatch_retains_all_physical_ownership() {
        let (mut batch, observer, state, handle) = owned_batch();
        let resource_drops = Arc::new(AtomicUsize::new(0));
        let events = Arc::new(velo::EventManager::local());
        let event = events.new_event().expect("create completion event");
        let notification = TransferCompleteNotification::from_awaiter(
            events
                .awaiter(event.handle())
                .expect("create completion awaiter"),
        );
        let task_resource_drops = Arc::clone(&resource_drops);
        let ownership_armed = Arc::new(tokio::sync::Notify::new());
        let task_armed = Arc::clone(&ownership_armed);
        let task = tokio::spawn(async move {
            let ownership = LocalPhysicalOwnership::new(
                &mut batch,
                RetainedResource {
                    drops: task_resource_drops,
                },
            );
            task_armed.notify_one();
            await_local_drain(&mut batch, ownership, notification).await
        });

        ownership_armed.notified().await;
        task.abort();
        assert!(
            task.await
                .expect_err("abort the local physical task")
                .is_cancelled()
        );

        assert_eq!(observer.use_count(), 2);
        assert_eq!(resource_drops.load(Ordering::SeqCst), 0);
        assert_eq!(handle.status(), TransferStatus::Transferring);
        assert!(
            state
                .lock()
                .unwrap()
                .error
                .as_deref()
                .is_some_and(|error| error.contains("ended without proven physical drain"))
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(25), handle.cancel().wait())
                .await
                .is_err(),
            "aborted local work must retain its cancellation unit",
        );
    }

    #[tokio::test]
    async fn panic_after_dispatch_before_drain_retains_all_physical_ownership() {
        let (mut batch, observer, state, handle) = owned_batch();
        let resource_drops = Arc::new(AtomicUsize::new(0));
        let ownership = LocalPhysicalOwnership::new(
            &mut batch,
            RetainedResource {
                drops: Arc::clone(&resource_drops),
            },
        );
        let panic_state = Arc::clone(&state);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _ownership = ownership;
            let _state_guard = panic_state.lock().unwrap();
            panic!("injected status publication panic");
        }));

        assert!(panic.is_err());
        assert!(batch.is_empty());
        assert_eq!(observer.use_count(), 2);
        assert_eq!(resource_drops.load(Ordering::SeqCst), 0);
        assert_eq!(handle.status(), TransferStatus::Transferring);
        assert!(
            state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .error
                .as_deref()
                .is_some_and(|error| error.contains("ended without proven physical drain"))
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(25), handle.cancel().wait())
                .await
                .is_err(),
            "a post-dispatch panic must retain its cancellation unit",
        );
    }
}
