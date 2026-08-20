// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production cancellation regressions for the offload pipeline.

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use futures::future::BoxFuture;
    use tokio::sync::{Notify, mpsc, watch};

    use super::super::batch::{BatchCollector, BatchConfig, TransferBatch};
    use super::super::container::{EvaluatedBlock, OffloadContainer, UpgradedContainer};
    use super::super::handle::{
        TransferHandle, TransferId, TransferState, TransferStatus, settle_transfer_unit,
    };
    use super::super::pending::PendingTracker;
    use super::super::pipeline::{ObjectTransferExecutor, PreconditionAwaiter, upgrade_batch};
    use super::super::queue::CancellableQueue;
    use super::super::source::{ExternalBlock, SourceBlock, SourceBlocks};
    use crate::leader::InstanceLeader;
    use crate::object::ObjectBlockOps;
    use crate::testing::{TestManagerBuilder, TestRegistryBuilder, create_messenger_tcp};
    use crate::{BlockId, G2, SequenceHash};
    use kvbm_common::LogicalLayoutHandle;

    fn test_hash(value: u64) -> SequenceHash {
        SequenceHash::new(value, None, 0)
    }

    fn external_blocks(ids: &[BlockId]) -> Vec<ExternalBlock<G2>> {
        ids.iter()
            .copied()
            .map(|id| ExternalBlock::new(id, test_hash(id as u64)))
            .collect()
    }

    fn new_container(
        transfer_id: TransferId,
        ids: &[BlockId],
        precondition: Option<velo::EventHandle>,
    ) -> (
        Arc<std::sync::Mutex<TransferState>>,
        TransferHandle,
        OffloadContainer<G2>,
    ) {
        let (state, handle) = TransferState::new(transfer_id, ids.to_vec());
        let state = Arc::new(std::sync::Mutex::new(state));
        let container = OffloadContainer::new(
            transfer_id,
            SourceBlocks::External(external_blocks(ids)),
            Arc::clone(&state),
            precondition,
        );
        (state, handle, container)
    }

    fn evaluated_container(
        transfer_id: TransferId,
        ids: &[BlockId],
    ) -> (
        Arc<std::sync::Mutex<TransferState>>,
        TransferHandle,
        OffloadContainer<G2>,
    ) {
        let (state, handle, mut container) = new_container(transfer_id, ids, None);
        drop(container.take_source());
        let evaluated = external_blocks(ids)
            .into_iter()
            .map(|block| EvaluatedBlock::new(SourceBlock::External(block), None))
            .collect();
        container.finish_evaluation(evaluated, Vec::new());
        {
            let mut state_guard = state.lock().unwrap();
            state_guard.total_expected_blocks = ids.len();
            state_guard.add_passed(ids.iter().copied());
            state_guard.set_status(TransferStatus::Queued);
        }
        (state, handle, container)
    }

    fn evaluated_container_for_state(
        transfer_id: TransferId,
        ids: &[BlockId],
        state: Arc<std::sync::Mutex<TransferState>>,
        cancellation: super::super::cancel::CancellationUnit,
        precondition: Option<velo::EventHandle>,
    ) -> OffloadContainer<G2> {
        let mut container = OffloadContainer::with_cancellation(
            transfer_id,
            SourceBlocks::External(external_blocks(ids)),
            Arc::clone(&state),
            precondition,
            cancellation,
        );
        drop(container.take_source());
        let evaluated = external_blocks(ids)
            .into_iter()
            .map(|block| EvaluatedBlock::new(SourceBlock::External(block), None))
            .collect();
        container.finish_evaluation(evaluated, Vec::new());
        {
            let mut state_guard = state.lock().unwrap();
            state_guard.total_expected_blocks += ids.len();
            state_guard.add_passed(ids.iter().copied());
            state_guard.set_status(TransferStatus::Queued);
        }
        container
    }

    fn evaluated_container_with_precondition(
        transfer_id: TransferId,
        ids: &[BlockId],
        precondition: velo::EventHandle,
        pending_tracker: &Arc<PendingTracker>,
    ) -> OffloadContainer<G2> {
        let (state, _handle) = TransferState::new(transfer_id, ids.to_vec());
        let state = Arc::new(std::sync::Mutex::new(state));
        let mut container = OffloadContainer::new(
            transfer_id,
            SourceBlocks::External(external_blocks(ids)),
            Arc::clone(&state),
            Some(precondition),
        );
        drop(container.take_source());
        let evaluated = external_blocks(ids)
            .into_iter()
            .map(|block| {
                let pending_guard = pending_tracker
                    .try_claim(block.sequence_hash)
                    .expect("each helper block claims a unique hash");
                EvaluatedBlock::new(SourceBlock::External(block), Some(pending_guard))
            })
            .collect();
        container.finish_evaluation(evaluated, Vec::new());
        {
            let mut state_guard = state.lock().unwrap();
            state_guard.total_expected_blocks = ids.len();
            state_guard.add_passed(ids.iter().copied());
            state_guard.set_status(TransferStatus::Queued);
        }
        container
    }

    async fn test_leader() -> Arc<InstanceLeader> {
        let messenger = create_messenger_tcp().await.expect("create messenger");
        let registry = TestRegistryBuilder::new().build();
        let g2_manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(8)
                .block_size(4)
                .registry(registry.clone())
                .build(),
        );

        Arc::new(
            InstanceLeader::builder()
                .messenger(messenger)
                .registry(registry)
                .g2_manager(g2_manager)
                .workers(Vec::new())
                .build()
                .expect("build test leader"),
        )
    }

    #[test]
    fn container_owns_the_whole_transfer_cancellation_unit() {
        let transfer_id = TransferId::new();
        let (_state, handle, container) = new_container(transfer_id, &[11, 12], None);

        assert_eq!(container.transfer_id(), transfer_id);
        assert_eq!(container.source_len(), 2);
        assert!(!container.is_cancelled());

        let _confirmation = handle.cancel();

        assert!(container.is_cancelled());
        assert_eq!(container.source_len(), 2);
    }

    #[tokio::test]
    async fn precommit_confirmation_waits_for_the_container_drop() {
        let transfer_id = TransferId::new();
        let (_state, handle, container) = new_container(transfer_id, &[13], None);
        let confirmation = handle.cancel();

        assert!(
            tokio::time::timeout(Duration::from_millis(50), confirmation.wait())
                .await
                .is_err(),
            "pre-commit confirmation must wait for the container drop"
        );
        drop(container);
        tokio::time::timeout(Duration::from_millis(250), handle.cancel().wait())
            .await
            .expect("container drop settles cancellation");
    }

    #[tokio::test]
    async fn precondition_awaiter_selects_cancellation_and_drops_the_container() {
        let leader = test_leader().await;
        let event = leader
            .messenger()
            .events()
            .new_event()
            .expect("create pending precondition");
        let transfer_id = TransferId::new();
        let (state, handle, container) = new_container(transfer_id, &[21], Some(event.handle()));
        let input = Arc::new(CancellableQueue::new());
        let output = Arc::new(CancellableQueue::new());
        let awaiter = PreconditionAwaiter::new(Arc::clone(&input), Arc::clone(&output), leader, 8);
        let task = tokio::spawn(awaiter.run());

        assert!(input.push(transfer_id, container));
        tokio::time::sleep(Duration::from_millis(10)).await;
        let confirmation = handle.cancel();

        tokio::time::timeout(Duration::from_millis(250), confirmation.wait())
            .await
            .expect("precondition cancellation settles");
        assert!(output.pop_valid().is_none());
        assert_eq!(state.lock().unwrap().status, TransferStatus::Cancelled);

        task.abort();
    }

    #[tokio::test]
    async fn precondition_awaiter_does_not_block_a_later_ready_container() {
        let leader = test_leader().await;
        let event = leader
            .messenger()
            .events()
            .new_event()
            .expect("create pending precondition");
        let pending_id = TransferId::new();
        let ready_id = TransferId::new();
        let (_pending_state, pending_handle, pending) =
            new_container(pending_id, &[22], Some(event.handle()));
        let (_ready_state, _ready_handle, ready) = new_container(ready_id, &[23], None);
        let input = Arc::new(CancellableQueue::new());
        let output = Arc::new(CancellableQueue::new());
        let awaiter = PreconditionAwaiter::new(Arc::clone(&input), Arc::clone(&output), leader, 8);
        let task = tokio::spawn(awaiter.run());

        assert!(input.push(pending_id, pending));
        assert!(input.push(ready_id, ready));

        let output_item = tokio::time::timeout(Duration::from_millis(250), async {
            loop {
                if let Some(item) = output.pop_valid() {
                    return item;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("ready container reaches the output");
        assert_eq!(output_item.transfer_id, ready_id);
        drop(output_item);

        tokio::time::timeout(Duration::from_millis(250), pending_handle.cancel().wait())
            .await
            .expect("pending container cancellation settles");
        task.abort();
    }

    #[tokio::test]
    async fn precondition_awaiter_enforces_its_concurrency_cap() {
        let leader = test_leader().await;
        let first_event = leader
            .messenger()
            .events()
            .new_event()
            .expect("create first pending precondition");
        let second_event = leader
            .messenger()
            .events()
            .new_event()
            .expect("create second pending precondition");
        let first_id = TransferId::new();
        let second_id = TransferId::new();
        let ready_id = TransferId::new();
        let (_first_state, first_handle, first) =
            new_container(first_id, &[24], Some(first_event.handle()));
        let (_second_state, second_handle, second) =
            new_container(second_id, &[25], Some(second_event.handle()));
        let (_ready_state, _ready_handle, ready) = new_container(ready_id, &[26], None);
        let input = Arc::new(CancellableQueue::new());
        let output = Arc::new(CancellableQueue::new());
        let awaiter = PreconditionAwaiter::new(Arc::clone(&input), Arc::clone(&output), leader, 2);
        let task = tokio::spawn(awaiter.run());

        assert!(input.push(first_id, first));
        assert!(input.push(second_id, second));
        assert!(input.push(ready_id, ready));
        tokio::time::timeout(Duration::from_millis(250), async {
            while input.len_approx() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("only two preconditions leave the queue");
        assert!(output.pop_valid().is_none());

        let first_confirmation = first_handle.cancel();
        let ready = tokio::time::timeout(Duration::from_millis(250), async {
            loop {
                if let Some(item) = output.pop_valid() {
                    return item;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("a released slot admits the ready container");
        assert_eq!(ready.transfer_id, ready_id);
        drop(ready);
        first_confirmation.wait().await;
        second_handle.cancel().wait().await;
        task.abort();
    }

    #[tokio::test]
    async fn precondition_awaiter_shutdown_drops_all_tracked_tasks() {
        let leader = test_leader().await;
        let first_event = leader
            .messenger()
            .events()
            .new_event()
            .expect("create first pending precondition");
        let second_event = leader
            .messenger()
            .events()
            .new_event()
            .expect("create second pending precondition");
        let pending_tracker = Arc::new(PendingTracker::new());
        let first_id = TransferId::new();
        let second_id = TransferId::new();
        let first = evaluated_container_with_precondition(
            first_id,
            &[27],
            first_event.handle(),
            &pending_tracker,
        );
        let second = evaluated_container_with_precondition(
            second_id,
            &[28],
            second_event.handle(),
            &pending_tracker,
        );
        let input = Arc::new(CancellableQueue::new());
        let output = Arc::new(CancellableQueue::new());
        let awaiter = PreconditionAwaiter::new(Arc::clone(&input), Arc::clone(&output), leader, 2);
        let task = tokio::spawn(awaiter.run());

        assert!(input.push(first_id, first));
        assert!(input.push(second_id, second));
        tokio::time::timeout(Duration::from_millis(250), async {
            while !input.is_empty_approx() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("both preconditions start");
        assert_eq!(pending_tracker.len(), 2);

        task.abort();
        let _ = task.await;
        tokio::time::timeout(Duration::from_millis(250), async {
            while !pending_tracker.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("awaiter shutdown drops all retained containers");
    }

    #[tokio::test]
    async fn aborted_precondition_task_fails_its_uncommitted_container() {
        let leader = test_leader().await;
        let event = leader
            .messenger()
            .events()
            .new_event()
            .expect("create pending precondition");
        let transfer_id = TransferId::new();
        let (_state, mut handle, container) =
            new_container(transfer_id, &[29], Some(event.handle()));
        let input = Arc::new(CancellableQueue::new());
        let output = Arc::new(CancellableQueue::new());
        let awaiter = PreconditionAwaiter::new(Arc::clone(&input), output, leader, 1);
        let task = tokio::spawn(awaiter.run());

        assert!(input.push(transfer_id, container));
        tokio::time::timeout(Duration::from_millis(250), async {
            while !input.is_empty_approx() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("precondition task owns the container");

        task.abort();
        task.await.expect_err("abort the precondition awaiter");

        let result = tokio::time::timeout(Duration::from_millis(250), handle.wait())
            .await
            .expect("aborted child publishes a terminal result")
            .expect("transfer handle returns a result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(
            result.error.as_deref(),
            Some("precommit offload container dropped before settlement")
        );
    }

    #[tokio::test]
    async fn panicking_precommit_task_fails_its_uncommitted_container() {
        let transfer_id = TransferId::new();
        let (_state, mut handle, container) = new_container(transfer_id, &[30], None);
        let task = tokio::spawn(async move {
            let _container = container;
            panic!("injected precommit task panic");
        });

        assert!(task.await.expect_err("precommit task panics").is_panic());
        let result = tokio::time::timeout(Duration::from_millis(250), handle.wait())
            .await
            .expect("panicking task publishes a terminal result")
            .expect("transfer handle returns a result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(
            result.error.as_deref(),
            Some("precommit offload container dropped before settlement")
        );
    }

    #[test]
    fn transfer_batch_sweeps_cancelled_containers_before_upgrade() {
        let cancelled_id = TransferId::new();
        let live_id = TransferId::new();
        let (cancelled_state, cancelled_handle, cancelled) =
            evaluated_container(cancelled_id, &[31, 32]);
        let (_live_state, _live_handle, live) = evaluated_container(live_id, &[41]);
        let batch = TransferBatch::from_containers(vec![cancelled, live]);

        let _confirmation = cancelled_handle.cancel();
        let resolved = upgrade_batch(batch);

        assert_eq!(resolved.blocks.len(), 1);
        assert_eq!(resolved.blocks[0].transfer_id, live_id);
        assert_eq!(resolved.blocks[0].block_id, 41);
        assert_eq!(
            cancelled_state.lock().unwrap().status,
            TransferStatus::Cancelled
        );
    }

    #[tokio::test]
    async fn batch_collector_never_splits_a_container() {
        let transfer_id = TransferId::new();
        let (_state, _handle, container) = evaluated_container(transfer_id, &[35, 36]);
        let input = Arc::new(CancellableQueue::new());
        let (output_tx, mut output_rx) = mpsc::channel(1);
        let (cancel_tx, cancel_rx) = watch::channel(HashSet::new());
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let config = BatchConfig::default().with_max_size(1);
        let collector = BatchCollector::new(
            config,
            Arc::clone(&input),
            output_tx,
            cancel_rx,
            shutdown_rx,
        );
        let task = tokio::spawn(collector.run());

        assert!(input.push(transfer_id, container));
        let batch = tokio::time::timeout(Duration::from_millis(250), output_rx.recv())
            .await
            .expect("collector flushes the oversized container")
            .expect("collector output stays open");

        assert_eq!(batch.container_len(), 1);
        assert_eq!(batch.len(), 2);
        drop(cancel_tx);
        task.abort();
    }

    #[test]
    fn cancellation_between_final_sweep_and_commitment_claim_wins() {
        let transfer_id = TransferId::new();
        let (state, handle, container) = evaluated_container(transfer_id, &[45]);
        let mut batch = TransferBatch::from_containers(vec![container]);

        assert_eq!(batch.sweep_cancelled(), 0);
        let _confirmation = handle.cancel();
        let container = batch
            .containers
            .pop()
            .expect("final sweep kept the live container");

        assert!(container.upgrade().is_none());
        assert_eq!(state.lock().unwrap().status, TransferStatus::Cancelled);
    }

    #[test]
    fn cancellation_after_upstream_commit_cannot_split_a_chain() {
        let transfer_id = TransferId::new();
        let (state, handle, upstream) = evaluated_container(transfer_id, &[46]);
        let upstream = upstream.upgrade().expect("upstream claims commitment");
        let mut continuations = upstream.cancellation.fan_out(1);
        let downstream = evaluated_container_for_state(
            transfer_id,
            &[47],
            Arc::clone(&state),
            continuations.pop().expect("one downstream route"),
            None,
        );
        let _confirmation = handle.cancel();
        let downstream = downstream
            .upgrade()
            .expect("the shared commitment applies to the downstream stage");

        assert_eq!(upstream.blocks.len(), 1);
        assert_eq!(downstream.blocks.len(), 1);
    }

    #[tokio::test]
    async fn postcommit_cancel_does_not_drop_a_waiting_downstream_stage() {
        let leader = test_leader().await;
        let event = leader
            .messenger()
            .events()
            .new_event()
            .expect("create downstream precondition");
        let transfer_id = TransferId::new();
        let (state, handle, upstream) = evaluated_container(transfer_id, &[48]);
        let upstream = upstream.upgrade().expect("upstream claims commitment");
        let mut continuations = upstream.cancellation.fan_out(1);
        let downstream = evaluated_container_for_state(
            transfer_id,
            &[49],
            Arc::clone(&state),
            continuations.pop().expect("one downstream route"),
            Some(event.handle()),
        );
        let input = Arc::new(CancellableQueue::new());
        let output = Arc::new(CancellableQueue::new());
        let awaiter = PreconditionAwaiter::new(Arc::clone(&input), Arc::clone(&output), leader, 1);
        let task = tokio::spawn(awaiter.run());

        assert!(input.push(transfer_id, downstream));
        tokio::time::sleep(Duration::from_millis(10)).await;
        let confirmation = handle.cancel();
        assert!(
            tokio::time::timeout(Duration::from_millis(25), confirmation.wait())
                .await
                .is_err(),
            "committed downstream work keeps cancellation pending",
        );

        event.trigger().expect("release downstream precondition");
        let downstream = tokio::time::timeout(Duration::from_millis(250), async {
            loop {
                if let Some(item) = output.pop_valid() {
                    return item.data;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("postcommit downstream work reaches the next stage");
        let downstream = downstream
            .upgrade()
            .expect("the downstream commitment remains valid");
        assert_eq!(downstream.blocks.len(), 1);
        drop(downstream);
        handle.cancel().wait().await;
        task.abort();
    }

    #[tokio::test]
    async fn failed_branch_enqueue_releases_only_its_child_unit() {
        let transfer_id = TransferId::new();
        let (state, handle) = TransferState::new(transfer_id, vec![50]);
        let state = Arc::new(std::sync::Mutex::new(state));
        let token = state.lock().unwrap().cancellation_token();
        let root = token.root_unit().expect("create root cancellation unit");
        assert!(root.claim_commitment());
        let mut branches = root.fan_out(2);
        let rejected = OffloadContainer::with_cancellation(
            transfer_id,
            SourceBlocks::External(external_blocks(&[50])),
            Arc::clone(&state),
            None,
            branches.pop().expect("rejected branch unit"),
        );
        let surviving = branches.pop().expect("surviving branch unit");
        let queue = CancellableQueue::new();
        queue.mark_cancelled(transfer_id);

        let confirmation = handle.cancel();
        assert!(!queue.push(transfer_id, rejected));
        assert!(
            tokio::time::timeout(Duration::from_millis(25), confirmation.wait())
                .await
                .is_err(),
            "the surviving branch keeps cancellation pending",
        );

        drop(surviving);
        handle.cancel().wait().await;
    }

    #[tokio::test]
    async fn logical_transfer_waits_for_all_route_units_before_terminal_failure() {
        let transfer_id = TransferId::new();
        let (mut state, mut handle) = TransferState::new(transfer_id, vec![50]);
        let token = state.cancellation_token();
        let root = token.root_unit().expect("create root cancellation unit");
        assert!(root.claim_commitment());
        state.add_passed([50]);
        state.mark_committed_blocks([50]);
        let state = Arc::new(std::sync::Mutex::new(state));
        let mut routes = root.fan_out(2);
        let first = routes.pop().expect("first route unit");
        let second = routes.pop().expect("second route unit");

        state.lock().unwrap().mark_completed([50]);
        let first_state = Arc::clone(&state);
        first.settle(move || {
            first_state.lock().unwrap().finish_logical_operation();
        });

        assert!(
            tokio::time::timeout(Duration::from_millis(25), handle.wait())
                .await
                .is_err(),
            "one completed route must not publish a terminal result",
        );

        state
            .lock()
            .unwrap()
            .record_error("late downstream route failure".to_string());
        let second_state = Arc::clone(&state);
        second.settle(move || {
            second_state.lock().unwrap().finish_logical_operation();
        });

        let result = tokio::time::timeout(Duration::from_millis(100), handle.wait())
            .await
            .expect("the last route publishes the terminal result")
            .expect("the handle returns a transfer result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(
            result.error.as_deref(),
            Some("late downstream route failure")
        );
    }

    #[tokio::test]
    async fn abandoned_committed_route_records_failure_before_sibling_completion() {
        let transfer_id = TransferId::new();
        let (state, mut handle, container) = evaluated_container(transfer_id, &[51]);
        let UpgradedContainer {
            blocks,
            cancellation,
            ..
        } = container.upgrade().expect("upgrade the complete container");
        drop(blocks);
        state.lock().unwrap().mark_completed([51]);

        let mut routes = cancellation.fan_out(2);
        let abandoned = routes.pop().expect("abandoned route unit");
        let completed = routes.pop().expect("completed route unit");
        drop(abandoned);
        settle_transfer_unit(completed, Arc::clone(&state));

        let result = tokio::time::timeout(Duration::from_millis(100), handle.wait())
            .await
            .expect("the final route publishes a terminal result")
            .expect("the handle returns a transfer result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(
            result.error.as_deref(),
            Some("committed offload route dropped before settlement")
        );
    }

    #[tokio::test]
    async fn precondition_failure_releases_guards_before_terminal_status() {
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(1)
                .block_size(4)
                .build(),
        );
        let token_block = crate::testing::create_sequential_block(0, manager.block_size());
        let sequence_hash = crate::testing::populate_manager_with_blocks(
            manager.as_ref(),
            std::slice::from_ref(&token_block),
        )
        .expect("populate source manager")[0];
        let source = manager
            .match_blocks(&[sequence_hash])
            .pop()
            .expect("match source block");
        let source_observer = source.clone();
        let pending_tracker = Arc::new(PendingTracker::new());
        let transfer_id = TransferId::new();
        let block_id = source.block_id();
        let (mut state, mut handle) = TransferState::new(transfer_id, vec![block_id]);
        state.total_expected_blocks = 1;
        state.add_passed([block_id]);
        state.set_status(TransferStatus::Queued);
        let state = Arc::new(std::sync::Mutex::new(state));
        let mut container = OffloadContainer::new(
            transfer_id,
            SourceBlocks::Strong(vec![source]),
            Arc::clone(&state),
            None,
        );
        let SourceBlocks::Strong(mut blocks) = container.take_source() else {
            panic!("strong source remains strong");
        };
        let pending_guard = pending_tracker
            .try_claim(sequence_hash)
            .expect("claim the source hash");
        container.finish_evaluation(
            vec![EvaluatedBlock::new(
                SourceBlock::Strong(blocks.pop().expect("one source block")),
                Some(pending_guard),
            )],
            Vec::new(),
        );

        container.fail("precondition poisoned: injected failure".to_string());

        let result = handle.wait().await.expect("failure publishes a result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert!(pending_tracker.is_empty());
        assert_eq!(source_observer.use_count(), 1);
    }

    #[tokio::test]
    async fn closed_batch_output_fails_the_retained_container() {
        let transfer_id = TransferId::new();
        let (_state, mut handle, container) = evaluated_container(transfer_id, &[52]);
        let input = Arc::new(CancellableQueue::new());
        let (output_tx, output_rx) = mpsc::channel(1);
        let (_cancel_tx, cancel_rx) = watch::channel(HashSet::new());
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let collector = BatchCollector::new(
            BatchConfig::default().with_max_size(1),
            Arc::clone(&input),
            output_tx,
            cancel_rx,
            shutdown_rx,
        );
        drop(output_rx);
        let task = tokio::spawn(collector.run());

        assert!(input.push(transfer_id, container));
        let result = tokio::time::timeout(Duration::from_millis(250), handle.wait())
            .await
            .expect("closed batch output publishes a terminal result")
            .expect("the handle returns a transfer result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(result.error.as_deref(), Some("batch output channel closed"));
        tokio::time::timeout(Duration::from_millis(100), handle.cancel().wait())
            .await
            .expect("batch rejection releases its cancellation unit");

        task.abort();
    }

    #[tokio::test]
    async fn post_upgrade_cancellation_waits_for_physical_completion() {
        let transfer_id = TransferId::new();
        let (state, handle, container) = evaluated_container(transfer_id, &[51]);
        let (input_tx, input_rx) = mpsc::channel(1);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let object_ops = Arc::new(GatedObjectBlockOps::default());
        let executor = ObjectTransferExecutor::new(
            input_rx,
            Arc::clone(&object_ops) as Arc<dyn ObjectBlockOps>,
            LogicalLayoutHandle::G2,
            false,
            1,
            None,
            shutdown_rx,
        );
        let executor_task = tokio::spawn(executor.run());

        input_tx
            .send(TransferBatch::from_containers(vec![container]))
            .await
            .expect("send transfer batch");
        object_ops.wait_until_started().await;

        let confirmation = handle.cancel();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), confirmation.wait())
                .await
                .is_err(),
            "post-upgrade cancellation must wait for physical work"
        );
        assert_eq!(object_ops.calls.load(Ordering::SeqCst), 1);

        object_ops.release.notify_one();
        drop(input_tx);
        tokio::time::timeout(Duration::from_millis(250), executor_task)
            .await
            .expect("physical executor drains")
            .expect("executor task completes");
        tokio::time::timeout(Duration::from_millis(250), handle.cancel().wait())
            .await
            .expect("cancellation confirms after physical completion");
        assert_eq!(state.lock().unwrap().status, TransferStatus::Complete);
    }

    #[test]
    fn surviving_containers_flat_map_only_after_upgrade() {
        let first_id = TransferId::new();
        let second_id = TransferId::new();
        let (_first_state, _first_handle, first) = evaluated_container(first_id, &[61, 62]);
        let (_second_state, _second_handle, second) = evaluated_container(second_id, &[71]);

        let resolved = upgrade_batch(TransferBatch::from_containers(vec![first, second]));

        let identities: Vec<_> = resolved
            .blocks
            .iter()
            .map(|block| (block.transfer_id, block.block_id))
            .collect();
        assert_eq!(
            identities,
            vec![(first_id, 61), (first_id, 62), (second_id, 71)]
        );
    }

    #[derive(Default)]
    struct GatedObjectBlockOps {
        started: Arc<Notify>,
        release: Arc<Notify>,
        entered: AtomicBool,
        calls: AtomicUsize,
    }

    impl GatedObjectBlockOps {
        async fn wait_until_started(&self) {
            while !self.entered.load(Ordering::SeqCst) {
                self.started.notified().await;
            }
        }
    }

    impl ObjectBlockOps for GatedObjectBlockOps {
        fn has_blocks(
            &self,
            keys: Vec<SequenceHash>,
        ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
            Box::pin(async move { keys.into_iter().map(|key| (key, None)).collect() })
        }

        fn put_blocks(
            &self,
            keys: Vec<SequenceHash>,
            _layout: LogicalLayoutHandle,
            _block_ids: Vec<BlockId>,
        ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.entered.store(true, Ordering::SeqCst);
            self.started.notify_waiters();
            let release = self.release.clone();
            Box::pin(async move {
                release.notified().await;
                keys.into_iter().map(Ok).collect()
            })
        }

        fn get_blocks(
            &self,
            keys: Vec<SequenceHash>,
            _layout: LogicalLayoutHandle,
            _block_ids: Vec<BlockId>,
        ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
            Box::pin(async move { keys.into_iter().map(Ok).collect() })
        }
    }
}
