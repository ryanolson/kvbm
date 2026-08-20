// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::offload::{ExternalBlock, PassAllPolicy};

mod object;

#[test]
fn test_pipeline_builder() {
    let config = PipelineBuilder::<(), ()>::new()
        .batch_size(32)
        .min_batch_size(8)
        .policy_timeout(Duration::from_millis(50))
        .auto_chain(true)
        .sweep_interval(Duration::from_millis(5))
        .build();

    assert_eq!(config.base.batch_config.max_batch_size, 32);
    assert_eq!(config.base.batch_config.min_batch_size, 8);
    assert_eq!(config.base.policy_timeout, Duration::from_millis(50));
    assert!(config.options.auto_chain);
    assert_eq!(config.base.sweep_interval, Duration::from_millis(5));
}

#[test]
fn pipeline_builder_records_logical_resource() {
    let resource = kvbm_common::LogicalResourceId(9);
    let config = PipelineBuilder::<(), ()>::new().resource(resource).build();
    assert_eq!(config.options.resource, Some(resource));
}

#[test]
fn test_pipeline_config_default() {
    let config = PipelineConfig::<(), ()>::default();
    assert!(config.base.policies.is_empty());
    assert!(!config.options.auto_chain);
    assert_eq!(config.base.sweep_interval, Duration::from_millis(10));
}

#[tokio::test]
async fn policy_evaluator_claims_a_same_hash_once_for_external_blocks() {
    let pending_tracker = Arc::new(PendingTracker::new());
    let input_queue = Arc::new(CancellableQueue::<OffloadContainer<()>>::new());
    let output_queue = Arc::new(CancellableQueue::<OffloadContainer<()>>::new());
    let (_cancel_tx, cancel_rx) = watch::channel(HashSet::new());
    let evaluator = PolicyEvaluator {
        policies: vec![Arc::new(PassAllPolicy::<()>::new())],
        timeout: Duration::from_millis(10),
        input_queue,
        output_queue: Arc::clone(&output_queue),
        cancel_rx,
        pending_tracker: Arc::clone(&pending_tracker),
    };
    let sequence_hash = SequenceHash::new(71, None, 0);
    let first_block_id = 11;
    let second_block_id = 12;
    let first_transfer_id = TransferId::new();
    let second_transfer_id = TransferId::new();
    let (first_state, _first_handle) = TransferState::new(first_transfer_id, vec![first_block_id]);
    let (second_state, second_handle) =
        TransferState::new(second_transfer_id, vec![second_block_id]);
    let first_state = Arc::new(std::sync::Mutex::new(first_state));
    let second_state = Arc::new(std::sync::Mutex::new(second_state));

    evaluator
        .evaluate(OffloadContainer::new(
            first_transfer_id,
            SourceBlocks::External(vec![ExternalBlock::new(first_block_id, sequence_hash)]),
            Arc::clone(&first_state),
            None,
        ))
        .await;

    let first = output_queue
        .pop_valid()
        .expect("the first container enters the policy output queue");
    assert_eq!(first.data.evaluated_len(), 1);
    assert!(pending_tracker.is_pending(&sequence_hash));
    assert_eq!(
        first_state.lock().unwrap().passed_blocks,
        vec![first_block_id]
    );

    evaluator
        .evaluate(OffloadContainer::new(
            second_transfer_id,
            SourceBlocks::External(vec![ExternalBlock::new(second_block_id, sequence_hash)]),
            Arc::clone(&second_state),
            None,
        ))
        .await;

    assert_eq!(second_handle.status(), TransferStatus::Complete);
    assert_eq!(
        second_state.lock().unwrap().filtered_out,
        vec![second_block_id]
    );
    assert!(
        output_queue.pop_valid().is_none(),
        "a losing duplicate claim must not enter the output queue"
    );

    drop(first);
    assert!(pending_tracker.is_empty());
}

#[tokio::test]
async fn policy_evaluator_filters_losing_strong_and_weak_claims() {
    let manager = Arc::new(
        crate::testing::TestManagerBuilder::<crate::G2>::new()
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
    let first_source = manager
        .match_blocks(&[sequence_hash])
        .pop()
        .expect("match source block");
    let block_id = first_source.block_id();
    let second_source = first_source.clone();
    let weak_source = first_source.downgrade();

    let pending_tracker = Arc::new(PendingTracker::new());
    let input_queue = Arc::new(CancellableQueue::<OffloadContainer<crate::G2>>::new());
    let output_queue = Arc::new(CancellableQueue::<OffloadContainer<crate::G2>>::new());
    let (_cancel_tx, cancel_rx) = watch::channel(HashSet::new());
    let evaluator = PolicyEvaluator {
        policies: vec![Arc::new(PassAllPolicy::<crate::G2>::new())],
        timeout: Duration::from_millis(10),
        input_queue,
        output_queue: Arc::clone(&output_queue),
        cancel_rx,
        pending_tracker: Arc::clone(&pending_tracker),
    };
    let first_transfer_id = TransferId::new();
    let second_transfer_id = TransferId::new();
    let weak_transfer_id = TransferId::new();
    let (first_state, _first_handle) = TransferState::new(first_transfer_id, vec![block_id]);
    let (second_state, second_handle) = TransferState::new(second_transfer_id, vec![block_id]);
    let (weak_state, weak_handle) = TransferState::new(weak_transfer_id, Vec::new());
    let first_state = Arc::new(std::sync::Mutex::new(first_state));
    let second_state = Arc::new(std::sync::Mutex::new(second_state));
    let weak_state = Arc::new(std::sync::Mutex::new(weak_state));

    evaluator
        .evaluate(OffloadContainer::new(
            first_transfer_id,
            SourceBlocks::Strong(vec![first_source]),
            Arc::clone(&first_state),
            None,
        ))
        .await;

    let first = output_queue
        .pop_valid()
        .expect("the first strong container enters the policy output queue");
    assert_eq!(first.data.evaluated_len(), 1);
    assert!(pending_tracker.is_pending(&sequence_hash));

    evaluator
        .evaluate(OffloadContainer::new(
            second_transfer_id,
            SourceBlocks::Strong(vec![second_source]),
            Arc::clone(&second_state),
            None,
        ))
        .await;

    assert_eq!(second_handle.status(), TransferStatus::Complete);
    assert_eq!(
        second_state.lock().unwrap().filtered_out,
        vec![block_id],
        "a losing strong claim records its source block ID"
    );
    assert!(output_queue.pop_valid().is_none());

    evaluator
        .evaluate(OffloadContainer::new(
            weak_transfer_id,
            SourceBlocks::Weak(vec![weak_source]),
            Arc::clone(&weak_state),
            None,
        ))
        .await;

    assert_eq!(weak_handle.status(), TransferStatus::Complete);
    assert!(weak_state.lock().unwrap().filtered_out.is_empty());
    assert!(output_queue.pop_valid().is_none());

    drop(first);
    assert!(pending_tracker.is_empty());
}

#[tokio::test]
async fn uncancelled_watcher_releases_its_queue_arc() {
    let transfer_id = TransferId::new();
    let (state, _handle) = TransferState::new(transfer_id, vec![1]);
    let state = Arc::new(std::sync::Mutex::new(state));
    let eval_queue = Arc::new(CancellableQueue::<OffloadContainer<()>>::new());
    let watcher_queue = Arc::new(CancellableQueue::<OffloadContainer<()>>::new());
    let queue_weak = Arc::downgrade(&watcher_queue);
    let (cancel_tx, _cancel_rx) = watch::channel(HashSet::new());
    let ingress = PipelineIngress {
        eval_queue: Arc::clone(&eval_queue),
        cancellation_queues: vec![Arc::clone(&watcher_queue)],
        transfers: Arc::new(DashMap::new()),
        registration_gate: Arc::new(ParkingMutex::new(())),
        cancel_tx,
        runtime: tokio::runtime::Handle::current(),
    };

    assert!(ingress.enqueue(
        transfer_id,
        SourceBlocks::External(vec![super::super::source::ExternalBlock::new(
            1,
            SequenceHash::new(1, None, 0),
        )]),
        Arc::clone(&state),
    ));
    drop(ingress);
    drop(watcher_queue);

    state.lock().unwrap().set_complete();
    tokio::time::timeout(Duration::from_millis(50), async {
        while queue_weak.upgrade().is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("terminal transfer releases its watcher");

    drop(eval_queue.pop_valid());
}

#[tokio::test]
async fn closed_ingress_fails_the_returned_container() {
    let transfer_id = TransferId::new();
    let (state, mut handle) = TransferState::new(transfer_id, vec![99]);
    let state = Arc::new(std::sync::Mutex::new(state));
    let eval_queue = Arc::new(CancellableQueue::<OffloadContainer<()>>::new());
    let cancellation_queue = Arc::new(CancellableQueue::<OffloadContainer<()>>::new());
    let (cancel_tx, _cancel_rx) = watch::channel(HashSet::new());
    let ingress = PipelineIngress {
        eval_queue: Arc::clone(&eval_queue),
        cancellation_queues: vec![cancellation_queue],
        transfers: Arc::new(DashMap::new()),
        registration_gate: Arc::new(ParkingMutex::new(())),
        cancel_tx,
        runtime: tokio::runtime::Handle::current(),
    };

    let drained = eval_queue.close_and_drain();
    assert!(drained.is_empty());

    assert!(!ingress.enqueue(
        transfer_id,
        SourceBlocks::External(vec![super::super::source::ExternalBlock::new(
            99,
            SequenceHash::new(99, None, 0),
        )]),
        state,
    ));

    let result = handle
        .wait()
        .await
        .expect("closed ingress publishes a transfer result");
    assert_eq!(result.status, TransferStatus::Failed);
    assert_eq!(result.error.as_deref(), Some(PRECOMMIT_SHUTDOWN_ERROR));
    assert!(eval_queue.is_empty_approx());
}

#[tokio::test]
async fn ingress_removes_each_terminal_transfer_state() {
    let eval_queue = Arc::new(CancellableQueue::<OffloadContainer<()>>::new());
    let cancellation_queue = Arc::new(CancellableQueue::<OffloadContainer<()>>::new());
    let transfers = Arc::new(DashMap::new());
    let (cancel_tx, _cancel_rx) = watch::channel(HashSet::new());
    let ingress = PipelineIngress {
        eval_queue: Arc::clone(&eval_queue),
        cancellation_queues: vec![cancellation_queue],
        transfers: Arc::clone(&transfers),
        registration_gate: Arc::new(ParkingMutex::new(())),
        cancel_tx,
        runtime: tokio::runtime::Handle::current(),
    };

    for block_id in 100..164 {
        let transfer_id = TransferId::new();
        let (state, _handle) = TransferState::new(transfer_id, vec![block_id]);
        let state = Arc::new(std::sync::Mutex::new(state));
        assert!(ingress.enqueue(
            transfer_id,
            SourceBlocks::External(vec![super::super::source::ExternalBlock::new(
                block_id,
                SequenceHash::new(block_id as u64, None, 0),
            )]),
            Arc::clone(&state),
        ));
        let container = eval_queue.pop_valid().expect("container is queued");
        state.lock().unwrap().set_complete();
        drop(container);
    }

    tokio::time::timeout(Duration::from_millis(250), async {
        while !transfers.is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("terminal transfer states leave the ingress map");
}

#[tokio::test]
async fn duplicate_transfer_id_cannot_replace_the_marker_owner() {
    let transfer_id = TransferId::new();
    let eval_queue = Arc::new(CancellableQueue::<OffloadContainer<()>>::new());
    let cancellation_queue = Arc::new(CancellableQueue::<OffloadContainer<()>>::new());
    let transfers = Arc::new(DashMap::new());
    let (cancel_tx, cancel_rx) = watch::channel(HashSet::new());
    let ingress = PipelineIngress {
        eval_queue: Arc::clone(&eval_queue),
        cancellation_queues: vec![Arc::clone(&cancellation_queue)],
        transfers: Arc::clone(&transfers),
        registration_gate: Arc::new(ParkingMutex::new(())),
        cancel_tx,
        runtime: tokio::runtime::Handle::current(),
    };
    let (old_state, _old_handle) = TransferState::new(transfer_id, vec![1]);
    let old_state = Arc::new(std::sync::Mutex::new(old_state));
    let (duplicate_state, mut duplicate_handle) = TransferState::new(transfer_id, vec![2]);
    let duplicate_state = Arc::new(std::sync::Mutex::new(duplicate_state));

    assert!(ingress.enqueue(
        transfer_id,
        SourceBlocks::External(vec![super::super::source::ExternalBlock::new(
            1,
            SequenceHash::new(1, None, 0),
        )]),
        Arc::clone(&old_state),
    ));
    cancellation_queue.mark_cancelled(transfer_id);
    ingress.cancel_tx.send_modify(|set| {
        set.insert(transfer_id);
    });
    assert!(!ingress.enqueue(
        transfer_id,
        SourceBlocks::External(vec![super::super::source::ExternalBlock::new(
            2,
            SequenceHash::new(2, None, 0),
        )]),
        Arc::clone(&duplicate_state),
    ));
    let duplicate_result = duplicate_handle
        .wait()
        .await
        .expect("duplicate registration publishes a result");
    assert_eq!(duplicate_result.status, TransferStatus::Failed);

    let current = transfers
        .get(&transfer_id)
        .expect("first registration remains");
    assert!(Arc::ptr_eq(current.value(), &old_state));
    drop(current);
    assert!(
        cancellation_queue.is_cancelled(transfer_id),
        "duplicate rejection must preserve the first queue marker",
    );
    assert!(
        cancel_rx.borrow().contains(&transfer_id),
        "duplicate rejection must preserve the first batch marker",
    );

    old_state.lock().unwrap().set_complete();
    tokio::time::timeout(Duration::from_millis(100), async {
        while !transfers.is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the first watcher removes its terminal state");
    assert!(!cancellation_queue.is_cancelled(transfer_id));
    assert!(!cancel_rx.borrow().contains(&transfer_id));
    while let Some(item) = eval_queue.pop_valid() {
        drop(item);
    }
}
