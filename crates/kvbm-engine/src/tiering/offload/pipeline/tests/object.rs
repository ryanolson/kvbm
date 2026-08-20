// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::*;

#[tokio::test]
async fn terminal_status_follows_source_and_pending_guard_release() {
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
    let source = manager
        .match_blocks(&[sequence_hash])
        .pop()
        .expect("match source block");
    let source_observer = source.clone();
    let pending_tracker = Arc::new(PendingTracker::new());
    let pending_guard = pending_tracker
        .try_claim(sequence_hash)
        .expect("claim the source hash");
    let transfer_id = TransferId::new();
    let block_id = source.block_id();
    let (mut state, handle) = TransferState::new(transfer_id, vec![block_id]);
    let cancellation = state
        .cancellation_token()
        .root_unit()
        .expect("create the root cancellation unit");
    assert!(cancellation.claim_commitment());
    state.add_passed([block_id]);
    state.mark_committed_blocks([block_id]);
    let state = Arc::new(std::sync::Mutex::new(state));
    let object_ops: Arc<dyn crate::object::ObjectBlockOps> = Arc::new(FailableObjectBlockOps {
        fail_hashes: std::collections::HashSet::new(),
    });
    let shared = SharedObjectExecutorState {
        object_ops,
        src_layout: LogicalLayoutHandle::G2,
        skip_transfers: false,
        lock_manager: None,
    };
    let mut batch = ResolvedBatch {
        blocks: vec![ResolvedBlock::<crate::G2> {
            transfer_id,
            block_id,
            sequence_hash,
            guard: Some(source),
            pending_guard: Some(pending_guard),
            state: Arc::clone(&state),
        }],
        evicted: Vec::new(),
        timing: TimingTrace::new(),
        cancellation_units: vec![ResolvedCancellationUnit {
            transfer_id,
            state: Arc::clone(&state),
            cancellation,
        }],
    };

    ObjectTransferExecutor::<crate::G2>::execute_transfer(&shared, &mut batch)
        .await
        .expect("execute object transfer");

    assert_eq!(handle.status(), TransferStatus::Complete);
    assert!(pending_tracker.is_empty());
    assert_eq!(source_observer.use_count(), 1);
}

/// Mock ObjectBlockOps that fails specific hashes.
struct FailableObjectBlockOps {
    fail_hashes: std::collections::HashSet<SequenceHash>,
}

#[derive(Clone, Copy)]
enum InterruptedObjectBehavior {
    Pending,
    Panic,
}

struct InterruptedObjectBlockOps {
    started: Arc<tokio::sync::Notify>,
    behavior: InterruptedObjectBehavior,
}

impl crate::object::ObjectBlockOps for FailableObjectBlockOps {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> futures::future::BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        Box::pin(async move { keys.into_iter().map(|h| (h, Some(1))).collect() })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> futures::future::BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        let fail_set = self.fail_hashes.clone();
        Box::pin(async move {
            keys.into_iter()
                .map(|h| if fail_set.contains(&h) { Err(h) } else { Ok(h) })
                .collect()
        })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> futures::future::BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Ok).collect() })
    }
}

impl crate::object::ObjectBlockOps for InterruptedObjectBlockOps {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> futures::future::BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        Box::pin(async move { keys.into_iter().map(|hash| (hash, None)).collect() })
    }

    fn put_blocks(
        &self,
        _keys: Vec<SequenceHash>,
        _layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> futures::future::BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        let started = Arc::clone(&self.started);
        let behavior = self.behavior;
        Box::pin(async move {
            started.notify_one();
            match behavior {
                InterruptedObjectBehavior::Pending => futures::future::pending().await,
                InterruptedObjectBehavior::Panic => {
                    panic!("injected object upload future panic")
                }
            }
        })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> futures::future::BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Ok).collect() })
    }
}

fn owned_object_transfer(
    object_ops: Arc<dyn crate::object::ObjectBlockOps>,
) -> (
    SharedObjectExecutorState,
    ResolvedBatch<crate::G2>,
    kvbm_logical::blocks::ImmutableBlock<crate::G2>,
    Arc<std::sync::Mutex<TransferState>>,
    super::super::super::handle::TransferHandle,
) {
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
    let source = manager
        .match_blocks(&[sequence_hash])
        .pop()
        .expect("match source block");
    let observer = source.clone();
    let transfer_id = TransferId::new();
    let block_id = source.block_id();
    let (mut state, handle) = TransferState::new(transfer_id, vec![block_id]);
    state.add_passed([block_id]);
    state.mark_in_flight([block_id]);
    let state = Arc::new(std::sync::Mutex::new(state));
    let cancellation = test_cancellation_unit(transfer_id, &state);
    let shared = SharedObjectExecutorState {
        object_ops,
        src_layout: LogicalLayoutHandle::G2,
        skip_transfers: false,
        lock_manager: None,
    };
    let batch = ResolvedBatch {
        blocks: vec![ResolvedBlock {
            transfer_id,
            block_id,
            sequence_hash,
            guard: Some(source),
            pending_guard: None,
            state: Arc::clone(&state),
        }],
        evicted: Vec::new(),
        timing: TimingTrace::new(),
        cancellation_units: vec![cancellation],
    };
    (shared, batch, observer, state, handle)
}

#[tokio::test]
async fn abort_after_object_upload_dispatch_retains_source_and_cancellation() {
    let started = Arc::new(tokio::sync::Notify::new());
    let (shared, mut batch, observer, state, handle) =
        owned_object_transfer(Arc::new(InterruptedObjectBlockOps {
            started: Arc::clone(&started),
            behavior: InterruptedObjectBehavior::Pending,
        }));
    let task =
        tokio::spawn(
            async move { ObjectTransferExecutor::execute_transfer(&shared, &mut batch).await },
        );

    started.notified().await;
    task.abort();
    assert!(
        task.await
            .expect_err("abort the object upload task")
            .is_cancelled()
    );

    assert_eq!(observer.use_count(), 2);
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
        "aborted object work must retain its cancellation unit",
    );
}

#[tokio::test]
async fn object_upload_future_panic_retains_source_and_cancellation() {
    let started = Arc::new(tokio::sync::Notify::new());
    let (shared, mut batch, observer, state, handle) =
        owned_object_transfer(Arc::new(InterruptedObjectBlockOps {
            started: Arc::clone(&started),
            behavior: InterruptedObjectBehavior::Panic,
        }));
    let task =
        tokio::spawn(
            async move { ObjectTransferExecutor::execute_transfer(&shared, &mut batch).await },
        );

    started.notified().await;
    assert!(
        task.await
            .expect_err("the object upload task must panic")
            .is_panic()
    );

    assert_eq!(observer.use_count(), 2);
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
        "a panicked object upload must retain its cancellation unit",
    );
}

fn test_hash(n: u64) -> SequenceHash {
    SequenceHash::new(n, None, 0)
}

fn test_cancellation_unit(
    transfer_id: TransferId,
    state: &Arc<std::sync::Mutex<TransferState>>,
) -> ResolvedCancellationUnit {
    let token = state.lock().unwrap().cancellation_token();
    let cancellation = token.root_unit().expect("create root cancellation unit");
    assert!(cancellation.claim_commitment());
    state.lock().unwrap().mark_committed();
    ResolvedCancellationUnit {
        transfer_id,
        state: Arc::clone(state),
        cancellation,
    }
}

#[tokio::test]
async fn test_execute_transfer_partial_failure() {
    use crate::offload::handle::{TransferState, TransferStatus};

    let hash_ok_1 = test_hash(1);
    let hash_fail = test_hash(2);
    let hash_ok_2 = test_hash(3);

    let fail_hashes = [hash_fail].into_iter().collect();
    let object_ops: Arc<dyn crate::object::ObjectBlockOps> =
        Arc::new(FailableObjectBlockOps { fail_hashes });

    let shared = SharedObjectExecutorState {
        object_ops,
        src_layout: LogicalLayoutHandle::G2,
        skip_transfers: false,
        lock_manager: None,
    };

    let transfer_id = crate::offload::handle::TransferId::new();
    let (mut state, handle) = TransferState::new(transfer_id, vec![10, 20, 30]);
    state.add_passed(vec![10, 20, 30]);
    state.mark_in_flight(vec![10, 20, 30]);
    let state_arc = Arc::new(std::sync::Mutex::new(state));

    let blocks = vec![
        ResolvedBlock::<crate::G2> {
            transfer_id,
            block_id: 10,
            sequence_hash: hash_ok_1,
            guard: None,
            pending_guard: None,
            state: state_arc.clone(),
        },
        ResolvedBlock::<crate::G2> {
            transfer_id,
            block_id: 20,
            sequence_hash: hash_fail,
            guard: None,
            pending_guard: None,
            state: state_arc.clone(),
        },
        ResolvedBlock::<crate::G2> {
            transfer_id,
            block_id: 30,
            sequence_hash: hash_ok_2,
            guard: None,
            pending_guard: None,
            state: state_arc.clone(),
        },
    ];

    let mut timing = TimingTrace::new();
    timing.mark_policy_complete();
    timing.mark_precondition_complete();

    let mut batch = ResolvedBatch {
        blocks,
        evicted: Vec::new(),
        timing,
        cancellation_units: vec![test_cancellation_unit(transfer_id, &state_arc)],
    };

    ObjectTransferExecutor::<crate::G2>::execute_transfer(&shared, &mut batch)
        .await
        .expect("execute_transfer should succeed");

    // Block 20 (`hash_fail`) must be in the failed list, not the completed list.
    let state_guard = state_arc.lock().unwrap();
    assert_eq!(state_guard.completed, vec![10, 30]);
    assert_eq!(state_guard.failed, vec![20]);
    assert_eq!(state_guard.in_flight.len(), 0);
    assert_eq!(state_guard.status, TransferStatus::Failed);
    assert!(state_guard.error.is_some());

    // The handle must have the same result.
    drop(state_guard);
    assert_eq!(handle.completed_blocks(), vec![10, 30]);
    assert_eq!(handle.failed_blocks(), vec![20]);
}

#[tokio::test]
async fn test_execute_transfer_all_success() {
    use crate::offload::handle::{TransferState, TransferStatus};

    let hash1 = test_hash(1);
    let hash2 = test_hash(2);

    let object_ops: Arc<dyn crate::object::ObjectBlockOps> = Arc::new(FailableObjectBlockOps {
        fail_hashes: std::collections::HashSet::new(),
    });

    let shared = SharedObjectExecutorState {
        object_ops,
        src_layout: LogicalLayoutHandle::G2,
        skip_transfers: false,
        lock_manager: None,
    };

    let transfer_id = crate::offload::handle::TransferId::new();
    let (mut state, handle) = TransferState::new(transfer_id, vec![10, 20]);
    state.add_passed(vec![10, 20]);
    state.mark_in_flight(vec![10, 20]);
    let state_arc = Arc::new(std::sync::Mutex::new(state));

    let blocks = vec![
        ResolvedBlock::<crate::G2> {
            transfer_id,
            block_id: 10,
            sequence_hash: hash1,
            guard: None,
            pending_guard: None,
            state: state_arc.clone(),
        },
        ResolvedBlock::<crate::G2> {
            transfer_id,
            block_id: 20,
            sequence_hash: hash2,
            guard: None,
            pending_guard: None,
            state: state_arc.clone(),
        },
    ];

    let mut timing = TimingTrace::new();
    timing.mark_policy_complete();
    timing.mark_precondition_complete();

    let mut batch = ResolvedBatch {
        blocks,
        evicted: Vec::new(),
        timing,
        cancellation_units: vec![test_cancellation_unit(transfer_id, &state_arc)],
    };

    ObjectTransferExecutor::<crate::G2>::execute_transfer(&shared, &mut batch)
        .await
        .expect("execute_transfer should succeed");

    let state_guard = state_arc.lock().unwrap();
    assert_eq!(state_guard.completed, vec![10, 20]);
    assert!(state_guard.failed.is_empty());
    assert_eq!(state_guard.status, TransferStatus::Complete);

    drop(state_guard);
    assert_eq!(handle.completed_blocks(), vec![10, 20]);
    assert!(handle.failed_blocks().is_empty());
}

/// Mixed batch: two transfer_ids, one partially fails, the other fully succeeds.
#[tokio::test]
async fn test_execute_transfer_mixed_transfers() {
    use crate::offload::handle::{TransferState, TransferStatus};

    let hash_a1 = test_hash(10);
    let hash_a2_fail = test_hash(20); // transfer A, will fail
    let hash_b1 = test_hash(30);
    let hash_b2 = test_hash(40);

    let fail_hashes = [hash_a2_fail].into_iter().collect();
    let object_ops: Arc<dyn crate::object::ObjectBlockOps> =
        Arc::new(FailableObjectBlockOps { fail_hashes });

    let shared = SharedObjectExecutorState {
        object_ops,
        src_layout: LogicalLayoutHandle::G2,
        skip_transfers: false,
        lock_manager: None,
    };

    // Transfer A: blocks 100, 200 (200 will fail)
    let tid_a = crate::offload::handle::TransferId::new();
    let (mut state_a, handle_a) = TransferState::new(tid_a, vec![100, 200]);
    state_a.add_passed(vec![100, 200]);
    state_a.mark_in_flight(vec![100, 200]);
    let state_a_arc = Arc::new(std::sync::Mutex::new(state_a));

    // Transfer B: blocks 300, 400 (both succeed)
    let tid_b = crate::offload::handle::TransferId::new();
    let (mut state_b, handle_b) = TransferState::new(tid_b, vec![300, 400]);
    state_b.add_passed(vec![300, 400]);
    state_b.mark_in_flight(vec![300, 400]);
    let state_b_arc = Arc::new(std::sync::Mutex::new(state_b));

    let blocks = vec![
        ResolvedBlock::<crate::G2> {
            transfer_id: tid_a,
            block_id: 100,
            sequence_hash: hash_a1,
            guard: None,
            pending_guard: None,
            state: state_a_arc.clone(),
        },
        ResolvedBlock::<crate::G2> {
            transfer_id: tid_a,
            block_id: 200,
            sequence_hash: hash_a2_fail,
            guard: None,
            pending_guard: None,
            state: state_a_arc.clone(),
        },
        ResolvedBlock::<crate::G2> {
            transfer_id: tid_b,
            block_id: 300,
            sequence_hash: hash_b1,
            guard: None,
            pending_guard: None,
            state: state_b_arc.clone(),
        },
        ResolvedBlock::<crate::G2> {
            transfer_id: tid_b,
            block_id: 400,
            sequence_hash: hash_b2,
            guard: None,
            pending_guard: None,
            state: state_b_arc.clone(),
        },
    ];

    let mut timing = TimingTrace::new();
    timing.mark_policy_complete();
    timing.mark_precondition_complete();

    let mut batch = ResolvedBatch {
        blocks,
        evicted: Vec::new(),
        timing,
        cancellation_units: vec![
            test_cancellation_unit(tid_a, &state_a_arc),
            test_cancellation_unit(tid_b, &state_b_arc),
        ],
    };

    ObjectTransferExecutor::<crate::G2>::execute_transfer(&shared, &mut batch)
        .await
        .expect("execute_transfer should succeed");

    // Transfer A: block 100 succeeded, block 200 failed
    let sa = state_a_arc.lock().unwrap();
    assert_eq!(sa.completed, vec![100]);
    assert_eq!(sa.failed, vec![200]);
    assert_eq!(sa.status, TransferStatus::Failed);
    assert!(sa.error.is_some());
    drop(sa);

    assert_eq!(handle_a.completed_blocks(), vec![100]);
    assert_eq!(handle_a.failed_blocks(), vec![200]);

    // Transfer B: both succeeded
    let sb = state_b_arc.lock().unwrap();
    assert_eq!(sb.completed, vec![300, 400]);
    assert!(sb.failed.is_empty());
    assert_eq!(sb.status, TransferStatus::Complete);
    drop(sb);

    assert_eq!(handle_b.completed_blocks(), vec![300, 400]);
    assert!(handle_b.failed_blocks().is_empty());
}
