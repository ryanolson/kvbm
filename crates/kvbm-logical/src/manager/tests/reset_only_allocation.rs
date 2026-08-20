use std::sync::{Arc, Mutex};

use super::*;
use crate::metrics::MetricsSnapshot;
use crate::pools::store::{DebugStoreSnapshot, SlotKind};
use crate::testing::create_test_manager_with_backend;
use crate::{BlockId, InactiveCandidate};

fn valued_manager(block_count: usize) -> BlockManager<TestBlockData> {
    create_test_manager_with_backend(block_count, |builder| {
        builder
            .block_size(4)
            .with_valued_lineage_backend(ScorerParams::default())
    })
}

fn register_inactive(
    manager: &BlockManager<TestBlockData>,
    token_start: u32,
) -> (BlockId, SequenceHash) {
    let token_block = create_iota_token_block(token_start, 4);
    let mutable = manager
        .allocate_blocks(1)
        .expect("allocate one block")
        .pop()
        .expect("one mutable block");
    let immutable = manager.register_block(mutable.complete(&token_block).expect("complete block"));
    let block = (immutable.block_id(), immutable.sequence_hash());
    drop(immutable);
    block
}

#[derive(Debug, PartialEq)]
struct LogicalStoreState {
    reset_len: usize,
    inactive_len: usize,
    available_blocks: usize,
    candidates: Vec<InactiveCandidate>,
    store: DebugStoreSnapshot,
}

fn state(manager: &BlockManager<TestBlockData>) -> LogicalStoreState {
    LogicalStoreState {
        reset_len: manager.reset_len(),
        inactive_len: manager.inactive_len(),
        available_blocks: manager.available_blocks(),
        candidates: manager.inactive_candidates(manager.total_blocks()),
        store: manager.store_for_test().debug_snapshot(),
    }
}

fn observer(calls: &Arc<Mutex<Vec<SequenceHash>>>) -> Arc<dyn BlockEvictionObserver> {
    let calls = Arc::clone(calls);
    Arc::new(move |hashes: &[SequenceHash]| {
        calls
            .lock()
            .expect("observer lock")
            .extend_from_slice(hashes);
    })
}

#[test]
fn reset_only_allocation_uses_reset_slots_without_eviction() {
    let manager = valued_manager(3);
    let (inactive_block_id, inactive_hash) = register_inactive(&manager, 10);
    let count = 2;
    let before = state(&manager);
    let metrics_before = manager.metrics().snapshot();
    let reset_ids = before.store.free.clone();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let eviction_observer = observer(&calls);
    manager.observe_evictions(&eviction_observer);

    let allocated = manager
        .allocate_blocks_from_reset(count)
        .expect("the reset pool has two slots");

    assert_eq!(
        allocated
            .iter()
            .map(|block| block.block_id())
            .collect::<Vec<_>>(),
        reset_ids,
        "the allocation uses only reset slots"
    );
    let after = state(&manager);
    assert_eq!(after.inactive_len, before.inactive_len);
    assert_eq!(after.candidates, before.candidates);
    assert_eq!(
        after.store.slots[inactive_block_id],
        SlotKind::Inactive(inactive_hash),
        "the inactive slot remains untouched"
    );
    assert_eq!(
        manager.metrics().snapshot(),
        MetricsSnapshot {
            allocations: metrics_before.allocations + count as u64,
            allocations_from_reset: metrics_before.allocations_from_reset + count as u64,
            inflight_mutable: metrics_before.inflight_mutable + count as i64,
            reset_pool_size: metrics_before.reset_pool_size - count as i64,
            ..metrics_before
        },
        "reset-only allocation updates only the reset allocation metrics"
    );

    drop(allocated);

    assert_eq!(
        manager.metrics().snapshot(),
        MetricsSnapshot {
            allocations: metrics_before.allocations + count as u64,
            allocations_from_reset: metrics_before.allocations_from_reset + count as u64,
            ..metrics_before
        },
        "mutable guard drops restore both pool gauges"
    );
    assert!(
        calls.lock().expect("observer lock").is_empty(),
        "reset-only allocation does not report an eviction"
    );
}

#[test]
fn reset_only_allocation_rejects_when_inactive_eviction_would_be_required() {
    let manager = valued_manager(2);
    register_inactive(&manager, 20);
    let before = state(&manager);
    let metrics_before = manager.metrics().snapshot();
    assert_eq!(before.reset_len, 1);
    assert_eq!(before.inactive_len, 1);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let eviction_observer = observer(&calls);
    manager.observe_evictions(&eviction_observer);

    assert!(
        manager.allocate_blocks_from_reset(2).is_none(),
        "the generic allocator could succeed only by evicting inactive state"
    );
    assert_eq!(
        state(&manager),
        before,
        "a rejected reset-only request leaves the logical store byte-for-byte unchanged"
    );
    assert_eq!(
        manager.metrics().snapshot(),
        metrics_before,
        "a rejected reset-only request leaves the full metric snapshot unchanged"
    );
    assert!(
        calls.lock().expect("observer lock").is_empty(),
        "a rejected reset-only request does not call observers"
    );
}

#[test]
fn reset_only_allocation_zero_count_is_a_noop() {
    let manager = valued_manager(2);
    register_inactive(&manager, 30);
    let before = state(&manager);
    let metrics_before = manager.metrics().snapshot();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let eviction_observer = observer(&calls);
    manager.observe_evictions(&eviction_observer);

    let allocated = manager
        .allocate_blocks_from_reset(0)
        .expect("zero-count allocation succeeds");

    assert!(allocated.is_empty());
    assert_eq!(state(&manager), before);
    assert_eq!(
        manager.metrics().snapshot(),
        metrics_before,
        "zero-count allocation leaves the full metric snapshot unchanged"
    );
    assert!(calls.lock().expect("observer lock").is_empty());
}
