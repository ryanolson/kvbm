use std::sync::Arc;

use dynamo_tokens::TokenBlockSequence;

use super::*;
use crate::pools::store::DebugStoreSnapshot;
use crate::testing::{TEST_SALT, create_test_manager_with_backend};
use crate::{BlockId, ExactAllocationError, ExactInactiveVictim, InactiveCandidate};

fn valued_manager(block_count: usize) -> BlockManager<TestBlockData> {
    create_test_manager_with_backend(block_count, |builder| {
        builder
            .block_size(1)
            .with_valued_lineage_backend(ScorerParams::default())
    })
}

fn inactive_chain(
    manager: &BlockManager<TestBlockData>,
    tokens: &[u32],
) -> Vec<(SequenceHash, BlockId)> {
    let blocks = TokenBlockSequence::from_slice(tokens, 1, Some(TEST_SALT))
        .blocks()
        .iter()
        .map(|token_block| {
            let mutable = manager
                .allocate_blocks(1)
                .expect("allocate one lineage block")
                .pop()
                .expect("one allocated block");
            let complete = mutable
                .complete(token_block)
                .expect("complete one lineage block");
            manager.register_block(complete)
        })
        .collect::<Vec<_>>();
    let source_blocks = blocks
        .iter()
        .map(|block| (block.sequence_hash(), block.block_id()))
        .collect();
    drop(blocks);
    source_blocks
}

fn victim(
    manager: &BlockManager<TestBlockData>,
    (seq_hash, block_id): (SequenceHash, BlockId),
) -> ExactInactiveVictim {
    let store = manager.store_for_test();
    ExactInactiveVictim {
        manager_id: manager.id(),
        block_id,
        seq_hash,
        generation: store.slot_generation_for_test(block_id),
        inactive_epoch: store.slot_inactive_epoch_for_test(block_id),
    }
}

fn leaf_to_root_victims(
    manager: &BlockManager<TestBlockData>,
    source_blocks: &[(SequenceHash, BlockId)],
) -> Vec<ExactInactiveVictim> {
    source_blocks
        .iter()
        .rev()
        .copied()
        .map(|block| victim(manager, block))
        .collect()
}

#[derive(Debug, PartialEq)]
struct PoolState {
    reset_len: usize,
    inactive_len: usize,
    available_blocks: usize,
    candidates: Vec<InactiveCandidate>,
    store: DebugStoreSnapshot,
}

fn unchanged_state(manager: &BlockManager<TestBlockData>) -> PoolState {
    PoolState {
        reset_len: manager.reset_len(),
        inactive_len: manager.inactive_len(),
        available_blocks: manager.available_blocks(),
        candidates: manager.inactive_candidates(manager.total_blocks()),
        store: manager.store_for_test().debug_snapshot(),
    }
}

#[test]
fn exact_reclaim_rejects_reset_capacity_drift_without_mutation() {
    let manager = valued_manager(3);
    let source_blocks = inactive_chain(&manager, &[10, 11]);
    let victims = leaf_to_root_victims(&manager, &source_blocks);
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(1, 0, &victims),
        Err(ExactAllocationError::ResetCapacityDrift {
            expected: 0,
            actual: 1,
        })
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_rejects_an_inactive_epoch_aba_without_mutation() {
    let manager = valued_manager(1);
    let source_blocks = inactive_chain(&manager, &[20]);
    let stale = victim(&manager, source_blocks[0]);

    let cached = manager.match_blocks(&[stale.seq_hash]);
    assert_eq!(cached.len(), 1, "the original tenure is a cache hit");
    drop(cached);
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(1, 0, &[stale]),
        Err(ExactAllocationError::StaleVictim { .. })
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_rejects_a_hash_mismatch_without_mutation() {
    let manager = valued_manager(2);
    let first = inactive_chain(&manager, &[25]);
    let second = inactive_chain(&manager, &[26]);
    let mismatched_hash = ExactInactiveVictim {
        seq_hash: second[0].0,
        ..victim(&manager, first[0])
    };
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(1, 0, &[mismatched_hash]),
        Err(ExactAllocationError::StaleVictim { .. })
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_rejects_a_reused_generation_without_mutation() {
    let manager = valued_manager(1);
    let first = inactive_chain(&manager, &[27]);
    let stale = victim(&manager, first[0]);

    let allocated = manager
        .allocate_blocks_with_exact_inactive(1, &[stale])
        .expect("evict the old generation before reuse");
    drop(allocated);
    let current = inactive_chain(&manager, &[28]);
    assert_eq!(current[0].1, stale.block_id);
    assert_ne!(
        manager
            .store_for_test()
            .slot_generation_for_test(stale.block_id),
        stale.generation
    );
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(1, 0, &[stale]),
        Err(ExactAllocationError::StaleVictim { .. })
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_rejects_a_wrong_manager_without_mutation() {
    let manager = valued_manager(2);
    let source_blocks = inactive_chain(&manager, &[30]);
    let other_manager = valued_manager(1);
    let wrong_manager = ExactInactiveVictim {
        manager_id: other_manager.id(),
        ..victim(&manager, source_blocks[0])
    };
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(1, 1, &[wrong_manager]),
        Err(ExactAllocationError::WrongManager { .. })
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_rejects_duplicate_slots_without_mutation() {
    let manager = valued_manager(2);
    let source_blocks = inactive_chain(&manager, &[40]);
    let victim = victim(&manager, source_blocks[0]);
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(1, 1, &[victim, victim]),
        Err(ExactAllocationError::DuplicateVictim { block_id }) if block_id == victim.block_id
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_rejects_insufficient_total_capacity_without_mutation() {
    let manager = valued_manager(3);
    let source_blocks = inactive_chain(&manager, &[50]);
    let victims = leaf_to_root_victims(&manager, &source_blocks);
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(4, 2, &victims),
        Err(ExactAllocationError::InsufficientCapacity {
            requested: 4,
            available: 3,
        })
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_rejects_a_zero_count_with_victims_without_mutation() {
    let manager = valued_manager(2);
    let source_blocks = inactive_chain(&manager, &[55]);
    let victims = leaf_to_root_victims(&manager, &source_blocks);
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(0, 1, &victims),
        Err(ExactAllocationError::ZeroCountWithVictims { supplied: 1 })
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_zero_count_empty_plan_is_a_reset_snapshot_noop() {
    let manager = valued_manager(2);
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(0, 1, &[]),
        Err(ExactAllocationError::ResetCapacityDrift {
            expected: 1,
            actual: 2,
        })
    ));
    assert_eq!(unchanged_state(&manager), before);

    let blocks = manager
        .allocate_blocks_with_exact_reclaim(0, 2, &[])
        .expect("an empty zero-count plan is a no-op");

    assert!(blocks.is_empty());
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_silent_defers_a_panicking_observer_until_the_token_notifies() {
    let manager = valued_manager(2);
    let source_blocks = inactive_chain(&manager, &[57]);
    let victims = leaf_to_root_victims(&manager, &source_blocks);
    let observer: Arc<dyn BlockEvictionObserver> = Arc::new(|_: &[SequenceHash]| {
        panic!("the observer must run only when the notification token is consumed")
    });
    manager.observe_evictions(&observer);

    let (allocated, notification) = manager
        .allocate_blocks_with_exact_reclaim_silent(1, 1, &victims)
        .expect("the silent transaction returns before observer delivery");

    assert_eq!(allocated.len(), 1);
    assert_eq!(manager.inactive_len(), 0);
    assert_eq!(manager.reset_len(), 1);
    assert!(
        manager.match_blocks(&[source_blocks[0].0]).is_empty(),
        "the physical eviction completed before deferred notification"
    );
    let reserve_ready = manager
        .allocate_blocks(1)
        .expect("the overshoot reset slot remains ready before notification");
    assert_eq!(reserve_ready.len(), 1);
    drop(reserve_ready);

    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| notification.notify())).is_err(),
        "the explicit notification owns the observer panic"
    );
    assert_eq!(manager.inactive_len(), 0);
    assert_eq!(manager.reset_len(), 1);
    drop(allocated);
}

#[test]
fn exact_reclaim_accepts_an_oversized_complete_entry_and_leaves_overshoot_reset() {
    let manager = valued_manager(3);
    let source_blocks = inactive_chain(&manager, &[60, 61, 62]);
    let victims = leaf_to_root_victims(&manager, &source_blocks);

    let allocated = manager
        .allocate_blocks_with_exact_reclaim(1, 0, &victims)
        .expect("reclaim the full entry before allocation");

    assert_eq!(allocated.len(), 1);
    assert_eq!(manager.inactive_len(), 0, "every authorized victim is gone");
    assert_eq!(
        manager.reset_len(),
        2,
        "overshoot remains available as reset"
    );
    assert_eq!(manager.available_blocks(), 2);
    drop(allocated);
}

#[test]
fn exact_reclaim_rejects_parent_before_child_without_mutation() {
    let manager = valued_manager(2);
    let source_blocks = inactive_chain(&manager, &[70, 71]);
    let parent_before_child = source_blocks
        .iter()
        .copied()
        .map(|block| victim(&manager, block))
        .collect::<Vec<_>>();
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(1, 0, &parent_before_child),
        Err(ExactAllocationError::InvalidVictimOrder { .. })
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_rejects_a_plan_that_omits_a_live_child_without_mutation() {
    let manager = valued_manager(2);
    let source_blocks = inactive_chain(&manager, &[80, 81]);
    let parent_only = [victim(&manager, source_blocks[0])];
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(1, 0, &parent_only),
        Err(ExactAllocationError::IncompleteVictimSet { .. })
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_rejects_a_leaf_only_plan_that_omits_real_ancestors_without_mutation() {
    let manager = valued_manager(3);
    let source_blocks = inactive_chain(&manager, &[85, 86, 87]);
    let leaf_only = [victim(
        &manager,
        *source_blocks.last().expect("lineage leaf"),
    )];
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(1, 0, &leaf_only),
        Err(ExactAllocationError::MissingVictimAncestor { .. })
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_rejects_a_plan_that_omits_a_higher_real_ancestor_without_mutation() {
    let manager = valued_manager(3);
    let source_blocks = inactive_chain(&manager, &[88, 89, 90]);
    let mut partial = leaf_to_root_victims(&manager, &source_blocks);
    partial.pop();
    let before = unchanged_state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_exact_reclaim(1, 0, &partial),
        Err(ExactAllocationError::MissingVictimAncestor { .. })
    ));
    assert_eq!(unchanged_state(&manager), before);
}

#[test]
fn exact_reclaim_validates_then_reclaims_in_leaf_to_root_order_before_allocation() {
    let manager = valued_manager(3);
    let source_blocks = inactive_chain(&manager, &[90, 91, 92]);
    let victims = leaf_to_root_victims(&manager, &source_blocks);

    let allocated = manager
        .allocate_blocks_with_exact_reclaim(2, 0, &victims)
        .expect("the complete leaf-to-root plan is valid");

    assert_eq!(allocated.len(), 2);
    assert_eq!(manager.inactive_len(), 0);
    assert_eq!(manager.reset_len(), 1);
    assert_eq!(manager.available_blocks(), 1);
    assert_eq!(
        manager.metrics().snapshot().evictions,
        3,
        "the transaction reports every reclaimed victim"
    );
    drop(allocated);
}

#[test]
fn exact_reclaim_accepts_disjoint_complete_entries() {
    let manager = valued_manager(4);
    let first = inactive_chain(&manager, &[100, 101]);
    let second = inactive_chain(&manager, &[200, 201]);
    let mut victims = leaf_to_root_victims(&manager, &first);
    victims.extend(leaf_to_root_victims(&manager, &second));

    let allocated = manager
        .allocate_blocks_with_exact_reclaim(2, 0, &victims)
        .expect("disjoint complete entries have independent removal orders");

    assert_eq!(allocated.len(), 2);
    assert_eq!(manager.inactive_len(), 0);
    assert_eq!(manager.reset_len(), 2);
    drop(allocated);
}
