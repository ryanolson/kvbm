use std::sync::Arc;

use dynamo_tokens::TokenBlockSequence;

use super::*;
use crate::pools::InactiveFeatures;
use crate::pools::store::DebugStoreSnapshot;
use crate::testing::{TEST_SALT, create_test_manager_with_backend};
use crate::{
    BlockId, ExactReclaimEntryPlan, ExactReclaimExecuteError, ExactReclaimNameError,
    ExactReclaimRefreshError, InactiveCandidate,
};

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

fn register_divergent(
    manager: &BlockManager<TestBlockData>,
    tokens: &[u32],
    position: usize,
) -> ImmutableBlock<TestBlockData> {
    let sequence = TokenBlockSequence::from_slice(tokens, 1, Some(TEST_SALT));
    let mutable = manager
        .allocate_blocks(1)
        .expect("allocate one divergent block")
        .pop()
        .expect("one allocated block");
    let complete = mutable
        .complete(&sequence.blocks()[position])
        .expect("complete one divergent block");
    manager.register_block(complete)
}

fn candidate_for(manager: &BlockManager<TestBlockData>, hash: SequenceHash) -> InactiveCandidate {
    manager
        .inactive_candidates(manager.total_blocks())
        .into_iter()
        .find(|candidate| candidate.seq_hash == hash)
        .expect("inactive leaf candidate")
}

fn named_entry(manager: &BlockManager<TestBlockData>, hash: SequenceHash) -> ExactReclaimEntryPlan {
    manager
        .name_complete_inactive_entry(candidate_for(manager, hash))
        .expect("name complete inactive entry")
}

#[derive(Debug, PartialEq)]
struct PoolState {
    reset_len: usize,
    inactive_len: usize,
    available_blocks: usize,
    candidates: Vec<InactiveCandidate>,
    store: DebugStoreSnapshot,
}

fn state(manager: &BlockManager<TestBlockData>) -> PoolState {
    PoolState {
        reset_len: manager.reset_len(),
        inactive_len: manager.inactive_len(),
        available_blocks: manager.available_blocks(),
        candidates: manager.inactive_candidates(manager.total_blocks()),
        store: manager.store_for_test().debug_snapshot(),
    }
}

#[test]
fn opaque_reclaim_rejects_naming_on_a_non_lineage_backend_without_mutation() {
    let manager =
        create_test_manager_with_backend(1, |builder| builder.block_size(1).with_lru_backend());
    let token = TokenBlockSequence::from_slice(&[1], 1, Some(TEST_SALT));
    let mutable = manager
        .allocate_blocks(1)
        .expect("allocate one block")
        .pop()
        .expect("one allocated block");
    let complete = mutable
        .complete(&token.blocks()[0])
        .expect("complete one block");
    let immutable = manager.register_block(complete);
    let hash = immutable.sequence_hash();
    let block_id = immutable.block_id();
    drop(immutable);
    let candidate = InactiveCandidate {
        seq_hash: hash,
        block_id,
        generation: manager.store_for_test().slot_generation_for_test(block_id),
        inactive_epoch: manager
            .store_for_test()
            .slot_inactive_epoch_for_test(block_id),
        features: InactiveFeatures {
            poisoned: false,
            is_leaf: true,
            age_ticks: None,
            freq_estimate: None,
            max_fanout: None,
            evict_rank: None,
        },
    };
    let before = state(&manager);

    assert_eq!(
        manager.name_complete_inactive_entry(candidate),
        Err(ExactReclaimNameError::UnsupportedBackend)
    );
    assert_eq!(state(&manager), before);
}

#[test]
fn opaque_reclaim_rejects_an_empty_refresh_without_mutation() {
    let manager = valued_manager(1);
    let before = state(&manager);

    assert!(matches!(
        manager.refresh_and_combine_exact_reclaim(&[]),
        Err(ExactReclaimRefreshError::EmptyPlan)
    ));
    assert_eq!(state(&manager), before);
}

#[test]
fn opaque_reclaim_rejects_a_stale_candidate_at_naming_without_mutation() {
    let manager = valued_manager(1);
    let source_blocks = inactive_chain(&manager, &[10]);
    let candidate = candidate_for(&manager, source_blocks[0].0);

    let cached = manager.match_blocks(&[candidate.seq_hash]);
    assert_eq!(cached.len(), 1, "the candidate becomes active");
    drop(cached);
    let before = state(&manager);

    assert_eq!(
        manager.name_complete_inactive_entry(candidate),
        Err(ExactReclaimNameError::StaleCandidate)
    );
    assert_eq!(state(&manager), before);
}

#[test]
fn opaque_reclaim_refreshes_the_full_private_lineage_closure() {
    let manager = valued_manager(3);
    let source_blocks = inactive_chain(&manager, &[20, 21, 22]);
    let entry = named_entry(&manager, source_blocks.last().expect("lineage leaf").0);

    let plan = manager
        .refresh_and_combine_exact_reclaim(&[entry])
        .expect("refresh complete lineage");
    assert_eq!(plan.reclaimed_blocks(), 3);

    let allocated = manager
        .allocate_blocks_with_fresh_exact_reclaim(1, plan)
        .expect("atomically reclaim the complete closure");
    assert_eq!(allocated.len(), 1);
    assert_eq!(manager.inactive_len(), 0);
    assert_eq!(manager.reset_len(), 2);
    drop(allocated);
}

#[test]
fn opaque_reclaim_combines_disjoint_entries() {
    let manager = valued_manager(4);
    let first = inactive_chain(&manager, &[30, 31]);
    let second = inactive_chain(&manager, &[40, 41]);
    let first_entry = named_entry(&manager, first.last().expect("first leaf").0);
    let second_entry = named_entry(&manager, second.last().expect("second leaf").0);

    let plan = manager
        .refresh_and_combine_exact_reclaim(&[first_entry, second_entry])
        .expect("combine disjoint entries");
    assert_eq!(plan.reclaimed_blocks(), 4);

    let allocated = manager
        .allocate_blocks_with_fresh_exact_reclaim(2, plan)
        .expect("reclaim both entries in one transaction");
    assert_eq!(allocated.len(), 2);
    assert_eq!(manager.inactive_len(), 0);
    assert_eq!(manager.reset_len(), 2);
    drop(allocated);
}

#[test]
fn opaque_reclaim_rejects_entries_with_a_shared_physical_slot() {
    let manager = valued_manager(8);
    let branch = inactive_chain(&manager, &[50, 51, 52, 53]);
    let sibling = register_divergent(&manager, &[50, 51, 99], 2);
    let sibling_hash = sibling.sequence_hash();
    drop(sibling);

    let branch_entry = named_entry(&manager, branch.last().expect("branch leaf").0);
    let sibling_entry = named_entry(&manager, sibling_hash);
    let before = state(&manager);

    assert!(matches!(
        manager.refresh_and_combine_exact_reclaim(&[branch_entry, sibling_entry]),
        Err(ExactReclaimRefreshError::SharedPhysicalSlot {
            first: 0,
            second: 1,
        })
    ));
    assert_eq!(state(&manager), before);
}

#[test]
fn opaque_reclaim_rejects_an_old_fresh_plan_then_refreshes_after_inactive_aba() {
    let manager = valued_manager(1);
    let source_blocks = inactive_chain(&manager, &[60]);
    let entry = named_entry(&manager, source_blocks[0].0);
    let stale_plan = manager
        .refresh_and_combine_exact_reclaim(std::slice::from_ref(&entry))
        .expect("initial fresh plan");

    let cached = manager.match_blocks(&[source_blocks[0].0]);
    assert_eq!(cached.len(), 1, "the original entry becomes active");
    drop(cached);
    let before = state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_fresh_exact_reclaim(1, stale_plan),
        Err(ExactReclaimExecuteError::StalePlan)
    ));
    assert_eq!(state(&manager), before);

    let refreshed = manager
        .refresh_and_combine_exact_reclaim(&[entry])
        .expect("the same registration survives inactive-epoch ABA");
    let allocated = manager
        .allocate_blocks_with_fresh_exact_reclaim(1, refreshed)
        .expect("the refreshed plan uses the current inactive epoch");
    assert_eq!(allocated.len(), 1);
    assert_eq!(manager.inactive_len(), 0);
    drop(allocated);
}

#[test]
fn opaque_reclaim_refresh_rejects_mutable_slot_reuse_even_with_the_same_hash() {
    let manager = valued_manager(1);
    let first = inactive_chain(&manager, &[70]);
    let entry = named_entry(&manager, first[0].0);

    let evicted = manager
        .allocate_blocks(1)
        .expect("evict the original generation");
    drop(evicted);
    let replacement = inactive_chain(&manager, &[70]);
    assert_eq!(replacement[0].1, first[0].1, "the physical slot was reused");
    let before = state(&manager);

    assert!(matches!(
        manager.refresh_and_combine_exact_reclaim(&[entry]),
        Err(ExactReclaimRefreshError::EntryUnavailable { index: 0 })
    ));
    assert_eq!(state(&manager), before);
}

#[test]
fn opaque_reclaim_rejects_reset_drift_without_mutation() {
    let manager = valued_manager(3);
    let source_blocks = inactive_chain(&manager, &[80, 81]);
    let entry = named_entry(&manager, source_blocks.last().expect("lineage leaf").0);
    let plan = manager
        .refresh_and_combine_exact_reclaim(&[entry])
        .expect("fresh plan captures one reset slot");
    let unrelated = manager
        .allocate_blocks(1)
        .expect("consume the reset slot before execute");
    let before = state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_fresh_exact_reclaim(1, plan),
        Err(ExactReclaimExecuteError::ResetCapacityDrift)
    ));
    assert_eq!(state(&manager), before);
    drop(unrelated);
}

#[test]
fn opaque_reclaim_rejects_insufficient_capacity_without_mutation() {
    let manager = valued_manager(2);
    let source_blocks = inactive_chain(&manager, &[85]);
    let entry = named_entry(&manager, source_blocks[0].0);
    let plan = manager
        .refresh_and_combine_exact_reclaim(&[entry])
        .expect("fresh plan");
    let before = state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_fresh_exact_reclaim(3, plan),
        Err(ExactReclaimExecuteError::InsufficientCapacity)
    ));
    assert_eq!(state(&manager), before);
}

#[test]
fn opaque_reclaim_rejects_zero_count_without_mutation() {
    let manager = valued_manager(2);
    let source_blocks = inactive_chain(&manager, &[86]);
    let entry = named_entry(&manager, source_blocks[0].0);
    let plan = manager
        .refresh_and_combine_exact_reclaim(&[entry])
        .expect("fresh plan");
    let before = state(&manager);

    assert!(matches!(
        manager.allocate_blocks_with_fresh_exact_reclaim(0, plan),
        Err(ExactReclaimExecuteError::InvalidRequest)
    ));
    assert_eq!(state(&manager), before);
}

#[test]
fn opaque_reclaim_rejects_a_fresh_plan_from_another_manager() {
    let owner = valued_manager(1);
    let source_blocks = inactive_chain(&owner, &[90]);
    let entry = named_entry(&owner, source_blocks[0].0);
    let plan = owner
        .refresh_and_combine_exact_reclaim(std::slice::from_ref(&entry))
        .expect("owner builds a fresh plan");
    let other = valued_manager(1);
    let before = state(&other);

    assert!(matches!(
        other.refresh_and_combine_exact_reclaim(&[entry]),
        Err(ExactReclaimRefreshError::WrongManager)
    ));
    assert_eq!(state(&other), before);

    assert!(matches!(
        other.allocate_blocks_with_fresh_exact_reclaim(1, plan),
        Err(ExactReclaimExecuteError::WrongManager)
    ));
    assert_eq!(state(&other), before);
}

#[test]
fn opaque_reclaim_silent_execution_defers_notification() {
    let manager = valued_manager(2);
    let source_blocks = inactive_chain(&manager, &[100]);
    let entry = named_entry(&manager, source_blocks[0].0);
    let plan = manager
        .refresh_and_combine_exact_reclaim(&[entry])
        .expect("fresh plan");
    let observer: Arc<dyn BlockEvictionObserver> = Arc::new(|_: &[SequenceHash]| {
        panic!("the observer runs only when the notification is consumed")
    });
    manager.observe_evictions(&observer);

    let (allocated, notification) = manager
        .allocate_blocks_with_fresh_exact_reclaim_silent(1, plan)
        .expect("silent transaction returns before observer delivery");

    assert_eq!(allocated.len(), 1);
    assert_eq!(manager.inactive_len(), 0);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| notification.notify())).is_err(),
        "the explicit notification owns the observer panic"
    );
    drop(allocated);
}
