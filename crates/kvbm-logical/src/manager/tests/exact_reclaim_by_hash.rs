use std::collections::HashSet;

use dynamo_tokens::TokenBlockSequence;

use super::*;
use crate::ExactReclaimNameError;
use crate::pools::store::DebugStoreSnapshot;
use crate::testing::{TEST_SALT, create_test_manager_with_backend};

fn valued_manager(block_count: usize) -> BlockManager<TestBlockData> {
    create_test_manager_with_backend(block_count, |builder| {
        builder
            .block_size(4)
            .with_valued_lineage_backend(ScorerParams::default())
    })
}

fn register_roots(
    manager: &BlockManager<TestBlockData>,
    count: usize,
) -> Vec<ImmutableBlock<TestBlockData>> {
    (0..count)
        .map(|seed| {
            let token = create_test_token_block_from_iota(seed as u32);
            let mutable = manager
                .allocate_blocks(1)
                .expect("allocate one root block")
                .pop()
                .expect("one mutable block");
            manager.register_block(mutable.complete(&token).expect("complete one root block"))
        })
        .collect()
}

fn register_chain(
    manager: &BlockManager<TestBlockData>,
    tokens: &[u32],
) -> Vec<ImmutableBlock<TestBlockData>> {
    let sequence = TokenBlockSequence::from_slice(tokens, 4, Some(TEST_SALT));
    sequence
        .blocks()
        .iter()
        .map(|token_block| {
            let mutable = manager
                .allocate_blocks(1)
                .expect("allocate one lineage block")
                .pop()
                .expect("one mutable block");
            manager.register_block(
                mutable
                    .complete(token_block)
                    .expect("complete one lineage block"),
            )
        })
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
fn manager_reports_exact_reclaim_backend_support() {
    let valued = valued_manager(1);
    let lru =
        create_test_manager_with_backend::<TestBlockData>(1, |builder| builder.with_lru_backend());

    assert!(valued.supports_exact_reclaim());
    assert!(!lru.supports_exact_reclaim());
}

#[test]
fn point_naming_rejects_an_unsupported_backend() {
    let manager =
        create_test_manager_with_backend::<TestBlockData>(1, |builder| builder.with_lru_backend());
    let hash = create_test_token_block_from_iota(1).kvbm_sequence_hash();

    assert_eq!(
        manager.name_complete_inactive_entry_by_hash(hash),
        Err(ExactReclaimNameError::UnsupportedBackend)
    );
}

#[test]
fn point_naming_rejects_an_active_leaf_without_pool_mutation() {
    let manager = valued_manager(2);
    let registered = register_chain(&manager, &[10, 11, 12, 13, 14, 15, 16, 17]);
    let leaf_hash = registered.last().expect("lineage leaf").sequence_hash();
    let before = state(&manager);

    assert_eq!(
        manager.name_complete_inactive_entry_by_hash(leaf_hash),
        Err(ExactReclaimNameError::StaleCandidate)
    );
    assert_eq!(state(&manager), before);
}

#[test]
fn point_naming_rejects_an_inactive_interior_root_without_pool_mutation() {
    let manager = valued_manager(2);
    let registered = register_chain(&manager, &[20, 21, 22, 23, 24, 25, 26, 27]);
    let root_hash = registered.first().expect("lineage root").sequence_hash();
    drop(registered);
    let before = state(&manager);

    assert_eq!(
        manager.name_complete_inactive_entry_by_hash(root_hash),
        Err(ExactReclaimNameError::NotCompleteInactiveEntry)
    );
    assert_eq!(state(&manager), before);
}

#[test]
fn point_naming_reaches_a_leaf_outside_the_bounded_candidate_snapshot() {
    const BLOCKS: usize = 4_100;
    let manager = valued_manager(BLOCKS);
    let registered = register_roots(&manager, BLOCKS);
    let hashes = registered
        .iter()
        .map(ImmutableBlock::sequence_hash)
        .collect::<Vec<_>>();
    drop(registered);

    let candidate_hashes = manager
        .inactive_candidates(manager.total_blocks())
        .into_iter()
        .map(|candidate| candidate.seq_hash)
        .collect::<HashSet<_>>();
    let outside = hashes
        .into_iter()
        .find(|hash| !candidate_hashes.contains(hash))
        .expect("the valued candidate snapshot is bounded");

    let name = manager
        .name_complete_inactive_entry_by_hash(outside)
        .expect("point naming reaches the exact inactive root");
    let plan = manager
        .refresh_and_combine_exact_reclaim(&[name])
        .expect("refresh the point-named root");

    assert_eq!(plan.reclaimed_blocks(), 1);
}
