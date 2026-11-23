// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::{TokenBlock, TokenBlockSequence};

use super::TestBlockData;
use crate::manager::BlockManager;
use crate::pools::RebindOutcome;
use crate::pools::store::LineageAudit;
use crate::pools::{BlockDuplicationPolicy, InactiveCandidate};
use crate::testing::{TEST_SALT, create_test_manager_with_backend};
use crate::{CompleteBlock, ImmutableBlock, KvbmSequenceHashProvider, SequenceHash};

#[test]
fn two_slot_reset_leaves_no_ghost() {
    const CAPACITY: usize = 2;
    const ANCESTORS: usize = 8;
    let manager = create_test_manager_with_backend::<TestBlockData>(CAPACITY, |builder| {
        builder.block_size(1).with_lineage_backend()
    });
    let tokens = (0..=ANCESTORS as u32).collect::<Vec<_>>();
    let sequence = TokenBlockSequence::from_slice(&tokens, 1, Some(TEST_SALT));
    let (tail, ancestors) = sequence.blocks().split_last().expect("nonempty chain");
    let cache = |token_block| {
        let mut blocks = manager
            .allocate_blocks_from_reset(1)
            .expect("one reset slot without evicting the tail");
        let complete = blocks.pop().unwrap().complete(token_block).unwrap();
        drop(manager.register_block(complete));
    };
    cache(tail);

    let store = manager.store_for_test();
    let mut after_reset = Vec::with_capacity(ANCESTORS);
    for ancestor in ancestors.iter().rev() {
        cache(ancestor);
        let blocks = manager.match_blocks(&[ancestor.kvbm_sequence_hash()]);
        assert_eq!(blocks.len(), 1);
        blocks[0].set_evict_on_reset(true);
        drop(blocks);

        let inactive = manager.inactive_len();
        assert_eq!(inactive, 1, "the tail stays inactive");
        assert_eq!(store.active_by_hash_len(), 0);
        let (nodes, _, slab_len, ..) = store.lineage_storage().unwrap();
        after_reset.push((nodes - inactive, slab_len));
    }

    assert!(
        after_reset
            .iter()
            .all(|&(ghosts, slab_len)| ghosts == 0 && slab_len <= CAPACITY),
        "unneeded ghosts or excess slab slots after reset, (ghosts, slab len): {after_reset:?}"
    );
}

type Manager = BlockManager<TestBlockData>;
type Live = ImmutableBlock<TestBlockData>;

fn new_manager(capacity: usize) -> Manager {
    create_test_manager_with_backend(capacity, |builder| {
        builder
            .block_size(1)
            .maximum_block_count(2 * capacity)
            .with_lineage_backend()
    })
}

fn tokens(values: &[u32]) -> TokenBlockSequence {
    TokenBlockSequence::from_slice(values, 1, Some(TEST_SALT))
}

fn stage(manager: &Manager, block: &TokenBlock) -> CompleteBlock<TestBlockData> {
    manager
        .allocate_blocks_from_reset(1)
        .unwrap()
        .pop()
        .unwrap()
        .complete(block)
        .unwrap()
}

fn register(manager: &Manager, block: &TokenBlock) -> Live {
    manager.register_block(stage(manager, block))
}

fn cache_chain(manager: &Manager, values: &[u32]) -> Vec<SequenceHash> {
    tokens(values)
        .blocks()
        .iter()
        .map(|block| {
            drop(register(manager, block));
            audit(manager);
            block.kvbm_sequence_hash()
        })
        .collect()
}

fn audit(manager: &Manager) -> LineageAudit {
    manager.store_for_test().audit_lineage(false)
}

fn reserve(manager: &Manager) -> (usize, usize, usize) {
    let (_, index, _, slab, policy) = manager.store_for_test().lineage_storage().unwrap();
    (index, slab, policy)
}

fn check_reserve(manager: &Manager, reserved: (usize, usize, usize)) {
    let (index, slab, policy) = reserve(manager);
    assert!(index <= reserved.0, "the index allocation grew");
    assert_eq!((slab, policy), (reserved.1, reserved.2));
}

fn reset(block: Live) {
    block.set_evict_on_reset(true);
    drop(block);
}

fn acquire(manager: &Manager, hash: SequenceHash) -> Live {
    manager.match_blocks(&[hash]).pop().expect("cached block")
}

fn candidate(manager: &Manager, hash: SequenceHash) -> Option<InactiveCandidate> {
    manager
        .inactive_candidates(manager.inactive_len())
        .into_iter()
        .find(|candidate| candidate.seq_hash == hash)
}

fn lineage(manager: &Manager, hash: SequenceHash) -> Option<Vec<SequenceHash>> {
    let plan = manager.preflight_inactive_lineage(candidate(manager, hash)?)?;
    Some(plan.source_hashes())
}

#[test]
fn capacity4_trace_stays_within_capacity() {
    let manager = new_manager(4);
    let reserved = reserve(&manager);
    cache_chain(&manager, &[10]);
    cache_chain(&manager, &[11]);
    let chain = tokens(&[1, 2, 3]);
    let chain = chain.blocks();
    for &position in &[2, 1] {
        let block = register(&manager, &chain[position]);
        audit(&manager);
        check_reserve(&manager, reserved);
        drop(block);
        audit(&manager);
        check_reserve(&manager, reserved);
    }
    let (blocks, evicted) = manager.allocate_blocks_with_evictions(2).unwrap();
    assert_eq!(evicted.len(), 2);
    drop(blocks);
    audit(&manager);
    let parent = register(&manager, &chain[0]);
    audit(&manager);
    drop(parent);
    assert_eq!(audit(&manager).nodes, 3);
    check_reserve(&manager, reserved);
    assert_eq!(
        lineage(&manager, chain[2].kvbm_sequence_hash())
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn child_without_registered_parent_waits() {
    let manager = new_manager(2);
    let chain = tokens(&[1, 2]);
    drop(register(&manager, &chain.blocks()[1]));
    let state = audit(&manager);
    assert_eq!((state.nodes, state.ghosts, state.waiting_keys), (1, 0, 1));
    assert!(lineage(&manager, chain.blocks()[1].kvbm_sequence_hash()).is_none());
}

#[test]
fn reset_of_live_parent_makes_children_wait() {
    let manager = new_manager(2);
    let chain = cache_chain(&manager, &[1, 2]);
    let child = candidate(&manager, chain[1]).unwrap();
    let parent = acquire(&manager, chain[0]);
    assert_eq!(audit(&manager).ghosts, 1);
    reset(parent);
    let state = audit(&manager);
    assert_eq!((state.nodes, state.ghosts, state.waiting_keys), (1, 0, 1));
    assert_eq!(
        candidate(&manager, chain[1]).unwrap().block_id,
        child.block_id
    );
    assert_eq!(acquire(&manager, chain[1]).block_id(), child.block_id);
    audit(&manager);
}

#[test]
fn tail_release_leaves_no_ghost_behind_a_hole() {
    for middle_live in [false, true] {
        let manager = new_manager(4);
        let chain = tokens(&[1, 2, 3]);
        let [root, middle, tail] = chain.blocks() else {
            unreachable!()
        };
        drop(register(&manager, tail));
        let middle = middle_live.then(|| register(&manager, middle));
        drop(register(&manager, root));
        audit(&manager);
        assert_eq!(
            manager.release_inactive_tail(if middle_live { 2 } else { 1 }),
            if middle_live { 2 } else { 1 }
        );
        assert_eq!(audit(&manager).ghosts, usize::from(middle_live));
        assert_eq!(manager.inactive_len(), 1);
        drop(middle);
        assert_eq!(audit(&manager).ghosts, 0);
        assert_eq!(manager.inactive_len(), if middle_live { 2 } else { 1 });
    }
}

#[test]
fn tail_release_retires_a_child_cached_before_its_parent() {
    let manager = new_manager(3);
    let chain = tokens(&[1, 2]);
    for block in chain.blocks().iter().rev() {
        drop(register(&manager, block));
        audit(&manager);
    }
    assert_eq!(manager.release_inactive_tail(1), 1);
    assert_eq!(audit(&manager).nodes, 0);
}

#[test]
fn chain_is_complete_after_reset_and_recache_of_its_parent() {
    let manager = new_manager(3);
    let chain = cache_chain(&manager, &[1, 2]);
    reset(acquire(&manager, chain[0]));
    assert_eq!(audit(&manager).waiting_keys, 1);
    cache_chain(&manager, &[1]);
    assert_eq!(lineage(&manager, chain[1]), Some(chain.clone()));
    assert!(candidate(&manager, chain[0]).is_none());
}

#[test]
fn suffix_cached_before_prefix_links_when_the_prefix_arrives() {
    let manager = new_manager(8);
    let chain = tokens(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let blocks = chain.blocks();
    for block in blocks[4..].iter().chain(&blocks[..4]) {
        drop(register(&manager, block));
        audit(&manager);
    }
    let leaf = blocks[7].kvbm_sequence_hash();
    assert_eq!(manager.inactive_candidates(8).len(), 1);
    assert_eq!(lineage(&manager, leaf).unwrap().len(), 8);
    assert!(manager.name_complete_inactive_entry_by_hash(leaf).is_ok());
}

#[test]
fn tail_release_does_not_cross_a_reset_block() {
    let manager = new_manager(4);
    let chain = tokens(&[1, 2, 3]);
    let [root, middle, tail] = chain.blocks() else {
        unreachable!()
    };
    for block in [tail, middle, root] {
        drop(register(&manager, block));
        audit(&manager);
    }
    reset(acquire(&manager, middle.kvbm_sequence_hash()));
    assert_eq!(audit(&manager).waiting_keys, 1);
    assert!(candidate(&manager, root.kvbm_sequence_hash()).is_some());
    assert_eq!(manager.release_inactive_tail(2), 2);
    assert_eq!(audit(&manager).nodes, 1);
    assert!(candidate(&manager, tail.kvbm_sequence_hash()).is_some());
    drop(register(&manager, middle));
    assert_eq!(audit(&manager).waiting_keys, 1);
    assert!(candidate(&manager, middle.kvbm_sequence_hash()).is_none());
}

#[test]
fn registration_beside_a_reset_parent_does_not_grow_the_slab() {
    let manager = new_manager(4);
    manager.set_capacity(8).unwrap();
    let reserved = reserve(&manager);
    for root in 10..16 {
        cache_chain(&manager, &[root]);
    }
    let chain = cache_chain(&manager, &[1, 2]);
    reset(acquire(&manager, chain[0]));
    audit(&manager);
    let root = register(&manager, &tokens(&[20]).blocks()[0]);
    assert_eq!(audit(&manager).nodes, 8);
    check_reserve(&manager, reserved);
    drop(root);
    check_reserve(&manager, reserved);
}

fn hash(current: u64, parent: Option<u64>, position: u64) -> SequenceHash {
    SequenceHash::new(current, parent, position)
}

fn stage_hash(
    manager: &Manager,
    hash: SequenceHash,
    parent: Option<u64>,
) -> CompleteBlock<TestBlockData> {
    let mutable = manager
        .allocate_blocks_from_reset(1)
        .unwrap()
        .pop()
        .unwrap();
    match parent {
        Some(full) => mutable.stage_with_parent(hash, 1, Some(full)),
        None => mutable.stage(hash, 1),
    }
    .unwrap()
}

fn cache_hash(manager: &Manager, hash: SequenceHash, parent: Option<u64>) {
    drop(manager.register_block(stage_hash(manager, hash, parent)));
}

fn colliding_roots() -> (SequenceHash, SequenceHash, SequenceHash) {
    let parent = 9;
    let colliding = parent ^ (1 << 54);
    (
        hash(parent, None, 0),
        hash(colliding, None, 0),
        hash(10, Some(parent), 1),
    )
}

#[test]
fn reader_refuses_a_link_under_another_hash() {
    let (true_parent, alias, child) = colliding_roots();
    let manager = new_manager(4);
    cache_hash(&manager, child, Some(true_parent.current_sequence_hash()));
    assert_eq!(audit(&manager).waiting_keys, 1);
    cache_hash(&manager, alias, None);
    assert!(lineage(&manager, child).is_none());
    let leaf = candidate(&manager, child).unwrap();
    assert!(manager.try_hold_inactive_lineage(leaf).is_none());
    assert!(manager.name_complete_inactive_entry(leaf).is_err());
    assert!(manager.name_complete_inactive_entry_by_hash(child).is_err());
    manager.release_inactive_tail(0);
    let root = manager.register_block(stage_hash(&manager, true_parent, None));
    cache_hash(&manager, child, Some(true_parent.current_sequence_hash()));
    assert!(
        lineage(&manager, child).is_none(),
        "the backing parent is live"
    );
    drop(root);
    assert_eq!(lineage(&manager, child), Some(vec![true_parent, child]));
    audit(&manager);

    let stale = manager.release_inactive_tail(0);
    assert_eq!(stale, 0);
    cache_hash(&manager, alias, None);
    cache_hash(&manager, child, None);
    let name = manager.name_complete_inactive_entry_by_hash(child).unwrap();
    drop(manager.register_block(stage_hash(
        &manager,
        child,
        Some(true_parent.current_sequence_hash()),
    )));
    assert!(manager.refresh_and_combine_exact_reclaim(&[name]).is_err());
    audit(&manager);

    // A live copy can share the key of the wrong inactive parent. Exercise
    // both a parent that remains in the index and one that is evicted first.
    for evict_alias in [false, true] {
        let manager = new_manager(4);
        let x = hash(20, None, 0);
        let b = hash(21, Some(19), 1);
        let alias = hash(21 ^ (1 << 54), Some(20), 1);
        let c = hash(22, Some(21), 2);
        cache_hash(&manager, x, None);
        cache_hash(&manager, alias, Some(20));
        let b = manager.register_block(stage_hash(&manager, b, Some(19)));
        if evict_alias {
            let last_reset = manager.allocate_blocks_from_reset(1).unwrap();
            let (_, evicted) = manager.allocate_blocks_with_evictions(1).unwrap();
            assert_eq!(evicted, vec![alias]);
            drop(last_reset);
        }
        cache_hash(&manager, c, Some(21));
        if evict_alias {
            cache_hash(&manager, alias, Some(20));
        }
        assert!(candidate(&manager, c).is_some());
        assert!(lineage(&manager, c).is_none());
        manager.store_for_test().audit_lineage(true);
        reset(b);
        manager.store_for_test().audit_lineage(true);
    }
}

#[test]
fn staged_blocks_use_the_parent_fragment() {
    let (parent, alias, child) = colliding_roots();
    let manager = new_manager(2);
    for (root, record) in [
        (parent, None),
        (parent, Some(parent.current_sequence_hash())),
        (alias, None),
    ] {
        assert_eq!(manager.release_inactive_tail(0), 0);
        cache_hash(&manager, root, None);
        cache_hash(&manager, child, record);
        assert_eq!(lineage(&manager, child), Some(vec![root, child]));
        audit(&manager);
    }
}

#[test]
fn registration_of_a_known_hash_gives_the_parent_record() {
    let (parent, alias, child) = colliding_roots();
    for batch in [false, true] {
        for existing_inactive in [false, true] {
            for first_full in [false, true] {
                for duplication in [
                    BlockDuplicationPolicy::Reject,
                    BlockDuplicationPolicy::Allow,
                ] {
                    let manager = create_test_manager_with_backend::<TestBlockData>(4, |builder| {
                        builder
                            .block_size(1)
                            .with_lineage_backend()
                            .duplication_policy(duplication)
                    });
                    let record = Some(parent.current_sequence_hash());
                    cache_hash(&manager, alias, None);
                    let first = manager.register_block(stage_hash(
                        &manager,
                        child,
                        if first_full { record } else { None },
                    ));
                    let existing = if existing_inactive {
                        drop(first);
                        None
                    } else {
                        Some(first)
                    };
                    let incoming =
                        stage_hash(&manager, child, if first_full { None } else { record });
                    if batch {
                        drop(manager.register_blocks(vec![incoming]));
                    } else {
                        drop(manager.register_block(incoming));
                    }
                    drop(existing);
                    assert!(candidate(&manager, child).is_some());
                    assert!(
                        lineage(&manager, child).is_none(),
                        "batch {batch}, inactive {existing_inactive}, first full {first_full}, {duplication:?}"
                    );
                    audit(&manager);
                }
            }
        }
    }
}

#[test]
fn parent_record_moves_at_a_rebind() {
    let (parent, alias, child) = colliding_roots();
    for site in [
        "inactive commit",
        "inactive commit_live",
        "direct live commit",
    ] {
        for full in [true, false] {
            let manager = new_manager(4);
            cache_hash(&manager, alias, None);
            // Leave the opposite provenance in slot 1. Occupy it while the
            // source takes slot 2, then free it for the rebind.
            let previous = stage_hash(
                &manager,
                child,
                (!full).then(|| parent.current_sequence_hash()),
            );
            drop(previous);
            let spacer = manager.allocate_blocks_from_reset(1).unwrap();
            let record = full.then(|| parent.current_sequence_hash());
            let source = manager.register_block(stage_hash(&manager, child, record));
            let source_id = source.block_id();
            drop(spacer);
            let outcome = match site {
                "inactive commit" => {
                    drop(source);
                    manager.prepare_rebind(source_id).unwrap().commit()
                }
                "inactive commit_live" => {
                    drop(source);
                    manager.prepare_rebind(source_id).unwrap().commit_live()
                }
                "direct live commit" => {
                    let result = manager.prepare_live_rebind(&source).unwrap().commit_live();
                    drop(source);
                    result
                }
                _ => unreachable!(),
            };
            assert!(
                matches!(outcome, RebindOutcome::Moved { src: 2, dst: 1, .. }),
                "{site}"
            );
            assert_eq!(candidate(&manager, child).unwrap().block_id, 1);
            assert_eq!(
                lineage(&manager, child),
                (!full).then(|| vec![alias, child]),
                "{site}, full record {full}"
            );
            audit(&manager);
        }
    }

    let manager = new_manager(4);
    cache_hash(&manager, alias, None);
    let primary = manager.register_block(stage_hash(&manager, child, None));
    let spacer = manager.allocate_blocks_from_reset(1).unwrap();
    let duplicate = manager.register_block(stage_hash(
        &manager,
        child,
        Some(parent.current_sequence_hash()),
    ));
    drop(spacer);
    assert!(matches!(
        manager
            .prepare_live_rebind(&duplicate)
            .unwrap()
            .commit_live(),
        RebindOutcome::Moved { src: 3, dst: 2, .. }
    ));
    let store = manager.store_for_test();
    assert!(!store.test_lineage_link(alias, child, duplicate.block_id()));
    assert!(store.test_lineage_link(parent, child, duplicate.block_id()));
    drop(duplicate);
    drop(primary);
    audit(&manager);
}

#[test]
fn zero_parent_hash_uses_the_full_check() {
    for current in [0, 1, 1 << 54] {
        let manager = new_manager(4);
        let parent = hash(current, None, 0);
        let child = hash(2, Some(current & 1), 1);
        cache_hash(&manager, parent, None);
        let spare = manager.allocate_blocks_from_reset(1).unwrap();
        let live = manager.register_block(stage_hash(&manager, child, Some(current & 1)));
        let source = live.block_id();
        drop(live);
        drop(spare);
        assert!(matches!(
            manager.prepare_rebind(source).unwrap().commit(),
            RebindOutcome::Moved { .. }
        ));
        assert_eq!(
            lineage(&manager, child),
            (current <= 1).then(|| vec![parent, child])
        );
        audit(&manager);
    }
}

#[test]
fn held_block_registered_again_by_hash_uses_the_parent_fragment() {
    for release_first in [false, true] {
        let manager = new_manager(4);
        let chain = cache_chain(&manager, &[1, 2]);
        let hold = manager
            .try_hold_inactive_lineage(candidate(&manager, chain[1]).unwrap())
            .unwrap();
        assert_eq!(audit(&manager).ghosts, 2);
        let replacement = manager.register_block(stage_hash(&manager, chain[1], None));
        if release_first {
            drop(replacement);
            drop(hold);
        } else {
            drop(hold);
            drop(replacement);
        }
        assert_eq!(lineage(&manager, chain[1]), Some(chain));
        audit(&manager);
    }
}

#[test]
fn two_hashes_on_one_key_leave_no_ghost() {
    let (parent, alias, child) = colliding_roots();
    for order in [[parent, alias], [alias, parent]] {
        let manager = new_manager(4);
        let [first, second] =
            order.map(|root| manager.register_block(stage_hash(&manager, root, None)));
        cache_hash(&manager, child, Some(parent.current_sequence_hash()));
        assert_eq!(manager.store_for_test().audit_lineage(true).ghosts, 1);
        reset(first);
        let state = manager.store_for_test().audit_lineage(true);
        assert_eq!((state.ghosts, state.waiting_keys), (0, 1));
        reset(second);
        let state = manager.store_for_test().audit_lineage(true);
        assert_eq!((state.ghosts, state.waiting_keys), (0, 1));
    }
}

#[test]
fn full_hash_collision_after_a_reset_certifies_the_other_parent() {
    collision_exposed_after_reset(0, true);
}

#[test]
fn fragment_collision_after_a_reset_exposes_a_false_lineage() {
    collision_exposed_after_reset(1 << 54, false);
    collision_exposed_after_reset(1 << 54, true);
    for record in [None, Some(3)] {
        let manager = new_manager(6);
        let x = hash(2, None, 0);
        let b = hash(3, Some(1), 1);
        let alias = hash(3 ^ (1 << 54), Some(2), 1);
        let c = hash(4, Some(3), 2);
        cache_hash(&manager, b, Some(1));
        cache_hash(&manager, c, record);
        reset(acquire(&manager, b));
        assert!(lineage(&manager, c).is_none());
        cache_hash(&manager, x, None);
        cache_hash(&manager, alias, Some(2));
        assert_eq!(
            lineage(&manager, c),
            record.is_none().then(|| vec![x, alias, c])
        );
        manager.store_for_test().audit_lineage(true);
    }
}

fn collision_exposed_after_reset(difference: u64, full_record: bool) {
    let manager = new_manager(8);
    let unrelated = hash(2, None, 0);
    let true_parent = hash(3, Some(1), 1);
    let alias = hash(3 ^ difference, Some(2), 1);
    assert_ne!(alias, true_parent);
    let above = hash(4, Some(true_parent.current_sequence_hash()), 2);
    let middle = hash(5, Some(4), 3);
    let leaf = hash(6, Some(5), 4);
    let accepts = difference == 0 || !full_record;
    for (block, record) in [
        (unrelated, None),
        (alias, Some(2)),
        (above, full_record.then_some(3)),
        (middle, Some(4)),
        (leaf, Some(5)),
    ] {
        cache_hash(&manager, block, record);
        manager.store_for_test().audit_lineage(true);
    }
    assert_eq!(
        lineage(&manager, leaf),
        accepts.then(|| vec![unrelated, alias, above, middle, leaf]),
    );
    reset(acquire(&manager, middle));
    manager.store_for_test().audit_lineage(true);
    assert_eq!(
        lineage(&manager, above),
        accepts.then(|| vec![unrelated, alias, above]),
    );
    assert!(lineage(&manager, leaf).is_none());
}

#[test]
fn batch_reset_and_inactive_orders_agree() {
    for parent_first in [true, false] {
        let manager = new_manager(3);
        let pair = tokens(&[1, 2]);
        let parent = register(&manager, &pair.blocks()[0]);
        let child = register(&manager, &pair.blocks()[1]);
        parent.set_evict_on_reset(true);
        let batch = if parent_first {
            vec![parent, child]
        } else {
            vec![child, parent]
        };
        manager.release_blocks(batch, None);
        let state = audit(&manager);
        assert_eq!((state.nodes, state.ghosts, state.waiting_keys), (1, 0, 1));
    }
}

#[test]
fn batch_with_repeated_hash_adds_one_node() {
    for existing_inactive in [false, true] {
        let manager = new_manager(4);
        let sequence = tokens(&[1]);
        let root = &sequence.blocks()[0];
        if existing_inactive {
            drop(register(&manager, root));
        }
        let batch = vec![stage(&manager, root), stage(&manager, root)];
        let registered = manager.register_blocks(batch);
        let state = audit(&manager);
        assert_eq!((state.nodes, state.ghosts), (1, 1));
        drop(registered);
        assert_eq!(audit(&manager).nodes, 1);
    }
}

#[test]
fn same_hash_registered_after_reset_adopts_children() {
    let manager = new_manager(4);
    let chain = cache_chain(&manager, &[1, 2]);
    reset(acquire(&manager, chain[0]));
    assert_eq!(audit(&manager).waiting_keys, 1);
    let sequence = tokens(&[1]);
    let parent = register(&manager, &sequence.blocks()[0]);
    let state = audit(&manager);
    assert_eq!((state.ghosts, state.waiting_keys), (1, 0));
    drop(parent);
    assert_eq!(lineage(&manager, chain[1]), Some(chain));
    audit(&manager);
}

#[test]
fn waiting_child_is_adopted_at_registration() {
    let manager = new_manager(4);
    let sequence = tokens(&[1, 2]);
    let child = sequence.blocks()[1].kvbm_sequence_hash();
    drop(register(&manager, &sequence.blocks()[1]));
    assert_eq!(audit(&manager).waiting_keys, 1);
    let parent = manager.register_blocks(vec![stage(&manager, &sequence.blocks()[0])]);
    let state = audit(&manager);
    assert_eq!((state.ghosts, state.waiting_keys), (1, 0));
    assert!(lineage(&manager, child).is_none());
    drop(parent);
    assert_eq!(lineage(&manager, child).unwrap().len(), 2);
    audit(&manager);
}

#[test]
fn parent_and_waiting_child_in_one_batch() {
    for order in [[0, 1], [1, 0]] {
        let manager = new_manager(4);
        let sequence = tokens(&[1, 2]);
        let blocks = sequence.blocks();
        drop(register(&manager, &blocks[1]));
        assert_eq!(audit(&manager).waiting_keys, 1);
        let batch =
            manager.register_blocks(order.map(|index| stage(&manager, &blocks[index])).into());
        let state = audit(&manager);
        assert_eq!((state.nodes, state.ghosts, state.waiting_keys), (2, 2, 0));
        drop(batch);
        audit(&manager);
    }
}

#[test]
fn held_leaf_child_commit_and_abort() {
    for commit in [false, true] {
        let manager = new_manager(4);
        let prefix = cache_chain(&manager, &[1, 2]);
        let hold = manager
            .try_hold_inactive_lineage(candidate(&manager, prefix[1]).unwrap())
            .unwrap();
        let tail = tokens(&[1, 2, 3]);
        let child = tail.blocks().last().unwrap();
        drop(register(&manager, child));
        assert_eq!(audit(&manager).waiting_keys, 0);
        if commit {
            hold.commit_victim_release();
        } else {
            drop(hold);
        }
        let state = audit(&manager);
        assert_eq!(state.ghosts, 0);
        assert_eq!(state.waiting_keys, usize::from(commit));
        let chain = lineage(&manager, child.kvbm_sequence_hash());
        assert_eq!(chain.map(|blocks| blocks.len()), (!commit).then_some(3));
    }
}

#[test]
fn held_hash_with_newer_copy() {
    for reset_newer in [false, true] {
        let manager = new_manager(4);
        let hashes = cache_chain(&manager, &[1, 2]);
        let hold = manager
            .try_hold_inactive_lineage(candidate(&manager, hashes[1]).unwrap())
            .unwrap();
        let seq = tokens(&[1, 2]);
        let newer = register(&manager, &seq.blocks()[1]);
        assert_eq!(audit(&manager).ghosts, 2);
        if reset_newer {
            reset(newer);
            assert_eq!(audit(&manager).ghosts, 2);
            drop(hold);
        } else {
            hold.commit_victim_release();
            assert_eq!(audit(&manager).ghosts, 1);
            drop(newer);
        }
        assert_eq!(lineage(&manager, hashes[1]), Some(hashes));
        audit(&manager);
    }
}

#[test]
fn held_hash_keeps_its_children_when_a_newer_copy_is_evicted() {
    for commit in [false, true] {
        let manager = new_manager(4);
        let chain = tokens(&[1, 2, 3]);
        let [root, held, child] = chain.blocks() else {
            unreachable!()
        };
        drop(register(&manager, root));
        drop(register(&manager, held));
        let hold = manager
            .try_hold_inactive_lineage(candidate(&manager, held.kvbm_sequence_hash()).unwrap())
            .unwrap();
        drop(register(&manager, held));
        let child = register(&manager, child);
        let (_, evictions) = manager.allocate_blocks_with_evictions(1).unwrap();
        assert_eq!(evictions, vec![held.kvbm_sequence_hash()]);
        drop(child);
        assert_eq!(audit(&manager).waiting_keys, 1);
        if commit {
            hold.commit_victim_release();
        } else {
            drop(hold);
        }
        assert_eq!(audit(&manager).waiting_keys, usize::from(commit));
    }
}

#[test]
fn evict_of_a_node_with_a_child_frees_it() {
    use crate::pools::{InactiveIndex, backends::LineageBackend};
    let mut backend = LineageBackend::new();
    let parent = hash(1, None, 0);
    let child = hash(2, Some(1), 1);
    backend.insert(parent, 0);
    backend.insert(child, 1);
    assert!(backend.take(parent, 0));
    assert_eq!(backend.lineage_storage().unwrap().0, 1);
    assert!(backend.complete_lineage(child, 1).is_none());
    backend.insert(parent, 0);
    assert_eq!(
        backend.complete_lineage(child, 1).unwrap(),
        vec![(parent, 0), (child, 1)]
    );
}

#[test]
fn failed_allocation_puts_victims_back_in_place() {
    let manager = new_manager(4);
    let chain = cache_chain(&manager, &[1, 2, 3, 4]);
    manager.set_allocation_ceiling(Some(1)).unwrap();
    assert!(manager.allocate_blocks(2).is_none());
    assert_eq!(lineage(&manager, chain[3]), Some(chain));
    audit(&manager);
}

#[test]
fn failed_allocation_leaves_a_live_ghost_unlinked() {
    let manager = new_manager(3);
    let chain = cache_chain(&manager, &[1, 2, 3]);
    let parent = acquire(&manager, chain[1]);
    assert!(candidate(&manager, chain[0]).is_none());
    manager.set_allocation_ceiling(Some(1)).unwrap();
    assert!(manager.allocate_blocks(2).is_none());
    assert!(candidate(&manager, chain[0]).is_some());
    assert!(lineage(&manager, chain[2]).is_none());
    audit(&manager);
    drop(parent);
    assert_eq!(lineage(&manager, chain[2]), Some(chain));
    audit(&manager);
}

#[test]
fn prepared_hold_and_exact_reclaim_survive_a_prune_and_a_failed_allocation() {
    let manager = new_manager(4);
    let chain = cache_chain(&manager, &[1, 2]);
    let other = tokens(&[1, 9, 10]);
    for block in &other.blocks()[1..] {
        drop(register(&manager, block));
    }
    let hash = chain[1];
    let proof = manager
        .preflight_inactive_lineage(candidate(&manager, hash).unwrap())
        .unwrap();
    let name = manager.name_complete_inactive_entry_by_hash(hash).unwrap();
    reset(acquire(&manager, other.blocks()[1].kvbm_sequence_hash()));
    audit(&manager);
    manager.set_allocation_ceiling(Some(0)).unwrap();
    assert!(manager.allocate_blocks(1).is_none());
    audit(&manager);
    manager.set_allocation_ceiling(None).unwrap();
    assert!(
        manager
            .refresh_and_combine_exact_reclaim(std::slice::from_ref(&name))
            .is_ok()
    );
    let hold = manager.try_hold_prepared_inactive_lineage(proof).unwrap();
    drop(hold);
    assert_eq!(lineage(&manager, hash), Some(chain));
    audit(&manager);
}

#[test]
fn reader_checks_mixed_lineage() {
    let mixed = new_manager(4);
    let a = hash(1, None, 0);
    let b = hash(2, Some(1), 1);
    let c = hash(3, Some(2), 2);
    cache_hash(&mixed, a, None);
    cache_hash(&mixed, b, None);
    cache_hash(&mixed, c, Some(2));
    assert_eq!(lineage(&mixed, c), Some(vec![a, b, c]));
    audit(&mixed);

    for (position, width) in [(256, 46), (65_536, 38)] {
        let manager = new_manager(3);
        let child = hash(5, Some(9), position);
        let parent = hash(9, Some(8), position - 1);
        let alias = hash(9 ^ (1 << width), Some(8), position - 1);
        let wrong_fragment = hash(10, Some(8), position - 1);
        let wrong_position = hash(9, Some(8), position - 2);
        assert_eq!(
            parent.parent_fragment_for_child_position(position),
            alias.parent_fragment_for_child_position(position)
        );
        assert_ne!(
            parent.current_sequence_hash(),
            alias.current_sequence_hash()
        );
        for record in [None, Some(9)] {
            let block = stage_hash(&manager, child, record);
            let check = |candidate| {
                manager
                    .store_for_test()
                    .test_lineage_link(candidate, child, block.block_id())
            };
            assert!(check(parent));
            assert_eq!(check(alias), record.is_none());
            assert!(!check(wrong_fragment));
            assert!(!check(wrong_position));
        }
        cache_hash(&manager, child, None);
        cache_hash(&manager, alias, None);
        assert_eq!(audit(&manager).waiting_keys, 1);
        assert!(candidate(&manager, alias).is_none());
    }
}

#[test]
fn promotion_by_another_hash_keeps_the_place() {
    for ancestor_present in [false, true] {
        let manager = new_manager(6);
        let a = hash(1, None, 0);
        let x = hash(2, None, 0);
        let b = hash(3, Some(1), 1);
        let alias = hash(3 ^ (1 << 54), Some(2), 1);
        let c = hash(4, Some(3), 2);
        // A is last and occupies the only tail slot.
        cache_hash(&manager, b, Some(1));
        cache_hash(&manager, c, Some(3));
        let live = acquire(&manager, b);
        cache_hash(&manager, x, None);
        cache_hash(&manager, alias, Some(2));
        if ancestor_present {
            cache_hash(&manager, a, None);
        }
        manager.store_for_test().audit_lineage(true);
        assert_eq!(manager.inactive_candidates(1)[0].seq_hash, c);
        assert!(lineage(&manager, c).is_none());
        if ancestor_present {
            let a_id = 3;
            let evicted = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let seen = evicted.clone();
            let observer: std::sync::Arc<dyn crate::manager::BlockEvictionObserver> =
                std::sync::Arc::new(move |hashes: &[SequenceHash]| {
                    seen.lock().unwrap().extend_from_slice(hashes)
                });
            manager.observe_evictions(&observer);
            manager.release_inactive_tail(a_id);
            assert_eq!(*evicted.lock().unwrap(), vec![c, alias, a]);
        } else {
            manager.release_inactive_tail(0);
        }
        reset(live);
        manager.store_for_test().audit_lineage(true);
    }
}

#[test]
fn other_hash_after_a_reset_hangs_under_its_own_parent() {
    for a_present in [false, true] {
        let manager = new_manager(8);
        let a = hash(1, None, 0);
        let x = hash(2, None, 0);
        let b = hash(3, Some(1), 1);
        let alias = hash(3 ^ (1 << 54), Some(2), 1);
        let c = hash(4, Some(3), 2);
        cache_hash(&manager, b, Some(1));
        cache_hash(&manager, c, Some(3));
        reset(acquire(&manager, b));
        cache_hash(&manager, alias, Some(2));
        if a_present {
            cache_hash(&manager, a, None);
        }
        cache_hash(&manager, x, None);
        manager.store_for_test().audit_lineage(true);
        assert!(candidate(&manager, x).is_none());
        assert!(lineage(&manager, c).is_none());
        let evicted = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = evicted.clone();
        let observer: std::sync::Arc<dyn crate::manager::BlockEvictionObserver> =
            std::sync::Arc::new(move |hashes: &[SequenceHash]| {
                seen.lock().unwrap().extend_from_slice(hashes)
            });
        manager.observe_evictions(&observer);
        manager.release_inactive_tail(if a_present { 3 } else { 2 });
        assert_eq!(*evicted.lock().unwrap(), vec![c, alias, x]);
        manager.store_for_test().audit_lineage(true);
    }
}

#[test]
fn prune_work_is_one_pass_per_list() {
    let pool = new_manager(67);
    let base = cache_chain(&pool, &[1, 2, 3]);
    for child in 10..74 {
        drop(register(&pool, &tokens(&[1, child]).blocks()[1]));
    }
    let a = acquire(&pool, base[0]);
    let b = acquire(&pool, base[1]);
    let before = audit(&pool);
    reset(b);
    let after = audit(&pool);
    assert_eq!(after.list_steps - before.list_steps, 65);
    assert_eq!(after.child_writes - before.child_writes, 1);
    reset(a);
    assert_eq!(audit(&pool).child_writes - after.child_writes, 64);

    let pool = new_manager(10_001);
    let parent = register(&pool, &tokens(&[1]).blocks()[0]);
    for child in 2..10_002 {
        drop(register(&pool, &tokens(&[1, child]).blocks()[1]));
    }
    let before = audit(&pool);
    reset(parent);
    let after = audit(&pool);
    assert_eq!(after.child_writes - before.child_writes, 10_000);
    assert!(after.list_steps - before.list_steps <= 1);
}

#[test]
fn tail_release_of_one_sibling_scans_its_list() {
    let manager = new_manager(128);
    let mut lower_slots = manager.allocate_blocks_from_reset(127).unwrap();
    drop(register(&manager, &tokens(&[1, 2]).blocks()[1]));
    lower_slots.reverse();
    for (index, mutable) in lower_slots.into_iter().enumerate() {
        let token = tokens(&[1, 3 + index as u32]);
        drop(manager.register_block(mutable.complete(&token.blocks()[1]).unwrap()));
    }
    let before = audit(&manager);
    manager.release_inactive_tail(127);
    let after = audit(&manager);
    assert_eq!(after.nodes, 127);
    assert_eq!(after.list_steps - before.list_steps, 128);
}

#[test]
fn tail_release_cascade_scans_the_parents_list() {
    let manager = new_manager(130);
    let pair = cache_chain(&manager, &[1, 2]);
    for child in 4..131 {
        drop(register(&manager, &tokens(&[1, child]).blocks()[1]));
    }
    drop(register(&manager, &tokens(&[1, 2, 3]).blocks()[2]));
    let parent = acquire(&manager, pair[1]);
    let before = audit(&manager);
    manager.release_inactive_tail(129);
    let after = audit(&manager);
    assert_eq!(after.nodes, before.nodes - 1);
    assert_eq!(after.list_steps - before.list_steps, 129);
    drop(parent);
    audit(&manager);
}

#[test]
fn valued_peek_orders_ties_by_arena_index() {
    use crate::pools::backends::ScorerParams;
    use crate::registry::BlockRegistry;
    use crate::tinylfu::FrequencyTracker;

    struct Frequencies(u128);

    impl FrequencyTracker<u128> for Frequencies {
        fn touch(&self, _key: u128) {}

        fn count(&self, key: u128) -> u32 {
            u32::from(self.0 == key)
        }
    }

    let a = hash(1, None, 0);
    let b = hash(2, None, 0);
    let manager = Manager::builder()
        .block_count(4)
        .block_size(1)
        .registry(
            BlockRegistry::builder()
                .frequency_tracker(std::sync::Arc::new(Frequencies(b.as_u128())))
                .build(),
        )
        .with_valued_lineage_backend(ScorerParams::default())
        .build()
        .unwrap();
    let a = manager.register_block(stage_hash(&manager, a, None));
    let b = manager.register_block(stage_hash(&manager, b, None));
    let (a_hash, b_hash) = (a.sequence_hash(), b.sequence_hash());
    drop(b);
    drop(a);
    assert_eq!(manager.inactive_candidates(1)[0].seq_hash, a_hash);
    let (_mutable, evicted) = manager.allocate_blocks_with_evictions(3).unwrap();
    assert_eq!(evicted, vec![b_hash]);
    audit(&manager);
}

fn writeback_fixture() -> (
    Manager,
    [SequenceHash; 3],
    Vec<SequenceHash>,
    SequenceHash,
    Vec<Live>,
) {
    let manager = new_manager(256);
    let sequence = tokens(&[1, 2, 3]);
    let chain = [0, 1, 2].map(|position| sequence.blocks()[position].kvbm_sequence_hash());
    for block in sequence.blocks() {
        drop(register(&manager, block));
    }
    let prefix = manager.match_blocks(&chain[..2]);
    assert_eq!(prefix.len(), 2);
    manager.release_blocks(prefix, Some(true));
    let copied_roots: Vec<_> = (10..40)
        .map(|token| {
            let root = register(&manager, &tokens(&[token]).blocks()[0]);
            root.sequence_hash()
        })
        .collect();
    drop(register(&manager, &sequence.blocks()[0]));
    let z = register(&manager, &tokens(&[40]).blocks()[0]).sequence_hash();
    let live = (100..323)
        .map(|token| register(&manager, &tokens(&[token]).blocks()[0]))
        .collect();
    assert_eq!(manager.available_blocks(), 33);
    (manager, chain, copied_roots, z, live)
}

#[test]
fn candidate_snapshot_holds_an_ancestor_above_a_reset_block() {
    let (manager, chain, _, z, mut live) = writeback_fixture();
    let snapshot = manager.inactive_candidates(32);
    assert!(
        snapshot
            .iter()
            .any(|candidate| candidate.seq_hash == chain[0])
    );
    assert!(!snapshot.iter().any(|candidate| candidate.seq_hash == z));
    audit(&manager);

    reset(live.pop().unwrap());
    drop(register(&manager, &tokens(&[1, 2]).blocks()[1]));
    let snapshot = manager.inactive_candidates(32);
    assert!(
        !snapshot
            .iter()
            .any(|candidate| candidate.seq_hash == chain[0])
    );
    assert!(snapshot.iter().any(|candidate| candidate.seq_hash == z));
    audit(&manager);
}

fn replay_writeback(
    manager: &Manager,
    mut g2_copies: std::collections::HashSet<SequenceHash>,
) -> [Vec<SequenceHash>; 5] {
    // The pinned selection fixes the window before validating each lineage
    // or checking G2. A submitted root gets a G2 copy for subsequent ticks.
    std::array::from_fn(|_| {
        assert!(manager.available_blocks() * 5 < 256);
        manager
            .inactive_candidates(32)
            .into_iter()
            .filter_map(|candidate| {
                let chain = manager.preflight_inactive_lineage(candidate)?;
                let hashes = chain.source_hashes();
                let leaf = *hashes.last()?;
                if !g2_copies.insert(leaf) {
                    return None;
                }
                Some(leaf)
            })
            .collect()
    })
}

#[test]
fn writeback_replay_skips_a_newer_uncopied_leaf() {
    let (manager, chain, older, z, _live) = writeback_fixture();
    let copies: std::collections::HashSet<_> = older.iter().copied().chain([chain[0]]).collect();
    let submissions = replay_writeback(&manager, copies.clone());
    println!(
        "Z submitted on each tick: {:?}",
        submissions.each_ref().map(|tick| tick.contains(&z))
    );
    assert!(submissions.iter().all(Vec::is_empty));
    reset(acquire(&manager, older[0]));
    let control = replay_writeback(&manager, copies);
    assert_eq!(control[0], vec![z]);
    assert!(control[1..].iter().all(Vec::is_empty));
    audit(&manager);
}

#[test]
fn eager_transition_keeps_the_ghost() {
    use std::time::{Duration, Instant};

    for via_registration in [false, true] {
        let manager = new_manager(4);
        let chain = cache_chain(&manager, &[1, 2]);
        let parent = acquire(&manager, chain[0]);
        parent.set_evict_on_reset(true);
        let store = manager.store_for_test();
        std::thread::scope(|scope| {
            let gate = store.pause_release_primary();
            let before = store.release_primary_arrivals();
            let release = scope.spawn(move || drop(parent));
            let deadline = Instant::now() + Duration::from_secs(5);
            while store.release_primary_arrivals() == before {
                assert!(Instant::now() < deadline, "release did not reach its gate");
                std::thread::yield_now();
            }
            let new_owner = if via_registration {
                manager.register_blocks(vec![stage(&manager, &tokens(&[1]).blocks()[0])])
            } else {
                manager.match_blocks(&[chain[0]])
            };
            let state = audit(&manager);
            assert_eq!((state.ghosts, state.waiting_keys), (1, 0));
            drop(gate);
            release.join().unwrap();
            drop(new_owner);
        });
        let state = audit(&manager);
        assert_eq!((state.ghosts, state.waiting_keys), (0, 1));
    }
}

#[test]
fn reset_then_registration_before_mark_absent() {
    use std::pin::pin;
    use std::sync::Arc;
    use std::time::Duration;

    use crate::events::{EventsManager, KvCacheEvent};
    use crate::registry::BlockRegistry;
    use futures::{FutureExt, StreamExt};

    for present_first in [false, true] {
        let events = Arc::new(EventsManager::builder().build());
        let mut stream = pin!(events.subscribe());
        let manager = Manager::builder()
            .block_count(3)
            .block_size(1)
            .registry(
                BlockRegistry::builder()
                    .event_manager(events.clone())
                    .build(),
            )
            .with_lineage_backend()
            .build()
            .unwrap();
        let seq = tokens(&[1, 2]);
        let parent = register(&manager, &seq.blocks()[0]);
        let child = register(&manager, &seq.blocks()[1]);
        let parent_hash = parent.sequence_hash();
        let child_hash = child.sequence_hash();
        let marker = parent.registration_handle();
        let child_marker = child.registration_handle();
        drop(child);
        while let Some(Some(_)) = stream.next().now_or_never() {}
        let old = std::thread::scope(|scope| {
            let (absent, allow_absent) = manager.store_for_test().pause_presence_marker(false);
            let release = scope.spawn(move || reset(parent));
            absent.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(audit(&manager).waiting_keys, 1);
            assert!(child_marker.has_block::<TestBlockData>());
            assert_eq!(lineage(&manager, child_hash), None);

            let (present, allow_present) = manager.store_for_test().pause_presence_marker(true);
            let register_again = scope.spawn(|| register(&manager, &seq.blocks()[0]));
            present.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(audit(&manager).waiting_keys, 0);
            assert!(marker.has_block::<TestBlockData>());
            let new = if present_first {
                allow_present.send(()).unwrap();
                let new = register_again.join().unwrap();
                allow_absent.send(()).unwrap();
                release.join().unwrap();
                new
            } else {
                allow_absent.send(()).unwrap();
                release.join().unwrap();
                allow_present.send(()).unwrap();
                register_again.join().unwrap()
            };
            assert!(marker.has_block::<TestBlockData>());
            assert!(
                stream.next().now_or_never().is_none(),
                "no event before the last owner drops"
            );
            new
        });
        reset(old);
        assert!(!marker.has_block::<TestBlockData>());
        drop(marker);
        assert_eq!(
            stream.next().now_or_never(),
            Some(Some(KvCacheEvent::Remove(parent_hash)))
        );
        assert!(stream.next().now_or_never().is_none());
        assert!(child_marker.has_block::<TestBlockData>());
        audit(&manager);
    }
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn churn_universe(collisions: bool) -> Vec<[SequenceHash; 8]> {
    (0..12)
        .map(|family| {
            let current = |position: u64| {
                if collisions {
                    (family << 54) | (position << 3) | (family % 3 + 1)
                } else {
                    1 + family * 8 + position
                }
            };
            std::array::from_fn(|position| {
                let position = position as u64;
                hash(
                    current(position),
                    (position > 0).then(|| current(position - 1)),
                    position,
                )
            })
        })
        .collect()
}

fn conflicts(manager: &Manager, universe: &[[SequenceHash; 8]], target: SequenceHash) -> bool {
    let position = target.position();
    let fragment = target.parent_fragment_for_child_position(position + 1);
    universe.iter().any(|chain| {
        let other = chain[position as usize];
        other != target
            && other.parent_fragment_for_child_position(position + 1) == fragment
            && manager.has_any_registered_hashes(&[other])
    })
}

fn parity_digest(collisions: bool, inspect: impl Fn(&Manager)) -> u128 {
    use crate::manager::BlockEvictionObserver;
    use std::sync::{Arc, Mutex};

    let manager = new_manager(24);
    let universe = churn_universe(collisions);
    let notifications = Arc::new(Mutex::new(Vec::new()));
    let write = notifications.clone();
    let observer: Arc<dyn BlockEvictionObserver> =
        Arc::new(move |hashes: &[SequenceHash]| write.lock().unwrap().extend_from_slice(hashes));
    manager.observe_evictions(&observer);
    let mut rng = 0x3403_cee7_7e98_da11;
    let mut digest = 0u128;
    let mut fold = |value| {
        digest = digest.rotate_left(7) ^ value;
        digest = digest.wrapping_mul(0x517c_c1b7_2722_0a95);
    };
    for _ in 0..2_000 {
        let random = next_random(&mut rng);
        let chain = &universe[((random >> 8) as usize) % universe.len()];
        let position = ((random >> 16) as usize) % chain.len();
        match random % 32 {
            0 => {
                manager.release_inactive_tail(((random >> 24) as usize) % 24);
            }
            1..=4 => {
                drop(manager.match_blocks(&chain[position..]));
            }
            _ => {
                let target = chain[position];
                if !manager.has_any_registered_hashes(&[target])
                    && !conflicts(&manager, &universe, target)
                    && let Some(mut slots) = manager.allocate_blocks(1)
                {
                    let block = slots.pop().unwrap().stage(target, 1).unwrap();
                    drop(manager.register_block(block));
                }
            }
        }
        for evicted in notifications.lock().unwrap().drain(..) {
            fold(evicted.as_u128());
        }
        fold(u128::MAX);
        for leaf in manager.inactive_candidates(24) {
            fold(leaf.seq_hash.as_u128());
            fold(leaf.block_id as u128);
        }
        fold(0);
        inspect(&manager);
    }
    digest
}

#[test]
fn reset_free_churn_matches_the_parent_commit() {
    for (collisions, expected) in [
        (false, 0x20cb_062b_d9b9_5f72_8b49_76fc_43a1_012a),
        (true, 0x9c64_2ba9_46bf_302c_d477_b864_fb5d_a14b),
    ] {
        let digest = parity_digest(collisions, |manager| {
            manager.store_for_test().audit_lineage(collisions);
        });
        println!("reset-free digest with collisions={collisions}: {digest:032x}");
        assert_eq!(digest, expected);
    }
}

#[test]
fn churn_at_full_occupancy() {
    churn_with_resets(false, false);
}

#[test]
fn collision_churn_keeps_the_reader_honest() {
    churn_with_resets(true, false);
    churn_with_resets(true, true);
}

fn churn_with_resets(collisions: bool, mixed: bool) {
    use std::collections::HashMap;
    let manager = new_manager(24);
    manager.set_capacity(48).unwrap();
    let reserved = reserve(&manager);
    let universe = churn_universe(collisions);
    let expected_parent: HashMap<_, _> = universe
        .iter()
        .flat_map(|chain| chain.windows(2).map(|pair| (pair[1], pair[0])))
        .collect();
    let uses_full = |hash: SequenceHash| !mixed || hash.position().is_multiple_of(2);
    let mut rng = 0xc30c_b9d3_4d1e_1325;
    let mut full = 0;
    let mut certificates = 0;
    let mut false_links = 0;
    for step in 0..4_000 {
        let random = next_random(&mut rng);
        let chain = &universe[((random >> 8) as usize) % universe.len()];
        let position = ((random >> 16) as usize) % chain.len();
        let target = chain[position];
        match random % 64 {
            0 if step >= 48 => {
                manager.release_inactive_tail(((random >> 24) as usize) % 48);
            }
            1..=4 if step >= 48 => {
                manager.release_blocks(manager.match_blocks(&[target]), Some(true));
            }
            5..=8 if step >= 48 => {
                drop(manager.match_blocks(&chain[position..]));
            }
            9..=12 if step >= 48 => {
                if let Some(hold) = candidate(&manager, target)
                    .and_then(|leaf| manager.try_hold_inactive_lineage(leaf))
                {
                    manager.store_for_test().audit_lineage(collisions);
                    if random & 0x100 != 0 {
                        hold.commit_victim_release();
                    } else {
                        drop(hold);
                    }
                }
            }
            13 if step >= 48 => {
                manager.set_allocation_ceiling(Some(0)).unwrap();
                assert!(manager.allocate_blocks(1).is_none());
                manager.set_allocation_ceiling(None).unwrap();
            }
            _ => {
                if !manager.has_any_registered_hashes(&[target])
                    && let Some(mut slots) = manager.allocate_blocks(1)
                {
                    let parent = expected_parent
                        .get(&target)
                        .filter(|_| uses_full(target))
                        .map(|parent| parent.current_sequence_hash());
                    let mutable = slots.pop().unwrap();
                    let staged = mutable.stage_with_parent(target, 1, parent).unwrap();
                    let guard = manager.register_block(staged);
                    if collisions && conflicts(&manager, &universe, target) {
                        reset(guard);
                    } else {
                        drop(guard);
                    }
                }
            }
        }
        manager.store_for_test().audit_lineage(collisions);
        check_reserve(&manager, reserved);
        full += usize::from(manager.inactive_len() == 48);
        for candidate in manager.inactive_candidates(48) {
            let Some(proof) = manager.preflight_inactive_lineage(candidate) else {
                continue;
            };
            certificates += 1;
            for pair in proof.source_hashes().windows(2) {
                if expected_parent.get(&pair[1]) != Some(&pair[0]) {
                    false_links += 1;
                    assert!(collisions && !uses_full(pair[1]));
                    assert_eq!(pair[0].position() + 1, pair[1].position());
                    assert_eq!(
                        pair[0].parent_fragment_for_child_position(pair[1].position()),
                        pair[1].parent_hash_fragment()
                    );
                }
            }
        }
    }
    println!(
        "churn collisions={collisions} mixed={mixed}, full steps {full}, certificates {certificates}, false links {false_links}"
    );
    assert!(certificates > 0);
    if !collisions {
        assert!(full > 0);
    }
}
