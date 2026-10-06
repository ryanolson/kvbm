use super::*;
use dynamo_tokens::TokenBlockSequence;
use crate::manager::InactiveBackendConfig;
use crate::pools::store::SlotKind;

type BackendBuilder =
    fn(BlockManagerConfigBuilder<TestBlockData>) -> BlockManagerConfigBuilder<TestBlockData>;

fn hashmap_backend(builder: BlockManagerConfigBuilder<TestBlockData>) -> BlockManagerConfigBuilder<TestBlockData> {
    builder.inactive_backend(InactiveBackendConfig::HashMap)
}

fn lru_backend(builder: BlockManagerConfigBuilder<TestBlockData>) -> BlockManagerConfigBuilder<TestBlockData> {
    builder.with_lru_backend()
}

fn multi_lru_backend(
    builder: BlockManagerConfigBuilder<TestBlockData>,
) -> BlockManagerConfigBuilder<TestBlockData> {
    builder.with_multi_lru_backend()
}

fn lineage_backend(
    builder: BlockManagerConfigBuilder<TestBlockData>,
) -> BlockManagerConfigBuilder<TestBlockData> {
    builder.with_lineage_backend()
}

fn manager_with_backend(
    block_count: usize,
    backend: BackendBuilder,
) -> BlockManager<TestBlockData> {
    testing::create_test_manager_with_backend(block_count, backend)
}

fn register_and_release(
    manager: &BlockManager<TestBlockData>,
    start: u32,
) -> (SequenceHash, BlockId) {
    let token_block = create_test_token_block_from_iota(start);
    let seq_hash = token_block.kvbm_sequence_hash();
    let mutable = manager
        .allocate_blocks(1)
        .expect("one block available")
        .into_iter()
        .next()
        .unwrap();
    let block_id = mutable.block_id();
    let immutable = manager.register_block(
        mutable
            .complete(&token_block)
            .expect("matching block size"),
    );
    drop(immutable);
    (seq_hash, block_id)
}

#[rstest]
#[case::hashmap(hashmap_backend)]
#[case::lru(lru_backend)]
#[case::multi_lru(multi_lru_backend)]
#[case::lineage(lineage_backend)]
fn prepare_commit_moves_the_inactive_registration_and_handle(
    #[case] backend: BackendBuilder,
) {
    let manager = manager_with_backend(4, backend);
    let token_block = create_test_token_block_from_iota(101);
    let seq_hash = token_block.kvbm_sequence_hash();
    let mutable = manager
        .allocate_blocks(1)
        .expect("one reset block")
        .into_iter()
        .next()
        .unwrap();
    let src = mutable.block_id();
    let immutable = manager.register_block(
        mutable
            .complete(&token_block)
            .expect("matching block size"),
    );
    let handle = immutable.registration_handle();
    drop(immutable);

    let before_prepare = manager.metrics().snapshot();
    let destination_generation = manager.store_for_test().slot_generation_for_test(1);
    let plan = manager.prepare_rebind(src).expect("inactive source and reset destination");
    let dst = plan.dst();
    let during_prepare = manager.metrics().snapshot();
    assert_eq!(during_prepare.allocations, before_prepare.allocations);
    assert_eq!(during_prepare.inflight_mutable, before_prepare.inflight_mutable);
    assert_eq!(during_prepare.evictions, before_prepare.evictions);
    assert_eq!(
        during_prepare.reset_pool_size,
        before_prepare.reset_pool_size - 1
    );
    let source_match = manager.match_blocks(&[seq_hash]);
    assert_eq!(source_match[0].block_id(), src);
    drop(source_match);
    assert_eq!(plan.src(), src);
    assert_eq!(plan.sequence_hash(), seq_hash);
    assert!(matches!(
        plan.commit(),
        RebindOutcome::Moved {
            seq_hash: moved_hash,
            src: moved_src,
            dst: moved_dst,
        } if moved_hash == seq_hash && moved_src == src && moved_dst == dst
    ));
    assert_eq!(
        manager.store_for_test().slot_generation_for_test(dst),
        destination_generation + 1
    );
    let after_commit = manager.metrics().snapshot();
    assert_eq!(after_commit.allocations, before_prepare.allocations);
    assert_eq!(after_commit.inflight_mutable, before_prepare.inflight_mutable);
    assert_eq!(after_commit.evictions, before_prepare.evictions);

    assert_eq!(manager.inactive_len(), 1);
    let snapshot = manager.store.debug_snapshot();
    assert_eq!(snapshot.slots[src], SlotKind::Reset);
    assert!(snapshot.free.contains(&src) || snapshot.fenced.contains(&src));
    let matched = manager.match_blocks(&[seq_hash]);
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].block_id(), dst);
    assert!(std::sync::Arc::ptr_eq(
        &handle.inner,
        &matched[0].registration_handle().inner
    ));
}

#[test]
fn direct_rebind_moves_a_copied_block_to_the_requested_reset_slot() {
    let mut manager = manager_with_backend(2, hashmap_backend);
    let (seq_hash, src) = register_and_release(&manager, 1_500);
    let generation = manager.store_for_test().slot_generation_for_test(1);
    assert!(manager.rebind(seq_hash, src, 1));
    assert_eq!(
        manager.store_for_test().slot_generation_for_test(1),
        generation + 1
    );

    let matched = manager.match_blocks(&[seq_hash]);
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].block_id(), 1);
    let snapshot = manager.store.debug_snapshot();
    assert_eq!(snapshot.slots[src], SlotKind::Reset);
}

#[test]
fn lineage_parent_rebind_preserves_chain_lookup_and_tail_release() {
    let manager = manager_with_backend(4, lineage_backend);
    let token_values: Vec<_> = (100..112).collect();
    let sequence =
        TokenBlockSequence::from_slice(&token_values, 4, Some(crate::testing::TEST_SALT));
    let mut registered = Vec::new();
    for (mutable, token_block) in manager
        .allocate_blocks(3)
        .unwrap()
        .into_iter()
        .zip(sequence.blocks())
    {
        registered.push(manager.register_block(mutable.complete(token_block).unwrap()));
    }
    let original: Vec<_> = registered
        .iter()
        .map(|block| (block.sequence_hash(), block.block_id()))
        .collect();
    drop(registered);

    let plan = manager.prepare_rebind(original[0].1).unwrap();
    let dst = plan.dst();
    assert!(matches!(plan.commit(), RebindOutcome::Moved { .. }));
    let hashes: Vec<_> = original.iter().map(|(hash, _)| *hash).collect();
    let matched = manager.match_blocks(&hashes);
    assert_eq!(
        matched.iter().map(|block| block.block_id()).collect::<Vec<_>>(),
        vec![dst, original[1].1, original[2].1]
    );
    drop(matched);

    let leaf = manager
        .inactive_candidates(manager.total_blocks())
        .into_iter()
        .find(|candidate| candidate.seq_hash == original[2].0)
        .unwrap();
    let hold = manager.try_hold_inactive_lineage(leaf).unwrap();
    assert_eq!(
        hold.source_blocks(),
        &[(original[0].0, dst), original[1], original[2]]
    );
    drop(hold);

    assert_eq!(manager.release_inactive_tail(3), 3);
    assert_eq!(manager.inactive_len(), 0);
    assert_eq!(manager.reset_len(), 4);
}

#[rstest]
#[case::hashmap(hashmap_backend)]
#[case::lru(lru_backend)]
#[case::multi_lru(multi_lru_backend)]
#[case::lineage(lineage_backend)]
fn rebind_preserves_backend_eviction_order(#[case] backend: BackendBuilder) {
    let control = manager_with_backend(4, backend);
    let moved = manager_with_backend(4, backend);
    let mut hashes = Vec::new();
    let mut ids = Vec::new();
    for start in [1_000, 1_004, 1_008] {
        let (hash, id) = register_and_release(&control, start);
        hashes.push(hash);
        ids.push(id);
        let (moved_hash, moved_id) = register_and_release(&moved, start);
        assert_eq!(moved_hash, hash);
        assert_eq!(moved_id, id);
    }

    let plan = moved.prepare_rebind(ids[1]).unwrap();
    assert!(matches!(plan.commit(), RebindOutcome::Moved { .. }));
    let expected = control
        .allocate_blocks_with_evictions(3)
        .expect("three blocks available")
        .1;
    let actual = moved
        .allocate_blocks_with_evictions(3)
        .expect("three blocks available")
        .1;
    assert_eq!(actual, expected);
    assert_eq!(expected.len(), 2);
    assert!(hashes.contains(&expected[0]));
}

#[test]
fn busy_rebind_can_retry_after_the_primary_releases() {
    let manager = manager_with_backend(2, lru_backend);
    let token_block = create_test_token_block_from_iota(2_000);
    let seq_hash = token_block.kvbm_sequence_hash();
    let mutable = manager
        .allocate_blocks(1)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let src = mutable.block_id();
    drop(manager.register_block(mutable.complete(&token_block).unwrap()));

    let plan = manager.prepare_rebind(src).unwrap();
    let matched = manager.match_blocks(&[seq_hash]);
    let plan = match plan.commit() {
        RebindOutcome::Busy(plan) => plan,
        outcome => panic!("expected Busy, got {outcome:?}"),
    };
    drop(matched);
    assert!(matches!(plan.commit(), RebindOutcome::Moved { .. }));
    assert_eq!(manager.inactive_len(), 1);
}

#[test]
fn stale_rebind_releases_its_destination_after_eviction_and_reregistration() {
    let manager = manager_with_backend(2, hashmap_backend);
    let token_block = create_test_token_block_from_iota(3_000);
    let seq_hash = token_block.kvbm_sequence_hash();
    let mutable = manager
        .allocate_blocks(1)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let src = mutable.block_id();
    drop(manager.register_block(mutable.complete(&token_block).unwrap()));

    let plan = manager.prepare_rebind(src).unwrap();
    let dst = plan.dst();
    let (mutables, evicted) = manager.allocate_blocks_with_evictions(1).unwrap();
    assert_eq!(mutables[0].block_id(), src);
    assert_eq!(evicted, vec![seq_hash]);
    let re_registered = manager.register_block(mutables.into_iter().next().unwrap().complete(&token_block).unwrap());
    drop(re_registered);

    assert!(matches!(plan.commit(), RebindOutcome::Stale));
    let snapshot = manager.store.debug_snapshot();
    assert!(snapshot.free.contains(&dst));
    assert_eq!(manager.inactive_len(), 1);
}

#[test]
fn held_rebind_can_retry_after_the_lineage_hold_restores() {
    let manager = manager_with_backend(2, lineage_backend);
    let (seq_hash, src) = register_and_release(&manager, 4_000);
    let candidate = manager.inactive_candidates(1).pop().unwrap();
    let plan = manager.prepare_rebind(src).unwrap();
    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("single-block lineage can be held");
    let plan = match plan.commit() {
        RebindOutcome::Busy(plan) => plan,
        outcome => panic!("expected Busy, got {outcome:?}"),
    };
    drop(hold);
    assert!(matches!(plan.commit(), RebindOutcome::Moved { .. }));
    assert_eq!(manager.match_blocks(&[seq_hash]).len(), 1);
}

#[test]
fn dropping_a_rebind_plan_restores_the_reset_gauge() {
    let manager = manager_with_backend(2, lru_backend);
    let (_, src) = register_and_release(&manager, 5_000);
    let before = manager.metrics().snapshot();
    assert_eq!(before.reset_pool_size, 1);
    assert_eq!(before.inactive_pool_size, 1);

    let plan = manager.prepare_rebind(src).unwrap();
    let dst = plan.dst();
    assert_eq!(manager.metrics().snapshot().reset_pool_size, 0);
    drop(plan);

    let after = manager.metrics().snapshot();
    assert_eq!(after.reset_pool_size, 1);
    assert_eq!(after.inactive_pool_size, 1);
    let snapshot = manager.store.debug_snapshot();
    assert_eq!(snapshot.slots[dst], SlotKind::Reset);
    assert!(snapshot.free.contains(&dst));
}

#[test]
fn prepare_rebind_reports_invalid_sources_and_missing_destinations() {
    let manager = manager_with_backend(1, lru_backend);
    assert!(matches!(
        manager.prepare_rebind(1),
        Err(RebindPrepareError::OutOfRange { block_id: 1 })
    ));
    assert!(matches!(
        manager.prepare_rebind(0),
        Err(RebindPrepareError::NotInactive { block_id: 0 })
    ));

    let (_, src) = register_and_release(&manager, 5_500);
    assert!(matches!(
        manager.prepare_rebind(src),
        Err(RebindPrepareError::NoDestination)
    ));
}

#[test]
fn lowering_ceiling_fences_a_prepared_rebind_destination() {
    let manager = manager_with_backend(3, lru_backend);
    let (_, src) = register_and_release(&manager, 6_000);
    let plan = manager.prepare_rebind(src).unwrap();
    let dst = plan.dst();
    manager.set_allocation_ceiling(Some(dst)).unwrap();

    assert!(matches!(plan.commit(), RebindOutcome::Fenced));
    let snapshot = manager.store.debug_snapshot();
    assert_eq!(snapshot.slots[dst], SlotKind::Reset);
    assert!(snapshot.fenced.contains(&dst));
    assert_eq!(manager.inactive_len(), 1);
}

#[test]
fn ceiling_rebalances_resets_and_reset_inactive_pool_keeps_fenced_slots() {
    let manager = manager_with_backend(4, lru_backend);
    let mut mutables = manager.allocate_blocks(4).unwrap();
    let mut registered = Vec::new();
    for (index, mutable) in mutables.drain(..).enumerate() {
        let token = create_test_token_block_from_iota(7_000 + index as u32 * 4);
        registered.push(manager.register_block(mutable.complete(&token).unwrap()));
    }
    drop(registered);
    manager.set_allocation_ceiling(Some(2)).unwrap();

    assert_eq!(manager.allocation_ceiling(), 2);
    manager.reset_inactive_pool().unwrap();
    let snapshot = manager.store.debug_snapshot();
    assert_eq!(snapshot.free, vec![0, 1]);
    assert_eq!(snapshot.fenced, vec![2, 3]);
    assert_eq!(manager.reset_len(), 4);
    assert_eq!(manager.store.available_len(), 2);
    assert_eq!(manager.occupied_blocks(), 0);
    assert!(manager.set_allocation_ceiling(Some(5)).is_err());
    assert!(manager.allocate_blocks_from_reset(3).is_none());
    let eligible = manager.allocate_blocks_from_reset(2).unwrap();
    assert_eq!(
        eligible.iter().map(|block| block.block_id()).collect::<Vec<_>>(),
        vec![0, 1]
    );
    drop(eligible);

    manager.set_allocation_ceiling(None).unwrap();
    assert_eq!(manager.allocation_ceiling(), 4);
    assert_eq!(manager.allocate_blocks_from_reset(4).unwrap().len(), 4);
}

#[test]
fn over_ceiling_evictions_are_fenced_while_eligible_victims_are_returned() {
    let manager = manager_with_backend(4, hashmap_backend);
    let mut mutables = manager.allocate_blocks(4).unwrap();
    let mut registered: Vec<_> = mutables
        .drain(..)
        .enumerate()
        .map(|(index, mutable)| {
            let token = create_test_token_block_from_iota(9_000 + index as u32 * 4);
            Some(manager.register_block(mutable.complete(&token).unwrap()))
        })
        .collect();
    let hashes: Vec<_> = registered
        .iter()
        .map(|block| block.as_ref().unwrap().sequence_hash())
        .collect();
    for block_id in [2, 3, 0, 1] {
        drop(registered[block_id].take());
    }
    manager.set_allocation_ceiling(Some(2)).unwrap();

    let (blocks, evicted) = manager.allocate_blocks_with_evictions(2).unwrap();
    assert_eq!(
        blocks.iter().map(|block| block.block_id()).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(evicted, vec![hashes[2], hashes[3], hashes[0], hashes[1]]);
    let snapshot = manager.store.debug_snapshot();
    assert_eq!(snapshot.fenced, vec![2, 3]);
    assert!(blocks.iter().all(|block| block.block_id() < 2));
}

#[test]
fn atomic_allocation_restores_fenced_victims_on_shortfall() {
    let manager = manager_with_backend(2, lru_backend);
    let (first_hash, _) = register_and_release(&manager, 9_100);
    register_and_release(&manager, 9_104);
    manager.set_allocation_ceiling(Some(0)).unwrap();
    let before = manager.store.debug_snapshot();

    assert!(manager.allocate_blocks_with_evictions(1).is_none());
    assert_eq!(manager.store.debug_snapshot(), before);
    assert_eq!(manager.inactive_len(), 2);

    manager.set_allocation_ceiling(None).unwrap();
    let (blocks, evicted) = manager.allocate_blocks_with_evictions(1).unwrap();
    assert_eq!(evicted, vec![first_hash]);
    assert!(blocks.iter().all(|block| block.block_id() < 2));
}

#[test]
fn inactive_tail_release_fences_over_ceiling_slots() {
    let manager = manager_with_backend(4, hashmap_backend);
    for start in [10_000, 10_004, 10_008, 10_012] {
        register_and_release(&manager, start);
    }
    manager.set_allocation_ceiling(Some(2)).unwrap();

    assert_eq!(manager.release_inactive_tail(2), 2);
    assert_eq!(manager.inactive_len(), 2);
    let snapshot = manager.store.debug_snapshot();
    assert_eq!(snapshot.fenced, vec![2, 3]);
    assert_eq!(snapshot.slots[2], SlotKind::Reset);
    assert_eq!(snapshot.slots[3], SlotKind::Reset);
    let blocks = manager.allocate_blocks(2).unwrap();
    assert!(blocks.iter().all(|block| block.block_id() < 2));
}

#[test]
fn draining_inactive_pool_fences_over_ceiling_slots() {
    let manager = manager_with_backend(4, hashmap_backend);
    for start in [10_100, 10_104, 10_108, 10_112] {
        register_and_release(&manager, start);
    }
    manager.set_allocation_ceiling(Some(2)).unwrap();

    assert_eq!(manager.drain_inactive_pool(), 4);
    assert_eq!(manager.inactive_len(), 0);
    assert_eq!(manager.reset_len(), 4);
    let snapshot = manager.store.debug_snapshot();
    assert_eq!(snapshot.free, vec![0, 1]);
    assert_eq!(snapshot.fenced, vec![2, 3]);
}

#[test]
fn rebind_upper_inactive_slots_before_capacity_shrink() {
    let manager = manager_with_backend(4, hashmap_backend);
    let mut mutables = manager.allocate_blocks(4).unwrap();
    let mut slots = Vec::new();
    for (index, mutable) in mutables.drain(..).enumerate() {
        if index < 2 {
            drop(mutable);
        } else {
            let token = create_test_token_block_from_iota(8_000 + index as u32 * 4);
            let immutable = manager.register_block(mutable.complete(&token).unwrap());
            slots.push((immutable.sequence_hash(), immutable.block_id()));
            drop(immutable);
        }
    }
    manager.set_allocation_ceiling(Some(2)).unwrap();

    for (_, src) in slots {
        let plan = manager.prepare_rebind(src).unwrap();
        assert!(plan.dst() < 2);
        assert!(matches!(plan.commit(), RebindOutcome::Moved { .. }));
    }
    manager.set_capacity(2).unwrap();
    assert_eq!(manager.total_blocks(), 2);
    manager.set_capacity(4).unwrap();
    assert_eq!(manager.allocation_ceiling(), 2);
    assert_eq!(manager.store.debug_snapshot().fenced, vec![2, 3]);
}
