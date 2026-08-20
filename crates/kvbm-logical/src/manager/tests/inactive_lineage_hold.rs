use std::sync::{Arc, Barrier};

use dynamo_tokens::TokenBlockSequence;

use super::*;
use crate::pools::{InactiveCandidate, InactiveFeatures};
use crate::testing::{TEST_SALT, create_test_manager_with_backend};
use crate::{BlockId, ImmutableBlock};

fn valued_manager(pages: usize) -> BlockManager<TestBlockData> {
    create_test_manager_with_backend(pages, |builder| {
        builder
            .block_size(1)
            .with_valued_lineage_backend(ScorerParams::default())
    })
}

fn hashes(tokens: &[u32]) -> Vec<SequenceHash> {
    TokenBlockSequence::from_slice(tokens, 1, Some(TEST_SALT))
        .blocks()
        .iter()
        .map(|block| block.kvbm_sequence_hash())
        .collect()
}

fn register_chain(
    manager: &BlockManager<TestBlockData>,
    tokens: &[u32],
) -> Vec<ImmutableBlock<TestBlockData>> {
    TokenBlockSequence::from_slice(tokens, 1, Some(TEST_SALT))
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
        .collect()
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

fn inactive_chain(
    manager: &BlockManager<TestBlockData>,
    tokens: &[u32],
) -> (Vec<(SequenceHash, BlockId)>, InactiveCandidate) {
    let blocks = register_chain(manager, tokens);
    let expected = blocks
        .iter()
        .map(|block| (block.sequence_hash(), block.block_id()))
        .collect::<Vec<_>>();
    drop(blocks);
    let candidate = manager
        .inactive_candidates(1)
        .into_iter()
        .next()
        .expect("the lineage leaf is a candidate");
    (expected, candidate)
}

#[test]
fn exact_candidate_holds_the_complete_lineage_exclusively() {
    let manager = valued_manager(8);
    let (expected, candidate) = inactive_chain(&manager, &[1, 2, 3]);
    let chain_hashes = expected.iter().map(|(hash, _)| *hash).collect::<Vec<_>>();

    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold the exact inactive lineage");

    assert_eq!(hold.manager_id(), manager.id());
    assert_eq!(hold.source_blocks(), expected.as_slice());
    assert_eq!(manager.inactive_len(), 0);
    assert_eq!(
        manager.metrics().snapshot().inflight_immutable,
        0,
        "a hold owns slots, not ImmutableBlock handles"
    );
    assert_eq!(
        manager.metrics().snapshot().held_residency,
        expected.len() as i64
    );
    assert!(
        manager.match_blocks(&chain_hashes).is_empty(),
        "a request cannot activate a pressure-held lineage"
    );

    drop(hold);
    assert_eq!(manager.inactive_len(), expected.len());
    assert_eq!(manager.metrics().snapshot().inflight_immutable, 0);
    assert_eq!(manager.metrics().snapshot().held_residency, 0);
    assert_eq!(manager.match_blocks(&chain_hashes).len(), expected.len());
}

#[test]
fn prepared_lineage_reports_the_exact_count_and_holds_that_lineage() {
    let manager = valued_manager(8);
    let (expected, candidate) = inactive_chain(&manager, &[4, 5, 6]);

    let prepared = manager
        .preflight_inactive_lineage(candidate)
        .expect("preflight the complete inactive lineage");
    assert_eq!(prepared.block_count(), expected.len());

    let hold = manager
        .try_hold_prepared_inactive_lineage(prepared)
        .expect("the unchanged prepared lineage acquires atomically");
    assert_eq!(hold.source_blocks(), expected.as_slice());
}

#[test]
fn point_preflight_by_hash_names_the_complete_inactive_leaf_lineage() {
    let manager = valued_manager(8);
    let (expected, candidate) = inactive_chain(&manager, &[30, 31, 32]);

    let prepared = manager
        .preflight_inactive_lineage_by_hash(candidate.seq_hash)
        .expect("preflight the inactive lineage by its leaf hash");
    assert_eq!(prepared.block_count(), expected.len());

    let hold = manager
        .try_hold_prepared_inactive_lineage(prepared)
        .expect("the unchanged point preflight acquires atomically");
    assert_eq!(hold.source_blocks(), expected.as_slice());
}

#[test]
fn point_preflight_by_hash_rejects_an_active_or_interior_hash_without_mutation() {
    let manager = valued_manager(8);
    let (expected, candidate) = inactive_chain(&manager, &[40, 41, 42]);
    let root_hash = expected.first().expect("lineage root").0;
    let before = manager.store_for_test().debug_snapshot();

    assert!(
        manager
            .preflight_inactive_lineage_by_hash(root_hash)
            .is_none(),
        "an interior hash must not name an evictable lineage"
    );
    assert_eq!(manager.store_for_test().debug_snapshot(), before);

    let active = manager.match_blocks(&[candidate.seq_hash]);
    assert_eq!(active.len(), 1, "activate the inactive leaf");
    let active_before = manager.store_for_test().debug_snapshot();
    assert!(
        manager
            .preflight_inactive_lineage_by_hash(candidate.seq_hash)
            .is_none(),
        "an active leaf must not name an inactive lineage"
    );
    assert_eq!(manager.store_for_test().debug_snapshot(), active_before);
    drop(active);
}

#[test]
fn stale_prepared_lineage_rejects_without_a_partial_hold() {
    let manager = valued_manager(8);
    let (expected, candidate) = inactive_chain(&manager, &[7, 8, 9]);
    let prepared = manager
        .preflight_inactive_lineage(candidate)
        .expect("preflight the complete inactive lineage");

    let hashes = expected.iter().map(|(hash, _)| *hash).collect::<Vec<_>>();
    let active_prefix = manager.match_blocks(&hashes[..2]);
    assert_eq!(active_prefix.len(), 2, "activate two preflight ancestors");
    let before = manager.store_for_test().debug_snapshot();

    assert!(
        manager
            .try_hold_prepared_inactive_lineage(prepared)
            .is_none(),
        "the preflight must reject after its exact lineage changes"
    );
    assert_eq!(manager.store_for_test().debug_snapshot(), before);
    drop(active_prefix);
}

#[test]
fn cross_manager_prepared_lineage_rejects_without_mutation() {
    let first = valued_manager(4);
    let second = valued_manager(4);
    let (first_expected, first_candidate) = inactive_chain(&first, &[10]);
    let (second_expected, second_candidate) = inactive_chain(&second, &[10]);
    assert_eq!(
        first_candidate, second_candidate,
        "the test requires a collision in the raw candidate fields"
    );
    let prepared = first
        .preflight_inactive_lineage(first_candidate)
        .expect("preflight the first manager");
    let first_before = first.store_for_test().debug_snapshot();
    let second_before = second.store_for_test().debug_snapshot();

    assert!(
        second
            .try_hold_prepared_inactive_lineage(prepared)
            .is_none(),
        "a descriptor from another manager has no mutation authority here"
    );
    assert_eq!(first.store_for_test().debug_snapshot(), first_before);
    assert_eq!(second.store_for_test().debug_snapshot(), second_before);
    assert_eq!(first.inactive_len(), first_expected.len());
    assert_eq!(second.inactive_len(), second_expected.len());
}

#[test]
fn stale_wrong_and_active_candidates_are_skipped_without_mutation() {
    let manager = valued_manager(8);
    let (expected, candidate) = inactive_chain(&manager, &[10, 11, 12]);
    let chain_hashes = expected.iter().map(|(hash, _)| *hash).collect::<Vec<_>>();

    let wrong_id = InactiveCandidate {
        block_id: (candidate.block_id + 1) % manager.total_blocks(),
        ..candidate
    };
    assert!(manager.try_hold_inactive_lineage(wrong_id).is_none());

    let other_hash = hashes(&[90])[0];
    let wrong_hash = InactiveCandidate {
        seq_hash: other_hash,
        ..candidate
    };
    assert!(manager.try_hold_inactive_lineage(wrong_hash).is_none());
    assert_eq!(manager.inactive_len(), expected.len());

    let request_hold = manager.match_blocks(&chain_hashes);
    assert_eq!(request_hold.len(), expected.len());
    assert!(manager.try_hold_inactive_lineage(candidate).is_none());
    drop(request_hold);
}

#[test]
fn a_stale_generation_cannot_hold_a_reused_lineage() {
    let manager = valued_manager(1);
    let (_expected, stale) = inactive_chain(&manager, &[15]);

    let current_blocks = register_chain(&manager, &[15]);
    drop(current_blocks);
    let current = manager
        .inactive_candidates(1)
        .into_iter()
        .next()
        .expect("the reused lineage is inactive");
    assert_eq!(current.block_id, stale.block_id);
    assert_eq!(current.seq_hash, stale.seq_hash);
    assert_ne!(current.generation, stale.generation);

    assert!(manager.try_hold_inactive_lineage(stale).is_none());
    assert_eq!(manager.inactive_candidates(1), vec![current]);
}

#[test]
fn an_inactive_epoch_rejects_a_cache_hit_drop_aba_for_a_lineage_hold() {
    let manager = valued_manager(1);
    let (_expected, stale) = inactive_chain(&manager, &[16]);

    let cached = manager.match_blocks(&[stale.seq_hash]);
    assert_eq!(
        cached.len(),
        1,
        "the original inactive tenure is a cache hit"
    );
    drop(cached);

    let current = manager
        .inactive_candidates(1)
        .into_iter()
        .next()
        .expect("the cache hit returned to inactive");
    assert_eq!(current.block_id, stale.block_id);
    assert_eq!(current.seq_hash, stale.seq_hash);
    assert_eq!(current.generation, stale.generation);
    assert_ne!(current.inactive_epoch, stale.inactive_epoch);

    assert!(manager.try_hold_inactive_lineage(stale).is_none());
    assert!(manager.try_hold_inactive_lineage(current).is_some());
}

#[test]
fn request_activation_and_pressure_hold_have_one_race_winner() {
    let manager = Arc::new(valued_manager(8));
    let (expected, candidate) = inactive_chain(&manager, &[20, 21, 22]);
    let chain_hashes = expected.iter().map(|(hash, _)| *hash).collect::<Vec<_>>();
    let start = Arc::new(Barrier::new(3));
    let finish = Arc::new(Barrier::new(3));

    let pressure = {
        let manager = Arc::clone(&manager);
        let start = Arc::clone(&start);
        let finish = Arc::clone(&finish);
        std::thread::spawn(move || {
            start.wait();
            let hold = manager.try_hold_inactive_lineage(candidate);
            let won = hold.is_some();
            finish.wait();
            drop(hold);
            won
        })
    };
    let request = {
        let manager = Arc::clone(&manager);
        let start = Arc::clone(&start);
        let finish = Arc::clone(&finish);
        std::thread::spawn(move || {
            start.wait();
            let hold = manager.match_blocks(&chain_hashes);
            let won = hold.len() == expected.len();
            finish.wait();
            drop(hold);
            won
        })
    };

    start.wait();
    finish.wait();
    let pressure_won = pressure.join().expect("pressure thread joins");
    let request_won = request.join().expect("request thread joins");
    assert_ne!(pressure_won, request_won, "exactly one owner wins the race");
}

#[test]
fn abort_preserves_the_lineage_and_commit_resets_only_the_victim() {
    let manager = valued_manager(8);
    let (expected, candidate) = inactive_chain(&manager, &[30, 31, 32]);
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_for_callback = Arc::clone(&observed);
    let observer: Arc<dyn BlockEvictionObserver> = Arc::new(move |hashes: &[SequenceHash]| {
        observed_for_callback
            .lock()
            .expect("eviction observation lock")
            .extend_from_slice(hashes);
    });
    manager.observe_evictions(&observer);
    let reset_before = manager.reset_len();

    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold before abort");
    drop(hold);
    assert_eq!(manager.inactive_len(), expected.len());
    assert_eq!(manager.reset_len(), reset_before);
    assert!(observed.lock().expect("observation lock").is_empty());

    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold before commit");
    hold.commit_victim_release();

    assert_eq!(manager.reset_len(), reset_before + 1);
    assert_eq!(manager.inactive_len(), expected.len() - 1);
    assert_eq!(
        *observed.lock().expect("observation lock"),
        vec![candidate.seq_hash],
        "commit notifies only the selected victim"
    );
    let hashes = expected.iter().map(|(hash, _)| *hash).collect::<Vec<_>>();
    assert_eq!(
        manager.match_blocks(&hashes).len(),
        expected.len() - 1,
        "only the selected leaf leaves G1"
    );
}

#[test]
fn silent_commit_finishes_source_mutation_before_a_panicking_observer() {
    let manager = valued_manager(8);
    let (expected, candidate) = inactive_chain(&manager, &[33, 34, 35]);
    let reset_before = manager.reset_len();
    let observer: Arc<dyn BlockEvictionObserver> =
        Arc::new(|_: &[SequenceHash]| panic!("test observer panics after the source commit"));
    manager.observe_evictions(&observer);

    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold before deferred commit");
    let notification = hold.commit_victim_release_silent();

    assert_eq!(manager.reset_len(), reset_before + 1);
    assert_eq!(manager.inactive_len(), expected.len() - 1);
    assert_eq!(
        manager
            .match_blocks(&expected.iter().map(|(hash, _)| *hash).collect::<Vec<_>>())
            .len(),
        expected.len() - 1,
        "the source state commits before an observer runs"
    );

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        notification.notify();
    }));
    assert!(panic.is_err(), "the observer panic reaches the caller");
    assert_eq!(
        manager.reset_len(),
        reset_before + 1,
        "a notification panic cannot roll back the committed victim"
    );
    assert_eq!(
        manager.inactive_len(),
        expected.len() - 1,
        "a notification panic cannot corrupt the restored support lineage"
    );
}

#[test]
fn silent_commit_defers_and_orders_notifications_after_all_sources_commit() {
    let first = valued_manager(4);
    let second = valued_manager(4);
    let (_first_blocks, first_candidate) = inactive_chain(&first, &[36]);
    let (_second_blocks, second_candidate) = inactive_chain(&second, &[37]);
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_for_callback = Arc::clone(&observed);
    let observer: Arc<dyn BlockEvictionObserver> = Arc::new(move |hashes: &[SequenceHash]| {
        observed_for_callback
            .lock()
            .expect("eviction observation lock")
            .extend_from_slice(hashes);
    });
    first.observe_evictions(&observer);
    second.observe_evictions(&observer);

    let first_notification = first
        .try_hold_inactive_lineage(first_candidate)
        .expect("hold first source")
        .commit_victim_release_silent();
    let second_notification = second
        .try_hold_inactive_lineage(second_candidate)
        .expect("hold second source")
        .commit_victim_release_silent();

    assert!(
        observed
            .lock()
            .expect("eviction observation lock")
            .is_empty(),
        "silent commits must not notify before every source commits"
    );
    second_notification.notify();
    first_notification.notify();
    assert_eq!(
        *observed.lock().expect("eviction observation lock"),
        vec![second_candidate.seq_hash, first_candidate.seq_hash],
        "the explicit notification phase controls observer order"
    );
}

#[test]
fn a_missing_inactive_ancestor_rejects_the_whole_hold() {
    let manager = valued_manager(8);
    let blocks = register_chain(&manager, &[40, 41, 42]);
    let hashes = blocks
        .iter()
        .map(ImmutableBlock::sequence_hash)
        .collect::<Vec<_>>();
    let active_prefix = manager.match_blocks(&hashes[..2]);
    drop(blocks);

    let candidate = manager
        .inactive_candidates(1)
        .into_iter()
        .next()
        .expect("the unique tail is inactive");
    assert_eq!(candidate.seq_hash, hashes[2]);
    assert!(manager.try_hold_inactive_lineage(candidate).is_none());
    assert_eq!(
        manager.inactive_len(),
        1,
        "the rejected tail stays inactive"
    );

    drop(active_prefix);
    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("the complete inactive lineage becomes holdable");
    assert_eq!(hold.source_blocks().len(), hashes.len());
}

#[test]
fn branch_abort_and_commit_preserve_shared_support_and_sibling() {
    let manager = valued_manager(8);
    let branch = hashes(&[50, 51, 52, 53]);
    let sibling = hashes(&[50, 51, 99]);
    let branch_blocks = register_chain(&manager, &[50, 51, 52, 53]);
    let expected = branch_blocks
        .iter()
        .map(|block| (block.sequence_hash(), block.block_id()))
        .collect::<Vec<_>>();
    let sibling_block = register_divergent(&manager, &[50, 51, 99], 2);
    drop(branch_blocks);
    drop(sibling_block);

    let candidate = manager
        .inactive_candidates(8)
        .into_iter()
        .find(|candidate| candidate.seq_hash == branch[3])
        .expect("the selected branch leaf is a candidate");
    let reset_before = manager.reset_len();

    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold the selected branch");
    assert_eq!(hold.source_blocks(), expected.as_slice());
    drop(hold);
    assert_eq!(manager.reset_len(), reset_before);

    let restored_branch = manager.match_blocks(&branch);
    let restored_sibling = manager.match_blocks(&[sibling[2]]);
    assert_eq!(restored_branch.len(), branch.len());
    assert_eq!(restored_sibling.len(), 1);
    drop(restored_branch);
    drop(restored_sibling);

    let candidate = manager
        .inactive_candidates(8)
        .into_iter()
        .find(|candidate| candidate.seq_hash == branch[3])
        .expect("the restored branch leaf has a new inactive tenure");
    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold the restored branch");
    hold.commit_victim_release();

    assert_eq!(manager.reset_len(), reset_before + 1);
    let retained_branch = manager.match_blocks(&branch);
    let retained_sibling = manager.match_blocks(&[sibling[2]]);
    assert_eq!(retained_branch.len(), branch.len() - 1);
    assert_eq!(retained_sibling.len(), 1);
    assert!(manager.match_blocks(&[branch[3]]).is_empty());
    drop(retained_branch);
    drop(retained_sibling);
}

#[test]
fn scan_and_ordinary_allocation_cannot_claim_held_slots() {
    let manager = valued_manager(2);
    let (source_blocks, candidate) = inactive_chain(&manager, &[60, 61]);
    let hashes = source_blocks
        .iter()
        .map(|(hash, _)| *hash)
        .collect::<Vec<_>>();
    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold all capacity");
    let before = manager.store_for_test().debug_snapshot();

    assert!(manager.scan_matches(&hashes, false).is_empty());
    assert!(manager.allocate_blocks(1).is_none());
    assert_eq!(manager.store_for_test().debug_snapshot(), before);

    drop(hold);
}

#[test]
fn non_lineage_backend_rejects_hold_without_mutation() {
    let manager = create_test_manager(1);
    let token_block = create_test_token_block_from_iota(70_000);
    let mutable = manager
        .allocate_blocks(1)
        .expect("allocate one LRU block")
        .pop()
        .expect("one allocated block");
    let immutable =
        manager.register_block(mutable.complete(&token_block).expect("complete LRU block"));
    let seq_hash = immutable.sequence_hash();
    let block_id = immutable.block_id();
    drop(immutable);
    let candidate = InactiveCandidate {
        seq_hash,
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
    let before = manager.store_for_test().debug_snapshot();

    assert!(manager.try_hold_inactive_lineage(candidate).is_none());
    assert_eq!(manager.store_for_test().debug_snapshot(), before);
    assert_eq!(manager.match_blocks(&[candidate.seq_hash]).len(), 1);
}

#[test]
fn observer_reentry_sees_an_absent_committed_victim_and_abort_emits_no_event() {
    let manager = Arc::new(valued_manager(4));
    let (_source_blocks, candidate) = inactive_chain(&manager, &[70, 71]);
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_for_callback = Arc::clone(&observed);
    let manager_for_callback = Arc::clone(&manager);
    let victim_hash = candidate.seq_hash;
    let observer: Arc<dyn BlockEvictionObserver> = Arc::new(move |hashes: &[SequenceHash]| {
        let presence = manager_for_callback
            .block_registry()
            .check_presence::<TestBlockData>(&[victim_hash]);
        let matched = manager_for_callback.match_blocks(&[victim_hash]);
        observed_for_callback
            .lock()
            .expect("observation lock")
            .push((hashes.to_vec(), presence, matched.len()));
        drop(matched);
    });
    manager.observe_evictions(&observer);

    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold before abort");
    drop(hold);
    assert!(observed.lock().expect("observation lock").is_empty());

    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold before commit");
    hold.commit_victim_release();

    assert_eq!(
        *observed.lock().expect("observation lock"),
        vec![(vec![victim_hash], vec![(victim_hash, false)], 0)]
    );
}

#[test]
fn held_blocks_remain_present_but_are_not_request_available() {
    let manager = valued_manager(4);
    let (source_blocks, candidate) = inactive_chain(&manager, &[80, 81]);
    let hashes = source_blocks
        .iter()
        .map(|(hash, _)| *hash)
        .collect::<Vec<_>>();
    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold the lineage");

    assert_eq!(
        manager
            .block_registry()
            .check_presence::<TestBlockData>(&hashes),
        hashes
            .iter()
            .copied()
            .map(|hash| (hash, true))
            .collect::<Vec<_>>()
    );
    assert!(manager.match_blocks(&hashes).is_empty());
    assert!(manager.scan_matches(&hashes, false).is_empty());

    drop(hold);
}

#[test]
fn abort_restores_a_held_lineage_after_same_hash_support_replacement() {
    let manager = valued_manager(4);
    let (source_blocks, candidate) = inactive_chain(&manager, &[90, 91]);
    let hashes = source_blocks
        .iter()
        .map(|(hash, _)| *hash)
        .collect::<Vec<_>>();
    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold the original lineage");

    let replacement = register_divergent(&manager, &[90, 91], 0);
    assert_eq!(replacement.sequence_hash(), hashes[0]);
    drop(replacement);

    assert!(manager.inactive_candidates(4).is_empty());
    assert_eq!(manager.inactive_advice(&hashes), vec![None, None]);
    let request_matches = manager.match_blocks(&hashes).len();
    let scan_matches = manager.scan_matches(&hashes, false).len();
    drop(hold);

    assert_eq!(request_matches, 0, "a held hash blocks prefix lookup");
    assert_eq!(scan_matches, 0, "a held hash blocks scan lookup");
    assert_eq!(manager.match_blocks(&hashes).len(), hashes.len());
    assert_eq!(manager.inactive_len(), hashes.len());
    assert_eq!(
        manager
            .block_registry()
            .check_presence::<TestBlockData>(&hashes),
        hashes
            .iter()
            .copied()
            .map(|hash| (hash, true))
            .collect::<Vec<_>>()
    );
}

#[test]
fn commit_releases_only_the_leaf_after_same_hash_support_replacement() {
    let manager = valued_manager(4);
    let (source_blocks, candidate) = inactive_chain(&manager, &[92, 93]);
    let hashes = source_blocks
        .iter()
        .map(|(hash, _)| *hash)
        .collect::<Vec<_>>();
    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold the original lineage");

    let replacement = register_divergent(&manager, &[92, 93], 0);
    assert_eq!(replacement.sequence_hash(), hashes[0]);
    drop(replacement);

    let request_matches = manager.match_blocks(&hashes).len();
    let scan_matches = manager.scan_matches(&hashes, false).len();
    hold.commit_victim_release();

    assert_eq!(request_matches, 0, "a held hash blocks prefix lookup");
    assert_eq!(scan_matches, 0, "a held hash blocks scan lookup");
    assert_eq!(manager.match_blocks(&hashes[..1]).len(), 1);
    assert!(manager.match_blocks(&hashes[1..]).is_empty());
    assert_eq!(
        manager
            .block_registry()
            .check_presence::<TestBlockData>(&hashes),
        vec![(hashes[0], true), (hashes[1], false)]
    );
}

#[test]
fn commit_keeps_a_same_hash_leaf_replacement_and_skips_its_eviction_notification() {
    let manager = valued_manager(2);
    let (source_blocks, candidate) = inactive_chain(&manager, &[94]);
    let hash = source_blocks[0].0;
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_for_callback = Arc::clone(&observed);
    let observer: Arc<dyn BlockEvictionObserver> = Arc::new(move |hashes: &[SequenceHash]| {
        observed_for_callback
            .lock()
            .expect("eviction observation lock")
            .extend_from_slice(hashes);
    });
    manager.observe_evictions(&observer);

    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold the original leaf");
    let replacement = register_divergent(&manager, &[94], 0);
    assert_eq!(replacement.sequence_hash(), hash);
    drop(replacement);

    hold.commit_victim_release();

    assert_eq!(manager.match_blocks(&[hash]).len(), 1);
    assert_eq!(
        manager
            .block_registry()
            .check_presence::<TestBlockData>(&[hash]),
        vec![(hash, true)]
    );
    assert!(
        observed
            .lock()
            .expect("eviction observation lock")
            .is_empty(),
        "a replacement stays resident, so the hash was not evicted"
    );
}

#[test]
fn a_second_hold_with_a_shared_held_prefix_is_rejected_without_mutation() {
    let manager = valued_manager(8);
    let (_first_blocks, first_candidate) = inactive_chain(&manager, &[95, 96]);
    let first_hold = manager
        .try_hold_inactive_lineage(first_candidate)
        .expect("hold the first lineage");

    let replacement_root = register_divergent(&manager, &[95, 97], 0);
    let replacement_leaf = register_divergent(&manager, &[95, 97], 1);
    let replacement_hashes = vec![
        replacement_root.sequence_hash(),
        replacement_leaf.sequence_hash(),
    ];
    drop(replacement_root);
    drop(replacement_leaf);

    let second_candidate = manager
        .inactive_candidates(8)
        .into_iter()
        .find(|candidate| candidate.seq_hash == replacement_hashes[1])
        .expect("the replacement leaf is inactive");
    let before = manager.store_for_test().debug_snapshot();

    assert!(
        manager
            .try_hold_inactive_lineage(second_candidate)
            .is_none()
    );
    assert_eq!(manager.store_for_test().debug_snapshot(), before);

    drop(first_hold);
    assert_eq!(
        manager.match_blocks(&replacement_hashes).len(),
        replacement_hashes.len(),
        "the rejected second hold leaves the replacement lineage available"
    );
}
