use std::sync::{Arc, Mutex};

use dynamo_tokens::TokenBlockSequence;

use super::*;
use crate::testing::{TEST_SALT, create_test_manager_with_backend};
use crate::{ExactAllocationError, ExactInactiveVictim, InactiveCandidate};

fn valued_manager(block_count: usize) -> BlockManager<TestBlockData> {
    create_test_manager_with_backend(block_count, |builder| {
        builder.with_valued_lineage_backend(ScorerParams::default())
    })
}

fn valued_lineage_manager(block_count: usize) -> BlockManager<TestBlockData> {
    create_test_manager_with_backend(block_count, |builder| {
        builder
            .block_size(1)
            .with_valued_lineage_backend(ScorerParams::default())
    })
}

fn register_inactive(manager: &BlockManager<TestBlockData>, token_start: u32) -> InactiveCandidate {
    let token_block = create_iota_token_block(token_start, 4);
    let mutable = manager
        .allocate_blocks(1)
        .expect("allocate one block")
        .pop()
        .expect("one mutable block");
    let immutable = manager.register_block(mutable.complete(&token_block).expect("complete block"));
    let hash = immutable.sequence_hash();
    drop(immutable);

    manager
        .inactive_candidates(manager.total_blocks())
        .into_iter()
        .find(|candidate| candidate.seq_hash == hash)
        .expect("the registered block is inactive")
}

#[test]
fn exact_allocation_evicts_only_the_named_inactive_victim() {
    let manager = valued_manager(3);
    let first = register_inactive(&manager, 10);
    let second = register_inactive(&manager, 100);
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observed_for_callback = Arc::clone(&observed);
    let observer: Arc<dyn BlockEvictionObserver> = Arc::new(move |hashes: &[SequenceHash]| {
        observed_for_callback
            .lock()
            .expect("observation lock")
            .extend_from_slice(hashes);
    });
    manager.observe_evictions(&observer);

    let allocated = manager
        .allocate_blocks_with_exact_inactive(2, &[first.exact_victim(manager.id())])
        .expect("allocate from the exact victim");

    assert_eq!(allocated.len(), 2);
    assert_eq!(manager.inactive_len(), 1);
    assert_eq!(manager.reset_len(), 0);
    assert_eq!(
        *observed.lock().expect("observation lock"),
        vec![first.seq_hash],
        "the observer sees only the exact victim"
    );
    assert!(
        manager.match_blocks(&[first.seq_hash]).is_empty(),
        "the exact victim is absent from the registry"
    );
    let remaining = manager.match_blocks(&[second.seq_hash]);
    assert_eq!(remaining.len(), 1, "an unlisted inactive block remains");
    drop(remaining);
    drop(allocated);
}

#[test]
fn invalid_exact_targets_leave_pool_state_unchanged() {
    let manager = valued_manager(2);
    let first = register_inactive(&manager, 200);
    let second = register_inactive(&manager, 300);
    let first_victim = first.exact_victim(manager.id());
    let before = (
        manager.reset_len(),
        manager.inactive_len(),
        manager.available_blocks(),
    );
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observed_for_callback = Arc::clone(&observed);
    let observer: Arc<dyn BlockEvictionObserver> = Arc::new(move |hashes: &[SequenceHash]| {
        observed_for_callback
            .lock()
            .expect("observation lock")
            .extend_from_slice(hashes);
    });
    manager.observe_evictions(&observer);

    let other_manager = valued_manager(1);
    let wrong_manager = ExactInactiveVictim {
        manager_id: other_manager.id(),
        ..first_victim
    };
    assert!(matches!(
        manager.allocate_blocks_with_exact_inactive(1, &[wrong_manager]),
        Err(ExactAllocationError::WrongManager { .. })
    ));
    assert_eq!(
        (
            manager.reset_len(),
            manager.inactive_len(),
            manager.available_blocks()
        ),
        before
    );

    assert!(matches!(
        manager.allocate_blocks_with_exact_inactive(
            1,
            &[first_victim, second.exact_victim(manager.id())]
        ),
        Err(ExactAllocationError::VictimCountMismatch { .. })
    ));
    assert_eq!(
        (
            manager.reset_len(),
            manager.inactive_len(),
            manager.available_blocks()
        ),
        before
    );

    assert!(matches!(
        manager.allocate_blocks_with_exact_inactive(2, &[first_victim, first_victim]),
        Err(ExactAllocationError::DuplicateVictim { .. })
    ));
    assert_eq!(
        (
            manager.reset_len(),
            manager.inactive_len(),
            manager.available_blocks()
        ),
        before
    );

    assert!(matches!(
        manager.allocate_blocks_with_exact_inactive(2, &[first_victim]),
        Err(ExactAllocationError::VictimCountMismatch { .. })
    ));
    assert_eq!(
        (
            manager.reset_len(),
            manager.inactive_len(),
            manager.available_blocks()
        ),
        before
    );

    let active = manager.match_blocks(&[first.seq_hash]);
    assert_eq!(active.len(), 1);
    let active_before = (
        manager.reset_len(),
        manager.inactive_len(),
        manager.available_blocks(),
    );
    assert!(matches!(
        manager.allocate_blocks_with_exact_inactive(1, &[first_victim]),
        Err(ExactAllocationError::ActiveVictim { .. })
    ));
    assert_eq!(
        (
            manager.reset_len(),
            manager.inactive_len(),
            manager.available_blocks()
        ),
        active_before
    );
    assert_eq!(manager.match_blocks(&[second.seq_hash]).len(), 1);
    drop(active);
    assert!(
        observed.lock().expect("observation lock").is_empty(),
        "a rejected target does not notify observers"
    );
}

#[test]
fn stale_generation_rejects_a_reused_slot_without_evicting_the_current_entry() {
    let manager = valued_manager(1);
    let token_block = create_iota_token_block(400, 4);
    let first = register_inactive(&manager, 400);
    let stale_victim = first.exact_victim(manager.id());

    let mutable = manager
        .allocate_blocks_with_exact_inactive(1, &[stale_victim])
        .expect("evict the first generation")
        .pop()
        .expect("one mutable block");
    let current = manager.register_block(mutable.complete(&token_block).expect("complete block"));
    drop(current);
    let current = manager
        .inactive_candidates(1)
        .into_iter()
        .next()
        .expect("the reused slot is inactive");
    assert_eq!(current.block_id, first.block_id);
    assert_eq!(current.seq_hash, first.seq_hash);
    assert_ne!(current.generation, first.generation);
    let before = (
        manager.reset_len(),
        manager.inactive_len(),
        manager.available_blocks(),
    );

    assert!(matches!(
        manager.allocate_blocks_with_exact_inactive(1, &[stale_victim]),
        Err(ExactAllocationError::StaleVictim { .. })
    ));
    assert_eq!(
        (
            manager.reset_len(),
            manager.inactive_len(),
            manager.available_blocks()
        ),
        before
    );
    assert_eq!(manager.inactive_candidates(1), vec![current]);
}

#[test]
fn inactive_epoch_rejects_a_cache_hit_drop_aba_without_evicting_the_current_entry() {
    let manager = valued_manager(1);
    let stale = register_inactive(&manager, 450);
    let stale_victim = stale.exact_victim(manager.id());

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

    let before = (
        manager.reset_len(),
        manager.inactive_len(),
        manager.available_blocks(),
    );
    assert!(matches!(
        manager.allocate_blocks_with_exact_inactive(1, &[stale_victim]),
        Err(ExactAllocationError::StaleVictim { .. })
    ));
    assert_eq!(
        (
            manager.reset_len(),
            manager.inactive_len(),
            manager.available_blocks()
        ),
        before
    );
    assert_eq!(manager.inactive_candidates(1), vec![current]);
}

#[test]
fn inactive_epoch_rejects_a_cache_hit_batch_release_aba() {
    let manager = valued_manager(1);
    let stale = register_inactive(&manager, 475);
    let stale_victim = stale.exact_victim(manager.id());

    let cached = manager.match_blocks(&[stale.seq_hash]);
    assert_eq!(
        cached.len(),
        1,
        "the original inactive tenure is a cache hit"
    );
    manager.release_blocks(cached, Some(false));

    let current = manager
        .inactive_candidates(1)
        .into_iter()
        .next()
        .expect("the batch release returned the cache hit to inactive");
    assert_eq!(current.block_id, stale.block_id);
    assert_eq!(current.seq_hash, stale.seq_hash);
    assert_eq!(current.generation, stale.generation);
    assert_ne!(current.inactive_epoch, stale.inactive_epoch);

    assert!(matches!(
        manager.allocate_blocks_with_exact_inactive(1, &[stale_victim]),
        Err(ExactAllocationError::StaleVictim { .. })
    ));
    assert_eq!(manager.inactive_candidates(1), vec![current]);
}

#[test]
fn a_candidate_that_became_a_lineage_interior_is_rejected() {
    let manager = valued_lineage_manager(2);
    let sequence = TokenBlockSequence::from_slice(&[500, 501], 1, Some(TEST_SALT));
    let parent_block = sequence.blocks()[0].clone();
    let child_block = sequence.blocks()[1].clone();

    let parent = manager
        .allocate_blocks(1)
        .expect("allocate parent")
        .pop()
        .expect("one parent block");
    let parent = manager.register_block(parent.complete(&parent_block).expect("complete parent"));
    drop(parent);
    let parent_candidate = manager
        .inactive_candidates(1)
        .into_iter()
        .next()
        .expect("the parent begins as a leaf");

    let child = manager
        .allocate_blocks(1)
        .expect("allocate child")
        .pop()
        .expect("one child block");
    let child = manager.register_block(child.complete(&child_block).expect("complete child"));
    drop(child);
    let before = (
        manager.reset_len(),
        manager.inactive_len(),
        manager.available_blocks(),
    );

    assert!(matches!(
        manager
            .allocate_blocks_with_exact_inactive(1, &[parent_candidate.exact_victim(manager.id())]),
        Err(ExactAllocationError::StaleVictim { .. })
    ));
    assert_eq!(
        (
            manager.reset_len(),
            manager.inactive_len(),
            manager.available_blocks()
        ),
        before
    );
    let candidates = manager.inactive_candidates(2);
    assert_eq!(candidates.len(), 1, "only the child remains evictable");
    assert_ne!(candidates[0].block_id, parent_candidate.block_id);
}
