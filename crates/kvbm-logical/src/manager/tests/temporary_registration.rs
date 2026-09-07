use crate::blocks::{BlockDuplicationPolicy, CompleteBlock, ImmutableBlock};
use crate::testing::{TestMeta, create_iota_token_block};
use crate::{BlockManager, BlockRegistry};

fn manager(count: usize) -> BlockManager<TestMeta> {
    BlockManager::builder()
        .block_count(count)
        .block_size(4)
        .registry(BlockRegistry::new())
        .with_lru_backend()
        .duplication_policy(BlockDuplicationPolicy::Reject)
        .build()
        .expect("test manager")
}

fn complete(manager: &BlockManager<TestMeta>, start: u32) -> CompleteBlock<TestMeta> {
    manager
        .allocate_blocks(1)
        .expect("free slot")
        .pop()
        .expect("one block")
        .complete(&create_iota_token_block(start, 4))
        .expect("complete block")
}

fn register_temporary(
    manager: &BlockManager<TestMeta>,
    mut block: CompleteBlock<TestMeta>,
) -> ImmutableBlock<TestMeta> {
    block.set_evict_on_reset(true);
    manager.register_block(block)
}

#[test]
fn temporary_registration_preserves_retained_primary() {
    let manager = manager(2);
    let retained = manager.register_block(complete(&manager, 100));
    let hash = retained.sequence_hash();
    let destination = complete(&manager, 100);
    assert_ne!(destination.block_id(), retained.block_id());

    let temporary = register_temporary(&manager, destination);
    assert_eq!(temporary.block_id(), retained.block_id());
    drop(retained);
    drop(temporary);

    let snapshot = manager.metrics().snapshot();
    assert_eq!(snapshot.inactive_pool_size, 1);
    assert_eq!(snapshot.reset_pool_size, 1);
    assert_eq!(manager.match_blocks(&[hash]).len(), 1);
}

#[test]
fn policy_adoption_survives_another_temporary_registration() {
    let manager = manager(3);
    let temporary = register_temporary(&manager, complete(&manager, 200));
    let hash = temporary.sequence_hash();
    let retained = manager.register_block(complete(&manager, 200));
    assert_eq!(retained.block_id(), temporary.block_id());
    retained.set_evict_on_reset(false);

    let another = register_temporary(&manager, complete(&manager, 200));
    assert_eq!(another.block_id(), retained.block_id());
    drop((retained, temporary, another));

    let snapshot = manager.metrics().snapshot();
    assert_eq!(snapshot.inactive_pool_size, 1);
    assert_eq!(snapshot.reset_pool_size, 2);
    assert_eq!(manager.match_blocks(&[hash]).len(), 1);
}

#[test]
fn temporary_registration_returns_to_free_after_last_lifecycle_pin() {
    let manager = manager(1);
    let temporary = register_temporary(&manager, complete(&manager, 300));
    let hash = temporary.sequence_hash();
    let pin = temporary.pin();
    let second_pin = pin.clone();
    drop(temporary);
    drop(pin);

    assert!(manager.allocate_blocks(1).is_none());
    assert_eq!(manager.match_blocks(&[hash]).len(), 1);
    drop(second_pin);

    let snapshot = manager.metrics().snapshot();
    assert_eq!(snapshot.inactive_pool_size, 0);
    assert_eq!(snapshot.reset_pool_size, 1);
    assert!(manager.match_blocks(&[hash]).is_empty());
}

#[test]
fn temporary_registration_does_not_affect_the_next_slot_tenant() {
    let manager = manager(1);
    let temporary = register_temporary(&manager, complete(&manager, 400));
    let id = temporary.block_id();
    drop(temporary);

    let retained = manager.register_block(complete(&manager, 500));
    let hash = retained.sequence_hash();
    assert_eq!(retained.block_id(), id);
    drop(retained);

    let snapshot = manager.metrics().snapshot();
    assert_eq!(snapshot.inactive_pool_size, 1);
    assert_eq!(snapshot.reset_pool_size, 0);
    assert_eq!(manager.match_blocks(&[hash]).len(), 1);
}

#[test]
fn staged_reset_flag_does_not_publish_a_hash() {
    let manager = manager(1);
    let mut block = complete(&manager, 600);
    let hash = block.sequence_hash();
    block.set_evict_on_reset(true);

    assert!(!manager.block_registry().is_registered(hash));
    assert!(manager.match_blocks(&[hash]).is_empty());
    assert_eq!(manager.available_blocks(), 0);
    drop(block);

    assert_eq!(manager.metrics().snapshot().reset_pool_size, 1);
    let retained = manager.register_block(complete(&manager, 700));
    drop(retained);
    assert_eq!(manager.metrics().snapshot().inactive_pool_size, 1);
}

#[rstest::rstest]
#[case(false)]
#[case(true)]
fn staged_reset_flag_rolls_back_to_the_manager_default(#[case] default_reset: bool) {
    let manager = crate::testing::create_test_manager_with_default_reset_on_release::<TestMeta>(
        1,
        default_reset,
    );
    let mut block = complete(&manager, 800);
    block.set_evict_on_reset(!default_reset);
    let reset = block.reset();
    let retained = manager.register_block(
        reset
            .complete(&create_iota_token_block(900, 4))
            .expect("complete reset block"),
    );
    drop(retained);

    let snapshot = manager.metrics().snapshot();
    assert_eq!(snapshot.reset_pool_size, i64::from(default_reset));
    assert_eq!(snapshot.inactive_pool_size, i64::from(!default_reset));
}

#[test]
fn staged_reset_flag_survives_a_registration_batch() {
    let manager = manager(3);
    let retained = manager.register_block(complete(&manager, 1000));
    let retained_hash = retained.sequence_hash();
    let mut collision = complete(&manager, 1000);
    let mut temporary = complete(&manager, 1100);
    let temporary_hash = temporary.sequence_hash();
    collision.set_evict_on_reset(true);
    temporary.set_evict_on_reset(true);

    let registered = manager.register_blocks(vec![collision, temporary]);
    assert_eq!(registered[0].block_id(), retained.block_id());
    drop((retained, registered));

    let snapshot = manager.metrics().snapshot();
    assert_eq!(snapshot.inactive_pool_size, 1);
    assert_eq!(snapshot.reset_pool_size, 2);
    assert_eq!(manager.match_blocks(&[retained_hash]).len(), 1);
    assert!(manager.match_blocks(&[temporary_hash]).is_empty());
}
