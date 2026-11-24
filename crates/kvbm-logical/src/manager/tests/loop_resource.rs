// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Base identity keeps one primary per lineage hash within one resource and
//! keeps equal lineage hashes independent across resource registries.

use super::*;
use crate::resources::{BlockManagerSet, LogicalResourceId};

type TestManager = BlockManager<TestBlockData>;

fn manager_with_registry(registry: BlockRegistry, block_count: usize) -> Arc<TestManager> {
    Arc::new(
        BlockManager::<TestBlockData>::builder()
            .block_count(block_count)
            .block_size(4)
            .registry(registry)
            .with_lru_backend()
            .duplication_policy(BlockDuplicationPolicy::Reject)
            .build()
            .expect("build manager"),
    )
}

fn register_token(
    manager: &TestManager,
    token: &dynamo_tokens::TokenBlock,
) -> ImmutableBlock<TestBlockData> {
    let block = manager
        .allocate_blocks(1)
        .expect("allocate one block")
        .pop()
        .expect("one mutable block")
        .complete(token)
        .expect("stage one block");
    manager.register_block(block)
}

fn stage_under_hash(
    manager: &TestManager,
    hash: SequenceHash,
) -> CompleteBlock<TestBlockData> {
    manager
        .allocate_blocks(1)
        .expect("allocate one block")
        .pop()
        .expect("one mutable block")
        .stage(hash, manager.block_size())
        .expect("stage one block under the given hash")
}

#[test]
fn loop_resource_same_hash_payload_collapses_onto_the_existing_primary() {
    let manager = manager_with_registry(BlockRegistry::new(), 2);
    let first_token = create_iota_token_block(100, 4);
    let hash = first_token.kvbm_sequence_hash();
    let primary = register_token(&manager, &first_token);
    let primary_id = primary.block_id();

    // A second payload. Its own token block hashes differently, but the
    // registration is staged under the primary's lineage hash, as a second
    // loop stream at the same position would be.
    let second_token = create_iota_token_block(200, 4);
    assert_ne!(second_token.kvbm_sequence_hash(), hash);
    let duplicate = stage_under_hash(&manager, hash);
    let duplicate_id = duplicate.block_id();
    assert_ne!(duplicate_id, primary_id);

    let returned = manager.register_block(duplicate);
    assert_eq!(
        returned.block_id(),
        primary_id,
        "a same-resource collision returns the existing primary"
    );
    assert_eq!(returned.sequence_hash(), hash);
    assert_eq!(manager.block_registry().registered_count(), 1);
    assert_eq!(manager.metrics().snapshot().registration_dedup, 1);

    let matched = manager.match_blocks(&[hash]);
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].block_id(), primary_id);
    assert_eq!(matched[0].sequence_hash(), hash);

    // The rejected duplicate slot went back to the reset pool.
    let returned_to_reset = manager
        .allocate_blocks(1)
        .expect("allocate the rejected duplicate")
        .pop()
        .expect("one mutable block");
    assert_eq!(returned_to_reset.block_id(), duplicate_id);
}

#[test]
fn loop_resource_distinct_registries_look_up_equal_hashes_independently() {
    let registry_a = BlockRegistry::new();
    let registry_b = BlockRegistry::new();
    let manager_a = manager_with_registry(registry_a.clone(), 2);
    let manager_b = manager_with_registry(registry_b.clone(), 2);
    let mut resources = BlockManagerSet::new();
    resources
        .insert(LogicalResourceId(1), Arc::clone(&manager_a))
        .expect("resource one");
    resources
        .insert(LogicalResourceId(2), Arc::clone(&manager_b))
        .expect("resource two");

    // Two different payloads that carry the same positional lineage hash.
    let token_a = create_iota_token_block(300, 4);
    let hash = token_a.kvbm_sequence_hash();
    let token_b = create_iota_token_block(400, 4);
    assert_ne!(token_b.kvbm_sequence_hash(), hash);

    let block_a = register_token(&manager_a, &token_a);
    let block_b = manager_b.register_block(stage_under_hash(&manager_b, hash));
    assert_eq!(block_a.sequence_hash(), hash);
    assert_eq!(block_b.sequence_hash(), hash);
    assert_eq!(block_b.block_id(), block_a.block_id());
    assert_ne!(
        block_a.pin().manager_id(),
        block_b.pin().manager_id(),
        "equal hashes live in different physical resources"
    );

    let hit_a = resources
        .get(LogicalResourceId(1))
        .expect("resource one")
        .match_blocks(&[hash]);
    let hit_b = resources
        .get(LogicalResourceId(2))
        .expect("resource two")
        .match_blocks(&[hash]);
    assert_eq!(hit_a.len(), 1);
    assert_eq!(hit_b.len(), 1);
    assert_eq!(hit_a[0].pin().manager_id(), manager_a.id());
    assert_eq!(hit_b[0].pin().manager_id(), manager_b.id());
    assert_eq!(registry_a.registered_count(), 1);
    assert_eq!(registry_b.registered_count(), 1);

    // Releasing resource one does not remove resource two's primary or its
    // registry entry.
    drop(hit_a);
    drop(block_a);
    assert!(registry_a.is_registered(hash));
    assert!(registry_b.is_registered(hash));
    let still_b = manager_b.match_blocks(&[hash]);
    assert_eq!(still_b.len(), 1);
    assert_eq!(still_b[0].pin().manager_id(), manager_b.id());
}

#[test]
fn loop_resource_legacy_base_lookup_is_unchanged() {
    let manager = create_test_manager(2);
    let token = create_iota_token_block(500, 4);
    let hash = token.kvbm_sequence_hash();
    let block = register_token(&manager, &token);

    let matched = manager.match_blocks(&[hash]);
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].block_id(), block.block_id());
    assert_eq!(matched[0].sequence_hash(), hash);
    assert_eq!(manager.block_registry().registered_count(), 1);
}
