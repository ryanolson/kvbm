// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::metrics::MetricsSnapshot;
use crate::pools::store::DebugStoreSnapshot;

#[derive(Debug, PartialEq)]
struct ManagerState {
    metrics: MetricsSnapshot,
    store: DebugStoreSnapshot,
}

fn state(manager: &BlockManager<TestBlockData>) -> ManagerState {
    ManagerState {
        metrics: manager.metrics().snapshot(),
        store: manager.store_for_test().debug_snapshot(),
    }
}

fn stage_block(
    manager: &BlockManager<TestBlockData>,
    token_start: u32,
) -> CompleteBlock<TestBlockData> {
    let token_block = create_test_token_block_from_iota(token_start);
    manager
        .allocate_blocks(1)
        .expect("allocate one block")
        .pop()
        .expect("one mutable block")
        .complete(&token_block)
        .expect("stage one block")
}

#[test]
fn fallible_registration_accepts_a_same_store_batch() {
    let manager = create_test_manager(2);
    let first = stage_block(&manager, 10);
    let second = stage_block(&manager, 20);
    let hashes = [first.sequence_hash(), second.sequence_hash()];
    let block_ids = [first.block_id(), second.block_id()];

    let registered = manager
        .try_register_blocks(vec![first, second])
        .expect("same-store blocks register");

    assert_eq!(
        registered
            .iter()
            .map(ImmutableBlock::block_id)
            .collect::<Vec<_>>(),
        block_ids
    );
    assert_eq!(manager.match_blocks(&hashes).len(), hashes.len());
}

#[test]
fn fallible_registration_rejects_foreign_blocks_without_mutation_or_stranding() {
    let manager = create_test_manager(2);
    let foreign_manager = create_test_manager(2);
    let local = stage_block(&manager, 30);
    let foreign = stage_block(&foreign_manager, 40);
    let local_hash = local.sequence_hash();
    let foreign_hash = foreign.sequence_hash();
    let local_before = state(&manager);
    let foreign_before = state(&foreign_manager);

    let rejected = match manager.try_register_blocks(vec![local, foreign]) {
        Err(error) => error,
        Ok(_) => panic!("foreign blocks must be rejected"),
    };

    assert_eq!(state(&manager), local_before);
    assert_eq!(state(&foreign_manager), foreign_before);

    let mut recovered = rejected.into_blocks();
    let local = recovered.remove(0);
    let foreign = recovered.remove(0);
    assert!(recovered.is_empty());

    let local_registered = manager
        .try_register_blocks(vec![local])
        .expect("the local block remains usable");
    let foreign_registered = foreign_manager
        .try_register_blocks(vec![foreign])
        .expect("the foreign block remains usable in its own store");
    assert_eq!(local_registered[0].sequence_hash(), local_hash);
    assert_eq!(foreign_registered[0].sequence_hash(), foreign_hash);
}
