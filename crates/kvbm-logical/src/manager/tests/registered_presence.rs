// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::metrics::MetricsSnapshot;
use crate::pools::store::DebugStoreSnapshot;
use crate::testing::create_test_manager_with_backend;

fn valued_manager(block_count: usize) -> BlockManager<TestBlockData> {
    create_test_manager_with_backend(block_count, |builder| {
        builder
            .block_size(4)
            .with_valued_lineage_backend(ScorerParams::default())
    })
}

fn register_block(
    manager: &BlockManager<TestBlockData>,
    token_start: u32,
) -> ImmutableBlock<TestBlockData> {
    let token_block = create_test_token_block_from_iota(token_start);
    let mutable = manager
        .allocate_blocks(1)
        .expect("allocate one block")
        .pop()
        .expect("one mutable block");
    manager.register_block(mutable.complete(&token_block).expect("complete one block"))
}

fn store_state(manager: &BlockManager<TestBlockData>) -> DebugStoreSnapshot {
    manager.store_for_test().debug_snapshot()
}

fn assert_non_mutating_presence_check(
    manager: &BlockManager<TestBlockData>,
    hashes: &[SequenceHash],
    expected: bool,
) {
    let candidates_before = manager.inactive_candidates(manager.total_blocks());
    let metrics_before: MetricsSnapshot = manager.metrics().snapshot();
    let store_before = store_state(manager);

    assert_eq!(manager.has_any_registered_hashes(hashes), expected);

    assert_eq!(
        manager.inactive_candidates(manager.total_blocks()),
        candidates_before,
        "the presence check does not reorder or touch inactive candidates"
    );
    assert_eq!(
        manager.metrics().snapshot(),
        metrics_before,
        "the presence check does not change metrics"
    );
    assert_eq!(
        store_state(manager),
        store_before,
        "the presence check does not change store state"
    );
}

#[test]
fn registered_presence_rejects_empty_and_absent_hashes_without_mutation() {
    let manager = valued_manager(2);
    let absent = create_test_token_block_from_iota(10).kvbm_sequence_hash();

    assert_non_mutating_presence_check(&manager, &[], false);
    assert_non_mutating_presence_check(&manager, &[absent], false);
}

#[test]
fn registered_presence_finds_an_active_hash_without_mutation() {
    let manager = valued_manager(2);
    let active = register_block(&manager, 20);
    let absent = create_test_token_block_from_iota(21).kvbm_sequence_hash();

    assert_non_mutating_presence_check(&manager, &[absent, active.sequence_hash()], true);
}

#[test]
fn registered_presence_finds_an_inactive_hash_without_mutation() {
    let manager = valued_manager(2);
    let inactive = register_block(&manager, 30);
    let hash = inactive.sequence_hash();
    drop(inactive);

    assert_non_mutating_presence_check(&manager, &[hash], true);
}

#[test]
fn registered_presence_finds_a_held_hash_that_request_lookup_hides() {
    let manager = valued_manager(2);
    let inactive = register_block(&manager, 40);
    let hash = inactive.sequence_hash();
    drop(inactive);
    let candidate = manager
        .inactive_candidates(1)
        .into_iter()
        .next()
        .expect("one inactive candidate");
    let hold = manager
        .try_hold_inactive_lineage(candidate)
        .expect("hold the inactive block");

    assert_non_mutating_presence_check(&manager, &[hash], true);
    assert!(
        manager.match_blocks(&[hash]).is_empty(),
        "a held hash is not request-available"
    );
    assert!(
        manager.scan_matches(&[hash], false).is_empty(),
        "a held hash is absent from scan lookup"
    );

    drop(hold);
}
