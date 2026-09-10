// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[rstest]
#[case(false)]
#[case(true)]
fn eager_lookup_preserves_the_retention_boundary(#[case] temporary: bool) {
    let manager = create_test_manager(4);
    let token = create_test_token_block_from_iota(71_000);
    let hash = token.kvbm_sequence_hash();
    let block = manager.allocate_blocks(1).unwrap().pop().unwrap();
    let mut complete = block.complete(&token).unwrap();
    complete.set_evict_on_reset(temporary);
    let immutable = manager.register_block(complete);
    let block_id = immutable.block_id();
    let store = manager.store_for_test();

    let gate = store.pause_release_primary();
    let arrivals = store.release_primary_arrivals();
    let dropping = std::thread::spawn(move || drop(immutable));
    while store.release_primary_arrivals() == arrivals {
        std::thread::yield_now();
    }

    // The original drop pauses before the store lock. A real lookup must
    // perform the eager transition and resurrection under that same lock.
    let before_lookup = manager.match_inactive_blocks(&[hash]);
    let lookup = manager.match_blocks(&[hash]);
    let during_lookup = manager.match_inactive_blocks(&[hash]);
    let eager_transitions = manager.metrics().snapshot().eager_primary_to_inactive_total;
    let inactive_during_lookup = manager.inactive_len();

    // Release the gate before assertions, so a failed assertion cannot
    // deadlock when the lookup pin drops during unwinding.
    drop(gate);
    dropping.join().unwrap();
    assert!(before_lookup.is_empty());
    assert_eq!(lookup.len(), 1);
    assert_eq!(lookup[0].block_id(), block_id);
    assert!(during_lookup.is_empty());
    assert_eq!(eager_transitions, 1);
    assert_eq!(inactive_during_lookup, 0);
    assert_eq!(manager.metrics().snapshot().release_primary_noop_total, 1);
    assert!(manager.match_inactive_blocks(&[hash]).is_empty());

    drop(lookup);
    let retained = manager.match_inactive_blocks(&[hash]);
    if temporary {
        assert!(retained.is_empty());
        assert_eq!(manager.reset_len(), 4);
        assert!(!manager.has_any_registered_hashes(&[hash]));
    } else {
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].block_id(), block_id);
        drop(retained);
        assert_eq!(manager.inactive_len(), 1);
        assert_eq!(manager.reset_len(), 3);
    }
    println!(
        "EAGER_RETENTION temporary={temporary} eager_transitions=1 exposed_inactive=0 release_noops=1"
    );
}
