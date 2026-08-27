// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inactive-residency accounting: every tenure that ends must land in
//! exactly one of the two terminal buckets — evicted or reused.
//!
//! The durations here are real wall-clock, so the assertions are on the
//! *partition* (which bucket, how many blocks) and on ordering relations
//! that a monotonic clock guarantees, never on absolute timings.

use std::thread::sleep;
use std::time::Duration;

use super::*;

/// Long enough to clear monotonic-clock granularity on every platform we
/// build for, short enough not to slow the suite.
const DWELL: Duration = Duration::from_millis(5);

fn register_one(manager: &BlockManager<TestBlockData>, token_start: u32) -> SequenceHash {
    let token_block = create_iota_token_block(token_start, 4);
    let mutable = manager
        .allocate_blocks(1)
        .expect("allocate one block")
        .pop()
        .expect("one mutable block");
    let immutable = manager.register_block(mutable.complete(&token_block).expect("complete block"));
    let seq_hash = immutable.sequence_hash();
    drop(immutable); // → inactive; the tenure starts here
    seq_hash
}

/// A tenure that ends in a cache hit is charged to the reused bucket only,
/// and its duration is at least the time the block actually dwelled.
#[test]
fn a_cache_hit_settles_into_the_reused_bucket() {
    let manager = create_test_manager(4);
    let seq_hash = register_one(&manager, 100);

    sleep(DWELL);
    let matched = manager.match_blocks(&[seq_hash]);
    assert_eq!(matched.len(), 1, "the inactive block must be re-matched");

    let snap = manager.metrics().snapshot();
    assert_eq!(snap.inactive_residency_reused_blocks, 1);
    assert_eq!(
        snap.inactive_residency_evicted_blocks, 0,
        "a hit must not be charged as an eviction"
    );
    assert!(
        snap.inactive_residency_reused_nanos >= DWELL.as_nanos() as u64,
        "reused residency {} ns is below the {:?} the block dwelled",
        snap.inactive_residency_reused_nanos,
        DWELL
    );
    assert!(
        snap.mean_reuse_age().is_some(),
        "one settled hit defines the reuse age"
    );
    assert!(
        snap.mean_eviction_age().is_none(),
        "the eviction age stays undefined until something is evicted"
    );
    assert_eq!(
        snap.wasted_residency_fraction(),
        Some(0.0),
        "with no evictions, none of the settled residency is wasted"
    );
}

/// A tenure that ends under allocation pressure is charged to the evicted
/// bucket only, and the mean is the pool's eviction age.
#[test]
fn an_eviction_settles_into_the_evicted_bucket() {
    // One block: registering a second forces the first out of inactive.
    let manager = create_test_manager(1);
    register_one(&manager, 200);

    sleep(DWELL);
    // Allocating the only block evicts the resident inactive one.
    let _forced = manager.allocate_blocks(1).expect("allocation evicts");

    let snap = manager.metrics().snapshot();
    assert_eq!(snap.inactive_residency_evicted_blocks, 1);
    assert_eq!(
        snap.inactive_residency_reused_blocks, 0,
        "an eviction must not be charged as a hit"
    );
    let age = snap.mean_eviction_age().expect("one eviction has settled");
    assert!(
        age >= DWELL,
        "eviction age {age:?} is below the {DWELL:?} the block dwelled"
    );
    assert!(snap.mean_reuse_age().is_none());
    assert_eq!(
        snap.wasted_residency_fraction(),
        Some(1.0),
        "with no hits, all settled residency is wasted"
    );
}

/// Every tenure that ends lands in exactly one bucket: across a mixed
/// workload the settled block counts sum to the tenures that actually ended.
#[test]
fn the_two_buckets_partition_every_ended_tenure() {
    let manager = create_test_manager(4);

    // Four tenures open, then two of them close by reuse.
    let hashes: Vec<SequenceHash> = (0..4).map(|i| register_one(&manager, i * 100)).collect();
    let reused = manager.match_blocks(&hashes[..2]);
    assert_eq!(reused.len(), 2);
    drop(reused); // → two fresh tenures open

    let snap = manager.metrics().snapshot();
    assert_eq!(snap.inactive_residency_reused_blocks, 2);
    assert_eq!(snap.inactive_residency_evicted_blocks, 0);

    // Drain the pool: every open tenure closes as an eviction. Two of the
    // four blocks are on their second tenure, so six tenures have existed
    // and all four still-open ones settle here.
    manager
        .reset_inactive_pool()
        .expect("drain the inactive pool");

    let snap = manager.metrics().snapshot();
    assert_eq!(snap.inactive_residency_evicted_blocks, 4);
    assert_eq!(
        snap.inactive_residency_reused_blocks + snap.inactive_residency_evicted_blocks,
        6,
        "every tenure that ended is charged exactly once"
    );
    assert_eq!(snap.inactive_pool_size, 0);
}

/// The stamp is per-tenure, not per-slot: a block that is reused and then
/// released again starts a *fresh* residency rather than carrying the first
/// tenure's start forward.
#[test]
fn a_resurrection_restarts_the_residency_clock() {
    let manager = create_test_manager(1);
    let seq_hash = register_one(&manager, 300);

    sleep(DWELL);
    let matched = manager.match_blocks(&[seq_hash]);
    assert_eq!(matched.len(), 1);
    let first_tenure = manager.metrics().snapshot().inactive_residency_reused_nanos;

    // Second tenure opens on drop and is evicted almost immediately.
    drop(matched);
    let _forced = manager.allocate_blocks(1).expect("allocation evicts");

    let snap = manager.metrics().snapshot();
    assert_eq!(snap.inactive_residency_evicted_blocks, 1);
    assert!(
        snap.inactive_residency_evicted_nanos < first_tenure,
        "second tenure ({} ns) must be timed from the resurrection, not from \
         the first tenure's start ({first_tenure} ns)",
        snap.inactive_residency_evicted_nanos
    );
}
