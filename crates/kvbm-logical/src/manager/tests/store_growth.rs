// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Growth of the live-primary hash index under the store lock.
//!
//! `BlockStore::active_by_hash` holds one entry per live primary. Allocating,
//! matching, registering and releasing all run under the one store mutex, so
//! an insertion that exhausts the map's growth budget rehashes every entry
//! with the lock held. These tests fill a lineage manager to its live
//! capacity, then churn it at capacity, and watch the map's `capacity()`:
//!
//! * a doubling (the map's allocation grows) fails, and
//! * an in-place rebuild — which tombstones can force even when the map has
//!   spare entries — is counted and bounded, because a rebuild also visits
//!   every bucket under the lock.
//!
//! The workload is fixed and seeded: every block stages a fresh one-token
//! block from a counter. Only `capacity()` and `len()` are observed.

use std::collections::VecDeque;

use super::*;
use crate::ImmutableBlock;
use crate::testing::{create_iota_token_block, create_test_manager_with_backend};

/// Live primaries held at once by the churn test.
///
/// At 4096 the map is at half the buckets a one-times reservation gives
/// (4096 entries need 8192 buckets), which is where tombstones make a
/// one-times map grow; a two-times reservation leaves it at a quarter load.
const LIVE: usize = 4096;

/// Blocks registered per batch — the manager's own registration chunk size.
/// Also one quarter of [`LIVE`], so the churn turns over whole batches.
const CHUNK: usize = 1024;

/// Turnovers of the churn phase. Each drops a quarter of the held guards,
/// reallocates that quarter, and registers and holds a quarter of fresh
/// hashes, so the index returns to [`LIVE`] entries every step.
const TURNOUTS: usize = 100;

/// In-place rebuilds this fixed workload is allowed, on aarch64 (16-byte
/// probe groups). Taken from the recorded run in the PR.
const REHASH_BOUND_AARCH64: usize = 0;
/// In-place rebuilds this fixed workload is allowed, on x86_64 (SSE2,
/// 16-byte probe groups). Taken from the recorded CI run in the PR.
const REHASH_BOUND_X86_64: usize = 0;

/// The recorded rebuild count for this host's probe-group width.
fn rehash_bound() -> usize {
    if cfg!(target_arch = "aarch64") {
        REHASH_BOUND_AARCH64
    } else {
        REHASH_BOUND_X86_64
    }
}

/// A lineage manager whose maximum is far above its live capacity — the way
/// a G1 manager is built and then grown with `set_capacity`.
fn lineage_manager(block_count: usize, maximum: usize) -> BlockManager<TestBlockData> {
    create_test_manager_with_backend(block_count, |builder| {
        builder
            .block_size(1)
            .maximum_block_count(maximum)
            .with_valued_lineage_backend(ScorerParams::default())
    })
}

/// What the live-primary index did over one fill and churn.
#[derive(Debug, Default)]
struct ChurnStats {
    /// Index `capacity()` right after the store was built.
    after_build: usize,
    /// Index `capacity()` right after the live fill.
    after_fill: usize,
    /// Highest index `capacity()` seen.
    peak_capacity: usize,
    /// Highest live-primary count seen.
    peak_live: usize,
    /// Highest `inactive_len()` seen.
    peak_inactive: usize,
    /// Steps that doubled the index past its build reservation.
    growths: usize,
    /// Steps where the index rebuilt in place: `capacity()` rose by more
    /// than that step's removals, which a per-removal group reset cannot do.
    rehashes: usize,
}

/// Allocate `count` blocks in [`CHUNK`] batches, stage every one with a
/// fresh one-token block, register the batch, and hold every guard.
fn hold_fresh(
    manager: &BlockManager<TestBlockData>,
    held: &mut VecDeque<Vec<ImmutableBlock<TestBlockData>>>,
    next_token: &mut u32,
    count: usize,
) {
    assert_eq!(count % CHUNK, 0, "allocate and register whole chunks");
    for _ in 0..count / CHUNK {
        let mutables = manager
            .allocate_blocks(CHUNK)
            .expect("a quarter of the store is free or evictable");
        let completes = mutables
            .into_iter()
            .map(|mutable| {
                let token_block = create_iota_token_block(*next_token, 1);
                *next_token += 1;
                mutable
                    .complete(&token_block)
                    .expect("meta block matches the store block size")
            })
            .collect::<Vec<_>>();
        let guards = manager.register_blocks(completes);
        assert_eq!(guards.len(), CHUNK, "one guard per held block");
        held.push_back(guards);
    }
}

/// Fill to `live` held primaries, then turn a quarter of them over
/// `turnouts` times, reading the live-primary index after every step.
fn fill_and_churn(
    manager: &BlockManager<TestBlockData>,
    live: usize,
    turnouts: usize,
) -> ChurnStats {
    assert_eq!(live % (4 * CHUNK), 0, "a quarter is whole chunks");
    assert!(turnouts > 0, "the churn runs at least one step");
    let quarter = live / 4;
    let store = manager.store_for_test();
    let after_build = store.active_by_hash_capacity();
    let mut stats = ChurnStats {
        after_build,
        after_fill: after_build,
        peak_capacity: after_build,
        peak_inactive: manager.inactive_len(),
        ..ChurnStats::default()
    };
    let mut held: VecDeque<Vec<ImmutableBlock<TestBlockData>>> = VecDeque::new();
    let mut next_token = 0u32;

    hold_fresh(manager, &mut held, &mut next_token, live);
    stats.after_fill = store.active_by_hash_capacity();
    stats.peak_capacity = stats.after_fill;
    stats.peak_live = store.active_by_hash_len();
    stats.peak_inactive = stats.peak_inactive.max(manager.inactive_len());
    assert_eq!(
        stats.peak_live, live,
        "the index holds one entry per live primary"
    );

    for _ in 0..turnouts {
        let before = store.active_by_hash_capacity();
        // Drop the oldest quarter: each release moves a block to inactive.
        for _ in 0..quarter / CHUNK {
            drop(held.pop_front());
        }
        assert_eq!(store.active_by_hash_len(), live - quarter);
        stats.peak_inactive = stats.peak_inactive.max(manager.inactive_len());
        // Allocate that quarter (it evicts the blocks just released), stage
        // fresh hashes, register and hold.
        hold_fresh(manager, &mut held, &mut next_token, quarter);
        assert_eq!(store.active_by_hash_len(), live);
        stats.peak_inactive = stats.peak_inactive.max(manager.inactive_len());
        let after = store.active_by_hash_capacity();
        if after > stats.after_build {
            stats.growths += 1;
        } else if after.saturating_sub(before) > quarter {
            stats.rehashes += 1;
        }
        stats.peak_capacity = stats.peak_capacity.max(after);
    }
    stats
}

fn report(label: &str, stats: &ChurnStats) {
    println!(
        "{label}: peak live primaries {}, peak inactive {}, capacity after build {}, \
         after fill {}, peak {}, growths {}, in-place rebuilds {}",
        stats.peak_live,
        stats.peak_inactive,
        stats.after_build,
        stats.after_fill,
        stats.peak_capacity,
        stats.growths,
        stats.rehashes,
    );
}

#[test]
fn active_by_hash_does_not_grow_while_live_primaries_fill_and_churn() {
    let manager = lineage_manager(LIVE, 64 * LIVE);
    let stats = fill_and_churn(&manager, LIVE, TURNOUTS);
    report("live churn", &stats);

    assert_eq!(
        stats.after_fill, stats.after_build,
        "the live fill must not grow the index: it was built at {}, ended the fill at {}",
        stats.after_build, stats.after_fill,
    );
    assert_eq!(
        stats.growths, 0,
        "the churn must not grow the index: capacity peaked at {} against a build reservation of {}",
        stats.peak_capacity, stats.after_build,
    );
    assert!(
        stats.rehashes <= rehash_bound(),
        "in-place rebuilds {} exceed the recorded {} for this workload",
        stats.rehashes,
        rehash_bound(),
    );
}

#[test]
fn set_capacity_reserves_active_by_hash() {
    let manager = lineage_manager(LIVE, 64 * LIVE);
    let store = manager.store_for_test();
    let built = store.active_by_hash_capacity();
    assert!(
        built >= 2 * LIVE,
        "the index is built to hold the live capacity at half load: {built} against {LIVE} blocks",
    );
    assert!(
        built < 8 * LIVE,
        "the index must be sized from the live capacity, not the maximum: {built}",
    );

    manager.set_capacity(4 * LIVE).expect("grow in place");
    let grown = store.active_by_hash_capacity();
    assert!(
        grown >= 2 * 4 * LIVE,
        "set_capacity reserves four-times room at half load: {grown}",
    );

    // A shrink keeps the reserve; growing back does not rebuild it.
    manager.set_capacity(LIVE).expect("shrink to the live size");
    assert_eq!(store.active_by_hash_capacity(), grown);
    manager.set_capacity(4 * LIVE).expect("grow back");
    assert_eq!(store.active_by_hash_capacity(), grown);

    let stats = fill_and_churn(&manager, 4 * LIVE, TURNOUTS);
    report("reserved 4x", &stats);
    assert_eq!(
        stats.after_fill, grown,
        "the fill uses the reserve instead of growing it",
    );
    assert_eq!(
        stats.growths, 0,
        "the churn uses the reserve instead of growing it: capacity peaked at {} against a reservation of {}",
        stats.peak_capacity, grown,
    );
}
