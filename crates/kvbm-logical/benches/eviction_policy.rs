// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Criterion microbenches for the valued leaf-eviction policy (EV-PR3).
//!
//! These drive the policy through the **public** [`BlockManager`] API (the
//! inactive backends themselves are `pub(crate)`), mirroring
//! `block_manager.rs`. Three groups:
//!
//! - **Per-event cost** of the valued backend: register (insert into inactive
//!   on drop), match (resurrect an inactive block), and allocate-one (force one
//!   eviction = one `next_victim` + graph remove).
//! - **Flat-with-n**: `allocate(1)` on the valued policy vs `LruBackend` at
//!   1k / 10k / 100k resident *leaves*. The valued policy's sampled-min
//!   `next_victim` is O(`k_sample`) — expected flat as the resident leaf count
//!   grows — versus LRU's O(1) `pop_lru`. (Benches don't assert; the shape of
//!   the curve is the result.)
//! - **Deep-chain eviction**: draining a 62.5k-deep single lineage chain via
//!   `allocate(len)`, the amortized-linear iterative prune / re-leaf path.
//!
//! Run: `cargo bench -p kvbm-logical --features testing --bench eviction_policy`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use dynamo_tokens::{TokenBlock, TokenBlockSequence};
use kvbm_logical::manager::ScorerParams;
use kvbm_logical::testing::{TestMeta, create_test_manager_with_backend};
use kvbm_logical::{BlockManager, KvbmSequenceHashProvider, SequenceHash};

/// `create_test_manager_with_backend` builds 4-token blocks.
const BLOCK_SIZE: u32 = 4;
const SALT: u64 = 1337;
/// Resident-leaf counts for the flat-with-n sweep.
const LEAF_COUNTS: [usize; 3] = [1_000, 10_000, 100_000];

type Configure = fn(
    kvbm_logical::manager::BlockManagerConfigBuilder<TestMeta>,
) -> kvbm_logical::manager::BlockManagerConfigBuilder<TestMeta>;

fn valued_backend() -> Configure {
    |b| b.with_valued_lineage_backend(ScorerParams::default())
}

/// One 1-block `TokenBlock` sequence with content unique to `i` (an independent
/// lineage root → its own evictable leaf).
fn leaf_block(i: usize) -> TokenBlock {
    let base = i as u32 * BLOCK_SIZE;
    let tokens: Vec<u32> = (base..base + BLOCK_SIZE).collect();
    TokenBlockSequence::from_slice(&tokens, BLOCK_SIZE, Some(SALT))
        .blocks()
        .first()
        .expect("one block")
        .clone()
}

/// Build a manager with `n` **independent** single-block leaves resident in the
/// inactive pool. Returns the manager and every leaf hash.
fn populate_leaves(n: usize, configure: Configure) -> (BlockManager<TestMeta>, Vec<SequenceHash>) {
    let manager = create_test_manager_with_backend::<TestMeta>(n, configure);
    let tbs: Vec<TokenBlock> = (0..n).map(leaf_block).collect();
    let hashes: Vec<SequenceHash> = tbs.iter().map(|tb| tb.kvbm_sequence_hash()).collect();
    let mutables = manager.allocate_blocks(n).expect("allocate n");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(tbs.iter())
        .map(|(m, tb)| m.complete(tb).expect("complete"))
        .collect();
    let immutables = manager.register_blocks(completes);
    // Drop the strong handles → all n blocks fall into the inactive pool.
    drop(immutables);
    (manager, hashes)
}

/// Build a manager holding one `depth`-deep single lineage chain resident in
/// the inactive pool (one leaf; its ancestors re-leaf as it drains).
fn populate_chain(depth: usize, configure: Configure) -> BlockManager<TestMeta> {
    let manager = create_test_manager_with_backend::<TestMeta>(depth, configure);
    let tokens: Vec<u32> = (0..depth as u32 * BLOCK_SIZE).collect();
    let tbs: Vec<TokenBlock> = TokenBlockSequence::from_slice(&tokens, BLOCK_SIZE, Some(SALT))
        .blocks()
        .to_vec();
    assert_eq!(tbs.len(), depth);
    let mutables = manager.allocate_blocks(depth).expect("allocate chain");
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(tbs.iter())
        .map(|(m, tb)| m.complete(tb).expect("complete"))
        .collect();
    let immutables = manager.register_blocks(completes);
    drop(immutables);
    manager
}

/// `allocate(1)` (one `next_victim` + graph remove) on the valued policy vs
/// LRU, swept over resident-leaf count. Expected flat with n for both.
fn bench_next_victim_flat(c: &mut Criterion) {
    let backends: &[(&str, Configure)] = &[
        ("valued", valued_backend()),
        ("lru", |b| b.with_lru_backend()),
    ];
    let mut group = c.benchmark_group("allocate_one_flat_with_n");
    for &(name, configure) in backends {
        for &n in &LEAF_COUNTS {
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, &n| {
                b.iter_batched(
                    || populate_leaves(n, configure),
                    |(manager, _hashes)| {
                        // One eviction from the inactive pool (reset pool empty).
                        black_box(manager.allocate_blocks(1))
                    },
                    criterion::BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

/// Per-event costs on the valued backend at a fixed 1k resident leaves.
fn bench_valued_per_event(c: &mut Criterion) {
    const N: usize = 1_000;
    let mut group = c.benchmark_group("valued_per_event");

    // match-one: resurrect a single inactive leaf; the returned Vec drops at
    // end of iteration, pushing it back — state restored, so `iter` is valid.
    let (manager, hashes) = populate_leaves(N, valued_backend());
    let probe = [hashes[N / 2]];
    group.bench_function("match_one", |b| {
        b.iter(|| black_box(manager.match_blocks(black_box(&probe))));
    });
    drop((manager, hashes));

    // allocate-one: force one eviction (`next_victim` + remove).
    group.bench_function("allocate_one", |b| {
        b.iter_batched(
            || populate_leaves(N, valued_backend()),
            |(manager, _)| black_box(manager.allocate_blocks(1)),
            criterion::BatchSize::LargeInput,
        );
    });

    // insert-one: drop one extra `ImmutableBlock`, whose `release_primary`
    // calls the backend's `insert` (Primary → Inactive).
    group.bench_function("insert_one", |b| {
        b.iter_batched(
            || {
                let (manager, _) = populate_leaves(N, valued_backend());
                let tb = leaf_block(N + 1);
                let mutable = manager
                    .allocate_blocks(1)
                    .expect("spare block")
                    .into_iter()
                    .next()
                    .expect("one");
                let complete = mutable.complete(&tb).expect("complete");
                let immutable = manager.register_block(complete);
                (manager, immutable)
            },
            |(manager, immutable)| {
                drop(black_box(immutable)); // Primary → Inactive: one insert.
                manager
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Drain a 62.5k-deep single lineage chain via `allocate(len)` — the
/// amortized-linear iterative prune / re-leaf path.
fn bench_deep_chain_eviction(c: &mut Criterion) {
    const DEPTH: usize = 62_500;
    let mut group = c.benchmark_group("deep_chain_eviction");
    group.sample_size(10);
    group.bench_function(BenchmarkId::new("valued", DEPTH), |b| {
        b.iter_batched(
            || populate_chain(DEPTH, valued_backend()),
            |manager| black_box(manager.allocate_blocks(DEPTH)),
            criterion::BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_next_victim_flat,
    bench_valued_per_event,
    bench_deep_chain_eviction
);
criterion_main!(benches);
