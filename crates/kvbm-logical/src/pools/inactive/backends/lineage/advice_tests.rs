// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only inactive candidate / feature-snapshot API on the lineage backend
//! (R7a §5): point [`advice`](InactiveIndex::advice) and bounded
//! [`peek_victims`](InactiveIndex::peek_victims), across **both** leaf policies
//! — `Valued` (the G1 construction) and `Tick` (today's G2 construction).
//!
//! The load-bearing properties under test:
//!
//! * membership + poison reporting, with the frequency sketch left untouched;
//! * poisoned leaves first, and — the dedup rule — never listed twice, since a
//!   poisoned leaf inhabits both `poison_dense` and `leaf_dense`;
//! * `Tick`'s peek is the exact eviction order;
//! * a peek is a *pure function of policy state*: repeatable, and it does not
//!   move the next real victim (no RNG consumption, no clock stamp);
//! * interior nodes are advice-visible but never eviction candidates, and a
//!   ghost placeholder is neither.

use std::collections::HashSet;
use std::sync::Arc;

use super::{LeafPolicy, LineageBackend, ScorerParams};
use crate::BlockId;
use crate::blocks::SequenceHash;
use crate::pools::InactiveIndex;
use crate::testing::BlockSequenceBuilder;
use crate::tinylfu::{FrequencyTracker, TinyLFUTracker};

/// A `count`-block single-owner chain, as `(block_id, seq_hash)` pairs.
fn chain(count: usize, offset: u32) -> Vec<(BlockId, SequenceHash)> {
    let tokens: Vec<u32> = (offset..offset + count as u32).collect();
    BlockSequenceBuilder::from_tokens(tokens)
        .with_block_size(1)
        .build()
}

/// `count` independent single-block root lineages, with **distinct** block ids.
///
/// `BlockSequenceBuilder` numbers every sequence from 0, so separately-built
/// roots would otherwise all carry `block_id 0` and any identity assertion over
/// them would be vacuous. The ids are assigned here instead.
fn independent_roots(count: usize) -> Vec<(BlockId, SequenceHash)> {
    (0..count)
        .map(|i| {
            let (_, hash) = BlockSequenceBuilder::from_tokens(vec![1000 + i as u32])
                .with_block_size(1)
                .build()
                .into_iter()
                .next()
                .expect("one root block per token");
            (i as BlockId, hash)
        })
        .collect()
}

/// Lineage backend on the valued leaf policy. `k_sample` is explicit because
/// the determinism test needs `K < resident leaves` to exercise the sampling
/// RNG at all.
fn valued_backend(
    sketch: Option<Arc<dyn FrequencyTracker<u128>>>,
    k_sample: usize,
) -> LineageBackend {
    valued_backend_seeded(sketch, k_sample, 0x51)
}

fn valued_backend_seeded(
    sketch: Option<Arc<dyn FrequencyTracker<u128>>>,
    k_sample: usize,
    seed: u64,
) -> LineageBackend {
    let params = ScorerParams {
        gamma: 0.6,
        n: 2,
        k_sample,
        t_blocks: None,
        seed,
    };
    LineageBackend::with_policy(0, LeafPolicy::valued(0, sketch, None, params))
}

/// `(hash, block_id)` identity of each peeked candidate.
fn identities(
    peeked: &[(SequenceHash, BlockId, crate::pools::InactiveFeatures)],
) -> Vec<(u128, BlockId)> {
    peeked
        .iter()
        .map(|(hash, id, _)| (hash.as_u128(), *id))
        .collect()
}

// ---------------------------------------------------------------------------
// §5.1 — point advice: poison, membership, and a untouched sketch
// ---------------------------------------------------------------------------

/// `advice` reports the poison bit and leaf/interior status for resident nodes,
/// `None` for absent or evicted ones, and reads the sketch without touching it.
#[test]
fn advice_reports_poison_and_membership_without_touching_the_sketch() {
    let sketch = Arc::new(TinyLFUTracker::<u128>::new(1 << 12));
    let mut backend = valued_backend(Some(sketch.clone()), 16);

    let a = chain(3, 0); // a0 → a1 → a2, single-owner
    let b = chain(2, 5000); // independent chain, ids offset to stay distinct
    for (id, hash) in &a {
        backend.insert(*hash, *id);
    }
    for (id, hash) in &b {
        backend.insert(*hash, *id + 10);
    }

    // Four genuine accesses on the poisoned leaf, none on anything else.
    for _ in 0..4 {
        sketch.touch(a[2].1.as_u128());
    }
    let watched: Vec<SequenceHash> = a.iter().chain(b.iter()).map(|(_, hash)| *hash).collect();
    let before: Vec<u32> = watched.iter().map(|h| sketch.count(h.as_u128())).collect();

    // Poison the whole single-owner suffix (leaf → root; nothing branches).
    backend.poison(a[2].1);

    let leaf = backend.advice(a[2].1).expect("a resident leaf has advice");
    assert!(
        leaf.poisoned,
        "the compaction-poisoned leaf reports poisoned"
    );
    assert!(leaf.is_leaf);
    assert_eq!(
        leaf.freq_estimate,
        Some(4),
        "the sketch estimate is read through, not synthesized"
    );
    assert_eq!(leaf.max_fanout, None, "no oracle attached");
    assert!(leaf.age_ticks.is_some(), "valued tracks a recency baseline");
    assert_eq!(
        leaf.evict_rank, None,
        "rank is only defined within one peek batch"
    );

    let interior = backend
        .advice(a[1].1)
        .expect("interior nodes are advice-visible");
    assert!(!interior.is_leaf, "a node with a child is not a leaf");
    assert!(
        interior.poisoned,
        "the suffix walk poisoned the interior too"
    );

    let unrelated = backend
        .advice(b[1].1)
        .expect("the sibling chain is resident");
    assert!(
        !unrelated.poisoned,
        "an unpoisoned lineage stays unpoisoned"
    );
    assert!(
        unrelated.freq_estimate.is_some(),
        "a tracked pool reports Some(0), not None, for an untouched block"
    );

    // Absent: never inserted.
    let absent = chain(1, 9000)[0].1;
    assert!(
        backend.advice(absent).is_none(),
        "an absent hash has no advice"
    );

    // Evicted: resurrected out of the index.
    assert!(
        backend.take(b[1].1, b[1].0 + 10),
        "resurrect the sibling leaf"
    );
    assert!(
        backend.advice(b[1].1).is_none(),
        "an evicted hash is no longer resident-inactive"
    );

    // Neither advice nor peek may act as an access.
    let _ = backend.peek_victims(8);
    for (hash, expected) in watched.iter().zip(before) {
        assert_eq!(
            sketch.count(hash.as_u128()),
            expected,
            "the read-only API touched the frequency sketch"
        );
    }
}

// ---------------------------------------------------------------------------
// §5.2 — peek ordering: poison first, exactly once, and Tick == eviction order
// ---------------------------------------------------------------------------

/// Poisoned leaves head the peek, and each appears exactly once.
///
/// The two poisoned leaves are also the two *oldest* — hence the two
/// lowest-scoring — so a scan phase that failed to skip `poison_idx.is_some()`
/// slots would re-pick precisely them and duplicate them. Choosing any other
/// pair would make the dedup check score-dependent and let it pass by luck.
#[test]
fn peek_lists_poisoned_leaves_first_and_never_twice() {
    let mut backend = valued_backend(None, 16);
    let roots = independent_roots(5); // roots[0] inserted first ⇒ oldest
    for (id, hash) in &roots {
        backend.insert(*hash, *id);
    }
    backend.poison(roots[0].1);
    backend.poison(roots[1].1);

    let peeked = backend.peek_victims(5);
    assert_eq!(peeked.len(), 5, "one entry per resident leaf");
    let distinct: HashSet<(u128, BlockId)> = identities(&peeked).into_iter().collect();
    assert_eq!(
        distinct.len(),
        5,
        "a poisoned leaf lives in both dense sets and must still be listed once"
    );

    let head: HashSet<u128> = peeked[..2].iter().map(|(h, _, _)| h.as_u128()).collect();
    assert_eq!(
        head,
        HashSet::from([roots[0].1.as_u128(), roots[1].1.as_u128()]),
        "the poisoned leaves come first"
    );
    assert!(peeked[..2].iter().all(|(_, _, f)| f.poisoned));
    assert!(peeked[2..].iter().all(|(_, _, f)| !f.poisoned));

    // Rank spans the byte range across the returned slice and never decreases.
    let ranks: Vec<u8> = peeked
        .iter()
        .map(|(_, _, f)| f.evict_rank.expect("valued exposes a peek-relative rank"))
        .collect();
    assert_eq!(ranks[0], 0, "rank 0 is the next victim");
    assert_eq!(*ranks.last().unwrap(), 255);
    assert!(ranks.windows(2).all(|w| w[0] <= w[1]));

    // A capped peek keeps the poison prefix rather than dropping it.
    let capped = backend.peek_victims(3);
    assert_eq!(capped.len(), 3);
    assert!(capped[..2].iter().all(|(_, _, f)| f.poisoned));

    // Cap below the poison count: the scan phase is skipped entirely and the
    // poison prefix is truncated, never displaced by a scored leaf.
    let single = backend.peek_victims(1);
    assert_eq!(single.len(), 1);
    assert!(
        single[0].2.poisoned,
        "the cap must not evict poison from the head"
    );

    // "Give me everything" is a plausible pressure-pass idiom and must not turn
    // into an overflowing (debug-panic) or multi-gigabyte reservation.
    let unbounded = backend.peek_victims(usize::MAX);
    assert_eq!(unbounded.len(), 5, "an unbounded request yields every leaf");
}

/// The opt-in `Fifo` leaf policy peeks its exact order too. Beyond R7a §3.2
/// (which names only `Tick` and `Valued`), but the head-first walk is exact and
/// cheap, and a supported backend silently reporting "no candidates" would
/// degrade the consumer contract without sanction. `Fifo` carries ordering but
/// no age, sketch, oracle, or poison — so those features come back absent.
#[test]
fn fifo_peek_matches_the_eviction_order() {
    let mut backend = LineageBackend::with_policy(0, LeafPolicy::fifo(0));
    let roots = independent_roots(4);
    for (id, hash) in &roots {
        backend.insert(*hash, *id);
    }

    let peeked = backend.peek_victims(4);
    assert_eq!(peeked.len(), 4);
    let features = peeked[0].2;
    assert!(features.is_leaf);
    assert!(!features.poisoned, "Fifo tracks no poison");
    assert_eq!(
        features.age_ticks, None,
        "Fifo has ordering state but no age"
    );
    assert_eq!(features.freq_estimate, None);
    assert_eq!(features.max_fanout, None);
    assert_eq!(features.evict_rank, Some(0), "Fifo is a total order");
    assert!(
        backend.advice(roots[2].1).is_some(),
        "a resident Fifo leaf is still advice-visible"
    );

    let ids = identities(&peeked);
    let drained: Vec<(u128, BlockId)> = backend
        .allocate_all()
        .into_iter()
        .map(|(hash, id)| (hash.as_u128(), id))
        .collect();
    assert_eq!(ids, drained, "Fifo's peek IS the eviction order");
}

/// `Tick` peeks the exact eviction order: the peek equals the prefix of a full
/// drain.
///
/// Independent roots, deliberately: draining a *chain* re-leafs the parent,
/// which returns at its own older tick and jumps ahead of the leaves peeked
/// behind it — the peek describes the current leaf set, not a replay of the
/// whole drain. The second half pins that weaker (still exact) guarantee: the
/// head of the peek is always the next victim.
#[test]
fn tick_peek_matches_the_eviction_order() {
    let mut backend = LineageBackend::with_capacity(0); // Tick is the default
    let roots = independent_roots(4);
    for (id, hash) in &roots {
        backend.insert(*hash, *id);
    }

    let peeked = identities(&backend.peek_victims(4));
    let drained: Vec<(u128, BlockId)> = backend
        .allocate_all()
        .into_iter()
        .map(|(hash, id)| (hash.as_u128(), id))
        .collect();
    assert_eq!(peeked.len(), 4);
    assert_eq!(peeked, drained, "Tick's peek IS the eviction order");

    // Branching graph: the peek head is still exactly the next real victim.
    let mut branched = LineageBackend::with_capacity(0);
    let long = chain(3, 0);
    let short = chain(1, 7000);
    for (id, hash) in &long {
        branched.insert(*hash, *id);
    }
    branched.insert(short[0].1, short[0].0 + 10);
    let head = branched.peek_victims(2);
    let victim = branched.allocate(1);
    assert_eq!(
        (head[0].0, head[0].1),
        victim[0],
        "the head of the peek is the next victim"
    );
}

// ---------------------------------------------------------------------------
// §5.3 / §3.4 — determinism and no observer effect
// ---------------------------------------------------------------------------

/// A peek is a pure function of policy state, and leaves the *real* victim
/// sequence identical to a pool that was never peeked.
///
/// Two separate failure modes are covered, because neither catches the other:
///
/// * comparing the two peeks **including their features** catches a `now`
///   stamp — that shifts every `age_ticks` while leaving the *order* intact
///   (the score is monotone in age), so an order-only comparison would pass;
/// * comparing against an unpeeked control pool catches RNG consumption. It
///   needs `K < resident leaves`, otherwise `next_victim` takes its exact-scan
///   branch, never draws, and the assertion is vacuous.
#[test]
fn peek_is_deterministic_and_does_not_perturb_the_next_victim() {
    let build_seeded = |seed: u64| {
        let mut backend = valued_backend_seeded(None, 4, seed); // K = 4 << 40 leaves
        for (id, hash) in independent_roots(40) {
            backend.insert(hash, id);
        }
        backend
    };
    let build = || build_seeded(0x51);
    let mut peeked_pool = build();
    let mut control = build();

    let first = peeked_pool.peek_victims(8);
    let second = peeked_pool.peek_victims(8);
    assert_eq!(first.len(), 8);
    assert!(
        first[0].2.age_ticks.is_some(),
        "the age field must be populated for the comparison below to bite"
    );
    assert_eq!(
        first, second,
        "repeated peeks must agree — features included, so a clock stamp shows up"
    );
    assert_eq!(peeked_pool.len(), 40, "a peek evicts nothing");

    let mut victims = Vec::new();
    for step in 0..6 {
        let peeked_victim = peeked_pool.allocate(1);
        assert_eq!(
            peeked_victim,
            control.allocate(1),
            "peeking changed the real victim at step {step}"
        );
        victims.extend(peeked_victim);
    }

    // Non-vacuity for the comparison above: with K < leaves the victim sequence
    // really is a function of the RNG stream, so a peek that consumed a draw
    // would have shifted it. A differently-seeded pool over the identical leaf
    // set diverges.
    let mut reseeded = build_seeded(0xA7F1);
    let other: Vec<_> = (0..6).flat_map(|_| reseeded.allocate(1)).collect();
    assert_ne!(
        victims, other,
        "the sampled victim sequence must depend on the RNG state"
    );
}

// ---------------------------------------------------------------------------
// §5.4 — interior nodes (and ghosts)
// ---------------------------------------------------------------------------

/// Interior nodes are visible through `advice` with `is_leaf: false` but are
/// never eviction candidates; a ghost placeholder is neither.
#[test]
fn interior_nodes_are_advice_visible_but_never_peeked() {
    let mut backend = valued_backend(None, 16);
    let c = chain(3, 0); // c0 → c1 → c2
    for (id, hash) in &c {
        backend.insert(*hash, *id);
    }

    let peeked = backend.peek_victims(10);
    assert_eq!(peeked.len(), 1, "only the leaf is an eviction candidate");
    assert_eq!(peeked[0].0, c[2].1);
    assert!(peeked[0].2.is_leaf);

    for (label, hash) in [("root", c[0].1), ("interior", c[1].1)] {
        let features = backend
            .advice(hash)
            .unwrap_or_else(|| panic!("{label} node is advice-visible"));
        assert!(!features.is_leaf, "{label} node is not a leaf");
        assert_eq!(features.evict_rank, None, "{label} node has no peek rank");
    }

    // Resurrect the interior block: it degrades to a Ghost placeholder, which
    // stores no hash and therefore has no advice.
    assert!(backend.take(c[1].1, c[1].0), "c1 resurrected out");
    assert!(
        backend.advice(c[1].1).is_none(),
        "a ghost placeholder is not resident-inactive"
    );
    assert!(
        backend.advice(c[0].1).is_some(),
        "the real ancestor above the ghost is still advice-visible"
    );
    assert_eq!(
        backend.peek_victims(10).len(),
        1,
        "the leaf is still the only candidate"
    );
}

// ---------------------------------------------------------------------------
// §5.6 — G2 shape: lineage + Tick, no sketch and no oracle
// ---------------------------------------------------------------------------

/// The G2 construction (lineage backend, `Tick` leaf policy, no frequency
/// sketch and no branch oracle) still populates features: the two
/// tracker-derived fields are `None`, everything else is real.
#[test]
fn tick_lineage_without_sketch_or_oracle_still_populates_features() {
    let mut backend = LineageBackend::with_capacity(8);
    let roots = independent_roots(3);
    for (id, hash) in &roots {
        backend.insert(*hash, *id);
    }

    let peeked = backend.peek_victims(3);
    assert_eq!(peeked.len(), 3);
    for (_, _, features) in &peeked {
        assert!(!features.poisoned, "Tick tracks no poison");
        assert!(features.is_leaf);
        assert!(
            features.age_ticks.is_some(),
            "Tick exposes a pool-logical age"
        );
        assert_eq!(features.freq_estimate, None, "no sketch attached");
        assert_eq!(features.max_fanout, None, "no oracle attached");
        assert!(features.evict_rank.is_some(), "Tick is a total order");
    }
    assert!(
        peeked[0].2.age_ticks >= peeked[2].2.age_ticks,
        "the head of the eviction order is the older block"
    );

    let point = backend.advice(roots[0].1).expect("resident root");
    assert!(point.is_leaf);
    assert!(point.age_ticks.is_some());
    assert_eq!(point.freq_estimate, None);
    assert_eq!(point.evict_rank, None);
}
