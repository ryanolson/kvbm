// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic agentic-coding trace harness + policy-comparison tests for the
//! valued leaf-eviction policy (EV-PR3).
//!
//! # What this exercises
//!
//! A [`Harness`] owns a `Box<dyn InactiveIndex>` plus the shared
//! [`TinyLFUTracker`] (sketch) and [`BranchPointTracker`] (oracle) `Arc`s, and
//! replays a trace of block-cache operations, threading the sketch/oracle
//! *exactly as the block registry would* so the valued policy sees the same
//! signals it would in production — without going through `BlockManager`.
//!
//! The tests compare the valued lineage backend against the `LruBackend`
//! (recency proxy) and the 4-tier `MultiLruBackend` (frequency / naive-LFU
//! proxy) on synthetic agentic-coding workloads: shared system prompts,
//! sub-agent forks, dormant multi-turn sessions, and compaction of completed
//! sessions.
//!
//! # Modelling conventions (identical across all backends, for fairness)
//!
//! * **Bounded cache.** Every backend is treated as a cache of `capacity`
//!   blocks. Admission is evict-before-insert: if the resident count is at
//!   capacity we evict one policy victim first, then insert. This both keeps
//!   the comparison fair and satisfies `LruBackend::insert`'s
//!   `len < cap` assertion. Different policies pick different victims — that
//!   divergence in the *retained* set is exactly what these tests measure.
//! * **Reuse = match-then-re-admit.** A prefill request matches the longest
//!   cached prefix (`find_matches`, which *removes* the hits, pulling them
//!   "active"), counts the hit rate, then re-admits every requested block —
//!   the reused hits plus the freshly-computed suffix. Each admitted block
//!   touches the sketch once (a genuine access), matching the registry's
//!   touch-on-match / touch-on-register behaviour. Fresh (never-resident)
//!   blocks additionally fire `oracle.on_block_registered`; reused blocks do
//!   not (they never left the registry). Evictions fire
//!   `oracle.on_block_removed`.
//! * **Compaction = `poison` only.** A completed session's single-owner tail
//!   is poisoned (`backend.poison`); the valued policy then drains it
//!   evict-first under later pressure, while `LruBackend`/`MultiLruBackend`
//!   ignore the hint and keep the (recently-touched / frequency-hot) dead
//!   tail until ordinary pressure ages it out. We deliberately do **not**
//!   also drop the poisoned tail uniformly from every backend: doing so would
//!   make the `poison`→no-op mutation change nothing, defeating the
//!   compaction kill-mutation guard ([`compaction_poison_is_load_bearing`]).
//!   The measured gap is the *timing* of dead-tail shedding, which is the
//!   real production effect.

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use super::{LeafPolicy, LineageBackend, ScorerParams};
use crate::BlockId;
use crate::blocks::SequenceHash;
use crate::branch_tracker::{BranchOracle, BranchPointTracker};
use crate::pools::InactiveIndex;
use crate::pools::backends::{LruBackend, MultiLruBackend};
use crate::testing::BlockSequenceBuilder;
use crate::tinylfu::{FrequencyTracker, TinyLFUTracker};

/// Sketch capacity — large enough that the decay policy does not halve counts
/// within a single test trace (counts stay a faithful access tally).
const SKETCH_CAP: usize = 1 << 16;

/// Which backend the harness drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    /// Valued lineage policy. `oracle`/`sketch` toggle the branch-fan boost
    /// and TinyLFU term — used to prove each signal is load-bearing.
    Valued { oracle: bool, sketch: bool },
    /// Plain recency LRU.
    Lru,
    /// 4-tier frequency-aware LRU (the naive-LFU / frequency baseline).
    MultiLru,
}

/// A cache harness over one inactive backend, threading the sketch + oracle.
struct Harness {
    backend: Box<dyn InactiveIndex>,
    sketch: Arc<TinyLFUTracker<u128>>,
    oracle: Arc<BranchPointTracker>,
    capacity: usize,
    /// Hashes currently registered in the oracle (resident registration set),
    /// so a fresh admit fires `on_block_registered` exactly once per resident
    /// period and a reuse does not.
    registered: HashSet<u128>,
    next_id: BlockId,
    hits: u64,
    reqs: u64,
}

impl Harness {
    fn new(policy: Policy, capacity: usize, seed: u64) -> Self {
        let sketch = Arc::new(TinyLFUTracker::<u128>::new(SKETCH_CAP));
        let oracle = Arc::new(BranchPointTracker::new());
        let cap = NonZeroUsize::new(capacity).expect("capacity > 0");
        let backend: Box<dyn InactiveIndex> = match policy {
            Policy::Valued {
                oracle: use_oracle,
                sketch: use_sketch,
            } => {
                let s: Option<Arc<dyn FrequencyTracker<u128>>> =
                    use_sketch.then(|| sketch.clone() as Arc<dyn FrequencyTracker<u128>>);
                let o: Option<Arc<dyn BranchOracle>> =
                    use_oracle.then(|| oracle.clone() as Arc<dyn BranchOracle>);
                // Exact-scan (k_sample huge) makes eviction the deterministic
                // true arg-min, so the quality tests never depend on sampling
                // luck. Production uses k_sample = 16; the benches cover that.
                let params = ScorerParams {
                    gamma: 0.6,
                    n: 2,
                    k_sample: usize::MAX,
                    t_blocks: None,
                    seed,
                };
                Box::new(LineageBackend::with_policy(
                    capacity,
                    LeafPolicy::valued(capacity, s, o, params),
                ))
            }
            Policy::Lru => Box::new(LruBackend::new(cap)),
            Policy::MultiLru => Box::new(
                MultiLruBackend::new_with_thresholds(
                    cap,
                    // Production default (`with_multi_lru_backend`): cold < 3,
                    // warm < 8, hot < 15, very-hot >= 15. Under the agentic
                    // trace a completed body block is touched only a handful of
                    // times (<= turns), so it lands in the cold/warm tiers, not
                    // very-hot — MultiLRU still keeps the fresher completed
                    // bodies over the older ongoing ones within a tier (LRU
                    // within tier), which is what the valued poison beats.
                    &[3, 8, 15],
                    sketch.clone() as Arc<dyn FrequencyTracker<u128>>,
                )
                .expect("valid thresholds"),
            ),
        };
        Self {
            backend,
            sketch,
            oracle,
            capacity,
            registered: HashSet::new(),
            next_id: 0,
            hits: 0,
            reqs: 0,
        }
    }

    fn mint_id(&mut self) -> BlockId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Evict policy victims until there is room for one more block.
    fn make_room(&mut self) {
        while self.backend.len() >= self.capacity {
            let evicted = self.backend.allocate(1);
            if evicted.is_empty() {
                break; // only ghosts/interior left — nothing evictable
            }
            for (hash, _id) in evicted {
                self.oracle.on_block_removed(hash);
                self.registered.remove(&hash.as_u128());
            }
        }
    }

    /// Admit a block that is *already registered* (a reused hit re-cached).
    fn admit_existing(&mut self, hash: SequenceHash, id: BlockId) {
        if self.backend.has(hash) {
            return; // defensive: never double-insert a resident block
        }
        self.make_room();
        self.backend.insert(hash, id);
    }

    /// Admit a *fresh* block (a cache miss / new registration). Mirrors the
    /// production lifecycle order — evict to make room FIRST, then register and
    /// insert. This matters for fairness: the eviction victim is chosen from
    /// the pre-existing resident set, so the valued policy does NOT get to
    /// protect a parent using the not-yet-registered new child's fan-out (a
    /// bias the baselines could not exploit). `on_block_removed` fires for every
    /// eviction inside `make_room`.
    fn admit_fresh(&mut self, hash: SequenceHash) {
        if self.backend.has(hash) {
            // Already resident (a prefix gap) — an access, so touch only.
            self.sketch.touch(hash.as_u128());
            return;
        }
        self.make_room();
        self.sketch.touch(hash.as_u128());
        if self.registered.insert(hash.as_u128()) {
            self.oracle.on_block_registered(hash);
        }
        let id = self.mint_id();
        self.backend.insert(hash, id);
    }

    /// Low-level register of a single block (used to build precise scenarios).
    fn register(&mut self, hash: SequenceHash) {
        self.admit_fresh(hash);
    }

    /// Low-level removal of a single block (models it leaving the registry:
    /// evicted or reused-away without being re-cached).
    fn remove(&mut self, hash: SequenceHash) {
        if let Some((h, _id)) = self.backend.find_match(hash, false) {
            self.oracle.on_block_removed(h);
            self.registered.remove(&h.as_u128());
        }
    }

    /// A prefill request over `hashes[..up_to]`: match the longest cached
    /// prefix, count the hit rate (when `measured`), then re-admit every
    /// requested block (reused hits + fresh suffix), in prefix order so
    /// parents precede children.
    fn prefill(&mut self, hashes: &[SequenceHash], up_to: usize, measured: bool) {
        let up_to = up_to.min(hashes.len());
        let hits = self.backend.find_matches(&hashes[..up_to], false);
        if measured {
            self.hits += hits.len() as u64;
            self.reqs += up_to as u64;
        }
        // Re-admit reused hits (still registered; a genuine access → touch).
        for (hash, id) in &hits {
            self.sketch.touch(hash.as_u128());
            self.admit_existing(*hash, *id);
        }
        // Freshly-computed suffix.
        for &hash in &hashes[hits.len()..up_to] {
            self.admit_fresh(hash);
        }
    }

    /// Compaction of a completed session: poison the single-owner tail suffix
    /// ending at `tail`. Valued drains it evict-first; other backends no-op.
    fn compact(&mut self, tail: SequenceHash) {
        self.backend.poison(tail);
    }

    fn hit_rate(&self) -> f64 {
        if self.reqs == 0 {
            0.0
        } else {
            self.hits as f64 / self.reqs as f64
        }
    }

    fn has(&self, hash: SequenceHash) -> bool {
        self.backend.has(hash)
    }
}

// ---------------------------------------------------------------------------
// Trace-token helpers
// ---------------------------------------------------------------------------

/// Token base for the shared system prompt (all sessions share it). The prompt
/// occupies `[0, SYS_SPAN)`; every unique body lives at or above `SYS_SPAN`.
const SYS_BASE: u32 = 0;
/// Reserved token span for the shared system prompt.
const SYS_SPAN: u32 = 2048;
/// Token spacing between distinct session bodies. `>=` the largest body/noise
/// token span so ranges never overlap (distinct content ⇒ distinct hashes),
/// yet small enough that thousands of session keys stay within `u32`.
const SESSION_STRIDE: u32 = 2048;

/// First token of the unique body for session `key`.
fn body_base(key: u32) -> u32 {
    SYS_SPAN + key * SESSION_STRIDE
}

/// Build the block-hash chain for a session = shared system prompt (`sys_blk`
/// blocks) followed by a unique body (`body_blk` blocks keyed by `body_id`),
/// at block size `b`. Two sessions with the same `sys_blk` share the prompt
/// blocks and branch at the first body block — a real lineage branch point.
fn session_hashes(sys_blk: usize, body_id: u32, body_blk: usize, b: usize) -> Vec<SequenceHash> {
    let bb = b as u32;
    let mut tokens: Vec<u32> = Vec::with_capacity((sys_blk + body_blk) * b);
    tokens.extend(SYS_BASE..SYS_BASE + sys_blk as u32 * bb);
    let base = body_base(body_id);
    tokens.extend(base..base + body_blk as u32 * bb);
    chain(&tokens, b)
}

/// A standalone single-block "session" (an independent leaf root) with unique
/// content keyed by `id`.
fn noise_block(id: u32, b: usize) -> SequenceHash {
    let base = body_base(id);
    let tokens: Vec<u32> = (base..base + b as u32).collect();
    chain(&tokens, b)[0]
}

fn chain(tokens: &[u32], b: usize) -> Vec<SequenceHash> {
    BlockSequenceBuilder::from_tokens(tokens.to_vec())
        .with_block_size(b)
        .build()
        .into_iter()
        .map(|(_, h)| h)
        .collect()
}

// ---------------------------------------------------------------------------
// Agentic trace: ongoing conversations re-requested across rounds, interleaved
// with multi-turn sessions that complete and get compacted.
// ---------------------------------------------------------------------------

/// Per-`b` agentic-trace sizing. Token sizes are fixed; block counts (and the
/// cache) scale with `b`, so at `b = 256` the chains are 16x shorter than at
/// `b = 16` while the cache stays a constant fraction of the working set.
struct AgenticCfg {
    sys_blk: usize,
    body_o_blk: usize,
    body_c_blk: usize,
    ongoing: u32,
    completed_per_round: u32,
    turns: usize,
    rounds: usize,
    capacity: usize,
}

fn agentic_cfg(b: usize) -> AgenticCfg {
    // Token-level sizes → block counts shrink 16x from b=16 to b=256.
    let blk = |toks: usize| (toks / b).max(1);
    let sys_blk = blk(64 * 16);
    let body_o_blk = blk(96 * 16);
    let body_c_blk = blk(96 * 16);
    let ongoing = 3;
    let completed_per_round = 3;
    // Cache holds the working prefix cache (system prompt + every ongoing
    // body) plus one completed body of slack — enough for the valued policy to
    // keep all ongoing conversations *iff* it sheds the compacted dead tails,
    // but too small to also hold the per-round completed churn, so recency
    // (LRU) and frequency (MultiLRU) evict live ongoing bodies instead.
    let capacity = (sys_blk + ongoing as usize * body_o_blk + body_c_blk).max(4);
    AgenticCfg {
        sys_blk,
        body_o_blk,
        body_c_blk,
        ongoing,
        completed_per_round,
        turns: 6,
        rounds: 5,
        capacity,
    }
}

/// Run the agentic trace against one policy and return the measured hit rate
/// over the *ongoing* (reuse) requests.
fn run_agentic(b: usize, policy: Policy, poison: bool) -> f64 {
    let cfg = agentic_cfg(b);
    let mut h = Harness::new(policy, cfg.capacity, 0xA6E7_1C05);

    let ongoing: Vec<Vec<SequenceHash>> = (0..cfg.ongoing)
        .map(|i| session_hashes(cfg.sys_blk, i, cfg.body_o_blk, b))
        .collect();
    let ongoing_len = cfg.sys_blk + cfg.body_o_blk;

    // Prime the ongoing conversations once (cold).
    for s in &ongoing {
        h.prefill(s, ongoing_len, false);
    }

    let mut completed_id = 1000u32;
    for _round in 0..cfg.rounds {
        // 1. Serve every ongoing conversation (MEASURED reuse).
        for s in &ongoing {
            h.prefill(s, ongoing_len, true);
        }
        // 2. Churn: multi-turn sessions that heat up, complete, and compact.
        for _ in 0..cfg.completed_per_round {
            let sess = session_hashes(cfg.sys_blk, completed_id, cfg.body_c_blk, b);
            completed_id += 1;
            let total = cfg.sys_blk + cfg.body_c_blk;
            for t in 1..=cfg.turns {
                let up = (cfg.sys_blk + (cfg.body_c_blk * t / cfg.turns)).min(total);
                h.prefill(&sess, up, false);
            }
            if poison {
                h.compact(*sess.last().unwrap());
            }
        }
    }
    h.hit_rate()
}

const VALUED: Policy = Policy::Valued {
    oracle: true,
    sketch: true,
};

// ---------------------------------------------------------------------------
// Branch-point / dormant survival scenario (targeted has() assertions).
// ---------------------------------------------------------------------------

/// Build a re-leafed branch point (a `k`-way fork whose children have all been
/// evicted), then apply single-lineage eviction pressure, and report whether
/// the branch-point block is still cached. Frequencies are equalised (every
/// block touched once) so the *only* signal that can protect the fork block is
/// the oracle's fan-out boost — making the oracle load-bearing.
fn branch_survives(b: usize, policy: Policy, k: u32) -> bool {
    let capacity = (k as usize) + 6;
    // Pressure sits between the no-oracle threshold (~capacity: s1 ages out on
    // pure recency) and the oracle threshold (~fan*capacity, fan = 1+ln(1+k)):
    // with the fan boost s1 survives, without it s1 is evicted.
    let pressure = capacity + capacity / 2;
    let mut h = Harness::new(policy, capacity, 0x5109_2BFE);

    // Shared prefix s0 -> s1 (s1 is the fork point).
    let prefix = session_hashes(2, 7, 0, b); // 2 shared blocks, empty body
    let s1 = prefix[1];
    h.register(prefix[0]);
    h.register(prefix[1]);

    // k sub-agent forks, each a single divergent block under s1.
    let mut children = Vec::new();
    for c in 0..k {
        let fork = session_hashes(2, 100 + c, 1, b);
        let child = fork[2];
        h.register(child);
        children.push(child);
    }
    // Sub-agents finish: their tails are reused-away / removed. s1 re-leafs but
    // keeps its max_fanout high-water mark in the oracle.
    for &child in &children {
        h.remove(child);
    }

    // Single-lineage churn pressure.
    for i in 0..pressure {
        h.register(noise_block(50_000 + i as u32, b));
    }
    h.has(s1)
}

// ---------------------------------------------------------------------------
// Single-block item traces (parity / Zipf / scan-resistance)
// ---------------------------------------------------------------------------

/// Small deterministic xorshift64* PRNG for trace generation.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Replay a sequence of single-block item accesses (each an independent root
/// leaf) and return the hit rate over accesses at index `>= measure_from`
/// (use `0` for the whole trace; use `len - window` to score only a final
/// window, e.g. the closing hot-set sweep of a scan trace).
fn run_item_trace(
    b: usize,
    policy: Policy,
    capacity: usize,
    accesses: &[u32],
    measure_from: usize,
) -> f64 {
    let mut h = Harness::new(policy, capacity, 0x1D2E_3F40);
    for (i, &item) in accesses.iter().enumerate() {
        let hash = [noise_block(item, b)];
        h.prefill(&hash, 1, i >= measure_from);
    }
    h.hit_rate()
}

/// Recency-dominated accesses: with `reuse_prob` re-touch one of the last
/// `window` items (strong temporal locality → LRU is near-optimal); otherwise
/// advance to a fresh item.
fn recency_accesses(len: usize, reuse_prob: f64, window: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut recent: Vec<u32> = Vec::new();
    let mut next_fresh = 0u32;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let item = if !recent.is_empty() && rng.unit() < reuse_prob {
            let w = window.min(recent.len());
            recent[recent.len() - 1 - rng.below(w as u32) as usize]
        } else {
            let it = next_fresh;
            next_fresh += 1;
            it
        };
        recent.push(item);
        out.push(item);
    }
    out
}

/// Zipf-ish accesses over `n` items (power-law skew toward low indices).
fn zipf_accesses(len: usize, n: u32, skew: f64, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    (0..len)
        .map(|_| ((n as f64) * rng.unit().powf(skew)) as u32 % n)
        .collect()
}

/// Scan trace: a small hot set (warmed to a high TinyLFU count) re-accessed on
/// a fixed cadence, flooded by runs of unique one-shot "scan" blocks. LRU is
/// polluted by the scan and evicts the hot set; a frequency-aware policy keeps
/// the high-count hot set and resists the pollution.
fn scan_accesses(hot: u32, warmup: usize, scan_run: usize, runs: usize) -> Vec<u32> {
    let mut out = Vec::new();
    let mut scan_id = 1_000_000u32;
    // Warm the hot set so its frequency count saturates before the scans.
    for _ in 0..warmup {
        for h in 0..hot {
            out.push(h);
        }
    }
    for _ in 0..runs {
        for h in 0..hot {
            out.push(h);
        }
        for _ in 0..scan_run {
            out.push(scan_id);
            scan_id += 1;
        }
    }
    // Final hot-set sweep: measures whether the hot set survived the scans.
    for h in 0..hot {
        out.push(h);
    }
    out
}

// ===========================================================================
// TESTS
// ===========================================================================

// ---- Headline 1: valued beats the frequency baseline (MultiLRU) ----

/// On the agentic trace the valued policy's compaction-driven shedding of dead
/// completed-session tails yields a higher reuse hit-rate than the 4-tier
/// frequency-aware `MultiLruBackend`, which hoards frequency-hot (but now dead)
/// completed sessions and evicts live ongoing conversations instead. Asserted
/// at both `b = 16` and `b = 256` (block chains 16x shorter at 256).
#[test]
fn valued_beats_frequency_baseline_on_agentic_trace() {
    for b in [16usize, 256] {
        let valued = run_agentic(b, VALUED, true);
        let multi_lru = run_agentic(b, Policy::MultiLru, true);
        assert!(
            valued > multi_lru + 0.1,
            "b={b}: valued reuse hit-rate {valued:.3} must beat MultiLRU {multi_lru:.3} \
             (frequency baseline) by a clear margin on the agentic trace"
        );
    }
}

// ---- Headline 2: valued beats LRU on post-compaction turnover ----

/// The valued policy's poison-directed shedding of compacted tails keeps the
/// ongoing conversations resident, so its post-compaction reuse hit-rate beats
/// plain recency LRU (which keeps the recently-touched dead tails). Both `b`.
#[test]
fn valued_beats_lru_on_post_compaction_turnover() {
    for b in [16usize, 256] {
        let valued = run_agentic(b, VALUED, true);
        let lru = run_agentic(b, Policy::Lru, true);
        assert!(
            valued > lru + 0.1,
            "b={b}: valued reuse hit-rate {valued:.3} must beat LRU {lru:.3} on \
             post-compaction turnover"
        );
    }
}

// ---- Headline 3: shared / branch-point blocks survive under valued ----

/// After single-lineage churn evicts a broadly-shared prompt's sub-agent forks,
/// the re-leafed branch-point block stays cached under the valued policy (its
/// oracle fan-out boost keeps it alive) where LRU ages it out. Both `b`.
#[test]
fn shared_branch_point_survives_under_valued_not_lru() {
    for b in [16usize, 256] {
        let k = 8; // a broadly-shared system prompt: 8-way sub-agent fork.
        assert!(
            branch_survives(b, VALUED, k),
            "b={b}: the shared branch-point block must stay cached under valued"
        );
        assert!(
            !branch_survives(b, Policy::Lru, k),
            "b={b}: LRU must age the shared branch-point block out under the same pressure"
        );
    }
}

// ---- Headline 4: dormant multi-turn session survives mid-conversation ----

/// A quiet multi-turn session that earlier forked a pair of sub-agents leaves a
/// branch-point block; while other sessions churn the cache, the valued policy
/// keeps that dormant session's fork block (fan protection) where LRU drops it.
/// A minimal (2-way) fork — the hardest case for the fan boost — and both `b`.
#[test]
fn dormant_forked_session_survives_under_valued_not_lru() {
    for b in [16usize, 256] {
        let k = 2; // a dormant session that forked exactly two sub-agents.
        assert!(
            branch_survives(b, VALUED, k),
            "b={b}: the dormant forked session's block must survive under valued"
        );
        assert!(
            !branch_survives(b, Policy::Lru, k),
            "b={b}: LRU must drop the dormant forked session's block under churn"
        );
    }
}

// ---- Kill-mutation (a): compaction poison is load-bearing ----

/// Replacing the compaction `poison` with a no-op must degrade the valued
/// policy's agentic reuse hit-rate: without poison the dead completed tails are
/// no longer shed evict-first, so they clog the cache and live ongoing bodies
/// are evicted — exactly the failure poison exists to prevent.
#[test]
fn compaction_poison_is_load_bearing() {
    for b in [16usize, 256] {
        let with_poison = run_agentic(b, VALUED, true);
        let without_poison = run_agentic(b, VALUED, false);
        assert!(
            with_poison > without_poison + 0.1,
            "b={b}: no-op compaction ({without_poison:.3}) must degrade valued vs \
             real poison ({with_poison:.3}) — poison must be load-bearing"
        );
    }
}

// ---- Kill-mutation (b): the branch oracle is load-bearing for survival ----

/// Constructing the valued backend WITHOUT the branch oracle (dropping the
/// `max_fanout` fan boost) must lose the shared-prompt survival property:
/// with equalised frequencies the fan boost is the only signal that can keep
/// the re-leafed branch point alive, so without it the block is evicted.
#[test]
fn branch_oracle_is_load_bearing_for_survival() {
    for b in [16usize, 256] {
        let k = 8;
        assert!(
            branch_survives(b, VALUED, k),
            "b={b}: with the oracle the branch point survives"
        );
        assert!(
            !branch_survives(
                b,
                Policy::Valued {
                    oracle: false,
                    sketch: true
                },
                k
            ),
            "b={b}: dropping the oracle (fan boost) must lose branch-point survival — \
             frequency alone (equalised here) does not protect it"
        );
    }
}

// ---- Parity: valued ≈ LRU where LRU is near-optimal ----

/// On a recency-dominated trace (strong temporal locality) the valued score
/// degenerates to recency, so its hit-rate matches LRU within ~1%.
#[test]
fn parity_with_lru_on_recency_trace() {
    let cap = 64;
    let accesses = recency_accesses(4000, 0.75, 24, 0x1111);
    let valued = run_item_trace(16, VALUED, cap, &accesses, 0);
    let lru = run_item_trace(16, Policy::Lru, cap, &accesses, 0);
    assert!(
        (valued - lru).abs() < 0.02,
        "recency parity: valued {valued:.3} vs LRU {lru:.3} must agree within ~1%"
    );
}

/// On a Zipf trace (where LRU is near-optimal) the valued policy must not
/// regress below LRU — its frequency term makes it match or slightly beat LRU.
/// This is a **no-regression** assertion, NOT a parity claim: the small win is
/// genuine frequency signal (the correct direction on a skewed reuse
/// distribution). The recency test above is the two-sided parity check.
#[test]
fn no_regression_vs_lru_on_zipf_trace() {
    let cap = 64;
    let accesses = zipf_accesses(4000, 400, 3.0, 0x2222);
    let valued = run_item_trace(16, VALUED, cap, &accesses, 0);
    let lru = run_item_trace(16, Policy::Lru, cap, &accesses, 0);
    assert!(
        valued >= lru,
        "zipf no-regression: valued {valued:.3} must not fall below LRU {lru:.3}"
    );
}

// ---- Scan resistance ≥ LRU ----

/// A hot set warmed to a high TinyLFU count must survive a flood of one-shot
/// scan blocks. We measure only the CLOSING hot-set sweep (the trailing `hot`
/// accesses) — the fraction of the hot set still resident after the last scan —
/// so the assertion cannot be satisfied vacuously by the long warmup. The
/// valued policy (frequency resists scan pollution) must retain the hot set at
/// least as well as LRU, and must actually retain most of it (not a degenerate
/// tie at zero).
#[test]
fn scan_resistance_final_sweep_at_least_lru() {
    let cap = 64;
    let hot = 32u32;
    let accesses = scan_accesses(hot, 20, 96, 6);
    // Score only the trailing hot-set sweep.
    let final_sweep = accesses.len() - hot as usize;
    let valued = run_item_trace(16, VALUED, cap, &accesses, final_sweep);
    let lru = run_item_trace(16, Policy::Lru, cap, &accesses, final_sweep);
    assert!(
        valued >= lru,
        "scan resistance: valued final-sweep survival {valued:.3} must be >= LRU {lru:.3}"
    );
    assert!(
        valued > 0.5,
        "scan resistance: valued must actually retain most of the hot set through the \
         scan (final-sweep survival {valued:.3}), not tie LRU at ~0"
    );
}

// ---- B5: deep-chain eviction is O(depth), proved by operation count ----

/// Draining a deep single-lineage chain via `allocate(len)` must be linear in
/// the chain depth. We prove it with a **machine-independent operation count**
/// (not wall-clock): the total prune-loop iterations executed by
/// `remove_node_at`. A linear leaf-first drain re-leafs one parent per evicted
/// block, so it does ~1 prune step per block ⇒ total ≈ depth and the 4x-depth
/// ratio is ≈ 4. An O(depth^2) regression (e.g. re-walking the whole chain per
/// eviction) would push the ratio toward ~16. Counting operations removes all
/// timer/CPU-frequency sensitivity, so the band can be tight.
#[test]
fn deep_chain_eviction_is_operation_count_linear() {
    fn prune_iters_for(depth: u32) -> u64 {
        // b = 1: one block per token → a linear chain of `depth` blocks.
        let hashes: Vec<SequenceHash> = BlockSequenceBuilder::from_tokens((0..depth).collect())
            .with_block_size(1)
            .build()
            .into_iter()
            .map(|(_, h)| h)
            .collect();
        let mut backend = LineageBackend::with_policy(
            depth as usize,
            LeafPolicy::valued(depth as usize, None, None, ScorerParams::default()),
        );
        for (id, &h) in hashes.iter().enumerate() {
            backend.insert(h, id);
        }
        // Only count the eviction drain, not the inserts (inserts don't prune).
        backend.reset_prune_iters();
        let evicted = backend.allocate(hashes.len());
        assert_eq!(evicted.len(), hashes.len(), "the whole chain must evict");
        backend.prune_iters()
    }

    let d = 4_000u32;
    let small = prune_iters_for(d);
    let large = prune_iters_for(4 * d);

    // Sanity: linear drain does ~1 prune step per evicted block.
    assert!(
        small >= d as u64 && small <= 2 * d as u64,
        "expected ~1 prune step per block: d={d} did {small} prune iterations"
    );

    let ratio = large as f64 / small as f64;
    assert!(
        (3.5..=4.5).contains(&ratio),
        "deep-chain eviction prune-iteration ratio for 4x depth was {ratio:.3} \
         (d={d}: {small} prune iters, 4d={0}: {large} prune iters); a linear \
         leaf-first drain is ~depth (ratio ~4), an O(depth^2) prune/re-leaf \
         regression would be ~16",
        4 * d
    );
}
