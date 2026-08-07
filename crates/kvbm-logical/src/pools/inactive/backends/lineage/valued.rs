// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Valued leaf-eviction policy for [`LineageBackend`](super::LineageBackend).
//!
//! Where [`Fifo`](super::LeafPolicy::Fifo)/[`Tick`](super::LeafPolicy::Tick) order leaves
//! by pure recency, `ValuedPolicy` scores each *sampled* leaf and evicts the lowest —
//! blending TinyLFU frequency, recency, a proximity-to-compaction discount, and a
//! branch-point fan-out boost, plus a hard poison bit that always evicts first.
//!
//! # Score (evict lowest; computed only for sampled candidates)
//!
//! ```text
//! score(leaf) = base × pen × fan          // a poisoned leaf bypasses this entirely
//!
//! base = (1 + f̂) / (now − last_touch + 1)   f̂ = sketch.count (0 with no sketch → pure recency)
//! pen  = 1 − γ·q^n·g(φ)                      // Term C; ∈ [1−γ, 1], only bites known-linear leaves
//!   q    = min(position_blocks / T_blocks, 1)   (0 when T is unset → pen ≡ 1)
//!   g(φ) = 1 iff max_fanout = Some(≤1); else 0   // Some(≥2) OR None ⇒ 0 = fail-open-to-shared
//! fan  = 1 + ln(1 + φ)   for φ = Some(_); 1 for None   // protects (re-leafed) branch points
//! ```
//!
//! The two `None`-paths are load-bearing: a leaf whose branch record is absent (a fresh
//! single-lineage tip, or a shared prompt whose record aged out of the bounded oracle) is
//! never *penalized* (`g = 0 ⇒ pen = 1`) and never *boosted* (`fan = 1`). The
//! evict-linear / protect-shared behavior is carried by the `fan` boost on genuine branch
//! points, not by penalizing everything without a record.
//!
//! # Poison (hard, evict-first)
//!
//! A leaf marked via [`mark_poisoned`](ValuedPolicy::mark_poisoned) joins `poison_dense`, a
//! second dense vector maintained exactly like the leaf vector (each entry back-indexed by
//! `LeafState::poison_idx`). [`next_victim`](ValuedPolicy::next_victim) returns its head
//! before any scoring. Because the vector holds ONLY current poisoned leaves — an entry is
//! swap-removed the instant its node is evicted or demoted — there are no stale entries to
//! validate and none can alias a recycled arena slot. A node poisoned while it is still
//! interior joins when it later re-leafs (via [`on_leaf_added`](ValuedPolicy::on_leaf_added)).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use super::eviction::LeafAdvice;
use crate::blocks::SequenceHash;
use crate::branch_tracker::BranchOracle;
use crate::tinylfu::FrequencyTracker;

/// Tunable scorer constants. Carries the `f64` γ, so — per the eviction plan's guardrail —
/// it is **never** embedded in the `Copy + Eq` config enums; it travels as a private
/// builder field and is handed to [`ValuedPolicy::new`] at construction. Public so serve
/// time (EV-PR5) can populate `t_blocks` and pass it through
/// `BlockManagerConfigBuilder::with_valued_lineage_backend`.
#[derive(Debug, Clone)]
pub struct ScorerParams {
    /// Term-C discount strength; `pen ∈ [1−γ, 1]`. Default `0.6`.
    pub gamma: f64,
    /// Convexity exponent on `q` (near-zero discount until genuinely near the wall).
    /// Default `2`.
    pub n: u32,
    /// Number of leaves sampled at victim time (`argmin` over them). Default `16`.
    /// `k ≥ resident leaves` degenerates to an exact scan (the neutral/parity path).
    pub k_sample: usize,
    /// Compaction budget in blocks. `None` leaves the Term-C penalty inert (`pen ≡ 1`);
    /// it is populated at serve time (EV-PR5), never in the model spec.
    pub t_blocks: Option<u64>,
    /// Seed for the victim-sampling RNG, for deterministic tests/replays.
    pub seed: u64,
}

impl Default for ScorerParams {
    fn default() -> Self {
        Self {
            gamma: 0.6,
            n: 2,
            k_sample: 16,
            t_blocks: None,
            // A fixed odd constant (golden-ratio derived) so an unseeded policy is still
            // deterministic run-to-run.
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Per-leaf ordering state, indexed by arena slot. Present for every `Real` node the
/// backend has announced via [`on_node_inserted`](ValuedPolicy::on_node_inserted); `None`
/// for free/ghost slots.
struct LeafState {
    seq_hash: SequenceHash,
    /// Logical tick at the node's most recent entry into the eviction order.
    last_touch: u64,
    /// Sticky until the slot recycles (`on_node_removed`).
    poisoned: bool,
    /// Index into `leaf_dense` while this node is an evictable leaf; `None` otherwise
    /// (interior, or removed). Enables O(1) uniform sampling and swap-remove.
    dense_idx: Option<u32>,
    /// Index into `poison_dense` while this node is a *poisoned* evictable leaf; `None`
    /// otherwise. Kept an exact inverse of `poison_dense`, so a recycled arena slot can
    /// never alias a stale poison entry.
    poison_idx: Option<u32>,
}

/// Sampled-min valued leaf policy. See the module docs.
pub(crate) struct ValuedPolicy {
    /// Per-slot state, addressed by arena index (parallel to the backend's slab).
    slots: Vec<Option<LeafState>>,
    /// Slot indices of currently-evictable leaves, for O(1) uniform sampling / swap-remove.
    leaf_dense: Vec<u32>,
    /// Slot indices of poisoned evictable leaves — evicted before any scoring. A dense
    /// vector maintained exactly like `leaf_dense` (each entry's `poison_idx` back-indexes
    /// it), so it holds ONLY current poisoned leaves: no stale entries to alias a recycled
    /// slot, and its size is bounded by the resident leaf count.
    poison_dense: Vec<u32>,
    /// Monotone logical clock; advances once per leaf entry (recency baseline).
    now: u64,
    /// xorshift64* state for victim sampling.
    rng: u64,
    sketch: Option<Arc<dyn FrequencyTracker<u128>>>,
    oracle: Option<Arc<dyn BranchOracle>>,
    params: ScorerParams,
}

impl ValuedPolicy {
    pub(crate) fn new(
        capacity: usize,
        sketch: Option<Arc<dyn FrequencyTracker<u128>>>,
        oracle: Option<Arc<dyn BranchOracle>>,
        params: ScorerParams,
    ) -> Self {
        // A zero seed would make xorshift emit only zeros; fall back to the default odd
        // constant so sampling still advances.
        let rng = if params.seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            params.seed
        };
        Self {
            slots: Vec::with_capacity(capacity),
            leaf_dense: Vec::with_capacity(capacity),
            poison_dense: Vec::new(),
            now: 0,
            rng,
            sketch,
            oracle,
            params,
        }
    }

    fn ensure(&mut self, idx: u32) {
        if idx as usize >= self.slots.len() {
            self.slots
                .resize_with(idx as usize + 1, || None::<LeafState>);
        }
    }

    /// xorshift64* — small, fast, deterministic. Never returns the same stream across
    /// distinct non-zero seeds.
    fn next_rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    // ---- backend-facing hooks (mirror the `LeafPolicy` surface) ----

    /// A slot became a `Real` node. Records its hash so the scorer/poison walk can reach
    /// the sketch and oracle; the node does not enter the eviction order until
    /// [`on_leaf_added`](Self::on_leaf_added).
    pub(crate) fn on_node_inserted(&mut self, idx: u32, seq_hash: SequenceHash) {
        self.ensure(idx);
        self.slots[idx as usize] = Some(LeafState {
            seq_hash,
            last_touch: self.now,
            poisoned: false,
            dense_idx: None,
            poison_idx: None,
        });
    }

    /// A `Real` node is now a leaf — stamp its recency and add it to the eviction order.
    /// A node poisoned while interior joins the poison set here, as it re-leafs.
    pub(crate) fn on_leaf_added(&mut self, idx: u32) {
        self.now += 1;
        let now = self.now;
        let dense_pos = self.leaf_dense.len() as u32;
        let poisoned = {
            let state = self.slots[idx as usize]
                .as_mut()
                .expect("ValuedPolicy: on_leaf_added before on_node_inserted");
            debug_assert!(
                state.dense_idx.is_none(),
                "ValuedPolicy: leaf {idx} added while already in the eviction order"
            );
            state.last_touch = now;
            state.dense_idx = Some(dense_pos);
            state.poisoned
        };
        self.leaf_dense.push(idx);
        if poisoned {
            self.add_poison(idx);
        }
    }

    /// A `Real` leaf gained a child — remove it from the eviction order (and the poison set)
    /// but keep its per-node state (recency, poison bit) for a possible later re-leafing.
    pub(crate) fn on_leaf_demoted(&mut self, idx: u32) {
        self.unlink_leaf(idx);
        self.unlink_poison(idx);
    }

    /// A `Real` node left the graph — drop it from both dense vectors and clear its state
    /// (poison included) so a recycled slot starts fresh.
    pub(crate) fn on_node_removed(&mut self, idx: u32) {
        self.unlink_leaf(idx);
        self.unlink_poison(idx);
        if (idx as usize) < self.slots.len() {
            self.slots[idx as usize] = None;
        }
    }

    /// Remove `idx` from `leaf_dense` (O(1) swap-remove) and clear its `dense_idx`. No-op
    /// if it is not currently a leaf.
    fn unlink_leaf(&mut self, idx: u32) {
        let Some(state) = self.slots.get_mut(idx as usize).and_then(|s| s.as_mut()) else {
            return;
        };
        let Some(dense_pos) = state.dense_idx.take() else {
            return;
        };
        let last = self.leaf_dense.len() as u32 - 1;
        self.leaf_dense.swap_remove(dense_pos as usize);
        // The element that was at the tail now sits at `dense_pos`; fix its back-index.
        if dense_pos != last {
            let moved = self.leaf_dense[dense_pos as usize];
            if let Some(moved_state) = self.slots[moved as usize].as_mut() {
                moved_state.dense_idx = Some(dense_pos);
            }
        }
    }

    /// Append `idx` to `poison_dense`, recording its back-index. Precondition: `idx` is a
    /// live leaf not already in the poison set.
    fn add_poison(&mut self, idx: u32) {
        let pos = self.poison_dense.len() as u32;
        let Some(state) = self.slots.get_mut(idx as usize).and_then(|s| s.as_mut()) else {
            return;
        };
        debug_assert!(
            state.poison_idx.is_none(),
            "ValuedPolicy: poison double-add for {idx}"
        );
        state.poison_idx = Some(pos);
        self.poison_dense.push(idx);
    }

    /// Remove `idx` from `poison_dense` (O(1) swap-remove) and clear its `poison_idx`. No-op
    /// if it is not currently a poisoned leaf. Leaves the sticky `poisoned` bit intact so a
    /// demote→re-leaf re-adds it.
    fn unlink_poison(&mut self, idx: u32) {
        let Some(state) = self.slots.get_mut(idx as usize).and_then(|s| s.as_mut()) else {
            return;
        };
        let Some(pos) = state.poison_idx.take() else {
            return;
        };
        let last = self.poison_dense.len() as u32 - 1;
        self.poison_dense.swap_remove(pos as usize);
        if pos != last {
            let moved = self.poison_dense[pos as usize];
            if let Some(moved_state) = self.slots[moved as usize].as_mut() {
                moved_state.poison_idx = Some(pos);
            }
        }
    }

    /// Slot index of the next block to evict, or `None` if no leaves. Poisoned leaves go
    /// first (relative order among them is immaterial — all are dead), then a sampled-min
    /// over `k_sample` uniformly-drawn leaves.
    pub(crate) fn next_victim(&mut self) -> Option<u32> {
        // Poisoned leaves first. `poison_dense` holds only current poisoned leaves, so its
        // head is always a valid victim — no lazy validation, no recycle aliasing.
        if let Some(&idx) = self.poison_dense.first() {
            return Some(idx);
        }

        let n = self.leaf_dense.len();
        if n == 0 {
            return None;
        }
        let k = self.params.k_sample.max(1).min(n);
        let mut best_idx: Option<u32> = None;
        let mut best_score = f64::INFINITY;
        if k == n {
            // Exact scan — also the neutral-params / parity path (K ≥ resident leaves).
            for i in 0..n {
                let slot = self.leaf_dense[i];
                let s = self.score_slot(slot);
                if s < best_score {
                    best_score = s;
                    best_idx = Some(slot);
                }
            }
        } else {
            // Sampling is with-replacement and `% n` carries a negligible modulo bias
            // (a 64-bit draw over `n ≤ total leaves`); for a victim heuristic the tiny
            // non-uniformity and occasional duplicate draw do not affect correctness.
            for _ in 0..k {
                let r = (self.next_rand() % n as u64) as usize;
                let slot = self.leaf_dense[r];
                let s = self.score_slot(slot);
                if s < best_score {
                    best_score = s;
                    best_idx = Some(slot);
                }
            }
        }
        best_idx
    }

    /// Value score for the leaf at slot `idx` (see the module docs). Lower ⇒ evict sooner.
    fn score_slot(&self, idx: u32) -> f64 {
        let state = match self.slots.get(idx as usize).and_then(|s| s.as_ref()) {
            Some(s) => s,
            None => return f64::INFINITY, // not a tracked leaf — never selected
        };
        let f_hat = self
            .sketch
            .as_ref()
            .map_or(0, |s| s.count(state.seq_hash.as_u128())) as f64;
        let age = self.now.saturating_sub(state.last_touch) as f64;
        let base = (1.0 + f_hat) / (age + 1.0);

        let phi = self
            .oracle
            .as_ref()
            .and_then(|o| o.max_fanout(state.seq_hash));

        // Term C: only bites a *known-linear* leaf (max_fanout Some(≤1)) once a compaction
        // budget is set. Absent record (None) or a branch point (Some(≥2)) ⇒ pen = 1.
        let pen = match (phi, self.params.t_blocks) {
            (Some(f), Some(t_blocks)) if f <= 1 && t_blocks > 0 => {
                let q = (state.seq_hash.position() as f64 / t_blocks as f64).min(1.0);
                1.0 - self.params.gamma * q.powi(self.params.n as i32)
            }
            _ => 1.0,
        };

        // Fan boost: protects (re-leafed) branch points; absent record ⇒ neutral 1.
        let fan = match phi {
            Some(f) => 1.0 + (1.0 + f as f64).ln(),
            None => 1.0,
        };

        base * pen * fan
    }

    // ---- poison support (driven by the backend's graph walk) ----

    /// Mark slot `idx` poisoned (sticky until recycle). If it is already an evictable leaf
    /// it joins the poison set immediately; if interior, it joins when it next re-leafs.
    /// No-op for an untracked (ghost/free) slot.
    // Production caller is EV-PR4 (`BlockManager::poison_lineage`); exercised by tests now.
    pub(crate) fn mark_poisoned(&mut self, idx: u32) {
        let is_leaf = {
            let Some(state) = self.slots.get_mut(idx as usize).and_then(|s| s.as_mut()) else {
                return;
            };
            if state.poisoned {
                return;
            }
            state.poisoned = true;
            state.dense_idx.is_some()
        };
        if is_leaf {
            self.add_poison(idx);
        }
    }

    /// Peak fan-out for `seq_hash` treated as a parent, via the attached oracle (`None` if
    /// no oracle, or no record). The backend's poison walk stops at the first ancestor with
    /// `max_fanout ≥ 2` (a shared branch point is never poisoned).
    pub(crate) fn max_fanout_of(&self, seq_hash: SequenceHash) -> Option<u32> {
        self.oracle.as_ref().and_then(|o| o.max_fanout(seq_hash))
    }

    // ---- read-only snapshot API (R7a §3.2 / §3.4) ----

    /// Read-only per-node advice for slot `idx` (leaf *or* interior — every
    /// `Real` node has a `LeafState`). Reads the sketch and oracle; never
    /// touches either. All-absent for an untracked (ghost/free) slot.
    pub(crate) fn advice_for(&self, idx: u32) -> LeafAdvice {
        let Some(state) = self.slots.get(idx as usize).and_then(|s| s.as_ref()) else {
            return LeafAdvice::default();
        };
        LeafAdvice {
            poisoned: state.poisoned,
            // Same recency baseline the scorer uses: pool-logical ticks, not time.
            age_ticks: Some(self.now.saturating_sub(state.last_touch)),
            freq_estimate: self
                .sketch
                .as_ref()
                .map(|s| s.count(state.seq_hash.as_u128())),
            max_fanout: self
                .oracle
                .as_ref()
                .and_then(|o| o.max_fanout(state.seq_hash)),
        }
    }

    /// Read-only victim peek: the poison set first (capped at `max`), then the
    /// lowest-scoring leaves, up to `max` slots total.
    ///
    /// # Why this is not `next_victim`
    ///
    /// [`next_victim`](Self::next_victim) draws `k_sample` leaves through the
    /// policy RNG. Reusing it here would advance that RNG, perturbing the next
    /// *real* eviction and breaking trace replay (R7a §3.4). This is instead a
    /// pure scan: `&self`, no RNG, no `now` stamp.
    ///
    /// # Bounds and the poison dedup rule
    ///
    /// The scan covers `min(leaf_dense.len(), MAX_PEEK_SCAN)` entries from the
    /// head of `leaf_dense` and keeps a k-min heap, so the cost is
    /// O(MAX_PEEK_SCAN + max·log max). Past that bound the peek may miss the
    /// true global minimum — a *coverage* bound, not a correctness one: these
    /// candidates feed an advisory consumer and the real eviction path is
    /// untouched.
    ///
    /// The window also truncates the **count**, not just the quality: the
    /// scored tail of the result can never exceed `MAX_PEEK_SCAN` however
    /// large `max` is, so a caller cannot read a short result as "the pool has
    /// no more leaves". The poison prefix is *not* subject to the window — it
    /// is capped only by `max` — so the total is
    /// `min(max, poison_dense.len()) + min(remaining, scanned non-poisoned)`.
    /// [`BlockManager::inactive_candidates`](crate::manager::BlockManager::inactive_candidates)
    /// states this bound in prose for consumers outside the crate.
    ///
    /// A poisoned leaf lives in **both** dense vectors, so the scan phase skips
    /// any slot whose `poison_idx` is `Some` — exact arena-slot identity, not a
    /// hash comparison. Without that skip a poisoned leaf (which is also,
    /// typically, a low scorer) would be listed twice in one peek. Skipped
    /// entries still consume the scan budget; that only trims coverage.
    pub(crate) fn peek_slots(&self, max: usize) -> Vec<u32> {
        if max == 0 {
            return Vec::new();
        }
        // Poisoned leaves first — they bypass scoring in `next_victim` too.
        let mut out: Vec<u32> = self.poison_dense.iter().take(max).copied().collect();
        let remaining = max - out.len();
        if remaining == 0 {
            return out;
        }

        let scan = self.leaf_dense.len().min(MAX_PEEK_SCAN);
        // Max-heap of the `remaining` best-so-far: push, then drop the worst.
        // Capacity is clamped to the scan bound: `max` is caller-supplied and
        // "give me everything" (`usize::MAX`) must not turn into an overflowing
        // or multi-gigabyte reservation — the heap can never exceed `scan`.
        let mut heap: BinaryHeap<ScoredLeaf> = BinaryHeap::with_capacity(remaining.min(scan) + 1);
        for &idx in &self.leaf_dense[..scan] {
            if self
                .slots
                .get(idx as usize)
                .and_then(|s| s.as_ref())
                .is_some_and(|s| s.poison_idx.is_some())
            {
                continue; // already listed above (dedup by arena-slot identity)
            }
            heap.push(ScoredLeaf {
                score: self.score_slot(idx),
                idx,
            });
            if heap.len() > remaining {
                heap.pop();
            }
        }
        // Ascending score — the eviction order among the kept candidates.
        out.extend(heap.into_sorted_vec().into_iter().map(|entry| entry.idx));
        out
    }

    /// Number of currently-evictable leaves. Test-only.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.leaf_dense.len()
    }
}

/// Coverage bound for the read-only [`ValuedPolicy::peek_slots`] scan: at most
/// this many `leaf_dense` entries are scored per peek. Documented as a
/// coverage bound, not a correctness bound — see `peek_slots`.
///
/// `pub(super)` so the backend's own tests can pin the boundary against the
/// constant instead of a magic number; it is crate-internal either way.
pub(super) const MAX_PEEK_SCAN: usize = 4096;

/// Heap entry for the bounded k-min peek scan.
///
/// Ordered by score then slot index, both through total orders
/// (`f64::total_cmp` orders every bit pattern, NaN included), so a peek is
/// deterministic for a given policy state and the max-heap always pops the
/// worst candidate kept so far. `PartialEq`/`PartialOrd` delegate to `cmp` to
/// keep the four comparison traits mutually consistent.
struct ScoredLeaf {
    score: f64,
    idx: u32,
}

impl Ord for ScoredLeaf {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then(self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for ScoredLeaf {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ScoredLeaf {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for ScoredLeaf {}

#[cfg(test)]
impl ValuedPolicy {
    /// Force the logical clock (for controlled recency in tests).
    fn test_set_now(&mut self, now: u64) {
        self.now = now;
    }

    /// Insert a leaf directly into the eviction order with an explicit `last_touch`,
    /// bypassing the `now`-advancing `on_leaf_added` so a test can pin recency exactly.
    fn test_add_leaf(&mut self, idx: u32, seq_hash: SequenceHash, last_touch: u64) {
        self.ensure(idx);
        let dense_pos = self.leaf_dense.len() as u32;
        self.slots[idx as usize] = Some(LeafState {
            seq_hash,
            last_touch,
            poisoned: false,
            dense_idx: Some(dense_pos),
            poison_idx: None,
        });
        self.leaf_dense.push(idx);
    }

    /// Direct score of the leaf at `idx` (lower ⇒ evict sooner).
    fn test_score(&self, idx: u32) -> f64 {
        self.score_slot(idx)
    }

    /// Whether slot `idx`'s node is currently marked poisoned.
    pub(crate) fn test_is_poisoned(&self, idx: u32) -> bool {
        self.slots
            .get(idx as usize)
            .and_then(|s| s.as_ref())
            .is_some_and(|s| s.poisoned)
    }

    /// Assert the dense-index bookkeeping is internally consistent: `leaf_dense`/`dense_idx`
    /// and `poison_dense`/`poison_idx` are each exact inverses, a poisoned-set entry is
    /// always also a leaf, and nothing else claims membership.
    fn test_check_invariants(&self) {
        for (pos, &idx) in self.leaf_dense.iter().enumerate() {
            let state = self.slots[idx as usize]
                .as_ref()
                .expect("leaf_dense points at a live slot");
            assert_eq!(
                state.dense_idx,
                Some(pos as u32),
                "dense_idx must be the inverse of leaf_dense"
            );
        }
        for (pos, &idx) in self.poison_dense.iter().enumerate() {
            let state = self.slots[idx as usize]
                .as_ref()
                .expect("poison_dense points at a live slot");
            assert_eq!(
                state.poison_idx,
                Some(pos as u32),
                "poison_idx must be the inverse of poison_dense"
            );
            assert!(state.poisoned, "poison_dense entry must be poisoned");
            assert!(
                state.dense_idx.is_some(),
                "poison_dense entry must also be a leaf"
            );
        }
        for (idx, slot) in self.slots.iter().enumerate() {
            if let Some(state) = slot {
                if let Some(dpos) = state.dense_idx {
                    assert_eq!(
                        self.leaf_dense[dpos as usize], idx as u32,
                        "a slot claiming dense_idx must sit there in leaf_dense"
                    );
                }
                if let Some(ppos) = state.poison_idx {
                    assert_eq!(
                        self.poison_dense[ppos as usize], idx as u32,
                        "a slot claiming poison_idx must sit there in poison_dense"
                    );
                }
                // Converse: every poisoned *leaf* must be registered in the poison set.
                if state.poisoned && state.dense_idx.is_some() {
                    assert!(
                        state.poison_idx.is_some(),
                        "a poisoned leaf must have a poison_dense entry"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tinylfu::TinyLFUTracker;
    use std::collections::HashMap;

    /// `SequenceHash` at an explicit position (the value scorer keys `q` on
    /// `position()` and the sketch/oracle on the hash identity).
    fn hash_at(current: u64, position: u64) -> SequenceHash {
        // A non-root needs a parent fragment; the exact value is irrelevant to these
        // policy-level tests, which key the stub oracle on the whole hash.
        let parent = if position == 0 {
            None
        } else {
            Some(current ^ 0xF00D)
        };
        SequenceHash::new(current, parent, position)
    }

    /// Oracle returning a caller-fixed `max_fanout` per hash (unset ⇒ `None`).
    #[derive(Default)]
    struct StubOracle {
        fanout: HashMap<SequenceHash, u32>,
    }
    impl BranchOracle for StubOracle {
        fn on_block_registered(&self, _hash: SequenceHash) {}
        fn on_block_removed(&self, _hash: SequenceHash) {}
        fn max_fanout(&self, hash: SequenceHash) -> Option<u32> {
            self.fanout.get(&hash).copied()
        }
    }

    fn params_with(gamma: f64, t_blocks: Option<u64>) -> ScorerParams {
        ScorerParams {
            gamma,
            n: 2,
            k_sample: 16,
            t_blocks,
            seed: 0x1234_5678_9ABC_DEF0,
        }
    }

    // ---- neutral params: score degenerates to recency (LRU) ----

    #[test]
    fn neutral_params_score_is_pure_recency() {
        // No sketch, no oracle, K >= leaves ⇒ exact scan; score = 1/(age+1).
        let mut p = ValuedPolicy::new(8, None, None, params_with(0.6, None));
        p.test_set_now(10);
        p.test_add_leaf(0, hash_at(100, 3), 1); // oldest
        p.test_add_leaf(1, hash_at(200, 3), 5);
        p.test_add_leaf(2, hash_at(300, 3), 3);
        // Oldest leaf (smallest last_touch) has the largest age ⇒ lowest score ⇒ victim.
        assert_eq!(p.next_victim(), Some(0));
        assert!(p.test_score(0) < p.test_score(2));
        assert!(p.test_score(2) < p.test_score(1));
    }

    // ---- kill-mutation (i): pen penalizes a deep, known-linear leaf ----

    #[test]
    fn killmut_pen_penalizes_deep_linear_leaf() {
        // Both leaves linear (max_fanout Some(1) ⇒ g=1) and identical recency/frequency;
        // ONLY position differs. The deeper leaf (q≈0.9) must score below the shallow one
        // (q≈0.1). Inverting `pen` (1 + γq^n instead of 1 −) flips this.
        let deep = hash_at(90, 90);
        let shallow = hash_at(10, 10);
        let mut oracle = StubOracle::default();
        oracle.fanout.insert(deep, 1);
        oracle.fanout.insert(shallow, 1);
        let mut p = ValuedPolicy::new(8, None, Some(Arc::new(oracle)), params_with(0.6, Some(100)));
        p.test_set_now(1000);
        p.test_add_leaf(0, deep, 500);
        p.test_add_leaf(1, shallow, 500); // identical recency
        assert!(
            p.test_score(0) < p.test_score(1),
            "deep known-linear leaf must score below the shallow one (pen)"
        );
        assert_eq!(p.next_victim(), Some(0));
    }

    // ---- kill-mutation (ii): a missing branch record must NOT penalize ----

    #[test]
    fn killmut_missing_record_is_not_penalized() {
        // A leaf whose oracle returns None (no record) must have pen == 1 AND fan == 1, i.e.
        // its score is exactly `base`, even deep in the sequence with a compaction budget
        // set. A mutation that penalizes None (g=1) would drop the score below base.
        let leaf = hash_at(90, 90);
        let oracle = StubOracle::default(); // returns None for everything
        let mut p = ValuedPolicy::new(8, None, Some(Arc::new(oracle)), params_with(0.6, Some(100)));
        p.test_set_now(1000);
        p.test_add_leaf(0, leaf, 500);
        let age = 1000.0 - 500.0;
        let base = 1.0 / (age + 1.0); // f̂ = 0 (no sketch)
        assert!(
            (p.test_score(0) - base).abs() < 1e-12,
            "a record-less leaf must be neither penalized nor boosted (score == base)"
        );
    }

    // ---- kill-mutation (iii): the fan boost protects higher-fanout branch points ----

    #[test]
    fn killmut_fanout_boost_protects_branch_point() {
        // Equal recency/frequency/position, penalty inert (T unset). Only max_fanout
        // differs: B=8, C=1. The fan boost must make score(B) > score(C) so B is evicted
        // later. Dropping the fan term makes them equal.
        let b = hash_at(50, 50);
        let c = hash_at(60, 50);
        let mut oracle = StubOracle::default();
        oracle.fanout.insert(b, 8);
        oracle.fanout.insert(c, 1);
        let mut p = ValuedPolicy::new(8, None, Some(Arc::new(oracle)), params_with(0.6, None));
        p.test_set_now(1000);
        p.test_add_leaf(0, b, 500);
        p.test_add_leaf(1, c, 500);
        assert!(
            p.test_score(0) > p.test_score(1),
            "higher fan-out branch point must score above the low-fanout leaf (fan boost)"
        );
        assert_eq!(
            p.next_victim(),
            Some(1),
            "the low-fanout leaf is evicted first"
        );
    }

    // ---- frequency raises value (TinyLFU term) ----

    #[test]
    fn frequency_raises_score_over_a_cold_leaf() {
        let hot = hash_at(1, 5);
        let cold = hash_at(2, 5);
        let sketch = Arc::new(TinyLFUTracker::<u128>::new(1 << 12));
        for _ in 0..8 {
            sketch.touch(hot.as_u128());
        }
        let mut p = ValuedPolicy::new(8, Some(sketch), None, params_with(0.6, None));
        p.test_set_now(100);
        p.test_add_leaf(0, hot, 50);
        p.test_add_leaf(1, cold, 50); // same recency
        assert!(
            p.test_score(0) > p.test_score(1),
            "the frequently-touched leaf must score higher than the cold one"
        );
        assert_eq!(p.next_victim(), Some(1));
    }

    // ---- seeded sampling determinism ----

    #[test]
    fn seeded_sampling_is_deterministic() {
        let build = || {
            let mut p = ValuedPolicy::new(64, None, None, params_with(0.6, None));
            p.test_set_now(10_000);
            // More leaves than K (=16) so sampling actually selects a subset.
            for i in 0..40u32 {
                p.test_add_leaf(i, hash_at(i as u64, 4), (i as u64) * 7 % 101);
            }
            p
        };
        let mut a = build();
        let mut b = build();
        // Same seed + identical state ⇒ identical sampled victim, repeatedly.
        for _ in 0..10 {
            assert_eq!(a.next_victim(), b.next_victim());
        }
    }

    // ---- poison FIFO ordering ----

    #[test]
    fn poisoned_leaf_evicts_before_any_scored_leaf() {
        let mut p = ValuedPolicy::new(8, None, None, params_with(0.6, None));
        p.test_set_now(100);
        // Leaf 0 is the FRESHEST (would be evicted LAST by recency)...
        p.test_add_leaf(0, hash_at(1, 5), 99);
        p.test_add_leaf(1, hash_at(2, 5), 10); // stale — would be the recency victim
        // ...but poisoning it forces it first.
        p.mark_poisoned(0);
        assert_eq!(p.next_victim(), Some(0), "poisoned leaf evicts first");
    }

    #[test]
    fn removing_a_poisoned_leaf_drops_it_from_the_poison_set() {
        let mut p = ValuedPolicy::new(8, None, None, params_with(0.6, None));
        p.test_set_now(100);
        p.test_add_leaf(0, hash_at(1, 5), 50);
        p.test_add_leaf(1, hash_at(2, 5), 10);
        p.mark_poisoned(0);
        // Slot 0 leaves the graph — it must leave the poison set too (no stale entry).
        p.on_node_removed(0);
        p.test_check_invariants();
        // No poisoned leaf remains, so the surviving leaf is scored normally.
        assert_eq!(p.next_victim(), Some(1));
    }

    /// A recycled arena slot must not inherit a stale poison entry: poison slot 0, evict it,
    /// then reuse slot 0 for a fresh UN-poisoned leaf — it must NOT be force-evicted first.
    #[test]
    fn recycled_slot_does_not_inherit_stale_poison() {
        let mut p = ValuedPolicy::new(8, None, None, params_with(0.6, None));
        p.test_set_now(100);
        p.test_add_leaf(0, hash_at(1, 5), 50); // will be poisoned then evicted
        p.test_add_leaf(1, hash_at(2, 5), 10); // stale — the true recency victim
        p.mark_poisoned(0);
        assert_eq!(p.next_victim(), Some(0));
        p.on_node_removed(0); // evict the poisoned leaf; slot 0 is now free

        // Recycle slot 0 for a fresh, un-poisoned, FRESH (evict-last) leaf.
        p.on_node_inserted(0, hash_at(3, 5));
        p.on_leaf_added(0);
        p.test_check_invariants();
        // With no live poison, the oldest leaf (slot 1) wins — NOT the recycled slot 0.
        assert_eq!(
            p.next_victim(),
            Some(1),
            "a recycled slot must not be evicted via a stale poison entry"
        );
    }

    // ---- leaf-index bookkeeping stays consistent under random hook sequences ----

    proptest::proptest! {
        #[test]
        fn leaf_dense_stays_consistent(ops in proptest::collection::vec(0u8..5, 0..300)) {
            use std::collections::BTreeSet;
            let mut p = ValuedPolicy::new(0, None, None, ScorerParams::default());
            let n_slots = 12u32;
            let mut real: BTreeSet<u32> = BTreeSet::new();   // slots that are Real nodes
            let mut leaf: BTreeSet<u32> = BTreeSet::new();   // slots currently in the order

            for (i, op) in ops.iter().enumerate() {
                let idx = (i as u32) % n_slots;
                match op {
                    0 => {
                        if !real.contains(&idx) {
                            p.on_node_inserted(idx, hash_at(idx as u64 + 1, 3));
                            real.insert(idx);
                        }
                    }
                    1 => {
                        if real.contains(&idx) && !leaf.contains(&idx) {
                            p.on_leaf_added(idx);
                            leaf.insert(idx);
                        }
                    }
                    2 => {
                        if leaf.contains(&idx) {
                            p.on_leaf_demoted(idx);
                            leaf.remove(&idx);
                        }
                    }
                    3 => {
                        if real.contains(&idx) {
                            p.on_node_removed(idx);
                            real.remove(&idx);
                            leaf.remove(&idx);
                        }
                    }
                    _ => {
                        // Poison a Real node (leaf or interior) — exercises `poison_dense`
                        // through demote/re-leaf/remove churn. Does not change leaf membership.
                        if real.contains(&idx) {
                            p.mark_poisoned(idx);
                        }
                    }
                }
                p.test_check_invariants();
                proptest::prop_assert_eq!(p.len(), leaf.len());
                let victim = p.next_victim();
                proptest::prop_assert_eq!(victim.is_some(), !leaf.is_empty());
                if let Some(v) = victim {
                    proptest::prop_assert!(leaf.contains(&v));
                }
            }
        }
    }
}
