// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pluggable leaf-eviction ordering for [`LineageBackend`](super::LineageBackend).
//!
//! The lineage graph only ever evicts *leaves*; *which* leaf goes first is
//! the one policy decision that varies. [`LeafPolicy`] isolates it behind a
//! small set of hooks the backend calls as nodes are inserted, change
//! leaf/interior status, and are removed:
//!
//! - [`on_node_inserted`](LeafPolicy::on_node_inserted) — a slot became a
//!   `Real` node (fresh insert or ghost promotion).
//! - [`on_leaf_added`](LeafPolicy::on_leaf_added) — a `Real` node is now a
//!   leaf and enters the eviction order.
//! - [`on_leaf_demoted`](LeafPolicy::on_leaf_demoted) — a `Real` leaf
//!   gained a child; still in the graph, no longer evictable. Per-node
//!   ordering state is *retained* (a future re-leafing may need it).
//! - [`on_node_removed`](LeafPolicy::on_node_removed) — a `Real` node left
//!   the graph (evicted or resurrected). Per-node ordering state is cleared.
//! - [`next_victim`](LeafPolicy::next_victim) — the eviction-order head.
//!
//! ## Variants
//!
//! - [`Fifo`](LeafPolicy::Fifo) — an intrusive FIFO over leaves: O(1) per
//!   hook, **zero heap allocation in steady state** (a pre-sized link
//!   array). A node that *re-becomes* a leaf appends at the tail.
//! - [`Tick`](LeafPolicy::Tick) — a `BTreeMap` ordered by a monotonic
//!   per-node insertion tick: a re-leafed node retains its *original*
//!   position. This is the historical lineage-backend behavior and the
//!   default. Costs O(log n) per hook and B-tree node churn — it is the
//!   only structure here that is not pre-sized.
//! - [`Valued`](LeafPolicy::Valued) — a sampled-min value scorer
//!   (frequency × recency × compaction-discount × fan-out boost) with a
//!   hard poison FIFO. O(K) per victim, no global order. See
//!   [`valued`](super::valued).
//!
//! `Fifo`/`Tick` ignore the `seq_hash` on `on_node_inserted` and the
//! poison hooks; only `Valued` uses them.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::valued::{ScorerParams, ValuedPolicy};
use crate::blocks::SequenceHash;
use crate::branch_tracker::BranchOracle;
use crate::tinylfu::FrequencyTracker;

/// Leaf-eviction ordering strategy for `LineageBackend`. See the module docs.
pub(crate) enum LeafPolicy {
    Fifo(FifoPolicy),
    Tick(TickPolicy),
    Valued(ValuedPolicy),
}

/// Policy-side half of a read-only
/// [`InactiveFeatures`](crate::pools::InactiveFeatures) snapshot for one arena
/// slot (R7a §2.2). The *structural* half — `is_leaf` — and the peek-relative
/// `evict_rank` belong to the backend's graph, not to the leaf policy, so the
/// backend fills those in.
///
/// The all-`None` [`Default`] is the honest answer for a policy that tracks no
/// such signal (and for an untracked slot), never a zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LeafAdvice {
    pub(crate) poisoned: bool,
    pub(crate) age_ticks: Option<u64>,
    pub(crate) freq_estimate: Option<u32>,
    pub(crate) max_fanout: Option<u32>,
}

impl LeafPolicy {
    /// Intrusive-FIFO policy, pre-sized for `capacity` slots.
    pub(crate) fn fifo(capacity: usize) -> Self {
        Self::Fifo(FifoPolicy::with_capacity(capacity))
    }

    /// Insertion-tick `BTreeMap` policy, pre-sized for `capacity` slots.
    pub(crate) fn tick(capacity: usize) -> Self {
        Self::Tick(TickPolicy::with_capacity(capacity))
    }

    /// Sampled-min valued policy, pre-sized for `capacity` slots. `sketch`/`oracle` are
    /// optional — with neither, the score degenerates to pure recency (LRU).
    pub(crate) fn valued(
        capacity: usize,
        sketch: Option<Arc<dyn FrequencyTracker<u128>>>,
        oracle: Option<Arc<dyn BranchOracle>>,
        params: ScorerParams,
    ) -> Self {
        Self::Valued(ValuedPolicy::new(capacity, sketch, oracle, params))
    }

    /// A slot just became a `Real` node (fresh insert or ghost promotion). `seq_hash` is
    /// only consumed by [`Valued`](Self::Valued) (frequency/oracle lookup keys).
    pub(crate) fn on_node_inserted(&mut self, idx: u32, seq_hash: SequenceHash) {
        match self {
            Self::Fifo(_) => {} // FIFO assigns nothing at insert time
            Self::Tick(p) => p.on_node_inserted(idx),
            Self::Valued(p) => p.on_node_inserted(idx, seq_hash),
        }
    }

    /// A `Real` node is now a leaf — add it to the eviction order.
    pub(crate) fn on_leaf_added(&mut self, idx: u32) {
        match self {
            Self::Fifo(p) => p.on_leaf_added(idx),
            Self::Tick(p) => p.on_leaf_added(idx),
            Self::Valued(p) => p.on_leaf_added(idx),
        }
    }

    /// A `Real` leaf gained a child — remove it from the eviction order but
    /// keep its per-node ordering state for a possible later re-leafing.
    pub(crate) fn on_leaf_demoted(&mut self, idx: u32) {
        match self {
            Self::Fifo(p) => p.unlink(idx),
            Self::Tick(p) => p.on_leaf_demoted(idx),
            Self::Valued(p) => p.on_leaf_demoted(idx),
        }
    }

    /// A `Real` node left the graph — drop it from the eviction order and
    /// clear its per-node state so a recycled slot starts fresh.
    pub(crate) fn on_node_removed(&mut self, idx: u32) {
        match self {
            Self::Fifo(p) => p.unlink(idx),
            Self::Tick(p) => p.on_node_removed(idx),
            Self::Valued(p) => p.on_node_removed(idx),
        }
    }

    /// Slot index of the next block to evict, or `None` if no leaves. `&mut self` because
    /// [`Valued`](Self::Valued) advances its sampling RNG.
    pub(crate) fn next_victim(&mut self) -> Option<u32> {
        match self {
            Self::Fifo(p) => p.next_victim(),
            Self::Tick(p) => p.next_victim(),
            Self::Valued(p) => p.next_victim(),
        }
    }

    /// Mark slot `idx` poisoned (evict-first). No-op for `Fifo`/`Tick`, which do not
    /// support poisoning.
    pub(crate) fn mark_poisoned(&mut self, idx: u32) {
        if let Self::Valued(p) = self {
            p.mark_poisoned(idx);
        }
    }

    /// Peak fan-out for `seq_hash` as a parent, via `Valued`'s oracle; `None` otherwise.
    /// The backend's poison walk stops at the first `≥ 2` ancestor.
    pub(crate) fn max_fanout_of(&self, seq_hash: SequenceHash) -> Option<u32> {
        match self {
            Self::Valued(p) => p.max_fanout_of(seq_hash),
            _ => None,
        }
    }

    /// Read-only peek: slot indices of the next victims in eviction order,
    /// worst-first, up to `max`. Takes `&self` — no clock stamp, no RNG draw,
    /// no reordering (the determinism invariant on
    /// [`InactiveIndex::peek_victims`](crate::pools::InactiveIndex::peek_victims)).
    ///
    /// `Fifo`/`Tick` return the exact head of their total order. `Valued` has
    /// no total order, so it returns its poison set followed by a bounded
    /// best-effort minimum — see [`ValuedPolicy::peek_slots`].
    pub(crate) fn peek_slots(&self, max: usize) -> Vec<u32> {
        match self {
            Self::Fifo(p) => p.peek_slots(max),
            Self::Tick(p) => p.peek_slots(max),
            Self::Valued(p) => p.peek_slots(max),
        }
    }

    /// Read-only per-node advice for slot `idx` — valid for interior nodes as
    /// well as leaves. All-`None`/`false` for a slot this policy does not
    /// track, and for policies that track no such signal (`Fifo` has ordering
    /// but no age; neither `Fifo` nor `Tick` has a sketch, oracle, or poison
    /// bit).
    pub(crate) fn advice_for(&self, idx: u32) -> LeafAdvice {
        match self {
            Self::Fifo(_) => LeafAdvice::default(),
            Self::Tick(p) => LeafAdvice {
                age_ticks: p.age_of(idx),
                ..LeafAdvice::default()
            },
            Self::Valued(p) => p.advice_for(idx),
        }
    }

    /// Number of currently-evictable leaves. Test-only.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Fifo(p) => p.len(),
            Self::Tick(p) => p.queue.len(),
            Self::Valued(p) => p.len(),
        }
    }

    /// Whether slot `idx` is marked poisoned (`Valued` only). Test-only —
    /// production readers get the bit through [`Self::advice_for`].
    #[cfg(test)]
    pub(crate) fn test_is_poisoned(&self, idx: u32) -> bool {
        match self {
            Self::Valued(p) => p.test_is_poisoned(idx),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// FIFO
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct FifoLink {
    prev: Option<u32>,
    next: Option<u32>,
}

/// Intrusive doubly-linked FIFO over leaf slots. `links[idx]` is `Some`
/// exactly while slot `idx` is a leaf in the list; the array is pre-sized
/// from the capacity hint, so steady-state hooks do not allocate.
pub(crate) struct FifoPolicy {
    links: Vec<Option<FifoLink>>,
    head: Option<u32>,
    tail: Option<u32>,
}

impl FifoPolicy {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            links: Vec::with_capacity(capacity),
            head: None,
            tail: None,
        }
    }

    /// Grow the link array to cover `idx` (only past the capacity hint).
    fn ensure(&mut self, idx: u32) {
        if idx as usize >= self.links.len() {
            self.links.resize(idx as usize + 1, None);
        }
    }

    fn on_leaf_added(&mut self, idx: u32) {
        self.ensure(idx);
        debug_assert!(
            self.links[idx as usize].is_none(),
            "FifoPolicy: leaf {idx} added while already linked"
        );
        self.links[idx as usize] = Some(FifoLink {
            prev: self.tail,
            next: None,
        });
        match self.tail {
            Some(t) => self.links[t as usize].as_mut().unwrap().next = Some(idx),
            None => self.head = Some(idx),
        }
        self.tail = Some(idx);
    }

    /// Unlink `idx` from the list. No-op if it is not currently a leaf —
    /// `on_node_removed` is called for interior nodes too.
    fn unlink(&mut self, idx: u32) {
        let Some(link) = self.links.get_mut(idx as usize).and_then(Option::take) else {
            return;
        };
        match link.prev {
            Some(p) => self.links[p as usize].as_mut().unwrap().next = link.next,
            None => self.head = link.next,
        }
        match link.next {
            Some(n) => self.links[n as usize].as_mut().unwrap().prev = link.prev,
            None => self.tail = link.prev,
        }
    }

    fn next_victim(&self) -> Option<u32> {
        self.head
    }

    /// Read-only walk from the head — the exact FIFO eviction order, at O(max).
    /// Not required by R7a §3.2 (which names only `Tick` and `Valued`), but the
    /// walk is exact and cheap, and a supported backend silently reporting "no
    /// candidates" would be an unsanctioned degrade of the consumer contract.
    fn peek_slots(&self, max: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(max.min(self.links.len()));
        let mut cur = self.head;
        while out.len() < max {
            let Some(idx) = cur else { break };
            out.push(idx);
            cur = self
                .links
                .get(idx as usize)
                .copied()
                .flatten()
                .and_then(|link| link.next);
        }
        out
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        let mut n = 0;
        let mut cur = self.head;
        while let Some(i) = cur {
            n += 1;
            cur = self.links[i as usize].unwrap().next;
        }
        n
    }
}

// ---------------------------------------------------------------------------
// Tick
// ---------------------------------------------------------------------------

/// `BTreeMap` of `(insertion_tick, slot)` ordered ascending. A node's tick
/// is assigned once, at Real-ification, and survives leaf→interior→leaf
/// transitions — so a re-leafed node returns to its original eviction
/// position. Reproduces the historical lineage-backend ordering exactly.
pub(crate) struct TickPolicy {
    /// `ticks[idx]` is the node's insertion tick while it is `Real`, `None`
    /// once removed (so a recycled slot is re-ticked on its next insert).
    ticks: Vec<Option<u64>>,
    /// Currently-evictable leaves, ordered by `(tick, slot)`. Ticks are
    /// unique per node, so `slot` only keeps the key `Ord` total — it never
    /// actually breaks a tie.
    queue: BTreeMap<(u64, u32), ()>,
    next_tick: u64,
}

impl TickPolicy {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            ticks: Vec::with_capacity(capacity),
            queue: BTreeMap::new(),
            next_tick: 0,
        }
    }

    fn ensure(&mut self, idx: u32) {
        if idx as usize >= self.ticks.len() {
            self.ticks.resize(idx as usize + 1, None);
        }
    }

    fn on_node_inserted(&mut self, idx: u32) {
        self.ensure(idx);
        let tick = self.next_tick;
        self.next_tick += 1;
        self.ticks[idx as usize] = Some(tick);
    }

    fn on_leaf_added(&mut self, idx: u32) {
        let tick =
            self.ticks[idx as usize].expect("TickPolicy: on_leaf_added before on_node_inserted");
        self.queue.insert((tick, idx), ());
    }

    fn on_leaf_demoted(&mut self, idx: u32) {
        // Leave `ticks[idx]` set — a later re-leafing restores this exact
        // `(tick, idx)` key, putting the node back in its original spot.
        if let Some(tick) = self.ticks[idx as usize] {
            self.queue.remove(&(tick, idx));
        }
    }

    fn on_node_removed(&mut self, idx: u32) {
        // `take()` clears the tick; the `queue.remove` is a no-op for an
        // interior node that was demoted before being removed.
        if let Some(tick) = self.ticks[idx as usize].take() {
            self.queue.remove(&(tick, idx));
        }
    }

    fn next_victim(&self) -> Option<u32> {
        self.queue.first_key_value().map(|(&(_, idx), _)| idx)
    }

    /// Read-only prefix of the `(tick, slot)` order — exact and deterministic,
    /// at O(max) B-tree steps.
    ///
    /// This is the eviction order of the *current* leaf set. Draining the pool
    /// can still diverge after the first victim: removing a leaf may re-leaf
    /// its parent, which returns at its own (older) tick and jumps ahead of the
    /// leaves peeked behind it.
    fn peek_slots(&self, max: usize) -> Vec<u32> {
        self.queue.keys().take(max).map(|&(_, idx)| idx).collect()
    }

    /// Pool-logical age of slot `idx`: ticks elapsed since it was Real-ified.
    /// `None` once the node leaves the graph (its tick is cleared). Never
    /// indexes — a read-only API must not inherit the mutating hooks' panic on
    /// an out-of-range slot.
    fn age_of(&self, idx: u32) -> Option<u64> {
        self.ticks
            .get(idx as usize)
            .copied()
            .flatten()
            .map(|tick| self.next_tick.saturating_sub(tick))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIFO: re-adding a node appends at the tail; unlink is order-preserving.
    #[test]
    fn fifo_order_and_unlink() {
        let mut p = FifoPolicy::with_capacity(8);
        for i in 0..4 {
            p.on_leaf_added(i);
        }
        assert_eq!(p.next_victim(), Some(0));
        assert_eq!(p.len(), 4);

        // Unlink the middle — order of the rest is preserved.
        p.unlink(2);
        assert_eq!(p.len(), 3);
        p.unlink(0);
        assert_eq!(p.next_victim(), Some(1));

        // Re-add 0 — it goes to the tail, not its old position.
        p.on_leaf_added(0);
        assert_eq!(p.next_victim(), Some(1)); // still 1 at head
        // Drain: 1, 3, 0.
        let mut order = Vec::new();
        while let Some(v) = p.next_victim() {
            order.push(v);
            p.unlink(v);
        }
        assert_eq!(order, vec![1, 3, 0]);
    }

    /// Tick: a demoted-then-re-added node returns to its ORIGINAL position.
    #[test]
    fn tick_re_leafed_node_keeps_original_position() {
        let mut p = TickPolicy::with_capacity(8);
        // Insert order (ticks): 0->t0, 1->t1, 2->t2.
        for i in 0..3 {
            p.on_node_inserted(i);
            p.on_leaf_added(i);
        }
        assert_eq!(p.next_victim(), Some(0));

        // Demote 0 (gained a child), then re-add it.
        p.on_leaf_demoted(0);
        assert_eq!(p.next_victim(), Some(1)); // 0 temporarily out
        p.on_leaf_added(0);
        // 0 keeps tick 0 → back at the head, ahead of 1 and 2.
        assert_eq!(p.next_victim(), Some(0));

        // Removing 0 entirely clears its tick; a recycled slot 0 re-ticks
        // to the END, not the front.
        p.on_node_removed(0);
        p.on_node_inserted(0);
        p.on_leaf_added(0);
        let mut order = Vec::new();
        while let Some(v) = p.next_victim() {
            order.push(v);
            p.on_node_removed(v);
        }
        assert_eq!(order, vec![1, 2, 0]);
    }
}
