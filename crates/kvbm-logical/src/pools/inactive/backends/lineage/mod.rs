// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lineage-aware inactive index — a slab-backed parent/child graph that
//! evicts only from leaves.
//!
//! # Structure
//!
//! Every node lives in a pre-sized slab. Real nodes represent inactive
//! blocks, and ghosts represent live blocks. A reset frees the ghost and
//! moves its children to an ordered waiting table until their parent returns.
//!
//! The node's place records its parent slab index or the fragment it waits
//! for. Child lists and waiting lists share the intrusive sibling link.
//!
//! A single `index: HashMap<(position, fragment), u32>` resolves a
//! `(position, fragment)` pair to a slot — needed because lineage
//! navigation is by *fragment* (a child's hash only carries its parent's
//! fragment, never the parent's full hash). A node is keyed by
//! `parent_fragment_for_child_position(position + 1)` — the fragment width
//! its children compute as `parent_hash_fragment` — so child→parent
//! lookups match exactly. The map uses `PairHasher` and reserves room for
//! churn at the current block capacity. Churn can still rebuild the map
//! in place when tombstones use up the growth budget.
//!
//! # Leaf eviction ordering
//!
//! *Which* leaf is evicted first is delegated to a pluggable [`LeafPolicy`]
//! (see [`eviction`]). The default is `Tick` — a `BTreeMap` keyed on a
//! per-node insertion tick, the historical behavior, where a node that
//! *re-becomes* a leaf returns to its original position. The `Fifo`
//! variant is O(1) and allocation-free but appends a re-leafed node at the
//! tail instead; it is opt-in via `with_lineage_backend_eviction`.

#[cfg(test)]
mod advice_tests;
mod eviction;
mod hold;
#[cfg(test)]
mod trace_tests;
mod valued;

pub(crate) use eviction::LeafPolicy;
pub use valued::ScorerParams;

use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasher, Hasher};

use dynamo_tokens::PositionalLineageHash;

use crate::BlockId;
use crate::blocks::SequenceHash;
use crate::pools::advice::{InactiveFeatures, evict_rank_for};
use crate::pools::store::{ExactReclaimPlanError, InactiveIndex, room};

// ---------------------------------------------------------------------------
// `(position, fragment)` index hasher
// ---------------------------------------------------------------------------

/// Hand-rolled mixer for the `(u64, u64)` index key. The fragment word is
/// already a well-mixed hash fragment and `position` is a small int;
/// SipHash over the pair would be wasted work on the lookup hot path. A
/// `(u64, u64)` derives `Hash` as two `write_u64` calls, so an FxHash-style
/// rotate-xor-multiply accumulator over those two words is sufficient and
/// cheap. `write` (the byte-slice path) is never exercised by `(u64, u64)`.
#[derive(Default)]
struct PairHasher(u64);

impl Hasher for PairHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write_u64(&mut self, v: u64) {
        // FxHash-style: rotate to spread bits across words, xor in the new
        // word, multiply by an odd constant to avalanche.
        const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(K);
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("(position, fragment) keys hash via write_u64, not the byte-slice path");
    }
}

#[derive(Default, Clone)]
struct PairBuildHasher;

impl BuildHasher for PairBuildHasher {
    type Hasher = PairHasher;
    fn build_hasher(&self) -> PairHasher {
        PairHasher::default()
    }
}

type IndexMap = HashMap<(u64, u64), u32, PairBuildHasher>;

// ---------------------------------------------------------------------------
// Slab
// ---------------------------------------------------------------------------

/// Payload of a slab slot.
enum SlotData {
    /// A real inactive block.
    Real {
        block_id: BlockId,
        seq_hash: SequenceHash,
    },
    /// Placeholder backed by a primary or held block.
    Ghost,
    /// Slot is on the free list; `next_sibling` is the free-list link.
    Free,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Place {
    Unlinked,
    Child(u32),
    Waiting(u64),
}

#[derive(Clone, Copy)]
enum Removal {
    Keep,
    Evict,
}

/// One arena slot. Graph edges are `u32` slab indices. `position` /
/// `fragment` are stored on every slot (real *and* ghost) because a ghost
/// has no `seq_hash` yet still needs its index key to remove itself during
/// pruning. Leaf-eviction ordering state lives in the [`LeafPolicy`], not
/// here, so the slot stays policy-agnostic.
struct LineageSlot {
    data: SlotData,
    position: u64,
    fragment: u64,
    place: Place,
    /// Head of this node's intrusive child list.
    first_child: Option<u32>,
    /// Next sibling in the parent's child list. Reused as the free-list
    /// link while the slot is `Free`.
    next_sibling: Option<u32>,
}

impl LineageSlot {
    fn key(&self) -> (u64, u64) {
        (self.position, self.fragment)
    }

    fn parent(&self) -> Option<u32> {
        match self.place {
            Place::Child(parent) => Some(parent),
            _ => None,
        }
    }

    fn is_leaf(&self) -> bool {
        self.first_child.is_none()
    }
}

pub(crate) struct LineageBackend {
    slots: Vec<LineageSlot>,
    /// Free-list head; links through `LineageSlot::next_sibling`.
    free_head: Option<u32>,
    /// `(position, fragment)` → slot index, for parent resolution on
    /// insert and target resolution on remove.
    index: IndexMap,
    /// Largest clean entry capacity observed so far.
    index_full: usize,
    /// Missing parent key to the head of its waiting list.
    waiting: BTreeMap<(u64, u64), u32>,
    /// Pluggable leaf-eviction ordering. The backend feeds it the
    /// inserted / leaf-added / leaf-demoted / removed transitions and asks
    /// it for the next eviction victim.
    leaves: LeafPolicy,
    /// Number of `Real` nodes (ghosts excluded).
    count: usize,
    /// Test-only count of loop iterations in `unplace` since the last reset.
    #[cfg(test)]
    prune_iters: u64,
    #[cfg(test)]
    list_steps: usize,
    #[cfg(test)]
    child_writes: usize,
}

impl Default for LineageBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl LineageBackend {
    /// Create with no pre-sized capacity and the default (`Tick`) eviction
    /// policy. Production builds go through [`with_policy`](Self::with_policy).
    pub(crate) fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create pre-sized for `capacity` real blocks with the default
    /// (`Tick`) eviction policy.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self::with_policy(capacity, LeafPolicy::tick(capacity))
    }

    /// Reserve the slab and leaf policy from the initial block capacity.
    pub(crate) fn with_policy(capacity: usize, mut leaves: LeafPolicy) -> Self {
        let slots = Vec::with_capacity(capacity);
        let index = IndexMap::with_capacity_and_hasher(room(capacity), PairBuildHasher);
        leaves.reserve(slots.capacity());
        Self {
            slots,
            free_head: None,
            index_full: index.capacity(),
            index,
            waiting: BTreeMap::new(),
            leaves,
            count: 0,
            #[cfg(test)]
            prune_iters: 0,
            #[cfg(test)]
            list_steps: 0,
            #[cfg(test)]
            child_writes: 0,
        }
    }

    // ---- slab alloc / free ----

    /// Place `slot` into a recycled or freshly-pushed arena cell.
    fn alloc_slot(&mut self, slot: LineageSlot) -> u32 {
        match self.free_head {
            Some(idx) => {
                self.free_head = self.slots[idx as usize].next_sibling;
                self.slots[idx as usize] = slot;
                idx
            }
            None => {
                let idx = self.slots.len();
                let capacity = self.slots.capacity();
                debug_assert!(
                    idx <= u32::MAX as usize,
                    "lineage slab exceeded u32 index space"
                );
                self.slots.push(slot);
                if idx == capacity {
                    self.leaves.reserve(self.slots.capacity());
                }
                idx as u32
            }
        }
    }

    /// Return a slot to the free list. Other fields are left stale — the
    /// next `alloc_slot` overwrites the cell wholesale.
    fn free_slot(&mut self, idx: u32) {
        self.slots[idx as usize].data = SlotData::Free;
        self.slots[idx as usize].next_sibling = self.free_head;
        self.free_head = Some(idx);
    }

    /// Unlink one node and return the list's new head.
    fn unlink_sibling(&mut self, head: u32, index: u32) -> Option<u32> {
        let mut current = head;
        let mut previous = None;
        loop {
            #[cfg(test)]
            {
                self.list_steps += 1;
            }
            if current == index {
                break;
            }
            previous = Some(current);
            current = self.slots[current as usize]
                .next_sibling
                .expect("node missing from its lineage list");
        }
        let next = self.slots[index as usize].next_sibling.take();
        if let Some(previous) = previous {
            self.slots[previous as usize].next_sibling = next;
            Some(head)
        } else {
            next
        }
    }

    fn adopt_waiting(&mut self, parent: u32) {
        let key = self.slots[parent as usize].key();
        let head = self.waiting.remove(&key);
        self.slots[parent as usize].first_child = head;
        let mut child = head;
        while let Some(index) = child {
            let slot = &mut self.slots[index as usize];
            slot.place = Place::Child(parent);
            child = slot.next_sibling;
        }
    }

    fn release_children(&mut self, parent: u32) {
        let slot = &mut self.slots[parent as usize];
        let key = slot.key();
        let Some(head) = slot.first_child.take() else {
            return;
        };
        let mut child = Some(head);
        while let Some(index) = child {
            let slot = &mut self.slots[index as usize];
            slot.place = Place::Waiting(key.1);
            child = slot.next_sibling;
            #[cfg(test)]
            {
                self.child_writes += 1;
            }
        }
        let previous = self.waiting.insert(key, head);
        debug_assert!(previous.is_none());
    }

    fn place_node(&mut self, index: u32, key: (u64, u64)) {
        if let Some(&parent) = self.index.get(&key) {
            let head = self.slots[parent as usize].first_child.replace(index);
            let slot = &mut self.slots[index as usize];
            slot.place = Place::Child(parent);
            slot.next_sibling = head;
            if head.is_none() && matches!(self.slots[parent as usize].data, SlotData::Real { .. }) {
                self.leaves.on_leaf_demoted(parent);
            }
        } else {
            let slot = &mut self.slots[index as usize];
            slot.place = Place::Waiting(key.1);
            slot.next_sibling = self.waiting.insert(key, index);
        }
    }

    /// A childless ghost stays in the index while its backing block is live.
    /// Unlink it so it does not keep its ancestors out of the leaf order.
    fn unplace(&mut self, mut index: u32) {
        loop {
            #[cfg(test)]
            {
                self.prune_iters += 1;
            }
            let slot = &mut self.slots[index as usize];
            match std::mem::replace(&mut slot.place, Place::Unlinked) {
                Place::Unlinked => return,
                Place::Waiting(fragment) => {
                    let key = (slot.position - 1, fragment);
                    let head = self.waiting[&key];
                    match self.unlink_sibling(head, index) {
                        None => {
                            self.waiting.remove(&key);
                        }
                        Some(next) if next != head => {
                            self.waiting.insert(key, next);
                        }
                        _ => {}
                    }
                    return;
                }
                Place::Child(parent) => {
                    let head = self.slots[parent as usize]
                        .first_child
                        .expect("linked child");
                    let next = self.unlink_sibling(head, index);
                    let parent_slot = &mut self.slots[parent as usize];
                    parent_slot.first_child = next;
                    if next.is_some() {
                        return;
                    }
                    if matches!(parent_slot.data, SlotData::Real { .. }) {
                        self.leaves.on_leaf_added(parent);
                        return;
                    }
                    debug_assert!(matches!(parent_slot.data, SlotData::Ghost));
                    index = parent;
                }
            }
        }
    }

    fn remove_slot(&mut self, index: u32) {
        self.release_children(index);
        self.unplace(index);
        self.index.remove(&self.slots[index as usize].key());
        self.free_slot(index);
    }

    fn add_node(&mut self, key: (u64, u64), data: SlotData) -> u32 {
        let index = self.alloc_slot(LineageSlot {
            data,
            position: key.0,
            fragment: key.1,
            place: Place::Unlinked,
            first_child: None,
            next_sibling: None,
        });
        self.index.insert(key, index);
        self.adopt_waiting(index);
        index
    }

    // ---- core mutation ----

    fn insert_inner(&mut self, seq_hash: SequenceHash, block_id: BlockId) {
        let position = seq_hash.position();
        // Key this node by the same fragment width its children will store
        // as their `parent_hash_fragment`, so child→parent lookups match
        // exactly. PLH fragment widths vary by mode (54/46/38 bits); a node
        // at `position` must be keyed at the width a child at `position + 1`
        // uses for its parent fragment — not this node's own
        // `current_hash_fragment` width.
        let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
        let parent_fragment = if position > 0 {
            Some(seq_hash.parent_hash_fragment())
        } else {
            None
        };

        // 1. Find-or-create this node. An existing entry must be a Ghost
        //    (a real-vs-real hit is a duplicate or a collision bug).
        let node_idx = match self.index.get(&(position, fragment)) {
            Some(&idx) => {
                match self.slots[idx as usize].data {
                    SlotData::Ghost => {
                        self.slots[idx as usize].data = SlotData::Real { block_id, seq_hash };
                        self.count += 1;
                        self.leaves.on_node_inserted(idx, seq_hash);
                    }
                    SlotData::Real {
                        seq_hash: existing, ..
                    } => {
                        if existing.as_u128() == seq_hash.as_u128() {
                            panic!(
                                "Duplicate insertion detected! position={}, fragment={:#x}, \
                                 hash={:#032x}.",
                                position,
                                fragment,
                                seq_hash.as_u128()
                            );
                        } else {
                            panic!(
                                "Hash collision detected! position={}, fragment={:#x}, \
                                 existing_hash={:#032x}, new_hash={:#032x}.",
                                position,
                                fragment,
                                existing.as_u128(),
                                seq_hash.as_u128()
                            );
                        }
                    }
                    SlotData::Free => unreachable!("index points at a freed slot"),
                }
                idx
            }
            None => {
                let idx =
                    self.add_node((position, fragment), SlotData::Real { block_id, seq_hash });
                self.count += 1;
                self.leaves.on_node_inserted(idx, seq_hash);
                idx
            }
        };

        // A ghost that already has a place keeps it when promoted.
        if let Some(fragment) = parent_fragment
            && self.slots[node_idx as usize].place == Place::Unlinked
        {
            self.place_node(node_idx, (position - 1, fragment));
        }

        // 3. If this node is a Real leaf, it enters the eviction order.
        //    (A promoted ghost already has children — not a leaf.)
        if self.slots[node_idx as usize].is_leaf() {
            self.leaves.on_leaf_added(node_idx);
        }
    }

    /// Look up a node by its full `SequenceHash` and, if it is the real
    /// block stored under that `(position, fragment)` key, remove it.
    ///
    /// The full-hash verification matters: the `(position, fragment)` key
    /// is not unique — distinct `PositionalLineageHash`es can share it
    /// (same fragment + position, different parent), so a key-only match
    /// would let a lookup for one PLH delete another's block.
    fn remove_by_hash(
        &mut self,
        lineage_hash: &PositionalLineageHash,
        removal: Removal,
    ) -> Option<(SequenceHash, BlockId)> {
        let position = lineage_hash.position();
        let fragment = lineage_hash.parent_fragment_for_child_position(position + 1);
        let idx = *self.index.get(&(position, fragment))?;
        match self.slots[idx as usize].data {
            SlotData::Real { seq_hash, .. } if seq_hash == *lineage_hash => {
                Some(self.remove_node_at(idx, removal))
            }
            _ => None,
        }
    }

    /// Keep the slot on a cache hit or hold. Eviction frees it.
    fn remove_node_at(&mut self, idx: u32, removal: Removal) -> (SequenceHash, BlockId) {
        let payload = match std::mem::replace(&mut self.slots[idx as usize].data, SlotData::Ghost) {
            SlotData::Real { seq_hash, block_id } => (seq_hash, block_id),
            _ => unreachable!("remove_node_at called on a non-Real slot"),
        };
        self.count -= 1;
        // The node has left the graph: drop it from the eviction order
        // (no-op if it was an interior node) and clear its policy state.
        self.leaves.on_node_removed(idx);

        match removal {
            Removal::Keep if self.slots[idx as usize].is_leaf() => self.unplace(idx),
            Removal::Keep => {}
            Removal::Evict => self.remove_slot(idx),
        }
        payload
    }

    /// Poison the single-owner suffix ending at the leaf `seq_hash`: walk leaf → parent,
    /// marking each node poisoned in the leaf policy (so it is evicted first once it is a
    /// leaf), and stop at — **without** poisoning — the first shared branch point. A no-op
    /// unless `seq_hash` names a resident `Real` node; the leaf policy itself ignores the
    /// marks unless it is the [`Valued`](eviction::LeafPolicy::Valued) arm.
    ///
    /// Interior shared ancestors are structurally unevictable non-leaves and are never
    /// poisoned; the walk halts at the first ancestor that is a branch point.
    // Reached in production via `InactiveIndex::poison`, wired to the client compaction
    // hint through `BlockManager::poison_lineage` (EV-PR4).
    fn poison_suffix(&mut self, seq_hash: SequenceHash) {
        let position = seq_hash.position();
        let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
        let Some(&start) = self.index.get(&(position, fragment)) else {
            return;
        };
        // Only the Real node stored under the FULL hash is a valid poison target (the
        // `(position, fragment)` key alone can collide across distinct PLHs).
        match self.slots[start as usize].data {
            SlotData::Real {
                seq_hash: stored, ..
            } if stored == seq_hash => {}
            _ => return,
        }

        let mut cur = start;
        loop {
            // Stop at a ghost placeholder: it has no stored hash, so its high-water branch
            // identity is unknowable (a formerly-shared branch point that was evicted becomes
            // a ghost). Walking past it could poison a real shared ancestor above. Ghosts are
            // absent in the common in-order case; stopping here only ever *under*-poisons.
            if matches!(self.slots[cur as usize].data, SlotData::Ghost) {
                break;
            }
            if self.is_branch_point(cur) {
                break; // shared branch point — never poisoned
            }
            self.leaves.mark_poisoned(cur);
            match self.slots[cur as usize].parent() {
                Some(parent) => cur = parent,
                None => break,
            }
        }
    }

    /// `(seq_hash, block_id)` of the `Real` node at `idx`; `None` for a ghost or
    /// freed slot.
    fn real_payload(&self, idx: u32) -> Option<(SequenceHash, BlockId)> {
        match self.slots.get(idx as usize)?.data {
            SlotData::Real { seq_hash, block_id } => Some((seq_hash, block_id)),
            _ => None,
        }
    }

    /// Read-only feature snapshot for the `Real` node at `idx`: the leaf
    /// policy's per-node advice plus the graph's own structural `is_leaf`, with
    /// the caller-supplied peek-relative rank. Mutates nothing.
    fn features_for(&self, idx: u32, evict_rank: Option<u8>) -> InactiveFeatures {
        let advice = self.leaves.advice_for(idx);
        InactiveFeatures {
            poisoned: advice.poisoned,
            // Structural truth from the graph, not from policy membership: an
            // interior node is unevictable no matter what the policy thinks.
            is_leaf: self.slots[idx as usize].is_leaf(),
            age_ticks: advice.age_ticks,
            freq_estimate: advice.freq_estimate,
            max_fanout: advice.max_fanout,
            evict_rank,
        }
    }

    /// A node is a (shared) branch point if it currently has ≥ 2 children, or — for a
    /// `Real` node — its monotone high-water `max_fanout` is ≥ 2 (a re-leafed branch point
    /// that may re-fork). A ghost has no hash, so only its current child count counts.
    fn is_branch_point(&self, idx: u32) -> bool {
        if let Some(first) = self.slots[idx as usize].first_child
            && self.slots[first as usize].next_sibling.is_some()
        {
            return true; // ≥ 2 live children right now
        }
        match self.slots[idx as usize].data {
            SlotData::Real { seq_hash, .. } => {
                self.leaves.max_fanout_of(seq_hash).is_some_and(|f| f >= 2)
            }
            _ => false,
        }
    }
}

impl InactiveIndex for LineageBackend {
    fn live_added(&mut self, hash: SequenceHash) {
        let position = hash.position();
        let key = (
            position,
            hash.parent_fragment_for_child_position(position + 1),
        );
        if !self.index.contains_key(&key) {
            self.add_node(key, SlotData::Ghost);
        }
    }

    fn forget_live(&mut self, hash: SequenceHash) {
        let position = hash.position();
        let key = (
            position,
            hash.parent_fragment_for_child_position(position + 1),
        );
        if let Some(&index) = self.index.get(&key)
            && matches!(self.slots[index as usize].data, SlotData::Ghost)
        {
            self.remove_slot(index);
        }
    }

    fn grow_capacity(&mut self, capacity: usize) {
        // Tombstones can lower the reported capacity without freeing buckets.
        self.index_full = self.index_full.max(self.index.capacity());
        let wanted = room(capacity);
        if wanted > self.index_full {
            self.index.reserve(wanted.saturating_sub(self.index.len()));
            self.index_full = self.index.capacity();
        }

        if capacity > self.slots.capacity() {
            self.slots.reserve_exact(capacity - self.slots.len());
            self.leaves.reserve(self.slots.capacity());
        }
    }

    #[cfg(test)]
    fn lineage_storage(&self) -> Option<(usize, usize, usize, usize, usize)> {
        Some((
            self.index.len(),
            self.index.capacity(),
            self.slots.len(),
            self.slots.capacity(),
            self.leaves.capacity(),
        ))
    }

    #[cfg(test)]
    fn audit_lineage(
        &self,
        live_keys: &std::collections::HashSet<(u64, u64)>,
        inactive: &std::collections::HashSet<(SequenceHash, BlockId)>,
        registered: usize,
        collisions: bool,
    ) -> Option<crate::pools::store::LineageAudit> {
        use std::collections::HashSet;

        let mut actual_inactive = HashSet::new();
        let mut linked = HashSet::new();
        let mut leaves = HashSet::new();
        let mut ghosts = 0;
        for (&key, &index) in &self.index {
            let slot = &self.slots[index as usize];
            assert_eq!(key, slot.key());
            match slot.data {
                SlotData::Ghost => {
                    assert!(live_keys.contains(&key), "unowned ghost at {key:?}");
                    assert!(!slot.is_leaf() || slot.place == Place::Unlinked);
                    ghosts += 1;
                }
                SlotData::Real { seq_hash, block_id } => {
                    assert!(actual_inactive.insert((seq_hash, block_id)));
                    if slot.is_leaf() {
                        leaves.insert(index);
                    }
                    if !collisions && slot.position > 0 {
                        let fragment = seq_hash.parent_hash_fragment();
                        let expected = self
                            .index
                            .get(&(slot.position - 1, fragment))
                            .map_or(Place::Waiting(fragment), |&parent| Place::Child(parent));
                        assert_eq!(slot.place, expected);
                    }
                }
                SlotData::Free => panic!("index points to a free slot"),
            }
            if slot.place == Place::Unlinked {
                assert!(slot.next_sibling.is_none());
            }
            let mut child = slot.first_child;
            while let Some(child_index) = child {
                let child_slot = &self.slots[child_index as usize];
                assert_eq!(child_slot.position, slot.position + 1);
                assert_eq!(child_slot.place, Place::Child(index));
                assert!(linked.insert(child_index), "repeated child link");
                child = child_slot.next_sibling;
            }
        }
        for (&key, &head) in &self.waiting {
            assert!(!self.index.contains_key(&key));
            let mut next = Some(head);
            while let Some(index) = next {
                let slot = &self.slots[index as usize];
                assert_eq!(slot.position, key.0 + 1);
                assert_eq!(slot.place, Place::Waiting(key.1));
                assert!(linked.insert(index), "repeated waiting link");
                next = slot.next_sibling;
            }
        }
        for (index, slot) in self.slots.iter().enumerate() {
            if !matches!(slot.data, SlotData::Free) {
                let index = index as u32;
                assert_eq!(self.index.get(&slot.key()), Some(&index));
                assert_eq!(linked.contains(&index), slot.place != Place::Unlinked);
            }
        }
        assert_eq!(&actual_inactive, inactive);
        assert_eq!(actual_inactive.len(), self.count);
        assert_eq!(self.index.len(), actual_inactive.len() + ghosts);
        assert!(self.index.len() <= registered);
        assert_eq!(self.leaves.len(), leaves.len());
        for leaf in self.leaves.peek_slots(leaves.len()) {
            assert!(leaves.remove(&leaf), "repeated or ineligible policy entry");
        }
        Some(crate::pools::store::LineageAudit {
            nodes: self.index.len(),
            ghosts,
            waiting_keys: self.waiting.len(),
            list_steps: self.list_steps,
            child_writes: self.child_writes,
        })
    }

    fn find_matches(
        &mut self,
        hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, BlockId)> {
        let mut matches = Vec::with_capacity(hashes.len());
        for hash in hashes {
            if let Some(pair) = self.remove_by_hash(hash, Removal::Keep) {
                matches.push(pair);
            } else {
                break;
            }
        }
        matches
    }

    fn find_match(&mut self, hash: SequenceHash, _touch: bool) -> Option<(SequenceHash, BlockId)> {
        self.remove_by_hash(&hash, Removal::Keep)
    }

    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        _touch: bool,
    ) -> Vec<(SequenceHash, BlockId)> {
        let mut matches = Vec::new();
        for hash in hashes {
            if let Some(pair) = self.remove_by_hash(hash, Removal::Keep) {
                matches.push(pair);
            }
        }
        matches
    }

    fn allocate(&mut self, count: usize) -> Vec<(SequenceHash, BlockId)> {
        let mut allocated = Vec::with_capacity(count);
        while allocated.len() < count {
            // Next leaf in policy order. `remove_node_at` drops it from the
            // policy and may expose its parent as the next victim.
            match self.leaves.next_victim() {
                Some(idx) => allocated.push(self.remove_node_at(idx, Removal::Evict)),
                None => break,
            }
        }
        allocated
    }

    fn insert(&mut self, seq_hash: SequenceHash, block_id: BlockId) {
        self.insert_inner(seq_hash, block_id);
    }

    fn rebind(&mut self, seq_hash: SequenceHash, src: BlockId, dst: BlockId) -> bool {
        let Some(index) = self.real_index(seq_hash, src) else {
            return false;
        };
        match &mut self.slots[index as usize].data {
            SlotData::Real { block_id, .. } => {
                *block_id = dst;
                true
            }
            _ => false,
        }
    }

    fn len(&self) -> usize {
        self.count
    }

    fn has(&self, seq_hash: SequenceHash) -> bool {
        let position = seq_hash.position();
        let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
        self.index.get(&(position, fragment)).is_some_and(|&idx| {
            match self.slots[idx as usize].data {
                SlotData::Real {
                    seq_hash: stored, ..
                } => stored == seq_hash,
                _ => false,
            }
        })
    }

    #[cfg(test)]
    fn contains(&self, seq_hash: SequenceHash, block_id: BlockId) -> bool {
        let position = seq_hash.position();
        let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
        self.index.get(&(position, fragment)).is_some_and(|&idx| {
            match self.slots[idx as usize].data {
                SlotData::Real {
                    seq_hash: stored,
                    block_id: stored_id,
                } => {
                    stored == seq_hash
                        && stored_id == block_id
                        && self.slots[idx as usize].is_leaf()
                }
                _ => false,
            }
        })
    }

    fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool {
        // Match on the full `SequenceHash` AND the block id — the
        // `(position, fragment)` key alone can collide across distinct PLHs.
        let position = seq_hash.position();
        let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
        let hit = self.index.get(&(position, fragment)).is_some_and(|&idx| {
            match self.slots[idx as usize].data {
                SlotData::Real {
                    seq_hash: stored,
                    block_id: stored_id,
                } => stored == seq_hash && stored_id == block_id,
                _ => false,
            }
        });
        if hit {
            self.remove_by_hash(&seq_hash, Removal::Evict).is_some()
        } else {
            false
        }
    }

    fn take_with_descendants(
        &mut self,
        seq_hash: SequenceHash,
        block_id: BlockId,
    ) -> Vec<(SequenceHash, BlockId)> {
        let position = seq_hash.position();
        let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
        let Some(&root) = self.index.get(&(position, fragment)) else {
            return Vec::new();
        };
        if !matches!(
            self.slots[root as usize].data,
            SlotData::Real {
                seq_hash: stored,
                block_id: stored_id,
            } if stored == seq_hash && stored_id == block_id
        ) {
            return Vec::new();
        }

        let mut order = Vec::new();
        let mut stack = vec![(root, false)];
        while let Some((idx, expanded)) = stack.pop() {
            if expanded {
                if matches!(self.slots[idx as usize].data, SlotData::Real { .. }) {
                    order.push(idx);
                }
                continue;
            }
            stack.push((idx, true));
            let mut children = Vec::new();
            let mut child = self.slots[idx as usize].first_child;
            while let Some(child_idx) = child {
                children.push(child_idx);
                child = self.slots[child_idx as usize].next_sibling;
            }
            for child_idx in children.into_iter().rev() {
                stack.push((child_idx, false));
            }
        }

        order
            .into_iter()
            .map(|idx| self.remove_node_at(idx, Removal::Evict))
            .collect()
    }

    fn take_complete_lineage(
        &mut self,
        seq_hash: SequenceHash,
        block_id: BlockId,
    ) -> Option<Vec<(SequenceHash, BlockId)>> {
        self.take_exact_complete_lineage(seq_hash, block_id)
    }

    fn complete_lineage(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
    ) -> Option<Vec<(SequenceHash, BlockId)>> {
        self.exact_complete_lineage(seq_hash, block_id)
    }

    fn supports_exact_reclaim(&self) -> bool {
        true
    }

    fn exact_block_id(&self, seq_hash: SequenceHash) -> Option<BlockId> {
        let position = seq_hash.position();
        let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
        let index = *self.index.get(&(position, fragment))?;
        match self.slots[index as usize].data {
            SlotData::Real {
                seq_hash: stored,
                block_id,
            } if stored == seq_hash => Some(block_id),
            _ => None,
        }
    }

    fn preflight_exact_reclaim(
        &self,
        victims: &[crate::ExactInactiveVictim],
    ) -> Result<(), ExactReclaimPlanError> {
        LineageBackend::preflight_exact_reclaim(self, victims)
    }

    fn poison(&mut self, seq_hash: SequenceHash) {
        self.poison_suffix(seq_hash);
    }

    #[cfg(test)]
    fn test_is_poisoned(&self, seq_hash: SequenceHash) -> bool {
        let position = seq_hash.position();
        let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
        match self.index.get(&(position, fragment)) {
            Some(&idx) => self.leaves.test_is_poisoned(idx),
            None => false,
        }
    }

    /// Up to `max` of the *currently evictable* leaves, worst-first by the
    /// leaf policy's ranking, with their features. Only leaves are ever
    /// returned — an interior node is structurally unevictable — so a consumer
    /// must reach interior nodes through [`Self::advice`] instead.
    ///
    /// # Why this is not a drain preview
    ///
    /// The ranking covers the leaf set as it stands. Evicting a leaf can
    /// re-leaf its parent (`on_leaf_added`) — a node this peek could not have
    /// listed, since it was interior at the time. `Tick`/`Fifo` re-admit it at
    /// its own older position, ahead of leaves listed behind it; `Valued`
    /// restamps `last_touch` there, so it re-enters as the *freshest* leaf and
    /// sinks to the back instead. Either way the drain diverges, so only the
    /// *head* is comparable to a real [`InactiveIndex::allocate`], and only where the
    /// policy is exact: `Tick` and `Fifo` always, `Valued` for its poison
    /// prefix (`Valued`'s unpoisoned pick is sampled, this scan is not). A
    /// result shorter than `max` means "no more candidates were ranked", never
    /// "the pool is empty" — `len` counts interior nodes too, and `Valued`
    /// scores only a bounded window of `leaf_dense` per call.
    ///
    /// `evict_rank` is the entry's position *within this returned slice*
    /// scaled to `[0, 255]`; it says nothing about the rest of the pool.
    /// Read-only: no clock stamp, no RNG draw, no reordering (R7a §3.4).
    fn peek_victims(&self, max: usize) -> Vec<(SequenceHash, BlockId, InactiveFeatures)> {
        let slots = self.leaves.peek_slots(max);
        let len = slots.len();
        slots
            .into_iter()
            .enumerate()
            .filter_map(|(rank, idx)| {
                let Some((seq_hash, block_id)) = self.real_payload(idx) else {
                    // The policy only ever holds `Real` leaves the backend
                    // announced; a ghost/free slot here is a bookkeeping bug.
                    debug_assert!(false, "leaf policy peeked a non-Real slot {idx}");
                    return None;
                };
                let features = self.features_for(idx, evict_rank_for(rank, len));
                Some((seq_hash, block_id, features))
            })
            .collect()
    }

    /// Point advice for `seq_hash`, O(1) through the `(position, fragment)`
    /// index. Membership is the same rule `poison_suffix` uses: the node must
    /// be `Real` *and* its stored full hash must equal `seq_hash` (the index
    /// key alone can collide across distinct PLHs). A ghost placeholder, a hash
    /// mismatch, or an absent key all report `None`.
    ///
    /// Interior nodes are visible here (with `is_leaf: false`) even though they
    /// never appear in [`Self::peek_victims`]. `evict_rank` is always `None` —
    /// rank is only defined within one peek batch.
    fn advice(&self, seq_hash: SequenceHash) -> Option<InactiveFeatures> {
        let position = seq_hash.position();
        let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
        let &idx = self.index.get(&(position, fragment))?;
        match self.slots[idx as usize].data {
            SlotData::Real {
                seq_hash: stored, ..
            } if stored == seq_hash => Some(self.features_for(idx, None)),
            _ => None,
        }
    }
}

#[cfg(test)]
impl LineageBackend {
    /// Count of loop iterations in `unplace` since the last reset.
    pub(crate) fn prune_iters(&self) -> u64 {
        self.prune_iters
    }

    /// Reset the `unplace` iteration count.
    pub(crate) fn reset_prune_iters(&mut self) {
        self.prune_iters = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::BlockSequenceBuilder;

    impl LineageBackend {
        /// Test-only: number of currently-evictable leaves.
        fn get_queue_len(&self) -> usize {
            self.leaves.len()
        }

        /// Test-only: whether the resident `Real` node for `seq_hash` is marked poisoned.
        fn test_is_poisoned(&self, seq_hash: SequenceHash) -> bool {
            let position = seq_hash.position();
            let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
            match self.index.get(&(position, fragment)) {
                Some(&idx) => self.leaves.test_is_poisoned(idx),
                None => false,
            }
        }

        /// Test-only: no live slots remain (all real + ghost nodes gone).
        fn is_graph_empty(&self) -> bool {
            self.index.is_empty()
        }
    }

    /// Build a chain of lineage hashes and return `(block_id, seq_hash)` pairs.
    fn create_chain(count: usize, offset: u32) -> Vec<(BlockId, SequenceHash)> {
        let tokens: Vec<u32> = (offset..offset + count as u32).collect();
        BlockSequenceBuilder::from_tokens(tokens)
            .with_block_size(1)
            .build()
    }

    fn create_blocks(count: usize) -> Vec<(BlockId, SequenceHash)> {
        create_chain(count, 0)
    }

    fn create_block(id: u32) -> (BlockId, SequenceHash) {
        BlockSequenceBuilder::from_tokens(vec![id])
            .with_block_size(1)
            .build()
            .into_iter()
            .next()
            .unwrap()
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn lineage_place_keeps_the_slab_cell_at_80_bytes() {
        assert_eq!(std::mem::size_of::<LineageSlot>(), 80);
    }

    #[test]
    fn lineage_policy_grows_with_slab() {
        const CAPACITY: usize = 4;
        let cases = [
            ("Tick", LeafPolicy::tick(CAPACITY)),
            ("Fifo", LeafPolicy::fifo(CAPACITY)),
            (
                "Valued",
                LeafPolicy::valued(CAPACITY, None, None, ScorerParams::default()),
            ),
        ];
        let roots = [10, 11, 12].map(|token| create_block(token).1);
        let chain = create_chain(2, 20);
        let parent = chain[0].1;
        let child = chain[1].1;

        for (name, policy) in cases {
            let mut backend = LineageBackend::with_policy(CAPACITY, policy);
            let capacities =
                |backend: &LineageBackend| (backend.slots.capacity(), backend.leaves.capacity());
            let mut trace = vec![capacities(&backend)];
            for (id, hash) in roots.into_iter().enumerate() {
                backend.insert(hash, id);
                trace.push(capacities(&backend));
            }

            // The child waits in slot 3. Register its parent in slot 4.
            backend.insert(child, 3);
            backend.live_added(parent);
            assert_eq!(backend.slots.len(), 5);
            trace.push(capacities(&backend));
            for hash in &roots[..2] {
                assert!(backend.remove_by_hash(hash, Removal::Evict).is_some());
                trace.push(capacities(&backend));
            }

            // Promoting the parent uses slot 4 with only three nodes left.
            let before = backend.leaves.capacity();
            backend.insert(parent, 0);
            trace.push(capacities(&backend));
            assert_eq!(backend.index.len(), 3);
            assert_eq!(backend.len(), 3);
            println!("{name} (slab, policy) capacity trace: {trace:?}");
            assert_eq!(
                backend.leaves.capacity(),
                before,
                "{name} grew on promotion"
            );
            for (slab, policy) in trace {
                assert!(policy >= slab, "{name} reserve {policy} trails slab {slab}");
            }

            backend.grow_capacity(3 * CAPACITY);
            assert!(backend.slots.capacity() >= 3 * CAPACITY);
            assert!(backend.leaves.capacity() >= backend.slots.capacity());
        }
    }

    #[test]
    fn test_leaf_insertion() {
        let mut backend = LineageBackend::new();
        let (id1, h1) = create_block(1);

        backend.insert(h1, id1);

        assert_eq!(backend.len(), 1);
        assert_eq!(backend.get_queue_len(), 1);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].1, 0);
        assert_eq!(backend.len(), 0);
        assert!(backend.is_graph_empty());
    }

    #[test]
    fn test_parent_child_insertion() {
        let mut backend = LineageBackend::new();

        let mut blocks = create_blocks(2);
        let (id1, h1) = blocks.remove(0);
        let (id2, h2) = blocks.remove(0);

        backend.insert(h1, id1);
        assert_eq!(backend.get_queue_len(), 1);

        backend.insert(h2, id2);
        assert_eq!(backend.len(), 2);
        assert_eq!(backend.get_queue_len(), 1);

        let allocated = backend.allocate(1);
        assert_eq!(allocated.len(), 1);
        assert_eq!(allocated[0].1, 1);

        assert_eq!(backend.get_queue_len(), 1);

        let allocated2 = backend.allocate(1);
        assert_eq!(allocated2.len(), 1);
        assert_eq!(allocated2[0].1, 0);
    }

    #[test]
    fn test_out_of_order_insertion() {
        let mut backend = LineageBackend::new();

        let mut chain = create_blocks(2);
        let (id2, h2) = chain.remove(1);
        backend.insert(h2, id2);
        assert_eq!(backend.len(), 1);
        assert_eq!(backend.get_queue_len(), 1);

        let mut chain2 = create_blocks(2);
        let (id1, h1) = chain2.remove(0);
        backend.insert(h1, id1);

        assert_eq!(backend.len(), 2);
        assert_eq!(backend.get_queue_len(), 1);

        let allocated = backend.allocate(1);
        assert_eq!(allocated[0].1, 1);

        assert_eq!(backend.get_queue_len(), 1);

        let allocated2 = backend.allocate(1);
        assert_eq!(allocated2[0].1, 0);
    }

    #[test]
    fn take_with_descendants_removes_leaf_to_root() {
        let mut backend = LineageBackend::new();
        let chain = create_chain(3, 0);

        for (id, hash) in &chain {
            backend.insert(*hash, *id);
        }

        assert_eq!(
            backend.take_with_descendants(chain[0].1, chain[0].0),
            vec![
                (chain[2].1, chain[2].0),
                (chain[1].1, chain[1].0),
                (chain[0].1, chain[0].0),
            ]
        );
        assert!(backend.is_graph_empty());
    }

    #[test]
    fn rebind_updates_a_parent_without_changing_its_lineage_position() {
        let mut backend = LineageBackend::new();
        let chain = create_chain(3, 0);
        for (id, hash) in &chain {
            backend.insert(*hash, *id);
        }

        let destination = 9;
        assert!(backend.rebind(chain[0].1, chain[0].0, destination));
        assert_eq!(
            backend.complete_lineage(chain[2].1, chain[2].0),
            Some(vec![
                (chain[0].1, destination),
                (chain[1].1, chain[1].0),
                (chain[2].1, chain[2].0),
            ])
        );
        assert_eq!(
            backend.take_with_descendants(chain[0].1, destination),
            vec![
                (chain[2].1, chain[2].0),
                (chain[1].1, chain[1].0),
                (chain[0].1, destination),
            ]
        );
    }

    #[test]
    fn take_with_descendants_stops_at_a_hole() {
        let mut backend = LineageBackend::new();
        let chain = create_chain(3, 0);

        backend.insert(chain[0].1, chain[0].0);
        backend.insert(chain[2].1, chain[2].0);

        assert_eq!(
            backend.take_with_descendants(chain[0].1, chain[0].0),
            vec![(chain[0].1, chain[0].0)]
        );
        assert_eq!(backend.len(), 1);
        assert!(backend.contains(chain[2].1, chain[2].0));
    }

    #[test]
    fn test_branching() {
        let mut backend = LineageBackend::new();

        let seq1 = create_chain(3, 0);
        let seq2 = create_chain(3, 5000);

        for (id, h) in seq1 {
            backend.insert(h, id);
        }
        for (id, h) in seq2 {
            backend.insert(h, id);
        }

        assert_eq!(backend.len(), 6);
        assert_eq!(backend.get_queue_len(), 2);

        let alloc1 = backend.allocate(1);
        assert_eq!(alloc1.len(), 1);
        assert_eq!(backend.len(), 5);

        assert_eq!(backend.get_queue_len(), 2);
    }

    /// Two interleaved 2-chains under the default `Tick` policy: a node
    /// that re-becomes a leaf returns to its *original* insertion-order
    /// position, so each chain's root is evicted right after its own leaf.
    /// (This is the historical ordering; `Fifo` would round-robin instead —
    /// see `re_leafed_node_goes_to_fifo_tail`.)
    #[test]
    fn test_interleaved_chains() {
        let mut backend = LineageBackend::new(); // default: Tick

        let mut chain1 = create_chain(2, 0);
        let (a_id, a_h) = chain1.remove(0);
        let (b_id, b_h) = chain1.remove(0);

        let mut chain2 = create_chain(2, 1000);
        let (x_id, x_h) = chain2.remove(0);
        let (y_id, y_h) = chain2.remove(0);

        backend.insert(a_h, a_id); // chain1 root  (block_id 0)
        backend.insert(b_h, b_id); // chain1 leaf  (block_id 1)
        backend.insert(x_h, x_id); // chain2 root  (block_id 0)
        backend.insert(y_h, y_id); // chain2 leaf  (block_id 1)

        assert_eq!(backend.len(), 4);
        assert_eq!(backend.get_queue_len(), 2);

        let alloc1 = backend.allocate(1);
        assert_eq!(alloc1[0].1, b_id); // B (tick 1)

        let alloc2 = backend.allocate(1);
        assert_eq!(alloc2[0].1, a_id); // A re-leafed, keeps tick 0 → next

        let alloc3 = backend.allocate(1);
        assert_eq!(alloc3[0].1, y_id); // Y (tick 3)

        let alloc4 = backend.allocate(1);
        assert_eq!(alloc4[0].1, x_id); // X re-leafed, keeps tick 2
    }

    #[test]
    fn test_remove_by_hash() {
        let mut backend = LineageBackend::new();

        let (id1, h1) = create_block(1);
        backend.insert(h1, id1);
        assert_eq!(backend.len(), 1);

        let removed = backend.remove_by_hash(&h1, Removal::Evict);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().1, 0);
        assert_eq!(backend.len(), 0);
        assert!(backend.is_graph_empty());
    }

    #[test]
    fn test_deep_chain_cleanup_iterative() {
        let depth = 1000;
        let mut backend = LineageBackend::new();

        let blocks = create_blocks(depth);
        let last_hash = blocks[depth - 1].1;
        for (id, h) in blocks {
            backend.insert(h, id);
        }

        assert_eq!(backend.len(), depth);
        assert_eq!(backend.get_queue_len(), 1);

        backend.remove_by_hash(&last_hash, Removal::Evict);

        assert_eq!(backend.len(), depth - 1);
        assert_eq!(backend.get_queue_len(), 1);

        backend = LineageBackend::new();

        let mut chain = create_blocks(101);
        let (leaf_id, leaf_h) = chain.remove(100);

        backend.insert(leaf_h, leaf_id);

        assert_eq!(backend.len(), 1);

        backend.remove_by_hash(&leaf_h, Removal::Evict);

        assert_eq!(backend.len(), 0);
        assert!(backend.is_graph_empty());
    }

    #[test]
    fn test_split_sequence_eviction() {
        let mut backend = LineageBackend::new();

        let branch1 = create_chain(5, 0);
        let branch2 = create_chain(5, 3000);

        for (id, h) in branch1 {
            backend.insert(h, id);
        }
        for (id, h) in branch2 {
            backend.insert(h, id);
        }

        assert_eq!(backend.len(), 10);
        assert_eq!(backend.get_queue_len(), 2);

        let alloc1 = backend.allocate(1);
        assert_eq!(alloc1.len(), 1);
        assert_eq!(backend.len(), 9);

        let alloc2 = backend.allocate(1);
        assert_eq!(alloc2.len(), 1);
        assert_eq!(backend.len(), 8);

        assert_eq!(backend.get_queue_len(), 2);

        backend.allocate(2);
        assert_eq!(backend.len(), 6);

        assert_eq!(backend.get_queue_len(), 2);
    }

    /// Regression: lookups must compare the full `SequenceHash`, not just
    /// the `(position, fragment)` index key. Two `PositionalLineageHash`es
    /// that share that key pair but have different parents must not collide
    /// — otherwise `find_matches` / `scan_matches` / `take` / `has` would
    /// return or remove the wrong block.
    #[test]
    fn lookup_rejects_same_position_fragment_but_different_full_hash() {
        let stored: SequenceHash = SequenceHash::new(0xAA, Some(0x11), 5);
        let impostor: SequenceHash = SequenceHash::new(0xAA, Some(0x22), 5);
        assert_eq!(stored.position(), impostor.position());
        assert_eq!(
            stored.parent_fragment_for_child_position(stored.position() + 1),
            impostor.parent_fragment_for_child_position(impostor.position() + 1)
        );
        assert_ne!(stored.as_u128(), impostor.as_u128());

        let mut backend = LineageBackend::new();
        backend.insert(stored, 42);

        assert!(backend.has(stored));
        assert!(
            !backend.has(impostor),
            "has() returned a false-positive for impostor PLH"
        );

        assert!(
            backend.remove_by_hash(&impostor, Removal::Evict).is_none(),
            "remove_by_hash matched impostor PLH and deleted stored block"
        );
        assert_eq!(backend.len(), 1);

        let scan_hits = backend.scan_matches(&[impostor], false);
        assert!(scan_hits.is_empty());
        assert_eq!(backend.len(), 1);

        let find_hits = backend.find_matches(&[impostor], false);
        assert!(find_hits.is_empty());
        assert_eq!(backend.len(), 1);

        assert!(!backend.take(impostor, 42));
        assert_eq!(backend.len(), 1);

        let removed = backend.remove_by_hash(&stored, Removal::Evict);
        assert_eq!(removed, Some((stored, 42)));
        assert_eq!(backend.len(), 0);
        assert!(backend.is_graph_empty());
    }

    /// Slab cells are recycled through the free list rather than reallocated.
    #[test]
    fn slab_recycles_freed_slots() {
        let mut backend = LineageBackend::with_capacity(8);

        for (id, h) in create_chain(8, 0) {
            backend.insert(h, id);
        }
        let high_water = backend.slots.len();
        assert_eq!(high_water, 8, "no ghosts for an in-order chain");

        backend.allocate(8);
        assert_eq!(backend.len(), 0);
        assert!(backend.is_graph_empty());

        for (id, h) in create_chain(8, 9000) {
            backend.insert(h, id);
        }
        assert_eq!(
            backend.slots.len(),
            high_water,
            "freed slots must be recycled, not reallocated"
        );
    }

    /// Under the `Fifo` policy a node that re-becomes a leaf is appended at
    /// the FIFO tail (the documented eviction-order difference vs. `Tick`).
    #[test]
    fn re_leafed_node_goes_to_fifo_tail() {
        let mut backend = LineageBackend::with_policy(0, LeafPolicy::fifo(0));
        let chain_a = create_chain(2, 0);
        let chain_b = create_chain(2, 7000);
        let (a0_id, a0_h) = chain_a[0];
        let (a1_id, a1_h) = chain_a[1];
        let (b0_id, b0_h) = chain_b[0];
        let (b1_id, b1_h) = chain_b[1];

        backend.insert(a0_h, a0_id);
        backend.insert(a1_h, a1_id);
        backend.insert(b0_h, b0_id);
        backend.insert(b1_h, b1_id);
        // Leaf FIFO: [A1, B1].

        // Remove A1 → A0 re-becomes a leaf and goes to the TAIL:
        // FIFO is now [B1, A0], not [A0, B1].
        assert!(backend.remove_by_hash(&a1_h, Removal::Evict).is_some());
        let _ = a0_id;
        let _ = b0_id;

        let order: Vec<BlockId> = backend.allocate(2).into_iter().map(|(_, id)| id).collect();
        assert_eq!(order, vec![b1_id, a0_id], "re-leafed A0 evicts after B1");
    }

    /// The same interleaved-chains scenario under `Fifo` round-robins the
    /// chains instead of keeping each root next to its own leaf.
    #[test]
    fn test_interleaved_chains_fifo() {
        let mut backend = LineageBackend::with_policy(0, LeafPolicy::fifo(0));

        let chain1 = create_chain(2, 0);
        let chain2 = create_chain(2, 1000);
        let (a_id, a_h) = chain1[0];
        let (b_id, b_h) = chain1[1];
        let (x_id, x_h) = chain2[0];
        let (y_id, y_h) = chain2[1];

        backend.insert(a_h, a_id);
        backend.insert(b_h, b_id);
        backend.insert(x_h, x_id);
        backend.insert(y_h, y_id);
        // Leaf FIFO: [B, Y].

        let order: Vec<BlockId> = backend.allocate(4).into_iter().map(|(_, id)| id).collect();
        // B; A re-leafs→tail; Y; X re-leafs→tail; A; X.
        assert_eq!(order, vec![b_id, y_id, a_id, x_id]);
    }

    // -----------------------------------------------------------------------
    // Valued leaf policy: neutral recency, poison suffix, branch-point stops.
    // -----------------------------------------------------------------------

    use crate::branch_tracker::BranchOracle;
    use std::collections::HashMap as StdHashMap;
    use std::sync::Arc;

    /// Oracle returning a caller-fixed `max_fanout` per hash (unset ⇒ `None`).
    #[derive(Default)]
    struct StubOracle {
        fanout: StdHashMap<SequenceHash, u32>,
    }
    impl BranchOracle for StubOracle {
        fn on_block_registered(&self, _hash: SequenceHash) {}
        fn on_block_removed(&self, _hash: SequenceHash) {}
        fn max_fanout(&self, hash: SequenceHash) -> Option<u32> {
            self.fanout.get(&hash).copied()
        }
    }

    fn valued_backend(oracle: Option<Arc<dyn BranchOracle>>) -> LineageBackend {
        let params = ScorerParams {
            gamma: 0.6,
            n: 2,
            k_sample: 16,
            t_blocks: None,
            seed: 0x51,
        };
        LineageBackend::with_policy(0, LeafPolicy::valued(0, None, oracle, params))
    }

    /// With no sketch/oracle and T unset the valued score is pure recency, so independent
    /// leaves evict oldest-first — parity with the recency `Tick`/`Fifo` behavior.
    #[test]
    fn valued_neutral_params_evict_oldest_first() {
        let mut backend = valued_backend(None);
        let a = create_block(1);
        let b = create_block(2);
        let c = create_block(3);
        backend.insert(a.1, a.0); // inserted first ⇒ oldest ⇒ evicted first
        backend.insert(b.1, b.0);
        backend.insert(c.1, c.0);
        assert_eq!(backend.len(), 3);
        let order: Vec<BlockId> = backend.allocate(3).into_iter().map(|(_, id)| id).collect();
        assert_eq!(order, vec![a.0, b.0, c.0], "recency order (oldest first)");
    }

    /// A poisoned single-owner suffix drains before any scored leaf, and the whole chain
    /// drains (the interior node re-leafs already poisoned).
    #[test]
    fn valued_poison_drains_single_owner_suffix_first() {
        let mut backend = valued_backend(None);
        // Chain A: a0 -> a1 (a1 leaf). Independent chain B: b0 -> b1.
        let chain_a = create_chain(2, 0);
        let chain_b = create_chain(2, 5000);
        for (id, h) in &chain_a {
            backend.insert(*h, *id);
        }
        for (id, h) in &chain_b {
            backend.insert(*h, *id);
        }
        // Poison chain A via its leaf a1: walk marks a1 (leaf) and a0 (interior single-owner).
        backend.poison(chain_a[1].1);

        // Even though chain B's leaf is older (would win on recency), the poisoned suffix
        // drains first, fully (a1 then re-leafed a0), before chain B.
        let order: Vec<BlockId> = backend.allocate(2).into_iter().map(|(_, id)| id).collect();
        assert_eq!(
            order,
            vec![chain_a[1].0, chain_a[0].0],
            "poisoned suffix drains leaf-then-root before any other chain"
        );
    }

    /// The poison walk halts at a *current* branch point (≥ 2 live children); the shared
    /// prefix and the sibling lineage are never poisoned.
    #[test]
    fn valued_poison_stops_at_current_branch_point() {
        let mut backend = valued_backend(None);
        // Shared prefix [t0, t1]; divergent leaves at position 2 (t2 vs t2').
        let chain1 = create_chain(3, 0); // tokens 0,1,2
        let mut b2 = BlockSequenceBuilder::from_tokens(vec![0, 1, 99])
            .with_block_size(1)
            .build();
        let leaf2 = b2.remove(2); // (id, hash) for the divergent block at position 2
        for (id, h) in &chain1 {
            backend.insert(*h, *id);
        }
        backend.insert(leaf2.1, leaf2.0); // second child of the position-1 block

        // Poison chain1's leaf (position 2). The walk marks only that leaf; its parent (the
        // position-1 block) has two live children ⇒ branch point ⇒ walk stops there.
        backend.poison(chain1[2].1);

        // Load-bearing: the poisoned suffix is EXACTLY the leaf. The shared prefix, its root,
        // and the sibling lineage must NOT be poisoned (deleting the branch-point stop would
        // poison the shared prefix and fail here).
        assert!(
            backend.test_is_poisoned(chain1[2].1),
            "the leaf is poisoned"
        );
        assert!(
            !backend.test_is_poisoned(chain1[1].1),
            "the shared prefix (branch point) must not be poisoned"
        );
        assert!(
            !backend.test_is_poisoned(chain1[0].1),
            "the shared root must not be poisoned"
        );
        assert!(
            !backend.test_is_poisoned(leaf2.1),
            "the sibling lineage must not be poisoned"
        );

        // First victim is the poisoned leaf; the sibling survives.
        let first = backend.allocate(1);
        assert_eq!(
            first[0].1, chain1[2].0,
            "only the poisoned suffix leaf goes first"
        );
        assert!(
            backend.has(leaf2.1),
            "the sibling lineage was not poisoned away"
        );
    }

    /// The walk also halts at a *re-leafed* branch point identified only by the monotone
    /// high-water `max_fanout` (current children == 1). Exercises `max_fanout_of`.
    #[test]
    fn valued_poison_stops_at_high_water_branch_point() {
        // Chain L(pos2) -> m(pos1) -> root(pos0); m currently has one child but a stub
        // oracle reports max_fanout = 2 (it forked and re-leafed), so it must be protected.
        let chain = create_chain(3, 0);
        let mid_hash = chain[1].1;
        let mut oracle = StubOracle::default();
        oracle.fanout.insert(mid_hash, 2);
        let mut backend = valued_backend(Some(Arc::new(oracle)));
        for (id, h) in &chain {
            backend.insert(*h, *id);
        }

        backend.poison(chain[2].1); // poison the leaf

        // Load-bearing: only the leaf is poisoned. The mid block (high-water branch point,
        // current fanout 1) and the root must NOT be poisoned — deleting the high-water stop
        // would poison the mid block and fail here.
        assert!(backend.test_is_poisoned(chain[2].1), "the leaf is poisoned");
        assert!(
            !backend.test_is_poisoned(mid_hash),
            "the high-water branch point must not be poisoned"
        );
        assert!(
            !backend.test_is_poisoned(chain[0].1),
            "the root must not be poisoned"
        );

        let first = backend.allocate(1);
        assert_eq!(first[0].1, chain[2].0, "only the leaf was poisoned");
        assert!(
            backend.has(mid_hash),
            "the high-water branch point was not poisoned away"
        );
    }

    /// The walk halts at a ghost placeholder (a resurrected interior node) rather than
    /// poisoning the real ancestor above it — a ghost's high-water branch identity is
    /// unknowable. Load-bearing: without the ghost stop the walk would poison b0.
    #[test]
    fn valued_poison_stops_at_ghost_ancestor() {
        let mut backend = valued_backend(None);
        let chain = create_chain(3, 0); // b0(pos0) -> b1(pos1) -> b2(pos2)
        for (id, h) in &chain {
            backend.insert(*h, *id);
        }
        // Resurrect the interior b1: it becomes a Ghost (still has child b2), so the chain
        // is now b2(Real leaf) -> ghost(b1) -> b0(Real root).
        assert!(
            backend.find_match(chain[1].1, false).is_some(),
            "b1 resurrected out"
        );

        backend.poison(chain[2].1);
        assert!(backend.test_is_poisoned(chain[2].1), "the leaf is poisoned");
        assert!(
            !backend.test_is_poisoned(chain[0].1),
            "the walk must stop at the ghost, not poison the real ancestor b0"
        );
    }

    #[test]
    fn pair_hasher_distinguishes_keys() {
        use std::collections::HashSet;
        let keys = [
            (0u64, 0u64),
            (0, 1),
            (1, 0),
            (1, 1),
            (5, 0xdead_beef),
            (5, 0xbeef_dead),
        ];
        let digests: HashSet<u64> = keys
            .iter()
            .map(|&k| {
                let mut h = PairBuildHasher.build_hasher();
                std::hash::Hash::hash(&k, &mut h);
                h.finish()
            })
            .collect();
        assert_eq!(
            digests.len(),
            keys.len(),
            "distinct pairs → distinct digests"
        );
    }
}
