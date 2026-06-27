// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-time branch-point tracking for the block registration path.
//!
//! A **branch point** is a positional node with more than one distinct child — the point
//! where multiple token sequences diverge from a shared prefix.
//!
//! [`BranchPointTracker`] maintains:
//! - Exact online parent-child accounting for all registered blocks
//! - A bounded historical cache of high-fanout / frequently-reused branch nodes
//!
//! Updates happen incrementally inside [`BlockRegistry::register_sequence_hash`] with
//! ~3 hash operations per registration. The tracker is wrapped in
//! `Arc<parking_lot::Mutex<_>>`, matching the [`TinyLFUTracker`](crate::tinylfu) pattern.

use std::collections::{HashMap, HashSet};

use crate::SequenceHash;

// ---------------------------------------------------------------------------
// ParentNode — lightweight per-node child set
// ---------------------------------------------------------------------------

/// Tracks the set of distinct child fragments for a single positional node.
struct ParentNode {
    /// Child fragments at `position + 1`.
    children: HashSet<u64>,
}

// ---------------------------------------------------------------------------
// BranchPointRecord — public metadata for a branch node
// ---------------------------------------------------------------------------

/// Metadata record for a branch point (a node with fanout > 1).
///
/// Stored in the bounded historical cache and returned by query methods.
#[derive(Debug, Clone)]
pub struct BranchPointRecord {
    /// Block position in the sequence.
    pub position: u64,
    /// Hash fragment identifying this node at its position.
    pub fragment: u64,
    /// Current number of distinct children.
    pub current_fanout: u16,
    /// Peak fanout ever observed for this node.
    pub max_fanout: u16,
    /// Number of times a new child was added (measures branch-point activity).
    pub observation_count: u32,
    /// Tick when this record was last updated.
    pub last_updated_tick: u64,
}

// ---------------------------------------------------------------------------
// BoundedBranchCache — fixed-capacity history of branch points
// ---------------------------------------------------------------------------

/// Fixed-capacity cache of [`BranchPointRecord`]s.
///
/// Eviction uses a frequency/recency hybrid score:
/// `observation_count / (current_tick - last_updated_tick + 1)`.
struct BoundedBranchCache {
    entries: HashMap<(u64, u64), BranchPointRecord>,
    capacity: usize,
}

impl BoundedBranchCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity,
        }
    }

    /// Score a record: higher is more valuable (frequently updated + recent).
    fn score(record: &BranchPointRecord, current_tick: u64) -> f64 {
        let age = current_tick.saturating_sub(record.last_updated_tick) + 1;
        record.observation_count as f64 / age as f64
    }

    /// Insert a new branch point (first time crossing fanout > 1).
    fn insert_new(
        &mut self,
        position: u64,
        fragment: u64,
        fanout: u16,
        tick: u64,
    ) {
        let key = (position, fragment);

        // Already tracked — just update
        if let Some(record) = self.entries.get_mut(&key) {
            record.current_fanout = fanout;
            record.max_fanout = record.max_fanout.max(fanout);
            record.observation_count = record.observation_count.saturating_add(1);
            record.last_updated_tick = tick;
            return;
        }

        // Evict if at capacity
        if self.entries.len() >= self.capacity && self.capacity > 0 {
            self.evict_lowest(tick);
        }

        if self.capacity > 0 {
            self.entries.insert(
                key,
                BranchPointRecord {
                    position,
                    fragment,
                    current_fanout: fanout,
                    max_fanout: fanout,
                    observation_count: 1,
                    last_updated_tick: tick,
                },
            );
        }
    }

    /// Update an existing branch point record (fanout increased beyond 2).
    fn update(
        &mut self,
        position: u64,
        fragment: u64,
        fanout: u16,
        tick: u64,
    ) {
        if let Some(record) = self.entries.get_mut(&(position, fragment)) {
            record.current_fanout = fanout;
            record.max_fanout = record.max_fanout.max(fanout);
            record.observation_count = record.observation_count.saturating_add(1);
            record.last_updated_tick = tick;
        } else {
            // Not in cache (may have been evicted). Re-insert.
            self.insert_new(position, fragment, fanout, tick);
        }
    }

    fn get(&self, position: u64, fragment: u64) -> Option<&BranchPointRecord> {
        self.entries.get(&(position, fragment))
    }

    /// Return top entries by score, descending.
    fn top_by_score(&self, limit: usize, current_tick: u64) -> Vec<BranchPointRecord> {
        let mut scored: Vec<_> = self
            .entries
            .values()
            .map(|r| (Self::score(r, current_tick), r))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(limit)
            .map(|(_, r)| r.clone())
            .collect()
    }

    /// Evict the entry with the lowest score.
    fn evict_lowest(&mut self, current_tick: u64) {
        if self.entries.is_empty() {
            return;
        }
        let victim = self
            .entries
            .iter()
            .map(|(&key, record)| (key, Self::score(record, current_tick)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(key, _)| key);

        if let Some(key) = victim {
            self.entries.remove(&key);
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

// ---------------------------------------------------------------------------
// BranchPointTracker — public API
// ---------------------------------------------------------------------------

/// Tracks parent-child relationships for all registered blocks and maintains
/// a bounded historical cache of branch points (nodes with fanout > 1).
///
/// Designed to sit inside [`BlockRegistry`](crate::registry::BlockRegistry) behind
/// `Arc<parking_lot::Mutex<BranchPointTracker>>`.
pub struct BranchPointTracker {
    /// Parent node tracking. Outer key: position, inner key: fragment at that position.
    /// Only nodes that have had at least one child registered appear here.
    parents: HashMap<u64, HashMap<u64, ParentNode>>,

    /// Bounded cache of historical branch-point records.
    history: BoundedBranchCache,

    /// Monotonic tick for recency ordering.
    tick: u64,
}

impl BranchPointTracker {
    /// Creates a new tracker with the given history cache capacity.
    pub fn new(history_capacity: usize) -> Self {
        Self {
            parents: HashMap::new(),
            history: BoundedBranchCache::new(history_capacity),
            tick: 0,
        }
    }

    /// Called when a new block is registered. Extracts the parent-child
    /// relationship from `seq_hash` and updates tracking state.
    ///
    /// Cost: O(1) amortized — one `HashMap` lookup + one `HashSet` insert.
    pub fn on_block_registered(&mut self, seq_hash: SequenceHash) {
        let position = seq_hash.position();

        // Root blocks (position 0) have no parent.
        if position == 0 {
            return;
        }

        let parent_position = position - 1;
        let parent_fragment = seq_hash.parent_hash_fragment();
        let current_fragment = seq_hash.current_hash_fragment();

        let parent_level = self.parents.entry(parent_position).or_default();
        let parent_node = parent_level
            .entry(parent_fragment)
            .or_insert_with(|| ParentNode {
                children: HashSet::new(),
            });

        let is_new_child = parent_node.children.insert(current_fragment);

        if is_new_child {
            let fanout = parent_node.children.len();

            if fanout == 2 {
                // Crossed from 1 → 2 children: new branch point.
                self.history
                    .insert_new(parent_position, parent_fragment, fanout as u16, self.tick);
            } else if fanout > 2 {
                // Existing branch point gained another child.
                self.history
                    .update(parent_position, parent_fragment, fanout as u16, self.tick);
            }
        }

        self.tick += 1;
    }

    // -- Point queries -------------------------------------------------------

    /// Returns `true` if the node at `(position, fragment)` currently has more
    /// than one child.
    pub fn is_branch_point(&self, position: u64, fragment: u64) -> bool {
        self.parents
            .get(&position)
            .and_then(|level| level.get(&fragment))
            .is_some_and(|node| node.children.len() > 1)
    }

    /// Returns the current fanout of a node, or `None` if untracked.
    pub fn current_fanout(&self, position: u64, fragment: u64) -> Option<usize> {
        self.parents
            .get(&position)
            .and_then(|level| level.get(&fragment))
            .map(|node| node.children.len())
    }

    // -- Enumeration ---------------------------------------------------------

    /// Returns all current branch points as `(position, fragment, fanout)` triples.
    pub fn current_branch_points(&self) -> Vec<(u64, u64, usize)> {
        let mut result = Vec::new();
        for (&position, level) in &self.parents {
            for (&fragment, node) in level {
                if node.children.len() > 1 {
                    result.push((position, fragment, node.children.len()));
                }
            }
        }
        result
    }

    /// Returns top-scoring historical branch points, descending by score.
    pub fn hot_branch_points(&self, limit: usize) -> Vec<BranchPointRecord> {
        self.history.top_by_score(limit, self.tick)
    }

    /// Returns branch points along a prefix path.
    ///
    /// The slice should contain the `SequenceHash` for each block position,
    /// ordered by position (index 0 = position 0, etc.). For each position
    /// that is a branch point, a [`BranchPointRecord`] is returned — from the
    /// history cache if available, otherwise synthesized from live state.
    pub fn branch_points_on_prefix(&self, prefix: &[SequenceHash]) -> Vec<BranchPointRecord> {
        let mut result = Vec::new();
        for seq_hash in prefix {
            let pos = seq_hash.position();
            let frag = seq_hash.current_hash_fragment();

            if let Some(record) = self.history.get(pos, frag) {
                result.push(record.clone());
            } else if self.is_branch_point(pos, frag) {
                let fanout = self.parents[&pos][&frag].children.len() as u16;
                result.push(BranchPointRecord {
                    position: pos,
                    fragment: frag,
                    current_fanout: fanout,
                    max_fanout: fanout,
                    observation_count: 0,
                    last_updated_tick: self.tick,
                });
            }
        }
        result
    }

    // -- Aggregate stats -----------------------------------------------------

    /// Returns the number of current branch points (fanout > 1).
    pub fn branch_point_count(&self) -> usize {
        self.parents
            .values()
            .flat_map(|level| level.values())
            .filter(|node| node.children.len() > 1)
            .count()
    }

    /// Returns the total number of tracked parent nodes (fanout >= 1).
    pub fn tracked_parent_count(&self) -> usize {
        self.parents.values().map(|level| level.len()).sum()
    }

    /// Returns the number of entries in the history cache.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{BlockSequenceBuilder, TestMeta};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a chain of registered blocks from a token sequence.
    /// Returns `(blocks, seq_hashes)` — blocks are consumed by the caller.
    fn build_chain(
        tokens: Vec<u32>,
        block_size: usize,
    ) -> Vec<SequenceHash> {
        BlockSequenceBuilder::<TestMeta>::from_tokens(tokens)
            .with_block_size(block_size)
            .build()
            .into_iter()
            .map(|(_, hash)| hash)
            .collect()
    }

    /// Register all hashes from a chain into the tracker.
    fn register_all(tracker: &mut BranchPointTracker, hashes: &[SequenceHash]) {
        for &hash in hashes {
            tracker.on_block_registered(hash);
        }
    }

    // -----------------------------------------------------------------------
    // 1. No branching
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_chain_no_branch_points() {
        let mut tracker = BranchPointTracker::new(1024);
        let hashes = build_chain((0..5).collect(), 1);
        register_all(&mut tracker, &hashes);

        assert_eq!(tracker.branch_point_count(), 0);
        for hash in &hashes {
            assert!(!tracker.is_branch_point(hash.position(), hash.current_hash_fragment()));
        }
    }

    // -----------------------------------------------------------------------
    // 2. Simple fork
    // -----------------------------------------------------------------------

    #[test]
    fn test_simple_fork_detected() {
        let mut tracker = BranchPointTracker::new(1024);

        // Chain A: tokens [0, 1, 2]  → positions 0, 1, 2
        // Chain B: tokens [0, 1, 99] → positions 0, 1, 2 (diverges at block 2)
        //
        // The block at position 1 in chain A has the same lineage hash as
        // position 1 in chain B (shared prefix [0, 1]). The blocks at
        // position 2 differ. So position 1 should become a branch point.
        let chain_a = build_chain(vec![0, 1, 2], 1);
        let chain_b = build_chain(vec![0, 1, 99], 1);

        register_all(&mut tracker, &chain_a);
        register_all(&mut tracker, &chain_b);

        // Position 1 should be a branch point (two distinct children at pos 2)
        let bp = tracker.current_branch_points();
        assert_eq!(bp.len(), 1, "Expected exactly one branch point, got: {:?}", bp);
        assert_eq!(bp[0].0, 1, "Branch point should be at position 1");
        assert_eq!(bp[0].2, 2, "Fanout should be 2");

        assert_eq!(tracker.branch_point_count(), 1);
    }

    // -----------------------------------------------------------------------
    // 3. Multiple forks at different positions
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_forks() {
        let mut tracker = BranchPointTracker::new(1024);

        // Shared prefix: [10, 20, 30, 40]
        // Fork 1 at position 1: chain [10, 20, 30, 40] vs [10, 20, 30, 41]
        //   → position 2 (block [30]) is branch point (children [40] and [41])
        // But we also want a fork at position 0:
        //   chain [10, 77, ...] diverges from [10, 20, ...]
        //   → position 0 (block [10]) is branch point

        let chain_a = build_chain(vec![10, 20, 30, 40], 1);
        let chain_b = build_chain(vec![10, 20, 30, 41], 1);
        let chain_c = build_chain(vec![10, 77], 1);

        register_all(&mut tracker, &chain_a);
        register_all(&mut tracker, &chain_b);
        register_all(&mut tracker, &chain_c);

        // We should have at least 2 branch points
        let bp_count = tracker.branch_point_count();
        assert!(
            bp_count >= 2,
            "Expected at least 2 branch points, got {}",
            bp_count
        );
    }

    // -----------------------------------------------------------------------
    // 4. High fanout
    // -----------------------------------------------------------------------

    #[test]
    fn test_high_fanout() {
        let mut tracker = BranchPointTracker::new(1024);

        // Shared root [42], then 10 distinct children
        for child_token in 100..110u32 {
            let chain = build_chain(vec![42, child_token], 1);
            register_all(&mut tracker, &chain);
        }

        assert_eq!(tracker.branch_point_count(), 1);

        // Find the branch point
        let bps = tracker.current_branch_points();
        assert_eq!(bps.len(), 1);
        assert_eq!(bps[0].2, 10, "Fanout should be 10");

        // History should track max_fanout
        let hot = tracker.hot_branch_points(1);
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0].max_fanout, 10);
        assert_eq!(hot[0].current_fanout, 10);
    }

    // -----------------------------------------------------------------------
    // 5. Cache capacity and eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_capacity_eviction() {
        let mut tracker = BranchPointTracker::new(2); // Only 2 slots

        // Create 3 independent branch points using different root tokens
        // BP1: root [1000], children [1001] and [1002]
        register_all(&mut tracker, &build_chain(vec![1000, 1001], 1));
        register_all(&mut tracker, &build_chain(vec![1000, 1002], 1));

        // BP2: root [2000], children [2001] and [2002]
        register_all(&mut tracker, &build_chain(vec![2000, 2001], 1));
        register_all(&mut tracker, &build_chain(vec![2000, 2002], 1));

        // BP3: root [3000], children [3001] and [3002]
        register_all(&mut tracker, &build_chain(vec![3000, 3001], 1));
        register_all(&mut tracker, &build_chain(vec![3000, 3002], 1));

        // All 3 are live branch points
        assert_eq!(tracker.branch_point_count(), 3);
        // But the history cache can only hold 2
        assert_eq!(tracker.history_len(), 2);
    }

    // -----------------------------------------------------------------------
    // 6. Cache scoring: frequent branch point survives eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_scoring_frequency_wins() {
        let mut tracker = BranchPointTracker::new(2);

        // BP1 (high frequency): root [500], keep adding children
        register_all(&mut tracker, &build_chain(vec![500, 501], 1));
        register_all(&mut tracker, &build_chain(vec![500, 502], 1)); // becomes branch point
        register_all(&mut tracker, &build_chain(vec![500, 503], 1)); // obs_count increases
        register_all(&mut tracker, &build_chain(vec![500, 504], 1));
        register_all(&mut tracker, &build_chain(vec![500, 505], 1));

        // BP2 (low frequency): root [600], only 2 children
        register_all(&mut tracker, &build_chain(vec![600, 601], 1));
        register_all(&mut tracker, &build_chain(vec![600, 602], 1));

        // BP3 (low frequency): root [700], only 2 children
        register_all(&mut tracker, &build_chain(vec![700, 701], 1));
        register_all(&mut tracker, &build_chain(vec![700, 702], 1));

        // BP1 should survive because it has the highest observation_count
        let hot = tracker.hot_branch_points(1);
        assert_eq!(hot.len(), 1);
        // The top entry should have observation_count >= 4 (initial + 3 more children)
        assert!(
            hot[0].observation_count >= 4,
            "Top entry should be the high-frequency one, obs_count={}",
            hot[0].observation_count
        );
    }

    // -----------------------------------------------------------------------
    // 7. Prefix query
    // -----------------------------------------------------------------------

    #[test]
    fn test_branch_points_on_prefix() {
        let mut tracker = BranchPointTracker::new(1024);

        // Chain: [10, 20, 30, 40, 50]
        let main_chain = build_chain(vec![10, 20, 30, 40, 50], 1);
        register_all(&mut tracker, &main_chain);

        // Fork at position 1 (block [20]): another child [99] from [10]
        register_all(&mut tracker, &build_chain(vec![10, 99], 1));

        // Fork at position 3 (block [40]): another child [88] from [30]
        register_all(&mut tracker, &build_chain(vec![10, 20, 30, 88], 1));

        let bp_on_path = tracker.branch_points_on_prefix(&main_chain);
        let positions: Vec<u64> = bp_on_path.iter().map(|r| r.position).collect();

        // Position 0 has 2 children ([20] and [99]) → branch point
        assert!(positions.contains(&0), "Expected branch point at position 0, got {:?}", positions);
        // Position 2 has 2 children ([40] and [88]) → branch point
        assert!(positions.contains(&2), "Expected branch point at position 2, got {:?}", positions);
    }

    // -----------------------------------------------------------------------
    // 8. Registry integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_integration() {
        use crate::registry::BlockRegistry;
        use std::sync::Arc;

        let tracker = Arc::new(parking_lot::Mutex::new(BranchPointTracker::new(1024)));
        let registry = BlockRegistry::builder()
            .branch_tracker(tracker.clone())
            .build();

        // Register blocks through the registry's normal path.
        // Chain A: [10, 20, 30]
        let chain_a = build_chain(vec![10, 20, 30], 1);
        for &hash in &chain_a {
            registry.register_sequence_hash(hash);
        }

        // Chain B: [10, 20, 99] — diverges at position 2
        let chain_b = build_chain(vec![10, 20, 99], 1);
        for &hash in &chain_b {
            registry.register_sequence_hash(hash);
        }

        let bt = tracker.lock();
        assert_eq!(bt.branch_point_count(), 1);
        let bps = bt.current_branch_points();
        assert_eq!(bps[0].0, 1, "Branch point should be at position 1");
        assert_eq!(bps[0].2, 2, "Fanout should be 2");
    }

    // -----------------------------------------------------------------------
    // 9. Re-registration idempotency
    // -----------------------------------------------------------------------

    #[test]
    fn test_reregistration_idempotent() {
        use crate::registry::BlockRegistry;
        use std::sync::Arc;

        let tracker = Arc::new(parking_lot::Mutex::new(BranchPointTracker::new(1024)));
        let registry = BlockRegistry::builder()
            .branch_tracker(tracker.clone())
            .build();

        let chain = build_chain(vec![10, 20], 1);
        for &hash in &chain {
            registry.register_sequence_hash(hash);
        }

        let parents_before = tracker.lock().tracked_parent_count();

        // Re-register the same hashes — should be a no-op for the tracker
        // because register_sequence_hash returns the existing handle.
        for &hash in &chain {
            registry.register_sequence_hash(hash);
        }

        let parents_after = tracker.lock().tracked_parent_count();
        assert_eq!(parents_before, parents_after, "Re-registration should not add parents");
    }

    // -----------------------------------------------------------------------
    // 10. Root blocks — no branch points
    // -----------------------------------------------------------------------

    #[test]
    fn test_root_blocks_no_branch_points() {
        let mut tracker = BranchPointTracker::new(1024);

        // Multiple independent root blocks (position 0)
        for token in [100, 200, 300, 400, 500] {
            let chain = build_chain(vec![token], 1);
            register_all(&mut tracker, &chain);
        }

        // Root blocks have no parent, so no branch points
        assert_eq!(tracker.branch_point_count(), 0);
        assert_eq!(tracker.tracked_parent_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Additional: current_fanout
    // -----------------------------------------------------------------------

    #[test]
    fn test_current_fanout() {
        let mut tracker = BranchPointTracker::new(1024);

        let chain_a = build_chain(vec![42, 100], 1);
        register_all(&mut tracker, &chain_a);

        let root_hash = chain_a[0];
        let pos = root_hash.position();
        let frag = root_hash.current_hash_fragment();

        assert_eq!(tracker.current_fanout(pos, frag), Some(1));

        register_all(&mut tracker, &build_chain(vec![42, 200], 1));
        assert_eq!(tracker.current_fanout(pos, frag), Some(2));

        register_all(&mut tracker, &build_chain(vec![42, 300], 1));
        assert_eq!(tracker.current_fanout(pos, frag), Some(3));

        // Untracked position
        assert_eq!(tracker.current_fanout(999, 0), None);
    }
}
