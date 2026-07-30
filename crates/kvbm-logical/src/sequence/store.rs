// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared three-collection lifecycle store backing both
//! [`ExternalBlockAssignments`](super::assignments::ExternalBlockAssignments) and
//! [`LogicalBlockAssignments`](super::assignments::LogicalBlockAssignments).

use indexmap::IndexMap;

use crate::BlockId;

/// Generic three-phase lifecycle store keyed by [`BlockId`].
///
/// Manages three ordered `IndexMap` collections representing the lifecycle
/// phases: **unassigned** (`U`) → **staged** (`S`) → **assigned** (`A`).
///
/// Both [`ExternalBlockAssignments`](super::assignments::ExternalBlockAssignments) and
/// [`LogicalBlockAssignments`](super::assignments::LogicalBlockAssignments) compose this
/// type internally and add their own type-specific transition logic on top.
pub(crate) struct BlockStore<U, S, A> {
    assigned: IndexMap<BlockId, A>,
    staged: IndexMap<BlockId, S>,
    unassigned: IndexMap<BlockId, U>,
}

impl<U, S, A> BlockStore<U, S, A> {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            assigned: IndexMap::new(),
            staged: IndexMap::new(),
            unassigned: IndexMap::new(),
        }
    }

    // -- Counts ---------------------------------------------------------------

    /// Returns the number of assigned entries.
    pub fn assigned_count(&self) -> usize {
        self.assigned.len()
    }

    /// Returns the number of staged entries.
    pub fn staged_count(&self) -> usize {
        self.staged.len()
    }

    /// Returns the number of unassigned entries.
    pub fn unassigned_count(&self) -> usize {
        self.unassigned.len()
    }

    // -- Queries --------------------------------------------------------------

    /// Returns `true` if all three collections are empty.
    pub fn is_empty(&self) -> bool {
        self.assigned.is_empty() && self.staged.is_empty() && self.unassigned.is_empty()
    }

    /// Checks whether a `BlockId` is present in any of the three collections.
    pub fn contains(&self, block_id: &BlockId) -> bool {
        self.assigned.contains_key(block_id)
            || self.staged.contains_key(block_id)
            || self.unassigned.contains_key(block_id)
    }

    /// Checks whether a `BlockId` is present in the assigned collection.
    #[allow(dead_code)]
    pub fn contains_assigned(&self, block_id: &BlockId) -> bool {
        self.assigned.contains_key(block_id)
    }

    /// Checks whether a `BlockId` is present in the staged collection.
    #[allow(dead_code)]
    pub fn contains_staged(&self, block_id: &BlockId) -> bool {
        self.staged.contains_key(block_id)
    }

    /// Checks whether a `BlockId` is present in the unassigned collection.
    #[allow(dead_code)]
    pub fn contains_unassigned(&self, block_id: &BlockId) -> bool {
        self.unassigned.contains_key(block_id)
    }

    // -- Index Access ---------------------------------------------------------

    /// Returns the assigned entry at the given index (insertion order).
    pub fn get_assigned(&self, index: usize) -> Option<(&BlockId, &A)> {
        self.assigned.get_index(index)
    }

    /// Returns the staged entry at the given index (staging order).
    pub fn get_staged(&self, index: usize) -> Option<(&BlockId, &S)> {
        self.staged.get_index(index)
    }

    /// Returns the unassigned entry at the given index (FIFO order).
    pub fn get_unassigned(&self, index: usize) -> Option<(&BlockId, &U)> {
        self.unassigned.get_index(index)
    }

    // -- Iteration ------------------------------------------------------------

    /// Iterates over assigned entries in positional order.
    pub fn assigned_iter(&self) -> impl Iterator<Item = (&BlockId, &A)> {
        self.assigned.iter()
    }

    /// Iterates over staged entries in staging order.
    pub fn staged_iter(&self) -> impl Iterator<Item = (&BlockId, &S)> {
        self.staged.iter()
    }

    /// Iterates over unassigned entries in FIFO order.
    pub fn unassigned_iter(&self) -> impl Iterator<Item = (&BlockId, &U)> {
        self.unassigned.iter()
    }

    // -- FIFO Pop -------------------------------------------------------------

    /// Removes and returns the first unassigned entry (FIFO).
    pub fn shift_unassigned(&mut self) -> Option<(BlockId, U)> {
        self.unassigned.shift_remove_index(0)
    }

    /// Removes and returns the first staged entry (FIFO).
    pub fn shift_staged(&mut self) -> Option<(BlockId, S)> {
        self.staged.shift_remove_index(0)
    }

    // -- LIFO Pop -------------------------------------------------------------

    /// Removes and returns the last unassigned entry (LIFO).
    pub fn pop_unassigned(&mut self) -> Option<(BlockId, U)> {
        self.unassigned.pop()
    }

    // -- Batch Drain ------------------------------------------------------------
    //
    // These exist to avoid the O(N²) cost of driving `shift_staged`/
    // `shift_unassigned` in a per-element loop: each `shift_remove_index(0)`
    // is itself O(len) because it shifts every following element down, so an
    // N-element loop is O(N²). Draining a contiguous prefix in one call is
    // O(len) total.

    /// Drains all staged entries in staging (FIFO) order, returning them as a
    /// `Vec`. O(N) total, versus O(N²) for an N-element loop of
    /// [`shift_staged`](Self::shift_staged).
    ///
    /// Returns an owned `Vec` (not the `Drain` iterator) so callers can
    /// re-borrow `self` (e.g. to call `insert_assigned`) while iterating the
    /// result.
    pub fn drain_staged(&mut self) -> Vec<(BlockId, S)> {
        self.staged.drain(..).collect()
    }

    /// Drains the first `min(n, unassigned.len())` entries from unassigned in
    /// FIFO order, returning them as a `Vec`. O(len) total (a single prefix
    /// shift), versus O(n · len) for an n-element loop of
    /// [`shift_unassigned`](Self::shift_unassigned).
    pub fn drain_unassigned_prefix(&mut self, n: usize) -> Vec<(BlockId, U)> {
        let n = n.min(self.unassigned.len());
        self.unassigned.drain(..n).collect()
    }

    /// Reinserts a batch of previously-drained unassigned entries at the
    /// front of the unassigned queue, preserving their relative (FIFO) order
    /// ahead of whatever remains.
    ///
    /// This is the recovery half of [`drain_unassigned_prefix`](Self::drain_unassigned_prefix):
    /// when a caller drains a prefix, processes entries one at a time, and
    /// hits an error partway through, the unprocessed tail of the drained
    /// batch must go back to the front of unassigned (not be lost, and not be
    /// appended to the back — that would violate FIFO order for anything
    /// still queued behind it).
    pub fn reinsert_unassigned_front(&mut self, front: Vec<(BlockId, U)>) {
        if front.is_empty() {
            return;
        }
        let remainder: IndexMap<BlockId, U> = std::mem::take(&mut self.unassigned);
        let mut rebuilt = IndexMap::with_capacity(front.len() + remainder.len());
        rebuilt.extend(front);
        rebuilt.extend(remainder);
        self.unassigned = rebuilt;
    }

    // -- Insert ---------------------------------------------------------------

    /// Inserts into the assigned collection.
    pub fn insert_assigned(&mut self, id: BlockId, val: A) {
        self.assigned.insert(id, val);
    }

    /// Inserts into the staged collection.
    pub fn insert_staged(&mut self, id: BlockId, val: S) {
        self.staged.insert(id, val);
    }

    /// Inserts into the unassigned collection.
    pub fn insert_unassigned(&mut self, id: BlockId, val: U) {
        self.unassigned.insert(id, val);
    }

    // -- Bulk -----------------------------------------------------------------

    /// Iterates over all block IDs across all three collections in lifecycle
    /// order: assigned → staged → unassigned.
    pub fn all_block_ids(&self) -> impl Iterator<Item = &BlockId> {
        self.assigned
            .keys()
            .chain(self.staged.keys())
            .chain(self.unassigned.keys())
    }

    /// Validates that none of the given `ids` collide with existing entries
    /// or with each other.
    ///
    /// Returns `Ok(())` if all IDs are unique, or `Err(id)` with the first
    /// duplicate found.
    pub fn validate_no_duplicates(
        &self,
        ids: impl Iterator<Item = BlockId>,
        count_hint: usize,
    ) -> Result<(), BlockId> {
        let mut seen = indexmap::IndexSet::with_capacity(count_hint);
        for id in ids {
            if self.contains(&id) || !seen.insert(id) {
                return Err(id);
            }
        }
        Ok(())
    }

    /// Clears all three collections.
    pub fn clear(&mut self) {
        self.assigned.clear();
        self.staged.clear();
        self.unassigned.clear();
    }

    /// Takes all assigned entries, returning them as a `Vec`.
    pub fn take_assigned(&mut self) -> Vec<(BlockId, A)> {
        std::mem::take(&mut self.assigned).into_iter().collect()
    }

    /// Takes all staged entries, returning them as a `Vec`.
    pub fn take_staged(&mut self) -> Vec<(BlockId, S)> {
        std::mem::take(&mut self.staged).into_iter().collect()
    }

    /// Takes all unassigned entries, returning them as a `Vec`.
    pub fn take_unassigned(&mut self) -> Vec<(BlockId, U)> {
        std::mem::take(&mut self.unassigned).into_iter().collect()
    }
}
