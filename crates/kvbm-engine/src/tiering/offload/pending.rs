// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pending transfer tracking for duplicate prevention.
//!
//! This module provides `PendingTracker` and `PendingGuard` types that work together
//! to track blocks that are currently in-flight through the transfer pipeline.
//!
//! # Problem
//!
//! When overlapping sequences are enqueued for transfer at roughly the same time,
//! the presence policy may allow duplicate transfers because:
//! - The first sequence's blocks haven't completed registration yet
//! - The second sequence sees the same blocks as "not present"
//!
//! # Solution
//!
//! The `PendingTracker` maintains a set of sequence hashes currently in the pipeline.
//! When blocks pass policy evaluation, `try_claim` atomically creates a `PendingGuard` that:
//! - Adds a previously absent sequence hash to the pending set
//! - Returns no guard when another block already owns that hash
//! - Automatically removes the owned hash on drop (RAII pattern)
//!
//! The `PresenceFilter` can check both the registry (completed transfers) and
//! the pending set (in-flight transfers) as early filters. `try_claim` remains
//! the unique-ownership boundary for every policy configuration.
//!
//! # Example
//!
//! ```ignore
//! let tracker = Arc::new(PendingTracker::new());
//!
//! // Claim a hash when a block passes policy.
//! let guard = tracker.try_claim(sequence_hash)?;
//!
//! // Guard travels with block through pipeline stages
//! queued_block.pending_guard = Some(guard);
//!
//! // When block completes or is cancelled, guard is dropped
//! // and hash is automatically removed from pending set
//! ```

use std::sync::Arc;

use dashmap::DashSet;

use crate::SequenceHash;

/// Tracks sequence hashes that are currently pending transfer.
///
/// This is shared between the pipeline and the presence policy via `Arc`.
/// Thread-safe for concurrent access from multiple pipeline stages.
#[derive(Debug, Default)]
pub struct PendingTracker {
    pending: DashSet<SequenceHash>,
}

impl PendingTracker {
    /// Create a new empty pending tracker.
    pub fn new() -> Self {
        Self {
            pending: DashSet::new(),
        }
    }

    /// Check if a sequence hash is currently pending transfer.
    ///
    /// Used by `PresenceFilter` to skip blocks that are already in-flight.
    pub fn is_pending(&self, hash: &SequenceHash) -> bool {
        self.pending.contains(hash)
    }

    /// Get the number of pending transfers.
    ///
    /// Useful for metrics and debugging.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Check if there are no pending transfers.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Atomically claim a sequence hash until the returned guard drops.
    ///
    /// Returns `None` if a live guard already owns the hash. The returned guard
    /// uses RAII to ensure the hash is removed when:
    /// - Transfer completes successfully
    /// - Transfer is cancelled
    /// - Block is evicted from pipeline
    /// - Any error causes the block to be dropped
    pub(crate) fn try_claim(self: &Arc<Self>, hash: SequenceHash) -> Option<PendingGuard> {
        if self.pending.insert(hash) {
            Some(PendingGuard {
                hash,
                tracker: Arc::clone(self),
            })
        } else {
            None
        }
    }
}

/// Extension trait for `Option<Arc<PendingTracker>>` to simplify pending checks.
///
/// Reduces the common pattern `self.pending_tracker.as_ref().is_some_and(|t| t.is_pending(&hash))`
/// to a single method call.
pub(crate) trait PendingCheck {
    fn is_hash_pending(&self, hash: &SequenceHash) -> bool;
}

impl PendingCheck for Option<Arc<PendingTracker>> {
    fn is_hash_pending(&self, hash: &SequenceHash) -> bool {
        self.as_ref().is_some_and(|t| t.is_pending(hash))
    }
}

/// RAII guard that removes a sequence hash from the pending set on drop.
///
/// This guard travels with the block through all pipeline stages and ensures
/// cleanup happens automatically regardless of how the transfer completes.
///
pub(crate) struct PendingGuard {
    hash: SequenceHash,
    tracker: Arc<PendingTracker>,
}

impl PendingGuard {
    #[cfg(test)]
    pub(crate) fn sequence_hash(&self) -> SequenceHash {
        self.hash
    }
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.tracker.pending.remove(&self.hash);
    }
}

impl std::fmt::Debug for PendingGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingGuard")
            .field("sequence_hash", &self.hash)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test SequenceHash with unique values.
    fn test_hash(id: u64) -> SequenceHash {
        SequenceHash::new(id, Some(0), id)
    }

    #[test]
    fn test_pending_tracker_new() {
        let tracker = PendingTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_pending_guard_inserts_and_removes() {
        let tracker = Arc::new(PendingTracker::new());
        let hash = test_hash(12345);

        assert!(!tracker.is_pending(&hash));

        {
            let _guard = tracker.try_claim(hash).expect("claim a new hash");
            assert!(tracker.is_pending(&hash));
            assert_eq!(tracker.len(), 1);
        }

        // Guard dropped, hash should be removed
        assert!(!tracker.is_pending(&hash));
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_multiple_guards_different_hashes() {
        let tracker = Arc::new(PendingTracker::new());
        let hash1 = test_hash(111);
        let hash2 = test_hash(222);
        let hash3 = test_hash(333);

        let guard1 = tracker.try_claim(hash1).expect("claim first hash");
        let guard2 = tracker.try_claim(hash2).expect("claim second hash");

        assert!(tracker.is_pending(&hash1));
        assert!(tracker.is_pending(&hash2));
        assert!(!tracker.is_pending(&hash3));
        assert_eq!(tracker.len(), 2);

        drop(guard1);
        assert!(!tracker.is_pending(&hash1));
        assert!(tracker.is_pending(&hash2));
        assert_eq!(tracker.len(), 1);

        drop(guard2);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_guard_sequence_hash_accessor() {
        let tracker = Arc::new(PendingTracker::new());
        let hash = test_hash(42);

        let guard = tracker.try_claim(hash).expect("claim hash");
        assert_eq!(guard.sequence_hash(), hash);
    }

    #[test]
    fn test_tracker_debug() {
        let tracker = PendingTracker::new();
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("PendingTracker"));
    }

    #[test]
    fn test_guard_debug() {
        let tracker = Arc::new(PendingTracker::new());
        let hash = test_hash(999);
        let guard = tracker.try_claim(hash).expect("claim hash");

        let debug_str = format!("{:?}", guard);
        assert!(debug_str.contains("PendingGuard"));
        assert!(debug_str.contains("sequence_hash"));
    }

    #[test]
    fn concurrent_claim_has_one_owner_until_drop() {
        const CLAIMERS: usize = 16;
        let tracker = Arc::new(PendingTracker::new());
        let hash = test_hash(555);
        let start = Arc::new(std::sync::Barrier::new(CLAIMERS));
        let claimers: Vec<_> = (0..CLAIMERS)
            .map(|_| {
                let tracker = Arc::clone(&tracker);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    tracker.try_claim(hash)
                })
            })
            .collect();
        let mut winners: Vec<_> = claimers
            .into_iter()
            .filter_map(|claimer| claimer.join().expect("claim thread does not panic"))
            .collect();

        assert_eq!(winners.len(), 1);
        assert!(tracker.is_pending(&hash));
        assert_eq!(tracker.len(), 1);

        assert!(tracker.try_claim(hash).is_none());
        assert!(tracker.is_pending(&hash));

        drop(winners.pop().expect("one winner"));
        assert!(!tracker.is_pending(&hash));

        let reclaim = tracker.try_claim(hash).expect("reclaim after owner drop");
        assert!(tracker.is_pending(&hash));
        drop(reclaim);
        assert!(tracker.is_empty());
    }
}
