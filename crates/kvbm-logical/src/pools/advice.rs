// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only advisory snapshot of the inactive pool (R7a).
//!
//! [`BlockManager::inactive_candidates`](crate::manager::BlockManager::inactive_candidates)
//! and
//! [`BlockManager::inactive_advice`](crate::manager::BlockManager::inactive_advice)
//! hand these types out to a *pressure pass* — an out-of-band consumer that
//! ranks residency, never the allocation path itself.
//!
//! # Staleness
//!
//! Every value here is a snapshot taken under the store mutex and is stale the
//! instant that lock drops: the named block may go active, be resurrected by a
//! cache hit, or be evicted before the consumer acts. Nothing in this module
//! confers authority over a block — a transfer must re-acquire it through the
//! ordinary match / pin / hold path, and a candidate that moved on is a
//! *skipped* candidate, never an error. Exact reclaim binds this snapshot to
//! a manager, a mutable generation, and an inactive epoch. It rejects a
//! moved candidate without mutation.
//!
//! # Non-mutation
//!
//! Producing these values must not touch the frequency sketch, resurrect a
//! block, reorder the eviction policy, or consume a sampling RNG — the
//! determinism invariant carried by the crate-private `InactiveIndex`
//! backend hooks that produce them.

use crate::BlockId;
use crate::blocks::SequenceHash;

/// One inactive block reported by a bounded, read-only peek, together with the
/// features the backend can cheaply expose for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InactiveCandidate {
    /// Hash the block is registered under while inactive.
    pub seq_hash: SequenceHash,
    /// Pool slot holding it. Together with the manager's
    /// [`ManagerId`](crate::ManagerId) this names a physical slot.
    pub block_id: BlockId,
    /// Mutable-allocation tenure of this physical slot.
    ///
    /// A fresh mutable allocation changes this value. Exact reclaim names bind
    /// this generation to the candidate's current manager and physical slot.
    pub generation: u64,
    /// Inactive residency tenure of this physical slot.
    ///
    /// A cache hit keeps the mutable-allocation generation, but it ends this
    /// tenure when its returned `ImmutableBlock` drops. Exact reclaim refresh
    /// resolves the current epoch and rejects a changed mutable generation.
    pub inactive_epoch: u64,
    /// Advisory features for this block; see [`InactiveFeatures`].
    pub features: InactiveFeatures,
}

/// Cheap, backend-exposable features of one resident-inactive block.
///
/// Every `Option` field is `None` when the backend does not track that signal
/// at all (rather than "zero"), so a consumer can tell "absent" from "measured
/// low" — the distinction the value model needs to fail open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InactiveFeatures {
    /// Compaction-poisoned (client-declared dead). Always `false` on backends
    /// without poison tracking.
    ///
    /// AUTHORITY BOUND: this bit is *sticky* and carries no provenance,
    /// confidence, or generation — the backend stores a bare `bool` set by a
    /// client compaction hint (see
    /// [`BlockManager::poison_lineage`](crate::manager::BlockManager::poison_lineage)),
    /// and nothing here proves the hint still describes the current
    /// generation of that lineage. A policy consuming this API must therefore
    /// treat the bit as **heuristic-grade evidence** — enough to rank a block
    /// down, never enough on its own to justify a hard, unrecoverable
    /// eviction. Exact-grade poison needs current-generation equality plus the
    /// branch-safety proof, and those exist only on the manager-side
    /// compaction path that *writes* the mark, never on this read path.
    pub poisoned: bool,
    /// Currently an evictable leaf (vs. an interior node, structurally
    /// protected by its descendants). Interior nodes are visible through
    /// [`BlockManager::inactive_advice`](crate::manager::BlockManager::inactive_advice)
    /// but never appear in a peek, since only leaves are eviction candidates.
    pub is_leaf: bool,
    /// Pool-logical age in the backend's own tick units — *not* wall time and
    /// not comparable across pools. `None` for backends with no ordering
    /// state.
    pub age_ticks: Option<u64>,
    /// Frequency-sketch estimate for this hash; `None` when the pool has no
    /// frequency tracker attached.
    pub freq_estimate: Option<u32>,
    /// Branch-oracle peak fan-out for this hash as a parent; `None` when there
    /// is no oracle *or* no record for the hash. Absent is not "linear" —
    /// consumers must fail open to "possibly shared".
    pub max_fanout: Option<u32>,
    /// Rank bucket in `[0, 255]` for this entry's position in the peek that
    /// produced it: the head of the batch is `0`, and for a batch of more than
    /// one entry the tail is `255`.
    ///
    /// **The bucket is quantized, so `0` does not name a unique block.** Any
    /// peek longer than 256 entries puts several candidates in each bucket —
    /// a 4096-entry batch lands its first 17 in bucket `0`. Only an entry's
    /// *position* in the returned slice identifies the head; filtering on
    /// `evict_rank == Some(0)` selects a group, not a block.
    ///
    /// **Nor is rank a promise about the next eviction** — it is the peek's
    /// own order. How tightly the head tracks the real victim depends on the
    /// backend's leaf policy:
    ///
    /// * total-order policies (lineage `Tick`, `Fifo`) walk their order
    ///   directly, so rank `0` *is* the block the pool would evict next;
    /// * the valued policy (the G1 construction) has no total order. Rank `0`
    ///   is the head of the poison prefix when one exists — also exact, since
    ///   real eviction drains poison first — but otherwise it is only the
    ///   *best-effort* minimum of the peek's bounded scan. The real eviction
    ///   path instead takes a sampled minimum over `k_sample` random draws
    ///   (that sampling is what the peek must not perturb; see
    ///   [`crate::pools::advice`]), so with more resident leaves than
    ///   `k_sample` the two routinely disagree.
    ///
    /// Consumers should read rank as "the pool ranks this low", never as "the
    /// pool is about to evict this".
    ///
    /// Coarse **by design**: a byte, not a score. Rhino's controller has its
    /// own value model, so exporting the backend's raw score would invite
    /// double-scoring and freeze backend internals into a public contract.
    ///
    /// `None` when the backend exposes no total order, **or** when the
    /// features came from a point
    /// [`BlockManager::inactive_advice`](crate::manager::BlockManager::inactive_advice)
    /// query — rank is only defined *within one peek batch* (ranking a single
    /// hash would cost the O(n) scan that point advice's O(1) contract
    /// forbids).
    pub evict_rank: Option<u8>,
}

/// Scale a 0-based position among `len` peeked candidates into the coarse
/// [`InactiveFeatures::evict_rank`] bucket: the head of the peek is always `0`
/// and, for `len > 1`, the tail is always `255`.
///
/// The rank is *relative to the returned slice*, not to the whole pool — a
/// peek of 4 out of 4000 blocks still spans the full byte range.
pub(crate) fn evict_rank_for(index: usize, len: usize) -> Option<u8> {
    if len == 0 || index >= len {
        return None;
    }
    if len == 1 {
        return Some(0);
    }
    // `index <= len - 1`, so the quotient is in [0, 255]; u64 math keeps the
    // product exact for any peek size, and the conversion saturates rather
    // than truncating (never reached, but the width contract is explicit).
    let scaled = (index as u64 * 255) / (len as u64 - 1);
    Some(u8::try_from(scaled).unwrap_or(u8::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evict_rank_spans_the_byte_range() {
        assert_eq!(evict_rank_for(0, 0), None, "empty peek has no rank");
        assert_eq!(evict_rank_for(1, 1), None, "index past the slice");
        assert_eq!(evict_rank_for(0, 1), Some(0), "a lone candidate is next");
        assert_eq!(evict_rank_for(0, 4), Some(0));
        assert_eq!(evict_rank_for(3, 4), Some(255));
        // Monotone non-decreasing across a batch.
        let ranks: Vec<u8> = (0..8).filter_map(|i| evict_rank_for(i, 8)).collect();
        assert_eq!(ranks.len(), 8);
        assert!(ranks.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(*ranks.last().unwrap(), 255);
    }

    /// The bucket is lossy once a batch outruns the byte, so rank `0` names a
    /// *group*, not the head. Pinned because the field doc says so and a
    /// consumer filtering on `Some(0)` would otherwise expect one block.
    #[test]
    fn rank_zero_is_a_bucket_not_a_unique_head() {
        // 4096 is the valued policy's peek scan window — a realistic batch.
        let zeros = (0..4096).filter(|&i| evict_rank_for(i, 4096) == Some(0));
        assert_eq!(
            zeros.count(),
            17,
            "a 4096-entry batch quantizes its first 17 positions into bucket 0"
        );
        // The head is still rank 0, and the boundary is where the doc says.
        assert_eq!(evict_rank_for(0, 4096), Some(0));
        assert_eq!(evict_rank_for(16, 4096), Some(0));
        assert_eq!(evict_rank_for(17, 4096), Some(1));
        // At or under the byte, every position gets its own bucket.
        let distinct: std::collections::HashSet<u8> =
            (0..256).filter_map(|i| evict_rank_for(i, 256)).collect();
        assert_eq!(distinct.len(), 256, "a 256-entry batch is lossless");
    }
}
