// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block lifecycle orchestration over the unified [`BlockStore`].
//!
//! [`BlockManager`] owns a single [`BlockStore`] and the [`BlockRegistry`].
//! All pool transitions go through the store's single mutex; the manager
//! adds the registry coordination, allocation eviction policy, and metrics.

mod builder;

#[cfg(test)]
mod tests;

pub use crate::pools::backends::ScorerParams;
pub use builder::{
    BlockManagerBuilderError, BlockManagerConfigBuilder, BlockManagerResetError,
    FrequencyTrackingCapacity, InactiveBackendConfig, LineageEviction,
};

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use parking_lot::RwLock;

use crate::blocks::{BlockMetadata, CompleteBlock, ImmutableBlock, MutableBlock};
use crate::metrics::BlockPoolMetrics;
use crate::pools::{
    BlockDuplicationPolicy, BlockStore, InactiveCandidate, InactiveFeatures, ReleaseOpts,
    SequenceHash,
};
use crate::registry::BlockRegistry;

/// Manages the full block lifecycle over the unified [`BlockStore`].
///
/// Construct via [`BlockManager::builder()`].
pub struct BlockManager<T: BlockMetadata> {
    pub(crate) store: Arc<BlockStore<T>>,
    pub(crate) block_registry: BlockRegistry,
    pub(crate) inactive_backend: InactiveBackendConfig,
    pub(crate) duplication_policy: BlockDuplicationPolicy,
    pub(crate) total_blocks: usize,
    pub(crate) block_size: usize,
    pub(crate) metrics: Arc<BlockPoolMetrics>,
    eviction_observers: RwLock<Vec<Weak<dyn BlockEvictionObserver>>>,
}

/// Batch callback fired after inactive slots have been evicted and reset.
pub trait BlockEvictionObserver: Send + Sync + 'static {
    fn on_blocks_evicted(&self, hashes: &[SequenceHash]);
}

impl<F> BlockEvictionObserver for F
where
    F: Fn(&[SequenceHash]) + Send + Sync + 'static,
{
    fn on_blocks_evicted(&self, hashes: &[SequenceHash]) {
        self(hashes);
    }
}

impl<T: BlockMetadata + Sync> BlockManager<T> {
    /// Create a new builder for `BlockManager`.
    pub fn builder() -> BlockManagerConfigBuilder<T> {
        BlockManagerConfigBuilder::default()
    }

    /// Stable, process-unique identifier for this manager's underlying
    /// [`BlockStore`](crate::pools::BlockStore). See [`crate::ManagerId`].
    /// Cheap (one field load via the store).
    ///
    /// Together with a [`BlockId`](crate::BlockId) this names a specific
    /// physical pool slot — the disambiguating runtime address that
    /// downstream consumers need after the policy parameter `T` has been
    /// type-erased through [`crate::LifecyclePinRef`].
    pub fn id(&self) -> crate::ManagerId {
        self.store.id()
    }

    /// Allocate `count` mutable blocks, drawing first from the reset pool
    /// and then evicting from the inactive pool if needed.
    ///
    /// Returns `None` if fewer than `count` blocks are available across both pools.
    pub fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>> {
        self.allocate_blocks_with_evictions(count)
            .map(|(blocks, _evicted)| blocks)
    }

    /// Like [`allocate_blocks`](Self::allocate_blocks) but also reports the
    /// [`SequenceHash`] of each block evicted from the inactive pool.
    pub fn allocate_blocks_with_evictions(
        &self,
        count: usize,
    ) -> Option<(Vec<MutableBlock<T>>, Vec<SequenceHash>)> {
        let allocation = self.store.allocate_atomic(count)?;
        self.notify_evictions(&allocation.1);
        Some(allocation)
    }

    /// Drain the inactive pool, returning all blocks to the reset pool.
    pub fn reset_inactive_pool(&self) -> Result<(), BlockManagerResetError> {
        let (blocks, evicted) = self.store.drain_inactive_to_mutable();
        self.notify_evictions(&evicted);
        drop(blocks);

        let reset_count = self.store.reset_len();
        if reset_count != self.total_blocks {
            return Err(BlockManagerResetError::BlockCountMismatch {
                expected: self.total_blocks,
                actual: reset_count,
            });
        }

        Ok(())
    }

    fn notify_evictions(&self, hashes: &[SequenceHash]) {
        if hashes.is_empty() {
            return;
        }
        let observers = {
            let mut registered = self.eviction_observers.write();
            let mut live = Vec::with_capacity(registered.len());
            registered.retain(|observer| {
                if let Some(observer) = observer.upgrade() {
                    live.push(observer);
                    true
                } else {
                    false
                }
            });
            live
        };
        for observer in observers {
            observer.on_blocks_evicted(hashes);
        }
    }

    /// Register a batch of completed blocks.
    pub fn register_blocks(&self, blocks: Vec<CompleteBlock<T>>) -> Vec<ImmutableBlock<T>> {
        blocks
            .into_iter()
            .map(|block| self.register_block(block))
            .collect()
    }

    /// Release a batch of immutable blocks under a *single* store-mutex
    /// acquisition — the batched inverse of [`register_blocks`](Self::register_blocks).
    ///
    /// Dropping N `ImmutableBlock`s one-at-a-time takes the store lock N
    /// times (once per `Drop` → `release_primary`); routing them through
    /// here takes it once. `reset_on_release`, when `Some(v)`, overrides
    /// every released block's per-slot reset flag *inside that same
    /// critical section* — `Some(true)` sends them straight to `Reset`
    /// (an eviction teardown), `Some(false)` forces the inactive pool,
    /// `None` leaves each slot's existing override (or the store-wide
    /// default) untouched (an ordinary finish). This replaces a separate
    /// per-block `ImmutableBlock::set_evict_on_reset` traversal that would
    /// otherwise take the lock once more per block.
    pub fn release_blocks(&self, blocks: Vec<ImmutableBlock<T>>, reset_on_release: Option<bool>) {
        self.store
            .release_blocks(blocks, ReleaseOpts { reset_on_release });
    }

    /// Mark the single-owner inactive lineage suffix ending at the leaf
    /// `seq_hash` for evict-first (compaction poison). Membership-based: a
    /// no-op unless the leaf is currently resident-inactive, and only the
    /// valued lineage backend acts on it — every other backend ignores it.
    /// The walk stops at (without poisoning) the first shared branch point, so
    /// blocks a live sibling still needs are never demoted. Wired to a client
    /// compaction hint at the runtime layer (EV-PR4); mirrors the additive
    /// [`release_blocks`](Self::release_blocks) / `has_inactive` wrappers.
    pub fn poison_lineage(&self, seq_hash: SequenceHash) {
        self.store.poison_lineage(seq_hash);
    }

    /// Test-only: whether `seq_hash`'s resident inactive node is marked
    /// poisoned. Lets tests observe [`Self::poison_lineage`] reaching the
    /// valued backend end-to-end (EV-PR4).
    #[cfg(test)]
    pub(crate) fn test_is_poisoned(&self, seq_hash: SequenceHash) -> bool {
        self.store.test_is_poisoned(seq_hash)
    }

    /// Register a single completed block and return an immutable handle.
    pub fn register_block(&self, block: CompleteBlock<T>) -> ImmutableBlock<T> {
        self.metrics.inc_registrations();
        let handle = self
            .block_registry
            .register_sequence_hash(block.sequence_hash());
        let inner = handle.register_block(block, self.duplication_policy, &self.store);
        ImmutableBlock::from_inner(inner)
    }

    /// Linear prefix match: walks `seq_hash` left-to-right, stopping on
    /// the first hash that hits neither the active nor the inactive pool.
    ///
    /// The whole active-or-inactive prefix is resolved under a **single**
    /// store-mutex acquisition via [`BlockStore::match_prefix_locked_batch`]
    /// — no per-hash registry radix-tree lookup, no per-hash store lock.
    /// Frequency-tracker touches are batched and applied *after* the store
    /// lock is released: every returned block is touched exactly once
    /// (including inactive resurrections).
    pub fn match_blocks(&self, seq_hash: &[SequenceHash]) -> Vec<ImmutableBlock<T>> {
        self.metrics
            .inc_match_hashes_requested(seq_hash.len() as u64);

        if seq_hash.is_empty() {
            self.metrics.inc_match_blocks_returned(0);
            return Vec::new();
        }

        // ONE store-lock acquisition for the whole active+inactive prefix.
        let inners = self.store.match_prefix_locked_batch(seq_hash);

        // Frequency-tracker touches, batched, AFTER the store lock is
        // released. Touches every returned hit exactly once — including
        // inactive resurrections, which the old `find_inactive_primaries`
        // path never touched.
        if self.block_registry.has_frequency_tracking() {
            for inner in &inners {
                self.block_registry.touch(inner.sequence_hash());
            }
        }

        let matched: Vec<ImmutableBlock<T>> =
            inners.into_iter().map(ImmutableBlock::from_inner).collect();

        self.metrics.inc_match_blocks_returned(matched.len() as u64);
        tracing::debug!(
            num_hashes = seq_hash.len(),
            total_matched = matched.len(),
            "match_blocks result"
        );
        tracing::trace!(matched = ?matched, "matched blocks");
        matched
    }

    /// Scatter-gather scan: finds all blocks matching any hash, without
    /// stopping on misses.
    pub fn scan_matches(
        &self,
        seq_hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<T>> {
        self.metrics
            .inc_scan_hashes_requested(seq_hashes.len() as u64);

        let mut result = HashMap::new();

        let active_found = self.scan_active_matches(seq_hashes, touch);
        for (hash, inner) in active_found {
            result.insert(hash, ImmutableBlock::from_inner(inner));
        }

        let remaining: Vec<SequenceHash> = seq_hashes
            .iter()
            .filter(|h| !result.contains_key(h))
            .copied()
            .collect();

        if !remaining.is_empty() {
            let inactive_found = self.store.scan_inactive_primaries(&remaining, touch);
            for (hash, inner) in inactive_found {
                result.insert(hash, ImmutableBlock::from_inner(inner));
            }
        }

        self.metrics.inc_scan_blocks_returned(result.len() as u64);

        result
    }

    /// Scan-style active lookup by sequence hash via the registry's
    /// stored Weak references — does not stop on miss.
    fn scan_active_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Arc<crate::blocks::ImmutableBlockInner<T>>)> {
        hashes
            .iter()
            .filter_map(|hash| {
                self.block_registry
                    .match_sequence_hash(*hash, touch)
                    .and_then(|handle| {
                        handle
                            .try_get_inner::<T>(&self.store, touch)
                            .map(|inner| (*hash, inner))
                    })
            })
            .collect()
    }

    /// Read-only snapshot of up to `max` inactive blocks that this pool's
    /// eviction policy ranks worst-first (poisoned leaves, then lowest-value
    /// or oldest). Non-destructive — it does not resurrect a block, touch its
    /// frequency, reorder the policy, or re-seed its sampling RNG.
    ///
    /// # What the order does and does not promise
    ///
    /// It ranks the blocks that are eviction candidates *right now*, which is
    /// not the same as replaying the pool's next `max` evictions:
    ///
    /// * on the lineage backend only leaves are candidates, and draining one
    ///   re-leafs its parent — a block this snapshot could not have listed,
    ///   because it was structurally unevictable when the peek ran. Where the
    ///   parent lands is policy-dependent: the total-order policies re-admit
    ///   it at its own older tick, *ahead* of candidates listed behind it,
    ///   while the valued policy restamps its recency so it re-enters as the
    ///   freshest leaf and sinks to the *back*. Either way the real drain
    ///   sequence diverges past the head. Interior nodes are absent here
    ///   entirely; reach them with [`inactive_advice`](Self::inactive_advice).
    /// * the head itself is exact for the total-order leaf policies (`Tick`,
    ///   `Fifo`), and for the poison prefix under the valued policy. Under the
    ///   valued policy with no poison it is a best-effort minimum, because real
    ///   eviction samples rather than scanning — see
    ///   [`InactiveFeatures::evict_rank`].
    ///
    /// A result shorter than `max` therefore does not mean the pool holds no
    /// further blocks: the valued policy scores only a bounded window of its
    /// leaf set per call (currently 4096 leaves, beyond any poison prefix,
    /// which is not subject to the window), and blocks that are structurally
    /// unevictable are never listed at all. Use
    /// [`inactive_len`](Self::inactive_len) for pool depth.
    ///
    /// Advisory only: entries are stale the instant the store lock drops, so a
    /// consumer must re-acquire authority over any block it acts on through
    /// the ordinary match / pin / hold path, and treat a candidate that went
    /// active or was evicted in between as *skipped*, not as an error.
    ///
    /// Empty on backends that expose no eviction order (`HashMap`, `Lru`,
    /// `MultiLru`); a consumer must degrade rather than depend on the signal.
    /// Intended for an out-of-band pressure pass — never the allocation path.
    pub fn inactive_candidates(&self, max: usize) -> Vec<InactiveCandidate> {
        self.store.inactive_candidates(max)
    }

    /// Membership-based point advice, positionally aligned with `hashes`.
    /// `None` = that hash is not currently resident-inactive in this pool
    /// (it is active, absent, or the backend tracks no features). Resolved
    /// under a single store-lock acquisition, with the same read-only
    /// guarantees as [`inactive_candidates`](Self::inactive_candidates).
    ///
    /// Unlike a peek, this reaches *interior* lineage nodes too (reported with
    /// `is_leaf: false`), so a consumer can ask about a block it already knows
    /// about rather than only about eviction candidates.
    pub fn inactive_advice(&self, hashes: &[SequenceHash]) -> Vec<Option<InactiveFeatures>> {
        self.store.inactive_advice(hashes)
    }

    /// Depth of the inactive (cached) pool: every registered block held there,
    /// reclaimable by eviction but not necessarily a candidate today.
    ///
    /// This is **not** the count of blocks the pool could free right now. On
    /// the lineage backend an interior node is structurally protected by its
    /// descendants, so it is counted here yet never offered by
    /// [`inactive_candidates`](Self::inactive_candidates) — a 3-block
    /// single-owner chain reports `3` with exactly one candidate. Freeing an
    /// interior node takes draining the leaves below it first. Size
    /// immediately free-able supply from the candidate list, not from this.
    ///
    /// Cheap: one store-lock acquisition. Named for the pool it reports, not
    /// for this type's `*_blocks` getters — it is the pressure-pass companion
    /// of [`reset_len`](Self::reset_len).
    pub fn inactive_len(&self) -> usize {
        self.store.inactive_len()
    }

    /// Number of blocks currently in the reset (free) pool. Cheap: one
    /// store-lock acquisition. `reset_len + inactive_len` is the same quantity
    /// as [`available_blocks`](Self::available_blocks), but that one reads both
    /// under a *single* lock — prefer it when the sum is what matters.
    pub fn reset_len(&self) -> usize {
        self.store.reset_len()
    }

    /// Total number of blocks managed (constant after construction).
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Blocks available for allocation (reset + inactive pools).
    ///
    /// Reads both pool sizes under a single store-lock acquisition so the
    /// returned value is a coherent snapshot, never an over- or under-count
    /// produced by a concurrent reset↔inactive transition.
    pub fn available_blocks(&self) -> usize {
        self.store.available_len()
    }

    /// Tokens per block (constant after construction).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Current duplication policy.
    pub fn duplication_policy(&self) -> &BlockDuplicationPolicy {
        &self.duplication_policy
    }

    /// Reference to the shared block registry.
    pub fn block_registry(&self) -> &BlockRegistry {
        &self.block_registry
    }

    /// Construction policy selected for this manager's inactive pool.
    pub const fn inactive_backend(&self) -> &InactiveBackendConfig {
        &self.inactive_backend
    }

    /// Observe future inactive-pool evictions while the caller retains `observer`.
    pub fn observe_evictions(&self, observer: &Arc<dyn BlockEvictionObserver>) {
        self.eviction_observers
            .write()
            .push(Arc::downgrade(observer));
    }

    /// Reference to the block pool metrics.
    pub fn metrics(&self) -> &Arc<BlockPoolMetrics> {
        &self.metrics
    }

    /// Test-only accessor for the underlying [`BlockStore`]. Used to
    /// reach test hooks like `BlockStore::pause_release_primary` from
    /// race-window tests.
    #[cfg(test)]
    pub(crate) fn store_for_test(&self) -> &Arc<BlockStore<T>> {
        &self.store
    }
}
