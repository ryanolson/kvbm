// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-mutex block store: unified bookkeeping for the reset, active,
//! and inactive pools.
//!
//! `BlockStore<T>` owns the entire block bookkeeping for a single
//! metadata tier. The unified mutex protects:
//!
//! - `slots: Vec<BlockSlot<T>>` — source of truth for every slot's state.
//! - `free: VecDeque<BlockId>` — reset pool (FIFO).
//! - `inactive: Box<dyn InactiveIndex>` — pluggable eviction-order index
//!   over slots in `Inactive` state.
//! - `active_by_hash: SeqHashMap<BlockId>` — primary block_id for each
//!   currently-registered hash (identity-hashed; see [`IdHasher`](super::IdHasher)).
//!
//! Active-pool lookup, slot transitions, and resurrection all happen
//! under one lock, so no across-lock gap can leave a hash unreachable
//! from both the active and inactive pools at the same time.
//!
//! # Lock ordering
//!
//! `BlockRegistrationHandle.attachments` (Mutex inside the registry) →
//! `BlockStore.inner` (Mutex). Never the reverse.

use std::collections::VecDeque;
use std::sync::{Arc, Weak};

// Under `#[cfg(test)]` use `tracing-mutex`'s parking_lot wrapper, which
// is API-identical to `parking_lot::Mutex` but builds a global
// lock-acquisition DAG and panics on order inversions or cycles. This
// turns the documented `attachments → store` ordering invariant into
// runtime-enforced behaviour during the test suite. In release/non-test
// builds the alias resolves to plain `parking_lot::Mutex` — zero cost.
#[cfg(not(test))]
use parking_lot::Mutex;
#[cfg(test)]
use tracing_mutex::parkinglot::Mutex;

use crate::BlockId;
use crate::blocks::{
    BlockDuplicationPolicy, BlockMetadata, CompleteBlock, ImmutableBlock, ImmutableBlockInner,
    MutableBlock, SequenceHash,
};
use crate::metrics::BlockPoolMetrics;
use crate::registry::BlockRegistrationHandle;

// Identity hashing for `SequenceHash`-keyed maps lives in `pools` — it is
// shared by `active_by_hash` here and by the inactive-pool backends.
use super::advice::{InactiveCandidate, InactiveFeatures};
use super::{ExactInactiveVictim, SeqHashMap};

#[cfg(test)]
mod exact_inactive;
mod exact_reclaim;
mod inactive_lineage_hold;

pub(crate) use exact_reclaim::ExactReclaimPlanError;
pub(crate) use inactive_lineage_hold::StoreInactiveLineageHold;

/// Index trait for inactive-pool eviction backends. T-free: backends only
/// need `(SequenceHash, BlockId)` pairs.
pub(crate) trait InactiveIndex: Send + Sync {
    /// Find blocks for the given hashes in order, stopping on first miss.
    /// Removes matched entries from the index.
    fn find_matches(
        &mut self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, BlockId)>;

    /// Find a single block matching `hash`. Default impl delegates to
    /// `find_matches`; backends override for an O(1) variant that
    /// avoids slice iteration on the single-block fast-path.
    fn find_match(&mut self, hash: SequenceHash, touch: bool) -> Option<(SequenceHash, BlockId)> {
        self.find_matches(&[hash], touch).into_iter().next()
    }

    /// Like `find_matches` but does not stop on miss.
    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, BlockId)>;

    /// Pull `count` blocks for eviction in policy order.
    fn allocate(&mut self, count: usize) -> Vec<(SequenceHash, BlockId)>;

    /// Make `block_id` evictable under `seq_hash`.
    fn insert(&mut self, seq_hash: SequenceHash, block_id: BlockId);

    fn len(&self) -> usize;

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn has(&self, seq_hash: SequenceHash) -> bool;

    /// Test exact inactive reclaim eligibility without changing policy state.
    ///
    /// A `true` result must guarantee that the next [`Self::take`] succeeds
    /// under the same store lock and that the pair is evictable under the
    /// backend policy. The default rejects exact reclaim until a backend
    /// provides that guarantee.
    #[cfg(test)]
    fn contains(&self, seq_hash: SequenceHash, block_id: BlockId) -> bool {
        let _ = (seq_hash, block_id);
        false
    }

    /// Remove a specific `block_id`/`seq_hash` pair if present.
    #[allow(dead_code)]
    fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool;

    /// Atomically remove the complete real lineage from root through the
    /// exact inactive leaf. Backends without lineage data return `None`.
    fn take_complete_lineage(
        &mut self,
        seq_hash: SequenceHash,
        block_id: BlockId,
    ) -> Option<Vec<(SequenceHash, BlockId)>> {
        let _ = (seq_hash, block_id);
        None
    }

    /// Read the complete real lineage without changing backend state.
    ///
    /// A backend that implements [`Self::take_complete_lineage`] must return
    /// the same list here while the caller holds the store lock. The pressure
    /// owner uses this preview to reject an overlap before extraction.
    fn complete_lineage(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
    ) -> Option<Vec<(SequenceHash, BlockId)>> {
        let _ = (seq_hash, block_id);
        None
    }

    /// Return whether this backend can resolve and validate exact reclaim.
    fn supports_exact_reclaim(&self) -> bool {
        false
    }

    /// Resolve one exact inactive entry leaf without exposing its block identity.
    fn exact_block_id(&self, seq_hash: SequenceHash) -> Option<BlockId> {
        let _ = seq_hash;
        None
    }

    /// Verify that `victims` names one complete, safe removal sequence.
    ///
    /// The store holds its mutex for this preflight and the later commit. A
    /// successful result must guarantee that each subsequent [`Self::take`]
    /// in the supplied order succeeds without a structural policy mutation
    /// before the first take. Backends without this proof reject non-empty
    /// plans. An empty plan needs no backend proof.
    fn preflight_exact_reclaim(
        &self,
        victims: &[ExactInactiveVictim],
    ) -> Result<(), ExactReclaimPlanError> {
        if victims.is_empty() {
            Ok(())
        } else {
            Err(ExactReclaimPlanError::Unsupported)
        }
    }

    /// Mark the single-owner lineage suffix ending at `seq_hash` for evict-first
    /// (compaction poison). Default is a no-op — only the lineage backend's valued leaf
    /// policy acts on it; every other backend ignores it. Wired to a client compaction
    /// hint through `BlockManager::poison_lineage` (EV-PR4).
    fn poison(&mut self, seq_hash: SequenceHash) {
        let _ = seq_hash;
    }

    /// Test-only: whether the resident node for `seq_hash` is marked poisoned.
    /// Default `false` — only the lineage backend's valued policy tracks poison
    /// marks. Mirrors [`Self::has`] so `BlockManager::test_is_poisoned` can
    /// observe the compaction-poison wiring end-to-end (EV-PR4).
    #[cfg(test)]
    fn test_is_poisoned(&self, seq_hash: SequenceHash) -> bool {
        let _ = seq_hash;
        false
    }

    /// Drain the entire index.
    fn allocate_all(&mut self) -> Vec<(SequenceHash, BlockId)> {
        let n = self.len();
        self.allocate(n)
    }

    /// Read-only ranked peek: up to `max` inactive blocks that this index
    /// currently ranks worst-first. Default: empty — a backend that exposes no
    /// order simply advertises no candidates, and the consumer degrades to
    /// its own registration-driven selection (R7a §4).
    ///
    /// # Scope of the order (implementor contract)
    ///
    /// The result ranks the blocks that are candidates *at this instant*; it
    /// is not required to predict the index's next `max` [`Self::allocate`]
    /// results. An index whose candidate set changes as it drains (the lineage
    /// backend re-leafs a parent when it evicts a leaf) will legitimately
    /// diverge from a real drain past the head. Implementors should keep the
    /// **head** exact wherever the policy admits an exact answer, and document
    /// where it cannot.
    ///
    /// Returning fewer than `max` entries is likewise not a claim that the
    /// index is exhausted — a bounded-scan implementation may report only what
    /// it scanned.
    ///
    /// # Determinism (load-bearing)
    ///
    /// Implementations MUST NOT mutate policy state — not the recency clock,
    /// not the ordering structures, and **not any sampling RNG**. An RNG
    /// advanced by a read-only peek would (a) perturb the *next real* victim
    /// draw, an observer effect on eviction, and (b) make a replayed trace
    /// non-reproducible from identical inputs. `&self` is the enforcement:
    /// the only interior mutability reachable from here is the shared
    /// frequency sketch / branch oracle, which must be *read* only (no
    /// `touch`).
    ///
    /// Results are advisory and stale the moment the store lock drops; see
    /// [`crate::pools::advice`].
    fn peek_victims(&self, max: usize) -> Vec<(SequenceHash, BlockId, InactiveFeatures)> {
        let _ = max;
        Vec::new()
    }

    /// Read-only point advice for `seq_hash`. `None` means "not resident in
    /// this index" — the block is active, absent, or the backend tracks no
    /// features. Default: `None`. Same non-mutation contract as
    /// [`Self::peek_victims`].
    fn advice(&self, seq_hash: SequenceHash) -> Option<InactiveFeatures> {
        let _ = seq_hash;
        None
    }
}

/// State of an individual slot. The variant determines all drop transitions
/// and resurrection semantics. Tracked under the unified store mutex.
///
/// `Primary`/`Duplicate` carry a `Weak<ImmutableBlockInner<T>>` so the
/// store can perform identity-checked drop transitions and serve active
/// lookups under the store mutex without consulting registry attachments.
#[allow(dead_code)]
pub(crate) enum SlotState<T: BlockMetadata> {
    /// In the `free` list; available for allocation.
    Reset,
    /// Held by a `MutableBlock`. Drop → `Reset`.
    Mutable,
    /// Held by a `CompleteBlock`. Drop → `Reset`.
    Staged { seq_hash: SequenceHash },
    /// Held by an `ImmutableBlock` whose inner is the canonical primary.
    /// Drop of last clone → `Inactive`.
    Primary {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
        inner: Weak<ImmutableBlockInner<T>>,
    },
    /// Held by an `ImmutableBlock` whose inner is a duplicate physical copy.
    /// Drop of last clone → `Reset` (with `mark_absent`).
    Duplicate {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
        inner: Weak<ImmutableBlockInner<T>>,
    },
    /// Idle, evictable, registered. In the inactive index under `seq_hash`.
    ///
    /// The per-block "reset on last drop" override lives in
    /// `BlockStoreInner::reset_on_release[block_id]` (a parallel `Vec<bool>`
    /// under the same mutex as this variant) rather than in the variant
    /// itself, so the override survives every transition into and out of
    /// `Inactive` without an extra hand-off:
    ///
    /// - `Primary → Inactive` (both `release_primary` and the
    ///   lookup-driven eager path): the bool is untouched, so the value
    ///   the last holder set via `set_evict_on_reset` is preserved.
    /// - `Inactive → Primary` (resurrection): the bool is untouched, so
    ///   the new `ImmutableBlockInner` reads the carried value on its
    ///   own drop.
    /// - `Inactive → Mutable` (eviction): the bool is reset to the
    ///   store-wide default, since a fresh tenant should not inherit a
    ///   previous holder's override.
    Inactive {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    },
    /// Temporarily owned by an out-of-band pressure action. The slot is
    /// registered but absent from both lookup maps and allocation pools.
    Held {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    },
}

impl<T: BlockMetadata> std::fmt::Debug for SlotState<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlotState::Reset => f.write_str("Reset"),
            SlotState::Mutable => f.write_str("Mutable"),
            SlotState::Staged { seq_hash } => f
                .debug_struct("Staged")
                .field("seq_hash", seq_hash)
                .finish(),
            SlotState::Primary { seq_hash, .. } => f
                .debug_struct("Primary")
                .field("seq_hash", seq_hash)
                .finish(),
            SlotState::Duplicate { seq_hash, .. } => f
                .debug_struct("Duplicate")
                .field("seq_hash", seq_hash)
                .finish(),
            SlotState::Inactive { seq_hash, .. } => f
                .debug_struct("Inactive")
                .field("seq_hash", seq_hash)
                .finish(),
            SlotState::Held { seq_hash, .. } => {
                f.debug_struct("Held").field("seq_hash", seq_hash).finish()
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct BlockSlot<T: BlockMetadata> {
    pub(crate) block_size: usize,
    /// Increments before every fresh mutable allocation for this slot.
    generation: u64,
    /// Increments every time this slot enters an inactive residency tenure.
    ///
    /// Unlike `generation`, this changes after a cache-hit resurrection
    /// returns to inactive. It lets exact reclaim reject an inactive →
    /// active → inactive ABA without changing fresh-allocation semantics.
    inactive_epoch: u64,
    pub(crate) state: SlotState<T>,
}

/// Inner state of a `BlockStore` — protected by a single mutex.
pub(crate) struct BlockStoreInner<T: BlockMetadata> {
    /// `slots[block_id]` — created at construction, never grows.
    slots: Vec<BlockSlot<T>>,
    /// Free list (reset pool). FIFO.
    free: VecDeque<BlockId>,
    /// Inactive eviction index (T-free).
    inactive: Box<dyn InactiveIndex>,
    /// Primary `block_id` for each currently-registered sequence hash.
    /// Updated atomically with the slot's `Primary`/`Inactive` state.
    /// Uses the identity [`IdHasher`](super::IdHasher) — the key is
    /// already a content hash.
    active_by_hash: SeqHashMap<BlockId>,
    /// Hashes whose canonical inactive lineage is owned by a pressure action.
    ///
    /// A concurrent registration can add a newer physical block with the
    /// same hash. Request lookup must still stop at this ownership fence
    /// until the action aborts or commits.
    held_by_hash: SeqHashMap<BlockId>,
    /// Per-slot "reset on last drop" override, indexed by `BlockId`.
    /// Length is fixed to `total_blocks` at construction.
    ///
    /// Written by [`crate::blocks::ImmutableBlock::set_evict_on_reset`]
    /// (which acquires this same mutex for the write); read by
    /// `release_primary` to choose the `Primary → Inactive` vs
    /// `Primary → Reset` transition. The eager `Primary → Inactive`
    /// path does *not* touch this field — it just transitions the slot
    /// — so a per-block override set by the dropping holder is
    /// preserved across the race window where a concurrent lookup
    /// beats `release_primary` to the mutex. The value rides through
    /// `Primary → Inactive → Primary` (resurrection) untouched, and is
    /// reset to the store-wide default on every transition into
    /// `Mutable` so a fresh tenant starts clean.
    ///
    /// Both writes and reads happen under the store mutex, so all
    /// visibility comes from the mutex's release-acquire semantics —
    /// no atomic-ordering subtleties.
    reset_on_release: Vec<bool>,
}

/// Single-mutex bookkeeping store for the reset, active, and inactive
/// pools.
pub(crate) struct BlockStore<T: BlockMetadata> {
    /// Stable, process-unique store identifier. Surfaced through
    /// `LifecyclePin::manager_id` so type-erased pins remain
    /// runtime-addressable to a unique physical pool.
    id: crate::ManagerId,
    inner: Mutex<BlockStoreInner<T>>,
    block_size: usize,
    total_blocks: usize,
    metrics: Arc<BlockPoolMetrics>,
    /// Store-wide default for the per-slot "reset on last drop" override.
    /// When `true`, every primary release bypasses the inactive pool and
    /// goes straight to `Reset` (mirrors `release_duplicate`). Individual
    /// holders can still override per-block via
    /// `ImmutableBlock::set_evict_on_reset`.
    default_reset_on_release: bool,
    /// Test-only hook to deterministically widen the
    /// "Arc strong=0 but `release_primary` not yet run" race window.
    /// `release_primary` acquires this gate *before* taking the store
    /// mutex; while a test holds the gate via
    /// `pause_release_primary()`, every `release_primary` call parks
    /// here without ever inspecting slot state, leaving the slot in
    /// `Primary { weak: dead }` for a concurrent lookup to observe.
    /// Production builds elide the field entirely.
    #[cfg(test)]
    release_primary_gate: Mutex<()>,

    /// Test-only arrival counter. Incremented at the very first
    /// instruction of every `release_primary` call (before the gate
    /// is contended). Tests use it to signal "the dropping thread has
    /// reached `release_primary` and is about to park", replacing
    /// scheduler-dependent sleeps in deterministic race tests.
    #[cfg(test)]
    release_primary_arrivals: std::sync::atomic::AtomicU64,
}

/// Options for [`BlockStore::release_blocks`].
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ReleaseOpts {
    /// When `Some(v)`, overwrite every released block's per-slot
    /// `reset_on_release` override with `v` inside the batch's single
    /// critical section — for every entry, regardless of whether it ends
    /// up released inline or deferred to its ordinary `Drop`. This
    /// replaces a separate per-block `ImmutableBlock::set_evict_on_reset`
    /// traversal (which would otherwise take the store mutex once per
    /// block, before the drop pass). `None` leaves each slot's existing
    /// override (from a prior `set_evict_on_reset` call, or the
    /// store-wide default) untouched.
    pub(crate) reset_on_release: Option<bool>,
}

/// Outcome counters for a [`BlockStore::release_blocks`] batch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ReleaseReport {
    /// Primary slots routed `Primary → Reset` inline, under the batch lock.
    pub(crate) primary_reset: usize,
    /// Primary slots routed `Primary → Inactive` inline, under the batch lock.
    pub(crate) primary_inactive: usize,
    /// Duplicate slots routed `Duplicate → Reset` inline, under the batch lock.
    pub(crate) duplicate_reset: usize,
    /// Entries left to their normal per-block `Drop` — either because the
    /// guard's `Inner` was still `Arc`-shared (a live duplicate's
    /// primary-keepalive, or another clone) at the moment we tried to
    /// take sole ownership via `Arc::try_unwrap`, or because the slot no
    /// longer matched this `Inner`'s identity by the time we reached the
    /// lock (defensive; see `release_blocks` docs).
    pub(crate) deferred_to_drop: usize,
}

#[allow(dead_code)]
impl ReleaseReport {
    /// Total entries actually released inline, under the batch's single
    /// lock acquisition (i.e. everything *not* deferred).
    pub(crate) fn released(&self) -> usize {
        self.primary_reset + self.primary_inactive + self.duplicate_reset
    }
}

/// Result of matching a still-alive `Arc`'s identity against its slot,
/// inside [`BlockStore::release_blocks`]'s single lock. Mirrors the
/// `self_ptr` check in `release_primary` / `release_duplicate`, but
/// captures the `(SequenceHash, BlockRegistrationHandle)` payload needed
/// to complete the transition instead of just a bool.
enum SlotIdentityMatch {
    Primary(SequenceHash, BlockRegistrationHandle),
    Duplicate(BlockRegistrationHandle),
}

/// Scratch entry for [`BlockStore::release_blocks`]'s single-pass
/// algorithm. `arc` is `Some` exactly while this entry is still
/// "unresolved" — not yet released inline and not yet swept into the
/// deferred/external set. See [`BlockStore::release_entry_at`].
struct ReleaseEntry<T: BlockMetadata> {
    block_id: BlockId,
    self_ptr: *const (),
    arc: Option<Arc<ImmutableBlockInner<T>>>,
}

#[allow(dead_code)]
impl<T: BlockMetadata + Sync> BlockStore<T> {
    pub(crate) fn new(
        total_blocks: usize,
        block_size: usize,
        inactive: Box<dyn InactiveIndex>,
        metrics: Arc<BlockPoolMetrics>,
        default_reset_on_release: bool,
    ) -> Arc<Self> {
        let mut slots = Vec::with_capacity(total_blocks);
        let mut free = VecDeque::with_capacity(total_blocks);
        for i in 0..total_blocks {
            slots.push(BlockSlot {
                block_size,
                generation: 0,
                inactive_epoch: 0,
                state: SlotState::Reset,
            });
            free.push_back(i);
        }
        let reset_on_release = vec![default_reset_on_release; total_blocks];
        Arc::new(Self {
            id: crate::ManagerId::next(),
            inner: Mutex::new(BlockStoreInner {
                slots,
                free,
                inactive,
                active_by_hash: SeqHashMap::default(),
                held_by_hash: SeqHashMap::default(),
                reset_on_release,
            }),
            block_size,
            total_blocks,
            metrics,
            default_reset_on_release,
            #[cfg(test)]
            release_primary_gate: Mutex::new(()),
            #[cfg(test)]
            release_primary_arrivals: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Store-wide default for the per-slot "reset on last drop" override.
    /// `true` makes registered blocks bypass the inactive pool on release
    /// (mirrors `release_duplicate`) unless a holder explicitly opts out
    /// via `ImmutableBlock::set_evict_on_reset(false)`.
    pub(crate) fn default_reset_on_release(&self) -> bool {
        self.default_reset_on_release
    }

    /// Set the per-block "reset on last drop" override for `block_id`.
    /// Backs [`crate::blocks::ImmutableBlock::set_evict_on_reset`].
    ///
    /// Acquires the store mutex so the write is published to any future
    /// mutex acquirer through release-acquire on the mutex itself —
    /// the per-slot value lives inside `BlockStoreInner` and is only
    /// read under that same mutex.
    pub(crate) fn store_reset_on_release(&self, block_id: BlockId, value: bool) {
        self.inner.lock().reset_on_release[block_id] = value;
    }

    /// Stable, process-unique identifier of this store. See
    /// [`crate::ManagerId`].
    pub(crate) fn id(&self) -> crate::ManagerId {
        self.id
    }

    /// Test-only: acquire a guard that pauses every subsequent
    /// `release_primary` *before* it takes the store mutex, leaving
    /// the slot in `Primary { weak: dead }` so a concurrent lookup
    /// can drive the eager `Primary → Inactive` branch
    /// deterministically. Drop the returned guard to resume.
    #[cfg(test)]
    pub(crate) fn pause_release_primary(&self) -> tracing_mutex::parkinglot::MutexGuard<'_, ()> {
        self.release_primary_gate.lock()
    }

    /// Test-only: number of times `release_primary` has been entered
    /// since construction. Useful as a signal in race tests to wait
    /// for a drop thread to reach the gate without a sleep.
    #[cfg(test)]
    pub(crate) fn release_primary_arrivals(&self) -> u64 {
        self.release_primary_arrivals
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub(crate) fn metrics(&self) -> &Arc<BlockPoolMetrics> {
        &self.metrics
    }

    pub(crate) fn reset_len(&self) -> usize {
        self.inner.lock().free.len()
    }

    pub(crate) fn inactive_len(&self) -> usize {
        self.inner.lock().inactive.len()
    }

    /// Atomic snapshot of `reset_len + inactive_len` under a single store-lock
    /// acquisition. Reading the two pools separately can yield a count that
    /// never existed (e.g. a concurrent reset→inactive promotion observed
    /// twice, inflating the total above `total_blocks`).
    pub(crate) fn available_len(&self) -> usize {
        let inner = self.inner.lock();
        inner.free.len() + inner.inactive.len()
    }

    /// Return whether any supplied hash has physical registered residency.
    ///
    /// This reads the active, inactive, and pressure-held indexes under one
    /// store lock. It does not make a held hash request-available, and it does
    /// not touch the inactive index or any metric.
    pub(crate) fn has_any_registered_hashes(&self, hashes: &[SequenceHash]) -> bool {
        let inner = self.inner.lock();
        hashes.iter().any(|hash| {
            inner.active_by_hash.contains_key(hash)
                || inner.held_by_hash.contains_key(hash)
                || inner.inactive.has(*hash)
        })
    }

    pub(crate) fn has_inactive(&self, seq_hash: SequenceHash) -> bool {
        let inner = self.inner.lock();
        !inner.held_by_hash.contains_key(&seq_hash) && inner.inactive.has(seq_hash)
    }

    /// Bounded read-only snapshot of up to `max` blocks the inactive index
    /// currently ranks worst-first. Non-destructive: nothing is resurrected,
    /// touched, reordered, or re-seeded — and the order describes the present
    /// candidate set rather than a replay of the next `max` evictions, so a
    /// short result is not "the index is empty"; see
    /// [`InactiveIndex::peek_victims`]. Empty on backends with no exposed
    /// order. Backs
    /// [`BlockManager::inactive_candidates`](crate::manager::BlockManager::inactive_candidates).
    pub(crate) fn inactive_candidates(&self, max: usize) -> Vec<InactiveCandidate> {
        let inner = self.inner.lock();
        let candidates = inner.inactive.peek_victims(max);
        candidates
            .into_iter()
            .filter(|(seq_hash, _, _)| !inner.held_by_hash.contains_key(seq_hash))
            .map(|(seq_hash, block_id, features)| InactiveCandidate {
                seq_hash,
                block_id,
                generation: inner.slots[block_id].generation,
                inactive_epoch: inner.slots[block_id].inactive_epoch,
                features,
            })
            .collect()
    }

    /// Membership-based point advice for each hash in `hashes`, positionally.
    /// `None` = not resident-inactive here. One store-lock acquisition for the
    /// whole slice, so the batch is a coherent snapshot rather than N
    /// independently-timed ones. Mirrors [`Self::has_inactive`].
    pub(crate) fn inactive_advice(&self, hashes: &[SequenceHash]) -> Vec<Option<InactiveFeatures>> {
        let inner = self.inner.lock();
        hashes
            .iter()
            .map(|&h| {
                (!inner.held_by_hash.contains_key(&h))
                    .then(|| inner.inactive.advice(h))
                    .flatten()
            })
            .collect()
    }

    /// Mark the single-owner inactive lineage suffix ending at `seq_hash` for
    /// evict-first (compaction poison). Membership-based: a no-op unless the
    /// leaf is currently resident-inactive, and only the valued lineage backend
    /// acts on the marks — every other backend ignores them (see
    /// [`InactiveIndex::poison`]). The additive inverse of the
    /// [`BlockManager::poison_lineage`](crate::manager::BlockManager::poison_lineage)
    /// wrapper (EV-PR4); mirrors [`Self::has_inactive`].
    pub(crate) fn poison_lineage(&self, seq_hash: SequenceHash) {
        self.inner.lock().inactive.poison(seq_hash);
    }

    /// Test-only: whether the inactive backend has `seq_hash`'s resident node
    /// marked poisoned. Mirrors [`Self::has_inactive`]; lets `BlockManager`
    /// tests observe the compaction-poison wiring (EV-PR4).
    #[cfg(test)]
    pub(crate) fn test_is_poisoned(&self, seq_hash: SequenceHash) -> bool {
        self.inner.lock().inactive.test_is_poisoned(seq_hash)
    }

    pub(crate) fn slot_block_size(&self, block_id: BlockId) -> usize {
        self.inner.lock().slots[block_id].block_size
    }

    // ---------- guard construction ----------

    /// Enter a new mutable tenure for a reset or inactive slot.
    fn allocate_mutable_slot(&self, inner: &mut BlockStoreInner<T>, block_id: BlockId) -> usize {
        let block_size = {
            let slot = &mut inner.slots[block_id];
            debug_assert!(matches!(
                slot.state,
                SlotState::Reset | SlotState::Inactive { .. }
            ));
            assert_ne!(slot.generation, u64::MAX, "block slot generation exhausted");
            slot.generation += 1;
            slot.state = SlotState::Mutable;
            slot.block_size
        };
        inner.reset_on_release[block_id] = self.default_reset_on_release;
        block_size
    }

    fn allocate_reset_blocks_locked(
        self: &Arc<Self>,
        inner: &mut BlockStoreInner<T>,
        count: usize,
    ) -> Vec<MutableBlock<T>> {
        debug_assert!(count <= inner.free.len());
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let id = inner.free.pop_front().unwrap();
            let block_size = self.allocate_mutable_slot(inner, id);
            out.push(MutableBlock::from_store(self.clone(), id, block_size));
        }
        self.metrics.dec_reset_pool_size_by(count as i64);
        self.metrics.inc_inflight_mutable_by(count as i64);
        self.metrics.inc_allocations(count as u64);
        self.metrics.inc_allocations_from_reset(count as u64);
        out
    }

    /// Allocate up to `count` MutableBlocks from the reset pool only.
    /// Returns however many were available (no eviction).
    pub(crate) fn allocate_reset_blocks(self: &Arc<Self>, count: usize) -> Vec<MutableBlock<T>> {
        let mut inner = self.inner.lock();
        let count = std::cmp::min(count, inner.free.len());
        self.allocate_reset_blocks_locked(&mut inner, count)
    }

    /// Allocate exactly `count` mutable blocks from the reset pool.
    ///
    /// A short reset pool returns `None` before any slot changes. This never
    /// inspects or changes the inactive pool.
    pub(crate) fn allocate_reset_blocks_atomic(
        self: &Arc<Self>,
        count: usize,
    ) -> Option<Vec<MutableBlock<T>>> {
        if count == 0 {
            return Some(Vec::new());
        }

        let mut inner = self.inner.lock();
        if inner.free.len() < count {
            return None;
        }
        Some(self.allocate_reset_blocks_locked(&mut inner, count))
    }

    /// All-or-nothing allocation across the reset and inactive pools under a
    /// single store-mutex acquisition. Returns `None` iff
    /// `free.len() + inactive.len() < count`; otherwise drains `count`
    /// blocks (reset first, then inactive) and reports the evicted hashes.
    /// No partial commits, no put-backs.
    pub(crate) fn allocate_atomic(
        self: &Arc<Self>,
        count: usize,
    ) -> Option<(Vec<MutableBlock<T>>, Vec<SequenceHash>)> {
        if count == 0 {
            return Some((Vec::new(), Vec::new()));
        }
        let mut inner = self.inner.lock();
        if inner.free.len() + inner.inactive.len() < count {
            return None;
        }

        let from_reset = std::cmp::min(count, inner.free.len());
        let from_inactive = count - from_reset;

        // Stage decisions on raw `BlockId`s first; only commit slot
        // transitions (and construct MutableBlock guards) once we know
        // the inactive backend returned the requested count.
        let mut reset_ids: Vec<BlockId> = Vec::with_capacity(from_reset);
        for _ in 0..from_reset {
            reset_ids.push(inner.free.pop_front().unwrap());
        }
        let evicted_pairs = if from_inactive > 0 {
            inner.inactive.allocate(from_inactive)
        } else {
            Vec::new()
        };
        // Defensive runtime check: any backend that violates the
        // `len() >= n ⇒ allocate(n).len() == n` invariant must not
        // leave us partially committed. Roll back and return None.
        // Restore FIFO order by re-inserting the popped reset IDs at the
        // front in reverse — `pop_front` consumed `[a, b, c]`, so
        // `push_front` in reverse `[c, b, a]` re-prepends `a, b, c`.
        if evicted_pairs.len() != from_inactive {
            for (h, id) in evicted_pairs {
                inner.inactive.insert(h, id);
            }
            for id in reset_ids.into_iter().rev() {
                inner.free.push_front(id);
            }
            self.metrics.inc_allocate_atomic_rollback();
            return None;
        }

        // Commit. Past this point we cannot fail.
        let mut blocks = Vec::with_capacity(count);
        for id in reset_ids {
            let block_size = self.allocate_mutable_slot(&mut inner, id);
            blocks.push(MutableBlock::from_store(self.clone(), id, block_size));
        }
        let mut evicted = Vec::with_capacity(from_inactive);
        let mut handles = Vec::with_capacity(from_inactive);
        for (seq_hash, block_id) in evicted_pairs {
            // Eviction discards the override; the slot leaves Inactive.
            let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
            let block_size = self.allocate_mutable_slot(&mut inner, block_id);
            blocks.push(MutableBlock::from_store(self.clone(), block_id, block_size));
            evicted.push(seq_hash);
            handles.push(handle);
        }

        self.metrics.dec_reset_pool_size_by(from_reset as i64);
        self.metrics.dec_inactive_pool_size_by(from_inactive as i64);
        self.metrics.inc_inflight_mutable_by(count as i64);
        self.metrics.inc_evictions(from_inactive as u64);
        self.metrics.inc_allocations(count as u64);
        self.metrics.inc_allocations_from_reset(from_reset as u64);

        drop(inner);
        // mark_absent::<T> takes the registry attachments lock — invoke
        // outside the store lock to honour the documented ordering.
        for h in handles {
            h.mark_absent::<T>();
        }
        Some((blocks, evicted))
    }

    /// Drain the inactive pool entirely into mutable blocks and report every
    /// lineage hash that ceased to be cached.
    pub(crate) fn drain_inactive_to_mutable(
        self: &Arc<Self>,
    ) -> (Vec<MutableBlock<T>>, Vec<SequenceHash>) {
        let mut inner = self.inner.lock();
        let drained = inner.inactive.allocate_all();
        let count = drained.len();
        let mut handles = Vec::with_capacity(count);
        let mut out = Vec::with_capacity(count);
        let mut evicted = Vec::with_capacity(count);
        for (seq_hash, block_id) in drained {
            // Eviction discards the override; the slot leaves Inactive.
            let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
            let block_size = self.allocate_mutable_slot(&mut inner, block_id);
            handles.push(handle);
            out.push(MutableBlock::from_store(self.clone(), block_id, block_size));
            evicted.push(seq_hash);
        }
        self.metrics.dec_inactive_pool_size_by(count as i64);
        self.metrics.inc_inflight_mutable_by(count as i64);
        drop(inner);
        for h in handles {
            h.mark_absent::<T>();
        }
        (out, evicted)
    }

    /// Promote inactive slots to `Primary`, building fresh
    /// `ImmutableBlockInner`s. Scan-style — does not stop on first miss.
    pub(crate) fn scan_inactive_primaries(
        self: &Arc<Self>,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Arc<ImmutableBlockInner<T>>)> {
        self.promote_inactive(hashes, touch, /*scan*/ true)
    }

    /// Pin the complete inactive set without an access-frequency update.
    pub(crate) fn match_inactive_primaries(
        self: &Arc<Self>,
        hashes: &[SequenceHash],
    ) -> Vec<(SequenceHash, Arc<ImmutableBlockInner<T>>)> {
        self.promote_inactive(hashes, false, /*scan*/ false)
    }

    /// Atomic active-or-inactive lookup by sequence hash. Replaces the
    /// previous `upgrade_or_resurrect` two-lock dance.
    pub(crate) fn acquire_for_hash(
        self: &Arc<Self>,
        seq_hash: SequenceHash,
        touch: bool,
    ) -> Option<Arc<ImmutableBlockInner<T>>> {
        let mut inner = self.inner.lock();
        self.acquire_for_hash_locked(&mut inner, seq_hash, touch)
    }

    /// Locked-form of [`acquire_for_hash`]. Walks one path under the
    /// caller's lock:
    /// 1. If `active_by_hash[seq_hash]` resolves to a Primary slot whose
    ///    `Weak` upgrades, return that strong `Arc`.
    /// 2. If the `Weak` is dead (last user is mid-drop), eagerly transition
    ///    `Primary → Inactive` ourselves, then fall through to (3).
    /// 3. If the inactive index has the hash, resurrect it.
    /// 4. Else `None`.
    fn acquire_for_hash_locked(
        self: &Arc<Self>,
        inner: &mut BlockStoreInner<T>,
        seq_hash: SequenceHash,
        touch: bool,
    ) -> Option<Arc<ImmutableBlockInner<T>>> {
        if inner.held_by_hash.contains_key(&seq_hash) {
            return None;
        }
        self.acquire_registered_for_hash_locked(inner, seq_hash, touch)
    }

    /// Registration-side lookup ignores a held ownership fence only to find
    /// a newer primary or duplicate that arrived after the hold began. The
    /// public request path always calls [`Self::acquire_for_hash_locked`].
    fn acquire_for_registration_locked(
        self: &Arc<Self>,
        inner: &mut BlockStoreInner<T>,
        seq_hash: SequenceHash,
        touch: bool,
    ) -> Option<Arc<ImmutableBlockInner<T>>> {
        self.acquire_registered_for_hash_locked(inner, seq_hash, touch)
    }

    /// Active-or-inactive lookup without the held ownership fence.
    fn acquire_registered_for_hash_locked(
        self: &Arc<Self>,
        inner: &mut BlockStoreInner<T>,
        seq_hash: SequenceHash,
        touch: bool,
    ) -> Option<Arc<ImmutableBlockInner<T>>> {
        // (1) Active path.
        if let Some(&block_id) = inner.active_by_hash.get(&seq_hash) {
            let live: Option<Arc<ImmutableBlockInner<T>>> = match &inner.slots[block_id].state {
                SlotState::Primary { inner: weak, .. } => weak.upgrade(),
                other => panic!("active_by_hash[{seq_hash:?}] = {block_id} but slot is {other:?}"),
            };
            if let Some(arc) = live {
                return Some(arc);
            }
            // (2) Eager Primary → Inactive transition. The original
            // Inner::drop will see slot != Primary and no-op.
            self.eager_primary_to_inactive_locked(inner, seq_hash, block_id);
            // Fall through to inactive path.
        }

        // (3) Inactive path. Single-hash fast-path through the
        // backend-specific `find_match` override (O(1) for hashmap/lru
        // backends) instead of allocating a one-element slice + Vec.
        let block_id = inner.inactive.find_match(seq_hash, touch)?.1;
        self.metrics.dec_inactive_pool_size();
        let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
        // Resurrection: the per-slot `reset_on_release` atomic carries
        // the previous holder's override across this transition
        // untouched, so the new `ImmutableBlockInner` will read the
        // right value on its own drop.
        let inner_arc =
            ImmutableBlockInner::new_primary(self.clone(), block_id, seq_hash, handle.clone());
        inner.slots[block_id].state = SlotState::Primary {
            seq_hash,
            handle,
            inner: Arc::downgrade(&inner_arc),
        };
        inner.active_by_hash.insert(seq_hash, block_id);
        Some(inner_arc)
    }

    /// Batched active-or-inactive prefix lookup under **one** store-mutex
    /// acquisition. Walks `hashes` left-to-right, stopping at the first
    /// hash that hits neither pool.
    ///
    /// Per hash this is exactly [`acquire_for_hash_locked`] — active hit /
    /// eager `Primary → Inactive` on a dead `Weak` / inactive resurrection —
    /// so the `self_ptr` race handling and eager-transition semantics are
    /// unchanged. This is literally the per-hash [`acquire_for_hash`] body
    /// hoisted above a single `lock()`, replacing the old N-acquisitions
    /// per-hash loop in `BlockManager::match_blocks`.
    ///
    /// Passes `touch = false`: the frequency tracker is **not** touched
    /// here. The caller is responsible for touching the returned hashes
    /// *after* this returns (store lock released) — see
    /// `BlockManager::match_blocks`. Keeping the explicit touch outside the
    /// store lock avoids widening the store-lock hold across the TinyLFU
    /// mutex. (The eager `Primary → Inactive` branch still nests
    /// store → frequency-tracker via `inactive.insert`, exactly as it does
    /// on the pre-existing per-call path — that ordering is consistent and
    /// not affected by this batching.)
    pub(crate) fn match_prefix_locked_batch(
        self: &Arc<Self>,
        hashes: &[SequenceHash],
    ) -> Vec<Arc<ImmutableBlockInner<T>>> {
        let mut inner = self.inner.lock();
        let mut out = Vec::with_capacity(hashes.len());
        for &h in hashes {
            match self.acquire_for_hash_locked(&mut inner, h, /*touch*/ false) {
                Some(arc) => out.push(arc),
                None => break,
            }
        }
        out
    }

    /// Atomic registration of a [`CompleteBlock`]: lookup-then-transition
    /// under one store-mutex acquisition. Closes the register-vs-register
    /// race for the same sequence hash.
    ///
    /// On `BlockDuplicationPolicy::Allow` returns a duplicate-backed
    /// `Arc<ImmutableBlockInner<T>>`; on `Reject` returns the existing
    /// primary's `Arc` and lets the supplied `block` guard release its
    /// slot back to the reset pool.
    pub(crate) fn register_completed_block(
        self: &Arc<Self>,
        block: CompleteBlock<T>,
        handle: BlockRegistrationHandle,
        policy: BlockDuplicationPolicy,
    ) -> Arc<ImmutableBlockInner<T>> {
        let block_id = block.block_id();
        let seq_hash = block.sequence_hash();
        debug_assert_eq!(seq_hash, handle.seq_hash());

        // Disarm the guard up front so the slot stays in `Staged` state
        // when we transition; we re-arm only on the Reject path.
        let mut block = block;
        block.disarm();

        let mut inner = self.inner.lock();
        let existing = self.acquire_for_registration_locked(&mut inner, seq_hash, false);

        // Whether we added a new presence-bearing slot (Primary or
        // Duplicate). Reject does not, since the slot returns to Reset.
        let mut presence_added = false;

        let result = if let Some(existing_primary) = existing {
            assert_ne!(
                existing_primary.block_id(),
                block_id,
                "register_completed_block: collision with same block_id {block_id}"
            );
            match policy {
                BlockDuplicationPolicy::Allow => {
                    debug_assert!(matches!(
                        inner.slots[block_id].state,
                        SlotState::Staged { .. }
                    ));
                    let inner_arc = ImmutableBlockInner::new_duplicate(
                        self.clone(),
                        block_id,
                        seq_hash,
                        handle.clone(),
                        existing_primary,
                    );
                    inner.slots[block_id].state = SlotState::Duplicate {
                        seq_hash,
                        handle: handle.clone(),
                        inner: Arc::downgrade(&inner_arc),
                    };
                    self.metrics.inc_duplicate_blocks();
                    presence_added = true;
                    inner_arc
                }
                BlockDuplicationPolicy::Reject => {
                    self.metrics.inc_registration_dedup();
                    // Re-arm so the block guard's drop releases the slot
                    // (Staged → Reset) when it falls out of scope below.
                    block.rearm();
                    existing_primary
                }
            }
        } else {
            // Fresh primary.
            debug_assert!(matches!(
                inner.slots[block_id].state,
                SlotState::Staged { .. }
            ));
            let inner_arc =
                ImmutableBlockInner::new_primary(self.clone(), block_id, seq_hash, handle.clone());
            inner.slots[block_id].state = SlotState::Primary {
                seq_hash,
                handle: handle.clone(),
                inner: Arc::downgrade(&inner_arc),
            };
            inner.active_by_hash.insert(seq_hash, block_id);
            presence_added = true;
            inner_arc
        };

        drop(inner);

        // mark_present takes the attachments lock; lock-order
        // (attachments → store) is satisfied because the store lock has
        // already been released. Skip on Reject — no new presence-bearing
        // slot was created.
        if presence_added {
            handle.mark_present::<T>();
        }

        // Block guard drops here: armed=false on Allow/fresh paths
        // (slot already transitioned), armed=true on Reject (releases
        // Staged → Reset).
        drop(block);
        result
    }

    /// Batched registration of completed blocks under **one** store-mutex
    /// acquisition — the register-side analogue of
    /// [`match_prefix_locked_batch`](Self::match_prefix_locked_batch)
    /// (batched lookup) and [`allocate_atomic`](Self::allocate_atomic)
    /// (batched commit). Mirrors [`register_completed_block`](Self::register_completed_block)
    /// per item; added *alongside* it, not as a replacement — existing
    /// single-block callers are unaffected.
    ///
    /// `blocks` and `handles` must be the same length and index-aligned:
    /// `handles[i]` backs `blocks[i]`. `policy` applies uniformly to the
    /// whole batch, matching `BlockManager`'s single store-wide
    /// `duplication_policy`.
    ///
    /// Because every lookup-then-transition in the batch runs under the
    /// same lock, this also closes the register-vs-register race *within*
    /// the batch itself: if two entries share a sequence hash, the second
    /// sees the first's `active_by_hash` update and is registered as a
    /// duplicate (or rejected), exactly as if the two had been registered
    /// one at a time.
    pub(crate) fn register_blocks(
        self: &Arc<Self>,
        mut blocks: Vec<CompleteBlock<T>>,
        handles: Vec<BlockRegistrationHandle>,
        policy: BlockDuplicationPolicy,
    ) -> Vec<Arc<ImmutableBlockInner<T>>> {
        assert_eq!(
            blocks.len(),
            handles.len(),
            "register_blocks: blocks/handles length mismatch"
        );
        if blocks.is_empty() {
            return Vec::new();
        }

        // Validate every block/handle pair *before* mutating any guard
        // state — matches the singular `register_block`'s
        // (registration.rs) validate-before-mutate ordering. A real,
        // always-on assertion (not `debug_assert!`): it must run in
        // release builds too, and it must run before any guard is
        // disarmed below, so a mismatch anywhere in the batch leaves
        // every `CompleteBlock` untouched (still armed) to unwind and
        // release itself normally on panic — no half-applied batch, no
        // `Staged` slots stranded mid-registration.
        for (block, handle) in blocks.iter().zip(handles.iter()) {
            assert_eq!(
                block.sequence_hash(),
                handle.seq_hash(),
                "register_blocks: attempted to register block {} with a different sequence hash than its handle",
                block.block_id(),
            );
        }

        // Disarm every guard up front so its slot stays `Staged` across
        // the transition; `rearm[i]` re-arms the Reject-dedup entries so
        // their guard drop still releases `Staged → Reset` once the lock
        // is gone — mirrors the single-item disarm/rearm dance above.
        for block in &mut blocks {
            block.disarm();
        }
        let mut rearm = vec![false; blocks.len()];
        let mut results = Vec::with_capacity(blocks.len());
        let mut present_handles = Vec::with_capacity(blocks.len());

        {
            let mut inner = self.inner.lock();
            for (i, (block, handle)) in blocks.iter().zip(handles.iter()).enumerate() {
                let block_id = block.block_id();
                let seq_hash = block.sequence_hash();

                let existing = self.acquire_for_registration_locked(&mut inner, seq_hash, false);

                let inner_arc = if let Some(existing_primary) = existing {
                    assert_ne!(
                        existing_primary.block_id(),
                        block_id,
                        "register_blocks: collision with same block_id {block_id}"
                    );
                    match policy {
                        BlockDuplicationPolicy::Allow => {
                            debug_assert!(matches!(
                                inner.slots[block_id].state,
                                SlotState::Staged { .. }
                            ));
                            let inner_arc = ImmutableBlockInner::new_duplicate(
                                self.clone(),
                                block_id,
                                seq_hash,
                                handle.clone(),
                                existing_primary,
                            );
                            inner.slots[block_id].state = SlotState::Duplicate {
                                seq_hash,
                                handle: handle.clone(),
                                inner: Arc::downgrade(&inner_arc),
                            };
                            self.metrics.inc_duplicate_blocks();
                            present_handles.push(handle.clone());
                            inner_arc
                        }
                        BlockDuplicationPolicy::Reject => {
                            self.metrics.inc_registration_dedup();
                            rearm[i] = true;
                            existing_primary
                        }
                    }
                } else {
                    debug_assert!(matches!(
                        inner.slots[block_id].state,
                        SlotState::Staged { .. }
                    ));
                    let inner_arc = ImmutableBlockInner::new_primary(
                        self.clone(),
                        block_id,
                        seq_hash,
                        handle.clone(),
                    );
                    inner.slots[block_id].state = SlotState::Primary {
                        seq_hash,
                        handle: handle.clone(),
                        inner: Arc::downgrade(&inner_arc),
                    };
                    inner.active_by_hash.insert(seq_hash, block_id);
                    present_handles.push(handle.clone());
                    inner_arc
                };
                results.push(inner_arc);
            }
        } // store lock released here

        // mark_present takes the attachments lock; lock-order
        // (attachments → store) is satisfied because the store lock has
        // already been released.
        for h in present_handles {
            h.mark_present::<T>();
        }

        // Guard drops here, same semantics as the single-item path:
        // armed=false on Allow/fresh (slot already transitioned),
        // armed=true on Reject (releases Staged → Reset).
        for (i, mut block) in blocks.into_iter().enumerate() {
            if rearm[i] {
                block.rearm();
            }
            drop(block);
        }

        results
    }

    /// Transition a slot into a new inactive residency tenure.
    ///
    /// The caller holds the store lock and must clear any active-map entry
    /// that names this slot. A hold abort restores a paused tenure and does
    /// not call this helper.
    fn enter_inactive_tenure_locked(
        &self,
        inner: &mut BlockStoreInner<T>,
        block_id: BlockId,
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    ) {
        let slot = &mut inner.slots[block_id];
        assert_ne!(
            slot.inactive_epoch,
            u64::MAX,
            "block slot inactive epoch exhausted"
        );
        slot.inactive_epoch += 1;
        slot.state = SlotState::Inactive { seq_hash, handle };
        inner.inactive.insert(seq_hash, block_id);
    }

    /// Internal helper: under the store lock, transition a Primary slot to
    /// Inactive without touching presence (the original Inner::drop's
    /// presence-side responsibilities are unchanged — it just no-ops the
    /// slot transition since we did it).
    fn eager_primary_to_inactive_locked(
        &self,
        inner: &mut BlockStoreInner<T>,
        seq_hash: SequenceHash,
        block_id: BlockId,
    ) {
        let handle = match &inner.slots[block_id].state {
            SlotState::Primary { handle, .. } => handle.clone(),
            other => panic!("eager_primary_to_inactive: slot {block_id} was {other:?}"),
        };
        // The per-block `reset_on_release` override lives in
        // `inner.reset_on_release[block_id]`, not in the dropping
        // `ImmutableBlockInner`. We leave it untouched here so the value
        // the holder set via `set_evict_on_reset` rides through this
        // race-window transition. Visibility: both `set_evict_on_reset`
        // and the eventual `release_primary` read go through this same
        // store mutex, so the value is published reliably regardless of
        // which thread wins the race for the lock.
        self.enter_inactive_tenure_locked(inner, block_id, seq_hash, handle);
        inner.active_by_hash.remove(&seq_hash);
        self.metrics.inc_inactive_pool_size();
        self.metrics.inc_eager_primary_to_inactive();
        tracing::trace!(
            ?seq_hash,
            block_id,
            "Eager Primary → Inactive (lookup-driven)"
        );
    }

    /// Common slot-transition core for find/scan inactive promotions.
    /// Unlike the previous two-step version, this builds the
    /// `ImmutableBlockInner` and writes its `Weak` into the slot under the
    /// same lock acquisition.
    fn promote_inactive(
        self: &Arc<Self>,
        hashes: &[SequenceHash],
        touch: bool,
        scan: bool,
    ) -> Vec<(SequenceHash, Arc<ImmutableBlockInner<T>>)> {
        let mut inner = self.inner.lock();
        if !scan {
            // Backend matching removes entries. A failed retention probe
            // must leave their order and inactive tenure intact.
            if hashes
                .iter()
                .any(|hash| inner.held_by_hash.contains_key(hash) || !inner.inactive.has(*hash))
            {
                return Vec::new();
            }
            if hashes.len() > 1 {
                let mut unique = SeqHashMap::default();
                if hashes.iter().any(|hash| unique.insert(*hash, ()).is_some()) {
                    return Vec::new();
                }
            }
        }
        let matched: Vec<(SequenceHash, BlockId)> = if scan {
            let visible_hashes = hashes
                .iter()
                .copied()
                .filter(|hash| !inner.held_by_hash.contains_key(hash))
                .collect::<Vec<_>>();
            inner.inactive.scan_matches(&visible_hashes, touch)
        } else {
            // First-hash fast-path: probe the head via the
            // backend-specific `find_match` override before allocating
            // the result Vec. Empty input or a head miss exits without
            // any allocation.
            let Some((&first_hash, rest)) = hashes.split_first() else {
                return Vec::new();
            };
            if inner.held_by_hash.contains_key(&first_hash) {
                return Vec::new();
            }
            let Some(first_pair) = inner.inactive.find_match(first_hash, touch) else {
                return Vec::new();
            };
            let mut matched = Vec::with_capacity(hashes.len());
            matched.push(first_pair);
            if !rest.is_empty() {
                let visible_rest = rest
                    .iter()
                    .copied()
                    .take_while(|hash| !inner.held_by_hash.contains_key(hash))
                    .collect::<Vec<_>>();
                matched.extend(inner.inactive.find_matches(&visible_rest, touch));
            }
            matched
        };
        self.metrics.dec_inactive_pool_size_by(matched.len() as i64);
        matched
            .into_iter()
            .map(|(seq_hash, block_id)| {
                let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
                // Resurrection: the per-slot `reset_on_release` atomic
                // carries the previous holder's override untouched.
                let inner_arc = ImmutableBlockInner::new_primary(
                    self.clone(),
                    block_id,
                    seq_hash,
                    handle.clone(),
                );
                inner.slots[block_id].state = SlotState::Primary {
                    seq_hash,
                    handle,
                    inner: Arc::downgrade(&inner_arc),
                };
                inner.active_by_hash.insert(seq_hash, block_id);
                (seq_hash, inner_arc)
            })
            .collect()
    }

    // ---------- guard transitions (called from guard methods / drops) ----------

    /// `Mutable` → `Reset` (MutableBlock dropped without a transition).
    pub(crate) fn release_mutable(&self, block_id: BlockId) {
        let mut inner = self.inner.lock();
        debug_assert!(matches!(inner.slots[block_id].state, SlotState::Mutable));
        inner.slots[block_id].state = SlotState::Reset;
        inner.free.push_back(block_id);
        self.metrics.inc_reset_pool_size();
        self.metrics.dec_inflight_mutable();
    }

    /// `Mutable` → `Staged` (MutableBlock::stage / ::complete).
    pub(crate) fn transition_to_staged(&self, block_id: BlockId, seq_hash: SequenceHash) {
        let mut inner = self.inner.lock();
        debug_assert!(matches!(inner.slots[block_id].state, SlotState::Mutable));
        inner.slots[block_id].state = SlotState::Staged { seq_hash };
        self.metrics.dec_inflight_mutable();
        self.metrics.inc_stagings();
    }

    /// `Staged` → `Mutable` (CompleteBlock::reset).
    pub(crate) fn transition_back_to_mutable(&self, block_id: BlockId) {
        let mut inner = self.inner.lock();
        debug_assert!(matches!(
            inner.slots[block_id].state,
            SlotState::Staged { .. }
        ));
        inner.slots[block_id].state = SlotState::Mutable;
        // Defensive: the Staged → Mutable rollback opens this slot to a
        // fresh tenant. Clear any leftover per-slot override.
        inner.reset_on_release[block_id] = self.default_reset_on_release;
        self.metrics.inc_inflight_mutable();
    }

    /// `Staged` → `Reset` (CompleteBlock dropped without a transition).
    pub(crate) fn release_staged(&self, block_id: BlockId) {
        let mut inner = self.inner.lock();
        debug_assert!(matches!(
            inner.slots[block_id].state,
            SlotState::Staged { .. }
        ));
        inner.slots[block_id].state = SlotState::Reset;
        inner.free.push_back(block_id);
        self.metrics.inc_reset_pool_size();
    }

    /// Drop transition for the last clone of a primary `ImmutableBlockInner`.
    ///
    /// Identity-checked against `self_ptr`. If a concurrent
    /// `acquire_for_hash` already eagerly transitioned the slot (or the
    /// slot has since been resurrected to a different Inner), this is a
    /// no-op.
    ///
    /// Reads `reset_on_release[block_id]` to select the destination:
    /// - `false` (default) → `SlotState::Inactive` + insert into the
    ///   inactive index, available for cache hits and cold eviction.
    /// - `true` → `SlotState::Reset` + push to free list +
    ///   `handle.mark_absent::<T>()`. Mirrors `release_duplicate`. The
    ///   block is *not* cached and cannot be matched/resurrected later.
    pub(crate) fn release_primary(&self, block_id: BlockId, self_ptr: *const ()) {
        // Test-only deterministic race-window widening:
        //   1. Bump the arrival counter so a coordinating test can
        //      observe "the drop has entered release_primary" without
        //      a sleep.
        //   2. Acquire the gate. While a test holds it, this call
        //      parks here *before* the store mutex is touched, so the
        //      slot remains in `Primary { weak: dead }` and a
        //      concurrent lookup can drive the eager-transition path.
        #[cfg(test)]
        self.release_primary_arrivals
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        #[cfg(test)]
        let _gate = self.release_primary_gate.lock();
        let handle_to_mark_absent = {
            let mut inner = self.inner.lock();
            let (seq_hash, handle) = match &inner.slots[block_id].state {
                SlotState::Primary {
                    seq_hash,
                    handle,
                    inner: weak,
                } if weak.as_ptr() as *const () == self_ptr => (*seq_hash, handle.clone()),
                // Eager lookup-driven transition already ran, OR this slot has
                // since been resurrected to a different Inner. No-op.
                _ => {
                    self.metrics.inc_release_primary_noop();
                    return;
                }
            };
            // Read the per-slot override. Both the writer
            // (`set_evict_on_reset`) and this read go through the store
            // mutex, so visibility comes from the mutex's
            // release-acquire — no atomic-ordering assumptions about
            // `Arc::drop` or `Weak::upgrade`.
            let reset_on_release = inner.reset_on_release[block_id];
            if reset_on_release {
                self.reset_slot_locked(&mut inner, block_id);
                // Only the primary owns the `active_by_hash` mapping;
                // duplicates have a different `block_id` under the same
                // hash and must never clear it.
                inner.active_by_hash.remove(&seq_hash);
                tracing::trace!(?seq_hash, block_id, "Primary released to reset pool");
                Some(handle)
            } else {
                // The atomic carries the holder's override into the
                // Inactive period untouched; a future resurrection will
                // inherit it via the same atomic.
                self.enter_inactive_tenure_locked(&mut inner, block_id, seq_hash, handle);
                inner.active_by_hash.remove(&seq_hash);
                self.metrics.inc_inactive_pool_size();
                tracing::trace!(?seq_hash, block_id, "Block stored in inactive pool");
                None
            }
        };
        // mark_absent takes the attachments lock; lock-order
        // (attachments → store) is satisfied because the store lock has
        // already been released. Matches the `release_duplicate` pattern.
        if let Some(handle) = handle_to_mark_absent {
            handle.mark_absent::<T>();
        }
    }

    /// Drop transition for the last clone of a duplicate `ImmutableBlockInner`:
    /// `Duplicate` → `Reset` (with `mark_absent::<T>`). Identity-checked.
    pub(crate) fn release_duplicate(&self, block_id: BlockId, self_ptr: *const ()) {
        let handle = {
            let mut inner = self.inner.lock();
            let handle = match &inner.slots[block_id].state {
                SlotState::Duplicate {
                    handle,
                    inner: weak,
                    ..
                } if weak.as_ptr() as *const () == self_ptr => handle.clone(),
                // Slot has moved on (this should not normally happen for
                // duplicates since they cannot be resurrected, but guard
                // defensively).
                _ => {
                    self.metrics.inc_release_duplicate_noop();
                    return;
                }
            };
            // Duplicates do NOT clear `active_by_hash` — that mapping
            // belongs to the primary, which has a different `block_id`
            // and is kept alive by `_primary_keepalive` until this drop.
            self.reset_slot_locked(&mut inner, block_id);
            handle
        };
        handle.mark_absent::<T>();
    }

    /// Batched release of a set of [`ImmutableBlock`] guards under **one**
    /// store-mutex acquisition — the teardown-side analogue of
    /// [`allocate_atomic`](Self::allocate_atomic): a request that today
    /// drops N `ImmutableBlock`s one at a time (each independently taking
    /// the store mutex via `ImmutableBlockInner::drop`) instead takes the
    /// lock once for the whole batch.
    ///
    /// # Equivalence contract
    ///
    /// This must leave the store in a state indistinguishable from
    /// dropping the same `blocks`, in the same order, one at a time via
    /// the pre-existing per-block path — **including** the reset (free)
    /// pool's FIFO order, which is observable: `allocate_reset_blocks`
    /// pops the front, so a different push order hands out a different
    /// block on the next allocation.
    ///
    /// The tricky case is a batch containing both a primary and its live
    /// duplicate. One-at-a-time, dropping the duplicate's guard is what
    /// (via ordinary `Drop` field-glue on `_primary_keepalive`) frees the
    /// primary — so the primary's *actual* release happens at whichever
    /// of {the primary's own guard, every co-batched duplicate targeting
    /// it} is **last** in drop order, not at the primary's own position.
    /// [`release_entry_at`](Self::release_entry_at) reproduces this
    /// exactly by explicitly, synchronously replaying the keepalive
    /// cascade at the right relative position instead of leaving it to
    /// `Drop`'s arbitrary timing (which is what an earlier version of
    /// this function did, and which a proptest caught reordering the
    /// free list relative to one-at-a-time drops).
    ///
    /// # Algorithm
    ///
    /// Phase 0 (no lock): extract every guard's backing
    /// `Arc<ImmutableBlockInner<T>>` (bypassing this guard's own `Drop`,
    /// whose only effect — the `inflight_immutable` metric decrement —
    /// `into_inner_for_batch_release` replicates), and capture each
    /// one's identity pointer (`Arc::as_ptr`) while still alive. Index
    /// every pointer by position (`target_position`) so a duplicate's
    /// `_primary_keepalive` can be resolved back to a co-batched primary
    /// entry, if any.
    ///
    /// Phase 1 (single lock): call
    /// [`release_entry_at`](Self::release_entry_at) for every position,
    /// in input order. See its docs for the identity-check +
    /// try-unwrap-or-defer + cascade-replay logic.
    ///
    /// Anything left holding an `Arc` after Phase 1 (a slot that already
    /// moved on before we reached the lock, or a guard that is genuinely
    /// still `Arc`-shared — another clone, or an *external*, not
    /// co-batched, duplicate/primary relationship) is swept into
    /// `deferred` and counted in `report.deferred_to_drop`.
    ///
    /// Phase 2/3 (post-lock): drop `deferred` and the released-and-defused
    /// `to_drop` values, then run `mark_absent` — preserving the
    /// documented `attachments → inner` ordering by never holding both
    /// locks at once. See `release_entry_at` for why some drops must
    /// wait until here rather than happening inline.
    pub(crate) fn release_blocks(
        &self,
        blocks: Vec<ImmutableBlock<T>>,
        opts: ReleaseOpts,
    ) -> ReleaseReport {
        let mut report = ReleaseReport::default();
        if blocks.is_empty() {
            return report;
        }

        // Phase 0 (no lock): extract each guard's backing Arc and its
        // identity pointer while the Arc is still alive.
        let mut entries: Vec<ReleaseEntry<T>> = Vec::with_capacity(blocks.len());
        for block in blocks {
            let block_id = block.block_id();
            let arc = block.into_inner_for_batch_release();
            let self_ptr = Arc::as_ptr(&arc) as *const ();
            entries.push(ReleaseEntry {
                block_id,
                self_ptr,
                arc: Some(arc),
            });
        }
        let target_position: std::collections::HashMap<*const (), usize> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.self_ptr, i))
            .collect();

        // Phase 1 (single lock): process every entry, in input order.
        let mut to_drop: Vec<ImmutableBlockInner<T>> = Vec::with_capacity(entries.len());
        let mut deferred: Vec<Arc<ImmutableBlockInner<T>>> = Vec::new();
        let mut pending_mark_absent: Vec<BlockRegistrationHandle> =
            Vec::with_capacity(entries.len());
        {
            let mut inner = self.inner.lock();
            for idx in 0..entries.len() {
                self.release_entry_at(
                    idx,
                    &mut entries,
                    &target_position,
                    &mut inner,
                    opts,
                    &mut report,
                    &mut to_drop,
                    &mut deferred,
                    &mut pending_mark_absent,
                );
            }

            // Anything still unresolved is genuinely externally shared
            // (or, defensively, a slot that had already moved on before
            // we reached the lock) — left to its ordinary `Drop`.
            for entry in &mut entries {
                if let Some(arc) = entry.arc.take() {
                    report.deferred_to_drop += 1;
                    deferred.push(arc);
                }
            }
        } // store lock released here

        // Phase 2: drop everything deferred or released-and-defused, now
        // that the lock is free. Safe even if a cascading Drop (e.g. an
        // externally-shared duplicate's `_primary_keepalive` hitting
        // zero once dropped here) needs the store lock itself — that is
        // exactly why these particular drops were not performed inline;
        // see `release_entry_at`.
        drop(deferred);
        drop(to_drop);

        // Phase 3: attachments-lock work, strictly after the store lock —
        // preserves the documented attachments → inner ordering by never
        // letting the two overlap.
        for handle in pending_mark_absent {
            handle.mark_absent::<T>();
        }

        report
    }

    /// Resolve and, if possible, release a single [`ReleaseEntry`] in
    /// [`release_blocks`](Self::release_blocks)'s batch, replaying the
    /// exact per-block identity-check + `try_unwrap`-or-defer logic
    /// (`release_primary`/`release_duplicate`), plus one addition: when
    /// a released *duplicate*'s `_primary_keepalive` targets another
    /// entry in the **same batch**, the keepalive is released
    /// synchronously, right here, instead of being left for that
    /// `Inner`'s ordinary field-drop glue to discover later (which is
    /// what made an earlier version of this function reorder the reset
    /// pool relative to one-at-a-time drops — see `release_blocks`'s
    /// docs).
    ///
    /// A no-op if `entries[idx].arc` is already `None` (already resolved
    /// — released, or merged into and resolved by a cascade from another
    /// entry).
    ///
    /// # Why the cascade must be handled explicitly, not via `Drop`
    ///
    /// If the duplicate's `Arc::try_unwrap` succeeds, we hold the
    /// duplicate's `Inner` by value. Its `_primary_keepalive` field, if
    /// dropped as ordinary field-drop glue, may make the primary's
    /// refcount hit zero and invoke the primary's **own**, non-defused
    /// `Drop` — which calls `release_primary`, reacquiring
    /// `self.inner`'s mutex. If that happened while this function still
    /// held the lock (as it must, to place the release at the correct
    /// relative position — see below), it would deadlock. So the
    /// keepalive is taken out via `take_primary_keepalive` (a safe field
    /// swap, not a drop) and handled explicitly:
    ///
    /// - If it points to a **co-batched** entry at position `j`:
    ///   - `j < idx` (already reached by the outer loop, so its own
    ///     `Arc` is either still held — untried or previously deferred
    ///     — or, defensively, already resolved by something else):
    ///     merge the two references (drop the redundant extraction —
    ///     provably safe, since the entry's own stored `Arc`, if
    ///     present, guarantees at least one reference survives that
    ///     drop) and retry position `j` **right now**. A successful
    ///     retry lands the release at position `idx` — matching
    ///     one-at-a-time order, where the cascade fires at the *later*
    ///     of the two positions.
    ///   - `j > idx` (not yet reached by the outer loop, so its `Arc` is
    ///     provably still untouched and therefore still holds a live
    ///     reference): drop the redundant extraction (provably not the
    ///     last reference) and do nothing further — the outer loop's
    ///     future visit to `j` will see the reduced count and correctly
    ///     release there, again matching one-at-a-time order.
    /// - If it points to an entry **not** in this batch, or (defensively)
    ///   to an already-`None` co-batched slot: we cannot prove this
    ///   drop isn't the last reference, so it is **not** dropped here —
    ///   it goes to `deferred` for the post-lock sweep, exactly like a
    ///   genuinely `Arc`-shared entry.
    #[allow(clippy::too_many_arguments)]
    fn release_entry_at(
        &self,
        idx: usize,
        entries: &mut [ReleaseEntry<T>],
        target_position: &std::collections::HashMap<*const (), usize>,
        inner: &mut BlockStoreInner<T>,
        opts: ReleaseOpts,
        report: &mut ReleaseReport,
        to_drop: &mut Vec<ImmutableBlockInner<T>>,
        deferred: &mut Vec<Arc<ImmutableBlockInner<T>>>,
        pending_mark_absent: &mut Vec<BlockRegistrationHandle>,
    ) {
        let Some(arc) = entries[idx].arc.take() else {
            return; // Already resolved.
        };
        let block_id = entries[idx].block_id;
        let self_ptr = entries[idx].self_ptr;

        // Applies to every entry we reach, matched or not — replaces a
        // separate pre-drop `set_evict_on_reset` traversal.
        if let Some(v) = opts.reset_on_release {
            inner.reset_on_release[block_id] = v;
        }

        let matched = match &inner.slots[block_id].state {
            SlotState::Primary {
                seq_hash,
                handle,
                inner: weak,
            } if weak.as_ptr() as *const () == self_ptr => {
                Some(SlotIdentityMatch::Primary(*seq_hash, handle.clone()))
            }
            SlotState::Duplicate {
                handle,
                inner: weak,
                ..
            } if weak.as_ptr() as *const () == self_ptr => {
                Some(SlotIdentityMatch::Duplicate(handle.clone()))
            }
            _ => None,
        };

        let Some(matched) = matched else {
            entries[idx].arc = Some(arc); // Put back for the final sweep.
            return;
        };

        let mut owned = match Arc::try_unwrap(arc) {
            Ok(owned) => owned,
            Err(arc) => {
                entries[idx].arc = Some(arc); // Still shared — put back;
                // may yet be resolved by a later cascade, else swept.
                return;
            }
        };

        match matched {
            SlotIdentityMatch::Primary(seq_hash, handle) => {
                if inner.reset_on_release[block_id] {
                    self.reset_slot_locked(inner, block_id);
                    // Only the primary owns the `active_by_hash` mapping.
                    inner.active_by_hash.remove(&seq_hash);
                    pending_mark_absent.push(handle);
                    report.primary_reset += 1;
                } else {
                    self.enter_inactive_tenure_locked(inner, block_id, seq_hash, handle);
                    inner.active_by_hash.remove(&seq_hash);
                    self.metrics.inc_inactive_pool_size();
                    report.primary_inactive += 1;
                }
            }
            SlotIdentityMatch::Duplicate(handle) => {
                // Duplicates do NOT clear `active_by_hash` — that
                // mapping belongs to the primary.
                self.reset_slot_locked(inner, block_id);
                pending_mark_absent.push(handle);
                report.duplicate_reset += 1;

                // Explicitly replay the keepalive cascade — see the
                // design note above for why this can't be left to
                // ordinary field-drop glue.
                if let Some(primary_arc) = owned.take_primary_keepalive() {
                    let target_ptr = Arc::as_ptr(&primary_arc) as *const ();
                    match target_position.get(&target_ptr) {
                        Some(&j) if j < idx => {
                            if let Some(existing) = entries[j].arc.take() {
                                // Two live references to the same Inner
                                // (`existing` + `primary_arc`); `existing`
                                // guarantees at least one survives, so
                                // dropping the redundant one can't be the
                                // last reference — safe under the lock.
                                entries[j].arc = Some(existing);
                                drop(primary_arc);
                                self.release_entry_at(
                                    j,
                                    entries,
                                    target_position,
                                    inner,
                                    opts,
                                    report,
                                    to_drop,
                                    deferred,
                                    pending_mark_absent,
                                );
                            } else {
                                // Defensive: already resolved by
                                // something else. Can't prove this
                                // isn't the last reference — must not
                                // drop while holding the lock.
                                deferred.push(primary_arc);
                            }
                        }
                        Some(&j) if j > idx => {
                            // entries[j].arc is provably still untouched
                            // (the outer loop hasn't reached it, and
                            // nothing before `idx` could have — cascades
                            // only ever target strictly earlier
                            // positions), so it still holds a live
                            // reference: dropping this redundant one
                            // can't be the last.
                            debug_assert!(entries[j].arc.is_some());
                            drop(primary_arc);
                        }
                        _ => {
                            // External to this batch (or, defensively,
                            // `j == idx`, which cannot happen — a
                            // duplicate never keeps itself alive).
                            // Cannot prove this isn't the last
                            // reference; must not risk dropping (and
                            // possibly reacquiring the store lock) while
                            // we still hold it.
                            deferred.push(primary_arc);
                        }
                    }
                }
            }
        }

        owned.defuse();
        to_drop.push(owned);
    }

    /// Slot transition shared by `release_primary` (when
    /// `reset_on_release = true`) and `release_duplicate`:
    /// `*` → `SlotState::Reset`, push to the free list, bump the
    /// reset-pool gauge. Does **not** touch `active_by_hash` — callers
    /// that own that mapping (the primary release path) must clear it
    /// themselves. Callers must invoke `handle.mark_absent::<T>()`
    /// *after* the store lock is released.
    fn reset_slot_locked(&self, inner: &mut BlockStoreInner<T>, block_id: BlockId) {
        inner.slots[block_id].state = SlotState::Reset;
        inner.free.push_back(block_id);
        self.metrics.inc_reset_pool_size();
    }
}

/// Per-slot summary for [`BlockStore::debug_snapshot`] — everything
/// [`SlotState`] carries except the `Weak`/`BlockRegistrationHandle`
/// payloads, which aren't meaningfully comparable across two independent
/// stores.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SlotKind {
    Reset,
    Mutable,
    Staged(SequenceHash),
    Primary(SequenceHash),
    Duplicate(SequenceHash),
    Inactive(SequenceHash),
    Held(SequenceHash),
}

/// Full test-only snapshot of a [`BlockStore`]'s bookkeeping, for
/// asserting two independently-operated stores end up byte-for-byte
/// equivalent (see the `release_blocks` state-equivalence proptest).
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DebugStoreSnapshot {
    /// Every slot's kind, in `block_id` order.
    pub(crate) slots: Vec<SlotKind>,
    /// The reset (free) pool, in exact FIFO order. Order-sensitive on
    /// purpose: `allocate_reset_blocks`/`allocate_atomic` pop the front,
    /// so a different push order hands a different physical block to
    /// the next allocation — that's observable, not just an internal
    /// bookkeeping detail. `release_blocks` must reproduce the exact
    /// push order that dropping the same guards one at a time would
    /// produce, including when a batch contains a primary and its live
    /// duplicate (see `release_blocks`'s and `release_entry_at`'s design
    /// notes for how the keepalive cascade is replayed at the correct
    /// relative position instead of being deferred to `Drop`'s arbitrary
    /// timing).
    pub(crate) free: Vec<BlockId>,
    /// The full `active_by_hash` map (primary `block_id` per registered
    /// hash). `HashMap` equality is set-like (order-independent), which
    /// is the right comparison for a map.
    pub(crate) active_by_hash: std::collections::HashMap<SequenceHash, BlockId>,
    /// Per-slot "reset on last drop" overrides, in `block_id` order.
    pub(crate) reset_on_release: Vec<bool>,
}

#[cfg(test)]
impl<T: BlockMetadata + Sync> BlockStore<T> {
    /// Test-only mutable-tenure counter for one physical slot.
    pub(crate) fn slot_generation_for_test(&self, block_id: BlockId) -> u64 {
        self.inner.lock().slots[block_id].generation
    }

    #[cfg(test)]
    pub(crate) fn slot_inactive_epoch_for_test(&self, block_id: BlockId) -> u64 {
        self.inner.lock().slots[block_id].inactive_epoch
    }

    /// Test-only deep snapshot of every piece of bookkeeping the unified
    /// mutex protects. Used to assert that `release_blocks` (batched) and
    /// dropping the same guards one at a time (per-block) leave the store
    /// in an indistinguishable state.
    pub(crate) fn debug_snapshot(&self) -> DebugStoreSnapshot {
        let inner = self.inner.lock();
        let slots = inner
            .slots
            .iter()
            .map(|slot| match &slot.state {
                SlotState::Reset => SlotKind::Reset,
                SlotState::Mutable => SlotKind::Mutable,
                SlotState::Staged { seq_hash } => SlotKind::Staged(*seq_hash),
                SlotState::Primary { seq_hash, .. } => SlotKind::Primary(*seq_hash),
                SlotState::Duplicate { seq_hash, .. } => SlotKind::Duplicate(*seq_hash),
                SlotState::Inactive { seq_hash, .. } => SlotKind::Inactive(*seq_hash),
                SlotState::Held { seq_hash, .. } => SlotKind::Held(*seq_hash),
            })
            .collect();
        // Exact FIFO order — see the `free` field docs.
        let free: Vec<BlockId> = inner.free.iter().copied().collect();
        let active_by_hash: std::collections::HashMap<SequenceHash, BlockId> = inner
            .active_by_hash
            .iter()
            .map(|(&h, &id)| (h, id))
            .collect();
        let reset_on_release = inner.reset_on_release.clone();
        DebugStoreSnapshot {
            slots,
            free,
            active_by_hash,
            reset_on_release,
        }
    }
}

impl<T: BlockMetadata> std::fmt::Debug for BlockStore<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockStore")
            .field("block_size", &self.block_size)
            .field("total_blocks", &self.total_blocks)
            .finish()
    }
}

// ---------- helpers ----------

/// Clone the [`BlockRegistrationHandle`] out of an `Inactive` slot
/// without consuming the slot itself. The per-block `reset_on_release`
/// override no longer rides in this variant — it lives in the
/// store-owned atomic array and is read directly via `BlockStore::reset_on_release`.
/// The caller must overwrite `slot.state` before releasing the store lock.
fn take_inactive_handle<T: BlockMetadata>(
    slot: &mut BlockSlot<T>,
    block_id: BlockId,
) -> BlockRegistrationHandle {
    match &slot.state {
        SlotState::Inactive { handle, .. } => handle.clone(),
        other => panic!("expected Inactive state for slot {block_id}, got {other:?}"),
    }
}

/// Hash → strong `Arc<ImmutableBlockInner<T>>` lookup. Walks active
/// then inactive under one store-mutex acquisition. `touch` propagates
/// to the inactive resurrection path so frequency tracking observes
/// the hit even when the active path absorbs it.
pub(crate) fn upgrade_or_resurrect<T: BlockMetadata + Sync>(
    handle: &BlockRegistrationHandle,
    store: &Arc<BlockStore<T>>,
    touch: bool,
) -> Option<Arc<ImmutableBlockInner<T>>> {
    store.acquire_for_hash(handle.seq_hash(), touch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pools::IdBuildHasher;

    /// A handful of distinct, realistically-constructed `SequenceHash`
    /// values. `SequenceHash` (`PositionalLineageHash`) packs
    /// `(current_hash, parent_hash, position)` into its backing `u128`,
    /// so varying any component yields a distinct key.
    fn sample_keys() -> Vec<SequenceHash> {
        vec![
            SequenceHash::new(0x1234, None, 0),
            SequenceHash::new(0x1234, Some(0x1234), 1),
            SequenceHash::new(0x5678, Some(0x1234), 2),
            SequenceHash::new(0xdead_beef, Some(0x5678), 3),
            SequenceHash::new(0xffff_ffff_ffff_ffff, Some(0xdead_beef), 255),
        ]
    }

    /// A `SeqHashMap` must round-trip `SequenceHash` keys. This locks in
    /// the assumption behind [`IdHasher`]: the derived `Hash` for
    /// `SequenceHash` forwards to `write_u128` (so `IdHasher::write`'s
    /// `unreachable!` is never hit — the test would panic there), and
    /// distinct keys do not collide into the same slot.
    #[test]
    fn seq_hash_map_round_trips_keys() {
        let keys = sample_keys();
        let mut map: SeqHashMap<u32> = SeqHashMap::default();

        for (i, &k) in keys.iter().enumerate() {
            map.insert(k, i as u32);
        }
        assert_eq!(map.len(), keys.len(), "no key collisions / overwrites");
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(map.get(&k).copied(), Some(i as u32), "round-trip key {i}");
        }

        // Overwrite + remove behave as a normal HashMap.
        map.insert(keys[0], 999);
        assert_eq!(map.get(&keys[0]).copied(), Some(999));
        assert_eq!(map.remove(&keys[1]), Some(1));
        assert!(!map.contains_key(&keys[1]));
    }

    /// `IdHasher` must produce distinct digests for distinct keys (no
    /// catastrophic folding collision among realistic values) and must
    /// run through `write_u128` — never the `write` byte-slice path
    /// (`hash_one` would panic in `IdHasher::write` if a key did not).
    #[test]
    fn id_hasher_distinguishes_distinct_keys() {
        use std::collections::HashSet;
        use std::hash::BuildHasher;

        let digests: HashSet<u64> = sample_keys()
            .iter()
            .map(|k| IdBuildHasher.hash_one(k))
            .collect();
        assert_eq!(
            digests.len(),
            5,
            "distinct keys must produce distinct digests"
        );
    }
}
