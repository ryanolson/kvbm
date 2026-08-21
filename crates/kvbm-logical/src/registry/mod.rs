// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global registry for block deduplication via weak references and sequence hash matching.
//!
//! The [`BlockRegistry`] is the central coordination point for block deduplication in the
//! KVBM system. It maps sequence hashes to registration handles using a
//! [`dynamo_tokens::PositionalRadixTree`], enabling efficient prefix-based lookups.
//!
//! # Architecture
//!
//! ```text
//! BlockRegistry
//!   └── PositionalRadixTree<Weak<BlockRegistrationHandleInner>>
//!         ├── seq_hash_1 → Handle → AttachmentStore (presence markers, weak refs, typed data)
//!         ├── seq_hash_2 → Handle → AttachmentStore
//!         └── ...
//! ```
//!
//! - **Handle**: One per sequence hash. Ties blocks across all pool tiers (active, inactive).
//! - **Attachments**: Arbitrary typed data stored on handles (unique or multiple per type).
//! - **Presence markers**: Track physical registered residency per tier. They
//!   include `Held` slots and do not prove request availability.
//! - **Weak references**: Enable block resurrection during pool transitions.
//!
//! # Future directions
//!
//! - Delegate pattern to decouple EventsManager from BlockRegistry
//! - Cross-pool touch tracking
//! - RAII attachment guards

mod attachments;
mod handle;
mod registration;

#[cfg(test)]
pub(crate) mod tests;

// Re-export public types
pub use attachments::{AttachmentError, TypedAttachments};
pub use handle::BlockRegistrationHandle;

use crate::{branch_tracker::BranchOracle, events::EventsManager, tinylfu::FrequencyTracker};

use crate::blocks::SequenceHash;

use std::sync::{Arc, Weak};

use handle::BlockRegistrationHandleInner;

pub(crate) type PositionalRadixTree<V> = dynamo_tokens::PositionalRadixTree<V, SequenceHash>;

// NOTE(B4): batched removal empties a position's per-hash bucket but leaves the now-empty
// position shard resident in the outer `DashMap<position, ..>`. This residual is accepted
// (registered *count* stays correct via `len()`, which sums inner-map sizes). Before any
// future structural change here, check upstream dynamo HEAD for a `tokens/radix.rs`
// empty-position-prune fix rather than adding a bespoke prune.

/// Builder for [`BlockRegistry`].
///
/// # Example
///
/// ```ignore
/// // Simple registry with no tracking
/// let registry = BlockRegistry::builder().build();
///
/// // With frequency tracking
/// let registry = BlockRegistry::builder()
///     .frequency_tracker(tracker)
///     .build();
///
/// // With both frequency tracking and event management
/// let registry = BlockRegistry::builder()
///     .frequency_tracker(tracker)
///     .event_manager(events_manager)
///     .build();
/// ```
#[derive(Default)]
pub struct BlockRegistryBuilder {
    frequency_tracker: Option<Arc<dyn FrequencyTracker<u128>>>,
    event_manager: Option<Arc<EventsManager>>,
    branch_oracle: Option<Arc<dyn BranchOracle>>,
}

impl BlockRegistryBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the frequency tracker for block access tracking.
    pub fn frequency_tracker(mut self, tracker: Arc<dyn FrequencyTracker<u128>>) -> Self {
        self.frequency_tracker = Some(tracker);
        self
    }

    /// Sets the events manager for distributed coordination.
    // TODO(delegate): Replace direct EventsManager coupling with a delegate/observer pattern.
    pub fn event_manager(mut self, manager: Arc<EventsManager>) -> Self {
        self.event_manager = Some(manager);
        self
    }

    /// Sets the branch oracle for branch-point fanout tracking. Unset, the registry is
    /// a no-op with respect to branch tracking (fail-closed).
    pub fn branch_oracle(mut self, oracle: Arc<dyn BranchOracle>) -> Self {
        self.branch_oracle = Some(oracle);
        self
    }

    /// Builds the BlockRegistry.
    pub fn build(self) -> BlockRegistry {
        BlockRegistry {
            frequency_tracker: self.frequency_tracker,
            event_manager: self.event_manager,
            branch_oracle: self.branch_oracle,
            prt: Arc::new(PositionalRadixTree::new()),
            #[cfg(test)]
            prefix_lock_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

/// Global registry for managing block registrations.
/// Tracks canonical blocks and provides registration handles.
#[derive(Clone)]
pub struct BlockRegistry {
    pub(crate) prt: Arc<PositionalRadixTree<Weak<BlockRegistrationHandleInner>>>,
    frequency_tracker: Option<Arc<dyn FrequencyTracker<u128>>>,
    // TODO(delegate): Replace direct EventsManager field with a delegate/observer trait.
    event_manager: Option<Arc<EventsManager>>,
    branch_oracle: Option<Arc<dyn BranchOracle>>,
    /// Test-only counter, bumped on each per-position `prefix()` (position-bucket lock)
    /// acquisition made by [`remove_batch`](Self::remove_batch). Lets a test assert P
    /// acquisitions for N removals across P positions; degrouping to per-hash removal
    /// flips the count to N.
    #[cfg(test)]
    prefix_lock_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl BlockRegistry {
    /// Creates a new builder for BlockRegistry.
    pub fn builder() -> BlockRegistryBuilder {
        BlockRegistryBuilder::new()
    }

    /// Creates a new BlockRegistry with no tracking.
    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn has_frequency_tracking(&self) -> bool {
        self.frequency_tracker.is_some()
    }

    pub fn touch(&self, seq_hash: SequenceHash) {
        if let Some(tracker) = &self.frequency_tracker {
            tracker.touch(seq_hash.as_u128());
        }
    }

    pub fn count(&self, seq_hash: SequenceHash) -> u32 {
        if let Some(tracker) = &self.frequency_tracker {
            tracker.count(seq_hash.as_u128())
        } else {
            0
        }
    }

    /// Check presence of sequence hashes for blocks with specific metadata type `T`.
    /// Returns `Vec<(SequenceHash, bool)>` where `bool` indicates whether a
    /// physically registered slot is currently believed to exist in the
    /// active, inactive, or held state for this tier.
    ///
    /// # Consistency model
    ///
    /// This view is a **refcounted shadow** of the authoritative
    /// `BlockStore<T>` state, *not* a linearizable snapshot. The store is
    /// the single source of truth for slot state and is updated under its
    /// own mutex; the registry-side presence count is then incremented
    /// (`mark_present`) or decremented (`mark_absent`) in a separate
    /// critical section that runs *after* the store mutex has been
    /// released — see `pools/store.rs::register_completed_block`,
    /// `allocate_atomic`, `release_duplicate`, and `drain_inactive_to_mutable`.
    ///
    /// In steady state and after every operation has fully completed, the
    /// shadow count agrees with the authoritative state because the
    /// per-slot increments and decrements commute (refcounted). However,
    /// while a registration, eviction, or duplicate drop is mid-flight,
    /// `check_presence` can briefly report the pre-update value. A `true`
    /// result proves physical registered residency only. It does not prove
    /// request availability. A held block remains present, but
    /// `BlockManager::match_blocks` and `BlockManager::scan_matches` must
    /// not return it. Callers who need request availability must use those
    /// store-backed operations or serialize against the mutating operation.
    ///
    /// Does NOT trigger frequency tracking.
    pub fn check_presence<T: crate::blocks::BlockMetadata>(
        &self,
        seq_hashes: &[SequenceHash],
    ) -> Vec<(SequenceHash, bool)> {
        seq_hashes
            .iter()
            .map(|&seq_hash| {
                let handle_result = self.match_sequence_hash(seq_hash, false);
                let present = handle_result
                    .as_ref()
                    .map(|handle| handle.has_block::<T>())
                    .unwrap_or(false);

                tracing::debug!(
                    ?seq_hash,
                    type_name = std::any::type_name::<T>(),
                    handle_found = handle_result.is_some(),
                    present,
                    "check_presence result"
                );

                (seq_hash, present)
            })
            .collect()
    }

    /// Check presence of sequence hashes for blocks with any of the specified metadata types.
    /// Returns `Vec<(SequenceHash, bool)>` where `bool` is true if a block
    /// exists for at least one of the supplied tier `TypeId`s.
    ///
    /// Same consistency caveats as [`check_presence`]: this is a
    /// refcounted shadow of physical registered residency, not a
    /// linearizable snapshot. A `true` result does not prove request
    /// availability. It can briefly disagree with the store during a mutation.
    ///
    /// Does NOT trigger frequency tracking.
    pub fn check_presence_any(
        &self,
        seq_hashes: &[SequenceHash],
        type_ids: &[std::any::TypeId],
    ) -> Vec<(SequenceHash, bool)> {
        seq_hashes
            .iter()
            .map(|&seq_hash| {
                let present = self
                    .match_sequence_hash(seq_hash, false)
                    .map(|handle| handle.has_any_block(type_ids))
                    .unwrap_or(false);
                (seq_hash, present)
            })
            .collect()
    }

    /// Register a sequence hash and get a registration handle.
    /// If the sequence is already registered, returns the existing handle.
    /// Otherwise, creates a new canonical registration.
    /// This method triggers frequency tracking.
    // TODO(delegate): This is where `on_block_registered` is called. Future delegate
    // pattern should replace the direct EventsManager call here.
    #[inline]
    pub fn register_sequence_hash(&self, seq_hash: SequenceHash) -> BlockRegistrationHandle {
        let map = self.prt.prefix(&seq_hash);
        let mut weak = map.entry(seq_hash).or_default();

        if let Some(inner) = weak.upgrade() {
            return BlockRegistrationHandle::from_inner(inner);
        }

        let inner = self.create_registration(seq_hash);
        *weak = Arc::downgrade(&inner);
        let handle = BlockRegistrationHandle::from_inner(inner);

        if let Some(event_manager) = &self.event_manager
            && let Err(e) = event_manager.on_block_registered(&handle)
        {
            tracing::warn!("Failed to register block with event manager: {}", e);
        }
        self.touch(seq_hash);
        if let Some(oracle) = &self.branch_oracle {
            oracle.on_block_registered(seq_hash);
        }

        handle
    }

    /// Acquire the per-position radix bucket (the outer-shard write guard) for `hash`.
    /// The only path `remove_batch` takes a position lock, so bumping the test counter
    /// *inside* the acquisition keeps the counter faithful to real acquisitions: any
    /// restructuring that moves this call into a per-handle loop (i.e. degroups to per-hash
    /// removal) turns P acquisitions into N and the lock-count test fails.
    #[inline]
    fn acquire_position(
        &self,
        hash: &SequenceHash,
    ) -> dashmap::mapref::one::RefMut<
        '_,
        u64,
        dashmap::DashMap<SequenceHash, Weak<BlockRegistrationHandleInner>>,
    > {
        #[cfg(test)]
        self.prefix_lock_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.prt.prefix(hash)
    }

    /// Batched, identity-checked removal of registry entries, grouped by position so each
    /// touched position's radix bucket is locked exactly ONCE -- versus the N independent
    /// per-position locks taken when N registration handles drop one at a time. Replaces
    /// those N singular `Drop`-path removals.
    ///
    /// **Precondition:** Each handle must belong to this registry.
    /// Removal still requires an exact pointer match.
    /// A stale or foreign handle cannot remove a current entry.
    ///
    /// Each handle is consumed by value (`remove_batch` releases the strong references it
    /// is handed). Within each position group, under a single position guard, a handle is
    /// removed only when it is the *last* strong reference to its registration
    /// (`strong_count == 1`, stable because the guard blocks concurrent registry upgrades)
    /// AND the stored `Weak` still points to that same inner (identity check -- a newer
    /// registration that replaced the slot is left intact). The removed handle is flagged
    /// immediately so its subsequent `Drop` is a no-op (no second per-position lock).
    ///
    /// `on_block_removed` fires exactly once per removed hash, and only for handles that
    /// carry an oracle -- transfer-created handles (`branch_oracle: None`, they never fired
    /// `on_block_registered`) are skipped, preserving the pairing invariant. The
    /// notifications are deferred until **after** the position guard is released (so a
    /// re-entrant `BranchOracle` impl can't deadlock) and after the flag is set (so a
    /// panicking impl can't double-panic through the group's drop). A handle that is *not*
    /// the last reference is left for its eventual last-drop to remove through the singular
    /// path.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn remove_batch(&self, handles: Vec<BlockRegistrationHandle>) {
        use std::collections::HashMap;

        // Group by position so each position is locked exactly once.
        let mut by_position: HashMap<u64, Vec<BlockRegistrationHandle>> = HashMap::new();
        for handle in handles {
            by_position
                .entry(handle.seq_hash().position())
                .or_default()
                .push(handle);
        }

        // Phase 1: under each position's guard, remove + FLAG every removable handle and
        // collect the hashes to notify. Flagging EVERY removed handle across ALL groups
        // before firing ANY notification is what makes a panicking `BranchOracle` safe: a
        // notification panic in Phase 2 unwinds and drops `by_position`, but every removed
        // inner is already flagged so its `Drop` is a no-op -- no still-unmarked handle in an
        // unprocessed group re-enters the singular path to fire the panicking oracle a second
        // time (a double-panic would abort). A handle removed only when it is the *last*
        // strong reference (`strong_count == 1`, stable under the guard) AND the stored `Weak`
        // still points to it (identity check); a `strong_count > 1` handle is left for its
        // eventual last-drop.
        let mut to_notify: Vec<SequenceHash> = Vec::new();
        for group in by_position.values() {
            // ONE lock acquisition per position (see `acquire_position`).
            let map = self.acquire_position(&group[0].seq_hash());
            for handle in group {
                let inner = &handle.inner;
                if Arc::strong_count(inner) == 1
                    && handle::remove_entry_if_identity(&map, handle.seq_hash(), Arc::as_ptr(inner))
                {
                    inner.mark_removed_via_batch();
                    if inner.branch_oracle.is_some() {
                        to_notify.push(handle.seq_hash());
                    }
                }
            }
            // Release this position's guard before the next acquisition and before any
            // notification (a re-entrant oracle would otherwise deadlock on this position).
            drop(map);
        }

        // Phase 2: every removed handle is flagged; fire notifications outside all guards. A
        // registered handle's `branch_oracle` is a clone of the registry's, so firing via
        // `self.branch_oracle` matches every collected hash; transfer-created handles were
        // skipped above (their inner oracle is `None`).
        if let Some(oracle) = self.branch_oracle.as_ref() {
            for hash in to_notify {
                oracle.on_block_removed(hash);
            }
        }
        // `by_position` drops here: every batch-removed inner is flagged -> `Drop` no-op; a
        // `strong_count > 1` handle's clone drop leaves the registration to its last owner.
    }

    /// Test-only: number of per-position lock acquisitions `remove_batch` has made since
    /// construction (see [`prefix_lock_count`](Self::prefix_lock_count)).
    #[cfg(test)]
    pub(crate) fn prefix_locks_taken(&self) -> usize {
        self.prefix_lock_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Internal method for transferring block registration without triggering frequency tracking.
    /// Used when copying blocks between pools where we don't want to count the transfer as a new access.
    #[allow(dead_code)]
    pub(crate) fn transfer_registration(&self, seq_hash: SequenceHash) -> BlockRegistrationHandle {
        let map = self.prt.prefix(&seq_hash);
        let mut weak = map.entry(seq_hash).or_default();

        match weak.upgrade() {
            Some(inner) => BlockRegistrationHandle::from_inner(inner),
            None => {
                // A transfer is a pool-to-pool move, not a new access: it deliberately
                // skips frequency tracking, and — unlike `register_sequence_hash` — it
                // does NOT fire `on_block_registered`. A fresh inner must therefore be
                // created WITHOUT a branch oracle; otherwise its `Drop` would fire an
                // *unpaired* `on_block_removed` (fanout underflow / phantom-record
                // delete), because no matching registration was ever observed. See
                // `registry/handle.rs`'s `Drop` impl, which fires the oracle.
                let inner = Arc::new(BlockRegistrationHandleInner::new(
                    seq_hash,
                    Arc::downgrade(&self.prt),
                    None,
                ));
                *weak = Arc::downgrade(&inner);
                BlockRegistrationHandle::from_inner(inner)
            }
        }
    }

    fn create_registration(&self, seq_hash: SequenceHash) -> Arc<BlockRegistrationHandleInner> {
        Arc::new(BlockRegistrationHandleInner::new(
            seq_hash,
            Arc::downgrade(&self.prt),
            self.branch_oracle.clone(),
        ))
    }

    /// Match a sequence hash and return a registration handle.
    /// This method triggers frequency tracking.
    #[inline]
    pub fn match_sequence_hash(
        &self,
        seq_hash: SequenceHash,
        touch: bool,
    ) -> Option<BlockRegistrationHandle> {
        let result = self
            .prt
            .prefix(&seq_hash)
            .get(&seq_hash)
            .and_then(|weak| weak.upgrade())
            .map(BlockRegistrationHandle::from_inner);

        if result.is_some() && touch {
            self.touch(seq_hash);
        }

        result
    }

    /// Check if a sequence is currently registered (has a canonical handle).
    #[inline]
    pub fn is_registered(&self, seq_hash: SequenceHash) -> bool {
        self.prt
            .prefix(&seq_hash)
            .get(&seq_hash)
            .map(|weak| weak.strong_count() > 0)
            .unwrap_or(false)
    }

    /// Get the current number of registered blocks.
    pub fn registered_count(&self) -> usize {
        self.prt.len()
    }

    /// Get the frequency tracker if frequency tracking is enabled.
    pub fn frequency_tracker(&self) -> Option<Arc<dyn FrequencyTracker<u128>>> {
        self.frequency_tracker.clone()
    }

    /// Get the branch oracle if branch-point tracking is enabled.
    pub fn branch_oracle(&self) -> Option<Arc<dyn BranchOracle>> {
        self.branch_oracle.clone()
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod remove_batch_tests {
    use super::{BlockRegistry, handle};
    use crate::blocks::SequenceHash;
    use crate::branch_tracker::{BranchOracle, BranchPointTracker};
    use crate::testing::BlockSequenceBuilder;
    use parking_lot::Mutex;
    use std::sync::Arc;

    fn build_chain(tokens: Vec<u32>) -> Vec<SequenceHash> {
        BlockSequenceBuilder::from_tokens(tokens)
            .with_block_size(1)
            .build()
            .into_iter()
            .map(|(_, hash)| hash)
            .collect()
    }

    /// Records every removal so a test can assert exactly-once-per-hash semantics.
    #[derive(Default)]
    struct CountingOracle {
        removed: Mutex<Vec<SequenceHash>>,
    }
    impl BranchOracle for CountingOracle {
        fn on_block_registered(&self, _hash: SequenceHash) {}
        fn on_block_removed(&self, hash: SequenceHash) {
            self.removed.lock().push(hash);
        }
        fn max_fanout(&self, _hash: SequenceHash) -> Option<u32> {
            None
        }
    }

    // (kill-mutation) per-position lock count: batch-remove N hashes across P positions;
    // the batch must acquire the per-position lock ONCE per position (P), not once per hash
    // (N). Degrouping `remove_batch` to per-hash removal flips this to N and fails.
    #[test]
    fn remove_batch_locks_once_per_position() {
        let registry = BlockRegistry::new();
        let n_chains = 3usize;
        let depth = 4usize; // positions 0..depth-1 => P = depth
        let mut handles = Vec::new();
        for c in 0..n_chains as u32 {
            let tokens: Vec<u32> = (0..depth as u32).map(|i| c * 1000 + i).collect();
            for hash in build_chain(tokens) {
                handles.push(registry.register_sequence_hash(hash));
            }
        }
        let n = handles.len();
        assert_eq!(n, n_chains * depth);
        assert_eq!(registry.registered_count(), n);

        registry.remove_batch(handles);

        assert_eq!(registry.registered_count(), 0, "all entries removed");
        assert_eq!(
            registry.prefix_locks_taken(),
            depth,
            "batch must lock ONCE per position ({depth}), not once per hash ({n})"
        );
    }

    // (kill-mutation) identity-checked removal: a registration `a` occupies slot X; the
    // shared batch/Drop call point `remove_if_identity` is handed X paired with a DIFFERENT
    // live handle `b`'s identity. The stored `Weak` points at `a`, not `b`, so the
    // compare-before-remove leaves `a` intact -- the replacement survives. Dropping the
    // `ptr::eq` compare deletes `a` and fails.
    #[test]
    fn identity_check_preserves_the_stored_registration() {
        let registry = BlockRegistry::new();
        let x = build_chain(vec![7])[0];
        let y = build_chain(vec![9])[0];
        let a = registry.register_sequence_hash(x); // slot X -> a
        let b = registry.register_sequence_hash(y); // separate live inner
        assert!(registry.is_registered(x));

        // Under the position guard for X, attempt removal of X but with b's identity.
        let map = registry.prt.prefix(&x);
        let removed = handle::remove_entry_if_identity(&map, x, Arc::as_ptr(&b.inner));
        drop(map);

        assert!(!removed, "mismatched identity must not remove");
        assert!(
            registry.is_registered(x),
            "identity check must leave the live registration for X intact"
        );
        drop((a, b));
    }

    #[test]
    fn identity_check_accepts_an_already_empty_slot() {
        let registry = BlockRegistry::new();
        let hash = build_chain(vec![7])[0];
        let map = registry.prt.prefix(&hash);

        let removed = handle::remove_entry_if_identity(&map, hash, std::ptr::null());

        assert!(!removed, "an empty slot is already removed");
    }

    // (kill-mutation) transfer pairing: `transfer_registration` builds a handle whose inner
    // carries `branch_oracle: None` because it never fired `on_block_registered`. Batch-
    // removing it must NOT fire `on_block_removed` -- an unpaired removal phantom-deletes a
    // live BranchPointRecord. Firing via the registry's oracle instead of the handle's own
    // (the pre-fix bug) re-fires it and this test fails; the entry must still be removed.
    #[test]
    fn batch_remove_of_transfer_created_handle_fires_no_unpaired_removal() {
        let oracle = Arc::new(CountingOracle::default());
        let registry = BlockRegistry::builder()
            .branch_oracle(oracle.clone() as Arc<dyn BranchOracle>)
            .build();

        // Fresh registration via transfer: no `on_block_registered`, inner oracle is None.
        let hash = build_chain(vec![1, 2])[1];
        let handle = registry.transfer_registration(hash);
        assert!(registry.is_registered(hash));

        registry.remove_batch(vec![handle]);

        assert!(
            oracle.removed.lock().is_empty(),
            "batch-removing a transfer-created (oracle-less) handle must not fire on_block_removed"
        );
        assert!(
            !registry.is_registered(hash),
            "the entry must still be deregistered from the radix tree"
        );
    }

    // Batch removes only the handles it is given -- unlisted live registrations are not
    // collateral.
    #[test]
    fn remove_batch_does_not_touch_unlisted_registrations() {
        let registry = BlockRegistry::new();
        let x = build_chain(vec![7])[0];
        let y = build_chain(vec![9])[0];
        let a = registry.register_sequence_hash(x);
        let survivor = registry.register_sequence_hash(y);

        registry.remove_batch(vec![a]);

        assert!(!registry.is_registered(x), "listed handle removed");
        assert!(registry.is_registered(y), "unlisted registration survives");
        drop(survivor);
    }

    // (kill-mutation) panic-safety: a `BranchOracle::on_block_removed` that panics partway
    // through the batch notification loop must surface as ONE caught unwind -- never a
    // double-panic abort. Every removed handle across ALL groups is flagged before ANY
    // notification fires, so unwinding drops the whole handle set as `Drop` no-ops; no
    // unprocessed group re-enters the singular path to fire the panicking oracle again.
    // Notifying group-by-group (the pre-fix structure) leaves later groups unflagged -> their
    // unwind-time `Drop` double-panics -> SIGABRT kills the whole test binary.
    #[test]
    fn panicking_oracle_surfaces_single_panic_not_abort() {
        struct PanicOnSecondRemoval {
            calls: Mutex<u32>,
        }
        impl BranchOracle for PanicOnSecondRemoval {
            fn on_block_registered(&self, _hash: SequenceHash) {}
            fn on_block_removed(&self, _hash: SequenceHash) {
                let mut calls = self.calls.lock();
                *calls += 1;
                assert!(*calls < 2, "oracle panics on its 2nd removal notification");
            }
            fn max_fanout(&self, _hash: SequenceHash) -> Option<u32> {
                None
            }
        }

        let oracle = Arc::new(PanicOnSecondRemoval {
            calls: Mutex::new(0),
        });
        let registry = BlockRegistry::builder()
            .branch_oracle(oracle.clone() as Arc<dyn BranchOracle>)
            .build();

        // Multiple chains across multiple positions => multiple groups, several removals.
        let mut handles = Vec::new();
        for c in 0..4u32 {
            for hash in build_chain(vec![c * 10, c * 10 + 1, c * 10 + 2]) {
                handles.push(registry.register_sequence_hash(hash));
            }
        }

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            registry.remove_batch(handles);
        }));
        assert!(
            outcome.is_err(),
            "the panicking oracle must surface as a single caught unwind"
        );
        // A double-panic abort would have killed the process before reaching here.
    }

    // Oracle reconciliation: on_block_removed fires exactly once per removed hash through
    // the batched path (no double-fire, no miss).
    #[test]
    fn batch_fires_on_block_removed_exactly_once_per_hash() {
        let oracle = Arc::new(CountingOracle::default());
        let registry = BlockRegistry::builder()
            .branch_oracle(oracle.clone() as Arc<dyn BranchOracle>)
            .build();

        let mut expected = Vec::new();
        let mut handles = Vec::new();
        for c in 0..5u32 {
            for hash in build_chain(vec![c * 10, c * 10 + 1, c * 10 + 2]) {
                expected.push(hash);
                handles.push(registry.register_sequence_hash(hash));
            }
        }

        registry.remove_batch(handles);

        let mut removed = oracle.removed.lock().clone();
        removed.sort();
        expected.sort();
        assert_eq!(
            removed, expected,
            "on_block_removed must fire exactly once per removed hash (no double, no miss)"
        );
        assert_eq!(registry.registered_count(), 0);
    }

    // Oracle reconciliation: the BranchOracle bounded-growth invariant still returns to
    // baseline (record_len / parents_len back to O(resident) == 0) through the batch.
    #[test]
    fn batch_removal_returns_branch_tracker_to_baseline() {
        let tracker = Arc::new(BranchPointTracker::new());
        let registry = BlockRegistry::builder()
            .branch_oracle(tracker.clone() as Arc<dyn BranchOracle>)
            .build();

        let mut handles = Vec::new();
        for c in 0..20u32 {
            // root + child => one non-root parent entry per lineage.
            for hash in build_chain(vec![c, c + 100]) {
                handles.push(registry.register_sequence_hash(hash));
            }
        }
        assert!(tracker.record_len() >= 1);
        assert_eq!(tracker.parents_len(), 20);

        registry.remove_batch(handles);

        assert_eq!(
            tracker.parents_len(),
            0,
            "parents map must return to baseline through the batched path"
        );
        assert_eq!(
            tracker.record_len(),
            0,
            "records map must return to baseline through the batched path"
        );
        assert_eq!(registry.registered_count(), 0);
    }

    // A singular drop races with batch removal on the same slots.
    // The final state must contain no phantom entry.
    #[test]
    fn concurrent_register_and_batch_remove_leave_no_phantom() {
        use std::thread;
        let registry = BlockRegistry::new();
        let hashes = build_chain((0..48u32).collect());

        let registrar = {
            let registry = registry.clone();
            let hashes = hashes.clone();
            thread::spawn(move || {
                for _ in 0..300 {
                    for &h in &hashes {
                        // Singular Drop path, racing the batch path on the same slots.
                        drop(registry.register_sequence_hash(h));
                    }
                }
            })
        };

        for _ in 0..300 {
            let batch: Vec<_> = hashes
                .iter()
                .map(|&h| registry.register_sequence_hash(h))
                .collect();
            registry.remove_batch(batch);
        }
        registrar.join().unwrap();

        // Every strong reference is gone at this point.
        // No slot can remain registered.
        assert_eq!(registry.registered_count(), 0);
    }
}
