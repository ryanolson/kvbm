// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded branch-point tracking, attachable to the block registry as an oracle.
//!
//! A **branch point** is a positional node with more than one distinct live child --
//! the point where multiple token sequences currently diverge from a shared prefix.
//!
//! [`BranchOracle`] is the attach point: [`BlockRegistry`](crate::registry::BlockRegistry)
//! holds an `Option<Arc<dyn BranchOracle>>`, exactly like its `frequency_tracker`. Unset,
//! behavior is a no-op ([`NoOpBranchOracle`]) -- fail closed, byte-identical to today.
//! Attached, [`BranchPointTracker`] observes every block registration/removal that flows
//! through the registry and answers `max_fanout` queries.
//!
//! # Bounded state
//!
//! State is bounded by the number of *currently resident* lineages, never by the number
//! of blocks ever registered:
//! - `parents`: one entry per resident non-root block (child hash -> parent key).
//! - `records`: one entry per resident block that currently has, or has ever had while
//!   still resident, at least one live child.
//!
//! `on_block_removed(hash)` drops exactly `hash`'s own entries from both maps:
//! - Its `parents[hash]` entry (its own bookkeeping as *someone else's child*), which
//!   also decrements that parent's `current_fanout` -- but **never** lowers the parent's
//!   `max_fanout`, which is a monotone high-water mark.
//! - Its own `records` entry (its bookkeeping as *a parent in its own right*), if any.
//!   A branch point that loses its last live child "re-leafs" (`current_fanout` -> 0)
//!   but its record and `max_fanout` persist -- the record is only forgotten when the
//!   branch-point block itself is removed.
//!
//! This keeps both maps at O(resident lineages) regardless of how much churn (register /
//! evict / re-register) has flowed through the tracker.
//!
//! # Positional keying rule
//!
//! Every map here that is indexed by a [`SequenceHash`] (`PositionalLineageHash`) keys on
//! the **combination `(position, hash-or-fragment)`**, never a bare hash -- matching the
//! registry's own [`PositionalRadixTree`](crate::registry) `(position, hash)` layout and
//! the inactive-pool lineage backend's `(position, fragment)` index.
//!
//! *Why:* a PLH packs `(mode, position, 64-bit current hash, parent-hash fragment)` into
//! 128 bits, and as `position` grows more bits are spent encoding it, so the stored
//! **parent-hash fragment is truncated** -- 54 bits below position 2^8, 46 below 2^16, and
//! only 38 bits at position >= 2^16 (~>= 1M tokens at typical block sizes). Concretely:
//! - [`Inner::records`] is keyed by `(parent_position, parent_fragment)`; the fragment is
//!   position-masked, hence never used alone.
//! - [`Inner::parents`] is keyed by `(child_position, child_hash)`.
//!
//! Two *distinct* parents that share a position and whose low fragment bits coincide
//! collide in `records` and merge. This is a **known, bounded limitation** that only bites
//! at the >= ~1M-token / 38-bit-fragment regime; it is the accepted pattern under the
//! current PLH bit budget and will be fully resolved when `PositionalLineageHash` widens
//! to 160--192 bits, letting the `SequenceHash` retain all 64 content bits. The
//! `(position, hash)` keying is deliberate, not a defect to be worked around by trying to
//! recover a full parent hash the child no longer carries.
//!
//! # Parent-before-child registration invariant
//!
//! In the block-registration flow blocks register in **prefix order** -- a parent block
//! registers before any of its children. So every inferred parent record `records[K_P]`
//! corresponds to a parent block `P` that has itself registered and will later fire
//! `on_block_removed(P)`, whose `self_as_parent_key(P) == K_P` drops the record. That is
//! what bounds `records` to O(resident lineages).
//!
//! A **pure orphan** child -- registered while its parent block `P` is *never* registered
//! -- is reachable through the raw [`BranchOracle`] API but does **not** occur in the
//! registration flow. Such a child still creates `records[K_P]`, and because `P` never
//! registers, `P` never fires the `on_block_removed(P)` that would reclaim it; the record
//! is pinned until then. Callers driving the oracle outside the prefix-ordered
//! registration path must preserve the parent-before-child discipline to keep the bound.
//!
//! A `Scatter` onboard names one concrete path into this case. `find_scatter`
//! (`kvbm-engine/src/p2p/control.rs`) returns every hash that any selected tier holds, gaps
//! included. A remote pull lands its committed hashes in local `G2`
//! (`kvbm-engine/src/remote/search/plan.rs`, `pull_from`). Under `SearchMode::Scatter`, that
//! pull can register a child block whose parent this instance never fetched. Prefix order is
//! a property of `Prefix` search, not of `Scatter` search.
//!
//! A registry has an oracle only if its builder receives one through `.branch_oracle(...)`.
//! The one production consumer installs the oracle on the registry that `G1`, `G2`, and
//! `G3` share for a `Full` resource under valued eviction. A remote pull that lands in
//! `G2` therefore feeds this oracle today. The only guard against an orphan is that every
//! production requester sends `SearchMode::Prefix`
//! (`kvbm-engine/src/remote/search/plan.rs:146` and
//! `kvbm-engine/src/remote/search/bundle/pull/transfer/mod.rs:99`), never `Scatter`. A
//! `Scatter` requester must restore the parent-before-child discipline, or it must accept a
//! pinned record.

use std::collections::HashMap;

use parking_lot::Mutex;

use crate::SequenceHash;

/// Identifies a node in its role as a *parent*.
///
/// Computed two symmetric ways that are guaranteed to agree (see
/// `PositionalLineageHash::parent_fragment_for_child_position`):
/// - From a child's own hash: `(child.position() - 1, child.parent_hash_fragment())`.
/// - From the parent's own hash: `(parent.position(), parent.parent_fragment_for_child_position(parent.position() + 1))`.
///
/// Fragment-truncated, so not globally unique in isolation -- always paired with
/// position, matching the radix tree's own backward-matching convention elsewhere in
/// this crate.
type ParentKey = (u64, u64);

/// Identifies a node in its role as a *child*, per the positional keying rule
/// (see the module docs): `(position, full child SequenceHash)`. Mirrors the
/// registry's [`PositionalRadixTree`](crate::registry) `(position, hash)`
/// layout. The child's own current-hash bits are not position-masked, but the
/// leading `position` element keeps this key consistent with `records`'
/// `(position, fragment)` key and with every other PLH-keyed index in the crate.
type ChildKey = (u64, SequenceHash);

/// Metadata for a node that currently has, or has ever had while resident, at least one
/// live child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchPointRecord {
    /// Block position of the parent node in the sequence.
    pub position: u64,
    /// Hash fragment identifying the parent node at its position.
    pub fragment: u64,
    /// Current number of distinct live children.
    pub current_fanout: u32,
    /// Peak fanout ever observed for this node while it has been resident. Monotone --
    /// never lowered by child removal, only forgotten when the node itself is removed.
    pub max_fanout: u32,
    /// Number of times a new child was registered under this node (measures
    /// branch-point activity; not decremented on removal).
    pub observation_count: u32,
}

/// Returns the key identifying `hash`'s parent, or `None` if `hash` is a root
/// (position 0, no parent).
fn parent_key_of(hash: SequenceHash) -> Option<ParentKey> {
    let position = hash.position();
    if position == 0 {
        return None;
    }
    Some((position - 1, hash.parent_hash_fragment()))
}

/// Returns the positional key identifying `hash` in its role as a *child*:
/// `(position, hash)`, per the module's positional-keying rule.
fn child_key_of(hash: SequenceHash) -> ChildKey {
    (hash.position(), hash)
}

/// Returns the key that `hash`'s own children would compute as their `parent_key_of`.
fn self_as_parent_key(hash: SequenceHash) -> ParentKey {
    let position = hash.position();
    (
        position,
        hash.parent_fragment_for_child_position(position + 1),
    )
}

/// Attach point for branch-point observation on the block registration path.
///
/// Implementors must tolerate idempotent re-registration (must not double-count a hash
/// that is already live) and idempotent/out-of-order removal (removing an unknown hash
/// is a no-op). [`NoOpBranchOracle`] is the fail-closed default when nothing is
/// attached, matching the registry's `frequency_tracker` pattern.
///
/// **Non-reentrancy:** a callback must not re-enter the [`BlockRegistry`](crate::registry)
/// it is attached to (no `register_sequence_hash` / `match_sequence_hash` / `is_registered`
/// / `remove_batch` on the same registry). The singular removal path fires
/// `on_block_removed` while holding the entry's position-radix guard, so re-entry
/// deadlocks. [`BranchPointTracker`] honors this — it only touches its own internal mutex.
pub trait BranchOracle: Send + Sync {
    /// Called when a block is newly registered in the registry.
    fn on_block_registered(&self, hash: SequenceHash);

    /// Called when a block's registration is fully dropped from the registry.
    fn on_block_removed(&self, hash: SequenceHash);

    /// Peak fanout ever observed for the node identified by `hash`, treating `hash` as
    /// a *parent* (not a child). Returns `None` if `hash` has never had a live child
    /// while resident -- callers that fail open to "shared" on `None` get that behavior
    /// automatically.
    fn max_fanout(&self, hash: SequenceHash) -> Option<u32>;
}

/// Fail-closed default: observes nothing, always reports `None`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpBranchOracle;

impl BranchOracle for NoOpBranchOracle {
    fn on_block_registered(&self, _hash: SequenceHash) {}
    fn on_block_removed(&self, _hash: SequenceHash) {}
    fn max_fanout(&self, _hash: SequenceHash) -> Option<u32> {
        None
    }
}

#[derive(Default)]
struct Inner {
    /// child key `(position, child hash)` -> its parent's key. One entry per resident
    /// non-root block. Keyed positionally per the module's `(position, hash)` rule.
    parents: HashMap<ChildKey, ParentKey>,
    /// parent key `(position, fragment)` -> branch-point record. One entry per resident
    /// block that currently has, or has ever had while resident, a live child.
    records: HashMap<ParentKey, BranchPointRecord>,
}

/// Tracks live parent/child fanout for the block registration path.
///
/// Wrapped around an internal `parking_lot::Mutex` so [`BranchOracle`]'s methods can
/// take `&self`, matching the [`TinyLFUTracker`](crate::tinylfu::TinyLFUTracker) pattern
/// used elsewhere in this crate.
#[derive(Default)]
pub struct BranchPointTracker {
    inner: Mutex<Inner>,
}

impl BranchPointTracker {
    /// Creates a new, empty tracker.
    pub fn new() -> Self {
        Self::default()
    }
}

impl BranchOracle for BranchPointTracker {
    fn on_block_registered(&self, hash: SequenceHash) {
        let Some(key) = parent_key_of(hash) else {
            return; // Root blocks have no parent -- nothing to record.
        };

        let child_key = child_key_of(hash);
        let mut inner = self.inner.lock();

        // Idempotent: an already-live child must not be double-counted.
        if inner.parents.contains_key(&child_key) {
            return;
        }
        inner.parents.insert(child_key, key);

        let record = inner.records.entry(key).or_insert(BranchPointRecord {
            position: key.0,
            fragment: key.1,
            current_fanout: 0,
            max_fanout: 0,
            observation_count: 0,
        });
        record.current_fanout += 1;
        record.max_fanout = record.max_fanout.max(record.current_fanout);
        record.observation_count = record.observation_count.saturating_add(1);
    }

    fn on_block_removed(&self, hash: SequenceHash) {
        let mut inner = self.inner.lock();

        // (1) Drop this block's own entry as a *child*: decrement its parent's live
        // fanout. `max_fanout` is a monotone high-water mark -- never lowered here.
        if let Some(key) = inner.parents.remove(&child_key_of(hash))
            && let Some(record) = inner.records.get_mut(&key)
        {
            record.current_fanout = record.current_fanout.saturating_sub(1);
        }

        // (2) Drop this block's own entry as a *parent* (a branch-point record), if it
        // has one. The record -- and its max_fanout high-water mark -- persists through
        // re-leafing (fanout dropping to zero via child removal above) and is only
        // forgotten here, when the block itself is removed.
        let own_key = self_as_parent_key(hash);
        inner.records.remove(&own_key);
    }

    fn max_fanout(&self, hash: SequenceHash) -> Option<u32> {
        let key = self_as_parent_key(hash);
        self.inner.lock().records.get(&key).map(|r| r.max_fanout)
    }
}

#[cfg(test)]
impl BranchPointTracker {
    /// Test-only: current live fanout for `hash` treated as a parent.
    pub(crate) fn current_fanout(&self, hash: SequenceHash) -> Option<u32> {
        let key = self_as_parent_key(hash);
        self.inner
            .lock()
            .records
            .get(&key)
            .map(|r| r.current_fanout)
    }

    /// Test-only: number of resident branch-point records (the bound this design
    /// targets: O(resident lineages), not O(all-ever-registered)).
    pub(crate) fn record_len(&self) -> usize {
        self.inner.lock().records.len()
    }

    /// Test-only: number of resident child->parent entries (the other bounded map).
    pub(crate) fn parents_len(&self) -> usize {
        self.inner.lock().parents.len()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::BlockRegistry;
    use crate::testing::BlockSequenceBuilder;
    use std::sync::Arc;

    fn build_chain(tokens: Vec<u32>, block_size: usize) -> Vec<SequenceHash> {
        BlockSequenceBuilder::from_tokens(tokens)
            .with_block_size(block_size)
            .build()
            .into_iter()
            .map(|(_, hash)| hash)
            .collect()
    }

    fn register_all(oracle: &dyn BranchOracle, hashes: &[SequenceHash]) {
        for &hash in hashes {
            oracle.on_block_registered(hash);
        }
    }

    /// Test helper: records every hash it observes, so we can assert the registry
    /// invokes the oracle with exactly the expected set (and no more).
    #[derive(Default)]
    struct RecordingOracle {
        registered: Mutex<Vec<SequenceHash>>,
        removed: Mutex<Vec<SequenceHash>>,
    }

    impl BranchOracle for RecordingOracle {
        fn on_block_registered(&self, hash: SequenceHash) {
            self.registered.lock().push(hash);
        }
        fn on_block_removed(&self, hash: SequenceHash) {
            self.removed.lock().push(hash);
        }
        fn max_fanout(&self, _hash: SequenceHash) -> Option<u32> {
            None
        }
    }

    // -----------------------------------------------------------------------
    // (a) No oracle attached => registry behavior is byte-identical to today
    //     (fail-closed no-op), and an attached NoOpBranchOracle behaves the same.
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_oracle_is_fail_closed_noop() {
        let registry = BlockRegistry::new();
        let chain = build_chain(vec![10, 20, 30], 1);
        // Hold the handles alive: BlockRegistry entries are Weak-ref-backed and are
        // removed as soon as the last strong handle drops.
        let handles: Vec<_> = chain
            .iter()
            .map(|&hash| {
                let handle = registry.register_sequence_hash(hash);
                assert_eq!(handle.seq_hash(), hash);
                handle
            })
            .collect();
        assert_eq!(registry.registered_count(), 3);
        drop(handles);
    }

    #[test]
    fn test_explicit_noop_oracle_matches_unset() {
        let registry = BlockRegistry::builder()
            .branch_oracle(Arc::new(NoOpBranchOracle))
            .build();
        let chain = build_chain(vec![10, 20, 30], 1);
        let handles: Vec<_> = chain
            .iter()
            .map(|&hash| {
                let handle = registry.register_sequence_hash(hash);
                assert_eq!(handle.seq_hash(), hash);
                handle
            })
            .collect();
        assert_eq!(registry.registered_count(), 3);
        // The NoOp oracle never records a fanout for anything.
        assert_eq!(NoOpBranchOracle.max_fanout(chain[0]), None);
        drop(handles);
    }

    // -----------------------------------------------------------------------
    // (b) Attached oracle receives exactly the registered hashes; fanout
    //     transitions 1 -> 2 -> K as K children register under one parent.
    // -----------------------------------------------------------------------

    #[test]
    fn test_oracle_receives_exactly_registered_hashes() {
        let oracle = Arc::new(RecordingOracle::default());
        let registry = BlockRegistry::builder()
            .branch_oracle(oracle.clone() as Arc<dyn BranchOracle>)
            .build();

        let chain_a = build_chain(vec![10, 20, 30], 1);
        // Hold the handles alive: BlockRegistry entries are Weak-ref-backed and are
        // removed as soon as the last strong handle drops, which would make a
        // "re-registration" actually be a fresh registration.
        let handles: Vec<_> = chain_a
            .iter()
            .map(|&hash| registry.register_sequence_hash(hash))
            .collect();
        assert_eq!(*oracle.registered.lock(), chain_a);

        // Re-registering the same hashes (while still live) must NOT call the oracle
        // again: the registry only notifies on genuinely new registrations (see
        // register_sequence_hash's early-return on `weak.upgrade()`).
        for &hash in &chain_a {
            registry.register_sequence_hash(hash);
        }
        assert_eq!(
            *oracle.registered.lock(),
            chain_a,
            "re-registration must not re-notify the oracle"
        );
        drop(handles);
    }

    #[test]
    fn test_fanout_transitions_1_to_2_to_k() {
        let tracker = BranchPointTracker::new();

        // Root token 42, then children registered one at a time: fanout should climb
        // 1, 2, ..., K as each new child is observed.
        let root = build_chain(vec![42], 1)[0];
        let k = 5u32;
        for (i, child_token) in (100..100 + k).enumerate() {
            let chain = build_chain(vec![42, child_token], 1);
            register_all(&tracker, &chain);
            let expected_fanout = (i as u32) + 1;
            assert_eq!(tracker.current_fanout(root), Some(expected_fanout));
            assert_eq!(tracker.max_fanout(root), Some(expected_fanout));
        }
        assert_eq!(tracker.current_fanout(root), Some(k));
    }

    // -----------------------------------------------------------------------
    // (c) Absent-record max_fanout query returns None.
    // -----------------------------------------------------------------------

    #[test]
    fn test_absent_record_max_fanout_is_none() {
        let tracker = BranchPointTracker::new();
        let never_registered = build_chain(vec![999], 1)[0];
        assert_eq!(tracker.max_fanout(never_registered), None);

        // A registered root with zero children also has no record yet.
        let root = build_chain(vec![7], 1)[0];
        tracker.on_block_registered(root);
        assert_eq!(tracker.max_fanout(root), None);
    }

    // -----------------------------------------------------------------------
    // (d) BOUNDED-GROWTH GUARD (kill-mutation): register N distinct lineages,
    //     remove them all, internal map sizes return to baseline -- not O(N).
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounded_growth_returns_to_baseline_after_removal() {
        let tracker = BranchPointTracker::new();

        let n = 200;
        let mut all_hashes = Vec::new();
        for i in 0..n {
            let root_token = 10_000 + i;
            let chain = build_chain(vec![root_token as u32, (root_token + 1) as u32], 1);
            register_all(&tracker, &chain);
            all_hashes.extend(chain);
        }

        assert_eq!(tracker.parents_len(), n as usize);
        assert!(tracker.record_len() >= 1);

        // Remove every block from every lineage (children first, then roots -- typical
        // eviction order, though the contract holds regardless of order).
        for &hash in &all_hashes {
            tracker.on_block_removed(hash);
        }

        assert_eq!(
            tracker.parents_len(),
            0,
            "parents map must return to baseline after full removal, not stay O(N)"
        );
        assert_eq!(
            tracker.record_len(),
            0,
            "records map must return to baseline after full removal, not stay O(N)"
        );
    }

    // -----------------------------------------------------------------------
    // (e) HIGH-WATER PRESERVATION ACROSS RE-LEAF (kill-mutation): max_fanout is a
    //     monotone high-water mark that survives a branch point losing all its
    //     live children, and is only forgotten when the branch-point block itself
    //     is removed.
    // -----------------------------------------------------------------------

    #[test]
    fn test_high_water_mark_survives_re_leaf() {
        let tracker = BranchPointTracker::new();

        let root = build_chain(vec![77], 1)[0];
        tracker.on_block_registered(root);

        let k = 4u32;
        let mut children = Vec::new();
        for child_token in 200..200 + k {
            let chain = build_chain(vec![77, child_token], 1);
            register_all(&tracker, &chain);
            children.push(chain[1]);
        }
        assert_eq!(tracker.max_fanout(root), Some(k));
        assert_eq!(tracker.current_fanout(root), Some(k));

        // Remove all but the last child.
        for &child in &children[..children.len() - 1] {
            tracker.on_block_removed(child);
        }
        assert_eq!(
            tracker.max_fanout(root),
            Some(k),
            "max_fanout must not drop while children are being removed"
        );
        assert_eq!(tracker.current_fanout(root), Some(1));

        // Remove the LAST child: re-leaf. max_fanout must STILL be K.
        tracker.on_block_removed(*children.last().unwrap());
        assert_eq!(
            tracker.max_fanout(root),
            Some(k),
            "max_fanout must persist through re-leaf (current_fanout -> 0)"
        );
        assert_eq!(tracker.current_fanout(root), Some(0));

        // A brand-new child under the same root: current_fanout resets to 1, max_fanout
        // stays at its high-water mark (>= k, here exactly k).
        let new_child_chain = build_chain(vec![77, 999], 1);
        register_all(&tracker, &new_child_chain);
        assert_eq!(tracker.current_fanout(root), Some(1));
        assert!(tracker.max_fanout(root).unwrap() >= k);

        // Removing the branch-point BLOCK ITSELF (root) drops its record entirely.
        tracker.on_block_removed(root);
        assert_eq!(tracker.max_fanout(root), None);
    }

    // -----------------------------------------------------------------------
    // Additional: root blocks never register a parent-side record for
    // themselves as a child (position 0 has no parent).
    // -----------------------------------------------------------------------

    #[test]
    fn test_root_blocks_have_no_parent_entry() {
        let tracker = BranchPointTracker::new();
        let root = build_chain(vec![55], 1)[0];
        tracker.on_block_registered(root);
        assert_eq!(tracker.parents_len(), 0, "root has no parent to record");
    }

    // -----------------------------------------------------------------------
    // Additional: registry removal path invokes on_block_removed when the last
    // strong reference to a registration handle is dropped.
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_drop_invokes_on_block_removed() {
        let oracle = Arc::new(RecordingOracle::default());
        let registry = BlockRegistry::builder()
            .branch_oracle(oracle.clone() as Arc<dyn BranchOracle>)
            .build();

        let chain = build_chain(vec![1, 2], 1);
        {
            let handle = registry.register_sequence_hash(chain[1]);
            assert_eq!(*oracle.registered.lock(), vec![chain[1]]);
            drop(handle);
        }
        assert_eq!(*oracle.removed.lock(), vec![chain[1]]);
    }

    // -----------------------------------------------------------------------
    // (f) TRANSFER PAIRING (kill-mutation): `transfer_registration` creates a
    //     fresh canonical handle WITHOUT firing `on_block_registered`. Its
    //     Drop must therefore NOT fire `on_block_removed` -- an unpaired
    //     removal underflows fanout / phantom-deletes a live record. The
    //     callbacks must be paired 1:1.
    // -----------------------------------------------------------------------

    #[test]
    fn test_transfer_registration_does_not_fire_unpaired_removal() {
        let oracle = Arc::new(RecordingOracle::default());
        let registry = BlockRegistry::builder()
            .branch_oracle(oracle.clone() as Arc<dyn BranchOracle>)
            .build();

        let chain = build_chain(vec![1, 2], 1);
        {
            // Fresh registration via transfer: no `on_block_registered`.
            let handle = registry.transfer_registration(chain[1]);
            assert!(
                oracle.registered.lock().is_empty(),
                "transfer_registration must not fire on_block_registered"
            );
            drop(handle);
        }
        // ... and, because it never registered, its Drop must not remove either.
        assert!(
            oracle.removed.lock().is_empty(),
            "transfer-created handle fired an unpaired on_block_removed"
        );
    }

    #[test]
    fn test_transfer_then_drop_does_not_phantom_delete_record() {
        // A live branch-point record can exist for a *non-resident* parent: it
        // is created by the first CHILD registering under it. Registering only
        // the child `b` of `a` creates `records[K_a]` (max_fanout = 1), and
        // `max_fanout(a)` reads it back via the parent<->child fragment symmetry
        // -- even though `a` itself is not resident.
        let tracker = Arc::new(BranchPointTracker::new());
        let registry = BlockRegistry::builder()
            .branch_oracle(tracker.clone() as Arc<dyn BranchOracle>)
            .build();

        let chain = build_chain(vec![7, 8], 1);
        let parent = chain[0];
        let _child = registry.register_sequence_hash(chain[1]);
        assert_eq!(
            tracker.max_fanout(parent),
            Some(1),
            "child registration must create the parent's branch-point record"
        );

        // Transfer-register the (non-resident) PARENT block. Under the bug its
        // Drop fires an unpaired `on_block_removed(parent)` whose
        // `self_as_parent_key(parent)` collides with `K_a`, phantom-deleting the
        // live record. Paired correctly, the record survives untouched.
        {
            let handle = registry.transfer_registration(parent);
            assert_eq!(
                tracker.max_fanout(parent),
                Some(1),
                "transfer must not observe a registration"
            );
            drop(handle);
        }
        assert_eq!(
            tracker.max_fanout(parent),
            Some(1),
            "transfer+drop phantom-deleted a live branch-point record (unpaired removal)"
        );
    }
}
