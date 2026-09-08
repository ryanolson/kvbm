// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block registration handle and its inner implementation.

use super::attachments::{AttachmentError, AttachmentStore, TypedAttachments};
use super::{BlockRegistry, PositionalRadixTree};

use crate::blocks::{BlockMetadata, SequenceHash};
use crate::branch_tracker::BranchOracle;
use crate::events::protocol::EventReleaseHandle;

use dashmap::DashMap;

use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

// Under `#[cfg(test)]`, swap in `tracing-mutex`'s parking_lot wrapper
// so the test suite enforces the documented `attachments → store`
// lock-acquisition ordering at runtime via a global DAG. Identical API
// in release builds; zero cost.
#[cfg(not(test))]
use parking_lot::Mutex;
#[cfg(test)]
use tracing_mutex::parkinglot::Mutex;

/// Handle that represents a block registration in the global registry.
/// This handle is cloneable and can be shared across pools.
#[derive(Clone, Debug)]
pub struct BlockRegistrationHandle {
    pub(crate) inner: Arc<BlockRegistrationHandleInner>,
}

/// Type alias for touch callback functions.
type TouchCallback = Arc<dyn Fn(SequenceHash) + Send + Sync>;

pub(crate) struct BlockRegistrationHandleInner {
    /// Sequence hash of the block
    seq_hash: SequenceHash,
    /// Attachments for the block
    pub(crate) attachments: Mutex<AttachmentStore>,
    /// Callbacks invoked when this handle is touched
    touch_callbacks: Mutex<Vec<TouchCallback>>,
    /// Weak reference to the registry - allows us to remove the block from the registry on drop
    registry: Weak<PositionalRadixTree<Weak<BlockRegistrationHandleInner>>>,
    /// Branch oracle to notify on removal (mirrors the registry's own field at the time
    /// this handle was created). `None` when branch tracking isn't attached -- fail-closed.
    /// A transfer-created inner is deliberately `None` (it never fired
    /// `on_block_registered`); [`BlockRegistry::remove_batch`](super::BlockRegistry::remove_batch)
    /// reads this to skip firing an unpaired removal for such handles.
    pub(super) branch_oracle: Option<Arc<dyn BranchOracle>>,
    /// Set by [`BlockRegistry::remove_batch`](super::BlockRegistry::remove_batch) once it
    /// has already deregistered this entry (and fired the oracle) under a single
    /// per-position lock. `Drop` checks this FIRST and returns before touching the
    /// registry, so the batched path collapses N per-position locks to P and the oracle
    /// never double-fires.
    removed_via_batch: AtomicBool,
}

impl std::fmt::Debug for BlockRegistrationHandleInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockRegistrationHandleInner")
            .field("seq_hash", &self.seq_hash)
            .field("attachments", &self.attachments)
            .field(
                "touch_callbacks",
                &format!("[{} callbacks]", self.touch_callbacks.lock().len()),
            )
            .finish()
    }
}

impl BlockRegistrationHandleInner {
    pub(super) fn new(
        seq_hash: SequenceHash,
        registry: Weak<PositionalRadixTree<Weak<BlockRegistrationHandleInner>>>,
        branch_oracle: Option<Arc<dyn BranchOracle>>,
    ) -> Self {
        Self {
            seq_hash,
            attachments: Mutex::new(AttachmentStore::new()),
            touch_callbacks: Mutex::new(Vec::new()),
            registry,
            branch_oracle,
            removed_via_batch: AtomicBool::new(false),
        }
    }

    /// Marks this registration as already removed by the batched path so the subsequent
    /// [`Drop`] is a no-op. Called under the entry's position guard, immediately before
    /// [`BlockRegistry::remove_batch`](super::BlockRegistry::remove_batch) releases the
    /// last strong reference.
    pub(super) fn mark_removed_via_batch(&self) {
        self.removed_via_batch.store(true, Ordering::Release);
    }

    /// Take the publisher of this registration's `Remove` event out of the
    /// attachment store. Both removal paths call this under the entry's
    /// position guard: dropping the returned handle there publishes the
    /// `Remove` inside the critical section, and
    /// [`EventReleaseHandle::disarm`] there suppresses the `Remove` of a
    /// registration that a newer one replaced.
    pub(super) fn take_event_release(&self) -> Option<EventReleaseHandle> {
        self.attachments.lock().event_release.take()
    }
}

/// Identity-checked removal of a single registry entry, performed under an already-held
/// position guard (`map`). This is the ONE shared *removal decision* for both the singular
/// [`Drop`] path and [`BlockRegistry::remove_batch`](super::BlockRegistry::remove_batch),
/// so an entry is deregistered on identical terms either way.
///
/// It does **not** fire [`BranchOracle::on_block_removed`] — each caller owns notification,
/// because they must fire it against *different* oracles and at *different* times:
/// - `Drop` fires the **handle's own** `branch_oracle` inline (under this guard). A
///   transfer-created inner carries `branch_oracle: None` (it never fired
///   `on_block_registered`), so its drop correctly fires nothing — the pairing invariant.
/// - `remove_batch` collects the removed hashes and fires the oracle **after** releasing
///   the guard (a public `BranchOracle` impl must not re-enter the registry, but deferring
///   the call keeps the batch path safe even if one does), skipping transfer-created
///   handles the same way (see its body).
///
/// The stored `Weak` pointer must match `identity`.
/// A different pointer identifies a replacement registration.
/// An absent slot means that another path already removed an entry.
/// Both cases leave the map unchanged.
pub(super) fn remove_entry_if_identity(
    map: &DashMap<SequenceHash, Weak<BlockRegistrationHandleInner>>,
    seq_hash: SequenceHash,
    identity: *const BlockRegistrationHandleInner,
) -> bool {
    let Some(weak_ref) = map.get(&seq_hash) else {
        return false;
    };
    let should_remove = std::ptr::eq(weak_ref.as_ptr(), identity);
    drop(weak_ref);
    if should_remove {
        map.remove(&seq_hash);
    }
    should_remove
}

impl Drop for BlockRegistrationHandleInner {
    #[inline]
    fn drop(&mut self) {
        // Batched removal already deregistered this entry (and fired the oracle) under one
        // per-position lock; skip the singular per-position lock entirely. This early
        // return is what lets `remove_batch` collapse N locks to P, and it keeps
        // `on_block_removed` firing exactly once per removed hash.
        if self.removed_via_batch.load(Ordering::Acquire) {
            // `remove_batch` released this registration's `Remove` under its own
            // position guard, so nothing is left to publish here.
            return;
        }
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        // The position-level write lock held by `prefix()` for the lifetime of `map`
        // serializes us against concurrent `register_sequence_hash` and
        // `transfer_registration` on this `seq_hash`, so the stored `Weak` is stable across
        // the identity check performed by `remove_entry_if_identity`.
        let map = registry.prefix(&self.seq_hash);
        if remove_entry_if_identity(&map, self.seq_hash, self as *const Self) {
            // Publish the `Remove` while `map` is held. A racing
            // `register_sequence_hash` publishes its `Create` under this same guard,
            // and the hub keeps one holder set per hash, so a `Remove` released after
            // that `Create` deletes a block this instance still holds. The event goes
            // out before the oracle call, so a panicking oracle cannot push it past
            // the guard.
            drop(self.take_event_release());
            // Fire the *handle's own* oracle (a transfer-created inner has `None` here and
            // so fires nothing — the pairing invariant). Held under `map`; `BranchOracle`
            // impls must not re-enter the registry (documented on the trait).
            if let Some(oracle) = &self.branch_oracle {
                oracle.on_block_removed(self.seq_hash);
            }
        } else if let Some(release) = self.take_event_release() {
            // A newer registration owns this slot, or already removed it. Its `Create`
            // is the authoritative one and its own drop publishes the `Remove`, so this
            // registration must publish nothing.
            release.disarm();
        }
    }
}

impl BlockRegistrationHandle {
    pub(crate) fn from_inner(inner: Arc<BlockRegistrationHandleInner>) -> Self {
        Self { inner }
    }

    pub fn seq_hash(&self) -> SequenceHash {
        self.inner.seq_hash
    }

    pub fn is_from_registry(&self, registry: &BlockRegistry) -> bool {
        self.inner
            .registry
            .upgrade()
            .map(|reg| Arc::ptr_eq(&reg, &registry.prt))
            .unwrap_or(false)
    }

    /// Store the publisher of this registration's `Remove` event.
    /// [`EventsManager::on_block_registered`](crate::events::EventsManager::on_block_registered)
    /// calls this right after it publishes the `Create`, under the position
    /// guard that `register_sequence_hash` holds.
    pub(crate) fn attach_event_release(&self, release: EventReleaseHandle) {
        self.inner.attachments.lock().event_release = Some(release);
    }

    /// Increment the physical-residency marker for tier `T`. Each
    /// registration transition (`Staged → Primary`, `Staged → Duplicate`)
    /// calls this exactly once. The marker remains set while a slot is
    /// `Primary`, `Duplicate`, `Inactive`, or `Held`.
    pub(crate) fn mark_present<T: BlockMetadata>(&self) {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();
        *attachments.presence_markers.entry(type_id).or_insert(0) += 1;
    }

    /// Decrement the physical-residency marker for tier `T`. Each
    /// presence-removing slot transition (`Inactive → Mutable` via
    /// eviction, `Held → Reset` via pressure commit, or `Duplicate → Reset`
    /// via last-duplicate drop) calls this exactly once. The entry is removed
    /// on reaching zero.
    pub(crate) fn mark_absent<T: BlockMetadata>(&self) {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();
        match attachments.presence_markers.get_mut(&type_id) {
            Some(count) => {
                debug_assert!(*count > 0, "mark_absent on zero-count presence marker");
                *count -= 1;
                if *count == 0 {
                    attachments.presence_markers.remove(&type_id);
                }
            }
            None => debug_assert!(false, "mark_absent with no presence marker present"),
        }
    }

    /// Returns `true` if a physical registered slot exists for this sequence
    /// hash and tier `T` (the refcount is greater than zero). This includes a
    /// `Held` slot and does not prove request availability.
    ///
    /// This is a **refcounted shadow** of authoritative `BlockStore<T>`
    /// state, not a linearizable snapshot. The store is updated under
    /// its own mutex; this counter is incremented/decremented in a
    /// separate critical section that runs after the store lock is
    /// released. In steady state the shadow agrees with the store; while
    /// a registration, eviction, or duplicate drop is mid-flight it can
    /// briefly report the pre-update value. A held slot remains present, but
    /// `BlockManager::match_blocks` and `BlockManager::scan_matches` cannot
    /// return it. Callers who need request availability must use those
    /// store-backed operations.
    pub fn has_block<T: BlockMetadata>(&self) -> bool {
        let type_id = TypeId::of::<T>();
        let attachments = self.inner.attachments.lock();
        attachments
            .presence_markers
            .get(&type_id)
            .copied()
            .unwrap_or(0)
            > 0
    }

    /// Returns `true` if physical registered residency exists for at least
    /// one specified metadata-tier `TypeId`. This does not prove request
    /// availability.
    pub fn has_any_block(&self, type_ids: &[TypeId]) -> bool {
        let attachments = self.inner.attachments.lock();
        type_ids.iter().any(|type_id| {
            attachments
                .presence_markers
                .get(type_id)
                .copied()
                .unwrap_or(0)
                > 0
        })
    }

    /// Register a callback to be invoked when this handle is touched.
    pub fn on_touch(&self, callback: Arc<dyn Fn(SequenceHash) + Send + Sync>) {
        self.inner.touch_callbacks.lock().push(callback);
    }

    /// Fire all registered touch callbacks with this handle's sequence hash.
    pub fn touch(&self) {
        let callbacks: Vec<_> = self.inner.touch_callbacks.lock().clone();
        let seq_hash = self.inner.seq_hash;
        for cb in &callbacks {
            cb(seq_hash);
        }
    }

    /// Get a typed accessor for attachments of type T
    pub fn get<T: Any + Send + Sync>(&self) -> TypedAttachments<'_, T> {
        TypedAttachments {
            handle: self,
            _phantom: PhantomData,
        }
    }

    /// Attach a unique value of type T to this handle.
    /// Only one value per type is allowed - subsequent calls will replace the previous value.
    /// Returns an error if type T is already registered as multiple attachment.
    pub fn attach_unique<T: Any + Send + Sync>(&self, value: T) -> Result<(), AttachmentError> {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();

        if let Some(super::attachments::AttachmentMode::Multiple) =
            attachments.type_registry.get(&type_id)
        {
            return Err(AttachmentError::TypeAlreadyRegisteredAsMultiple(type_id));
        }

        attachments
            .unique_attachments
            .insert(type_id, Box::new(value));
        attachments
            .type_registry
            .insert(type_id, super::attachments::AttachmentMode::Unique);

        Ok(())
    }

    /// Attach a value of type T to this handle.
    /// Multiple values per type are allowed - this will append to existing values.
    /// Returns an error if type T is already registered as unique attachment.
    pub fn attach<T: Any + Send + Sync>(&self, value: T) -> Result<(), AttachmentError> {
        let type_id = TypeId::of::<T>();
        let mut attachments = self.inner.attachments.lock();

        if let Some(super::attachments::AttachmentMode::Unique) =
            attachments.type_registry.get(&type_id)
        {
            return Err(AttachmentError::TypeAlreadyRegisteredAsUnique(type_id));
        }

        attachments
            .multiple_attachments
            .entry(type_id)
            .or_default()
            .push(Box::new(value));
        attachments
            .type_registry
            .insert(type_id, super::attachments::AttachmentMode::Multiple);

        Ok(())
    }
}
