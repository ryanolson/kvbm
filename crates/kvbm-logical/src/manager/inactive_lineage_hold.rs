// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owner-mediated inactive lineage holds.

use std::marker::PhantomData;
use std::sync::{Arc, Weak};

use parking_lot::RwLock;

use crate::blocks::BlockMetadata;
use crate::pools::InactiveCandidate;
use crate::pools::store::StoreInactiveLineageHold;
use crate::{BlockId, ManagerId, SequenceHash};

use super::BlockEvictionObserver;

/// Opaque proof that one inactive candidate still names one complete lineage.
///
/// A preflight records the exact root-to-leaf source set under the store lock.
/// [`crate::BlockManager::try_hold_prepared_inactive_lineage`] consumes it and
/// rechecks the manager, candidate, and full source set under that same lock
/// before it mutates the pool.
#[must_use = "consume the preflight through its owning BlockManager"]
pub struct InactiveLineagePreflight<T: BlockMetadata> {
    manager_id: ManagerId,
    candidate: InactiveCandidate,
    source_blocks: Vec<(SequenceHash, BlockId)>,
    marker: PhantomData<fn() -> T>,
}

impl<T: BlockMetadata> InactiveLineagePreflight<T> {
    pub(super) fn new(
        manager_id: ManagerId,
        candidate: InactiveCandidate,
        source_blocks: Vec<(SequenceHash, BlockId)>,
    ) -> Self {
        Self {
            manager_id,
            candidate,
            source_blocks,
            marker: PhantomData,
        }
    }

    /// Exact number of blocks that the later hold must claim.
    pub fn block_count(&self) -> usize {
        self.source_blocks.len()
    }

    /// Logical hashes in root-to-leaf order.
    ///
    /// This exposes no physical slot identity. The manager keeps that identity
    /// inside the proof until a later exact hold succeeds.
    pub fn source_hashes(&self) -> Vec<SequenceHash> {
        self.source_blocks.iter().map(|(hash, _)| *hash).collect()
    }

    /// Inactive candidate that this proof binds.
    pub const fn candidate(&self) -> InactiveCandidate {
        self.candidate
    }

    pub(super) fn matches_manager(&self, manager_id: ManagerId) -> bool {
        self.manager_id == manager_id
    }

    pub(super) fn into_parts(self) -> (InactiveCandidate, Vec<(SequenceHash, BlockId)>) {
        (self.candidate, self.source_blocks)
    }
}

impl<T: BlockMetadata> std::fmt::Debug for InactiveLineagePreflight<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InactiveLineagePreflight")
            .field("block_count", &self.source_blocks.len())
            .finish_non_exhaustive()
    }
}

/// Exclusive ownership of a complete inactive lineage.
///
/// Drop aborts the action and restores the full lineage. Commit releases
/// only the selected leaf and restores every support block.
#[must_use = "dropping the hold aborts the pressure action"]
pub struct InactiveLineageHold<T: BlockMetadata> {
    manager_id: ManagerId,
    store_hold: StoreInactiveLineageHold<T>,
    eviction_notifier: EvictionNotifier,
}

impl<T: BlockMetadata> InactiveLineageHold<T> {
    pub(super) fn new(
        manager_id: ManagerId,
        store_hold: StoreInactiveLineageHold<T>,
        eviction_notifier: EvictionNotifier,
    ) -> Self {
        Self {
            manager_id,
            store_hold,
            eviction_notifier,
        }
    }

    /// Manager that owns every source slot in this hold.
    pub const fn manager_id(&self) -> ManagerId {
        self.manager_id
    }

    /// Complete source blocks in root-to-leaf order.
    pub fn source_blocks(&self) -> &[(SequenceHash, BlockId)] {
        self.store_hold.source_blocks()
    }

    /// Restore every support block and release the selected leaf without an
    /// observer callback.
    ///
    /// The returned token owns the complete notification. The store mutation
    /// ends before this method returns, so a later observer panic cannot undo
    /// or corrupt the committed source state.
    pub fn commit_victim_release_silent(self) -> EvictionNotification {
        let Self {
            manager_id: _,
            store_hold,
            eviction_notifier,
        } = self;
        let hashes = store_hold.commit_victim_release().into_iter().collect();
        eviction_notifier.deferred_notification(hashes)
    }

    /// Restore every support block and release the selected leaf.
    pub fn commit_victim_release(self) {
        self.commit_victim_release_silent().notify();
    }
}

/// One deferred eviction observer notification.
///
/// The token is one-shot. Its private fields own every observer and hash that
/// the notification needs. Calling [`Self::notify`] consumes the token.
#[must_use = "call notify after every related source commit completes"]
pub struct EvictionNotification {
    eviction_notifier: EvictionNotifier,
    hashes: Vec<SequenceHash>,
}

impl EvictionNotification {
    /// Notify observers after the source mutation has completed.
    pub fn notify(self) {
        self.eviction_notifier.notify(&self.hashes);
    }
}

#[derive(Clone, Default)]
pub(super) struct EvictionNotifier {
    observers: Arc<RwLock<Vec<Weak<dyn BlockEvictionObserver>>>>,
}

impl EvictionNotifier {
    /// Create a one-shot notification without calling an observer.
    pub(super) fn deferred_notification(&self, hashes: Vec<SequenceHash>) -> EvictionNotification {
        EvictionNotification {
            eviction_notifier: self.clone(),
            hashes,
        }
    }

    pub(super) fn observe(&self, observer: &Arc<dyn BlockEvictionObserver>) {
        self.observers.write().push(Arc::downgrade(observer));
    }

    pub(super) fn notify(&self, hashes: &[SequenceHash]) {
        if hashes.is_empty() {
            return;
        }
        let observers = {
            let mut registered = self.observers.write();
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
}
