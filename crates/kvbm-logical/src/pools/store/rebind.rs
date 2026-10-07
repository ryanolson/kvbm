use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

use crate::blocks::{BlockMetadata, ImmutableBlock, ImmutableBlockInner};
use crate::registry::BlockRegistrationHandle;
use crate::{BlockId, SequenceHash};

use super::{BlockStore, SlotState, effective_ceiling, push_reset_id_locked};

/// A source block could not be prepared for a rebind.
#[derive(Debug, thiserror::Error)]
pub enum RebindPrepareError {
    /// The physical source block is outside the store's capacity.
    #[error("block id {block_id} is out of range")]
    OutOfRange { block_id: BlockId },
    /// The source is not currently in the inactive pool.
    #[error("block id {block_id} is not inactive")]
    NotInactive { block_id: BlockId },
    /// The source is not a live primary or duplicate block.
    #[error("block id {block_id} is not a live registered block")]
    NotLive { block_id: BlockId },
    /// No allocation-eligible reset block is available as a destination.
    #[error("no reset destination is available below the allocation ceiling")]
    NoDestination,
}

/// A reservation for copying a registered block to a new physical slot.
///
/// Copy the source data to [`Self::dst`] before you call [`Self::commit`] or
/// [`Self::commit_live`].
#[must_use = "a prepared rebind reserves its destination until committed or dropped"]
pub struct RebindPlan<T: BlockMetadata> {
    store: Arc<BlockStore<T>>,
    seq_hash: SequenceHash,
    src: BlockId,
    dst: BlockId,
    src_generation: u64,
    handle: BlockRegistrationHandle,
    source_inner: Option<Weak<ImmutableBlockInner<T>>>,
    armed: bool,
}

impl<T: BlockMetadata> std::fmt::Debug for RebindPlan<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RebindPlan")
            .field("seq_hash", &self.seq_hash)
            .field("src", &self.src)
            .field("dst", &self.dst)
            .field("src_generation", &self.src_generation)
            .field("armed", &self.armed)
            .finish()
    }
}

/// Result of committing an inactive or live rebind plan.
#[derive(Debug)]
pub enum RebindOutcome<T: BlockMetadata> {
    /// The registration moved to the destination without changing its identity.
    Moved {
        seq_hash: SequenceHash,
        src: BlockId,
        dst: BlockId,
    },
    /// The source is active or held during `commit`.
    /// Its last live holder is dropping during `commit_live`.
    /// The plan remains reserved for a retry.
    Busy(RebindPlan<T>),
    /// The source changed tenure or identity.
    /// The store released the reserved destination.
    Stale,
    /// The allocation ceiling was lowered over the reserved destination.
    Fenced,
}

impl<T: BlockMetadata> RebindPlan<T> {
    /// Physical source block captured when the plan was prepared.
    pub fn src(&self) -> BlockId {
        self.src
    }

    /// Reserved physical destination block.
    pub fn dst(&self) -> BlockId {
        self.dst
    }

    /// Registered sequence hash captured from the inactive source.
    pub fn sequence_hash(&self) -> SequenceHash {
        self.seq_hash
    }

    /// Commit an inactive-source move after copying its data to the destination.
    pub fn commit(mut self) -> RebindOutcome<T> {
        let store = Arc::clone(&self.store);
        let mut inner = store.inner.lock();
        assert!(
            matches!(inner.slots[self.dst].state, SlotState::Mutable),
            "rebind plan no longer owns its mutable destination"
        );

        if self.dst >= effective_ceiling(&inner) {
            release_reserved_destination(&store, &mut inner, self.dst);
            self.armed = false;
            return RebindOutcome::Fenced;
        }

        let source_identity = match &inner.slots[self.src].state {
            SlotState::Inactive { seq_hash, handle } => {
                *seq_hash == self.seq_hash
                    && inner.slots[self.src].generation == self.src_generation
                    && Arc::ptr_eq(&handle.inner, &self.handle.inner)
            }
            _ => false,
        };

        if source_identity {
            let rebound = inner.inactive.rebind(self.seq_hash, self.src, self.dst);
            if !rebound {
                debug_assert!(
                    rebound,
                    "inactive source identity was missing from its index"
                );
                release_reserved_destination(&store, &mut inner, self.dst);
                self.armed = false;
                return RebindOutcome::Stale;
            }

            let reset_on_release = inner.reset_on_release[self.src];
            store.set_inactive_tenure_state_locked(
                &mut inner,
                self.dst,
                self.seq_hash,
                self.handle.clone(),
            );
            inner.reset_on_release[self.dst] = reset_on_release;
            inner.reset_on_release[self.src] = store.default_reset_on_release;
            store.reset_slot_locked(&mut inner, self.src);
            self.armed = false;
            return RebindOutcome::Moved {
                seq_hash: self.seq_hash,
                src: self.src,
                dst: self.dst,
            };
        }

        let same_active_tenure = matches!(
            &inner.slots[self.src].state,
            SlotState::Primary { seq_hash, .. }
                | SlotState::Duplicate { seq_hash, .. }
                | SlotState::Held { seq_hash, .. }
                if *seq_hash == self.seq_hash
                    && inner.slots[self.src].generation == self.src_generation
        );
        if same_active_tenure {
            return RebindOutcome::Busy(self);
        }

        release_reserved_destination(&store, &mut inner, self.dst);
        self.armed = false;
        RebindOutcome::Stale
    }

    /// Commit after copying source data to the destination.
    ///
    /// Unlike [`Self::commit`], this moves a live primary or duplicate in
    /// place, so all existing strong and weak handles observe the destination.
    pub fn commit_live(mut self) -> RebindOutcome<T> {
        let store = Arc::clone(&self.store);
        let mut inner = store.inner.lock();
        assert!(
            matches!(inner.slots[self.dst].state, SlotState::Mutable),
            "rebind plan no longer owns its mutable destination"
        );

        if self.dst >= effective_ceiling(&inner) {
            release_reserved_destination(&store, &mut inner, self.dst);
            self.armed = false;
            return RebindOutcome::Fenced;
        }

        let inactive_identity = match &inner.slots[self.src].state {
            SlotState::Inactive { seq_hash, handle } => {
                *seq_hash == self.seq_hash
                    && inner.slots[self.src].generation == self.src_generation
                    && Arc::ptr_eq(&handle.inner, &self.handle.inner)
            }
            _ => false,
        };
        if inactive_identity {
            let rebound = inner.inactive.rebind(self.seq_hash, self.src, self.dst);
            if !rebound {
                debug_assert!(
                    rebound,
                    "inactive source identity was missing from its index"
                );
                release_reserved_destination(&store, &mut inner, self.dst);
                self.armed = false;
                return RebindOutcome::Stale;
            }

            let reset_on_release = inner.reset_on_release[self.src];
            store.set_inactive_tenure_state_locked(
                &mut inner,
                self.dst,
                self.seq_hash,
                self.handle.clone(),
            );
            inner.reset_on_release[self.dst] = reset_on_release;
            inner.reset_on_release[self.src] = store.default_reset_on_release;
            store.reset_slot_locked(&mut inner, self.src);
            self.armed = false;
            return RebindOutcome::Moved {
                seq_hash: self.seq_hash,
                src: self.src,
                dst: self.dst,
            };
        }

        let active_source = match &inner.slots[self.src].state {
            SlotState::Primary {
                seq_hash,
                handle,
                inner: weak,
            } if *seq_hash == self.seq_hash
                && inner.slots[self.src].generation == self.src_generation
                && Arc::ptr_eq(&handle.inner, &self.handle.inner)
                && self
                    .source_inner
                    .as_ref()
                    .is_none_or(|source| source.as_ptr() == weak.as_ptr()) =>
            {
                Some((true, weak.clone(), handle.clone()))
            }
            SlotState::Duplicate {
                seq_hash,
                handle,
                inner: weak,
            } if *seq_hash == self.seq_hash
                && inner.slots[self.src].generation == self.src_generation
                && Arc::ptr_eq(&handle.inner, &self.handle.inner)
                && self
                    .source_inner
                    .as_ref()
                    .is_none_or(|source| source.as_ptr() == weak.as_ptr()) =>
            {
                Some((false, weak.clone(), handle.clone()))
            }
            _ => None,
        };
        let Some((is_primary, source_inner, handle)) = active_source else {
            release_reserved_destination(&store, &mut inner, self.dst);
            self.armed = false;
            return RebindOutcome::Stale;
        };
        let Some(live_inner) = source_inner.upgrade() else {
            drop(inner);
            return RebindOutcome::Busy(self);
        };

        let moved_state = if is_primary {
            SlotState::Primary {
                seq_hash: self.seq_hash,
                handle,
                inner: source_inner,
            }
        } else {
            SlotState::Duplicate {
                seq_hash: self.seq_hash,
                handle,
                inner: source_inner,
            }
        };
        inner.slots[self.dst].state = moved_state;
        let reset_on_release = inner.reset_on_release[self.src];
        inner.reset_on_release[self.dst] = reset_on_release;
        inner.reset_on_release[self.src] = store.default_reset_on_release;
        if is_primary {
            let previous = inner.active_by_hash.insert(self.seq_hash, self.dst);
            debug_assert_eq!(previous, Some(self.src));
        }
        store.reset_slot_locked(&mut inner, self.src);
        live_inner.block_id.store(self.dst, Ordering::Release);
        self.armed = false;

        drop(inner);
        drop(live_inner);
        RebindOutcome::Moved {
            seq_hash: self.seq_hash,
            src: self.src,
            dst: self.dst,
        }
    }
}

impl<T: BlockMetadata> Drop for RebindPlan<T> {
    fn drop(&mut self) {
        if self.armed {
            let mut inner = self.store.inner.lock();
            assert!(
                matches!(inner.slots[self.dst].state, SlotState::Mutable),
                "dropped rebind plan no longer owns its mutable destination"
            );
            release_reserved_destination(&self.store, &mut inner, self.dst);
            self.armed = false;
        }
    }
}

impl<T: BlockMetadata> BlockStore<T> {
    pub(crate) fn prepare_rebind(
        self: &Arc<Self>,
        src: BlockId,
    ) -> Result<RebindPlan<T>, RebindPrepareError> {
        let mut inner = self.inner.lock();
        if src >= inner.capacity {
            return Err(RebindPrepareError::OutOfRange { block_id: src });
        }

        let (seq_hash, handle) = match &inner.slots[src].state {
            SlotState::Inactive { seq_hash, handle } => (*seq_hash, handle.clone()),
            _ => return Err(RebindPrepareError::NotInactive { block_id: src }),
        };
        let dst = inner
            .free
            .pop_first()
            .ok_or(RebindPrepareError::NoDestination)?;
        let src_generation = inner.slots[src].generation;
        let _ = self.allocate_mutable_slot(&mut inner, dst);
        self.metrics.dec_reset_pool_size();

        Ok(RebindPlan {
            store: Arc::clone(self),
            seq_hash,
            src,
            dst,
            src_generation,
            handle,
            source_inner: None,
            armed: true,
        })
    }
}

impl<T: BlockMetadata + Sync> BlockStore<T> {
    pub(crate) fn live_blocks_from(&self, first: BlockId) -> Vec<ImmutableBlock<T>> {
        let inner = self.inner.lock();
        if first >= inner.capacity {
            return Vec::new();
        }

        // Upgrades only add strong references; they do not drop Arcs under this lock.
        let mut blocks = Vec::new();
        for slot in &inner.slots[first..inner.capacity] {
            let block_inner = match &slot.state {
                SlotState::Primary { inner, .. } | SlotState::Duplicate { inner, .. } => inner,
                _ => continue,
            };
            if let Some(block_inner) = block_inner.upgrade() {
                blocks.push(ImmutableBlock::from_inner(block_inner));
            }
        }
        blocks
    }

    pub(crate) fn prepare_live_rebind(
        self: &Arc<Self>,
        block: &ImmutableBlock<T>,
    ) -> Result<RebindPlan<T>, RebindPrepareError> {
        let mut inner = self.inner.lock();
        let src = block.block_id();
        if src >= inner.capacity {
            return Err(RebindPrepareError::NotLive { block_id: src });
        }

        let block_inner = block.inner_arc();
        let (seq_hash, handle) = match &inner.slots[src].state {
            SlotState::Primary {
                seq_hash,
                handle,
                inner: weak,
            }
            | SlotState::Duplicate {
                seq_hash,
                handle,
                inner: weak,
            } if weak.as_ptr() == Arc::as_ptr(block_inner) => (*seq_hash, handle.clone()),
            _ => return Err(RebindPrepareError::NotLive { block_id: src }),
        };
        let dst = inner
            .free
            .pop_first()
            .ok_or(RebindPrepareError::NoDestination)?;
        let src_generation = inner.slots[src].generation;
        let _ = self.allocate_mutable_slot(&mut inner, dst);
        self.metrics.dec_reset_pool_size();

        Ok(RebindPlan {
            store: Arc::clone(self),
            seq_hash,
            src,
            dst,
            src_generation,
            handle,
            source_inner: Some(Arc::downgrade(block_inner)),
            armed: true,
        })
    }
}

fn release_reserved_destination<T: BlockMetadata>(
    store: &BlockStore<T>,
    inner: &mut super::BlockStoreInner<T>,
    dst: BlockId,
) {
    inner.reset_on_release[dst] = store.default_reset_on_release;
    inner.slots[dst].state = SlotState::Reset;
    push_reset_id_locked(inner, dst);
    store.metrics.inc_reset_pool_size();
}
