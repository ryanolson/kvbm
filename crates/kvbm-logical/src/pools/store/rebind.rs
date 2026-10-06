use std::sync::Arc;

use crate::blocks::BlockMetadata;
use crate::registry::BlockRegistrationHandle;
use crate::{BlockId, SequenceHash};

use super::{BlockStore, SlotState, effective_ceiling, push_reset_id_locked};

/// A source block could not be prepared for an inactive rebind.
#[derive(Debug, thiserror::Error)]
pub enum RebindPrepareError {
    /// The physical source block is outside the store's capacity.
    #[error("block id {block_id} is out of range")]
    OutOfRange { block_id: BlockId },
    /// The source is not currently in the inactive pool.
    #[error("block id {block_id} is not inactive")]
    NotInactive { block_id: BlockId },
    /// No allocation-eligible reset block is available as a destination.
    #[error("no reset destination is available below the allocation ceiling")]
    NoDestination,
}

/// A reservation for copying an inactive block to a new physical slot.
///
/// The source remains matchable until commit. Copy the source data to
/// [`Self::dst`] before consuming this plan with [`Self::commit`].
#[must_use = "a prepared rebind reserves its destination until committed or dropped"]
pub struct RebindPlan<T: BlockMetadata> {
    store: Arc<BlockStore<T>>,
    seq_hash: SequenceHash,
    src: BlockId,
    dst: BlockId,
    src_generation: u64,
    handle: BlockRegistrationHandle,
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

/// Result of committing an inactive rebind plan.
#[derive(Debug)]
pub enum RebindOutcome<T: BlockMetadata> {
    /// The registration moved to the destination without changing policy order.
    Moved {
        seq_hash: SequenceHash,
        src: BlockId,
        dst: BlockId,
    },
    /// The source is temporarily active or held; retry this plan after release.
    Busy(RebindPlan<T>),
    /// The source changed tenure or identity; the reserved destination was released.
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

    /// Commit the move after the caller has copied source data to the destination.
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
                debug_assert!(rebound, "inactive source identity was missing from its index");
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
            armed: true,
        })
    }

    pub(crate) fn rebind(&self, seq_hash: SequenceHash, src: BlockId, dst: BlockId) -> bool {
        let mut inner = self.inner.lock();
        if src >= inner.capacity
            || dst >= inner.capacity
            || src == dst
            || dst >= effective_ceiling(&inner)
            || !inner.free.contains(&dst)
            || !matches!(
                &inner.slots[src].state,
                SlotState::Inactive { seq_hash: stored, .. } if *stored == seq_hash
            )
        {
            return false;
        }

        let handle = match &inner.slots[src].state {
            SlotState::Inactive { handle, .. } => handle.clone(),
            _ => unreachable!(),
        };
        let rebound = inner.inactive.rebind(seq_hash, src, dst);
        if !rebound {
            debug_assert!(rebound, "inactive source identity was missing from its index");
            return false;
        }

        let reset_on_release = inner.reset_on_release[src];
        inner.free.remove(&dst);
        let _ = self.allocate_mutable_slot(&mut inner, dst);
        self.set_inactive_tenure_state_locked(&mut inner, dst, seq_hash, handle);
        inner.reset_on_release[dst] = reset_on_release;
        inner.reset_on_release[src] = self.default_reset_on_release;
        self.reset_slot_locked(&mut inner, src);
        self.metrics.dec_reset_pool_size();
        true
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
