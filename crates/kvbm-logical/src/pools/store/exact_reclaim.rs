// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact inactive reclaim transactions for the unified block store.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::blocks::{BlockMetadata, MutableBlock};
use crate::pools::{
    ExactAllocationError, ExactInactiveVictim, ExactReclaimEntryPlan, ExactReclaimNameError,
    ExactReclaimRefreshError, FreshExactReclaimPlan,
};
use crate::{BlockId, SequenceHash};

use super::{BlockStore, SlotState, take_inactive_handle};

/// A backend-local rejection for an ordered exact-reclaim preflight.
///
/// The store owns slot identities and capacity. The inactive backend owns
/// structural removal validity, including parent and child order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExactReclaimPlanError {
    /// The backend cannot prove an exact ordered plan.
    Unsupported,
    /// A supplied identity is not present in the backend.
    MissingVictim { block_id: BlockId },
    /// A selected parent appears before one of its selected live children.
    ParentBeforeChild {
        parent_block_id: BlockId,
        child_block_id: BlockId,
    },
    /// A selected parent has a live child outside the supplied plan.
    OmittedLiveChild {
        parent_block_id: BlockId,
        child_block_id: BlockId,
    },
    /// A selected victim does not include one of its real ancestors.
    MissingRealAncestor {
        child_block_id: BlockId,
        parent_block_id: BlockId,
    },
}

impl<T: BlockMetadata> BlockStore<T> {
    pub(crate) fn supports_exact_reclaim(&self) -> bool {
        self.inner.lock().inactive.supports_exact_reclaim()
    }

    /// Name one current inactive leaf from its logical hash.
    ///
    /// This point query does not depend on a bounded eviction snapshot. The
    /// returned name keeps all physical identity private.
    pub(crate) fn name_complete_inactive_entry_by_hash(
        &self,
        seq_hash: SequenceHash,
    ) -> Result<ExactReclaimEntryPlan, ExactReclaimNameError> {
        let inner = self.inner.lock();
        if !inner.inactive.supports_exact_reclaim() {
            return Err(ExactReclaimNameError::UnsupportedBackend);
        }
        let Some(block_id) = inner.inactive.exact_block_id(seq_hash) else {
            return Err(ExactReclaimNameError::StaleCandidate);
        };
        let Some(slot) = inner.slots.get(block_id) else {
            return Err(ExactReclaimNameError::StaleCandidate);
        };
        if !matches!(&slot.state, SlotState::Inactive { seq_hash: stored, .. } if *stored == seq_hash)
        {
            return Err(ExactReclaimNameError::StaleCandidate);
        }
        if inner
            .inactive
            .complete_lineage(seq_hash, block_id)
            .is_none()
        {
            return Err(ExactReclaimNameError::NotCompleteInactiveEntry);
        }
        Ok(ExactReclaimEntryPlan::new(
            self.id,
            seq_hash,
            block_id,
            slot.generation,
        ))
    }

    /// Refresh named entries and combine their complete physical closures.
    ///
    /// This reads every entry, the reset capacity, and the backend removal
    /// proof under one store lock. The returned plan owns fresh private
    /// inactive identities in leaf-to-root order.
    pub(crate) fn refresh_and_combine_exact_reclaim(
        &self,
        entries: &[ExactReclaimEntryPlan],
    ) -> Result<FreshExactReclaimPlan, ExactReclaimRefreshError> {
        if entries.is_empty() {
            return Err(ExactReclaimRefreshError::EmptyPlan);
        }

        let inner = self.inner.lock();
        if !inner.inactive.supports_exact_reclaim() {
            return Err(ExactReclaimRefreshError::UnsupportedBackend);
        }
        if entries.iter().any(|entry| entry.manager_id != self.id) {
            return Err(ExactReclaimRefreshError::WrongManager);
        }

        let expected_reset_slots = inner.free.len();
        let mut physical_slots = HashMap::new();
        let mut victims = Vec::new();

        for (entry_index, entry) in entries.iter().enumerate() {
            let Some(leaf) = inner.slots.get(entry.leaf_block_id) else {
                return Err(ExactReclaimRefreshError::EntryUnavailable { index: entry_index });
            };
            match &leaf.state {
                SlotState::Inactive { seq_hash, .. }
                    if *seq_hash == entry.leaf_hash && leaf.generation == entry.leaf_generation => {
                }
                _ => {
                    return Err(ExactReclaimRefreshError::EntryUnavailable { index: entry_index });
                }
            }

            let Some(source_blocks) = inner
                .inactive
                .complete_lineage(entry.leaf_hash, entry.leaf_block_id)
            else {
                return Err(ExactReclaimRefreshError::EntryUnavailable { index: entry_index });
            };

            for (seq_hash, block_id) in source_blocks.into_iter().rev() {
                let Some(slot) = inner.slots.get(block_id) else {
                    return Err(ExactReclaimRefreshError::IncompleteCombinedPlan);
                };
                if !matches!(&slot.state, SlotState::Inactive { seq_hash: stored, .. } if *stored == seq_hash)
                {
                    return Err(ExactReclaimRefreshError::IncompleteCombinedPlan);
                }
                if let Some(&first) = physical_slots.get(&block_id) {
                    return Err(ExactReclaimRefreshError::SharedPhysicalSlot {
                        first,
                        second: entry_index,
                    });
                }
                physical_slots.insert(block_id, entry_index);
                victims.push(ExactInactiveVictim {
                    manager_id: self.id,
                    block_id,
                    seq_hash,
                    generation: slot.generation,
                    inactive_epoch: slot.inactive_epoch,
                });
            }
        }

        inner
            .inactive
            .preflight_exact_reclaim(&victims)
            .map_err(map_exact_reclaim_refresh_error)?;

        Ok(FreshExactReclaimPlan::new(
            self.id,
            expected_reset_slots,
            victims,
        ))
    }

    /// Atomically reclaim every supplied inactive victim and then allocate
    /// `count` mutable slots from the resulting reset capacity.
    ///
    /// This differs from the test-only exact inactive protocol. That protocol
    /// consumes exactly the inactive shortage. This transaction
    /// accepts a complete backend-validated removal plan that can reclaim more
    /// than the immediate allocation needs. It verifies the reset snapshot,
    /// every victim identity, and the backend removal order before it changes
    /// any pool or slot.
    pub(crate) fn allocate_exact_reclaim(
        self: &Arc<Self>,
        count: usize,
        expected_reset_slots: usize,
        victims: &[ExactInactiveVictim],
    ) -> Result<(Vec<MutableBlock<T>>, Vec<SequenceHash>), ExactAllocationError> {
        let mut inner = self.inner.lock();

        if inner.free.len() != expected_reset_slots {
            return Err(ExactAllocationError::ResetCapacityDrift {
                expected: expected_reset_slots,
                actual: inner.free.len(),
            });
        }
        if count == 0 && !victims.is_empty() {
            return Err(ExactAllocationError::ZeroCountWithVictims {
                supplied: victims.len(),
            });
        }

        let mut seen_block_ids = HashSet::with_capacity(victims.len());
        for victim in victims {
            if victim.manager_id != self.id {
                return Err(ExactAllocationError::WrongManager {
                    expected: self.id,
                    actual: victim.manager_id,
                });
            }
            if !seen_block_ids.insert(victim.block_id) {
                return Err(ExactAllocationError::DuplicateVictim {
                    block_id: victim.block_id,
                });
            }
        }

        let available = inner.free.len().saturating_add(victims.len());
        if available < count {
            return Err(ExactAllocationError::InsufficientCapacity {
                requested: count,
                available,
            });
        }

        // Validate every external identity before asking the backend to prove
        // the supplied removal order. The store lock closes all races with
        // resurrection and ordinary allocation until the later commit.
        for victim in victims {
            let Some(slot) = inner.slots.get(victim.block_id) else {
                return Err(ExactAllocationError::StaleVictim {
                    block_id: victim.block_id,
                });
            };
            match &slot.state {
                SlotState::Inactive { seq_hash, .. }
                    if *seq_hash == victim.seq_hash
                        && slot.generation == victim.generation
                        && slot.inactive_epoch == victim.inactive_epoch => {}
                SlotState::Inactive { .. } => {
                    return Err(ExactAllocationError::StaleVictim {
                        block_id: victim.block_id,
                    });
                }
                _ => {
                    return Err(ExactAllocationError::ActiveVictim {
                        block_id: victim.block_id,
                    });
                }
            }
        }

        inner
            .inactive
            .preflight_exact_reclaim(victims)
            .map_err(map_exact_reclaim_plan_error)?;

        // The backend preflight and all later removals share this store lock.
        // A successful preflight therefore makes every take below infallible.
        let mut handles = Vec::with_capacity(victims.len());
        let mut evicted = Vec::with_capacity(victims.len());
        for victim in victims {
            assert!(
                inner.inactive.take(victim.seq_hash, victim.block_id),
                "preflighted exact reclaim victim is absent from the inactive index"
            );
            let handle = take_inactive_handle(&mut inner.slots[victim.block_id], victim.block_id);
            inner.slots[victim.block_id].state = SlotState::Reset;
            inner.reset_on_release[victim.block_id] = self.default_reset_on_release;
            inner.free.push_back(victim.block_id);
            evicted.push(victim.seq_hash);
            handles.push(handle);
        }

        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            let block_id = inner
                .free
                .pop_front()
                .expect("validated exact reclaim capacity was lost under the store lock");
            let block_size = self.allocate_mutable_slot(&mut inner, block_id);
            blocks.push(MutableBlock::from_store(self.clone(), block_id, block_size));
        }

        let from_existing_reset = std::cmp::min(count, expected_reset_slots);
        self.metrics.inc_reset_pool_size_by(victims.len() as i64);
        self.metrics.dec_reset_pool_size_by(count as i64);
        self.metrics.dec_inactive_pool_size_by(victims.len() as i64);
        self.metrics.inc_inflight_mutable_by(count as i64);
        self.metrics.inc_evictions(victims.len() as u64);
        self.metrics.inc_allocations(count as u64);
        self.metrics
            .inc_allocations_from_reset(from_existing_reset as u64);

        drop(inner);
        // Registry attachment mutation takes its own lock. It must happen
        // after the store lock releases.
        for handle in handles {
            handle.mark_absent::<T>();
        }
        Ok((blocks, evicted))
    }
}

fn map_exact_reclaim_plan_error(error: ExactReclaimPlanError) -> ExactAllocationError {
    match error {
        ExactReclaimPlanError::Unsupported => ExactAllocationError::UnsupportedReclaimPlan,
        ExactReclaimPlanError::MissingVictim { block_id } => {
            ExactAllocationError::StaleVictim { block_id }
        }
        ExactReclaimPlanError::ParentBeforeChild {
            parent_block_id,
            child_block_id,
        } => ExactAllocationError::InvalidVictimOrder {
            parent_block_id,
            child_block_id,
        },
        ExactReclaimPlanError::OmittedLiveChild {
            parent_block_id,
            child_block_id,
        } => ExactAllocationError::IncompleteVictimSet {
            parent_block_id,
            child_block_id,
        },
        ExactReclaimPlanError::MissingRealAncestor {
            child_block_id,
            parent_block_id,
        } => ExactAllocationError::MissingVictimAncestor {
            child_block_id,
            parent_block_id,
        },
    }
}

fn map_exact_reclaim_refresh_error(error: ExactReclaimPlanError) -> ExactReclaimRefreshError {
    match error {
        ExactReclaimPlanError::Unsupported => ExactReclaimRefreshError::UnsupportedBackend,
        ExactReclaimPlanError::MissingVictim { .. }
        | ExactReclaimPlanError::ParentBeforeChild { .. }
        | ExactReclaimPlanError::OmittedLiveChild { .. }
        | ExactReclaimPlanError::MissingRealAncestor { .. } => {
            ExactReclaimRefreshError::IncompleteCombinedPlan
        }
    }
}
