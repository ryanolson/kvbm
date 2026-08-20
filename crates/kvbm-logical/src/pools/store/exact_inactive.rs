// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-only exact inactive candidate and victim protocols.

use std::collections::HashSet;
use std::sync::Arc;

use crate::SequenceHash;
use crate::blocks::{BlockMetadata, MutableBlock};
use crate::pools::{
    ExactAllocationError, ExactInactiveVictim, ExactReclaimEntryPlan, ExactReclaimNameError,
    InactiveCandidate,
};

use super::{BlockStore, SlotState, take_inactive_handle};

impl<T: BlockMetadata> BlockStore<T> {
    /// Name one current inactive leaf without exposing its support slots.
    ///
    /// The retained name binds the physical leaf and mutable generation. It
    /// intentionally does not bind the inactive epoch. A later refresh can
    /// therefore accept an inactive-to-active-to-inactive ABA in the same
    /// mutable allocation tenure.
    pub(crate) fn name_complete_inactive_entry(
        &self,
        candidate: InactiveCandidate,
    ) -> Result<ExactReclaimEntryPlan, ExactReclaimNameError> {
        let inner = self.inner.lock();
        if !inner.inactive.supports_exact_reclaim() {
            return Err(ExactReclaimNameError::UnsupportedBackend);
        }

        let Some(slot) = inner.slots.get(candidate.block_id) else {
            return Err(ExactReclaimNameError::StaleCandidate);
        };
        match &slot.state {
            SlotState::Inactive { seq_hash, .. }
                if *seq_hash == candidate.seq_hash
                    && slot.generation == candidate.generation
                    && slot.inactive_epoch == candidate.inactive_epoch => {}
            _ => return Err(ExactReclaimNameError::StaleCandidate),
        }

        if inner
            .inactive
            .complete_lineage(candidate.seq_hash, candidate.block_id)
            .is_none()
        {
            return Err(ExactReclaimNameError::NotCompleteInactiveEntry);
        }

        Ok(ExactReclaimEntryPlan::new(
            self.id,
            candidate.seq_hash,
            candidate.block_id,
            candidate.generation,
        ))
    }

    /// Allocate `count` mutable blocks from reset slots and exact inactive
    /// victims only. The caller must supply one identity for every inactive
    /// slot that the request needs after reset capacity.
    pub(crate) fn allocate_exact_inactive(
        self: &Arc<Self>,
        count: usize,
        victims: &[ExactInactiveVictim],
    ) -> Result<(Vec<MutableBlock<T>>, Vec<SequenceHash>), ExactAllocationError> {
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

        let mut inner = self.inner.lock();
        let from_reset = std::cmp::min(count, inner.free.len());
        let from_inactive = count - from_reset;
        if victims.len() != from_inactive {
            return Err(ExactAllocationError::VictimCountMismatch {
                needed: from_inactive,
                supplied: victims.len(),
            });
        }
        if inner.inactive.len() < from_inactive {
            return Err(ExactAllocationError::VictimCountMismatch {
                needed: from_inactive,
                supplied: inner.inactive.len(),
            });
        }

        // Validate every external identity before mutating a pool or slot.
        // The store mutex closes the race with resurrection and ordinary
        // allocation, so a validated exact pair remains valid until commit.
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
                        && slot.inactive_epoch == victim.inactive_epoch =>
                {
                    if !inner.inactive.contains(victim.seq_hash, victim.block_id) {
                        return Err(ExactAllocationError::StaleVictim {
                            block_id: victim.block_id,
                        });
                    }
                }
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

        let mut blocks = Vec::with_capacity(count);
        for _ in 0..from_reset {
            let block_id = inner
                .free
                .pop_front()
                .expect("reset capacity was validated");
            let block_size = self.allocate_mutable_slot(&mut inner, block_id);
            blocks.push(MutableBlock::from_store(self.clone(), block_id, block_size));
        }

        let mut evicted = Vec::with_capacity(from_inactive);
        let mut handles = Vec::with_capacity(from_inactive);
        for victim in victims {
            assert!(
                inner.inactive.take(victim.seq_hash, victim.block_id),
                "validated inactive victim is absent from the index"
            );
            let handle = take_inactive_handle(&mut inner.slots[victim.block_id], victim.block_id);
            let block_size = self.allocate_mutable_slot(&mut inner, victim.block_id);
            blocks.push(MutableBlock::from_store(
                self.clone(),
                victim.block_id,
                block_size,
            ));
            evicted.push(victim.seq_hash);
            handles.push(handle);
        }

        self.metrics.dec_reset_pool_size_by(from_reset as i64);
        self.metrics.dec_inactive_pool_size_by(from_inactive as i64);
        self.metrics.inc_inflight_mutable_by(count as i64);
        self.metrics.inc_evictions(from_inactive as u64);
        self.metrics.inc_allocations(count as u64);
        self.metrics.inc_allocations_from_reset(from_reset as u64);

        drop(inner);
        // This obtains the registry attachments lock. Do it after the store
        // mutex releases to preserve the documented lock ordering.
        for handle in handles {
            handle.mark_absent::<T>();
        }
        Ok((blocks, evicted))
    }
}
