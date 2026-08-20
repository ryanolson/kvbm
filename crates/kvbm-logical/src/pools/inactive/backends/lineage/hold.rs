// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic complete-lineage extraction for pressure ownership.

use std::collections::HashMap;

use crate::pools::store::ExactReclaimPlanError;
use crate::{BlockId, ExactInactiveVictim, SequenceHash};

use super::{LineageBackend, SlotData};

type CompleteLineage = (Vec<u32>, Vec<(SequenceHash, BlockId)>);

impl LineageBackend {
    /// Validate one supplied leaf-to-root removal plan without mutation.
    ///
    /// Each selected parent must list every currently live child earlier in
    /// the plan. This makes every later removal a leaf operation and prevents
    /// an omitted child from becoming orphaned by its parent removal.
    pub(super) fn preflight_exact_reclaim(
        &self,
        victims: &[ExactInactiveVictim],
    ) -> Result<(), ExactReclaimPlanError> {
        let mut plan_positions = HashMap::with_capacity(victims.len());
        let mut plan_indices = Vec::with_capacity(victims.len());

        for (position, victim) in victims.iter().enumerate() {
            let index = self.real_index(victim.seq_hash, victim.block_id).ok_or(
                ExactReclaimPlanError::MissingVictim {
                    block_id: victim.block_id,
                },
            )?;
            if plan_positions.insert(index, position).is_some() {
                return Err(ExactReclaimPlanError::MissingVictim {
                    block_id: victim.block_id,
                });
            }
            plan_indices.push(index);
        }

        for (position, (&index, victim)) in plan_indices.iter().zip(victims).enumerate() {
            self.validate_nearest_real_ancestor(index, victim.block_id, position, &plan_positions)?;

            let mut child = self.slots[index as usize].first_child;
            while let Some(child_index) = child {
                match self.slots[child_index as usize].data {
                    SlotData::Real { block_id, .. } => {
                        let Some(&child_position) = plan_positions.get(&child_index) else {
                            return Err(ExactReclaimPlanError::OmittedLiveChild {
                                parent_block_id: victim.block_id,
                                child_block_id: block_id,
                            });
                        };
                        if child_position > position {
                            return Err(ExactReclaimPlanError::ParentBeforeChild {
                                parent_block_id: victim.block_id,
                                child_block_id: block_id,
                            });
                        }
                    }
                    SlotData::Ghost => {
                        self.validate_ghost_children(
                            child_index,
                            victim.block_id,
                            position,
                            &plan_positions,
                        )?;
                    }
                    SlotData::Free => unreachable!("a live child list cannot point to a free slot"),
                }
                child = self.slots[child_index as usize].next_sibling;
            }
        }

        Ok(())
    }

    fn validate_nearest_real_ancestor(
        &self,
        index: u32,
        child_block_id: BlockId,
        child_position: usize,
        plan_positions: &HashMap<u32, usize>,
    ) -> Result<(), ExactReclaimPlanError> {
        let mut parent = self.slots[index as usize].parent;
        while let Some(parent_index) = parent {
            match self.slots[parent_index as usize].data {
                SlotData::Real { block_id, .. } => {
                    let Some(&parent_position) = plan_positions.get(&parent_index) else {
                        return Err(ExactReclaimPlanError::MissingRealAncestor {
                            child_block_id,
                            parent_block_id: block_id,
                        });
                    };
                    if parent_position < child_position {
                        return Err(ExactReclaimPlanError::ParentBeforeChild {
                            parent_block_id: block_id,
                            child_block_id,
                        });
                    }
                    return Ok(());
                }
                SlotData::Ghost => {}
                SlotData::Free => {
                    unreachable!("a lineage ancestor cannot be a free slot")
                }
            }
            parent = self.slots[parent_index as usize].parent;
        }
        Ok(())
    }

    fn validate_ghost_children(
        &self,
        ghost_index: u32,
        parent_block_id: BlockId,
        parent_position: usize,
        plan_positions: &HashMap<u32, usize>,
    ) -> Result<(), ExactReclaimPlanError> {
        let mut pending = vec![ghost_index];
        while let Some(index) = pending.pop() {
            match self.slots[index as usize].data {
                SlotData::Real { block_id, .. } => {
                    let Some(&child_position) = plan_positions.get(&index) else {
                        return Err(ExactReclaimPlanError::OmittedLiveChild {
                            parent_block_id,
                            child_block_id: block_id,
                        });
                    };
                    if child_position > parent_position {
                        return Err(ExactReclaimPlanError::ParentBeforeChild {
                            parent_block_id,
                            child_block_id: block_id,
                        });
                    }
                }
                SlotData::Ghost => {
                    let mut child = self.slots[index as usize].first_child;
                    while let Some(child_index) = child {
                        pending.push(child_index);
                        child = self.slots[child_index as usize].next_sibling;
                    }
                }
                SlotData::Free => unreachable!("a live child list cannot point to a free slot"),
            }
        }
        Ok(())
    }

    /// Read one exact inactive leaf and every real ancestor through root.
    ///
    /// This performs the complete validation used by extraction, but it does
    /// not change the graph or its eviction policy state.
    pub(super) fn exact_complete_lineage(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
    ) -> Option<Vec<(SequenceHash, BlockId)>> {
        self.complete_lineage_indices(seq_hash, block_id)
            .map(|(_, source_blocks)| source_blocks)
    }

    /// Remove one exact inactive leaf and every real ancestor through root.
    /// Validation finishes before the first graph mutation.
    pub(super) fn take_exact_complete_lineage(
        &mut self,
        seq_hash: SequenceHash,
        block_id: BlockId,
    ) -> Option<Vec<(SequenceHash, BlockId)>> {
        let (indices_leaf_first, source_blocks) =
            self.complete_lineage_indices(seq_hash, block_id)?;

        for index in indices_leaf_first {
            self.remove_node_at(index);
        }
        Some(source_blocks)
    }

    fn complete_lineage_indices(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
    ) -> Option<CompleteLineage> {
        let position = seq_hash.position();
        let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
        let target = *self.index.get(&(position, fragment))?;
        let target_slot = &self.slots[target as usize];
        if !target_slot.is_leaf() {
            return None;
        }
        match target_slot.data {
            SlotData::Real {
                seq_hash: stored_hash,
                block_id: stored_id,
            } if stored_hash == seq_hash && stored_id == block_id => {}
            _ => return None,
        }

        let mut indices_leaf_first = Vec::with_capacity(position as usize + 1);
        let mut current = Some(target);
        while let Some(index) = current {
            let slot = &self.slots[index as usize];
            if !matches!(slot.data, SlotData::Real { .. }) {
                return None;
            }
            indices_leaf_first.push(index);
            current = slot.parent;
            if current.is_none() && slot.position != 0 {
                return None;
            }
        }

        let source_blocks = indices_leaf_first
            .iter()
            .rev()
            .map(|&index| match self.slots[index as usize].data {
                SlotData::Real { seq_hash, block_id } => (seq_hash, block_id),
                _ => unreachable!("complete lineage validation accepted a non-real node"),
            })
            .collect::<Vec<_>>();

        Some((indices_leaf_first, source_blocks))
    }

    fn real_index(&self, seq_hash: SequenceHash, block_id: BlockId) -> Option<u32> {
        let position = seq_hash.position();
        let fragment = seq_hash.parent_fragment_for_child_position(position + 1);
        let index = *self.index.get(&(position, fragment))?;
        match self.slots[index as usize].data {
            SlotData::Real {
                seq_hash: stored_hash,
                block_id: stored_id,
            } if stored_hash == seq_hash && stored_id == block_id => Some(index),
            _ => None,
        }
    }
}
