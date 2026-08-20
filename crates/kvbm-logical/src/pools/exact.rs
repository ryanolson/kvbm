// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generation-bound identities for exact inactive reclaim.

use std::fmt;

use crate::{BlockId, ManagerId, SequenceHash};

#[cfg(test)]
use super::InactiveCandidate;

/// One caller-authorized inactive victim for an exact allocation.
///
/// All fields identify one inactive tenure. A later mutable allocation changes
/// `generation`. A cache-hit resurrection changes `inactive_epoch`. Either
/// transition invalidates this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExactInactiveVictim {
    /// Manager that owns the physical slot.
    pub(crate) manager_id: ManagerId,
    /// Physical slot in the manager.
    pub(crate) block_id: BlockId,
    /// Hash registered in the inactive slot.
    pub(crate) seq_hash: SequenceHash,
    /// Mutable-allocation tenure of the physical slot.
    pub(crate) generation: u64,
    /// Inactive residency tenure of the physical slot.
    pub(crate) inactive_epoch: u64,
}

#[cfg(test)]
impl ExactInactiveVictim {
    /// Create an exact victim identity from an inactive candidate.
    pub(crate) const fn from_candidate(
        manager_id: ManagerId,
        candidate: InactiveCandidate,
    ) -> Self {
        Self {
            manager_id,
            block_id: candidate.block_id,
            seq_hash: candidate.seq_hash,
            generation: candidate.generation,
            inactive_epoch: candidate.inactive_epoch,
        }
    }
}

#[cfg(test)]
impl InactiveCandidate {
    /// Bind this snapshot to its manager for exact inactive reclaim.
    pub(crate) const fn exact_victim(self, manager_id: ManagerId) -> ExactInactiveVictim {
        ExactInactiveVictim::from_candidate(manager_id, self)
    }
}

/// Rejection for an exact inactive allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExactAllocationError {
    /// A victim belongs to another manager.
    WrongManager {
        expected: ManagerId,
        actual: ManagerId,
    },
    /// More than one requested victim names the same slot.
    DuplicateVictim { block_id: BlockId },
    /// The supplied victim count does not match the inactive count needed.
    #[cfg(test)]
    VictimCountMismatch { needed: usize, supplied: usize },
    /// The slot still exists, but its inactive hash or generation differs.
    StaleVictim { block_id: BlockId },
    /// The requested slot is no longer inactive.
    ActiveVictim { block_id: BlockId },
    /// The reset-pool snapshot changed before the transaction began.
    ResetCapacityDrift { expected: usize, actual: usize },
    /// Reset capacity plus the complete reclaim plan cannot satisfy the request.
    InsufficientCapacity { requested: usize, available: usize },
    /// A zero-count allocation cannot include eviction victims.
    ZeroCountWithVictims { supplied: usize },
    /// The inactive backend cannot validate an ordered reclaim plan.
    UnsupportedReclaimPlan,
    /// A parent appears before one of its live children in the reclaim plan.
    InvalidVictimOrder {
        parent_block_id: BlockId,
        child_block_id: BlockId,
    },
    /// The reclaim plan omits a live child of a selected parent.
    IncompleteVictimSet {
        parent_block_id: BlockId,
        child_block_id: BlockId,
    },
    /// The reclaim plan omits a real ancestor of a selected victim.
    MissingVictimAncestor {
        child_block_id: BlockId,
        parent_block_id: BlockId,
    },
}

impl fmt::Display for ExactAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongManager { expected, actual } => {
                write!(f, "victim manager {actual:?} does not match {expected:?}")
            }
            Self::DuplicateVictim { block_id } => {
                write!(f, "victim slot {block_id} appears more than once")
            }
            #[cfg(test)]
            Self::VictimCountMismatch { needed, supplied } => {
                write!(
                    f,
                    "exact allocation needs {needed} victims but received {supplied}"
                )
            }
            Self::StaleVictim { block_id } => {
                write!(f, "victim slot {block_id} has a stale identity")
            }
            Self::ActiveVictim { block_id } => {
                write!(f, "victim slot {block_id} is not inactive")
            }
            Self::ResetCapacityDrift { expected, actual } => write!(
                f,
                "reset capacity changed from {expected} to {actual} before exact reclaim"
            ),
            Self::InsufficientCapacity {
                requested,
                available,
            } => write!(
                f,
                "exact reclaim needs {requested} destination slots but only {available} are authorized"
            ),
            Self::ZeroCountWithVictims { supplied } => write!(
                f,
                "zero-count exact reclaim cannot consume {supplied} eviction victims"
            ),
            Self::UnsupportedReclaimPlan => {
                f.write_str("inactive backend does not support ordered exact reclaim")
            }
            Self::InvalidVictimOrder {
                parent_block_id,
                child_block_id,
            } => write!(
                f,
                "parent slot {parent_block_id} appears before child slot {child_block_id}"
            ),
            Self::IncompleteVictimSet {
                parent_block_id,
                child_block_id,
            } => write!(
                f,
                "reclaim plan omits live child slot {child_block_id} of parent slot {parent_block_id}"
            ),
            Self::MissingVictimAncestor {
                child_block_id,
                parent_block_id,
            } => write!(
                f,
                "reclaim plan omits parent slot {parent_block_id} for child slot {child_block_id}"
            ),
        }
    }
}

impl std::error::Error for ExactAllocationError {}
