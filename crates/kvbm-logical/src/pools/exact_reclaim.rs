// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opaque entry names and fresh snapshots for exact inactive reclaim.

use std::fmt;

use crate::{BlockId, ManagerId, SequenceHash};

use super::{ExactAllocationError, ExactInactiveVictim};

/// A retained name for one complete inactive cache entry.
///
/// The name binds its original leaf slot and mutable allocation generation.
/// It does not bind the inactive epoch. A cache hit can therefore end and
/// restart the inactive tenure before [`BlockManager::refresh_and_combine_exact_reclaim`](crate::BlockManager::refresh_and_combine_exact_reclaim).
/// A later mutable allocation, even with the same hash, invalidates this name.
///
/// ```compile_fail
/// let entry: kvbm_logical::ExactReclaimEntryPlan = unreachable!();
/// let _ = entry.leaf_block_id;
/// ```
#[must_use]
#[derive(Clone, PartialEq, Eq)]
pub struct ExactReclaimEntryPlan {
    pub(crate) manager_id: ManagerId,
    pub(crate) leaf_hash: SequenceHash,
    pub(crate) leaf_block_id: BlockId,
    pub(crate) leaf_generation: u64,
}

impl ExactReclaimEntryPlan {
    pub(crate) const fn new(
        manager_id: ManagerId,
        leaf_hash: SequenceHash,
        leaf_block_id: BlockId,
        leaf_generation: u64,
    ) -> Self {
        Self {
            manager_id,
            leaf_hash,
            leaf_block_id,
            leaf_generation,
        }
    }
}

impl fmt::Debug for ExactReclaimEntryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ExactReclaimEntryPlan(..)")
    }
}

/// A one-use exact-reclaim snapshot.
///
/// A manager creates this value by resolving one or more retained entry
/// names under its store lock. The private identities include the current
/// inactive epochs. The executor consumes this value.
///
/// ```compile_fail
/// let plan: kvbm_logical::FreshExactReclaimPlan = unreachable!();
/// let _ = plan.victims_leaf_to_root;
/// ```
///
/// Raw victims are private to `kvbm-logical`.
///
/// ```compile_fail
/// fn expose_raw_victim(_: kvbm_logical::ExactInactiveVictim) {}
/// ```
///
/// Raw exact executors are also private to `kvbm-logical`.
///
/// ```compile_fail
/// fn execute_raw<T: kvbm_logical::BlockMetadata + Sync>(
///     manager: &kvbm_logical::BlockManager<T>,
/// ) {
///     let _ = manager.allocate_blocks_with_exact_reclaim(0, 0, &[]);
/// }
/// ```
#[must_use]
pub struct FreshExactReclaimPlan {
    pub(crate) manager_id: ManagerId,
    pub(crate) expected_reset_slots: usize,
    pub(crate) victims_leaf_to_root: Vec<ExactInactiveVictim>,
}

impl FreshExactReclaimPlan {
    pub(crate) const fn new(
        manager_id: ManagerId,
        expected_reset_slots: usize,
        victims_leaf_to_root: Vec<ExactInactiveVictim>,
    ) -> Self {
        Self {
            manager_id,
            expected_reset_slots,
            victims_leaf_to_root,
        }
    }

    /// Return the number of physical slots that this plan reclaims.
    pub fn reclaimed_blocks(&self) -> usize {
        self.victims_leaf_to_root.len()
    }
}

impl fmt::Debug for FreshExactReclaimPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FreshExactReclaimPlan")
            .field("reclaimed_blocks", &self.reclaimed_blocks())
            .finish_non_exhaustive()
    }
}

/// Rejection while naming one inactive entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactReclaimNameError {
    /// The backend cannot resolve complete inactive lineages.
    UnsupportedBackend,
    /// The candidate changed before the name operation acquired the store lock.
    StaleCandidate,
    /// The candidate does not name a complete inactive lineage.
    NotCompleteInactiveEntry,
}

impl fmt::Display for ExactReclaimNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBackend => {
                f.write_str("inactive backend does not support exact reclaim")
            }
            Self::StaleCandidate => {
                f.write_str("inactive candidate changed before exact reclaim naming")
            }
            Self::NotCompleteInactiveEntry => {
                f.write_str("inactive candidate does not name a complete cache entry")
            }
        }
    }
}

impl std::error::Error for ExactReclaimNameError {}

/// Rejection while refreshing and combining retained entry names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactReclaimRefreshError {
    /// Exact reclaim needs at least one named entry.
    EmptyPlan,
    /// A retained entry belongs to another manager.
    WrongManager,
    /// The entry at `index` no longer has its original leaf registration.
    EntryUnavailable { index: usize },
    /// Two entries resolve to the same physical slot.
    SharedPhysicalSlot { first: usize, second: usize },
    /// The combined entries omit a live child or required ancestor.
    IncompleteCombinedPlan,
    /// The backend cannot resolve complete inactive lineages.
    UnsupportedBackend,
}

impl fmt::Display for ExactReclaimRefreshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPlan => f.write_str("exact reclaim needs at least one entry"),
            Self::WrongManager => f.write_str("exact reclaim entry belongs to another manager"),
            Self::EntryUnavailable { index } => {
                write!(
                    f,
                    "exact reclaim entry {index} no longer has its original leaf registration"
                )
            }
            Self::SharedPhysicalSlot { first, second } => {
                write!(
                    f,
                    "exact reclaim entries {first} and {second} share a physical slot"
                )
            }
            Self::IncompleteCombinedPlan => {
                f.write_str("combined exact reclaim entries do not form a complete removal plan")
            }
            Self::UnsupportedBackend => {
                f.write_str("inactive backend does not support exact reclaim")
            }
        }
    }
}

impl std::error::Error for ExactReclaimRefreshError {}

/// Rejection while executing a fresh exact-reclaim snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactReclaimExecuteError {
    /// The fresh plan belongs to another manager.
    WrongManager,
    /// Reset capacity changed after refresh.
    ResetCapacityDrift,
    /// A physical identity or lineage changed after refresh.
    StalePlan,
    /// The reset capacity and reclaim plan cannot satisfy the request.
    InsufficientCapacity,
    /// The allocation request cannot use this non-empty plan.
    InvalidRequest,
    /// The backend no longer supports exact reclaim.
    UnsupportedBackend,
}

impl fmt::Display for ExactReclaimExecuteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongManager => {
                f.write_str("fresh exact reclaim plan belongs to another manager")
            }
            Self::ResetCapacityDrift => {
                f.write_str("reset capacity changed after exact reclaim refresh")
            }
            Self::StalePlan => f.write_str("fresh exact reclaim plan changed before execution"),
            Self::InsufficientCapacity => {
                f.write_str("exact reclaim plan cannot satisfy the allocation request")
            }
            Self::InvalidRequest => {
                f.write_str("allocation request cannot use this exact reclaim plan")
            }
            Self::UnsupportedBackend => {
                f.write_str("inactive backend does not support exact reclaim")
            }
        }
    }
}

impl std::error::Error for ExactReclaimExecuteError {}

impl From<ExactAllocationError> for ExactReclaimExecuteError {
    fn from(error: ExactAllocationError) -> Self {
        match error {
            ExactAllocationError::WrongManager { .. } => Self::WrongManager,
            ExactAllocationError::ResetCapacityDrift { .. } => Self::ResetCapacityDrift,
            ExactAllocationError::InsufficientCapacity { .. } => Self::InsufficientCapacity,
            ExactAllocationError::ZeroCountWithVictims { .. } => Self::InvalidRequest,
            ExactAllocationError::UnsupportedReclaimPlan => Self::UnsupportedBackend,
            #[cfg(test)]
            ExactAllocationError::VictimCountMismatch { .. } => Self::StalePlan,
            ExactAllocationError::DuplicateVictim { .. }
            | ExactAllocationError::StaleVictim { .. }
            | ExactAllocationError::ActiveVictim { .. }
            | ExactAllocationError::InvalidVictimOrder { .. }
            | ExactAllocationError::IncompleteVictimSet { .. }
            | ExactAllocationError::MissingVictimAncestor { .. } => Self::StalePlan,
        }
    }
}
