// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capacity ownership for G2 destination allocations.
//!
//! The facade separates G2 admission from the logical block manager. A
//! capacity implementation receives an explicit allocation intent and holds a
//! lease until its staged blocks become registered or drop.

mod allocation;
mod capacity;
mod policy_route;
mod reclaim;
mod request;

pub use allocation::{
    G2Allocation, G2CacheExtensionResidencyLease, G2ExactAllocation, G2ExactRegistrationOwner,
    G2ExactStagedAllocation, G2RegisteredAllocation, G2StagedAllocation,
};
pub(crate) use allocation::{
    RequiredStagingAllocation, RequiredStagingPublishedAllocation, RequiredStagingStagedAllocation,
    reserve_required_staging,
};
pub use capacity::{
    DirectG2Capacity, G2Capacity, G2CapacityDecision, G2CapacityError, G2CapacitySet,
    direct_g2_capacity, direct_g2_capacity_set, reserve_compatibility,
};
pub use policy_route::{
    PolicyCancelDisposition, PolicyG1G2BoundRoute, PolicyG1G2CancelHandle, PolicyG1G2Completion,
    PolicyG1G2Execution, PolicyG1G2ExecutionError, PolicyG1G2Installation, PolicyG1G2Reservation,
    PolicyG1G2Route, PolicyG1G2SourceSettlement, PolicyG1G2SubmitError,
    PolicyG1G2ValidatedInstallation, PolicyG1SourceMetadata, PolicyPhysicalCompletion,
    PolicyPhysicalTerminal,
};
#[cfg(test)]
pub(crate) use policy_route::{PolicyG1G2TransferExecutor, PolicyG1G2TransferReceipt};
pub use reclaim::{
    G2PendingReclaim, G2PendingReclaimCompletion, G2ReclaimCompletion, G2ReclaimPlan,
};
pub use request::{G2AllocationKind, G2CapacityRequest, G2CapacityRequirement};

/// Guard held for a compatibility allocation and its staged successor.
pub(crate) trait G2LeaseGuard: Send + Sync {}

impl G2LeaseGuard for () {}

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
