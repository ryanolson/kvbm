// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G2 allocation and registration ownership.

mod compatibility;
mod exact;
mod required_staging;
mod storage;

pub use compatibility::{G2Allocation, G2StagedAllocation};
pub(crate) use exact::G2ExactPublishedRequiredStaging;
pub use exact::{
    G2CacheExtensionResidencyLease, G2ExactAllocation, G2ExactRegistrationOwner,
    G2ExactStagedAllocation, G2RegisteredAllocation,
};
pub(crate) use required_staging::{
    RequiredStagingAllocation, RequiredStagingPublishedAllocation, RequiredStagingStagedAllocation,
    reserve_required_staging,
};
