// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G2 capacity request types.

/// The reservation policy required by a G2 destination allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum G2AllocationKind {
    /// A G1→G2 cache extension that can borrow idle reserve capacity.
    CacheExtension,
    /// A restore or remote pull that must reserve capacity before transfer.
    RequiredStaging,
}

/// The guarantee that a caller requires from a G2 reservation.
///
/// Compatibility requests use the legacy logical-manager allocation path.
/// They do not establish an exact-reclaim guarantee. Exact-reclaim requests
/// require a policy adapter that owns a physical reclaim transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum G2CapacityRequirement {
    /// Preserve the legacy best-effort manager behavior.
    Compatibility,
    /// Require an exact, policy-owned reclaim transaction.
    ExactReclaim,
}

/// One request for G2 destination capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct G2CapacityRequest {
    kind: G2AllocationKind,
    count: usize,
    requirement: G2CapacityRequirement,
}

impl G2CapacityRequest {
    /// Build a compatibility request for a legacy route.
    pub const fn compatibility(kind: G2AllocationKind, count: usize) -> Self {
        Self {
            kind,
            count,
            requirement: G2CapacityRequirement::Compatibility,
        }
    }

    /// Build a request that requires exact physical reclaim authority.
    pub const fn exact_reclaim(kind: G2AllocationKind, count: usize) -> Self {
        Self {
            kind,
            count,
            requirement: G2CapacityRequirement::ExactReclaim,
        }
    }

    /// Return the allocation policy intent.
    pub const fn kind(self) -> G2AllocationKind {
        self.kind
    }

    /// Return the requested block count.
    pub const fn count(self) -> usize {
        self.count
    }

    /// Return the required capacity guarantee.
    pub const fn requirement(self) -> G2CapacityRequirement {
        self.requirement
    }
}
