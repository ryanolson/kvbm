// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pending exact-reclaim ownership.

use super::{G2CapacityError, G2CapacityRequest, G2ExactAllocation};

/// Public summary of one adapter-owned reclaim plan.
///
/// Exact physical identities remain private in the completion owner.
///
/// ```compile_fail
/// use kvbm_engine::g2_capacity::G2ReclaimPlan;
///
/// fn expose_targets(plan: &G2ReclaimPlan) {
///     let _ = plan.targets();
/// }
/// ```
pub struct G2ReclaimPlan {
    request: G2CapacityRequest,
    target_count: usize,
}

/// A capacity claim that needs reclaim completion.
#[must_use = "if you drop this pending reclaim, it cancels its adapter-owned reservation"]
pub struct G2PendingReclaim {
    plan: G2ReclaimPlan,
    completion: Box<dyn G2PendingReclaimCompletion>,
}

/// Result of a pending reclaim completion.
pub enum G2ReclaimCompletion {
    /// The adapter acquired exact mutable destination slots.
    Granted(G2ExactAllocation),
    /// Reclaim needs another attempt with its owner and permit intact.
    PendingReclaim(G2PendingReclaim),
    /// The adapter rejected completion without capacity.
    Rejected(G2CapacityError),
}

/// A pending reclaim completion owned by one source capacity adapter.
pub trait G2PendingReclaimCompletion: Send {
    /// Complete this pending reclaim transaction.
    fn complete(self: Box<Self>) -> G2ReclaimCompletion;
}

impl G2ReclaimPlan {
    /// Construct a public summary of an adapter-owned reclaim plan.
    pub const fn new(request: G2CapacityRequest, target_count: usize) -> Self {
        Self {
            request,
            target_count,
        }
    }

    /// Return the request that produced this plan.
    pub const fn request(&self) -> G2CapacityRequest {
        self.request
    }

    /// Return the number of private exact targets.
    pub const fn target_count(&self) -> usize {
        self.target_count
    }
}

impl G2PendingReclaim {
    /// Construct a pending reclaim with its source adapter capability.
    pub fn new(plan: G2ReclaimPlan, completion: Box<dyn G2PendingReclaimCompletion>) -> Self {
        Self { plan, completion }
    }

    /// Inspect the public reclaim plan summary.
    pub const fn plan(&self) -> &G2ReclaimPlan {
        &self.plan
    }

    /// Consume the claim and run its completion step.
    pub fn complete(self) -> G2ReclaimCompletion {
        self.completion.complete()
    }
}
