// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use kvbm_engine::G2;
use kvbm_engine::g2_capacity::{
    G2AllocationKind, G2CapacityError, G2CapacityRequest, G2CapacityRequirement, G2ExactAllocation,
    G2ExactRegistrationOwner, G2PendingReclaim, G2PendingReclaimCompletion, G2ReclaimCompletion,
    G2ReclaimPlan,
};
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock};

struct ExternalOwner;
struct ExternalCompletion;

impl G2ExactRegistrationOwner for ExternalOwner {
    fn register_blocks(
        &mut self,
        _blocks: Vec<CompleteBlock<G2>>,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        Err(G2CapacityError::Rejected(
            "external SPI compile fixture".to_string(),
        ))
    }

    fn rollback_required_staging(&mut self, blocks: Vec<ImmutableBlock<G2>>) {
        for block in &blocks {
            block.set_evict_on_reset(true);
        }
        drop(blocks);
    }

    fn retain_cache_extension(self: Box<Self>) {}
}

impl G2PendingReclaimCompletion for ExternalCompletion {
    fn complete(self: Box<Self>) -> G2ReclaimCompletion {
        G2ReclaimCompletion::Rejected(G2CapacityError::Rejected(
            "external pending SPI compile fixture".to_string(),
        ))
    }
}

#[test]
fn external_adapter_can_bind_an_exact_registration_owner() {
    let request = G2CapacityRequest::exact_reclaim(G2AllocationKind::RequiredStaging, 0);

    let allocation = G2ExactAllocation::new(request, Vec::new(), Box::new(ExternalOwner))
        .expect("external adapter allocation");

    assert_eq!(
        allocation.requirement(),
        G2CapacityRequirement::ExactReclaim
    );
}

#[test]
fn external_adapter_can_bind_a_pending_completion_owner() {
    let request = G2CapacityRequest::exact_reclaim(G2AllocationKind::RequiredStaging, 2);
    let pending =
        G2PendingReclaim::new(G2ReclaimPlan::new(request, 3), Box::new(ExternalCompletion));

    assert_eq!(pending.plan().request(), request);
    assert_eq!(pending.plan().target_count(), 3);
}
