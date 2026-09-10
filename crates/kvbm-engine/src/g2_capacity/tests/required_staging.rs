// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::executor::block_on;
use kvbm_logical::blocks::ImmutableBlock;

use super::*;
use crate::g2_capacity::test_support::RecordingG2Capacity;

struct PendingGrantCapacity {
    inner: ExactRegistrationCapacity,
}

struct RepeatedPendingCapacity {
    manager: Arc<BlockManager<G2>>,
    attempts: Arc<AtomicUsize>,
    permit_drops: Arc<AtomicUsize>,
}

struct FaultyExactCapacity {
    inner: ExactRegistrationCapacity,
    granted_kind: G2AllocationKind,
    granted_count: Option<usize>,
    pending: bool,
}

struct FaultyCompatibilityCapacity {
    manager: Arc<BlockManager<G2>>,
    granted_kind: G2AllocationKind,
    granted_count: Option<usize>,
}

impl G2Capacity for PendingGrantCapacity {
    fn reserve(&self, request: G2CapacityRequest) -> Result<G2CapacityDecision, G2CapacityError> {
        let allocation = expect_exact_allocation(self.inner.reserve(request)?);
        Ok(G2CapacityDecision::PendingReclaim(G2PendingReclaim::new(
            G2ReclaimPlan::new(request, 1),
            Box::new(GrantedCompletion { allocation }),
        )))
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn manager_id(&self) -> kvbm_logical::ManagerId {
        self.inner.manager_id()
    }

    fn register_compatibility(
        &self,
        allocation: G2StagedAllocation,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        self.inner.register_compatibility(allocation)
    }

    fn match_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.inner.match_blocks(hashes)
    }

    fn match_inactive_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.inner.match_inactive_blocks(hashes)
    }

    fn has_any_registered_hashes(&self, hashes: &[SequenceHash]) -> bool {
        self.inner.has_any_registered_hashes(hashes)
    }

    fn scan_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<G2>> {
        self.inner.scan_matches(hashes, touch)
    }
}

impl G2Capacity for RepeatedPendingCapacity {
    fn reserve(&self, request: G2CapacityRequest) -> Result<G2CapacityDecision, G2CapacityError> {
        Ok(G2CapacityDecision::PendingReclaim(G2PendingReclaim::new(
            G2ReclaimPlan::new(request, 1),
            Box::new(RetryCompletion {
                request,
                attempts: Arc::clone(&self.attempts),
                permit: DropGuard(Arc::clone(&self.permit_drops)),
            }),
        )))
    }

    fn block_size(&self) -> usize {
        self.manager.block_size()
    }

    fn manager_id(&self) -> kvbm_logical::ManagerId {
        self.manager.id()
    }

    fn register_compatibility(
        &self,
        _allocation: G2StagedAllocation,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        Err(G2CapacityError::Rejected(
            "repeated-pending capacity has no compatibility route".to_string(),
        ))
    }

    fn match_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.manager.match_blocks(hashes)
    }

    fn match_inactive_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.manager.match_inactive_blocks(hashes)
    }

    fn has_any_registered_hashes(&self, hashes: &[SequenceHash]) -> bool {
        self.manager.has_any_registered_hashes(hashes)
    }

    fn scan_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<G2>> {
        self.manager.scan_matches(hashes, touch)
    }
}

impl G2Capacity for FaultyExactCapacity {
    fn reserve(&self, request: G2CapacityRequest) -> Result<G2CapacityDecision, G2CapacityError> {
        let allocation =
            expect_exact_allocation(self.inner.reserve(G2CapacityRequest::exact_reclaim(
                self.granted_kind,
                self.granted_count.unwrap_or(request.count()),
            ))?);
        if self.pending {
            Ok(G2CapacityDecision::PendingReclaim(G2PendingReclaim::new(
                G2ReclaimPlan::new(request, 1),
                Box::new(GrantedCompletion { allocation }),
            )))
        } else {
            Ok(G2CapacityDecision::ExactGranted(allocation))
        }
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn manager_id(&self) -> kvbm_logical::ManagerId {
        self.inner.manager_id()
    }

    fn register_compatibility(
        &self,
        allocation: G2StagedAllocation,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        self.inner.register_compatibility(allocation)
    }

    fn match_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.inner.match_blocks(hashes)
    }

    fn match_inactive_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.inner.match_inactive_blocks(hashes)
    }

    fn has_any_registered_hashes(&self, hashes: &[SequenceHash]) -> bool {
        self.inner.has_any_registered_hashes(hashes)
    }

    fn scan_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<G2>> {
        self.inner.scan_matches(hashes, touch)
    }
}

impl G2Capacity for FaultyCompatibilityCapacity {
    fn reserve(&self, request: G2CapacityRequest) -> Result<G2CapacityDecision, G2CapacityError> {
        if request.requirement() == G2CapacityRequirement::ExactReclaim {
            return Err(G2CapacityError::ExactReclaimUnsupported(request));
        }
        self.manager
            .allocate_blocks(self.granted_count.unwrap_or(request.count()))
            .map(|blocks| {
                G2CapacityDecision::Granted(G2Allocation::direct(self.granted_kind, blocks))
            })
            .ok_or(G2CapacityError::Unavailable(request))
    }

    fn block_size(&self) -> usize {
        self.manager.block_size()
    }

    fn manager_id(&self) -> kvbm_logical::ManagerId {
        self.manager.id()
    }

    fn register_compatibility(
        &self,
        allocation: G2StagedAllocation,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        allocation
            .register_with(|blocks| self.manager.try_register_blocks(blocks))
            .map_err(|_| G2CapacityError::Rejected("foreign compatibility allocation".into()))
    }

    fn match_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.manager.match_blocks(hashes)
    }

    fn match_inactive_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.manager.match_inactive_blocks(hashes)
    }

    fn has_any_registered_hashes(&self, hashes: &[SequenceHash]) -> bool {
        self.manager.has_any_registered_hashes(hashes)
    }

    fn scan_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<G2>> {
        self.manager.scan_matches(hashes, touch)
    }
}

#[test]
fn required_staging_falls_back_only_when_exact_reclaim_is_unsupported() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .build(),
    );
    let capacity = Arc::new(DirectG2Capacity::new(Arc::clone(&manager)));

    let allocation = reserve_required_staging(capacity, 2).expect("compatibility reservation");

    assert_eq!(
        allocation.requirement(),
        G2CapacityRequirement::Compatibility
    );
    assert_eq!(allocation.len(), 2);
    assert_eq!(manager.available_blocks(), 0);

    drop(allocation);

    assert_eq!(manager.available_blocks(), 2);
}

#[test]
fn required_staging_preserves_the_exact_owner_through_transfer_and_publication() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .build(),
    );
    let capacity = Arc::new(ExactRegistrationCapacity::new(Arc::clone(&manager)));
    let hashes = [
        SequenceHash::new(81, None, 0),
        SequenceHash::new(82, None, 1),
    ];

    let allocation =
        reserve_required_staging(capacity.clone(), hashes.len()).expect("exact reservation");
    assert_eq!(
        allocation.requirement(),
        G2CapacityRequirement::ExactReclaim
    );
    assert_eq!(
        capacity.requests(),
        vec![G2CapacityRequest::exact_reclaim(
            G2AllocationKind::RequiredStaging,
            hashes.len(),
        )]
    );

    let allocation = block_on(
        allocation.transfer_with(|blocks| async move { Ok::<_, std::convert::Infallible>(blocks) }),
    )
    .expect("identity transfer");
    let staged = allocation
        .stage_all(&hashes, capacity.block_size())
        .expect("stage exact allocation");

    assert_eq!(staged.block_ids().len(), hashes.len());
    assert_eq!(capacity.exact_registration_count(), 0);
    assert_eq!(capacity.exact_permit_drop_count(), 0);

    let registered = staged.publish().expect("source-bound publication");

    assert_eq!(registered.len(), hashes.len());
    assert_eq!(capacity.exact_registration_count(), 1);
    assert_eq!(capacity.exact_permit_drop_count(), 1);
    assert_eq!(
        capacity.compatibility_registrations.load(Ordering::Relaxed),
        0
    );
}

#[test]
fn unregistered_required_staging_rolls_back_on_drop() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let capacity = Arc::new(ExactRegistrationCapacity::new(Arc::clone(&manager)));
    let hash = SequenceHash::new(83, None, 0);
    let staged = reserve_required_staging(capacity.clone(), 1)
        .expect("exact reservation")
        .stage_all(&[hash], capacity.block_size())
        .expect("stage exact allocation");

    assert_eq!(staged.block_ids().len(), 1);
    assert_eq!(manager.available_blocks(), 0);
    assert!(manager.match_blocks(&[hash]).is_empty());

    drop(staged);

    assert_eq!(manager.available_blocks(), 1);
    assert!(manager.match_blocks(&[hash]).is_empty());
    assert_eq!(capacity.exact_registration_count(), 0);
    assert_eq!(capacity.exact_permit_drop_count(), 1);
}

#[test]
fn compatibility_publication_uses_the_capacity_bound_at_reservation() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let foreign_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let capacity = Arc::new(RecordingG2Capacity::new(Arc::clone(&manager)));
    let foreign = RecordingG2Capacity::new(Arc::clone(&foreign_manager));
    let hash = SequenceHash::new(84, None, 0);

    let registered = reserve_required_staging(capacity.clone(), 1)
        .expect("compatibility reservation")
        .stage_all(&[hash], capacity.block_size())
        .expect("stage compatibility allocation")
        .publish()
        .expect("publish through bound capacity");

    assert_eq!(registered.len(), 1);
    assert_eq!(capacity.registration_count(), 1);
    assert_eq!(foreign.registration_count(), 0);
    assert_eq!(manager.match_blocks(&[hash]).len(), 1);
    assert!(foreign_manager.match_blocks(&[hash]).is_empty());
}

/// F31: a compatibility rollback must reset only the slot it staged. Under
/// `Reject`, `register_compatibility` hands back the pre-existing retained
/// primary instead of the newly staged block, so an unconditional
/// `set_evict_on_reset(true)` sweep over the returned blocks flags a slot
/// this allocation never staged and never owned.
#[test]
fn compatibility_rollback_keeps_a_collided_retained_primary() {
    let manager = Arc::new(
        BlockManager::<G2>::builder()
            .block_count(2)
            .block_size(4)
            .registry(kvbm_logical::BlockRegistry::new())
            .with_lru_backend()
            .duplication_policy(kvbm_logical::blocks::BlockDuplicationPolicy::Reject)
            .build()
            .expect("build a Reject-policy manager"),
    );
    let hash = SequenceHash::new(1, None, 0);
    let retained_slot = manager
        .allocate_blocks(1)
        .expect("free slot for the retained primary")
        .pop()
        .expect("one allocated block")
        .stage(hash, manager.block_size())
        .expect("stage the retained primary");
    let retained = manager.register_block(retained_slot);

    let capacity = Arc::new(DirectG2Capacity::new(Arc::clone(&manager)));
    let published = reserve_required_staging(capacity.clone(), 1)
        .expect("compatibility reservation")
        .stage_all(&[hash], capacity.block_size())
        .expect("stage the colliding block")
        .publish_reversible()
        .expect("register the collision through the compatibility route");

    // Roll back the unregistered transaction, then release the retained
    // primary. A correct rollback never touches the retained slot's flag.
    drop(published);
    drop(retained);

    assert_eq!(manager.match_blocks(&[hash]).len(), 1);
}

#[test]
fn required_staging_rejects_a_partial_hash_set_and_rolls_back_every_slot() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .build(),
    );
    let capacity = Arc::new(ExactRegistrationCapacity::new(Arc::clone(&manager)));

    let error = match reserve_required_staging(capacity.clone(), 2)
        .expect("exact reservation")
        .stage_all(&[SequenceHash::new(85, None, 0)], capacity.block_size())
    {
        Ok(_) => panic!("a partial hash set must fail as one transaction"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("2 blocks for 1 hashes"));
    assert_eq!(manager.available_blocks(), 2);
    assert_eq!(capacity.exact_registration_count(), 0);
    assert_eq!(capacity.exact_permit_drop_count(), 1);
}

#[test]
fn one_pending_reclaim_completion_can_grant_the_exact_allocation() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let capacity = Arc::new(PendingGrantCapacity {
        inner: ExactRegistrationCapacity::new(Arc::clone(&manager)),
    });
    let hash = SequenceHash::new(86, None, 0);

    let allocation = reserve_required_staging(capacity.clone(), 1)
        .expect("one completion must grant exact capacity");
    assert_eq!(
        allocation.requirement(),
        G2CapacityRequirement::ExactReclaim
    );

    let registered = allocation
        .stage_all(&[hash], capacity.block_size())
        .expect("stage granted allocation")
        .publish()
        .expect("publish granted allocation");

    assert_eq!(registered.len(), 1);
    assert_eq!(capacity.inner.exact_registration_count(), 1);
    assert_eq!(capacity.inner.exact_permit_drop_count(), 1);
}

#[test]
fn repeated_pending_reclaim_is_canceled_after_the_bounded_attempt() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let attempts = Arc::new(AtomicUsize::new(0));
    let permit_drops = Arc::new(AtomicUsize::new(0));
    let capacity = Arc::new(RepeatedPendingCapacity {
        manager,
        attempts: Arc::clone(&attempts),
        permit_drops: Arc::clone(&permit_drops),
    });

    let error = match reserve_required_staging(capacity, 1) {
        Ok(_) => panic!("a repeated pending reclaim must not escape as an allocation"),
        Err(error) => error,
    };

    assert_eq!(error, G2CapacityError::PendingReclaim { target_count: 1 });
    assert_eq!(attempts.load(Ordering::Relaxed), 1);
    assert_eq!(permit_drops.load(Ordering::Relaxed), 1);
}

#[test]
fn exact_grant_with_the_wrong_allocation_kind_is_rejected_and_rolled_back() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let capacity = Arc::new(FaultyExactCapacity {
        inner: ExactRegistrationCapacity::new(Arc::clone(&manager)),
        granted_kind: G2AllocationKind::CacheExtension,
        granted_count: None,
        pending: false,
    });

    let error = match reserve_required_staging(capacity.clone(), 1) {
        Ok(_) => panic!("a CacheExtension exact grant must not satisfy RequiredStaging"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("CacheExtension allocation for a RequiredStaging request")
    );
    assert_eq!(manager.available_blocks(), 1);
    assert_eq!(capacity.inner.exact_permit_drop_count(), 1);
}

#[test]
fn pending_exact_grant_with_the_wrong_allocation_kind_is_rejected_and_rolled_back() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let capacity = Arc::new(FaultyExactCapacity {
        inner: ExactRegistrationCapacity::new(Arc::clone(&manager)),
        granted_kind: G2AllocationKind::CacheExtension,
        granted_count: None,
        pending: true,
    });

    let error = match reserve_required_staging(capacity.clone(), 1) {
        Ok(_) => panic!("a pending CacheExtension grant must not satisfy RequiredStaging"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("CacheExtension allocation for a RequiredStaging request")
    );
    assert_eq!(manager.available_blocks(), 1);
    assert_eq!(capacity.inner.exact_permit_drop_count(), 1);
}

#[test]
fn compatibility_grant_with_the_wrong_allocation_kind_is_rejected_and_rolled_back() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let capacity = Arc::new(FaultyCompatibilityCapacity {
        manager: Arc::clone(&manager),
        granted_kind: G2AllocationKind::CacheExtension,
        granted_count: None,
    });

    let error = match reserve_required_staging(capacity, 1) {
        Ok(_) => panic!("a CacheExtension compatibility grant must not satisfy RequiredStaging"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("CacheExtension allocation for a RequiredStaging request")
    );
    assert_eq!(manager.available_blocks(), 1);
}

#[test]
fn exact_grant_with_the_wrong_count_is_rejected_and_rolled_back() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .build(),
    );
    let capacity = Arc::new(FaultyExactCapacity {
        inner: ExactRegistrationCapacity::new(Arc::clone(&manager)),
        granted_kind: G2AllocationKind::RequiredStaging,
        granted_count: Some(1),
        pending: false,
    });

    let error = match reserve_required_staging(capacity.clone(), 2) {
        Ok(_) => panic!("a partial exact grant must not satisfy RequiredStaging"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("1 blocks for a 2-block RequiredStaging request")
    );
    assert_eq!(manager.available_blocks(), 2);
    assert_eq!(capacity.inner.exact_permit_drop_count(), 1);
}

#[test]
fn compatibility_grant_with_the_wrong_count_is_rejected_and_rolled_back() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .build(),
    );
    let capacity = Arc::new(FaultyCompatibilityCapacity {
        manager: Arc::clone(&manager),
        granted_kind: G2AllocationKind::RequiredStaging,
        granted_count: Some(1),
    });

    let error = match reserve_required_staging(capacity, 2) {
        Ok(_) => panic!("a partial compatibility grant must not satisfy RequiredStaging"),
        Err(error) => error,
    };

    assert!(
        error
            .to_string()
            .contains("1 blocks for a 2-block RequiredStaging request")
    );
    assert_eq!(manager.available_blocks(), 2);
}
