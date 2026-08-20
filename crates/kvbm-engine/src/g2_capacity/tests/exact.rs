// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use futures::executor::block_on;

use super::*;

#[test]
fn direct_adapter_rejects_exact_reclaim() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let capacity = DirectG2Capacity::new(manager);
    let request = G2CapacityRequest::exact_reclaim(G2AllocationKind::RequiredStaging, 1);

    let error = match capacity.reserve(request) {
        Err(error) => error,
        Ok(_) => panic!("direct adapter must not claim exact-reclaim authority"),
    };

    assert_eq!(error, G2CapacityError::ExactReclaimUnsupported(request));
}

#[test]
fn exact_staging_uses_only_its_source_registration_owner() {
    let source_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let wrong_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let source = ExactRegistrationCapacity::new(Arc::clone(&source_manager));
    let wrong = ExactRegistrationCapacity::new(wrong_manager);
    let request = G2CapacityRequest::exact_reclaim(G2AllocationKind::RequiredStaging, 1);
    let allocation = expect_exact_allocation(source.reserve(request).expect("exact reservation"));

    let registered = allocation
        .stage_all(&[SequenceHash::new(71, None, 0)], source.block_size())
        .expect("stage exact allocation")
        .register()
        .expect("source-bound exact registration");

    assert_eq!(registered.blocks().len(), 1);
    assert_eq!(source.exact_registration_count(), 1);
    assert_eq!(wrong.exact_registration_count(), 0);
    assert_eq!(
        source.compatibility_registrations.load(Ordering::Relaxed),
        0
    );
}

#[test]
fn exact_only_capacity_rejects_foreign_compatibility_registration() {
    let source_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let exact_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let source = DirectG2Capacity::new(Arc::clone(&source_manager));
    let exact = ExactRegistrationCapacity::new(exact_manager);
    let hashes = [SequenceHash::new(74, None, 0)];
    let staged = reserve_compatibility(&source, G2AllocationKind::RequiredStaging, 1)
        .expect("compatibility allocation")
        .stage_all(&hashes, 4)
        .expect("stage compatibility allocation");

    let error = exact
        .register_compatibility(staged)
        .expect_err("exact-only capacity must reject compatibility registration");

    assert_eq!(
        error,
        G2CapacityError::Rejected(
            "exact-only capacity rejects compatibility registration".to_string()
        )
    );
    assert!(source_manager.match_blocks(&hashes).is_empty());
    assert!(source_manager.allocate_blocks(1).is_some());
}

#[test]
fn exact_owner_survives_pending_completion_transfer_and_staging() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let capacity = ExactRegistrationCapacity::new(Arc::clone(&manager));
    let request = G2CapacityRequest::exact_reclaim(G2AllocationKind::RequiredStaging, 1);
    let allocation = expect_exact_allocation(capacity.reserve(request).expect("exact reservation"));
    let pending = G2PendingReclaim::new(
        G2ReclaimPlan::new(request, 0),
        Box::new(GrantedCompletion { allocation }),
    );
    let allocation = match pending.complete() {
        G2ReclaimCompletion::Granted(allocation) => allocation,
        G2ReclaimCompletion::PendingReclaim(_) | G2ReclaimCompletion::Rejected(_) => {
            panic!("completion must grant exact capacity")
        }
    };
    let allocation = block_on(
        allocation.transfer_with(|blocks| async move { Ok::<_, std::convert::Infallible>(blocks) }),
    )
    .expect("identity transfer");

    let registered = allocation
        .stage_all(&[SequenceHash::new(72, None, 0)], capacity.block_size())
        .expect("stage exact allocation")
        .register()
        .expect("source-bound exact registration");

    assert_eq!(registered.blocks().len(), 1);
    assert_eq!(capacity.exact_registration_count(), 1);
    assert_eq!(capacity.exact_permit_drop_count(), 1);
}

#[test]
fn exact_cache_residency_remains_in_source_owner_state() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let capacity = ExactRegistrationCapacity::new(Arc::clone(&manager));
    let request = G2CapacityRequest::exact_reclaim(G2AllocationKind::CacheExtension, 1);
    let allocation = expect_exact_allocation(capacity.reserve(request).expect("exact reservation"));

    let registered = allocation
        .stage_all(&[SequenceHash::new(73, None, 0)], 4)
        .expect("stage exact allocation")
        .register()
        .expect("register exact allocation");

    assert_eq!(capacity.retained_permit_count(), 1);
    assert_eq!(capacity.exact_permit_drop_count(), 0);

    drop(registered);

    assert_eq!(capacity.exact_permit_drop_count(), 0);

    capacity.clear_retained_permits();

    assert_eq!(capacity.exact_permit_drop_count(), 1);
}

#[test]
fn dropping_exact_allocation_cancels_one_owner_permit_once() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let capacity = ExactRegistrationCapacity::new(manager);
    let request = G2CapacityRequest::exact_reclaim(G2AllocationKind::RequiredStaging, 1);
    let allocation = expect_exact_allocation(capacity.reserve(request).expect("exact reservation"));

    drop(allocation);

    assert_eq!(capacity.exact_permit_drop_count(), 1);
}

#[test]
fn exact_allocation_rejects_a_compatibility_request() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let request = G2CapacityRequest::compatibility(G2AllocationKind::RequiredStaging, 1);
    let blocks = manager.allocate_blocks(1).expect("test allocation");
    let permit_drops = Arc::new(AtomicUsize::new(0));

    let error = match G2ExactAllocation::new(
        request,
        blocks,
        Box::new(ExactRegistrationOwner {
            manager,
            registrations: Arc::new(AtomicUsize::new(0)),
            permit: Arc::new(ExactPermitDropGuard {
                drops: Arc::clone(&permit_drops),
                observer: Arc::new(Mutex::new(None)),
            }),
            retained_permits: Arc::new(Mutex::new(Vec::new())),
        }),
    ) {
        Err(error) => error,
        Ok(_) => panic!("an exact allocation must reject a compatibility request"),
    };

    assert_eq!(
        error,
        G2CapacityError::RequirementMismatch {
            expected: G2CapacityRequirement::ExactReclaim,
            actual: G2CapacityRequirement::Compatibility,
        }
    );
    assert_eq!(permit_drops.load(Ordering::Relaxed), 1);
}
