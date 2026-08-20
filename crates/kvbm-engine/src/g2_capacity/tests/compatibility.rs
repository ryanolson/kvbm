// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn allocation_releases_its_capacity_guard() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let released = Arc::new(AtomicUsize::new(0));
    let mutables = manager.allocate_blocks(1).expect("test allocation");
    let allocation = G2Allocation::new(
        G2AllocationKind::RequiredStaging,
        mutables,
        Arc::new(DropGuard(Arc::clone(&released))),
    );

    drop(allocation);

    assert_eq!(released.load(Ordering::Relaxed), 1);
}

#[test]
fn direct_adapter_registers_a_staged_lease() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .build(),
    );
    let capacity = DirectG2Capacity::new(Arc::clone(&manager));
    let hashes = [
        SequenceHash::new(11, None, 0),
        SequenceHash::new(12, Some(11), 1),
    ];

    let staged = reserve_compatibility(&capacity, G2AllocationKind::RequiredStaging, hashes.len())
        .expect("test allocation")
        .stage_all(&hashes, capacity.block_size())
        .expect("stage allocation");
    let registered = capacity
        .register_compatibility(staged)
        .expect("compatibility registration");

    assert_eq!(registered.len(), hashes.len());
    assert_eq!(manager.match_blocks(&hashes).len(), hashes.len());
}

#[test]
fn direct_adapter_rejects_blocks_from_another_manager_without_mutation() {
    let source_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let target_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let source = DirectG2Capacity::new(Arc::clone(&source_manager));
    let target = DirectG2Capacity::new(Arc::clone(&target_manager));
    let hash = SequenceHash::new(13, None, 0);
    let staged = reserve_compatibility(&source, G2AllocationKind::RequiredStaging, 1)
        .expect("source allocation")
        .stage_all(&[hash], source.block_size())
        .expect("stage source allocation");
    let target_before = target_manager.metrics().snapshot();

    let error = target
        .register_compatibility(staged)
        .expect_err("the target must reject foreign blocks");

    assert_eq!(
        error,
        G2CapacityError::Rejected(
            "compatibility registration blocks belong to another G2 manager".to_string()
        )
    );
    assert_eq!(target_manager.metrics().snapshot(), target_before);
    assert_eq!(source_manager.available_blocks(), 1);
    assert_eq!(target_manager.available_blocks(), 1);
    assert!(source_manager.match_blocks(&[hash]).is_empty());
    assert!(target_manager.match_blocks(&[hash]).is_empty());
}

#[test]
fn staged_lease_survives_the_registration_callback() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let released = Arc::new(AtomicUsize::new(0));
    let mutables = manager.allocate_blocks(1).expect("test allocation");
    let staged = G2Allocation::new(
        G2AllocationKind::CacheExtension,
        mutables,
        Arc::new(DropGuard(Arc::clone(&released))),
    )
    .stage_all(&[SequenceHash::new(1, None, 0)], 4)
    .expect("stage allocation");

    let registered = staged.register_with(|blocks| {
        assert_eq!(released.load(Ordering::Relaxed), 0);
        manager.register_blocks(blocks)
    });

    assert_eq!(registered.len(), 1);
    assert_eq!(released.load(Ordering::Relaxed), 1);
}

#[test]
fn cache_extension_registration_retains_its_residency_lease() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    );
    let released = Arc::new(AtomicUsize::new(0));
    let mutables = manager.allocate_blocks(1).expect("test allocation");
    let staged = G2Allocation::new(
        G2AllocationKind::CacheExtension,
        mutables,
        Arc::new(DropGuard(Arc::clone(&released))),
    )
    .stage_all(&[SequenceHash::new(1, None, 0)], 4)
    .expect("stage allocation");

    let (registered, residency) = staged
        .register_with_cache_residency(|blocks| manager.register_blocks(blocks))
        .expect("cache extension registration");

    assert_eq!(registered.len(), 1);
    assert_eq!(residency.len(), 1);
    assert_eq!(released.load(Ordering::Relaxed), 0);

    drop(residency);

    assert_eq!(released.load(Ordering::Relaxed), 1);
}

#[test]
fn block_lease_shares_cancel_once_after_entry_reassembly() {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(3)
            .block_size(4)
            .build(),
    );
    let released = Arc::new(AtomicUsize::new(0));
    let mutables = manager.allocate_blocks(3).expect("test allocation");
    let hashes = [
        SequenceHash::new(21, None, 0),
        SequenceHash::new(22, Some(21), 1),
        SequenceHash::new(23, Some(22), 2),
    ];
    let entries = G2Allocation::new(
        G2AllocationKind::CacheExtension,
        mutables,
        Arc::new(DropGuard(Arc::clone(&released))),
    )
    .stage_all(&hashes, 4)
    .expect("stage allocation")
    .into_entries();

    assert_eq!(released.load(Ordering::Relaxed), 0);

    let staged = G2StagedAllocation::from_entries(G2AllocationKind::CacheExtension, entries);
    let (_, residency) = staged
        .register_with_cache_residency(|blocks| manager.register_blocks(blocks))
        .expect("cache extension registration");

    assert_eq!(residency.len(), 3);
    assert_eq!(residency.unique_guard_count(), 1);
    assert_eq!(released.load(Ordering::Relaxed), 0);

    drop(residency);

    assert_eq!(released.load(Ordering::Relaxed), 1);
}
