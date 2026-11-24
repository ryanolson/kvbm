// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::thread;

use dynamo_tokens::TokenBlockSequence;
use kvbm_logical::manager::BlockManagerConfigBuilder;
use kvbm_logical::testing::{
    count_store_lock_acquisitions, create_iota_token_block, create_test_manager,
    create_test_manager_with_backend,
};
use kvbm_logical::{BlockManager, InactiveLineagePreflight};

fn prepared_lineage() -> (BlockManager<()>, InactiveLineagePreflight<()>) {
    let manager =
        create_test_manager_with_backend(2, BlockManagerConfigBuilder::with_lineage_backend);
    let sequence = TokenBlockSequence::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None);
    let pins = manager
        .allocate_blocks_from_reset(2)
        .unwrap()
        .into_iter()
        .zip(sequence.blocks())
        .map(|(slot, block)| manager.register_block(slot.complete(block).unwrap()))
        .collect::<Vec<_>>();
    let leaf = pins.last().unwrap().sequence_hash();
    drop(pins);
    let prepared = manager.preflight_inactive_lineage_by_hash(leaf).unwrap();
    assert_eq!(prepared.block_count(), 2);
    (manager, prepared)
}

#[test]
fn prepared_pin_counts_one_lock_for_multiple_blocks() {
    let (manager, prepared) = prepared_lineage();
    let selected = prepared.source_hashes();

    // Keep the pins alive past the count so it measures the call alone.
    let (pins, count) = count_store_lock_acquisitions(|_| {
        manager.pin_prepared_inactive_lineage(&prepared, &selected)
    });

    assert_eq!(pins.unwrap().len(), selected.len());
    assert_eq!(count, 1);
}

#[test]
fn prepared_pin_rejection_counts_one_lock() {
    let (manager, prepared) = prepared_lineage();
    let selected = prepared.source_hashes();
    let (other_manager, _) = prepared_lineage();
    let (wrong_manager, count) = count_store_lock_acquisitions(|_| {
        other_manager.pin_prepared_inactive_lineage(&prepared, &selected)
    });
    assert!(wrong_manager.is_none());
    assert_eq!(count, 1);

    let pins = manager
        .pin_prepared_inactive_lineage(&prepared, &selected)
        .unwrap();
    let (stale, count) = count_store_lock_acquisitions(|_| {
        manager.pin_prepared_inactive_lineage(&prepared, &selected)
    });
    assert!(stale.is_none());
    assert_eq!(count, 1);
    drop(pins);
}

#[test]
fn armed_mutable_drop_counts() {
    let manager = create_test_manager::<()>(1);
    let blocks = manager.allocate_blocks_from_reset(1).unwrap();

    let ((), count) = count_store_lock_acquisitions(|_| drop(blocks));

    assert_eq!(count, 1);
    assert_eq!(manager.available_blocks(), 1);
}

#[test]
fn thread_without_store_access_counts_zero() {
    let manager = create_test_manager::<()>(1);
    let ((), count) = thread::scope(|threads| {
        count_store_lock_acquisitions(|_| {
            threads
                .spawn(|| assert_eq!(manager.available_blocks(), 1))
                .join()
                .unwrap();
        })
    });

    assert_eq!(count, 0);
}

#[test]
fn normal_return_counts_across_pools() {
    let first = create_test_manager::<()>(1);
    let second = create_test_manager::<()>(2);

    let (available, count) =
        count_store_lock_acquisitions(|_| first.available_blocks() + second.available_blocks());

    assert_eq!(available, 3);
    assert_eq!(count, 2);
}

#[test]
fn error_cleanup_counts_staged_drop() {
    let manager = create_test_manager::<()>(1);
    let staged = manager
        .allocate_blocks_from_reset(1)
        .unwrap()
        .pop()
        .unwrap()
        .complete(&create_iota_token_block(0, 4))
        .unwrap();

    let (result, count) = count_store_lock_acquisitions(|_| {
        let _held = staged;
        manager
            .allocate_blocks_from_reset(1)
            .ok_or("no reset blocks")?;
        Ok(())
    });

    assert_eq!(result, Err("no reset blocks"));
    assert_eq!(count, 2);
    assert_eq!(manager.available_blocks(), 1);
}

#[test]
fn unwind_counts_armed_mutable_drop() {
    let manager = create_test_manager::<()>(1);
    let blocks = manager.allocate_blocks_from_reset(1).unwrap();

    let (result, count): (thread::Result<()>, usize) = count_store_lock_acquisitions(|_| {
        catch_unwind(AssertUnwindSafe(move || {
            let _held = blocks;
            std::panic::panic_any("unwind with an armed mutable block");
        }))
    });

    assert_eq!(
        result.unwrap_err().downcast_ref::<&str>(),
        Some(&"unwind with an armed mutable block")
    );
    assert_eq!(count, 1);
    assert_eq!(manager.available_blocks(), 1);
}

#[test]
fn immutable_drop_counts_only_for_last_owner() {
    let manager = create_test_manager::<()>(1);
    let staged = manager
        .allocate_blocks_from_reset(1)
        .unwrap()
        .pop()
        .unwrap()
        .complete(&create_iota_token_block(0, 4))
        .unwrap();
    let block = manager.register_block(staged);
    let clone = block.clone();

    assert_eq!(count_store_lock_acquisitions(|_| drop(clone)), ((), 0));
    assert_eq!(count_store_lock_acquisitions(|_| drop(block)), ((), 1));
}

#[test]
fn nested_scope_restores_outer_count_after_unwind() {
    let manager = create_test_manager::<()>(1);

    let ((), count) = count_store_lock_acquisitions(|outer| {
        assert_eq!(manager.available_blocks(), 1);
        let result = catch_unwind(AssertUnwindSafe(|| {
            count_store_lock_acquisitions(|inner| {
                assert_eq!(manager.available_blocks(), 1);
                assert_eq!(inner.count(), 1);
                assert_eq!(outer.count(), 2);
                std::panic::panic_any("unwind across a count scope");
            })
        }));
        assert_eq!(
            result.unwrap_err().downcast_ref::<&str>(),
            Some(&"unwind across a count scope")
        );
        assert_eq!(
            count_store_lock_acquisitions(|_| manager.available_blocks()),
            (1, 1)
        );
    });

    assert_eq!(count, 3);
    assert_eq!(count_store_lock_acquisitions(|_| ()), ((), 0));
}

#[test]
fn worker_cannot_read_another_threads_scope() {
    let manager = create_test_manager::<()>(1);

    let ((), count) = count_store_lock_acquisitions(|scope_count| {
        let error = thread::scope(|threads| {
            threads
                .spawn(|| {
                    assert_eq!(manager.available_blocks(), 1);
                    scope_count.count()
                })
                .join()
                .expect_err("a different thread must reject the count read")
        });
        assert_eq!(
            error.downcast_ref::<&str>(),
            Some(&"store lock count belongs to a different thread")
        );
    });

    assert_eq!(count, 0);
}
