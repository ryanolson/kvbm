// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing")]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::thread;

use kvbm_logical::testing::{
    count_store_lock_acquisitions, create_iota_token_block, create_test_manager,
};

#[test]
fn armed_mutable_drop_counts() {
    let manager = create_test_manager::<()>(1);
    let blocks = manager.allocate_blocks_from_reset(1).unwrap();

    let ((), count) = count_store_lock_acquisitions(|| drop(blocks));

    assert_eq!(count, 1);
    assert_eq!(manager.available_blocks(), 1);
}

#[test]
fn thread_without_store_access_counts_zero() {
    let manager = create_test_manager::<()>(1);
    let ((), count) = thread::scope(|threads| {
        let measure = || {
            threads
                .spawn(|| assert_eq!(manager.available_blocks(), 1))
                .join()
                .unwrap();
        };
        count_store_lock_acquisitions(measure)
    });

    assert_eq!(count, 0);
}

#[test]
fn normal_return_counts_across_pools() {
    let first = create_test_manager::<()>(1);
    let second = create_test_manager::<()>(2);

    let (available, count) =
        count_store_lock_acquisitions(|| first.available_blocks() + second.available_blocks());

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

    let (result, count) = count_store_lock_acquisitions(|| {
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

    let (result, count): (thread::Result<()>, usize) = count_store_lock_acquisitions(|| {
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

    assert_eq!(count_store_lock_acquisitions(|| drop(clone)), ((), 0));
    assert_eq!(count_store_lock_acquisitions(|| drop(block)), ((), 1));
}

#[test]
fn nested_scope_restores_outer_count_after_unwind() {
    let manager = create_test_manager::<()>(1);

    let ((), count) = count_store_lock_acquisitions(|| {
        assert_eq!(manager.available_blocks(), 1);
        let result = catch_unwind(AssertUnwindSafe(|| {
            count_store_lock_acquisitions(|| {
                assert_eq!(manager.available_blocks(), 1);
                std::panic::panic_any("unwind across a count scope");
            })
        }));
        assert_eq!(
            result.unwrap_err().downcast_ref::<&str>(),
            Some(&"unwind across a count scope")
        );
        assert_eq!(
            count_store_lock_acquisitions(|| manager.available_blocks()),
            (1, 1)
        );
    });

    assert_eq!(count, 3);
    assert_eq!(count_store_lock_acquisitions(|| ()), ((), 0));
}
