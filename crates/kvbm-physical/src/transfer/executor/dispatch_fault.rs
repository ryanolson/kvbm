// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Point {
    CompletionEvent,
    RaggedCopy,
    TransformUpload,
    VectorizedFree,
    BeforeWholeBlockBatch,
    BeforeTransformUpload,
    PlannerBatch,
    BeforeGraphLaunch,
    VectorizedAlloc,
}

thread_local! {
    static FAULT: Cell<Option<(Point, usize)>> = const { Cell::new(None) };
}

pub(crate) fn inject<T>(point: Point, call: usize, dispatch: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FAULT.set(None);
        }
    }

    assert!(FAULT.replace(Some((point, call))).is_none());
    let _reset = Reset;
    let result = dispatch();
    assert!(
        FAULT.get().is_none(),
        "dispatch did not reach the injected failure"
    );
    result
}

pub(super) fn check(point: Point) -> anyhow::Result<()> {
    if let Some((target, count)) = FAULT.get()
        && target == point
    {
        if count == 1 {
            FAULT.set(None);
            anyhow::bail!("injected CUDA dispatch failure");
        }
        FAULT.set(Some((target, count - 1)));
    }
    Ok(())
}
