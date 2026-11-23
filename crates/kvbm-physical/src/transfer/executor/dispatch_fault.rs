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
    NixlPost,
    NixlStatusEvent,
    HostCopy,
}

thread_local! {
    pub(crate) static NIXL_POSTS: Cell<usize> = const { Cell::new(0) };
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
            match target {
                Point::NixlPost => anyhow::bail!("injected NIXL post failure"),
                Point::NixlStatusEvent => anyhow::bail!("injected NIXL status event failure"),
                Point::HostCopy => anyhow::bail!("injected host dispatch failure"),
                _ => anyhow::bail!("injected CUDA dispatch failure"),
            }
        }
        FAULT.set(Some((target, count - 1)));
    }
    Ok(())
}

pub(super) fn after_nixl_post(
    agent: &kvbm_memory::nixl::NixlAgent,
    request: &kvbm_memory::nixl::XferRequest,
    pending: bool,
) -> anyhow::Result<bool> {
    NIXL_POSTS.set(NIXL_POSTS.get() + 1);
    let fault = check(Point::NixlPost);
    let status_event_will_fail = FAULT
        .get()
        .is_some_and(|(point, _)| point == Point::NixlStatusEvent);
    if pending && (fault.is_err() || status_event_will_fail) {
        // Keep fixture buffers safe even when the old dispatcher returns a bare error.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !agent.get_xfer_status(request)?.is_success() {
            assert!(
                std::time::Instant::now() < deadline,
                "test NIXL post did not drain"
            );
            std::thread::yield_now();
        }
    }
    fault?;
    Ok(pending)
}
