// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::{Cell, RefCell};
use std::sync::Arc;

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
    NixlQueueClosed,
    StagedNixlQueueClosed,
    NixlEventShutdown,
    NixlDroppedReceipt,
    NixlSuccessfulReceipt,
    NixlStatusEvent,
    HostCopy,
}

thread_local! {
    pub(crate) static NIXL_REQUEST_PROBE: RefCell<Option<Arc<()>>> = const { RefCell::new(None) };
    pub(crate) static NIXL_POSTS: Cell<usize> = const { Cell::new(0) };
    static FAULT: Cell<Option<(Point, usize)>> = const { Cell::new(None) };
}

struct Fault;

impl Fault {
    fn install(point: Point, call: usize) -> Self {
        assert!(FAULT.replace(Some((point, call))).is_none());
        Self
    }

    fn assert_used(&self) {
        assert!(
            FAULT.get().is_none(),
            "dispatch did not reach the injected failure"
        );
    }
}

impl Drop for Fault {
    fn drop(&mut self) {
        FAULT.set(None);
    }
}

pub(crate) fn inject<T>(point: Point, call: usize, dispatch: impl FnOnce() -> T) -> T {
    let fault = Fault::install(point, call);
    let result = dispatch();
    fault.assert_used();
    result
}

pub(crate) async fn inject_async<T>(point: Point, dispatch: impl Future<Output = T>) -> T {
    let fault = Fault::install(point, 1);
    let result = dispatch.await;
    fault.assert_used();
    result
}

pub(crate) fn check(point: Point) -> anyhow::Result<()> {
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
    match FAULT.get().map(|(point, _)| point) {
        Some(
            point @ (Point::NixlPost
            | Point::NixlQueueClosed
            | Point::StagedNixlQueueClosed
            | Point::NixlEventShutdown
            | Point::NixlDroppedReceipt),
        ) => {
            assert!(pending, "the fault requires a pending request");
            if point == Point::NixlPost {
                check(point)?;
            }
            if matches!(point, Point::NixlEventShutdown | Point::NixlDroppedReceipt) {
                let _ = check(point);
            }
        }
        Some(Point::NixlSuccessfulReceipt) => {
            let _ = check(Point::NixlSuccessfulReceipt);
        }
        Some(Point::NixlStatusEvent) if pending => {
            // The late-reservation mutation still posts. Drain the fixture before
            // the injected reservation error lets the fixture release its buffers.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !agent.get_xfer_status(request)?.is_success() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "test NIXL post did not drain"
                );
                std::thread::yield_now();
            }
        }
        _ => {}
    }
    Ok(pending)
}
