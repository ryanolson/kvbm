// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn retry_keeps_reclaim_completion_owner_and_permit() {
    let request = G2CapacityRequest::exact_reclaim(G2AllocationKind::RequiredStaging, 1);
    let attempts = Arc::new(AtomicUsize::new(0));
    let permit_drops = Arc::new(AtomicUsize::new(0));
    let pending = G2PendingReclaim::new(
        G2ReclaimPlan::new(request, 1),
        Box::new(RetryCompletion {
            request,
            attempts: Arc::clone(&attempts),
            permit: DropGuard(Arc::clone(&permit_drops)),
        }),
    );

    let pending = match G2CapacityDecision::PendingReclaim(pending) {
        G2CapacityDecision::PendingReclaim(pending) => pending,
        G2CapacityDecision::Granted(_) | G2CapacityDecision::ExactGranted(_) => {
            panic!("test decision must retain pending reclaim")
        }
    };
    let retry = match pending.complete() {
        G2ReclaimCompletion::PendingReclaim(retry) => retry,
        G2ReclaimCompletion::Granted(_) | G2ReclaimCompletion::Rejected(_) => {
            panic!("reset drift must preserve a pending reclaim")
        }
    };

    assert_eq!(attempts.load(Ordering::Relaxed), 1);
    assert_eq!(retry.plan().target_count(), 1);
    assert_eq!(permit_drops.load(Ordering::Relaxed), 0);

    drop(retry);

    assert_eq!(permit_drops.load(Ordering::Relaxed), 1);
}
