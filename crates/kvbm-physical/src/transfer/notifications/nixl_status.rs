// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL request retention and status polling.

use anyhow::{Result, anyhow};
use kvbm_memory::nixl::{Agent as NixlAgent, XferRequest};
use kvbm_observability::SharedKvbmObservability;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::CompletionChecker;
use crate::transfer::executor::registration::RegistrationGuards;

struct NixlRequestResources {
    request: XferRequest,
    _registrations: RegistrationGuards,
    #[cfg(all(test, feature = "testing-kvbm"))]
    _request_probe: Option<std::sync::Arc<()>>,
}

/// Owns a request and its registrations from post through drain proof.
/// An unproven request keeps both for the process lifetime, even if its
/// checker, queue entry, or receipt is dropped.
pub(crate) struct NixlRequestHold {
    resources: Option<NixlRequestResources>,
    observability: Option<SharedKvbmObservability>,
    proven: AtomicBool,
}

impl NixlRequestHold {
    pub(crate) fn new(
        request: XferRequest,
        registrations: RegistrationGuards,
        observability: Option<SharedKvbmObservability>,
    ) -> Self {
        Self {
            resources: Some(NixlRequestResources {
                request,
                _registrations: registrations,
                #[cfg(all(test, feature = "testing-kvbm"))]
                _request_probe: crate::transfer::executor::dispatch_fault::NIXL_REQUEST_PROBE
                    .take(),
            }),
            observability,
            proven: AtomicBool::new(false),
        }
    }

    pub(crate) fn request(&self) -> &XferRequest {
        &self
            .resources
            .as_ref()
            .expect("request hold is present")
            .request
    }

    pub(crate) fn confirm_drain(&self) {
        self.proven.store(true, Ordering::Relaxed);
    }
}

impl Drop for NixlRequestHold {
    fn drop(&mut self) {
        if self.proven.load(Ordering::Relaxed) {
            return;
        }
        std::mem::forget(self.resources.take());
        if let Some(metrics) = &self.observability {
            metrics
                .transfer_metrics()
                .transfer_failures_total
                .with_label_values(&["nixl", "post", "retained_request"])
                .inc();
        }
        static RETAINED_REQUESTS: AtomicU64 = AtomicU64::new(0);
        if let Some(count) = count_retained_request(&RETAINED_REQUESTS) {
            tracing::error!(
                retained_requests = count,
                "Retained NIXL request without a drain proof"
            );
        }
    }
}

fn count_retained_request(counter: &AtomicU64) -> Option<u64> {
    let count = counter.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    count.is_power_of_two().then_some(count)
}

/// Completion checker that polls NIXL transfer status.
pub(crate) struct NixlStatusChecker {
    agent: NixlAgent,
    hold: NixlRequestHold,
}

impl NixlStatusChecker {
    pub(crate) fn new(agent: NixlAgent, hold: NixlRequestHold) -> Self {
        Self { agent, hold }
    }
}

impl CompletionChecker for NixlStatusChecker {
    fn is_complete(&self) -> Result<bool> {
        match self.agent.get_xfer_status(self.hold.request()) {
            Ok(status) => Ok(status.is_success()),
            Err(e) => Err(anyhow!("NIXL transfer status check failed: {}", e)),
        }
    }

    fn completion_notified(&self) {
        // A failed Velo trigger leaves the receipt unproven and the hold retained.
        self.hold.confirm_drain();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_nixl_requests_log_at_doubling_counts() {
        let counter = AtomicU64::new(0);
        let logged: Vec<_> = (0..10)
            .filter_map(|_| count_retained_request(&counter))
            .collect();
        assert_eq!(counter.load(Ordering::Relaxed), 10);
        assert_eq!(logged, [1, 2, 4, 8]);
    }
}
