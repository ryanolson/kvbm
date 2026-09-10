// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use kvbm_common::SequenceHash;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;

use super::*;
use crate::G2;
use crate::testing::managers::TestManagerBuilder;

mod compatibility;
mod exact;
mod policy_route;
mod reclaim;
mod required_staging;
mod routes;

struct DropGuard(Arc<AtomicUsize>);

type ExactPermitDropObserver = Arc<dyn Fn() + Send + Sync>;

struct ExactPermitDropGuard {
    drops: Arc<AtomicUsize>,
    observer: Arc<Mutex<Option<ExactPermitDropObserver>>>,
}

struct ExactRegistrationOwner {
    manager: Arc<BlockManager<G2>>,
    registrations: Arc<AtomicUsize>,
    permit: Arc<ExactPermitDropGuard>,
    retained_permits: Arc<Mutex<Vec<Arc<ExactPermitDropGuard>>>>,
}

struct ExactRegistrationCapacity {
    manager: Arc<BlockManager<G2>>,
    requests: Mutex<Vec<G2CapacityRequest>>,
    compatibility_registrations: AtomicUsize,
    exact_registrations: Arc<AtomicUsize>,
    exact_permit_drops: Arc<AtomicUsize>,
    exact_permit_drop_observer: Arc<Mutex<Option<ExactPermitDropObserver>>>,
    retained_permits: Arc<Mutex<Vec<Arc<ExactPermitDropGuard>>>>,
}

struct GrantedCompletion {
    allocation: G2ExactAllocation,
}

struct RetryCompletion {
    request: G2CapacityRequest,
    attempts: Arc<AtomicUsize>,
    permit: DropGuard,
}

impl G2LeaseGuard for DropGuard {}

impl G2LeaseGuard for ExactPermitDropGuard {}

impl Drop for DropGuard {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for ExactPermitDropGuard {
    fn drop(&mut self) {
        let observer = self
            .observer
            .lock()
            .expect("exact permit observer lock")
            .clone();
        if let Some(observer) = observer {
            observer();
        }
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl G2ExactRegistrationOwner for ExactRegistrationOwner {
    fn register_blocks(
        &mut self,
        blocks: Vec<kvbm_logical::blocks::CompleteBlock<G2>>,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        self.registrations.fetch_add(1, Ordering::Relaxed);
        Ok(self.manager.register_blocks(blocks))
    }

    fn rollback_required_staging(&mut self, blocks: Vec<ImmutableBlock<G2>>) {
        self.manager.release_blocks(blocks, Some(true));
    }

    fn retain_cache_extension(self: Box<Self>) {
        let Self {
            permit,
            retained_permits,
            ..
        } = *self;
        retained_permits
            .lock()
            .expect("retained permits lock")
            .push(permit);
    }
}

impl ExactRegistrationCapacity {
    fn new(manager: Arc<BlockManager<G2>>) -> Self {
        Self {
            manager,
            requests: Mutex::new(Vec::new()),
            compatibility_registrations: AtomicUsize::new(0),
            exact_registrations: Arc::new(AtomicUsize::new(0)),
            exact_permit_drops: Arc::new(AtomicUsize::new(0)),
            exact_permit_drop_observer: Arc::new(Mutex::new(None)),
            retained_permits: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn exact_registration_count(&self) -> usize {
        self.exact_registrations.load(Ordering::Relaxed)
    }

    fn requests(&self) -> Vec<G2CapacityRequest> {
        self.requests.lock().expect("requests lock").clone()
    }

    fn retained_permit_count(&self) -> usize {
        self.retained_permits
            .lock()
            .expect("retained permits lock")
            .len()
    }

    fn exact_permit_drop_count(&self) -> usize {
        self.exact_permit_drops.load(Ordering::Relaxed)
    }

    fn observe_exact_permit_drop(&self, observer: ExactPermitDropObserver) {
        *self
            .exact_permit_drop_observer
            .lock()
            .expect("exact permit observer lock") = Some(observer);
    }

    fn clear_retained_permits(&self) {
        self.retained_permits
            .lock()
            .expect("retained permits lock")
            .clear();
    }
}

impl G2Capacity for ExactRegistrationCapacity {
    fn reserve(&self, request: G2CapacityRequest) -> Result<G2CapacityDecision, G2CapacityError> {
        self.requests.lock().expect("requests lock").push(request);
        if request.requirement() == G2CapacityRequirement::Compatibility {
            return Err(G2CapacityError::Rejected(
                "exact-only capacity rejects compatibility allocation".to_string(),
            ));
        }
        let blocks = self
            .manager
            .allocate_blocks(request.count())
            .ok_or(G2CapacityError::Unavailable(request))?;
        G2ExactAllocation::new(
            request,
            blocks,
            Box::new(ExactRegistrationOwner {
                manager: Arc::clone(&self.manager),
                registrations: Arc::clone(&self.exact_registrations),
                permit: Arc::new(ExactPermitDropGuard {
                    drops: Arc::clone(&self.exact_permit_drops),
                    observer: Arc::clone(&self.exact_permit_drop_observer),
                }),
                retained_permits: Arc::clone(&self.retained_permits),
            }),
        )
        .map(G2CapacityDecision::ExactGranted)
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
        self.compatibility_registrations
            .fetch_add(1, Ordering::Relaxed);
        Err(G2CapacityError::Rejected(
            "exact-only capacity rejects compatibility registration".to_string(),
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

impl G2PendingReclaimCompletion for GrantedCompletion {
    fn complete(self: Box<Self>) -> G2ReclaimCompletion {
        let Self { allocation } = *self;
        G2ReclaimCompletion::Granted(allocation)
    }
}

impl G2PendingReclaimCompletion for RetryCompletion {
    fn complete(self: Box<Self>) -> G2ReclaimCompletion {
        let Self {
            request,
            attempts,
            permit,
        } = *self;
        attempts.fetch_add(1, Ordering::Relaxed);
        G2ReclaimCompletion::PendingReclaim(G2PendingReclaim::new(
            G2ReclaimPlan::new(request, 1),
            Box::new(Self {
                request,
                attempts,
                permit,
            }),
        ))
    }
}

fn expect_exact_allocation(decision: G2CapacityDecision) -> G2ExactAllocation {
    match decision {
        G2CapacityDecision::ExactGranted(allocation) => allocation,
        G2CapacityDecision::Granted(_) | G2CapacityDecision::PendingReclaim(_) => {
            panic!("test source must grant exact capacity")
        }
    }
}
