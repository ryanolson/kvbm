// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test capacity fixtures.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use kvbm_common::SequenceHash;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;

use super::{
    G2Allocation, G2AllocationKind, G2Capacity, G2CapacityDecision, G2CapacityError,
    G2CapacityRequest, G2CapacityRequirement, G2LeaseGuard, G2StagedAllocation,
};
use crate::G2;

struct Lease {
    live: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

/// Test capacity that records allocation intent and lease lifetime.
pub(crate) struct RecordingG2Capacity {
    manager: Arc<BlockManager<G2>>,
    kinds: Mutex<Vec<G2AllocationKind>>,
    allocations: AtomicUsize,
    registrations: AtomicUsize,
    registrations_with_live_lease: AtomicUsize,
    live_leases: Arc<AtomicUsize>,
    lease_drops: Arc<AtomicUsize>,
}

impl G2LeaseGuard for Lease {}

impl Drop for Lease {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl RecordingG2Capacity {
    pub(crate) fn new(manager: Arc<BlockManager<G2>>) -> Self {
        Self {
            manager,
            kinds: Mutex::new(Vec::new()),
            allocations: AtomicUsize::new(0),
            registrations: AtomicUsize::new(0),
            registrations_with_live_lease: AtomicUsize::new(0),
            live_leases: Arc::new(AtomicUsize::new(0)),
            lease_drops: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub(crate) fn allocation_kinds(&self) -> Vec<G2AllocationKind> {
        self.kinds.lock().expect("kinds lock").clone()
    }

    pub(crate) fn allocation_count(&self) -> usize {
        self.allocations.load(Ordering::Relaxed)
    }

    pub(crate) fn registration_count(&self) -> usize {
        self.registrations.load(Ordering::Relaxed)
    }

    pub(crate) fn registrations_with_live_lease(&self) -> usize {
        self.registrations_with_live_lease.load(Ordering::Relaxed)
    }

    pub(crate) fn lease_drop_count(&self) -> usize {
        self.lease_drops.load(Ordering::Relaxed)
    }
}

impl G2Capacity for RecordingG2Capacity {
    fn reserve(&self, request: G2CapacityRequest) -> Result<G2CapacityDecision, G2CapacityError> {
        if request.requirement() == G2CapacityRequirement::ExactReclaim {
            return Err(G2CapacityError::ExactReclaimUnsupported(request));
        }
        self.kinds.lock().expect("kinds lock").push(request.kind());
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.manager
            .allocate_blocks(request.count())
            .map(|blocks| {
                self.live_leases.fetch_add(1, Ordering::Relaxed);
                G2CapacityDecision::Granted(G2Allocation::new(
                    request.kind(),
                    blocks,
                    Arc::new(Lease {
                        live: Arc::clone(&self.live_leases),
                        drops: Arc::clone(&self.lease_drops),
                    }),
                ))
            })
            .ok_or(G2CapacityError::Unavailable(request))
    }

    fn block_size(&self) -> usize {
        self.manager.block_size()
    }

    fn manager_id(&self) -> kvbm_logical::ManagerId {
        self.manager.id()
    }

    fn register_compatibility(
        &self,
        allocation: G2StagedAllocation,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        Ok(allocation.register_with(|blocks| {
            if self.live_leases.load(Ordering::Relaxed) != 0 {
                self.registrations_with_live_lease
                    .fetch_add(1, Ordering::Relaxed);
            }
            self.registrations.fetch_add(1, Ordering::Relaxed);
            self.manager.register_blocks(blocks)
        }))
    }

    fn match_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.manager.match_blocks(hashes)
    }

    fn scan_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<G2>> {
        self.manager.scan_matches(hashes, touch)
    }
}
