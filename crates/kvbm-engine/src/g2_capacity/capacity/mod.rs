// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capacity adapters and resource routing.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::ManagerId;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;
use kvbm_logical::resources::BlockManagerSet;

use super::{
    G2Allocation, G2AllocationKind, G2CapacityRequest, G2CapacityRequirement, G2ExactAllocation,
    G2PendingReclaim, G2StagedAllocation,
};
use crate::G2;

/// Narrow G2 capacity interface for destination routes.
///
/// Compatibility routes use [`Self::register_compatibility`]. Exact
/// allocations use their source-bound registration owner.
pub trait G2Capacity: Send + Sync {
    /// Reserve G2 destination capacity for one explicit request.
    fn reserve(&self, request: G2CapacityRequest) -> Result<G2CapacityDecision, G2CapacityError>;

    /// Return the fixed G2 block size.
    fn block_size(&self) -> usize;

    /// Return the exact logical manager that owns destination block IDs.
    fn manager_id(&self) -> ManagerId;

    /// Register a compatibility allocation while its lease remains held.
    fn register_compatibility(
        &self,
        allocation: G2StagedAllocation,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError>;

    /// Find matching registered G2 blocks.
    fn match_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>>;

    /// Find matching registered G2 blocks with optional access tracking.
    fn scan_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<G2>>;
}

/// Result of a capacity reservation.
pub enum G2CapacityDecision {
    /// Compatibility capacity is ready.
    Granted(G2Allocation),
    /// Exact capacity and its registration owner are ready.
    ExactGranted(G2ExactAllocation),
    /// The adapter must finish an exact reclaim first.
    PendingReclaim(G2PendingReclaim),
}

/// Capacity errors that do not consume a successful allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum G2CapacityError {
    /// The compatibility manager has no destination slots.
    Unavailable(G2CapacityRequest),
    /// The adapter does not provide exact-reclaim authority.
    ExactReclaimUnsupported(G2CapacityRequest),
    /// A compatibility route received a pending reclaim.
    PendingReclaim { target_count: usize },
    /// The adapter rejected an operation without capacity.
    Rejected(String),
    /// An allocation route received the wrong requirement.
    RequirementMismatch {
        /// The requirement that the route needs.
        expected: G2CapacityRequirement,
        /// The requirement that the capacity returned.
        actual: G2CapacityRequirement,
    },
}

/// Resource-keyed G2 capacity policies.
pub struct G2CapacitySet {
    capacities: BTreeMap<LogicalResourceId, Arc<dyn G2Capacity>>,
}

/// Compatibility-only capacity backed by a logical G2 manager.
///
/// The facade does not expose its manager.
///
/// ```compile_fail
/// use kvbm_engine::g2_capacity::DirectG2Capacity;
///
/// fn inspect(capacity: &DirectG2Capacity) {
///     let _ = capacity.manager();
/// }
/// ```
#[derive(Clone)]
pub struct DirectG2Capacity {
    manager: Arc<BlockManager<G2>>,
}

impl fmt::Display for G2CapacityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(request) => write!(
                formatter,
                "G2 capacity is unavailable for {:?} request of {} blocks",
                request.kind(),
                request.count()
            ),
            Self::ExactReclaimUnsupported(request) => write!(
                formatter,
                "G2 capacity does not support exact reclaim for {:?} request of {} blocks",
                request.kind(),
                request.count()
            ),
            Self::PendingReclaim { target_count } => write!(
                formatter,
                "G2 route needs completion for {target_count} exact reclaim targets"
            ),
            Self::Rejected(reason) => write!(formatter, "G2 capacity rejected request: {reason}"),
            Self::RequirementMismatch { expected, actual } => write!(
                formatter,
                "G2 capacity returned {actual:?} capacity for a {expected:?} route"
            ),
        }
    }
}

impl std::error::Error for G2CapacityError {}

impl G2CapacitySet {
    /// Create an empty resource-keyed capacity set.
    pub fn new() -> Self {
        Self {
            capacities: BTreeMap::new(),
        }
    }

    /// Insert or replace one resource capacity policy.
    pub fn insert(
        &mut self,
        resource: LogicalResourceId,
        capacity: Arc<dyn G2Capacity>,
    ) -> Option<Arc<dyn G2Capacity>> {
        self.capacities.insert(resource, capacity)
    }

    /// Return one resource capacity policy.
    pub fn get(&self, resource: LogicalResourceId) -> Option<&Arc<dyn G2Capacity>> {
        self.capacities.get(&resource)
    }

    /// Iterate policies in resource order.
    pub fn iter(&self) -> impl Iterator<Item = (LogicalResourceId, &Arc<dyn G2Capacity>)> + '_ {
        self.capacities
            .iter()
            .map(|(&resource, capacity)| (resource, capacity))
    }

    /// Return the number of resource policies.
    pub fn len(&self) -> usize {
        self.capacities.len()
    }

    /// Return true when this set has no policies.
    pub fn is_empty(&self) -> bool {
        self.capacities.is_empty()
    }
}

impl Default for G2CapacitySet {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectG2Capacity {
    /// Wrap an existing logical G2 manager.
    pub fn new(manager: Arc<BlockManager<G2>>) -> Self {
        Self { manager }
    }
}

impl G2Capacity for DirectG2Capacity {
    fn reserve(&self, request: G2CapacityRequest) -> Result<G2CapacityDecision, G2CapacityError> {
        if request.requirement() == G2CapacityRequirement::ExactReclaim {
            return Err(G2CapacityError::ExactReclaimUnsupported(request));
        }
        self.manager
            .allocate_blocks(request.count())
            .map(|blocks| G2CapacityDecision::Granted(G2Allocation::direct(request.kind(), blocks)))
            .ok_or(G2CapacityError::Unavailable(request))
    }

    fn block_size(&self) -> usize {
        self.manager.block_size()
    }

    fn manager_id(&self) -> ManagerId {
        self.manager.id()
    }

    fn register_compatibility(
        &self,
        allocation: G2StagedAllocation,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        allocation
            .register_with(|blocks| self.manager.try_register_blocks(blocks))
            .map_err(|_| {
                G2CapacityError::Rejected(
                    "compatibility registration blocks belong to another G2 manager".to_string(),
                )
            })
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

/// Build the default compatibility facade for one G2 manager.
pub fn direct_g2_capacity(manager: Arc<BlockManager<G2>>) -> Arc<dyn G2Capacity> {
    Arc::new(DirectG2Capacity::new(manager))
}

/// Build direct capacity facades for each G2 manager.
pub fn direct_g2_capacity_set(managers: &BlockManagerSet<G2>) -> G2CapacitySet {
    let mut capacities = G2CapacitySet::new();
    for (resource, manager) in managers.iter() {
        capacities.insert(resource, direct_g2_capacity(Arc::clone(manager)));
    }
    capacities
}

/// Reserve immediate capacity for a compatibility route.
pub fn reserve_compatibility(
    capacity: &dyn G2Capacity,
    kind: G2AllocationKind,
    count: usize,
) -> Result<G2Allocation, G2CapacityError> {
    match capacity.reserve(G2CapacityRequest::compatibility(kind, count))? {
        G2CapacityDecision::Granted(allocation) => Ok(allocation),
        G2CapacityDecision::ExactGranted(_) => Err(G2CapacityError::RequirementMismatch {
            expected: G2CapacityRequirement::Compatibility,
            actual: G2CapacityRequirement::ExactReclaim,
        }),
        G2CapacityDecision::PendingReclaim(pending) => Err(G2CapacityError::PendingReclaim {
            target_count: pending.plan().target_count(),
        }),
    }
}
