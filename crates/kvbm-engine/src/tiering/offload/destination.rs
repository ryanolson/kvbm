// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Destination adapters for block-manager pipelines.
//!
//! G2 uses [`G2Capacity`] for every destination allocation. G3 retains the
//! direct logical-manager adapter because G2 capacity policy does not apply
//! to G2→G3 offload.

use std::sync::Arc;

use anyhow::{Result, ensure};
use kvbm_logical::blocks::{BlockMetadata, ImmutableBlock, MutableBlock};
use kvbm_logical::manager::BlockManager;

use crate::g2_capacity::{
    DirectG2Capacity, G2Allocation, G2AllocationKind, G2Capacity, G2StagedAllocation,
    reserve_compatibility,
};
use crate::{BlockId, G2, G3, SequenceHash};

/// Private destination input for a block-manager pipeline.
///
/// Construct this type implicitly from `Arc<BlockManager<G2>>`,
/// `Arc<dyn G2Capacity>`, or `Arc<BlockManager<G3>>`. A raw G2 manager maps
/// to [`DirectG2Capacity`] for compatibility.
pub(crate) struct PipelineDestination<Dst: BlockMetadata> {
    destination: Arc<dyn BlockDestination<Dst>>,
}

impl PipelineDestination<G2> {
    /// Use an injected G2 capacity policy for destination allocations.
    pub(crate) fn g2_capacity(capacity: Arc<dyn G2Capacity>) -> Self {
        Self {
            destination: Arc::new(G2CapacityDestination {
                capacity,
                kind: G2AllocationKind::CacheExtension,
            }),
        }
    }
}

impl From<Arc<BlockManager<G2>>> for PipelineDestination<G2> {
    fn from(manager: Arc<BlockManager<G2>>) -> Self {
        Self::g2_capacity(Arc::new(DirectG2Capacity::new(manager)))
    }
}

impl From<Arc<dyn G2Capacity>> for PipelineDestination<G2> {
    fn from(capacity: Arc<dyn G2Capacity>) -> Self {
        Self::g2_capacity(capacity)
    }
}

impl From<Arc<BlockManager<G3>>> for PipelineDestination<G3> {
    fn from(manager: Arc<BlockManager<G3>>) -> Self {
        Self {
            destination: Arc::new(ManagerDestination { manager }),
        }
    }
}

impl<Dst: BlockMetadata> PipelineDestination<Dst> {
    pub(crate) fn into_inner(self) -> Arc<dyn BlockDestination<Dst>> {
        self.destination
    }
}

pub(crate) trait BlockDestination<Dst: BlockMetadata>: Send + Sync {
    fn allocate(&self, count: usize) -> Result<Option<Box<dyn DestinationAllocation<Dst>>>>;
}

pub(crate) trait DestinationAllocation<Dst: BlockMetadata>: Send {
    fn block_ids(&self) -> Vec<BlockId>;
    fn register(self: Box<Self>, hashes: &[SequenceHash]) -> Result<Vec<ImmutableBlock<Dst>>>;
}

struct G2CapacityDestination {
    capacity: Arc<dyn G2Capacity>,
    kind: G2AllocationKind,
}

impl BlockDestination<G2> for G2CapacityDestination {
    fn allocate(&self, count: usize) -> Result<Option<Box<dyn DestinationAllocation<G2>>>> {
        let allocation = reserve_compatibility(self.capacity.as_ref(), self.kind, count)
            .map_err(|error| anyhow::anyhow!("G2 capacity reservation: {error}"))?;
        Ok(Some(Box::new(G2DestinationAllocation {
            allocation,
            capacity: Arc::clone(&self.capacity),
        }) as Box<dyn DestinationAllocation<G2>>))
    }
}

struct G2DestinationAllocation {
    allocation: G2Allocation,
    capacity: Arc<dyn G2Capacity>,
}

impl DestinationAllocation<G2> for G2DestinationAllocation {
    fn block_ids(&self) -> Vec<BlockId> {
        self.allocation.block_ids()
    }

    fn register(self: Box<Self>, hashes: &[SequenceHash]) -> Result<Vec<ImmutableBlock<G2>>> {
        let staged: G2StagedAllocation = self
            .allocation
            .stage_all(hashes, self.capacity.block_size())?;
        Ok(self.capacity.register_compatibility(staged)?)
    }
}

struct ManagerDestination<Dst: BlockMetadata> {
    manager: Arc<BlockManager<Dst>>,
}

impl<Dst: BlockMetadata> BlockDestination<Dst> for ManagerDestination<Dst> {
    fn allocate(&self, count: usize) -> Result<Option<Box<dyn DestinationAllocation<Dst>>>> {
        Ok(self.manager.allocate_blocks(count).map(|blocks| {
            Box::new(ManagerAllocation {
                blocks,
                manager: Arc::clone(&self.manager),
            }) as Box<dyn DestinationAllocation<Dst>>
        }))
    }
}

struct ManagerAllocation<Dst: BlockMetadata> {
    blocks: Vec<MutableBlock<Dst>>,
    manager: Arc<BlockManager<Dst>>,
}

impl<Dst: BlockMetadata> DestinationAllocation<Dst> for ManagerAllocation<Dst> {
    fn block_ids(&self) -> Vec<BlockId> {
        self.blocks.iter().map(MutableBlock::block_id).collect()
    }

    fn register(self: Box<Self>, hashes: &[SequenceHash]) -> Result<Vec<ImmutableBlock<Dst>>> {
        ensure!(
            self.blocks.len() == hashes.len(),
            "destination allocation has {} blocks for {} hashes",
            self.blocks.len(),
            hashes.len()
        );
        let staged = self
            .blocks
            .into_iter()
            .zip(hashes.iter())
            .map(|(block, hash)| {
                block
                    .stage(*hash, self.manager.block_size())
                    .map_err(|error| anyhow::anyhow!("stage destination block: {error:#}"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self.manager.register_blocks(staged))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::g2_capacity::{
        G2CapacityDecision, G2CapacityError, G2CapacityRequest, G2CapacityRequirement, G2LeaseGuard,
    };
    use crate::testing::managers::TestManagerBuilder;

    struct Lease {
        drops: Arc<AtomicUsize>,
    }

    impl G2LeaseGuard for Lease {}

    impl Drop for Lease {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct RecordingCapacity {
        manager: Arc<BlockManager<G2>>,
        kinds: Mutex<Vec<G2AllocationKind>>,
        lease_drops: Arc<AtomicUsize>,
        registration_saw_live_lease: AtomicUsize,
        registration_finished: AtomicUsize,
    }

    impl RecordingCapacity {
        fn new(manager: Arc<BlockManager<G2>>) -> Self {
            Self {
                manager,
                kinds: Mutex::new(Vec::new()),
                lease_drops: Arc::new(AtomicUsize::new(0)),
                registration_saw_live_lease: AtomicUsize::new(0),
                registration_finished: AtomicUsize::new(0),
            }
        }
    }

    impl G2Capacity for RecordingCapacity {
        fn reserve(
            &self,
            request: G2CapacityRequest,
        ) -> Result<G2CapacityDecision, G2CapacityError> {
            if request.requirement() == G2CapacityRequirement::ExactReclaim {
                return Err(G2CapacityError::ExactReclaimUnsupported(request));
            }
            self.kinds.lock().expect("kinds lock").push(request.kind());
            self.manager
                .allocate_blocks(request.count())
                .map(|blocks| {
                    G2CapacityDecision::Granted(G2Allocation::new(
                        request.kind(),
                        blocks,
                        Arc::new(Lease {
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
            if self.lease_drops.load(Ordering::Relaxed) == 0 {
                self.registration_saw_live_lease.store(1, Ordering::Relaxed);
            }
            let registered =
                allocation.register_with(|blocks| self.manager.register_blocks(blocks));
            self.registration_finished
                .store(self.lease_drops.load(Ordering::Relaxed), Ordering::Relaxed);
            Ok(registered)
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

    #[test]
    fn g1_to_g2_destination_uses_cache_extension_and_keeps_lease_through_register() {
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(1)
                .block_size(4)
                .build(),
        );
        let capacity = Arc::new(RecordingCapacity::new(Arc::clone(&manager)));
        let destination = PipelineDestination::<G2>::g2_capacity(capacity.clone()).into_inner();
        let hash = SequenceHash::new(31, None, 0);

        let registered = destination
            .allocate(1)
            .expect("G1 to G2 capacity reservation")
            .expect("G1 to G2 must ask the injected capacity")
            .register(&[hash])
            .expect("G1 to G2 registration");

        assert_eq!(registered.len(), 1);
        assert_eq!(
            *capacity.kinds.lock().expect("kinds lock"),
            vec![G2AllocationKind::CacheExtension]
        );
        assert_eq!(
            capacity.registration_saw_live_lease.load(Ordering::Relaxed),
            1,
            "the capacity lease must survive logical registration"
        );
        assert_eq!(
            capacity.registration_finished.load(Ordering::Relaxed),
            1,
            "the lease must release only after logical registration"
        );
        assert_eq!(manager.match_blocks(&[hash]).len(), 1);
    }
}
