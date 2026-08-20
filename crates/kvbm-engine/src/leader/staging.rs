// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared G3→G2 staging logic.
//!
//! Core staging kernel: allocate G2 destinations, transfer G3→G2, register the
//! new G2 blocks. Used by the control-plane `transfer` module and the parked
//! G4/async-search machinery. Each caller handles its own post-staging
//! bookkeeping (updating holders, sending notifications, etc.).

use std::sync::Arc;

use anyhow::Result;

use crate::g2_capacity::{G2Capacity, reserve_required_staging};
use crate::{BlockId, G2, G3, worker::group::ParallelWorkers};
use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_physical::transfer::TransferOptions;

use super::blocks::BlockHolder;

/// Result of staging G3 blocks to G2.
pub struct StagingResult {
    /// Newly created G2 blocks (registered with the G2 manager).
    pub new_g2_blocks: Vec<ImmutableBlock<G2>>,
}

/// Stage G3 blocks to G2.
///
/// Core staging kernel: allocate G2 destinations → execute local transfer (G3→G2)
/// → register new G2 blocks with the source sequence hashes → return new blocks.
///
/// The caller is responsible for:
/// - Clearing the G3 holder (`take_all()`)
/// - Adding new blocks to the G2 holder (`extend()`)
/// - Sending any notifications to peers
pub async fn stage_g3_to_g2(
    g3_blocks: &BlockHolder<G3>,
    g2_capacity: Arc<dyn G2Capacity>,
    parallel_worker: &dyn ParallelWorkers,
) -> Result<StagingResult> {
    if g3_blocks.is_empty() {
        return Ok(StagingResult {
            new_g2_blocks: Vec::new(),
        });
    }

    let src_ids: Vec<BlockId> = g3_blocks.blocks().iter().map(|b| b.block_id()).collect();

    let dst_allocation = reserve_restored_g2_allocation(Arc::clone(&g2_capacity), src_ids.len())?;
    let src_ids: Arc<[BlockId]> = Arc::from(src_ids);
    let dst_allocation = dst_allocation
        .transfer_with(|blocks| async move {
            let dst_ids: Vec<BlockId> = blocks.iter().map(|block| block.block_id()).collect();
            let notification = parallel_worker.execute_local_transfer(
                LogicalLayoutHandle::G3,
                LogicalLayoutHandle::G2,
                src_ids,
                Arc::from(dst_ids),
                TransferOptions::default(),
            )?;
            notification.await?;
            Ok::<_, anyhow::Error>(blocks)
        })
        .await?;

    // Register new G2 blocks using the G3 blocks' metadata (sequence hashes)
    let hashes = g3_blocks
        .blocks()
        .iter()
        .map(|block| block.sequence_hash())
        .collect::<Vec<_>>();
    let new_g2_blocks = dst_allocation
        .stage_all(&hashes, g2_capacity.block_size())?
        .publish()?;

    Ok(StagingResult { new_g2_blocks })
}

fn reserve_restored_g2_allocation(
    g2_capacity: Arc<dyn G2Capacity>,
    count: usize,
) -> Result<crate::g2_capacity::RequiredStagingAllocation> {
    reserve_required_staging(g2_capacity, count)
        .map_err(|error| anyhow::anyhow!("Failed to reserve G2 blocks: {error}"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::g2_capacity::{
        DirectG2Capacity, G2AllocationKind, G2CapacityDecision, G2CapacityError, G2CapacityRequest,
        G2StagedAllocation,
    };
    use crate::testing::managers::TestManagerBuilder;

    struct RestoreCapacity {
        direct: DirectG2Capacity,
        kinds: Mutex<Vec<G2AllocationKind>>,
        requirements: Mutex<Vec<crate::g2_capacity::G2CapacityRequirement>>,
    }

    impl G2Capacity for RestoreCapacity {
        fn reserve(
            &self,
            request: G2CapacityRequest,
        ) -> Result<G2CapacityDecision, G2CapacityError> {
            self.kinds.lock().expect("kinds lock").push(request.kind());
            self.requirements
                .lock()
                .expect("requirements lock")
                .push(request.requirement());
            self.direct.reserve(request)
        }

        fn block_size(&self) -> usize {
            self.direct.block_size()
        }

        fn manager_id(&self) -> kvbm_logical::ManagerId {
            self.direct.manager_id()
        }

        fn register_compatibility(
            &self,
            allocation: G2StagedAllocation,
        ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
            self.direct.register_compatibility(allocation)
        }

        fn match_blocks(&self, hashes: &[crate::SequenceHash]) -> Vec<ImmutableBlock<G2>> {
            self.direct.match_blocks(hashes)
        }

        fn scan_matches(
            &self,
            hashes: &[crate::SequenceHash],
            touch: bool,
        ) -> HashMap<crate::SequenceHash, ImmutableBlock<G2>> {
            self.direct.scan_matches(hashes, touch)
        }
    }

    #[test]
    fn g3_to_g2_restore_uses_required_staging_capacity() {
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(1)
                .block_size(4)
                .build(),
        );
        let capacity = Arc::new(RestoreCapacity {
            direct: DirectG2Capacity::new(Arc::clone(&manager)),
            kinds: Mutex::new(Vec::new()),
            requirements: Mutex::new(Vec::new()),
        });
        let hash = crate::SequenceHash::new(41, None, 0);
        let allocation =
            reserve_restored_g2_allocation(capacity.clone(), 1).expect("restore allocation");
        let registered = allocation
            .stage_all(&[hash], capacity.block_size())
            .expect("stage restore allocation")
            .publish()
            .expect("publish restore allocation");

        assert_eq!(registered.len(), 1);
        assert_eq!(
            *capacity.kinds.lock().expect("kinds lock"),
            vec![
                G2AllocationKind::RequiredStaging,
                G2AllocationKind::RequiredStaging,
            ]
        );
        assert_eq!(
            *capacity.requirements.lock().expect("requirements lock"),
            vec![
                crate::g2_capacity::G2CapacityRequirement::ExactReclaim,
                crate::g2_capacity::G2CapacityRequirement::Compatibility,
            ]
        );
        assert_eq!(manager.match_blocks(&[hash]).len(), 1);
    }
}
