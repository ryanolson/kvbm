// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure transfer bookkeeping for replicated G1 and striped G2.

use anyhow::{Result, bail};
use kvbm_common::{BlockId, StripedBlockPlacement};

/// Plans owner-only copies and dynamic-root replication without moving bytes.
#[derive(Debug, Clone, Copy)]
pub(super) struct ReplicatedTransferPlanner {
    placement: StripedBlockPlacement,
}

impl ReplicatedTransferPlanner {
    /// Construct a planner for a non-empty TP worker group.
    pub(super) fn new(world_size: usize) -> Result<Self> {
        Ok(Self {
            placement: StripedBlockPlacement::new(world_size)?,
        })
    }

    /// Select this rank's G1→G2 copies and translate G2 IDs to local slots.
    pub(super) fn plan_offload(
        self,
        rank: usize,
        g1_block_ids: &[BlockId],
        global_g2_block_ids: &[BlockId],
    ) -> Result<LocalOffloadPlan> {
        self.validate_pairs(g1_block_ids, global_g2_block_ids)?;
        self.validate_rank(rank)?;

        let mut pairs = g1_block_ids
            .iter()
            .copied()
            .zip(global_g2_block_ids.iter().copied())
            .filter_map(|(g1, global_g2)| {
                let (owner, local_g2) = self.placement.resolve(global_g2);
                (owner == rank).then_some((local_g2, g1))
            })
            .collect::<Vec<_>>();
        pairs.sort_unstable_by_key(|(local_g2, _)| *local_g2);

        Ok(LocalOffloadPlan {
            g1_block_ids: pairs.iter().map(|(_, g1)| *g1).collect(),
            local_g2_block_ids: pairs.iter().map(|(g2, _)| *g2).collect(),
        })
    }

    /// Group G2→G1 copies by owner. Every rank derives the same rank-ordered
    /// batches and therefore enters broadcasts in the same order.
    pub(super) fn plan_onboard(
        self,
        global_g2_block_ids: &[BlockId],
        g1_block_ids: &[BlockId],
    ) -> Result<Vec<ReplicaOnboardPlan>> {
        self.validate_pairs(global_g2_block_ids, g1_block_ids)?;

        let mut by_owner = vec![Vec::new(); self.placement.world_size()];
        for (global_g2, g1) in global_g2_block_ids
            .iter()
            .copied()
            .zip(g1_block_ids.iter().copied())
        {
            let (owner, local_g2) = self.placement.resolve(global_g2);
            by_owner[owner].push((local_g2, g1));
        }

        Ok(by_owner
            .into_iter()
            .enumerate()
            .filter_map(|(root_rank, mut pairs)| {
                if pairs.is_empty() {
                    return None;
                }
                pairs.sort_unstable_by_key(|(local_g2, _)| *local_g2);
                Some(ReplicaOnboardPlan {
                    root_rank,
                    local_g2_block_ids: pairs.iter().map(|(g2, _)| *g2).collect(),
                    g1_block_ids: pairs.iter().map(|(_, g1)| *g1).collect(),
                })
            })
            .collect())
    }

    fn validate_pairs(self, left: &[BlockId], right: &[BlockId]) -> Result<()> {
        if left.len() != right.len() {
            bail!(
                "replicated transfer requires equal block-ID counts, got {} and {}",
                left.len(),
                right.len()
            );
        }
        Ok(())
    }

    fn validate_rank(self, rank: usize) -> Result<()> {
        if rank >= self.placement.world_size() {
            bail!(
                "rank {rank} is outside replicated worker group of size {}",
                self.placement.world_size()
            );
        }
        Ok(())
    }
}

/// Rank-local G1→G2 work for one logical offload request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LocalOffloadPlan {
    g1_block_ids: Vec<BlockId>,
    local_g2_block_ids: Vec<BlockId>,
}

impl LocalOffloadPlan {
    pub(super) fn g1_block_ids(&self) -> &[BlockId] {
        &self.g1_block_ids
    }

    pub(super) fn local_g2_block_ids(&self) -> &[BlockId] {
        &self.local_g2_block_ids
    }
}

/// One owner copy followed by a broadcast from `root_rank` to all replicas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ReplicaOnboardPlan {
    root_rank: usize,
    local_g2_block_ids: Vec<BlockId>,
    g1_block_ids: Vec<BlockId>,
}

impl ReplicaOnboardPlan {
    pub(super) fn root_rank(&self) -> usize {
        self.root_rank
    }

    pub(super) fn local_g2_block_ids(&self) -> &[BlockId] {
        &self.local_g2_block_ids
    }

    pub(super) fn g1_block_ids(&self) -> &[BlockId] {
        &self.g1_block_ids
    }
}

#[cfg(test)]
mod tests {
    use super::ReplicatedTransferPlanner;

    #[test]
    fn offload_selects_only_the_local_g2_stripe() {
        let planner = ReplicatedTransferPlanner::new(2).expect("valid TP size");
        let global_g2 = [0, 1, 2, 3, 4];
        let replicated_g1 = [10, 11, 12, 13, 14];

        let rank0 = planner
            .plan_offload(0, &replicated_g1, &global_g2)
            .expect("rank 0 plan");
        assert_eq!(rank0.g1_block_ids(), &[10, 12, 14]);
        assert_eq!(rank0.local_g2_block_ids(), &[0, 1, 2]);

        let rank1 = planner
            .plan_offload(1, &replicated_g1, &global_g2)
            .expect("rank 1 plan");
        assert_eq!(rank1.g1_block_ids(), &[11, 13]);
        assert_eq!(rank1.local_g2_block_ids(), &[0, 1]);
    }

    #[test]
    fn onboard_groups_dynamic_broadcast_roots_in_collective_order() {
        let planner = ReplicatedTransferPlanner::new(2).expect("valid TP size");
        let plans = planner
            .plan_onboard(&[3, 0, 2, 1], &[23, 20, 22, 21])
            .expect("onboard plan");

        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].root_rank(), 0);
        assert_eq!(plans[0].local_g2_block_ids(), &[0, 1]);
        assert_eq!(plans[0].g1_block_ids(), &[20, 22]);
        assert_eq!(plans[1].root_rank(), 1);
        assert_eq!(plans[1].local_g2_block_ids(), &[0, 1]);
        assert_eq!(plans[1].g1_block_ids(), &[21, 23]);
    }

    #[test]
    fn planner_rejects_mismatched_pairs_and_invalid_rank() {
        let planner = ReplicatedTransferPlanner::new(2).expect("valid TP size");
        assert!(planner.plan_onboard(&[0], &[10, 11]).is_err());
        assert!(planner.plan_offload(2, &[10], &[0]).is_err());
    }
}
