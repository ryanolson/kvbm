// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logical-to-physical placement for replicated KV caches.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// Deterministically stripes canonical blocks across a worker group.
///
/// A logical [`crate::BlockId`] remains global and continues to be keyed by
/// its [`crate::SequenceHash`]. The placement only translates that global ID
/// to the one worker and worker-local slot that physically owns the lower-tier
/// copy. No per-block placement table is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StripedBlockPlacement {
    world_size: usize,
}

impl StripedBlockPlacement {
    /// Build a placement for a non-empty worker group.
    pub fn new(world_size: usize) -> Result<Self> {
        if world_size == 0 {
            bail!("striped block placement requires at least one worker");
        }
        Ok(Self { world_size })
    }

    /// Number of workers contributing physical lower-tier capacity.
    pub fn world_size(self) -> usize {
        self.world_size
    }

    /// Resolve a global block ID to `(owner_rank, owner_local_block_id)`.
    pub fn resolve(self, global_block_id: crate::BlockId) -> (usize, crate::BlockId) {
        (
            global_block_id % self.world_size,
            global_block_id / self.world_size,
        )
    }

    /// Reconstruct a global block ID from its owner and worker-local slot.
    pub fn global(
        self,
        owner_rank: usize,
        local_block_id: crate::BlockId,
    ) -> Result<crate::BlockId> {
        self.validate_rank(owner_rank)?;
        local_block_id
            .checked_mul(self.world_size)
            .and_then(|base| base.checked_add(owner_rank))
            .ok_or_else(|| anyhow::anyhow!("striped block ID overflow"))
    }

    /// Logical capacity contributed by equal physical storage on every rank.
    pub fn global_capacity(self, per_rank_capacity: usize) -> Result<usize> {
        per_rank_capacity
            .checked_mul(self.world_size)
            .ok_or_else(|| anyhow::anyhow!("striped block capacity overflow"))
    }

    /// Physical slots required on one rank for a global logical capacity.
    pub fn local_capacity(self, global_capacity: usize, rank: usize) -> Result<usize> {
        self.validate_rank(rank)?;
        if rank >= global_capacity {
            return Ok(0);
        }
        Ok(1 + (global_capacity - 1 - rank) / self.world_size)
    }

    fn validate_rank(self, rank: usize) -> Result<()> {
        if rank >= self.world_size {
            bail!(
                "rank {rank} is outside striped worker group of size {}",
                self.world_size
            );
        }
        Ok(())
    }
}
