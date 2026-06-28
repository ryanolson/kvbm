// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Collective communication operations for distributed workers.
//!
//! This module provides infrastructure for collective operations needed by
//! replicated data workers. It defines the [`CollectiveOps`] trait and provides
//! multiple implementations:
//!
//! - [`StubCollectiveOps`]: No-op implementation for testing and single-worker scenarios
//! - [`NcclCollectives`]: NCCL-based implementation for GPU collective operations (requires `nccl` feature)
//!
//! # Architecture
//!
//! In MLA (Multi-head Latent Attention) scenarios, G1 KV blocks are replicated
//! across workers while lower-tier blocks are striped across the worker group.
//! The rank that owns a lower-tier block loads it, then broadcasts from a
//! dynamically selected root.
//!
//! ```text
//! Global G2 block N ──→ owner = N % world_size
//! Owner G2 ──→ owner G1 ───broadcast(root=owner)──→ every G1 replica
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use kvbm_engine::collectives::{CollectiveOps, StubCollectiveOps};
//!
//! let collective = StubCollectiveOps::new(events);
//!
//! // Broadcast G1 blocks from rank 0 to all ranks
//! let notification = collective.broadcast(
//!     0,
//!     LogicalLayoutHandle::G1,
//!     LogicalLayoutHandle::G1,
//!     &src_block_ids,
//!     &dst_block_ids,
//!     Some(0..32),
//! )?;
//! notification.await_completion()?;
//! ```

mod stub;

#[cfg(feature = "nccl")]
mod bootstrap;
#[cfg(feature = "nccl")]
mod nccl;
#[cfg(feature = "nccl")]
mod nccl_ffi;

pub use stub::StubCollectiveOps;

#[cfg(feature = "nccl")]
pub use bootstrap::NcclBootstrap;
#[cfg(feature = "nccl")]
pub use nccl::{CudaEventRegistrar, LayoutResolver, NcclCollectives};

use std::ops::Range;

use anyhow::Result;

use crate::BlockId;
use kvbm_common::{LogicalLayoutHandle, LogicalResourceId};
use kvbm_physical::transfer::TransferCompleteNotification;

/// Collective communication operations for distributed workers.
///
/// This trait defines the collective operations needed by replicated data workers
/// to broadcast data across ranks. Implementations may use NCCL, NIXL, or other
/// collective communication libraries.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to allow sharing across threads.
/// NCCL operations are inherently thread-safe when used correctly (one stream
/// per communicator per thread).
pub trait CollectiveOps: Send + Sync {
    /// Broadcast blocks from the selected root to all other ranks.
    ///
    /// This operation transfers the specified blocks from the source layout on
    /// root rank to the destination layout on all other ranks. Optionally, a layer
    /// range can be specified to transfer only a subset of layers (for pipelined
    /// loading).
    ///
    /// # Arguments
    /// * `root_rank` - Rank whose source blocks contain the canonical data
    /// * `src` - The source logical layout (typically G1 on the selected root)
    /// * `dst` - The destination logical layout (typically G1 on all ranks)
    /// * `src_block_ids` - The block IDs to read from on the source
    /// * `dst_block_ids` - The block IDs to write to on the destination
    /// * `layer_range` - Optional range of layers to transfer. If None, all layers are transferred.
    ///
    /// # Returns
    /// A notification that completes when the broadcast is done on all ranks.
    ///
    /// # Synchronization
    ///
    /// This is a collective operation - all ranks must call this method with
    /// the same arguments for the broadcast to complete correctly. The returned
    /// notification signals local completion; global completion is guaranteed
    /// by the collective semantics of the underlying implementation.
    fn broadcast(
        &self,
        root_rank: usize,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: &[BlockId],
        dst_block_ids: &[BlockId],
        layer_range: Option<Range<usize>>,
    ) -> Result<TransferCompleteNotification>;

    /// Broadcast blocks within one logical resource's physical G1 layout.
    fn broadcast_for_resource(
        &self,
        resource: LogicalResourceId,
        root_rank: usize,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: &[BlockId],
        dst_block_ids: &[BlockId],
        layer_range: Option<Range<usize>>,
    ) -> Result<TransferCompleteNotification> {
        if resource != LogicalResourceId::default() {
            anyhow::bail!(
                "collective does not implement broadcasts for non-default resource {resource:?}"
            );
        }
        self.broadcast(
            root_rank,
            src,
            dst,
            src_block_ids,
            dst_block_ids,
            layer_range,
        )
    }

    /// Get the rank of this worker in the collective group.
    fn rank(&self) -> usize;

    /// Get the total number of workers in the collective group.
    fn world_size(&self) -> usize;
}
