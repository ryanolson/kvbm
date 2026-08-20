// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Main offload engine coordinating pipelines.
//!
//! The `OffloadEngine` is a standalone component that manages block offloading
//! between storage tiers (G1→G2, G2→G3, G2→G4).
//!
//! # Example
//! ```ignore
//! use std::sync::Arc;
//! use kvbm_engine::{G1, G2, G3};
//! use kvbm_engine::offload::{
//!     OffloadEngine, PipelineBuilder, PresenceAndLFUFilter, PresenceFilter,
//! };
//!
//! let engine = OffloadEngine::builder(leader.clone())
//!     .with_g3_manager(g3_manager.clone())
//!     .with_g1_to_g2_pipeline(
//!         PipelineBuilder::<G1, G2>::new()
//!             .policy(Arc::new(PresenceFilter::<G1, G2>::new(registry.clone())))
//!             .batch_size(32)
//!             .auto_chain(true)
//!             .build()
//!     )
//!     .with_g2_to_g3_pipeline(
//!         PipelineBuilder::<G2, G3>::new()
//!             .policy(Arc::new(PresenceAndLFUFilter::with_default_threshold(registry.clone())))
//!             .batch_size(64)
//!             .build()
//!     )
//!     .build()?;
//!
//! let mut handle = engine.enqueue_g2_to_g3(blocks)?;
//! let result = handle.wait().await?;
//! ```

use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use tokio::task::JoinHandle;

use crate::g2_capacity::G2Capacity;
use crate::leader::InstanceLeader;
use crate::object::ObjectBlockOps;
use crate::{BlockId, G1, G2, G3};
use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::blocks::BlockMetadata;
use kvbm_logical::manager::BlockManager;

use super::chain_router;
use super::handle::{TransferHandle, TransferId, TransferState};
use super::pipeline::{
    ObjectPipeline, ObjectPipelineConfig, Pipeline, PipelineConfig, RegisterObserver,
};
use super::remote_g4::{self, RemoteG4OffloadRequest};
use super::source::SourceBlocks;

/// Central coordinator for offload pipelines.
///
/// The engine manages multiple pipelines (G1→G2, G2→G3, G1→G3, G2→G4) and
/// provides a unified interface for enqueueing blocks for offload.
///
/// # Storage Tier Model
///
/// - G1→G2: `G2Capacity` destination facade (host memory)
/// - G2→G3: `BlockManager<G3>` destination (disk/NVMe)
/// - G1→G3: `BlockManager<G3>` destination (disk, bypass-host) — used when
///   G2 is intentionally unconfigured (`cache.bypass_host_cache() == true`).
///   Requires GDS support for direct GPU↔disk transfers; the strategy layer
///   selects the direct path when `TransferCapabilities::allow_gds` is set.
/// - G2→G4: ObjectBlockOps destination (object storage like S3)
///
/// # Distributed G2→G4 Offloading
///
/// Use `with_g2_to_g4_pipeline()` when the leader has an `ObjectBlockOps`
/// implementation. The implementation resolves the logical G2 source layout.
///
/// Use `with_enable_remote_g4(true)` when workers own the object upload path.
/// The leader then sends committed G2 blocks through its worker group.
#[allow(dead_code)]
pub struct OffloadEngine {
    /// Reference to the instance leader for transfers
    leader: Arc<InstanceLeader>,
    /// G1→G2 pipeline (BlockManager destination)
    g1_to_g2: Option<Pipeline<G1, G2>>,
    /// G2→G3 pipeline (BlockManager destination)
    g2_to_g3: Option<Pipeline<G2, G3>>,
    /// G1→G3 pipeline (BlockManager destination, host-bypass mode)
    g1_to_g3: Option<Pipeline<G1, G3>>,
    /// G2→G4 pipeline (Object storage destination) - for local mode only
    g2_to_g4: Option<ObjectPipeline<G2>>,
    /// Weak active-transfer tracking with terminal pruning.
    transfers: TransferRegistry,
    /// Chain router task handle (routes G1→G2 output to downstream pipelines)
    _chain_router_handle: Option<JoinHandle<()>>,
    /// Remote G4 offload task handle (for distributed mode)
    _remote_g4_offload_handle: Option<JoinHandle<()>>,
}

impl OffloadEngine {
    /// Create a new builder for the offload engine.
    pub fn builder(leader: Arc<InstanceLeader>) -> OffloadEngineBuilder {
        OffloadEngineBuilder::new(leader)
    }

    /// Enqueue blocks for G1→G2 offload.
    ///
    /// Returns a `TransferHandle` for tracking progress and cancellation.
    pub fn enqueue_g1_to_g2(&self, blocks: impl Into<SourceBlocks<G1>>) -> Result<TransferHandle> {
        let pipeline = self
            .g1_to_g2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G1→G2 pipeline not configured"))?;

        self.enqueue_to_pipeline(pipeline, blocks.into())
    }

    /// Enqueue blocks for G1→G2 offload with a precondition event.
    ///
    /// The precondition event must be satisfied before the batch is processed
    /// by the transfer executor. This enables coordination with worker forward passes.
    ///
    /// Returns a `TransferHandle` for tracking progress and cancellation.
    pub fn enqueue_g1_to_g2_with_precondition(
        &self,
        blocks: impl Into<SourceBlocks<G1>>,
        precondition: Option<velo::EventHandle>,
    ) -> Result<TransferHandle> {
        let pipeline = self
            .g1_to_g2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G1→G2 pipeline not configured"))?;

        self.enqueue_to_pipeline_with_precondition(pipeline, blocks.into(), precondition)
    }

    /// Enqueue blocks for G2→G3 offload.
    ///
    /// Returns a `TransferHandle` for tracking progress and cancellation.
    pub fn enqueue_g2_to_g3(&self, blocks: impl Into<SourceBlocks<G2>>) -> Result<TransferHandle> {
        let pipeline = self
            .g2_to_g3
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G2→G3 pipeline not configured"))?;

        self.enqueue_to_pipeline(pipeline, blocks.into())
    }

    /// Enqueue blocks for G1→G3 direct offload (host-bypass mode).
    ///
    /// Used when `cache.bypass_host_cache()` is true: blocks are written
    /// directly to disk via GDS without staging through host memory.
    ///
    /// Returns a `TransferHandle` for tracking progress and cancellation.
    pub fn enqueue_g1_to_g3(&self, blocks: impl Into<SourceBlocks<G1>>) -> Result<TransferHandle> {
        let pipeline = self
            .g1_to_g3
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G1→G3 pipeline not configured"))?;

        self.enqueue_to_pipeline(pipeline, blocks.into())
    }

    /// Enqueue blocks for G1→G3 direct offload with a precondition event.
    ///
    /// Host-bypass equivalent of `enqueue_g1_to_g2_with_precondition` — used
    /// in scheduler offload when G2 is bypassed.
    pub fn enqueue_g1_to_g3_with_precondition(
        &self,
        blocks: impl Into<SourceBlocks<G1>>,
        precondition: Option<velo::EventHandle>,
    ) -> Result<TransferHandle> {
        let pipeline = self
            .g1_to_g3
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G1→G3 pipeline not configured"))?;

        self.enqueue_to_pipeline_with_precondition(pipeline, blocks.into(), precondition)
    }

    /// Enqueue blocks for G2→G4 offload (object storage).
    ///
    /// Returns a `TransferHandle` for tracking progress and cancellation.
    pub fn enqueue_g2_to_g4(&self, blocks: impl Into<SourceBlocks<G2>>) -> Result<TransferHandle> {
        let pipeline = self
            .g2_to_g4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G2→G4 pipeline not configured"))?;

        self.enqueue_to_object_pipeline(pipeline, blocks.into())
    }

    /// Create transfer state, store it, and return the components needed for enqueueing.
    fn create_transfer<T: BlockMetadata>(
        &self,
        source: &SourceBlocks<T>,
    ) -> (
        TransferId,
        Arc<std::sync::Mutex<TransferState>>,
        TransferHandle,
    ) {
        let input_block_ids = self.extract_block_ids(source);
        let transfer_id = TransferId::new();
        let (state, handle) = TransferState::new(transfer_id, input_block_ids);
        let state = Arc::new(std::sync::Mutex::new(state));
        self.transfers.insert(transfer_id, &state);
        (transfer_id, state, handle)
    }

    /// Internal: enqueue to a specific pipeline.
    fn enqueue_to_pipeline<Src: BlockMetadata, Dst: BlockMetadata>(
        &self,
        pipeline: &Pipeline<Src, Dst>,
        source: SourceBlocks<Src>,
    ) -> Result<TransferHandle> {
        let (transfer_id, state, handle) = self.create_transfer(&source);
        if !pipeline.enqueue(transfer_id, source, state) {
            tracing::warn!("Transfer {} was cancelled before enqueueing", transfer_id);
        }
        Ok(handle)
    }

    /// Internal: enqueue to a specific pipeline with a precondition.
    fn enqueue_to_pipeline_with_precondition<Src: BlockMetadata, Dst: BlockMetadata>(
        &self,
        pipeline: &Pipeline<Src, Dst>,
        source: SourceBlocks<Src>,
        precondition: Option<velo::EventHandle>,
    ) -> Result<TransferHandle> {
        let (transfer_id, state, handle) = self.create_transfer(&source);
        state.lock().unwrap().precondition = precondition;
        if !pipeline.enqueue(transfer_id, source, state) {
            tracing::warn!("Transfer {} was cancelled before enqueueing", transfer_id);
        }
        Ok(handle)
    }

    /// Internal: enqueue to an object pipeline (G2→G4).
    fn enqueue_to_object_pipeline(
        &self,
        pipeline: &ObjectPipeline<G2>,
        source: SourceBlocks<G2>,
    ) -> Result<TransferHandle> {
        let (transfer_id, state, handle) = self.create_transfer(&source);
        if !pipeline.enqueue(transfer_id, source, state) {
            tracing::warn!("Transfer {} was cancelled before enqueueing", transfer_id);
        }
        Ok(handle)
    }

    /// Extract block IDs from source blocks.
    ///
    /// For External/Strong blocks, returns the known block IDs.
    /// For Weak blocks, returns empty vec (IDs determined at upgrade time).
    fn extract_block_ids<T: BlockMetadata>(&self, source: &SourceBlocks<T>) -> Vec<BlockId> {
        match source {
            SourceBlocks::External(blocks) => blocks.iter().map(|b| b.block_id).collect(),
            SourceBlocks::Strong(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
            SourceBlocks::Weak(_) => Vec::new(), // IDs not available without upgrade
        }
    }

    /// Release a completed transfer's resources.
    ///
    /// This is optional - transfers are automatically cleaned up,
    /// but call this to release resources earlier.
    pub fn release_transfer(&self, transfer_id: TransferId) {
        self.transfers.remove(&transfer_id);
    }

    /// Get the number of active transfers.
    pub fn active_transfer_count(&self) -> usize {
        self.transfers.active_count()
    }

    /// Check if G1→G2 pipeline is configured.
    pub fn has_g1_to_g2(&self) -> bool {
        self.g1_to_g2.is_some()
    }

    /// Register an observer on the G1→G2 pipeline. Returns `Err` if the
    /// pipeline isn't configured. The observer fires after each batch's
    /// destination-tier register step. The callback receives the registered
    /// immutable G2 blocks for that batch.
    pub fn add_g1_to_g2_register_observer(&self, observer: RegisterObserver<G2>) -> Result<()> {
        let pipeline = self
            .g1_to_g2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("G1→G2 pipeline not configured"))?;
        pipeline.add_register_observer(observer);
        Ok(())
    }

    /// Check if G2→G3 pipeline is configured.
    pub fn has_g2_to_g3(&self) -> bool {
        self.g2_to_g3.is_some()
    }

    /// Check if G1→G3 pipeline is configured (host-bypass mode).
    pub fn has_g1_to_g3(&self) -> bool {
        self.g1_to_g3.is_some()
    }

    /// Check if G2→G4 pipeline is configured.
    pub fn has_g2_to_g4(&self) -> bool {
        self.g2_to_g4.is_some()
    }
}

#[derive(Default)]
struct TransferRegistry {
    states: DashMap<TransferId, std::sync::Weak<std::sync::Mutex<TransferState>>>,
}

impl TransferRegistry {
    fn insert(&self, transfer_id: TransferId, state: &Arc<std::sync::Mutex<TransferState>>) {
        self.prune();
        self.states.insert(transfer_id, Arc::downgrade(state));
    }

    fn remove(&self, transfer_id: &TransferId) {
        self.states.remove(transfer_id);
    }

    fn active_count(&self) -> usize {
        self.prune();
        self.states.len()
    }

    fn prune(&self) {
        self.states.retain(|_, weak_state| {
            let Some(state) = weak_state.upgrade() else {
                return false;
            };
            let is_terminal = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .status
                .is_terminal();
            !is_terminal
        });
    }
}

/// Builder for OffloadEngine.
pub struct OffloadEngineBuilder {
    leader: Arc<InstanceLeader>,
    g2_capacity: Option<Arc<dyn G2Capacity>>,
    g3_manager: Option<Arc<BlockManager<G3>>>,
    /// Object storage operations for G4 (replaces `BlockManager<G4>`)
    object_ops: Option<Arc<dyn ObjectBlockOps>>,
    g1_to_g2_config: Option<PipelineConfig<G1, G2>>,
    g2_to_g3_config: Option<PipelineConfig<G2, G3>>,
    g1_to_g3_config: Option<PipelineConfig<G1, G3>>,
    /// G2→G4 uses ObjectPipelineConfig (no destination BlockManager)
    g2_to_g4_config: Option<ObjectPipelineConfig<G2>>,
    /// Optional runtime handle override (defaults to leader.runtime())
    runtime: Option<tokio::runtime::Handle>,
    /// Enable remote G4 offloading via workers' ObjectBlockOps (for distributed mode)
    enable_remote_g4: bool,
}

impl OffloadEngineBuilder {
    /// Create a new builder with the given instance leader.
    pub fn new(leader: Arc<InstanceLeader>) -> Self {
        Self {
            leader,
            g2_capacity: None,
            g3_manager: None,
            object_ops: None,
            g1_to_g2_config: None,
            g2_to_g3_config: None,
            g1_to_g3_config: None,
            g2_to_g4_config: None,
            runtime: None,
            enable_remote_g4: false,
        }
    }

    /// Set an explicit runtime handle for spawning pipeline tasks.
    ///
    /// If not set, defaults to `leader.runtime()`. Use this when you need
    /// pipeline tasks to run on a specific runtime (e.g., in tests).
    pub fn with_runtime(mut self, runtime: tokio::runtime::Handle) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Set the G2 destination capacity facade.
    ///
    /// This facade owns G1→G2 admission. If omitted, the builder uses the
    /// leader's primary capacity facade.
    pub fn with_g2_capacity(mut self, capacity: Arc<dyn G2Capacity>) -> Self {
        self.g2_capacity = Some(capacity);
        self
    }

    /// Set the G3 block manager.
    pub fn with_g3_manager(mut self, manager: Arc<BlockManager<G3>>) -> Self {
        self.g3_manager = Some(manager);
        self
    }

    /// Set object storage operations for G4.
    ///
    /// G4 is object storage (S3, MinIO, etc.) and uses `ObjectBlockOps`
    /// instead of a `BlockManager`. This replaces `with_g4_manager`.
    /// The implementation resolves `LogicalLayoutHandle::G2` internally.
    pub fn with_object_ops(mut self, object_ops: Arc<dyn ObjectBlockOps>) -> Self {
        self.object_ops = Some(object_ops);
        self
    }

    /// Configure G1→G2 pipeline.
    pub fn with_g1_to_g2_pipeline(mut self, config: PipelineConfig<G1, G2>) -> Self {
        self.g1_to_g2_config = Some(config);
        self
    }

    /// Configure G2→G3 pipeline.
    pub fn with_g2_to_g3_pipeline(mut self, config: PipelineConfig<G2, G3>) -> Self {
        self.g2_to_g3_config = Some(config);
        self
    }

    /// Configure G1→G3 pipeline (host-bypass mode).
    ///
    /// Use this when G2 is intentionally unconfigured. Requires a G3 manager
    /// (`with_g3_manager`) and GDS-capable transfer capabilities at the worker
    /// side for the direct path to fire.
    pub fn with_g1_to_g3_pipeline(mut self, config: PipelineConfig<G1, G3>) -> Self {
        self.g1_to_g3_config = Some(config);
        self
    }

    /// Configure G2→G4 pipeline (object storage).
    ///
    /// Uses `ObjectPipelineConfig` instead of `PipelineConfig` since G4
    /// is object storage, not a BlockManager destination.
    ///
    /// Use `with_enable_remote_g4(true)` when workers own the object upload path.
    pub fn with_g2_to_g4_pipeline(mut self, config: ObjectPipelineConfig<G2>) -> Self {
        self.g2_to_g4_config = Some(config);
        self
    }

    /// Enable remote G4 offloading via workers' ObjectBlockOps.
    ///
    /// This enables G2→G4 work where:
    /// 1. G1→G2 chain output is routed to a remote offload task
    /// 2. The task calls workers' ObjectBlockOps::put_blocks() via RPC
    /// 3. Workers upload blocks from their local G2 to object storage
    /// 4. Per-block results are returned and logged
    ///
    /// This is mutually exclusive with `with_g2_to_g4_pipeline()` - use one or the other.
    pub fn with_enable_remote_g4(mut self, enable: bool) -> Self {
        self.enable_remote_g4 = enable;
        self
    }

    /// Build the offload engine.
    pub fn build(self) -> Result<OffloadEngine> {
        if self.enable_remote_g4 && self.g2_to_g4_config.is_some() {
            anyhow::bail!("local and remote G2-to-G4 offload modes are mutually exclusive");
        }
        // Get the runtime handle for spawning background tasks
        // Use explicit override if provided, otherwise get from leader
        let runtime = self.runtime.unwrap_or_else(|| self.leader.runtime());

        // Build G1→G2 pipeline if configured
        // Note: G1 is externally owned (vLLM GPU cache), so no G1 manager needed.
        // Pipeline works with ExternalBlock<G1> which contains block_id + sequence_hash.
        let mut g1_to_g2 = if let Some(config) = self.g1_to_g2_config {
            let resource = config
                .options
                .resource
                .unwrap_or_else(|| self.leader.primary_g2_resource());
            let manager = self.leader.g2_manager_for(resource).ok_or_else(|| {
                anyhow::anyhow!("no G2 manager configured for offload resource {resource:?}")
            })?;
            let g2_capacity = match self.g2_capacity.clone() {
                Some(capacity) => capacity,
                None => self.leader.g2_capacity_for(resource).ok_or_else(|| {
                    anyhow::anyhow!("no G2 capacity configured for offload resource {resource:?}")
                })?,
            };
            anyhow::ensure!(
                g2_capacity.manager_id() == manager.id(),
                "G2 capacity manager does not match selected resource {resource:?}"
            );

            Some(Pipeline::new(
                config,
                g2_capacity,
                self.leader.clone(),
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                runtime.clone(),
            )?)
        } else {
            None
        };

        // Build G2→G3 pipeline if configured
        let g2_to_g3 = if let Some(config) = self.g2_to_g3_config {
            let g3_manager = self
                .g3_manager
                .clone()
                .ok_or_else(|| anyhow::anyhow!("G3 manager required for G2→G3 pipeline"))?;

            Some(Pipeline::new(
                config,
                g3_manager,
                self.leader.clone(),
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G3,
                runtime.clone(),
            )?)
        } else {
            None
        };

        // Build G1→G3 pipeline (host-bypass) if configured.
        // Standalone — no chain-router needed since G3 is the terminal tier.
        let g1_to_g3 = if let Some(config) = self.g1_to_g3_config {
            let g3_manager = self
                .g3_manager
                .clone()
                .ok_or_else(|| anyhow::anyhow!("G3 manager required for G1→G3 pipeline"))?;

            Some(Pipeline::new(
                config,
                g3_manager,
                self.leader.clone(),
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G3,
                runtime.clone(),
            )?)
        } else {
            None
        };

        // Build G2→G4 pipeline if configured (object storage destination)
        // Note: For distributed mode, use enable_remote_g4 instead
        let g2_to_g4 = if let Some(config) = self.g2_to_g4_config {
            let object_ops = self
                .object_ops
                .ok_or_else(|| anyhow::anyhow!("ObjectBlockOps required for G2→G4 pipeline"))?;

            // ObjectPipeline takes LogicalLayoutHandle - the ObjectBlockOps implementation
            // resolves this to a physical layout internally
            Some(ObjectPipeline::new(
                config,
                object_ops,
                LogicalLayoutHandle::G2,
                self.leader.clone(),
                runtime.clone(),
            )?)
        } else {
            None
        };

        // Create channel for remote G4 offload if enabled
        let (remote_g4_tx, remote_g4_rx) = if self.enable_remote_g4 {
            let (tx, rx) = tokio::sync::mpsc::channel::<RemoteG4OffloadRequest>(64);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        // Wire up auto-chaining from G1→G2 to downstream G2→G3/G2→G4 pipelines
        let chain_router_handle = if let Some(ref mut g1_to_g2_pipeline) = g1_to_g2 {
            if g1_to_g2_pipeline.auto_chain() {
                if let Some(chain_rx) = g1_to_g2_pipeline.take_chain_rx() {
                    // Get references to downstream pipeline queues
                    let g2_to_g3_queue = g2_to_g3.as_ref().map(Pipeline::ingress);
                    let g2_to_g4_queue = g2_to_g4.as_ref().map(ObjectPipeline::ingress);

                    // Check if we have any downstream target (local pipelines or remote G4)
                    let has_g2_to_g4_local = g2_to_g4_queue.is_some();
                    let has_g2_to_g4_remote = remote_g4_tx.is_some();

                    // Only spawn if there's at least one downstream target
                    if g2_to_g3_queue.is_some() || has_g2_to_g4_local || has_g2_to_g4_remote {
                        tracing::debug!(
                            has_g2_to_g3 = g2_to_g3_queue.is_some(),
                            has_g2_to_g4_local,
                            has_g2_to_g4_remote,
                            "Spawning chain router for G1→G2 auto-chaining"
                        );
                        Some(runtime.spawn(chain_router::run(
                            chain_rx,
                            g2_to_g3_queue,
                            g2_to_g4_queue,
                            remote_g4_tx,
                        )))
                    } else {
                        tracing::debug!(
                            "G1→G2 auto_chain enabled but no downstream pipelines configured"
                        );
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Spawn remote G4 offload task if enabled
        let remote_g4_offload_handle = if let Some(rx) = remote_g4_rx {
            tracing::info!("Enabling remote G4 offload via workers' ObjectBlockOps");
            Some(runtime.spawn(remote_g4::run(rx, self.leader.clone())))
        } else {
            None
        };

        Ok(OffloadEngine {
            leader: self.leader,
            g1_to_g2,
            g2_to_g3,
            g1_to_g3,
            g2_to_g4,
            transfers: TransferRegistry::default(),
            _chain_router_handle: chain_router_handle,
            _remote_g4_offload_handle: remote_g4_offload_handle,
        })
    }
}

#[cfg(test)]
mod tests;
