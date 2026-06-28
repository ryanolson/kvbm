// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replicated data worker for MLA (Multi-head Latent Attention) scenarios.
//!
//! In MLA architectures, G1 KV blocks are replicated across all workers rather
//! than sharded. Lower tiers are striped across the worker group so each logical
//! block has exactly one lower-tier owner.
//!
//! # Architecture
//!
//! ```text
//! Global G2 block N ──→ owner = N % world_size, local = N / world_size
//! Owner G2          ──→ owner G1 ───broadcast(root=owner)──→ every G1 replica
//! ```
//!
//! # Transfer Semantics
//!
//! | Operation | Behavior |
//! |-----------|----------|
//! | G2 → G1 (onboard) | Each G2 owner transfers its batch, then broadcasts from that owner |
//! | G1 → G2 (offload) | Each rank writes only the global G2 blocks it owns |
//! | G2 ↔ G3 | Not yet supported by this worker |
//! | G1 → G1 (local) | All ranks execute (data is replicated) |

mod planner;

use planner::ReplicatedTransferPlanner;

use super::*;

use crate::KvbmRuntime;
use crate::collectives::CollectiveOps;
use anyhow::{Context, Result, bail, ensure};

use std::sync::Arc;

/// Replicated data worker for MLA scenarios.
///
/// G1 is replicated on every rank while G2 is striped across ranks. When loading
/// data to G1, each G2 owner transfers its local batch and broadcasts that batch
/// to the same G1 block IDs on every other rank.
///
/// # Requirements
///
/// - Every worker must have equal-sized local G2 storage
/// - Replicated G1 allocators must assign the same destination IDs on every rank
/// - A [`CollectiveOps`] implementation must be provided for broadcasting
///
/// # Trait Implementations
///
/// - [`WorkerTransfers`]: Specialized routing based on source/destination tiers
pub struct ReplicatedDataWorker {
    inner: Arc<PhysicalWorker>,
    runtime: Arc<KvbmRuntime>,
    collective: Arc<dyn CollectiveOps>,
    planner: ReplicatedTransferPlanner,
    rank: usize,
}

impl ReplicatedDataWorker {
    /// Create a new ReplicatedDataWorker.
    ///
    /// # Arguments
    /// * `worker` - The rank-local physical worker with G1 and G2 layouts
    /// * `runtime` - Runtime used to sequence owner copies and collectives
    /// * `collective` - The collective ops implementation for broadcasting
    pub fn new(
        worker: Arc<PhysicalWorker>,
        runtime: Arc<KvbmRuntime>,
        collective: Arc<dyn CollectiveOps>,
    ) -> Result<Self> {
        let rank = worker
            .rank()
            .context("replicated data worker requires a physical worker rank")?;
        ensure!(
            rank == collective.rank(),
            "physical worker rank {rank} does not match collective rank {}",
            collective.rank()
        );
        let planner = ReplicatedTransferPlanner::new(collective.world_size())?;

        Ok(Self {
            inner: worker,
            runtime,
            collective,
            planner,
            rank,
        })
    }

    /// Get access to the underlying SpmdWorker.
    pub fn inner(&self) -> &PhysicalWorker {
        &self.inner
    }

    /// Get the rank of the underlying worker.
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl WorkerTransfers for ReplicatedDataWorker {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let resource = self
            .inner
            .resource_handles()
            .map(ResourceLayoutHandles::primary)
            .unwrap_or_default();
        self.execute_local_transfer_for_resource(
            resource,
            src,
            dst,
            src_block_ids,
            dst_block_ids,
            options,
        )
    }

    fn execute_local_transfer_for_resource(
        &self,
        resource: LogicalResourceId,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        match (src, dst) {
            (LogicalLayoutHandle::G1, LogicalLayoutHandle::G1) => {
                self.inner.execute_local_transfer_for_resource(
                    resource,
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    options,
                )
            }
            (LogicalLayoutHandle::G1, LogicalLayoutHandle::G2) => {
                let plan = self.planner.plan_offload(
                    self.rank(),
                    src_block_ids.as_ref(),
                    dst_block_ids.as_ref(),
                )?;
                if plan.g1_block_ids().is_empty() {
                    return Ok(TransferCompleteNotification::completed());
                }

                self.inner.execute_local_transfer_for_resource(
                    resource,
                    src,
                    dst,
                    Arc::from(plan.g1_block_ids()),
                    Arc::from(plan.local_g2_block_ids()),
                    options,
                )
            }
            (LogicalLayoutHandle::G2, LogicalLayoutHandle::G1) => {
                let plans = self
                    .planner
                    .plan_onboard(src_block_ids.as_ref(), dst_block_ids.as_ref())?;
                if plans.is_empty() {
                    return Ok(TransferCompleteNotification::completed());
                }

                let event_system = self.runtime.event_system();
                let event = event_system.new_event()?;
                let awaiter = event_system.awaiter(event.handle())?;
                let inner = Arc::clone(&self.inner);
                let collective = Arc::clone(&self.collective);
                let rank = self.rank();
                let layer_range = options.layer_range.clone();

                self.runtime.tokio().spawn(async move {
                    let result = execute_onboard_plans(
                        inner,
                        collective,
                        rank,
                        resource,
                        plans,
                        options,
                        layer_range,
                    )
                    .await;
                    match result {
                        Ok(()) => {
                            let _ = event.trigger();
                        }
                        Err(error) => {
                            let _ = event.poison(error.to_string());
                        }
                    }
                });

                Ok(TransferCompleteNotification::from_awaiter(awaiter))
            }
            _ => bail!(
                "replicated data worker does not yet support local transfer {src:?} -> {dst:?}"
            ),
        }
    }

    #[expect(unused_variables)]
    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!("replicated data worker remote onboard is not yet implemented")
    }

    #[expect(unused_variables)]
    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst: RemoteDescriptor,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!("replicated data worker remote offload is not yet implemented")
    }

    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        // Use the shared implementation
        self.inner.connect_remote(instance_id, metadata)
    }

    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool {
        self.inner.has_remote_metadata(instance_id)
    }

    #[expect(unused_variables)]
    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!("replicated data worker instance remote onboard is not yet implemented")
    }

    fn execute_remote_pull_plan(
        &self,
        plan: crate::leader::dispatch::WorkerPullPlan,
    ) -> Result<TransferCompleteNotification> {
        ensure!(
            plan.source_layout == LogicalLayoutHandle::G2
                && plan.dst_layout == LogicalLayoutHandle::G2,
            "replicated remote pull must land once in striped G2; got {:?} -> {:?}",
            plan.source_layout,
            plan.dst_layout
        );
        self.inner.execute_remote_pull_plan(plan)
    }
}

impl Worker for ReplicatedDataWorker {
    fn g1_handle(&self) -> Option<LayoutHandle> {
        self.inner.g1_handle()
    }

    fn g2_handle(&self) -> Option<LayoutHandle> {
        self.inner.g2_handle()
    }

    fn g3_handle(&self) -> Option<LayoutHandle> {
        self.inner.g3_handle()
    }

    fn export_metadata(&self) -> Result<SerializedLayoutResponse> {
        Worker::export_metadata(self.inner.as_ref())
    }

    fn import_metadata(&self, metadata: SerializedLayout) -> Result<ImportMetadataResponse> {
        Worker::import_metadata(self.inner.as_ref(), metadata)
    }
}

impl ObjectBlockOps for ReplicatedDataWorker {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        ObjectBlockOps::has_blocks(self.inner.as_ref(), keys)
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        src_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        ObjectBlockOps::put_blocks(self.inner.as_ref(), keys, src_layout, block_ids)
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        dst_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        ObjectBlockOps::get_blocks(self.inner.as_ref(), keys, dst_layout, block_ids)
    }
}

async fn execute_onboard_plans(
    inner: Arc<PhysicalWorker>,
    collective: Arc<dyn CollectiveOps>,
    rank: usize,
    resource: LogicalResourceId,
    plans: Vec<planner::ReplicaOnboardPlan>,
    options: kvbm_physical::transfer::TransferOptions,
    layer_range: Option<std::ops::Range<usize>>,
) -> Result<()> {
    for plan in plans {
        let g1_block_ids: Arc<[BlockId]> = Arc::from(plan.g1_block_ids());

        if rank == plan.root_rank() {
            inner
                .execute_local_transfer_for_resource(
                    resource,
                    LogicalLayoutHandle::G2,
                    LogicalLayoutHandle::G1,
                    Arc::from(plan.local_g2_block_ids()),
                    Arc::clone(&g1_block_ids),
                    options.clone(),
                )?
                .await
                .with_context(|| {
                    format!("rank {rank} failed to load its striped G2 batch before broadcast")
                })?;
        }

        collective
            .broadcast_for_resource(
                resource,
                plan.root_rank(),
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G1,
                g1_block_ids.as_ref(),
                g1_block_ids.as_ref(),
                layer_range.clone(),
            )?
            .await
            .with_context(|| {
                format!(
                    "replicated G1 broadcast from rank {} failed",
                    plan.root_rank()
                )
            })?;
    }

    Ok(())
}

#[cfg(test)]
mod trait_tests {
    use super::*;

    fn assert_worker<T: Worker>() {}

    #[test]
    fn replicated_data_policy_is_a_complete_worker() {
        assert_worker::<ReplicatedDataWorker>();
    }
}
