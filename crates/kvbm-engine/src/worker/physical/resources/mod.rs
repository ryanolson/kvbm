// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-resource selection of tensor-sharded or replicated transfer behavior.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use futures::future::BoxFuture;
use kvbm_common::{LogicalLayoutHandle, LogicalResourceId};
use kvbm_physical::manager::{LayoutHandle, SerializedLayout, WorkerDataPlacement};
use kvbm_physical::transfer::{TransferCompleteNotification, TransferOptions};

use crate::KvbmRuntime;
use crate::collectives::CollectiveOps;
use crate::leader::dispatch::WorkerPullPlan;
use crate::object::ObjectBlockOps;
use crate::worker::{
    BlockId, ConnectRemoteResponse, ImportMetadataResponse, InstanceId, RemoteDescriptor,
    SequenceHash, SerializedLayoutResponse, Worker, WorkerTransfers,
};

use super::{PhysicalWorker, ReplicatedDataWorker};

/// One physical worker whose local transfer policy is selected by resource.
pub struct ResourceDispatchWorker {
    inner: Arc<PhysicalWorker>,
    replicated: Option<ReplicatedDataWorker>,
    placements: ResourceTransferPlacements,
}

impl ResourceDispatchWorker {
    /// Build a dispatcher over a resource-owned physical worker.
    ///
    /// A collective is required when any resource uses replicated G1 with a
    /// striped lower tier. Tensor-sharded-only workers do not need one.
    pub fn new(
        worker: Arc<PhysicalWorker>,
        runtime: Arc<KvbmRuntime>,
        collective: Option<Arc<dyn CollectiveOps>>,
        placements: Vec<(LogicalResourceId, WorkerDataPlacement)>,
    ) -> Result<Self> {
        let primary = worker
            .resource_handles()
            .map(|handles| handles.primary())
            .unwrap_or_default();
        let placements = ResourceTransferPlacements::new(primary, placements)?;
        if let Some(handles) = worker.resource_handles() {
            let physical_resources = handles
                .iter()
                .map(|(resource, _)| resource)
                .collect::<Vec<_>>();
            ensure!(
                physical_resources == placements.resources(),
                "resource transfer placements do not match physical resource handles"
            );
        } else {
            ensure!(
                placements.resources() == vec![LogicalResourceId::default()],
                "legacy physical worker can dispatch only resource zero"
            );
        }

        let replicated = if placements.has_replicated() {
            let collective = collective
                .context("replicated resource transfer placement requires collective operations")?;
            Some(ReplicatedDataWorker::new(
                Arc::clone(&worker),
                runtime,
                collective,
            )?)
        } else {
            None
        };

        Ok(Self {
            inner: worker,
            replicated,
            placements,
        })
    }

    fn placement(&self, resource: LogicalResourceId) -> Result<WorkerDataPlacement> {
        self.placements
            .get(resource)
            .ok_or_else(|| anyhow::anyhow!("no local transfer placement for resource {resource:?}"))
    }

    fn replicated(&self) -> Result<&ReplicatedDataWorker> {
        self.replicated
            .as_ref()
            .context("replicated transfer policy was not constructed")
    }
}

impl WorkerTransfers for ResourceDispatchWorker {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.execute_local_transfer_for_resource(
            self.placements.primary(),
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
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        match self.placement(resource)? {
            WorkerDataPlacement::TensorSharded => self.inner.execute_local_transfer_for_resource(
                resource,
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                options,
            ),
            WorkerDataPlacement::ReplicatedG1StripedLower => {
                self.replicated()?.execute_local_transfer_for_resource(
                    resource,
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    options,
                )
            }
        }
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        match self.placement(self.placements.primary())? {
            WorkerDataPlacement::TensorSharded => {
                self.inner
                    .execute_remote_onboard(src, dst, dst_block_ids, options)
            }
            WorkerDataPlacement::ReplicatedG1StripedLower => self
                .replicated()?
                .execute_remote_onboard(src, dst, dst_block_ids, options),
        }
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst: RemoteDescriptor,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        match self.placement(self.placements.primary())? {
            WorkerDataPlacement::TensorSharded => {
                self.inner
                    .execute_remote_offload(src, src_block_ids, dst, options)
            }
            WorkerDataPlacement::ReplicatedG1StripedLower => self
                .replicated()?
                .execute_remote_offload(src, src_block_ids, dst, options),
        }
    }

    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        self.inner.connect_remote(instance_id, metadata)
    }

    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool {
        self.inner.has_remote_metadata(instance_id)
    }

    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        match self.placement(self.placements.primary())? {
            WorkerDataPlacement::TensorSharded => self.inner.execute_remote_onboard_for_instance(
                instance_id,
                remote_logical_type,
                src_block_ids,
                dst,
                dst_block_ids,
                options,
            ),
            WorkerDataPlacement::ReplicatedG1StripedLower => {
                self.replicated()?.execute_remote_onboard_for_instance(
                    instance_id,
                    remote_logical_type,
                    src_block_ids,
                    dst,
                    dst_block_ids,
                    options,
                )
            }
        }
    }

    fn execute_remote_onboard_for_instance_rank(
        &self,
        instance_id: InstanceId,
        remote_rank: usize,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        match self.placement(self.placements.primary())? {
            WorkerDataPlacement::TensorSharded => {
                self.inner.execute_remote_onboard_for_instance_rank(
                    instance_id,
                    remote_rank,
                    remote_logical_type,
                    src_block_ids,
                    dst,
                    dst_block_ids,
                    options,
                )
            }
            WorkerDataPlacement::ReplicatedG1StripedLower => {
                self.replicated()?.execute_remote_onboard_for_instance_rank(
                    instance_id,
                    remote_rank,
                    remote_logical_type,
                    src_block_ids,
                    dst,
                    dst_block_ids,
                    options,
                )
            }
        }
    }

    fn execute_remote_pull_plan(
        &self,
        plan: WorkerPullPlan,
    ) -> Result<TransferCompleteNotification> {
        match self.placement(plan.dst_resource)? {
            WorkerDataPlacement::TensorSharded => self.inner.execute_remote_pull_plan(plan),
            WorkerDataPlacement::ReplicatedG1StripedLower => {
                self.replicated()?.execute_remote_pull_plan(plan)
            }
        }
    }
}

impl Worker for ResourceDispatchWorker {
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

impl ObjectBlockOps for ResourceDispatchWorker {
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

struct ResourceTransferPlacements {
    primary: LogicalResourceId,
    placements: BTreeMap<LogicalResourceId, WorkerDataPlacement>,
}

impl ResourceTransferPlacements {
    fn new(
        primary: LogicalResourceId,
        placements: Vec<(LogicalResourceId, WorkerDataPlacement)>,
    ) -> Result<Self> {
        let expected_len = placements.len();
        let placements = placements.into_iter().collect::<BTreeMap<_, _>>();
        ensure!(
            placements.len() == expected_len,
            "duplicate resource transfer placement"
        );
        ensure!(
            placements.contains_key(&primary),
            "primary resource {primary:?} has no transfer placement"
        );
        Ok(Self {
            primary,
            placements,
        })
    }

    fn primary(&self) -> LogicalResourceId {
        self.primary
    }

    fn get(&self, resource: LogicalResourceId) -> Option<WorkerDataPlacement> {
        self.placements.get(&resource).copied()
    }

    fn resources(&self) -> Vec<LogicalResourceId> {
        self.placements.keys().copied().collect()
    }

    fn has_replicated(&self) -> bool {
        self.placements
            .values()
            .any(|placement| *placement == WorkerDataPlacement::ReplicatedG1StripedLower)
    }
}

#[cfg(test)]
mod tests {
    use kvbm_common::LogicalResourceId;
    use kvbm_physical::manager::WorkerDataPlacement;

    use super::ResourceTransferPlacements;

    #[test]
    fn placement_map_routes_each_resource_and_rejects_invalid_sets() {
        let primary = LogicalResourceId(2);
        let mla = LogicalResourceId(7);
        let placements = ResourceTransferPlacements::new(
            primary,
            vec![
                (primary, WorkerDataPlacement::TensorSharded),
                (mla, WorkerDataPlacement::ReplicatedG1StripedLower),
            ],
        )
        .unwrap();

        assert_eq!(placements.primary(), primary);
        assert_eq!(
            placements.get(primary),
            Some(WorkerDataPlacement::TensorSharded)
        );
        assert_eq!(
            placements.get(mla),
            Some(WorkerDataPlacement::ReplicatedG1StripedLower)
        );
        assert!(
            ResourceTransferPlacements::new(
                primary,
                vec![
                    (primary, WorkerDataPlacement::TensorSharded),
                    (primary, WorkerDataPlacement::ReplicatedG1StripedLower),
                ],
            )
            .is_err()
        );
        assert!(
            ResourceTransferPlacements::new(
                primary,
                vec![(mla, WorkerDataPlacement::ReplicatedG1StripedLower)],
            )
            .is_err()
        );
    }
}
