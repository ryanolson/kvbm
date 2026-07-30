// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinated;
#[doc = include_str!("../../docs/worker-group.md")]
pub mod group;
mod physical;
mod protocol;
pub mod velo;

pub use coordinated::CoordinatedWorker;
pub use physical::{PhysicalWorker, PhysicalWorkerBuilder};
#[cfg(feature = "collectives")]
pub use physical::{ReplicatedDataWorker, ResourceDispatchWorker};

/// Compatibility alias for [`PhysicalWorker`].
pub use physical::PhysicalWorker as DirectWorker;

use anyhow::Result;
use futures::future::BoxFuture;
use std::{collections::BTreeMap, pin::Pin, sync::Arc};

use crate::object::ObjectBlockOps;
pub use crate::{BlockId, InstanceId, SequenceHash};
pub use kvbm_common::{LogicalLayoutHandle, LogicalResourceId};
pub use kvbm_physical::{
    manager::{LayoutHandle, RdmaLayoutDescriptors, SerializedLayout, WorkerDataPlacement},
    transfer::{PayloadDigest, TransferCompleteNotification},
};

pub use velo::{VeloWorkerClient, VeloWorkerService, VeloWorkerServiceBuilder};

/// Boxed future for serialized layout responses - allows both typed_unary and raw unary results
pub type SerializedResponseAwaiter = Pin<Box<dyn Future<Output = Result<SerializedLayout>> + Send>>;
/// Boxed future for import metadata responses
pub type ImportMetadataResponseAwaiter =
    Pin<Box<dyn Future<Output = Result<Vec<LayoutHandle>>> + Send>>;

/// Authoritative rank-local routing for local transfers, keyed by KV resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LocalTransferPlacements {
    primary: LogicalResourceId,
    resources: BTreeMap<LogicalResourceId, WorkerDataPlacement>,
}

impl LocalTransferPlacements {
    pub(crate) fn new(
        primary: LogicalResourceId,
        placements: Vec<(LogicalResourceId, WorkerDataPlacement)>,
    ) -> Result<Self> {
        let expected_len = placements.len();
        let resources = placements.into_iter().collect::<BTreeMap<_, _>>();
        anyhow::ensure!(
            resources.len() == expected_len,
            "duplicate resource transfer placement"
        );
        anyhow::ensure!(
            resources.contains_key(&primary),
            "primary resource {primary:?} has no transfer placement"
        );
        Ok(Self { primary, resources })
    }

    pub(crate) fn from_metadata(metadata: &RdmaLayoutDescriptors) -> Result<Option<Self>> {
        if let Some(resources) = metadata.resource_parallelism.as_ref() {
            return Self::new(
                resources.primary(),
                resources
                    .iter()
                    .map(|entry| (entry.resource, entry.placement))
                    .collect(),
            )
            .map(Some);
        }
        metadata
            .worker_data_placement
            .map(|placement| {
                let primary = metadata
                    .resource_layouts
                    .as_ref()
                    .map(|resources| resources.primary())
                    .unwrap_or_default();
                Self::new(primary, vec![(primary, placement)])
            })
            .transpose()
    }

    pub(crate) fn primary(&self) -> LogicalResourceId {
        self.primary
    }

    pub(crate) fn get(&self, resource: LogicalResourceId) -> Option<WorkerDataPlacement> {
        self.resources.get(&resource).copied()
    }

    pub(crate) fn resources(&self) -> Vec<LogicalResourceId> {
        self.resources.keys().copied().collect()
    }

    pub(crate) fn has_replicated(&self) -> bool {
        self.resources
            .values()
            .any(|placement| *placement == WorkerDataPlacement::ReplicatedG1StripedLower)
    }

    pub(crate) fn onboard_requires_serialization(
        &self,
        resource: Option<LogicalResourceId>,
    ) -> bool {
        self.get(resource.unwrap_or(self.primary))
            .is_none_or(|placement| placement == WorkerDataPlacement::ReplicatedG1StripedLower)
    }
}

pub use protocol::*;

pub trait WorkerTransfers: Send + Sync {
    /// Whether a local G2 -> G1 dispatch must be serialized with other
    /// onboards for this worker group.
    ///
    /// Replicated G1 placements enter collectives synchronously during
    /// dispatch, so every rank must observe one logical onboard at a time and
    /// in the same order. `resource == None` asks about the worker's selected
    /// primary resource. Implementations that cannot prove their routing is
    /// independent retain the conservative default.
    fn local_onboard_requires_serialization(&self, resource: Option<LogicalResourceId>) -> bool {
        let _ = resource;
        true
    }

    /// Permanently abort rank-local collective state after one member of a
    /// replicated transfer group fails.
    ///
    /// Workers without collectives complete this operation immediately. RPC
    /// clients override it to poison their local route before forwarding the
    /// abort to the worker process.
    fn abort_local_collectives(&self, reason: String) -> Result<TransferCompleteNotification> {
        let _ = reason;
        Ok(TransferCompleteNotification::completed())
    }

    /// Execute a local transfer between two logical layouts.
    ///
    /// # Arguments
    /// * `src` - The source layout handle
    /// * `dst` - The destination layout handle
    /// * `src_block_ids` - The source block IDs
    /// * `dst_block_ids` - The destination block IDs
    /// * `options` - Transfer options (layer range, bounce buffers, etc.)
    ///
    /// # Returns
    /// A future that completes when the transfer is complete
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification>;

    /// Execute a local transfer for one logical KV resource.
    ///
    /// The default preserves pre-resource workers for resource zero and fails
    /// closed for every non-default resource. Resource-aware physical and
    /// parallel workers override this method.
    fn execute_local_transfer_for_resource(
        &self,
        resource: LogicalResourceId,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        if resource != LogicalResourceId::default() {
            anyhow::bail!(
                "local transfer for non-default resource {resource:?} is not implemented by this worker"
            );
        }
        self.execute_local_transfer(src, dst, src_block_ids, dst_block_ids, options)
    }

    /// Execute a remote transfer from a remote layout to a local logical layout.
    ///
    /// This represents a NIXL transfer.
    ///
    /// # Arguments
    /// * `src` - Remote sources can take several forms, see [`RemoteDescriptor`]
    /// * `dst` - The destination layout handle
    /// * `dst_block_ids` - The destination block IDs
    /// * `options` - Transfer options (layer range, bounce buffers, etc.)
    ///
    /// # Returns
    /// A future that completes when the transfer is complete
    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification>;

    /// Execute a remote offload from a local logical layout to a remote descriptor.
    ///
    /// This represents a NIXL offload.
    ///
    /// # Arguments
    /// * `src` - The source layout handle
    /// * `dst` - The destination remote descriptor
    /// * `src_block_ids` - The source block IDs
    /// * `options` - Transfer options (layer range, bounce buffers, etc.)
    ///
    /// # Returns
    /// A future that completes when the offload is complete
    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst: RemoteDescriptor,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification>;

    /// Connect to a remote instance by importing its metadata and storing handle mappings.
    ///
    /// This method stores the handle mappings internally for later use by
    /// `execute_remote_onboard_for_instance`. The metadata is also imported into
    /// the underlying transfer manager so NIXL knows about the remote.
    ///
    /// # Arguments
    /// * `instance_id` - The unique identifier of the remote instance
    /// * `metadata` - Serialized layout metadata from the remote instance.
    ///   For DirectWorker, expects exactly 1 element.
    ///   For ReplicatedWorker, expects one element per worker (in rank order).
    ///
    /// # Returns
    /// A response that completes when the metadata has been imported and mappings stored.
    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse>;

    /// Check if remote metadata has been imported for an instance.
    ///
    /// Returns true if `connect_remote` has been successfully called for this instance.
    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool;

    /// Execute a remote onboard transfer using stored handle mapping.
    ///
    /// This method looks up the remote handle from the stored mapping
    /// (established via `connect_remote`) and executes the transfer.
    ///
    /// # Arguments
    /// * `instance_id` - The remote instance to pull from
    /// * `remote_logical_type` - The logical layout type on the remote (e.g., G2)
    /// * `src_block_ids` - Block IDs on the remote to pull
    /// * `dst` - Local destination logical layout
    /// * `dst_block_ids` - Local destination block IDs
    /// * `options` - Transfer options
    ///
    /// # Errors
    /// Returns error if remote metadata hasn't been imported for this instance.
    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification>;

    /// Rank-aware variant of [`Self::execute_remote_onboard_for_instance`]
    /// — pulls from a specific remote rank under a peer leader (AB-1c).
    ///
    /// Workers that maintain a rank-aware remote-handle map should override
    /// this to look up `(instance_id, remote_rank, remote_logical_type)`.
    /// The default impl bails because the legacy single-handle-per-instance
    /// path cannot disambiguate ranks. This method is the building block the
    /// AB-2/AB-3 cross-parallelism dispatcher will call when targeting an
    /// asymmetric-TP peer rank by rank.
    #[allow(clippy::too_many_arguments)]
    fn execute_remote_onboard_for_instance_rank(
        &self,
        instance_id: InstanceId,
        remote_rank: usize,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        // Default impl: not implemented. Workers that don't track per-rank
        // remote handles must error out rather than mis-route by ignoring
        // the rank.
        let _ = (
            instance_id,
            remote_rank,
            remote_logical_type,
            src_block_ids,
            dst,
            dst_block_ids,
            options,
        );
        anyhow::bail!(
            "execute_remote_onboard_for_instance_rank: rank-aware dispatch not implemented \
             for this Worker impl (instance={instance_id}, remote_rank={remote_rank})"
        )
    }

    /// AB-3: execute a [`crate::leader::dispatch::WorkerPullPlan`] — a
    /// multi-shard pull resolved by the peer leader's
    /// cross-parallelism planner.
    ///
    /// The plan carries one or more [`crate::leader::dispatch::PullShard`]s,
    /// each addressing a specific remote rank under
    /// `plan.remote_instance` with coordinate-space slices on both the
    /// local destination and remote source sides. Workers that own a
    /// rank-aware remote-handle map and a sliced-transfer entry point
    /// override this; the default impl bails because the legacy
    /// transfer path cannot honour `axis_slices`.
    fn execute_remote_pull_plan(
        &self,
        plan: crate::leader::dispatch::WorkerPullPlan,
    ) -> Result<TransferCompleteNotification> {
        let _ = plan;
        anyhow::bail!(
            "execute_remote_pull_plan: multi-shard pull dispatch not implemented for this Worker impl"
        )
    }
}

pub trait Worker: WorkerTransfers + ObjectBlockOps + Send + Sync {
    /// Compute actual-byte digests for local host-tier blocks in caller order.
    ///
    /// The default fails closed. Physical workers implement the host/pinned G2
    /// path and remote worker clients forward it through the worker RPC plane.
    fn compute_host_payload_digests(
        &self,
        resource: LogicalResourceId,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<Vec<PayloadDigest>>> {
        Box::pin(async move {
            anyhow::bail!(
                "host payload digest is not implemented for resource {resource:?} blocks {block_ids:?}"
            )
        })
    }

    /// Get the G1 layout handle for this worker (if configured).
    ///
    /// Returns None if no G1 layout has been registered with this worker.
    fn g1_handle(&self) -> Option<LayoutHandle>;

    /// Get the G2 layout handle for this worker (if configured).
    ///
    /// Returns None if no G2 layout has been registered with this worker.
    fn g2_handle(&self) -> Option<LayoutHandle>;

    /// Get the G3 layout handle for this worker (if configured).
    ///
    /// Returns None if no G3 layout has been registered with this worker.
    fn g3_handle(&self) -> Option<LayoutHandle>;

    /// Export the local metadata for this worker.
    ///
    /// # Returns
    /// A [`kvbm_physical::manager::SerializedLayout`] containing the local metadata
    fn export_metadata(&self) -> Result<SerializedLayoutResponse>;

    /// Import the remote metadata for this worker.
    ///
    /// # Arguments
    /// * `metadata` - A [`kvbm_physical::manager::SerializedLayout`] containing the remote metadata
    ///
    /// # Returns
    /// A vector of [`kvbm_physical::manager::LayoutHandle`] for the imported remote layouts
    fn import_metadata(&self, metadata: SerializedLayout) -> Result<ImportMetadataResponse>;
}
