// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::object::ObjectBlockOps;
use crate::worker::LocalTransferPlacements;
use futures::future::BoxFuture;
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::OnceLock;

#[derive(Clone)]
pub struct VeloWorkerClient {
    messenger: Arc<Messenger>,
    remote: InstanceId,
    g1_handle: Arc<OnceLock<LayoutHandle>>,
    g2_handle: Arc<OnceLock<LayoutHandle>>,
    g3_handle: Arc<OnceLock<LayoutHandle>>,
    /// The remote physical worker's authoritative local-transfer routing,
    /// configured from the leader's initialization plan and checked against
    /// stamped worker metadata when available.
    local_transfer_placements: Arc<OnceLock<LocalTransferPlacements>>,
    /// Fatal replicated-transfer failure observed by this client. Setting it
    /// rejects later G2 -> G1 requests even if the abort RPC cannot reach the
    /// worker process.
    collective_failure: Arc<OnceLock<String>>,
    /// Track which remote instances we've connected to for has_remote_metadata()
    connected_instances: Arc<RwLock<HashSet<InstanceId>>>,
}

impl VeloWorkerClient {
    fn install_local_transfer_placements(
        &self,
        placements: LocalTransferPlacements,
        source: &str,
    ) -> Result<()> {
        if let Err(placements) = self.local_transfer_placements.set(placements) {
            anyhow::ensure!(
                self.local_transfer_placements.get() == Some(&placements),
                "{source} transfer placements disagree with the configured worker routing"
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_local_transfer(
        &self,
        resource: Option<LogicalResourceId>,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        if src == LogicalLayoutHandle::G2
            && dst == LogicalLayoutHandle::G1
            && self.local_onboard_requires_serialization(resource)
            && let Some(reason) = self.collective_failure.get()
        {
            anyhow::bail!("local collective transfer group is aborted: {reason}");
        }
        // Create a single local event for this operation
        let event = self.messenger.events().new_event()?;
        let awaiter = self.messenger.events().awaiter(event.handle())?;

        // Convert to serializable options
        // TODO: Extract bounce buffer handle if present in options.bounce_buffer
        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
            metric_route: options.metric_route,
        };

        // Create the message
        let message = LocalTransferMessage {
            resource,
            src,
            dst,
            src_block_ids: src_block_ids.to_vec(),
            dst_block_ids: dst_block_ids.to_vec(),
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        // Spawn a task for the remote instance
        let velo = self.messenger.clone();
        let remote_instance = self.remote;

        // Use unary (not am_sync) to wait for transfer completion
        self.messenger.tracker().spawn_on(
            async move {
                let result = velo
                    .unary(handler_names::LOCAL_TRANSFER)?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.messenger.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }
}

impl WorkerTransfers for VeloWorkerClient {
    fn local_onboard_requires_serialization(&self, resource: Option<LogicalResourceId>) -> bool {
        self.local_transfer_placements
            .get()
            .is_none_or(|placements| placements.onboard_requires_serialization(resource))
    }

    fn abort_local_collectives(&self, reason: String) -> Result<TransferCompleteNotification> {
        let _ = self.collective_failure.set(reason.clone());
        let event = self.messenger.events().new_event()?;
        let awaiter = self.messenger.events().awaiter(event.handle())?;
        let bytes = Bytes::from(serde_json::to_vec(&AbortLocalCollectivesMessage {
            reason,
        })?);
        let velo = Arc::clone(&self.messenger);
        let remote = self.remote;

        self.messenger.tracker().spawn_on(
            async move {
                let result = velo
                    .unary(handler_names::ABORT_LOCAL_COLLECTIVES)?
                    .raw_payload(bytes)
                    .instance(remote)
                    .send()
                    .await;
                match result {
                    Ok(_) => event.trigger(),
                    Err(error) => event.poison(error.to_string()),
                }
            },
            self.messenger.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.dispatch_local_transfer(None, src, dst, src_block_ids, dst_block_ids, options)
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
        self.dispatch_local_transfer(
            Some(resource),
            src,
            dst,
            src_block_ids,
            dst_block_ids,
            options,
        )
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let event = self.messenger.events().new_event()?;
        let awaiter = self.messenger.events().awaiter(event.handle())?;

        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
            metric_route: options.metric_route,
        };

        let message = RemoteOnboardMessage {
            src,
            dst,
            dst_block_ids: dst_block_ids.to_vec(),
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        let velo = self.messenger.clone();
        let remote_instance = self.remote;

        self.messenger.tracker().spawn_on(
            async move {
                // Use unary instead of am_sync for explicit response handling
                let result = velo
                    .unary(handler_names::REMOTE_ONBOARD)?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.messenger.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst: RemoteDescriptor,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let event = self.messenger.events().new_event()?;
        let awaiter = self.messenger.events().awaiter(event.handle())?;

        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
            metric_route: options.metric_route,
        };

        let message = RemoteOffloadMessage {
            src,
            dst,
            src_block_ids: src_block_ids.to_vec(),
            options,
        };

        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        let velo = self.messenger.clone();
        let remote_instance = self.remote;

        self.messenger.tracker().spawn_on(
            async move {
                // Use unary instead of am_sync for explicit response handling
                let result = velo
                    .unary(handler_names::REMOTE_OFFLOAD)?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.messenger.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        // Serialize metadata to bytes (SerializedLayout uses bincode internally)
        let serialized_metadata: Vec<Vec<u8>> =
            metadata.iter().map(|m| m.as_bytes().to_vec()).collect();

        let message = ConnectRemoteMessage {
            instance_id,
            metadata: serialized_metadata,
        };
        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        // Create event for completion tracking
        let event = self.messenger.events().new_event()?;
        let awaiter = self.messenger.events().awaiter(event.handle())?;

        let velo = self.messenger.clone();
        let remote_instance = self.remote;
        let connected = self.connected_instances.clone();
        let target_instance = instance_id;

        self.messenger.tracker().spawn_on(
            async move {
                let result = velo
                    .unary(handler_names::CONNECT_REMOTE)?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => {
                        // Track that we've connected to this instance
                        connected.write().insert(target_instance);
                        event.trigger()
                    }
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.messenger.runtime(),
        );

        Ok(ConnectRemoteResponse::from_awaiter(awaiter))
    }

    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool {
        // Check if we've successfully connected to this instance
        self.connected_instances.read().contains(&instance_id)
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
        let message = ExecuteRemoteOnboardForInstanceMessage {
            instance_id,
            remote_logical_type,
            src_block_ids,
            dst,
            dst_block_ids: dst_block_ids.to_vec(),
            options: SerializableTransferOptions::from(options),
        };
        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        // Create event for completion tracking
        let event = self.messenger.events().new_event()?;
        let awaiter = self.messenger.events().awaiter(event.handle())?;

        let velo = self.messenger.clone();
        let remote_instance = self.remote;

        self.messenger.tracker().spawn_on(
            async move {
                let result = velo
                    .unary(handler_names::REMOTE_ONBOARD_FOR_INSTANCE)?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.messenger.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_remote_pull_plan(
        &self,
        plan: crate::leader::dispatch::WorkerPullPlan,
    ) -> Result<TransferCompleteNotification> {
        let message = RemotePullPlanMessage { plan };
        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        let event = self.messenger.events().new_event()?;
        let awaiter = self.messenger.events().awaiter(event.handle())?;

        let velo = self.messenger.clone();
        let remote_instance = self.remote;

        self.messenger.tracker().spawn_on(
            async move {
                let result = velo
                    .unary(handler_names::REMOTE_PULL_PLAN)?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.messenger.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
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
        let message = ExecuteRemoteOnboardForInstanceRankMessage {
            instance_id,
            remote_rank,
            remote_logical_type,
            src_block_ids,
            dst,
            dst_block_ids: dst_block_ids.to_vec(),
            options: SerializableTransferOptions::from(options),
        };
        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        let event = self.messenger.events().new_event()?;
        let awaiter = self.messenger.events().awaiter(event.handle())?;

        let velo = self.messenger.clone();
        let remote_instance = self.remote;

        self.messenger.tracker().spawn_on(
            async move {
                let result = velo
                    .unary(handler_names::REMOTE_ONBOARD_FOR_INSTANCE_RANK)?
                    .raw_payload(bytes)
                    .instance(remote_instance)
                    .send()
                    .await;

                match result {
                    Ok(_) => event.trigger(),
                    Err(e) => event.poison(e.to_string()),
                }
            },
            self.messenger.runtime(),
        );

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }
}

impl Worker for VeloWorkerClient {
    fn compute_host_payload_digests(
        &self,
        resource: LogicalResourceId,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<Vec<PayloadDigest>>> {
        let message = HostPayloadDigestsMessage {
            resource,
            block_ids,
        };
        let bytes = match serde_json::to_vec(&message) {
            Ok(bytes) => Bytes::from(bytes),
            Err(error) => return Box::pin(async move { Err(error.into()) }),
        };
        let messenger = Arc::clone(&self.messenger);
        let remote = self.remote;
        Box::pin(async move {
            let response = messenger
                .unary(handler_names::HOST_PAYLOAD_DIGESTS)?
                .raw_payload(bytes)
                .instance(remote)
                .send()
                .await?;
            serde_json::from_slice(&response).map_err(Into::into)
        })
    }

    fn g1_handle(&self) -> Option<LayoutHandle> {
        self.g1_handle.get().copied()
    }

    fn g2_handle(&self) -> Option<LayoutHandle> {
        self.g2_handle.get().copied()
    }

    fn g3_handle(&self) -> Option<LayoutHandle> {
        self.g3_handle.get().copied()
    }

    fn export_metadata(&self) -> Result<SerializedLayoutResponse> {
        // Use unary (not typed_unary) to avoid JSON serialization of bincode data
        let unary_result = self
            .messenger
            .unary(handler_names::EXPORT_METADATA)?
            .instance(self.remote)
            .send();

        // Wrap UnaryResult to convert Bytes to SerializedLayout
        let future = async move {
            let bytes = unary_result.await?;
            Ok(SerializedLayout::from_bytes(bytes.to_vec()))
        };

        Ok(SerializedLayoutResponse::from_boxed(Box::pin(future)))
    }

    fn import_metadata(&self, metadata: SerializedLayout) -> Result<ImportMetadataResponse> {
        // Use raw_payload to avoid JSON serialization of bincode data
        let unary_result = self
            .messenger
            .unary(handler_names::IMPORT_METADATA)?
            .raw_payload(Bytes::from(metadata.as_bytes().to_vec()))
            .instance(self.remote)
            .send();

        // Response is JSON-serialized Vec<LayoutHandle>
        let future = async move {
            let bytes = unary_result.await?;
            serde_json::from_slice(&bytes).map_err(|e| {
                anyhow::anyhow!("Failed to deserialize import_metadata response: {}", e)
            })
        };

        Ok(ImportMetadataResponse::from_boxed(Box::pin(future)))
    }
}

impl VeloWorkerClient {
    /// Create a new VeloWorkerClient for communicating with a remote worker.
    pub fn new(messenger: Arc<Messenger>, remote: InstanceId) -> Self {
        Self {
            messenger,
            remote,
            g1_handle: Arc::new(OnceLock::new()),
            g2_handle: Arc::new(OnceLock::new()),
            g3_handle: Arc::new(OnceLock::new()),
            local_transfer_placements: Arc::new(OnceLock::new()),
            collective_failure: Arc::new(OnceLock::new()),
            connected_instances: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Configure layout handles from serialized metadata.
    ///
    /// Call this after worker initialization when handles are known from WorkerLayoutResponse.
    /// This allows the VeloWorkerClient to provide layout handles like DirectWorker does.
    ///
    /// # Arguments
    /// * `metadata` - SerializedLayout from WorkerLayoutResponse.metadata
    ///
    /// # Example
    /// ```ignore
    /// let response: WorkerLayoutResponse = worker.initialize(config).await?;
    /// worker_client.configure_layout_handles(&response.metadata)?;
    /// ```
    pub fn configure_layout_handles(&self, metadata: &SerializedLayout) -> Result<()> {
        let unpacked = metadata.unpack()?;
        if let Some(placements) = LocalTransferPlacements::from_metadata(&unpacked)? {
            self.install_local_transfer_placements(placements, "serialized worker metadata")?;
        }
        for desc in &unpacked.layouts {
            match desc.logical_type {
                LogicalLayoutHandle::G1 => {
                    self.g1_handle.set(desc.handle).ok();
                }
                LogicalLayoutHandle::G2 => {
                    self.g2_handle.set(desc.handle).ok();
                }
                LogicalLayoutHandle::G3 => {
                    self.g3_handle.set(desc.handle).ok();
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Configure the worker's actual local-transfer routing selected during
    /// initialization. The leader uses this policy to serialize only routes
    /// that enter rank-wide collectives.
    pub fn configure_local_transfer_placements(
        &self,
        primary: LogicalResourceId,
        placements: Vec<(LogicalResourceId, WorkerDataPlacement)>,
    ) -> Result<()> {
        let placements = LocalTransferPlacements::new(primary, placements)?;
        self.install_local_transfer_placements(placements, "leader initialization")
    }

    /// Get the layout configuration from the remote worker.
    ///
    /// This calls the `kvbm.worker.get_layout_config` handler on the remote worker.
    /// Used by the leader during Phase 3 to gather G1 layout configs from all workers
    /// and validate they match before creating G2/G3 layouts.
    ///
    /// # Returns
    /// A typed unary result that resolves to the layout configuration
    pub fn get_layout_config(&self) -> Result<::velo::TypedUnaryResult<LayoutConfig>> {
        let instance = self.remote;

        let awaiter = self
            .messenger
            .typed_unary::<LayoutConfig>("kvbm.worker.get_layout_config")?
            .instance(instance)
            .send();

        Ok(awaiter)
    }

    /// Configure additional layouts (G2, G3) on the remote worker.
    ///
    /// This calls the `kvbm.worker.configure_layouts` handler on the remote worker.
    /// The worker will create host/pinned cache (G2) and optionally disk cache (G3)
    /// based on the provided configuration.
    ///
    /// # Arguments
    /// * `config` - Leader-provided configuration specifying block counts and backends
    ///
    /// # Returns
    /// A typed unary result that resolves to the worker's response with updated metadata
    pub fn configure_layouts(
        &self,
        config: LeaderLayoutConfig,
    ) -> Result<::velo::TypedUnaryResult<WorkerLayoutResponse>> {
        let instance = self.remote;

        let awaiter = self
            .messenger
            .typed_unary::<WorkerLayoutResponse>("kvbm.worker.configure_layouts")?
            .payload(config)?
            .instance(instance)
            .send();

        Ok(awaiter)
    }
}

impl ObjectBlockOps for VeloWorkerClient {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        let message = ObjectHasBlocksMessage { keys: keys.clone() };
        let bytes = match serde_json::to_vec(&message) {
            Ok(b) => Bytes::from(b),
            Err(_) => {
                return Box::pin(async move { keys.into_iter().map(|k| (k, None)).collect() });
            }
        };

        let velo = self.messenger.clone();
        let remote = self.remote;

        Box::pin(async move {
            let result = velo
                .unary(handler_names::OBJECT_HAS_BLOCKS)
                .ok()
                .map(|u| u.raw_payload(bytes).instance(remote).send());

            match result {
                Some(unary_result) => match unary_result.await {
                    Ok(response_bytes) => {
                        match serde_json::from_slice::<ObjectHasBlocksResponse>(&response_bytes) {
                            Ok(response) => response.results,
                            Err(_) => keys.into_iter().map(|k| (k, None)).collect(),
                        }
                    }
                    Err(_) => keys.into_iter().map(|k| (k, None)).collect(),
                },
                None => keys.into_iter().map(|k| (k, None)).collect(),
            }
        })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        src_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // For remote workers, we send the logical layout handle - they resolve it locally
        let message = ObjectPutBlocksMessage {
            keys: keys.clone(),
            layout: src_layout,
            block_ids,
        };
        let bytes = match serde_json::to_vec(&message) {
            Ok(b) => Bytes::from(b),
            Err(_) => return Box::pin(async move { keys.into_iter().map(Err).collect() }),
        };

        let velo = self.messenger.clone();
        let remote = self.remote;

        Box::pin(async move {
            let result = velo
                .unary(handler_names::OBJECT_PUT_BLOCKS)
                .ok()
                .map(|u| u.raw_payload(bytes).instance(remote).send());

            match result {
                Some(unary_result) => match unary_result.await {
                    Ok(response_bytes) => {
                        match serde_json::from_slice::<ObjectPutGetBlocksResponse>(&response_bytes)
                        {
                            Ok(response) => response.into_results(),
                            Err(_) => keys.into_iter().map(Err).collect(),
                        }
                    }
                    Err(_) => keys.into_iter().map(Err).collect(),
                },
                None => keys.into_iter().map(Err).collect(),
            }
        })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        dst_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // For remote workers, we send the logical layout handle - they resolve it locally
        let message = ObjectGetBlocksMessage {
            keys: keys.clone(),
            layout: dst_layout,
            block_ids,
        };
        let bytes = match serde_json::to_vec(&message) {
            Ok(b) => Bytes::from(b),
            Err(_) => return Box::pin(async move { keys.into_iter().map(Err).collect() }),
        };

        let velo = self.messenger.clone();
        let remote = self.remote;

        Box::pin(async move {
            let result = velo
                .unary(handler_names::OBJECT_GET_BLOCKS)
                .ok()
                .map(|u| u.raw_payload(bytes).instance(remote).send());

            match result {
                Some(unary_result) => match unary_result.await {
                    Ok(response_bytes) => {
                        match serde_json::from_slice::<ObjectPutGetBlocksResponse>(&response_bytes)
                        {
                            Ok(response) => response.into_results(),
                            Err(_) => keys.into_iter().map(Err).collect(),
                        }
                    }
                    Err(_) => keys.into_iter().map(Err).collect(),
                },
                None => keys.into_iter().map(Err).collect(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use kvbm_physical::manager::{
        ParallelismDescriptor, ResourceLayoutDescriptor, ResourceLayouts,
        ResourceParallelismDescriptor, ResourceParallelismDescriptors, WorkerAddress,
    };

    use super::*;
    use crate::testing::create_messenger_tcp;

    #[tokio::test]
    async fn tensor_sharded_velo_worker_does_not_serialize_local_onboards() -> Result<()> {
        let client = VeloWorkerClient::new(create_messenger_tcp().await?, InstanceId::new_v4());
        let resource = LogicalResourceId(3);
        client.configure_local_transfer_placements(
            resource,
            vec![(resource, WorkerDataPlacement::TensorSharded)],
        )?;

        assert!(!client.local_onboard_requires_serialization(None));
        assert!(!client.local_onboard_requires_serialization(Some(resource)));
        Ok(())
    }

    #[tokio::test]
    async fn mixed_velo_worker_serializes_only_replicated_resource_onboards() -> Result<()> {
        let client = VeloWorkerClient::new(create_messenger_tcp().await?, InstanceId::new_v4());
        let attention = LogicalResourceId(2);
        let latent = LogicalResourceId(7);
        client.configure_local_transfer_placements(
            attention,
            vec![
                (attention, WorkerDataPlacement::TensorSharded),
                (latent, WorkerDataPlacement::ReplicatedG1StripedLower),
            ],
        )?;

        assert!(!client.local_onboard_requires_serialization(None));
        assert!(!client.local_onboard_requires_serialization(Some(attention)));
        assert!(client.local_onboard_requires_serialization(Some(latent)));
        Ok(())
    }

    #[tokio::test]
    async fn legacy_metadata_preserves_nonzero_primary_transfer_placement() -> Result<()> {
        let client = VeloWorkerClient::new(create_messenger_tcp().await?, InstanceId::new_v4());
        let primary = LogicalResourceId(11);
        let resource_layouts = ResourceLayouts::new(
            primary,
            vec![ResourceLayoutDescriptor::new(primary, Vec::new())],
        )?;
        let metadata = SerializedLayout::pack_with_resources(
            WorkerAddress::new(11, "legacy-placement".to_string()),
            Vec::new(),
            Vec::new(),
            Some(ParallelismDescriptor::single_worker(1)),
            Some(WorkerDataPlacement::TensorSharded),
            Some(resource_layouts),
        )?;

        client.configure_layout_handles(&metadata)?;

        assert!(!client.local_onboard_requires_serialization(None));
        assert!(!client.local_onboard_requires_serialization(Some(primary)));
        Ok(())
    }

    #[tokio::test]
    async fn mixed_resource_metadata_configures_each_transfer_placement() -> Result<()> {
        let client = VeloWorkerClient::new(create_messenger_tcp().await?, InstanceId::new_v4());
        let attention = LogicalResourceId(2);
        let latent = LogicalResourceId(7);
        let descriptor = ParallelismDescriptor::single_worker(1);
        let resources = ResourceParallelismDescriptors::new(
            attention,
            vec![
                ResourceParallelismDescriptor::new(
                    attention,
                    descriptor.clone(),
                    WorkerDataPlacement::TensorSharded,
                ),
                ResourceParallelismDescriptor::new(
                    latent,
                    descriptor.clone(),
                    WorkerDataPlacement::ReplicatedG1StripedLower,
                ),
            ],
        )?;
        let metadata = SerializedLayout::pack_with_resource_parallelism(
            WorkerAddress::new(12, "mixed-placement".to_string()),
            Vec::new(),
            Vec::new(),
            Some(descriptor),
            Some(WorkerDataPlacement::TensorSharded),
            None,
            Some(resources),
        )?;

        client.configure_layout_handles(&metadata)?;

        assert!(!client.local_onboard_requires_serialization(Some(attention)));
        assert!(client.local_onboard_requires_serialization(Some(latent)));
        Ok(())
    }

    #[tokio::test]
    async fn stamped_metadata_must_match_explicit_initialization_placement() -> Result<()> {
        let client = VeloWorkerClient::new(create_messenger_tcp().await?, InstanceId::new_v4());
        let resource = LogicalResourceId::default();
        client.configure_local_transfer_placements(
            resource,
            vec![(resource, WorkerDataPlacement::TensorSharded)],
        )?;
        let metadata = SerializedLayout::pack_with_resources(
            WorkerAddress::new(13, "conflicting-placement".to_string()),
            Vec::new(),
            Vec::new(),
            Some(ParallelismDescriptor::single_worker(1)),
            Some(WorkerDataPlacement::ReplicatedG1StripedLower),
            None,
        )?;

        let error = client
            .configure_layout_handles(&metadata)
            .expect_err("conflicting placement metadata must fail closed");
        assert!(error.to_string().contains("disagree"));
        Ok(())
    }

    #[tokio::test]
    async fn fatal_collective_failure_rejects_later_onboards_before_rpc_dispatch() -> Result<()> {
        let client = VeloWorkerClient::new(create_messenger_tcp().await?, InstanceId::new_v4());
        client
            .collective_failure
            .set("injected rank-group failure".to_string())
            .unwrap();

        let error = client
            .execute_local_transfer(
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G1,
                Arc::from([0]),
                Arc::from([0]),
                TransferOptions::default(),
            )
            .err()
            .expect("a poisoned client must fail before contacting its worker");
        assert!(error.to_string().contains("injected rank-group failure"));
        Ok(())
    }

    #[tokio::test]
    async fn fatal_collective_failure_preserves_independent_tensor_resource() -> Result<()> {
        let client = VeloWorkerClient::new(create_messenger_tcp().await?, InstanceId::new_v4());
        let attention = LogicalResourceId(2);
        let latent = LogicalResourceId(7);
        client.configure_local_transfer_placements(
            attention,
            vec![
                (attention, WorkerDataPlacement::TensorSharded),
                (latent, WorkerDataPlacement::ReplicatedG1StripedLower),
            ],
        )?;
        client
            .collective_failure
            .set("injected replicated-resource failure".to_string())
            .unwrap();

        let tensor_route = client.execute_local_transfer_for_resource(
            attention,
            LogicalLayoutHandle::G2,
            LogicalLayoutHandle::G1,
            Arc::from([0]),
            Arc::from([0]),
            TransferOptions::default(),
        );
        assert!(
            tensor_route.is_ok(),
            "an independent tensor-sharded route must remain dispatchable"
        );
        let replicated_error = client
            .execute_local_transfer_for_resource(
                latent,
                LogicalLayoutHandle::G2,
                LogicalLayoutHandle::G1,
                Arc::from([0]),
                Arc::from([0]),
                TransferOptions::default(),
            )
            .err()
            .expect("the replicated route must fail closed");
        assert!(
            replicated_error
                .to_string()
                .contains("injected replicated-resource failure")
        );
        Ok(())
    }
}
