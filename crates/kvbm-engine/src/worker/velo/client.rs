// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::check::{IncarnationReply, LocalTransferCheckTiming, LocalTransferRecheck};
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
    /// The worker of the remote process that this client drives. `None` is
    /// the one worker of a process whose service has no worker id.
    worker_id: Option<u64>,
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
    /// The wait and deadline of the check of a lost local-transfer reply.
    check_timing: LocalTransferCheckTiming,
    /// The remote worker's incarnation, learned before the first transfer.
    incarnation: Arc<IncarnationBinding>,
}

/// The incarnation of one remote worker, learned once.
///
/// A failed request is not remembered: the next transfer asks again. The
/// client cannot tell an older worker without the handler from a worker that
/// has not registered it yet, so it never stops asking.
#[derive(Default)]
struct IncarnationBinding {
    known: OnceLock<uuid::Uuid>,
    /// Serializes the request, so concurrent first transfers share one.
    request: tokio::sync::Mutex<()>,
}

impl IncarnationBinding {
    /// The worker's incarnation, asking for it if it is not known yet. `None`
    /// when it is not known by `deadline`, for example from an older worker
    /// without the handler. The whole call, including the wait for another
    /// transfer's request, stays within `deadline`.
    async fn bind(
        &self,
        messenger: &Messenger,
        remote: InstanceId,
        worker_id: Option<u64>,
        deadline: std::time::Duration,
    ) -> Option<uuid::Uuid> {
        self.bind_with(deadline, || {
            Self::ask(messenger, remote, worker_id, deadline)
        })
        .await
    }

    async fn bind_with<F, Fut>(&self, deadline: std::time::Duration, ask: F) -> Option<uuid::Uuid>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<uuid::Uuid>>,
    {
        if let Some(incarnation) = self.known.get() {
            return Some(*incarnation);
        }
        let attempt = async {
            let _request = self.request.lock().await;
            if let Some(incarnation) = self.known.get() {
                return Ok(*incarnation);
            }
            let incarnation = ask().await?;
            let _ = self.known.set(incarnation);
            Ok(incarnation)
        };
        let error = match tokio::time::timeout(deadline, attempt).await {
            Ok(Ok(incarnation)) => return Some(incarnation),
            Ok(Err(error)) => error,
            Err(_) => anyhow::anyhow!("no answer within {deadline:?}"),
        };
        tracing::debug!(
            error = %format!("{error:#}"),
            "worker did not give its incarnation; this local transfer runs unbound"
        );
        None
    }

    async fn ask(
        messenger: &Messenger,
        remote: InstanceId,
        worker_id: Option<u64>,
        deadline: std::time::Duration,
    ) -> Result<uuid::Uuid> {
        let handler = handler_names::scoped(handler_names::WORKER_INCARNATION, worker_id);
        let bytes = messenger
            .unary(&handler)?
            .raw_payload(Bytes::new())
            .instance(remote)
            .timeout(deadline)
            .send()
            .await?;
        Ok(serde_json::from_slice::<IncarnationReply>(&bytes)?.incarnation)
    }
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
        let worker_id = self.worker_id;
        if src == LogicalLayoutHandle::G2
            && dst == LogicalLayoutHandle::G1
            && self.local_onboard_requires_serialization(resource)
            && let Some(reason) = self.collective_failure.get()
        {
            anyhow::bail!("local collective transfer group is aborted: {reason}");
        }
        // Convert to serializable options
        // TODO: Extract bounce buffer handle if present in options.bounce_buffer
        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
            metric_route: options.metric_route,
            copy_engine: options.copy_engine.into(),
        };

        // Create the message
        let transfer_id = uuid::Uuid::new_v4();
        let message = LocalTransferMessage {
            transfer_id: Some(transfer_id),
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
        let timing = self.check_timing;
        let binding = Arc::clone(&self.incarnation);

        // The task sends the drain outcome of this operation to the receipt.
        let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel();

        // Use unary (not am_sync) to wait for transfer completion
        self.messenger.tracker().spawn_on(
            async move {
                // Bind the worker incarnation before the copy is sent, so a
                // later check can tell the same worker from any other.
                let incarnation = binding
                    .bind(&velo, remote_instance, worker_id, timing.deadline)
                    .await;
                let recheck = LocalTransferRecheck {
                    instance: remote_instance,
                    worker_id,
                    incarnation,
                    transfer_id,
                };
                let handler = handler_names::scoped(handler_names::LOCAL_TRANSFER, worker_id);
                let outcome = match velo.unary(&handler) {
                    Ok(request) => {
                        let reply = request
                            .raw_payload(bytes)
                            .instance(remote_instance)
                            .send()
                            .await;
                        match reply {
                            Ok(reply) => LocalTransferReply::decode(&reply),
                            // velo cannot prove that a failed send stayed
                            // unsent, so every error here is a lost reply. No
                            // incarnation was bound, so no check can decide:
                            // the outcome is unknown at once.
                            Err(_) if recheck.incarnation.is_none() => recheck.unbound(),
                            // The request can be with the worker, and the
                            // worker can have an active copy. Wait, then ask
                            // the same worker once.
                            Err(error) => {
                                tracing::warn!(
                                    %recheck,
                                    error = %error,
                                    "local transfer reply lost; checking with the worker"
                                );
                                tokio::time::sleep(timing.wait).await;
                                recheck.check(&velo, timing.deadline).await
                            }
                        }
                    }
                    // The request did not leave this process.
                    Err(error) => TransferDrainOutcome::DrainedWithError(anyhow::anyhow!(
                        "local transfer RPC was not sent: {error}"
                    )),
                };
                let _ = outcome_tx.send(outcome);
            },
            self.messenger.runtime(),
        );

        Ok(TransferCompleteNotification::from_drain_future(
            async move {
                let outcome = outcome_rx.await.unwrap_or_else(|_| {
                    TransferDrainOutcome::Unproven(anyhow::anyhow!(
                        "local transfer RPC task stopped before it reported an outcome"
                    ))
                });
                match outcome {
                    TransferDrainOutcome::Unproven(error)
                        if error.downcast_ref::<LocalTransferRecheck>().is_none() =>
                    {
                        TransferDrainOutcome::Unproven(error.context(LocalTransferWorker {
                            instance: remote_instance,
                            worker_id,
                        }))
                    }
                    outcome => outcome,
                }
            },
        ))
    }
}

impl WorkerTransfers for VeloWorkerClient {
    fn local_onboard_requires_serialization(&self, resource: Option<LogicalResourceId>) -> bool {
        self.local_transfer_placements
            .get()
            .is_none_or(|placements| placements.onboard_requires_serialization(resource))
    }

    fn abort_local_collectives(&self, reason: String) -> Result<TransferCompleteNotification> {
        let worker_id = self.worker_id;
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
                    .unary(&handler_names::scoped(
                        handler_names::ABORT_LOCAL_COLLECTIVES,
                        worker_id,
                    ))?
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
        let worker_id = self.worker_id;
        let event = self.messenger.events().new_event()?;
        let awaiter = self.messenger.events().awaiter(event.handle())?;

        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
            metric_route: options.metric_route,
            copy_engine: options.copy_engine.into(),
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
                    .unary(&handler_names::scoped(
                        handler_names::REMOTE_ONBOARD,
                        worker_id,
                    ))?
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
        let worker_id = self.worker_id;
        let event = self.messenger.events().new_event()?;
        let awaiter = self.messenger.events().awaiter(event.handle())?;

        let options = SerializableTransferOptions {
            layer_range: options.layer_range,
            nixl_write_notification: options.nixl_write_notification,
            bounce_buffer_handle: None,
            bounce_buffer_block_ids: None,
            metric_route: options.metric_route,
            copy_engine: options.copy_engine.into(),
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
                    .unary(&handler_names::scoped(
                        handler_names::REMOTE_OFFLOAD,
                        worker_id,
                    ))?
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
        let worker_id = self.worker_id;
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
                    .unary(&handler_names::scoped(
                        handler_names::CONNECT_REMOTE,
                        worker_id,
                    ))?
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
        let worker_id = self.worker_id;
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
                    .unary(&handler_names::scoped(
                        handler_names::REMOTE_ONBOARD_FOR_INSTANCE,
                        worker_id,
                    ))?
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
        let worker_id = self.worker_id;
        let message = RemotePullPlanMessage { plan };
        let bytes = Bytes::from(serde_json::to_vec(&message)?);

        let event = self.messenger.events().new_event()?;
        let awaiter = self.messenger.events().awaiter(event.handle())?;

        let velo = self.messenger.clone();
        let remote_instance = self.remote;

        self.messenger.tracker().spawn_on(
            async move {
                let result = velo
                    .unary(&handler_names::scoped(
                        handler_names::REMOTE_PULL_PLAN,
                        worker_id,
                    ))?
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
        let worker_id = self.worker_id;
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
                    .unary(&handler_names::scoped(
                        handler_names::REMOTE_ONBOARD_FOR_INSTANCE_RANK,
                        worker_id,
                    ))?
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
        let worker_id = self.worker_id;
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
                .unary(&handler_names::scoped(
                    handler_names::HOST_PAYLOAD_DIGESTS,
                    worker_id,
                ))?
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
        let worker_id = self.worker_id;
        // Use unary (not typed_unary) to avoid JSON serialization of bincode data
        let unary_result = self
            .messenger
            .unary(&handler_names::scoped(
                handler_names::EXPORT_METADATA,
                worker_id,
            ))?
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
        let worker_id = self.worker_id;
        // Use raw_payload to avoid JSON serialization of bincode data
        let unary_result = self
            .messenger
            .unary(&handler_names::scoped(
                handler_names::IMPORT_METADATA,
                worker_id,
            ))?
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
        Self::with_worker_id(messenger, remote, None)
    }

    /// A client of worker `worker_id` in the remote process. The process
    /// serves each of its workers with
    /// [`VeloWorkerService::new_for_worker`](super::VeloWorkerService::new_for_worker)
    /// on one messenger, and the worker id selects the handlers of one of
    /// them.
    pub fn new_for_worker(messenger: Arc<Messenger>, remote: InstanceId, worker_id: u64) -> Self {
        Self::with_worker_id(messenger, remote, Some(worker_id))
    }

    fn with_worker_id(
        messenger: Arc<Messenger>,
        remote: InstanceId,
        worker_id: Option<u64>,
    ) -> Self {
        Self {
            messenger,
            remote,
            worker_id,
            g1_handle: Arc::new(OnceLock::new()),
            g2_handle: Arc::new(OnceLock::new()),
            g3_handle: Arc::new(OnceLock::new()),
            local_transfer_placements: Arc::new(OnceLock::new()),
            collective_failure: Arc::new(OnceLock::new()),
            connected_instances: Arc::new(RwLock::new(HashSet::new())),
            check_timing: LocalTransferCheckTiming::default(),
            incarnation: Arc::new(IncarnationBinding::default()),
        }
    }

    /// Set the wait and deadline of the check of a lost local-transfer reply.
    /// The defaults are [`LocalTransferCheckTiming::default`].
    pub fn with_local_transfer_check(mut self, timing: LocalTransferCheckTiming) -> Self {
        self.check_timing = timing;
        self
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
        let worker_id = self.worker_id;
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
                .unary(&handler_names::scoped(
                    handler_names::OBJECT_HAS_BLOCKS,
                    worker_id,
                ))
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
        let worker_id = self.worker_id;
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
                .unary(&handler_names::scoped(
                    handler_names::OBJECT_PUT_BLOCKS,
                    worker_id,
                ))
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
        let worker_id = self.worker_id;
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
                .unary(&handler_names::scoped(
                    handler_names::OBJECT_GET_BLOCKS,
                    worker_id,
                ))
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

    /// One process serves two or more workers on one messenger. Each worker
    /// has its own set of handlers, and a client with a worker id reaches
    /// the handlers of that worker only. A client without a worker id
    /// reaches the plain handlers of a process with one worker.
    #[tokio::test]
    async fn a_client_with_a_worker_id_reaches_the_handlers_of_that_worker() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let pair = crate::testing::messenger::create_messenger_pair_tcp().await?;
        let workers = [None, Some(1), Some(2)];
        let calls: Vec<Arc<AtomicUsize>> = workers
            .iter()
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect();
        for (worker, counter) in workers.iter().zip(&calls) {
            let counter = Arc::clone(counter);
            let handler = ::velo::Handler::unary_handler_async(
                handler_names::scoped(handler_names::LOCAL_TRANSFER, *worker),
                move |ctx| {
                    let counter = Arc::clone(&counter);
                    async move {
                        let _message: LocalTransferMessage = serde_json::from_slice(&ctx.payload)?;
                        counter.fetch_add(1, Ordering::Relaxed);
                        Ok(Some(Bytes::new()))
                    }
                },
            )
            .build();
            pair.messenger_b.register_handler(handler)?;
        }
        let counts = || {
            calls
                .iter()
                .map(|counter| counter.load(Ordering::Relaxed))
                .collect::<Vec<_>>()
        };
        let remote = pair.messenger_b.instance_id();
        let transfer = |client: VeloWorkerClient| {
            client.execute_local_transfer(
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                Arc::<[BlockId]>::from(vec![0]),
                Arc::<[BlockId]>::from(vec![0]),
                TransferOptions::default(),
            )
        };

        transfer(VeloWorkerClient::new_for_worker(
            Arc::clone(&pair.messenger_a),
            remote,
            2,
        ))?
        .await?;
        assert_eq!(counts(), [0, 0, 1]);
        transfer(VeloWorkerClient::new_for_worker(
            Arc::clone(&pair.messenger_a),
            remote,
            1,
        ))?
        .await?;
        assert_eq!(counts(), [0, 1, 1]);
        transfer(VeloWorkerClient::new(Arc::clone(&pair.messenger_a), remote))?.await?;
        assert_eq!(counts(), [1, 1, 1]);
        Ok(())
    }

    /// The outcome of a failed RPC is unknown, because the worker can have an
    /// active copy. A reply carries the drain outcome of the worker.
    #[tokio::test]
    async fn a_local_transfer_receipt_separates_unknown_and_drained_failures() -> Result<()> {
        let pair = crate::testing::messenger::create_messenger_pair_tcp().await?;
        let rpc_error = ::velo::Handler::unary_handler_async(
            handler_names::scoped(handler_names::LOCAL_TRANSFER, Some(1)),
            |_ctx| async { Err::<Option<Bytes>, _>(anyhow::anyhow!("injected RPC failure")) },
        )
        .build();
        pair.messenger_b.register_handler(rpc_error)?;
        let drained = ::velo::Handler::unary_handler_async(
            handler_names::scoped(handler_names::LOCAL_TRANSFER, Some(2)),
            |_ctx| async {
                let reply = LocalTransferReply::DrainedWithError {
                    error: "injected drained failure".to_string(),
                };
                Ok(Some(Bytes::from(serde_json::to_vec(&reply)?)))
            },
        )
        .build();
        pair.messenger_b.register_handler(drained)?;
        let remote = pair.messenger_b.instance_id();
        let transfer = |worker_id| {
            VeloWorkerClient::new_for_worker(Arc::clone(&pair.messenger_a), remote, worker_id)
                .with_local_transfer_check(LocalTransferCheckTiming {
                    wait: std::time::Duration::from_millis(10),
                    deadline: std::time::Duration::from_millis(200),
                })
                .execute_local_transfer(
                    LogicalLayoutHandle::G1,
                    LogicalLayoutHandle::G2,
                    Arc::<[BlockId]>::from(vec![0]),
                    Arc::<[BlockId]>::from(vec![0]),
                    TransferOptions::default(),
                )
        };

        assert!(matches!(
            transfer(1)?.await_drain().await,
            TransferDrainOutcome::Unproven(_)
        ));
        match transfer(2)?.await_drain().await {
            TransferDrainOutcome::DrainedWithError(error) => {
                assert!(error.to_string().contains("injected drained failure"));
            }
            TransferDrainOutcome::Completed => panic!("the worker failure must reach the receipt"),
            TransferDrainOutcome::Unproven(error) => {
                panic!("the worker reply proved drain: {error:#}")
            }
        }
        Ok(())
    }

    #[test]
    fn a_local_transfer_reply_keeps_the_drain_outcome() {
        let round_trip = |outcome| {
            let wire = serde_json::to_vec(&LocalTransferReply::from_outcome(outcome)).unwrap();
            LocalTransferReply::decode(&wire)
        };
        assert!(matches!(
            round_trip(TransferDrainOutcome::Completed),
            TransferDrainOutcome::Completed
        ));
        assert!(matches!(
            round_trip(TransferDrainOutcome::DrainedWithError(anyhow::anyhow!(
                "drained"
            ))),
            TransferDrainOutcome::DrainedWithError(_)
        ));
        assert!(matches!(
            round_trip(TransferDrainOutcome::Unproven(anyhow::anyhow!("unknown"))),
            TransferDrainOutcome::Unproven(_)
        ));
        // A handler that does not send this type replies with no bytes on success.
        assert!(matches!(
            LocalTransferReply::decode(&[]),
            TransferDrainOutcome::Completed
        ));
        assert!(matches!(
            LocalTransferReply::decode(b"not a reply"),
            TransferDrainOutcome::Unproven(_)
        ));
    }

    /// A worker that never answers holds the binding for at most its
    /// deadline, and so does a second transfer waiting behind the first.
    #[tokio::test]
    async fn incarnation_binding_stays_within_its_deadline() {
        let binding = Arc::new(IncarnationBinding::default());
        let deadline = std::time::Duration::from_millis(200);
        let started = std::time::Instant::now();
        let silent = || std::future::pending::<Result<uuid::Uuid>>();
        let (first, second) = tokio::join!(
            binding.bind_with(deadline, silent),
            binding.bind_with(deadline, silent)
        );
        let elapsed = started.elapsed();
        assert_eq!((first, second), (None, None));
        assert!(
            elapsed < deadline + std::time::Duration::from_millis(150),
            "binding took {elapsed:?}, past its {deadline:?} deadline"
        );
        let known = uuid::Uuid::new_v4();
        assert_eq!(
            binding
                .bind_with(deadline, || async move { Ok(known) })
                .await,
            Some(known),
            "a later transfer asks again"
        );
    }

    #[test]
    fn a_worker_id_scopes_each_handler_name() {
        assert_eq!(
            handler_names::scoped(handler_names::LOCAL_TRANSFER, None),
            handler_names::LOCAL_TRANSFER
        );
        assert_eq!(
            handler_names::scoped(handler_names::LOCAL_TRANSFER, Some(3)),
            "kvbm.worker.local_transfer#3"
        );
        assert_ne!(
            handler_names::scoped(handler_names::LOCAL_TRANSFER, Some(1)),
            handler_names::scoped(handler_names::LOCAL_TRANSFER, Some(11))
        );
    }
}
