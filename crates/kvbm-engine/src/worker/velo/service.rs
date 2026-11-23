// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use kvbm_physical::manager::SerializedLayout;

use super::check::{CheckMessage, IncarnationReply, LocalTransferRecords};
use super::{
    AbortLocalCollectivesMessage, Arc, ConnectRemoteMessage, DirectWorker,
    ExecuteRemoteOnboardForInstanceMessage, ExecuteRemoteOnboardForInstanceRankMessage,
    HostPayloadDigestsMessage, LocalTransferMessage, LocalTransferReply, ObjectGetBlocksMessage,
    ObjectHasBlocksMessage, ObjectHasBlocksResponse, ObjectPutBlocksMessage,
    ObjectPutGetBlocksResponse, RemoteOffloadMessage, RemoteOnboardMessage, RemotePullPlanMessage,
    Result, TransferCompleteNotification, TransferDrainOutcome, TransferOptions, Worker,
    WorkerTransfers, handler_names,
};
use crate::object::ObjectBlockOps;

use bytes::Bytes;
use derive_builder::Builder;

use ::velo::{Handler, Messenger};

/// Dispatch one decoded `LOCAL_TRANSFER` request to `transfers`.
///
/// An error from this function occurs before the worker launches a physical
/// operation.
fn dispatch_local_transfer(
    message: LocalTransferMessage,
    physical_worker: &DirectWorker,
    transfers: &dyn WorkerTransfers,
) -> Result<TransferCompleteNotification> {
    // Convert options and resolve bounce buffer if present
    let bounce_buffer_parts = message.options.bounce_buffer_parts();
    let mut options: TransferOptions = message.options.into();
    if let Some((handle, block_ids)) = bounce_buffer_parts {
        options.bounce_buffer = Some(physical_worker.create_bounce_buffer(handle, block_ids)?);
    }

    match message.resource {
        Some(resource) => transfers.execute_local_transfer_for_resource(
            resource,
            message.src,
            message.dst,
            Arc::from(message.src_block_ids),
            Arc::from(message.dst_block_ids),
            options,
        ),
        None => transfers.execute_local_transfer(
            message.src,
            message.dst,
            Arc::from(message.src_block_ids),
            Arc::from(message.dst_block_ids),
            options,
        ),
    }
}

/// Dispatches one decoded local transfer on a worker.
pub(super) type LocalTransferDispatch =
    Arc<dyn Fn(LocalTransferMessage) -> Result<TransferCompleteNotification> + Send + Sync>;

/// Run one local transfer and keep its record.
///
/// The transfer is recorded as active before it is dispatched, and its drain
/// outcome is recorded after the drain. A dispatch error launched nothing, so
/// it is recorded as drained with an error.
async fn run_local_transfer(
    records: &LocalTransferRecords,
    dispatch: &LocalTransferDispatch,
    message: LocalTransferMessage,
) -> TransferDrainOutcome {
    let transfer_id = message.transfer_id;
    if let Some(transfer_id) = transfer_id {
        records.begin(transfer_id);
    }
    let outcome = match dispatch(message) {
        Ok(notification) => notification.await_drain().await,
        Err(error) => TransferDrainOutcome::DrainedWithError(error),
    };
    if let Some(transfer_id) = transfer_id {
        records.finish(transfer_id, &outcome);
    }
    outcome
}

fn local_transfer_reply(outcome: TransferDrainOutcome) -> Result<Option<Bytes>> {
    // The reply carries the outcome. An RPC error tells the client that the
    // outcome is unknown.
    let reply = LocalTransferReply::from_outcome(outcome);
    Ok(Some(Bytes::from(serde_json::to_vec(&reply)?)))
}

/// Register the local-transfer handlers of one worker: `LOCAL_TRANSFER`,
/// `LOCAL_TRANSFER_CHECK`, and `WORKER_INCARNATION`, scoped by `worker_id`.
///
/// `records` holds this worker's incarnation and transfer records. Each
/// worker of a process has its own.
pub(super) fn register_local_transfer_handlers(
    messenger: &Messenger,
    worker_id: Option<u64>,
    records: Arc<LocalTransferRecords>,
    dispatch: LocalTransferDispatch,
) -> Result<()> {
    let transfer_records = Arc::clone(&records);
    let transfer = Handler::unary_handler_async(
        handler_names::scoped(handler_names::LOCAL_TRANSFER, worker_id),
        move |ctx| {
            let records = Arc::clone(&transfer_records);
            let dispatch = Arc::clone(&dispatch);
            async move {
                let message: LocalTransferMessage = match serde_json::from_slice(&ctx.payload) {
                    Ok(message) => message,
                    Err(error) => {
                        return local_transfer_reply(TransferDrainOutcome::DrainedWithError(
                            error.into(),
                        ));
                    }
                };
                #[cfg(any(test, feature = "testing"))]
                let deferred = records.hooks.defer_next.lock().take();
                #[cfg(any(test, feature = "testing"))]
                if let Some(gate) = deferred {
                    tokio::spawn(async move {
                        let _ = gate.await;
                        let _ = run_local_transfer(&records, &dispatch, message).await;
                    });
                    anyhow::bail!("injected lost reply before the transfer was recorded");
                }
                #[cfg(any(test, feature = "testing"))]
                let lose_running_reply = records.hooks.lose_running_reply.lock().take();
                #[cfg(any(test, feature = "testing"))]
                if let Some(gate) = lose_running_reply {
                    tokio::spawn(async move {
                        let _ = run_local_transfer(&records, &dispatch, message).await;
                    });
                    gate.await?;
                    anyhow::bail!("injected lost reply for a running transfer");
                }
                let outcome = run_local_transfer(&records, &dispatch, message).await;
                #[cfg(test)]
                {
                    use std::sync::atomic::Ordering;
                    if records.hooks.evict_after_next.swap(false, Ordering::AcqRel) {
                        records.evict_all();
                    }
                    if records.hooks.lose_next_reply.swap(false, Ordering::AcqRel) {
                        anyhow::bail!("injected lost reply after the transfer finished");
                    }
                }
                local_transfer_reply(outcome)
            }
        },
    )
    .build();
    messenger.register_handler(transfer)?;

    let check_records = Arc::clone(&records);
    let check = Handler::unary_handler_async(
        handler_names::scoped(handler_names::LOCAL_TRANSFER_CHECK, worker_id),
        move |ctx| {
            let records = Arc::clone(&check_records);
            async move {
                #[cfg(test)]
                records
                    .hooks
                    .checks
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                let message: CheckMessage = serde_json::from_slice(&ctx.payload)?;
                let reply = records.check(message.transfer_id);
                Ok(Some(Bytes::from(serde_json::to_vec(&reply)?)))
            }
        },
    )
    .build();
    messenger.register_handler(check)?;

    let incarnation = Handler::unary_handler_async(
        handler_names::scoped(handler_names::WORKER_INCARNATION, worker_id),
        move |_ctx| {
            let reply = IncarnationReply {
                incarnation: records.incarnation(),
            };
            async move { Ok(Some(Bytes::from(serde_json::to_vec(&reply)?))) }
        },
    )
    .build();
    messenger.register_handler(incarnation)?;
    Ok(())
}

/// Builder for VeloWorkerService - provides flexibility in construction.
///
/// Use this builder when you need to:
/// - Pass a pre-built DirectWorker (when caller manages layout registration)
/// - Pass a pre-built TransferManager (service creates DirectWorker)
/// - Have more control over worker configuration
#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct VeloWorkerService {
    messenger: Arc<Messenger>,
    worker: Arc<DirectWorker>,
    /// Optional transfer policy layered over the physical worker. Metadata,
    /// bounce buffers, and object operations remain owned by `worker`.
    #[builder(default)]
    transfers: Option<Arc<dyn WorkerTransfers>>,
    /// The id of this worker in a process that serves two or more workers on
    /// one messenger. `None` is the one worker of a process, with the plain
    /// handler names.
    #[builder(default)]
    worker_id: Option<u64>,
    /// This worker's incarnation and local-transfer records.
    #[builder(setter(skip), default = "LocalTransferRecords::new()")]
    records: Arc<LocalTransferRecords>,
}

impl VeloWorkerService {
    pub fn new(messenger: Arc<Messenger>, worker: Arc<DirectWorker>) -> Result<Self> {
        let service = Self {
            messenger,
            worker,
            transfers: None,
            worker_id: None,
            records: LocalTransferRecords::new(),
        };
        service.register_handlers()?;
        Ok(service)
    }

    /// Construct the service of worker `worker_id` in a process that serves
    /// two or more workers on one messenger. Each worker registers its own
    /// handlers, and
    /// [`VeloWorkerClient::new_for_worker`](super::VeloWorkerClient::new_for_worker)
    /// with the same id reaches this worker. `transfers` is the optional
    /// transfer policy of [`Self::new_with_transfers`].
    pub fn new_for_worker(
        messenger: Arc<Messenger>,
        worker_id: u64,
        worker: Arc<DirectWorker>,
        transfers: Option<Arc<dyn WorkerTransfers>>,
    ) -> Result<Self> {
        let service = Self {
            messenger,
            worker,
            transfers,
            worker_id: Some(worker_id),
            records: LocalTransferRecords::new(),
        };
        service.register_handlers()?;
        Ok(service)
    }

    /// The id of this worker, if its process serves two or more workers.
    pub fn worker_id(&self) -> Option<u64> {
        self.worker_id
    }

    fn handler_name(&self, name: &str) -> String {
        handler_names::scoped(name, self.worker_id)
    }

    /// Construct a service with a transfer policy layered over the physical
    /// worker, for example replicated-G1/striped-G2 MLA routing.
    pub fn new_with_transfers(
        messenger: Arc<Messenger>,
        worker: Arc<DirectWorker>,
        transfers: Arc<dyn WorkerTransfers>,
    ) -> Result<Self> {
        let service = Self {
            messenger,
            worker,
            transfers: Some(transfers),
            worker_id: None,
            records: LocalTransferRecords::new(),
        };
        service.register_handlers()?;
        Ok(service)
    }

    /// Access the underlying DirectWorker.
    ///
    /// This is useful for:
    /// - Registering additional layouts after service creation
    /// - Exporting metadata for handshake
    /// - Accessing the TransferManager
    pub fn worker(&self) -> &Arc<DirectWorker> {
        &self.worker
    }

    fn transfers(&self) -> Arc<dyn WorkerTransfers> {
        self.transfers
            .clone()
            .unwrap_or_else(|| self.worker.clone())
    }

    /// Register all worker handlers with Velo
    fn register_handlers(&self) -> Result<()> {
        self.register_local_transfer_handler()?;
        self.register_abort_local_collectives_handler()?;
        self.register_remote_onboard_handler()?;
        self.register_remote_offload_handler()?;
        self.register_import_metadata_handler()?;
        self.register_export_metadata_handler()?;
        self.register_connect_remote_handler()?;
        self.register_execute_remote_onboard_for_instance_handler()?;
        self.register_execute_remote_onboard_for_instance_rank_handler()?;
        self.register_execute_remote_pull_plan_handler()?;
        self.register_host_payload_digests_handler()?;
        // Object storage handlers
        self.register_object_has_blocks_handler()?;
        self.register_object_put_blocks_handler()?;
        self.register_object_get_blocks_handler()?;
        Ok(())
    }

    fn register_abort_local_collectives_handler(&self) -> Result<()> {
        let transfers = self.transfers();
        let handler = Handler::unary_handler_async(
            self.handler_name(handler_names::ABORT_LOCAL_COLLECTIVES),
            move |ctx| {
                let transfers = Arc::clone(&transfers);
                async move {
                    let message: AbortLocalCollectivesMessage =
                        serde_json::from_slice(&ctx.payload)?;
                    transfers.abort_local_collectives(message.reason)?.await?;
                    Ok(Some(Bytes::new()))
                }
            },
        )
        .build();
        self.messenger.register_handler(handler)?;
        Ok(())
    }

    fn register_host_payload_digests_handler(&self) -> Result<()> {
        let worker = Arc::clone(&self.worker);
        let handler = Handler::unary_handler_async(
            self.handler_name(handler_names::HOST_PAYLOAD_DIGESTS),
            move |ctx| {
                let worker = Arc::clone(&worker);
                async move {
                    let message: HostPayloadDigestsMessage = serde_json::from_slice(&ctx.payload)?;
                    let digests = worker
                        .compute_host_payload_digests(message.resource, message.block_ids)
                        .await?;
                    Ok(Some(Bytes::from(serde_json::to_vec(&digests)?)))
                }
            },
        )
        .build();
        self.messenger.register_handler(handler)?;
        Ok(())
    }

    fn register_local_transfer_handler(&self) -> Result<()> {
        let transfers = self.transfers();
        let physical_worker = self.worker.clone();
        let dispatch: LocalTransferDispatch = Arc::new(move |message| {
            dispatch_local_transfer(message, &physical_worker, &*transfers)
        });
        register_local_transfer_handlers(
            &self.messenger,
            self.worker_id,
            Arc::clone(&self.records),
            dispatch,
        )
    }

    fn register_remote_onboard_handler(&self) -> Result<()> {
        let transfers = self.transfers();
        let physical_worker = self.worker.clone();

        // Use unary_handler_async for explicit response (works with unary client)
        let handler = Handler::unary_handler_async(
            self.handler_name(handler_names::REMOTE_ONBOARD),
            move |ctx| {
                let transfers = transfers.clone();
                let physical_worker = physical_worker.clone();

                async move {
                    let message: RemoteOnboardMessage = serde_json::from_slice(&ctx.payload)?;

                    // Convert options and resolve bounce buffer if present
                    let bounce_buffer_parts = message.options.bounce_buffer_parts();
                    let mut options: TransferOptions = message.options.into();
                    if let Some((handle, block_ids)) = bounce_buffer_parts {
                        options.bounce_buffer =
                            Some(physical_worker.create_bounce_buffer(handle, block_ids)?);
                    }

                    let notification = transfers.execute_remote_onboard(
                        message.src,
                        message.dst,
                        Arc::from(message.dst_block_ids),
                        options,
                    )?;

                    notification.await?;

                    Ok(Some(Bytes::new()))
                }
            },
        )
        .build();

        self.messenger.register_handler(handler)?;
        Ok(())
    }

    fn register_remote_offload_handler(&self) -> Result<()> {
        let transfers = self.transfers();
        let physical_worker = self.worker.clone();

        // Use unary_handler_async for explicit response (works with unary client)
        let handler = Handler::unary_handler_async(
            self.handler_name(handler_names::REMOTE_OFFLOAD),
            move |ctx| {
                let transfers = transfers.clone();
                let physical_worker = physical_worker.clone();

                async move {
                    let message: RemoteOffloadMessage = serde_json::from_slice(&ctx.payload)?;

                    // Convert options and resolve bounce buffer if present
                    let bounce_buffer_parts = message.options.bounce_buffer_parts();
                    let mut options: TransferOptions = message.options.into();
                    if let Some((handle, block_ids)) = bounce_buffer_parts {
                        options.bounce_buffer =
                            Some(physical_worker.create_bounce_buffer(handle, block_ids)?);
                    }

                    let notification = transfers.execute_remote_offload(
                        message.src,
                        Arc::from(message.src_block_ids),
                        message.dst,
                        options,
                    )?;

                    notification.await?;

                    Ok(Some(Bytes::new()))
                }
            },
        )
        .build();

        self.messenger.register_handler(handler)?;
        Ok(())
    }

    fn register_import_metadata_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = Handler::unary_handler(
            self.handler_name(handler_names::IMPORT_METADATA),
            move |ctx| {
                let metadata = SerializedLayout::from_bytes(ctx.payload.to_vec());
                let handles = worker.import_metadata(metadata)?;
                Ok(Some(Bytes::from(serde_json::to_vec(&handles)?)))
            },
        )
        .build();

        self.messenger.register_handler(handler)?;
        Ok(())
    }

    fn register_export_metadata_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = Handler::unary_handler(
            self.handler_name(handler_names::EXPORT_METADATA),
            move |_ctx| {
                let response = worker.export_metadata()?;
                Ok(Some(Bytes::from(response.as_bytes().to_vec())))
            },
        )
        .build();

        self.messenger.register_handler(handler)?;
        Ok(())
    }

    /// Register handler for connect_remote - stores remote instance metadata in local worker
    fn register_connect_remote_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = Handler::unary_handler(
            self.handler_name(handler_names::CONNECT_REMOTE),
            move |ctx| {
                let message: ConnectRemoteMessage = serde_json::from_slice(&ctx.payload)?;

                // Deserialize metadata (SerializedLayout stored as raw bytes)
                let metadata: Vec<SerializedLayout> = message
                    .metadata
                    .into_iter()
                    .map(SerializedLayout::from_bytes)
                    .collect();

                // Call DirectWorker.connect_remote()
                worker.connect_remote(message.instance_id, metadata)?;

                // Return empty response to signal success
                Ok(Some(Bytes::new()))
            },
        )
        .build();

        self.messenger.register_handler(handler)?;
        Ok(())
    }

    /// Register handler for execute_remote_onboard_for_instance - pulls from remote using instance ID
    fn register_execute_remote_onboard_for_instance_handler(&self) -> Result<()> {
        let transfers = self.transfers();
        let physical_worker = self.worker.clone();

        let handler = Handler::unary_handler_async(
            self.handler_name(handler_names::REMOTE_ONBOARD_FOR_INSTANCE),
            move |ctx| {
                let transfers = transfers.clone();
                let physical_worker = physical_worker.clone();
                async move {
                    let message: ExecuteRemoteOnboardForInstanceMessage =
                        serde_json::from_slice(&ctx.payload)?;

                    // Convert options and resolve bounce buffer if present
                    let bounce_buffer_parts = message.options.bounce_buffer_parts();
                    let mut options: TransferOptions = message.options.into();
                    if let Some((handle, block_ids)) = bounce_buffer_parts {
                        options.bounce_buffer =
                            Some(physical_worker.create_bounce_buffer(handle, block_ids)?);
                    }

                    let notification = transfers.execute_remote_onboard_for_instance(
                        message.instance_id,
                        message.remote_logical_type,
                        message.src_block_ids,
                        message.dst,
                        Arc::from(message.dst_block_ids),
                        options,
                    )?;

                    notification.await?;
                    Ok(Some(Bytes::new()))
                }
            },
        )
        .build();

        self.messenger.register_handler(handler)?;
        Ok(())
    }

    /// Register handler for execute_remote_onboard_for_instance_rank — the
    /// rank-aware variant introduced by AB-1c.
    fn register_execute_remote_onboard_for_instance_rank_handler(&self) -> Result<()> {
        let transfers = self.transfers();
        let physical_worker = self.worker.clone();

        let handler = Handler::unary_handler_async(
            self.handler_name(handler_names::REMOTE_ONBOARD_FOR_INSTANCE_RANK),
            move |ctx| {
                let transfers = transfers.clone();
                let physical_worker = physical_worker.clone();
                async move {
                    let message: ExecuteRemoteOnboardForInstanceRankMessage =
                        serde_json::from_slice(&ctx.payload)?;

                    let bounce_buffer_parts = message.options.bounce_buffer_parts();
                    let mut options: TransferOptions = message.options.into();
                    if let Some((handle, block_ids)) = bounce_buffer_parts {
                        options.bounce_buffer =
                            Some(physical_worker.create_bounce_buffer(handle, block_ids)?);
                    }

                    let notification = transfers.execute_remote_onboard_for_instance_rank(
                        message.instance_id,
                        message.remote_rank,
                        message.remote_logical_type,
                        message.src_block_ids,
                        message.dst,
                        Arc::from(message.dst_block_ids),
                        options,
                    )?;

                    notification.await?;
                    Ok(Some(Bytes::new()))
                }
            },
        )
        .build();

        self.messenger.register_handler(handler)?;
        Ok(())
    }

    /// Register handler for execute_remote_pull_plan — the AB-3
    /// multi-shard cross-parallelism pull entrypoint.
    fn register_execute_remote_pull_plan_handler(&self) -> Result<()> {
        let transfers = self.transfers();

        let handler = Handler::unary_handler_async(
            self.handler_name(handler_names::REMOTE_PULL_PLAN),
            move |ctx| {
                let transfers = transfers.clone();
                async move {
                    let message: RemotePullPlanMessage = serde_json::from_slice(&ctx.payload)?;
                    let notification = transfers.execute_remote_pull_plan(message.plan)?;
                    notification.await?;
                    Ok(Some(Bytes::new()))
                }
            },
        )
        .build();

        self.messenger.register_handler(handler)?;
        Ok(())
    }

    // ========================================================================
    // Object Storage Handlers
    // ========================================================================

    /// Register handler for object_has_blocks - check if blocks exist in object storage
    fn register_object_has_blocks_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = Handler::unary_handler_async(
            self.handler_name(handler_names::OBJECT_HAS_BLOCKS),
            move |ctx| {
                let worker = worker.clone();

                async move {
                    let message: ObjectHasBlocksMessage = serde_json::from_slice(&ctx.payload)?;

                    // Call DirectWorker's ObjectBlockOps implementation
                    let results = worker.has_blocks(message.keys).await;

                    let response = ObjectHasBlocksResponse { results };
                    Ok(Some(Bytes::from(serde_json::to_vec(&response)?)))
                }
            },
        )
        .build();

        self.messenger.register_handler(handler)?;
        Ok(())
    }

    /// Register handler for object_put_blocks - upload blocks to object storage
    fn register_object_put_blocks_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = Handler::unary_handler_async(
            self.handler_name(handler_names::OBJECT_PUT_BLOCKS),
            move |ctx| {
                let worker = worker.clone();

                async move {
                    let message: ObjectPutBlocksMessage = serde_json::from_slice(&ctx.payload)?;

                    // Call DirectWorker's ObjectBlockOps implementation
                    // DirectWorker resolves logical handle to physical layout internally
                    let results = worker
                        .put_blocks(message.keys, message.layout, message.block_ids)
                        .await;

                    let response = ObjectPutGetBlocksResponse::from_results(results);
                    Ok(Some(Bytes::from(serde_json::to_vec(&response)?)))
                }
            },
        )
        .build();

        self.messenger.register_handler(handler)?;
        Ok(())
    }

    /// Register handler for object_get_blocks - download blocks from object storage
    fn register_object_get_blocks_handler(&self) -> Result<()> {
        let worker = self.worker.clone();

        let handler = Handler::unary_handler_async(
            self.handler_name(handler_names::OBJECT_GET_BLOCKS),
            move |ctx| {
                let worker = worker.clone();

                async move {
                    let message: ObjectGetBlocksMessage = serde_json::from_slice(&ctx.payload)?;

                    // Call DirectWorker's ObjectBlockOps implementation
                    // DirectWorker resolves logical handle to physical layout internally
                    let results = worker
                        .get_blocks(message.keys, message.layout, message.block_ids)
                        .await;

                    let response = ObjectPutGetBlocksResponse::from_results(results);
                    Ok(Some(Bytes::from(serde_json::to_vec(&response)?)))
                }
            },
        )
        .build();

        self.messenger.register_handler(handler)?;
        Ok(())
    }
}
