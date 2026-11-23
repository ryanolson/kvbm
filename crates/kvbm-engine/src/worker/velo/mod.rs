// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo-based RPC implementation for distributed worker communication.
//!
//! # RPC Pattern Guidelines
//!
//! This module uses only two Velo RPC patterns:
//!
//! 1. **`am_send` (fire-and-forget)**: Use when no response is needed.
//!    - Client sends message and returns immediately
//!    - Handler processes asynchronously, no response sent back
//!    - Use `Handler::am_handler` or `am_handler_async`
//!
//! 2. **`unary` (request-response)**: Use when waiting for completion.
//!    - Client sends request and awaits response
//!    - Handler returns `Ok(Some(Bytes))` or `Ok(None)` which is sent back
//!    - Use `Handler::unary_handler` or `unary_handler_async`
//!
//! # Why Not `am_sync`?
//!
//! We avoid `am_sync` due to observed issues where it does not reliably
//! receive completion signals when paired with `am_handler_async`. While
//! `am_sync` should theoretically behave like `unary` (both await completion),
//! in practice pairing `am_sync` client with `am_handler_async` handler caused
//! indefinite blocking during RDMA transfer tests.
//!
//! The root cause appears to be a mismatch in how responses are routed:
//! - `am_handler_async` returns `Result<()>` - the return value is NOT sent back
//! - `unary_handler_async` returns `Result<Option<Bytes>>` - the return value IS sent back
//!
//! Until the `am_sync` completion path is validated, prefer the simpler and
//! more predictable patterns: `am_send` for fire-and-forget, `unary` for
//! request-response.
//!
//! # The local-transfer contract
//!
//! A local transfer is a `LOCAL_TRANSFER` unary RPC. The worker runs the copy
//! and replies with its drain outcome: completed, drained with an error, or
//! unproven.
//!
//! - **Transfer id.** The client gives each transfer a random id, sent in the
//!   request. The worker records the id as active when the request arrives,
//!   before it dispatches the copy, and records the drain outcome after the
//!   drain. A dispatch error launched nothing, so it is recorded as drained
//!   with an error.
//! - **Incarnation.** Each worker service has an incarnation: a random id
//!   fixed when the service starts. Before its first transfer, the client asks
//!   for it with `WORKER_INCARNATION`, under the check deadline, and keeps it
//!   once it has an answer.
//!   Every transfer is bound to the incarnation the client held when it sent
//!   the transfer.
//! - **The check.** If the RPC fails after the request was handed to velo,
//!   the copy may still run: velo cannot prove that a failed send stayed
//!   unsent. Only a failure to build the request proves nothing left the
//!   process, and gives `DrainedWithError`. The client waits (`LocalTransferCheckTiming::wait`), then
//!   asks the same instance and worker once with `LOCAL_TRANSFER_CHECK`, under
//!   an explicit velo deadline (`LocalTransferCheckTiming::deadline`). The
//!   defaults are 2 s and 5 s; `messenger.local_transfer_check` in
//!   `kvbm-config` overrides them. A copy still active at the check stays
//!   unknown; the engine pauses the request, and only a later check of the
//!   same target lifts it.
//! - **What decides.** Only a record from the same incarnation decides: a
//!   record "completed" gives `Completed`, and a record "drained with error"
//!   gives `DrainedWithError`. Everything else gives `Unproven`: no answer by
//!   the deadline, a record still active, no record (the check came first, or
//!   the record was evicted), a different incarnation, or no bound
//!   incarnation. A reconnect, a restart or a replacement worker proves
//!   nothing, because kvbm has no signal that the original copy stopped.
//!   After an ambiguous RPC error, a fresh handler list may show that the
//!   worker answered but lacks the check handler. The list uses the same
//!   deadline and cannot prove drain.
//! - **The re-check capability.** An `Unproven` outcome from this path
//!   carries a [`LocalTransferRecheck`] as error context: the instance, the
//!   worker id, the bound incarnation and the transfer id.
//!   [`LocalTransferRecheck::check`] repeats the same check, and it is the only
//!   route that can later prove the outcome.
//! - **A copy on several workers.** An SPMD or replicated copy folds one
//!   receipt per worker. The folded `Unproven` error is an
//!   `AggregateTransferFailure` that keeps every part's typed error, so each
//!   worker's capability survives. [`UnresolvedTransfer::from_unproven`] lists
//!   the unresolved participants, walking nested aggregates, including those
//!   inside a part marked drained. Only an explicit drained leaf is proven.
//!   [`UnresolvedTransfer::recheck_worker`] checks one worker's participants,
//!   [`UnresolvedTransfer::apply`] applies one check's outcome, and
//!   [`UnresolvedTransfer::recheck`] checks them all. Each leaves the other
//!   participants and the drained failures as they are.
//!   [`UnresolvedTransfer::into_outcome`] settles only when every participant
//!   is resolved. A participant with no capability, or of a kind kvbm-engine
//!   does not know, never resolves.
//! - **Records.** A worker keeps at most 65,536 records, and keeps a finished
//!   record for 10 minutes. When full, it evicts the oldest finished record
//!   first. An evicted record answers "unknown".
//! - **Compatibility.** All wire changes are additive. A request from an older
//!   client has no transfer id, and the worker keeps no record for it. An older
//!   worker has no `WORKER_INCARNATION` handler; velo rejects the call at once.
//!   The client cannot tell that from a worker still starting, so it sends the
//!   transfer unbound and asks again on the next one. The whole incarnation
//!   request, including any wait behind another transfer's request, stays
//!   within the check deadline. A lost reply from an unbound transfer is
//!   `Unproven` at once, with no incarnation in its capability and no check
//!   sent, because no answer could decide it.

mod check;
mod client;
#[cfg(test)]
mod lost_reply_tests;
mod service;
mod unresolved;

pub(crate) use check::CheckEvidence;
pub use check::{
    DEFAULT_LOCAL_TRANSFER_CHECK_DEADLINE, DEFAULT_LOCAL_TRANSFER_CHECK_WAIT,
    LocalTransferCheckTiming, LocalTransferRecheck,
};
pub use client::VeloWorkerClient;
pub use service::{VeloWorkerService, VeloWorkerServiceBuilder};
pub use unresolved::{UnresolvedParticipant, UnresolvedTransfer};

/// Identifies the remote worker even when a receipt has no re-check capability.
#[derive(Debug, thiserror::Error)]
#[error("local transfer on instance {instance}, worker {worker_id:?}")]
pub(crate) struct LocalTransferWorker {
    pub(crate) instance: velo::InstanceId,
    pub(crate) worker_id: Option<u64>,
}

/// Canonical handler names for the Velo RPCs this worker exposes.
///
/// Centralising these as constants keeps the client (`client.rs`) and
/// service (`service.rs`) in lock-step — a rename or addition lands in
/// exactly one place and the compiler enforces both sides see the same
/// string. Adding a new RPC: add a const here, register a handler in
/// `service.rs`, and use the const at the call site in `client.rs`.
pub(crate) mod handler_names {
    pub const LOCAL_TRANSFER: &str = "kvbm.worker.local_transfer";
    /// Ask a worker for the record of one local transfer.
    pub const LOCAL_TRANSFER_CHECK: &str = "kvbm.worker.local_transfer_check";
    /// Ask a worker for its incarnation.
    pub const WORKER_INCARNATION: &str = "kvbm.worker.incarnation";
    pub const ABORT_LOCAL_COLLECTIVES: &str = "kvbm.worker.abort_local_collectives";
    pub const REMOTE_ONBOARD: &str = "kvbm.worker.remote_onboard";
    pub const REMOTE_OFFLOAD: &str = "kvbm.worker.remote_offload";
    pub const IMPORT_METADATA: &str = "kvbm.worker.import_metadata";
    pub const EXPORT_METADATA: &str = "kvbm.worker.export_metadata";
    pub const CONNECT_REMOTE: &str = "kvbm.worker.connect_remote";
    pub const REMOTE_ONBOARD_FOR_INSTANCE: &str = "kvbm.worker.remote_onboard_for_instance";
    pub const REMOTE_ONBOARD_FOR_INSTANCE_RANK: &str =
        "kvbm.worker.remote_onboard_for_instance_rank";
    /// AB-3: multi-shard cross-parallelism pull. Carries a
    /// `crate::leader::dispatch::WorkerPullPlan` whose `shards` list
    /// drives one or more sliced reads from rank-aware remote handles.
    pub const REMOTE_PULL_PLAN: &str = "kvbm.worker.remote_pull_plan";
    pub const HOST_PAYLOAD_DIGESTS: &str = "kvbm.worker.host_payload_digests";
    pub const OBJECT_HAS_BLOCKS: &str = "kvbm.worker.object_has_blocks";
    pub const OBJECT_PUT_BLOCKS: &str = "kvbm.worker.object_put_blocks";
    pub const OBJECT_GET_BLOCKS: &str = "kvbm.worker.object_get_blocks";

    /// The handler name of `name` for one worker of a process.
    ///
    /// A process with one worker registers the names above as they are
    /// (`worker` is `None`). A process that serves two or more workers on one
    /// messenger gives each worker an id, and each worker registers its own
    /// set of handlers under `name#id`. A client selects a worker by the same
    /// id, so no message needs a worker field.
    pub fn scoped(name: &str, worker: Option<u64>) -> String {
        match worker {
            None => name.to_string(),
            Some(id) => format!("{name}#{id}"),
        }
    }
}

use super::DirectWorker;
use super::*;
use kvbm_common::KvbmTransferRoute;
use kvbm_physical::layout::LayoutConfig;
use kvbm_physical::transfer::{CopyEngine, TransferDrainOutcome, TransferOptions};

use ::velo::Messenger;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

// Serializable transfer options for remote operations
#[derive(Serialize, Deserialize, Clone)]
struct SerializableTransferOptions {
    layer_range: Option<std::ops::Range<usize>>,
    nixl_write_notification: Option<u64>,
    bounce_buffer_handle: Option<LayoutHandle>,
    bounce_buffer_block_ids: Option<Vec<BlockId>>,
    metric_route: Option<KvbmTransferRoute>,
    /// The CUDA copy primitive that the sender selected. A message without
    /// the field lets the receiver choose.
    #[serde(default)]
    copy_engine: WireCopyEngine,
}

/// The wire form of [`CopyEngine`].
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
enum WireCopyEngine {
    #[default]
    Auto,
    VectorizedKernel,
    MemcpyBatch,
}

impl From<CopyEngine> for WireCopyEngine {
    fn from(engine: CopyEngine) -> Self {
        match engine {
            CopyEngine::Auto => Self::Auto,
            CopyEngine::VectorizedKernel => Self::VectorizedKernel,
            CopyEngine::MemcpyBatch => Self::MemcpyBatch,
        }
    }
}

impl From<WireCopyEngine> for CopyEngine {
    fn from(engine: WireCopyEngine) -> Self {
        match engine {
            WireCopyEngine::Auto => Self::Auto,
            WireCopyEngine::VectorizedKernel => Self::VectorizedKernel,
            WireCopyEngine::MemcpyBatch => Self::MemcpyBatch,
        }
    }
}

impl From<SerializableTransferOptions> for TransferOptions {
    fn from(opts: SerializableTransferOptions) -> Self {
        TransferOptions {
            layer_range: opts.layer_range,
            nixl_write_notification: opts.nixl_write_notification,
            // bounce_buffer requires TransportManager to resolve handle to layout
            bounce_buffer: None,
            cuda_stream: None,
            // KV layout overrides are not serialized; they must be set locally
            src_kv_layout: None,
            dst_kv_layout: None,
            metric_route: opts.metric_route,
            // use_planner is a per-side optimization toggle; the wire form
            // intentionally does not propagate it. Each receiver picks its
            // own planner policy. c6 made the receiver-side rule load-
            // bearing: `executor::execute_transfer` auto-promotes
            // use_planner = true when `requires_transform(src, dst)` is
            // true, so the `false` hardcode here is safe for the
            // cross-leader case (Universal↔Universal under c3 semantics
            // → requires_transform = false → auto-promote never fires).
            use_planner: false,
            copy_engine: opts.copy_engine.into(),
        }
    }
}

impl SerializableTransferOptions {
    /// Extract bounce buffer handle and block IDs if present
    fn bounce_buffer_parts(&self) -> Option<(LayoutHandle, Vec<BlockId>)> {
        match (&self.bounce_buffer_handle, &self.bounce_buffer_block_ids) {
            (Some(handle), Some(block_ids)) => Some((*handle, block_ids.clone())),
            _ => None,
        }
    }
}

impl From<TransferOptions> for SerializableTransferOptions {
    fn from(opts: TransferOptions) -> Self {
        // Extract bounce buffer parts if present using into_parts()
        let (bounce_buffer_handle, bounce_buffer_block_ids) = opts
            .bounce_buffer
            .map(|bb| {
                let (handle, block_ids) = bb.into_parts();
                (Some(handle), Some(block_ids))
            })
            .unwrap_or((None, None));

        Self {
            layer_range: opts.layer_range,
            nixl_write_notification: opts.nixl_write_notification,
            bounce_buffer_handle,
            bounce_buffer_block_ids,
            metric_route: opts.metric_route,
            copy_engine: opts.copy_engine.into(),
        }
    }
}

// Message types for remote worker operations
#[derive(Serialize, Deserialize)]
struct LocalTransferMessage {
    /// The id the client gave this transfer. The worker keeps a record under
    /// it. A message from an older client has none, and keeps no record.
    #[serde(default)]
    transfer_id: Option<uuid::Uuid>,
    #[serde(default)]
    resource: Option<kvbm_common::LogicalResourceId>,
    src: LogicalLayoutHandle,
    dst: LogicalLayoutHandle,
    src_block_ids: Vec<BlockId>,
    dst_block_ids: Vec<BlockId>,
    options: SerializableTransferOptions,
}

/// The reply to a `LOCAL_TRANSFER` request.
///
/// The reply carries the drain certainty of the worker, so that the client
/// can tell a drained failure from an unproven one. An RPC error has no
/// reply. The client then reports [`TransferDrainOutcome::Unproven`], because
/// the worker can have an active copy.
#[derive(Serialize, Deserialize)]
enum LocalTransferReply {
    /// The transfer completed.
    Completed,
    /// The transfer failed, and no physical operation is active.
    DrainedWithError { error: String },
    /// The transfer failed, and the worker did not prove that its work drained.
    Unproven { error: String },
}

impl LocalTransferReply {
    fn from_outcome(outcome: TransferDrainOutcome) -> Self {
        match outcome {
            TransferDrainOutcome::Completed => Self::Completed,
            TransferDrainOutcome::DrainedWithError(error) => Self::DrainedWithError {
                error: format!("{error:#}"),
            },
            TransferDrainOutcome::Unproven(error) => Self::Unproven {
                error: format!("{error:#}"),
            },
        }
    }

    fn into_outcome(self) -> TransferDrainOutcome {
        match self {
            Self::Completed => TransferDrainOutcome::Completed,
            Self::DrainedWithError { error } => {
                TransferDrainOutcome::DrainedWithError(anyhow::anyhow!(error))
            }
            Self::Unproven { error } => TransferDrainOutcome::Unproven(anyhow::anyhow!(error)),
        }
    }

    /// Decode the reply bytes of a `LOCAL_TRANSFER` request.
    ///
    /// An empty reply is the success reply of a handler that does not send
    /// this type. A reply that does not decode proves nothing.
    fn decode(bytes: &[u8]) -> TransferDrainOutcome {
        if bytes.is_empty() {
            return TransferDrainOutcome::Completed;
        }
        match serde_json::from_slice::<Self>(bytes) {
            Ok(reply) => reply.into_outcome(),
            Err(error) => TransferDrainOutcome::Unproven(anyhow::anyhow!(
                "local transfer reply did not decode: {error}"
            )),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct AbortLocalCollectivesMessage {
    reason: String,
}

#[derive(Serialize, Deserialize)]
struct HostPayloadDigestsMessage {
    resource: LogicalResourceId,
    block_ids: Vec<BlockId>,
}

#[cfg(test)]
mod local_transfer_message_tests {
    use super::*;
    use kvbm_common::LogicalResourceId;

    fn message(resource: LogicalResourceId) -> LocalTransferMessage {
        LocalTransferMessage {
            transfer_id: None,
            resource: Some(resource),
            src: LogicalLayoutHandle::G1,
            dst: LogicalLayoutHandle::G2,
            src_block_ids: vec![1],
            dst_block_ids: vec![2],
            options: SerializableTransferOptions {
                layer_range: None,
                nixl_write_notification: None,
                bounce_buffer_handle: None,
                bounce_buffer_block_ids: None,
                metric_route: None,
                copy_engine: WireCopyEngine::Auto,
            },
        }
    }

    #[test]
    fn a_legacy_local_transfer_message_has_no_transfer_id() {
        let mut legacy = serde_json::to_value(message(LogicalResourceId(7))).unwrap();
        legacy.as_object_mut().unwrap().remove("transfer_id");
        let decoded: LocalTransferMessage = serde_json::from_value(legacy).unwrap();
        assert_eq!(decoded.transfer_id, None);
    }

    #[test]
    fn local_transfer_message_round_trips_resource() {
        let encoded = serde_json::to_vec(&message(LogicalResourceId(7))).unwrap();
        let decoded: LocalTransferMessage = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.resource, Some(LogicalResourceId(7)));
    }

    #[test]
    fn legacy_local_transfer_message_omits_resource_routing() {
        let mut encoded = serde_json::to_value(message(LogicalResourceId(7))).unwrap();
        encoded.as_object_mut().unwrap().remove("resource");
        let decoded: LocalTransferMessage = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.resource, None);
    }
}

#[derive(Serialize, Deserialize)]
struct RemoteOnboardMessage {
    src: RemoteDescriptor,
    dst: LogicalLayoutHandle,
    dst_block_ids: Vec<BlockId>,
    options: SerializableTransferOptions,
}

#[derive(Serialize, Deserialize)]
struct RemoteOffloadMessage {
    src: LogicalLayoutHandle,
    dst: RemoteDescriptor,
    src_block_ids: Vec<BlockId>,
    options: SerializableTransferOptions,
}

/// Message for connect_remote RPC - stores remote instance metadata in local worker
#[derive(Serialize, Deserialize)]
struct ConnectRemoteMessage {
    instance_id: InstanceId,
    /// Metadata serialized as raw bytes (SerializedLayout uses bincode internally)
    metadata: Vec<Vec<u8>>,
}

/// Message for execute_remote_onboard_for_instance RPC - pulls from remote using instance ID
#[derive(Serialize, Deserialize)]
struct ExecuteRemoteOnboardForInstanceMessage {
    instance_id: InstanceId,
    remote_logical_type: LogicalLayoutHandle,
    src_block_ids: Vec<BlockId>,
    dst: LogicalLayoutHandle,
    dst_block_ids: Vec<BlockId>,
    options: SerializableTransferOptions,
}

/// Rank-aware variant for AB-1c. Targets a specific remote rank under
/// `instance_id` so the worker can resolve via its
/// `remote_handles_rank` map and serve asymmetric-TP pulls.
#[derive(Serialize, Deserialize)]
struct ExecuteRemoteOnboardForInstanceRankMessage {
    instance_id: InstanceId,
    remote_rank: usize,
    remote_logical_type: LogicalLayoutHandle,
    src_block_ids: Vec<BlockId>,
    dst: LogicalLayoutHandle,
    dst_block_ids: Vec<BlockId>,
    options: SerializableTransferOptions,
}

/// Multi-shard pull message for AB-3. The plan was produced by the
/// peer leader's [`crate::leader::dispatch::plan_pull`] and carries
/// everything one local worker needs to execute its share of an
/// asymmetric-TP transfer: a target [`InstanceId`], source/destination
/// [`LogicalLayoutHandle`]s, paired `(src_block_id, dst_block_id)`
/// vectors that apply to every shard, the per-remote-rank shard list
/// with coordinate-space slices on both sides, and a
/// [`crate::leader::dispatch::WirePullOptions`].
///
/// On the wire we wrap the plan in a one-field struct so future
/// additions (e.g. retry hints) can land without re-encoding existing
/// fields — serde_json named fields handle the forward-compat
/// gracefully, but the wrapper keeps the door open for sibling fields
/// without forcing a fresh top-level type.
#[derive(Serialize, Deserialize)]
struct RemotePullPlanMessage {
    plan: crate::leader::dispatch::WorkerPullPlan,
}

// ============================================================================
// Object Storage Message Types
// ============================================================================

/// Message for object_has_blocks RPC - check if blocks exist in object storage
#[derive(Serialize, Deserialize)]
struct ObjectHasBlocksMessage {
    keys: Vec<SequenceHash>,
}

/// Response for object_has_blocks RPC
#[derive(Serialize, Deserialize)]
struct ObjectHasBlocksResponse {
    results: Vec<(SequenceHash, Option<usize>)>,
}

/// Message for object_put_blocks RPC - upload blocks to object storage
#[derive(Serialize, Deserialize)]
struct ObjectPutBlocksMessage {
    keys: Vec<SequenceHash>,
    layout: LogicalLayoutHandle,
    block_ids: Vec<BlockId>,
}

/// Message for object_get_blocks RPC - download blocks from object storage
#[derive(Serialize, Deserialize)]
struct ObjectGetBlocksMessage {
    keys: Vec<SequenceHash>,
    layout: LogicalLayoutHandle,
    block_ids: Vec<BlockId>,
}

/// Response for object put/get operations
#[derive(Serialize, Deserialize)]
struct ObjectPutGetBlocksResponse {
    /// Ok(key) for success, Err(key) for failure - serialized as (bool, key)
    results: Vec<(bool, SequenceHash)>,
}

impl ObjectPutGetBlocksResponse {
    fn from_results(results: Vec<Result<SequenceHash, SequenceHash>>) -> Self {
        Self {
            results: results
                .into_iter()
                .map(|r| match r {
                    Ok(k) => (true, k),
                    Err(k) => (false, k),
                })
                .collect(),
        }
    }

    fn into_results(self) -> Vec<Result<SequenceHash, SequenceHash>> {
        self.results
            .into_iter()
            .map(|(ok, k)| if ok { Ok(k) } else { Err(k) })
            .collect()
    }
}
