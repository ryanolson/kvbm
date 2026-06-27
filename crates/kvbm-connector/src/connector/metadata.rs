// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader→worker per-step signal: the control payload + the connector wire envelope.
//!
//! [`ConnectorMetadata`] is the connector control payload the leader hands the
//! worker each scheduler step — distinct from the serde wire type
//! [`crate::common::KvConnectorMetadata`], which carries
//! the forward-pass / intra-pass transfer plan. The fields the worker consumes
//! are `evicted_requests` (the request ids it must fence before reusing their G1
//! block ids) and `fences` (the per-eviction-generation [`EvictionFence`]s it
//! awaits per rank, REFACTOR.md §5).
//!
//! [`WireMetadata`] is the connector wire envelope: both payloads ride vLLM's
//! scheduler→worker metadata bytes together. The leader seals it
//! (`Leader::serialize_metadata` — draining the staged fences) and the worker
//! unseals it (`Worker::bind_serialized_metadata`); the PyO3 binding only moves
//! the bytes.

use serde::{Deserialize, Serialize};

use kvbm_protocols::connector::{EvictionFence, RequestId};

use crate::common::KvConnectorMetadata;

/// Per-step leader→worker control payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConnectorMetadata {
    /// Monotonically increasing scheduler iteration.
    pub iteration: u64,
    /// Requests preempted this step; the worker fences each before reuse.
    pub evicted_requests: Vec<RequestId>,
    /// Eviction fences minted this step. Each [`EvictionFence`] carries one
    /// per-(generation, worker) [`kvbm_protocols::connector::FenceToken`]; the
    /// worker reads its rank's token and `await_fence`s it before its next
    /// forward pass reuses the evicted G1 ids.
    pub fences: Vec<EvictionFence>,
}

/// The connector leader→worker wire envelope: the transfer plan plus the
/// connector control payload, serialized as one metadata blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WireMetadata {
    /// The forward-pass / intra-pass transfer plan (legacy wire shape:
    /// completion events, intra-pass load/store block ids).
    pub(crate) plan: KvConnectorMetadata,
    /// The connector control payload (eviction fences + evicted request ids).
    pub(crate) control: ConnectorMetadata,
}

/// Control-only view of the [`WireMetadata`] envelope, for readers that need
/// the fences but must not pay for (or depend on) the transfer plan — the
/// worker's `handle_preemptions` hook. serde simply skips the `plan` field the
/// struct lacks, so the same bytes parse under either shape.
#[derive(Debug, Deserialize)]
pub(crate) struct WireControl {
    /// The connector control payload (eviction fences + evicted request ids).
    pub(crate) control: ConnectorMetadata,
}
