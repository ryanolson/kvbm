// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoints for the in-process kv-router consolidator.

use crate::EventSource;

/// Parameters for the in-process kv-router consolidator.
///
/// Supplied at leader construction time and forwarded to the
/// `InstanceLeader::with_consolidator` wiring inside `initialize_async`. All three
/// fields must be non-empty strings; `vllm_zmq_endpoint` is optional and when
/// `None` the consolidator runs in KVBM-only mode (no ZMQ ingress).
pub struct ConsolidatorEndpoints {
    /// ZMQ endpoint vLLM publishes G1 events on (e.g. `"tcp://127.0.0.1:5557"`).
    /// `None` disables ZMQ ingress.
    pub vllm_zmq_endpoint: Option<String>,
    /// ZMQ endpoint the consolidator binds for egress (e.g. `"tcp://0.0.0.0:57001"`).
    pub egress_endpoint: String,
    /// Origin tag for ZMQ-ingress events (`Vllm`, `Trtllm`, or `Kvbm`).
    pub engine_source: EventSource,
}
