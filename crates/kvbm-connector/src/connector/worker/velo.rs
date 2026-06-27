// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! connector worker-side velo receive path — the completion handlers the leader's
//! [`VeloWorkerSink`](crate::connector::leader::VeloWorkerSink)
//! fans engine completions out to.
//!
//! Each handler captures the worker's [`WorkerCompletionState`] — the **same**
//! `Arc` that `get_finished` drains and `bind_metadata`'s `await_fences` waits on
//! (the REFACTOR.md §delegate-lifetime-inversion: a fresh `Arc` anywhere here
//! would silently strand completions). The keys + payloads are connector-specific
//! and shared with the leader sink, so both ends agree on one worker protocol.

use std::sync::Arc;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use velo::{Handler, Messenger};

use kvbm_engine::worker::LeaderLayoutConfig;
use kvbm_protocols::connector::{EngineWorkerSink, FenceToken, LoadOutcome, SaveOutcome};

use crate::KvbmRuntime;

use super::gpu::GpuState;
use super::state::WorkerCompletionState;

// The single shared spot for the connector worker handler keys (the leader sink sends
// to exactly these). Kept distinct from the legacy `kvbm.connector.worker.*`
// keys so the two worker implementations never collide in one process.
pub(crate) const ONBOARD_COMPLETE_HANDLER: &str = "kvbm.connector.worker.onboard_complete";
pub(crate) const OFFLOAD_COMPLETE_HANDLER: &str = "kvbm.connector.worker.offload_complete";
pub(crate) const FAILED_ONBOARD_HANDLER: &str = "kvbm.connector.worker.failed_onboard";
pub(crate) const FENCE_COMPLETE_HANDLER: &str = "kvbm.connector.worker.fence_complete";

// The leader-driven init keys. These deliberately MATCH the legacy worker keys
// (`connector/worker/velo/mod.rs`): the connector leader drives workers through the
// `ConnectorWorkerClient`, so the connector worker must answer on the same handlers.
// Only one worker implementation registers per process, so the shared keys never
// collide at runtime.
const INITIALIZE_HANDLER: &str = "kvbm.connector.worker.initialize";
const GET_LAYOUT_CONFIG_HANDLER: &str = "kvbm.connector.worker.get_layout_config";

/// Onboard (load) success — the request's external KV is resident.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OnboardComplete {
    pub(crate) request_id: String,
}

/// Offload (save) terminal — fired once per request (leader-gated drain).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OffloadComplete {
    pub(crate) request_id: String,
}

/// Onboard failure, naming the failed G1 dest block ids. There is no
/// empty-means-total convention: a total failure arrives already resolved to
/// the load's full dest set (`LoadOutcome` has no id-less failure), because
/// vLLM invalidates failed loads by block id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FailedOnboard {
    pub(crate) request_id: String,
    pub(crate) block_ids: Vec<usize>,
}

/// Register the four engine→worker completion handlers, each capturing
/// `completion` (the worker's live [`WorkerCompletionState`]). Fire-and-forget;
/// a registration failure is logged, not propagated (mirrors the legacy service).
pub(crate) fn register_completion_handlers(
    messenger: &Arc<Messenger>,
    completion: Arc<WorkerCompletionState>,
) {
    let onboard = Arc::clone(&completion);
    let onboard_handler = Handler::typed_unary_async(ONBOARD_COMPLETE_HANDLER, move |ctx| {
        let completion = Arc::clone(&onboard);
        async move {
            let msg: OnboardComplete = ctx.input;
            completion.record_load_finished(&msg.request_id, LoadOutcome::Done);
            Ok(())
        }
    })
    .build();
    if let Err(e) = messenger.register_handler(onboard_handler) {
        tracing::error!("failed to register connector onboard_complete handler: {e}");
    }

    let offload = Arc::clone(&completion);
    let offload_handler = Handler::typed_unary_async(OFFLOAD_COMPLETE_HANDLER, move |ctx| {
        let completion = Arc::clone(&offload);
        async move {
            let msg: OffloadComplete = ctx.input;
            completion.record_save_finished(&msg.request_id, SaveOutcome::Done);
            Ok(())
        }
    })
    .build();
    if let Err(e) = messenger.register_handler(offload_handler) {
        tracing::error!("failed to register connector offload_complete handler: {e}");
    }

    let failed = Arc::clone(&completion);
    let failed_handler = Handler::typed_unary_async(FAILED_ONBOARD_HANDLER, move |ctx| {
        let completion = Arc::clone(&failed);
        async move {
            let msg: FailedOnboard = ctx.input;
            let outcome = LoadOutcome::FailedPartial {
                block_ids: msg.block_ids,
            };
            completion.record_load_finished(&msg.request_id, outcome);
            Ok(())
        }
    })
    .build();
    if let Err(e) = messenger.register_handler(failed_handler) {
        tracing::error!("failed to register connector failed_onboard handler: {e}");
    }

    let fence_handler = Handler::typed_unary_async(FENCE_COMPLETE_HANDLER, move |ctx| {
        let completion = Arc::clone(&completion);
        async move {
            let token: FenceToken = ctx.input;
            completion.record_fence_complete(token);
            Ok(())
        }
    })
    .build();
    if let Err(e) = messenger.register_handler(fence_handler) {
        tracing::error!("failed to register connector fence_complete handler: {e}");
    }
}

/// Register the leader-driven init handlers (faithful port of the legacy
/// `velo::service` registrations onto the connector [`GpuState`]):
///
/// * `initialize` — completes deferred NIXL registration and builds the
///   `DirectWorker` + `WorkerEngine` (injecting the worker's completion sink —
///   the delegate-lifetime inversion) + `VeloWorkerService`.
/// * `get_layout_config` — serves the registration-time layout config.
pub(crate) fn register_init_handlers(
    messenger: &Arc<Messenger>,
    runtime: Arc<KvbmRuntime>,
    gpu: Arc<GpuState>,
    completion: Arc<dyn EngineWorkerSink>,
) {
    let init_gpu = Arc::clone(&gpu);
    let init_handler = Handler::typed_unary_async(INITIALIZE_HANDLER, move |ctx| {
        let gpu = Arc::clone(&init_gpu);
        let runtime = Arc::clone(&runtime);
        let completion = Arc::clone(&completion);
        async move {
            let config: LeaderLayoutConfig = ctx.input;
            gpu.initialize(&runtime, config, completion)
        }
    })
    .build();
    if let Err(e) = messenger.register_handler(init_handler) {
        tracing::error!("failed to register connector initialize handler: {e}");
    }

    let layout_handler = Handler::unary_handler_async(GET_LAYOUT_CONFIG_HANDLER, move |_ctx| {
        let gpu = Arc::clone(&gpu);
        async move {
            Ok(Some(Bytes::from(serde_json::to_vec(
                &gpu.layout_config()?,
            )?)))
        }
    })
    .build();
    if let Err(e) = messenger.register_handler(layout_handler) {
        tracing::error!("failed to register connector get_layout_config handler: {e}");
    }
}
