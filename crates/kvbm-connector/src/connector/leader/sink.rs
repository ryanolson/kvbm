// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-side [`EngineWorkerSink`]: the engine's push face onto the worker velo
//! peers.
//!
//! The leader engine observes an action terminal and calls one of the sink's
//! methods; [`VeloWorkerSink`] forwards it as a velo RPC to the matching worker
//! velo handlers (see
//! [`worker::velo`](crate::connector::worker::velo)), which land it
//! in each worker's `WorkerCompletionState` that `get_finished` drains. This is
//! the leader half of the delegate loop; the worker registers the receiving
//! handlers at `Worker::new`.
//!
//! Trait-contract notes: the methods are SYNC fire-and-forget (`&self -> ()`,
//! invoked with no engine lock held), but velo sends are async — so each method
//! spawns the send(s) on the runtime and logs failures. A dropped completion
//! surfaces as a worker hang, never silent corruption, so loud logging is the
//! correct failure mode. `mark_load_finished` / `mark_save_finished` broadcast to
//! ALL ranks; `mark_fence_complete` targets the single worker named by
//! `token.rank` (rank-indexing of `worker_ids` is load-bearing).

use std::sync::Arc;

use tokio::runtime::Handle;
use velo::Messenger;

use kvbm_protocols::connector::{
    EngineWorkerSink, FenceToken, LoadOutcome, RequestId, SaveOutcome,
};

use crate::InstanceId;
use crate::connector::worker::velo::{
    FAILED_ONBOARD_HANDLER, FENCE_COMPLETE_HANDLER, FailedOnboard, OFFLOAD_COMPLETE_HANDLER,
    ONBOARD_COMPLETE_HANDLER, OffloadComplete, OnboardComplete,
};

/// Fans engine completions out to the worker velo peers. Holds a clone of the
/// rank-indexed worker velo ids + the runtime handle to spawn the async sends.
pub(crate) struct VeloWorkerSink {
    messenger: Arc<Messenger>,
    /// Worker velo instance ids, indexed by TP rank (the order `register_worker`
    /// accumulated them). `mark_fence_complete` indexes this by `token.rank`.
    worker_ids: Vec<InstanceId>,
    tokio: Handle,
}

impl VeloWorkerSink {
    pub(crate) fn new(
        messenger: Arc<Messenger>,
        worker_ids: Vec<InstanceId>,
        tokio: Handle,
    ) -> Arc<Self> {
        Arc::new(Self {
            messenger,
            worker_ids,
            tokio,
        })
    }
}

/// One fire-and-forget unary velo send (mirrors the legacy `ConnectorWorkerClient`
/// send shape: `unary(key).payload(msg).instance(target).send()`).
async fn send<P: serde::Serialize>(
    messenger: &Messenger,
    key: &str,
    payload: P,
    target: InstanceId,
) -> anyhow::Result<()> {
    messenger
        .unary(key)?
        .payload(payload)?
        .instance(target)
        .send()
        .await?;
    Ok(())
}

impl EngineWorkerSink for VeloWorkerSink {
    fn mark_load_finished(&self, req: &RequestId, outcome: LoadOutcome) {
        let messenger = Arc::clone(&self.messenger);
        let ids = self.worker_ids.clone();
        let req = req.clone();
        self.tokio.spawn(async move {
            for id in ids {
                let res = match &outcome {
                    LoadOutcome::Done => {
                        let msg = OnboardComplete {
                            request_id: req.clone(),
                        };
                        send(&messenger, ONBOARD_COMPLETE_HANDLER, msg, id).await
                    }
                    LoadOutcome::FailedPartial { block_ids } => {
                        let msg = FailedOnboard {
                            request_id: req.clone(),
                            block_ids: block_ids.clone(),
                        };
                        send(&messenger, FAILED_ONBOARD_HANDLER, msg, id).await
                    }
                };
                if let Err(e) = res {
                    tracing::warn!(%req, ?id, "connector mark_load_finished velo send failed: {e}");
                }
            }
        });
    }

    fn mark_save_finished(&self, req: &RequestId, _outcome: SaveOutcome) {
        // No failed-offload RPC exists: a save failure collapses to "done" (the
        // KV stays resident in GPU and vLLM does not gate on save); failure
        // block_ids are dropped. The terminal still fires once per request.
        let messenger = Arc::clone(&self.messenger);
        let ids = self.worker_ids.clone();
        let req = req.clone();
        self.tokio.spawn(async move {
            for id in ids {
                let msg = OffloadComplete {
                    request_id: req.clone(),
                };
                if let Err(e) = send(&messenger, OFFLOAD_COMPLETE_HANDLER, msg, id).await {
                    tracing::warn!(%req, ?id, "connector mark_save_finished velo send failed: {e}");
                }
            }
        });
    }

    fn mark_fence_complete(&self, token: FenceToken) {
        let Some(&target) = self.worker_ids.get(token.rank as usize) else {
            tracing::error!(
                rank = token.rank,
                "connector mark_fence_complete: rank out of range"
            );
            return;
        };
        let messenger = Arc::clone(&self.messenger);
        self.tokio.spawn(async move {
            if let Err(e) = send(&messenger, FENCE_COMPLETE_HANDLER, token, target).await {
                tracing::warn!(
                    rank = token.rank,
                    ?target,
                    "connector mark_fence_complete velo send failed: {e}"
                );
            }
        });
    }
}
