// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The prefill-worker side of velo dispatch.
//!
//! [`register_prefill_worker`] installs the [`PREFILL_DISPATCH_HANDLER`] on a
//! worker. The handler puts each request into a local channel and acks it at
//! once, so the worker's engine can take the request with no poll. The same
//! handler serves the hub and a decode worker that sends to the prefill worker
//! directly.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;
use velo::{Handler, Messenger, TypedContext};
use velo_ext::WorkerId;

use super::protocol::{
    PREFILL_DISPATCH_HANDLER, PREFILL_DONE_HANDLER, PrefillDispatchRequest,
    PrefillDispatchResponse, PrefillDone,
};

/// A request that a dispatcher gave to this worker.
pub struct PrefillAssignment {
    pub request: PrefillDispatchRequest,
    pub done: PrefillDoneNotice,
}

/// Tells the dispatcher of a request that the request ended. It sends
/// [`PREFILL_DONE_HANDLER`] when it is dropped, so every end of the request
/// (success, failure or cancel) frees the dispatcher's capacity.
pub struct PrefillDoneNotice {
    messenger: Arc<Messenger>,
    dispatcher: Option<WorkerId>,
    request_id: String,
}

impl PrefillDoneNotice {
    /// The notice for the request in `ctx`, sent back to its sender.
    pub fn new(ctx: &TypedContext<PrefillDispatchRequest>) -> Self {
        Self {
            messenger: Arc::clone(&ctx.msg),
            dispatcher: Some(ctx.sender_worker_id()),
            request_id: ctx.input.request_id.clone(),
        }
    }

    /// Drop the notice without a send. Use it only when the dispatcher
    /// already knows that the worker did not take the request.
    pub fn disarm(mut self) {
        self.dispatcher = None;
    }
}

impl Drop for PrefillDoneNotice {
    fn drop(&mut self) {
        let Some(dispatcher) = self.dispatcher.take() else {
            return;
        };
        let done = PrefillDone {
            request_id: std::mem::take(&mut self.request_id),
        };
        match self
            .messenger
            .am_send(PREFILL_DONE_HANDLER)
            .and_then(|builder| builder.payload(&done))
        {
            // The frame is handed to the transport before `send` returns;
            // dropping the result does not cancel it.
            Ok(builder) => drop(builder.worker(dispatcher).send()),
            Err(error) => tracing::warn!(
                request_id = %done.request_id,
                %error,
                "failed to build the prefill done notice"
            ),
        }
    }
}

/// Register the [`PREFILL_DISPATCH_HANDLER`] on `messenger`. The worker takes
/// requests from the returned channel. When `capacity` requests wait in the
/// channel, the handler rejects the next one.
pub fn register_prefill_worker(
    messenger: &Messenger,
    capacity: usize,
) -> Result<mpsc::Receiver<PrefillAssignment>> {
    let (tx, rx) = mpsc::channel(capacity);
    let handler =
        Handler::typed_unary_async::<PrefillDispatchRequest, PrefillDispatchResponse, _, _>(
            PREFILL_DISPATCH_HANDLER,
            move |ctx| {
                let done = PrefillDoneNotice::new(&ctx);
                let assignment = PrefillAssignment {
                    request: ctx.input,
                    done,
                };
                let reply = match tx.try_send(assignment) {
                    Ok(()) => PrefillDispatchResponse {
                        ok: true,
                        error: None,
                    },
                    Err(error) => {
                        let reason = match &error {
                            mpsc::error::TrySendError::Full(_) => "the prefill worker is full",
                            mpsc::error::TrySendError::Closed(_) => "the prefill worker stopped",
                        };
                        error.into_inner().done.disarm();
                        PrefillDispatchResponse {
                            ok: false,
                            error: Some(reason.to_string()),
                        }
                    }
                };
                async move { Ok(reply) }
            },
        )
        .build();
    messenger
        .register_handler(handler)
        .map_err(|error| anyhow::anyhow!("register {PREFILL_DISPATCH_HANDLER}: {error}"))?;
    Ok(rx)
}
