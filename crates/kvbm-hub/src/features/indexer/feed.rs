// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Carrier-feed PUB socket and publisher loop.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{Context as _, Result};
use dynamo_kv_router::carrier_feed::{
    CarrierFeedFrame, CARRIER_FEED_TOPIC, encode_frame,
};
use futures::SinkExt;
use tmq::{
    Context, Multipart,
    publish::{Publish, publish},
};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_util::sync::CancellationToken;

const ZMQ_LINGER_MS: i32 = 0;

/// Publisher-side counters for the carrier feed.
#[derive(Debug, Default)]
pub struct FeedCounters {
    /// Successfully encoded and sent feed frames.
    pub feed_published: AtomicU64,
    /// Feed frames that failed encoding or sending.
    pub feed_send_errors: AtomicU64,
}

/// Binds a `PUB` socket to `endpoint` (e.g. `tcp://0.0.0.0:0`).
pub fn bind_pub_socket(endpoint: &str) -> Result<Publish> {
    let ctx = Context::new();
    publish(&ctx)
        .set_linger(ZMQ_LINGER_MS)
        .bind(endpoint)
        .with_context(|| format!("binding carrier-feed PUB socket to {endpoint}"))
}

/// Publishes queued carrier-feed frames until cancelled or the queue closes.
///
/// A malformed frame or a transient socket failure is counted and logged, but
/// never terminates the publisher task.
pub async fn run_feed_publisher(
    mut socket: Publish,
    mut rx: UnboundedReceiver<CarrierFeedFrame>,
    cancel: CancellationToken,
    counters: Arc<FeedCounters>,
) {
    tracing::info!("carrier feed publisher started");
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            frame = rx.recv() => {
                let Some(frame) = frame else {
                    break;
                };
                let payload = match encode_frame(&frame) {
                    Ok(payload) => payload,
                    Err(error) => {
                        counters.feed_send_errors.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(%error, "carrier feed frame encoding failed");
                        continue;
                    }
                };
                let multipart = Multipart::from(vec![CARRIER_FEED_TOPIC.to_vec(), payload]);
                let send = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => break,
                    result = socket.send(multipart) => result,
                };
                match send {
                    Ok(()) => {
                        counters.feed_published.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(error) => {
                        counters.feed_send_errors.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(%error, "carrier feed frame send failed");
                    }
                }
            }
        }
    }
    tracing::info!("carrier feed publisher stopped");
}
