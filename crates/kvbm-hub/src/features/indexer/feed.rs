// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Carrier-feed event-plane publishers.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context as _, Result};
use dynamo_kv_router::carrier_feed::{
    CARRIER_FEED_TOPIC, CarrierFeedFrame, encode_frame as encode_carrier_feed_frame,
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

/// Binds a ZMQ `PUB` socket to `endpoint` (e.g. `tcp://0.0.0.0:0`).
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
                let payload = match encode_payload(&frame, &counters) {
                    Ok(payload) => payload,
                    Err(()) => continue,
                };
                let multipart = Multipart::from(vec![CARRIER_FEED_TOPIC.to_vec(), payload]);
                let send = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => break,
                    result = socket.send(multipart) => result,
                };
                record_send_result(send, &counters);
            }
        }
    }
    tracing::info!("carrier feed publisher stopped");
}

pub async fn run_nats_feed_publisher(
    client: async_nats::Client,
    subject: String,
    mut rx: UnboundedReceiver<CarrierFeedFrame>,
    cancel: CancellationToken,
    counters: Arc<FeedCounters>,
) {
    tracing::info!(%subject, "carrier feed NATS publisher started");
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            frame = rx.recv() => {
                let Some(frame) = frame else { break };
                let payload = match encode_payload(&frame, &counters) {
                    Ok(payload) => payload,
                    Err(()) => continue,
                };
                let result = client
                    .publish(subject.clone(), payload.into())
                    .await
                    .map(|_| ());
                record_send_result(result, &counters);
            }
        }
    }
    tracing::info!("carrier feed NATS publisher stopped");
}

fn encode_payload(frame: &CarrierFeedFrame, counters: &FeedCounters) -> Result<Vec<u8>, ()> {
    encode_carrier_feed_frame(frame).map_err(|error| {
        counters.feed_send_errors.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(%error, "carrier feed frame encoding failed");
    })
}

fn record_send_result<E: std::fmt::Display>(
    result: std::result::Result<(), E>,
    counters: &FeedCounters,
) {
    match result {
        Ok(()) => {
            counters.feed_published.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            counters.feed_send_errors.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(%error, "carrier feed frame send failed");
        }
    }
}
