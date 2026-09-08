// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-index hub publisher wiring.
//!
//! The hub's KV-index ZMQ ingest endpoint is discovered by the hub handshake
//! ([`super::hub_handshake`]) from the aggregate `GET /v1/config`. When the
//! Indexer feature is effective and block-size-compatible, the connector
//! connects a ZMQ `PUB` socket to that endpoint and wires a [`Publisher`] into
//! the block-registry [`EventsManager`] so block create/remove events flow to
//! the hub's index. The connector also registers `Feature::Indexer` with the
//! hub so the hub's `on_unregister` sweep reclaims this instance's entries.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use bytes::Bytes;
use futures::SinkExt;
use futures::future::BoxFuture;
use kvbm_logical::pubsub::Publisher;
use tmq::{Context as ZmqContext, Multipart, publish::Publish, publish::publish};
use tokio::sync::{Mutex, mpsc};

/// Subject/topic frame prepended to each published batch.
pub const SUBJECT: &str = "kvbm.kv_index";

const ZMQ_LINGER_MS: i32 = 0;

/// [`Publisher`] backed by a ZMQ `PUB` socket connected to the hub's `SUB`
/// ingest socket.
///
/// The [`Publisher`] contract is synchronous (`publish` returns immediately),
/// but the tmq socket send is async. A bounded mpsc channel bridges the two: a
/// background task owns the socket and drains the channel. Backpressure drops
/// the newest batch, the one that `publish` hands off, and never blocks the
/// event pipeline, because KV index freshness is advisory. The channel keeps
/// what it already queued.
pub struct ZmqHubPublisher {
    tx: mpsc::Sender<Bytes>,
    dropped_batches: AtomicU64,
}

impl ZmqHubPublisher {
    /// Connects a `PUB` socket to `endpoint` and spawns the send task.
    pub fn connect(endpoint: &str) -> Result<Self> {
        let ctx = ZmqContext::new();
        let socket = publish(&ctx)
            .set_linger(ZMQ_LINGER_MS)
            .connect(endpoint)
            .with_context(|| format!("connecting indexer PUB socket to {endpoint}"))?;
        let socket = Arc::new(Mutex::new(socket));

        let (tx, mut rx) = mpsc::channel::<Bytes>(1024);
        tokio::spawn(async move {
            while let Some(payload) = rx.recv().await {
                if let Err(e) = send_batch(&socket, payload).await {
                    tracing::warn!(error = %e, "indexer PUB send failed");
                }
            }
            tracing::info!("indexer PUB send task stopped");
        });

        Ok(Self {
            tx,
            dropped_batches: AtomicU64::new(0),
        })
    }

    /// Cumulative count of batches dropped because `publish` found the
    /// channel full. The index is advisory, so the publisher does not
    /// repair the loss. The counter only makes the loss visible.
    pub fn dropped_batches(&self) -> u64 {
        self.dropped_batches.load(Ordering::Relaxed)
    }
}

async fn send_batch(socket: &Arc<Mutex<Publish>>, payload: Bytes) -> Result<()> {
    let frames: Vec<Vec<u8>> = vec![SUBJECT.as_bytes().to_vec(), payload.to_vec()];
    socket.lock().await.send(Multipart::from(frames)).await?;
    Ok(())
}

impl Publisher for ZmqHubPublisher {
    fn publish(&self, _subject: &str, payload: Bytes) -> Result<()> {
        // Non-blocking hand-off. On a full channel, drop the batch (advisory
        // index) rather than stalling the broadcast subscriber.
        match self.tx.try_send(payload) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                let dropped = self.dropped_batches.fetch_add(1, Ordering::Relaxed) + 1;
                tracing::warn!(dropped, "indexer publish channel full; dropping batch");
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                anyhow::bail!("indexer publish channel closed")
            }
        }
    }

    fn flush(&self) -> BoxFuture<'static, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl ZmqHubPublisher {
        /// Publisher over a caller-held channel, with no drain task and no
        /// ZMQ socket. Exercises the backpressure path of `publish` directly.
        /// The test holds the receiver, so a full channel is reachable
        /// without a slow or absent hub on a real socket.
        fn for_channel(capacity: usize) -> (Self, mpsc::Receiver<Bytes>) {
            let (tx, rx) = mpsc::channel(capacity);
            (
                Self {
                    tx,
                    dropped_batches: AtomicU64::new(0),
                },
                rx,
            )
        }
    }

    #[test]
    fn a_full_publish_channel_drops_the_newest_batch_and_counts_it() {
        let (publisher, mut rx) = ZmqHubPublisher::for_channel(1024);

        for i in 0..1025u32 {
            let payload = Bytes::from(i.to_be_bytes().to_vec());
            publisher
                .publish(SUBJECT, payload)
                .expect("publish must not error on a full channel");
        }

        assert_eq!(publisher.dropped_batches(), 1);

        let mut received = Vec::new();
        while let Ok(payload) = rx.try_recv() {
            received.push(payload);
        }
        assert_eq!(
            received.len(),
            1024,
            "the channel still holds the first 1024 batches"
        );
        assert_eq!(
            received.first(),
            Some(&Bytes::from(0u32.to_be_bytes().to_vec()))
        );
        assert_eq!(
            received.last(),
            Some(&Bytes::from(1023u32.to_be_bytes().to_vec()))
        );
    }
}
