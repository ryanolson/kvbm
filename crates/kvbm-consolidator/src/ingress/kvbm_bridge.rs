// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVBM ingress bridge: pumps [`kvbm_logical::events::protocol::KvCacheEvent`]s from
//! an [`kvbm_logical::events::EventsManager`] subscription into the tracker.
//!
//! `KvCacheEvent::Create(seq_hash)` → `tracker.handle_kvbm_store(seq_hash, [], 0, None)`.
//! Carrier creates and snapshots are not consumed by this block-only bridge.
//! The bridge has no tokens/block_size/lora_name to forward — the canonical key
//! is the PLH, and a placeholder Store would be invalid downstream — so the
//! tracker registers blocks without publishing.
//!
//! `KvCacheEvent::Remove(seq_hash)` → `tracker.handle_kvbm_remove(seq_hash)`.
//!
//! The bridge accepts an `impl Stream<Item = KvCacheEvent>` so callers can pass
//! `EventsManager::subscribe()` directly (it returns a Stream, not a broadcast
//! Receiver). Tests can wrap a `broadcast::Receiver` with `BroadcastStream` if they
//! want broadcast semantics.

use std::sync::Arc;

use futures::{Stream, StreamExt};
use kvbm_logical::events::protocol::KvCacheEvent;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::tracker::Tracker;

/// Emit an audit log for a single KVBM event.
///
/// Called once per block event, before the tracker write.
pub fn audit_kvbm_event(event: &KvCacheEvent) {
    match event {
        KvCacheEvent::Create(seq_hash) => {
            tracing::info!(
                target: "kvbm_consolidator_audit",
                event = "ingress_kvbm",
                kind = "store",
                seq_hash = format!("{:032x}", seq_hash.as_u128()),
                num_blocks = 1usize,
            );
        }
        KvCacheEvent::Remove(seq_hash) => {
            tracing::info!(
                target: "kvbm_consolidator_audit",
                event = "ingress_kvbm",
                kind = "remove",
                seq_hash = format!("{:032x}", seq_hash.as_u128()),
                num_blocks = 1usize,
            );
        }
        KvCacheEvent::Snapshot(_) => {
            tracing::debug!("consolidator: dropping unsupported KVBM snapshot event");
        }
    }
}

/// Spawn the bridge task. Returns a `JoinHandle` that completes when `cancel` fires or
/// the upstream stream ends.
pub fn spawn<S>(
    stream: S,
    tracker: Arc<RwLock<Tracker>>,
    cancel: CancellationToken,
) -> JoinHandle<()>
where
    S: Stream<Item = KvCacheEvent> + Send + 'static,
{
    tokio::spawn(async move {
        tokio::pin!(stream);
        loop {
            tokio::select! {
                biased;

                _ = cancel.cancelled() => break,

                next = stream.next() => match next {
                    Some(KvCacheEvent::Create(seq_hash)) => {
                        audit_kvbm_event(&KvCacheEvent::Create(seq_hash));
                        tracker.write().await.handle_kvbm_store(seq_hash, Vec::new(), 0, None);
                    }
                    Some(KvCacheEvent::Remove(seq_hash)) => {
                        audit_kvbm_event(&KvCacheEvent::Remove(seq_hash));
                        tracker.write().await.handle_kvbm_remove(seq_hash);
                    }
                    Some(event @ KvCacheEvent::Snapshot(_)) => {
                        tracing::debug!(?event, "consolidator: dropping unsupported KVBM snapshot event");
                    }
                    None => break,
                },
            }
        }
    })
}
