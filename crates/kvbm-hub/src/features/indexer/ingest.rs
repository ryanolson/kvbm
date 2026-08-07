// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ ingest loop: dispatch published frames by topic, decode them, and apply
//! them to the [`PositionalIndex`] or the [`TierPlacementProjection`].
//!
//! # Why this dispatches on the topic frame
//!
//! R7b §6 assumes a new ZMQ subject means "old subscribers never see unknown
//! frames". That is false for this hub: [`bind_sub_socket`](super::zmq) calls
//! `subscribe(b"")`, i.e. *every* topic, so a new subject lands in this same
//! loop. Isolation has to come from the decoder. Without the dispatch below,
//! every tier-placement batch would arrive as a warn-logged "undecodable batch"
//! — and a mis-decode, rather than a drop, would corrupt the block index.
//! `kvbm-logical`'s cross-decode tests assert neither stream can be read as the
//! other, which is the guardrail behind this dispatch.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use futures::StreamExt;
use kvbm_logical::events::KvbmCacheEvents;
use kvbm_protocols::tier_protocol::{
    TIER_PLACEMENT_SUBJECT, TierPlacementBatchV1, TierPlacementRejection,
};
use tmq::subscribe::Subscribe;
use tokio_util::sync::CancellationToken;

use super::index::PositionalIndex;
use super::tier_placement::TierPlacementProjection;

/// ZMQ topic frame the legacy KV index stream publishes under. Matches
/// `kvbm-connector`'s `ZmqHubPublisher::SUBJECT`.
pub const LEGACY_INDEX_SUBJECT: &str = "kvbm.kv_index";

/// Per-reason ingest drop counters.
///
/// R7b §8 makes these mandatory, and the split matters operationally: an
/// unsupported version means "upgrade the consumer", an undecodable frame means
/// "the link is damaged", and an unknown topic means "someone is publishing
/// something this build does not know about". One counter for all three would
/// answer none of those questions.
#[derive(Debug, Default)]
pub struct IngestCounters {
    /// Legacy batches applied to the positional index.
    pub legacy_applied: AtomicU64,
    /// Legacy frames that did not deserialize.
    pub legacy_undecodable: AtomicU64,
    /// Frames with no topic frame, routed to the legacy decoder for
    /// pre-dispatch compatibility.
    pub legacy_untopiced: AtomicU64,
    /// Tier-placement batches handed to the projection.
    pub tier_accepted: AtomicU64,
    /// Tier-placement frames that did not deserialize.
    pub tier_undecodable: AtomicU64,
    /// Tier-placement frames announcing a schema version this build cannot read.
    pub tier_unsupported_version: AtomicU64,
    /// Tier-placement frames that decoded but violate the schema contract.
    pub tier_invalid: AtomicU64,
    /// Frames on a topic this build does not handle.
    pub unknown_topic: AtomicU64,
    /// Multipart messages with no frames at all.
    pub empty_multipart: AtomicU64,
}

impl IngestCounters {
    fn record_tier_rejection(&self, rejection: TierPlacementRejection) {
        let counter = match rejection {
            TierPlacementRejection::Undecodable => &self.tier_undecodable,
            TierPlacementRejection::UnsupportedVersion => &self.tier_unsupported_version,
            TierPlacementRejection::Invalid => &self.tier_invalid,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// Everything the ingest loop writes into.
pub struct IngestSinks {
    /// Legacy positional block index.
    pub index: Arc<PositionalIndex>,
    /// Advisory tier-placement projection.
    pub tier_placements: Arc<TierPlacementProjection>,
    /// Per-reason drop counters.
    pub counters: Arc<IngestCounters>,
}

/// Drains the bound `SUB` socket until cancelled.
///
/// Frame 0 is the topic; the last frame is the msgpack payload. Malformed
/// frames are counted and skipped — one bad message never tears down ingest.
pub async fn run_ingest_loop(mut sub: Subscribe, sinks: IngestSinks, cancel: CancellationToken) {
    tracing::info!("indexer ingest loop started");
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            msg = sub.next() => match msg {
                Some(Ok(multipart)) => {
                    let Some(payload) = multipart.iter().last() else {
                        sinks.counters.empty_multipart.fetch_add(1, Ordering::Relaxed);
                        tracing::trace!("indexer: empty multipart, skipping");
                        continue;
                    };
                    let first = multipart.iter().next().map(|frame| &**frame);
                    let topic = topic_of(&sinks, multipart.len(), first);
                    dispatch(&sinks, topic, payload);
                }
                Some(Err(e)) => {
                    tracing::warn!(error = %e, "indexer: ZMQ recv error");
                }
                None => {
                    tracing::info!("indexer: SUB stream ended");
                    break;
                }
            },
        }
    }
    tracing::info!("indexer ingest loop stopped");
}

/// Pick the topic a multipart message should be routed under.
///
/// A single-frame message carries no topic. Every publisher in this tree
/// prepends one, but the pre-dispatch loop decoded whatever it received as
/// legacy, so an untopiced frame keeps that meaning rather than silently
/// becoming an unknown-topic drop. It cannot be mistaken for a tier frame: those
/// always carry their subject, and the cross-decode tests in `kvbm-logical`
/// prove neither stream decodes as the other regardless.
fn topic_of<'a>(sinks: &IngestSinks, frames: usize, first: Option<&'a [u8]>) -> &'a [u8] {
    match first {
        Some(topic) if frames > 1 => topic,
        _ => {
            sinks
                .counters
                .legacy_untopiced
                .fetch_add(1, Ordering::Relaxed);
            LEGACY_INDEX_SUBJECT.as_bytes()
        }
    }
}

/// Route one frame by topic. Split out from the loop so the dispatch table is
/// directly testable without a socket.
pub(super) fn dispatch(sinks: &IngestSinks, topic: &[u8], payload: &[u8]) {
    if topic == LEGACY_INDEX_SUBJECT.as_bytes() {
        match rmp_serde::from_slice::<KvbmCacheEvents>(payload) {
            Ok(batch) => {
                sinks.index.apply(batch);
                sinks
                    .counters
                    .legacy_applied
                    .fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                sinks
                    .counters
                    .legacy_undecodable
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(%error, bytes = payload.len(), "indexer: undecodable batch, dropping");
            }
        }
        return;
    }
    if topic == TIER_PLACEMENT_SUBJECT.as_bytes() {
        // One gate: decode and validate whole, then hand a guaranteed-valid
        // batch to the projection. A version rejection is counted apart from a
        // corrupt frame, which is why this is `from_decoded` rather than a bare
        // deserialize.
        match TierPlacementBatchV1::from_decoded(rmp_serde::from_slice::<TierPlacementBatchV1>(
            payload,
        )) {
            Ok(batch) => {
                sinks.tier_placements.apply_delta(&batch);
                sinks.counters.tier_accepted.fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                sinks.counters.record_tier_rejection(error.rejection());
                tracing::debug!(%error, bytes = payload.len(), "indexer: tier placement frame dropped");
            }
        }
        return;
    }
    sinks.counters.unknown_topic.fetch_add(1, Ordering::Relaxed);
    tracing::trace!(
        topic = %String::from_utf8_lossy(topic),
        bytes = payload.len(),
        "indexer: unknown topic, dropping"
    );
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::RwLock;

    use kvbm_common::LogicalResourceId;
    use kvbm_logical::SequenceHash;
    use kvbm_logical::events::KvCacheEvents;
    use kvbm_protocols::cache_manifest::{CacheManifestId, RegistrationEpoch};
    use kvbm_protocols::tier_protocol::{
        InstanceId, KeyRange, PhysicalPlacementMode, PlacementScope, TIER_PLACEMENT_SCHEMA_VERSION,
        TierDepth, TierPlacementBatchV1, TierPlacementOp,
    };

    use super::*;

    fn sinks(instance: InstanceId) -> IngestSinks {
        let registered = Arc::new(RwLock::new(HashSet::from([instance])));
        IngestSinks {
            index: Arc::new(PositionalIndex::new(128, 4).unwrap()),
            tier_placements: Arc::new(TierPlacementProjection::new(registered)),
            counters: Arc::new(IngestCounters::default()),
        }
    }

    fn legacy_payload(instance: InstanceId, hash: SequenceHash) -> Vec<u8> {
        rmp_serde::to_vec(&KvbmCacheEvents {
            events: KvCacheEvents::Create(vec![hash]),
            instance_id: instance.as_u128(),
        })
        .unwrap()
    }

    fn tier_batch(instance: InstanceId, version: u16) -> TierPlacementBatchV1 {
        TierPlacementBatchV1 {
            v: version,
            cache: CacheManifestId::from_bytes([5; 32]),
            instance_id: instance,
            registration_epoch: RegistrationEpoch::new(),
            seq: 1,
            snapshot_generation: 1,
            ops: vec![TierPlacementOp::Ready {
                scope: PlacementScope::unitary(LogicalResourceId(0)),
                tier: TierDepth(1),
                placement: PhysicalPlacementMode::Whole,
                generation: 1,
                keys: KeyRange::Hashes(vec![SequenceHash::root(7)]),
            }],
        }
    }

    /// The `subscribe(b"")` finding, asserted: both streams land in this loop,
    /// and only the topic frame keeps them apart.
    #[test]
    fn each_topic_reaches_only_its_own_consumer() {
        let instance = InstanceId::new_v4();
        let hash = SequenceHash::root(3);
        let sinks = sinks(instance);

        dispatch(
            &sinks,
            LEGACY_INDEX_SUBJECT.as_bytes(),
            &legacy_payload(instance, hash),
        );
        assert!(sinks.index.query(&[hash]).is_some());
        assert_eq!(sinks.counters.legacy_applied.load(Ordering::Relaxed), 1);
        assert_eq!(sinks.counters.tier_accepted.load(Ordering::Relaxed), 0);

        let tier = rmp_serde::to_vec(&tier_batch(instance, TIER_PLACEMENT_SCHEMA_VERSION)).unwrap();
        dispatch(&sinks, TIER_PLACEMENT_SUBJECT.as_bytes(), &tier);
        assert_eq!(sinks.counters.tier_accepted.load(Ordering::Relaxed), 1);
        // The tier frame did not reach the block index...
        assert_eq!(sinks.counters.legacy_applied.load(Ordering::Relaxed), 1);
        assert_eq!(sinks.counters.legacy_undecodable.load(Ordering::Relaxed), 0);
        // ...and the block index still holds only what the legacy frame put
        // there.
        assert!(sinks.index.query(&[hash]).is_some());
    }

    #[test]
    fn an_unknown_topic_touches_neither_consumer() {
        let instance = InstanceId::new_v4();
        let sinks = sinks(instance);
        let hash = SequenceHash::root(3);

        dispatch(
            &sinks,
            b"kvbm.some_future_stream",
            &legacy_payload(instance, hash),
        );
        assert_eq!(sinks.counters.unknown_topic.load(Ordering::Relaxed), 1);
        assert_eq!(sinks.counters.legacy_applied.load(Ordering::Relaxed), 0);
        assert_eq!(sinks.counters.tier_accepted.load(Ordering::Relaxed), 0);
        assert!(sinks.index.query(&[hash]).is_none());
    }

    /// A version rejection and a corrupt frame must never share a counter: one
    /// means "upgrade the consumer", the other means "the link is damaged".
    #[test]
    fn tier_rejections_are_counted_by_reason() {
        let instance = InstanceId::new_v4();
        let sinks = sinks(instance);

        let future =
            rmp_serde::to_vec(&tier_batch(instance, TIER_PLACEMENT_SCHEMA_VERSION + 1)).unwrap();
        dispatch(&sinks, TIER_PLACEMENT_SUBJECT.as_bytes(), &future);
        assert_eq!(
            sinks
                .counters
                .tier_unsupported_version
                .load(Ordering::Relaxed),
            1
        );

        let mut g1 = tier_batch(instance, TIER_PLACEMENT_SCHEMA_VERSION);
        g1.ops = vec![TierPlacementOp::Ready {
            scope: PlacementScope::unitary(LogicalResourceId(0)),
            tier: TierDepth::G1,
            placement: PhysicalPlacementMode::Whole,
            generation: 1,
            keys: KeyRange::Hashes(vec![SequenceHash::root(7)]),
        }];
        dispatch(
            &sinks,
            TIER_PLACEMENT_SUBJECT.as_bytes(),
            &rmp_serde::to_vec(&g1).unwrap(),
        );
        assert_eq!(sinks.counters.tier_invalid.load(Ordering::Relaxed), 1);

        dispatch(&sinks, TIER_PLACEMENT_SUBJECT.as_bytes(), b"not msgpack");
        assert_eq!(sinks.counters.tier_undecodable.load(Ordering::Relaxed), 1);

        assert_eq!(sinks.counters.tier_accepted.load(Ordering::Relaxed), 0);
    }

    /// Every publisher in this tree prepends a topic frame, so this path is
    /// defensive — but it is the one behaviour the dispatch could silently
    /// change, so it is pinned rather than assumed. `topic_of` is the loop's
    /// own selection logic, exercised without a socket.
    #[test]
    fn an_untopiced_frame_keeps_its_pre_dispatch_legacy_meaning() {
        let instance = InstanceId::new_v4();
        let sinks = sinks(instance);
        let hash = SequenceHash::root(3);
        let payload = legacy_payload(instance, hash);

        let topic = topic_of(&sinks, 1, None);
        assert_eq!(topic, LEGACY_INDEX_SUBJECT.as_bytes());
        assert_eq!(sinks.counters.legacy_untopiced.load(Ordering::Relaxed), 1);
        dispatch(&sinks, topic, &payload);
        assert!(sinks.index.query(&[hash]).is_some());

        // A two-frame message uses its topic, and does not count as untopiced.
        assert_eq!(
            topic_of(&sinks, 2, Some(TIER_PLACEMENT_SUBJECT.as_bytes())),
            TIER_PLACEMENT_SUBJECT.as_bytes()
        );
        assert_eq!(sinks.counters.legacy_untopiced.load(Ordering::Relaxed), 1);
    }

    /// Misrouting must be loud, not silently corrupting. A tier frame delivered
    /// on the legacy topic has to fail the legacy decoder rather than land in
    /// the block index as garbage.
    #[test]
    fn a_misrouted_tier_frame_cannot_corrupt_the_block_index() {
        let instance = InstanceId::new_v4();
        let sinks = sinks(instance);
        let tier = rmp_serde::to_vec(&tier_batch(instance, TIER_PLACEMENT_SCHEMA_VERSION)).unwrap();

        dispatch(&sinks, LEGACY_INDEX_SUBJECT.as_bytes(), &tier);
        assert_eq!(sinks.counters.legacy_undecodable.load(Ordering::Relaxed), 1);
        assert_eq!(sinks.counters.legacy_applied.load(Ordering::Relaxed), 0);
    }
}
