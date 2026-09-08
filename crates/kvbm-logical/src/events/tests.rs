// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the events pipeline.
//!
//! These tests verify the end-to-end flow from BlockRegistry through
//! EventsManager, EventBatcher, and KvbmCacheEventsPublisher.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use futures::StreamExt;
use futures::future::BoxFuture;
use tokio::sync::mpsc;

use super::batcher::BatchingConfig;
use super::manager::EventsManager;
use super::protocol::{KvCacheEvent, KvCacheEvents, KvbmCacheEvents};
use super::publisher::KvbmCacheEventsPublisher;
use crate::pubsub::Publisher;
use crate::registry::BlockRegistry;
use crate::{KvbmSequenceHashProvider, SequenceHash};
use dynamo_tokens::TokenBlockSequence;

fn create_seq_hash_at_position(position: usize) -> SequenceHash {
    let tokens_per_block = 4;
    let total_tokens = (position + 1) * tokens_per_block;
    let tokens: Vec<u32> = (0..total_tokens as u32).collect();
    let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(1337));
    seq.blocks()[position].kvbm_sequence_hash()
}

/// Mock publisher that captures published events via channel.
struct MockPublisher {
    captured_tx: mpsc::UnboundedSender<KvbmCacheEvents>,
}

impl MockPublisher {
    fn new(captured_tx: mpsc::UnboundedSender<KvbmCacheEvents>) -> Self {
        Self { captured_tx }
    }
}

impl Publisher for MockPublisher {
    fn publish(&self, _subject: &str, payload: Bytes) -> Result<()> {
        let events: KvbmCacheEvents = rmp_serde::from_slice(&payload)?;
        self.captured_tx.send(events).ok();
        Ok(())
    }

    fn flush(&self) -> BoxFuture<'static, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

/// Full pipeline test: BlockRegistry -> EventsManager -> Batcher -> Publisher
#[tokio::test]
async fn test_full_event_pipeline() {
    // 1. Setup - AllEventsPolicy is the default
    let manager = Arc::new(EventsManager::builder().build());
    let registry = BlockRegistry::new();

    // 2. Create mock publisher that captures events
    let (captured_tx, mut captured_rx) = mpsc::unbounded_channel();
    let mock_publisher = Arc::new(MockPublisher::new(captured_tx));

    // 3. Build pipeline
    let _publisher = KvbmCacheEventsPublisher::builder()
        .instance_id(12345)
        .event_stream(manager.subscribe())
        .publisher(mock_publisher)
        .batching_config(BatchingConfig::default().with_window(Duration::from_millis(50)))
        .build()
        .unwrap();

    // 4. Register blocks (triggers Create events)
    let seq_hashes: Vec<_> = (0..5).map(create_seq_hash_at_position).collect();
    let handles: Vec<_> = seq_hashes
        .iter()
        .map(|&hash| {
            let handle = registry.register_sequence_hash(hash);
            manager.on_block_registered(&handle);
            handle
        })
        .collect();

    // 5. Wait for batch window
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 6. Verify Create batch received
    let batch = tokio::time::timeout(Duration::from_millis(200), captured_rx.recv())
        .await
        .unwrap()
        .unwrap();

    assert!(matches!(batch.events, KvCacheEvents::Create(_)));
    assert_eq!(batch.instance_id, 12345);

    // Verify sorted by position ascending
    if let KvCacheEvents::Create(hashes) = &batch.events {
        assert_eq!(hashes.len(), 5);
        for i in 1..hashes.len() {
            assert!(
                hashes[i - 1].position() <= hashes[i].position(),
                "Create events should be sorted ascending by position"
            );
        }
    }

    // 7. Drop handles (triggers Remove events)
    drop(handles);
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 8. Verify Remove batch received
    let batch = tokio::time::timeout(Duration::from_millis(200), captured_rx.recv())
        .await
        .unwrap()
        .unwrap();

    assert!(matches!(batch.events, KvCacheEvents::Remove(_)));

    // Verify sorted by position descending
    if let KvCacheEvents::Remove(hashes) = &batch.events {
        assert_eq!(hashes.len(), 5);
        for i in 1..hashes.len() {
            assert!(
                hashes[i - 1].position() >= hashes[i].position(),
                "Remove events should be sorted descending by position"
            );
        }
    }
}

/// Test that type switches cause immediate flush
#[tokio::test]
async fn test_type_switch_flushes_batch() {
    let manager = Arc::new(EventsManager::builder().build());
    let registry = BlockRegistry::new();

    let (captured_tx, mut captured_rx) = mpsc::unbounded_channel();
    let mock_publisher = Arc::new(MockPublisher::new(captured_tx));

    // Use long window so we know flushes are due to type switch, not timeout
    let _publisher = KvbmCacheEventsPublisher::builder()
        .instance_id(12345)
        .event_stream(manager.subscribe())
        .publisher(mock_publisher)
        .batching_config(BatchingConfig::default().with_window(Duration::from_secs(60)))
        .build()
        .unwrap();

    // Register block (Create event)
    let hash1 = create_seq_hash_at_position(10);
    let handle1 = registry.register_sequence_hash(hash1);
    manager.on_block_registered(&handle1);

    // Drop block (Remove event) - should flush pending Create first
    drop(handle1);

    // Register another block (Create event) - should flush pending Remove
    let hash2 = create_seq_hash_at_position(20);
    let handle2 = registry.register_sequence_hash(hash2);
    manager.on_block_registered(&handle2);

    // Give time for events to propagate
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Should receive: Create batch (flushed on type switch to Remove)
    let batch1 = tokio::time::timeout(Duration::from_millis(200), captured_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(batch1.events, KvCacheEvents::Create(_)),
        "First batch should be Create"
    );

    // Should receive: Remove batch (flushed on type switch to Create)
    let batch2 = tokio::time::timeout(Duration::from_millis(200), captured_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(batch2.events, KvCacheEvents::Remove(_)),
        "Second batch should be Remove"
    );

    drop(handle2);
}

/// Test max batch size triggers flush
#[tokio::test]
async fn test_max_batch_size_flush() {
    let manager = Arc::new(EventsManager::builder().build());
    let registry = BlockRegistry::new();

    let (captured_tx, mut captured_rx) = mpsc::unbounded_channel();
    let mock_publisher = Arc::new(MockPublisher::new(captured_tx));

    let _publisher = KvbmCacheEventsPublisher::builder()
        .instance_id(12345)
        .event_stream(manager.subscribe())
        .publisher(mock_publisher)
        .batching_config(
            BatchingConfig::default()
                .with_window(Duration::from_secs(60)) // Long window
                .with_max_size(NonZeroUsize::new(3).unwrap()),
        )
        .build()
        .unwrap();

    // Register 5 blocks
    let handles: Vec<_> = (0..5)
        .map(|i| {
            let hash = create_seq_hash_at_position(i);
            let handle = registry.register_sequence_hash(hash);
            manager.on_block_registered(&handle);
            handle
        })
        .collect();

    // Give time for events to propagate
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Should receive first batch with 3 events (max size reached)
    let batch1 = tokio::time::timeout(Duration::from_millis(200), captured_rx.recv())
        .await
        .unwrap()
        .unwrap();
    if let KvCacheEvents::Create(hashes) = &batch1.events {
        assert_eq!(
            hashes.len(),
            3,
            "First batch should have max_size (3) events"
        );
    } else {
        panic!("Expected Create batch");
    }

    // Drop handles to allow remove events to proceed
    drop(handles);
}

/// Test multiple subscribers receive same events
#[tokio::test]
async fn test_multiple_subscribers() {
    let manager = Arc::new(EventsManager::builder().build());

    let mut stream1 = Box::pin(manager.subscribe());
    let mut stream2 = Box::pin(manager.subscribe());

    let registry = BlockRegistry::new();
    let hash = create_seq_hash_at_position(42);
    let handle = registry.register_sequence_hash(hash);
    manager.on_block_registered(&handle);

    // Both streams should receive the Create event
    let event1 = tokio::time::timeout(Duration::from_millis(100), stream1.next())
        .await
        .unwrap()
        .unwrap();
    let event2 = tokio::time::timeout(Duration::from_millis(100), stream2.next())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event1, KvCacheEvent::Create(hash));
    assert_eq!(event2, KvCacheEvent::Create(hash));

    // Drop handle to trigger Remove
    drop(handle);

    // Both should receive Remove
    let event1 = tokio::time::timeout(Duration::from_millis(100), stream1.next())
        .await
        .unwrap()
        .unwrap();
    let event2 = tokio::time::timeout(Duration::from_millis(100), stream2.next())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event1, KvCacheEvent::Remove(hash));
    assert_eq!(event2, KvCacheEvent::Remove(hash));
}

/// Test that events are properly serialized with msgpack
#[tokio::test]
async fn test_msgpack_serialization() {
    let hash = create_seq_hash_at_position(10);
    let batch = KvbmCacheEvents {
        events: KvCacheEvents::Create(vec![hash]),
        instance_id: 12345,
    };

    // Serialize with msgpack
    let bytes = rmp_serde::to_vec(&batch).unwrap();

    // Deserialize
    let decoded: KvbmCacheEvents = rmp_serde::from_slice(&bytes).unwrap();

    assert_eq!(decoded.instance_id, 12345);
    assert!(matches!(decoded.events, KvCacheEvents::Create(ref h) if h.len() == 1));
}

// ---------------------------------------------------------------------------
// R7b §7.8 — the legacy stream is byte-identical before and after
// ---------------------------------------------------------------------------

/// Fixed legacy batch, built from PLH parts rather than from a tokenized
/// sequence.
///
/// `create_seq_hash_at_position` routes through `TokenBlockSequence` and the
/// block-hash function, so a golden built on it would break on an upstream
/// `dynamo-tokens` hash change and read as a kvbm wire regression. §7.8 asks
/// whether *kvbm's encoding* moved, so the fixture pins the hash values
/// directly.
fn golden_legacy_batch() -> KvbmCacheEvents {
    let root = SequenceHash::new(0x0102_0304_0506_0708, None, 0);
    let child = SequenceHash::new(0x1112_1314_1516_1718, Some(0x0102_0304_0506_0708), 1);
    KvbmCacheEvents {
        events: KvCacheEvents::Create(vec![root, child]),
        instance_id: 0x0000_0000_0000_0000_DEAD_BEEF_CAFE_F00D,
    }
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The legacy wire encoding is frozen. R7b adds a *separate* stream on a
/// separate subject; if this assertion moves, the consolidator
/// (`kvbm_bridge.rs`) and the frozen dynamo `lib/kvbm-*` copy have been broken,
/// which is exactly what "legacy stream untouched" is supposed to prevent.
///
/// This is a byte comparison on purpose: a round-trip assertion still passes
/// after an encoding shift, so it does not test what §7.8 asks.
#[test]
fn legacy_wire_encoding_is_byte_identical() {
    // Layout: 2-element array (the struct's fields, positionally)
    //   [0] map {"Create": [<plh 16B>, <plh 16B>]}  — externally-tagged enum
    //   [1] 16-byte big-endian instance_id (u128)
    const GOLDEN_MSGPACK: &str = concat!(
        "9281a643726561746592",
        "c41000004080c1014181c200000000000000",
        "c41000444484c5054585c602030405060708",
        "c4100000000000000000deadbeefcafef00d",
    );
    let encoded = rmp_serde::to_vec(&golden_legacy_batch()).unwrap();
    assert_eq!(to_hex(&encoded), GOLDEN_MSGPACK);

    // ...and the golden bytes still decode to the same value.
    let decoded: KvbmCacheEvents = rmp_serde::from_slice(&encoded).unwrap();
    assert_eq!(decoded, golden_legacy_batch());
}

// ---------------------------------------------------------------------------
// Cross-decode guard: the two streams can never be confused for each other
// ---------------------------------------------------------------------------

fn tier_batch_msgpack() -> Vec<u8> {
    use kvbm_protocols::cache_manifest::{CacheManifestId, RegistrationEpoch};
    use kvbm_protocols::tier_protocol::{
        InstanceId, KeyRange, PhysicalPlacementMode, PlacementScope, TIER_PLACEMENT_SCHEMA_VERSION,
        TierDepth, TierPlacementBatchV1, TierPlacementOp,
    };

    let batch = TierPlacementBatchV1 {
        v: TIER_PLACEMENT_SCHEMA_VERSION,
        cache: CacheManifestId::from_bytes([5; 32]),
        instance_id: InstanceId::new_v4(),
        registration_epoch: RegistrationEpoch::new(),
        seq: 4,
        snapshot_generation: 1,
        ops: vec![TierPlacementOp::Ready {
            scope: PlacementScope::unitary(kvbm_common::LogicalResourceId(0)),
            tier: TierDepth(1),
            placement: PhysicalPlacementMode::Whole,
            generation: 2,
            keys: KeyRange::Hashes(vec![SequenceHash::new(0x2122_2324_2526_2728, None, 0)]),
        }],
    };
    rmp_serde::to_vec(&batch).unwrap()
}

/// A tier-placement frame must never decode as a legacy `KvbmCacheEvents`.
///
/// The hub's legacy indexer subscribes to `b""` — every ZMQ topic lands in the
/// same ingest loop — so "a new subject means old subscribers never see the new
/// frames" is false as written. Isolation has to come from the decoder, and a
/// silent mis-decode here would corrupt the block index rather than merely drop
/// a message. `rmp-serde` encodes structs positionally, so the 7-field tier
/// envelope should fail the 2-field legacy visitor — but that is asserted here,
/// not assumed.
#[test]
fn tier_frames_do_not_decode_as_legacy_events() {
    let tier_bytes = tier_batch_msgpack();
    assert!(
        rmp_serde::from_slice::<KvbmCacheEvents>(&tier_bytes).is_err(),
        "tier frame silently decoded as a legacy cache-event batch"
    );
}

/// ...and the converse, so a topic-dispatch bug in either direction is loud.
#[test]
fn legacy_frames_do_not_decode_as_tier_batches() {
    use kvbm_protocols::tier_protocol::TierPlacementBatchV1;

    let legacy_bytes = rmp_serde::to_vec(&golden_legacy_batch()).unwrap();
    assert!(
        rmp_serde::from_slice::<TierPlacementBatchV1>(&legacy_bytes).is_err(),
        "legacy frame silently decoded as a tier-placement batch"
    );
}
