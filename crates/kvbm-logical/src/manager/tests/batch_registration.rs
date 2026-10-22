// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The batch registration path gives the same result as one
//! `register_block` call for each block, in input order.

use std::pin::Pin;

use futures::{FutureExt, Stream, StreamExt};

use super::*;
use crate::events::{EventsManager, KvCacheEvent};
use crate::metrics::MetricsSnapshot;
use crate::pools::store::DebugStoreSnapshot;

type Events = Pin<Box<dyn Stream<Item = KvCacheEvent> + Send>>;

#[derive(Clone, Copy, Debug)]
enum Backend {
    Lru,
    Lineage,
}

fn manager(
    policy: BlockDuplicationPolicy,
    backend: Backend,
    block_count: usize,
) -> (BlockManager<TestBlockData>, Events) {
    let events = Arc::new(EventsManager::builder().build());
    let stream: Events = Box::pin(events.subscribe());
    let builder = BlockManager::<TestBlockData>::builder()
        .block_count(block_count)
        .block_size(4)
        .registry(BlockRegistry::builder().event_manager(events).build())
        .duplication_policy(policy);
    let builder = match backend {
        Backend::Lru => builder.with_lru_backend(),
        Backend::Lineage => builder.with_lineage_backend(),
    };
    (builder.build().expect("build manager"), stream)
}

fn drain(events: &mut Events) -> Vec<KvCacheEvent> {
    let mut drained = Vec::new();
    while let Some(Some(event)) = events.next().now_or_never() {
        drained.push(event);
    }
    drained
}

/// A chain of `count` blocks; `repeat` maps an index to an earlier index
/// whose tokens it reuses, so the batch holds a repeated hash.
fn token_blocks(count: usize, repeat: &[(usize, usize)]) -> Vec<dynamo_tokens::TokenBlock> {
    let tokens: Vec<u32> = (0..(count as u32 * 4)).collect();
    let sequence = dynamo_tokens::TokenBlockSequence::from_slice(&tokens, 4, Some(7));
    let mut blocks = sequence.blocks().to_vec();
    for &(index, source) in repeat {
        blocks[index] = blocks[source].clone();
    }
    blocks
}

fn stage(
    manager: &BlockManager<TestBlockData>,
    blocks: &[dynamo_tokens::TokenBlock],
) -> Vec<CompleteBlock<TestBlockData>> {
    manager
        .allocate_blocks(blocks.len())
        .expect("allocate")
        .into_iter()
        .zip(blocks)
        .map(|(block, token_block)| block.complete(token_block).expect("stage"))
        .collect()
}

#[derive(Debug, PartialEq)]
struct Outcome {
    registered: Vec<(BlockId, SequenceHash)>,
    events: Vec<KvCacheEvent>,
    metrics: MetricsSnapshot,
    store: DebugStoreSnapshot,
    present: Vec<bool>,
}

/// Registers `blocks` after a block at `held` is already registered, one
/// call for each block or as one batch.
fn register(
    policy: BlockDuplicationPolicy,
    backend: Backend,
    blocks: &[dynamo_tokens::TokenBlock],
    held: usize,
    batch: bool,
) -> Outcome {
    let (manager, mut events) = manager(policy, backend, 2 * blocks.len() + 2);
    let _held = manager.register_block(stage(&manager, &blocks[held..=held]).remove(0));
    let staged = stage(&manager, blocks);
    let registered = if batch {
        manager.register_blocks(staged)
    } else {
        staged
            .into_iter()
            .map(|block| manager.register_block(block))
            .collect()
    };
    let hashes: Vec<_> = registered.iter().map(|b| b.sequence_hash()).collect();
    let present = hashes
        .iter()
        .map(|hash| {
            manager
                .block_registry()
                .match_sequence_hash(*hash, false)
                .is_some_and(|handle| handle.has_block::<TestBlockData>())
        })
        .collect();
    let outcome = Outcome {
        registered: registered
            .iter()
            .map(|block| (block.block_id(), block.sequence_hash()))
            .collect(),
        events: drain(&mut events),
        metrics: manager.metrics().snapshot(),
        store: manager.store_for_test().debug_snapshot(),
        present,
    };
    drop(registered);
    outcome
}

#[rstest]
fn batch_matches_per_block_registration(
    #[values(BlockDuplicationPolicy::Allow, BlockDuplicationPolicy::Reject)]
    policy: BlockDuplicationPolicy,
    #[values(Backend::Lru, Backend::Lineage)] backend: Backend,
) {
    // Index 5 repeats index 1 in the same batch; index 3 is already
    // registered before the batch.
    let blocks = token_blocks(8, &[(5, 1)]);
    let per_block = register(policy, backend, &blocks, 3, false);
    let batch = register(policy, backend, &blocks, 3, true);
    assert_eq!(batch, per_block);
    assert_eq!(batch.registered.len(), blocks.len());
}

#[rstest]
fn batch_larger_than_one_chunk_matches_per_block_registration(
    #[values(BlockDuplicationPolicy::Allow, BlockDuplicationPolicy::Reject)]
    policy: BlockDuplicationPolicy,
    #[values(Backend::Lru, Backend::Lineage)] backend: Backend,
) {
    // A hash of the first chunk repeats in the second chunk.
    let count = REGISTER_CHUNK_BLOCKS + 5;
    let blocks = token_blocks(count, &[(REGISTER_CHUNK_BLOCKS + 2, 4)]);
    let per_block = register(policy, backend, &blocks, 0, false);
    let batch = register(policy, backend, &blocks, 0, true);
    assert_eq!(batch, per_block);
}

#[test]
fn empty_batch_changes_nothing() {
    let (manager, mut events) = manager(BlockDuplicationPolicy::Reject, Backend::Lineage, 4);
    let before = (
        manager.metrics().snapshot(),
        manager.store_for_test().debug_snapshot(),
    );
    assert!(manager.register_blocks(Vec::new()).is_empty());
    assert_eq!(
        (
            manager.metrics().snapshot(),
            manager.store_for_test().debug_snapshot()
        ),
        before
    );
    assert!(drain(&mut events).is_empty());
}
