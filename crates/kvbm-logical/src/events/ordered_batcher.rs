// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Order-preserving append-only batching (R7b §2).
//!
//! [`OrderedBatcher`] turns a stream of items into a stream of `Vec<T>`
//! batches, flushing on size, on a time window, or at end of input. That is the
//! whole contract, and the omissions are the point:
//!
//! - **Never reorders.** Items leave in arrival order, across batches and
//!   within them.
//! - **Never coalesces.** A `Ready` followed by a `Remove` for the same key
//!   yields *both*, in that order. On a sequenced, recoverable stream, sequence
//!   integrity is what substitutes for coalescing: dropping a pair because it
//!   "cancels out" would make the consumer's view depend on batch boundaries,
//!   and a consumer that later replays from a snapshot could not tell the
//!   difference between "nothing happened" and "two things happened".
//! - **Never drops.** Drop accounting belongs to the transport, which is where
//!   the per-reason counters live.
//!
//! Contrast [`EventBatcher`](super::batcher::EventBatcher), which serves the
//! legacy index stream and deliberately does all three: it sorts by position,
//! splits batches by event type, and exists to make radix-tree application
//! cheap. That is correct for a stream whose consumer is a set, and wrong for a
//! stream whose consumer is a sequence.
//!
//! The batcher is generic over the item type, which is both an ergonomic
//! convenience and the mechanical guarantee: it cannot inspect `T`, so it has
//! no way to decide two items are redundant. It also means this layer adds no
//! dependency edge — the tier-placement wire types live in `kvbm-protocols`
//! (see `kvbm_protocols::tier_protocol`), which this transport-free crate does
//! not and should not depend on.

use async_stream::stream;
use futures::Stream;
use futures::StreamExt;
use tokio::pin;

use super::batcher::BatchingConfig;

/// Order-preserving, append-only batcher.
///
/// Reuses [`BatchingConfig`] so the tier stream and the legacy stream are tuned
/// through one knob set.
#[derive(Debug, Clone, Default)]
pub struct OrderedBatcher {
    config: BatchingConfig,
}

impl OrderedBatcher {
    /// Creates a batcher with the given cadence.
    #[must_use]
    pub fn new(config: BatchingConfig) -> Self {
        Self { config }
    }

    /// The cadence this batcher flushes on.
    #[must_use]
    pub fn config(&self) -> &BatchingConfig {
        &self.config
    }

    /// Transform an input stream into a stream of batches.
    ///
    /// A batch is emitted when `max_batch_size` items have accumulated, when
    /// `window_duration` elapses with at least one item buffered, or when the
    /// input ends with items still buffered. Empty batches are never emitted.
    pub fn batch<S, T>(self, input: S) -> impl Stream<Item = Vec<T>> + Send
    where
        S: Stream<Item = T> + Send + 'static,
        T: Send + 'static,
    {
        let config = self.config;
        let max_batch_size = config.max_batch_size.get();

        stream! {
            pin!(input);

            let mut current: Vec<T> = Vec::with_capacity(max_batch_size);
            let mut deadline = tokio::time::Instant::now() + config.window_duration;

            loop {
                let timeout = tokio::time::sleep_until(deadline);

                tokio::select! {
                    biased;

                    maybe_item = input.next() => {
                        match maybe_item {
                            Some(item) => {
                                // Append only. No inspection of `item`, so no
                                // opportunity to coalesce or reorder.
                                current.push(item);

                                if current.len() >= max_batch_size {
                                    yield std::mem::replace(
                                        &mut current,
                                        Vec::with_capacity(max_batch_size),
                                    );
                                    deadline = tokio::time::Instant::now() + config.window_duration;
                                }
                            }
                            None => {
                                if !current.is_empty() {
                                    yield std::mem::take(&mut current);
                                }
                                break;
                            }
                        }
                    }

                    _ = timeout => {
                        if !current.is_empty() {
                            yield std::mem::replace(
                                &mut current,
                                Vec::with_capacity(max_batch_size),
                            );
                        }
                        deadline = tokio::time::Instant::now() + config.window_duration;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::time::Duration;

    use futures::stream;

    use super::*;

    /// Stand-in for `kvbm_protocols::tier_protocol::TierPlacementOp`, which
    /// this crate cannot see. Only the Ready/Remove shape matters to the
    /// batching contract.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Op {
        Ready(u32),
        Remove(u32),
    }

    #[tokio::test]
    async fn ready_then_remove_for_the_same_key_is_never_coalesced() {
        // The R7b §2 requirement, stated as a regression guard: identical
        // scope/tier/keys on both sides, so a coalescing batcher would emit one
        // item (or none) instead of two.
        let config = BatchingConfig::default().with_window(Duration::from_secs(60));
        let batcher = OrderedBatcher::new(config);

        let input = stream::iter(vec![Op::Ready(7), Op::Remove(7)]);
        let mut output = Box::pin(batcher.batch(input));

        let batch = output.next().await.expect("one batch");
        assert_eq!(batch, vec![Op::Ready(7), Op::Remove(7)]);
        assert!(output.next().await.is_none());
    }

    #[tokio::test]
    async fn arrival_order_survives_batch_boundaries() {
        let config = BatchingConfig::default()
            .with_window(Duration::from_secs(60))
            .with_max_size(NonZeroUsize::new(2).unwrap());
        let batcher = OrderedBatcher::new(config);

        // Descending keys: a batcher that sorted (as the legacy EventBatcher
        // does) would reverse these.
        let input = stream::iter(vec![
            Op::Ready(9),
            Op::Remove(5),
            Op::Ready(3),
            Op::Remove(9),
            Op::Ready(1),
        ]);
        let mut output = Box::pin(batcher.batch(input));

        assert_eq!(
            output.next().await.expect("first batch"),
            vec![Op::Ready(9), Op::Remove(5)]
        );
        assert_eq!(
            output.next().await.expect("second batch"),
            vec![Op::Ready(3), Op::Remove(9)]
        );
        assert_eq!(output.next().await.expect("tail batch"), vec![Op::Ready(1)]);
        assert!(output.next().await.is_none());
    }

    #[tokio::test]
    async fn type_switches_do_not_split_batches() {
        // The legacy batcher flushes on Create -> Remove; this one must not,
        // because a split there would be a reordering hazard the moment the
        // transport interleaves batches.
        let config = BatchingConfig::default().with_window(Duration::from_secs(60));
        let batcher = OrderedBatcher::new(config);

        let input = stream::iter(vec![
            Op::Ready(1),
            Op::Remove(2),
            Op::Ready(3),
            Op::Remove(4),
        ]);
        let mut output = Box::pin(batcher.batch(input));

        assert_eq!(
            output.next().await.expect("single batch"),
            vec![Op::Ready(1), Op::Remove(2), Op::Ready(3), Op::Remove(4)]
        );
        assert!(output.next().await.is_none());
    }

    #[tokio::test]
    async fn flushes_on_max_size() {
        let config = BatchingConfig::default()
            .with_window(Duration::from_secs(60))
            .with_max_size(NonZeroUsize::new(3).unwrap());
        let batcher = OrderedBatcher::new(config);

        let input = stream::iter((0..5).map(Op::Ready));
        let mut output = Box::pin(batcher.batch(input));

        assert_eq!(output.next().await.expect("full batch").len(), 3);
        assert_eq!(output.next().await.expect("tail batch").len(), 2);
        assert!(output.next().await.is_none());
    }

    #[tokio::test]
    async fn flushes_on_window_expiry() {
        let config = BatchingConfig::default().with_window(Duration::from_millis(50));
        let batcher = OrderedBatcher::new(config);

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let input = tokio_stream::wrappers::ReceiverStream::new(rx);
        let mut output = Box::pin(batcher.batch(input));

        tx.send(Op::Ready(1)).await.expect("send");
        let batch = tokio::time::timeout(Duration::from_millis(500), output.next())
            .await
            .expect("window flush")
            .expect("batch");
        assert_eq!(batch, vec![Op::Ready(1)]);

        drop(tx);
    }

    #[tokio::test]
    async fn empty_input_yields_no_batches() {
        let batcher = OrderedBatcher::new(BatchingConfig::default());
        let mut output = Box::pin(batcher.batch(stream::iter(Vec::<Op>::new())));
        assert!(output.next().await.is_none());
    }
}
