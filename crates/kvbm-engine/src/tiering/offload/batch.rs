// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Batch collection for complete offload containers.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, watch};

use kvbm_logical::blocks::BlockMetadata;

use super::container::OffloadContainer;
use super::handle::TransferId;
use super::pipeline::shutdown::PRECOMMIT_SHUTDOWN_ERROR;
use super::queue::CancellableQueue;

/// Timing trace for a transfer batch.
#[derive(Debug, Clone)]
pub(crate) struct TimingTrace {
    /// When the request entered the pipeline.
    pub enqueued_at: Instant,
    /// When policy evaluation completed.
    pub policy_complete_at: Option<Instant>,
    /// When the precondition completed.
    pub precondition_complete_at: Option<Instant>,
    /// When the container entered a batch.
    pub batched_at: Option<Instant>,
    /// When physical work started.
    pub transfer_start_at: Option<Instant>,
    /// When physical work completed.
    pub transfer_complete_at: Option<Instant>,
}

impl TimingTrace {
    /// Create a trace with the enqueue time set to now.
    pub fn new() -> Self {
        Self {
            enqueued_at: Instant::now(),
            policy_complete_at: None,
            precondition_complete_at: None,
            batched_at: None,
            transfer_start_at: None,
            transfer_complete_at: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn mark_policy_complete(&mut self) {
        self.policy_complete_at = Some(Instant::now());
    }

    #[cfg(test)]
    pub(crate) fn mark_precondition_complete(&mut self) {
        self.precondition_complete_at = Some(Instant::now());
    }

    pub fn mark_batched(&mut self) {
        self.batched_at = Some(Instant::now());
    }

    pub fn mark_transfer_start(&mut self) {
        self.transfer_start_at = Some(Instant::now());
    }

    pub fn mark_transfer_complete(&mut self) {
        self.transfer_complete_at = Some(Instant::now());
    }

    pub fn total_duration(&self) -> Option<Duration> {
        self.transfer_complete_at
            .map(|end| end.duration_since(self.enqueued_at))
    }

    pub fn policy_duration(&self) -> Option<Duration> {
        self.policy_complete_at
            .map(|end| end.duration_since(self.enqueued_at))
    }

    pub fn precondition_duration(&self) -> Option<Duration> {
        match (self.policy_complete_at, self.precondition_complete_at) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    pub fn transfer_duration(&self) -> Option<Duration> {
        match (self.transfer_start_at, self.transfer_complete_at) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }
}

impl Default for TimingTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for batch collection.
#[derive(Debug, Clone)]
pub(crate) struct BatchConfig {
    /// Maximum blocks per batch.
    pub max_batch_size: usize,
    /// Time before a partial batch flushes.
    pub flush_interval: Duration,
    /// Minimum blocks for an interval flush.
    pub min_batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1024,
            flush_interval: Duration::from_millis(10),
            min_batch_size: 8,
        }
    }
}

impl BatchConfig {
    #[cfg(test)]
    pub(crate) fn with_max_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    #[cfg(test)]
    pub(crate) fn with_flush_interval(mut self, interval: Duration) -> Self {
        self.flush_interval = interval;
        self
    }

    #[cfg(test)]
    pub(crate) fn with_min_size(mut self, size: usize) -> Self {
        self.min_batch_size = size;
        self
    }
}

/// A batch of whole cancellation containers.
pub(crate) struct TransferBatch<T: BlockMetadata> {
    /// Containers remain intact until the weak-to-strong upgrade boundary.
    pub(crate) containers: Vec<OffloadContainer<T>>,
    /// Batch timing data.
    pub timing: TimingTrace,
}

impl<T: BlockMetadata> TransferBatch<T> {
    pub fn new() -> Self {
        Self {
            containers: Vec::new(),
            timing: TimingTrace::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            containers: Vec::with_capacity(capacity),
            timing: TimingTrace::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_containers(containers: Vec<OffloadContainer<T>>) -> Self {
        Self {
            containers,
            timing: TimingTrace::new(),
        }
    }

    pub(crate) fn push_container(&mut self, container: OffloadContainer<T>) {
        self.containers.push(container);
    }

    /// Count blocks without flattening the containers.
    pub fn len(&self) -> usize {
        self.containers
            .iter()
            .map(OffloadContainer::evaluated_len)
            .sum()
    }

    pub(crate) fn container_len(&self) -> usize {
        self.containers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }

    /// Drop cancelled containers before the first weak-to-strong upgrade.
    pub(crate) fn sweep_cancelled(&mut self) -> usize {
        let mut removed = 0;
        let mut live = Vec::with_capacity(self.containers.len());
        for container in std::mem::take(&mut self.containers) {
            if container.is_cancelled() {
                removed += 1;
            } else {
                live.push(container);
            }
        }
        self.containers = live;
        removed
    }

    /// Fail each retained container after its source and pending guards release.
    pub(crate) fn fail(self, error: &str) {
        for container in self.containers {
            container.fail(error.to_string());
        }
    }
}

impl<T: BlockMetadata> Default for TransferBatch<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Sender for completed batches.
pub(crate) type BatchOutput<T> = mpsc::Sender<TransferBatch<T>>;
/// Receiver for completed batches.
pub(crate) type BatchOutputRx<T> = mpsc::Receiver<TransferBatch<T>>;

/// Collect precondition-ready containers into complete batches.
pub(crate) struct BatchCollector<T: BlockMetadata> {
    config: BatchConfig,
    input_queue: Arc<CancellableQueue<OffloadContainer<T>>>,
    output_tx: BatchOutput<T>,
    cancel_rx: watch::Receiver<HashSet<TransferId>>,
    shutdown_rx: watch::Receiver<bool>,
    current_batch: TransferBatch<T>,
}

impl<T: BlockMetadata> BatchCollector<T> {
    pub fn new(
        config: BatchConfig,
        input_queue: Arc<CancellableQueue<OffloadContainer<T>>>,
        output_tx: BatchOutput<T>,
        cancel_rx: watch::Receiver<HashSet<TransferId>>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        let max_batch_size = config.max_batch_size;
        Self {
            config,
            input_queue,
            output_tx,
            cancel_rx,
            shutdown_rx,
            current_batch: TransferBatch::with_capacity(max_batch_size),
        }
    }

    pub async fn run(mut self) {
        let mut flush_timer = tokio::time::interval(self.config.flush_interval);
        flush_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            if self.shutdown_requested() {
                self.fail_precommit_work();
                break;
            }

            if self.input_queue.is_closed() {
                self.fail_precommit_work();
                break;
            }

            while let Some(item) = self.input_queue.pop_valid() {
                self.handle_container(item.data).await;
            }

            if self.input_queue.is_closed() {
                self.fail_precommit_work();
                break;
            }

            tokio::select! {
                _ = self.input_queue.notified() => {}
                _ = flush_timer.tick() => self.try_flush().await,
                result = self.cancel_rx.changed() => {
                    if result.is_err() {
                        if self.shutdown_requested() {
                            self.fail_precommit_work();
                        } else {
                            self.flush_if_not_empty().await;
                        }
                        break;
                    }
                }
                result = self.shutdown_rx.changed() => {
                    let _ = result;
                    self.fail_precommit_work();
                        break;
                }
            }
        }
    }

    async fn handle_container(&mut self, container: OffloadContainer<T>) {
        if container.is_cancelled() {
            return;
        }

        let transfer_id = container.transfer_id();
        let input_len = container.source_len();
        let container_blocks = container.evaluated_len();
        let state = container.state();

        if !self.current_batch.is_empty()
            && self.current_batch.len() + container_blocks > self.config.max_batch_size
        {
            self.flush().await;
        }

        self.current_batch.push_container(container);
        if self.current_batch.len() >= self.config.max_batch_size {
            self.flush().await;
        }

        let should_flush = {
            let mut state = state.lock().unwrap();
            state.blocks_processed += input_len;
            state.blocks_processed >= state.total_expected_blocks && state.total_expected_blocks > 0
        };
        if should_flush && !self.current_batch.is_empty() {
            tracing::debug!(%transfer_id, batch_size = self.current_batch.len(), "Per-transfer sentinel flush");
            self.flush().await;
        }
    }

    async fn try_flush(&mut self) {
        if self.current_batch.len() >= self.config.min_batch_size {
            self.flush().await;
        }
    }

    async fn flush_if_not_empty(&mut self) {
        if !self.current_batch.is_empty() {
            self.flush().await;
        }
    }

    async fn flush(&mut self) {
        nvtx_range!("offload::batch");
        if self.current_batch.is_empty() {
            return;
        }

        let mut batch = std::mem::replace(
            &mut self.current_batch,
            TransferBatch::with_capacity(self.config.max_batch_size),
        );
        batch.timing.mark_batched();
        if self.shutdown_requested() {
            batch.fail(PRECOMMIT_SHUTDOWN_ERROR);
            return;
        }

        tokio::select! {
            reservation = self.output_tx.reserve() => {
                match reservation {
                    Ok(reservation) if !self.shutdown_requested() => reservation.send(batch),
                    Ok(_) => batch.fail(PRECOMMIT_SHUTDOWN_ERROR),
                    Err(_) => {
                        tracing::warn!("Batch output channel closed");
                        batch.fail("batch output channel closed");
                    }
                }
            }
            changed = self.shutdown_rx.changed() => {
                let _ = changed;
                batch.fail(PRECOMMIT_SHUTDOWN_ERROR);
            }
        }
    }

    fn shutdown_requested(&self) -> bool {
        *self.shutdown_rx.borrow()
    }

    fn fail_precommit_work(&mut self) {
        while let Some(item) = self.input_queue.pop() {
            item.data.fail(PRECOMMIT_SHUTDOWN_ERROR.to_string());
        }
        let batch = std::mem::replace(
            &mut self.current_batch,
            TransferBatch::with_capacity(self.config.max_batch_size),
        );
        batch.fail(PRECOMMIT_SHUTDOWN_ERROR);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_config_builder_sets_each_value() {
        let config = BatchConfig::default()
            .with_max_size(128)
            .with_min_size(16)
            .with_flush_interval(Duration::from_millis(50));

        assert_eq!(config.max_batch_size, 128);
        assert_eq!(config.min_batch_size, 16);
        assert_eq!(config.flush_interval, Duration::from_millis(50));
    }

    #[test]
    fn transfer_batch_starts_empty() {
        let batch: TransferBatch<()> = TransferBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.container_len(), 0);
    }

    #[tokio::test]
    async fn batch_collector_exits_on_closed_cancel_channel() {
        let input_queue = Arc::new(CancellableQueue::<OffloadContainer<()>>::new());
        let (output_tx, mut output_rx) = mpsc::channel::<TransferBatch<()>>(10);
        let (cancel_tx, cancel_rx) = watch::channel(HashSet::new());
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let collector = BatchCollector::new(
            BatchConfig::default(),
            input_queue,
            output_tx,
            cancel_rx,
            shutdown_rx,
        );

        drop(cancel_tx);
        tokio::spawn(async move {
            collector.run().await;
        });

        let result = tokio::time::timeout(Duration::from_millis(50), output_rx.recv()).await;
        assert!(result.is_err() || result.unwrap().is_none());
    }
}
