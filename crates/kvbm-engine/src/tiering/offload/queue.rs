// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cancellable queue implementation using crossbeam SegQueue.
//!
//! Provides a concurrent queue wrapper that supports active cancellation via
//! a sweeper task that can iterate through queued items and remove those
//! belonging to cancelled transfers.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crossbeam_queue::SegQueue;
use dashmap::DashSet;
use parking_lot::Mutex;
use tokio::sync::Notify;

use super::handle::TransferId;

/// A queued item with its associated transfer ID.
pub(crate) struct QueueItem<T> {
    /// The transfer this item belongs to
    pub transfer_id: TransferId,
    /// The actual data
    pub data: T,
}

impl<T> QueueItem<T> {
    /// Create a new queue item.
    pub fn new(transfer_id: TransferId, data: T) -> Self {
        Self { transfer_id, data }
    }
}

/// A concurrent queue that supports active cancellation via sweeping.
///
/// Unlike mpsc channels where cancellation can only be checked at dequeue time,
/// this queue allows a dedicated sweeper task to iterate through queued items
/// and remove those belonging to cancelled transfers. This ensures that
/// `ImmutableBlock` guards are dropped promptly when a transfer is cancelled.
///
/// # Architecture
///
/// ```text
/// Producer ──► [SegQueue] ◄── Consumer
///                  ▲
///                  │
///             [Sweeper Task]
///                  │
///            (removes cancelled items)
/// ```
pub(crate) struct CancellableQueue<T> {
    /// The underlying lock-free queue
    inner: SegQueue<QueueItem<T>>,
    /// Set of cancelled transfer IDs
    cancelled: DashSet<TransferId>,
    /// Approximate length for monitoring (not exact due to concurrent access)
    len: AtomicUsize,
    /// Wakes consumers without waiting for a wall-clock poll interval.
    notify: Notify,
    /// Serializes producer admission, sweeping, and queue closure.
    producer_gate: Mutex<()>,
    /// Rejects work after the pipeline owner starts shutdown.
    closed: AtomicBool,
    /// Pauses one sweep before requeue for deterministic concurrency tests.
    #[cfg(test)]
    before_sweep_requeue: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl<T> CancellableQueue<T> {
    /// Create a new cancellable queue.
    pub fn new() -> Self {
        Self {
            inner: SegQueue::new(),
            cancelled: DashSet::new(),
            len: AtomicUsize::new(0),
            notify: Notify::new(),
            producer_gate: Mutex::new(()),
            closed: AtomicBool::new(false),
            #[cfg(test)]
            before_sweep_requeue: Mutex::new(None),
        }
    }

    /// Push an item onto the queue.
    ///
    /// If cancellation or closure rejects the item, this method drops it.
    /// The return value is true only when the queue accepts the item.
    #[cfg(test)]
    pub(crate) fn push(&self, transfer_id: TransferId, data: T) -> bool {
        self.push_or_return(transfer_id, data).is_ok()
    }

    /// Push an item, or return ownership after cancellation or closure.
    pub(crate) fn push_or_return(&self, transfer_id: TransferId, data: T) -> Result<(), T> {
        let _producer = self.producer_gate.lock();
        if self.closed.load(Ordering::Acquire) {
            return Err(data);
        }

        // Fast path: check if already cancelled before queuing
        if self.cancelled.contains(&transfer_id) {
            return Err(data);
        }

        self.inner.push(QueueItem::new(transfer_id, data));
        self.len.fetch_add(1, Ordering::Relaxed);
        self.notify.notify_one();
        Ok(())
    }

    /// Wait until a producer pushes new work or cancellation changes.
    pub async fn notified(&self) {
        self.notify.notified().await;
    }

    /// Close producer admission and return all queued payloads.
    ///
    /// The returned vector keeps each payload alive after the admission gate
    /// releases. The caller owns terminal handling for these payloads.
    pub(crate) fn close_and_drain(&self) -> Vec<T> {
        let drained = {
            let _producer = self.producer_gate.lock();
            self.closed.store(true, Ordering::Release);

            let mut drained = Vec::new();
            while let Some(QueueItem { data, .. }) = self.inner.pop() {
                drained.push(data);
            }
            if !drained.is_empty() {
                self.len.fetch_sub(drained.len(), Ordering::Relaxed);
            }
            drained
        };
        self.notify.notify_waiters();
        self.notify.notify_one();
        drained
    }

    /// Return true after producer admission closes.
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn pause_next_sweep_before_requeue(&self, pause: impl FnOnce() + Send + 'static) {
        *self.before_sweep_requeue.lock() = Some(Box::new(pause));
    }

    /// Hold producer admission for a deterministic queue close-and-drain test.
    #[cfg(test)]
    pub(crate) fn lock_admission_for_test(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.producer_gate.lock()
    }

    /// Pop an item from the queue.
    ///
    /// Returns `None` if the queue is empty.
    /// Items from cancelled transfers may still be returned - use `pop_valid()`
    /// if you want to skip cancelled items automatically.
    pub fn pop(&self) -> Option<QueueItem<T>> {
        let item = self.inner.pop();
        if item.is_some() {
            self.len.fetch_sub(1, Ordering::Relaxed);
        }
        item
    }

    /// Pop a valid (non-cancelled) item from the queue.
    ///
    /// Skips and drops items belonging to cancelled transfers.
    /// Returns `None` if no valid items are available.
    pub fn pop_valid(&self) -> Option<QueueItem<T>> {
        loop {
            match self.inner.pop() {
                Some(item) => {
                    self.len.fetch_sub(1, Ordering::Relaxed);
                    if self.cancelled.contains(&item.transfer_id) {
                        // Drop cancelled item and try again
                        continue;
                    }
                    return Some(item);
                }
                None => return None,
            }
        }
    }

    /// Mark a transfer as cancelled.
    ///
    /// Items belonging to this transfer will be:
    /// - Dropped immediately if pushed after this call
    /// - Removed by the sweeper task if already in the queue
    /// - Skipped by `pop_valid()` if dequeued
    pub fn mark_cancelled(&self, transfer_id: TransferId) {
        self.cancelled.insert(transfer_id);
        self.notify.notify_waiters();
    }

    /// Check if a transfer has been cancelled.
    #[cfg(test)]
    pub(crate) fn is_cancelled(&self, transfer_id: TransferId) -> bool {
        self.cancelled.contains(&transfer_id)
    }

    /// Remove cancelled items from the queue.
    ///
    /// This is called by the sweeper task to actively remove items from
    /// cancelled transfers, ensuring their resources (like `ImmutableBlock` guards)
    /// are released promptly.
    ///
    /// Returns the number of items removed.
    ///
    /// # Implementation Note
    ///
    /// This performs a full drain-and-requeue operation. While not ideal for
    /// very large queues, it ensures correctness with the lock-free SegQueue.
    /// For typical offload workloads (batches of 64-256 blocks), this is efficient.
    pub fn sweep(&self) -> usize {
        let removed = {
            let _producer = self.producer_gate.lock();
            if self.closed.load(Ordering::Acquire) {
                return 0;
            }

            if self.cancelled.is_empty() {
                return 0;
            }

            // Drain all items and requeue non-cancelled ones.
            let mut removed = Vec::new();
            let mut kept = Vec::new();

            while let Some(item) = self.inner.pop() {
                if self.cancelled.contains(&item.transfer_id) {
                    removed.push(item);
                } else {
                    kept.push(item);
                }
            }

            #[cfg(test)]
            if let Some(pause) = self.before_sweep_requeue.lock().take() {
                pause();
            }

            let restored_work = !kept.is_empty();

            // Requeue kept items.
            for item in kept {
                self.inner.push(item);
            }

            // A concurrent consumer can observe the queue as empty while this
            // sweep owns the retained items. Store one availability permit after
            // requeue so that a consumer which has not yet registered cannot
            // remain asleep until unrelated work arrives.
            if restored_work {
                self.notify.notify_one();
            }

            if !removed.is_empty() {
                self.len.fetch_sub(removed.len(), Ordering::Relaxed);
            }

            removed
        };

        let removed_count = removed.len();
        drop(removed);
        removed_count
    }

    /// Clear the cancelled set for a specific transfer.
    ///
    /// Called when a transfer is fully complete to clean up the cancelled set.
    pub fn clear_cancelled(&self, transfer_id: TransferId) {
        self.cancelled.remove(&transfer_id);
    }

    /// Get the approximate queue length.
    ///
    /// This is not exact due to concurrent modifications but useful for monitoring.
    #[cfg(test)]
    pub(crate) fn len_approx(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Check if the queue is approximately empty.
    #[cfg(test)]
    pub(crate) fn is_empty_approx(&self) -> bool {
        self.len_approx() == 0
    }

    /// Get the number of cancelled transfers being tracked.
    #[cfg(test)]
    pub(crate) fn cancelled_count(&self) -> usize {
        self.cancelled.len()
    }
}

impl<T> Default for CancellableQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, mpsc};

    use tokio::sync::oneshot;

    use super::*;

    #[test]
    fn test_basic_push_pop() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        assert!(queue.push(id, 42));
        assert_eq!(queue.len_approx(), 1);

        let item = queue.pop().unwrap();
        assert_eq!(item.transfer_id, id);
        assert_eq!(item.data, 42);
        assert_eq!(queue.len_approx(), 0);
    }

    #[test]
    fn test_cancelled_push_rejected() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        queue.mark_cancelled(id);
        assert!(!queue.push(id, 42));
        assert_eq!(queue.len_approx(), 0);
    }

    #[tokio::test]
    async fn close_and_drain_wakes_the_consumer_and_rejects_late_work() {
        let queue = Arc::new(CancellableQueue::<i32>::new());
        let waiting_queue = Arc::clone(&queue);
        let waiter = tokio::spawn(async move {
            waiting_queue.notified().await;
            waiting_queue.is_closed()
        });
        tokio::task::yield_now().await;

        let drained = queue.close_and_drain();

        assert!(waiter.await.expect("queue waiter must finish"));
        assert!(drained.is_empty());
        assert!(!queue.push(TransferId::new(), 42));
        assert!(queue.is_empty_approx());
    }

    #[test]
    fn close_and_drain_returns_queued_payloads_and_rejects_late_work() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let first_id = TransferId::new();
        let second_id = TransferId::new();

        assert!(queue.push(first_id, 7));
        assert!(queue.push(second_id, 9));

        let mut drained = queue.close_and_drain();
        drained.sort_unstable();

        assert!(queue.is_closed());
        assert_eq!(drained, vec![7, 9]);
        assert!(queue.is_empty_approx());
        assert!(queue.pop().is_none());
        assert_eq!(queue.push_or_return(TransferId::new(), 11), Err(11));
    }

    #[test]
    fn close_and_drain_waits_for_sweep_and_returns_live_work() {
        enum Item {
            Live(u8),
            #[allow(dead_code)]
            Cancelled,
        }

        let queue = Arc::new(CancellableQueue::new());
        let live_id = TransferId::new();
        let cancelled_id = TransferId::new();
        let (drained_tx, drained_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();

        assert!(queue.push(live_id, Item::Live(7)));
        assert!(queue.push(cancelled_id, Item::Cancelled));
        queue.mark_cancelled(cancelled_id);
        queue.pause_next_sweep_before_requeue(move || {
            drained_tx.send(()).expect("test observes the sweep drain");
            release_rx.recv().expect("test releases the sweep");
        });

        let sweep_queue = Arc::clone(&queue);
        let sweep = std::thread::spawn(move || sweep_queue.sweep());
        drained_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("sweep removes live work before it waits");

        let close_queue = Arc::clone(&queue);
        let (close_started_tx, close_started_rx) = mpsc::channel();
        let (close_finished_tx, close_finished_rx) = mpsc::channel();
        let close = std::thread::spawn(move || {
            close_started_tx
                .send(())
                .expect("test starts close and drain");
            let drained = close_queue.close_and_drain();
            close_finished_tx
                .send(())
                .expect("test observes close and drain");
            drained
        });
        close_started_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("close and drain starts after the sweep drains live work");

        let close_completed_while_live_work_was_held = close_finished_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_ok();

        release_tx.send(()).expect("test releases the sweep");
        assert_eq!(sweep.join().expect("sweep thread exits"), 1);
        if !close_completed_while_live_work_was_held {
            close_finished_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .expect("close and drain finishes after the sweep requeues live work");
        }
        let drained = close.join().expect("close and drain thread exits");

        assert!(
            !close_completed_while_live_work_was_held,
            "close and drain must wait for the sweep to requeue live work"
        );
        assert_eq!(drained.len(), 1);
        assert!(matches!(drained.into_iter().next(), Some(Item::Live(7))));
        assert!(queue.is_empty_approx());
        assert!(queue.pop().is_none());
    }

    #[test]
    fn sweep_after_close_and_drain_leaves_no_queued_work() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let live_id = TransferId::new();
        let cancelled_id = TransferId::new();

        assert!(queue.push(live_id, 7));
        assert!(queue.push(cancelled_id, 9));
        queue.mark_cancelled(cancelled_id);
        let mut drained = queue.close_and_drain();
        drained.sort_unstable();

        assert_eq!(queue.sweep(), 0);
        assert_eq!(drained, vec![7, 9]);
        assert!(queue.is_empty_approx());
        assert!(queue.pop().is_none());
    }

    #[test]
    fn sweep_drops_cancelled_items_after_releasing_the_admission_gate() {
        struct DropProbe {
            queue: std::sync::Weak<CancellableQueue<DropProbe>>,
            gate_available: mpsc::Sender<bool>,
        }

        impl Drop for DropProbe {
            fn drop(&mut self) {
                let gate_available = self
                    .queue
                    .upgrade()
                    .and_then(|queue| queue.producer_gate.try_lock().map(|_| ()))
                    .is_some();
                self.gate_available
                    .send(gate_available)
                    .expect("test observes the cancelled item drop");
            }
        }

        let queue = Arc::new(CancellableQueue::new());
        let cancelled_id = TransferId::new();
        let (gate_available_tx, gate_available_rx) = mpsc::channel();

        assert!(queue.push(
            cancelled_id,
            DropProbe {
                queue: Arc::downgrade(&queue),
                gate_available: gate_available_tx,
            },
        ));
        queue.mark_cancelled(cancelled_id);

        assert_eq!(queue.sweep(), 1);
        assert!(
            gate_available_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .expect("cancelled item drops during sweep"),
            "a cancelled item destructor must not run under the admission gate"
        );
    }

    #[test]
    fn close_and_drain_returns_payloads_before_their_destructors_run() {
        struct DropProbe {
            queue: std::sync::Weak<CancellableQueue<DropProbe>>,
            gate_available: mpsc::Sender<bool>,
        }

        impl Drop for DropProbe {
            fn drop(&mut self) {
                let gate_available = self
                    .queue
                    .upgrade()
                    .and_then(|queue| queue.producer_gate.try_lock().map(|_| ()))
                    .is_some();
                self.gate_available
                    .send(gate_available)
                    .expect("test observes the drained item drop");
            }
        }

        let queue = Arc::new(CancellableQueue::new());
        let (gate_available_tx, gate_available_rx) = mpsc::channel();

        assert!(queue.push(
            TransferId::new(),
            DropProbe {
                queue: Arc::downgrade(&queue),
                gate_available: gate_available_tx,
            },
        ));

        let drained = queue.close_and_drain();
        assert!(
            gate_available_rx.try_recv().is_err(),
            "close and drain must return the payload before it drops"
        );
        drop(drained);
        assert!(
            gate_available_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .expect("drained item drops after close and drain"),
            "a drained item destructor must not run under the admission gate"
        );
    }

    #[test]
    fn test_pop_valid_skips_cancelled() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id1 = TransferId::new();
        let id2 = TransferId::new();

        queue.push(id1, 1);
        queue.push(id2, 2);
        queue.push(id1, 3);

        queue.mark_cancelled(id1);

        // pop_valid should skip items from id1
        let item = queue.pop_valid().unwrap();
        assert_eq!(item.transfer_id, id2);
        assert_eq!(item.data, 2);

        // No more valid items
        assert!(queue.pop_valid().is_none());
    }

    #[test]
    fn test_sweep_removes_cancelled() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id1 = TransferId::new();
        let id2 = TransferId::new();

        queue.push(id1, 1);
        queue.push(id2, 2);
        queue.push(id1, 3);
        queue.push(id2, 4);

        assert_eq!(queue.len_approx(), 4);

        queue.mark_cancelled(id1);
        let removed = queue.sweep();

        assert_eq!(removed, 2);
        assert_eq!(queue.len_approx(), 2);

        // Remaining items should be from id2
        let item1 = queue.pop().unwrap();
        let item2 = queue.pop().unwrap();
        assert_eq!(item1.transfer_id, id2);
        assert_eq!(item2.transfer_id, id2);
    }

    #[test]
    fn test_sweep_empty_cancelled_set() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        queue.push(id, 1);
        queue.push(id, 2);

        // Sweep with no cancelled transfers should be a no-op
        let removed = queue.sweep();
        assert_eq!(removed, 0);
        assert_eq!(queue.len_approx(), 2);
    }

    #[test]
    fn test_clear_cancelled() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        queue.mark_cancelled(id);
        assert!(queue.is_cancelled(id));
        assert_eq!(queue.cancelled_count(), 1);

        queue.clear_cancelled(id);
        assert!(!queue.is_cancelled(id));
        assert_eq!(queue.cancelled_count(), 0);
    }

    /// Test multiple transfer IDs with interleaved cancellation.
    #[test]
    fn test_multiple_transfers_interleaved() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id1 = TransferId::new();
        let id2 = TransferId::new();
        let id3 = TransferId::new();

        // Push items from different transfers
        queue.push(id1, 1);
        queue.push(id2, 2);
        queue.push(id1, 3);
        queue.push(id3, 4);
        queue.push(id2, 5);
        queue.push(id3, 6);

        assert_eq!(queue.len_approx(), 6);

        // Cancel id2
        queue.mark_cancelled(id2);
        let removed = queue.sweep();
        assert_eq!(removed, 2); // items 2 and 5
        assert_eq!(queue.len_approx(), 4);

        // Cancel id1
        queue.mark_cancelled(id1);
        let removed = queue.sweep();
        assert_eq!(removed, 2); // items 1 and 3
        assert_eq!(queue.len_approx(), 2);

        // Remaining should be from id3
        let item1 = queue.pop().unwrap();
        let item2 = queue.pop().unwrap();
        assert_eq!(item1.transfer_id, id3);
        assert_eq!(item2.transfer_id, id3);
    }

    /// Test sweep with empty queue.
    #[test]
    fn test_sweep_empty_queue() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        queue.mark_cancelled(id);
        let removed = queue.sweep();
        assert_eq!(removed, 0);
        assert!(queue.is_empty_approx());
    }

    /// Test pop_valid exhausts queue of only cancelled items.
    #[test]
    fn test_pop_valid_exhausts_cancelled() {
        let queue: CancellableQueue<i32> = CancellableQueue::new();
        let id = TransferId::new();

        queue.push(id, 1);
        queue.push(id, 2);
        queue.push(id, 3);

        queue.mark_cancelled(id);

        // pop_valid should return None after exhausting cancelled items
        assert!(queue.pop_valid().is_none());
        // Queue should be empty now (items were dropped during pop_valid)
        assert_eq!(queue.len_approx(), 0);
    }

    /// Test that cancelled items are dropped (not leaked) during sweep.
    #[test]
    fn test_sweep_drops_items() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct DropCounter {
            counter: Arc<AtomicUsize>,
        }

        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drop_count = Arc::new(AtomicUsize::new(0));
        let queue: CancellableQueue<DropCounter> = CancellableQueue::new();
        let id = TransferId::new();

        queue.push(
            id,
            DropCounter {
                counter: drop_count.clone(),
            },
        );
        queue.push(
            id,
            DropCounter {
                counter: drop_count.clone(),
            },
        );
        queue.push(
            id,
            DropCounter {
                counter: drop_count.clone(),
            },
        );

        assert_eq!(drop_count.load(Ordering::SeqCst), 0);

        queue.mark_cancelled(id);
        let removed = queue.sweep();

        assert_eq!(removed, 3);
        assert_eq!(drop_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn sweep_wakes_consumer_after_it_temporarily_observes_empty_queue() {
        enum Item {
            Kept(u8),
            #[allow(dead_code)]
            Cancelled,
        }

        let queue = Arc::new(CancellableQueue::new());
        let kept_id = TransferId::new();
        let cancelled_id = TransferId::new();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();

        assert!(queue.push(kept_id, Item::Kept(7)));
        assert!(queue.push(cancelled_id, Item::Cancelled));

        // `Notify` retains one producer permit. Consume it before this test
        // registers the consumer that must wake only after requeue.
        queue.notified().await;
        queue.mark_cancelled(cancelled_id);
        queue.pause_next_sweep_before_requeue(move || {
            entered_tx.send(()).expect("test observes the sweep drain");
            release_rx.recv().expect("test releases the sweep");
        });

        let sweep_queue = Arc::clone(&queue);
        let sweep = std::thread::spawn(move || sweep_queue.sweep());
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("sweep drains the cancelled item after it retains live work");

        let (observed_empty_tx, observed_empty_rx) = oneshot::channel();
        let (start_wait_tx, start_wait_rx) = oneshot::channel();
        let consumer_queue = Arc::clone(&queue);
        let consumer = tokio::spawn(async move {
            assert!(
                consumer_queue.pop_valid().is_none(),
                "consumer observes the sweep's temporary empty queue"
            );
            observed_empty_tx
                .send(())
                .expect("test observes the empty queue");
            start_wait_rx
                .await
                .expect("test starts the consumer wait after requeue");
            consumer_queue.notified().await;
            consumer_queue
                .pop_valid()
                .expect("requeue restores live work")
        });
        observed_empty_rx
            .await
            .expect("consumer observes the temporary empty queue");

        release_tx.send(()).expect("release requeue");
        assert_eq!(sweep.join().expect("sweep thread exits"), 1);
        start_wait_tx
            .send(())
            .expect("start consumer wait after durable requeue notification");

        let item = tokio::time::timeout(std::time::Duration::from_secs(1), consumer)
            .await
            .expect("requeue wakes the sleeping consumer")
            .expect("consumer task exits");
        assert_eq!(item.transfer_id, kept_id);
        assert!(matches!(item.data, Item::Kept(7)));
    }
}
