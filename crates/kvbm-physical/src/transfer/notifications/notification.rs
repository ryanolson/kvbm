// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer completion notification handle.

use anyhow::{Error, Result};
use futures::future::{Either, Ready, ready};
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use velo::{EventAwaiter, EventManager};

/// The drain certainty after a transfer completion receipt resolves.
///
/// A failed receipt can still prove that every launched physical operation
/// drained. Callers that own source memory must retain it only for
/// [`Self::Unproven`] outcomes.
#[must_use]
pub enum TransferDrainOutcome {
    /// Every physical operation completed successfully.
    Completed,
    /// Every launched physical operation drained, but dispatch or a nested
    /// receipt reported failure.
    DrainedWithError(Error),
    /// At least one physical completion did not prove that its work drained.
    Unproven(Error),
}

impl TransferDrainOutcome {
    fn into_result(self) -> Result<()> {
        match self {
            Self::Completed => Ok(()),
            Self::DrainedWithError(error) | Self::Unproven(error) => Err(error),
        }
    }
}

pub enum TransferAwaiter {
    Local(EventAwaiter),
    Aggregate(Pin<Box<dyn Future<Output = TransferDrainOutcome> + Send>>),
    // Sync(SyncResult),
}

impl std::future::Future for TransferAwaiter {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            Self::Local(waiter) => Pin::new(waiter).poll(cx),
            Self::Aggregate(waiter) => waiter
                .as_mut()
                .poll(cx)
                .map(TransferDrainOutcome::into_result),
            // Self::Sync(sync) => Pin::new(sync).poll(cx),
        }
    }
}

/// Notification handle for an in-progress transfer.
///
/// This object can be awaited to block until the transfer completes.
/// The transfer is tracked by a background handler that polls for completion
/// or processes notification events.
///
/// Uses `futures::Either` to avoid event system overhead for synchronous completions.
/// One pending transfer uses `EventAwaiter` without extra aggregate event state.
/// An aggregate receipt uses an owned future and does not start a background task.
pub struct TransferCompleteNotification {
    awaiter: Either<Ready<Result<()>>, TransferAwaiter>,
}

impl TransferCompleteNotification {
    /// Create a notification that is already completed (for synchronous transfers).
    ///
    /// This is useful for transfers that complete immediately without needing
    /// background polling, such as memcpy operations.
    ///
    /// This is extremely efficient - no allocations, locks, or event system overhead.
    pub fn completed() -> Self {
        Self {
            awaiter: Either::Left(ready(Ok(()))),
        }
    }

    /// Create a notification from a `LocalEventWaiter`.
    ///
    /// This is the primary way to construct a notification when you already
    /// have an event waiter from the event system.
    pub fn from_awaiter(awaiter: EventAwaiter) -> Self {
        Self {
            awaiter: Either::Right(TransferAwaiter::Local(awaiter)),
        }
    }

    // /// Create a notification from a synchronous active message result.
    // pub fn from_sync_result(sync: SyncResult) -> Self {
    //     Self {
    //         awaiter: Either::Right(TransferAwaiter::Sync(sync)),
    //     }
    // }

    /// Check if the notification can yield the current task.
    ///
    /// The internal `Left` arm is ready. The `Right` arm can require a wakeup.
    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, Either::Right(_))
    }

    /// Await the receipt and preserve whether physical drain was proven.
    ///
    /// Ordinary `.await` preserves the legacy `Result<()>` contract. Use this
    /// method when source ownership depends on the distinction between a
    /// drained failure and an ambiguous completion failure.
    pub async fn await_drain(self) -> TransferDrainOutcome {
        match self.awaiter {
            Either::Left(awaiter) => match awaiter.await {
                Ok(()) => TransferDrainOutcome::Completed,
                Err(error) => TransferDrainOutcome::Unproven(error),
            },
            Either::Right(TransferAwaiter::Local(awaiter)) => match awaiter.await {
                Ok(()) => TransferDrainOutcome::Completed,
                Err(error) => TransferDrainOutcome::Unproven(error),
            },
            Either::Right(TransferAwaiter::Aggregate(awaiter)) => awaiter.await,
        }
    }

    /// Aggregate multiple notifications into one that completes when all are done.
    ///
    /// This is useful when a transfer is split across multiple workers and you want
    /// to wait for all of them to complete.
    ///
    /// # Arguments
    /// * `notifications` - The notifications to aggregate
    /// * `events` - The event system retained for API compatibility
    /// * `runtime` - The runtime handle retained for API compatibility
    ///
    /// # Behavior
    /// - If the list is empty, returns an already-completed notification
    /// - If there's only one, returns it directly
    /// - Otherwise, returns a notification that directly owns all child notifications
    pub fn aggregate(
        notifications: Vec<Self>,
        events: &Arc<EventManager>,
        runtime: &tokio::runtime::Handle,
    ) -> Result<Self> {
        Self::aggregate_results(notifications.into_iter().map(Ok).collect(), events, runtime)
    }

    /// Aggregate dispatch results without abandoning transfers that launched
    /// before a later synchronous dispatch error.
    ///
    /// The returned receipt owns every successful notification. It polls all
    /// notifications to completion before it returns a combined failure.
    pub fn aggregate_results(
        results: Vec<Result<Self>>,
        _events: &Arc<EventManager>,
        _runtime: &tokio::runtime::Handle,
    ) -> Result<Self> {
        let mut notifications = Vec::with_capacity(results.len());
        let mut dispatch_errors = Vec::new();
        for result in results {
            match result {
                Ok(notification) => notifications.push(notification),
                Err(error) => dispatch_errors.push(error),
            }
        }
        if notifications.is_empty() {
            return errors_or_completed(dispatch_errors);
        }
        if notifications.len() == 1 && dispatch_errors.is_empty() {
            return Ok(notifications.into_iter().next().unwrap());
        }

        // Preserve the allocation-free success path for completed notifications.
        // Dispatch errors require a receipt, even when all notifications are ready.
        if dispatch_errors.is_empty() && notifications.iter().all(|n| !n.could_yield()) {
            return Ok(Self::completed());
        }

        Ok(Self {
            awaiter: Either::Right(TransferAwaiter::Aggregate(Box::pin(
                await_all_notifications(notifications, dispatch_errors),
            ))),
        })
    }
}

/// Awaits all transfer notifications and returns their combined result.
///
/// This function awaits ALL notifications regardless of individual failures,
/// then combines synchronous dispatch and asynchronous completion errors.
async fn await_all_notifications(
    notifications: Vec<TransferCompleteNotification>,
    mut errors: Vec<Error>,
) -> TransferDrainOutcome {
    let outcomes = futures::future::join_all(
        notifications
            .into_iter()
            .map(TransferCompleteNotification::await_drain),
    )
    .await;
    let mut unproven = false;

    for outcome in outcomes {
        match outcome {
            TransferDrainOutcome::Completed => {}
            TransferDrainOutcome::DrainedWithError(error) => errors.push(error),
            TransferDrainOutcome::Unproven(error) => {
                unproven = true;
                errors.push(error);
            }
        }
    }

    if errors.is_empty() {
        TransferDrainOutcome::Completed
    } else if unproven {
        TransferDrainOutcome::Unproven(anyhow::anyhow!(combined_error_message(&errors)))
    } else {
        TransferDrainOutcome::DrainedWithError(anyhow::anyhow!(combined_error_message(&errors)))
    }
}

fn errors_or_completed(errors: Vec<anyhow::Error>) -> Result<TransferCompleteNotification> {
    if errors.is_empty() {
        Ok(TransferCompleteNotification::completed())
    } else {
        Err(anyhow::anyhow!(combined_error_message(&errors)))
    }
}

fn combined_error_message(errors: &[anyhow::Error]) -> String {
    errors
        .iter()
        .map(|error| format!("{error:#}"))
        .collect::<Vec<_>>()
        .join("; ")
}

impl std::future::IntoFuture for TransferCompleteNotification {
    type Output = Result<()>;
    type IntoFuture = Either<Ready<Result<()>>, TransferAwaiter>;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::{Result, anyhow};
    use velo::EventManager;

    use super::{TransferCompleteNotification, TransferDrainOutcome};

    #[tokio::test]
    async fn dispatch_error_after_child_drain_preserves_drain_proof() -> Result<()> {
        let events = Arc::new(EventManager::local());
        let aggregate = TransferCompleteNotification::aggregate_results(
            vec![
                Ok(TransferCompleteNotification::completed()),
                Err(anyhow!("later synchronous dispatch failed")),
            ],
            &events,
            &tokio::runtime::Handle::current(),
        )?;

        match aggregate.await_drain().await {
            TransferDrainOutcome::DrainedWithError(error) => {
                assert!(
                    error
                        .to_string()
                        .contains("later synchronous dispatch failed")
                );
            }
            TransferDrainOutcome::Completed => panic!("dispatch failure must reach the receipt"),
            TransferDrainOutcome::Unproven(error) => {
                panic!("all child receipts drained, not unproven: {error:#}")
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn launched_ready_notification_defers_dispatch_error_to_receipt() -> Result<()> {
        let events = Arc::new(EventManager::local());

        let aggregate = TransferCompleteNotification::aggregate_results(
            vec![
                Ok(TransferCompleteNotification::completed()),
                Err(anyhow!("later synchronous dispatch failed")),
            ],
            &events,
            &tokio::runtime::Handle::current(),
        )
        .expect("a launched notification must always produce a completion receipt");

        let failure = aggregate
            .await
            .expect_err("the receipt must report the later dispatch error");
        assert!(
            failure
                .to_string()
                .contains("later synchronous dispatch failed")
        );
        Ok(())
    }

    #[tokio::test]
    async fn unavailable_aggregate_events_cannot_abandon_launched_notification() -> Result<()> {
        let notification_events = Arc::new(EventManager::local());
        let delayed_event = notification_events.new_event()?;
        let delayed = TransferCompleteNotification::from_awaiter(
            notification_events.awaiter(delayed_event.handle())?,
        );
        let unavailable_aggregate_events = Arc::new(EventManager::local());
        unavailable_aggregate_events.force_shutdown("aggregate events unavailable");

        let aggregate = TransferCompleteNotification::aggregate_results(
            vec![
                Ok(delayed),
                Err(anyhow!("later synchronous dispatch failed")),
            ],
            &unavailable_aggregate_events,
            &tokio::runtime::Handle::current(),
        )
        .expect("event setup cannot fail after a worker returns a receipt");
        let mut completion = tokio::spawn(aggregate.into_future());

        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut completion)
                .await
                .is_err(),
            "a synchronous error must not abandon an already-launched transfer"
        );
        delayed_event.trigger()?;
        let failure = tokio::time::timeout(Duration::from_secs(1), completion)
            .await??
            .expect_err("the deferred synchronous dispatch failure must poison completion");
        assert!(
            failure
                .to_string()
                .contains("later synchronous dispatch failed")
        );
        Ok(())
    }

    #[tokio::test]
    async fn dispatch_and_completion_errors_are_combined_after_drain() -> Result<()> {
        let events = Arc::new(EventManager::local());
        let delayed_event = events.new_event()?;
        let delayed =
            TransferCompleteNotification::from_awaiter(events.awaiter(delayed_event.handle())?);

        let aggregate = TransferCompleteNotification::aggregate_results(
            vec![Ok(delayed), Err(anyhow!("synchronous dispatch failure"))],
            &events,
            &tokio::runtime::Handle::current(),
        )?;
        delayed_event.poison("asynchronous completion failure")?;
        let failure = aggregate
            .await
            .expect_err("both terminal failures must poison aggregate completion");
        let message = failure.to_string();
        assert!(message.contains("synchronous dispatch failure"));
        assert!(message.contains("asynchronous completion failure"));
        Ok(())
    }

    #[tokio::test]
    async fn completion_error_keeps_aggregate_drain_unproven() -> Result<()> {
        let events = Arc::new(EventManager::local());
        let delayed_event = events.new_event()?;
        let delayed =
            TransferCompleteNotification::from_awaiter(events.awaiter(delayed_event.handle())?);

        let aggregate = TransferCompleteNotification::aggregate_results(
            vec![Ok(delayed), Err(anyhow!("synchronous dispatch failure"))],
            &events,
            &tokio::runtime::Handle::current(),
        )?;
        delayed_event.poison("asynchronous completion failure")?;

        match aggregate.await_drain().await {
            TransferDrainOutcome::Unproven(error) => {
                let message = error.to_string();
                assert!(message.contains("synchronous dispatch failure"));
                assert!(message.contains("asynchronous completion failure"));
            }
            TransferDrainOutcome::Completed => panic!("completion failure must reach the receipt"),
            TransferDrainOutcome::DrainedWithError(error) => {
                panic!("completion failure removes drain proof: {error:#}")
            }
        }
        Ok(())
    }
}
