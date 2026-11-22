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

/// One failed part of an aggregate receipt, with its drain certainty.
///
/// A caller that meets a variant it does not know must treat the part as
/// unproven.
#[derive(Debug)]
#[non_exhaustive]
pub enum TransferPartFailure {
    /// The part launched nothing, or every operation it launched drained.
    DrainedWithError(Error),
    /// The part did not prove that its work drained.
    Unproven(Error),
}

impl TransferPartFailure {
    /// The error of this part.
    pub fn error(&self) -> &Error {
        match self {
            Self::DrainedWithError(error) | Self::Unproven(error) => error,
        }
    }

    /// Take the error of this part.
    pub fn into_error(self) -> Error {
        match self {
            Self::DrainedWithError(error) | Self::Unproven(error) => error,
        }
    }

    /// Whether this part has no drain proof.
    pub fn is_unproven(&self) -> bool {
        matches!(self, Self::Unproven(_))
    }
}

/// The error of an aggregate receipt that failed: every failed part, in
/// order, each with its own typed error.
///
/// The aggregate keeps each part's [`anyhow::Error`] whole, so a caller can
/// still downcast a part's context, for example the re-check capability of a
/// worker whose reply was lost. Find it with
/// `error.downcast_ref::<AggregateTransferFailure>()`. Its text is the
/// optional label, then each part's error (`{:#}`) joined with `"; "`.
#[derive(Debug)]
pub struct AggregateTransferFailure {
    label: Option<String>,
    parts: Vec<TransferPartFailure>,
}

impl AggregateTransferFailure {
    /// An aggregate failure with an optional label before the part errors.
    pub fn new(label: Option<String>, parts: Vec<TransferPartFailure>) -> Self {
        Self { label, parts }
    }

    /// The failed parts, in order.
    pub fn parts(&self) -> &[TransferPartFailure] {
        &self.parts
    }

    /// Take the failed parts, in order.
    pub fn into_parts(self) -> Vec<TransferPartFailure> {
        self.parts
    }

    /// Whether any part has no drain proof.
    pub fn is_unproven(&self) -> bool {
        self.parts.iter().any(TransferPartFailure::is_unproven)
    }

    /// The drain outcome of an aggregate whose failed parts are `parts`:
    /// `Completed` when there are none, `Unproven` when any part is
    /// unproven, and `DrainedWithError` otherwise.
    pub fn outcome(label: Option<String>, parts: Vec<TransferPartFailure>) -> TransferDrainOutcome {
        if parts.is_empty() {
            return TransferDrainOutcome::Completed;
        }
        let failure = Self::new(label, parts);
        if failure.is_unproven() {
            TransferDrainOutcome::Unproven(Error::new(failure))
        } else {
            TransferDrainOutcome::DrainedWithError(Error::new(failure))
        }
    }
}

impl std::fmt::Display for AggregateTransferFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(label) = &self.label {
            write!(f, "{label}: ")?;
        }
        for (index, part) in self.parts.iter().enumerate() {
            if index > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{:#}", part.error())?;
        }
        Ok(())
    }
}

impl std::error::Error for AggregateTransferFailure {}

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

    /// Create a notification from a future that reports its own drain outcome.
    ///
    /// Use this constructor when the source of the receipt knows whether the
    /// physical work drained, for example a worker RPC reply. The returned
    /// receipt gives that outcome from [`Self::await_drain`] without change.
    pub fn from_drain_future<F>(outcome: F) -> Self
    where
        F: Future<Output = TransferDrainOutcome> + Send + 'static,
    {
        Self {
            awaiter: Either::Right(TransferAwaiter::Aggregate(Box::pin(outcome))),
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
    dispatch_errors: Vec<Error>,
) -> TransferDrainOutcome {
    let outcomes = futures::future::join_all(
        notifications
            .into_iter()
            .map(TransferCompleteNotification::await_drain),
    )
    .await;
    // A synchronous dispatch error launched nothing, so it is drained.
    let mut parts: Vec<_> = dispatch_errors
        .into_iter()
        .map(TransferPartFailure::DrainedWithError)
        .collect();
    for outcome in outcomes {
        match outcome {
            TransferDrainOutcome::Completed => {}
            TransferDrainOutcome::DrainedWithError(error) => {
                parts.push(TransferPartFailure::DrainedWithError(error));
            }
            TransferDrainOutcome::Unproven(error) => {
                parts.push(TransferPartFailure::Unproven(error));
            }
        }
    }
    AggregateTransferFailure::outcome(None, parts)
}

fn errors_or_completed(errors: Vec<anyhow::Error>) -> Result<TransferCompleteNotification> {
    if errors.is_empty() {
        Ok(TransferCompleteNotification::completed())
    } else {
        Err(Error::new(AggregateTransferFailure::new(
            None,
            errors
                .into_iter()
                .map(TransferPartFailure::DrainedWithError)
                .collect(),
        )))
    }
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

    use super::{
        AggregateTransferFailure, TransferCompleteNotification, TransferDrainOutcome,
        TransferPartFailure,
    };

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
    async fn drain_future_preserves_its_outcome() {
        let drained = TransferCompleteNotification::from_drain_future(async {
            TransferDrainOutcome::DrainedWithError(anyhow!("clean failure"))
        });
        assert!(drained.could_yield());
        assert!(matches!(
            drained.await_drain().await,
            TransferDrainOutcome::DrainedWithError(_)
        ));

        let unproven = TransferCompleteNotification::from_drain_future(async {
            TransferDrainOutcome::Unproven(anyhow!("lost reply"))
        });
        assert!(matches!(
            unproven.await_drain().await,
            TransferDrainOutcome::Unproven(_)
        ));

        let failed = TransferCompleteNotification::from_drain_future(async {
            TransferDrainOutcome::Unproven(anyhow!("lost reply"))
        });
        let error = failed.await.expect_err("an unproven outcome is a failure");
        assert!(error.to_string().contains("lost reply"));
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

    /// A typed context a caller attaches to one part's error.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Worker(u32);

    impl std::fmt::Display for Worker {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "worker {}", self.0)
        }
    }

    fn unproven_on(worker: u32) -> TransferCompleteNotification {
        TransferCompleteNotification::from_drain_future(async move {
            TransferDrainOutcome::Unproven(anyhow!("lost reply").context(Worker(worker)))
        })
    }

    fn workers_of(failure: &AggregateTransferFailure) -> Vec<(bool, Option<Worker>)> {
        failure
            .parts()
            .iter()
            .map(|part| {
                (
                    part.is_unproven(),
                    part.error().downcast_ref::<Worker>().copied(),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn an_unproven_aggregate_keeps_every_part_error_typed() -> Result<()> {
        let events = Arc::new(EventManager::local());
        let aggregate = TransferCompleteNotification::aggregate_results(
            vec![
                Ok(unproven_on(1)),
                Err(anyhow!("dispatch failed")),
                Ok(unproven_on(2)),
            ],
            &events,
            &tokio::runtime::Handle::current(),
        )?;

        let TransferDrainOutcome::Unproven(error) = aggregate.await_drain().await else {
            panic!("an unproven part keeps the aggregate unproven");
        };
        let failure = error
            .downcast_ref::<AggregateTransferFailure>()
            .expect("the aggregate error keeps its parts");
        assert_eq!(
            workers_of(failure),
            vec![
                (false, None),
                (true, Some(Worker(1))),
                (true, Some(Worker(2))),
            ]
        );
        assert_eq!(
            error.to_string(),
            "dispatch failed; worker 1: lost reply; worker 2: lost reply"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_completed_part_leaves_only_the_unproven_part() -> Result<()> {
        let events = Arc::new(EventManager::local());
        let delayed_event = events.new_event()?;
        let delayed =
            TransferCompleteNotification::from_awaiter(events.awaiter(delayed_event.handle())?);
        let aggregate = TransferCompleteNotification::aggregate_results(
            vec![Ok(delayed), Ok(unproven_on(2))],
            &events,
            &tokio::runtime::Handle::current(),
        )?;
        delayed_event.trigger()?;

        let TransferDrainOutcome::Unproven(error) = aggregate.await_drain().await else {
            panic!("an unproven part keeps the aggregate unproven");
        };
        let failure = error
            .downcast_ref::<AggregateTransferFailure>()
            .expect("the aggregate error keeps its parts");
        assert_eq!(workers_of(failure), vec![(true, Some(Worker(2)))]);
        Ok(())
    }

    #[test]
    fn aggregate_failure_text_is_the_label_and_the_joined_parts() {
        let failure = AggregateTransferFailure::new(
            Some("group failed".to_owned()),
            vec![
                TransferPartFailure::DrainedWithError(anyhow!("inner").context("outer")),
                TransferPartFailure::Unproven(anyhow!("lost")),
            ],
        );
        assert_eq!(failure.to_string(), "group failed: outer: inner; lost");
        assert!(failure.is_unproven());
    }
}
