// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use futures::stream::{FuturesUnordered, StreamExt};
use kvbm_physical::transfer::{
    AggregateTransferFailure, TransferCompleteNotification, TransferDrainOutcome,
    TransferPartFailure,
};

/// Serializes replicated collective onboards and permanently fails the rank
/// group after its first divergent completion.
pub(super) struct ReplicatedOnboardCoordinator {
    sequence: Arc<tokio::sync::Mutex<()>>,
    failure: Arc<OnceLock<String>>,
    runtime: tokio::runtime::Handle,
}

impl ReplicatedOnboardCoordinator {
    pub(super) fn new(runtime: tokio::runtime::Handle) -> Self {
        Self {
            sequence: Arc::new(tokio::sync::Mutex::new(())),
            failure: Arc::new(OnceLock::new()),
            runtime,
        }
    }

    pub(super) fn poison(&self, reason: String) {
        let _ = self.failure.set(reason);
    }

    pub(super) fn execute<Dispatch, Abort>(
        &self,
        dispatch: Dispatch,
        abort: Abort,
    ) -> Result<TransferCompleteNotification>
    where
        Dispatch: FnOnce() -> Vec<Result<TransferCompleteNotification>> + Send + 'static,
        Abort: FnOnce(String) + Send + 'static,
    {
        self.execute_with_wait_hook(dispatch, abort, None)
    }

    /// The returned receipt resolves only after every rank receipt resolved.
    /// A group failure thus proves that no accepted rank copy is active,
    /// unless the outcome is [`TransferDrainOutcome::Unproven`].
    fn execute_with_wait_hook<Dispatch, Abort>(
        &self,
        dispatch: Dispatch,
        abort: Abort,
        wait_hook: Option<Box<dyn FnOnce() + Send>>,
    ) -> Result<TransferCompleteNotification>
    where
        Dispatch: FnOnce() -> Vec<Result<TransferCompleteNotification>> + Send + 'static,
        Abort: FnOnce(String) + Send + 'static,
    {
        if let Some(reason) = self.failure.get() {
            anyhow::bail!("replicated onboard group is aborted: {reason}");
        }
        let sequence = Arc::clone(&self.sequence);
        let failure = Arc::clone(&self.failure);

        let task = self.runtime.spawn(async move {
            let mut sequence_lock = Box::pin(sequence.lock());
            let sequence_guard = match futures::poll!(&mut sequence_lock) {
                std::task::Poll::Ready(guard) => guard,
                std::task::Poll::Pending => {
                    if let Some(wait_hook) = wait_hook {
                        wait_hook();
                    }
                    sequence_lock.await
                }
            };
            if let Some(reason) = failure.get() {
                // No rank received this operation.
                return TransferDrainOutcome::DrainedWithError(anyhow::anyhow!(
                    "replicated onboard group is aborted: {reason}"
                ));
            }

            let mut sequence_guard = Some(sequence_guard);
            await_rank_notifications(dispatch(), |error| {
                let candidate = format!("replicated onboard rank group failed: {error:#}");
                let first_failure = failure.set(candidate.clone()).is_ok();
                let reason = failure.get().cloned().unwrap_or(candidate);
                // The failure is permanent, and each later operation fails
                // before its dispatch. The sequence lock has no more function.
                drop(sequence_guard.take());
                if first_failure {
                    // The abort releases ranks that wait in their collective.
                    abort(reason);
                }
            })
            .await
        });

        Ok(TransferCompleteNotification::from_drain_future(
            async move {
                match task.await {
                    Ok(outcome) => outcome,
                    Err(error) => TransferDrainOutcome::Unproven(anyhow::anyhow!(
                        "replicated onboard task stopped before it reported an outcome: {error}"
                    )),
                }
            },
        ))
    }
}

/// Poll every rank concurrently until every rank receipt resolves.
///
/// The first failure calls `on_first_failure` immediately. Serial waiting can
/// hide a failed RPC behind an earlier rank blocked in its collective. The
/// function does not return at the first failure, because the other ranks can
/// have an active copy. A synchronous dispatch error launched no copy.
async fn await_rank_notifications<OnFailure>(
    notifications: Vec<Result<TransferCompleteNotification>>,
    on_first_failure: OnFailure,
) -> TransferDrainOutcome
where
    OnFailure: FnOnce(&anyhow::Error),
{
    let mut on_first_failure = Some(on_first_failure);
    // A synchronous dispatch error launched nothing, so it is drained.
    let mut parts = Vec::new();
    let mut pending = FuturesUnordered::new();
    for notification in notifications {
        match notification {
            Ok(notification) => pending.push(notification.await_drain()),
            Err(error) => parts.push(TransferPartFailure::DrainedWithError(error)),
        }
    }
    if let Some(part) = parts.first()
        && let Some(on_first_failure) = on_first_failure.take()
    {
        on_first_failure(part.error());
    }
    while let Some(outcome) = pending.next().await {
        let part = match outcome {
            TransferDrainOutcome::Completed => continue,
            TransferDrainOutcome::DrainedWithError(error) => {
                TransferPartFailure::DrainedWithError(error)
            }
            TransferDrainOutcome::Unproven(error) => TransferPartFailure::Unproven(error),
        };
        if let Some(on_first_failure) = on_first_failure.take() {
            on_first_failure(part.error());
        }
        parts.push(part);
    }

    AggregateTransferFailure::outcome(
        Some("replicated onboard rank group failed".to_owned()),
        parts,
    )
}

pub(super) fn dispatch_collective_aborts<T, Abort>(
    workers: Vec<T>,
    reason: String,
    runtime: tokio::runtime::Handle,
    abort: Arc<Abort>,
) where
    T: Send + 'static,
    Abort: Fn(T, String) -> Result<TransferCompleteNotification> + Send + Sync + 'static,
{
    for (rank, worker) in workers.into_iter().enumerate() {
        let reason = reason.clone();
        let runtime = runtime.clone();
        let abort = Arc::clone(&abort);
        let spawn = std::thread::Builder::new()
            .name(format!("kvbm-collective-abort-rank-{rank}"))
            .spawn(move || match abort(worker, reason) {
                Ok(notification) => {
                    std::mem::drop(runtime.spawn(async move {
                        if let Err(error) = notification.await {
                            tracing::warn!(
                                rank,
                                error = %error,
                                "rank-local collective abort did not complete"
                            );
                        }
                    }));
                }
                Err(error) => {
                    tracing::warn!(
                        rank,
                        error = %error,
                        "failed to dispatch rank-local collective abort"
                    );
                }
            });
        if let Err(error) = spawn {
            tracing::warn!(
                rank,
                error = %error,
                "failed to spawn rank-local collective abort thread"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Barrier, mpsc};
    use std::time::Duration;

    #[tokio::test]
    async fn dispatches_one_logical_operation_at_a_time() {
        let events = Arc::new(::velo::EventManager::local());
        let coordinator = ReplicatedOnboardCoordinator::new(tokio::runtime::Handle::current());
        let first_event = events.new_event().unwrap();
        let first_notification = TransferCompleteNotification::from_awaiter(
            events.awaiter(first_event.handle()).unwrap(),
        );
        let first_started = Arc::new(AtomicBool::new(false));
        let second_started = Arc::new(AtomicBool::new(false));

        let first = coordinator
            .execute(
                {
                    let first_started = Arc::clone(&first_started);
                    move || {
                        first_started.store(true, Ordering::SeqCst);
                        vec![Ok(first_notification)]
                    }
                },
                |_| {},
            )
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while !first_started.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let (second_waiting, second_waiting_rx) = tokio::sync::oneshot::channel();
        let second = coordinator
            .execute_with_wait_hook(
                {
                    let second_started = Arc::clone(&second_started);
                    move || {
                        second_started.store(true, Ordering::SeqCst);
                        vec![Ok(TransferCompleteNotification::completed())]
                    }
                },
                |_| {},
                Some(Box::new(move || {
                    let _ = second_waiting.send(());
                })),
            )
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), second_waiting_rx)
            .await
            .expect("the second onboard did not poll the held sequence lock")
            .expect("the second onboard dropped its wait signal");
        assert!(!second_started.load(Ordering::SeqCst));

        first_event.trigger().unwrap();
        first.await.unwrap();
        second.await.unwrap();
        assert!(second_started.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn failure_releases_sequence_and_aborts_every_rank_concurrently() {
        let events = Arc::new(::velo::EventManager::local());
        let coordinator = ReplicatedOnboardCoordinator::new(tokio::runtime::Handle::current());
        let pending_event = events.new_event().unwrap();
        let pending_notification = TransferCompleteNotification::from_awaiter(
            events.awaiter(pending_event.handle()).unwrap(),
        );
        let failed_event = events.new_event().unwrap();
        let failed_notification = TransferCompleteNotification::from_awaiter(
            events.awaiter(failed_event.handle()).unwrap(),
        );
        failed_event.poison("injected rank RPC failure").unwrap();
        let abort_ranks = (0..4)
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect::<Vec<_>>();
        let observed_ranks = abort_ranks.clone();
        let cleanup_barrier = Arc::new(Barrier::new(abort_ranks.len() + 1));
        let release_cleanup = Arc::clone(&cleanup_barrier);
        let (abort_entered, abort_entered_rx) = mpsc::channel();
        let runtime = tokio::runtime::Handle::current();

        let completion = coordinator
            .execute(
                move || vec![Ok(pending_notification), Ok(failed_notification)],
                move |reason| {
                    assert!(reason.contains("injected rank RPC failure"));
                    dispatch_collective_aborts(
                        abort_ranks,
                        reason,
                        runtime,
                        Arc::new(move |calls: Arc<AtomicUsize>, _reason| {
                            calls.fetch_add(1, Ordering::SeqCst);
                            abort_entered.send(()).unwrap();
                            cleanup_barrier.wait();
                            Ok(TransferCompleteNotification::completed())
                        }),
                    );
                },
            )
            .unwrap();
        // The failed rank starts the abort, but the group receipt waits for
        // the rank that is in its collective.
        let mut completion = Box::pin(std::future::IntoFuture::into_future(completion));
        assert!(
            tokio::time::timeout(Duration::from_millis(100), completion.as_mut())
                .await
                .is_err(),
            "the group receipt must wait for a peer that can have an active copy"
        );
        for _ in 0..observed_ranks.len() {
            abort_entered_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("every rank must enter abort concurrently");
        }
        assert!(
            observed_ranks
                .iter()
                .all(|calls| calls.load(Ordering::SeqCst) == 1)
        );

        let sequence_guard =
            tokio::time::timeout(Duration::from_secs(1), coordinator.sequence.lock())
                .await
                .expect("the sequence lock must be released after fatal group poisoning");
        drop(sequence_guard);
        release_cleanup.wait();
        let later = coordinator.execute(
            || vec![Ok(TransferCompleteNotification::completed())],
            |_| {},
        );
        assert!(
            later
                .err()
                .is_some_and(|error| error.to_string().contains("aborted"))
        );

        pending_event.trigger().unwrap();
        let error = tokio::time::timeout(Duration::from_secs(1), completion)
            .await
            .expect("the group receipt must resolve after the last rank resolves")
            .expect_err("the logical onboard must surface the rank failure");
        assert!(error.to_string().contains("injected rank RPC failure"));
    }

    #[tokio::test]
    async fn failed_rank_keeps_group_receipt_pending_until_pending_rank_resolves() {
        let events = Arc::new(::velo::EventManager::local());
        let coordinator = ReplicatedOnboardCoordinator::new(tokio::runtime::Handle::current());
        let pending_event = events.new_event().unwrap();
        let pending_notification = TransferCompleteNotification::from_awaiter(
            events.awaiter(pending_event.handle()).unwrap(),
        );
        let aborts = Arc::new(AtomicUsize::new(0));
        let observed_aborts = Arc::clone(&aborts);

        let completion = coordinator
            .execute(
                move || {
                    vec![
                        Err(anyhow::anyhow!("injected rank dispatch failure")),
                        Ok(pending_notification),
                    ]
                },
                move |_| {
                    aborts.fetch_add(1, Ordering::SeqCst);
                },
            )
            .unwrap();
        let mut drain = tokio::spawn(completion.await_drain());

        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut drain)
                .await
                .is_err(),
            "the group receipt must stay pending while a rank receipt is pending"
        );
        assert_eq!(
            observed_aborts.load(Ordering::SeqCst),
            1,
            "the first failure must start the abort before the last rank resolves"
        );

        pending_event.trigger().unwrap();
        let outcome = tokio::time::timeout(Duration::from_secs(1), drain)
            .await
            .expect("the group receipt must resolve after the last rank resolves")
            .unwrap();
        match outcome {
            TransferDrainOutcome::DrainedWithError(error) => {
                assert!(error.to_string().contains("injected rank dispatch failure"));
            }
            TransferDrainOutcome::Completed => panic!("the rank failure must reach the receipt"),
            TransferDrainOutcome::Unproven(error) => {
                panic!("every rank receipt resolved with drain proof: {error:#}")
            }
        }
    }

    #[tokio::test]
    async fn unproven_rank_outcome_makes_group_outcome_unproven() {
        let coordinator = ReplicatedOnboardCoordinator::new(tokio::runtime::Handle::current());
        let completion = coordinator
            .execute(
                || {
                    vec![
                        Ok(TransferCompleteNotification::completed()),
                        Ok(TransferCompleteNotification::from_drain_future(async {
                            TransferDrainOutcome::Unproven(anyhow::anyhow!(
                                "injected lost RPC reply"
                            ))
                        })),
                    ]
                },
                |_| {},
            )
            .unwrap();

        match completion.await_drain().await {
            TransferDrainOutcome::Unproven(error) => {
                assert!(error.to_string().contains("injected lost RPC reply"));
            }
            TransferDrainOutcome::Completed => panic!("the rank failure must reach the receipt"),
            TransferDrainOutcome::DrainedWithError(error) => {
                panic!("a lost reply does not prove drain: {error:#}")
            }
        }
    }

    /// Two ranks end unproven, each with its own typed context. The group
    /// receipt keeps both, and its text keeps the group label.
    #[tokio::test]
    async fn an_unproven_group_keeps_every_rank_error_typed() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct Rank(u32);
        impl std::fmt::Display for Rank {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "rank {}", self.0)
            }
        }
        let unproven = |rank| {
            TransferCompleteNotification::from_drain_future(async move {
                TransferDrainOutcome::Unproven(anyhow::anyhow!("lost reply").context(Rank(rank)))
            })
        };
        let coordinator = ReplicatedOnboardCoordinator::new(tokio::runtime::Handle::current());
        let completion = coordinator
            .execute(
                move || {
                    vec![
                        Ok(unproven(0)),
                        Ok(TransferCompleteNotification::completed()),
                        Ok(unproven(2)),
                    ]
                },
                |_| {},
            )
            .unwrap();

        let outcome = tokio::time::timeout(Duration::from_secs(5), completion.await_drain())
            .await
            .expect("the group receipt resolves");
        let TransferDrainOutcome::Unproven(error) = outcome else {
            panic!("an unproven rank keeps the group unproven");
        };
        let failure = error
            .downcast_ref::<kvbm_physical::transfer::AggregateTransferFailure>()
            .expect("the group error keeps its rank errors");
        let mut ranks: Vec<_> = failure
            .parts()
            .iter()
            .map(|part| {
                assert!(part.is_unproven());
                part.error().downcast_ref::<Rank>().copied()
            })
            .collect();
        ranks.sort_by_key(|rank| rank.map(|Rank(rank)| rank));
        assert_eq!(ranks, vec![Some(Rank(0)), Some(Rank(2))]);
        assert!(
            error
                .to_string()
                .starts_with("replicated onboard rank group failed: rank ")
        );
    }
}
