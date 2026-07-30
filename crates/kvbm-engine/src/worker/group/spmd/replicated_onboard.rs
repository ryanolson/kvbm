// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use futures::stream::{FuturesUnordered, StreamExt};
use kvbm_physical::transfer::TransferCompleteNotification;

/// Serializes replicated collective onboards and permanently fails the rank
/// group after its first divergent completion.
pub(super) struct ReplicatedOnboardCoordinator {
    sequence: Arc<tokio::sync::Mutex<()>>,
    failure: Arc<OnceLock<String>>,
    events: Arc<::velo::EventManager>,
    runtime: tokio::runtime::Handle,
}

impl ReplicatedOnboardCoordinator {
    pub(super) fn new(events: Arc<::velo::EventManager>, runtime: tokio::runtime::Handle) -> Self {
        Self {
            sequence: Arc::new(tokio::sync::Mutex::new(())),
            failure: Arc::new(OnceLock::new()),
            events,
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
        let event = self.events.new_event()?;
        let awaiter = self.events.awaiter(event.handle())?;
        let sequence = Arc::clone(&self.sequence);
        let failure = Arc::clone(&self.failure);

        self.runtime.spawn(async move {
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
            let result = match failure.get() {
                Some(reason) => Err(anyhow::anyhow!(
                    "replicated onboard group is aborted: {reason}"
                )),
                None => await_rank_notifications(dispatch()).await,
            };
            let Err(error) = result else {
                let _ = event.trigger();
                return;
            };

            let candidate = format!("replicated onboard rank group failed: {error:#}");
            let first_failure = failure.set(candidate.clone()).is_ok();
            let reason = failure.get().cloned().unwrap_or(candidate);
            let _ = event.poison(reason.clone());
            drop(sequence_guard);
            if first_failure {
                abort(reason);
            }
        });

        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }
}

/// Poll every rank concurrently and surface the first failure. Serial waiting
/// can hide a failed RPC behind an earlier rank blocked in its collective.
async fn await_rank_notifications(
    notifications: Vec<Result<TransferCompleteNotification>>,
) -> Result<()> {
    let mut pending = FuturesUnordered::new();
    for notification in notifications {
        let notification = notification?;
        pending.push(async move { notification.await });
    }
    while let Some(result) = pending.next().await {
        result?;
    }
    Ok(())
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
        let coordinator = ReplicatedOnboardCoordinator::new(
            Arc::clone(&events),
            tokio::runtime::Handle::current(),
        );
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
        let coordinator = ReplicatedOnboardCoordinator::new(
            Arc::clone(&events),
            tokio::runtime::Handle::current(),
        );
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
        let error = tokio::time::timeout(Duration::from_secs(1), completion)
            .await
            .expect("a failed rank must not wait for a peer stuck in its collective")
            .expect_err("the logical onboard must surface the rank failure");
        assert!(error.to_string().contains("injected rank RPC failure"));
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
    }
}
