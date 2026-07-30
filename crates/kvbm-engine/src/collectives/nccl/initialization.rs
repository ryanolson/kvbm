// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;

pub(super) fn initialize_rank_group<T, Initialize, Abort>(
    rank_count: usize,
    timeout: Duration,
    initialize: Initialize,
    abort: Abort,
) -> Result<Vec<T>>
where
    T: Send + Sync,
    Initialize: Fn(usize, &AtomicBool, Instant) -> Result<T> + Sync,
    Abort: Fn(&T, &str) -> Result<()> + Sync,
{
    let cancelled = AtomicBool::new(false);
    let deadline = Instant::now() + timeout;
    let results = std::thread::scope(|scope| {
        let initializers = (0..rank_count)
            .map(|rank| {
                let cancelled = &cancelled;
                let initialize = &initialize;
                scope.spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        initialize(rank, cancelled, deadline)
                    }))
                    .unwrap_or_else(|_| {
                        Err(anyhow::anyhow!("NCCL rank {rank} initializer panicked"))
                    });
                    if result.is_err() {
                        cancelled.store(true, Ordering::Release);
                    }
                    (rank, result)
                })
            })
            .collect::<Vec<_>>();

        initializers
            .into_iter()
            .map(|initializer| {
                initializer
                    .join()
                    .expect("rank initializer catches and converts panics")
            })
            .collect::<Vec<_>>()
    });

    let first_failure = results.iter().find_map(|(rank, result)| {
        result
            .as_ref()
            .err()
            .map(|error| (*rank, format!("{error:#}")))
    });
    let Some((failed_rank, error)) = first_failure else {
        return results.into_iter().map(|(_, result)| result).collect();
    };

    cancelled.store(true, Ordering::Release);
    let reason = format!("NCCL rank {failed_rank} initialization failed: {error}");
    let cleanup_errors = std::thread::scope(|scope| {
        let aborts = results
            .iter()
            .filter_map(|(rank, result)| {
                let collective = result.as_ref().ok()?;
                let abort = &abort;
                let reason = &reason;
                Some((*rank, scope.spawn(move || abort(collective, reason))))
            })
            .collect::<Vec<_>>();
        aborts
            .into_iter()
            .filter_map(|(rank, abort)| match abort.join() {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(format!("rank {rank}: {error:#}")),
                Err(_) => Some(format!("rank {rank}: abort panicked")),
            })
            .collect::<Vec<_>>()
    });

    if cleanup_errors.is_empty() {
        anyhow::bail!(reason)
    }
    anyhow::bail!(
        "{reason}; initialized-rank cleanup also failed: {}",
        cleanup_errors.join("; ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    #[test]
    fn rank_failure_cancels_peers_and_aborts_completed_ranks() {
        let peer_observed_cancellation = AtomicBool::new(false);
        let aborted = AtomicUsize::new(0);
        let result = initialize_rank_group(
            3,
            Duration::from_secs(5),
            |rank, cancelled, deadline| match rank {
                0 => Ok(rank),
                1 => anyhow::bail!("injected initializer failure"),
                2 => {
                    while !cancelled.load(Ordering::Acquire) {
                        assert!(Instant::now() < deadline, "peer cancellation timed out");
                        std::thread::yield_now();
                    }
                    peer_observed_cancellation.store(true, Ordering::Release);
                    anyhow::bail!("cancelled after peer failure")
                }
                _ => unreachable!(),
            },
            |rank, reason| {
                assert_eq!(*rank, 0);
                assert!(reason.contains("injected initializer failure"));
                aborted.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .expect_err("one failed rank must fail the complete initialization group");

        assert!(result.to_string().contains("injected initializer failure"));
        assert!(peer_observed_cancellation.load(Ordering::Acquire));
        assert_eq!(aborted.load(Ordering::SeqCst), 1);
    }
}
