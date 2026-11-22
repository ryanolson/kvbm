// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A copy on two workers whose replies are lost.
//!
//! The copy runs through [`SpmdParallelWorkers`] on both of its routes: a
//! G2 to G1 onboard goes through the replicated onboard coordinator, and a
//! G1 to G2 copy through the plain receipt aggregate. Both workers run on one
//! messenger with distinct worker ids.

use super::*;
use crate::worker::group::SpmdParallelWorkers;
use crate::worker::{UnresolvedTransfer, Worker, WorkerTransfers};

const OTHER: u64 = WORKER + 1;
/// Bounds every await in these tests.
const LIMIT: Duration = Duration::from_secs(10);

/// The two routes of a local copy through the SPMD group.
const ROUTES: [(LogicalLayoutHandle, LogicalLayoutHandle); 2] = [
    (LogicalLayoutHandle::G2, LogicalLayoutHandle::G1),
    (LogicalLayoutHandle::G1, LogicalLayoutHandle::G2),
];

struct TwoWorkers {
    pair: MessengerPair,
    first: Arc<LocalTransferRecords>,
    second: Arc<LocalTransferRecords>,
}

impl TwoWorkers {
    /// Two workers with the production handlers over stub copies that
    /// complete.
    async fn start() -> Result<Self> {
        let pair = create_messenger_pair_tcp().await?;
        let first = LocalTransferRecords::new();
        let second = LocalTransferRecords::new();
        for (worker_id, records) in [(WORKER, &first), (OTHER, &second)] {
            register_local_transfer_handlers(
                &pair.messenger_b,
                Some(worker_id),
                Arc::clone(records),
                copy_ending(|| TransferDrainOutcome::Completed),
            )?;
        }
        Ok(Self {
            pair,
            first,
            second,
        })
    }

    /// A fresh SPMD group of one client per worker. A failed onboard aborts
    /// its group for good, so each copy takes a new one.
    fn group(&self) -> SpmdParallelWorkers {
        let workers = [WORKER, OTHER]
            .into_iter()
            .map(|worker_id| {
                Arc::new(
                    VeloWorkerClient::new_for_worker(
                        Arc::clone(&self.pair.messenger_a),
                        self.pair.messenger_b.instance_id(),
                        worker_id,
                    )
                    .with_local_transfer_check(TIMING),
                ) as Arc<dyn Worker>
            })
            .collect();
        SpmdParallelWorkers::new(
            workers,
            Arc::new(::velo::EventManager::local()),
            tokio::runtime::Handle::current(),
        )
    }

    /// Run one copy on both workers along `route` and await its outcome.
    async fn copy(
        &self,
        (src, dst): (LogicalLayoutHandle, LogicalLayoutHandle),
    ) -> Result<TransferDrainOutcome> {
        let receipt = self.group().execute_local_transfer(
            src,
            dst,
            Arc::<[BlockId]>::from(vec![0]),
            Arc::<[BlockId]>::from(vec![0]),
            TransferOptions::default(),
        )?;
        Ok(tokio::time::timeout(LIMIT, receipt.await_drain()).await?)
    }

    fn lose_and_evict(records: &LocalTransferRecords) {
        records
            .hooks
            .evict_after_next
            .store(true, Ordering::Release);
        records.hooks.lose_next_reply.store(true, Ordering::Release);
    }
}

/// The participants of an `Unproven` outcome. Panics on any other outcome.
fn participants_of(outcome: TransferDrainOutcome) -> UnresolvedTransfer {
    match outcome {
        TransferDrainOutcome::Unproven(error) => UnresolvedTransfer::from_unproven(error),
        other => panic!("expected Unproven, got {}", describe(&other)),
    }
}

/// `(worker id, incarnation)` of every capability, in worker id order.
fn named_workers(unresolved: &UnresolvedTransfer) -> Vec<(Option<u64>, Option<uuid::Uuid>)> {
    let mut named: Vec<_> = unresolved
        .rechecks()
        .map(|recheck| (recheck.worker_id, recheck.incarnation))
        .collect();
    named.sort();
    named
}

#[tokio::test]
async fn both_unproven_workers_keep_a_recheck_each() -> Result<()> {
    let workers = TwoWorkers::start().await?;
    for route in ROUTES {
        TwoWorkers::lose_and_evict(&workers.first);
        TwoWorkers::lose_and_evict(&workers.second);

        let unresolved = participants_of(workers.copy(route).await?);

        assert_eq!(
            unresolved.participants().len(),
            2,
            "route {route:?}: every unproven worker is a participant"
        );
        assert_eq!(
            named_workers(&unresolved),
            vec![
                (Some(WORKER), Some(workers.first.incarnation())),
                (Some(OTHER), Some(workers.second.incarnation())),
            ],
            "route {route:?}: each participant names its own worker"
        );
        for recheck in unresolved.rechecks() {
            assert_eq!(recheck.instance, workers.pair.messenger_b.instance_id());
        }
    }
    Ok(())
}

#[tokio::test]
#[ignore = "the aggregate re-check sends requests to every worker"]
async fn a_probe_for_one_worker_only_checks_that_worker() -> Result<()> {
    let workers = tokio::time::timeout(LIMIT, TwoWorkers::start()).await??;
    TwoWorkers::lose_and_evict(&workers.first);
    TwoWorkers::lose_and_evict(&workers.second);
    let mut unresolved = participants_of(workers.copy(ROUTES[1]).await?);
    let first_checks = checks(&workers.first);
    let second_checks = checks(&workers.second);

    // The first worker's probe owns this pass. The second worker has its own probe.
    tokio::time::timeout(
        LIMIT,
        unresolved.recheck(&workers.pair.messenger_a, TIMING.deadline),
    )
    .await?;

    assert_eq!(checks(&workers.first) - first_checks, 1);
    assert_eq!(checks(&workers.second) - second_checks, 0);
    Ok(())
}

#[tokio::test]
async fn a_completed_worker_and_an_unproven_worker_leave_only_the_unproven_one() -> Result<()> {
    let workers = TwoWorkers::start().await?;
    for route in ROUTES {
        TwoWorkers::lose_and_evict(&workers.second);

        let unresolved = participants_of(workers.copy(route).await?);

        assert_eq!(
            unresolved.participants().len(),
            1,
            "route {route:?}: only the unproven worker is unresolved"
        );
        assert_eq!(
            named_workers(&unresolved),
            vec![(Some(OTHER), Some(workers.second.incarnation()))],
            "route {route:?}"
        );
    }
    Ok(())
}

/// Both checks come before the copies are recorded. One worker's copy then
/// completes: a re-check resolves it, but the other stays unknown, so the
/// transfer stays unproven. It settles once the other copy completes too.
#[tokio::test]
async fn the_transfer_settles_only_after_every_participant_is_resolved() -> Result<()> {
    let workers = TwoWorkers::start().await?;
    for route in ROUTES {
        let (open_first, gate) = tokio::sync::oneshot::channel();
        *workers.first.hooks.defer_next.lock() = Some(gate);
        let (open_second, gate) = tokio::sync::oneshot::channel();
        *workers.second.hooks.defer_next.lock() = Some(gate);
        let first_checks = checks(&workers.first);
        let second_checks = checks(&workers.second);

        let mut unresolved = participants_of(workers.copy(route).await?);
        assert_eq!(unresolved.participants().len(), 2, "route {route:?}");
        let first_id = unresolved
            .rechecks()
            .find(|recheck| recheck.worker_id == Some(WORKER))
            .expect("the first worker is a participant")
            .transfer_id;
        let second_id = unresolved
            .rechecks()
            .find(|recheck| recheck.worker_id == Some(OTHER))
            .expect("the second worker is a participant")
            .transfer_id;

        open_first.send(()).expect("the first copy is waiting");
        tokio::time::timeout(
            LIMIT,
            wait_for_record(&workers.first, first_id, Record::Completed),
        )
        .await?;
        tokio::time::timeout(
            LIMIT,
            unresolved.recheck(&workers.pair.messenger_a, TIMING.deadline),
        )
        .await?;

        assert!(!unresolved.is_resolved(), "route {route:?}");
        assert_eq!(
            named_workers(&unresolved),
            vec![(Some(OTHER), Some(workers.second.incarnation()))],
            "route {route:?}: the unknown worker is still unresolved"
        );

        open_second.send(()).expect("the second copy is waiting");
        tokio::time::timeout(
            LIMIT,
            wait_for_record(&workers.second, second_id, Record::Completed),
        )
        .await?;
        tokio::time::timeout(
            LIMIT,
            unresolved.recheck(&workers.pair.messenger_a, TIMING.deadline),
        )
        .await?;

        assert!(unresolved.is_resolved(), "route {route:?}");
        let outcome = unresolved.into_outcome();
        assert!(
            matches!(outcome, TransferDrainOutcome::Completed),
            "route {route:?}: got {}",
            describe(&outcome)
        );
        assert_eq!(
            checks(&workers.first) - first_checks,
            2,
            "route {route:?}: a resolved participant is not checked again"
        );
        assert_eq!(
            checks(&workers.second) - second_checks,
            3,
            "route {route:?}"
        );
    }
    Ok(())
}

/// A participant with no capability never resolves, even when every other
/// participant does.
#[tokio::test]
async fn a_participant_without_a_capability_keeps_the_transfer_unproven() -> Result<()> {
    let workers = TwoWorkers::start().await?;
    let (open, gate) = tokio::sync::oneshot::channel();
    *workers.first.hooks.defer_next.lock() = Some(gate);
    let client = VeloWorkerClient::new_for_worker(
        Arc::clone(&workers.pair.messenger_a),
        workers.pair.messenger_b.instance_id(),
        WORKER,
    )
    .with_local_transfer_check(TIMING);
    let receipt = TransferCompleteNotification::aggregate_results(
        vec![
            transfer(&client),
            Ok(TransferCompleteNotification::from_drain_future(async {
                TransferDrainOutcome::Unproven(anyhow::anyhow!("in-process drain failed"))
            })),
        ],
        &Arc::new(::velo::EventManager::local()),
        &tokio::runtime::Handle::current(),
    )?;
    let mut unresolved = participants_of(tokio::time::timeout(LIMIT, receipt.await_drain()).await?);
    assert_eq!(unresolved.participants().len(), 2);
    let transfer_id = unresolved
        .rechecks()
        .next()
        .expect("the velo worker is a participant")
        .transfer_id;

    open.send(()).expect("the copy is waiting");
    tokio::time::timeout(
        LIMIT,
        wait_for_record(&workers.first, transfer_id, Record::Completed),
    )
    .await?;
    tokio::time::timeout(
        LIMIT,
        unresolved.recheck(&workers.pair.messenger_a, TIMING.deadline),
    )
    .await?;

    assert_eq!(unresolved.participants().len(), 1);
    assert!(unresolved.participants()[0].recheck().is_none());
    let outcome = unresolved.into_outcome();
    match &outcome {
        TransferDrainOutcome::Unproven(error) => {
            assert!(format!("{error:#}").contains("in-process drain failed"));
        }
        other => panic!("expected Unproven, got {}", describe(other)),
    }
    Ok(())
}
