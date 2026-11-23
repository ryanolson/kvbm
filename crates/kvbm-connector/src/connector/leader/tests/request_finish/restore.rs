// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, anyhow};
use kvbm_engine::worker::velo::serve_test_local_transfers;
use kvbm_physical::transfer::{TransferCompleteNotification, TransferDrainOutcome};
use tokio::sync::mpsc;

use crate::common::SchedulerOutput;

use super::*;

mod scheduler;
use scheduler::SchedulerCallbacks;

#[derive(Clone, Copy, Debug)]
enum DeliveryOrder {
    ReceiveFirst,
    SendFirst,
    StoppedAfterPreemption,
}

struct HeldCopy {
    src: Vec<usize>,
    dst: Vec<usize>,
    prove: oneshot::Sender<TransferDrainOutcome>,
}

impl HeldCopy {
    fn complete(self) {
        assert!(self.prove.send(TransferDrainOutcome::Completed).is_ok());
    }

    fn fail_drained(self) {
        assert!(
            self.prove
                .send(TransferDrainOutcome::DrainedWithError(anyhow!(
                    "held copy failed"
                )))
                .is_ok()
        );
    }
}

async fn next_copy(
    copies: &mut mpsc::UnboundedReceiver<(LogicalLayoutHandle, HeldCopy)>,
    direction: LogicalLayoutHandle,
) -> Result<HeldCopy> {
    let (layout, copy) = tokio::time::timeout(WAIT, copies.recv())
        .await?
        .context("worker must receive a transfer")?;
    assert_eq!(layout, direction);
    Ok(copy)
}

async fn deliver(
    scheduler: &mut SchedulerCallbacks,
    leader: &Leader,
    delivered: &mut Vec<FinishedRequests>,
    result: FinishedRequests,
) -> Result<()> {
    scheduler.deliver(leader, &result).await?;
    delivered.push(result);
    Ok(())
}

// The engine and sink run on one Tokio thread. By the time we drain the wire
// receive, its sink callback has returned and the engine has settled delivery.
#[rstest::rstest]
#[case(DeliveryOrder::ReceiveFirst, false)]
#[case(DeliveryOrder::SendFirst, false)]
#[case(DeliveryOrder::StoppedAfterPreemption, false)]
#[case(DeliveryOrder::SendFirst, true)]
#[tokio::test]
async fn preempted_restore_has_one_request_release(
    #[case] order: DeliveryOrder,
    #[case] restored_failure: bool,
) -> Result<()> {
    let RestoreHarness {
        leader,
        driver,
        completions,
        releases,
        mut pending_copy,
        _worker,
    } = RestoreHarness::new(true).await?;
    let mut old_blocks = DESTINATIONS.to_vec();
    old_blocks.push(12);
    leader.build_connector_meta(sched_new(1, REQUEST, old_blocks.clone(), 0, 3 * BS))?;
    driver.finish_forward_pass(1);
    let save = next_copy(&mut pending_copy, LogicalLayoutHandle::G1).await?;
    assert_eq!(save.src, old_blocks);

    let token = preempt(&leader)?;
    assert!(!completions.fence_recorded(&token));
    assert_no_finished(&completions);

    let request = HashSet::from([REQUEST.to_owned()]);
    let external = BS * DESTINATIONS.len();
    let mut scheduler = SchedulerCallbacks::new()?;
    let restored = leader.get_num_new_matched_tokens(REQUEST, 0)?;
    if matches!(order, DeliveryOrder::StoppedAfterPreemption) {
        assert_eq!(
            restored,
            (None, false),
            "stop arrives before any restore was admitted"
        );
    }
    let mut delivered = Vec::new();
    let pending;
    let waiting_for_remote;
    if restored == (None, false) {
        // With admission closed, neither allocation nor dispatch can begin.
        assert!(pending_copy.try_recv().is_err());
        assert!(!completions.fence_recorded(&token));
        if !matches!(order, DeliveryOrder::SendFirst) {
            // Finish while restore is deferred. The old save owns the release.
            waiting_for_remote = false;
            pending = if matches!(order, DeliveryOrder::StoppedAfterPreemption) {
                scheduler.stop_preempted(&leader).await?
            } else {
                scheduler.abort(&leader, waiting_for_remote).await?
            };
            assert!(pending);
            assert_no_finished(&completions);
            save.complete();
            let send = wait_for_finished(&completions).await?;
            assert!(send.onboarding.is_empty());
            assert_eq!(send.offloading, request);
            deliver(&mut scheduler, &leader, &mut delivered, send).await?;
        } else {
            // With no abort yet, the old save opens only its eviction fence.
            save.complete();
            eventually(|| completions.fence_recorded(&token)).await?;
            assert_no_finished(&completions);
            assert_eq!(
                leader.get_num_new_matched_tokens(REQUEST, 0)?,
                (Some(external), true)
            );
            leader.allocate(REQUEST, DESTINATIONS.to_vec(), external)?;
            let load = next_copy(&mut pending_copy, LogicalLayoutHandle::G2).await?;
            if restored_failure {
                load.fail_drained();
            } else {
                load.complete();
            }
            let recv = wait_for_finished(&completions).await?;
            assert_eq!(recv.onboarding, request);
            assert!(recv.offloading.is_empty());
            if restored_failure {
                assert_eq!(completions.drain_failed(), HashSet::from(DESTINATIONS));
            }
            waiting_for_remote = true;
            pending = scheduler.abort(&leader, waiting_for_remote).await?;
            deliver(&mut scheduler, &leader, &mut delivered, recv).await?;
        }
    } else {
        // Follow the admitted restore through the worker even on the broken
        // path so both outputs reach the pinned scheduler.
        assert_eq!(restored, (Some(external), true));
        leader.allocate(REQUEST, DESTINATIONS.to_vec(), external)?;
        let load = next_copy(&mut pending_copy, LogicalLayoutHandle::G2).await?;
        assert_eq!(load.dst, DESTINATIONS);
        assert!(load.dst.iter().all(|id| save.src.contains(id)));
        load.complete();
        let recv = wait_for_finished(&completions).await?;
        assert_eq!(recv.onboarding, request);
        assert!(recv.offloading.is_empty());
        assert!(!completions.fence_recorded(&token));
        waiting_for_remote = true;
        pending = scheduler.abort(&leader, waiting_for_remote).await?;
        assert!(pending);
        let recv = if matches!(order, DeliveryOrder::ReceiveFirst) {
            deliver(&mut scheduler, &leader, &mut delivered, recv).await?;
            assert!(!completions.fence_recorded(&token));
            None
        } else {
            Some(recv)
        };
        save.complete();
        let send = wait_for_finished(&completions).await?;
        assert!(send.onboarding.is_empty());
        assert_eq!(send.offloading, request);
        deliver(&mut scheduler, &leader, &mut delivered, send).await?;
        if let Some(recv) = recv {
            deliver(&mut scheduler, &leader, &mut delivered, recv).await?;
        }
    }

    eventually(|| completions.fence_recorded(&token)).await?;
    leader.update_connector_output(HashSet::new(), HashSet::new())?;
    assert!(!leader.has_slot(REQUEST));
    assert_no_finished(&completions);
    let entries: usize = delivered
        .iter()
        .map(|output| output.offloading.len() + output.onboarding.len())
        .sum();
    assert_eq!(releases.load(Ordering::Relaxed), entries);
    scheduler.assert_one_free().await?;
    assert_eq!(
        entries, 1,
        "{order:?} delivered {delivered:?}, pending={pending}, waiting_for_remote={waiting_for_remote}"
    );
    Ok(())
}

struct RestoreHarness {
    leader: Leader,
    driver: Arc<dyn WorkerEngineDriver>,
    completions: Arc<WorkerCompletionState>,
    releases: Arc<AtomicUsize>,
    pending_copy: mpsc::UnboundedReceiver<(LogicalLayoutHandle, HeldCopy)>,
    _worker: Arc<Messenger>,
}

impl RestoreHarness {
    async fn new(enable_offload: bool) -> Result<Self> {
        let pair = create_messenger_pair_tcp().await?;
        let (copies, pending_copy) = mpsc::unbounded_channel();
        serve_test_local_transfers(&pair.messenger_b, move |layout, src, dst| {
            let (prove, wait) = oneshot::channel();
            copies
                .send((layout, HeldCopy { src, dst, prove }))
                .map_err(|_| anyhow!("test stopped observing transfers"))?;
            Ok(TransferCompleteNotification::from_drain_future(
                async move {
                    wait.await.unwrap_or_else(|_| {
                        TransferDrainOutcome::Unproven(anyhow!("test dropped proof"))
                    })
                },
            ))
        })?;

        let completions = Arc::new(WorkerCompletionState::new());
        register_completion_handlers(&pair.messenger_b, completions.clone());
        let worker_id = pair.messenger_b.instance_id();
        let releases = Arc::new(AtomicUsize::new(0));
        let sink = Arc::new(CountingSink {
            sink: VeloWorkerSink::new(
                pair.messenger_a.clone(),
                vec![worker_id],
                pair.messenger_a.runtime().clone(),
            ),
            releases: releases.clone(),
            hold_load_delivery: Mutex::new(None),
        });
        let (leader, driver) =
            build_test_leader(pair.messenger_a, worker_id, sink, enable_offload)?;
        Ok(Self {
            leader,
            driver,
            completions,
            releases,
            pending_copy,
            _worker: pair.messenger_b,
        })
    }
}

fn preempt(leader: &Leader) -> Result<FenceToken> {
    let mut preempt = SchedulerOutput::new(2);
    preempt.preempted_req_ids.push(REQUEST.to_owned());
    leader.build_connector_meta(preempt)?;
    let metadata = leader.build_metadata();
    assert_eq!(metadata.evicted_requests, vec![REQUEST.to_owned()]);
    let [fence] = metadata.fences.as_slice() else {
        bail!("expected one eviction fence, got {:?}", metadata.fences);
    };
    let [token] = fence.per_worker.as_slice() else {
        bail!("expected one worker fence, got {:?}", fence.per_worker);
    };
    assert_eq!(fence.request_id, REQUEST);
    Ok(*token)
}

#[rstest::rstest]
#[case(false)]
#[case(true)]
#[tokio::test]
async fn failed_evicted_onboard_keeps_restore_deferred_until_proof(
    #[case] abort_while_deferred: bool,
) -> Result<()> {
    let RestoreHarness {
        leader,
        completions,
        releases,
        mut pending_copy,
        _worker,
        ..
    } = RestoreHarness::new(false).await?;
    let external = BS * DESTINATIONS.len();
    assert_eq!(
        leader.get_num_new_matched_tokens(REQUEST, 0)?,
        (Some(external), true)
    );
    leader.allocate(REQUEST, DESTINATIONS.to_vec(), external)?;
    let load = next_copy(&mut pending_copy, LogicalLayoutHandle::G2).await?;
    let token = preempt(&leader)?;
    assert_eq!(
        leader.get_num_new_matched_tokens(REQUEST, 0)?,
        (None, false)
    );
    assert!(!completions.fence_recorded(&token));
    assert_no_finished(&completions);
    let scheduler = if abort_while_deferred {
        let mut scheduler = SchedulerCallbacks::new()?;
        assert!(scheduler.abort(&leader, false).await?);
        assert_no_finished(&completions);
        Some(scheduler)
    } else {
        None
    };
    load.fail_drained();
    eventually(|| completions.fence_recorded(&token)).await?;
    if let Some(mut scheduler) = scheduler {
        let result = wait_for_finished(&completions).await?;
        assert_eq!(result.onboarding, HashSet::from([REQUEST.to_owned()]));
        assert!(result.offloading.is_empty());
        assert_eq!(completions.drain_failed(), HashSet::from(DESTINATIONS));
        scheduler.deliver(&leader, &result).await?;
        scheduler.assert_one_free().await?;
        assert_eq!(releases.load(Ordering::Relaxed), 1);
    } else {
        assert_eq!(releases.load(Ordering::Relaxed), 0);
        leader.update_connector_output(HashSet::new(), HashSet::new())?;
        assert_eq!(
            leader.get_num_new_matched_tokens(REQUEST, 0)?,
            (Some(external), true)
        );
    }
    assert_no_finished(&completions);
    Ok(())
}
