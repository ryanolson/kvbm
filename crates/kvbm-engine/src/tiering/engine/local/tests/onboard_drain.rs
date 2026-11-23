// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A local onboard whose worker reply is lost while the copy may still run.
//!
//! The worker accepts the G2 to G1 copy and signals that it started. The test
//! then owns the running copy as a guard, separate from the receipt. Losing
//! the reply resolves the receipt `Unproven` while that guard is still held:
//! the copy may still read its G2 sources and write its G1 destinations. Until
//! a later check proves the copy drained, the onboard must keep its sources,
//! keep its destinations from the scheduler, and stay `Pending`.
//!
//! Each test waits for the onboard driver task to return before it reads the
//! sink. The driver writes the handle's cell before it calls the sink, so a
//! wait on the handle alone could read the sink too early.

use std::time::Duration;

use super::super::super::test_support::{
    DeliveredOutcome, HangingParallelWorkers, build_leader_with_resources,
    build_leader_with_workers, wait_onboard_driver_finished,
};
use super::onboard_pause::{eventually, tick};
use super::*;
use kvbm_common::LogicalResourceId;
use kvbm_logical::{BlockManager, MutableBlock};
use kvbm_physical::transfer::{TransferCompleteNotification, TransferDrainOutcome};
use kvbm_protocols::connector::LoadOutcome;
use tokio::sync::{mpsc, oneshot};

use crate::testing::messenger::create_messenger_pair_tcp;
use crate::worker::group::SpmdParallelWorkers;
use crate::worker::velo::{
    HeldLocalTransfers, LocalTransferCheckTiming, VeloWorkerClient, serve_held_local_transfers,
};

const RESOURCE: LogicalResourceId = LogicalResourceId(0);
const G2_BLOCKS: usize = 2;
const G1_DESTINATION: BlockId = 50;
const REQUEST: &str = "lost-reply-onboard";

struct Onboard {
    engine: Arc<LocalConnectorEngine>,
    hanging: Arc<HangingParallelWorkers>,
    sink: Arc<RecordingSink>,
    g2: Arc<BlockManager<crate::G2>>,
}

async fn build() -> Result<Onboard> {
    let hanging = HangingParallelWorkers::new(RESOURCE);
    let (leader, g2) = build_leader_with_workers(RESOURCE, G2_BLOCKS, BS, hanging.clone()).await?;
    let sink = RecordingSink::new();
    let engine = LocalConnectorEngine::new(Arc::new(leader), sink.clone(), BS, true);
    Ok(Onboard {
        engine,
        hanging,
        sink,
        g2,
    })
}

/// Register one G2 block and hand it to a matched search. Returns the search
/// handle and the source block id. The test keeps no reference to the block:
/// only the onboard holds it once it starts.
pub(super) fn matched_search(
    engine: &Arc<LocalConnectorEngine>,
    g2: &BlockManager<crate::G2>,
    request: &str,
) -> Result<(FindBlocksHandle, BlockId)> {
    let tokens = crate::testing::token_blocks::create_token_sequence(1, BS, 7);
    let block = g2
        .allocate_blocks(1)
        .ok_or_else(|| anyhow::anyhow!("no free G2 block"))?
        .pop()
        .expect("one block");
    let block = block
        .complete(&tokens.blocks()[0])
        .map_err(|error| anyhow::anyhow!("complete G2 block: {error:?}"))?;
    let source = g2.register_block(block);
    let source_id = source.block_id();
    let ready = FindMatchesResult::Ready(ReadyResult::new(
        vec![source],
        MatchBreakdown {
            host_blocks: 1,
            ..MatchBreakdown::default()
        },
    ));

    let search_id = SearchId::new();
    engine.searches.insert(
        search_id,
        SearchState {
            request_id: request.into(),
            status: Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 1 })),
            onboarding: OnboardingState::new(
                0,
                BS + 1,
                OnboardingShard {
                    start_block: 0,
                    num_queried_blocks: 1,
                    find_session: ready,
                },
            ),
            buffer: plhs_for(0, 1),
        },
    );
    let owner: Arc<dyn LeaderEngine> = engine.clone();
    let handle = FindBlocksHandle::search(request.into(), search_id, Arc::downgrade(&owner));
    Ok((handle, source_id))
}

/// Take every block the G2 pool can give out now and return their ids. A
/// block the onboard still holds is not among them; a released source is,
/// because allocation reuses inactive blocks. Panics if the pool cannot give
/// out the blocks it reports available, so the reuse checks never pass on an
/// empty allocation.
pub(super) fn reusable_g2_ids(g2: &BlockManager<crate::G2>) -> Vec<BlockId> {
    let available = g2.available_blocks();
    let blocks = g2.allocate_blocks(available).unwrap_or_else(|| {
        panic!("the G2 pool could not allocate its {available} available blocks")
    });
    blocks.iter().map(|block| block.block_id()).collect()
}

/// Panic once with every primary failure, so a base run reports all of them.
fn assert_primary(failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "primary checks failed:\n  - {}",
        failures.join("\n  - ")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn onboard_blocks_keeps_sources_and_destinations_after_a_lost_reply() -> Result<()> {
    let fixture = build().await?;
    let (search, source_id) = matched_search(&fixture.engine, &fixture.g2, REQUEST)?;
    let mut onboard = fixture
        .engine
        .clone()
        .onboard_blocks(&search, &[G1_DESTINATION], BS)?;
    let action_id = *onboard.id();

    fixture.hanging.wait_started().await;
    let copy = fixture.hanging.take_running_copy()?;
    fixture.hanging.lose_reply(&copy)?;
    wait_onboard_driver_finished(action_id).await;
    assert!(fixture.hanging.copy_running(), "the copy is still running");

    let mut failures = Vec::new();
    let reusable = reusable_g2_ids(&fixture.g2);
    if reusable.contains(&source_id) {
        failures.push(format!(
            "the G2 source block {source_id} was reusable while the copy may still read it \
             (reusable ids {reusable:?})"
        ));
    }
    let loads = fixture.sink.loads();
    if !loads.is_empty() {
        failures.push(format!(
            "the scheduler was told to release the G1 destinations while the copy may still \
             write them (sink loads {loads:?})"
        ));
    }
    if onboard.is_complete() {
        failures.push(format!(
            "the onboard must stay Pending after a lost reply, got {:?}",
            onboard.outcome()
        ));
    }
    assert_primary(failures);

    let fence = onboard.take_physical_drain_fence();
    assert!(
        fence.as_ref().is_some_and(|fence| !fence.is_complete()),
        "the onboard must expose an incomplete physical drain fence"
    );
    assert!(
        fixture
            .engine
            .actions
            .get(&action_id)
            .is_some_and(|record| record.physical_pending),
        "the onboard's physical work must stay pending"
    );
    drop(copy);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn onboard_resource_blocks_keeps_destinations_after_a_lost_reply() -> Result<()> {
    let fixture = build().await?;
    let mut onboard = fixture.engine.clone().onboard_resource_blocks(
        &REQUEST.to_owned(),
        vec![ResourceOnboard {
            resource: RESOURCE,
            source_block_ids: vec![0],
            destination_block_ids: vec![G1_DESTINATION],
        }],
    )?;
    let action_id = *onboard.id();

    fixture.hanging.wait_started().await;
    let copy = fixture.hanging.take_running_copy()?;
    fixture.hanging.lose_reply(&copy)?;
    wait_onboard_driver_finished(action_id).await;
    assert!(fixture.hanging.copy_running(), "the copy is still running");

    let mut failures = Vec::new();
    let loads = fixture.sink.loads();
    if !loads.is_empty() {
        failures.push(format!(
            "the scheduler was told to release the G1 destinations while the copy may still \
             write them (sink loads {loads:?})"
        ));
    }
    if onboard.is_complete() {
        failures.push(format!(
            "the onboard must stay Pending after a lost reply, got {:?}",
            onboard.outcome()
        ));
    }
    assert_primary(failures);

    let fence = onboard.take_physical_drain_fence();
    assert!(
        fence.as_ref().is_some_and(|fence| !fence.is_complete()),
        "the onboard must expose an incomplete physical drain fence"
    );
    assert!(
        fixture
            .engine
            .actions
            .get(&action_id)
            .is_some_and(|record| record.physical_pending),
        "the onboard's physical work must stay pending"
    );
    drop(copy);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn onboard_blocks_releases_at_once_after_a_drained_failure() -> Result<()> {
    let fixture = build().await?;
    let (search, source_id) = matched_search(&fixture.engine, &fixture.g2, REQUEST)?;
    let onboard = fixture
        .engine
        .clone()
        .onboard_blocks(&search, &[G1_DESTINATION], BS)?;
    let action_id = *onboard.id();

    fixture.hanging.wait_started().await;
    let copy = fixture.hanging.take_running_copy()?;
    // While the copy runs, the onboard must hold its source. Without this
    // check, an onboard that dropped the source before dispatch would also
    // pass the reuse check below.
    let reusable = reusable_g2_ids(&fixture.g2);
    assert!(
        !reusable.contains(&source_id),
        "the G2 source {source_id} was reusable while the copy was running \
         (reusable ids {reusable:?})"
    );
    fixture.hanging.fail_drained(copy)?;
    wait_onboard_driver_finished(action_id).await;
    assert_eq!(
        fixture.hanging.delivered(),
        Some(DeliveredOutcome::DrainedWithError),
        "the control must deliver a drained failure, not an unproven one"
    );

    let failed = LoadOutcome::FailedPartial {
        block_ids: vec![G1_DESTINATION],
    };
    assert_eq!(onboard.outcome(), Some(failed.clone()));
    assert_eq!(fixture.sink.loads(), vec![(REQUEST.to_owned(), failed)]);
    let reusable = reusable_g2_ids(&fixture.g2);
    assert!(
        reusable.contains(&source_id),
        "a drained failure must return the G2 source {source_id} (reusable ids {reusable:?})"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_onboard_handle_keeps_the_fence_until_drain() -> Result<()> {
    let fixture = build().await?;
    let (search, source_id) = matched_search(&fixture.engine, &fixture.g2, REQUEST)?;
    let mut onboard = fixture
        .engine
        .clone()
        .onboard_blocks(&search, &[G1_DESTINATION], BS)?;
    let action_id = *onboard.id();
    fixture.hanging.wait_started().await;
    let fence = onboard
        .take_physical_drain_fence()
        .expect("the local onboard must give its caller a physical fence");
    drop(onboard);
    assert!(!fence.is_complete());
    assert!(fixture.hanging.copy_running());
    assert!(!reusable_g2_ids(&fixture.g2).contains(&source_id));
    assert!(fixture.sink.loads().is_empty());
    assert!(
        fixture
            .engine
            .actions
            .get(&action_id)
            .is_some_and(|record| record.physical_pending),
        "handle drop must leave physical work pending"
    );

    fixture.hanging.release()?;
    wait_onboard_driver_finished(action_id).await;
    assert!(fence.is_complete());
    assert!(reusable_g2_ids(&fixture.g2).contains(&source_id));
    assert_eq!(
        fixture.sink.loads(),
        vec![(REQUEST.into(), LoadOutcome::Done)]
    );
    assert!(!fixture.engine.actions.contains_key(&action_id));
    Ok(())
}

const TAIL: LogicalResourceId = LogicalResourceId(1);

struct ResourceWorker {
    _server: Arc<velo::Messenger>,
    transfers: HeldLocalTransfers,
    copies: mpsc::UnboundedReceiver<(LogicalResourceId, oneshot::Sender<TransferDrainOutcome>)>,
}

impl ResourceWorker {
    async fn copy(&mut self, expected: LogicalResourceId) -> oneshot::Sender<TransferDrainOutcome> {
        let mut copy = None;
        eventually("the resource copy to start", || {
            copy = self.copies.try_recv().ok();
            copy.is_some()
        })
        .await;
        let (resource, proof) = copy.unwrap();
        assert_eq!(resource, expected);
        proof
    }
}

struct ResourceRestore {
    engine: Arc<LocalConnectorEngine>,
    sink: Arc<RecordingSink>,
    worker: ResourceWorker,
    handle: OnboardHandle,
    caller_pins: Vec<MutableBlock<crate::G2>>,
}

async fn resource_restore(lose_first_reply: bool) -> Result<ResourceRestore> {
    let pair = create_messenger_pair_tcp().await?;
    let (started, copies) = mpsc::unbounded_channel();
    let worker_transfers = serve_held_local_transfers(
        &pair.messenger_b,
        0,
        lose_first_reply,
        move |resource, _, _| {
            let (proof, receipt) = oneshot::channel();
            started.send((resource, proof))?;
            Ok(TransferCompleteNotification::from_drain_future(
                async move {
                    receipt
                        .await
                        .unwrap_or_else(|error| TransferDrainOutcome::Unproven(error.into()))
                },
            ))
        },
    )?;
    let client = VeloWorkerClient::new_for_worker(
        pair.messenger_a.clone(),
        pair.messenger_b.instance_id(),
        0,
    )
    .with_local_transfer_check(LocalTransferCheckTiming {
        wait: Duration::from_millis(10),
        ..LocalTransferCheckTiming::default()
    });
    let workers = Arc::new(SpmdParallelWorkers::new(
        vec![Arc::new(client)],
        Arc::new(velo::EventManager::local()),
        pair.messenger_a.runtime().clone(),
    ));
    let resources = [RESOURCE, TAIL];
    let (leader, managers) =
        build_leader_with_resources(pair.messenger_a, &resources, 1, BS, workers)?;
    let sink = RecordingSink::new();
    let engine = LocalConnectorEngine::new(Arc::new(leader), sink.clone(), BS, true);
    let caller_pins: Vec<_> = managers
        .iter()
        .map(|manager| manager.allocate_blocks(1).unwrap().pop().unwrap())
        .collect();
    let transfers = resources
        .into_iter()
        .zip(&caller_pins)
        .enumerate()
        .map(|(index, (resource, pin))| ResourceOnboard {
            resource,
            source_block_ids: vec![pin.block_id()],
            destination_block_ids: vec![G1_DESTINATION + index],
        })
        .collect();
    let handle = engine
        .clone()
        .onboard_resource_blocks(&REQUEST.into(), transfers)?;
    Ok(ResourceRestore {
        engine,
        sink,
        worker: ResourceWorker {
            _server: pair.messenger_b,
            transfers: worker_transfers,
            copies,
        },
        handle,
        caller_pins,
    })
}

#[tokio::test]
async fn resource_onboard_resumes_the_tail_after_proof() -> Result<()> {
    let ResourceRestore {
        engine,
        sink,
        mut worker,
        mut handle,
        caller_pins,
    } = resource_restore(true).await?;
    let first = worker.copy(RESOURCE).await;
    wait_onboard_driver_finished(*handle.id()).await;
    assert!(
        handle.paused().is_some(),
        "lost reply must pause the resource restore"
    );
    assert!(sink.loads().is_empty());
    assert!(
        worker.copies.try_recv().is_err(),
        "the tail must wait for proof"
    );
    let fence = handle.take_physical_drain_fence().expect("resource fence");
    assert!(!fence.is_complete());

    let weak = Arc::downgrade(&engine);
    drop(engine);
    let engine = weak
        .upgrade()
        .expect("the paused restore must hold the engine");
    let admission = tokio::time::timeout(
        Duration::from_secs(5),
        engine.resource_onboard_admission.lock(),
    )
    .await?;
    assert!(first.send(TransferDrainOutcome::Completed).is_ok());
    eventually("the worker records first-copy proof", || {
        !worker.transfers.active()
    })
    .await;
    tokio::time::pause();
    tick(super::super::super::paused::PROBE_INTERVAL).await;
    eventually("the first proof clears the pause", || {
        handle.paused().is_none()
    })
    .await;
    assert!(
        worker.copies.try_recv().is_err(),
        "the tail must wait for admission"
    );
    assert!(
        handle.outcome().is_none(),
        "one proof cannot complete the restore"
    );
    assert!(!fence.is_complete());
    drop(admission);
    drop(engine);

    let tail = worker.copy(TAIL).await;
    assert!(sink.loads().is_empty());
    assert!(!fence.is_complete());
    assert!(tail.send(TransferDrainOutcome::Completed).is_ok());
    eventually("terminal after tail proof", || !sink.loads().is_empty()).await;
    assert_eq!(handle.outcome(), Some(LoadOutcome::Done));
    assert!(fence.is_complete());
    assert_eq!(sink.loads(), vec![(REQUEST.into(), LoadOutcome::Done)]);
    drop(caller_pins);
    Ok(())
}

#[tokio::test]
async fn resource_onboard_pauses_again_after_a_second_lost_reply() -> Result<()> {
    use super::super::super::paused::PROBE_INTERVAL;

    let mut restore = resource_restore(true).await?;
    let action = *restore.handle.id();
    let first = restore.worker.copy(RESOURCE).await;
    wait_onboard_driver_finished(action).await;
    let first_pause = restore
        .handle
        .paused()
        .expect("the first lost reply must pause");
    let fence = restore
        .handle
        .take_physical_drain_fence()
        .expect("resource fence");
    assert!(!fence.is_complete());
    assert!(restore.sink.loads().is_empty());

    restore.worker.transfers.lose_next_reply();
    assert!(first.send(TransferDrainOutcome::Completed).is_ok());
    eventually("the worker records first-copy proof", || {
        !restore.worker.transfers.active()
    })
    .await;
    tokio::time::pause();
    tick(PROBE_INTERVAL).await;
    let tail = restore.worker.copy(TAIL).await;
    tokio::time::resume();
    wait_onboard_driver_finished(action).await;
    let second_pause = restore
        .handle
        .paused()
        .expect("the second lost reply must reinsert the pause");
    assert!(second_pause.since() > first_pause.since());
    assert_eq!(second_pause.worker(), first_pause.worker());
    assert!(restore.worker.transfers.active());
    assert!(restore.handle.outcome().is_none());
    assert!(!fence.is_complete());
    assert!(restore.sink.loads().is_empty());

    assert!(tail.send(TransferDrainOutcome::Completed).is_ok());
    eventually("the worker records tail proof", || {
        !restore.worker.transfers.active()
    })
    .await;
    tokio::time::pause();
    tick(PROBE_INTERVAL).await;
    eventually("the re-paused restore settles", || {
        !restore.sink.loads().is_empty()
    })
    .await;
    assert!(restore.handle.paused().is_none());
    assert_eq!(restore.handle.outcome(), Some(LoadOutcome::Done));
    assert!(fence.is_complete());
    assert_eq!(
        restore.sink.loads(),
        vec![(REQUEST.into(), LoadOutcome::Done)]
    );
    assert!(restore.worker.copies.try_recv().is_err());
    Ok(())
}

async fn resource_failure_keeps_its_destination_ids(lost_reply: bool) -> Result<()> {
    let mut restore = resource_restore(lost_reply).await?;
    let proof = restore.worker.copy(RESOURCE).await;
    let action = *restore.handle.id();
    if lost_reply {
        wait_onboard_driver_finished(action).await;
        assert!(
            restore.handle.paused().is_some(),
            "lost reply must pause the resource restore"
        );
        assert!(restore.sink.loads().is_empty());
    }
    let fence = restore
        .handle
        .take_physical_drain_fence()
        .expect("resource fence");
    assert!(!fence.is_complete());
    assert!(
        proof
            .send(TransferDrainOutcome::DrainedWithError(anyhow::anyhow!(
                "worker copy failed"
            )))
            .is_ok()
    );
    if lost_reply {
        eventually("the worker records the drained failure", || {
            !restore.worker.transfers.active()
        })
        .await;
        tokio::time::pause();
        tick(super::super::super::paused::PROBE_INTERVAL).await;
        eventually("resource failure terminal", || {
            !restore.sink.loads().is_empty()
        })
        .await;
    } else {
        wait_onboard_driver_finished(action).await;
    }
    assert!(fence.is_complete());
    assert_eq!(
        restore.engine.poll_action(&action),
        ActionStatus::Failed(ActionFailure::Resource {
            resource: RESOURCE,
            block_ids: Some(vec![G1_DESTINATION]),
        }),
    );
    assert_eq!(
        restore.sink.loads(),
        vec![(
            REQUEST.into(),
            LoadOutcome::FailedPartial {
                block_ids: vec![G1_DESTINATION]
            }
        )],
    );
    assert!(restore.handle.paused().is_none());
    assert!(restore.worker.copies.try_recv().is_err());
    Ok(())
}

#[tokio::test]
async fn resource_onboard_reports_the_failed_resource_after_recheck() -> Result<()> {
    resource_failure_keeps_its_destination_ids(true).await
}

#[tokio::test]
async fn resource_onboard_releases_at_once_after_a_drained_failure() -> Result<()> {
    resource_failure_keeps_its_destination_ids(false).await
}
