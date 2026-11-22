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

use super::super::super::test_support::{
    HangingParallelWorkers, build_leader_with_workers, wait_onboard_driver_finished,
};
use super::*;
use kvbm_common::LogicalResourceId;
use kvbm_logical::BlockManager;

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
fn matched_search(onboard: &Onboard) -> Result<(FindBlocksHandle, BlockId)> {
    let tokens = crate::testing::token_blocks::create_token_sequence(1, BS, 7);
    let block = onboard
        .g2
        .allocate_blocks(1)
        .ok_or_else(|| anyhow::anyhow!("no free G2 block"))?
        .pop()
        .expect("one block");
    let block = block
        .complete(&tokens.blocks()[0])
        .map_err(|error| anyhow::anyhow!("complete G2 block: {error:?}"))?;
    let source = onboard.g2.register_block(block);
    let source_id = source.block_id();
    let ready = FindMatchesResult::Ready(ReadyResult::new(
        vec![source],
        MatchBreakdown {
            host_blocks: 1,
            ..MatchBreakdown::default()
        },
    ));

    let search_id = SearchId::new();
    onboard.engine.searches.insert(
        search_id,
        SearchState {
            request_id: REQUEST.into(),
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
    let owner: Arc<dyn LeaderEngine> = onboard.engine.clone();
    let handle = FindBlocksHandle::search(REQUEST.into(), search_id, Arc::downgrade(&owner));
    Ok((handle, source_id))
}

/// Take every block the G2 pool can give out now and return their ids. A
/// block the onboard still holds is not among them; a released source is,
/// because allocation reuses inactive blocks.
fn reusable_g2_ids(g2: &BlockManager<crate::G2>) -> Vec<BlockId> {
    let available = g2.available_blocks();
    g2.allocate_blocks(available)
        .map(|blocks| blocks.iter().map(|block| block.block_id()).collect())
        .unwrap_or_default()
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
#[ignore = "a local onboard does not yet keep its resources when a lost worker reply leaves the copy unproven"]
async fn onboard_blocks_keeps_sources_and_destinations_after_a_lost_reply() -> Result<()> {
    let fixture = build().await?;
    let (search, source_id) = matched_search(&fixture)?;
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
#[ignore = "a local onboard does not yet keep its resources when a lost worker reply leaves the copy unproven"]
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
    let (search, source_id) = matched_search(&fixture)?;
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
