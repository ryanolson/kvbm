// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Hold each offload transfer until the test selects a terminal state.
struct PendingOffloadSubmit {
    submits: StdMutex<usize>,
    status: watch::Sender<TransferStatus>,
}

impl PendingOffloadSubmit {
    fn new() -> Arc<Self> {
        let (status, _initial) = watch::channel(TransferStatus::Queued);
        Arc::new(Self {
            submits: StdMutex::new(0),
            status,
        })
    }

    fn submit_count(&self) -> usize {
        *self.submits.lock().unwrap()
    }

    fn finish(&self, status: TransferStatus) {
        self.status.send_replace(status);
    }
}

impl OffloadSubmit for PendingOffloadSubmit {
    fn supports_resource(&self, _resource: kvbm_common::LogicalResourceId) -> bool {
        true
    }

    fn submit_g1_to_g2(
        &self,
        _resource: Option<kvbm_common::LogicalResourceId>,
        _blocks: Vec<ExternalBlock<crate::G1>>,
        _precondition: Option<velo::EventHandle>,
    ) -> Result<Box<dyn OffloadTransfer>> {
        *self.submits.lock().unwrap() += 1;
        Ok(Box::new(PendingOffloadTransfer {
            status: self.status.subscribe(),
        }))
    }
}

struct PendingOffloadTransfer {
    status: watch::Receiver<TransferStatus>,
}

impl OffloadTransfer for PendingOffloadTransfer {
    fn status(&self) -> TransferStatus {
        *self.status.borrow()
    }

    fn completed_blocks(&self) -> Vec<BlockId> {
        Vec::new()
    }

    fn failed_blocks(&self) -> Vec<BlockId> {
        Vec::new()
    }

    fn wait_terminal(&self) -> futures::future::BoxFuture<'static, ()> {
        let mut status = self.status.clone();
        Box::pin(async move {
            loop {
                if matches!(
                    *status.borrow_and_update(),
                    TransferStatus::Complete | TransferStatus::Cancelled | TransferStatus::Failed
                ) {
                    return;
                }
                if status.changed().await.is_err() {
                    return;
                }
            }
        })
    }
}

async fn wait_for_save_count(sink: &RecordingSink, expected: usize) {
    for _ in 0..200 {
        if sink.saves().len() == expected {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("save terminal did not reach the expected count");
}

async fn wait_for_fence_count(sink: &RecordingSink, expected: usize) {
    for _ in 0..200 {
        if sink.fences().len() == expected {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("fence terminal did not reach the expected count");
}

async fn wait_for_action_removal(engine: &LocalConnectorEngine, action_id: ActionId) {
    for _ in 0..200 {
        if !engine.actions.contains_key(&action_id) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("action record did not release after its terminal");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn poll_does_not_remove_dropped_buffered_save() -> Result<()> {
    let leader = Arc::new(build_test_leader().await?);
    let sink = RecordingSink::new();
    let submit = PendingOffloadSubmit::new();
    let engine = LocalConnectorEngine::with_offload_submit(
        leader,
        sink.clone(),
        BS,
        true,
        submit.clone(),
        None,
    );

    let request: RequestId = "poll-buffered-drop".into();
    let handle = engine.clone().offload(&request, vec![(h(1), 10usize)])?;
    let action_id = *handle.id();
    drop(handle);

    // The handle is dropped while its record is still live (the flush has not
    // run), so the record's completion cell no longer upgrades. Polling cannot
    // prove that the dead cell has no physical work, so the unknown status
    // fails closed instead of reporting `Complete`.
    assert_eq!(
        engine.poll_action(&action_id),
        ActionStatus::Failed(ActionFailure::AllBlocks)
    );
    assert!(
        engine.actions.contains_key(&action_id),
        "poll must retain pending physical work"
    );
    assert!(
        engine
            .by_request
            .get(&request)
            .is_some_and(|actions| actions.ids.as_slice() == [action_id]),
        "poll must retain the request index"
    );

    engine
        .take_offload_drain(&request)
        .expect("offload registered a drain")
        .commit();
    assert!(sink.saves().is_empty(), "the drain must wait for the save");

    engine.finish_forward_pass(0);
    assert_eq!(submit.submit_count(), 1);
    submit.finish(TransferStatus::Complete);
    wait_for_save_count(&sink, 1).await;
    wait_for_action_removal(&engine, action_id).await;
    assert!(!engine.by_request.contains_key(&request));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn eviction_settles_a_dropped_buffered_save_without_dispatch() -> Result<()> {
    evict_dropped_save(false).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn eviction_keeps_a_dropped_flushed_save_fenced_until_proof() -> Result<()> {
    evict_dropped_save(true).await
}

async fn evict_dropped_save(flushed: bool) -> Result<()> {
    let leader = Arc::new(build_test_leader().await?);
    let sink = RecordingSink::new();
    let submit = PendingOffloadSubmit::new();
    let engine = LocalConnectorEngine::with_offload_submit(
        leader,
        sink.clone(),
        BS,
        true,
        submit.clone(),
        None,
    );

    let request: RequestId = "evict-buffered-drop".into();
    let handle = engine.clone().offload(&request, vec![(h(1), 10usize)])?;
    let action_id = *handle.id();
    if flushed {
        engine.finish_forward_pass(0);
        assert_eq!(submit.submit_count(), 1);
    }
    drop(handle);

    let eviction = engine.evict(&request);
    assert!(!eviction.fence.per_worker.is_empty());
    let fence = eviction.handle.expect("eviction armed a fence");
    if flushed {
        assert!(!fence.is_complete());
        assert!(sink.fences().is_empty());
        submit.finish(TransferStatus::Complete);
    } else {
        assert!(
            fence.is_complete(),
            "eviction can settle an unsubmitted save"
        );
        engine.finish_forward_pass(0);
        assert_eq!(submit.submit_count(), 0);
    }
    wait_for_fence_count(&sink, eviction.fence.per_worker.len()).await;
    assert!(fence.is_complete());
    wait_for_action_removal(&engine, action_id).await;
    assert!(!engine.by_request.contains_key(&request));
    assert!(sink.saves().is_empty());
    Ok(())
}

#[tokio::test]
async fn direct_restore_refuses_until_the_evicted_save_drains() -> Result<()> {
    use crate::tiering::engine::test_support::{HangingParallelWorkers, build_leader_with_workers};

    let resource = LogicalResourceId(0);
    let worker = HangingParallelWorkers::new(resource);
    let (leader, _g2) = build_leader_with_workers(resource, 2, BS, worker.clone()).await?;
    let sink = RecordingSink::new();
    let submit = PendingOffloadSubmit::new();
    let engine = LocalConnectorEngine::with_offload_submit(
        Arc::new(leader),
        sink.clone(),
        BS,
        true,
        submit.clone(),
        None,
    );
    let request = RequestId::from("direct-restore");
    let old = engine.clone().offload(&request, vec![(h(1), 10)])?;
    engine.finish_forward_pass(0);
    assert_eq!(submit.submit_count(), 1);
    let eviction = engine.evict(&request);
    let resources = || {
        vec![ResourceOnboard {
            resource,
            source_block_ids: vec![0],
            destination_block_ids: vec![10],
        }]
    };
    assert!(matches!(
        engine
            .clone()
            .onboard_resource_blocks(&request, resources()),
        Err(LeaderEngineError::PriorGenerationDraining)
    ));
    assert!(!worker.copy_running());
    assert_eq!(engine.actions.len(), 1);
    assert!(!eviction.handle.as_ref().unwrap().is_complete());

    submit.finish(TransferStatus::Complete);
    wait_for_fence_count(&sink, eviction.fence.per_worker.len()).await;
    let restored = engine
        .clone()
        .onboard_resource_blocks(&request, resources())?;
    worker.wait_started().await;
    assert!(matches!(
        engine.clone().offload(&request, vec![(h(1), 10)]),
        Err(LeaderEngineError::OnboardAlreadyInFlight)
    ));
    worker.release()?;
    crate::tiering::engine::test_support::wait_onboard_driver_finished(*restored.id()).await;
    assert_eq!(restored.outcome(), Some(LoadOutcome::Done));
    drop((old, restored));
    assert!(!engine.by_request.contains_key(&request));
    Ok(())
}

#[tokio::test]
async fn evict_invalidates_a_search_offer_without_pending_copies() -> Result<()> {
    use crate::tiering::engine::test_support::{HangingParallelWorkers, build_leader_with_workers};
    use kvbm_protocols::connector::CacheScope;

    let resource = LogicalResourceId(0);
    let worker = HangingParallelWorkers::new(resource);
    let (leader, g2) = build_leader_with_workers(resource, 2, BS, worker).await?;
    let mut sources = g2.allocate_blocks(1).unwrap();
    let staged = sources.pop().unwrap().stage(h(1), BS)?;
    let _source = g2.register_block(staged);
    let engine = LocalConnectorEngine::new(Arc::new(leader), RecordingSink::new(), BS, true);
    let request = FindBlocksRequest {
        request_id: "stale-restore-offer".into(),
        cache: CacheScope::LegacyPrimary,
        sequence_hashes: Arc::from([h(1)]),
        num_computed_tokens: 0,
        total_tokens: BS + 1,
        transfer_params: None,
        local_prefill_estimate: None,
    };
    let FindBlocksOutcome::Resolved {
        minted: Some(offer),
        ..
    } = engine.clone().find_blocks(&request, None)?
    else {
        panic!("the first match must mint an offer");
    };
    assert!(engine.evict(&request.request_id).handle.is_none());
    assert!(matches!(
        engine.clone().onboard_blocks(&offer, &[10], BS),
        Err(LeaderEngineError::SearchNotMatched)
    ));
    assert!(engine.actions.is_empty());
    drop(offer);
    assert!(engine.admission_offers.is_empty());
    assert!(!engine.by_request.contains_key(&request.request_id));
    assert!(matches!(
        engine.clone().find_blocks(&request, None)?,
        FindBlocksOutcome::Resolved {
            minted: Some(_),
            ..
        }
    ));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_running_save_holds_request_drain_until_terminal() -> Result<()> {
    let leader = Arc::new(build_test_leader().await?);
    let sink = RecordingSink::new();
    let submit = PendingOffloadSubmit::new();
    let engine = LocalConnectorEngine::with_offload_submit(
        leader,
        sink.clone(),
        BS,
        true,
        submit.clone(),
        None,
    );

    let request: RequestId = "running-drop".into();
    let handle = engine.clone().offload(&request, vec![(h(1), 10usize)])?;
    let action_id = *handle.id();
    engine.finish_forward_pass(0);
    assert_eq!(submit.submit_count(), 1);

    drop(handle);
    engine
        .take_offload_drain(&request)
        .expect("offload registered a drain")
        .commit();
    assert!(sink.saves().is_empty());

    submit.finish(TransferStatus::Complete);
    wait_for_save_count(&sink, 1).await;
    assert_eq!(sink.saves(), vec![(request.clone(), SaveOutcome::Done)]);
    wait_for_action_removal(&engine, action_id).await;
    assert!(!engine.by_request.contains_key(&request));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn duplicate_save_terminal_keeps_first_outcome_and_releases_barriers_once() -> Result<()> {
    let leader = Arc::new(build_test_leader().await?);
    let sink = RecordingSink::new();
    let engine = LocalConnectorEngine::new(leader, sink.clone(), BS, true);

    let request: RequestId = "duplicate-save".into();
    let action_id = ActionId::new();
    let cell = Arc::new(Mutex::new(ActionStatus::Pending));
    engine.actions.insert(
        action_id,
        ActionRecord::new_save(request.clone(), Arc::downgrade(&cell)),
    );
    engine
        .by_request
        .insert(request.clone(), vec![action_id].into());
    engine.offload_drains.insert(request.clone(), ());

    let fence = engine.evict(&request).fence;
    engine
        .take_offload_drain(&request)
        .expect("offload registered a drain")
        .commit();

    engine.finish_save_action(action_id, &request, ActionStatus::Complete);
    assert_eq!(*cell.lock().expect("cell"), ActionStatus::Complete);
    assert_eq!(sink.saves(), vec![(request.clone(), SaveOutcome::Done)]);
    assert_eq!(sink.fences().len(), fence.per_worker.len());

    engine.finish_save_action(
        action_id,
        &request,
        ActionStatus::Failed(ActionFailure::AllBlocks),
    );
    assert_eq!(*cell.lock().expect("cell"), ActionStatus::Complete);
    assert_eq!(sink.saves(), vec![(request.clone(), SaveOutcome::Done)]);
    assert_eq!(sink.fences().len(), fence.per_worker.len());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_settles_dropped_buffered_offload() -> Result<()> {
    let leader = Arc::new(build_test_leader().await?);
    let sink = RecordingSink::new();
    let engine = LocalConnectorEngine::with_offload_submit(
        leader,
        sink.clone(),
        BS,
        true,
        PendingOffloadSubmit::new(),
        None,
    );

    let request: RequestId = "shutdown-buffered".into();
    let handle = engine.clone().offload(&request, vec![(h(1), 10usize)])?;
    let action_id = *handle.id();
    drop(handle);
    engine
        .take_offload_drain(&request)
        .expect("offload registered a drain")
        .commit();
    assert!(sink.saves().is_empty());

    engine.shutdown();

    assert_eq!(sink.saves(), vec![(request.clone(), SaveOutcome::Done)]);
    assert!(!engine.actions.contains_key(&action_id));
    assert!(!engine.by_request.contains_key(&request));
    assert!(engine.offload_buffer.lock().unwrap().is_empty());
    Ok(())
}
