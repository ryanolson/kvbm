// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use super::super::super::test_support::{
    HangingParallelWorkers, build_leader_with_workers, wait_onboard_driver_finished,
};
use super::onboard_drain::matched_search;
use super::*;
use kvbm_common::LogicalResourceId;
use tokio::sync::oneshot;

const REQUEST: &str = "rq";
const DESTINATION: BlockId = 50;
const TEST_WAIT: Duration = Duration::from_secs(5);

struct HeldLoad {
    engine: Arc<LocalConnectorEngine>,
    sink: Arc<RecordingSink>,
    worker: Arc<HangingParallelWorkers>,
    onboard: OnboardHandle,
    fence: FenceHandle,
    request: RequestId,
    _search: FindBlocksHandle,
}

impl HeldLoad {
    async fn start(prior_offload: bool) -> Result<Self> {
        let sink = RecordingSink::new();
        let (load, ()) =
            Self::start_with_sink(prior_offload, sink.clone(), sink, |_, _| ()).await?;
        Ok(load)
    }

    async fn start_with_sink<T>(
        prior_offload: bool,
        sink: Arc<RecordingSink>,
        worker_sink: Arc<dyn EngineWorkerSink>,
        before_onboard: impl FnOnce(&LocalConnectorEngine, &RequestId) -> T,
    ) -> Result<(Self, T)> {
        let resource = LogicalResourceId(0);
        let worker = HangingParallelWorkers::new(resource);
        let (leader, g2) = build_leader_with_workers(resource, 2, BS, worker.clone()).await?;
        let engine = LocalConnectorEngine::new(Arc::new(leader), worker_sink, BS, false);
        let request = RequestId::from(REQUEST);
        let prior = before_onboard(&engine, &request);
        let (search, _) = matched_search(&engine, &g2, &request)?;
        let mut onboard = engine.clone().onboard_blocks(&search, &[DESTINATION], BS)?;
        worker.wait_started().await;
        let fence = onboard
            .take_physical_drain_fence()
            .expect("local physical fence");
        if prior_offload {
            // An earlier save left a registration. The held copy is the only
            // action still in flight.
            engine.offload_drains.insert(request.clone(), ());
        }
        Ok((
            Self {
                engine,
                sink,
                worker,
                onboard,
                fence,
                request,
                _search: search,
            },
            prior,
        ))
    }

    fn gate_after_receive_claim(&self) -> (oneshot::Receiver<()>, std::sync::mpsc::Sender<()>) {
        self.gate_settlement(|record| &mut record.on_physical_load_claimed)
    }

    fn gate_after_fence_publish(&self) -> (oneshot::Receiver<()>, std::sync::mpsc::Sender<()>) {
        self.gate_settlement(|record| &mut record.on_physical_load_published)
    }

    fn gate_settlement(
        &self,
        observer: impl FnOnce(&mut ActionRecord) -> &mut Option<Box<dyn FnOnce() + Send + Sync>>,
    ) -> (oneshot::Receiver<()>, std::sync::mpsc::Sender<()>) {
        let (entered, wait) = oneshot::channel();
        let (resume, gate) = std::sync::mpsc::channel();
        let gate = Mutex::new(gate);
        let mut record = self
            .engine
            .actions
            .get_mut(self.onboard.id())
            .expect("the copy action is still tracked");
        *observer(&mut record) = Some(Box::new(move || {
            assert!(entered.send(()).is_ok());
            gate.into_inner()
                .expect("test gate")
                .recv_timeout(TEST_WAIT)
                .expect("test resumes the physical terminal");
        }));
        (wait, resume)
    }

    fn assert_before_proof(&self) {
        assert!(self.worker.copy_running());
        assert!(!self.fence.is_complete());
        assert!(self.sink.loads().is_empty());
        assert!(self.sink.saves().is_empty());
    }

    fn assert_single_send(&self) {
        assert!(self.fence.is_complete());
        assert_eq!(
            (self.sink.loads(), self.sink.saves()),
            (vec![], vec![(self.request.clone(), SaveOutcome::Done)]),
            "request finish must produce one finished_sending and no finished_recving"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_release_accounts_for_the_published_fence_before_receive_delivery() -> Result<()> {
    let load = HeldLoad::start(false).await?;
    let (published, resume) = load.gate_after_fence_publish();
    load.assert_before_proof();
    load.worker.release()?;
    tokio::time::timeout(TEST_WAIT, published).await??;
    assert!(load.fence.is_complete());

    let needs_drain = load.engine.begin_request_finish(&load.request);
    let before_delivery = (load.sink.loads(), load.sink.saves());
    resume.send(())?;
    wait_onboard_driver_finished(*load.onboard.id()).await;
    let after_delivery = (load.sink.loads(), load.sink.saves());

    assert_eq!(before_delivery, (vec![], vec![]));
    assert!(
        needs_drain,
        "finish reported no unsettled work before the receive was delivered: {after_delivery:?}"
    );
    load.assert_single_send();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_release_orders_finish_after_the_receive_claim() -> Result<()> {
    let load = HeldLoad::start(true).await?;
    let (claimed, resume) = load.gate_after_receive_claim();
    let (published, deliver) = load.gate_after_fence_publish();
    load.assert_before_proof();
    load.worker.release()?;
    tokio::time::timeout(TEST_WAIT, claimed).await??;
    assert!(load.onboard.is_complete());
    assert!(!load.fence.is_complete());
    assert!(
        !load
            .engine
            .actions
            .get(load.onboard.id())
            .unwrap()
            .physical_pending
    );

    let engine = load.engine.clone();
    let request = load.request.clone();
    let (starting, started) = oneshot::channel();
    let mut finish = tokio::task::spawn_blocking(move || {
        assert!(starting.send(()).is_ok());
        engine.begin_request_finish(&request)
    });
    tokio::time::timeout(TEST_WAIT, started).await??;
    let returned_while_held = tokio::time::timeout(Duration::from_millis(200), &mut finish)
        .await
        .ok();
    let returned_before_publish = returned_while_held.is_some();
    let before_publish = (load.sink.loads(), load.sink.saves());
    resume.send(())?;
    tokio::time::timeout(TEST_WAIT, published).await??;
    let needs_drain = match returned_while_held {
        Some(result) => result?,
        None => tokio::time::timeout(TEST_WAIT, finish).await??,
    };
    let before_delivery = (load.sink.loads(), load.sink.saves());
    deliver.send(())?;
    wait_onboard_driver_finished(*load.onboard.id()).await;

    let after_publish = (load.sink.loads(), load.sink.saves());
    assert_eq!(
        (
            returned_before_publish,
            needs_drain,
            before_publish,
            before_delivery,
            after_publish
        ),
        (
            false,
            true,
            (vec![], vec![]),
            (vec![], vec![]),
            (vec![], vec![(load.request.clone(), SaveOutcome::Done)]),
        ),
        "finish waits for publication, then takes the undelivered receive into its drain"
    );
    assert!(load.fence.is_complete());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_release_folds_receive_into_finish_before_proof() -> Result<()> {
    let load = HeldLoad::start(true).await?;
    load.assert_before_proof();
    assert!(load.engine.begin_request_finish(&load.request));
    load.assert_before_proof();
    load.worker.release()?;
    wait_onboard_driver_finished(*load.onboard.id()).await;
    load.assert_single_send();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_release_sends_once_for_an_evicted_load_without_offloads() -> Result<()> {
    let load = HeldLoad::start(false).await?;
    let eviction = load.engine.evict(&load.request);
    let eviction_fence = eviction.handle.expect("eviction arms its own fence");
    assert!(!eviction_fence.is_complete());
    let needs_drain = load.engine.begin_request_finish(&load.request);
    load.assert_before_proof();
    assert!(load.sink.fences().is_empty());
    load.worker.release()?;
    wait_onboard_driver_finished(*load.onboard.id()).await;
    assert!(eviction_fence.is_complete());
    assert_eq!(load.sink.fences(), eviction.fence.per_worker);
    load.assert_single_send();
    assert!(needs_drain, "a load alone requires the request drain");
    Ok(())
}

struct HeldReceiveSink {
    sink: Arc<RecordingSink>,
    entered: Mutex<Option<oneshot::Sender<()>>>,
    resume: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl EngineWorkerSink for HeldReceiveSink {
    fn mark_load_finished(&self, request: &RequestId, outcome: LoadOutcome) {
        assert!(
            self.entered
                .lock()
                .unwrap()
                .take()
                .unwrap()
                .send(())
                .is_ok()
        );
        self.resume
            .lock()
            .unwrap()
            .recv_timeout(TEST_WAIT)
            .expect("test resumes sink delivery");
        self.sink.mark_load_finished(request, outcome);
    }

    fn mark_save_finished(&self, request: &RequestId, outcome: SaveOutcome) {
        self.sink.mark_save_finished(request, outcome);
    }

    fn mark_fence_complete(&self, token: FenceToken) {
        self.sink.mark_fence_complete(token);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_release_joins_an_in_progress_receive_and_keeps_the_old_eviction_fence()
-> Result<()> {
    let sink = RecordingSink::new();
    let (enter, entered) = oneshot::channel();
    let (resume, blocked) = std::sync::mpsc::channel();
    let gated = Arc::new(HeldReceiveSink {
        sink: sink.clone(),
        entered: Mutex::new(Some(enter)),
        resume: Mutex::new(blocked),
    });
    let (load, (old_save, eviction)) =
        HeldLoad::start_with_sink(true, sink, gated, |engine, request| {
            // The prior generation's save has its own eviction fence. Drop the
            // visible handle before restoring a new load for the same request.
            let save = ActionId::new();
            engine
                .actions
                .insert(save, ActionRecord::new_save(request.clone(), Weak::new()));
            engine.by_request.insert(request.clone(), vec![save]);
            let eviction = engine.evict(request);
            engine.release_action(&save);
            (save, eviction)
        })
        .await?;
    let old_fence = eviction.handle.expect("the old save has a fence");
    load.assert_before_proof();
    load.worker.release()?;
    tokio::time::timeout(TEST_WAIT, entered).await??;
    assert!(load.fence.is_complete());
    let join = load.engine.begin_request_finish(&load.request);
    let before = (load.sink.loads(), load.sink.saves());
    let action = *load.onboard.id();
    drop(load.onboard);
    let retained = load.engine.actions.contains_key(&action);
    resume.send(())?;
    wait_onboard_driver_finished(action).await;
    assert!(join);
    assert!(retained, "an active delivery must outlive the handle");
    assert!(!load.engine.actions.contains_key(&action));
    assert_eq!(before, (vec![], vec![]));
    assert_eq!(
        load.sink.loads(),
        vec![(load.request.clone(), LoadOutcome::Done)]
    );
    assert!(load.sink.saves().is_empty());
    assert!(!old_fence.is_complete());
    assert!(load.sink.fences().is_empty());

    load.engine
        .finish_save_action(old_save, &load.request, ActionStatus::Complete);
    assert!(old_fence.is_complete());
    assert_eq!(load.sink.fences(), eviction.fence.per_worker);
    assert!(load.sink.saves().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_release_reports_ready_after_receive_delivery() -> Result<()> {
    let load = HeldLoad::start(true).await?;
    load.assert_before_proof();
    load.worker.release()?;
    wait_onboard_driver_finished(*load.onboard.id()).await;
    assert!(load.fence.is_complete());
    let received = vec![(load.request.clone(), LoadOutcome::Done)];
    assert_eq!(load.sink.loads(), received);

    assert!(!load.engine.begin_request_finish(&load.request));
    assert!(load.engine.take_offload_drain(&load.request).is_none());
    assert_eq!(load.sink.loads(), received);
    assert!(load.sink.saves().is_empty());
    Ok(())
}
