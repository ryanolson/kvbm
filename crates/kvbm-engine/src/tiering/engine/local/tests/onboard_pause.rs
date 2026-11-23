// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{Read, Seek};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::anyhow;
use bytes::Bytes;
use futures::future::BoxFuture;
use kvbm_logical::BlockManager;
use kvbm_physical::transfer::{TransferCompleteNotification, TransferDrainOutcome};
use kvbm_protocols::connector::FenceHandle;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Instant, advance, timeout};
use velo::{Handler, Messenger};

use super::super::super::paused::PROBE_INTERVAL;
use super::super::super::test_support::{
    build_leader_with_messenger, wait_onboard_driver_finished,
};
use super::onboard_drain::{matched_search, reusable_g2_ids};
use super::*;
use crate::testing::messenger::create_messenger_tcp;
use crate::worker::group::SpmdParallelWorkers;
use crate::worker::velo::{
    HeldLocalTransfers, LocalTransferCheckTiming, VeloWorkerClient, handler_names,
    serve_held_local_transfers,
};

const REQUEST: &str = "paused-onboard";
const DEST: BlockId = 50;
const WAIT: Duration = Duration::from_secs(2);
const DEADLINE: Duration = Duration::from_secs(5);
const WALL_LIMIT: Duration = Duration::from_secs(5);
const SAMPLE_COUNT: &str = "kvbm_paused_onboards";
const SAMPLE_START: &str = "kvbm_onboard_oldest_pause_timestamp_seconds";
const SAMPLE_ATTEMPT: &str = "kvbm_onboard_last_probe_attempt_timestamp_seconds";
const SAMPLE_ANSWER: &str = "kvbm_onboard_last_probe_answer_timestamp_seconds";
const SAMPLE_CHECKS: &str = "kvbm_onboard_probe_checks_total";
const SAMPLE_STUCK: &str = "kvbm_onboard_probe_stuck_total";
const SILENT: &str = "worker_not_answering";
const NO_PROOF: &str = "worker_no_proof";
const NO_CAPABILITY: &str = "no_recovery_capability";
const IMMEDIATE: LocalTransferCheckTiming = LocalTransferCheckTiming {
    wait: Duration::ZERO,
    deadline: DEADLINE,
};

struct Fixture {
    engine: Arc<LocalConnectorEngine>,
    sink: Arc<RecordingSink>,
    g2: Arc<BlockManager<G2>>,
    registry: prometheus::Registry,
    relay: Arc<Messenger>,
    server: Arc<Messenger>,
}

impl Fixture {
    async fn new(timing: LocalTransferCheckTiming, workers: u64) -> Result<Self> {
        let messenger = timeout(WALL_LIMIT, create_messenger_tcp()).await??;
        let relay = timeout(WALL_LIMIT, create_messenger_tcp()).await??;
        let server = timeout(WALL_LIMIT, create_messenger_tcp()).await??;
        link(&messenger, &relay)?;
        link(&relay, &server)?;
        let clients = (0..workers)
            .map(|id| {
                Arc::new(
                    VeloWorkerClient::new_for_worker(messenger.clone(), relay.instance_id(), id)
                        .with_local_transfer_check(timing),
                ) as Arc<dyn crate::worker::Worker>
            })
            .collect();
        let workers = Arc::new(SpmdParallelWorkers::new(
            clients,
            Arc::new(velo::EventManager::local()),
            messenger.runtime().clone(),
        ));
        let (leader, g2) =
            build_leader_with_messenger(messenger, LogicalResourceId(0), 2, BS, workers)?;
        let registry = leader.observability().unwrap().registry().clone();
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::new(Arc::new(leader), sink.clone(), BS, true);
        Ok(Self {
            engine,
            sink,
            g2,
            registry,
            relay,
            server,
        })
    }

    fn worker(&self, id: u64, bind_incarnation: bool) -> Result<TestWorker> {
        let (started, copies) = mpsc::unbounded_channel();
        let transfers = serve_held_local_transfers(
            &self.server,
            id,
            false,
            move |resource, sources, destinations| {
                assert_eq!(resource, LogicalResourceId(0));
                let (finish, receipt) = oneshot::channel();
                started.send(HeldCopy {
                    sources,
                    destinations,
                    finish,
                })?;
                Ok(TransferCompleteNotification::from_drain_future(
                    async move {
                        receipt
                            .await
                            .unwrap_or_else(|error| TransferDrainOutcome::Unproven(error.into()))
                    },
                ))
            },
        )?;
        let worker = TestWorker {
            id,
            relay: self.relay.clone(),
            server: self.server.clone(),
            transfers,
            copies,
            probe_reply: Arc::new(Mutex::new(ProbeReply::Forward)),
            checks: Arc::new(AtomicUsize::new(0)),
        };
        worker.forward(handler_names::LOCAL_TRANSFER)?;
        if bind_incarnation {
            worker.forward(handler_names::WORKER_INCARNATION)?;
        }
        Ok(worker)
    }

    fn onboard(&self) -> Result<Onboard> {
        let (search, source) = matched_search(&self.engine, &self.g2, REQUEST)?;
        let mut handle = self.engine.clone().onboard_blocks(&search, &[DEST], BS)?;
        let fence = handle
            .take_physical_drain_fence()
            .expect("local drain fence");
        Ok(Onboard {
            handle,
            fence,
            source,
        })
    }

    fn assert_held(&self, onboard: &Onboard) {
        assert_eq!(onboard.handle.outcome(), None);
        assert!(!onboard.fence.is_complete());
        assert!(self.sink.loads().is_empty());
        assert_eq!(self.g2.available_blocks(), 1);
        assert!(!reusable_g2_ids(&self.g2).contains(&onboard.source));
    }

    fn assert_paused(&self, onboard: &Onboard, worker: &TestWorker, state: &str) {
        self.assert_held(onboard);
        assert_eq!(onboard.handle.paused().unwrap().worker(), worker.label());
        assert_eq!(worker.metric(self, SAMPLE_COUNT, &[state]), Some(1.0));
    }

    async fn assert_settled(&self, onboard: &Onboard, outcome: LoadOutcome) {
        eventually("scheduler release", || !self.sink.loads().is_empty()).await;
        assert_eq!(onboard.handle.outcome(), Some(outcome.clone()));
        assert!(onboard.handle.paused().is_none());
        assert!(onboard.fence.is_complete());
        assert_eq!(self.sink.loads(), vec![(REQUEST.into(), outcome)]);
        assert_eq!(self.g2.available_blocks(), 2);
        assert!(reusable_g2_ids(&self.g2).contains(&onboard.source));
    }
}

fn link(a: &Messenger, b: &Messenger) -> Result<()> {
    a.register_peer(b.peer_info())?;
    b.register_peer(a.peer_info())?;
    Ok(())
}

struct Onboard {
    handle: OnboardHandle,
    fence: FenceHandle,
    source: BlockId,
}

impl Onboard {
    async fn wait_for_pause(&self) {
        eventually("production onboard pause", || {
            self.handle.paused().is_some() || self.handle.is_complete()
        })
        .await;
        assert!(
            self.handle.paused().is_some(),
            "lost RPC must park the onboard"
        );
        wait_onboard_driver_finished(*self.handle.id()).await;
    }
}

struct HeldCopy {
    sources: Vec<BlockId>,
    destinations: Vec<BlockId>,
    finish: oneshot::Sender<TransferDrainOutcome>,
}

#[derive(Clone, Copy)]
enum ProbeReply {
    Forward,
    Silent,
    Delayed(Duration),
    WrongIncarnation,
    Error,
}

struct TestWorker {
    id: u64,
    relay: Arc<Messenger>,
    server: Arc<Messenger>,
    transfers: HeldLocalTransfers,
    copies: mpsc::UnboundedReceiver<HeldCopy>,
    probe_reply: Arc<Mutex<ProbeReply>>,
    checks: Arc<AtomicUsize>,
}

impl TestWorker {
    fn label(&self) -> String {
        format!("{}#{}", self.relay.instance_id(), self.id)
    }

    fn forward(&self, name: &str) -> Result<()> {
        let name = handler_names::scoped(name, Some(self.id));
        let route = self.route(name.clone());
        self.relay.register_handler(
            Handler::unary_handler_async(&name, move |ctx| route(ctx.payload)).build(),
        )
    }

    // The relay can withhold a check or advertise no handler. Forwarded
    // checks still read the original worker's record for the transfer ID.
    fn register_check(&self) -> Result<()> {
        let name = self.check_name();
        let route = self.route(name.clone());
        let reply = self.probe_reply.clone();
        let checks = self.checks.clone();
        self.relay.register_handler(
            Handler::unary_handler_async(&name, move |ctx| {
                let reply = *reply.lock().unwrap();
                let delay_until = match reply {
                    ProbeReply::Delayed(delay) => Some(Instant::now() + delay),
                    _ => None,
                };
                checks.fetch_add(1, Ordering::Relaxed);
                let forward = route(ctx.payload);
                async move {
                    match reply {
                        ProbeReply::Silent => return std::future::pending().await,
                        ProbeReply::Error => return Err(anyhow!("injected check RPC error")),
                        ProbeReply::WrongIncarnation => {
                            return Ok(Some(
                                serde_json::to_vec(&serde_json::json!({
                                    "incarnation": Uuid::new_v4(),
                                    "record": "Completed",
                                }))?
                                .into(),
                            ));
                        }
                        _ => {}
                    }
                    if let Some(until) = delay_until {
                        tokio::time::sleep_until(until).await;
                    }
                    forward.await
                }
            })
            .build(),
        )
    }

    fn route(
        &self,
        name: String,
    ) -> impl Fn(Bytes) -> BoxFuture<'static, Result<Option<Bytes>>> + use<> {
        let relay = Arc::downgrade(&self.relay);
        let server = self.server.instance_id();
        move |payload| {
            let relay = relay.clone();
            let name = name.clone();
            Box::pin(async move {
                let relay = relay.upgrade().expect("the relay is serving the request");
                let bytes = relay
                    .unary(&name)?
                    .raw_payload(payload)
                    .instance(server)
                    .send()
                    .await?;
                Ok(Some(bytes))
            })
        }
    }

    fn check_name(&self) -> String {
        handler_names::scoped(handler_names::LOCAL_TRANSFER_CHECK, Some(self.id))
    }

    fn reply(&self, reply: ProbeReply) {
        *self.probe_reply.lock().unwrap() = reply;
    }

    async fn wait_for_check(&self, count: usize) {
        eventually("check reaches the named worker", || {
            self.checks.load(Ordering::Relaxed) >= count
        })
        .await;
        assert_eq!(self.checks.load(Ordering::Relaxed), count);
    }

    fn lose_next_reply(&self) -> oneshot::Sender<()> {
        self.transfers.gate_reply_loss()
    }

    async fn running_copy(&mut self, onboard: &Onboard) -> HeldCopy {
        let copy = timeout(WALL_LIMIT, self.copies.recv())
            .await
            .expect("worker starts the copy")
            .expect("worker keeps the dispatch channel open");
        assert_eq!(copy.sources, vec![onboard.source]);
        assert_eq!(copy.destinations, vec![DEST]);
        assert!(self.transfers.active());
        copy
    }

    async fn finish_copy(&self, copy: HeldCopy, outcome: TransferDrainOutcome) {
        assert!(
            copy.finish.send(outcome).is_ok(),
            "worker holds the running copy"
        );
        eventually("worker records the drain", || !self.transfers.active()).await;
    }

    fn metric(&self, fixture: &Fixture, family: &str, labels: &[&str]) -> Option<f64> {
        let label = self.label();
        let labels = [&[label.as_str()][..], labels].concat();
        metric(&fixture.registry, family, &labels)
    }
}

// Wait for dispatch before opening the Velo error gate. The copy keeps
// running independently of that reply.
async fn start_held(
    fixture: &Fixture,
    worker: &mut TestWorker,
) -> Result<(Onboard, HeldCopy, oneshot::Sender<()>)> {
    let lose_reply = worker.lose_next_reply();
    let onboard = fixture.onboard()?;
    let copy = worker.running_copy(&onboard).await;
    fixture.assert_held(&onboard);
    Ok((onboard, copy, lose_reply))
}

async fn lose_reply_and_pause(
    fixture: &Fixture,
    worker: &mut TestWorker,
) -> Result<(Onboard, HeldCopy)> {
    let (onboard, copy, lose_reply) = start_held(fixture, worker).await?;
    lose_reply.send(()).expect("the transfer RPC is waiting");
    onboard.wait_for_pause().await;
    assert!(worker.transfers.active());
    tokio::time::pause();
    Ok((onboard, copy))
}

fn metric(registry: &prometheus::Registry, family: &str, labels: &[&str]) -> Option<f64> {
    registry
        .gather()
        .into_iter()
        .find(|metric| metric.name() == family)?
        .get_metric()
        .iter()
        .find(|metric| {
            labels.iter().all(|value| {
                metric
                    .get_label()
                    .iter()
                    .any(|label| label.value() == *value)
            })
        })
        .map(|metric| {
            if family.ends_with("_total") {
                metric.get_counter().value()
            } else {
                metric.get_gauge().value()
            }
        })
}

pub(super) async fn tick(duration: Duration) {
    // Tokio's timer wheel rounds deadlines up to the next millisecond.
    advance(duration + Duration::from_millis(1)).await;
    tokio::task::yield_now().await;
}

pub(super) async fn eventually(reason: &str, mut check: impl FnMut() -> bool) {
    let limit = std::time::Instant::now() + WALL_LIMIT;
    while !check() {
        // Keep the fake clock fixed while loopback I/O wakes a task.
        assert!(
            std::time::Instant::now() < limit,
            "timed out waiting for {reason}"
        );
        tokio::task::yield_now().await;
    }
}

struct Logs {
    file: File,
    _capture: tracing::subscriber::DefaultGuard,
    _quiet: tracing::Dispatch,
}

impl Logs {
    fn capture() -> Result<Self> {
        let file = tempfile::tempfile()?;
        // A second dispatch keeps scoped capture interest visible to parallel tests.
        let quiet = tracing::Dispatch::new(
            tracing_subscriber::fmt()
                .with_max_level(tracing::level_filters::LevelFilter::OFF)
                .finish(),
        );
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_writer(file.try_clone()?)
            .finish();
        Ok(Self {
            file,
            _capture: tracing::subscriber::set_default(subscriber),
            _quiet: quiet,
        })
    }

    fn read(&mut self) -> String {
        self.file.rewind().unwrap();
        let mut text = String::new();
        self.file.read_to_string(&mut text).unwrap();
        text
    }

    async fn wait_for_loss(&mut self) {
        eventually("the Velo client sees the lost RPC reply", || {
            self.read()
                .contains("local transfer reply lost; checking with the worker")
        })
        .await;
    }

    fn assert_event(&mut self, message: &str, worker: &TestWorker, state: &str) {
        let log = self.read();
        assert!(
            log.lines().any(|line| line.contains(message)
                && line.contains(&worker.label())
                && line.contains(state)),
            "{log}"
        );
    }
}

#[tokio::test]
async fn paused_onboard_retains_holds_by_the_initial_check_deadline() -> Result<()> {
    let mut logs = Logs::capture()?;
    let fixture = Fixture::new(LocalTransferCheckTiming::default(), 1).await?;
    let mut worker = fixture.worker(0, true)?;
    worker.register_check()?;
    worker.reply(ProbeReply::Silent);
    let (onboard, copy, lose_reply) = start_held(&fixture, &mut worker).await?;

    tokio::time::pause();
    let lost_at = Instant::now();
    lose_reply.send(()).unwrap();
    logs.wait_for_loss().await;
    let before_deadline = Duration::from_millis(10);
    tick(WAIT - before_deadline).await;
    assert_eq!(worker.checks.load(Ordering::Relaxed), 0);
    tick(before_deadline).await;
    worker.wait_for_check(1).await;
    tick(DEADLINE - before_deadline).await;
    fixture.assert_held(&onboard);
    assert!(onboard.handle.paused().is_none());
    assert_eq!(worker.metric(&fixture, SAMPLE_COUNT, &[SILENT]), None);

    tick(before_deadline).await;
    onboard.wait_for_pause().await;
    assert!(lost_at.elapsed() <= WAIT + DEADLINE + Duration::from_millis(4));
    fixture.assert_paused(&onboard, &worker, SILENT);
    assert!(worker.transfers.active());
    let since = onboard.handle.paused().unwrap().since();
    let start = worker.metric(&fixture, SAMPLE_START, &[SILENT]).unwrap();
    let attempt = worker.metric(&fixture, SAMPLE_ATTEMPT, &[]).unwrap();
    assert!((start - attempt - DEADLINE.as_secs_f64()).abs() < 0.01);
    assert_eq!(worker.metric(&fixture, SAMPLE_ANSWER, &[]), None);
    assert_eq!(Instant::now().into_std(), since);
    tick(Duration::from_secs(1)).await;
    assert_eq!(
        worker.metric(&fixture, SAMPLE_START, &[SILENT]),
        Some(start)
    );
    assert_eq!(
        Instant::now().into_std() - since,
        Duration::from_millis(1001)
    );
    logs.assert_event("onboard paused", &worker, SILENT);

    worker.reply(ProbeReply::Forward);
    worker
        .finish_copy(copy, TransferDrainOutcome::Completed)
        .await;
    tick(PROBE_INTERVAL).await;
    fixture.assert_settled(&onboard, LoadOutcome::Done).await;
    Ok(())
}

#[tokio::test]
async fn paused_onboard_resumes_after_arrival_proof() -> Result<()> {
    let mut logs = Logs::capture()?;
    let fixture = Fixture::new(IMMEDIATE, 1).await?;
    let mut worker = fixture.worker(0, true)?;
    worker.register_check()?;
    let (onboard, copy) = lose_reply_and_pause(&fixture, &mut worker).await?;
    fixture.assert_paused(&onboard, &worker, NO_PROOF);

    worker
        .finish_copy(copy, TransferDrainOutcome::Completed)
        .await;
    tick(PROBE_INTERVAL).await;
    fixture.assert_settled(&onboard, LoadOutcome::Done).await;
    assert_eq!(
        worker.metric(&fixture, SAMPLE_COUNT, &[NO_PROOF]),
        Some(0.0)
    );
    assert_eq!(
        worker.metric(&fixture, SAMPLE_CHECKS, &["completed"]),
        Some(1.0)
    );
    assert!(logs.read().lines().any(|line| {
        line.contains("onboard pause cleared") && line.contains("result=\"completed\"")
    }));
    Ok(())
}

#[tokio::test]
async fn paused_onboard_fails_after_drained_error_proof() -> Result<()> {
    let fixture = Fixture::new(IMMEDIATE, 1).await?;
    let mut worker = fixture.worker(0, true)?;
    worker.register_check()?;
    let (onboard, copy) = lose_reply_and_pause(&fixture, &mut worker).await?;
    fixture.assert_paused(&onboard, &worker, NO_PROOF);

    worker
        .finish_copy(
            copy,
            TransferDrainOutcome::DrainedWithError(anyhow!("injected copy failure")),
        )
        .await;
    tick(PROBE_INTERVAL).await;
    fixture.assert_settled(&onboard, failed_load()).await;
    assert_eq!(
        worker.metric(&fixture, SAMPLE_CHECKS, &["drained_with_error"]),
        Some(1.0)
    );
    Ok(())
}

fn failed_load() -> LoadOutcome {
    LoadOutcome::FailedPartial {
        block_ids: vec![DEST],
    }
}

#[tokio::test]
async fn paused_onboard_distinguishes_a_worker_without_proof() -> Result<()> {
    let fixture = Fixture::new(IMMEDIATE, 1).await?;
    let mut worker = fixture.worker(0, true)?;
    worker.register_check()?;
    worker.reply(ProbeReply::Silent);
    let (onboard, copy, lose_reply) = start_held(&fixture, &mut worker).await?;
    lose_reply.send(()).unwrap();
    worker.wait_for_check(1).await;
    tokio::time::pause();
    tick(DEADLINE).await;
    onboard.wait_for_pause().await;
    fixture.assert_paused(&onboard, &worker, SILENT);

    worker.reply(ProbeReply::WrongIncarnation);
    tick(PROBE_INTERVAL).await;
    eventually("wrong incarnation is recorded as no proof", || {
        worker.metric(&fixture, SAMPLE_COUNT, &[NO_PROOF]) == Some(1.0)
            || onboard.fence.is_complete()
    })
    .await;
    fixture.assert_paused(&onboard, &worker, NO_PROOF);
    assert_eq!(worker.metric(&fixture, SAMPLE_COUNT, &[SILENT]), Some(0.0));
    assert!(worker.metric(&fixture, SAMPLE_ANSWER, &[]).is_some());
    assert!(worker.transfers.active());

    worker.reply(ProbeReply::Forward);
    worker
        .finish_copy(copy, TransferDrainOutcome::Completed)
        .await;
    tick(PROBE_INTERVAL).await;
    fixture.assert_settled(&onboard, LoadOutcome::Done).await;
    Ok(())
}

#[tokio::test]
async fn paused_onboard_watchdog_reports_a_wedged_probe() -> Result<()> {
    let mut logs = Logs::capture()?;
    let fixture = Fixture::new(IMMEDIATE, 1).await?;
    let mut worker = fixture.worker(0, true)?;
    worker.register_check()?;
    let (onboard, copy) = lose_reply_and_pause(&fixture, &mut worker).await?;
    fixture.engine.paused_onboards.stop_probes();
    let since = onboard.handle.paused().unwrap().since();
    let start = worker.metric(&fixture, SAMPLE_START, &[NO_PROOF]);
    let attempt = worker.metric(&fixture, SAMPLE_ATTEMPT, &[]);
    tick(PROBE_INTERVAL * 3).await;
    assert_eq!(worker.metric(&fixture, SAMPLE_STUCK, &[]), Some(1.0));
    assert_eq!(worker.metric(&fixture, SAMPLE_START, &[NO_PROOF]), start);
    assert_eq!(worker.metric(&fixture, SAMPLE_ATTEMPT, &[]), attempt);
    assert!(Instant::now().into_std() - since >= PROBE_INTERVAL * 3);
    logs.assert_event("onboard probe is stuck", &worker, "since_attempt_seconds");
    logs.assert_event("onboards remain paused", &worker, NO_PROOF);

    tick(PROBE_INTERVAL).await;
    assert_eq!(worker.metric(&fixture, SAMPLE_STUCK, &[]), Some(1.0));
    fixture.assert_paused(&onboard, &worker, NO_PROOF);
    assert!(worker.transfers.active());
    worker
        .finish_copy(copy, TransferDrainOutcome::Completed)
        .await;
    Ok(())
}

#[tokio::test]
async fn paused_onboard_waits_for_every_worker_before_releasing() -> Result<()> {
    let fixture = Fixture::new(IMMEDIATE, 2).await?;
    let mut first = fixture.worker(0, true)?;
    let mut second = fixture.worker(1, true)?;
    first.register_check()?;
    second.register_check()?;
    let lose_second_reply = second.lose_next_reply();
    let (onboard, first_copy, lose_first_reply) = start_held(&fixture, &mut first).await?;
    let second_copy = second.running_copy(&onboard).await;
    lose_first_reply.send(()).unwrap();
    lose_second_reply.send(()).unwrap();
    onboard.wait_for_pause().await;
    tokio::time::pause();
    assert_eq!(first.metric(&fixture, SAMPLE_COUNT, &[NO_PROOF]), Some(1.0));
    assert_eq!(
        second.metric(&fixture, SAMPLE_COUNT, &[NO_PROOF]),
        Some(1.0)
    );

    first
        .finish_copy(
            first_copy,
            TransferDrainOutcome::DrainedWithError(anyhow!("first worker drained")),
        )
        .await;
    tick(PROBE_INTERVAL).await;
    eventually("both workers' recovery checks settle", || {
        first.metric(&fixture, SAMPLE_COUNT, &[NO_PROOF]) == Some(0.0)
            && second.metric(&fixture, SAMPLE_CHECKS, &[NO_PROOF]) == Some(2.0)
    })
    .await;
    fixture.assert_paused(&onboard, &second, NO_PROOF);
    assert!(second.transfers.active());

    second
        .finish_copy(second_copy, TransferDrainOutcome::Completed)
        .await;
    tick(PROBE_INTERVAL).await;
    fixture.assert_settled(&onboard, failed_load()).await;
    assert_eq!(
        first.metric(&fixture, SAMPLE_CHECKS, &["drained_with_error"]),
        Some(1.0)
    );
    assert_eq!(
        second.metric(&fixture, SAMPLE_CHECKS, &["completed"]),
        Some(1.0)
    );
    assert_eq!(
        second.metric(&fixture, SAMPLE_COUNT, &[NO_PROOF]),
        Some(0.0)
    );
    Ok(())
}

#[tokio::test]
async fn paused_onboard_pins_engine_through_teardown_until_proven() -> Result<()> {
    let mut logs = Logs::capture()?;
    let fixture = Fixture::new(IMMEDIATE, 1).await?;
    let mut worker = fixture.worker(0, true)?;
    worker.register_check()?;
    let baseline = Arc::strong_count(&fixture.engine);
    let (first, first_copy) = lose_reply_and_pause(&fixture, &mut worker).await?;
    assert_eq!(Arc::strong_count(&fixture.engine), baseline + 1);
    worker
        .finish_copy(first_copy, TransferDrainOutcome::Completed)
        .await;
    tick(PROBE_INTERVAL).await;
    fixture.assert_settled(&first, LoadOutcome::Done).await;
    assert_eq!(Arc::strong_count(&fixture.engine), baseline);
    drop(first);
    tokio::time::resume();

    let prior_loads = fixture.sink.loads();
    let lose_reply = worker.lose_next_reply();
    let second = fixture.onboard()?;
    let second_copy = worker.running_copy(&second).await;
    lose_reply.send(()).unwrap();
    second.wait_for_pause().await;
    assert_eq!(Arc::strong_count(&fixture.engine), baseline + 1);
    tokio::time::pause();

    let eviction = fixture.engine.evict(&REQUEST.into());
    let eviction_fence = eviction.handle.unwrap();
    let tokens = eviction.fence.per_worker;
    assert!(!tokens.is_empty());
    let Onboard {
        handle,
        fence,
        source,
    } = second;
    assert!(!eviction_fence.is_complete());
    assert!(!fence.is_complete());
    let weak_engine = Arc::downgrade(&fixture.engine);
    let Fixture {
        engine,
        g2,
        registry,
        sink,
        ..
    } = fixture;
    drop(handle);
    drop(engine);
    assert!(weak_engine.upgrade().is_some());
    assert!(!reusable_g2_ids(&g2).contains(&source));
    assert_eq!(
        metric(&registry, SAMPLE_COUNT, &[&worker.label(), NO_PROOF]),
        Some(1.0)
    );
    assert_eq!(sink.loads(), prior_loads);
    assert!(sink.fences().is_empty());

    tick(PROBE_INTERVAL).await;
    assert!(
        weak_engine
            .upgrade()
            .unwrap()
            .paused_onboards
            .teardown_reported()
    );
    logs.assert_event(
        "connector owners dropped with paused onboards",
        &worker,
        NO_PROOF,
    );
    eventually("active copy is checked after teardown", || {
        metric(&registry, SAMPLE_CHECKS, &[&worker.label(), NO_PROOF]) == Some(3.0)
    })
    .await;
    worker
        .finish_copy(second_copy, TransferDrainOutcome::Completed)
        .await;
    tick(PROBE_INTERVAL).await;
    eventually("last engine owner released", || {
        weak_engine.upgrade().is_none()
    })
    .await;
    assert!(reusable_g2_ids(&g2).contains(&source));
    assert!(eviction_fence.is_complete());
    assert!(fence.is_complete());
    assert_eq!(sink.loads(), prior_loads);
    assert_eq!(sink.fences(), tokens);
    assert_eq!(
        metric(&registry, SAMPLE_COUNT, &[&worker.label(), NO_PROOF]),
        None
    );
    Ok(())
}

#[tokio::test]
async fn paused_onboard_without_a_capability_stays_fatal() -> Result<()> {
    let mut logs = Logs::capture()?;
    let fixture = Fixture::new(IMMEDIATE, 1).await?;
    let mut worker = fixture.worker(0, false)?;
    worker.register_check()?;
    let (onboard, copy) = lose_reply_and_pause(&fixture, &mut worker).await?;
    fixture.assert_paused(&onboard, &worker, NO_CAPABILITY);

    worker
        .finish_copy(copy, TransferDrainOutcome::Completed)
        .await;
    worker.forward(handler_names::WORKER_INCARNATION)?;
    tick(PROBE_INTERVAL * 3).await;
    fixture.assert_paused(&onboard, &worker, NO_CAPABILITY);
    assert_eq!(worker.checks.load(Ordering::Relaxed), 0);
    assert_eq!(worker.metric(&fixture, SAMPLE_ATTEMPT, &[]), None);
    assert_eq!(worker.metric(&fixture, SAMPLE_STUCK, &[]), None);
    logs.assert_event("onboard paused", &worker, NO_CAPABILITY);
    Ok(())
}

#[tokio::test]
async fn paused_onboard_honors_configured_probe_deadline() -> Result<()> {
    let fixture = Fixture::new(configured_timing(), 1).await?;
    let mut worker = fixture.worker(0, true)?;
    worker.register_check()?;
    let slow = Duration::from_secs(7);
    worker.reply(ProbeReply::Delayed(slow));
    let (onboard, copy, lose_reply) = start_held(&fixture, &mut worker).await?;
    lose_reply.send(()).unwrap();
    worker.wait_for_check(1).await;
    tokio::time::pause();

    tick(DEADLINE).await;
    fixture.assert_held(&onboard);
    assert!(onboard.handle.paused().is_none());
    tick(slow - DEADLINE).await;
    onboard.wait_for_pause().await;
    fixture.assert_paused(&onboard, &worker, NO_PROOF);
    assert_eq!(worker.metric(&fixture, SAMPLE_CHECKS, &[SILENT]), None);

    worker
        .finish_copy(copy, TransferDrainOutcome::Completed)
        .await;
    tick(PROBE_INTERVAL).await;
    worker.wait_for_check(2).await;
    tick(slow).await;
    fixture.assert_settled(&onboard, LoadOutcome::Done).await;
    assert_eq!(worker.metric(&fixture, SAMPLE_CHECKS, &[SILENT]), None);
    Ok(())
}

fn configured_timing() -> LocalTransferCheckTiming {
    LocalTransferCheckTiming::from(&kvbm_config::LocalTransferCheckConfig {
        wait_ms: 0,
        deadline_ms: 20_000,
    })
}

#[tokio::test]
async fn local_onboard_pause_names_the_reporting_worker() -> Result<()> {
    let mut logs = Logs::capture()?;
    let fixture = Fixture::new(IMMEDIATE, 1).await?;
    let worker = fixture.worker(0, true)?;
    fixture.relay.register_handler(
        Handler::unary_handler_async(
            handler_names::scoped(handler_names::LOCAL_TRANSFER, Some(worker.id)),
            |_| async {
                Ok(Some(Bytes::from_static(
                    br#"{"Unproven":{"error":"injected worker drain failure"}}"#,
                )))
            },
        )
        .build(),
    )?;
    let onboard = fixture.onboard()?;
    onboard.wait_for_pause().await;
    fixture.assert_paused(&onboard, &worker, NO_CAPABILITY);
    assert_eq!(worker.metric(&fixture, SAMPLE_ATTEMPT, &[]), None);
    logs.assert_event("onboard paused", &worker, NO_CAPABILITY);
    Ok(())
}

#[tokio::test]
async fn local_onboard_recovery_uses_the_configured_probe_deadline() -> Result<()> {
    let fixture = Fixture::new(configured_timing(), 1).await?;
    let mut worker = fixture.worker(0, true)?;
    worker.register_check()?;
    let (onboard, copy) = lose_reply_and_pause(&fixture, &mut worker).await?;
    fixture.assert_paused(&onboard, &worker, NO_PROOF);
    assert_eq!(
        worker.metric(&fixture, SAMPLE_CHECKS, &[NO_PROOF]),
        Some(1.0)
    );

    let slow = Duration::from_secs(7);
    worker.reply(ProbeReply::Delayed(slow));
    tick(PROBE_INTERVAL).await;
    worker.wait_for_check(2).await;
    tick(slow).await;
    eventually("slow active reply or timeout", || {
        worker.metric(&fixture, SAMPLE_CHECKS, &[NO_PROOF]) == Some(2.0)
            || worker.metric(&fixture, SAMPLE_CHECKS, &[SILENT]).is_some()
    })
    .await;
    assert_eq!(worker.metric(&fixture, SAMPLE_CHECKS, &[SILENT]), None);
    fixture.assert_paused(&onboard, &worker, NO_PROOF);

    worker
        .finish_copy(copy, TransferDrainOutcome::Completed)
        .await;
    tick(PROBE_INTERVAL).await;
    worker.wait_for_check(3).await;
    tick(slow).await;
    fixture.assert_settled(&onboard, LoadOutcome::Done).await;
    Ok(())
}

#[tokio::test]
async fn paused_onboard_watchdog_allows_configured_probe_budget() -> Result<()> {
    let mut logs = Logs::capture()?;
    let timing = configured_timing();
    let fixture = Fixture::new(timing, 1).await?;
    let mut worker = fixture.worker(0, true)?;
    worker.register_check()?;
    let (onboard, copy) = lose_reply_and_pause(&fixture, &mut worker).await?;
    worker.reply(ProbeReply::Silent);
    tick(PROBE_INTERVAL).await;
    worker.wait_for_check(2).await;
    fixture.engine.paused_onboards.stop_probes();

    tick(timing.deadline + Duration::from_secs(1)).await;
    assert_eq!(worker.metric(&fixture, SAMPLE_STUCK, &[]), None);
    assert!(!logs.read().contains("onboard probe is stuck"));
    tick(PROBE_INTERVAL + Duration::from_secs(1)).await;
    assert_eq!(worker.metric(&fixture, SAMPLE_STUCK, &[]), Some(1.0));
    logs.assert_event("onboard probe is stuck", &worker, "since_attempt_seconds");
    fixture.assert_paused(&onboard, &worker, NO_PROOF);
    assert!(worker.transfers.active());
    worker
        .finish_copy(copy, TransferDrainOutcome::Completed)
        .await;
    Ok(())
}

#[tokio::test]
async fn paused_onboard_reports_missing_handler_until_it_registers() -> Result<()> {
    let fixture = Fixture::new(IMMEDIATE, 1).await?;
    let mut worker = fixture.worker(0, true)?;
    let (onboard, copy) = lose_reply_and_pause(&fixture, &mut worker).await?;
    fixture.assert_paused(&onboard, &worker, NO_PROOF);
    assert!(worker.metric(&fixture, SAMPLE_ANSWER, &[]).is_some());

    worker.register_check()?;
    worker
        .finish_copy(copy, TransferDrainOutcome::Completed)
        .await;
    tick(PROBE_INTERVAL).await;
    fixture.assert_settled(&onboard, LoadOutcome::Done).await;
    worker.wait_for_check(1).await;
    Ok(())
}

#[tokio::test]
async fn paused_onboard_keeps_transport_failure_waiting() -> Result<()> {
    let mut logs = Logs::capture()?;
    let fixture = Fixture::new(LocalTransferCheckTiming::default(), 1).await?;
    let mut worker = fixture.worker(0, true)?;
    worker.register_check()?;
    let (onboard, copy, lose_reply) = start_held(&fixture, &mut worker).await?;
    tokio::time::pause();
    lose_reply.send(()).unwrap();
    logs.wait_for_loss().await;
    let messenger = fixture.engine.leader.messenger();
    messenger.unregister_peer(fixture.relay.instance_id());
    tick(WAIT).await;
    onboard.wait_for_pause().await;
    fixture.assert_paused(&onboard, &worker, SILENT);
    assert_eq!(worker.metric(&fixture, SAMPLE_ANSWER, &[]), None);
    assert_eq!(worker.checks.load(Ordering::Relaxed), 0);
    assert!(worker.transfers.active());

    messenger.register_peer(fixture.relay.peer_info())?;
    worker
        .finish_copy(copy, TransferDrainOutcome::Completed)
        .await;
    tick(PROBE_INTERVAL).await;
    fixture.assert_settled(&onboard, LoadOutcome::Done).await;
    Ok(())
}

#[tokio::test]
async fn paused_onboard_uses_fresh_handler_list_instead_of_cached_list() -> Result<()> {
    let fixture = Fixture::new(IMMEDIATE, 2).await?;
    let mut absent = fixture.worker(0, true)?;
    let mut present = fixture.worker(1, true)?;
    present.register_check()?;
    present.reply(ProbeReply::Error);
    // The handshake advertises the opposite of the actual handler list.
    let absent_name = absent.check_name();
    let present_name = present.check_name();
    let stale = Arc::new(AtomicBool::new(true));
    fixture.relay.register_streaming_handler(
        Handler::typed_unary_async("_hello", {
            let stale = stale.clone();
            let absent = absent_name.clone();
            let present = present_name.clone();
            move |ctx: velo::TypedContext<serde_json::Value>| {
                let mut handlers = ctx.msg.list_local_handlers();
                if stale.load(Ordering::Relaxed) {
                    handlers.retain(|name| *name != present);
                    handlers.push(absent.clone());
                }
                async move { Ok::<_, anyhow::Error>(serde_json::json!({"handlers": handlers})) }
            }
        })
        .build(),
    )?;
    let messenger = fixture.engine.leader.messenger();
    let instance = fixture.relay.instance_id();
    timeout(DEADLINE, messenger.refresh_handlers(instance)).await??;
    let cached = timeout(DEADLINE, messenger.available_handlers(instance)).await??;
    assert!(cached.contains(&absent_name));
    assert!(!cached.contains(&present_name));

    let lose_present_reply = present.lose_next_reply();
    let (onboard, absent_copy, lose_absent_reply) = start_held(&fixture, &mut absent).await?;
    let present_copy = present.running_copy(&onboard).await;
    lose_absent_reply.send(()).unwrap();
    lose_present_reply.send(()).unwrap();
    onboard.wait_for_pause().await;
    fixture.assert_held(&onboard);
    assert_eq!(
        absent.metric(&fixture, SAMPLE_COUNT, &[NO_PROOF]),
        Some(1.0)
    );
    assert!(absent.metric(&fixture, SAMPLE_ANSWER, &[]).is_some());
    assert_eq!(present.metric(&fixture, SAMPLE_COUNT, &[SILENT]), Some(1.0));
    assert_eq!(present.metric(&fixture, SAMPLE_ANSWER, &[]), None);

    assert!(absent.transfers.active());
    assert!(present.transfers.active());
    absent.register_check()?;
    present.reply(ProbeReply::Forward);
    stale.store(false, Ordering::Relaxed);
    timeout(DEADLINE, messenger.refresh_handlers(instance)).await??;
    absent
        .finish_copy(absent_copy, TransferDrainOutcome::Completed)
        .await;
    present
        .finish_copy(present_copy, TransferDrainOutcome::Completed)
        .await;
    tokio::time::pause();
    tick(PROBE_INTERVAL).await;
    fixture.assert_settled(&onboard, LoadOutcome::Done).await;
    Ok(())
}
