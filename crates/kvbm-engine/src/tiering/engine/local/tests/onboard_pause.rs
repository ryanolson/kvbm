// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{Read, Seek};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::anyhow;
use kvbm_logical::BlockManager;
use kvbm_physical::transfer::{
    AggregateTransferFailure, TransferDrainOutcome, TransferPartFailure,
};
use kvbm_protocols::connector::FenceHandle;
use tokio::time::{Instant, advance, timeout};
use velo::Messenger;

use super::super::super::paused::{PROBE_INTERVAL, PauseCell, PausedLoad};
use super::super::super::test_support::{CompletedParallelWorkers, build_leader_with_messenger};
use super::onboard_drain::matched_search;
use super::*;
use crate::testing::messenger::create_messenger_tcp;
use crate::worker::LocalTransferRecheck;
use crate::worker::group::{ParallelWorkers, SpmdParallelWorkers};
use crate::worker::velo::{LocalTransferCheckTiming, VeloWorkerClient, handler_names};

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

struct Fixture {
    engine: Arc<LocalConnectorEngine>,
    sink: Arc<RecordingSink>,
    g2: Arc<BlockManager<G2>>,
    registry: prometheus::Registry,
    worker: Arc<Messenger>,
}

impl Fixture {
    async fn new() -> Result<Self> {
        Self::with_workers(|_, _| Arc::new(CompletedParallelWorkers)).await
    }

    async fn with_velo_worker(timing: LocalTransferCheckTiming) -> Result<Self> {
        Self::with_workers(|messenger, worker| {
            let client =
                VeloWorkerClient::new_for_worker(messenger.clone(), worker.instance_id(), 0)
                    .with_local_transfer_check(timing);
            Arc::new(SpmdParallelWorkers::new(
                vec![Arc::new(client)],
                Arc::new(velo::EventManager::local()),
                messenger.runtime().clone(),
            ))
        })
        .await
    }

    async fn with_workers(
        workers: impl FnOnce(&Arc<Messenger>, &Arc<Messenger>) -> Arc<dyn ParallelWorkers>,
    ) -> Result<Self> {
        let messenger = timeout(WALL_LIMIT, create_messenger_tcp()).await??;
        let worker = timeout(WALL_LIMIT, create_messenger_tcp()).await??;
        worker.register_peer(messenger.peer_info())?;
        messenger.register_peer(worker.peer_info())?;
        let workers = workers(&messenger, &worker);
        let (leader, g2) =
            build_leader_with_messenger(messenger, LogicalResourceId(0), 2, BS, workers)?;
        let sink = RecordingSink::new();
        let registry = leader.observability().unwrap().registry().clone();
        let engine = LocalConnectorEngine::new(Arc::new(leader), sink.clone(), BS, true);
        Ok(Self {
            engine,
            sink,
            g2,
            registry,
            worker,
        })
    }

    async fn worker(&self, id: u64) -> Result<TestWorker> {
        let incarnation = Uuid::new_v4();
        let check = LocalTransferRecheck {
            instance: self.worker.instance_id(),
            worker_id: Some(id),
            incarnation: Some(incarnation),
            transfer_id: Uuid::new_v4(),
        };
        let answer = Arc::new(Mutex::new((
            Duration::ZERO,
            Some(reply(incarnation, serde_json::json!("Unknown"))),
        )));
        let checks = Arc::new(AtomicUsize::new(0));
        let handler = velo::Handler::unary_handler_async(
            handler_names::scoped(handler_names::LOCAL_TRANSFER_CHECK, Some(id)),
            {
                let answer = answer.clone();
                let checks = checks.clone();
                move |_| {
                    let (delay, answer) = answer.lock().unwrap().clone();
                    let reply_at = Instant::now() + delay;
                    checks.fetch_add(1, Ordering::Relaxed);
                    async move {
                        if !delay.is_zero() {
                            tokio::time::sleep_until(reply_at).await;
                        }
                        Ok(answer)
                    }
                }
            },
        )
        .build();
        self.worker.register_handler(handler)?;
        // Establish the route before pausing the clock. The tests control the
        // deadline of the check itself.
        let initial = check.check(self.engine.leader.messenger(), DEADLINE).await;
        assert!(matches!(initial, TransferDrainOutcome::Unproven(_)));
        assert_eq!(checks.load(Ordering::Relaxed), 1);
        *answer.lock().unwrap() = (PROBE_INTERVAL * 4, None);
        Ok(TestWorker {
            answer,
            check,
            checks,
        })
    }

    fn serve_local_onboard(
        &self,
        worker: &TestWorker,
        response: Option<bytes::Bytes>,
    ) -> Result<()> {
        let incarnation = serde_json::to_vec(&serde_json::json!({
            "incarnation": worker.check.incarnation.unwrap(),
        }))?;
        for (name, reply) in [
            (handler_names::WORKER_INCARNATION, Some(incarnation.into())),
            (handler_names::LOCAL_TRANSFER, response),
        ] {
            let handler = velo::Handler::unary_handler_async(
                handler_names::scoped(name, worker.check.worker_id),
                move |_| {
                    let reply = reply.clone();
                    async move {
                        reply
                            .map(Some)
                            .ok_or_else(|| anyhow!("injected lost reply"))
                    }
                },
            )
            .build();
            self.worker.register_handler(handler)?;
        }
        Ok(())
    }

    fn start_local_onboard(&self) -> Result<(FindBlocksHandle, OnboardHandle)> {
        let (search, _) = matched_search(&self.engine, &self.g2, REQUEST)?;
        let onboard = self.engine.clone().onboard_blocks(&search, &[DEST], BS)?;
        Ok((search, onboard))
    }

    fn mint(&self) -> Action {
        let id = ActionId::new();
        let status = Arc::new(Mutex::new(ActionStatus::Pending));
        let physical = Arc::new(AtomicBool::new(false));
        let pause = PauseCell::default();
        self.engine.actions.insert(
            id,
            ActionRecord::new(REQUEST.into(), Arc::downgrade(&status))
                .with_physical_drain(tokio_util::sync::CancellationToken::new()),
        );
        self.engine
            .by_request
            .entry(REQUEST.into())
            .or_default()
            .push(id);
        let owner: Arc<dyn LeaderEngine> = self.engine.clone();
        let handle = OnboardHandle::new_with_physical_drain(
            id,
            Arc::downgrade(&owner),
            status,
            vec![DEST],
            FenceHandle::new(physical.clone()),
        )
        .with_pause_cell(pause.clone());
        Action {
            handle,
            physical,
            pause,
        }
    }

    fn load(&self, action: &Action) -> PausedLoad {
        PausedLoad {
            action: *action.handle.id(),
            request: REQUEST.into(),
            destinations: vec![DEST],
            physical_drain: action.physical.clone(),
            pause: action.pause.clone(),
            holds: Box::new(self.g2.allocate_blocks(1).expect("one G2 block")),
            resource: None,
        }
    }

    fn lose_reply(&self, action: &Action, worker: &TestWorker, deadline: Duration) {
        let engine = self.engine.clone();
        let load = self.load(action);
        let check = worker.check.clone();
        tokio::spawn(async move {
            // Inject a lost reply at the test driver's receipt boundary, then
            // run the real bounded proof check.
            tokio::time::sleep(WAIT).await;
            let TransferDrainOutcome::Unproven(error) =
                check.check(engine.leader.messenger(), deadline).await
            else {
                panic!("the initial check must leave the outcome unproven");
            };
            engine.pause_load_action(load, error);
        });
    }

    async fn pause(&self, worker: &TestWorker) -> Action {
        self.pause_after_check(worker, DEADLINE, DEADLINE).await
    }

    async fn pause_after_check(
        &self,
        worker: &TestWorker,
        deadline: Duration,
        check_elapsed: Duration,
    ) -> Action {
        let action = self.mint();
        self.lose_reply(&action, worker, deadline);
        tokio::task::yield_now().await;
        tick(WAIT).await;
        eventually("initial check reaches the worker", || {
            worker.checks.load(Ordering::Relaxed) >= 2
        })
        .await;
        tick(check_elapsed).await;
        eventually("paused handle", || action.handle.paused().is_some()).await;
        action
    }

    fn assert_paused(&self, action: &Action, worker: &TestWorker) {
        assert_eq!(action.handle.outcome(), None, "the scheduler stays pending");
        assert_eq!(action.handle.paused().unwrap().worker(), worker.label());
        assert!(self.sink.loads().is_empty(), "no scheduler release");
        assert!(
            !action.physical.load(Ordering::Acquire),
            "the fence stays armed"
        );
        assert_eq!(self.g2.available_blocks(), 1, "the engine owns the block");
        assert_eq!(
            metric(&self.registry, SAMPLE_COUNT, &["worker_not_answering"]),
            Some(1.0)
        );
    }
}

struct Action {
    handle: OnboardHandle,
    physical: Arc<AtomicBool>,
    pause: PauseCell,
}

struct TestWorker {
    answer: Arc<Mutex<(Duration, Option<bytes::Bytes>)>>,
    check: LocalTransferRecheck,
    checks: Arc<AtomicUsize>,
}

impl TestWorker {
    fn label(&self) -> String {
        format!("{}#{}", self.check.instance, self.check.worker_id.unwrap())
    }

    fn answer(&self, record: serde_json::Value) {
        self.answer_as(self.check.incarnation.unwrap(), record);
    }

    fn answer_as(&self, incarnation: Uuid, record: serde_json::Value) {
        *self.answer.lock().unwrap() = (Duration::ZERO, Some(reply(incarnation, record)));
    }

    fn answer_after(&self, delay: Duration, record: serde_json::Value) {
        *self.answer.lock().unwrap() =
            (delay, Some(reply(self.check.incarnation.unwrap(), record)));
    }

    fn go_silent(&self) {
        *self.answer.lock().unwrap() = (PROBE_INTERVAL * 4, None);
    }

    fn unproven(&self) -> anyhow::Error {
        anyhow!("lost copy receipt").context(self.check.clone())
    }
}

fn reply(incarnation: Uuid, record: serde_json::Value) -> bytes::Bytes {
    serde_json::to_vec(&serde_json::json!({"incarnation": incarnation, "record": record}))
        .unwrap()
        .into()
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

#[tokio::test]
#[allow(clippy::disallowed_macros)]
async fn paused_onboard_retains_holds_by_the_initial_check_deadline() -> Result<()> {
    let fixture = Fixture::new().await?;
    let worker = fixture.worker(0).await?;
    tokio::time::pause();
    let lost_at = Instant::now();
    let action = fixture.pause(&worker).await;
    assert!(lost_at.elapsed() <= WAIT + DEADLINE + Duration::from_millis(2));
    fixture.assert_paused(&action, &worker);
    let since = action.handle.paused().unwrap().since();
    let start = metric(&fixture.registry, SAMPLE_START, &["worker_not_answering"]).unwrap();
    let attempt = metric(&fixture.registry, SAMPLE_ATTEMPT, &[]).unwrap();
    assert!((start - attempt - DEADLINE.as_secs_f64()).abs() < 0.01);
    assert_eq!(metric(&fixture.registry, SAMPLE_ANSWER, &[]), None);
    assert_eq!(Instant::now().into_std() - since, Duration::ZERO);
    tick(Duration::from_secs(1)).await;
    assert_eq!(
        metric(&fixture.registry, SAMPLE_START, &["worker_not_answering"]),
        Some(start)
    );
    assert_eq!(
        Instant::now().into_std() - since,
        Duration::from_millis(1001)
    );
    let metrics = prometheus::TextEncoder::new().encode_to_string(&fixture.registry.gather())?;
    for line in metrics
        .lines()
        .filter(|line| line.contains("worker_not_answering") || line.starts_with(SAMPLE_ATTEMPT))
        .filter(|line| !line.starts_with('#'))
    {
        println!("{line}");
    }

    worker.answer(serde_json::json!("Completed"));
    tick(PROBE_INTERVAL).await;
    eventually("logical terminal", || action.handle.is_complete()).await;
    Ok(())
}

#[tokio::test]
async fn paused_onboard_resumes_after_arrival_proof() -> Result<()> {
    let fixture = Fixture::new().await?;
    let worker = fixture.worker(0).await?;
    tokio::time::pause();
    let action = fixture.pause(&worker).await;
    fixture.assert_paused(&action, &worker);
    worker.answer(serde_json::json!("Completed"));
    tick(PROBE_INTERVAL).await;
    eventually("physical terminal", || {
        action.physical.load(Ordering::Acquire)
    })
    .await;
    assert_eq!(action.handle.outcome(), Some(LoadOutcome::Done));
    assert!(action.handle.paused().is_none());
    assert_eq!(
        fixture.sink.loads(),
        vec![(REQUEST.into(), LoadOutcome::Done)]
    );
    assert_eq!(fixture.g2.available_blocks(), 2);
    assert_eq!(
        metric(&fixture.registry, SAMPLE_COUNT, &["worker_not_answering"]),
        Some(0.0)
    );
    assert_eq!(
        metric(
            &fixture.registry,
            "kvbm_onboard_probe_checks_total",
            &["completed"]
        ),
        Some(1.0)
    );
    Ok(())
}

#[tokio::test]
async fn paused_onboard_fails_after_drained_error_proof() -> Result<()> {
    let fixture = Fixture::new().await?;
    let worker = fixture.worker(0).await?;
    tokio::time::pause();
    let action = fixture.pause(&worker).await;
    fixture.assert_paused(&action, &worker);
    worker.answer(serde_json::json!({"DrainedWithError": {"error": "injected worker failure"}}));
    tick(PROBE_INTERVAL).await;
    eventually("physical terminal", || {
        action.physical.load(Ordering::Acquire)
    })
    .await;
    let failed = LoadOutcome::FailedPartial {
        block_ids: vec![DEST],
    };
    assert_eq!(action.handle.outcome(), Some(failed.clone()));
    assert!(action.handle.paused().is_none());
    assert_eq!(fixture.sink.loads(), vec![(REQUEST.into(), failed)]);
    assert_eq!(fixture.g2.available_blocks(), 2);
    Ok(())
}

#[tokio::test]
async fn paused_onboard_distinguishes_a_worker_without_proof() -> Result<()> {
    let fixture = Fixture::new().await?;
    let worker = fixture.worker(0).await?;
    tokio::time::pause();
    let action = fixture.pause(&worker).await;
    worker.answer_as(Uuid::new_v4(), serde_json::json!("Completed"));
    tick(PROBE_INTERVAL).await;
    eventually("pause metrics", || {
        metric(&fixture.registry, SAMPLE_COUNT, &["worker_no_proof"]) == Some(1.0)
    })
    .await;
    assert_eq!(
        metric(&fixture.registry, SAMPLE_COUNT, &["worker_not_answering"]),
        Some(0.0)
    );
    assert!(metric(&fixture.registry, SAMPLE_ANSWER, &[]).is_some());
    assert_eq!(action.handle.outcome(), None);
    assert_eq!(fixture.g2.available_blocks(), 1);
    worker.answer(serde_json::json!("Completed"));
    tick(PROBE_INTERVAL).await;
    eventually("logical terminal", || action.handle.is_complete()).await;
    Ok(())
}

#[tokio::test]
async fn paused_onboard_watchdog_reports_a_wedged_probe() -> Result<()> {
    let fixture = Fixture::new().await?;
    let worker = fixture.worker(0).await?;
    tokio::time::pause();
    let action = fixture.pause(&worker).await;
    fixture.engine.paused_onboards.stop_probes();
    let since = action.handle.paused().unwrap().since();
    let start = metric(&fixture.registry, SAMPLE_START, &["worker_not_answering"]);
    let attempt = metric(&fixture.registry, SAMPLE_ATTEMPT, &[]);
    tick(PROBE_INTERVAL * 3).await;
    assert_eq!(
        metric(&fixture.registry, "kvbm_onboard_probe_stuck_total", &[]),
        Some(1.0)
    );
    assert_eq!(
        metric(&fixture.registry, SAMPLE_START, &["worker_not_answering"]),
        start
    );
    assert_eq!(metric(&fixture.registry, SAMPLE_ATTEMPT, &[]), attempt);
    assert!(Instant::now().into_std() - since >= PROBE_INTERVAL * 3);
    tick(PROBE_INTERVAL).await;
    assert_eq!(
        metric(&fixture.registry, "kvbm_onboard_probe_stuck_total", &[]),
        Some(1.0)
    );
    fixture.assert_paused(&action, &worker);
    Ok(())
}

#[tokio::test]
async fn paused_onboard_waits_for_every_worker_before_releasing() -> Result<()> {
    let fixture = Fixture::new().await?;
    let first = fixture.worker(0).await?;
    let second = fixture.worker(1).await?;
    tokio::time::pause();
    let action = fixture.mint();
    let error = anyhow::Error::new(AggregateTransferFailure::new(
        None,
        [first.unproven(), second.unproven()]
            .into_iter()
            .map(TransferPartFailure::Unproven)
            .collect(),
    ));
    first.answer(serde_json::json!({"DrainedWithError": {"error": "first worker drained"}}));
    fixture
        .engine
        .pause_load_action(fixture.load(&action), error);
    tokio::task::yield_now().await;
    tick(PROBE_INTERVAL).await;
    eventually("pause metrics", || {
        metric(
            &fixture.registry,
            SAMPLE_COUNT,
            &[&first.label(), "worker_not_answering"],
        ) == Some(0.0)
    })
    .await;
    assert_eq!(action.handle.outcome(), None);
    assert_eq!(action.handle.paused().unwrap().worker(), second.label());
    assert!(!action.physical.load(Ordering::Acquire));
    assert_eq!(fixture.g2.available_blocks(), 1);
    second.answer(serde_json::json!("Completed"));
    tick(DEADLINE).await;
    tick(PROBE_INTERVAL).await;
    eventually("physical terminal", || {
        action.physical.load(Ordering::Acquire)
    })
    .await;
    assert_eq!(
        action.handle.outcome(),
        Some(LoadOutcome::FailedPartial {
            block_ids: vec![DEST]
        })
    );
    assert_eq!(fixture.g2.available_blocks(), 2);
    Ok(())
}

#[tokio::test]
async fn paused_onboard_pins_engine_through_teardown_until_proven() -> Result<()> {
    let fixture = Fixture::new().await?;
    let worker = fixture.worker(0).await?;
    tokio::time::pause();
    let baseline = Arc::strong_count(&fixture.engine);
    let first = fixture.pause(&worker).await;
    assert_eq!(Arc::strong_count(&fixture.engine), baseline + 1);
    worker.answer(serde_json::json!("Completed"));
    tick(PROBE_INTERVAL).await;
    eventually("first physical terminal", || {
        first.physical.load(Ordering::Acquire)
    })
    .await;
    assert_eq!(Arc::strong_count(&fixture.engine), baseline);
    drop(first);

    let second = fixture.mint();
    worker.go_silent();
    fixture
        .engine
        .pause_load_action(fixture.load(&second), worker.unproven());
    assert_eq!(Arc::strong_count(&fixture.engine), baseline + 1);
    let eviction_fence = fixture.engine.evict(&REQUEST.into()).handle.unwrap();
    assert!(!eviction_fence.is_complete());
    let weak_engine = Arc::downgrade(&fixture.engine);
    let Fixture {
        engine,
        g2,
        registry,
        worker: transport,
        ..
    } = fixture;
    drop(second);
    drop(engine);
    assert!(weak_engine.upgrade().is_some());
    assert_eq!(g2.available_blocks(), 1);
    assert_eq!(
        metric(&registry, SAMPLE_COUNT, &["worker_not_answering"]),
        Some(1.0)
    );

    tick(PROBE_INTERVAL).await;
    assert!(
        weak_engine
            .upgrade()
            .unwrap()
            .paused_onboards
            .teardown_reported()
    );
    eventually("the post-teardown probe reaches the worker", || {
        worker.checks.load(Ordering::Relaxed) >= 4
    })
    .await;
    // Keep the worker service alive after the connector is gone.
    worker.answer(serde_json::json!("Completed"));
    tick(DEADLINE).await;
    tick(PROBE_INTERVAL).await;
    eventually("engine drop", || weak_engine.upgrade().is_none()).await;
    assert_eq!(g2.available_blocks(), 2);
    assert!(eviction_fence.is_complete());
    drop(transport);
    Ok(())
}

#[tokio::test]
async fn paused_onboard_without_a_capability_stays_fatal() -> Result<()> {
    let fixture = Fixture::new().await?;
    tokio::time::pause();
    let action = fixture.mint();
    fixture
        .engine
        .pause_load_action(fixture.load(&action), anyhow!("in-process drain failed"));
    tick(PROBE_INTERVAL * 3).await;
    assert_eq!(action.handle.paused().unwrap().worker(), "unknown");
    assert_eq!(action.handle.outcome(), None);
    assert_eq!(fixture.g2.available_blocks(), 1);
    assert_eq!(
        metric(&fixture.registry, SAMPLE_COUNT, &["no_recovery_capability"]),
        Some(1.0)
    );
    assert_eq!(metric(&fixture.registry, SAMPLE_ATTEMPT, &[]), None);
    assert_eq!(
        metric(&fixture.registry, "kvbm_onboard_probe_stuck_total", &[]),
        None
    );
    Ok(())
}

#[tokio::test]
async fn paused_onboard_honors_configured_probe_deadline() -> Result<()> {
    let fixture = Fixture::new().await?;
    let worker = fixture.worker(0).await?;
    let config = kvbm_config::LocalTransferCheckConfig {
        deadline_ms: 20_000,
        ..Default::default()
    };
    let deadline = crate::worker::velo::LocalTransferCheckTiming::from(&config).deadline;
    let reply_delay = Duration::from_secs(7);
    tokio::time::pause();

    worker.answer_after(reply_delay, serde_json::json!("Active"));
    let action = fixture
        .pause_after_check(&worker, deadline, reply_delay)
        .await;
    let no_answer = || {
        metric(
            &fixture.registry,
            "kvbm_onboard_probe_checks_total",
            &["worker_not_answering"],
        )
    };
    assert_eq!(
        metric(&fixture.registry, SAMPLE_COUNT, &["worker_no_proof"]),
        Some(1.0)
    );

    worker.answer_after(reply_delay, serde_json::json!("Completed"));
    tick(PROBE_INTERVAL).await;
    eventually("recovery probe reaches the worker", || {
        worker.checks.load(Ordering::Relaxed) == 3
    })
    .await;
    tick(reply_delay).await;
    eventually("recovery proof or probe deadline", || {
        action.physical.load(Ordering::Acquire) || no_answer().is_some()
    })
    .await;
    assert_eq!(action.handle.outcome(), Some(LoadOutcome::Done));
    assert_eq!(no_answer(), None);
    assert_eq!(fixture.g2.available_blocks(), 2);
    Ok(())
}

#[tokio::test]
async fn local_onboard_pause_names_the_reporting_worker() -> Result<()> {
    let mut log = tempfile::tempfile()?;
    // Keep a second dispatch alive so tracing checks the capture's interest
    // when another test thread registers the event first.
    let _quiet = tracing::Dispatch::new(
        tracing_subscriber::fmt()
            .with_max_level(tracing::level_filters::LevelFilter::OFF)
            .finish(),
    );
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_writer(log.try_clone()?)
        .finish();
    let _capture = tracing::subscriber::set_default(subscriber);
    let fixture = Fixture::with_velo_worker(LocalTransferCheckTiming::default()).await?;
    let worker = fixture.worker(0).await?;
    fixture.serve_local_onboard(
        &worker,
        Some(bytes::Bytes::from_static(
            br#"{"Unproven":{"error":"injected worker drain failure"}}"#,
        )),
    )?;
    let (_search, onboard) = fixture.start_local_onboard()?;
    eventually("local onboard disposition", || {
        onboard.paused().is_some() || onboard.is_complete()
    })
    .await;

    let pause = onboard.paused().expect("the local onboard must pause");
    let label = worker.label();
    assert_eq!(pause.worker(), label);
    assert!(onboard.outcome().is_none());
    assert!(fixture.sink.loads().is_empty());
    assert_eq!(
        metric(
            &fixture.registry,
            SAMPLE_COUNT,
            &[&label, "no_recovery_capability"]
        ),
        Some(1.0)
    );
    assert_eq!(metric(&fixture.registry, SAMPLE_ATTEMPT, &[]), None);
    log.rewind()?;
    let mut lines = String::new();
    log.read_to_string(&mut lines)?;
    assert!(
        lines.lines().any(|line| {
            line.contains("onboard paused")
                && line.contains(&format!("worker={label}"))
                && line.contains("no_recovery_capability")
        }),
        "{lines}"
    );
    Ok(())
}

#[tokio::test]
async fn local_onboard_recovery_uses_the_configured_probe_deadline() -> Result<()> {
    let timing = LocalTransferCheckTiming::from(&kvbm_config::LocalTransferCheckConfig {
        wait_ms: 0,
        deadline_ms: 20_000,
    });
    let fixture = Fixture::with_velo_worker(timing).await?;
    let worker = fixture.worker(0).await?;
    fixture.serve_local_onboard(&worker, None)?;
    worker.answer(serde_json::json!("Active"));
    let (_search, mut onboard) = fixture.start_local_onboard()?;
    // Use wall time until the TCP reply is lost and the first check responds.
    // Advance only the probe timers once the driver has parked the copy.
    eventually("local onboard disposition", || {
        onboard.paused().is_some() || onboard.is_complete()
    })
    .await;
    let pause = onboard.paused().expect("the local onboard must pause");
    assert_eq!(pause.worker(), worker.label());
    assert!(fixture.sink.loads().is_empty());
    assert_eq!(fixture.g2.available_blocks(), 1);
    let fence = onboard.take_physical_drain_fence().expect("local fence");
    assert!(!fence.is_complete());
    let check_count = |state: &str| metric(&fixture.registry, SAMPLE_CHECKS, &[state]);
    assert_eq!(check_count("worker_no_proof"), Some(1.0));

    let slow_reply = Duration::from_secs(7);
    tokio::time::pause();
    worker.answer_after(slow_reply, serde_json::json!("Active"));
    tick(PROBE_INTERVAL).await;
    eventually("first recovery check reached the worker", || {
        worker.checks.load(Ordering::Relaxed) == 3
    })
    .await;
    tick(slow_reply).await;
    eventually("first recovery check disposition", || {
        check_count("worker_no_proof") == Some(2.0) || check_count("worker_not_answering").is_some()
    })
    .await;
    assert_eq!(check_count("worker_not_answering"), None);
    assert_eq!(
        metric(&fixture.registry, SAMPLE_COUNT, &["worker_no_proof"]),
        Some(1.0)
    );
    assert!(onboard.paused().is_some());
    assert!(!fence.is_complete());
    assert_eq!(fixture.g2.available_blocks(), 1);

    worker.answer_after(slow_reply, serde_json::json!("Completed"));
    tick(PROBE_INTERVAL).await;
    eventually("second recovery check reached the worker", || {
        worker.checks.load(Ordering::Relaxed) == 4
    })
    .await;
    tick(slow_reply).await;
    eventually("onboard terminal", || !fixture.sink.loads().is_empty()).await;
    assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
    assert!(onboard.paused().is_none());
    assert!(fence.is_complete());
    assert_eq!(fixture.g2.available_blocks(), 2);
    assert_eq!(
        fixture.sink.loads(),
        vec![(REQUEST.into(), LoadOutcome::Done)]
    );
    Ok(())
}

#[tokio::test]
async fn paused_onboard_watchdog_allows_configured_probe_budget() -> Result<()> {
    let fixture = Fixture::new().await?;
    let worker = fixture.worker(0).await?;
    let deadline = Duration::from_secs(20);
    tokio::time::pause();

    let action = fixture.pause_after_check(&worker, deadline, deadline).await;
    tick(PROBE_INTERVAL).await;
    eventually("first recovery probe reaches the worker", || {
        worker.checks.load(Ordering::Relaxed) == 3
    })
    .await;
    fixture.engine.paused_onboards.stop_probes();

    let stuck = || metric(&fixture.registry, "kvbm_onboard_probe_stuck_total", &[]);
    tick(deadline + Duration::from_secs(1)).await;
    assert_eq!(
        stuck(),
        None,
        "the next attempt may take the full deadline and interval"
    );
    tick(PROBE_INTERVAL + Duration::from_secs(1)).await;
    assert_eq!(stuck(), Some(1.0));
    assert_eq!(action.handle.outcome(), None);
    Ok(())
}

fn check_target(fixture: &Fixture, worker: u64) -> LocalTransferRecheck {
    LocalTransferRecheck {
        instance: fixture.worker.instance_id(),
        worker_id: Some(worker),
        incarnation: Some(Uuid::new_v4()),
        transfer_id: Uuid::new_v4(),
    }
}

async fn check_and_pause(fixture: &Fixture, check: &LocalTransferRecheck) -> Action {
    let action = fixture.mint();
    let TransferDrainOutcome::Unproven(error) = check
        .check(fixture.engine.leader.messenger(), DEADLINE)
        .await
    else {
        panic!("this check cannot prove the copy drained");
    };
    fixture
        .engine
        .pause_load_action(fixture.load(&action), error);
    action
}

#[tokio::test]
async fn paused_onboard_reports_missing_handler_until_it_registers() -> Result<()> {
    let fixture = Fixture::new().await?;
    let check = check_target(&fixture, 99);
    let action = check_and_pause(&fixture, &check).await;

    assert_eq!(
        metric(&fixture.registry, SAMPLE_COUNT, &["worker_no_proof"]),
        Some(1.0)
    );
    assert_eq!(action.handle.outcome(), None);
    assert_eq!(fixture.g2.available_blocks(), 1);
    assert!(metric(&fixture.registry, SAMPLE_ANSWER, &[]).is_some());

    let payload = reply(check.incarnation.unwrap(), serde_json::json!("Completed"));
    let handler = velo::Handler::unary_handler_async(
        handler_names::scoped(handler_names::LOCAL_TRANSFER_CHECK, check.worker_id),
        move |_| {
            let payload = payload.clone();
            async move { Ok(Some(payload)) }
        },
    )
    .build();
    fixture.worker.register_handler(handler)?;
    tokio::time::pause();
    tick(PROBE_INTERVAL).await;
    eventually("the newly registered handler proves the copy", || {
        action.physical.load(Ordering::Acquire)
    })
    .await;
    assert_eq!(action.handle.outcome(), Some(LoadOutcome::Done));
    assert_eq!(fixture.g2.available_blocks(), 2);
    Ok(())
}

#[tokio::test]
async fn paused_onboard_keeps_transport_failure_waiting() -> Result<()> {
    let fixture = Fixture::new().await?;
    let worker = fixture.worker(0).await?;
    fixture
        .engine
        .leader
        .messenger()
        .unregister_peer(worker.check.instance);
    let action = check_and_pause(&fixture, &worker.check).await;
    fixture.assert_paused(&action, &worker);
    assert_eq!(metric(&fixture.registry, SAMPLE_ANSWER, &[]), None);
    Ok(())
}

#[tokio::test]
async fn paused_onboard_uses_fresh_handler_list_instead_of_cached_list() -> Result<()> {
    let fixture = Fixture::new().await?;
    let absent = check_target(&fixture, 1);
    let present = check_target(&fixture, 2);
    let absent_name = handler_names::scoped(handler_names::LOCAL_TRANSFER_CHECK, absent.worker_id);
    let present_name =
        handler_names::scoped(handler_names::LOCAL_TRANSFER_CHECK, present.worker_id);
    fixture.worker.register_handler(
        velo::Handler::unary_handler_async(&present_name, |_| async {
            Err::<Option<bytes::Bytes>, _>(anyhow!("the check failed"))
        })
        .build(),
    )?;
    // Simulate a stale handshake view. The worker's actual handler list has
    // the opposite answer for both checks.
    let cached_name = absent_name.clone();
    fixture.worker.register_streaming_handler(
        velo::Handler::typed_unary_async(
            "_hello",
            move |_: velo::TypedContext<serde_json::Value>| {
                let name = cached_name.clone();
                async move { Ok::<_, anyhow::Error>(serde_json::json!({"handlers": [name]})) }
            },
        )
        .build(),
    )?;
    let messenger = fixture.engine.leader.messenger();
    timeout(DEADLINE, messenger.refresh_handlers(absent.instance)).await??;
    let cached = timeout(DEADLINE, messenger.available_handlers(absent.instance)).await??;
    assert!(cached.contains(&absent_name));
    assert!(!cached.contains(&present_name));

    let absent_action = check_and_pause(&fixture, &absent).await;
    let present_action = check_and_pause(&fixture, &present).await;
    let absent_worker = format!("{}#1", absent.instance);
    let present_worker = format!("{}#2", present.instance);
    assert_eq!(
        [
            metric(
                &fixture.registry,
                SAMPLE_COUNT,
                &[&absent_worker, "worker_no_proof"]
            ),
            metric(
                &fixture.registry,
                SAMPLE_COUNT,
                &[&present_worker, "worker_not_answering"]
            ),
        ],
        [Some(1.0), Some(1.0)]
    );
    assert_eq!(absent_action.handle.outcome(), None);
    assert_eq!(present_action.handle.outcome(), None);
    assert_eq!(fixture.g2.available_blocks(), 0);
    Ok(())
}
