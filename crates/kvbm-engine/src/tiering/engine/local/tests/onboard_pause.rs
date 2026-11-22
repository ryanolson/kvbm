// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
use super::super::super::test_support::{CompletedParallelWorkers, build_leader_with_workers};
use super::*;
use crate::testing::messenger::create_messenger_tcp;
use crate::worker::LocalTransferRecheck;
use crate::worker::velo::handler_names;

const REQUEST: &str = "paused-onboard";
const DEST: BlockId = 50;
const WAIT: Duration = Duration::from_secs(2);
const DEADLINE: Duration = Duration::from_secs(5);
const WALL_LIMIT: Duration = Duration::from_secs(5);
const SAMPLE_COUNT: &str = "kvbm_paused_onboards";
const SAMPLE_START: &str = "kvbm_onboard_oldest_pause_timestamp_seconds";
const SAMPLE_ATTEMPT: &str = "kvbm_onboard_last_probe_attempt_timestamp_seconds";
const SAMPLE_ANSWER: &str = "kvbm_onboard_last_probe_answer_timestamp_seconds";

struct Fixture {
    engine: Arc<LocalConnectorEngine>,
    sink: Arc<RecordingSink>,
    g2: Arc<BlockManager<G2>>,
    registry: prometheus::Registry,
    worker: Arc<Messenger>,
}

impl Fixture {
    async fn new() -> Result<Self> {
        let (leader, g2) = timeout(
            WALL_LIMIT,
            build_leader_with_workers(
                LogicalResourceId(0),
                2,
                BS,
                Arc::new(CompletedParallelWorkers),
            ),
        )
        .await??;
        let worker = timeout(WALL_LIMIT, create_messenger_tcp()).await??;
        worker.register_peer(leader.messenger().peer_info())?;
        leader.messenger().register_peer(worker.peer_info())?;
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
        let answer = Arc::new(Mutex::new(Some(reply(
            incarnation,
            serde_json::json!("Unknown"),
        ))));
        let checks = Arc::new(AtomicUsize::new(0));
        let handler = velo::Handler::unary_handler_async(
            handler_names::scoped(handler_names::LOCAL_TRANSFER_CHECK, Some(id)),
            {
                let answer = answer.clone();
                let checks = checks.clone();
                move |_| {
                    let answer = answer.lock().unwrap().clone();
                    checks.fetch_add(1, Ordering::Relaxed);
                    async move {
                        if answer.is_none() {
                            tokio::time::sleep(PROBE_INTERVAL * 4).await;
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
        *answer.lock().unwrap() = None;
        Ok(TestWorker {
            answer,
            check,
            checks,
        })
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
        }
    }

    fn lose_reply(&self, action: &Action, worker: &TestWorker) {
        let engine = self.engine.clone();
        let load = self.load(action);
        let check = worker.check.clone();
        tokio::spawn(async move {
            // Inject a lost reply at the test driver's receipt boundary, then
            // run the real bounded proof check with the silent worker.
            tokio::time::sleep(WAIT).await;
            let TransferDrainOutcome::Unproven(error) =
                check.check(engine.leader.messenger(), DEADLINE).await
            else {
                panic!("the silent worker cannot prove its copy drained");
            };
            engine.pause_load_action(load, error);
        });
    }

    async fn pause(&self, worker: &TestWorker) -> Action {
        let action = self.mint();
        self.lose_reply(&action, worker);
        tokio::task::yield_now().await;
        tick(WAIT).await;
        eventually("initial check reaches the worker", || {
            worker.checks.load(Ordering::Relaxed) >= 2
        })
        .await;
        tick(DEADLINE).await;
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
    answer: Arc<Mutex<Option<bytes::Bytes>>>,
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
        *self.answer.lock().unwrap() = Some(reply(incarnation, record));
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

async fn tick(duration: Duration) {
    // Tokio's timer wheel rounds deadlines up to the next millisecond.
    advance(duration + Duration::from_millis(1)).await;
    tokio::task::yield_now().await;
}

async fn eventually(reason: &str, check: impl Fn() -> bool) {
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
    *worker.answer.lock().unwrap() = None;
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
