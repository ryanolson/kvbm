// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::io::{Read, Seek};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, bail};
use bytes::Bytes;
use kvbm_common::LogicalLayoutHandle;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::offload::{OffloadEngine, PipelineBuilder};
use kvbm_engine::testing::{TestManagerBuilder, create_messenger_pair_tcp};
use kvbm_engine::worker::VeloWorkerClient;
use kvbm_engine::worker::group::SpmdParallelWorkers;
use kvbm_engine::worker::velo::LocalTransferCheckTiming;
use kvbm_engine::{ConnectorEngineConfig, G2, build_local_connector_engine};
use kvbm_logical::BlockRegistry;
use kvbm_protocols::connector::{
    EngineWorkerSink, FenceToken, LoadOutcome, RequestId, SaveOutcome, WorkerEngineDriver,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::oneshot;
use uuid::Uuid;
use velo::{Handler, Messenger};

use crate::common::FinishedStatus;
use crate::connector::leader::sink::VeloWorkerSink;
use crate::connector::worker::velo::register_completion_handlers;
use crate::connector::worker::{FinishedRequests, WorkerCompletionState};

use super::{BS, Leader, make_request, sched_new};

mod restore;

const REQUEST: &str = "abort-onboard";
const DESTINATIONS: [usize; 2] = [10, 11];
const WAIT: Duration = Duration::from_secs(3);
const SINK_WAIT: Duration = Duration::from_secs(15);

#[derive(Deserialize)]
struct TransferId {
    transfer_id: Uuid,
}

#[derive(Deserialize)]
struct TransferRequest {
    transfer_id: Uuid,
    src: LogicalLayoutHandle,
}

struct HeldSave {
    started: oneshot::Receiver<()>,
    reply: oneshot::Sender<TransferReply>,
}

enum TransferReply {
    Completed,
    Lost,
}

/// The wire-side copy is active until the test publishes proof. Dropping the
/// unary response exercises the production Velo client and its recheck.
struct HeldOnboard {
    leader: Leader,
    completions: Arc<WorkerCompletionState>,
    copy: Arc<Mutex<Option<(Uuid, Value)>>>,
    reply: oneshot::Sender<TransferReply>,
    releases: Arc<AtomicUsize>,
    driver: Arc<dyn WorkerEngineDriver>,
    save: HeldSave,
    _server: Arc<Messenger>,
}

struct CountingSink {
    sink: Arc<VeloWorkerSink>,
    releases: Arc<AtomicUsize>,
    hold_load_delivery: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
}

impl EngineWorkerSink for CountingSink {
    fn mark_load_finished(&self, request: &RequestId, outcome: LoadOutcome) {
        self.releases.fetch_add(1, Ordering::Relaxed);
        self.sink.mark_load_finished(request, outcome);
        let gate = self.hold_load_delivery.lock().unwrap().take();
        if let Some(gate) = gate {
            gate.recv_timeout(SINK_WAIT)
                .expect("test releases the load sink callback");
        }
    }

    fn mark_save_finished(&self, request: &RequestId, outcome: SaveOutcome) {
        self.releases.fetch_add(1, Ordering::Relaxed);
        self.sink.mark_save_finished(request, outcome);
    }

    fn mark_fence_complete(&self, token: FenceToken) {
        self.sink.mark_fence_complete(token);
    }
}

impl HeldOnboard {
    async fn start() -> Result<Self> {
        Self::start_with_load_delivery(None, false).await
    }

    async fn start_with_load_delivery(
        hold_load_delivery: Option<std::sync::mpsc::Receiver<()>>,
        enable_offload: bool,
    ) -> Result<Self> {
        let pair = create_messenger_pair_tcp().await?;
        let worker_id = pair.messenger_b.instance_id();
        let incarnation = Uuid::new_v4();
        let copy = Arc::new(Mutex::new(None));
        let (started, accepted) = oneshot::channel();
        let (reply, reply_result) = oneshot::channel();
        let transfer = Arc::new(Mutex::new(Some((started, reply_result))));
        let (save_started, save_accepted) = oneshot::channel();
        let (save_reply, save_result) = oneshot::channel();
        let save = Arc::new(Mutex::new(Some((save_started, save_result))));
        let active = copy.clone();
        pair.messenger_b.register_handler(
            Handler::unary_handler_async("kvbm.worker.local_transfer", move |ctx| {
                let active = active.clone();
                let transfer = transfer.clone();
                let save = save.clone();
                async move {
                    let request: TransferRequest = serde_json::from_slice(&ctx.payload)?;
                    let gate = match request.src {
                        LogicalLayoutHandle::G1 => save.lock().unwrap().take(),
                        LogicalLayoutHandle::G2 => {
                            *active.lock().unwrap() = Some((request.transfer_id, json!("Active")));
                            transfer.lock().unwrap().take()
                        }
                        _ => bail!("unexpected transfer source {:?}", request.src),
                    };
                    let Some((started, reply_result)) = gate else {
                        bail!("only one transfer per direction should reach this worker");
                    };
                    let _ = started.send(());
                    match reply_result.await? {
                        TransferReply::Completed => Ok(Some(Bytes::from_static(b"\"Completed\""))),
                        TransferReply::Lost => bail!("worker lost the reply for the active copy"),
                    }
                }
            })
            .build(),
        )?;
        pair.messenger_b.register_handler(
            Handler::unary_handler_async("kvbm.worker.incarnation", move |_| async move {
                Ok(Some(Bytes::from(serde_json::to_vec(
                    &json!({ "incarnation": incarnation }),
                )?)))
            })
            .build(),
        )?;
        let records = copy.clone();
        pair.messenger_b.register_handler(
            Handler::unary_handler_async("kvbm.worker.local_transfer_check", move |ctx| {
                let records = records.clone();
                async move {
                    let request: TransferId = serde_json::from_slice(&ctx.payload)?;
                    let record = records
                        .lock()
                        .unwrap()
                        .as_ref()
                        .filter(|(id, _)| *id == request.transfer_id)
                        .map_or_else(|| json!("Unknown"), |(_, record)| record.clone());
                    Ok(Some(Bytes::from(serde_json::to_vec(
                        &json!({ "incarnation": incarnation, "record": record }),
                    )?)))
                }
            })
            .build(),
        )?;

        let completions = Arc::new(WorkerCompletionState::new());
        register_completion_handlers(&pair.messenger_b, completions.clone());
        let sink = VeloWorkerSink::new(
            pair.messenger_a.clone(),
            vec![worker_id],
            pair.messenger_a.runtime().clone(),
        );
        let releases = Arc::new(AtomicUsize::new(0));
        let sink = Arc::new(CountingSink {
            sink,
            releases: releases.clone(),
            hold_load_delivery: Mutex::new(hold_load_delivery),
        });
        let (leader, driver) =
            build_test_leader(pair.messenger_a, worker_id, sink, enable_offload)?;
        let external = BS * DESTINATIONS.len();
        assert_eq!(
            leader.get_num_new_matched_tokens(REQUEST, 0)?,
            (Some(external), true)
        );
        leader.allocate(REQUEST, DESTINATIONS.to_vec(), external)?;
        tokio::time::timeout(WAIT, accepted).await??;
        Ok(Self {
            leader,
            completions,
            copy,
            reply,
            releases,
            driver,
            save: HeldSave {
                started: save_accepted,
                reply: save_reply,
            },
            _server: pair.messenger_b,
        })
    }
}

fn build_test_leader(
    messenger: Arc<Messenger>,
    worker_id: velo::InstanceId,
    sink: Arc<dyn EngineWorkerSink>,
    enable_offload: bool,
) -> Result<(Leader, Arc<dyn WorkerEngineDriver>)> {
    let client = VeloWorkerClient::new(messenger.clone(), worker_id).with_local_transfer_check(
        LocalTransferCheckTiming {
            wait: Duration::ZERO,
            deadline: WAIT,
        },
    );
    let workers = Arc::new(SpmdParallelWorkers::new(
        vec![Arc::new(client)],
        Arc::new(velo::EventManager::local()),
        messenger.runtime().clone(),
    ));
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(if enable_offload { 5 } else { 2 })
            .block_size(BS)
            .build(),
    );
    let instance = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(BlockRegistry::new())
            .g2_manager(manager.clone())
            .parallel_worker(workers)
            .build()?,
    );
    let offload = if enable_offload {
        let pipeline = PipelineBuilder::new().batch_size(3).min_batch_size(3);
        Some(Arc::new(
            OffloadEngine::builder(instance.clone())
                .with_g1_to_g2_pipeline(pipeline.build())
                .build()?,
        ))
    } else {
        None
    };
    let (engine, driver) = build_local_connector_engine(
        instance,
        sink,
        ConnectorEngineConfig {
            block_size: BS,
            remote: Default::default(),
            resource_policies: Default::default(),
            resource_component_bytes: Default::default(),
        },
        offload,
    );
    let leader = Leader::with_engine(engine, BS);
    leader.create_slot(make_request(REQUEST, (0..12).collect()))?;
    let hashes = leader
        .state
        .lock()
        .get_mut(REQUEST)
        .unwrap()
        .all_sequence_hashes();
    let sources = manager.allocate_blocks(DESTINATIONS.len()).unwrap();
    let completed = sources
        .into_iter()
        .zip(hashes)
        .map(|(source, hash)| source.stage(hash, BS))
        .collect::<Result<Vec<_>, _>>()?;
    manager.register_blocks(completed);
    Ok((leader, driver))
}

async fn eventually(mut predicate: impl FnMut() -> bool) -> Result<()> {
    tokio::time::timeout(WAIT, async {
        while !predicate() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    Ok(())
}

fn assert_no_finished(completions: &WorkerCompletionState) {
    let finished = completions.drain_finished();
    assert!(finished.offloading.is_empty());
    assert!(finished.onboarding.is_empty());
    assert!(completions.drain_failed().is_empty());
}

async fn prove(
    copy: &Mutex<Option<(Uuid, Value)>>,
    failed: bool,
    completions: &WorkerCompletionState,
) -> Result<FinishedRequests> {
    copy.lock().unwrap().as_mut().unwrap().1 = if failed {
        json!({ "DrainedWithError": { "error": "destination did not arrive" } })
    } else {
        json!("Completed")
    };
    // Advance beyond the engine's pause recheck interval.
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(11)).await;
    tokio::time::resume();
    wait_for_finished(completions).await
}

async fn wait_for_finished(completions: &WorkerCompletionState) -> Result<FinishedRequests> {
    let mut finished = None;
    eventually(|| {
        let observed = completions.drain_finished();
        if observed.offloading.is_empty() && observed.onboarding.is_empty() {
            return false;
        }
        finished = Some(observed);
        true
    })
    .await?;
    Ok(finished.unwrap())
}

#[rstest::rstest]
#[case(false)]
#[case(true)]
#[tokio::test]
async fn request_finished_paused_load_releases_once_after_proof(
    #[case] failed: bool,
) -> Result<()> {
    let HeldOnboard {
        leader,
        completions,
        copy,
        reply,
        releases,
        _server,
        ..
    } = HeldOnboard::start().await?;
    assert!(reply.send(TransferReply::Lost).is_ok());
    eventually(|| {
        leader
            .state
            .lock()
            .get_mut(REQUEST)
            .unwrap()
            .onboard
            .as_ref()
            .unwrap()
            .paused()
            .is_some()
    })
    .await?;
    let worker = leader
        .state
        .lock()
        .get_mut(REQUEST)
        .unwrap()
        .onboard
        .as_ref()
        .unwrap()
        .paused()
        .unwrap()
        .worker()
        .to_owned();
    let mut log = tempfile::tempfile()?;
    let observer = tracing::Dispatch::new(
        tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .with_writer(log.try_clone()?)
            .finish(),
    );
    // Keep a second dispatch alive so parallel tests can't cache this callsite
    // as disabled while the scoped observer captures the connector events.
    let _other = tracing::Dispatch::new(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::ERROR)
            .finish(),
    );
    tracing::dispatcher::with_default(&observer, || {
        assert_eq!(leader.request_finished(REQUEST), FinishedStatus::Pending);
        leader.update_connector_output(HashSet::new(), HashSet::new())
    })?;
    assert_no_finished(&completions);
    assert_eq!(releases.load(Ordering::Relaxed), 0);
    assert!(leader.has_slot(REQUEST));

    let finished = prove(&copy, failed, &completions).await?;
    tracing::dispatcher::with_default(&observer, || {
        leader.update_connector_output(HashSet::new(), HashSet::new())
    })?;
    assert!(
        leader.has_slot(REQUEST),
        "a drained handle must remain until its worker signal arrives"
    );
    let expected = HashSet::from([REQUEST.to_owned()]);
    if failed {
        assert_eq!(finished.onboarding, expected);
        assert!(finished.offloading.is_empty());
        assert_eq!(completions.drain_failed(), HashSet::from(DESTINATIONS));
    } else {
        assert_eq!(finished.offloading, expected);
        assert!(finished.onboarding.is_empty());
    }
    tracing::dispatcher::with_default(&observer, || {
        leader.update_connector_output(finished.offloading, finished.onboarding)
    })?;
    assert!(!leader.has_slot(REQUEST));
    assert_no_finished(&completions);
    assert_eq!(releases.load(Ordering::Relaxed), 1);

    log.rewind()?;
    let mut lines = String::new();
    log.read_to_string(&mut lines)?;
    for state in ["unproven", "resolved"] {
        let events: Vec<_> = lines
            .lines()
            .filter(|line| {
                line.contains(&format!("pause_state={state:?}"))
                    && line.contains(&format!("request_id={REQUEST}"))
                    && line.contains(&format!("worker={worker}"))
            })
            .collect();
        assert_eq!(events.len(), 1, "{state} event for {REQUEST}: {lines}");
    }
    Ok(())
}

#[rstest::rstest]
#[case(false)]
#[case(true)]
#[tokio::test]
async fn request_finished_preempted_load_without_offloads_releases_once(
    #[case] holderless: bool,
) -> Result<()> {
    let HeldOnboard {
        leader,
        completions,
        copy,
        reply,
        releases,
        _server,
        ..
    } = HeldOnboard::start().await?;
    if holderless {
        // A find may have released the consumed lifecycle before preemption.
        leader.state.lock().get_mut(REQUEST).unwrap().proposal = None;
    }
    leader.on_evicted(REQUEST)?;
    assert_eq!(leader.request_finished(REQUEST), FinishedStatus::Pending);
    assert_no_finished(&completions);
    assert_eq!(releases.load(Ordering::Relaxed), 0);
    leader.update_connector_output(HashSet::new(), HashSet::new())?;
    assert!(leader.has_slot(REQUEST));
    assert!(reply.send(TransferReply::Lost).is_ok());
    eventually(|| {
        let state = leader.state.lock();
        let slot = state.get(REQUEST).unwrap();
        slot.drain_holders
            .iter()
            .filter_map(|(_, onboard, _)| onboard.as_ref())
            .chain(slot.evicted_onboards.iter().map(|(onboard, _)| onboard))
            .any(|onboard| onboard.paused().is_some())
    })
    .await?;
    assert_no_finished(&completions);
    let finished = prove(&copy, false, &completions).await?;
    assert!(finished.onboarding.is_empty());
    assert_eq!(finished.offloading, HashSet::from([REQUEST.to_owned()]));
    leader.update_connector_output(finished.offloading, finished.onboarding)?;
    assert!(!leader.has_slot(REQUEST));
    assert_no_finished(&completions);
    assert_eq!(releases.load(Ordering::Relaxed), 1);
    Ok(())
}

#[rstest::rstest]
#[case(false)]
#[case(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_finished_after_consuming_the_receive(#[case] save_pending: bool) -> Result<()> {
    let (resume, gate) = std::sync::mpsc::channel();
    let HeldOnboard {
        leader,
        completions,
        reply,
        driver,
        save,
        releases,
        _server,
        ..
    } = HeldOnboard::start_with_load_delivery(Some(gate), save_pending).await?;
    assert!(reply.send(TransferReply::Completed).is_ok());
    let received = wait_for_finished(&completions).await?;
    let request = HashSet::from([REQUEST.to_owned()]);
    assert_eq!(received.onboarding, request);
    assert!(received.offloading.is_empty());
    leader.update_connector_output(received.offloading, received.onboarding)?;

    // vLLM can schedule this request after consuming the receive, even when
    // the engine's sink callback is still returning on another thread.
    if save_pending {
        let mut blocks = DESTINATIONS.to_vec();
        blocks.push(12);
        leader.build_connector_meta(sched_new(1, REQUEST, blocks, 2 * BS, BS))?;
        assert_eq!(leader.state.lock().get(REQUEST).unwrap().offloads.len(), 1);
        driver.finish_forward_pass(1);
        tokio::time::timeout(WAIT, save.started).await??;
    }
    let finish = leader.request_finished(REQUEST);
    resume.send(())?;
    if !save_pending {
        assert_eq!(finish, FinishedStatus::Finished);
        assert!(!leader.has_slot(REQUEST));
        assert_no_finished(&completions);
        return Ok(());
    }
    assert_eq!(finish, FinishedStatus::Pending);
    assert_no_finished(&completions);
    assert!(leader.has_slot(REQUEST));

    assert!(save.reply.send(TransferReply::Completed).is_ok());
    let released = wait_for_finished(&completions)
        .await
        .expect("finish must arm the in-flight save after the load receive was consumed");
    assert_eq!(released.offloading, request);
    assert!(released.onboarding.is_empty());
    leader.update_connector_output(released.offloading, released.onboarding)?;
    assert!(!leader.has_slot(REQUEST));
    assert_no_finished(&completions);
    assert_eq!(releases.load(Ordering::Relaxed), 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_finished_before_consuming_the_receive_joins_it() -> Result<()> {
    let (resume, gate) = std::sync::mpsc::channel();
    let HeldOnboard {
        leader,
        completions,
        reply,
        releases,
        _server,
        ..
    } = HeldOnboard::start_with_load_delivery(Some(gate), false).await?;
    assert!(reply.send(TransferReply::Completed).is_ok());
    let received = wait_for_finished(&completions).await?;
    assert_eq!(received.onboarding, HashSet::from([REQUEST.to_owned()]));
    assert!(received.offloading.is_empty());

    // The engine-core loop handles an abort before updating from this frame.
    assert_eq!(leader.request_finished(REQUEST), FinishedStatus::Pending);
    assert!(leader.has_slot(REQUEST));
    assert_no_finished(&completions);
    leader.update_connector_output(received.offloading, received.onboarding)?;
    resume.send(())?;
    assert!(!leader.has_slot(REQUEST));
    assert_no_finished(&completions);
    assert_eq!(releases.load(Ordering::Relaxed), 1);
    Ok(())
}
