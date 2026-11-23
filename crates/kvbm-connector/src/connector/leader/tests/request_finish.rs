// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::io::{Read, Seek};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, bail};
use bytes::Bytes;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::testing::{TestManagerBuilder, create_messenger_pair_tcp};
use kvbm_engine::worker::VeloWorkerClient;
use kvbm_engine::worker::group::SpmdParallelWorkers;
use kvbm_engine::worker::velo::LocalTransferCheckTiming;
use kvbm_engine::{ConnectorEngineConfig, G2, build_local_connector_engine};
use kvbm_logical::BlockRegistry;
use kvbm_protocols::connector::{
    EngineWorkerSink, FenceToken, LoadOutcome, RequestId, SaveOutcome,
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

use super::{BS, Leader, make_request};

const REQUEST: &str = "abort-onboard";
const DESTINATIONS: [usize; 2] = [10, 11];
const WAIT: Duration = Duration::from_secs(3);

#[derive(Deserialize)]
struct TransferId {
    transfer_id: Uuid,
}

/// The wire-side copy is active until the test publishes proof. Dropping the
/// unary response exercises the production Velo client and its recheck.
struct HeldOnboard {
    leader: Leader,
    completions: Arc<WorkerCompletionState>,
    copy: Arc<Mutex<Option<(Uuid, Value)>>>,
    lose_reply: oneshot::Sender<()>,
    releases: Arc<AtomicUsize>,
    _server: Arc<Messenger>,
}

struct CountingSink {
    sink: Arc<VeloWorkerSink>,
    releases: Arc<AtomicUsize>,
}

impl EngineWorkerSink for CountingSink {
    fn mark_load_finished(&self, request: &RequestId, outcome: LoadOutcome) {
        self.releases.fetch_add(1, Ordering::Relaxed);
        self.sink.mark_load_finished(request, outcome);
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
        let pair = create_messenger_pair_tcp().await?;
        let worker_id = pair.messenger_b.instance_id();
        let incarnation = Uuid::new_v4();
        let copy = Arc::new(Mutex::new(None));
        let (started, accepted) = oneshot::channel();
        let (lose_reply, lost) = oneshot::channel();
        let transfer = Arc::new(Mutex::new(Some((started, lost))));
        let active = copy.clone();
        pair.messenger_b.register_handler(
            Handler::unary_handler_async("kvbm.worker.local_transfer", move |ctx| {
                let active = active.clone();
                let transfer = transfer.lock().unwrap().take();
                async move {
                    let request: TransferId = serde_json::from_slice(&ctx.payload)?;
                    let Some((started, lost)) = transfer else {
                        bail!("only one transfer should reach this worker");
                    };
                    *active.lock().unwrap() = Some((request.transfer_id, json!("Active")));
                    let _ = started.send(());
                    lost.await?;
                    bail!("worker lost the reply for the active copy")
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
        });
        let client = VeloWorkerClient::new(pair.messenger_a.clone(), worker_id)
            .with_local_transfer_check(LocalTransferCheckTiming {
                wait: Duration::ZERO,
                deadline: WAIT,
            });
        let workers = Arc::new(SpmdParallelWorkers::new(
            vec![Arc::new(client)],
            Arc::new(velo::EventManager::local()),
            pair.messenger_a.runtime().clone(),
        ));
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(2)
                .block_size(BS)
                .build(),
        );
        let instance = InstanceLeader::builder()
            .messenger(pair.messenger_a)
            .registry(BlockRegistry::new())
            .g2_manager(manager.clone())
            .parallel_worker(workers)
            .build()?;
        let (engine, _) = build_local_connector_engine(
            Arc::new(instance),
            sink,
            ConnectorEngineConfig {
                block_size: BS,
                remote: Default::default(),
                resource_policies: Default::default(),
                resource_component_bytes: Default::default(),
            },
            None,
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
        let _sources = manager.register_blocks(completed);
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
            lose_reply,
            releases,
            _server: pair.messenger_b,
        })
    }
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
        lose_reply,
        releases,
        _server,
    } = HeldOnboard::start().await?;
    assert!(lose_reply.send(()).is_ok());
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
        lose_reply,
        releases,
        _server,
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
    assert!(lose_reply.send(()).is_ok());
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
