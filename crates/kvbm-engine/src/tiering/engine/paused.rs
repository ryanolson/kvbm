// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Onboard ownership while a submitted copy has no drain proof.

mod metrics;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use kvbm_physical::transfer::TransferDrainOutcome;
use kvbm_protocols::connector::{
    ActionFailure, ActionId, ActionStatus, RequestId, ResourceOnboard,
};
use parking_lot::Mutex as RegistryMutex;
use tokio::time::{Instant, sleep_until};
use velo::InstanceId;

use crate::remote::cd::{prefill::PrefillRequestState, state::CdRequestState};
use crate::worker::velo::{
    CheckEvidence, DEFAULT_LOCAL_TRANSFER_CHECK_DEADLINE, LocalTransferWorker,
};
use crate::worker::{LocalTransferRecheck, UnresolvedTransfer};

use super::local::LocalConnectorEngine;
use super::onboard::include_undispatched_destinations;
use metrics::PauseMetrics;

pub(super) const PROBE_INTERVAL: Duration = Duration::from_secs(10);
const PROBE_SCHEDULING_MARGIN: Duration = Duration::from_secs(1);
pub(super) type PauseCell = Arc<Mutex<Option<(std::time::Instant, String)>>>;

/// Everything the driver must retain until all copies have stopped. A pause
/// holds memory and blocks until proof or restart. vLLM preemption waits on the
/// eviction fence, which stops that worker's forward pass. During process
/// exit, the OS tears down the holds with the process. Local workers die with
/// it or must restart before the memory can be reused.
pub(super) struct PausedLoad {
    pub(super) action: ActionId,
    pub(super) request: RequestId,
    pub(super) destinations: Vec<usize>,
    pub(super) physical_drain: Arc<AtomicBool>,
    pub(super) pause: PauseCell,
    pub(super) holds: Box<dyn Send>,
    pub(super) continuation: Option<LoadContinuation>,
}

pub(super) enum LoadContinuation {
    Resource(ResourceContinuation),
    Bundle(BundleContinuation),
    CdDecode(CdDecodeContinuation),
    CdPrefill(Arc<PrefillRequestState>),
}

/// The current resource and the work that has yet to reach admission.
pub(super) struct ResourceContinuation {
    pub(super) failure: ActionFailure,
    pub(super) remaining: std::vec::IntoIter<ResourceOnboard>,
}

/// The current resource's failure and the destinations no copy ever reached.
pub(super) struct BundleContinuation {
    pub(super) failure: ActionFailure,
    pub(super) unfilled: Vec<usize>,
    pub(super) logical_done: bool,
}

impl BundleContinuation {
    fn outcome(self, completed: bool) -> Option<ActionStatus> {
        if self.logical_done {
            return None;
        }
        let status = if completed {
            ActionStatus::Complete
        } else {
            ActionStatus::Failed(self.failure)
        };
        Some(include_undispatched_destinations(status, self.unfilled))
    }
}

/// The originating CD lifecycle and the destinations still missing at pause.
pub(super) struct CdDecodeContinuation {
    pub(super) state: Arc<CdRequestState>,
    pub(super) unfilled: Vec<usize>,
    pub(super) pending_copy: HashSet<usize>,
}

impl CdDecodeContinuation {
    pub(super) fn new(state: Arc<CdRequestState>) -> Self {
        Self {
            state,
            unfilled: Vec::new(),
            pending_copy: HashSet::new(),
        }
    }

    fn outcome(&self, completed: bool) -> ActionStatus {
        // A pause ends CD dispatch. Proof of completion preserves the copied
        // ids. A drained error leaves them in the set vLLM will recompute.
        let block_ids: Vec<_> = self
            .unfilled
            .iter()
            .copied()
            .filter(|id| !completed || !self.pending_copy.contains(id))
            .collect();
        if block_ids.is_empty() {
            ActionStatus::Complete
        } else {
            ActionStatus::Failed(ActionFailure::Partial { block_ids })
        }
    }
}

struct PausedRecord {
    load: PausedLoad,
    since: Instant,
    unresolved: UnresolvedTransfer,
    participants: Vec<Participant>,
}

impl PausedRecord {
    fn publish_to_handle(&self) {
        let worker = self.participants[0].worker.to_string();
        *self.load.pause.lock().expect("pause mutex poisoned") =
            Some((self.since.into_std(), worker));
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Worker {
    Velo(InstanceId, Option<u64>),
    Unknown,
}

impl Worker {
    fn of(recheck: &LocalTransferRecheck) -> Self {
        Self::Velo(recheck.instance, recheck.worker_id)
    }
}

impl fmt::Display for Worker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Velo(instance, Some(worker)) => write!(f, "{instance}#{worker}"),
            Self::Velo(instance, None) => write!(f, "{instance}"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum PauseState {
    WorkerNotAnswering,
    WorkerNoProof,
    NoRecoveryCapability,
}

impl PauseState {
    const ALL: [Self; 3] = [
        Self::WorkerNotAnswering,
        Self::WorkerNoProof,
        Self::NoRecoveryCapability,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::WorkerNotAnswering => "worker_not_answering",
            Self::WorkerNoProof => "worker_no_proof",
            Self::NoRecoveryCapability => "no_recovery_capability",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::WorkerNotAnswering => "waiting for the worker to answer",
            Self::WorkerNoProof => "fatal pause until drain proof or restart",
            Self::NoRecoveryCapability => "fatal pause until restart",
        }
    }
}

struct Participant {
    worker: Worker,
    recheck: Option<LocalTransferRecheck>,
    deadline: Duration,
    state: PauseState,
    last_attempt: Option<Instant>,
    last_answer: Option<Instant>,
}

impl Participant {
    fn new(recheck: Option<LocalTransferRecheck>, error: &anyhow::Error) -> Self {
        let worker = recheck
            .as_ref()
            .map(Worker::of)
            .or_else(|| {
                let origin = error.downcast_ref::<LocalTransferWorker>()?;
                Some(Worker::Velo(origin.instance, origin.worker_id))
            })
            .unwrap_or(Worker::Unknown);
        let recheck = recheck.filter(|check| check.incarnation.is_some());
        let evidence = CheckEvidence::from_error(error);
        let state = match (&recheck, evidence.answered) {
            (None, _) => PauseState::NoRecoveryCapability,
            (_, Some(_)) => PauseState::WorkerNoProof,
            (_, None) => PauseState::WorkerNotAnswering,
        };
        Self {
            worker,
            recheck,
            deadline: evidence
                .deadline
                .unwrap_or(DEFAULT_LOCAL_TRANSFER_CHECK_DEADLINE),
            state,
            last_attempt: evidence.attempted,
            last_answer: evidence.answered,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct WorkerSummary {
    count: usize,
    oldest: Option<Instant>,
}

impl WorkerSummary {
    fn add(&mut self, since: Instant) {
        self.count += 1;
        self.oldest = Some(self.oldest.map_or(since, |oldest| oldest.min(since)));
    }
}

struct Probe {
    last_attempt: Option<Instant>,
    deadline: Duration,
    stuck: bool,
    #[cfg(test)]
    task: Option<tokio::task::AbortHandle>,
}

impl Probe {
    fn new(deadline: Duration) -> Self {
        Self {
            last_attempt: None,
            deadline,
            stuck: false,
            #[cfg(test)]
            task: None,
        }
    }

    fn stuck_after(&self) -> Duration {
        // A silent check can use its entire deadline before the loop sleeps.
        PROBE_INTERVAL
            .saturating_add(self.deadline)
            .max(PROBE_INTERVAL * 2)
            .saturating_add(PROBE_SCHEDULING_MARGIN)
    }
}

#[derive(Default)]
struct Registry {
    records: HashMap<ActionId, PausedRecord>,
    probes: HashMap<Worker, Probe>,
    watchdog_running: bool,
    teardown_reported: bool,
    // A last public handle drop cannot release memory a worker may still touch.
    pin: Option<Arc<LocalConnectorEngine>>,
}

impl Registry {
    fn summarize(&self) -> HashMap<Worker, [WorkerSummary; 3]> {
        let mut workers: HashMap<Worker, [WorkerSummary; 3]> = HashMap::new();
        for record in self.records.values() {
            let states: HashSet<_> = record
                .participants
                .iter()
                .map(|part| (part.worker, part.state))
                .collect();
            for (worker, state) in states {
                workers.entry(worker).or_default()[state as usize].add(record.since);
            }
        }
        workers
    }

    fn report_teardown(&mut self, engine: &Arc<LocalConnectorEngine>, now: Instant) {
        // The observer and the registry pin account for these two owners.
        if self.teardown_reported || Arc::strong_count(engine) != 2 {
            return;
        }
        self.teardown_reported = true;
        let summary = self.summarize();
        let mut workers = Vec::new();
        let mut oldest = Duration::ZERO;
        for (worker, states) in summary {
            for (state, summary) in PauseState::ALL.into_iter().zip(states) {
                if let Some(started) = summary.oldest {
                    oldest = oldest.max(now - started);
                    workers.push((worker.to_string(), state.label(), summary.count));
                }
            }
        }
        tracing::error!(
            count = self.records.len(),
            oldest_pause_seconds = oldest.as_secs_f64(),
            ?workers,
            "connector owners dropped with paused onboards",
        );
    }
}

pub(super) struct PausedOnboards {
    inner: RegistryMutex<Registry>,
    metrics: Option<PauseMetrics>,
}

impl PausedOnboards {
    pub(super) fn new(registry: Option<&prometheus::Registry>) -> Self {
        let metrics = registry.and_then(|registry| {
            let engine = uuid::Uuid::new_v4().to_string();
            match PauseMetrics::new(registry, &engine) {
                Ok(metrics) => Some(metrics),
                Err(error) => {
                    tracing::error!(%error, "could not register onboard pause metrics");
                    None
                }
            }
        });
        Self {
            inner: RegistryMutex::default(),
            metrics,
        }
    }

    fn publish(&self, inner: &Registry, workers: impl IntoIterator<Item = Worker>) {
        if let Some(metrics) = &self.metrics {
            let summaries = inner.summarize();
            for worker in workers {
                let summary = summaries.get(&worker).copied().unwrap_or_default();
                metrics.publish_pauses(&worker.to_string(), &summary);
            }
        }
    }

    pub(super) fn insert(
        &self,
        engine: &Arc<LocalConnectorEngine>,
        load: PausedLoad,
        error: anyhow::Error,
    ) {
        let since = Instant::now();
        let unresolved = UnresolvedTransfer::from_unproven(error);
        let participants = unresolved
            .participants()
            .iter()
            .map(|part| Participant::new(part.recheck().cloned(), part.error()))
            .collect();
        let record = PausedRecord {
            load,
            since,
            unresolved,
            participants,
        };
        let mut inner = self.inner.lock();
        assert!(
            !inner.records.contains_key(&record.load.action),
            "onboard paused twice"
        );
        if inner.records.is_empty() {
            inner.pin = Some(engine.clone());
            inner.teardown_reported = false;
        }
        let mut workers = HashSet::new();
        let mut start = Vec::new();
        for participant in &record.participants {
            let worker = participant.worker;
            let state = participant.state;
            tracing::warn!(
                request_id = %record.load.request,
                action_id = ?record.load.action,
                %worker,
                state = state.label(),
                reason = state.description(),
                "onboard paused",
            );
            workers.insert(worker);
            if let Some(metrics) = &self.metrics
                && let Some(attempted) = participant.last_attempt
            {
                metrics.attempt(&worker.to_string(), attempted);
                metrics.check(&worker.to_string(), state.label(), participant.last_answer);
            }
            if participant.recheck.is_some()
                && let std::collections::hash_map::Entry::Vacant(entry) = inner.probes.entry(worker)
            {
                entry.insert(Probe::new(participant.deadline));
                start.push(worker);
            }
        }
        record.publish_to_handle();
        inner.records.insert(record.load.action, record);
        self.publish(&inner, workers);

        let runtime = engine.leader.runtime();
        for worker in start {
            let task = runtime.spawn(probe_worker(
                Arc::downgrade(engine),
                worker,
                since + PROBE_INTERVAL,
            ));
            #[cfg(test)]
            {
                inner.probes.get_mut(&worker).unwrap().task = Some(task.abort_handle());
            }
            drop(task);
        }
        if !inner.watchdog_running {
            inner.watchdog_running = true;
            runtime.spawn(watch_pauses(Arc::downgrade(engine), since + PROBE_INTERVAL));
        }
    }

    fn probe_targets(&self, worker: Worker) -> Vec<(ActionId, LocalTransferRecheck)> {
        let mut inner = self.inner.lock();
        let mut targets = Vec::new();
        for (action, record) in &inner.records {
            targets.extend(record.participants.iter().filter_map(|participant| {
                (participant.worker == worker)
                    .then_some(participant.recheck.as_ref())
                    .flatten()
                    .map(|check| (*action, check.clone()))
            }));
        }
        if targets.is_empty() {
            inner.probes.remove(&worker);
        }
        targets.sort_unstable_by_key(|(action, check)| (action.0, check.transfer_id));
        targets
    }

    fn start_check(
        &self,
        worker: Worker,
        action: ActionId,
        check: &LocalTransferRecheck,
    ) -> Option<Duration> {
        let mut inner = self.inner.lock();
        let now = Instant::now();
        let participant = inner
            .records
            .get_mut(&action)?
            .participants
            .iter_mut()
            .find(|part| part.recheck.as_ref() == Some(check))?;
        participant.last_attempt = Some(now);
        let deadline = participant.deadline;
        let probe = inner.probes.get_mut(&worker).expect("registered probe");
        probe.last_attempt = Some(now);
        probe.deadline = deadline;
        probe.stuck = false;
        if let Some(metrics) = &self.metrics {
            metrics.attempt(&worker.to_string(), now);
        }
        Some(deadline)
    }

    fn apply_check(
        &self,
        engine: &Arc<LocalConnectorEngine>,
        action: ActionId,
        check: &LocalTransferRecheck,
        outcome: TransferDrainOutcome,
    ) -> (Option<PausedRecord>, bool) {
        let worker = Worker::of(check);
        let (result, answered, state) = match &outcome {
            TransferDrainOutcome::Completed => ("completed", Some(Instant::now()), None),
            TransferDrainOutcome::DrainedWithError(_) => {
                ("drained_with_error", Some(Instant::now()), None)
            }
            TransferDrainOutcome::Unproven(error) => {
                let answered = CheckEvidence::from_error(error).answered;
                let state = if answered.is_some() {
                    PauseState::WorkerNoProof
                } else {
                    PauseState::WorkerNotAnswering
                };
                (state.label(), answered, Some(state))
            }
        };
        let mut inner = self.inner.lock();
        inner.report_teardown(engine, Instant::now());
        if let Some(metrics) = &self.metrics {
            metrics.check(&worker.to_string(), result, answered);
        }
        let Some(record) = inner.records.get_mut(&action) else {
            return (None, answered.is_some());
        };
        let Some(index) = record
            .participants
            .iter()
            .position(|part| part.recheck.as_ref() == Some(check))
        else {
            return (None, answered.is_some());
        };
        assert!(
            record.unresolved.apply(check, outcome),
            "tracked participant is unresolved"
        );
        if let Some(state) = state {
            let part = &mut record.participants[index];
            part.state = state;
            part.last_answer = answered.or(part.last_answer);
        } else {
            record.participants.remove(index);
        }

        let settled = if record.unresolved.is_resolved() {
            inner.records.remove(&action)
        } else {
            record.publish_to_handle();
            None
        };
        self.publish(&inner, [worker]);
        let pin = if inner.records.is_empty() {
            inner.pin.take()
        } else {
            None
        };
        drop(inner);
        drop(pin);
        (settled, answered.is_some())
    }

    #[cfg(test)]
    pub(super) fn stop_probes(&self) {
        for probe in self.inner.lock().probes.values() {
            probe.task.as_ref().unwrap().abort();
        }
    }

    #[cfg(test)]
    pub(super) fn teardown_reported(&self) -> bool {
        self.inner.lock().teardown_reported
    }
}

impl LocalConnectorEngine {
    /// Park an unproven load whose action already has a physical fence.
    pub(super) fn pause_load_action(self: &Arc<Self>, load: PausedLoad, error: anyhow::Error) {
        self.paused_onboards.insert(self, load, error);
    }

    fn resolve_paused_load(self: &Arc<Self>, record: PausedRecord) {
        let mut load = record.load;
        *load.pause.lock().expect("pause mutex poisoned") = None;
        let (completed, result) = match record.unresolved.into_outcome() {
            TransferDrainOutcome::Completed => (true, "completed"),
            TransferDrainOutcome::DrainedWithError(error) => {
                tracing::debug!(request_id = %load.request, %error, "paused onboard proved drained with error");
                (false, "drained_with_error")
            }
            TransferDrainOutcome::Unproven(_) => unreachable!("unresolved onboard cannot settle"),
        };
        let status = match load.continuation.take() {
            Some(LoadContinuation::Bundle(bundle)) => bundle.outcome(completed),
            Some(LoadContinuation::CdDecode(cd)) => {
                let status = cd.outcome(completed);
                load.continuation = Some(LoadContinuation::CdDecode(cd));
                Some(status)
            }
            Some(LoadContinuation::Resource(resource))
                if completed && resource.remaining.len() != 0 =>
            {
                tracing::info!(
                    request_id = %load.request,
                    action_id = ?load.action,
                    pause_seconds = record.since.elapsed().as_secs_f64(),
                    "onboard pause cleared, resuming remaining resources",
                );
                self.spawn_resource_onboard(load, resource.remaining);
                return;
            }
            Some(LoadContinuation::Resource(resource)) if !completed => {
                Some(include_undispatched_destinations(
                    ActionStatus::Failed(resource.failure),
                    resource
                        .remaining
                        .flat_map(|transfer| transfer.destination_block_ids),
                ))
            }
            continuation => {
                load.continuation = continuation;
                Some(if completed {
                    ActionStatus::Complete
                } else {
                    ActionStatus::Failed(ActionFailure::AllBlocks)
                })
            }
        };
        let action = load.action;
        let request = load.request.clone();
        load.finish(self, status);
        tracing::info!(
            request_id = %request,
            action_id = ?action,
            result,
            pause_seconds = record.since.elapsed().as_secs_f64(),
            "onboard pause cleared",
        );
    }
}

async fn probe_worker(engine: Weak<LocalConnectorEngine>, worker: Worker, mut next: Instant) {
    let mut round = 0usize;
    loop {
        sleep_until(next).await;
        let mut targets = match engine.upgrade() {
            Some(engine) => engine.paused_onboards.probe_targets(worker),
            None => return,
        };
        if targets.is_empty() {
            return;
        }
        // If the worker is silent, take a different transfer first next time.
        let shift = round % targets.len();
        targets.rotate_left(shift);
        round = round.wrapping_add(1);
        for (action, check) in targets {
            let (messenger, deadline) = {
                let Some(engine) = engine.upgrade() else {
                    return;
                };
                let Some(deadline) = engine.paused_onboards.start_check(worker, action, &check)
                else {
                    continue;
                };
                (engine.leader.messenger().clone(), deadline)
            };
            let outcome = check.check(&messenger, deadline).await;
            let Some(engine) = engine.upgrade() else {
                return;
            };
            let (settled, answered) = engine
                .paused_onboards
                .apply_check(&engine, action, &check, outcome);
            if let Some(record) = settled {
                engine.resolve_paused_load(record);
            }
            if !answered {
                break;
            }
        }
        next = Instant::now() + PROBE_INTERVAL;
    }
}

async fn watch_pauses(engine: Weak<LocalConnectorEngine>, mut next: Instant) {
    loop {
        sleep_until(next).await;
        let Some(engine) = engine.upgrade() else {
            return;
        };
        let paused = &engine.paused_onboards;
        let now = Instant::now();
        next = now + PROBE_INTERVAL;
        let (summaries, stuck) = {
            let mut inner = paused.inner.lock();
            if inner.records.is_empty() {
                inner.watchdog_running = false;
                return;
            }

            inner.report_teardown(&engine, now);
            let summaries = inner.summarize();
            let mut stuck = Vec::new();
            for (worker, probe) in &mut inner.probes {
                let Some(states) = summaries.get(worker) else {
                    continue;
                };
                let Some(oldest) = states[..2].iter().filter_map(|state| state.oldest).min() else {
                    continue;
                };
                let reference = probe.last_attempt.map_or(oldest, |last| last.max(oldest));
                if !probe.stuck && now - reference >= probe.stuck_after() {
                    probe.stuck = true;
                    if let Some(metrics) = &paused.metrics {
                        metrics.stuck(&worker.to_string());
                    }
                    stuck.push((*worker, now - reference));
                }
            }
            (summaries, stuck)
        };
        for (worker, elapsed) in stuck {
            tracing::error!(%worker, since_attempt_seconds = elapsed.as_secs_f64(), "onboard probe is stuck");
        }
        for (worker, states) in summaries {
            for (state, summary) in PauseState::ALL.into_iter().zip(states) {
                if let Some(oldest) = summary.oldest {
                    tracing::warn!(
                        %worker,
                        state = state.label(),
                        reason = state.description(),
                        count = summary.count,
                        oldest_pause_seconds = (now - oldest).as_secs_f64(),
                        "onboards remain paused",
                    );
                }
            }
        }
    }
}

#[cfg(test)]
#[test]
fn pause_without_any_worker_identity_is_unrecoverable() {
    // Velo always attaches the reporting worker to a lost transfer reply.
    let error = anyhow::anyhow!("in-process drain failed");
    let participant = Participant::new(None, &error);
    assert_eq!(participant.worker, Worker::Unknown);
    assert_eq!(participant.state, PauseState::NoRecoveryCapability);
    assert!(participant.recheck.is_none());
    assert!(participant.last_attempt.is_none());
}
