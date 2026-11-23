// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The check of a local transfer whose reply was lost.
//!
//! See the local-transfer contract in the module docs of [`super`].

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ::velo::{InstanceId, Messenger, RequestTimeout};
use bytes::Bytes;
use kvbm_physical::transfer::TransferDrainOutcome;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::time::Instant as ClockInstant;
use uuid::Uuid;

use super::handler_names;

/// The default time the leader waits after a lost reply before it checks.
pub const DEFAULT_LOCAL_TRANSFER_CHECK_WAIT: Duration = Duration::from_secs(2);
/// The default deadline of one check, and of the incarnation request.
pub const DEFAULT_LOCAL_TRANSFER_CHECK_DEADLINE: Duration = Duration::from_secs(5);

/// The most records a worker keeps. Older finished records are evicted first.
pub(super) const RECORD_CAPACITY: usize = 65_536;
/// How long a worker keeps a finished record.
pub(super) const RECORD_MAX_AGE: Duration = Duration::from_secs(600);

/// How long the leader waits after a lost reply, and how long one check may
/// take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalTransferCheckTiming {
    /// The time between the lost reply and the check.
    pub wait: Duration,
    /// The deadline of the check RPC, and of the incarnation request.
    pub deadline: Duration,
}

impl Default for LocalTransferCheckTiming {
    fn default() -> Self {
        Self {
            wait: DEFAULT_LOCAL_TRANSFER_CHECK_WAIT,
            deadline: DEFAULT_LOCAL_TRANSFER_CHECK_DEADLINE,
        }
    }
}

impl From<&kvbm_config::LocalTransferCheckConfig> for LocalTransferCheckTiming {
    fn from(config: &kvbm_config::LocalTransferCheckConfig) -> Self {
        Self {
            wait: Duration::from_millis(config.wait_ms),
            deadline: Duration::from_millis(config.deadline_ms),
        }
    }
}

/// What a lost local transfer needs to be checked again: the worker that took
/// the copy, the worker incarnation the client bound before it sent the copy,
/// and the transfer id.
///
/// It is attached as context to the error of a
/// [`TransferDrainOutcome::Unproven`] receipt whose reply was lost. Find it
/// with `error.downcast_ref::<LocalTransferRecheck>()`, and repeat the check
/// with [`Self::check`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalTransferRecheck {
    /// The velo instance of the worker that took the copy.
    pub instance: InstanceId,
    /// The worker id within that process, if it serves two or more workers.
    pub worker_id: Option<u64>,
    /// The incarnation bound before the copy. `None` when the client could not
    /// learn it; a check then proves nothing.
    pub incarnation: Option<Uuid>,
    /// The id of the transfer.
    pub transfer_id: Uuid,
}

impl fmt::Display for LocalTransferRecheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "local transfer {} on worker {}",
            self.transfer_id, self.instance
        )?;
        if let Some(worker_id) = self.worker_id {
            write!(f, "#{worker_id}")?;
        }
        match self.incarnation {
            Some(incarnation) => write!(f, " (incarnation {incarnation}) has no drain proof"),
            None => write!(f, " (incarnation unknown) has no drain proof"),
        }
    }
}

/// Timing of the check that produced an unproven outcome.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CheckEvidence {
    pub(crate) attempted: Option<ClockInstant>,
    pub(crate) answered: Option<ClockInstant>,
    pub(crate) deadline: Option<Duration>,
}

#[derive(Debug, thiserror::Error)]
#[error("{reason}")]
struct UnprovenCheck {
    evidence: CheckEvidence,
    reason: String,
}

enum CheckAnswer {
    Reply(Bytes),
    MissingHandler,
}

#[derive(Deserialize)]
struct HandlerList {
    handlers: Vec<String>,
}

impl CheckEvidence {
    pub(crate) fn from_error(error: &anyhow::Error) -> Self {
        error
            .downcast_ref::<UnprovenCheck>()
            .map_or_else(Self::default, |check| check.evidence)
    }
}

impl LocalTransferRecheck {
    /// Ask the captured worker once, under `deadline`, whether the transfer
    /// drained.
    ///
    /// Only a record from the same incarnation decides: "completed" gives
    /// [`TransferDrainOutcome::Completed`], "drained with error" gives
    /// [`TransferDrainOutcome::DrainedWithError`]. Every other answer, and no
    /// answer, gives [`TransferDrainOutcome::Unproven`] with this capability
    /// attached again. Without a bound incarnation no answer could decide, so
    /// no check is sent.
    pub async fn check(&self, messenger: &Messenger, deadline: Duration) -> TransferDrainOutcome {
        let Some(incarnation) = self.incarnation else {
            return self.unbound();
        };
        let mut evidence = CheckEvidence {
            attempted: Some(ClockInstant::now()),
            answered: None,
            deadline: Some(deadline),
        };
        // The outer timeout also bounds any handshake velo runs first, which
        // the request's own timeout may not cover.
        match tokio::time::timeout(deadline, self.ask(messenger, deadline)).await {
            Ok(Ok(answer)) => {
                evidence.answered = Some(ClockInstant::now());
                match answer {
                    CheckAnswer::Reply(bytes) => match serde_json::from_slice(&bytes) {
                        Ok(reply) => self.decide(incarnation, reply, evidence),
                        Err(error) => {
                            self.unproven(evidence, format!("invalid check reply: {error}"))
                        }
                    },
                    CheckAnswer::MissingHandler => self.unproven(
                        evidence,
                        "the reachable worker has no transfer check handler",
                    ),
                }
            }
            Ok(Err(error)) => {
                self.unproven(evidence, format!("the check got no answer: {error:#}"))
            }
            Err(_) => self.unproven(
                evidence,
                format!("the check got no answer within {deadline:?}"),
            ),
        }
    }

    async fn ask(&self, messenger: &Messenger, deadline: Duration) -> anyhow::Result<CheckAnswer> {
        let handler = handler_names::scoped(handler_names::LOCAL_TRANSFER_CHECK, self.worker_id);
        let payload = serde_json::to_vec(&CheckMessage {
            transfer_id: self.transfer_id,
        })?;
        let result = messenger
            .unary(&handler)?
            .raw_payload(Bytes::from(payload))
            .instance(self.instance)
            .timeout(deadline)
            .send()
            .await;
        let error = match result {
            Ok(bytes) => return Ok(CheckAnswer::Reply(bytes)),
            Err(error) if error.is::<RequestTimeout>() => return Err(error),
            Err(error) => error,
        };

        // Velo flattens remote errors and transport errors into the same result.
        // Only a fresh list that lacks the handler proves the peer answered.
        // The caller's outer timeout covers both requests.
        let list = messenger
            .typed_unary_streaming::<HandlerList>("_list_handlers")
            .payload(())?
            .instance(self.instance)
            .send()
            .await;
        match list {
            Ok(list) if !list.handlers.contains(&handler) => Ok(CheckAnswer::MissingHandler),
            _ => Err(error),
        }
    }

    fn decide(
        &self,
        incarnation: Uuid,
        reply: CheckReply,
        evidence: CheckEvidence,
    ) -> TransferDrainOutcome {
        if reply.incarnation != incarnation {
            return self.unproven(
                evidence,
                format!(
                    "the check reached incarnation {}, not the one that took the copy",
                    reply.incarnation
                ),
            );
        }
        match reply.record {
            Record::Completed => TransferDrainOutcome::Completed,
            Record::DrainedWithError { error } => {
                TransferDrainOutcome::DrainedWithError(anyhow::anyhow!(error))
            }
            Record::Active => self.unproven(evidence, "the worker reports the copy still active"),
            Record::Unproven { error } => {
                self.unproven(evidence, format!("the worker has no drain proof: {error}"))
            }
            Record::Unknown => self.unproven(evidence, "the worker has no record of the transfer"),
        }
    }

    /// The outcome of a transfer with no bound incarnation: no check can
    /// decide it, so none is sent.
    pub(super) fn unbound(&self) -> TransferDrainOutcome {
        self.unproven(
            CheckEvidence::default(),
            "the worker incarnation was not bound before the copy",
        )
    }

    /// An unknown outcome that carries this capability.
    pub(super) fn unproven(
        &self,
        evidence: CheckEvidence,
        reason: impl fmt::Display,
    ) -> TransferDrainOutcome {
        TransferDrainOutcome::Unproven(
            anyhow::Error::new(UnprovenCheck {
                evidence,
                reason: reason.to_string(),
            })
            .context(self.clone()),
        )
    }
}

/// The `LOCAL_TRANSFER_CHECK` request.
#[derive(Serialize, Deserialize)]
pub(super) struct CheckMessage {
    pub(super) transfer_id: Uuid,
}

/// The `LOCAL_TRANSFER_CHECK` reply: the answering incarnation and its record.
#[derive(Serialize, Deserialize)]
pub(super) struct CheckReply {
    pub(super) incarnation: Uuid,
    pub(super) record: Record,
}

/// The `WORKER_INCARNATION` reply.
#[derive(Serialize, Deserialize)]
pub(super) struct IncarnationReply {
    pub(super) incarnation: Uuid,
}

/// A worker's record of one transfer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) enum Record {
    /// The request arrived; the copy may run.
    Active,
    /// The copy completed.
    Completed,
    /// The copy failed, and no physical operation is active.
    DrainedWithError { error: String },
    /// The copy failed without proof that its work drained.
    Unproven { error: String },
    /// No record: never seen, or evicted.
    Unknown,
}

impl Record {
    fn from_outcome(outcome: &TransferDrainOutcome) -> Self {
        match outcome {
            TransferDrainOutcome::Completed => Self::Completed,
            TransferDrainOutcome::DrainedWithError(error) => Self::DrainedWithError {
                error: format!("{error:#}"),
            },
            TransferDrainOutcome::Unproven(error) => Self::Unproven {
                error: format!("{error:#}"),
            },
        }
    }

    fn is_terminal(&self) -> bool {
        !matches!(self, Self::Active)
    }
}

struct Entry {
    record: Record,
    /// When the record last changed.
    at: Instant,
}

#[derive(Default)]
struct Registry {
    entries: HashMap<Uuid, Entry>,
    /// Ids in arrival order, used to evict an active record when no finished
    /// record is left. Ids whose entry is gone or finished are skipped.
    arrivals: VecDeque<Uuid>,
    /// Finished ids in the order they finished, with the time each finished.
    /// An id whose entry changed since is skipped.
    finished: VecDeque<(Uuid, Instant)>,
}

impl Registry {
    fn is_live_finished(&self, id: &Uuid, at: Instant) -> bool {
        self.entries
            .get(id)
            .is_some_and(|entry| entry.record.is_terminal() && entry.at == at)
    }

    /// Drop finished records older than `max_age`.
    fn age_out(&mut self, now: Instant, max_age: Duration) {
        while let Some(&(id, at)) = self.finished.front() {
            if now.duration_since(at) < max_age {
                break;
            }
            self.finished.pop_front();
            if self.is_live_finished(&id, at) {
                self.entries.remove(&id);
            }
        }
    }

    /// Drop one record: the oldest finished record, or else the oldest active
    /// one.
    fn evict_one(&mut self) {
        while let Some((id, at)) = self.finished.pop_front() {
            if self.is_live_finished(&id, at) {
                self.entries.remove(&id);
                return;
            }
        }
        while let Some(id) = self.arrivals.pop_front() {
            if self.entries.remove(&id).is_some() {
                return;
            }
        }
    }

    /// Keep `arrivals` near the number of active records.
    fn compact_arrivals(&mut self, capacity: usize) {
        while self.arrivals.front().is_some_and(|id| {
            self.entries
                .get(id)
                .is_none_or(|entry| entry.record.is_terminal())
        }) {
            self.arrivals.pop_front();
        }
        if self.arrivals.len() > capacity.saturating_mul(2) {
            let entries = &self.entries;
            self.arrivals.retain(|id| {
                entries
                    .get(id)
                    .is_some_and(|entry| !entry.record.is_terminal())
            });
        }
    }
}

/// One worker's transfer records, and the incarnation they belong to.
///
/// The incarnation is a random id fixed when the worker's service starts.
/// The registry is bounded by [`RECORD_CAPACITY`] and [`RECORD_MAX_AGE`].
pub(super) struct LocalTransferRecords {
    incarnation: Uuid,
    registry: Mutex<Registry>,
    capacity: usize,
    max_age: Duration,
    #[cfg(test)]
    pub(super) hooks: TestHooks,
}

impl LocalTransferRecords {
    pub(super) fn new() -> Arc<Self> {
        Self::with_incarnation(Uuid::new_v4())
    }

    pub(super) fn with_incarnation(incarnation: Uuid) -> Arc<Self> {
        Arc::new(Self {
            incarnation,
            registry: Mutex::new(Registry::default()),
            capacity: RECORD_CAPACITY,
            max_age: RECORD_MAX_AGE,
            #[cfg(test)]
            hooks: TestHooks::default(),
        })
    }

    pub(super) fn incarnation(&self) -> Uuid {
        self.incarnation
    }

    /// Record `transfer_id` as active. Call before the copy is dispatched.
    pub(super) fn begin(&self, transfer_id: Uuid) {
        let now = Instant::now();
        let mut registry = self.registry.lock();
        registry.age_out(now, self.max_age);
        registry.compact_arrivals(self.capacity);
        while registry.entries.len() >= self.capacity {
            registry.evict_one();
        }
        registry.entries.insert(
            transfer_id,
            Entry {
                record: Record::Active,
                at: now,
            },
        );
        registry.arrivals.push_back(transfer_id);
    }

    /// Record the drain outcome of `transfer_id`.
    pub(super) fn finish(&self, transfer_id: Uuid, outcome: &TransferDrainOutcome) {
        #[cfg(test)]
        {
            // Bind the hook first: an `if let` on the guard would hold the
            // hooks lock while the hook runs.
            let hook = self.hooks.before_finish_lock.lock().take();
            if let Some(hook) = hook {
                hook();
            }
        }
        let mut registry = self.registry.lock();
        // Read the clock under the lock, so the finished queue stays in time
        // order and aging can stop at the first record that is too new.
        let now = Instant::now();
        if let Some(entry) = registry.entries.get_mut(&transfer_id) {
            entry.record = Record::from_outcome(outcome);
            entry.at = now;
            registry.finished.push_back((transfer_id, now));
        }
    }

    /// The reply to a check of `transfer_id`.
    pub(super) fn check(&self, transfer_id: Uuid) -> CheckReply {
        let mut registry = self.registry.lock();
        registry.age_out(Instant::now(), self.max_age);
        let record = registry
            .entries
            .get(&transfer_id)
            .map_or(Record::Unknown, |entry| entry.record.clone());
        CheckReply {
            incarnation: self.incarnation,
            record,
        }
    }

    /// The ids of the transfers recorded as active.
    #[cfg(test)]
    pub(super) fn active_ids(&self) -> Vec<Uuid> {
        let registry = self.registry.lock();
        registry
            .entries
            .iter()
            .filter(|(_, entry)| entry.record == Record::Active)
            .map(|(id, _)| *id)
            .collect()
    }

    /// The number of records held.
    #[cfg(test)]
    pub(super) fn record_count(&self) -> usize {
        self.registry.lock().entries.len()
    }

    /// The finish times in the order the finished queue holds them.
    #[cfg(test)]
    fn finished_times(&self) -> Vec<Instant> {
        self.registry
            .lock()
            .finished
            .iter()
            .map(|(_, at)| *at)
            .collect()
    }

    #[cfg(test)]
    pub(super) fn evict_all(&self) {
        let mut registry = self.registry.lock();
        registry.entries.clear();
        registry.arrivals.clear();
        registry.finished.clear();
    }
}

/// Fault seams for the tests of this module and the client.
#[cfg(test)]
#[derive(Default)]
pub(super) struct TestHooks {
    /// Drop the reply of the next transfer after its outcome is recorded.
    pub(super) lose_next_reply: std::sync::atomic::AtomicBool,
    /// Hold the next transfer, unrecorded, until this gate opens, and fail
    /// its reply at once.
    pub(super) defer_next: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    /// Run the next transfer and fail its reply when this gate opens.
    pub(super) lose_running_reply: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    /// Evict every record after the next transfer is recorded.
    pub(super) evict_after_next: std::sync::atomic::AtomicBool,
    /// Checks received.
    pub(super) checks: std::sync::atomic::AtomicUsize,
    /// Run once in the next `finish`, before it takes the registry lock.
    pub(super) before_finish_lock: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records(capacity: usize, max_age: Duration) -> LocalTransferRecords {
        LocalTransferRecords {
            incarnation: Uuid::new_v4(),
            registry: Mutex::new(Registry::default()),
            capacity,
            max_age,
            hooks: TestHooks::default(),
        }
    }

    #[test]
    fn the_default_timing_matches_the_config_default() {
        assert_eq!(
            LocalTransferCheckTiming::default(),
            LocalTransferCheckTiming::from(&kvbm_config::LocalTransferCheckConfig::default())
        );
    }

    #[test]
    fn a_record_moves_from_active_to_its_outcome() {
        let records = records(4, RECORD_MAX_AGE);
        let id = Uuid::new_v4();
        assert_eq!(records.check(id).record, Record::Unknown);
        records.begin(id);
        assert_eq!(records.check(id).record, Record::Active);
        records.finish(id, &TransferDrainOutcome::Completed);
        assert_eq!(records.check(id).record, Record::Completed);
    }

    #[test]
    fn a_full_registry_evicts_finished_records_before_active_ones() {
        let records = records(2, RECORD_MAX_AGE);
        let active = Uuid::new_v4();
        let finished = Uuid::new_v4();
        records.begin(active);
        records.begin(finished);
        records.finish(finished, &TransferDrainOutcome::Completed);
        let next = Uuid::new_v4();
        records.begin(next);
        assert_eq!(records.check(active).record, Record::Active);
        assert_eq!(records.check(finished).record, Record::Unknown);
        assert_eq!(records.check(next).record, Record::Active);
    }

    #[test]
    fn a_full_registry_of_active_records_evicts_the_oldest() {
        let records = records(2, RECORD_MAX_AGE);
        let (first, second, third) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        records.begin(first);
        records.begin(second);
        records.begin(third);
        assert_eq!(records.check(first).record, Record::Unknown);
        assert_eq!(records.check(second).record, Record::Active);
        assert_eq!(records.check(third).record, Record::Active);
    }

    /// Two transfers finish at the same time. The one that read the clock
    /// first takes the lock second. The finished queue must still be in time
    /// order, or aging stops at the newer record and keeps an expired one.
    #[test]
    fn concurrent_finishes_keep_the_finished_queue_in_time_order() {
        let records = Arc::new(records(8, RECORD_MAX_AGE));
        let (early, late) = (Uuid::new_v4(), Uuid::new_v4());
        records.begin(early);
        records.begin(late);
        let (release, held) = std::sync::mpsc::channel::<()>();
        *records.hooks.before_finish_lock.lock() = Some(Box::new(move || {
            held.recv().expect("released");
        }));
        let early_finish = {
            let records = Arc::clone(&records);
            std::thread::spawn(move || records.finish(early, &TransferDrainOutcome::Completed))
        };
        while records.hooks.before_finish_lock.lock().is_some() {
            std::thread::yield_now();
        }
        std::thread::sleep(Duration::from_millis(5));
        records.finish(late, &TransferDrainOutcome::Completed);
        release.send(()).expect("the early finish is held");
        early_finish.join().expect("early finish");

        let times = records.finished_times();
        assert!(
            times.windows(2).all(|pair| pair[0] <= pair[1]),
            "the finished queue is out of time order: {times:?}"
        );
    }

    #[test]
    fn finished_records_age_out_but_active_ones_stay() {
        let records = records(8, Duration::ZERO);
        let active = Uuid::new_v4();
        let finished = Uuid::new_v4();
        records.begin(active);
        records.begin(finished);
        records.finish(finished, &TransferDrainOutcome::Completed);
        assert_eq!(records.check(finished).record, Record::Unknown);
        assert_eq!(records.check(active).record, Record::Active);
    }
}
