// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{SystemTime, UNIX_EPOCH};

use prometheus::core::{Collector, Desc};
use prometheus::proto::MetricFamily;
use prometheus::{GaugeVec, IntCounterVec, IntGaugeVec, Opts, Registry};
use tokio::time::Instant;

use super::{PauseState, WorkerSummary};

#[derive(Clone)]
struct Families {
    count: IntGaugeVec,
    oldest: GaugeVec,
    attempt: GaugeVec,
    answer: GaugeVec,
    checks: IntCounterVec,
    stuck: IntCounterVec,
}

impl Families {
    fn collectors(&self) -> [&dyn Collector; 6] {
        [
            &self.count,
            &self.oldest,
            &self.attempt,
            &self.answer,
            &self.checks,
            &self.stuck,
        ]
    }
}

impl Collector for Families {
    fn desc(&self) -> Vec<&Desc> {
        self.collectors()
            .iter()
            .flat_map(|metric| metric.desc())
            .collect()
    }

    fn collect(&self) -> Vec<MetricFamily> {
        self.collectors()
            .iter()
            .flat_map(|metric| metric.collect())
            .collect()
    }
}

pub(super) struct PauseMetrics {
    registry: Registry,
    families: Families,
    clock_start: Instant,
    unix_start: f64,
}

impl PauseMetrics {
    pub(super) fn new(registry: &Registry, engine: &str) -> Result<Self, prometheus::Error> {
        let opts = |name, help| Opts::new(name, help).const_label("engine", engine);
        let families = Families {
            count: IntGaugeVec::new(
                opts("kvbm_paused_onboards", "Onboards waiting for a drain proof"),
                &["worker", "state"],
            )?,
            oldest: GaugeVec::new(
                opts(
                    "kvbm_onboard_oldest_pause_timestamp_seconds",
                    "Start time of the oldest onboard pause, or zero",
                ),
                &["worker", "state"],
            )?,
            attempt: GaugeVec::new(
                opts(
                    "kvbm_onboard_last_probe_attempt_timestamp_seconds",
                    "Start time of the last drain check",
                ),
                &["worker"],
            )?,
            answer: GaugeVec::new(
                opts(
                    "kvbm_onboard_last_probe_answer_timestamp_seconds",
                    "Time of the last drain check answer",
                ),
                &["worker"],
            )?,
            checks: IntCounterVec::new(
                opts(
                    "kvbm_onboard_probe_checks_total",
                    "Onboard drain checks finished by result",
                ),
                &["worker", "result"],
            )?,
            stuck: IntCounterVec::new(
                opts(
                    "kvbm_onboard_probe_stuck_total",
                    "Onboard probe loops that missed their next attempt budget",
                ),
                &["worker"],
            )?,
        };
        registry.register(Box::new(families.clone()))?;
        Ok(Self {
            registry: registry.clone(),
            families,
            clock_start: Instant::now(),
            unix_start: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        })
    }

    fn timestamp(&self, time: Instant) -> f64 {
        if time < self.clock_start {
            self.unix_start - (self.clock_start - time).as_secs_f64()
        } else {
            self.unix_start + (time - self.clock_start).as_secs_f64()
        }
    }

    pub(super) fn publish_pauses(&self, worker: &str, snapshot: &[WorkerSummary; 3]) {
        for (state, summary) in PauseState::ALL.into_iter().zip(snapshot) {
            let labels = &[worker, state.label()];
            self.families
                .count
                .with_label_values(labels)
                .set(summary.count as i64);
            self.families
                .oldest
                .with_label_values(labels)
                .set(summary.oldest.map_or(0.0, |time| self.timestamp(time)));
        }
    }

    pub(super) fn attempt(&self, worker: &str, now: Instant) {
        let gauge = self.families.attempt.with_label_values(&[worker]);
        gauge.set(gauge.get().max(self.timestamp(now)));
    }

    pub(super) fn check(&self, worker: &str, result: &str, answered: Option<Instant>) {
        self.families
            .checks
            .with_label_values(&[worker, result])
            .inc();
        if let Some(answered) = answered {
            let gauge = self.families.answer.with_label_values(&[worker]);
            gauge.set(gauge.get().max(self.timestamp(answered)));
        }
    }

    pub(super) fn stuck(&self, worker: &str) {
        self.families.stuck.with_label_values(&[worker]).inc();
    }
}

impl Drop for PauseMetrics {
    fn drop(&mut self) {
        if let Err(error) = self.registry.unregister(Box::new(self.families.clone())) {
            tracing::error!(%error, "could not unregister onboard pause metrics");
        }
    }
}
