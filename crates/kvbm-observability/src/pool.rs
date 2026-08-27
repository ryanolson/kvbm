// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw atomic counters and gauges for a single block pool type.
//!
//! All increment/decrement methods use `Ordering::Relaxed` for zero overhead on the hot path.
//! The [`MetricsAggregator`] reads these atomics at scrape time and builds Prometheus protos.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

/// Raw atomic metrics for a single block pool (one per `BlockManager<T>`).
///
/// Counters are monotonically increasing `AtomicU64`.
/// Gauges are bidirectional `AtomicI64`.
pub struct BlockPoolMetrics {
    type_label: String,

    // Counters (monotonic)
    allocations: AtomicU64,
    allocations_from_reset: AtomicU64,
    evictions: AtomicU64,
    registrations: AtomicU64,
    duplicate_blocks: AtomicU64,
    registration_dedup: AtomicU64,
    stagings: AtomicU64,
    match_hashes_requested: AtomicU64,
    match_blocks_returned: AtomicU64,
    scan_hashes_requested: AtomicU64,
    scan_blocks_returned: AtomicU64,

    // Inactive-residency accounting. Each terminal outcome of an inactive
    // tenure contributes the tenure's wall-clock duration to one of these
    // pairs, so `sum / blocks` is an exact mean for that outcome:
    //
    // * evicted — residency that ended in eviction: cache capacity spent on
    //   a block that was never reused. The mean is the pool's **eviction
    //   age**; a *falling* eviction age is the capacity-pressure signal.
    // * reused — residency that ended in a cache hit. The counterpart that
    //   makes the evicted number readable: without it, a large eviction age
    //   cannot be told apart from a pool so oversized it barely evicts.
    //
    // Together with the block-time currently accrued by resident blocks,
    // the two pairs partition all inactive residency.
    inactive_residency_evicted_nanos: AtomicU64,
    inactive_residency_evicted_blocks: AtomicU64,
    inactive_residency_reused_nanos: AtomicU64,
    inactive_residency_reused_blocks: AtomicU64,

    // Audit counters for normally-rare branches. These exist primarily
    // so tests can assert "this code path actually fired" rather than
    // inferring it from emergent behaviour, and so production
    // dashboards detect regressions if these spike.
    eager_primary_to_inactive_total: AtomicU64,
    allocate_atomic_rollback_total: AtomicU64,
    release_primary_noop_total: AtomicU64,
    release_duplicate_noop_total: AtomicU64,

    // Gauges (bidirectional)
    inflight_mutable: AtomicI64,
    inflight_immutable: AtomicI64,
    held_residency: AtomicI64,
    reset_pool_size: AtomicI64,
    inactive_pool_size: AtomicI64,
}

impl BlockPoolMetrics {
    /// Create a new `BlockPoolMetrics` with the given type label (e.g. `"G1"`).
    pub fn new(type_label: String) -> Self {
        Self {
            type_label,
            allocations: AtomicU64::new(0),
            allocations_from_reset: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            registrations: AtomicU64::new(0),
            duplicate_blocks: AtomicU64::new(0),
            registration_dedup: AtomicU64::new(0),
            stagings: AtomicU64::new(0),
            match_hashes_requested: AtomicU64::new(0),
            match_blocks_returned: AtomicU64::new(0),
            scan_hashes_requested: AtomicU64::new(0),
            scan_blocks_returned: AtomicU64::new(0),
            inactive_residency_evicted_nanos: AtomicU64::new(0),
            inactive_residency_evicted_blocks: AtomicU64::new(0),
            inactive_residency_reused_nanos: AtomicU64::new(0),
            inactive_residency_reused_blocks: AtomicU64::new(0),
            eager_primary_to_inactive_total: AtomicU64::new(0),
            allocate_atomic_rollback_total: AtomicU64::new(0),
            release_primary_noop_total: AtomicU64::new(0),
            release_duplicate_noop_total: AtomicU64::new(0),
            inflight_mutable: AtomicI64::new(0),
            inflight_immutable: AtomicI64::new(0),
            held_residency: AtomicI64::new(0),
            reset_pool_size: AtomicI64::new(0),
            inactive_pool_size: AtomicI64::new(0),
        }
    }

    /// The pool type label (e.g. `"G1"`, `"G2"`).
    #[inline(always)]
    pub fn type_label(&self) -> &str {
        &self.type_label
    }

    // ---- Counter increments ----

    #[inline(always)]
    pub fn inc_allocations(&self, n: u64) {
        self.allocations.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_allocations_from_reset(&self, n: u64) {
        self.allocations_from_reset.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_evictions(&self, n: u64) {
        self.evictions.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_registrations(&self) {
        self.registrations.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_duplicate_blocks(&self) {
        self.duplicate_blocks.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_registration_dedup(&self) {
        self.registration_dedup.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_stagings(&self) {
        self.stagings.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_match_hashes_requested(&self, n: u64) {
        self.match_hashes_requested.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_match_blocks_returned(&self, n: u64) {
        self.match_blocks_returned.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_scan_hashes_requested(&self, n: u64) {
        self.scan_hashes_requested.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_scan_blocks_returned(&self, n: u64) {
        self.scan_blocks_returned.fetch_add(n, Ordering::Relaxed);
    }

    // ---- Inactive-residency accounting ----

    /// Record `blocks` inactive tenures that ended in eviction, together
    /// with their summed residency.
    ///
    /// Callers settle a whole batch in one call: the store reads its clock
    /// once per critical section, so a batched eviction contributes one
    /// summed duration rather than N separate adds. An empty batch is a
    /// no-op, so a caller that settles unconditionally pays a predictable
    /// branch rather than two atomic RMWs on shared lines.
    #[inline(always)]
    pub fn add_inactive_residency_evicted(&self, nanos: u64, blocks: u64) {
        if blocks == 0 {
            return;
        }
        self.inactive_residency_evicted_nanos
            .fetch_add(nanos, Ordering::Relaxed);
        self.inactive_residency_evicted_blocks
            .fetch_add(blocks, Ordering::Relaxed);
    }

    /// Record `blocks` inactive tenures that ended in a cache hit, together
    /// with their summed residency. See
    /// [`Self::add_inactive_residency_evicted`].
    #[inline(always)]
    pub fn add_inactive_residency_reused(&self, nanos: u64, blocks: u64) {
        if blocks == 0 {
            return;
        }
        self.inactive_residency_reused_nanos
            .fetch_add(nanos, Ordering::Relaxed);
        self.inactive_residency_reused_blocks
            .fetch_add(blocks, Ordering::Relaxed);
    }

    // ---- Gauge operations ----

    #[inline(always)]
    pub fn inc_inflight_mutable(&self) {
        self.inflight_mutable.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_inflight_mutable_by(&self, n: i64) {
        self.inflight_mutable.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_inflight_mutable(&self) {
        self.inflight_mutable.fetch_sub(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_inflight_mutable_by(&self, n: i64) {
        self.inflight_mutable.fetch_sub(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_inflight_immutable(&self) {
        self.inflight_immutable.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_inflight_immutable_by(&self, n: i64) {
        self.inflight_immutable.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_inflight_immutable(&self) {
        self.inflight_immutable.fetch_sub(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_inflight_immutable_by(&self, n: i64) {
        self.inflight_immutable.fetch_sub(n, Ordering::Relaxed);
    }

    /// Count registered slots owned by an out-of-band pressure action.
    #[inline(always)]
    pub fn inc_held_residency_by(&self, n: i64) {
        self.held_residency.fetch_add(n, Ordering::Relaxed);
    }

    /// Remove registered slots from out-of-band pressure ownership.
    #[inline(always)]
    pub fn dec_held_residency_by(&self, n: i64) {
        self.held_residency.fetch_sub(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_reset_pool_size(&self, size: i64) {
        self.reset_pool_size.store(size, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_reset_pool_size(&self) {
        self.reset_pool_size.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_reset_pool_size_by(&self, n: i64) {
        self.reset_pool_size.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_reset_pool_size(&self) {
        self.reset_pool_size.fetch_sub(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_reset_pool_size_by(&self, n: i64) {
        self.reset_pool_size.fetch_sub(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn set_inactive_pool_size(&self, size: i64) {
        self.inactive_pool_size.store(size, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_inactive_pool_size(&self) {
        self.inactive_pool_size.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn inc_inactive_pool_size_by(&self, n: i64) {
        self.inactive_pool_size.fetch_add(n, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_inactive_pool_size(&self) {
        self.inactive_pool_size.fetch_sub(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn dec_inactive_pool_size_by(&self, n: i64) {
        self.inactive_pool_size.fetch_sub(n, Ordering::Relaxed);
    }

    // ---- Audit counters ----

    /// Lookup-driven `Primary → Inactive` transition fired when the
    /// active-pool `Weak` was dead. Hitting this is exclusively a
    /// race-window event; tests assert it ticks under stress.
    #[inline(always)]
    pub fn inc_eager_primary_to_inactive(&self) {
        self.eager_primary_to_inactive_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// `allocate_atomic` rolled back due to an inactive backend
    /// returning fewer pairs than `len()` advertised. Should never
    /// happen with shipped backends; tests assert it fires when wired
    /// against an under-allocating fake.
    #[inline(always)]
    pub fn inc_allocate_atomic_rollback(&self) {
        self.allocate_atomic_rollback_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// `release_primary` no-op'd because the slot was no longer
    /// `Primary` for this Inner (a concurrent lookup eagerly transitioned
    /// it, or it was resurrected to a different Inner).
    #[inline(always)]
    pub fn inc_release_primary_noop(&self) {
        self.release_primary_noop_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// `release_duplicate` no-op'd because the slot's `Duplicate` weak
    /// no longer matches this Inner.
    #[inline(always)]
    pub fn inc_release_duplicate_noop(&self) {
        self.release_duplicate_noop_total
            .fetch_add(1, Ordering::Relaxed);
    }

    // ---- Snapshot for stats collector ----

    /// Take a point-in-time snapshot of all metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            allocations: self.allocations.load(Ordering::Relaxed),
            allocations_from_reset: self.allocations_from_reset.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            registrations: self.registrations.load(Ordering::Relaxed),
            duplicate_blocks: self.duplicate_blocks.load(Ordering::Relaxed),
            registration_dedup: self.registration_dedup.load(Ordering::Relaxed),
            stagings: self.stagings.load(Ordering::Relaxed),
            match_hashes_requested: self.match_hashes_requested.load(Ordering::Relaxed),
            match_blocks_returned: self.match_blocks_returned.load(Ordering::Relaxed),
            scan_hashes_requested: self.scan_hashes_requested.load(Ordering::Relaxed),
            scan_blocks_returned: self.scan_blocks_returned.load(Ordering::Relaxed),
            inactive_residency_evicted_nanos: self
                .inactive_residency_evicted_nanos
                .load(Ordering::Relaxed),
            inactive_residency_evicted_blocks: self
                .inactive_residency_evicted_blocks
                .load(Ordering::Relaxed),
            inactive_residency_reused_nanos: self
                .inactive_residency_reused_nanos
                .load(Ordering::Relaxed),
            inactive_residency_reused_blocks: self
                .inactive_residency_reused_blocks
                .load(Ordering::Relaxed),
            eager_primary_to_inactive_total: self
                .eager_primary_to_inactive_total
                .load(Ordering::Relaxed),
            allocate_atomic_rollback_total: self
                .allocate_atomic_rollback_total
                .load(Ordering::Relaxed),
            release_primary_noop_total: self.release_primary_noop_total.load(Ordering::Relaxed),
            release_duplicate_noop_total: self.release_duplicate_noop_total.load(Ordering::Relaxed),
            inflight_mutable: self.inflight_mutable.load(Ordering::Relaxed),
            inflight_immutable: self.inflight_immutable.load(Ordering::Relaxed),
            held_residency: self.held_residency.load(Ordering::Relaxed),
            reset_pool_size: self.reset_pool_size.load(Ordering::Relaxed),
            inactive_pool_size: self.inactive_pool_size.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of all atomic metrics, used by the stats collector and prometheus collector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricsSnapshot {
    pub allocations: u64,
    pub allocations_from_reset: u64,
    pub evictions: u64,
    pub registrations: u64,
    pub duplicate_blocks: u64,
    pub registration_dedup: u64,
    pub stagings: u64,
    pub match_hashes_requested: u64,
    pub match_blocks_returned: u64,
    pub scan_hashes_requested: u64,
    pub scan_blocks_returned: u64,
    pub inactive_residency_evicted_nanos: u64,
    pub inactive_residency_evicted_blocks: u64,
    pub inactive_residency_reused_nanos: u64,
    pub inactive_residency_reused_blocks: u64,
    pub eager_primary_to_inactive_total: u64,
    pub allocate_atomic_rollback_total: u64,
    pub release_primary_noop_total: u64,
    pub release_duplicate_noop_total: u64,
    pub inflight_mutable: i64,
    pub inflight_immutable: i64,
    pub held_residency: i64,
    pub reset_pool_size: i64,
    pub inactive_pool_size: i64,
}

impl MetricsSnapshot {
    /// Mean inactive residency of the blocks this pool has evicted — the
    /// pool's **eviction age**.
    ///
    /// This is the capacity-pressure signal, and it reads *inversely* to
    /// intuition: a long eviction age means freed blocks linger, i.e. the
    /// pool has headroom; a *falling* eviction age means blocks are being
    /// recycled soon after they are freed, i.e. thrash. Compare it against
    /// [`Self::mean_reuse_age`] — an eviction age well below the reuse age
    /// means the pool is discarding blocks before the workload's own reuse
    /// distance, so more capacity would convert directly into hits.
    ///
    /// `None` before the first eviction.
    ///
    /// Cumulative since process start. For a live signal, difference two
    /// snapshots (or use `rate()` over both exported counters) rather than
    /// reading this directly.
    pub fn mean_eviction_age(&self) -> Option<Duration> {
        mean_residency(
            self.inactive_residency_evicted_nanos,
            self.inactive_residency_evicted_blocks,
        )
    }

    /// Mean inactive residency of the blocks this pool has served as cache
    /// hits — how long a block typically waits before it is reused.
    ///
    /// `None` before the first inactive hit. Same cumulative caveat as
    /// [`Self::mean_eviction_age`].
    pub fn mean_reuse_age(&self) -> Option<Duration> {
        mean_residency(
            self.inactive_residency_reused_nanos,
            self.inactive_residency_reused_blocks,
        )
    }

    /// Fraction of settled inactive residency that ended in eviction rather
    /// than reuse, in `[0.0, 1.0]` — the share of cache-residency block-time
    /// spent on blocks that were never reused.
    ///
    /// Unlike the two ages this is dimensionless, so it is comparable across
    /// pools of different sizes and across tiers. `None` until at least one
    /// tenure has settled.
    pub fn wasted_residency_fraction(&self) -> Option<f64> {
        let evicted = self.inactive_residency_evicted_nanos;
        let total = evicted.checked_add(self.inactive_residency_reused_nanos)?;
        (total > 0).then(|| evicted as f64 / total as f64)
    }
}

/// Mean residency, or `None` when no tenure has settled into this bucket.
fn mean_residency(nanos: u64, blocks: u64) -> Option<Duration> {
    (blocks > 0).then(|| Duration::from_nanos(nanos / blocks))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_increments() {
        let m = BlockPoolMetrics::new("G1".to_string());

        m.inc_allocations(5);
        m.inc_allocations(3);
        m.inc_evictions(2);
        m.inc_registrations();
        m.inc_duplicate_blocks();
        m.inc_registration_dedup();
        m.inc_stagings();

        let snap = m.snapshot();
        assert_eq!(snap.allocations, 8);
        assert_eq!(snap.evictions, 2);
        assert_eq!(snap.registrations, 1);
        assert_eq!(snap.duplicate_blocks, 1);
        assert_eq!(snap.registration_dedup, 1);
        assert_eq!(snap.stagings, 1);
    }

    #[test]
    fn test_gauge_bidirectional() {
        let m = BlockPoolMetrics::new("G2".to_string());

        m.inc_inflight_mutable();
        m.inc_inflight_mutable();
        m.dec_inflight_mutable();

        m.inc_inflight_immutable();
        m.inc_inflight_immutable();
        m.inc_inflight_immutable();
        m.dec_inflight_immutable();

        let snap = m.snapshot();
        assert_eq!(snap.inflight_mutable, 1);
        assert_eq!(snap.inflight_immutable, 2);
    }

    #[test]
    fn test_held_residency_gauge() {
        let m = BlockPoolMetrics::new("G2".to_string());

        m.inc_held_residency_by(3);
        m.dec_held_residency_by(1);

        assert_eq!(m.snapshot().held_residency, 2);
    }

    #[test]
    fn test_pool_size_gauges() {
        let m = BlockPoolMetrics::new("G1".to_string());

        m.set_reset_pool_size(100);
        m.set_inactive_pool_size(50);

        let snap = m.snapshot();
        assert_eq!(snap.reset_pool_size, 100);
        assert_eq!(snap.inactive_pool_size, 50);

        m.set_reset_pool_size(80);
        let snap = m.snapshot();
        assert_eq!(snap.reset_pool_size, 80);

        // Test inc/dec for reset pool size
        m.inc_reset_pool_size();
        m.inc_reset_pool_size();
        m.dec_reset_pool_size();
        let snap = m.snapshot();
        assert_eq!(snap.reset_pool_size, 81);

        // Test inc/dec for inactive pool size
        m.inc_inactive_pool_size();
        m.inc_inactive_pool_size();
        m.inc_inactive_pool_size();
        m.dec_inactive_pool_size();
        let snap = m.snapshot();
        assert_eq!(snap.inactive_pool_size, 52);
    }

    #[test]
    fn inactive_residency_means_are_undefined_until_a_tenure_settles() {
        let m = BlockPoolMetrics::new("G1".to_string());
        let snap = m.snapshot();

        assert_eq!(snap.mean_eviction_age(), None);
        assert_eq!(snap.mean_reuse_age(), None);
        assert_eq!(snap.wasted_residency_fraction(), None);
    }

    #[test]
    fn inactive_residency_means_divide_by_their_own_denominator() {
        let m = BlockPoolMetrics::new("G1".to_string());

        // Two batches into each bucket, mirroring how the store settles:
        // one summed duration plus its block count per critical section.
        m.add_inactive_residency_evicted(Duration::from_millis(300).as_nanos() as u64, 2);
        m.add_inactive_residency_evicted(Duration::from_millis(100).as_nanos() as u64, 2);
        m.add_inactive_residency_reused(Duration::from_millis(600).as_nanos() as u64, 4);

        let snap = m.snapshot();
        assert_eq!(snap.inactive_residency_evicted_blocks, 4);
        assert_eq!(
            snap.mean_eviction_age(),
            Some(Duration::from_millis(100)),
            "400ms over 4 evicted blocks"
        );
        assert_eq!(
            snap.mean_reuse_age(),
            Some(Duration::from_millis(150)),
            "600ms over 4 reused blocks"
        );
        // 400ms wasted of 1000ms settled.
        assert_eq!(snap.wasted_residency_fraction(), Some(0.4));
    }

    /// The eviction age reads *inversely* to intuition, and this pin exists
    /// so nobody "fixes" the direction: the thrashing pool is the one with
    /// the SHORTER age, because it recycles freed blocks sooner.
    #[test]
    fn a_thrashing_pool_reports_the_shorter_eviction_age() {
        let roomy = BlockPoolMetrics::new("roomy".to_string());
        roomy.add_inactive_residency_evicted(Duration::from_secs(30).as_nanos() as u64, 10);

        let thrashing = BlockPoolMetrics::new("thrashing".to_string());
        thrashing.add_inactive_residency_evicted(Duration::from_millis(200).as_nanos() as u64, 10);

        assert!(
            thrashing.snapshot().mean_eviction_age() < roomy.snapshot().mean_eviction_age(),
            "a falling eviction age is the capacity-pressure signal"
        );
    }

    #[test]
    fn test_type_label() {
        let m = BlockPoolMetrics::new("MyPool".to_string());
        assert_eq!(m.type_label(), "MyPool");
    }

    #[test]
    fn test_match_scan_counters() {
        let m = BlockPoolMetrics::new("G1".to_string());

        m.inc_match_hashes_requested(10);
        m.inc_match_blocks_returned(7);
        m.inc_scan_hashes_requested(20);
        m.inc_scan_blocks_returned(15);

        let snap = m.snapshot();
        assert_eq!(snap.match_hashes_requested, 10);
        assert_eq!(snap.match_blocks_returned, 7);
        assert_eq!(snap.scan_hashes_requested, 20);
        assert_eq!(snap.scan_blocks_returned, 15);
    }
}
