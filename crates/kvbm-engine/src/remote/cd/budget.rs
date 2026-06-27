// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional-disaggregation admission state: the inflight-token budget and
//! the CD circuit-breaker tier cell.

use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};

use kvbm_protocols::disagg::BreakerTier;

// ============================================================================
// Inflight token budget
// ============================================================================

#[derive(Debug)]
pub(crate) struct InflightBudget {
    available: AtomicUsize,
    capacity: usize,
}

impl InflightBudget {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            available: AtomicUsize::new(capacity),
            capacity,
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn available(&self) -> usize {
        self.available.load(Ordering::Acquire)
    }

    pub(crate) fn try_reserve(&self, n: usize) -> bool {
        if self.capacity == usize::MAX {
            return true;
        }
        let mut current = self.available.load(Ordering::Acquire);
        loop {
            if current < n {
                return false;
            }
            match self.available.compare_exchange_weak(
                current,
                current - n,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    pub(crate) fn release(&self, n: usize) {
        if self.capacity == usize::MAX {
            return;
        }
        let prev = self.available.fetch_add(n, Ordering::Release);
        debug_assert!(
            prev.saturating_add(n) <= self.capacity,
            "InflightBudget release overflow: prev={prev} +n={n} capacity={cap}",
            cap = self.capacity
        );
    }
}

// ============================================================================
// CD circuit-breaker tier cell
// ============================================================================

/// Decode-local cache of the hub-pushed CD circuit-breaker tier.
///
/// The hub's prefill-router breaker computes the tier and PUSHES `(tier, epoch)`
/// to this decode (via the connector's velo tier-signal handler, which writes
/// this cell); the engine reads the cached value synchronously inside its search
/// path with a single relaxed atomic load (zero hot-path RPC). The state is
/// ABSOLUTE (latest-by-recency via `epoch`), never a toggle: [`Self::apply`]
/// adopts an inbound `(tier, epoch)` ONLY when `epoch >= cached_epoch`, so an
/// older push (e.g. a reorder, or a duplicate) can never regress a newer tier,
/// and a RESTARTED hub — whose epoch the hub seeds above any prior value —
/// always wins.
///
/// Default is [`BreakerTier::Calm`] @ epoch 0: a decode that never received a
/// push (breaker disabled, or pre-first-push) behaves EXACTLY as today.
///
/// `pub` (re-exported at [`crate::cd`]): the connector's velo tier-signal
/// handler writes this cell via [`Self::apply`]; [`Self::tier`] /
/// [`Self::epoch`] are the read half (the handler logs the cached epoch on a
/// stale push, and the connector's handler-body tests assert the visible
/// state).
#[derive(Debug)]
pub struct TierCell {
    /// Current tier as a `u8` (0=Calm, 1=Warm, 2=Hot). Lock-free hot-path read.
    tier: AtomicU8,
    /// Highest epoch applied so far (monotone non-decreasing via `apply`). Lock-free read.
    epoch: AtomicU64,
    /// Serializes concurrent `apply` writers so the `(epoch, tier)` pair updates
    /// atomically with respect to other pushes. A bare epoch-CAS + a separate
    /// tier-store lets a lower-epoch push land its tier AFTER a higher-epoch push
    /// won the epoch (the two tier stores are unordered) — leaving `epoch=high`
    /// with `tier=low`, a clobber that defeats latest-by-recency. The write lock
    /// closes that. It is OFF the hot path (pushes are rare — per tier transition
    /// / on decode registration), and readers ([`Self::tier`] / [`Self::epoch`])
    /// stay lock-free atomic loads.
    write_lock: std::sync::Mutex<()>,
}

impl Default for TierCell {
    fn default() -> Self {
        Self {
            tier: AtomicU8::new(tier_to_u8(BreakerTier::Calm)),
            epoch: AtomicU64::new(0),
            write_lock: std::sync::Mutex::new(()),
        }
    }
}

impl TierCell {
    /// Current cached tier (single relaxed load — the hot-path read).
    pub fn tier(&self) -> BreakerTier {
        tier_from_u8(self.tier.load(Ordering::Relaxed))
    }

    /// Highest epoch applied so far.
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Apply an inbound `(tier, epoch)` push iff `epoch >= cached_epoch`
    /// (idempotent, absolute state). Returns `true` when the push was applied
    /// (including an equal-epoch re-apply, which keeps the handler ack simple),
    /// `false` when it was rejected as stale.
    ///
    /// Writers are SERIALIZED on `write_lock` so the `(epoch, tier)` pair updates
    /// atomically with respect to other concurrent pushes: a recency-rejected
    /// (stale / lower-epoch) push stores NOTHING, and — crucially — a lower-epoch
    /// push can never land its tier AFTER a higher-epoch push (which a bare
    /// epoch-CAS + separate tier-store CANNOT prevent, since the two tier stores
    /// from different writers are unordered). The lock is off the hot path
    /// (pushes are rare); the reader ([`Self::tier`]) takes no lock and always
    /// observes the tier last published by the most-recent winning push.
    pub fn apply(&self, tier: BreakerTier, epoch: u64) -> bool {
        // Serialize writers (rare velo pushes) — see the field doc for why a
        // lock-free epoch-CAS is insufficient under concurrent pushes. A poisoned
        // lock is benign here (the guarded data is two atomics, not invariants),
        // so recover the guard rather than panic.
        let _guard = self
            .write_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let cached = self.epoch.load(Ordering::Acquire);
        if epoch < cached {
            // Strictly older ⇒ recency-rejected: store NOTHING.
            return false;
        }
        // Winning (or equal-epoch idempotent) push: publish the tier, then commit
        // the epoch (epoch as the commit marker). Both Release; readers of `tier`
        // are Relaxed and never gate on `epoch`, so a hypothetical epoch-then-tier
        // reader still sees a tier no older than the epoch it observed.
        self.tier.store(tier_to_u8(tier), Ordering::Release);
        self.epoch.store(epoch, Ordering::Release);
        true
    }
}

fn tier_to_u8(t: BreakerTier) -> u8 {
    match t {
        BreakerTier::Calm => 0,
        BreakerTier::Warm => 1,
        BreakerTier::Hot => 2,
    }
}

fn tier_from_u8(v: u8) -> BreakerTier {
    match v {
        0 => BreakerTier::Calm,
        1 => BreakerTier::Warm,
        _ => BreakerTier::Hot,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflight_budget_unlimited_capacity_skips_atomic() {
        let budget = InflightBudget::new(usize::MAX);
        assert!(budget.try_reserve(1_000_000));
        assert_eq!(budget.available(), usize::MAX);
        budget.release(1_000_000);
        assert_eq!(budget.available(), usize::MAX);
    }

    #[test]
    fn inflight_budget_reserve_then_release_balances() {
        let budget = InflightBudget::new(256);
        assert!(budget.try_reserve(64));
        assert_eq!(budget.available(), 192);
        budget.release(64);
        assert_eq!(budget.available(), 256);
    }

    #[test]
    fn inflight_budget_exhausted_reservation_returns_false() {
        let budget = InflightBudget::new(100);
        assert!(budget.try_reserve(64));
        assert!(!budget.try_reserve(64));
        assert_eq!(budget.available(), 36);
    }

    #[test]
    fn inflight_budget_partial_reservation_succeeds_when_fits() {
        let budget = InflightBudget::new(100);
        assert!(budget.try_reserve(64));
        assert!(budget.try_reserve(32));
        assert_eq!(budget.available(), 4);
        assert!(!budget.try_reserve(8));
        assert!(budget.try_reserve(4));
        assert_eq!(budget.available(), 0);
    }

    #[test]
    fn inflight_budget_zero_reservation_is_a_noop() {
        let budget = InflightBudget::new(64);
        assert!(budget.try_reserve(0));
        assert_eq!(budget.available(), 64);
    }

    // --- TierCell: latest-by-recency (epoch-gated) absolute state ------------

    #[test]
    fn tier_cell_default_is_calm_epoch_zero() {
        // A decode that never received a push behaves exactly as today.
        let cache = TierCell::default();
        assert_eq!(cache.tier(), BreakerTier::Calm);
        assert_eq!(cache.epoch(), 0);
    }

    #[test]
    fn tier_cell_higher_epoch_push_applies() {
        let cache = TierCell::default();
        assert!(cache.apply(BreakerTier::Hot, 5), "higher epoch must apply");
        assert_eq!(cache.tier(), BreakerTier::Hot);
        assert_eq!(cache.epoch(), 5);
    }

    #[test]
    fn tier_cell_lower_epoch_push_leaves_tier_unchanged() {
        // The latest-by-recency invariant: a stale (strictly-lower-epoch) push
        // stores NOTHING — neither the tier nor the epoch may regress. NOTE: this
        // sequential test documents the intended SEMANTICS; it passes on both the
        // old store-before-CAS code and the fixed store-after-CAS code, because
        // sequentially the `epoch < cached` early-return fires before any store
        // either way. The clobber the fix removes is a CONCURRENT interleave (a
        // stale push reads the old cached epoch, stores its tier, then loses the
        // CAS — leaving a high epoch paired with a stale tier); a deterministic
        // unit test can't force that interleave. This guards the absolute-state
        // contract.
        let cache = TierCell::default();
        assert!(cache.apply(BreakerTier::Hot, 50));
        assert_eq!(cache.tier(), BreakerTier::Hot);
        assert_eq!(cache.epoch(), 50);

        // Stale push: CALM @ epoch 10 (< 50) — rejected, tier NOT regressed.
        assert!(
            !cache.apply(BreakerTier::Calm, 10),
            "stale (lower-epoch) push must be rejected"
        );
        assert_eq!(
            cache.tier(),
            BreakerTier::Hot,
            "tier must not be clobbered by a recency-rejected push"
        );
        assert_eq!(cache.epoch(), 50, "epoch must not regress");
    }

    #[test]
    fn tier_cell_equal_epoch_push_is_idempotent_reapply() {
        // Equal epoch is accepted (keeps the handler ack simple) and re-applies
        // the tier — an idempotent no-op when the tier matches, and a re-store
        // of the same epoch's tier when it differs (a benign duplicate).
        let cache = TierCell::default();
        assert!(cache.apply(BreakerTier::Warm, 7));
        assert_eq!(cache.tier(), BreakerTier::Warm);

        assert!(
            cache.apply(BreakerTier::Warm, 7),
            "equal-epoch re-apply is accepted"
        );
        assert_eq!(cache.tier(), BreakerTier::Warm);
        assert_eq!(cache.epoch(), 7);
    }
}
