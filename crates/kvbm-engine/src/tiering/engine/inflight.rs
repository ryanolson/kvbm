// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-flight onboard hash guard (the engine analogue of vLLM's native
//! offloading connector `_blocks_being_loaded` set).
//!
//! Hashes whose onboard to G1 is committed are recorded here, ONCE per match
//! lifecycle at its USAA-time onboard mint, keyed by the lifecycle's
//! generation id ([`InflightKey`]). Any OTHER request whose match window
//! intersects them defers its `find_blocks` poll instead of racing a duplicate
//! load — vLLM parks the deferred request in skipped-waiting and re-polls
//! every step. The entry clears when the lifecycle's RAII release fires —
//! aligning deferral-end with vLLM's block-registration step (`cache_blocks`
//! runs on the same scheduler pass that processes `finished_recving`), so the
//! gap between an engine-side load terminal and the loaded blocks becoming
//! vLLM-visible can never invite a duplicate load.
//!
//! Race tolerance is by design: once vLLM registers the loaded blocks, future
//! windows exclude them anyway (`num_computed` grows past them), so a late
//! clear only costs deferral latency, never correctness.
//!
//! ## Lifecycle-keyed recording, release-funnel clearing
//!
//! The guard exists to dedup loads INTO G1 — so it records exactly when a load
//! is committed (USAA) and holds until the loaded blocks are connector-visible
//! (the lifecycle handle's release). It is NOT cleared at the engine action
//! terminal: a load that is engine-terminal but not yet vLLM-registered is
//! still a duplicate-load hazard, and the eviction drain-holder deliberately
//! keeps the deferral alive past the terminal until the held lifecycle drops.
//!
//! ### Record sites (the three onboard mints, one per lifecycle kind)
//!
//! 1. **local onboard** — `LocalConnectorEngine::local_onboard`, keyed
//!    [`InflightKey::Search`] by the driving search generation, recording the
//!    hit window `buffer[computed .. computed + hit_blocks]` (the
//!    absolute-indexed hash buffer, sliced to the matched span).
//! 2. **cd onboard** — `LocalConnectorEngine::cd_onboard`, keyed by the SAME
//!    driving [`InflightKey::Search`] generation (passed down from
//!    `local_onboard`; the connector's parked handle carries that id, so the
//!    clear binds to the release that actually fires), recording the unified
//!    window `window_hashes[..unified]` (the local+remote committed span). The
//!    cd bail paths (`mint_failed_onboard`) record NOTHING — they are born
//!    terminal with nothing in flight.
//! 3. **prefill onboard** — `LocalConnectorEngine::prefill_onboard_by_id`,
//!    keyed [`InflightKey::Prefill`] by the accept generation, recording the
//!    EXTERNAL suffix of the decode-provided window
//!    `expected_hashes[len - external_blocks ..]` — exactly the hashes the
//!    USAA kick copies into G1.
//!
//! One record per lifecycle is structural: the engine refuses a second onboard
//! per latched generation (`local_onboard` consumes the search latch;
//! `claim_usaa` refuses prefill re-entry), so a key can never double-record.
//! Offload (save) actions never record. The zero-external prefill
//! fall-through's INTERNAL search records under its own `Search` key when its
//! delegated onboard mints.
//!
//! ### Clear sites: the two release funnels
//!
//! `release_search` clears `Search` keys; `release_prefill_session`
//! (`prefill_release`) clears `Prefill` keys — unconditionally at function
//! top, BEFORE any map/generation guard, because several teardown paths
//! remove the engine-side lifecycle state while the connector still parks the
//! handle (pre-USAA replay, pipeline failure, kick failure) and the eventual
//! handle drop must still clear. Clearing by the releasing handle's OWN
//! generation id keeps a stale release harmless: it can only touch its own
//! (already-cleared) entry, never a fresh re-latch's.
//!
//! Every path a recorded lifecycle can end funnels through one of the two:
//!
//! * connector handle drop (reap — inline or finishing sweep — and leader
//!   teardown) → kind-routed RAII → `release_search` /
//!   `release_prefill_session`;
//! * recv-side release (`update_connector_output` on a terminal onboard) →
//!   handle drop → same;
//! * `release_parked` instruction (zero-refine / `Lost` / empty window,
//!   Issue-A-gated on the in-flight onboard) → handle drop → same;
//! * eviction drain-holder drop (the holder retains the handle until the held
//!   onboard is terminal, so the deferral OUTLIVES the action terminal until
//!   the connector-visible release) → handle drop → same;
//! * `evict()`'s engine-internal prefill teardown → `prefill_release`
//!   directly (the connector's stale holder handle later no-ops on the
//!   `AcceptId` guard). This is the one pre-terminal clear: an evicted
//!   prefill's in-flight kick is fence-armed, so the worker-side hazard is
//!   covered, and the early re-admit costs at most a duplicate load into a
//!   different dest (the vLLM race tolerance above);
//! * the zero-external fall-through's internal binding →
//!   `prefill_release`'s `take_local_search` → `release_search`.
//!
//! Residual exposure: a connector that leaks a handle (never drops it) leaks
//! the entry — the same exposure class as a leaked never-terminal action.
//!
//! ## Deliberate deviation from vLLM's plain-set shape
//!
//! vLLM tracks loading blocks in a plain set. We keep a **refcounted multiset**
//! instead: overlapping recordings are reachable (two lifecycles can commit
//! onboards covering a shared hash). A plain set would drop tracking for the
//! still-live second lifecycle the instant the first releases, re-admitting an
//! overlapping minter while a load is live. Refcounting keeps a hash deferred
//! until the LAST lifecycle covering it releases.
//!
//! Keying by generation id (not request id) keeps evict + restore correct: a
//! restored request's fresh lifecycle records under a NEW generation while the
//! old generation's entry drains in its holder, so the same request id may own
//! several live entries at once.

use std::collections::HashMap;

use kvbm_common::SequenceHash;
use kvbm_protocols::connector::{AcceptId, SearchId};

/// Generation key for one recorded match lifecycle — the same id the
/// lifecycle's RAII release carries, so record and clear bind to the same
/// generation by construction.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub(super) enum InflightKey {
    /// A local (or CD-unified) match lifecycle; cleared by `release_search`.
    Search(SearchId),
    /// A dispatched remote-prefill lifecycle; cleared by
    /// `release_prefill_session`.
    Prefill(AcceptId),
}

/// Refcounted multiset of hashes under in-flight onboard, keyed for clearing
/// by the owning lifecycle's [`InflightKey`].
#[derive(Debug, Default)]
pub(super) struct InflightOnboards {
    /// Refcounted multiset: how many recorded lifecycles cover each hash. A
    /// hash with a non-zero count defers any non-exempt window that touches it.
    hashes: HashMap<SequenceHash, u32>,
    /// Per-lifecycle snapshot of the hashes it recorded, so [`Self::clear`]
    /// decrements exactly the multiset entries [`Self::record`] incremented.
    by_lifecycle: HashMap<InflightKey, Vec<SequenceHash>>,
    /// Optional gauge mirroring `hashes.len()` (distinct in-flight hashes).
    /// `None` for the gauge-less bare-leader / test path. Set-to-len at the end
    /// of every mutator, so it cannot desync from the multiset.
    gauge: Option<prometheus::IntGauge>,
}

impl InflightOnboards {
    /// Gauge-less guard. The production engine builds via [`Self::with_gauge`];
    /// only the in-module tests construct without a gauge.
    #[cfg(test)]
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Build a guard whose distinct-hash count drives `gauge` (the engine wires
    /// in `CompatMetrics::inflight_onboard_hashes`; `None` leaves it inert).
    pub(super) fn with_gauge(gauge: Option<prometheus::IntGauge>) -> Self {
        Self {
            gauge,
            ..Default::default()
        }
    }

    /// Record a lifecycle's onboard-covered hashes, incrementing the multiset
    /// refcount for each. A double-record of the same `key` is impossible by
    /// construction (one onboard per latched generation) — it is a
    /// `debug_assert` + no-op so a misuse never corrupts the refcounts.
    pub(super) fn record(&mut self, key: InflightKey, hashes: Vec<SequenceHash>) {
        if self.by_lifecycle.contains_key(&key) {
            debug_assert!(false, "double-record of in-flight onboard {key:?}");
            return;
        }
        for &h in &hashes {
            *self.hashes.entry(h).or_insert(0) += 1;
        }
        self.by_lifecycle.insert(key, hashes);
        self.sync_gauge();
    }

    /// Clear a lifecycle's recorded hashes, decrementing (saturating) the
    /// multiset refcount for each and removing entries that reach zero.
    /// Idempotent: returns `false` for an unknown key (already cleared or
    /// never recorded), so calling it at every release site is safe.
    pub(super) fn clear(&mut self, key: &InflightKey) -> bool {
        let Some(hashes) = self.by_lifecycle.remove(key) else {
            return false;
        };
        for h in hashes {
            if let Some(count) = self.hashes.get_mut(&h) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.hashes.remove(&h);
                }
            }
        }
        self.sync_gauge();
        true
    }

    /// Set the gauge (if wired) to the current distinct-hash count. `record` and
    /// `clear` are the only mutators of `hashes`, so set-at-end is always exact
    /// — including the refcount-shared-hash case where a naive per-call delta
    /// would over/under-count.
    fn sync_gauge(&self) {
        if let Some(g) = &self.gauge {
            g.set(self.hashes.len() as i64);
        }
    }

    /// Whether any hash in `window` is currently under an in-flight onboard —
    /// the deferral predicate the `find_blocks` router consults on every
    /// non-exempt arm (fresh mint AND refresh).
    pub(super) fn overlaps(&self, window: &[SequenceHash]) -> bool {
        window.iter().any(|h| self.hashes.contains_key(h))
    }

    /// True when nothing is in flight (no covered hashes, no tracked
    /// lifecycles).
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.hashes.is_empty() && self.by_lifecycle.is_empty()
    }

    /// Number of distinct hashes currently under at least one in-flight onboard.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.hashes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(i: u64) -> SequenceHash {
        SequenceHash::new(i, None, i)
    }

    fn key() -> InflightKey {
        InflightKey::Search(SearchId::new())
    }

    #[test]
    fn empty_guard_never_overlaps() {
        let guard = InflightOnboards::new();
        assert!(guard.is_empty());
        assert_eq!(guard.len(), 0);
        assert!(!guard.overlaps(&[hash(0), hash(1)]));
    }

    #[test]
    fn record_then_overlaps_then_clear() {
        let mut guard = InflightOnboards::new();
        let a = key();
        guard.record(a, vec![hash(0), hash(1)]);

        assert!(guard.overlaps(&[hash(1)]), "recorded hash overlaps");
        assert!(!guard.overlaps(&[hash(2)]), "unrecorded hash does not");
        assert_eq!(guard.len(), 2);

        assert!(guard.clear(&a), "clearing a known lifecycle returns true");
        assert!(guard.is_empty());
        assert!(!guard.overlaps(&[hash(0), hash(1)]));
    }

    /// Both key kinds key independent entries: a prefill generation's clear
    /// never touches a search generation's record.
    #[test]
    fn search_and_prefill_keys_are_independent() {
        let mut guard = InflightOnboards::new();
        let s = InflightKey::Search(SearchId::new());
        let p = InflightKey::Prefill(AcceptId::new());
        guard.record(s, vec![hash(0)]);
        guard.record(p, vec![hash(1)]);

        assert!(guard.clear(&p));
        assert!(guard.overlaps(&[hash(0)]), "the search record survives");
        assert!(!guard.overlaps(&[hash(1)]));
        assert!(guard.clear(&s));
        assert!(guard.is_empty());
    }

    #[test]
    fn clear_unknown_lifecycle_is_idempotent_noop() {
        let mut guard = InflightOnboards::new();
        let a = key();
        guard.record(a, vec![hash(0)]);

        assert!(guard.clear(&a));
        // Second clear of the same (now-unknown) lifecycle is a no-op.
        assert!(!guard.clear(&a), "idempotent: false for an unknown key");
        assert!(!guard.clear(&key()), "false for a never-recorded key");
        assert!(guard.is_empty());
    }

    /// The refcount deviation from vLLM's plain set: two distinct lifecycles
    /// cover a shared hash. Releasing the FIRST must keep the hash deferred for
    /// the still-live second; only releasing the LAST re-admits an overlapping
    /// minter.
    #[test]
    fn overlapping_records_refcount_independently() {
        let mut guard = InflightOnboards::new();
        let a = key();
        let b = key();

        guard.record(a, vec![hash(0), hash(1)]);
        guard.record(b, vec![hash(1), hash(2)]); // shares hash(1)

        assert!(guard.overlaps(&[hash(1)]));
        assert_eq!(guard.len(), 3, "hash(0), hash(1), hash(2)");

        // Clear A: hash(0) drops to zero, hash(1) still held by B.
        assert!(guard.clear(&a));
        assert!(!guard.overlaps(&[hash(0)]), "A's exclusive hash cleared");
        assert!(
            guard.overlaps(&[hash(1)]),
            "shared hash stays deferred while B is live"
        );
        assert!(guard.overlaps(&[hash(2)]));
        assert_eq!(guard.len(), 2);

        // Clear B: now nothing is in flight.
        assert!(guard.clear(&b));
        assert!(guard.is_empty());
        assert!(!guard.overlaps(&[hash(1)]));
    }

    /// The gauge mirrors the distinct-hash count exactly across record/clear,
    /// including the refcount-shared-hash path (set-to-len, not naive +/-len).
    #[test]
    fn gauge_tracks_distinct_hash_count() {
        let gauge = prometheus::IntGauge::new("test_inflight_onboard_hashes", "t").unwrap();
        let mut guard = InflightOnboards::with_gauge(Some(gauge.clone()));
        assert_eq!(gauge.get(), 0);

        let a = key();
        guard.record(a, vec![hash(0), hash(1)]);
        assert_eq!(gauge.get(), 2);

        let b = key();
        guard.record(b, vec![hash(1), hash(2)]); // shares hash(1)
        assert_eq!(gauge.get(), 3, "distinct count, not naive +len");

        assert!(guard.clear(&a));
        assert_eq!(gauge.get(), 2, "hash(1) still held by b");

        assert!(guard.clear(&b));
        assert_eq!(gauge.get(), 0);
    }

    #[test]
    fn double_record_same_lifecycle_is_noop_in_release() {
        // The debug_assert fires in debug builds; here we only assert the
        // refcounts are not corrupted (the second record is a no-op).
        let mut guard = InflightOnboards::new();
        let a = key();
        guard.record(a, vec![hash(0)]);
        // A second record of the SAME lifecycle would double-count without the
        // guard. Run it only in release builds where the debug_assert is gone.
        if !cfg!(debug_assertions) {
            guard.record(a, vec![hash(0), hash(5)]);
            assert_eq!(guard.len(), 1, "the duplicate record was ignored");
            assert!(!guard.overlaps(&[hash(5)]));
        }
        assert!(guard.clear(&a));
        assert!(
            guard.is_empty(),
            "single clear fully drains a single record"
        );
    }
}
