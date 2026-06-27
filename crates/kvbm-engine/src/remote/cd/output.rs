// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill OUTPUT capture: the engine-owned G2 register observer.
//!
//! As the prefill worker computes the rest of the prompt, its G1 blocks
//! offload to G2 and the offload pipeline's register step fires its
//! observers. [`PrefillOutputObserver`] is the engine's hook on that seam
//! (ported from the legacy connector's `ConditionalDecodeG2Observer`): it
//! matches freshly-registered blocks against each accepted lifecycle's
//! expected-output set and forwards the matches to that lifecycle's
//! [`PrefillRequestState::commit_output`], which publishes them into the
//! parked session the decode side is draining.
//!
//! Two disciplines carried over from the legacy observer, both load-bearing:
//!
//! * **Generation scoping.** Every piece of observer state is keyed by
//!   `(request_id, AcceptId)`. The release-time finalize drain runs AFTER the
//!   per-request map entry is removed, so a re-accepted same-rid lifecycle can
//!   install a fresh entry while the old drain still polls — generation
//!   scoping keeps the two independent (the same-key-different-generation
//!   discipline used throughout this tree).
//! * **Inflight accounting.** [`Self::observe`] bumps a per-generation
//!   inflight counter UNDER the same lock that makes the residual removal
//!   visible, and decrements only after the dispatch returns. Without it,
//!   [`Self::has_pending`] would read `false` in the window between
//!   "residual emptied" and "session.commit + make_available landed", and the
//!   release drain could finalize the session ahead of the final output
//!   commits on the wire.

use std::collections::{HashMap, HashSet};
use std::sync::Weak;

use parking_lot::Mutex;

use kvbm_logical::blocks::ImmutableBlock;
use kvbm_protocols::connector::AcceptId;
use kvbm_protocols::connector::RequestId;

use super::prefill::PrefillRequestState;
use crate::{G2, SequenceHash};

/// Engine-owned observer over the offload pipeline's G1→G2 register step.
/// One instance per CD runtime, registered once at engine construction.
pub(crate) struct PrefillOutputObserver {
    /// One mutex over BOTH maps on purpose: `observe` must make the residual
    /// removal and the inflight bump visible atomically, or a concurrent
    /// `has_pending` caller could observe "residual gone, dispatch uncounted".
    inner: Mutex<ObserverState>,
}

struct ObserverState {
    /// Per-request expected-output residuals, exactly one live generation per
    /// request id (track replaces). Emptied entries are dropped — their
    /// inflight count keeps `has_pending` honest until the dispatch returns.
    entries: HashMap<RequestId, OutputEntry>,
    /// Per-generation in-flight `commit_output` dispatch counter.
    inflight: HashMap<(RequestId, AcceptId), u32>,
}

struct OutputEntry {
    /// The accept generation this residual belongs to.
    accept_id: AcceptId,
    /// Output hashes not yet seen registered.
    expected: HashSet<SequenceHash>,
    /// The ORIGINATING lifecycle, bound at track time. Dispatch upgrades this
    /// `Weak` and never re-fetches by request id — a fresh same-rid lifecycle
    /// can never receive a stale generation's blocks.
    state: Weak<PrefillRequestState>,
}

impl PrefillOutputObserver {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(ObserverState {
                entries: HashMap::new(),
                inflight: HashMap::new(),
            }),
        }
    }

    /// Track a lifecycle's expected-output set, replacing any prior (older
    /// generation) entry for the same request id. An empty set tracks nothing
    /// (the lifecycle owes no output); any stale entry is still replaced away.
    pub(crate) fn track(
        &self,
        request_id: RequestId,
        accept_id: AcceptId,
        expected: HashSet<SequenceHash>,
        state: Weak<PrefillRequestState>,
    ) {
        let mut inner = self.inner.lock();
        if expected.is_empty() {
            inner.entries.remove(&request_id);
            return;
        }
        inner.entries.insert(
            request_id,
            OutputEntry {
                accept_id,
                expected,
                state,
            },
        );
    }

    /// Drop the residual entry for `request_id` ONLY when it belongs to
    /// `accept_id` — a stale generation's untrack must not evict a fresh
    /// same-rid entry. Idempotent; leaves any inflight count to be balanced
    /// by its own dispatcher.
    pub(crate) fn untrack(&self, request_id: &RequestId, accept_id: AcceptId) {
        let mut inner = self.inner.lock();
        if inner
            .entries
            .get(request_id)
            .is_some_and(|e| e.accept_id == accept_id)
        {
            inner.entries.remove(request_id);
        }
    }

    /// Remove specific hashes from a generation's residual (the already-in-G2
    /// sweep at accept publishes those blocks itself). Drops the entry when
    /// the residual empties, mirroring `observe`'s emptied-entry drop.
    pub(crate) fn untrack_hashes(
        &self,
        request_id: &RequestId,
        accept_id: AcceptId,
        hashes: &[SequenceHash],
    ) {
        let mut inner = self.inner.lock();
        let Some(entry) = inner.entries.get_mut(request_id) else {
            return;
        };
        if entry.accept_id != accept_id {
            return;
        }
        for hash in hashes {
            entry.expected.remove(hash);
        }
        if entry.expected.is_empty() {
            inner.entries.remove(request_id);
        }
    }

    /// Whether generation `(request_id, accept_id)` still owes output: a
    /// non-empty residual OR an in-flight dispatch. The release drain polls
    /// this before finalizing the parked session.
    pub(crate) fn has_pending(&self, request_id: &RequestId, accept_id: AcceptId) -> bool {
        let inner = self.inner.lock();
        if inner
            .entries
            .get(request_id)
            .is_some_and(|e| e.accept_id == accept_id && !e.expected.is_empty())
        {
            return true;
        }
        inner
            .inflight
            .iter()
            .any(|((rid, aid), count)| rid == request_id && *aid == accept_id && *count > 0)
    }

    /// The generation's residual size (0 when absent) — watchdog diagnostics.
    pub(crate) fn residual(&self, request_id: &RequestId, accept_id: AcceptId) -> usize {
        self.inner
            .lock()
            .entries
            .get(request_id)
            .filter(|e| e.accept_id == accept_id)
            .map(|e| e.expected.len())
            .unwrap_or(0)
    }

    /// The register-step hook: match a freshly-registered batch against every
    /// residual, then dispatch the matches to their originating lifecycles.
    ///
    /// Runs synchronously on the offload executor's transfer task — the lock
    /// section is bounded (no session I/O) and the dispatches are sync bounded
    /// sends, so the callback never blocks the pipeline on anything slower
    /// than the session channel.
    pub(crate) fn observe(&self, blocks: &[ImmutableBlock<G2>]) {
        let mut matched: HashMap<RequestId, MatchedBatch> = HashMap::new();
        {
            let mut inner = self.inner.lock();
            for block in blocks {
                let hash = block.sequence_hash();
                // Linear scan over the live entries — N is the number of
                // concurrently accepted prefills (small). Mirrors the legacy
                // observer; swap for a hash→rid reverse index only if
                // profiling demands it.
                //
                // First match CLAIMS the block (the `break`): if two live
                // lifecycles expect the same output hash, the iterator-ordered
                // first one wins and the other's residual keeps the hash —
                // now G2-resident, it never re-registers, so that lifecycle
                // drains only via the release watchdog and its decode peer
                // bails on under-delivery (bounded failure, not a hang).
                // Exact legacy parity; revisit (clone to every matching
                // entry — decode dedups) only if production hits the case.
                for (rid, entry) in inner.entries.iter_mut() {
                    if entry.expected.remove(&hash) {
                        matched
                            .entry(rid.clone())
                            .or_insert_with(|| MatchedBatch {
                                accept_id: entry.accept_id,
                                state: entry.state.clone(),
                                blocks: Vec::new(),
                            })
                            .blocks
                            .push(block.clone());
                        break;
                    }
                }
            }
            // Emptied entries leave the map; their inflight bump below keeps
            // `has_pending` true until the dispatch lands.
            inner.entries.retain(|_, e| !e.expected.is_empty());
            // Bump inflight UNDER the lock that made the residual removal
            // visible, so no `has_pending` caller can see the gap.
            for (rid, batch) in &matched {
                *inner
                    .inflight
                    .entry((rid.clone(), batch.accept_id))
                    .or_default() += 1;
            }
        }

        // Dispatch with NO observer lock held: commit_output takes the
        // lifecycle's session-slot lock and performs session sends.
        for (rid, batch) in matched {
            match batch.state.upgrade() {
                Some(state) => state.commit_output(batch.blocks),
                None => {
                    // The lifecycle died after the residual was assembled —
                    // drop the pins and lazily evict the dead entry (its
                    // remaining residual can never be delivered).
                    tracing::debug!(
                        request_id = %rid,
                        "prefill output observer: lifecycle gone; dropping matched blocks"
                    );
                    self.untrack(&rid, batch.accept_id);
                }
            }
            self.finish_dispatch(&rid, batch.accept_id);
        }
    }

    /// Balance one inflight dispatch, dropping the counter entry at zero.
    fn finish_dispatch(&self, request_id: &RequestId, accept_id: AcceptId) {
        let mut inner = self.inner.lock();
        let key = (request_id.clone(), accept_id);
        if let Some(count) = inner.inflight.get_mut(&key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                inner.inflight.remove(&key);
            }
        }
    }

    /// Test accessor: whether generation `(request_id, accept_id)` has a
    /// residual entry.
    #[cfg(test)]
    pub(crate) fn is_tracked(&self, request_id: &RequestId, accept_id: AcceptId) -> bool {
        self.inner
            .lock()
            .entries
            .get(request_id)
            .is_some_and(|e| e.accept_id == accept_id)
    }

    /// Test accessor: the generation's in-flight dispatch count.
    #[cfg(test)]
    pub(crate) fn inflight_count(&self, request_id: &RequestId, accept_id: AcceptId) -> u32 {
        self.inner
            .lock()
            .inflight
            .iter()
            .find(|((rid, aid), _)| rid == request_id && *aid == accept_id)
            .map(|(_, count)| *count)
            .unwrap_or(0)
    }
}

/// One per-lifecycle slice of an observed batch, assembled under the lock and
/// dispatched outside it.
struct MatchedBatch {
    accept_id: AcceptId,
    state: Weak<PrefillRequestState>,
    blocks: Vec<ImmutableBlock<G2>>,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    use kvbm_logical::manager::BlockManager;

    use super::*;
    use crate::testing::managers::{TestManagerBuilder, TestRegistryBuilder};

    const BS: usize = 4;

    fn g2_manager(count: usize) -> Arc<BlockManager<G2>> {
        let registry = TestRegistryBuilder::new().build();
        Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(count)
                .block_size(BS)
                .registry(registry)
                .build(),
        )
    }

    fn h(i: u64) -> SequenceHash {
        SequenceHash::new(i, None, i)
    }

    /// Register blocks staged by explicit hashes (no token chain needed).
    fn blocks_for(
        manager: &Arc<BlockManager<G2>>,
        hashes: &[SequenceHash],
    ) -> Vec<ImmutableBlock<G2>> {
        let mutables = manager.allocate_blocks(hashes.len()).expect("alloc");
        let completes: Vec<_> = mutables
            .into_iter()
            .zip(hashes)
            .map(|(b, hash)| b.stage(*hash, BS).expect("stage"))
            .collect();
        manager.register_blocks(completes)
    }

    fn state() -> Arc<PrefillRequestState> {
        Arc::new(PrefillRequestState::new(
            AcceptId::new(),
            Vec::new(),
            Arc::new(AtomicUsize::new(0)),
            uuid::Uuid::new_v4(),
        ))
    }

    /// Generation scoping: an old-generation untrack must not evict a fresh
    /// same-rid entry, and `has_pending` answers per generation (the
    /// same-key-different-generation shape).
    #[test]
    fn untrack_and_has_pending_are_generation_scoped() {
        let observer = PrefillOutputObserver::new();
        let rid: RequestId = "rq".to_string();
        let aid1 = AcceptId::new();
        let aid2 = AcceptId::new();
        let s1 = state();
        let s2 = state();

        observer.track(rid.clone(), aid1, [h(1)].into(), Arc::downgrade(&s1));
        // Re-accept of the same rid replaces the entry with the fresh
        // generation.
        observer.track(rid.clone(), aid2, [h(2)].into(), Arc::downgrade(&s2));

        assert!(!observer.has_pending(&rid, aid1), "old generation is gone");
        assert!(observer.has_pending(&rid, aid2));

        // The old drain's untrack must not touch the fresh entry.
        observer.untrack(&rid, aid1);
        assert!(observer.is_tracked(&rid, aid2));
        assert!(observer.has_pending(&rid, aid2));

        observer.untrack(&rid, aid2);
        assert!(!observer.has_pending(&rid, aid2));
    }

    /// untrack_hashes shrinks only the matching generation's residual and
    /// drops the entry once empty.
    #[test]
    fn untrack_hashes_is_generation_scoped_and_drops_emptied() {
        let observer = PrefillOutputObserver::new();
        let rid: RequestId = "rq".to_string();
        let aid = AcceptId::new();
        let stale = AcceptId::new();
        let s = state();

        observer.track(rid.clone(), aid, [h(1), h(2)].into(), Arc::downgrade(&s));
        observer.untrack_hashes(&rid, stale, &[h(1)]);
        assert_eq!(observer.residual(&rid, aid), 2, "stale generation no-ops");

        observer.untrack_hashes(&rid, aid, &[h(1)]);
        assert_eq!(observer.residual(&rid, aid), 1);
        observer.untrack_hashes(&rid, aid, &[h(2)]);
        assert!(!observer.is_tracked(&rid, aid), "emptied entry dropped");
    }

    /// A dead lifecycle at dispatch time: the matched blocks are dropped, the
    /// inflight counter is balanced, and the dead entry is lazily evicted —
    /// even when its residual was only partially matched.
    #[test]
    fn observe_with_dead_lifecycle_drops_balances_and_untracks() {
        let observer = PrefillOutputObserver::new();
        let mgr = g2_manager(4);
        let rid: RequestId = "rq".to_string();
        let aid = AcceptId::new();

        let blocks = blocks_for(&mgr, &[h(10), h(11)]);
        let expected: HashSet<SequenceHash> = blocks.iter().map(|b| b.sequence_hash()).collect();

        let s = state();
        observer.track(rid.clone(), aid, expected, Arc::downgrade(&s));
        drop(s);

        // Match only 1 of 2: the residual is non-empty, but the dead Weak
        // must still evict the whole entry (it can never be delivered).
        observer.observe(&blocks[..1]);
        assert!(!observer.is_tracked(&rid, aid), "dead entry lazily evicted");
        assert_eq!(observer.inflight_count(&rid, aid), 0, "inflight balanced");
        assert!(!observer.has_pending(&rid, aid));

        // A late re-observe is a quiet no-op.
        observer.observe(&blocks);
        assert!(!observer.has_pending(&rid, aid));
    }
}
