// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! R7b §7 recovery matrix for the tier-placement consumer.
//!
//! Every assertion here checks one property. A projection answers empty after
//! it detects a continuity failure. These tests do not claim that every
//! transport loss is detectable. A final dropped batch has no later sequence
//! evidence. All reads use the public [`TierPlacementProjection::holders`] API.
//! Thus, a test cannot bypass the `valid` filter through an internal map.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{CacheManifestId, RegistrationEpoch};
use kvbm_protocols::tier_protocol::{
    InstanceId, KeyRange, PhysicalPlacementMode, PlacementScope, SealOutcome,
    TIER_MEDIUM_CAP_DIRECT_SERVABLE, TIER_PLACEMENT_SCHEMA_VERSION, TierDepth, TierMedium,
    TierPlacementBatchV1, TierPlacementEntry, TierPlacementManifest, TierPlacementOp,
    TierPlacementRejection, TierPlacementSequencer, TierPlacementSnapshotV1,
};

use super::SnapshotRequester;
use super::projection::{
    DeltaOutcome, DiscardReason, InvalidationReason, ProjectionLimits, SnapshotInstall,
    TierPlacementProjection, TierPlacementProjectionError,
};

const RESOURCE: LogicalResourceId = LogicalResourceId(3);
const OTHER_RESOURCE: LogicalResourceId = LogicalResourceId(9);
/// Depth 1, conventionally G2. Spelled locally: the wire type deliberately
/// names no depth but G1.
const G2: TierDepth = TierDepth(1);
const G3: TierDepth = TierDepth(2);

mod gate9;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[derive(Default)]
struct RecordingRequester {
    requests: Mutex<Vec<(CacheManifestId, InstanceId)>>,
}

impl RecordingRequester {
    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}

impl SnapshotRequester for RecordingRequester {
    fn request(&self, cache: CacheManifestId, instance: InstanceId) {
        self.requests.lock().unwrap().push((cache, instance));
    }
}

/// A projection with a manually advanced clock and a recording requester.
///
/// The clock advances rather than being frozen on purpose: the snapshot-request
/// rate limiter is time-based, and a frozen clock would silently suppress the
/// second request in every recovery test — the tests would then pass for the
/// wrong reason.
struct Harness {
    projection: TierPlacementProjection,
    requester: Arc<RecordingRequester>,
    registered: Arc<RwLock<HashSet<InstanceId>>>,
    clock: Arc<AtomicU64>,
    cache: CacheManifestId,
    instance: InstanceId,
    epoch: RegistrationEpoch,
}

impl Harness {
    fn new() -> Self {
        Self::with_limits(ProjectionLimits::default())
    }

    fn with_limits(limits: ProjectionLimits) -> Self {
        let instance = InstanceId::new_v4();
        let registered = Arc::new(RwLock::new(HashSet::from([instance])));
        let clock = Arc::new(AtomicU64::new(1_000));
        let clock_handle = Arc::clone(&clock);
        let projection = TierPlacementProjection::for_test(
            Arc::clone(&registered),
            Arc::new(move || clock_handle.load(Ordering::Relaxed)),
            limits,
        );
        let requester = Arc::new(RecordingRequester::default());
        assert!(projection.set_requester(Arc::clone(&requester) as Arc<dyn SnapshotRequester>));
        Self {
            projection,
            requester,
            registered,
            clock,
            cache: CacheManifestId::from_bytes([9; 32]),
            instance,
            epoch: RegistrationEpoch::new(),
        }
    }

    /// Advance past the snapshot-request rate limit window.
    fn tick(&self) {
        self.clock.fetch_add(5_000, Ordering::Relaxed);
    }

    fn batch(&self, seq: u64, generation: u64, ops: Vec<TierPlacementOp>) -> TierPlacementBatchV1 {
        TierPlacementBatchV1 {
            v: TIER_PLACEMENT_SCHEMA_VERSION,
            cache: self.cache,
            instance_id: self.instance,
            registration_epoch: self.epoch,
            seq,
            snapshot_generation: generation,
            ops,
        }
    }

    /// `media` is derived from `entries` rather than fixed: the snapshot header
    /// must describe every depth its body names, so a hard-coded G2-only header
    /// would make every multi-depth fixture fail validation for a reason the
    /// test is not about.
    fn snapshot(
        &self,
        generation: u64,
        seq_floor: u64,
        entries: Vec<TierPlacementEntry>,
    ) -> TierPlacementSnapshotV1 {
        TierPlacementSnapshotV1 {
            v: TIER_PLACEMENT_SCHEMA_VERSION,
            cache: self.cache,
            instance_id: self.instance,
            registration_epoch: self.epoch,
            snapshot_generation: generation,
            seq_floor,
            media: media_for(&entries),
            manifests: Vec::new(),
            entries,
        }
    }

    fn install(&self, snapshot: &TierPlacementSnapshotV1) -> SnapshotInstall {
        self.projection
            .install_snapshot(snapshot, self.epoch)
            .expect("snapshot installs")
    }

    fn holds(&self, hash: SequenceHash) -> bool {
        !self
            .projection
            .holders(self.cache, scope(RESOURCE), hash)
            .is_empty()
    }
}

fn scope(resource: LogicalResourceId) -> PlacementScope {
    PlacementScope::unitary(resource)
}

/// One medium per depth the entries name.
///
/// The snapshot header must describe every depth its body references, so a
/// hard-coded header would make multi-depth fixtures fail validation for a
/// reason no test here is about.
fn media_for(entries: &[TierPlacementEntry]) -> Vec<TierMedium> {
    let mut media: Vec<TierMedium> = Vec::new();
    for depth in entries.iter().map(|entry| entry.tier) {
        if !media.iter().any(|medium| medium.depth == depth) {
            media.push(TierMedium {
                depth,
                medium: "pinned-host".to_string(),
                capabilities: TIER_MEDIUM_CAP_DIRECT_SERVABLE,
            });
        }
    }
    media
}

/// A genuine PLH chain, built from parts so the fixtures depend on neither the
/// tokenizer nor a particular block-hash function.
fn chain(len: usize) -> Vec<SequenceHash> {
    let mut hashes = Vec::with_capacity(len);
    let mut current = SequenceHash::root(0xA11C_E000_0000_0001);
    hashes.push(current);
    for step in 1..len {
        current = current.extend(0xB0B0_0000_0000_0000 + step as u64);
        hashes.push(current);
    }
    hashes
}

fn ready(tier: TierDepth, generation: u64, hashes: Vec<SequenceHash>) -> TierPlacementOp {
    TierPlacementOp::Ready {
        scope: scope(RESOURCE),
        tier,
        placement: PhysicalPlacementMode::Whole,
        generation,
        keys: KeyRange::Hashes(hashes),
    }
}

fn remove(tier: TierDepth, generation: u64, hashes: Vec<SequenceHash>) -> TierPlacementOp {
    TierPlacementOp::Remove {
        scope: scope(RESOURCE),
        tier,
        generation,
        keys: KeyRange::Hashes(hashes),
    }
}

fn entry(tier: TierDepth, generation: u64, hashes: Vec<SequenceHash>) -> TierPlacementEntry {
    TierPlacementEntry {
        scope: scope(RESOURCE),
        tier,
        placement: PhysicalPlacementMode::Whole,
        generation,
        keys: KeyRange::Hashes(hashes),
    }
}

// ---------------------------------------------------------------------------
// §7.2 — gap ⇒ invalidate ⇒ snapshot ⇒ resume at seq_floor + 1
// ---------------------------------------------------------------------------

#[test]
fn gap_invalidates_requests_a_snapshot_and_resumes_at_the_floor() {
    let harness = Harness::new();
    let keys = chain(6);

    // A publisher's first delta finds no projection: create, ask, drop.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(1, 1, vec![ready(G2, 1, vec![keys[0]])])),
        DeltaOutcome::Invalidated(InvalidationReason::NoProjection)
    );
    assert_eq!(harness.requester.count(), 1);

    // The snapshot lands and the projection starts answering.
    harness.install(&harness.snapshot(1, 1, vec![entry(G2, 1, vec![keys[0]])]));
    assert!(harness.holds(keys[0]));

    // seq 2 applies in order.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(2, 1, vec![ready(G2, 1, vec![keys[1]])])),
        DeltaOutcome::Applied { seq: 2, ready: 2 }
    );
    assert!(harness.holds(keys[1]));

    // seq 4 is a gap: seq 3 was lost.
    harness.tick();
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(4, 1, vec![ready(G2, 1, vec![keys[2]])])),
        DeltaOutcome::Invalidated(InvalidationReason::SequenceGap)
    );
    assert_eq!(harness.requester.count(), 2);

    // Everything the projection knew is now unanswerable — including the keys
    // it had legitimately applied. Temporary miss, never stale success.
    assert!(!harness.holds(keys[0]));
    assert!(!harness.holds(keys[1]));

    // Further deltas are dropped while invalid, and (inside the rate-limit
    // window) do not amplify into more requests.
    for seq in 5..=7 {
        assert_eq!(
            harness.projection.apply_delta(&harness.batch(
                seq,
                1,
                vec![ready(G2, 1, vec![keys[3]])]
            )),
            DeltaOutcome::Discarded(DiscardReason::AwaitingSnapshot)
        );
    }
    assert_eq!(harness.requester.count(), 2, "rate limiter caps requests");
    assert_eq!(
        harness
            .projection
            .counters()
            .snapshot_requests_suppressed
            .load(Ordering::Relaxed),
        3
    );

    // Install at seq_floor = 6 ⇒ resume at 7.
    harness.install(&harness.snapshot(2, 6, vec![entry(G2, 1, vec![keys[4]])]));
    assert!(harness.holds(keys[4]));
    assert!(!harness.holds(keys[0]), "replace-all, not merge");

    // A late seq 5 is subsumed by the floor and discarded as a gap rather than
    // applied out of order.
    harness.tick();
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(5, 2, vec![ready(G2, 1, vec![keys[5]])])),
        DeltaOutcome::Invalidated(InvalidationReason::SequenceGap)
    );
    assert!(!harness.holds(keys[5]));

    harness.install(&harness.snapshot(3, 6, vec![entry(G2, 1, vec![keys[4]])]));
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(7, 3, vec![ready(G2, 1, vec![keys[5]])])),
        DeltaOutcome::Applied { seq: 7, ready: 2 }
    );
    assert!(harness.holds(keys[5]));
}

// ---------------------------------------------------------------------------
// §7.3 — an epoch bump replaces all prior state
// ---------------------------------------------------------------------------

#[test]
fn epoch_bump_replaces_all_prior_state_and_old_epoch_deltas_are_discarded() {
    let mut harness = Harness::new();
    let keys = chain(4);
    harness.install(&harness.snapshot(4, 10, vec![entry(G2, 1, vec![keys[0]])]));
    assert!(harness.holds(keys[0]));

    let old_epoch = harness.epoch;
    // The publisher restarts: new epoch, and its sequencer restarts at
    // generation 1 — well below the installed generation of 4.
    harness.epoch = RegistrationEpoch::new();

    harness.tick();
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(1, 1, vec![ready(G2, 1, vec![keys[1]])])),
        DeltaOutcome::Invalidated(InvalidationReason::UnknownEpoch)
    );
    assert!(!harness.holds(keys[0]), "prior epoch's state is gone");

    // The new epoch's snapshot must install even though its generation is lower
    // than the one installed for the previous epoch. Comparing generations
    // before epochs would strand this projection invalid forever, which is the
    // permanent failure §4 forbids.
    assert_eq!(
        harness.install(&harness.snapshot(1, 0, vec![entry(G2, 1, vec![keys[1]])])),
        SnapshotInstall::Installed {
            installed_generation: 1,
            seq_floor: 0,
        }
    );
    assert!(harness.holds(keys[1]));

    // A straggling delta from the old epoch cannot touch the new projection.
    let stale = TierPlacementBatchV1 {
        registration_epoch: old_epoch,
        ..harness.batch(1, 1, vec![ready(G2, 1, vec![keys[2]])])
    };
    harness.tick();
    assert_eq!(
        harness.projection.apply_delta(&stale),
        DeltaOutcome::Invalidated(InvalidationReason::UnknownEpoch)
    );
    assert!(!harness.holds(keys[2]));
}

/// The security half of the epoch rule: the delta plane is unauthenticated, so
/// a forged batch must not be able to install an epoch.
#[test]
fn a_forged_epoch_delta_cannot_flip_an_installed_projection() {
    let harness = Harness::new();
    let keys = chain(3);
    harness.install(&harness.snapshot(1, 0, vec![entry(G2, 1, vec![keys[0]])]));
    assert!(harness.holds(keys[0]));

    let forged = TierPlacementBatchV1 {
        registration_epoch: RegistrationEpoch::new(),
        ..harness.batch(1, 1, vec![ready(G2, 1, vec![keys[1]])])
    };

    // Round one: the forgery invalidates (fail-safe) but does not adopt its
    // epoch...
    harness.tick();
    assert_eq!(
        harness.projection.apply_delta(&forged),
        DeltaOutcome::Invalidated(InvalidationReason::UnknownEpoch)
    );
    assert!(!harness.holds(keys[0]));

    // ...so a genuine delta at the real epoch cannot be made to look wrong by
    // it, and a repeated forgery keeps hitting the same rule instead of
    // ping-ponging the projection between two "installed" epochs.
    for _ in 0..3 {
        harness.tick();
        assert_eq!(
            harness.projection.apply_delta(&forged),
            DeltaOutcome::Invalidated(InvalidationReason::UnknownEpoch)
        );
    }
    assert!(
        !harness.holds(keys[1]),
        "forged state never becomes visible"
    );

    // The authorized snapshot is still the only thing that can make it answer.
    harness.install(&harness.snapshot(2, 0, vec![entry(G2, 1, vec![keys[2]])]));
    assert!(harness.holds(keys[2]));
}

// ---------------------------------------------------------------------------
// §7.4 — generation ordering, both directions
// ---------------------------------------------------------------------------

#[test]
fn a_remove_cannot_evict_a_newer_ready_and_a_late_ready_cannot_resurrect_one() {
    let harness = Harness::new();
    let keys = chain(3);
    harness.install(&harness.snapshot(1, 0, Vec::new()));

    // Ready at generation 7.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(1, 1, vec![ready(G2, 7, vec![keys[0]])])),
        DeltaOutcome::Applied { seq: 1, ready: 1 }
    );

    // Remove(gen=5) is a late invalidation of a copy that has since been
    // replaced. It must not remove the generation-7 copy.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(2, 1, vec![remove(G2, 5, vec![keys[0]])])),
        DeltaOutcome::Applied { seq: 2, ready: 1 }
    );
    let holders = harness
        .projection
        .holders(harness.cache, scope(RESOURCE), keys[0]);
    assert_eq!(holders.len(), 1);
    assert_eq!(holders[0].generation, 7);

    // Remove at the same generation does evict.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(3, 1, vec![remove(G2, 7, vec![keys[0]])])),
        DeltaOutcome::Applied { seq: 3, ready: 0 }
    );
    assert!(!harness.holds(keys[0]));

    // The other direction, part one: a Ready at an older generation cannot
    // *overwrite* a newer record. This is the Occupied case — the record is
    // still present when the late Ready lands.
    harness
        .projection
        .apply_delta(&harness.batch(4, 1, vec![ready(G2, 7, vec![keys[1]])]));
    harness
        .projection
        .apply_delta(&harness.batch(5, 1, vec![ready(G2, 5, vec![keys[1]])]));
    let holders = harness
        .projection
        .holders(harness.cache, scope(RESOURCE), keys[1]);
    assert_eq!(holders.len(), 1);
    assert_eq!(
        holders[0].generation, 7,
        "an older Ready must not overwrite a newer one"
    );
}

/// R7b §7 test 4, the half the Occupied case above cannot reach: *"late
/// `Ready(gen=5)` after `Remove(gen=7)` is discarded"*.
///
/// Once the Remove has landed the key is absent, so a late Ready meets an empty
/// slot — and an unconditional insert there resurrects a copy the publisher
/// already said is gone, which `holders()` then reports as a live holder. That
/// is a stale success, the one failure this module exists to make impossible.
///
/// Both arrival shapes are covered, because they need different things to be
/// true. The intra-batch one needs no transport reordering at all: R7b §2 lists
/// several independent emission pipelines (offload commit, the eviction observer
/// batch, bundle invalidation), the batcher deliberately neither reorders nor
/// coalesces, so a source-side inversion is transmitted faithfully inside one
/// batch.
#[test]
fn a_late_ready_after_a_remove_cannot_resurrect_the_removed_copy() {
    let keys = chain(3);

    // Shape 1: one batch carrying the inversion.
    let harness = Harness::new();
    harness.install(&harness.snapshot(7, 0, vec![entry(G2, 7, vec![keys[0]])]));
    assert!(harness.holds(keys[0]));
    assert_eq!(
        harness.projection.apply_delta(&harness.batch(
            1,
            7,
            vec![remove(G2, 7, vec![keys[0]]), ready(G2, 5, vec![keys[0]])],
        )),
        DeltaOutcome::Applied { seq: 1, ready: 0 }
    );
    assert!(
        !harness.holds(keys[0]),
        "a Ready older than the Remove that preceded it resurrected the copy"
    );

    // Shape 2: the inversion split across batches, with the projection observed
    // in the correct empty state in between.
    let harness = Harness::new();
    harness.install(&harness.snapshot(1, 0, Vec::new()));
    harness
        .projection
        .apply_delta(&harness.batch(1, 1, vec![ready(G2, 7, vec![keys[1]])]));
    harness
        .projection
        .apply_delta(&harness.batch(2, 1, vec![remove(G2, 7, vec![keys[1]])]));
    assert!(!harness.holds(keys[1]));
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(3, 1, vec![ready(G2, 5, vec![keys[1]])])),
        DeltaOutcome::Applied { seq: 3, ready: 0 }
    );
    assert!(!harness.holds(keys[1]));

    // Equality loses, matching Remove's `<=` in the other direction: at one
    // generation the removal is the final word whichever order the two arrive
    // in. The cost is a re-offload at an unchanged generation staying
    // unadvertised until the next snapshot — a temporary miss, deliberately
    // preferred to a possible stale success.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(4, 1, vec![ready(G2, 7, vec![keys[1]])])),
        DeltaOutcome::Applied { seq: 4, ready: 0 }
    );
    assert!(!harness.holds(keys[1]));

    // A genuinely newer copy is not suppressed: the tombstone bounds the past,
    // it does not close the key.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(5, 1, vec![ready(G2, 8, vec![keys[1]])])),
        DeltaOutcome::Applied { seq: 5, ready: 1 }
    );
    assert!(harness.holds(keys[1]));

    // And the tombstone is gone with it, so a later Remove at a generation the
    // *record* beats still loses.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(6, 1, vec![remove(G2, 7, vec![keys[1]])])),
        DeltaOutcome::Applied { seq: 6, ready: 1 }
    );
    assert!(harness.holds(keys[1]));
}

/// A Remove that *loses* must write nothing at all: a key is live or
/// tombstoned, never both.
///
/// The observable consequence is retention, not ordering — worth stating
/// precisely, because the obvious framing ("a losing tombstone would suppress
/// the Ready that beat it") does not hold. A losing Remove carries a generation
/// *below* the record it met, so any Ready its tombstone could suppress would
/// have lost to that record anyway, and any Remove strong enough to clear the
/// record raises the tombstone above it. What a losing tombstone does do is
/// count the key twice against `max_ready_per_instance`, and that guard is
/// fail-safe: exceeding it invalidates and empties the projection. So a
/// publisher emitting late invalidations — the normal shape after an eviction
/// race — would burn retention budget it never used and blank an otherwise
/// healthy projection.
///
/// The guard is therefore where this is asserted, tight enough that one
/// spurious tombstone is the difference.
#[test]
fn a_losing_remove_consumes_neither_retention_budget_nor_the_record() {
    let harness = Harness::with_limits(ProjectionLimits {
        max_ready_per_instance: 2,
        ..ProjectionLimits::default()
    });
    let keys = chain(2);
    harness.install(&harness.snapshot(1, 0, Vec::new()));

    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(1, 1, vec![ready(G2, 7, keys.clone())])),
        DeltaOutcome::Applied { seq: 1, ready: 2 },
        "exactly at the retention guard"
    );

    // Loses against the generation-7 record, so it must be a no-op in both
    // dimensions: the record stays, and nothing new is retained.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(2, 1, vec![remove(G2, 5, vec![keys[0]])])),
        DeltaOutcome::Applied { seq: 2, ready: 2 },
        "a losing Remove that retained a tombstone would push past the guard here"
    );
    assert!(harness.holds(keys[0]));
    assert!(harness.projection.is_valid(harness.cache, harness.instance));

    // And a refresh still lands, so the no-op did not leave a suppressing
    // tombstone either.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(3, 1, vec![ready(G2, 7, vec![keys[0]])])),
        DeltaOutcome::Applied { seq: 3, ready: 2 }
    );
    assert!(harness.holds(keys[0]));
}

/// Tombstones are retained state, so they are inside the capacity guard rather
/// than beside it, and a snapshot install drops them: a delta sealed before an
/// install carries the older generation and is discarded as stale, so no
/// reordering can cross the boundary for a tombstone to guard against.
#[test]
fn tombstones_are_bounded_by_the_guard_and_cleared_by_an_install() {
    let harness = Harness::with_limits(ProjectionLimits {
        max_ready_per_instance: 4,
        ..ProjectionLimits::default()
    });
    let keys = chain(6);
    harness.install(&harness.snapshot(1, 0, Vec::new()));

    // Two live records plus two tombstones is exactly the guard.
    harness.projection.apply_delta(&harness.batch(
        1,
        1,
        vec![
            ready(G2, 1, keys[0..4].to_vec()),
            remove(G2, 1, keys[0..2].to_vec()),
        ],
    ));
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(2, 1, vec![remove(G2, 1, vec![keys[4]])])),
        DeltaOutcome::Invalidated(InvalidationReason::CapacityExceeded),
        "tombstones count against the retained-state guard"
    );

    // The install replaces everything, tombstones included, so the key that was
    // tombstoned before is servable again from the snapshot body.
    harness.tick();
    harness.install(&harness.snapshot(2, 10, vec![entry(G2, 1, vec![keys[0]])]));
    assert!(harness.holds(keys[0]));
}

#[test]
fn depths_are_tracked_independently() {
    // The same block Ready at two depths is two records: a G3 eviction must not
    // silently remove the G2 copy, which a depth-collapsing key would do.
    let harness = Harness::new();
    let keys = chain(2);
    harness.install(&harness.snapshot(1, 0, Vec::new()));
    harness.projection.apply_delta(&harness.batch(
        1,
        1,
        vec![ready(G2, 1, vec![keys[0]]), ready(G3, 1, vec![keys[0]])],
    ));
    assert_eq!(
        harness
            .projection
            .holders(harness.cache, scope(RESOURCE), keys[0])
            .len(),
        2
    );

    harness
        .projection
        .apply_delta(&harness.batch(2, 1, vec![remove(G3, 1, vec![keys[0]])]));
    let holders = harness
        .projection
        .holders(harness.cache, scope(RESOURCE), keys[0]);
    assert_eq!(holders.len(), 1);
    assert_eq!(holders[0].tier, G2);
}

#[test]
fn deltas_older_than_the_installed_generation_are_discarded_without_invalidating() {
    let harness = Harness::new();
    let keys = chain(2);
    harness.install(&harness.snapshot(5, 3, vec![entry(G2, 1, vec![keys[0]])]));

    // A delta still in flight from before the snapshot.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(4, 4, vec![ready(G2, 1, vec![keys[1]])])),
        DeltaOutcome::Discarded(DiscardReason::StaleGeneration)
    );
    assert!(harness.holds(keys[0]), "still answering");
    assert!(!harness.holds(keys[1]));
    assert_eq!(harness.requester.count(), 0, "stale is not a loss");

    // A delta from a generation the hub has not installed *is* a loss: the
    // publisher snapshotted and the hub never got it.
    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(4, 6, vec![ready(G2, 1, vec![keys[1]])])),
        DeltaOutcome::Invalidated(InvalidationReason::GenerationAhead)
    );
    assert_eq!(harness.requester.count(), 1);
    assert!(!harness.holds(keys[0]));
}

// ---------------------------------------------------------------------------
// §7.5 — ManifestInterval resolution
// ---------------------------------------------------------------------------

fn interval_op(
    manifest_id: u64,
    start: u32,
    len: u32,
    resource: LogicalResourceId,
) -> TierPlacementOp {
    TierPlacementOp::Ready {
        scope: scope(resource),
        tier: G2,
        placement: PhysicalPlacementMode::Whole,
        generation: 1,
        keys: KeyRange::ManifestInterval {
            manifest_id,
            start,
            len,
        },
    }
}

#[test]
fn an_installed_manifest_resolves_an_interval_exactly() {
    let harness = Harness::new();
    let keys = chain(8);
    let mut snapshot = harness.snapshot(1, 0, Vec::new());
    snapshot.manifests = vec![TierPlacementManifest {
        manifest_id: 42,
        resource: RESOURCE,
        hashes: keys.clone(),
    }];
    harness.install(&snapshot);

    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(1, 1, vec![interval_op(42, 2, 3, RESOURCE)])),
        DeltaOutcome::Applied { seq: 1, ready: 3 }
    );
    for (offset, key) in keys[2..5].iter().enumerate() {
        assert!(harness.holds(*key), "position {} resolved", offset + 2);
    }
    assert!(!harness.holds(keys[1]), "interval start is exclusive below");
    assert!(!harness.holds(keys[5]), "interval end is exclusive above");
}

#[test]
fn an_unresolvable_interval_is_a_sequence_gap_never_inferred_membership() {
    let keys = chain(8);
    // Each case is a different way for an interval to be unresolvable. None of
    // them may produce a partial or guessed membership: a positional lineage
    // hash carries one parent *fragment*, so there is nothing to walk back
    // through even in principle.
    let cases: Vec<(&str, TierPlacementOp)> = vec![
        ("unknown manifest id", interval_op(999, 0, 2, RESOURCE)),
        ("wrong resource", interval_op(42, 0, 2, OTHER_RESOURCE)),
        ("past the end", interval_op(42, 6, 5, RESOURCE)),
        ("start past the end", interval_op(42, 40, 1, RESOURCE)),
    ];

    for (label, op) in cases {
        let harness = Harness::new();
        let mut snapshot = harness.snapshot(1, 0, Vec::new());
        snapshot.manifests = vec![TierPlacementManifest {
            manifest_id: 42,
            resource: RESOURCE,
            hashes: keys.clone(),
        }];
        harness.install(&snapshot);

        assert_eq!(
            harness
                .projection
                .apply_delta(&harness.batch(1, 1, vec![op])),
            DeltaOutcome::Invalidated(InvalidationReason::UnresolvedInterval),
            "{label}"
        );
        assert_eq!(harness.requester.count(), 1, "{label}: snapshot requested");
        for key in &keys {
            assert!(!harness.holds(*key), "{label}: nothing was inferred");
        }
    }
}

#[test]
fn an_unresolvable_interval_does_not_partially_apply_the_batch() {
    // Key resolution happens before any mutation, so an op that resolves cannot
    // land just because it happened to sit before the one that did not.
    let harness = Harness::new();
    let keys = chain(4);
    let mut snapshot = harness.snapshot(1, 0, Vec::new());
    snapshot.manifests = vec![TierPlacementManifest {
        manifest_id: 42,
        resource: RESOURCE,
        hashes: keys.clone(),
    }];
    harness.install(&snapshot);

    assert_eq!(
        harness.projection.apply_delta(&harness.batch(
            1,
            1,
            vec![
                ready(G2, 1, vec![keys[0]]),
                interval_op(999, 0, 1, RESOURCE),
            ],
        )),
        DeltaOutcome::Invalidated(InvalidationReason::UnresolvedInterval)
    );
    // Re-install a clean projection and confirm the first op left no trace.
    harness.install(&harness.snapshot(2, 0, Vec::new()));
    assert!(!harness.holds(keys[0]));
}

// ---------------------------------------------------------------------------
// §7.6 — Reserved / Copying never reach the consumer
// ---------------------------------------------------------------------------

#[test]
fn local_only_states_are_undecodable_and_leave_the_projection_untouched() {
    let harness = Harness::new();
    let keys = chain(2);
    harness.install(&harness.snapshot(1, 0, vec![entry(G2, 1, vec![keys[0]])]));

    // There is no `Copying` variant, so this is an unknown-variant decode
    // failure — a structural guarantee rather than a runtime check.
    let forged = serde_json::json!({
        "v": 1,
        "cache": harness.cache,
        "instance_id": harness.instance,
        "registration_epoch": harness.epoch,
        "seq": 1,
        "snapshot_generation": 1,
        "ops": [{ "Copying": {
            "scope": { "resource": RESOURCE, "lane": 0 },
            "tier": 1,
            "placement": "Whole",
            "generation": 1,
            "keys": { "Hashes": [keys[1]] },
        }}],
    });
    let error = TierPlacementBatchV1::decode_json(&serde_json::to_vec(&forged).unwrap())
        .expect_err("Copying is not a wire state");
    assert_eq!(error.rejection(), TierPlacementRejection::Undecodable);

    // ...and the projection, never having seen it, still answers exactly what
    // the snapshot installed.
    assert!(harness.holds(keys[0]));
    assert!(!harness.holds(keys[1]));
}

#[test]
fn a_g1_batch_is_dropped_without_invalidating() {
    // `validate()` rejects depth 0 before the projection ever sees it. The drop
    // deliberately does *not* invalidate: the publisher's sequencer does not
    // consume a number for a batch it refused to send, so a batch the hub
    // rejects is followed by a seq the hub reads as a gap and recovers from.
    // Two invalidation policies for one condition would just be two ways to be
    // wrong.
    let harness = Harness::new();
    let keys = chain(2);
    harness.install(&harness.snapshot(1, 0, vec![entry(G2, 1, vec![keys[0]])]));

    let outcome = harness.projection.apply_delta(&harness.batch(
        1,
        1,
        vec![ready(TierDepth::G1, 1, vec![keys[1]])],
    ));
    assert_eq!(
        outcome,
        DeltaOutcome::Discarded(DiscardReason::Rejected(TierPlacementRejection::Invalid))
    );
    assert!(harness.holds(keys[0]), "still valid");
    assert_eq!(harness.requester.count(), 0);
}

// ---------------------------------------------------------------------------
// §7.7 — snapshot install is atomic under concurrent delta arrival
// ---------------------------------------------------------------------------

#[test]
fn snapshot_install_is_atomic_against_a_concurrent_delta() {
    let instance = InstanceId::new_v4();
    let registered = Arc::new(RwLock::new(HashSet::from([instance])));
    let cache = CacheManifestId::from_bytes([9; 32]);
    let epoch = RegistrationEpoch::new();
    let keys = chain(8);

    // The hook fires after the replacement has been built and before the swap —
    // exactly the window a torn read would have to live in. It is armed only
    // for the racing install; the seeding install below runs on this thread and
    // would otherwise block on its own gate.
    let armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let hook_gate = Arc::clone(&gate);
    let hook_armed = Arc::clone(&armed);
    let projection = Arc::new(TierPlacementProjection::with_install_hook(
        registered,
        Arc::new(move || {
            if !hook_armed.load(Ordering::Relaxed) {
                return;
            }
            let (lock, cv) = &*hook_gate;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = cv.wait(released).unwrap();
            }
        }),
    ));
    projection.set_requester(Arc::new(RecordingRequester::default()));

    let batch = |seq: u64, hashes: Vec<SequenceHash>| TierPlacementBatchV1 {
        v: TIER_PLACEMENT_SCHEMA_VERSION,
        cache,
        instance_id: instance,
        registration_epoch: epoch,
        seq,
        snapshot_generation: 1,
        ops: vec![ready(G2, 1, hashes)],
    };
    let snapshot =
        |generation: u64, seq_floor: u64, hashes: Vec<SequenceHash>| TierPlacementSnapshotV1 {
            v: TIER_PLACEMENT_SCHEMA_VERSION,
            cache,
            instance_id: instance,
            registration_epoch: epoch,
            snapshot_generation: generation,
            seq_floor,
            media: media_for(&[entry(G2, 1, hashes.clone())]),
            manifests: Vec::new(),
            entries: vec![entry(G2, 1, hashes)],
        };

    // Seed a projection holding keys 0..2, then install a snapshot whose set
    // differs in *three* keys — with a one-key difference a partial application
    // would not be observable and the assertion could not fail.
    projection
        .install_snapshot(&snapshot(1, 0, keys[0..3].to_vec()), epoch)
        .unwrap();
    armed.store(true, Ordering::Relaxed);

    std::thread::scope(|threads| {
        let installer = threads
            .spawn(|| projection.install_snapshot(&snapshot(2, 10, keys[5..8].to_vec()), epoch));

        // Let the installer reach the hook, then race a delta into the window.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let delta = projection.apply_delta(&batch(1, vec![keys[3], keys[4]]));
        assert_eq!(delta, DeltaOutcome::Applied { seq: 1, ready: 5 });

        let (lock, cv) = &*gate;
        *lock.lock().unwrap() = true;
        cv.notify_all();

        assert_eq!(
            installer.join().unwrap().unwrap(),
            SnapshotInstall::Installed {
                installed_generation: 2,
                seq_floor: 10,
            }
        );
    });

    // The post-install state is exactly the snapshot's — not a blend with the
    // delta that raced it. Replace-all means the delta's keys are gone, and
    // `last_seq` is the floor, so the publisher's seq 11 resumes cleanly.
    for key in &keys[5..8] {
        assert!(
            !projection.holders(cache, scope(RESOURCE), *key).is_empty(),
            "snapshot key present"
        );
    }
    for key in keys[0..5].iter() {
        assert!(
            projection.holders(cache, scope(RESOURCE), *key).is_empty(),
            "pre-install and raced-delta keys are gone"
        );
    }
    // `last_seq == seq_floor`: the raced delta's sequence position was subsumed,
    // so continuity resumes at the floor rather than at the delta.
    assert_eq!(
        projection.apply_delta(&TierPlacementBatchV1 {
            seq: 11,
            snapshot_generation: 2,
            ..batch(11, vec![keys[0]])
        }),
        DeltaOutcome::Applied { seq: 11, ready: 4 }
    );
}

// ---------------------------------------------------------------------------
// The invariant, stated directly
// ---------------------------------------------------------------------------

#[test]
fn an_invalid_projection_answers_empty_and_the_same_query_answers_after_install() {
    let harness = Harness::new();
    let keys = chain(2);
    harness.install(&harness.snapshot(1, 0, vec![entry(G2, 1, vec![keys[0]])]));

    let query = || {
        harness
            .projection
            .holders(harness.cache, scope(RESOURCE), keys[0])
    };
    assert_eq!(query().len(), 1, "valid ⇒ answers");

    harness.tick();
    harness
        .projection
        .apply_delta(&harness.batch(99, 1, vec![ready(G2, 1, vec![keys[1]])]));
    assert!(!harness.projection.is_valid(harness.cache, harness.instance));
    assert!(
        query().is_empty(),
        "invalid ⇒ empty, not the pre-gap answer"
    );

    harness.install(&harness.snapshot(2, 99, vec![entry(G2, 1, vec![keys[0]])]));
    assert_eq!(query().len(), 1, "install ⇒ answers again");
}

#[test]
fn a_hub_with_no_transport_stays_invalid_rather_than_serving_stale_state() {
    // `attach` installs a requester only when the hub has velo. A discovery-only
    // hub therefore cannot recover — and must degrade to a permanent empty, not
    // to a stale-serving fallback.
    let instance = InstanceId::new_v4();
    let registered = Arc::new(RwLock::new(HashSet::from([instance])));
    let projection = TierPlacementProjection::new(registered);
    let cache = CacheManifestId::from_bytes([9; 32]);
    let epoch = RegistrationEpoch::new();
    let keys = chain(2);

    projection
        .install_snapshot(
            &TierPlacementSnapshotV1 {
                v: TIER_PLACEMENT_SCHEMA_VERSION,
                cache,
                instance_id: instance,
                registration_epoch: epoch,
                snapshot_generation: 1,
                seq_floor: 0,
                media: media_for(&[entry(G2, 1, vec![keys[0]])]),
                manifests: Vec::new(),
                entries: vec![entry(G2, 1, vec![keys[0]])],
            },
            epoch,
        )
        .unwrap();
    assert_eq!(projection.holders(cache, scope(RESOURCE), keys[0]).len(), 1);

    // No requester installed: the request is a no-op, but the invalidation is
    // real.
    projection.apply_delta(&TierPlacementBatchV1 {
        v: TIER_PLACEMENT_SCHEMA_VERSION,
        cache,
        instance_id: instance,
        registration_epoch: epoch,
        seq: 77,
        snapshot_generation: 1,
        ops: vec![ready(G2, 1, vec![keys[1]])],
    });
    assert!(
        projection
            .holders(cache, scope(RESOURCE), keys[0])
            .is_empty()
    );
}

// ---------------------------------------------------------------------------
// Lifecycle and admission
// ---------------------------------------------------------------------------

#[test]
fn deregistration_drops_every_cache_for_that_instance() {
    let harness = Harness::new();
    let other_cache = CacheManifestId::from_bytes([11; 32]);
    let keys = chain(2);
    harness.install(&harness.snapshot(1, 0, vec![entry(G2, 1, vec![keys[0]])]));
    harness.install(&TierPlacementSnapshotV1 {
        cache: other_cache,
        ..harness.snapshot(1, 0, vec![entry(G2, 1, vec![keys[1]])])
    });
    assert!(harness.holds(keys[0]));
    assert_eq!(
        harness
            .projection
            .holders(other_cache, scope(RESOURCE), keys[1])
            .len(),
        1
    );

    harness.projection.remove_instance(harness.instance);
    assert!(!harness.holds(keys[0]));
    assert!(
        harness
            .projection
            .holders(other_cache, scope(RESOURCE), keys[1])
            .is_empty()
    );
}

#[test]
fn deltas_from_unregistered_instances_are_counted_and_dropped() {
    // Gating creation on the registered set bounds the map with state the hub
    // authenticates. A bare numeric cap would let an unauthenticated flood of
    // random ids fill it and starve genuine instances — a *permanent* empty
    // rather than a temporary one.
    let harness = Harness::new();
    let keys = chain(1);
    harness.registered.write().unwrap().clear();

    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(1, 1, vec![ready(G2, 1, vec![keys[0]])])),
        DeltaOutcome::Discarded(DiscardReason::UnregisteredInstance)
    );
    assert_eq!(
        harness
            .projection
            .counters()
            .discarded_unregistered
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(harness.requester.count(), 0);
}

/// The registered-instance gate bounds the *instance* half of the key. `cache`
/// is the other half, it is publisher-chosen, and the delta plane that carries
/// it is unauthenticated — so a delta must never create an entry.
///
/// If it did, one registered instance id plus N forged cache ids would fill
/// `max_instances` with entries that are inert by construction (only an
/// authorized install writes `installed_epoch`, so they can never become valid)
/// and that nothing ages out. And the damage would outlast the flood: the
/// snapshot install is the only route back to valid, and it refuses at the same
/// bound.
#[test]
fn forged_cache_ids_neither_fill_the_projection_nor_block_a_genuine_install() {
    let harness = Harness::with_limits(ProjectionLimits {
        max_instances: 2,
        ..ProjectionLimits::default()
    });
    let keys = chain(2);

    for byte in 0..8u8 {
        let mut forged = harness.batch(1, 1, vec![ready(G2, 1, vec![keys[0]])]);
        forged.cache = CacheManifestId::from_bytes([byte; 32]);
        forged.registration_epoch = RegistrationEpoch::new();
        assert_eq!(
            harness.projection.apply_delta(&forged),
            DeltaOutcome::Invalidated(InvalidationReason::NoProjection),
            "a delta for an unknown (cache, instance) reports the miss and creates nothing"
        );
    }
    // And the amplification is bounded too: the request rate limit is keyed on
    // the instance, which the registered set bounds, not on the forged cache.
    assert_eq!(harness.requester.count(), 1);

    // The genuine cache installs, past a bound eight forged frames would have
    // exhausted, and answers.
    harness.install(&harness.snapshot(1, 0, vec![entry(G2, 1, vec![keys[0]])]));
    assert!(harness.holds(keys[0]));

    // Deregistration prunes the stamps alongside the projections, so a
    // re-registered publisher is not silently rate-limited by its predecessor.
    harness.projection.remove_instance(harness.instance);
    let mut forged = harness.batch(1, 1, vec![ready(G2, 1, vec![keys[0]])]);
    forged.cache = CacheManifestId::from_bytes([0; 32]);
    harness.projection.apply_delta(&forged);
    assert_eq!(harness.requester.count(), 2);
}

/// An internal fault must not be counted as normal recovery back-pressure: a
/// healthy projection bumps `discarded_awaiting_snapshot` on every delta while
/// it waits for a snapshot, so burying a poisoned lock there would make the
/// per-reason counters unable to answer the question they exist for.
#[test]
fn a_poisoned_state_lock_is_reported_as_unavailable_not_as_back_pressure() {
    let harness = Harness::new();
    let keys = chain(1);
    harness.install(&harness.snapshot(1, 0, vec![entry(G2, 1, vec![keys[0]])]));
    harness.projection.poison_state_for_test();

    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(1, 1, vec![ready(G2, 1, vec![keys[0]])])),
        DeltaOutcome::Discarded(DiscardReason::Unavailable)
    );
    let counters = harness.projection.counters();
    assert_eq!(counters.discarded_unavailable.load(Ordering::Relaxed), 1);
    assert_eq!(
        counters.discarded_awaiting_snapshot.load(Ordering::Relaxed),
        0
    );

    // The install path already separated the two; assert the pair stays
    // consistent rather than trusting that it does.
    assert_eq!(
        harness
            .projection
            .install_snapshot(&harness.snapshot(2, 0, Vec::new()), harness.epoch),
        Err(TierPlacementProjectionError::Unavailable)
    );
    // And the read path degrades to empty rather than panicking.
    assert!(!harness.holds(keys[0]));
    assert!(!harness.projection.is_valid(harness.cache, harness.instance));
}

/// `ready_placement` is the single-instance question CT-2a asks once it has
/// chosen a peer. It answers the shallowest depth that instance holds, ignores
/// every other holder, and inherits the validity filter.
///
/// Also the one exercise of the public `with_limits` constructor: a
/// `ProjectionLimits` an external caller can name but cannot pass would be dead
/// public surface.
#[test]
fn ready_placement_answers_for_one_instance_only_and_respects_validity() {
    let mine = InstanceId::new_v4();
    let theirs = InstanceId::new_v4();
    let registered = Arc::new(RwLock::new(HashSet::from([mine, theirs])));
    let projection =
        TierPlacementProjection::with_limits(Arc::clone(&registered), ProjectionLimits::default());
    let cache = CacheManifestId::from_bytes([9; 32]);
    let keys = chain(1);

    let snapshot = |instance: InstanceId, epoch: RegistrationEpoch, tiers: &[TierDepth]| {
        let entries: Vec<TierPlacementEntry> = tiers
            .iter()
            .map(|tier| entry(*tier, 1, vec![keys[0]]))
            .collect();
        TierPlacementSnapshotV1 {
            v: TIER_PLACEMENT_SCHEMA_VERSION,
            cache,
            instance_id: instance,
            registration_epoch: epoch,
            snapshot_generation: 1,
            seq_floor: 0,
            media: media_for(&entries),
            manifests: Vec::new(),
            entries,
        }
    };
    let my_epoch = RegistrationEpoch::new();
    let their_epoch = RegistrationEpoch::new();
    projection
        .install_snapshot(&snapshot(mine, my_epoch, &[G3, G2]), my_epoch)
        .expect("installs");
    projection
        .install_snapshot(&snapshot(theirs, their_epoch, &[G2]), their_epoch)
        .expect("installs");

    assert_eq!(
        projection.holders(cache, scope(RESOURCE), keys[0]).len(),
        3,
        "two instances, three placements"
    );
    let placement = projection
        .ready_placement(cache, mine, scope(RESOURCE), keys[0])
        .expect("mine is a holder");
    assert_eq!(placement.instance, mine);
    assert_eq!(placement.tier, G2, "shallowest depth this instance holds");

    // A different instance's placements never leak into the answer.
    assert_eq!(
        projection
            .ready_placement(cache, theirs, scope(RESOURCE), keys[0])
            .expect("theirs is a holder")
            .instance,
        theirs
    );
    assert!(
        projection
            .ready_placement(cache, InstanceId::new_v4(), scope(RESOURCE), keys[0])
            .is_none()
    );

    // Same validity filter as `holders`: an invalid projection answers nothing.
    projection.remove_instance(mine);
    assert!(
        projection
            .ready_placement(cache, mine, scope(RESOURCE), keys[0])
            .is_none()
    );
}

/// The reader supplies hash, resource and lane; depth is the fourth key
/// component and it does not have one. The projection's depth set stands in for
/// it, and it is deliberately a monotone *superset* — `Remove` never withdraws a
/// depth.
///
/// So the property to pin is that a member with no records behind it is inert:
/// it costs a lookup that finds nothing and cannot make an answer wrong, in
/// either direction.
#[test]
fn the_depth_index_is_a_superset_that_never_changes_an_answer() {
    let harness = Harness::new();
    let keys = chain(2);
    harness.install(&harness.snapshot(1, 0, Vec::new()));

    harness.projection.apply_delta(&harness.batch(
        1,
        1,
        vec![ready(G2, 1, vec![keys[0]]), ready(G3, 1, vec![keys[0]])],
    ));
    assert_eq!(
        harness
            .projection
            .holders(harness.cache, scope(RESOURCE), keys[0])
            .len(),
        2
    );

    // Emptying a depth leaves it in the index. The answers must not notice.
    harness
        .projection
        .apply_delta(&harness.batch(2, 1, vec![remove(G3, 1, vec![keys[0]])]));
    let holders = harness
        .projection
        .holders(harness.cache, scope(RESOURCE), keys[0]);
    assert_eq!(holders.len(), 1);
    assert_eq!(holders[0].tier, G2);
    assert_eq!(
        harness
            .projection
            .ready_placement(harness.cache, harness.instance, scope(RESOURCE), keys[0])
            .expect("still a holder at G2")
            .tier,
        G2
    );

    // A key that never existed at any depth answers nothing, however many
    // depths the index carries.
    assert!(!harness.holds(keys[1]));
    // Nor does another scope pick up this key's records.
    assert!(
        harness
            .projection
            .holders(harness.cache, scope(OTHER_RESOURCE), keys[0])
            .is_empty()
    );
    assert!(
        harness
            .projection
            .holders(
                harness.cache,
                PlacementScope {
                    resource: RESOURCE,
                    lane: 1
                },
                keys[0]
            )
            .is_empty()
    );

    // And a depth can come back without a reinstall.
    harness
        .projection
        .apply_delta(&harness.batch(3, 1, vec![ready(G3, 2, vec![keys[0]])]));
    assert_eq!(
        harness
            .projection
            .holders(harness.cache, scope(RESOURCE), keys[0])
            .len(),
        2
    );
}

/// A single-hash read must not cost the ready set.
///
/// A wall-clock assertion, which normally earns a flaky test — this one is
/// defensible because the margin is four orders of magnitude, not two. The
/// lookup loop is ~1 ms with the depth index (each read is one outer-map visit,
/// one depth, one hash lookup) against a 5 s budget. Restoring the scan this
/// replaced was measured at 15.7 s for exactly this loop, so the test fails on
/// the regression and has ~10,000x headroom on a loaded machine.
///
/// Worth an explicit gate because the cost is invisible from the outside: a
/// scan-based read is *correct*, just quadratic in fleet size x ready set, and
/// CT-2a calls this once per candidate block with the read lock held.
#[test]
fn a_single_hash_read_does_not_scan_the_ready_set() {
    const READY: usize = 100_000;
    const LOOKUPS: usize = 5_000;
    const BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

    let harness = Harness::new();
    let keys = chain(READY);
    harness.install(&harness.snapshot(1, 0, vec![entry(G2, 1, keys.clone())]));

    let started = std::time::Instant::now();
    for index in 0..LOOKUPS {
        let hash = keys[index * (READY / LOOKUPS)];
        let holders = harness
            .projection
            .holders(harness.cache, scope(RESOURCE), hash);
        assert_eq!(holders.len(), 1);
        assert_eq!(holders[0].tier, G2);
    }
    // A miss is the same cost and the same answer.
    assert!(
        harness
            .projection
            .holders(
                harness.cache,
                scope(RESOURCE),
                SequenceHash::root(0xDEAD_BEEF)
            )
            .is_empty()
    );
    let elapsed = started.elapsed();
    assert!(
        elapsed < BUDGET,
        "{LOOKUPS} single-hash reads over {READY} ready records took {elapsed:?}; \
         the read path is scanning the ready set again"
    );
}

#[test]
fn a_ready_set_beyond_the_guard_empties_the_projection_rather_than_truncating_it() {
    let harness = Harness::with_limits(ProjectionLimits {
        max_ready_per_instance: 3,
        ..ProjectionLimits::default()
    });
    let keys = chain(6);
    harness.install(&harness.snapshot(1, 0, vec![entry(G2, 1, vec![keys[0]])]));

    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(1, 1, vec![ready(G2, 1, keys.clone())])),
        DeltaOutcome::Invalidated(InvalidationReason::CapacityExceeded)
    );
    // Truncation would be a partial membership claim — the thing this module is
    // built to never do.
    for key in &keys {
        assert!(!harness.holds(*key));
    }
}

// ---------------------------------------------------------------------------
// Snapshot endpoint semantics
// ---------------------------------------------------------------------------

#[test]
fn a_snapshot_claiming_another_epoch_is_refused_without_changing_state() {
    let harness = Harness::new();
    let keys = chain(2);
    harness.install(&harness.snapshot(1, 0, vec![entry(G2, 1, vec![keys[0]])]));

    let mut forged = harness.snapshot(2, 0, vec![entry(G2, 1, vec![keys[1]])]);
    forged.registration_epoch = RegistrationEpoch::new();
    assert_eq!(
        harness
            .projection
            .install_snapshot(&forged, harness.epoch)
            .expect_err("body epoch must match the credential's"),
        TierPlacementProjectionError::EpochMismatch
    );
    assert!(harness.holds(keys[0]), "state unchanged");
    assert!(!harness.holds(keys[1]));
}

#[test]
fn a_periodic_push_the_hub_already_holds_is_accepted_but_installs_nothing() {
    let harness = Harness::new();
    let keys = chain(2);
    let snapshot = harness.snapshot(4, 10, vec![entry(G2, 1, vec![keys[0]])]);
    harness.install(&snapshot);

    // R7b §3 has the publisher pushing every 60 s whether or not anything was
    // lost, so a re-push of the installed generation is the common case, not an
    // error.
    assert_eq!(
        harness.install(&snapshot),
        SnapshotInstall::AlreadyCurrent {
            installed_generation: 4,
        }
    );
    assert!(harness.holds(keys[0]));

    // An older generation is likewise refused rather than rolling state back.
    assert_eq!(
        harness.install(&harness.snapshot(3, 10, vec![entry(G2, 1, vec![keys[1]])])),
        SnapshotInstall::AlreadyCurrent {
            installed_generation: 4,
        }
    );
    assert!(!harness.holds(keys[1]));
}

#[test]
fn a_re_push_at_the_installed_generation_un_sticks_an_invalid_projection() {
    // Same generation, but the projection is invalid: re-installing is harmless
    // (identical state) and strictly better than waiting for the publisher's
    // next generation bump.
    let harness = Harness::new();
    let keys = chain(2);
    let snapshot = harness.snapshot(4, 10, vec![entry(G2, 1, vec![keys[0]])]);
    harness.install(&snapshot);

    harness.tick();
    harness
        .projection
        .apply_delta(&harness.batch(99, 4, vec![ready(G2, 1, vec![keys[1]])]));
    assert!(!harness.projection.is_valid(harness.cache, harness.instance));

    assert_eq!(
        harness.install(&snapshot),
        SnapshotInstall::Installed {
            installed_generation: 4,
            seq_floor: 10,
        }
    );
    assert!(harness.holds(keys[0]));
    // The rate-limit stamp is cleared on install, so the *next* loss can ask
    // immediately instead of waiting out a window it did not use.
    let before = harness.requester.count();
    harness
        .projection
        .apply_delta(&harness.batch(99, 4, vec![ready(G2, 1, vec![keys[1]])]));
    assert_eq!(harness.requester.count(), before + 1);
}

/// Seal a batch that must not be deferred.
fn seal(publisher: &mut TierPlacementSequencer, ops: Vec<TierPlacementOp>) -> TierPlacementBatchV1 {
    match publisher.seal(ops).expect("valid ops") {
        SealOutcome::Batch(batch) => batch,
        SealOutcome::Deferred(_) => panic!("the emission gate is armed here"),
    }
}

/// A publisher that keeps emitting across its own snapshot push does not merely
/// blink — it re-gaps after *every* install, indefinitely.
///
/// Driven by the real [`TierPlacementSequencer`] rather than by hand-written
/// sequence numbers, because the number the publisher would actually stamp next
/// is the whole point. An earlier version of this test re-applied the *same*
/// batch at the same `seq` after the install and called the result "heals";
/// that models a retransmission ZMQ pub/sub does not provide. The publisher's
/// sequencer already consumed that number for the batch the hub dropped, so the
/// real next batch is `seq_floor + 2` and it gaps — and the recovery snapshot
/// that follows loses the same race again.
#[test]
fn an_ungated_publisher_re_gaps_after_every_snapshot_install() {
    let harness = Harness::new();
    let keys = chain(4);
    let mut publisher = TierPlacementSequencer::new(harness.cache, harness.instance, harness.epoch);

    let boot = publisher
        .snapshot(media_for(&[]), Vec::new(), Vec::new())
        .expect("valid snapshot");
    // Ungated: the publisher releases emission without waiting for the install.
    assert!(publisher.note_snapshot_installed(boot.snapshot_generation));
    harness.install(&boot);
    let first = seal(&mut publisher, vec![ready(G2, 1, vec![keys[0]])]);
    assert_eq!(
        harness.projection.apply_delta(&first),
        DeltaOutcome::Applied { seq: 1, ready: 1 }
    );

    for round in 0..3 {
        harness.tick();
        let entries = vec![entry(G2, 1, vec![keys[0]])];
        let snapshot = publisher
            .snapshot(media_for(&entries), Vec::new(), entries)
            .expect("valid snapshot");
        assert!(publisher.note_snapshot_installed(snapshot.snapshot_generation));

        // The delta that beats the HTTP push. Dropped, and never retransmitted.
        let outran = seal(&mut publisher, vec![ready(G2, 1, vec![keys[1]])]);
        harness.projection.apply_delta(&outran);

        // The push lands: valid again, for exactly one delta's worth of time.
        harness.install(&snapshot);
        assert!(harness.projection.is_valid(harness.cache, harness.instance));

        let next = seal(&mut publisher, vec![ready(G2, 1, vec![keys[2]])]);
        assert_eq!(
            harness.projection.apply_delta(&next),
            DeltaOutcome::Invalidated(InvalidationReason::SequenceGap),
            "round {round}: the outrun delta is gone for good, so the next one gaps"
        );
        assert!(
            !harness.holds(keys[0]),
            "round {round}: back to answering empty"
        );
    }
}

/// The publisher-side fix: sealing a snapshot arms an emission gate, so the
/// first post-snapshot delta cannot precede the state it describes. The ops are
/// held, not dropped — `seal` hands them back and consumes no sequence number —
/// and the re-seal after the ack lands exactly on the consumer's resume point.
#[test]
fn the_publisher_emission_gate_closes_the_outrun_race() {
    let harness = Harness::new();
    let keys = chain(4);
    let mut publisher = TierPlacementSequencer::new(harness.cache, harness.instance, harness.epoch);

    let boot = publisher
        .snapshot(media_for(&[]), Vec::new(), Vec::new())
        .expect("valid snapshot");
    let SnapshotInstall::Installed {
        installed_generation,
        ..
    } = harness.install(&boot)
    else {
        panic!("the bootstrap snapshot installs");
    };
    assert!(publisher.note_snapshot_installed(installed_generation));

    for round in 0..3 {
        harness.tick();
        let entries = vec![entry(G2, 1, vec![keys[0]])];
        let snapshot = publisher
            .snapshot(media_for(&entries), Vec::new(), entries)
            .expect("valid snapshot");

        let held = vec![ready(G2, 1, vec![keys[1]])];
        assert_eq!(
            publisher
                .seal(held.clone())
                .expect("deferral is not an error"),
            SealOutcome::Deferred(held.clone()),
            "round {round}: emission is held across the push window"
        );

        let SnapshotInstall::Installed {
            installed_generation,
            seq_floor,
        } = harness.install(&snapshot)
        else {
            panic!("round {round}: the snapshot installs");
        };
        assert!(publisher.note_snapshot_installed(installed_generation));

        let released = seal(&mut publisher, held);
        assert_eq!(released.seq, seq_floor + 1, "round {round}");
        assert_eq!(
            harness.projection.apply_delta(&released),
            DeltaOutcome::Applied {
                seq: seq_floor + 1,
                ready: 2,
            },
            "round {round}: no gap, no generation-ahead"
        );
        assert!(harness.holds(keys[0]), "round {round}");
        assert!(harness.holds(keys[1]), "round {round}");
    }

    // Nothing was lost across three pushes: no invalidation of any kind fired.
    let counters = harness.projection.counters();
    assert_eq!(
        counters
            .invalidated_generation_ahead
            .load(Ordering::Relaxed),
        0
    );
    assert_eq!(counters.invalidated_sequence_gap.load(Ordering::Relaxed), 0);
}

#[test]
fn a_snapshot_carrying_a_manifest_interval_is_refused() {
    let harness = Harness::new();
    let mut inexact = harness.snapshot(1, 0, Vec::new());
    inexact.entries = vec![TierPlacementEntry {
        scope: scope(RESOURCE),
        tier: G2,
        placement: PhysicalPlacementMode::Whole,
        generation: 1,
        keys: KeyRange::ManifestInterval {
            manifest_id: 1,
            start: 0,
            len: 2,
        },
    }];
    // A consumer installing a *recovery* snapshot has by definition discarded
    // the projection an interval would resolve against, so an inexact snapshot
    // is unrecoverable by construction.
    assert!(matches!(
        harness
            .projection
            .install_snapshot(&inexact, harness.epoch)
            .expect_err("snapshots are exact"),
        TierPlacementProjectionError::Invalid(_)
    ));
}
