// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! R7b §7 recovery matrix for the tier-placement consumer.
//!
//! Every assertion here is ultimately about one property: *a projection that
//! cannot prove continuity answers empty, and never answers stale*. The reads
//! all go through the public [`TierPlacementProjection::holders`] a CT-2a caller
//! would use, rather than a test accessor into the map — an assertion that
//! reached inside the state would keep passing if the read path forgot its
//! `valid` filter, which is the one bug that matters.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{CacheManifestId, RegistrationEpoch};
use kvbm_protocols::tier_protocol::{
    InstanceId, KeyRange, PhysicalPlacementMode, PlacementScope, TIER_MEDIUM_CAP_DIRECT_SERVABLE,
    TIER_PLACEMENT_SCHEMA_VERSION, TierDepth, TierMedium, TierPlacementBatchV1, TierPlacementEntry,
    TierPlacementManifest, TierPlacementOp, TierPlacementRejection, TierPlacementSnapshotV1,
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
            media: vec![TierMedium {
                depth: G2,
                medium: "pinned-host".to_string(),
                capabilities: TIER_MEDIUM_CAP_DIRECT_SERVABLE,
            }],
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

    // The other direction: a Ready at an older generation cannot resurrect a
    // copy over a newer record.
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
            media: Vec::new(),
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
                media: Vec::new(),
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
