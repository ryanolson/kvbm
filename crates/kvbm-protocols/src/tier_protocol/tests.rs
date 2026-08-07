// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use kvbm_common::{LogicalResourceId, SequenceHash};

use crate::cache_manifest::{BundleResourceLineage, CacheManifestId, RegistrationEpoch};

use super::{
    InstanceId, KeyRange, PhysicalPlacementMode, PlacementScope, SealOutcome,
    TIER_MEDIUM_CAP_DIRECT_SERVABLE, TIER_PLACEMENT_MAX_MANIFESTS, TIER_PLACEMENT_MAX_MEDIA,
    TIER_PLACEMENT_MAX_OPS_PER_BATCH, TIER_PLACEMENT_MAX_SNAPSHOT_ENTRIES,
    TIER_PLACEMENT_SCHEMA_VERSION, TIER_PLACEMENT_SUBJECT, TierDepth, TierMedium,
    TierPlacementBatchV1, TierPlacementEntry, TierPlacementError, TierPlacementManifest,
    TierPlacementOp, TierPlacementRejection, TierPlacementSequencer, TierPlacementSnapshotV1,
};

const RESOURCE: LogicalResourceId = LogicalResourceId(3);

/// Depth 1, conventionally G2. Spelled locally rather than as a `TierDepth`
/// associated constant: the wire type deliberately names no depth but G1.
const G2: TierDepth = TierDepth(1);

fn cache() -> CacheManifestId {
    CacheManifestId::from_bytes([9; 32])
}

fn instance() -> InstanceId {
    InstanceId::new_v4()
}

/// A genuine PLH chain of `len` blocks, built from parts so the fixtures do not
/// depend on the tokenizer or on a particular block-hash function.
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

fn scope() -> PlacementScope {
    PlacementScope::unitary(RESOURCE)
}

fn ready(tier: TierDepth, generation: u64, keys: KeyRange) -> TierPlacementOp {
    TierPlacementOp::Ready {
        scope: scope(),
        tier,
        placement: PhysicalPlacementMode::Whole,
        generation,
        keys,
    }
}

fn remove(tier: TierDepth, generation: u64, keys: KeyRange) -> TierPlacementOp {
    TierPlacementOp::Remove {
        scope: scope(),
        tier,
        generation,
        keys,
    }
}

fn entry(tier: TierDepth, keys: KeyRange) -> TierPlacementEntry {
    TierPlacementEntry {
        scope: scope(),
        tier,
        placement: PhysicalPlacementMode::TpShards { count: 4 },
        generation: 7,
        keys,
    }
}

fn batch(ops: Vec<TierPlacementOp>) -> TierPlacementBatchV1 {
    TierPlacementBatchV1 {
        v: TIER_PLACEMENT_SCHEMA_VERSION,
        cache: cache(),
        instance_id: instance(),
        registration_epoch: RegistrationEpoch::new(),
        seq: 11,
        snapshot_generation: 2,
        ops,
    }
}

fn snapshot(entries: Vec<TierPlacementEntry>) -> TierPlacementSnapshotV1 {
    TierPlacementSnapshotV1 {
        v: TIER_PLACEMENT_SCHEMA_VERSION,
        cache: cache(),
        instance_id: instance(),
        registration_epoch: RegistrationEpoch::new(),
        snapshot_generation: 3,
        seq_floor: 10,
        media: g2_medium(),
        manifests: Vec::new(),
        entries,
    }
}

/// A single G2 medium, the shape nearly every snapshot fixture needs now that
/// [`TierPlacementSnapshotV1::validate`] requires header coverage for every
/// depth the entries name.
fn g2_medium() -> Vec<TierMedium> {
    vec![TierMedium {
        depth: G2,
        medium: "pinned-host".to_string(),
        capabilities: TIER_MEDIUM_CAP_DIRECT_SERVABLE,
    }]
}

/// Unwrap a seal that must have produced a batch.
fn sealed(outcome: SealOutcome) -> TierPlacementBatchV1 {
    match outcome {
        SealOutcome::Batch(batch) => batch,
        SealOutcome::Deferred(_) => panic!("the emission gate is not armed here"),
    }
}

fn manifest(manifest_id: u64, len: usize) -> TierPlacementManifest {
    TierPlacementManifest {
        manifest_id,
        resource: RESOURCE,
        hashes: chain(len),
    }
}

// ---------------------------------------------------------------------------
// §7.1 — round-trip serde for every message, in both codecs
// ---------------------------------------------------------------------------

#[test]
fn batch_round_trips_json_and_msgpack() {
    let original = batch(vec![
        ready(G2, 4, KeyRange::Hashes(chain(3))),
        remove(
            TierDepth(2),
            5,
            KeyRange::ManifestInterval {
                manifest_id: 77,
                start: 2,
                len: 8,
            },
        ),
    ]);
    original.validate().expect("fixture is valid");

    let json = serde_json::to_vec(&original).expect("json encode");
    assert_eq!(
        TierPlacementBatchV1::decode_json(&json).expect("json decode"),
        original
    );

    // The delta plane is msgpack over ZMQ; the control plane is JSON. Both must
    // survive, so the spec's "serde JSON" phrasing is not read as codec-only.
    let packed = rmp_serde::to_vec(&original).expect("msgpack encode");
    let decoded =
        TierPlacementBatchV1::from_decoded(rmp_serde::from_slice::<TierPlacementBatchV1>(&packed))
            .expect("msgpack decode");
    assert_eq!(decoded, original);
}

#[test]
fn snapshot_round_trips_json_and_msgpack() {
    let original = snapshot(vec![entry(G2, KeyRange::Hashes(chain(4)))]);
    original.validate().expect("fixture is valid");

    let json = serde_json::to_vec(&original).expect("json encode");
    assert_eq!(
        TierPlacementSnapshotV1::decode_json(&json).expect("json decode"),
        original
    );

    let packed = rmp_serde::to_vec(&original).expect("msgpack encode");
    let decoded = TierPlacementSnapshotV1::from_decoded(rmp_serde::from_slice::<
        TierPlacementSnapshotV1,
    >(&packed))
    .expect("msgpack decode");
    assert_eq!(decoded, original);
}

#[test]
fn every_leaf_type_round_trips() {
    for mode in [
        PhysicalPlacementMode::Whole,
        PhysicalPlacementMode::TpShards { count: 8 },
        PhysicalPlacementMode::DataStripes { count: 3 },
    ] {
        let json = serde_json::to_string(&mode).expect("encode");
        assert_eq!(
            serde_json::from_str::<PhysicalPlacementMode>(&json).expect("decode"),
            mode
        );
    }

    let depth = TierDepth(4);
    assert_eq!(
        serde_json::from_str::<TierDepth>(&serde_json::to_string(&depth).expect("encode"))
            .expect("decode"),
        depth
    );
    // Numeric on the wire, not a tagged enum: adding a tier is not a version bump.
    assert_eq!(serde_json::to_string(&depth).expect("encode"), "4");

    let scope = PlacementScope {
        resource: RESOURCE,
        lane: 2,
    };
    assert_eq!(
        serde_json::from_str::<PlacementScope>(&serde_json::to_string(&scope).expect("encode"))
            .expect("decode"),
        scope
    );

    let medium = TierMedium {
        depth: TierDepth(2),
        medium: "nvme".to_string(),
        capabilities: 0b1011,
    };
    assert_eq!(
        serde_json::from_str::<TierMedium>(&serde_json::to_string(&medium).expect("encode"))
            .expect("decode"),
        medium
    );
}

#[test]
fn subject_and_version_are_frozen_constants() {
    // Both ends of the seam share these; a rename is a two-repo change (R10).
    assert_eq!(TIER_PLACEMENT_SUBJECT, "kvbm.tier_placements");
    assert_eq!(TIER_PLACEMENT_SCHEMA_VERSION, 1);
}

// ---------------------------------------------------------------------------
// §7.1 — unknown version rejected whole, and counted apart from corruption
// ---------------------------------------------------------------------------

#[test]
fn future_version_is_rejected_whole_after_decoding_cleanly() {
    // Built by stamping v=2 onto a v1-shaped body on purpose: a genuinely
    // v2-shaped payload would fail to *decode* (msgpack encodes structs
    // positionally), which would prove the wrong thing. The point of splitting
    // `validate()` out of `Deserialize` is that "too new" and "corrupt" stay
    // separately countable.
    let mut future = batch(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))]);
    future.v = TIER_PLACEMENT_SCHEMA_VERSION + 1;

    let packed = rmp_serde::to_vec(&future).expect("encode");
    let decoded = rmp_serde::from_slice::<TierPlacementBatchV1>(&packed);
    assert!(decoded.is_ok(), "the frame itself must decode");

    let error = TierPlacementBatchV1::from_decoded(decoded).expect_err("version must be rejected");
    assert_eq!(
        error,
        TierPlacementError::UnsupportedVersion {
            found: 2,
            supported: TIER_PLACEMENT_SCHEMA_VERSION,
        }
    );
    assert_eq!(
        error.rejection(),
        TierPlacementRejection::UnsupportedVersion
    );

    // Corruption is a different counter.
    let corrupt =
        TierPlacementBatchV1::from_decoded(rmp_serde::from_slice::<TierPlacementBatchV1>(&[
            0xC1, 0x00, 0x7F,
        ]))
        .expect_err("corrupt bytes must be rejected");
    assert_eq!(corrupt.rejection(), TierPlacementRejection::Undecodable);
}

#[test]
fn future_version_snapshot_is_rejected_whole() {
    let mut future = snapshot(vec![entry(G2, KeyRange::Hashes(chain(2)))]);
    future.v = 9;
    assert_eq!(
        future.validate().expect_err("version must be rejected"),
        TierPlacementError::UnsupportedVersion {
            found: 9,
            supported: TIER_PLACEMENT_SCHEMA_VERSION,
        }
    );
}

#[test]
fn one_bad_op_rejects_the_whole_batch() {
    // Index 0 is fine; index 1 is not. Nothing is applied, and the error names
    // the offending index rather than silently dropping it.
    let mixed = batch(vec![
        ready(G2, 1, KeyRange::Hashes(chain(2))),
        ready(TierDepth::G1, 2, KeyRange::Hashes(chain(2))),
    ]);
    assert_eq!(
        mixed.validate().expect_err("batch must be rejected"),
        TierPlacementError::G1NotPublishable { index: 1 }
    );

    let json = serde_json::to_vec(&mixed).expect("encode");
    assert!(TierPlacementBatchV1::decode_json(&json).is_err());
}

// ---------------------------------------------------------------------------
// §7.6 — Reserved/Copying (and PositionRun) are structurally unspeakable
// ---------------------------------------------------------------------------

#[test]
fn local_only_states_have_no_wire_representation() {
    // `Reserved` and `Copying` must never reduce a remote copy's alternative
    // cost, so they are not variants at all — an encoder cannot express one and
    // a decoder rejects it as an unknown variant.
    for payload in [
        r#"{"Copying":{"scope":{"resource":3,"lane":0},"tier":1,"placement":"Whole","generation":1,"keys":{"Hashes":[]}}}"#,
        r#"{"Reserved":{"scope":{"resource":3,"lane":0},"tier":1,"placement":"Whole","generation":1,"keys":{"Hashes":[]}}}"#,
    ] {
        assert!(
            serde_json::from_str::<TierPlacementOp>(payload).is_err(),
            "local-only state decoded as a wire op: {payload}"
        );
    }
}

#[test]
fn position_run_is_not_a_key_range() {
    // Regression guard for the 2026-08-05 correction (finding 4): a terminal
    // hash plus a length is not recoverable membership, because a PLH carries
    // one parent *fragment*, not the ancestor chain. `PositionRun` was removed
    // and must not come back by way of a tolerant decoder.
    let payload = r#"{"PositionRun":{"end_hash":1,"len":4}}"#;
    assert!(serde_json::from_str::<KeyRange>(payload).is_err());
}

// ---------------------------------------------------------------------------
// §1 rule 3 / §7.5 — G1 is not publishable; snapshots carry exact membership
// ---------------------------------------------------------------------------

#[test]
fn g1_is_rejected_everywhere() {
    assert!(!TierDepth::G1.is_publishable());
    assert!(G2.is_publishable());
    assert_eq!(TierDepth::G1.depth(), 0);

    assert_eq!(
        batch(vec![ready(TierDepth::G1, 1, KeyRange::Hashes(chain(2)))])
            .validate()
            .expect_err("ready at G1"),
        TierPlacementError::G1NotPublishable { index: 0 }
    );
    assert_eq!(
        batch(vec![remove(TierDepth::G1, 1, KeyRange::Hashes(chain(2)))])
            .validate()
            .expect_err("remove at G1"),
        TierPlacementError::G1NotPublishable { index: 0 }
    );
    assert_eq!(
        snapshot(vec![entry(TierDepth::G1, KeyRange::Hashes(chain(2)))])
            .validate()
            .expect_err("snapshot entry at G1"),
        TierPlacementError::G1NotPublishable { index: 0 }
    );

    let mut g1_medium = snapshot(vec![]);
    g1_medium.media = vec![TierMedium {
        depth: TierDepth::G1,
        medium: "device".to_string(),
        capabilities: 0,
    }];
    assert_eq!(
        g1_medium.validate().expect_err("snapshot medium at G1"),
        TierPlacementError::MediumNotPublishable { index: 0 }
    );
}

#[test]
fn snapshots_reject_manifest_intervals() {
    // A consumer installing a recovery snapshot has already discarded the
    // projection it would need to resolve an interval, so exactness is the only
    // safe encoding there.
    let inexact = snapshot(vec![entry(
        G2,
        KeyRange::ManifestInterval {
            manifest_id: 1,
            start: 0,
            len: 4,
        },
    )]);
    assert_eq!(
        inexact.validate().expect_err("snapshot must be exact"),
        TierPlacementError::InexactSnapshotKeys { index: 0 }
    );

    // Deltas may use either form.
    assert!(
        batch(vec![ready(
            G2,
            1,
            KeyRange::ManifestInterval {
                manifest_id: 1,
                start: 0,
                len: 4,
            },
        )])
        .validate()
        .is_ok()
    );
}

#[test]
fn empty_and_overflowing_key_ranges_are_rejected() {
    assert_eq!(
        batch(vec![ready(G2, 1, KeyRange::Hashes(vec![]))])
            .validate()
            .expect_err("empty hashes"),
        TierPlacementError::EmptyKeys { index: 0 }
    );
    assert_eq!(
        batch(vec![ready(
            G2,
            1,
            KeyRange::ManifestInterval {
                manifest_id: 4,
                start: 0,
                len: 0,
            },
        )])
        .validate()
        .expect_err("zero-length interval"),
        TierPlacementError::EmptyKeys { index: 0 }
    );
    assert_eq!(
        batch(vec![ready(
            G2,
            1,
            KeyRange::ManifestInterval {
                manifest_id: 4,
                start: u32::MAX,
                len: 2,
            },
        )])
        .validate()
        .expect_err("overflowing interval"),
        TierPlacementError::IntervalOverflow {
            index: 0,
            manifest_id: 4,
            start: u32::MAX,
            len: 2,
        }
    );
}

#[test]
fn duplicate_medium_depth_is_rejected() {
    let mut duplicated = snapshot(vec![]);
    duplicated.media = vec![
        TierMedium {
            depth: G2,
            medium: "pinned-host".to_string(),
            capabilities: 0,
        },
        TierMedium {
            depth: G2,
            medium: "also-host".to_string(),
            capabilities: 0,
        },
    ];
    assert_eq!(
        duplicated.validate().expect_err("duplicate depth"),
        TierPlacementError::DuplicateMediumDepth { depth: G2 }
    );
}

// ---------------------------------------------------------------------------
// ManifestInterval resolves against the exact bundle-lineage constructor
// ---------------------------------------------------------------------------

#[test]
fn snapshot_hashes_install_as_an_exact_lineage_manifest() {
    // This is the consumer-side install path an interval later resolves
    // against: the snapshot's exact run feeds `BundleResourceLineage::new`,
    // which validates adjacent position and parent-fragment continuity.
    let hashes = chain(5);
    let installed =
        BundleResourceLineage::new(RESOURCE, hashes.clone()).expect("snapshot run is a real chain");
    assert_eq!(installed.hashes(), hashes.as_slice());

    // ...and a run that is not a real chain cannot become a manifest, so an
    // interval can never resolve against fabricated membership.
    let mut broken = chain(5);
    broken.remove(2);
    assert!(BundleResourceLineage::new(RESOURCE, broken).is_err());
}

#[test]
fn snapshot_manifests_are_the_only_way_an_interval_becomes_resolvable() {
    // Without an installed manifest, an interval delta is unresolvable by
    // construction — nothing else on the wire binds an id to a chain. The
    // snapshot header is that binding, and it validates through the same
    // lineage constructor a bundle advertisement does.
    let mut snap = snapshot(vec![entry(G2, KeyRange::Hashes(chain(3)))]);
    snap.manifests = vec![manifest(42, 6)];
    snap.validate().expect("manifest is a real chain");

    let installed = snap.manifests[0].lineage().expect("lineage builds");
    assert_eq!(installed.resource(), RESOURCE);
    assert_eq!(installed.hashes(), chain(6).as_slice());
}

#[test]
fn malformed_manifests_reject_the_whole_snapshot() {
    // Rejection happens in `validate()`, before any install: a snapshot install
    // is replace-all, so a manifest that failed halfway would leave a consumer
    // holding state assembled from a body it had already decided to reject.
    let mut broken = snapshot(vec![]);
    let mut hashes = chain(5);
    hashes.remove(2);
    broken.manifests = vec![TierPlacementManifest {
        manifest_id: 7,
        resource: RESOURCE,
        hashes,
    }];
    assert!(matches!(
        broken.validate().expect_err("non-chain manifest"),
        TierPlacementError::InvalidManifest { manifest_id: 7, .. }
    ));
    assert_eq!(
        broken
            .validate()
            .expect_err("non-chain manifest")
            .rejection(),
        TierPlacementRejection::Invalid
    );

    let mut duplicated = snapshot(vec![]);
    duplicated.manifests = vec![manifest(3, 2), manifest(3, 4)];
    assert_eq!(
        duplicated.validate().expect_err("duplicate id"),
        TierPlacementError::DuplicateManifestId { manifest_id: 3 }
    );

    let mut too_many = snapshot(vec![]);
    too_many.manifests = (0..=TIER_PLACEMENT_MAX_MANIFESTS as u64)
        .map(|id| manifest(id, 1))
        .collect();
    assert!(matches!(
        too_many.validate().expect_err("manifest count"),
        TierPlacementError::TooLarge {
            what: "snapshot manifests",
            ..
        }
    ));
}

#[test]
fn snapshot_manifests_default_to_empty_for_publishers_that_never_use_intervals() {
    // The snapshot rides the JSON control plane (map-encoded), so unlike the
    // frozen delta envelope it may carry a `#[serde(default)]` field. A body
    // written before the field existed still decodes.
    let legacy = serde_json::json!({
        "v": TIER_PLACEMENT_SCHEMA_VERSION,
        "cache": CacheManifestId::from_bytes([9; 32]),
        "instance_id": InstanceId::new_v4(),
        "registration_epoch": RegistrationEpoch::new(),
        "snapshot_generation": 3,
        "seq_floor": 10,
        "media": [],
        "entries": [],
    });
    let decoded = TierPlacementSnapshotV1::decode_json(&serde_json::to_vec(&legacy).unwrap())
        .expect("field defaults");
    assert!(decoded.manifests.is_empty());
}

// ---------------------------------------------------------------------------
// Publisher-side sequencing
// ---------------------------------------------------------------------------

#[test]
fn sequencer_numbers_batches_monotonically_from_one() {
    let mut sequencer = TierPlacementSequencer::new(cache(), instance(), RegistrationEpoch::new());
    assert_eq!(sequencer.last_seq(), 0);
    assert_eq!(sequencer.snapshot_generation(), 0);

    for expected in 1..=4u64 {
        let batch = sealed(
            sequencer
                .seal(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))])
                .expect("valid ops"),
        );
        assert_eq!(batch.seq, expected);
        assert_eq!(batch.v, TIER_PLACEMENT_SCHEMA_VERSION);
        assert_eq!(batch.cache, sequencer.cache());
        assert_eq!(batch.instance_id, sequencer.instance_id());
        assert_eq!(batch.registration_epoch, sequencer.registration_epoch());
        assert_eq!(batch.snapshot_generation, 0);
        assert_eq!(sequencer.last_seq(), expected);
    }
}

#[test]
fn sequencer_refuses_g1_without_burning_a_sequence_number() {
    let mut sequencer = TierPlacementSequencer::new(cache(), instance(), RegistrationEpoch::new());
    sequencer
        .seal(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))])
        .expect("valid ops");

    assert_eq!(
        sequencer
            .seal(vec![ready(TierDepth::G1, 1, KeyRange::Hashes(chain(2)))])
            .expect_err("publisher-side G1 assertion"),
        TierPlacementError::G1NotPublishable { index: 0 }
    );

    // A rejected batch never reached the wire, so consuming its number would
    // manufacture a gap and trigger fleet-wide snapshot traffic for a purely
    // local bug.
    let next = sealed(
        sequencer
            .seal(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))])
            .expect("valid ops"),
    );
    assert_eq!(next.seq, 2);
}

#[test]
fn snapshot_bumps_generation_and_hands_the_consumer_a_resume_point() {
    let mut sequencer = TierPlacementSequencer::new(cache(), instance(), RegistrationEpoch::new());
    for _ in 0..3 {
        sequencer
            .seal(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))])
            .expect("valid ops");
    }

    let snap = sequencer
        .snapshot(
            g2_medium(),
            vec![manifest(1, 3)],
            vec![entry(G2, KeyRange::Hashes(chain(3)))],
        )
        .expect("valid snapshot");

    assert_eq!(snap.snapshot_generation, 1);
    assert_eq!(snap.seq_floor, 3);
    assert_eq!(snap.manifests, vec![manifest(1, 3)]);
    assert_eq!(sequencer.snapshot_generation(), 1);

    assert!(sequencer.note_snapshot_installed(snap.snapshot_generation));
    // The consumer resumes at seq_floor + 1; that is exactly the next sealed seq.
    let resumed = sealed(
        sequencer
            .seal(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))])
            .expect("valid ops"),
    );
    assert_eq!(resumed.seq, snap.seq_floor + 1);
    assert_eq!(resumed.snapshot_generation, snap.snapshot_generation);
}

#[test]
fn rejected_snapshot_does_not_bump_the_generation_or_arm_the_gate() {
    let mut sequencer = TierPlacementSequencer::new(cache(), instance(), RegistrationEpoch::new());
    assert!(
        sequencer
            .snapshot(
                g2_medium(),
                vec![],
                vec![entry(
                    G2,
                    KeyRange::ManifestInterval {
                        manifest_id: 1,
                        start: 0,
                        len: 2,
                    },
                )],
            )
            .is_err()
    );
    assert_eq!(sequencer.snapshot_generation(), 0);
    // A snapshot that never reached the wire must not hold deltas hostage: the
    // consumer will never acknowledge a generation it was never offered, so an
    // armed gate here would stop emission permanently.
    assert!(!sequencer.awaiting_snapshot_install());
    assert_eq!(
        sealed(
            sequencer
                .seal(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))])
                .expect("valid ops")
        )
        .seq,
        1
    );
}

/// R7b §5 header contract: `media` describes every depth the entries name, and
/// it is the only place capability bits travel. Without coverage a consumer must
/// guess whether a depth is direct-servable or staging-required, which is the
/// cost input CT-2a reads.
#[test]
fn a_snapshot_entry_at_an_undescribed_depth_is_rejected() {
    let mut bare = snapshot(vec![entry(TierDepth(2), KeyRange::Hashes(chain(2)))]);
    assert_eq!(
        bare.validate().expect_err("depth 2 is not in the header"),
        TierPlacementError::UndescribedDepth {
            index: 0,
            depth: TierDepth(2),
        }
    );
    assert_eq!(
        bare.validate().expect_err("bucket").rejection(),
        TierPlacementRejection::Invalid
    );

    // An empty header is fine as long as the body references no depth.
    let mut empty = snapshot(vec![]);
    empty.media = Vec::new();
    empty.validate().expect("no entries, nothing to describe");

    bare.media.push(TierMedium {
        depth: TierDepth(2),
        medium: "nvme".to_string(),
        capabilities: 0,
    });
    bare.validate().expect("described now");
}

/// The race the gate exists for: an ungated publisher's first post-snapshot
/// delta beats its own snapshot to the consumer, the consumer drops it as
/// `GenerationAhead`, and pub/sub never retransmits it — so the *next* delta
/// gaps against the resume point forever after.
///
/// Asserted on the numbers rather than on a consumer, because those are what a
/// consumer keys on: while the gate is armed nothing is sealed and no sequence
/// number is consumed, and the batch released afterwards is exactly
/// `seq_floor + 1` at the new generation.
#[test]
fn a_sealed_snapshot_holds_deltas_until_the_install_is_acknowledged() {
    let mut sequencer = TierPlacementSequencer::new(cache(), instance(), RegistrationEpoch::new());
    for _ in 0..2 {
        sealed(
            sequencer
                .seal(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))])
                .expect("valid ops"),
        );
    }

    let snap = sequencer
        .snapshot(
            g2_medium(),
            vec![],
            vec![entry(G2, KeyRange::Hashes(chain(3)))],
        )
        .expect("valid snapshot");
    assert!(sequencer.awaiting_snapshot_install());

    let ops = vec![ready(G2, 2, KeyRange::Hashes(chain(2)))];
    let deferred = sequencer
        .seal(ops.clone())
        .expect("deferral is not an error");
    assert_eq!(deferred, SealOutcome::Deferred(ops.clone()));
    // Deferral consumes nothing, so a re-seal after the ack cannot leave a hole.
    assert_eq!(sequencer.last_seq(), snap.seq_floor);

    // A mismatched acknowledgement does not release the gate: a lost or
    // superseded push must not let deltas outrun the state they describe.
    assert!(!sequencer.note_snapshot_installed(snap.snapshot_generation + 1));
    assert!(sequencer.awaiting_snapshot_install());
    assert!(matches!(
        sequencer.seal(ops.clone()).expect("still deferred"),
        SealOutcome::Deferred(_)
    ));

    assert!(sequencer.note_snapshot_installed(snap.snapshot_generation));
    assert!(!sequencer.awaiting_snapshot_install());
    let released = sealed(sequencer.seal(ops).expect("valid ops"));
    assert_eq!(released.seq, snap.seq_floor + 1);
    assert_eq!(released.snapshot_generation, snap.snapshot_generation);
}

/// A second snapshot sealed while the first is unacknowledged supersedes it, and
/// only the newer generation releases the gate. The older ack is stale by then
/// and must not.
#[test]
fn a_superseding_snapshot_re_arms_the_gate_at_the_higher_generation() {
    let mut sequencer = TierPlacementSequencer::new(cache(), instance(), RegistrationEpoch::new());
    let first = sequencer
        .snapshot(g2_medium(), vec![], vec![])
        .expect("valid snapshot");
    let second = sequencer
        .snapshot(g2_medium(), vec![], vec![])
        .expect("valid snapshot");
    assert_eq!(second.snapshot_generation, first.snapshot_generation + 1);

    assert!(!sequencer.note_snapshot_installed(first.snapshot_generation));
    assert!(sequencer.awaiting_snapshot_install());
    assert!(sequencer.note_snapshot_installed(second.snapshot_generation));

    let released = sealed(
        sequencer
            .seal(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))])
            .expect("valid ops"),
    );
    assert_eq!(released.snapshot_generation, second.snapshot_generation);
    assert_eq!(released.seq, second.seq_floor + 1);
}

// ---------------------------------------------------------------------------
// Publisher-forced divergence (the dropped-`Remove` resync)
// ---------------------------------------------------------------------------

/// The mechanism, asserted on the numbers a consumer keys on: a divergent
/// publisher's next batch skips exactly one sequence number, which is a gap by
/// the consumer's `last_seq + 1` rule. Emission is deliberately *not* held —
/// holding it would leave the consumer valid-and-stale until an HTTP push
/// lands, whereas emitting makes it invalid-and-empty on the next batch.
#[test]
fn marking_divergent_burns_one_sequence_number_and_keeps_emitting() {
    let mut sequencer = TierPlacementSequencer::new(cache(), instance(), RegistrationEpoch::new());
    let first = sealed(
        sequencer
            .seal(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))])
            .expect("valid ops"),
    );
    assert_eq!(first.seq, 1);
    assert!(!sequencer.needs_snapshot());

    assert!(sequencer.mark_divergent(), "arms on the first call");
    assert!(sequencer.needs_snapshot());
    // Idempotent while armed: one gap invalidates exactly as thoroughly as ten,
    // and a burst of dropped Removes must not burn a number per drop.
    for _ in 0..5 {
        assert!(!sequencer.mark_divergent());
    }

    let after = sealed(
        sequencer
            .seal(vec![remove(G2, 1, KeyRange::Hashes(chain(2)))])
            .expect("valid ops"),
    );
    assert_eq!(
        after.seq,
        first.seq + 2,
        "exactly one number burned, however many drops were reported"
    );
    assert_eq!(
        after.snapshot_generation, 0,
        "divergence is a sequence-plane signal; it must not touch the generation"
    );
}

/// The non-obvious interaction: a burned number must not survive as a hole
/// across the snapshot that repairs it. `seq_floor` is derived from `last_seq`,
/// so the consumer resumes at exactly the number the next `seal` stamps.
#[test]
fn a_snapshot_after_divergence_leaves_no_hole_and_disarms_the_request() {
    let mut sequencer = TierPlacementSequencer::new(cache(), instance(), RegistrationEpoch::new());
    sealed(
        sequencer
            .seal(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))])
            .expect("valid ops"),
    );
    sequencer.mark_divergent();

    let snap = sequencer
        .snapshot(
            g2_medium(),
            vec![],
            vec![entry(G2, KeyRange::Hashes(chain(2)))],
        )
        .expect("valid snapshot");
    assert!(
        !sequencer.needs_snapshot(),
        "a sealed snapshot is the resync; nothing is still owed"
    );
    assert_eq!(snap.seq_floor, 2, "the burned number is inside the floor");

    assert!(sequencer.note_snapshot_installed(snap.snapshot_generation));
    let resumed = sealed(
        sequencer
            .seal(vec![ready(G2, 1, KeyRange::Hashes(chain(2)))])
            .expect("valid ops"),
    );
    assert_eq!(
        resumed.seq,
        snap.seq_floor + 1,
        "the consumer's resume point and the publisher's next number agree"
    );
}

/// Divergence recorded *after* a snapshot is sealed describes state that
/// snapshot does not carry, so it must arm a fresh one rather than ride along.
#[test]
fn divergence_after_a_seal_survives_the_install_acknowledgement() {
    let mut sequencer = TierPlacementSequencer::new(cache(), instance(), RegistrationEpoch::new());
    let snap = sequencer
        .snapshot(g2_medium(), vec![], vec![])
        .expect("valid snapshot");
    assert!(sequencer.mark_divergent());
    assert!(sequencer.note_snapshot_installed(snap.snapshot_generation));
    assert!(
        sequencer.needs_snapshot(),
        "the installed snapshot predates the loss it cannot describe"
    );
}

/// The un-wedge. A publisher whose push can never succeed must not hold the
/// gate forever: silence leaves the consumer serving the previous generation as
/// *valid*, which is the one failure it cannot detect. Abandoning resumes
/// emission at the bumped generation, which the consumer reads as
/// `GenerationAhead` and answers empty for.
#[test]
fn abandoning_an_unpushable_snapshot_releases_emission_and_stays_divergent() {
    let mut sequencer = TierPlacementSequencer::new(cache(), instance(), RegistrationEpoch::new());
    let ops = vec![ready(G2, 1, KeyRange::Hashes(chain(2)))];
    let snap = sequencer
        .snapshot(g2_medium(), vec![], vec![])
        .expect("valid snapshot");
    assert!(matches!(
        sequencer
            .seal(ops.clone())
            .expect("deferral is not an error"),
        SealOutcome::Deferred(_)
    ));

    assert!(sequencer.abandon_pending_snapshot());
    assert!(!sequencer.awaiting_snapshot_install());
    assert!(
        sequencer.needs_snapshot(),
        "abandoning does not make the publisher continuous again"
    );
    assert!(
        !sequencer.abandon_pending_snapshot(),
        "nothing left to abandon"
    );

    let resumed = sealed(sequencer.seal(ops).expect("valid ops"));
    assert_eq!(
        resumed.snapshot_generation, snap.snapshot_generation,
        "emission resumes at the generation no consumer installed — visible, not silent"
    );
    // The generation bump is what invalidates a consumer still on the old
    // generation; it rejects on generation before it reads the sequence. The
    // burned number covers the case the failed push cannot distinguish — the
    // install landed and only the response was lost — where the consumer is at
    // the new generation and the gap is the only thing left to tell it to
    // resync.
    assert_eq!(
        resumed.seq,
        snap.seq_floor + 2,
        "the abandon burned one number on top of the deferred (unconsumed) seal"
    );
}

// ---------------------------------------------------------------------------
// R6 — anti-amplification bounds
// ---------------------------------------------------------------------------

#[test]
fn oversized_messages_are_rejected() {
    // Guards, not capacity policy: a bound a real publisher could hit would
    // turn the spec's "temporary miss" into a permanent one, so the limits sit
    // orders of magnitude above the default 1024-item batching cadence.
    let op = ready(G2, 1, KeyRange::Hashes(chain(1)));
    let too_many = batch(vec![op; TIER_PLACEMENT_MAX_OPS_PER_BATCH + 1]);
    assert_eq!(
        too_many.validate().expect_err("op count"),
        TierPlacementError::TooLarge {
            what: "batch ops",
            count: TIER_PLACEMENT_MAX_OPS_PER_BATCH + 1,
            limit: TIER_PLACEMENT_MAX_OPS_PER_BATCH,
        }
    );
    assert_eq!(
        too_many.validate().expect_err("op count").rejection(),
        TierPlacementRejection::Invalid
    );

    // Key totals are summed across ops, so many small intervals cannot slip
    // past a per-op check.
    let wide = batch(vec![
        ready(
            G2,
            1,
            KeyRange::ManifestInterval {
                manifest_id: 1,
                start: 0,
                len: u32::MAX / 2,
            },
        );
        3
    ]);
    assert!(matches!(
        wide.validate().expect_err("key count"),
        TierPlacementError::TooLarge {
            what: "batch keys",
            ..
        }
    ));

    let mut wide_snapshot = snapshot(vec![]);
    wide_snapshot.media = (0..=TIER_PLACEMENT_MAX_MEDIA)
        .map(|depth| TierMedium {
            depth: TierDepth((depth % 200 + 1) as u8),
            medium: "m".to_string(),
            capabilities: 0,
        })
        .collect();
    assert_eq!(
        wide_snapshot.validate().expect_err("media count"),
        TierPlacementError::TooLarge {
            what: "snapshot media",
            count: TIER_PLACEMENT_MAX_MEDIA + 1,
            limit: TIER_PLACEMENT_MAX_MEDIA,
        }
    );
}

// The property that matters about the bounds: an ordinary publisher can never
// trip one. A full batching window (the default 1024-item flush) of 256-block
// runs sits ~64x under the op bound and ~4x under the key bound. These are
// compile-time assertions so a future retune of the constants cannot quietly
// bring a guard down into reachable territory.
const _: () = assert!(1024 < TIER_PLACEMENT_MAX_OPS_PER_BATCH);
const _: () = assert!(1024 * 256 <= super::TIER_PLACEMENT_MAX_KEYS_PER_MESSAGE);
const _: () = assert!(TIER_PLACEMENT_MAX_SNAPSHOT_ENTRIES >= 1 << 16);
