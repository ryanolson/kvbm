// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CT-0 gate 9: one composed proof of exact lineage recovery.

use super::*;

const INITIAL_MANIFEST: u64 = 41;
const RECOVERY_MANIFEST: u64 = 42;

fn interval(manifest_id: u64, start: u32, len: u32) -> TierPlacementOp {
    TierPlacementOp::Ready {
        scope: scope(RESOURCE),
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

fn exact_snapshot(
    harness: &Harness,
    generation: u64,
    seq_floor: u64,
    manifest_id: u64,
    lineage: &[SequenceHash],
) -> TierPlacementSnapshotV1 {
    let mut snapshot = harness.snapshot(
        generation,
        seq_floor,
        vec![entry(G2, 1, lineage[..2].to_vec())],
    );
    snapshot.manifests = vec![TierPlacementManifest {
        manifest_id,
        resource: RESOURCE,
        hashes: lineage.to_vec(),
    }];
    snapshot
}

#[test]
fn gap_snapshot_and_resync_preserve_exact_recoverable_lineage() {
    let harness = Harness::new();
    let lineage = chain(7);

    let mut disconnected = exact_snapshot(&harness, 1, 0, INITIAL_MANIFEST, &lineage);
    disconnected.manifests[0].hashes.remove(3);
    assert!(matches!(
        harness
            .projection
            .install_snapshot(&disconnected, harness.epoch),
        Err(TierPlacementProjectionError::Invalid(_))
    ));
    assert!(!harness.projection.is_valid(harness.cache, harness.instance));

    harness.install(&exact_snapshot(&harness, 1, 0, INITIAL_MANIFEST, &lineage));
    assert_eq!(
        harness.projection.apply_delta(&harness.batch(
            1,
            1,
            vec![interval(INITIAL_MANIFEST, 2, 2)]
        )),
        DeltaOutcome::Applied { seq: 1, ready: 4 }
    );
    assert!(lineage[..4].iter().all(|hash| harness.holds(*hash)));

    assert_eq!(
        harness
            .projection
            .apply_delta(&harness.batch(3, 1, vec![ready(G2, 1, vec![lineage[4]])])),
        DeltaOutcome::Invalidated(InvalidationReason::SequenceGap)
    );
    assert_eq!(harness.requester.count(), 1);
    assert!(lineage.iter().all(|hash| !harness.holds(*hash)));

    harness.install(&exact_snapshot(&harness, 2, 3, RECOVERY_MANIFEST, &lineage));
    assert!(lineage[..2].iter().all(|hash| harness.holds(*hash)));
    assert!(lineage[2..].iter().all(|hash| !harness.holds(*hash)));

    assert_eq!(
        harness.projection.apply_delta(&harness.batch(
            4,
            2,
            vec![interval(INITIAL_MANIFEST, 2, 2)]
        )),
        DeltaOutcome::Invalidated(InvalidationReason::UnresolvedInterval),
        "replace-all recovery must not retain the old lineage manifest"
    );
    assert_eq!(harness.requester.count(), 2);
    assert!(lineage.iter().all(|hash| !harness.holds(*hash)));

    harness.install(&exact_snapshot(&harness, 3, 4, RECOVERY_MANIFEST, &lineage));
    assert_eq!(
        harness.projection.apply_delta(&harness.batch(
            5,
            3,
            vec![interval(RECOVERY_MANIFEST, 2, 3)]
        )),
        DeltaOutcome::Applied { seq: 5, ready: 5 }
    );
    assert!(lineage[..5].iter().all(|hash| harness.holds(*hash)));
    assert!(lineage[5..].iter().all(|hash| !harness.holds(*hash)));
}
