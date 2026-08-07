// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publisher-side sequence/generation bookkeeping for the tier-placement
//! stream.
//!
//! This is the whole publisher contract the rhino-side wiring (CT-2) needs:
//! pure, synchronous, transport-free. It owns the two counters a consumer uses
//! to detect loss — `seq` and `snapshot_generation` — so no publisher has to
//! reimplement the monotonicity rules, and it applies the same `validate()` the
//! consumer applies, so a publisher cannot put a message on the wire that its
//! own consumer would reject (R7b §7.6's publisher-side assertion).
//!
//! Not included here, deliberately: the emission points, the batching cadence,
//! the ZMQ socket, and the periodic-snapshot timer. Those are rhino-side.

use crate::cache_manifest::{CacheManifestId, RegistrationEpoch};

use super::{
    InstanceId, TIER_PLACEMENT_SCHEMA_VERSION, TierMedium, TierPlacementBatchV1,
    TierPlacementEntry, TierPlacementError, TierPlacementManifest, TierPlacementOp,
    TierPlacementSnapshotV1,
};

/// Stamps identity, sequence, and generation onto outbound tier-placement
/// messages.
///
/// One sequencer per `(cache, instance, registration_epoch)`. A new
/// registration epoch means a new sequencer: sequence numbers are only
/// monotone *within* an epoch, and a consumer discards all prior state when the
/// epoch changes, so restarting the counters is correct rather than merely
/// tolerated.
#[derive(Debug, Clone)]
pub struct TierPlacementSequencer {
    cache: CacheManifestId,
    instance_id: InstanceId,
    registration_epoch: RegistrationEpoch,
    snapshot_generation: u64,
    next_seq: u64,
}

impl TierPlacementSequencer {
    /// Start a sequencer at generation 0, with the first sealed batch carrying
    /// `seq == 1`.
    ///
    /// `seq` starts at 1 so that a consumer's "expect `last_seq + 1`" rule
    /// works from an initial `last_seq` of 0 with no special case.
    #[must_use]
    pub const fn new(
        cache: CacheManifestId,
        instance_id: InstanceId,
        registration_epoch: RegistrationEpoch,
    ) -> Self {
        Self {
            cache,
            instance_id,
            registration_epoch,
            snapshot_generation: 0,
            next_seq: 1,
        }
    }

    /// Cache identity this sequencer stamps.
    #[must_use]
    pub const fn cache(&self) -> CacheManifestId {
        self.cache
    }

    /// Instance identity this sequencer stamps.
    #[must_use]
    pub const fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    /// Registration epoch this sequencer stamps.
    #[must_use]
    pub const fn registration_epoch(&self) -> RegistrationEpoch {
        self.registration_epoch
    }

    /// Current snapshot generation. Deltas carry this value.
    #[must_use]
    pub const fn snapshot_generation(&self) -> u64 {
        self.snapshot_generation
    }

    /// Sequence number of the most recently sealed batch (0 before the first).
    #[must_use]
    pub const fn last_seq(&self) -> u64 {
        self.next_seq - 1
    }

    /// Seal a delta batch.
    ///
    /// The sequence number is consumed **only on success**: a batch rejected by
    /// `validate()` never reaches the wire, so consuming its number would
    /// manufacture a gap the consumer would then recover from, turning a local
    /// publisher bug into fleet-wide snapshot traffic.
    pub fn seal(
        &mut self,
        ops: Vec<TierPlacementOp>,
    ) -> Result<TierPlacementBatchV1, TierPlacementError> {
        let batch = TierPlacementBatchV1 {
            v: TIER_PLACEMENT_SCHEMA_VERSION,
            cache: self.cache,
            instance_id: self.instance_id,
            registration_epoch: self.registration_epoch,
            seq: self.next_seq,
            snapshot_generation: self.snapshot_generation,
            ops,
        };
        batch.validate()?;
        self.next_seq += 1;
        Ok(batch)
    }

    /// Seal a full snapshot, bumping the generation.
    ///
    /// `seq_floor` is the last sealed delta sequence, so the consumer resumes
    /// at `seq_floor + 1` — exactly the number the next [`Self::seal`] will
    /// stamp. The generation is bumped only on success, for the same reason
    /// `seal` does not consume a sequence number on failure.
    ///
    /// `manifests` are the lineage chains subsequent deltas may address with
    /// [`KeyRange::ManifestInterval`](super::KeyRange::ManifestInterval). A
    /// publisher that always sends exact hashes passes an empty vector.
    pub fn snapshot(
        &mut self,
        media: Vec<TierMedium>,
        manifests: Vec<TierPlacementManifest>,
        entries: Vec<TierPlacementEntry>,
    ) -> Result<TierPlacementSnapshotV1, TierPlacementError> {
        let snapshot = TierPlacementSnapshotV1 {
            v: TIER_PLACEMENT_SCHEMA_VERSION,
            cache: self.cache,
            instance_id: self.instance_id,
            registration_epoch: self.registration_epoch,
            snapshot_generation: self.snapshot_generation + 1,
            seq_floor: self.last_seq(),
            media,
            manifests,
            entries,
        };
        snapshot.validate()?;
        self.snapshot_generation = snapshot.snapshot_generation;
        Ok(snapshot)
    }
}
