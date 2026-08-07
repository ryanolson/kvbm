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

/// Outcome of sealing a delta batch.
///
/// A plain `Result` cannot express "not now, ask again": deferral is not an
/// error, and adding a variant to [`TierPlacementError`] would put a
/// flow-control state into a wire-rejection type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SealOutcome {
    /// Sealed and ready to publish.
    Batch(TierPlacementBatchV1),
    /// A snapshot is sealed but not yet acknowledged installed, so no sequence
    /// number was consumed and the ops are handed straight back.
    ///
    /// The caller must retain them and re-seal **these ops first** after
    /// [`TierPlacementSequencer::note_snapshot_installed`]; the stream is
    /// order-preserving, so emitting anything ahead of them would transmit an
    /// inversion the consumer faithfully applies.
    Deferred(Vec<TierPlacementOp>),
}

/// Stamps identity, sequence, and generation onto outbound tier-placement
/// messages.
///
/// One sequencer per `(cache, instance, registration_epoch)`. A new
/// registration epoch means a new sequencer: sequence numbers are only
/// monotone *within* an epoch, and a consumer discards all prior state when the
/// epoch changes, so restarting the counters is correct rather than merely
/// tolerated.
///
/// # Why sealing a snapshot gates delta emission
///
/// A snapshot bumps the generation, and the deltas after it carry the new one.
/// Snapshots travel the reliable slow plane (HTTP) and deltas the lossy fast one
/// (ZMQ), so an ungated publisher's first post-snapshot delta routinely beats
/// its own snapshot to the consumer. The consumer reads that as
/// `GenerationAhead`, invalidates, and drops it — and the delta is never
/// retransmitted, because pub/sub has no retransmission. When the snapshot then
/// installs, the consumer resumes at `seq_floor + 1`, the next real delta is
/// `seq_floor + 2`, and it gaps. Repeat every push: under sustained emission the
/// projection is valid only for the instant between an install and the next
/// delta, which is an availability collapse rather than the bounded window R7b
/// §3 intends.
///
/// So [`Self::snapshot`] arms a gate and [`Self::seal`] defers until
/// [`Self::note_snapshot_installed`] confirms the install landed. This is the
/// publisher half of the contract; it does not change how a consumer behaves
/// against a publisher that ignores it (that case still degrades to empty
/// answers, never to stale ones).
#[derive(Debug, Clone)]
pub struct TierPlacementSequencer {
    cache: CacheManifestId,
    instance_id: InstanceId,
    registration_epoch: RegistrationEpoch,
    snapshot_generation: u64,
    next_seq: u64,
    /// Generation of a sealed-but-unacknowledged snapshot.
    pending_install: Option<u64>,
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
            pending_install: None,
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

    /// Whether a sealed snapshot is still waiting to be acknowledged installed.
    ///
    /// While this is true, [`Self::seal`] defers every batch.
    #[must_use]
    pub const fn awaiting_snapshot_install(&self) -> bool {
        self.pending_install.is_some()
    }

    /// Record that the consumer installed the snapshot at `installed_generation`,
    /// releasing delta emission.
    ///
    /// Feed it the hub's `TierPlacementSnapshotResponse::installed_generation`.
    /// Returns whether this released the gate; a stale or mismatched
    /// acknowledgement leaves it armed, so a lost or superseded push cannot let
    /// deltas outrun the state they describe.
    pub fn note_snapshot_installed(&mut self, installed_generation: u64) -> bool {
        if self.pending_install == Some(installed_generation) {
            self.pending_install = None;
            return true;
        }
        false
    }

    /// Seal a delta batch.
    ///
    /// The sequence number is consumed **only on success**: a batch rejected by
    /// `validate()` never reaches the wire, so consuming its number would
    /// manufacture a gap the consumer would then recover from, turning a local
    /// publisher bug into fleet-wide snapshot traffic. Deferral consumes nothing
    /// either, for the same reason.
    pub fn seal(&mut self, ops: Vec<TierPlacementOp>) -> Result<SealOutcome, TierPlacementError> {
        if self.awaiting_snapshot_install() {
            return Ok(SealOutcome::Deferred(ops));
        }
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
        Ok(SealOutcome::Batch(batch))
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
    ///
    /// On success the emission gate arms: [`Self::seal`] defers until
    /// [`Self::note_snapshot_installed`] confirms this generation landed. A
    /// publisher whose push fails re-pushes *this* snapshot, or seals a fresh one
    /// (which supersedes it and re-arms at the higher generation); either way
    /// deltas stay held, because a delta at a generation the consumer has not
    /// installed is a delta the consumer will drop.
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
        self.pending_install = Some(snapshot.snapshot_generation);
        Ok(snapshot)
    }
}
