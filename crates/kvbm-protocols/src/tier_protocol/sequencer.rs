// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publisher-side sequence/generation bookkeeping for the tier-placement
//! stream.
//!
//! This is the whole publisher contract the rhino-side wiring (CT-2) needs:
//! pure, synchronous, transport-free. It owns `seq` and
//! `snapshot_generation`, which expose observed gaps and generation changes.
//! Thus, no publisher must reimplement the monotonicity rules. It also applies
//! the same `validate()` that the consumer applies. A publisher cannot put a
//! message on the wire that its own consumer rejects.
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
    ///
    /// # The publisher's obligations, so this arm cannot wedge
    ///
    /// A publisher that discards this vector silently violates the contract:
    /// the ops are physically gone, the sequence is still contiguous, and the
    /// consumer therefore stays *valid* while believing state the publisher has
    /// already changed. For a `Remove` that is a stale success — the exact
    /// failure the tier stream exists to prevent. So:
    ///
    /// - **Retain** the ops in a bounded buffer and re-seal them, in order,
    ///   ahead of anything newer, once the gate releases.
    /// - **If the buffer overflows and ops must be dropped, call
    ///   [`TierPlacementSequencer::mark_divergent`] first.** That converts a
    ///   silent hole into a sequence gap the consumer recovers from. Dropping
    ///   without it is the one thing a publisher must never do.
    /// - **If the snapshot push keeps failing**, stop retrying and call
    ///   [`TierPlacementSequencer::abandon_pending_snapshot`] rather than
    ///   holding the gate forever. Emission that never resumes leaves the
    ///   consumer serving the *previous* generation as valid indefinitely;
    ///   resuming at an uninstalled generation makes it invalid and empty. Both
    ///   lose reuse, only one of them lies.
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
/// publisher half of the contract. If a publisher ignores it, the generation
/// race invalidates the projection and produces empty answers.
///
/// # Why there is a second lever: publisher-side divergence
///
/// The gate above handles the case where the publisher *knows* what it sent.
/// It does not cover the case where the publisher knows it lost an op it never
/// sent — a `Remove` dropped by a full transmit queue, or retained ops
/// discarded under [`SealOutcome::Deferred`]. Nothing on the wire records
/// those: the sequence stays contiguous, so the consumer stays valid and keeps
/// advertising a copy that is gone. Under-reporting a removal *is*
/// over-reporting residency, so the usual "the stream under-reports, never
/// over-reports" argument holds for `Ready` and inverts for `Remove`.
///
/// [`Self::mark_divergent`] is how a publisher says so. It burns one sequence
/// number, which the consumer reads as a gap by the rule it already applies —
/// invalidate, ask for a snapshot, answer empty until one installs. No new wire
/// field, no new consumer state machine, and unforgeable in the useful
/// direction: only the publisher can decide its own stream diverged.
///
/// This lever cannot detect a batch lost after the transport accepts it. A
/// later batch exposes that gap. A successful snapshot repairs the state.
/// Until either event occurs, a lost terminal `Remove` can remain visible as
/// advisory placement data.
#[derive(Debug, Clone)]
pub struct TierPlacementSequencer {
    cache: CacheManifestId,
    instance_id: InstanceId,
    registration_epoch: RegistrationEpoch,
    snapshot_generation: u64,
    next_seq: u64,
    /// Generation of a sealed-but-unacknowledged snapshot.
    pending_install: Option<u64>,
    /// The publisher has lost an op it never transmitted, so the consumer's
    /// view is (or is about to become) wrong in the stale-success direction.
    /// Cleared by sealing a snapshot, which replaces the consumer's state
    /// wholesale.
    divergent: bool,
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
            divergent: false,
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

    /// Whether the publisher owes the consumer a snapshot *now*, rather than at
    /// the next periodic push.
    ///
    /// True from [`Self::mark_divergent`] until a snapshot is sealed. A
    /// publisher polls this in its emission loop and pushes out of band; the
    /// periodic timer remains the backstop for the case where the poll is
    /// starved.
    #[must_use]
    pub const fn needs_snapshot(&self) -> bool {
        self.divergent
    }

    /// Declare that this publisher's delta stream no longer describes its own
    /// state, forcing the consumer to resync.
    ///
    /// Call it when an op is lost *without* reaching the wire — a dropped
    /// `Remove`, or retained [`SealOutcome::Deferred`] ops discarded under
    /// buffer pressure. Two effects, both immediate:
    ///
    /// 1. **One sequence number is burned.** The next sealed batch is
    ///    `last_seq + 2` from the consumer's perspective, which is a gap by the
    ///    rule the consumer already applies: invalidate, request a snapshot,
    ///    answer empty. This is the whole point — the consumer stops answering
    ///    with the stale copy on the *very next delta*, on the lossy plane,
    ///    without waiting for the HTTP push to land.
    /// 2. **[`Self::needs_snapshot`] arms**, so the publisher pushes a snapshot
    ///    at the earliest opportunity instead of at the next 60 s tick.
    ///
    /// **[`Self::seal`] deliberately keeps emitting while divergent.** Holding
    /// deltas back would leave the consumer valid-and-stale for the whole push
    /// round trip; emitting makes it invalid-and-empty on the next batch. Empty
    /// costs a redundant transfer, stale costs a transfer from a copy that no
    /// longer exists, so the fail-safe direction is to keep talking.
    ///
    /// Idempotent while armed: a burst of dropped ops burns one number, because
    /// one gap invalidates exactly as thoroughly as ten. Returns whether this
    /// call armed it.
    pub fn mark_divergent(&mut self) -> bool {
        if self.divergent {
            return false;
        }
        self.divergent = true;
        // Saturating rather than wrapping: at the u64 ceiling (2^64 sealed
        // batches in one epoch) a wrap would restart the sequence inside a live
        // epoch, which the consumer reads as a replay rather than a gap.
        self.next_seq = self.next_seq.saturating_add(1);
        true
    }

    /// Give up on a sealed-but-unacknowledged snapshot, releasing the emission
    /// gate and declaring divergence.
    ///
    /// The wedge this exists to prevent: a publisher whose push cannot succeed
    /// (hub unreachable, credential rejected, feature disabled) holds
    /// `pending_install` forever, [`Self::seal`] defers forever, and the stream
    /// goes silent — leaving the consumer serving the *previous* generation as
    /// valid for as long as the process lives. Silence is the one failure mode
    /// the consumer cannot detect.
    ///
    /// Releasing instead resumes emission at the bumped generation, which the
    /// consumer reads as `GenerationAhead`: invalidate, request, answer empty.
    /// The publisher stays divergent, so it retries a snapshot when it can.
    ///
    /// **The generation bump is what invalidates here**, not the sequence
    /// number [`Self::mark_divergent`] burns on the way through — a consumer
    /// still holding the previous generation rejects on generation before it
    /// ever looks at the sequence. The burn is not wasted, though: it covers
    /// the case a failed push cannot distinguish, where the install actually
    /// landed and only the response was lost. There the consumer *is* at the
    /// new generation, and the burn is what tells it to resync rather than
    /// accept a stream the publisher no longer vouches for. Abandoning means
    /// "I do not know what the consumer has", so it costs one extra resync in
    /// that case and buys correctness in the other — and the publisher owes a
    /// snapshot either way, because [`Self::needs_snapshot`] stays armed.
    ///
    /// Returns whether a pending install was actually armed.
    pub fn abandon_pending_snapshot(&mut self) -> bool {
        if self.pending_install.take().is_none() {
            return false;
        }
        self.mark_divergent();
        true
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
    ///
    /// Divergence ([`Self::mark_divergent`]) does **not** hold emission: the
    /// batch goes out carrying the burned gap, which is what makes the consumer
    /// stop answering with state the publisher knows is wrong.
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
    /// installed is a delta the consumer will drop. A publisher that gives up
    /// entirely must call [`Self::abandon_pending_snapshot`] rather than hold
    /// the gate: see that method for why silence is worse than resuming.
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
        // A snapshot is replace-all state assembled from the publisher's own
        // residency map, so it repairs whatever divergence prompted it. Cleared
        // on seal rather than on install: divergence recorded *after* this call
        // describes state this snapshot does not carry, and must arm a fresh
        // one.
        self.divergent = false;
        Ok(snapshot)
    }
}
