// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned tier-placement event stream (R7b §2) — wire types only.
//!
//! This is the advisory placement stream that rides *alongside* the legacy
//! `kvbm_logical::events::KvbmCacheEvents` index stream. The legacy stream is
//! untouched: it carries no tier, resource, lane, generation, sequence number,
//! or recovery, and it keeps flowing unchanged for the consolidator and the
//! hub's legacy indexer.
//!
//! # Design rules encoded here (R7b §1)
//!
//! 1. **Advisory with recovery.** `seq`, [`RegistrationEpoch`], and
//!    `snapshot_generation` expose a continuity failure when evidence reaches
//!    the consumer. A snapshot repairs detected loss. Authority stays with
//!    owner-side bundle or manager state. This stream never becomes a source of
//!    truth. A consumer that detects a continuity failure must answer **empty**.
//!    A dropped terminal batch has no later sequence evidence. It can leave
//!    stale advisory data until a successful snapshot installs. A caller must
//!    acquire the exact owner `BundleLease` before it uses a holder.
//! 2. **G1 is not publishable.** [`TierDepth::G1`] (depth 0) is rejected by
//!    every `validate()` on this module, publisher side and consumer side.
//!    G1-only visibility goes through the bundle-advertisement path with its
//!    staging preconditions.
//! 3. **Self-describing before interpretation.** Version, cache identity,
//!    instance, epoch and sequence all precede the ops, so a batch can be
//!    rejected without interpreting a single placement.
//! 4. **`Reserved` / `Copying` never reach the wire.** They are local states.
//!    The op enum has no such variant, so an encoder cannot express one and a
//!    decoder rejects it as an unknown variant — a structural guarantee, not a
//!    convention.
//!
//! # Why this module lives in `kvbm-protocols`, not `kvbm-logical`
//!
//! The R7b spec heads §2 with `kvbm-logical/src/events/tier_protocol.rs`. That
//! placement is not reachable without forking identity types:
//! [`CacheManifestId`] and [`RegistrationEpoch`] live here, and `kvbm-logical`
//! does not depend on this crate (nor should it — it is the transport-free
//! lifecycle core, and this crate pulls `velo-ext`). Re-declaring those
//! newtypes in `kvbm-logical` would fork exactly the identities the epoch /
//! manifest machinery exists to keep single-valued. This crate already holds
//! every input the schema needs — including [`BundleResourceLineage`], the
//! exact lineage constructor `ManifestInterval` resolves against — and both
//! ends of the seam (the rhino-side publisher and `kvbm-hub`) already depend on
//! it. The reusable *batching* half of §2 does land in `kvbm-logical`, as
//! `kvbm_logical::events::OrderedBatcher`.
//!
//! [`BundleResourceLineage`]: crate::cache_manifest::BundleResourceLineage
//!
//! # The envelope is frozen at the v1 shape
//!
//! The delta plane is msgpack over ZMQ, and `rmp-serde` encodes structs
//! *positionally* (as arrays). A trailing `#[serde(default)]` field therefore
//! survives old→new decode but **not** new→old. Consequently
//! [`TierPlacementBatchV1`] must never grow a field: schema growth goes through
//! `v` (whole-batch reject, then a `…V2` type), or through the snapshot, which
//! travels the JSON/HTTP control plane. Do not "just add a field" here.
//!
//! # `PositionRun` is deliberately absent
//!
//! An earlier draft of this schema carried
//! `PositionRun { end_hash: SequenceHash, len: u32 }` and claimed a consumer
//! could reconstruct membership by walking back from the terminal hash. It
//! cannot, and the omission is load-bearing (2026-08-05 correction, finding 4 —
//! `agent-docs/research/thermonuclear-review-2026-08-05.md`):
//!
//! - A [`SequenceHash`] (a positional lineage hash) carries its own full
//!   sequence hash plus **one** parent *fragment* — not the chain of ancestor
//!   hashes. A terminal hash and a length are not recoverable membership.
//! - The consumer that most needs the compression is the one installing a
//!   *recovery* snapshot, and it has by definition already discarded the
//!   projection it would need to walk the chain.
//! - Inferring membership from fragments in a possibly-colliding global index
//!   is the "stale success" failure this stream is designed to make impossible.
//!
//! [`KeyRange::ManifestInterval`] replaces it: an interval is only ever
//! resolved against a lineage manifest the consumer has **already installed and
//! validated** via [`BundleResourceLineage::new`], which checks adjacent-hash
//! position continuity and parent-fragment continuity. A consumer without that
//! manifest MUST treat the op as a sequence gap and request a snapshot. And
//! because a recovery snapshot is exactly the case where no manifest is
//! installed yet, snapshots carry exact [`KeyRange::Hashes`] only —
//! [`TierPlacementSnapshotV1::validate`] enforces it.
//!
//! [`BundleResourceLineage::new`]: crate::cache_manifest::BundleResourceLineage::new
//!
//! # Where manifests come from (v1: snapshots only)
//!
//! A manifest enters a consumer exactly one way in v1: the
//! [`TierPlacementSnapshotV1::manifests`] header. R7b §2 also permits a manifest
//! to be established by prior `Hashes` runs in the delta stream; that is
//! **deferred**, because it needs a publisher-side rule for assigning a stable
//! `manifest_id` to a run that CT-2 has not defined, and a consumer that guessed
//! the rule would resolve intervals against a manifest the publisher never
//! meant. Until then `ManifestInterval` is only usable after a snapshot, which
//! is also when it pays: steady state, not recovery.
//!
//! Note the asymmetry with the frozen delta envelope above: the snapshot travels
//! the JSON/HTTP control plane, which is map-encoded, so it *may* grow a
//! `#[serde(default)]` field. The freeze applies to
//! [`TierPlacementBatchV1`] alone.

use std::fmt;

use kvbm_common::{LogicalResourceId, SequenceHash};
use serde::{Deserialize, Serialize};

use crate::cache_manifest::{BundleResourceLineage, CacheManifestId, RegistrationEpoch};

mod sequencer;

pub use sequencer::{SealOutcome, TierPlacementSequencer};

#[cfg(test)]
mod tests;

/// Schema version of the types in this module. Consumers reject `v` greater
/// than this **whole**, never per-op.
pub const TIER_PLACEMENT_SCHEMA_VERSION: u16 = 1;

/// ZMQ topic frame for the tier-placement delta stream.
///
/// Shared by publisher and consumer so the two ends cannot drift into two
/// hard-coded literals. Note that a subscriber filtering on `b""` (as the hub's
/// legacy indexer does) receives *every* topic, so a consumer must dispatch on
/// the topic frame rather than assume isolation.
pub const TIER_PLACEMENT_SUBJECT: &str = "kvbm.tier_placements";

/// Instance identity on this stream.
///
/// This is `velo_ext::InstanceId`, the identity the hub already keys its
/// registry, bundle ownership and `on_unregister` path on — not the legacy
/// `kvbm_logical::events::InstanceId` (a bare `u128`).
pub type InstanceId = velo_ext::InstanceId;

/// Anti-amplification bound on ops in one delta batch.
///
/// The `*_MAX_*` constants below bound the state a *single* message can make a
/// consumer retain. They are guards, not capacity policy: every value is orders
/// of magnitude above any plausible publisher (the default batching cadence
/// flushes at 1024 items), because a bound a real publisher can hit would turn
/// the spec's "temporary miss" into a permanent one — a projection that can
/// never install answers empty forever. Aggregate, cross-message hub limits are
/// a separate concern and belong to the consumer, not the wire.
///
/// These also do not bound peak *decode* allocation: a hostile length prefix is
/// spent inside the deserializer before `validate()` runs. That is the
/// transport's frame-size limit to enforce.
pub const TIER_PLACEMENT_MAX_OPS_PER_BATCH: usize = 1 << 16;
/// Anti-amplification bound on entries in one snapshot.
pub const TIER_PLACEMENT_MAX_SNAPSHOT_ENTRIES: usize = 1 << 20;
/// Anti-amplification bound on keys covered by one message, summed across every
/// op or entry (an interval counts its `len`).
pub const TIER_PLACEMENT_MAX_KEYS_PER_MESSAGE: usize = 1 << 20;
/// Anti-amplification bound on media described by one snapshot header. Each
/// entry is one policy depth, and depth is a `u8`.
pub const TIER_PLACEMENT_MAX_MEDIA: usize = 256;
/// Anti-amplification bound on lineage manifests installed by one snapshot.
pub const TIER_PLACEMENT_MAX_MANIFESTS: usize = 1 << 12;

/// Capability bit: the medium can serve a transfer directly (no staging copy).
pub const TIER_MEDIUM_CAP_DIRECT_SERVABLE: u32 = 1 << 0;
/// Capability bit: a transfer from this medium must be staged through a faster
/// tier first.
pub const TIER_MEDIUM_CAP_STAGING_REQUIRED: u32 = 1 << 1;

/// Wire tier identity — a numeric depth, **never** an enum.
///
/// Encoding the depth numerically means adding a tier, or running several media
/// at one policy depth, never needs a protocol version bump (2026-08-05
/// correction, finding 5: the earlier `enum TierClass { G2, G3, G4 }` was not
/// GX-extensible). Depth 0 is G1 and is never publishable on this stream;
/// depth 1 is conventionally G2, and so on. Medium and capability metadata
/// travel separately and once, in [`TierMedium`] on the snapshot header, and
/// are versioned independently of placement deltas.
///
/// Only [`TierDepth::G1`] gets a named constant, because depth 0 is a protocol
/// *rule* — it is the one depth with wire semantics. There is deliberately no
/// `G2`/`G3` alias: naming depths would smuggle back the 1:1 depth↔generation
/// assumption the enum encoding died of, and it is exactly wrong once several
/// media share one policy depth. Write the number.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct TierDepth(pub u8);

impl TierDepth {
    /// Device-resident depth. Never publishable on this stream (§1 rule 3).
    pub const G1: Self = Self(0);

    /// Numeric depth.
    #[must_use]
    pub const fn depth(self) -> u8 {
        self.0
    }

    /// Whether this depth may appear on the tier-placement stream.
    #[must_use]
    pub const fn is_publishable(self) -> bool {
        self.0 != Self::G1.0
    }
}

impl fmt::Display for TierDepth {
    /// Renders the G-notation operators read in logs. This is a display
    /// convention layered over the numeric depth, not a wire encoding — the
    /// wire always carries the number.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "G{}", self.0.saturating_add(1))
    }
}

/// Per-depth medium description, published in the snapshot header only.
///
/// Deltas carry a bare [`TierDepth`]; this is how a consumer learns what that
/// depth *is*. Unknown capability bits are ignored by consumers, so the bit
/// vocabulary can grow without a version bump.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TierMedium {
    /// Depth this medium backs.
    pub depth: TierDepth,
    /// Free-form stable label ("pinned-host", "nvme", "object", ...).
    pub medium: String,
    /// Coarse capability bits; see `TIER_MEDIUM_CAP_*`. Unknown bits ignored.
    pub capabilities: u32,
}

/// Where a placement sits in the logical-resource / execution-lane space.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlacementScope {
    /// Logical KV resource this placement covers.
    pub resource: LogicalResourceId,
    /// Execution lane (ADP); 0 for unitary deployments.
    pub lane: u8,
}

impl PlacementScope {
    /// Scope for a unitary (single-lane) deployment.
    #[must_use]
    pub const fn unitary(resource: LogicalResourceId) -> Self {
        Self { resource, lane: 0 }
    }
}

/// How one logical copy is physically laid out on the owner.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PhysicalPlacementMode {
    /// One whole copy on the owner.
    Whole,
    /// TP shards — all `count` are required for one logical copy.
    TpShards {
        /// Number of shards making up the copy.
        count: u16,
    },
    /// ReplicatedData stripes — disjoint stripes, owner-restored.
    DataStripes {
        /// Number of stripes making up the copy.
        count: u16,
    },
}

/// Placement keys.
///
/// See the module docs for why a terminal-hash run (`PositionRun`) is not a
/// legal encoding here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KeyRange {
    /// Exact membership. The only legal form in snapshots; always legal in
    /// deltas.
    Hashes(Vec<SequenceHash>),
    /// Delta-only optimization: the positional interval `[start, start + len)`
    /// of an immutable lineage manifest the consumer has already installed and
    /// validated under `manifest_id`. A consumer without that manifest, or one
    /// whose manifest does not cover the interval, MUST treat the op as a
    /// sequence gap and request a snapshot — never guess membership.
    ManifestInterval {
        /// Publisher-assigned identity of the installed lineage manifest.
        manifest_id: u64,
        /// First position of the interval, in manifest coordinates.
        start: u32,
        /// Number of positions covered. Never zero.
        len: u32,
    },
}

impl KeyRange {
    /// Whether this range states membership exactly (no manifest lookup).
    #[must_use]
    pub const fn is_exact(&self) -> bool {
        matches!(self, Self::Hashes(_))
    }

    /// Number of keys this range covers, for the per-message bound. An
    /// interval counts its `len`: that is how many keys the consumer will
    /// retain once it resolves.
    fn key_count(&self) -> usize {
        match self {
            Self::Hashes(hashes) => hashes.len(),
            Self::ManifestInterval { len, .. } => *len as usize,
        }
    }

    /// Shape validation that needs no consumer state.
    ///
    /// Interval resolution against an installed manifest is a *consumer*
    /// concern and is deliberately not attempted here.
    fn validate(&self, index: usize) -> Result<(), TierPlacementError> {
        match self {
            Self::Hashes(hashes) => {
                if hashes.is_empty() {
                    return Err(TierPlacementError::EmptyKeys { index });
                }
                Ok(())
            }
            Self::ManifestInterval {
                manifest_id,
                start,
                len,
            } => {
                if *len == 0 {
                    return Err(TierPlacementError::EmptyKeys { index });
                }
                if start.checked_add(*len).is_none() {
                    return Err(TierPlacementError::IntervalOverflow {
                        index,
                        manifest_id: *manifest_id,
                        start: *start,
                        len: *len,
                    });
                }
                Ok(())
            }
        }
    }
}

/// One placement transition.
///
/// Only `Ready` and `Remove` exist. `Reserved` and `Copying` are local states
/// that must not reduce any remote copy's alternative cost, so they have no
/// wire representation at all.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TierPlacementOp {
    /// A placement became Ready (transferable now).
    Ready {
        /// Resource / lane the placement covers.
        scope: PlacementScope,
        /// Depth the copy is resident at.
        tier: TierDepth,
        /// Physical layout of the copy.
        placement: PhysicalPlacementMode,
        /// Bundle/lineage generation, matching
        /// `BundleAdvertisementRecord.generation` semantics. A `Remove` with an
        /// older generation cannot remove a newer `Ready`.
        generation: u64,
        /// Keys covered.
        keys: KeyRange,
    },
    /// A placement is gone.
    Remove {
        /// Resource / lane the placement covered.
        scope: PlacementScope,
        /// Depth the copy was resident at.
        tier: TierDepth,
        /// Generation this removal speaks for.
        generation: u64,
        /// Keys covered.
        keys: KeyRange,
    },
}

impl TierPlacementOp {
    /// Resource / lane this op speaks for.
    #[must_use]
    pub const fn scope(&self) -> PlacementScope {
        match self {
            Self::Ready { scope, .. } | Self::Remove { scope, .. } => *scope,
        }
    }

    /// Depth this op speaks for.
    #[must_use]
    pub const fn tier(&self) -> TierDepth {
        match self {
            Self::Ready { tier, .. } | Self::Remove { tier, .. } => *tier,
        }
    }

    /// Generation this op speaks for.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        match self {
            Self::Ready { generation, .. } | Self::Remove { generation, .. } => *generation,
        }
    }

    /// Keys this op covers.
    #[must_use]
    pub const fn keys(&self) -> &KeyRange {
        match self {
            Self::Ready { keys, .. } | Self::Remove { keys, .. } => keys,
        }
    }

    fn validate(&self, index: usize) -> Result<(), TierPlacementError> {
        let tier = self.tier();
        if !tier.is_publishable() {
            return Err(TierPlacementError::G1NotPublishable { index });
        }
        self.keys().validate(index)
    }
}

/// One Ready placement in a snapshot body.
///
/// Structurally a `TierPlacementOp::Ready` without the variant tag: a snapshot
/// is by definition the complete Ready set, so a `Remove` entry is unspeakable
/// rather than merely rejected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TierPlacementEntry {
    /// Resource / lane the placement covers.
    pub scope: PlacementScope,
    /// Depth the copy is resident at.
    pub tier: TierDepth,
    /// Physical layout of the copy.
    pub placement: PhysicalPlacementMode,
    /// Bundle/lineage generation of the copy.
    pub generation: u64,
    /// Keys covered. Exact membership only — see the module docs.
    pub keys: KeyRange,
}

impl TierPlacementEntry {
    fn validate(&self, index: usize) -> Result<(), TierPlacementError> {
        if !self.tier.is_publishable() {
            return Err(TierPlacementError::G1NotPublishable { index });
        }
        if !self.keys.is_exact() {
            return Err(TierPlacementError::InexactSnapshotKeys { index });
        }
        self.keys.validate(index)
    }
}

/// An immutable lineage manifest a snapshot installs under `manifest_id`, so
/// that subsequent deltas can address its positions with
/// [`KeyRange::ManifestInterval`] instead of repeating exact hashes.
///
/// This is the *only* way a manifest reaches a consumer in v1 (see the module
/// docs). Without it `ManifestInterval` is unresolvable by construction: nothing
/// else on this wire binds an id to a hash chain, so every interval would gap,
/// request a snapshot, install no manifest, and gap again — a livelock, not the
/// "temporary miss" R7b §4 promises.
///
/// The hashes are carried raw rather than as a [`BundleResourceLineage`] so that
/// a malformed chain is a *validation* failure with a precise error, not a
/// deserialization failure that reads as a corrupt frame.
///
/// [`BundleResourceLineage`]: crate::cache_manifest::BundleResourceLineage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TierPlacementManifest {
    /// Publisher-assigned identity. Unique within one snapshot; stable for as
    /// long as the publisher references it from deltas.
    pub manifest_id: u64,
    /// Logical resource the chain belongs to. An interval whose op names a
    /// different resource is rejected by the consumer.
    pub resource: LogicalResourceId,
    /// The exact ordered chain. Validated as a real lineage (adjacent-hash
    /// position and parent-fragment continuity) by [`Self::lineage`].
    pub hashes: Vec<SequenceHash>,
}

impl TierPlacementManifest {
    /// Build (and thereby validate) the lineage this manifest describes.
    ///
    /// Delegates to [`BundleResourceLineage::new`] rather than re-implementing
    /// continuity checks, so a manifest installed here and a bundle lineage
    /// advertised through the directory can never disagree about what a valid
    /// chain is.
    ///
    /// [`BundleResourceLineage::new`]: crate::cache_manifest::BundleResourceLineage::new
    pub fn lineage(&self) -> Result<BundleResourceLineage, TierPlacementError> {
        BundleResourceLineage::new(self.resource, self.hashes.clone()).map_err(|source| {
            TierPlacementError::InvalidManifest {
                manifest_id: self.manifest_id,
                detail: source.to_string(),
            }
        })
    }
}

/// Wire envelope, v1. Fields are ordered for cheap reject-before-interpret of
/// stale publishers.
///
/// **Never grow this struct** — see the module docs on positional msgpack
/// encoding. Growth goes through `v`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TierPlacementBatchV1 {
    /// Schema version. Consumers reject `v > 1` batches whole.
    pub v: u16,
    /// Hash-domain / manifest identity. A consumer indexes placements per
    /// `(cache, instance)`; mismatched cache ids never merge.
    pub cache: CacheManifestId,
    /// Publisher instance.
    pub instance_id: InstanceId,
    /// Hub-minted lifecycle identity of this publisher process. A new epoch
    /// invalidates all prior advisory state for `instance_id`.
    pub registration_epoch: RegistrationEpoch,
    /// Monotone per `(instance, epoch)`. A gap means the consumer discards this
    /// instance's advisory projection and requests a snapshot.
    pub seq: u64,
    /// Increments each time the publisher installs a new full snapshot. Deltas
    /// carrying an older generation than the consumer's installed snapshot are
    /// discarded.
    pub snapshot_generation: u64,
    /// Ordered placement transitions. Order is significant: the batching layer
    /// never reorders and never coalesces.
    pub ops: Vec<TierPlacementOp>,
}

impl TierPlacementBatchV1 {
    /// Version + per-op shape validation.
    ///
    /// Whole-batch: the first failure rejects the batch, so no consumer can
    /// observe a partially-applied batch. Version rejection is a *distinct*
    /// error from an undecodable frame — see [`TierPlacementRejection`] — so
    /// the two can be counted separately, which is why this is a plain method
    /// rather than a hand-written `Deserialize`.
    pub fn validate(&self) -> Result<(), TierPlacementError> {
        if self.v > TIER_PLACEMENT_SCHEMA_VERSION {
            return Err(TierPlacementError::UnsupportedVersion {
                found: self.v,
                supported: TIER_PLACEMENT_SCHEMA_VERSION,
            });
        }
        if self.ops.len() > TIER_PLACEMENT_MAX_OPS_PER_BATCH {
            return Err(TierPlacementError::TooLarge {
                what: "batch ops",
                count: self.ops.len(),
                limit: TIER_PLACEMENT_MAX_OPS_PER_BATCH,
            });
        }
        let mut keys = 0usize;
        for (index, op) in self.ops.iter().enumerate() {
            op.validate(index)?;
            keys = keys.saturating_add(op.keys().key_count());
        }
        if keys > TIER_PLACEMENT_MAX_KEYS_PER_MESSAGE {
            return Err(TierPlacementError::TooLarge {
                what: "batch keys",
                count: keys,
                limit: TIER_PLACEMENT_MAX_KEYS_PER_MESSAGE,
            });
        }
        Ok(())
    }

    /// Codec-agnostic decode gate: hand it the result of *any* deserializer and
    /// get back a batch that is guaranteed validated.
    ///
    /// Consumers should route every inbound frame through this (or
    /// [`Self::decode_json`]) rather than calling the deserializer directly, so
    /// there is exactly one place where "decode, then validate whole" holds.
    pub fn from_decoded<E: fmt::Display>(
        decoded: Result<Self, E>,
    ) -> Result<Self, TierPlacementError> {
        let batch = decoded.map_err(|error| TierPlacementError::Decode {
            detail: error.to_string(),
        })?;
        batch.validate()?;
        Ok(batch)
    }

    /// JSON convenience wrapper over [`Self::from_decoded`].
    pub fn decode_json(bytes: &[u8]) -> Result<Self, TierPlacementError> {
        Self::from_decoded(serde_json::from_slice::<Self>(bytes))
    }
}

/// Full placement state for one `(cache, instance)`, pushed on the reliable
/// plane at registration, on hub request, and periodically.
///
/// Snapshot install is transactional per `(cache, instance)`: replace-all, bump
/// the installed generation, then resume delta application at
/// `seq > seq_floor`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TierPlacementSnapshotV1 {
    /// Schema version. Rejected whole above [`TIER_PLACEMENT_SCHEMA_VERSION`].
    pub v: u16,
    /// Hash-domain / manifest identity.
    pub cache: CacheManifestId,
    /// Publisher instance.
    pub instance_id: InstanceId,
    /// Publisher lifecycle identity this state belongs to.
    pub registration_epoch: RegistrationEpoch,
    /// The generation subsequent deltas carry.
    pub snapshot_generation: u64,
    /// Deltas with `seq <= seq_floor` are subsumed by this snapshot.
    pub seq_floor: u64,
    /// Medium metadata for every depth referenced by `entries`. This is the
    /// only place medium/capability data travels, and it travels once.
    ///
    /// Coverage of the *snapshot body* is enforced, not merely documented:
    /// [`Self::validate`] rejects a snapshot whose entries name a depth the
    /// header omits.
    ///
    /// It does not extend to deltas. [`TierPlacementOp::validate`] rejects only
    /// G1, so a publisher that brings a new depth online may `Ready` at it
    /// before its next snapshot describes it. That is deliberate — rejecting
    /// such a delta would add an invalidation surface for a condition the next
    /// periodic snapshot closes on its own — but it means a consumer must treat
    /// a depth with no [`TierMedium`] as *capabilities unknown*, never as
    /// capabilities zero.
    pub media: Vec<TierMedium>,
    /// Lineage manifests this snapshot installs, for later
    /// [`KeyRange::ManifestInterval`] deltas. `#[serde(default)]` so a publisher
    /// that never uses intervals can omit the field entirely; the snapshot is
    /// JSON/map-encoded on the control plane, so this is decode-compatible in
    /// both directions.
    #[serde(default)]
    pub manifests: Vec<TierPlacementManifest>,
    /// The complete Ready set. Exact keys only.
    pub entries: Vec<TierPlacementEntry>,
}

impl TierPlacementSnapshotV1 {
    /// Version, medium and per-entry validation. Whole-snapshot: a single bad
    /// entry rejects the install rather than half-applying it.
    pub fn validate(&self) -> Result<(), TierPlacementError> {
        if self.v > TIER_PLACEMENT_SCHEMA_VERSION {
            return Err(TierPlacementError::UnsupportedVersion {
                found: self.v,
                supported: TIER_PLACEMENT_SCHEMA_VERSION,
            });
        }
        if self.media.len() > TIER_PLACEMENT_MAX_MEDIA {
            return Err(TierPlacementError::TooLarge {
                what: "snapshot media",
                count: self.media.len(),
                limit: TIER_PLACEMENT_MAX_MEDIA,
            });
        }
        if self.entries.len() > TIER_PLACEMENT_MAX_SNAPSHOT_ENTRIES {
            return Err(TierPlacementError::TooLarge {
                what: "snapshot entries",
                count: self.entries.len(),
                limit: TIER_PLACEMENT_MAX_SNAPSHOT_ENTRIES,
            });
        }
        let mut seen: Vec<TierDepth> = Vec::with_capacity(self.media.len());
        for (index, medium) in self.media.iter().enumerate() {
            if !medium.depth.is_publishable() {
                return Err(TierPlacementError::MediumNotPublishable { index });
            }
            if seen.contains(&medium.depth) {
                return Err(TierPlacementError::DuplicateMediumDepth {
                    depth: medium.depth,
                });
            }
            seen.push(medium.depth);
        }
        if self.manifests.len() > TIER_PLACEMENT_MAX_MANIFESTS {
            return Err(TierPlacementError::TooLarge {
                what: "snapshot manifests",
                count: self.manifests.len(),
                limit: TIER_PLACEMENT_MAX_MANIFESTS,
            });
        }
        // Validate every manifest here, not at install time: a snapshot install
        // is replace-all, so a manifest that fails halfway through would leave
        // the consumer holding a projection assembled from a body it had already
        // decided to reject.
        let mut manifest_ids: Vec<u64> = Vec::with_capacity(self.manifests.len());
        let mut manifest_keys = 0usize;
        for manifest in &self.manifests {
            if manifest_ids.contains(&manifest.manifest_id) {
                return Err(TierPlacementError::DuplicateManifestId {
                    manifest_id: manifest.manifest_id,
                });
            }
            manifest_ids.push(manifest.manifest_id);
            manifest.lineage()?;
            manifest_keys = manifest_keys.saturating_add(manifest.hashes.len());
        }
        if manifest_keys > TIER_PLACEMENT_MAX_KEYS_PER_MESSAGE {
            return Err(TierPlacementError::TooLarge {
                what: "snapshot manifest keys",
                count: manifest_keys,
                limit: TIER_PLACEMENT_MAX_KEYS_PER_MESSAGE,
            });
        }
        let mut keys = 0usize;
        for (index, entry) in self.entries.iter().enumerate() {
            entry.validate(index)?;
            // Every depth the body references must be described by the header.
            // The snapshot is the *only* place medium/capability metadata
            // travels (R7b §2), so an undescribed depth leaves the consumer
            // unable to tell direct-servable from staging-required and forced to
            // guess a cost — the one thing the header exists to prevent. `seen`
            // is at most 256 entries (depth is a `u8`), so the scan is bounded.
            if !seen.contains(&entry.tier) {
                return Err(TierPlacementError::UndescribedDepth {
                    index,
                    depth: entry.tier,
                });
            }
            keys = keys.saturating_add(entry.keys.key_count());
        }
        if keys > TIER_PLACEMENT_MAX_KEYS_PER_MESSAGE {
            return Err(TierPlacementError::TooLarge {
                what: "snapshot keys",
                count: keys,
                limit: TIER_PLACEMENT_MAX_KEYS_PER_MESSAGE,
            });
        }
        Ok(())
    }

    /// Codec-agnostic decode gate; see [`TierPlacementBatchV1::from_decoded`].
    pub fn from_decoded<E: fmt::Display>(
        decoded: Result<Self, E>,
    ) -> Result<Self, TierPlacementError> {
        let snapshot = decoded.map_err(|error| TierPlacementError::Decode {
            detail: error.to_string(),
        })?;
        snapshot.validate()?;
        Ok(snapshot)
    }

    /// JSON convenience wrapper over [`Self::from_decoded`].
    pub fn decode_json(bytes: &[u8]) -> Result<Self, TierPlacementError> {
        Self::from_decoded(serde_json::from_slice::<Self>(bytes))
    }
}

/// Coarse rejection class, for the per-reason drop counters R7b §8 makes
/// mandatory.
///
/// An unsupported version and a corrupt frame are operationally different
/// events — one means "upgrade the consumer", the other means "the link is
/// damaged" — so they must never share a counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TierPlacementRejection {
    /// The bytes did not deserialize at all.
    Undecodable,
    /// The frame decoded but announced a schema version we cannot interpret.
    UnsupportedVersion,
    /// The frame decoded at a known version but violates the schema contract.
    Invalid,
}

/// Why a tier-placement message was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TierPlacementError {
    /// The bytes did not deserialize.
    #[error("tier placement message could not be decoded: {detail}")]
    Decode {
        /// Deserializer-supplied detail.
        detail: String,
    },
    /// Version above what this build understands; the message is dropped whole.
    #[error("unsupported tier placement schema version {found} (supported: {supported})")]
    UnsupportedVersion {
        /// Version announced by the publisher.
        found: u16,
        /// Highest version this build interprets.
        supported: u16,
    },
    /// A G1 (depth 0) placement appeared on the stream.
    #[error("tier placement at index {index} names G1, which is not publishable on this stream")]
    G1NotPublishable {
        /// Index of the offending op/entry.
        index: usize,
    },
    /// A snapshot header described a medium at G1 (depth 0).
    #[error("snapshot medium at index {index} names G1, which is not publishable on this stream")]
    MediumNotPublishable {
        /// Index of the offending medium.
        index: usize,
    },
    /// A snapshot entry used a manifest interval instead of exact membership.
    #[error("snapshot entry at index {index} must carry exact hashes, not a manifest interval")]
    InexactSnapshotKeys {
        /// Index of the offending entry.
        index: usize,
    },
    /// An op or entry covered no keys.
    #[error("tier placement at index {index} covers no keys")]
    EmptyKeys {
        /// Index of the offending op/entry.
        index: usize,
    },
    /// `start + len` overflowed `u32`.
    #[error(
        "manifest interval at index {index} overflows: manifest {manifest_id} start {start} len {len}"
    )]
    IntervalOverflow {
        /// Index of the offending op/entry.
        index: usize,
        /// Manifest the interval referenced.
        manifest_id: u64,
        /// Interval start.
        start: u32,
        /// Interval length.
        len: u32,
    },
    /// The snapshot header described the same depth twice.
    #[error("snapshot header describes depth {depth} more than once")]
    DuplicateMediumDepth {
        /// The repeated depth.
        depth: TierDepth,
    },
    /// A snapshot entry names a depth the header did not describe.
    #[error(
        "snapshot entry at index {index} names depth {depth}, which the header does not describe"
    )]
    UndescribedDepth {
        /// Index of the offending entry.
        index: usize,
        /// The undescribed depth.
        depth: TierDepth,
    },
    /// The snapshot header installed the same manifest id twice.
    #[error("snapshot header installs manifest {manifest_id} more than once")]
    DuplicateManifestId {
        /// The repeated manifest id.
        manifest_id: u64,
    },
    /// A snapshot manifest is not a valid lineage chain.
    #[error("snapshot manifest {manifest_id} is not a valid lineage: {detail}")]
    InvalidManifest {
        /// Manifest that failed to build.
        manifest_id: u64,
        /// Lineage-constructor detail.
        detail: String,
    },
    /// The message exceeds an anti-amplification bound.
    #[error("tier placement message carries {count} {what}, above the limit of {limit}")]
    TooLarge {
        /// Which collection overflowed.
        what: &'static str,
        /// Observed size.
        count: usize,
        /// Bound.
        limit: usize,
    },
}

impl TierPlacementError {
    /// Counter bucket for this rejection.
    #[must_use]
    pub const fn rejection(&self) -> TierPlacementRejection {
        match self {
            Self::Decode { .. } => TierPlacementRejection::Undecodable,
            Self::UnsupportedVersion { .. } => TierPlacementRejection::UnsupportedVersion,
            Self::G1NotPublishable { .. }
            | Self::MediumNotPublishable { .. }
            | Self::InexactSnapshotKeys { .. }
            | Self::EmptyKeys { .. }
            | Self::IntervalOverflow { .. }
            | Self::DuplicateMediumDepth { .. }
            | Self::UndescribedDepth { .. }
            | Self::DuplicateManifestId { .. }
            | Self::InvalidManifest { .. }
            | Self::TooLarge { .. } => TierPlacementRejection::Invalid,
        }
    }
}
