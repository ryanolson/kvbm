// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-`(cache, instance)` tier-placement projection and its recovery rules.
//!
//! # The state machine, in one paragraph
//!
//! A projection is either **valid** or **invalid**. Valid means that all observed
//! deltas form one sequence at the installed generation and epoch. Invalid means
//! that the projection answers nothing and asks for a snapshot. A sequence gap,
//! an unauthorized epoch, or a future generation makes it invalid. Only a
//! credential-authorized snapshot makes it valid. The snapshot replaces all
//! state and resumes deltas at `seq_floor + 1`.
//!
//! A terminal transport loss has no later sequence evidence. It does not clear
//! `valid`. Thus, callers must treat every holder as advisory and verify the
//! exact owner before use.
//!
//! # Why the map is keyed on `(cache, instance)` and not `(cache, instance,
//! epoch)`
//!
//! R7b §4's "`(instance, epoch) → last_seq, installed_generation`" names the
//! tracked tuple, not the map key. "Unknown/new epoch ⇒ invalidate and replace"
//! is a *transition on an existing entry*; keying by epoch would instead leak
//! one entry per publisher restart, which a crash-looping publisher turns into
//! unbounded growth. The epoch lives inside the entry.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{BundleResourceLineage, CacheManifestId, RegistrationEpoch};
use kvbm_protocols::tier_protocol::{
    InstanceId, KeyRange, PhysicalPlacementMode, PlacementScope, TierDepth, TierPlacementBatchV1,
    TierPlacementError, TierPlacementOp, TierPlacementRejection, TierPlacementSnapshotV1,
};

use super::{SnapshotRequester, UnavailableSnapshotRequester};

/// Default minimum interval between snapshot requests for one
/// `(cache, instance)`.
///
/// Without this, a persistently lossy link converts every dropped batch into an
/// active message — the delta plane's whole justification is that losing it is
/// cheap, and an unbounded request rate would give that cost straight back.
const DEFAULT_SNAPSHOT_REQUEST_MIN_INTERVAL_MS: u64 = 1_000;

/// Identity of one Ready placement inside an instance's projection.
///
/// Depth is part of the key: the same block can legitimately be Ready at two
/// depths at once (a G2 copy that has also been written to G3), and collapsing
/// them would let a G3 eviction silently remove the G2 copy.
type PlacementKey = (SequenceHash, LogicalResourceId, u8, TierDepth);

/// Placements retained for one instance: the live Ready set, plus the removal
/// tombstones that keep a late `Ready` from resurrecting a copy the publisher
/// already said is gone.
///
/// # Why two maps rather than one `Ready | Removed` record
///
/// [`TierPlacementProjection::holders`] reads `ready` and cannot see
/// `tombstones` at all. With a single map carrying a `Removed` variant, a read
/// path that forgot one filter would answer with removed copies — precisely the
/// stale success this module exists to prevent. Splitting them makes that bug
/// unrepresentable rather than merely tested. It also keeps `ready.len()` the
/// honest live-placement count for `DeltaOutcome::Applied`.
///
/// # The invariant
///
/// **A key is in at most one of the two maps.** Ready wins ⇒ tombstone dropped,
/// record inserted. Remove wins ⇒ record dropped, tombstone raised. A Remove
/// that *loses* (an older generation than the record it met) writes nothing at
/// all.
///
/// The cost of breaking that last rule is retention, not ordering: a losing
/// Remove carries a generation below the record it met, so a tombstone it left
/// could only suppress a Ready that would have lost to that record anyway. What
/// it would do is count the key twice against `max_ready_per_instance` — and
/// that guard empties the projection rather than truncating it, so a publisher
/// emitting ordinary late invalidations would burn budget it never used and
/// blank a healthy projection.
///
/// # What bounds the tombstones
///
/// They share `max_ready_per_instance` with the live set, so the worst-case
/// retained state is unchanged and overflow is the same fail-safe (invalidate
/// and clear, never truncate). Every snapshot install drops them, and R7b §3
/// makes installs periodic. That is not merely convenient, it is correct: a
/// delta sealed before a snapshot carries the older generation and is discarded
/// as stale, and a delta sealed after it speaks for post-snapshot truth, so no
/// reordering can cross the install boundary and no tombstone needs to.
/// # Why the depth set exists
///
/// A reader knows the hash, the resource and the lane; it does **not** know the
/// depth, which is the fourth component of the key. Without something standing
/// in for it, answering one hash means iterating the whole ready map — up to
/// `max_ready_per_instance` (1 Mi) entries per instance, per single-hash lookup,
/// under the read lock. `depths` supplies the missing component, so a read is a
/// handful of hash lookups instead.
///
/// It is a monotone superset by design: `apply_remove` never withdraws a depth.
/// A stale member costs one wasted lookup that finds nothing, which cannot make
/// a read wrong, and it is bounded twice over — by the `u8` depth space, and by
/// the replace-all install that rebuilds it. Refcounting it exactly would buy
/// nothing and would add a counter that could drift out of step with the map it
/// describes.
#[derive(Debug, Default)]
struct PlacementSet {
    ready: HashMap<PlacementKey, ReadyRecord>,
    /// Highest generation a `Remove` has spoken for at a key whose Ready is
    /// gone.
    tombstones: HashMap<PlacementKey, u64>,
    /// Ascending superset of the depths the live records use.
    depths: BTreeSet<TierDepth>,
}

impl PlacementSet {
    /// Total records retained, live and tombstoned, for the capacity guard.
    fn retained(&self) -> usize {
        self.ready.len().saturating_add(self.tombstones.len())
    }

    fn clear(&mut self) {
        self.ready.clear();
        self.tombstones.clear();
        self.depths.clear();
    }

    /// Live records for one `(hash, resource, lane)`, shallowest depth first.
    fn at_scope(
        &self,
        hash: SequenceHash,
        scope: PlacementScope,
    ) -> impl Iterator<Item = (TierDepth, &ReadyRecord)> {
        self.depths.iter().filter_map(move |tier| {
            self.ready
                .get(&(hash, scope.resource, scope.lane, *tier))
                .map(|record| (*tier, record))
        })
    }

    /// Apply a `Ready` at `generation`, honouring both order rules.
    fn apply_ready(&mut self, key: PlacementKey, generation: u64, record: ReadyRecord) {
        // A `Remove` already spoke for this key at a generation this `Ready`
        // does not beat, so the copy is gone and a late `Ready` must not
        // resurrect it (R7b §7 test 4). Equality loses on purpose: `Remove`
        // applies `<=` in the other direction, so at one generation the removal
        // is the final word whichever order the two arrive in. The cost is that
        // a block evicted and re-offloaded at the *same* bundle generation stays
        // unadvertised until the next snapshot — a reuse loss deliberately
        // accepted under "temporary miss, never stale success", and one a
        // publisher avoids by bumping the generation when it re-offloads.
        if self
            .tombstones
            .get(&key)
            .is_some_and(|removed| *removed >= generation)
        {
            return;
        }
        // A late `Ready` cannot overwrite a newer copy either.
        if self
            .ready
            .get(&key)
            .is_some_and(|existing| existing.generation > generation)
        {
            return;
        }
        self.tombstones.remove(&key);
        self.depths.insert(key.3);
        self.ready.insert(key, record);
    }

    /// Apply a `Remove` at `generation`.
    fn apply_remove(&mut self, key: PlacementKey, generation: u64) {
        // A late invalidation cannot remove a newer copy (R7b §4).
        if self
            .ready
            .get(&key)
            .is_some_and(|existing| existing.generation > generation)
        {
            return;
        }
        self.ready.remove(&key);
        let tombstone = self.tombstones.entry(key).or_insert(generation);
        *tombstone = (*tombstone).max(generation);
    }
}

/// Capacity guards for the projection.
///
/// [`BundleDirectory`](super::super::bundle::BundleDirectory) is fastidious
/// about bounding what a publisher can make the hub retain; an unbounded
/// per-instance ready set here would be the soft spot in an otherwise bounded
/// surface. These are guards, not policy: overflow invalidates the projection
/// (empty answers) rather than evicting, because a partially-retained ready set
/// is precisely the stale-success failure this module exists to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionLimits {
    /// Maximum `(cache, instance)` entries across the whole projection. A
    /// backstop behind the registered-instance admission gate.
    pub max_instances: usize,
    /// Maximum placement records retained for one instance, live Ready records
    /// and removal tombstones together.
    pub max_ready_per_instance: usize,
    /// Maximum lineage manifests retained for one instance.
    pub max_manifests_per_instance: usize,
    /// Minimum interval between snapshot requests for one `(cache, instance)`.
    pub snapshot_request_min_interval_ms: u64,
}

impl Default for ProjectionLimits {
    fn default() -> Self {
        Self {
            max_instances: 4_096,
            max_ready_per_instance: 1 << 20,
            max_manifests_per_instance: 1 << 12,
            snapshot_request_min_interval_ms: DEFAULT_SNAPSHOT_REQUEST_MIN_INTERVAL_MS,
        }
    }
}

/// One instance's Ready placement for a key, as answered to a reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierPlacementHolder {
    /// Instance holding the copy.
    pub instance: InstanceId,
    /// Depth the copy is resident at.
    pub tier: TierDepth,
    /// Physical layout of the copy.
    pub placement: PhysicalPlacementMode,
    /// Bundle/lineage generation the copy was published at.
    pub generation: u64,
    /// When the hub observed this placement, for advisory age.
    pub observed_at_unix_ms: u64,
}

/// Why a projection stopped answering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidationReason {
    /// No projection existed for this `(cache, instance)` yet.
    NoProjection,
    /// The batch's epoch is not the one an authorized snapshot installed.
    UnknownEpoch,
    /// `seq != last_seq + 1` — a delta was lost, duplicated, or reordered.
    SequenceGap,
    /// The publisher has installed a snapshot the hub never received.
    GenerationAhead,
    /// A `ManifestInterval` referenced a manifest the hub does not hold, or one
    /// that does not cover the interval.
    UnresolvedInterval,
    /// Applying the batch would exceed a projection capacity guard.
    CapacityExceeded,
}

/// Why a batch was dropped without changing the projection's validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscardReason {
    /// The publisher is not in the hub's registered-instance set.
    UnregisteredInstance,
    /// `snapshot_generation` is older than the installed one.
    StaleGeneration,
    /// The projection is invalid and waiting for a snapshot.
    AwaitingSnapshot,
    /// The projection's state lock is poisoned.
    ///
    /// Distinct from [`Self::AwaitingSnapshot`] on purpose: that one is normal
    /// recovery back-pressure a healthy projection bumps on every delta while it
    /// waits, and burying an internal fault inside it would make the per-reason
    /// counters unable to answer the one question they exist for. The snapshot
    /// path already separates the two ([`TierPlacementProjectionError::Unavailable`],
    /// surfaced as a 503).
    Unavailable,
    /// The batch failed [`TierPlacementBatchV1::validate`].
    Rejected(TierPlacementRejection),
}

/// What applying a delta did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaOutcome {
    /// The batch applied in order; the projection still answers.
    Applied {
        /// Sequence number now installed.
        seq: u64,
        /// Ready placements the projection holds for this instance afterwards.
        ready: usize,
    },
    /// The projection stopped answering and a snapshot was (or would have been)
    /// requested.
    Invalidated(InvalidationReason),
    /// The batch changed nothing.
    Discarded(DiscardReason),
}

/// What installing a snapshot did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotInstall {
    /// State was replaced; deltas resume at `seq_floor + 1`.
    Installed {
        /// Generation now installed.
        installed_generation: u64,
        /// Sequence floor deltas resume above.
        seq_floor: u64,
    },
    /// A periodic push the hub already has (or has superseded). No state change.
    AlreadyCurrent {
        /// Generation the hub holds.
        installed_generation: u64,
    },
}

/// Why a snapshot install was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TierPlacementProjectionError {
    /// The snapshot body violates the wire contract.
    #[error("invalid tier placement snapshot: {0}")]
    Invalid(#[from] TierPlacementError),
    /// The snapshot claims an epoch other than the credential's.
    #[error("tier placement snapshot epoch does not match the authorized registration epoch")]
    EpochMismatch,
    /// Installing would exceed a capacity guard.
    #[error("tier placement snapshot exceeds the {what} limit of {limit}")]
    Capacity {
        /// Which guard was hit.
        what: &'static str,
        /// The guard's value.
        limit: usize,
    },
    /// The projection lock is poisoned.
    #[error("tier placement projection is unavailable")]
    Unavailable,
}

/// Per-reason drop/apply counters. R7b §8 makes these mandatory: the delta plane
/// is allowed to lose messages, so "how often, and why" is the only way to tell
/// a healthy advisory stream from a broken one.
#[derive(Debug, Default)]
pub struct TierPlacementCounters {
    /// Batches applied in order.
    pub applied_batches: AtomicU64,
    /// Placement ops applied.
    pub applied_ops: AtomicU64,
    /// Invalidations, by reason.
    pub invalidated_no_projection: AtomicU64,
    /// Invalidations caused by an epoch no authorized snapshot installed.
    pub invalidated_unknown_epoch: AtomicU64,
    /// Invalidations caused by a sequence gap.
    pub invalidated_sequence_gap: AtomicU64,
    /// Invalidations caused by a generation ahead of the installed one.
    pub invalidated_generation_ahead: AtomicU64,
    /// Invalidations caused by an unresolvable manifest interval.
    pub invalidated_unresolved_interval: AtomicU64,
    /// Invalidations caused by a capacity guard.
    pub invalidated_capacity: AtomicU64,
    /// Batches from instances that never registered the indexer feature.
    pub discarded_unregistered: AtomicU64,
    /// Batches carrying a generation older than the installed one.
    pub discarded_stale_generation: AtomicU64,
    /// Batches dropped while waiting for a snapshot.
    pub discarded_awaiting_snapshot: AtomicU64,
    /// Batches dropped because the projection's state lock is poisoned.
    pub discarded_unavailable: AtomicU64,
    /// Batches that failed `validate()` at the projection boundary.
    pub discarded_rejected: AtomicU64,
    /// Snapshot requests actually emitted.
    pub snapshot_requests: AtomicU64,
    /// Snapshot requests suppressed by the rate limiter.
    pub snapshot_requests_suppressed: AtomicU64,
    /// Snapshots installed (state replaced).
    pub snapshots_installed: AtomicU64,
    /// Snapshots accepted but already current (periodic push).
    pub snapshots_already_current: AtomicU64,
}

impl TierPlacementCounters {
    fn record_invalidation(&self, reason: InvalidationReason) {
        let counter = match reason {
            InvalidationReason::NoProjection => &self.invalidated_no_projection,
            InvalidationReason::UnknownEpoch => &self.invalidated_unknown_epoch,
            InvalidationReason::SequenceGap => &self.invalidated_sequence_gap,
            InvalidationReason::GenerationAhead => &self.invalidated_generation_ahead,
            InvalidationReason::UnresolvedInterval => &self.invalidated_unresolved_interval,
            InvalidationReason::CapacityExceeded => &self.invalidated_capacity,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn record_discard(&self, reason: DiscardReason) {
        let counter = match reason {
            DiscardReason::UnregisteredInstance => &self.discarded_unregistered,
            DiscardReason::StaleGeneration => &self.discarded_stale_generation,
            DiscardReason::AwaitingSnapshot => &self.discarded_awaiting_snapshot,
            DiscardReason::Unavailable => &self.discarded_unavailable,
            DiscardReason::Rejected(_) => &self.discarded_rejected,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReadyRecord {
    generation: u64,
    placement: PhysicalPlacementMode,
    observed_at_unix_ms: u64,
}

#[derive(Debug, Default)]
struct InstanceProjection {
    /// Written **only** by a credential-authorized snapshot install.
    ///
    /// The ZMQ delta plane carries no credential. If a delta could install an
    /// epoch, a forged batch could set a bogus one, make every genuine delta
    /// mismatch, and ping-pong the projection between invalid and installed
    /// while amplifying snapshot requests. Failure would stay fail-safe (empty
    /// answers), but the amplification would be real. With this rule the state
    /// machine is monotone with respect to *authenticated* input.
    installed_epoch: Option<RegistrationEpoch>,
    last_seq: u64,
    installed_generation: u64,
    valid: bool,
    manifests: HashMap<u64, BundleResourceLineage>,
    placements: PlacementSet,
    snapshot_requested_at_ms: Option<u64>,
}

impl InstanceProjection {
    /// Stop answering, keeping the placement maps.
    ///
    /// They are unreadable while invalid, and an install replaces them wholesale
    /// anyway, so freeing them here would only add allocator churn on a flapping
    /// link. Capacity overflow is the one case that clears, because there the
    /// size *is* the problem.
    fn invalidate(&mut self) {
        self.valid = false;
    }

    fn clear(&mut self) {
        self.valid = false;
        self.placements.clear();
        self.manifests.clear();
    }
}

/// Advisory placement projection for every publishing instance.
pub struct TierPlacementProjection {
    state: RwLock<HashMap<(CacheManifestId, InstanceId), InstanceProjection>>,
    /// Admission gate for entry *creation*: the registered-instance set
    /// `IndexerManager` already maintains through `on_register`/`on_unregister`.
    ///
    /// A numeric bound alone has a nastier failure mode than it looks: an
    /// unauthenticated flood of random instance ids fills the map, and then
    /// *genuine* instances cannot get an entry — a permanent empty answer rather
    /// than a temporary one. Gating creation on registration bounds the map by
    /// state the hub already authenticates.
    ///
    /// This gate bounds the *instance* axis only. The `cache` half of the key is
    /// publisher-chosen and unauthenticated, which is why a delta never creates
    /// an entry at all — see [`Self::apply_delta`].
    registered: Arc<RwLock<std::collections::HashSet<InstanceId>>>,
    /// Rate-limit stamps for snapshot requests triggered by a delta for a
    /// `(cache, instance)` the projection holds no entry for.
    ///
    /// Keyed by instance **alone**. A `(cache, instance)` stamp map would be the
    /// same unbounded growth the no-create rule closes, since `cache` is
    /// unauthenticated; the registered-instance set bounds this one, and
    /// [`Self::remove_instance`] prunes it. The lock is never held together with
    /// `state`.
    unknown_request_stamps: RwLock<HashMap<InstanceId, u64>>,
    requester: OnceLock<Arc<dyn SnapshotRequester>>,
    clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    limits: ProjectionLimits,
    counters: TierPlacementCounters,
    #[cfg(test)]
    install_before_swap: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl std::fmt::Debug for TierPlacementProjection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TierPlacementProjection")
            .field(
                "instances",
                &self.state.read().map(|state| state.len()).unwrap_or(0),
            )
            .field("limits", &self.limits)
            .finish()
    }
}

impl TierPlacementProjection {
    /// Build a projection sharing `registered` with its owning manager, with the
    /// default capacity guards.
    #[must_use]
    pub fn new(registered: Arc<RwLock<std::collections::HashSet<InstanceId>>>) -> Self {
        Self::with_limits(registered, ProjectionLimits::default())
    }

    /// Build a projection with explicit capacity guards.
    #[must_use]
    pub fn with_limits(
        registered: Arc<RwLock<std::collections::HashSet<InstanceId>>>,
        limits: ProjectionLimits,
    ) -> Self {
        Self::with_parts(
            registered,
            Arc::new(super::super::bundle::unix_time_ms),
            limits,
        )
    }

    fn with_parts(
        registered: Arc<RwLock<std::collections::HashSet<InstanceId>>>,
        clock: Arc<dyn Fn() -> u64 + Send + Sync>,
        limits: ProjectionLimits,
    ) -> Self {
        Self {
            state: RwLock::new(HashMap::new()),
            registered,
            unknown_request_stamps: RwLock::new(HashMap::new()),
            requester: OnceLock::new(),
            clock,
            limits,
            counters: TierPlacementCounters::default(),
            #[cfg(test)]
            install_before_swap: None,
        }
    }

    /// Install the transport-backed snapshot requester. Idempotent; returns
    /// `false` if one was already installed.
    pub fn set_requester(&self, requester: Arc<dyn SnapshotRequester>) -> bool {
        self.requester.set(requester).is_ok()
    }

    /// Per-reason counters.
    #[must_use]
    pub fn counters(&self) -> &TierPlacementCounters {
        &self.counters
    }

    /// Whether this `(cache, instance)` projection currently answers queries.
    #[must_use]
    pub fn is_valid(&self, cache: CacheManifestId, instance: InstanceId) -> bool {
        self.state
            .read()
            .ok()
            .and_then(|state| {
                state
                    .get(&(cache, instance))
                    .map(|projection| projection.valid)
            })
            .unwrap_or(false)
    }

    /// Instances holding `hash` Ready at `scope`, across every valid projection
    /// for `cache`.
    ///
    /// This is the read path CT-2a consumes. It filters on `valid` before it
    /// consults any ready map. This filter enforces empty answers after a
    /// detected failure. A reader cannot bypass it because no other public API
    /// exposes a ready record.
    ///
    /// Sorted by depth then instance so a caller's tie-break is deterministic.
    ///
    /// # Cost
    ///
    /// A few hash lookups per projection, via
    /// [`PlacementSet::at_scope`] — not a scan of any ready map. What remains
    /// linear is the number of `(cache, instance)` entries, because the outer map
    /// is keyed on the pair and a cache cannot narrow it. That factor is bounded
    /// by `max_instances` (4096) and is the fleet size, unlike the ready-set size
    /// it replaced, which was bounded only by `max_ready_per_instance` (1 Mi).
    /// If a deployment ever makes 4096 lookups per candidate block matter, the
    /// next step is a per-cache index, not a bigger scan.
    #[must_use]
    pub fn holders(
        &self,
        cache: CacheManifestId,
        scope: PlacementScope,
        hash: SequenceHash,
    ) -> Vec<TierPlacementHolder> {
        let Ok(state) = self.state.read() else {
            return Vec::new();
        };
        let mut holders: Vec<TierPlacementHolder> = state
            .iter()
            .filter(|((entry_cache, _), projection)| *entry_cache == cache && projection.valid)
            .flat_map(|((_, instance), projection)| {
                projection
                    .placements
                    .at_scope(hash, scope)
                    .map(move |(tier, record)| TierPlacementHolder {
                        instance: *instance,
                        tier,
                        placement: record.placement,
                        generation: record.generation,
                        observed_at_unix_ms: record.observed_at_unix_ms,
                    })
            })
            .collect();
        holders.sort_by(|left, right| {
            left.tier
                .cmp(&right.tier)
                .then_with(|| left.instance.as_u128().cmp(&right.instance.as_u128()))
        });
        holders
    }

    /// The shallowest depth at which `instance` holds `hash` Ready, or `None`
    /// when the projection is invalid or does not hold it.
    ///
    /// Answered from that instance's entry alone. Deriving it from
    /// [`Self::holders`] would run the whole cross-instance walk to answer a
    /// single-instance question, and the depth set already iterates ascending, so
    /// the first hit *is* the shallowest.
    #[must_use]
    pub fn ready_placement(
        &self,
        cache: CacheManifestId,
        instance: InstanceId,
        scope: PlacementScope,
        hash: SequenceHash,
    ) -> Option<TierPlacementHolder> {
        let state = self.state.read().ok()?;
        let projection = state.get(&(cache, instance))?;
        // Same `valid` gate as `holders`. After the projection detects a
        // continuity failure, no public path can expose its retained records.
        if !projection.valid {
            return None;
        }
        projection
            .placements
            .at_scope(hash, scope)
            .next()
            .map(|(tier, record)| TierPlacementHolder {
                instance,
                tier,
                placement: record.placement,
                generation: record.generation,
                observed_at_unix_ms: record.observed_at_unix_ms,
            })
    }

    /// Drop every projection for `instance`, across every cache.
    ///
    /// Parity with `PositionalIndex::remove_instance` and
    /// `BundleDirectory::remove_owner`: a deregistered publisher's advisory
    /// state is not merely stale, it is about a process that no longer exists.
    pub fn remove_instance(&self, instance: InstanceId) {
        if let Ok(mut state) = self.state.write() {
            state.retain(|(_, entry_instance), _| *entry_instance != instance);
        }
        // Pruned separately, and never under the `state` lock: the two are
        // independent maps and holding both would be the only place in this
        // module with a lock order to get wrong.
        if let Ok(mut stamps) = self.unknown_request_stamps.write() {
            stamps.remove(&instance);
        }
    }

    /// Apply one delta batch.
    ///
    /// Whole-batch: a batch either applies completely or changes nothing, so no
    /// reader can observe half of one. Key resolution therefore happens before
    /// any mutation.
    pub fn apply_delta(&self, batch: &TierPlacementBatchV1) -> DeltaOutcome {
        if let Err(error) = batch.validate() {
            // Defensive: ingest already rejects here and counts by bucket. A
            // rejected batch does *not* invalidate — the publisher's sequencer
            // does not consume a number for a batch it never sent, and a batch
            // the hub rejects is followed by a seq the hub reads as a gap and
            // recovers from. Two invalidation policies for one condition would
            // just be two ways to be wrong.
            return self.discard(DiscardReason::Rejected(error.rejection()));
        }
        if !self.is_registered(batch.instance_id) {
            // Visible rather than silent: a deployment publishing tier
            // placements without declaring `Feature::Indexer` loses everything,
            // and that should show up as a counter, not as an empty directory.
            return self.discard(DiscardReason::UnregisteredInstance);
        }

        let now = (self.clock)();
        let key = (batch.cache, batch.instance_id);
        let Ok(mut state) = self.state.write() else {
            return self.discard(DiscardReason::Unavailable);
        };

        // A delta never *creates* an entry. The registered-instance gate above
        // bounds the instance axis, but `cache` is the other half of the key and
        // is publisher-chosen on an unauthenticated plane: creating here would
        // let one registered instance id plus N forged cache ids fill
        // `max_instances` with entries that are inert by construction (only an
        // authorized install writes `installed_epoch`, so they can never become
        // valid) and that nothing ages out. Genuine caches would then be
        // permanently unable to get an entry — and the documented recovery path
        // would be blocked too, because `install_snapshot` refuses at the same
        // bound. So the map grows only through authorized installs, and its
        // single job on this path — carrying the request rate-limit stamp — moves
        // to an instance-keyed side map the registered set already bounds.
        let Some(projection) = state.get_mut(&key) else {
            drop(state);
            self.request_unknown_snapshot(batch.cache, batch.instance_id, now);
            self.counters
                .record_invalidation(InvalidationReason::NoProjection);
            return DeltaOutcome::Invalidated(InvalidationReason::NoProjection);
        };
        let outcome = Self::apply_to(projection, batch, now, &self.limits);
        match outcome {
            DeltaOutcome::Invalidated(reason) => {
                Self::request_snapshot(
                    &self.counters,
                    self.requester.get(),
                    projection,
                    batch.cache,
                    batch.instance_id,
                    now,
                    self.limits.snapshot_request_min_interval_ms,
                );
                self.counters.record_invalidation(reason);
            }
            DeltaOutcome::Discarded(DiscardReason::AwaitingSnapshot) => {
                // Re-ask, rate-limited: the first request may have been lost,
                // and without a retry the projection would wait for the
                // publisher's periodic push (up to a minute) even though it is
                // still receiving traffic from that publisher.
                Self::request_snapshot(
                    &self.counters,
                    self.requester.get(),
                    projection,
                    batch.cache,
                    batch.instance_id,
                    now,
                    self.limits.snapshot_request_min_interval_ms,
                );
                self.counters
                    .record_discard(DiscardReason::AwaitingSnapshot);
            }
            DeltaOutcome::Discarded(reason) => self.counters.record_discard(reason),
            DeltaOutcome::Applied { .. } => {
                self.counters
                    .applied_batches
                    .fetch_add(1, Ordering::Relaxed);
                self.counters
                    .applied_ops
                    .fetch_add(batch.ops.len() as u64, Ordering::Relaxed);
            }
        }
        outcome
    }

    /// Install a full snapshot for `(snapshot.cache, snapshot.instance_id)`.
    ///
    /// `authorized_epoch` is the epoch the *credential check* returned, not the
    /// one in the body — the body is data, the credential is authority. The two
    /// must agree, and disagreement is a 409 rather than a silent adoption.
    pub fn install_snapshot(
        &self,
        snapshot: &TierPlacementSnapshotV1,
        authorized_epoch: RegistrationEpoch,
    ) -> Result<SnapshotInstall, TierPlacementProjectionError> {
        snapshot.validate()?;
        if snapshot.registration_epoch != authorized_epoch {
            return Err(TierPlacementProjectionError::EpochMismatch);
        }
        if snapshot.manifests.len() > self.limits.max_manifests_per_instance {
            return Err(TierPlacementProjectionError::Capacity {
                what: "manifests per instance",
                limit: self.limits.max_manifests_per_instance,
            });
        }

        // Build the replacement outside the lock. The install then becomes a
        // pair of moves under the write lock, so a concurrent delta or reader
        // sees exactly the pre-install or exactly the post-install state — never
        // a body half-decoded into live state.
        let now = (self.clock)();
        let mut manifests = HashMap::with_capacity(snapshot.manifests.len());
        for manifest in &snapshot.manifests {
            manifests.insert(manifest.manifest_id, manifest.lineage()?);
        }
        let mut ready: HashMap<PlacementKey, ReadyRecord> = HashMap::new();
        let mut depths: BTreeSet<TierDepth> = BTreeSet::new();
        for entry in &snapshot.entries {
            let KeyRange::Hashes(hashes) = &entry.keys else {
                // Unreachable after `validate()`; kept as a hard floor rather
                // than an `expect`, because "snapshots are exact" is the premise
                // the whole recovery story rests on.
                return Err(TierPlacementProjectionError::Invalid(
                    TierPlacementError::InexactSnapshotKeys { index: 0 },
                ));
            };
            for hash in hashes {
                if ready.len() >= self.limits.max_ready_per_instance {
                    return Err(TierPlacementProjectionError::Capacity {
                        what: "ready placements per instance",
                        limit: self.limits.max_ready_per_instance,
                    });
                }
                ready.insert(
                    (*hash, entry.scope.resource, entry.scope.lane, entry.tier),
                    ReadyRecord {
                        generation: entry.generation,
                        placement: entry.placement,
                        observed_at_unix_ms: now,
                    },
                );
                depths.insert(entry.tier);
            }
        }

        #[cfg(test)]
        if let Some(hook) = &self.install_before_swap {
            hook();
        }

        let mut state = self
            .state
            .write()
            .map_err(|_| TierPlacementProjectionError::Unavailable)?;
        let key = (snapshot.cache, snapshot.instance_id);
        if !state.contains_key(&key) && state.len() >= self.limits.max_instances {
            return Err(TierPlacementProjectionError::Capacity {
                what: "projected instances",
                limit: self.limits.max_instances,
            });
        }
        let projection = state.entry(key).or_default();

        // Epoch first, generation second. A publisher restart mints a new epoch
        // and restarts its generation counter at 1; comparing generations first
        // would reject that snapshot as stale against an installed generation of
        // 7 and strand the projection invalid forever — the exact permanent
        // failure R7b §4 forbids.
        let same_epoch = projection.installed_epoch == Some(authorized_epoch);
        if same_epoch
            && (snapshot.snapshot_generation < projection.installed_generation
                || (snapshot.snapshot_generation == projection.installed_generation
                    && projection.valid))
        {
            self.counters
                .snapshots_already_current
                .fetch_add(1, Ordering::Relaxed);
            return Ok(SnapshotInstall::AlreadyCurrent {
                installed_generation: projection.installed_generation,
            });
        }

        projection.manifests = manifests;
        // Replace-all, tombstones included: a delta sealed before this snapshot
        // carries the older generation and is discarded as stale, and one sealed
        // after it speaks for post-snapshot truth, so no reordering can cross
        // the install boundary for a tombstone to guard against.
        projection.placements = PlacementSet {
            ready,
            tombstones: HashMap::new(),
            depths,
        };
        projection.installed_generation = snapshot.snapshot_generation;
        projection.last_seq = snapshot.seq_floor;
        projection.installed_epoch = Some(authorized_epoch);
        projection.valid = true;
        // Clearing the rate-limit stamp matters: a second gap inside the request
        // window would otherwise be unable to ask, and a lost snapshot after that
        // would strand the projection until the publisher's periodic push.
        projection.snapshot_requested_at_ms = None;
        self.counters
            .snapshots_installed
            .fetch_add(1, Ordering::Relaxed);
        Ok(SnapshotInstall::Installed {
            installed_generation: snapshot.snapshot_generation,
            seq_floor: snapshot.seq_floor,
        })
    }

    fn is_registered(&self, instance: InstanceId) -> bool {
        self.registered
            .read()
            .map(|set| set.contains(&instance))
            .unwrap_or(false)
    }

    fn discard(&self, reason: DiscardReason) -> DeltaOutcome {
        self.counters.record_discard(reason);
        DeltaOutcome::Discarded(reason)
    }

    /// Ask for a snapshot for a `(cache, instance)` the projection holds no
    /// entry for.
    ///
    /// Must be called with the `state` lock released — see
    /// [`Self::unknown_request_stamps`].
    fn request_unknown_snapshot(&self, cache: CacheManifestId, instance: InstanceId, now: u64) {
        let Ok(mut stamps) = self.unknown_request_stamps.write() else {
            return;
        };
        if let Some(last) = stamps.get(&instance)
            && now.saturating_sub(*last) < self.limits.snapshot_request_min_interval_ms
        {
            self.counters
                .snapshot_requests_suppressed
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        stamps.insert(instance, now);
        drop(stamps);
        self.counters
            .snapshot_requests
            .fetch_add(1, Ordering::Relaxed);
        match self.requester.get() {
            Some(requester) => requester.request(cache, instance),
            None => UnavailableSnapshotRequester.request(cache, instance),
        }
    }

    fn request_snapshot(
        counters: &TierPlacementCounters,
        requester: Option<&Arc<dyn SnapshotRequester>>,
        projection: &mut InstanceProjection,
        cache: CacheManifestId,
        instance: InstanceId,
        now: u64,
        min_interval_ms: u64,
    ) {
        if let Some(last) = projection.snapshot_requested_at_ms
            && now.saturating_sub(last) < min_interval_ms
        {
            counters
                .snapshot_requests_suppressed
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        projection.snapshot_requested_at_ms = Some(now);
        counters.snapshot_requests.fetch_add(1, Ordering::Relaxed);
        match requester {
            Some(requester) => requester.request(cache, instance),
            None => UnavailableSnapshotRequester.request(cache, instance),
        }
    }

    /// The transition table, applied to an existing entry.
    fn apply_to(
        projection: &mut InstanceProjection,
        batch: &TierPlacementBatchV1,
        now: u64,
        limits: &ProjectionLimits,
    ) -> DeltaOutcome {
        if projection.installed_epoch != Some(batch.registration_epoch) {
            // Covers both "never installed" and "publisher restarted". Clears
            // rather than merely invalidating: state from a different epoch is
            // about a different process lifetime, so retaining it has no value
            // even as a cache.
            projection.clear();
            return DeltaOutcome::Invalidated(InvalidationReason::UnknownEpoch);
        }
        if !projection.valid {
            return DeltaOutcome::Discarded(DiscardReason::AwaitingSnapshot);
        }
        if batch.snapshot_generation < projection.installed_generation {
            return DeltaOutcome::Discarded(DiscardReason::StaleGeneration);
        }
        if batch.snapshot_generation > projection.installed_generation {
            projection.invalidate();
            return DeltaOutcome::Invalidated(InvalidationReason::GenerationAhead);
        }
        // `checked_add`, not `saturating_add`: at the u64 ceiling a saturating
        // successor equals `last_seq` itself, so a replayed batch would be
        // accepted as the next one — an overflow silently converted into a
        // duplicate application instead of the gap it is. Unreachable in
        // practice (it needs 2^64 sealed batches from one instance/epoch), but
        // the arithmetic should not be the thing standing between here and a
        // duplicate.
        if projection.last_seq.checked_add(1) != Some(batch.seq) {
            projection.invalidate();
            return DeltaOutcome::Invalidated(InvalidationReason::SequenceGap);
        }

        // Phase 1: resolve every op's keys. An unresolvable interval is a
        // sequence gap by R7b §4 — never an inferred membership, and never a
        // partially applied batch.
        let mut resolved: Vec<(&TierPlacementOp, Vec<SequenceHash>)> =
            Vec::with_capacity(batch.ops.len());
        for op in &batch.ops {
            let Some(keys) = resolve_keys(&projection.manifests, op) else {
                projection.invalidate();
                return DeltaOutcome::Invalidated(InvalidationReason::UnresolvedInterval);
            };
            resolved.push((op, keys));
        }

        // Phase 2: apply.
        for (op, keys) in resolved {
            match op {
                TierPlacementOp::Ready {
                    scope,
                    tier,
                    placement,
                    generation,
                    ..
                } => {
                    for hash in keys {
                        projection.placements.apply_ready(
                            (hash, scope.resource, scope.lane, *tier),
                            *generation,
                            ReadyRecord {
                                generation: *generation,
                                placement: *placement,
                                observed_at_unix_ms: now,
                            },
                        );
                    }
                }
                TierPlacementOp::Remove {
                    scope,
                    tier,
                    generation,
                    ..
                } => {
                    for hash in keys {
                        projection
                            .placements
                            .apply_remove((hash, scope.resource, scope.lane, *tier), *generation);
                    }
                }
            }
        }

        if projection.placements.retained() > limits.max_ready_per_instance {
            // The size *is* the problem here, so this is the one invalidation
            // that frees the map.
            projection.clear();
            return DeltaOutcome::Invalidated(InvalidationReason::CapacityExceeded);
        }

        projection.last_seq = batch.seq;
        DeltaOutcome::Applied {
            seq: batch.seq,
            ready: projection.placements.ready.len(),
        }
    }

    #[cfg(test)]
    pub(super) fn for_test(
        registered: Arc<RwLock<std::collections::HashSet<InstanceId>>>,
        clock: Arc<dyn Fn() -> u64 + Send + Sync>,
        limits: ProjectionLimits,
    ) -> Self {
        Self::with_parts(registered, clock, limits)
    }

    /// Poison the state lock, so the tests can assert what a caller sees when it
    /// is. There is no other way to reach the fault arms, and "an internal fault
    /// is counted as an internal fault" is exactly the property worth pinning.
    #[cfg(test)]
    pub(super) fn poison_state_for_test(&self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = self.state.write().expect("not yet poisoned");
            panic!("poisoning the projection state lock");
        }));
        assert!(self.state.write().is_err(), "lock should be poisoned");
    }

    #[cfg(test)]
    pub(super) fn with_install_hook(
        registered: Arc<RwLock<std::collections::HashSet<InstanceId>>>,
        hook: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        let mut projection = Self::with_parts(
            registered,
            Arc::new(super::super::bundle::unix_time_ms),
            ProjectionLimits::default(),
        );
        projection.install_before_swap = Some(hook);
        projection
    }
}

/// Resolve an op's [`KeyRange`] to exact hashes, or `None` when the manifest is
/// missing or does not cover the interval.
///
/// Missing manifest ⇒ `None` ⇒ sequence gap. This is the load-bearing half of
/// the 2026-08-05 correction: a positional lineage hash carries its own hash and
/// **one** parent fragment, so nothing here could reconstruct membership from a
/// terminal hash even if it wanted to. Guessing would be a stale success.
fn resolve_keys(
    manifests: &HashMap<u64, BundleResourceLineage>,
    op: &TierPlacementOp,
) -> Option<Vec<SequenceHash>> {
    match op.keys() {
        KeyRange::Hashes(hashes) => Some(hashes.clone()),
        KeyRange::ManifestInterval {
            manifest_id,
            start,
            len,
        } => {
            let lineage = manifests.get(manifest_id)?;
            // A manifest for a different resource is not a coordinate system
            // this op can be read in, so it is a gap rather than a coincidence.
            if lineage.resource() != op.scope().resource {
                return None;
            }
            let start = *start as usize;
            let end = start.checked_add(*len as usize)?;
            lineage.hashes().get(start..end).map(<[_]>::to_vec)
        }
    }
}
