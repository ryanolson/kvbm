// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Feature-owned wire protocol for the KV indexer.
//!
//! All paths are **relative** — the server nests them under
//! `/v1/features/{ROUTE_PREFIX}` (see
//! [`FeatureManager::route_prefix`](crate::features::FeatureManager::route_prefix)).
//! Nothing here lives in the central [`crate::protocol::paths`]; the feature
//! owns its whole namespace.

use kvbm_common::LogicalResourceId;
use kvbm_logical::SequenceHash;
use kvbm_protocols::cache_manifest::{
    BundleKey, BundleResourceLineage, CacheManifestId, RegistrationEpoch, ResourceRequirement,
};
use kvbm_protocols::tier_protocol::{PhysicalPlacementMode, TierDepth, TierPlacementSnapshotV1};
use serde::{Deserialize, Serialize};
use velo_ext::InstanceId;

use crate::protocol::MutationCredential;

/// URL segment the server nests this feature's routers under
/// (`/v1/features/indexer/...`).
pub const ROUTE_PREFIX: &str = "indexer";

/// Velo active-message handler name for the client → hub block lookup. The hub
/// installs this handler on its own velo in
/// [`IndexerManager::attach`](super::manager::IndexerManager); a
/// [`IndexerLookupClient`](super::client::IndexerLookupClient) calls it. Follows
/// the `kvbm_hub_*` convention shared with the heartbeat handler.
pub const QUERY_HANDLER: &str = "kvbm_hub_indexer_query";
pub const BUNDLE_PUBLISH_HANDLER: &str = "kvbm_hub_bundle_publish";
pub const BUNDLE_INVALIDATE_HANDLER: &str = "kvbm_hub_bundle_invalidate";
pub const BUNDLE_QUERY_HANDLER: &str = "kvbm_hub_bundle_query";

/// Relative route paths (mounted under `/v1/features/indexer`).
pub mod paths {
    /// `GET /config` — indexer configuration + ZMQ ingest endpoint. A `200`
    /// also serves as the capability probe used by connectors.
    pub const CONFIG: &str = "/config";

    /// `GET /instances` — the set of instances that declared `Feature::Indexer`
    /// at registration. Not every registered instance necessarily emits KV
    /// events, so this is the *registered* (participating) set.
    pub const INSTANCES: &str = "/instances";

    /// `GET /hashes/by_position/{pos}` — dump the index bucket at `pos`.
    pub const BY_POSITION: &str = "/hashes/by_position/{pos}";

    /// `POST /query` — resolve a block-hash sequence to the holding instances.
    pub const QUERY: &str = "/query";

    /// `POST /tier-placements/snapshot` — install a publisher's full
    /// tier-placement state (R7b §3).
    ///
    /// **Control plane only.** It is a mutation, so it is mounted by
    /// `control_router` and deliberately absent from the read-only discovery
    /// port. R7b §3 writes the path as `/v1/tier-placements/snapshot`; feature
    /// routers declare relative paths and own their namespace, so the effective
    /// path is `/v1/features/indexer/tier-placements/snapshot`.
    pub const TIER_PLACEMENT_SNAPSHOT: &str = "/tier-placements/snapshot";
}

/// One Ready placement asserted by a bundle advertisement (R7b §5).
///
/// This is the *owner-asserted, credential-authenticated* view of placement, as
/// distinct from the advisory ZMQ projection. Query rows are derived from these,
/// not from the projection, so a directory read never has to take a second lock
/// or reconcile two sources mid-answer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadyPlacement {
    /// Logical resource the copy covers.
    pub resource: LogicalResourceId,
    /// Execution lane (ADP); 0 for unitary deployments.
    pub lane: u8,
    /// Depth the copy is resident at.
    pub tier: TierDepth,
    /// Physical layout of the copy.
    pub placement: PhysicalPlacementMode,
}

/// Response for `GET /config`. Doubles as the capability probe: a successful
/// `200` tells a connector the indexer is present and where to publish.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexerConfigResponse {
    /// Maximum sequence length (tokens) the index is sized for.
    pub max_seq_len: usize,
    /// Block size (tokens per block). Must match the publisher's page size.
    pub block_size: usize,
    /// Number of position buckets (`max_seq_len / block_size`).
    pub num_positions: usize,
    /// ZMQ endpoint a publisher connects its `PUB` socket to
    /// (e.g. `tcp://127.0.0.1:54231`). Empty when ingest is not yet bound.
    pub zmq_endpoint: String,
}

/// Response for `GET /instances`. The set of instances that declared
/// `Feature::Indexer` at registration, as decimal `u128` strings (matching the
/// holder ids in [`IndexEntry::instances`]). Lets an operator distinguish
/// "registered to participate" from "actually holding indexed blocks".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct InstancesResponse {
    /// Registered (participating) instance ids, decimal `u128`, sorted.
    pub instances: Vec<String>,
}

/// One indexed block: a positional-lineage hash and the instances holding it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexEntry {
    /// Human-readable PLH (`position:current[:parent]`, base58).
    pub hash: String,
    /// Raw 128-bit PLH as a decimal string (jq-safe; avoids JSON number
    /// precision loss for values above 2^53).
    pub hash_u128: String,
    /// Block position decoded from the PLH.
    pub position: u64,
    /// Instance ids (decimal `u128` strings) currently holding this block.
    pub instances: Vec<String>,
}

/// Response for `GET /hashes/by_position/{pos}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ByPositionResponse {
    /// The queried position.
    pub position: usize,
    /// Entries indexed at that position.
    pub entries: Vec<IndexEntry>,
}

/// Request body for `POST /query`.
///
/// `hashes` are the block-sequence PLHs in position order (low → high). The
/// indexer walks them high → low and returns the deepest one present.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueryRequest {
    /// Positional-lineage hashes of the candidate block sequence.
    pub hashes: Vec<SequenceHash>,
}

/// Response body for `POST /query`. `hit` is `None` when no supplied hash is
/// indexed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueryResponse {
    /// The deepest matching block, or `None` if nothing matched.
    pub hit: Option<IndexEntry>,
}

/// Typed result of the velo [`QUERY_HANDLER`] lookup.
///
/// Unlike the HTTP [`IndexEntry`] (which stringifies ids for jq-safety), this
/// stays typed end-to-end over the velo plane: the matched [`SequenceHash`] and
/// the holder [`InstanceId`]s feed straight into peer discovery. Paired so the
/// matched hash and its holders cannot drift — a full miss is the `None` arm of
/// the `Option<FindBlocksHit>` the handler and
/// [`IndexerLookupClient`](super::client::IndexerLookupClient) return.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FindBlocksHit {
    /// Deepest candidate hash present in the index.
    pub matched: SequenceHash,
    /// Instances currently holding `matched`. Always non-empty.
    pub candidates: Vec<InstanceId>,
}

/// One complete-bundle owner record stored by the hub directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleAdvertisementRecord {
    pub key: BundleKey,
    pub generation: u64,
    pub owner: InstanceId,
    /// Hub-minted lifecycle identity of `owner`. The wire field remains
    /// optional for decode compatibility, but new hub publications reject
    /// missing or mismatched values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registration_epoch: Option<RegistrationEpoch>,
    pub requirements: Vec<ResourceRequirement>,
    pub lineages: Vec<BundleResourceLineage>,
    /// Publisher-requested expiry. The hub clamps this to its server-owned
    /// maximum advertisement lifetime before storing or leasing the record.
    pub expires_at_unix_ms: u64,
    /// Where this bundle is Ready, per resource/lane (R7b §5). Empty means the
    /// publisher predates the field — read it as "unknown", never as "nowhere".
    #[serde(default)]
    pub placements: Vec<ReadyPlacement>,
    /// Honest holder-estimated cost to make the bundle transferable: 0 when it
    /// is already Ready off-device, nonzero when a stage is required.
    ///
    /// R7b §5 says a G1-only advertisement MUST set this. It is enforced as
    /// advisory in this change, because the field is `#[serde(default)]` for
    /// decode compatibility and a hard rejection would break every publisher
    /// that predates it — the same rollout shape `registration_epoch` took, and
    /// for the same reason. `None` means unknown; it does **not** mean zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_cost_hint_us: Option<u64>,
    /// When the hub accepted this advertisement, for advisory-age computations.
    ///
    /// Hub-stamped at publish, never publisher-supplied: it is a freshness
    /// signal, and a publisher-supplied timestamp is both spoofable and subject
    /// to clock skew. `expires_at_unix_ms` cannot substitute — the hub *clamps*
    /// it, so it says nothing about when the record arrived.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advertised_at_unix_ms: Option<u64>,
}

impl BundleAdvertisementRecord {
    /// Depth a peer would have to reach to obtain the **whole** bundle, if the
    /// advertisement claims every resource the bundle requires.
    ///
    /// # Why not simply the shallowest claimed depth
    ///
    /// A bundle is only usable with *all* of its required resources, so the
    /// depth that governs a puller's cost is the deepest one it must reach, not
    /// the shallowest one it may. Reporting the minimum across all placements
    /// lets a record whose `PrefixHistory` sits at depth 1 and whose other
    /// required resource sits at depth 3 advertise `ready_tier = 1`, understating
    /// the stage cost by two tiers to the CT-2a consumer that reads this as a
    /// cost hint. So: per required resource, take the shallowest publishable
    /// depth it is claimed at; the answer is the deepest of those.
    ///
    /// Depth 0 (G1) is filtered throughout: it is not a remotely-Ready
    /// placement, and R7b §1 rule 3 keeps it off the placement stream entirely.
    ///
    /// # `None` is overloaded, and deliberately so
    ///
    /// It means "no publishable depth can be stated for the whole bundle" —
    /// which covers both a publisher that predates the field (empty
    /// `placements`, R7b §5's "unknown, never nowhere") and one that claims some
    /// resources but not all. Distinguishing them would require the consumer to
    /// act on a partial claim, and the only safe action on a partial claim is the
    /// same as on no claim: do not assume a depth.
    #[must_use]
    pub fn ready_tier(&self) -> Option<TierDepth> {
        let mut deepest: Option<TierDepth> = None;
        for requirement in &self.requirements {
            let resource = requirement.resource();
            let shallowest = self
                .placements
                .iter()
                .filter(|placement| placement.resource == resource)
                .map(|placement| placement.tier)
                .filter(|tier| tier.is_publishable())
                .min()?;
            deepest =
                Some(deepest.map_or(shallowest, |current: TierDepth| current.max(shallowest)));
        }
        deepest
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundlePublishRequest {
    pub credential: MutationCredential,
    pub advertisement: BundleAdvertisementRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleInvalidateRequest {
    pub credential: MutationCredential,
    pub key: BundleKey,
    pub generation: u64,
    pub owner: InstanceId,
    /// Publisher-requested replay-protection horizon. For absent keys the hub
    /// clamps this to a server-owned duration and enforces bounded admission.
    pub retain_until_unix_ms: u64,
}

/// Owner-scoped invalidation supplied by a connector. The indexer client adds
/// its registration credential when constructing the wire request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleInvalidationRecord {
    pub key: BundleKey,
    pub generation: u64,
    pub owner: InstanceId,
    /// Requested replay-protection horizon; the hub treats this as an
    /// untrusted hint and applies its own retention bound.
    pub retain_until_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleQueryRequest {
    pub manifest: CacheManifestId,
    pub requirements: Vec<ResourceRequirement>,
    pub candidates: Vec<BundleKey>,
    pub now_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleQueryHit {
    pub advertisement: BundleAdvertisementRecord,
    pub lease_id: uuid::Uuid,
    pub lease_expires_at_unix_ms: u64,
    /// Shallowest publishable depth the winning owner claims (R7b §5).
    ///
    /// **Echoed from the advertisement, not joined against the advisory
    /// projection.** The advertisement is the authenticated, owner-asserted
    /// source; consulting the ZMQ projection here would put a second lock inside
    /// the bundle query path and open the torn-read surface the transactional
    /// snapshot install exists to close.
    #[serde(default)]
    pub ready_tier: Option<TierDepth>,
    /// When the hub accepted the winning advertisement, for FleetAdvisory age.
    #[serde(default)]
    pub advertised_at_unix_ms: Option<u64>,
}

/// Why no complete bundle could satisfy a directory query.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BundleQueryMissReason {
    NotFound,
    Incompatible,
    Incomplete,
    Expired,
}

/// Complete-bundle directory response, preserving actionable miss telemetry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)] // keep the public wire model direct and allocation-free on hits.
#[serde(rename_all = "snake_case")]
pub enum BundleQueryOutcome {
    Hit(BundleQueryHit),
    Miss(BundleQueryMissReason),
}

/// Body of `POST /v1/features/indexer/tier-placements/snapshot`.
///
/// The credential travels in the body, matching this feature's existing
/// [`BundlePublishRequest`] / [`BundleInvalidateRequest`] pattern. The
/// header-based path (`MUTATION_CREDENTIAL_HEADER` +
/// `HubServerState::credentials`) is not reachable from a feature router:
/// feature routes are built `.with_state(Arc<IndexerManager>)` and never see
/// `HubServerState`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TierPlacementSnapshotRequest {
    /// Owner's registration credential.
    pub credential: MutationCredential,
    /// Full placement state to install.
    pub snapshot: TierPlacementSnapshotV1,
}

/// Result of a snapshot install.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TierPlacementSnapshotResponse {
    /// `false` when the hub already held this generation — the expected answer
    /// to a periodic push that arrived while nothing had been lost, not an
    /// error.
    pub installed: bool,
    /// Generation the hub holds after the call.
    pub installed_generation: u64,
    /// Sequence floor deltas resume above.
    pub seq_floor: u64,
}

#[cfg(test)]
mod tests {
    use kvbm_protocols::cache_manifest::{BundleKey, ResourceRequirement, ResourceRole};

    use super::*;

    /// Field names R7b §5 added. Naming them here pins the wire contract: a
    /// rename breaks this test rather than silently defaulting on every
    /// deployed publisher.
    const ADDED_RECORD_FIELDS: [&str; 3] =
        ["placements", "stage_cost_hint_us", "advertised_at_unix_ms"];
    const ADDED_HIT_FIELDS: [&str; 2] = ["ready_tier", "advertised_at_unix_ms"];

    fn record() -> BundleAdvertisementRecord {
        let resource = LogicalResourceId(1);
        let hashes = vec![SequenceHash::root(1), SequenceHash::root(1).extend(2)];
        BundleAdvertisementRecord {
            key: BundleKey::from_parts(CacheManifestId::from_bytes([41; 32]), hashes[1], 8)
                .unwrap(),
            generation: 3,
            owner: InstanceId::new_v4(),
            registration_epoch: Some(RegistrationEpoch::new()),
            requirements: vec![
                ResourceRequirement::new(resource, ResourceRole::PrefixHistory, 4).unwrap(),
            ],
            lineages: vec![BundleResourceLineage::new(resource, hashes).unwrap()],
            expires_at_unix_ms: 10_000,
            placements: vec![ReadyPlacement {
                resource,
                lane: 0,
                tier: TierDepth(2),
                placement: PhysicalPlacementMode::TpShards { count: 4 },
            }],
            stage_cost_hint_us: Some(250),
            advertised_at_unix_ms: Some(5_000),
        }
    }

    /// Strip the added keys to reconstruct the pre-R7b wire form.
    ///
    /// Deliberately derived rather than frozen as a literal: a hand-pinned blob
    /// would embed `BundleKey`/`BundleResourceLineage`/`RegistrationEpoch`
    /// encodings, so an unrelated upstream change to any of those would fail
    /// here and read as an R7b compatibility regression. What this change can
    /// actually break is the *added fields*, and that is what is pinned —
    /// `ADDED_*_FIELDS` names them, and the assertions below prove both
    /// directions across exactly that delta.
    fn strip(value: &serde_json::Value, fields: &[&str]) -> serde_json::Value {
        let mut value = value.clone();
        let map = value.as_object_mut().expect("record encodes as a JSON map");
        for field in fields {
            assert!(
                map.remove(*field).is_some(),
                "{field} must be present before stripping"
            );
        }
        value
    }

    #[test]
    fn old_advertisement_bytes_still_decode_into_the_widened_record() {
        let new = serde_json::to_value(record()).unwrap();
        let old = strip(&new, &ADDED_RECORD_FIELDS);
        let decoded: BundleAdvertisementRecord = serde_json::from_value(old).unwrap();
        assert!(decoded.placements.is_empty());
        assert_eq!(decoded.stage_cost_hint_us, None);
        assert_eq!(decoded.advertised_at_unix_ms, None);
        // Everything a pre-R7b publisher did send survives untouched.
        assert_eq!(decoded.key, record().key);
        assert_eq!(decoded.generation, 3);
        assert_eq!(decoded.expires_at_unix_ms, 10_000);
    }

    #[test]
    fn new_advertisement_bytes_decode_on_the_old_struct_shape() {
        // The codec that matters is `serde_json` — velo's typed handlers use it,
        // so these types are map-encoded and unknown fields are ignored by
        // default. Asserted rather than assumed: the tier delta envelope is
        // positional msgpack and does *not* have this property, which is why it
        // is frozen and grows through `v` instead.
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct OldRecord {
            key: BundleKey,
            generation: u64,
            owner: InstanceId,
            #[serde(default)]
            registration_epoch: Option<RegistrationEpoch>,
            requirements: Vec<ResourceRequirement>,
            lineages: Vec<BundleResourceLineage>,
            expires_at_unix_ms: u64,
        }

        let new = serde_json::to_value(record()).unwrap();
        let decoded: OldRecord = serde_json::from_value(new).unwrap();
        assert_eq!(decoded.generation, 3);
        assert_eq!(decoded.expires_at_unix_ms, 10_000);
    }

    #[test]
    fn query_hit_row_extensions_decode_in_both_directions() {
        let hit = BundleQueryHit {
            advertisement: record(),
            lease_id: uuid::Uuid::new_v4(),
            lease_expires_at_unix_ms: 2_000,
            ready_tier: Some(TierDepth(2)),
            advertised_at_unix_ms: Some(5_000),
        };
        let new = serde_json::to_value(&hit).unwrap();
        let old = strip(&new, &ADDED_HIT_FIELDS);
        let decoded: BundleQueryHit = serde_json::from_value(old).unwrap();
        assert_eq!(decoded.ready_tier, None);
        assert_eq!(decoded.advertised_at_unix_ms, None);
        assert_eq!(decoded.lease_expires_at_unix_ms, 2_000);

        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct OldHit {
            advertisement: BundleAdvertisementRecord,
            lease_id: uuid::Uuid,
            lease_expires_at_unix_ms: u64,
        }
        let decoded: OldHit = serde_json::from_value(new).unwrap();
        assert_eq!(decoded.lease_expires_at_unix_ms, 2_000);
    }

    /// `ResourceRequirement`'s own contract is "every listed resource is
    /// mandatory", so the depth that governs a puller's cost is the deepest one
    /// it must reach for the whole bundle, not the shallowest one some resource
    /// happens to sit at.
    #[test]
    fn ready_tier_is_the_deepest_depth_the_whole_bundle_requires() {
        let history = LogicalResourceId(1);
        let capsule = LogicalResourceId(2);
        let mut advertisement = record();
        advertisement.requirements = vec![
            ResourceRequirement::new(history, ResourceRole::PrefixHistory, 4).unwrap(),
            ResourceRequirement::new(capsule, ResourceRole::BoundaryCapsule, 4).unwrap(),
        ];
        advertisement.placements = vec![
            ReadyPlacement {
                resource: history,
                lane: 0,
                // G1 is not a remotely-Ready placement and R7b §1 rule 3 keeps
                // it off the stream, so it must not win any comparison.
                tier: TierDepth::G1,
                placement: PhysicalPlacementMode::Whole,
            },
            ReadyPlacement {
                resource: history,
                lane: 0,
                tier: TierDepth(1),
                placement: PhysicalPlacementMode::Whole,
            },
            ReadyPlacement {
                resource: capsule,
                lane: 0,
                tier: TierDepth(3),
                placement: PhysicalPlacementMode::Whole,
            },
        ];
        assert_eq!(
            advertisement.ready_tier(),
            Some(TierDepth(3)),
            "the shallowest depth alone would understate the stage cost by two tiers"
        );

        // Per resource it is still the shallowest: a second, deeper copy of the
        // capsule does not make the bundle costlier than its cheapest complete
        // set.
        advertisement.placements.push(ReadyPlacement {
            resource: capsule,
            lane: 0,
            tier: TierDepth(2),
            placement: PhysicalPlacementMode::Whole,
        });
        assert_eq!(advertisement.ready_tier(), Some(TierDepth(2)));

        // A required resource with no publishable placement makes the whole
        // answer unstatable, not "as good as the resources that are claimed".
        advertisement
            .placements
            .retain(|placement| placement.resource != capsule);
        assert_eq!(advertisement.ready_tier(), None);

        advertisement.placements = vec![ReadyPlacement {
            resource: history,
            lane: 0,
            tier: TierDepth::G1,
            placement: PhysicalPlacementMode::Whole,
        }];
        assert_eq!(
            advertisement.ready_tier(),
            None,
            "a G1-only advertisement is not ready at any pullable tier"
        );

        // Empty is the pre-R7b publisher: "unknown", which reads out the same
        // way as incomplete — the only safe action on either is to assume no
        // depth.
        advertisement.placements.clear();
        assert_eq!(advertisement.ready_tier(), None);
    }
}
