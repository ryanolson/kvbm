// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the KV indexer feature.
//!
//! Owns a [`PositionalIndex`], binds the ZMQ ingest socket during
//! [`FeatureManager::attach`], and exports its own HTTP surface under
//! `/v1/features/indexer` (the server nests it via
//! [`FeatureManager::route_prefix`]).

use std::collections::HashSet;
use std::sync::{Arc, OnceLock, RwLock};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
};
use futures::future::BoxFuture;
use kvbm_protocols::cache_manifest::RegistrationEpoch;
use tokio::task::JoinHandle;
use velo_ext::{InstanceId, PeerInfo};

use super::bundle::{BundleDirectory, BundleDirectoryError};
use super::index::PositionalIndex;
use super::ingest::{IngestCounters, IngestSinks, run_ingest_loop};
use super::protocol::{
    self, ByPositionResponse, IndexerConfigResponse, InstancesResponse, QueryRequest,
    QueryResponse, TierPlacementSnapshotRequest, TierPlacementSnapshotResponse,
};
use super::tier_placement::{
    SnapshotInstall, TierPlacementProjection, TierPlacementProjectionError, VeloSnapshotRequester,
};
use super::zmq::{bind_sub_socket, bound_endpoint, port_of};
use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::protocol::{Feature, FeatureKey, MutationCredential};

/// Default host advertised in `GET /config`'s `zmq_endpoint` when none is
/// configured. Single-host / loopback deployments work out of the box;
/// multi-host deployments must set an explicit advertise host.
const DEFAULT_ADVERTISE_HOST: &str = "127.0.0.1";
const DEFAULT_BUNDLE_LEASE_TTL_MS: u64 = 30_000;

/// Hub-side KV block index feature manager.
pub struct IndexerManager {
    index: Arc<PositionalIndex>,
    bundle_directory: Arc<BundleDirectory>,
    /// ZMQ bind spec (e.g. `tcp://0.0.0.0:0`).
    zmq_bind: String,
    /// Host advertised to publishers in `GET /config`.
    advertise_host: String,
    /// Resolved advertised endpoint (`tcp://host:port`), set during `attach`.
    endpoint: OnceLock<String>,
    /// Ingest task handle (set once spawned during `attach`).
    ingest_task: OnceLock<JoinHandle<()>>,
    /// Instances that declared `Feature::Indexer` at registration. Tracked
    /// separately from the index contents: an instance can register
    /// (participate) before — or without ever — emitting KV events, so this is
    /// the *registered* set, not the *emitting* set. Maintained by
    /// `on_register`/`on_unregister`; `GET /instances` sorts the output for a
    /// stable response (`InstanceId` is not `Ord`).
    ///
    /// Shared (not cloned) with [`Self::tier_placements`], which uses it as the
    /// admission gate for creating a projection: one registered set, so the two
    /// halves of the feature cannot disagree about who is participating.
    instances: Arc<RwLock<HashSet<InstanceId>>>,
    /// Advisory tier-placement projection (R7b §4).
    tier_placements: Arc<TierPlacementProjection>,
    /// Per-reason ZMQ ingest drop counters.
    ingest_counters: Arc<IngestCounters>,
}

impl std::fmt::Debug for IndexerManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexerManager")
            .field("max_seq_len", &self.index.max_seq_len())
            .field("block_size", &self.index.block_size())
            .field("num_positions", &self.index.num_positions())
            .field("endpoint", &self.endpoint.get())
            .finish()
    }
}

impl IndexerManager {
    /// Builds a manager sized for `max_seq_len`/`block_size`, binding ingest to
    /// `zmq_bind` (defaults to `tcp://0.0.0.0:0`) and advertising
    /// `advertise_host` (defaults to `127.0.0.1`).
    pub fn new(
        max_seq_len: usize,
        block_size: usize,
        zmq_bind: Option<String>,
        advertise_host: Option<String>,
    ) -> anyhow::Result<Self> {
        let index = Arc::new(PositionalIndex::new(max_seq_len, block_size)?);
        let instances = Arc::new(RwLock::new(HashSet::new()));
        Ok(Self {
            index,
            bundle_directory: Arc::new(BundleDirectory::new(DEFAULT_BUNDLE_LEASE_TTL_MS)),
            zmq_bind: zmq_bind.unwrap_or_else(|| "tcp://0.0.0.0:0".to_string()),
            advertise_host: advertise_host.unwrap_or_else(|| DEFAULT_ADVERTISE_HOST.to_string()),
            endpoint: OnceLock::new(),
            ingest_task: OnceLock::new(),
            tier_placements: Arc::new(TierPlacementProjection::new(Arc::clone(&instances))),
            ingest_counters: Arc::new(IngestCounters::default()),
            instances,
        })
    }

    /// Shared tier-placement projection handle (for tests / CT-2a consumers).
    #[must_use]
    pub fn tier_placements(&self) -> &Arc<TierPlacementProjection> {
        &self.tier_placements
    }

    /// Per-reason ZMQ ingest drop counters.
    #[must_use]
    pub fn ingest_counters(&self) -> &Arc<IngestCounters> {
        &self.ingest_counters
    }

    /// Authorize and install a publisher's full tier-placement state.
    ///
    /// Order is load-bearing: shape, then authority, then epoch agreement, then
    /// the transactional install. Authorizing before validating would let an
    /// unauthenticated caller learn whether a credential is valid from the
    /// shape of the rejection.
    fn install_tier_placement_snapshot(
        &self,
        request: &TierPlacementSnapshotRequest,
    ) -> Result<TierPlacementSnapshotResponse, (StatusCode, String)> {
        request
            .snapshot
            .validate()
            .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
        let authorized_epoch = self
            .bundle_directory
            .authorize_owner_epoch(request.snapshot.instance_id, &request.credential)
            .map_err(|error| {
                let status = match error {
                    BundleDirectoryError::UnknownOwner { .. }
                    | BundleDirectoryError::UnauthorizedOwner { .. } => StatusCode::UNAUTHORIZED,
                    BundleDirectoryError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
                    _ => StatusCode::CONFLICT,
                };
                (status, error.to_string())
            })?;
        let installed = self
            .tier_placements
            .install_snapshot(&request.snapshot, authorized_epoch)
            .map_err(|error| {
                let status = match error {
                    TierPlacementProjectionError::Invalid(_) => StatusCode::BAD_REQUEST,
                    TierPlacementProjectionError::EpochMismatch => StatusCode::CONFLICT,
                    TierPlacementProjectionError::Capacity { .. } => {
                        StatusCode::INSUFFICIENT_STORAGE
                    }
                    TierPlacementProjectionError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
                };
                (status, error.to_string())
            })?;
        Ok(match installed {
            SnapshotInstall::Installed {
                installed_generation,
                seq_floor,
            } => TierPlacementSnapshotResponse {
                installed: true,
                installed_generation,
                seq_floor,
            },
            // A periodic push the hub already has. Not an error: R7b §3 has the
            // publisher pushing every 60 s regardless of whether anything was
            // lost, so the common case is exactly this.
            SnapshotInstall::AlreadyCurrent {
                installed_generation,
            } => TierPlacementSnapshotResponse {
                installed: false,
                installed_generation,
                seq_floor: request.snapshot.seq_floor,
            },
        })
    }

    /// Snapshot of the registered instance set as decimal `u128` strings
    /// (matching the holder ids in [`IndexEntry::instances`]), sorted for a
    /// stable response.
    fn instances_response(&self) -> InstancesResponse {
        let mut instances: Vec<String> = self
            .instances
            .read()
            .map(|s| s.iter().map(|id| id.as_u128().to_string()).collect())
            .unwrap_or_default();
        instances.sort();
        InstancesResponse { instances }
    }

    /// Shared index handle (for tests / introspection).
    pub fn index(&self) -> &Arc<PositionalIndex> {
        &self.index
    }

    /// Resolved advertised ZMQ endpoint, once `attach` has bound it.
    pub fn endpoint(&self) -> Option<&String> {
        self.endpoint.get()
    }

    /// Returns the current complete-bundle advertisement count for test gates.
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn bundle_advertisement_count(&self) -> anyhow::Result<usize> {
        self.bundle_directory
            .advertisement_count()
            .map_err(anyhow::Error::from)
    }

    fn config_response(&self) -> IndexerConfigResponse {
        IndexerConfigResponse {
            max_seq_len: self.index.max_seq_len(),
            block_size: self.index.block_size(),
            num_positions: self.index.num_positions(),
            zmq_endpoint: self.endpoint.get().cloned().unwrap_or_default(),
        }
    }
}

impl FeatureManager for IndexerManager {
    fn key(&self) -> FeatureKey {
        FeatureKey::Indexer
    }

    fn config_requirements(&self) -> crate::features::FeatureConfigRequirements {
        // The publisher's page size must match the index's block size or events
        // hash/bucket wrong. `max_seq_len` is NOT a must-match: a larger value
        // simply grows the index (see `on_register`).
        crate::features::FeatureConfigRequirements {
            block_size: true,
            block_layout: false,
        }
    }

    fn requires_runtime_summary(&self) -> bool {
        // KV-index is new (introduced with the runtime summary): mandate it so
        // a publisher cannot register without its block_size being checked
        // against the hub's index block size.
        true
    }

    fn authoritative_block_size(&self) -> Option<usize> {
        // The index block size is the source of truth publishers must match.
        // Reconciled into `primary` at startup so validation never depends on
        // the operator having also set `primary` explicitly.
        Some(self.index.block_size())
    }

    fn descriptor(&self, _primary: &crate::protocol::PrimaryConfig) -> serde_json::Value {
        // Advertise the ZMQ ingest endpoint + sizing so the connector can wire
        // its publisher straight from the aggregate config (no separate probe).
        serde_json::to_value(self.config_response()).unwrap_or(serde_json::Value::Null)
    }

    fn route_prefix(&self) -> Option<&'static str> {
        Some(protocol::ROUTE_PREFIX)
    }

    fn attach<'a>(&'a self, ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async move {
            let sub = bind_sub_socket(&self.zmq_bind)
                .map_err(|e| FeatureError::Other(anyhow::anyhow!("indexer bind: {e}")))?;
            let bound = bound_endpoint(&sub)
                .map_err(|e| FeatureError::Other(anyhow::anyhow!("indexer endpoint: {e}")))?;
            let port = port_of(&bound)
                .map_err(|e| FeatureError::Other(anyhow::anyhow!("indexer port: {e}")))?;
            let advertised = format!("tcp://{}:{}", self.advertise_host, port);
            tracing::info!(
                bound = %bound,
                advertised = %advertised,
                max_seq_len = self.index.max_seq_len(),
                block_size = self.index.block_size(),
                "indexer ingest bound"
            );
            let _ = self.endpoint.set(advertised);

            let sinks = IngestSinks {
                index: Arc::clone(&self.index),
                tier_placements: Arc::clone(&self.tier_placements),
                counters: Arc::clone(&self.ingest_counters),
            };
            let task = tokio::spawn(run_ingest_loop(sub, sinks, ctx.cancel));
            let _ = self.ingest_task.set(task);

            // Expose the velo-plane block lookup (`QUERY_HANDLER`) when the hub
            // runs with a transport. Discovery-only hubs skip it — clients fall
            // back to the HTTP `POST /query` surface.
            if let Some(velo) = ctx.velo.as_ref() {
                let messenger = velo.messenger();
                messenger
                    .register_handler(super::handlers::create_query_handler(Arc::clone(
                        &self.index,
                    )))
                    .map_err(|e| {
                        FeatureError::Other(anyhow::anyhow!("indexer query handler: {e}"))
                    })?;
                for handler in
                    super::handlers::create_bundle_handlers(Arc::clone(&self.bundle_directory))
                {
                    messenger.register_handler(handler).map_err(|error| {
                        FeatureError::Other(anyhow::anyhow!("bundle directory handler: {error}"))
                    })?;
                }
                // The hub is the *caller* on the snapshot-request handler, not
                // the callee: the responder is the publisher (CT-2, rhino side).
                self.tier_placements
                    .set_requester(Arc::new(VeloSnapshotRequester::new(messenger.clone())));
            } else {
                // A discovery-only hub can never ask for a snapshot, so a
                // projection that loses continuity stays invalid and answers
                // empty forever. That is the correct degradation (temporary miss
                // semantics, indefinitely), not a reason to serve stale state.
                tracing::warn!(
                    "indexer attached without a transport: tier placement projections cannot \
                     request snapshots and will answer empty after any loss"
                );
            }
            Ok(())
        })
    }

    fn on_register<'a>(
        &'a self,
        instance_id: InstanceId,
        feature: &'a Feature,
    ) -> BoxFuture<'a, Result<(), FeatureError>> {
        // The client declares `Feature::Indexer` so the hub can reclaim its
        // index entries on unregister (`on_unregister` → `remove_instance`).
        // The index itself is populated out-of-band via the ZMQ ingest socket,
        // so there is nothing to do here beyond accepting the (empty) payload
        // and rejecting a misrouted key. Block-size / max-seq-len consistency
        // is validated centrally via `RuntimeConfigSummary`.
        Box::pin(async move {
            match feature {
                Feature::Indexer(cfg) => {
                    // Grow the index to fit this registrant's max_seq_len (never
                    // shrinks). Block-size consistency is validated centrally.
                    if let Some(max_seq_len) = cfg.max_seq_len {
                        self.index.grow_to_max_seq_len(max_seq_len);
                    }
                    // Track the registered (participating) instance so
                    // `GET /instances` can report it even before it emits any
                    // KV events.
                    if let Ok(mut set) = self.instances.write() {
                        set.insert(instance_id);
                    }
                    tracing::debug!(
                        instance = %instance_id,
                        max_seq_len = ?cfg.max_seq_len,
                        num_positions = self.index.num_positions(),
                        "indexer participation registered"
                    );
                    Ok(())
                }
                _ => Err(FeatureError::KeyMismatch {
                    manager: FeatureKey::Indexer,
                    payload: feature.key(),
                }),
            }
        })
    }

    fn stage_registration(
        &self,
        instance_id: InstanceId,
        credential: &MutationCredential,
        registration_epoch: RegistrationEpoch,
        participates: bool,
    ) -> Result<(), FeatureError> {
        self.bundle_directory
            .stage_owner_transition(
                instance_id,
                participates.then(|| credential.clone()),
                registration_epoch,
            )
            .map_err(anyhow::Error::from)?;
        Ok(())
    }

    fn commit_registration(
        &self,
        instance_id: InstanceId,
        _credential: &MutationCredential,
        registration_epoch: RegistrationEpoch,
        incarnation: crate::registry::RegistryIncarnation,
        _participates: bool,
    ) -> Result<(), FeatureError> {
        self.bundle_directory
            .bind_owner_registration(instance_id, registration_epoch, incarnation)
            .map_err(anyhow::Error::from)?;
        // Drop the advisory projection at every registration commit, not only at
        // `on_unregister`. A publisher that restarts and re-registers under the
        // *same* instance id with an unchanged feature set never reaches
        // `on_unregister` (the registration transaction fires it only for
        // features present before and absent now), so without this the previous
        // process lifetime's Ready set would keep answering `valid` under a new
        // `RegistrationEpoch` — an unbounded stale success across every restart
        // that reuses its id. Self-healing on the next epoch-stamped delta is not
        // a substitute: a publisher that has not yet emitted a placement change,
        // or that runs with the tier stream off, would never emit one.
        //
        // Unconditional on purpose. A first registration has nothing to drop, a
        // re-registration without the feature is already covered by
        // `on_unregister`, and dropping advisory state is always safe — including
        // on the rollback path, where the publisher will push a snapshot for
        // whichever epoch it ends up holding.
        //
        // Ordering note: the epoch is bound above, so a snapshot authorized
        // against the *new* epoch can install between the two statements and be
        // dropped here. That is the safe direction — the projection is left
        // empty, not stale — and it self-heals on the hub's next snapshot request
        // or the publisher's periodic push. Dropping first would not fix it
        // either; it would only move the window.
        self.tier_placements.remove_instance(instance_id);
        Ok(())
    }

    fn on_unregister(&self, instance_id: InstanceId) {
        // Bridge the registry's velo InstanceId to the u128 the events wire
        // format carries (publishers stamp `velo_id.as_u128()`).
        self.index.remove_instance(instance_id.as_u128());
        self.bundle_directory.remove_owner(instance_id);
        // Advisory placement state is about a process that no longer exists, so
        // it is dropped outright rather than aged out.
        self.tier_placements.remove_instance(instance_id);
        if let Ok(mut set) = self.instances.write() {
            set.remove(&instance_id);
        }
    }

    fn on_register_any<'a>(
        &'a self,
        instance_id: InstanceId,
        _peer: &'a PeerInfo,
        incarnation: crate::registry::RegistryIncarnation,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Err(error) = self
                .bundle_directory
                .finalize_owner_registration(instance_id, incarnation)
            {
                tracing::error!(
                    instance = %instance_id,
                    %incarnation,
                    %error,
                    "indexer owner registration could not be finalized"
                );
            }
        })
    }

    fn control_router(self: Arc<Self>) -> Router {
        read_routes().merge(control_routes()).with_state(self)
    }

    fn public_router(self: Arc<Self>) -> Router {
        read_routes().with_state(self)
    }
}

/// Routes mounted on both ports. `POST /query` is here despite its verb: it is
/// a lookup whose argument set is too large for a URL, not a mutation.
fn read_routes() -> Router<Arc<IndexerManager>> {
    Router::new()
        .route(protocol::paths::CONFIG, get(get_config))
        .route(protocol::paths::INSTANCES, get(get_instances))
        .route(protocol::paths::BY_POSITION, get(get_by_position))
        .route(protocol::paths::QUERY, post(post_query))
}

/// Routes mounted on the control port only.
///
/// The two routers used to be the same function, so anything added to it landed
/// on the read-only discovery port as well. The snapshot install is a
/// credential-authorized mutation and must not be reachable there.
fn control_routes() -> Router<Arc<IndexerManager>> {
    Router::new().route(
        protocol::paths::TIER_PLACEMENT_SNAPSHOT,
        post(post_tier_placement_snapshot),
    )
}

async fn get_config(State(mgr): State<Arc<IndexerManager>>) -> Json<IndexerConfigResponse> {
    Json(mgr.config_response())
}

async fn get_instances(State(mgr): State<Arc<IndexerManager>>) -> Json<InstancesResponse> {
    Json(mgr.instances_response())
}

async fn get_by_position(
    State(mgr): State<Arc<IndexerManager>>,
    Path(pos): Path<usize>,
) -> Json<ByPositionResponse> {
    Json(mgr.index.by_position(pos))
}

async fn post_query(
    State(mgr): State<Arc<IndexerManager>>,
    Json(req): Json<QueryRequest>,
) -> Json<QueryResponse> {
    Json(QueryResponse {
        hit: mgr.index.query(&req.hashes),
    })
}

async fn post_tier_placement_snapshot(
    State(mgr): State<Arc<IndexerManager>>,
    Json(req): Json<TierPlacementSnapshotRequest>,
) -> Result<Json<TierPlacementSnapshotResponse>, (StatusCode, String)> {
    mgr.install_tier_placement_snapshot(&req).map(Json)
}

#[cfg(test)]
mod tests {
    use kvbm_common::{LogicalResourceId, SequenceHash};
    use kvbm_protocols::cache_manifest::{
        BundleKey, BundleResourceLineage, CacheManifestId, ResourceRequirement, ResourceRole,
    };

    use super::*;
    use crate::features::indexer::protocol::{
        BundleAdvertisementRecord, BundlePublishRequest, BundleQueryMissReason, BundleQueryOutcome,
        BundleQueryRequest,
    };

    #[tokio::test]
    async fn reregister_without_indexer_removes_owner_and_old_mutation_authority() {
        let manager = IndexerManager::new(128, 4, None, None).unwrap();
        #[cfg(feature = "test-support")]
        assert_eq!(manager.bundle_advertisement_count().unwrap(), 0);
        let owner = InstanceId::new_v4();
        let feature = Feature::Indexer(Default::default());
        manager.on_register(owner, &feature).await.unwrap();

        let old_credential = MutationCredential::generate();
        let registration_epoch = crate::features::indexer::bundle::test_registration_epoch(owner);
        manager
            .bundle_directory
            .register_owner(owner, old_credential.clone())
            .unwrap();
        let resource = LogicalResourceId(1);
        let requirements =
            vec![ResourceRequirement::new(resource, ResourceRole::PrefixHistory, 4).unwrap()];
        let hashes = vec![SequenceHash::root(1), SequenceHash::root(1).extend(2)];
        let manifest = CacheManifestId::from_bytes([41; 32]);
        let key = BundleKey::from_parts(manifest, hashes[1], 8).unwrap();
        let publish = BundlePublishRequest {
            credential: old_credential.clone(),
            advertisement: BundleAdvertisementRecord {
                key,
                generation: 1,
                owner,
                registration_epoch: Some(registration_epoch),
                requirements: requirements.clone(),
                lineages: vec![BundleResourceLineage::new(resource, hashes).unwrap()],
                expires_at_unix_ms: 10_000,
                placements: Vec::new(),
                stage_cost_hint_us: None,
                advertised_at_unix_ms: None,
            },
        };
        manager.bundle_directory.publish(publish.clone()).unwrap();
        #[cfg(feature = "test-support")]
        assert_eq!(manager.bundle_advertisement_count().unwrap(), 1);

        let replacement_epoch = RegistrationEpoch::new();
        manager
            .stage_registration(
                owner,
                &MutationCredential::generate(),
                replacement_epoch,
                false,
            )
            .unwrap();
        manager
            .commit_registration(
                owner,
                &MutationCredential::generate(),
                replacement_epoch,
                crate::registry::RegistryIncarnation::from_u64(2),
                false,
            )
            .unwrap();
        manager
            .bundle_directory
            .finalize_owner_registration(owner, crate::registry::RegistryIncarnation::from_u64(2))
            .unwrap();
        manager.on_unregister(owner);

        assert!(matches!(
            manager.bundle_directory.publish(publish),
            Err(super::super::bundle::BundleDirectoryError::UnknownOwner { .. })
        ));
        assert_eq!(
            manager.bundle_directory.query(BundleQueryRequest {
                manifest,
                requirements,
                candidates: vec![key],
                now_unix_ms: 1_000,
            }),
            BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
        );
        assert!(manager.instances_response().instances.is_empty());
    }

    /// A publisher that restarts and re-registers under the *same* instance id
    /// with an unchanged feature set never reaches `on_unregister` — the
    /// registration transaction fires that only for features present before and
    /// absent now. So the advisory projection has to be dropped on the
    /// registration commit, or the dead process lifetime's Ready set keeps
    /// answering `valid` under a brand-new `RegistrationEpoch`: an unbounded
    /// stale success across every restart that reuses its id.
    ///
    /// Routed through `IndexerManager` rather than the projection directly,
    /// because the bug was in the lifecycle wiring, not in the projection.
    #[tokio::test]
    async fn re_registering_the_same_instance_drops_its_tier_placement_projection() {
        use kvbm_protocols::tier_protocol::{KeyRange, PhysicalPlacementMode};
        use kvbm_protocols::tier_protocol::{
            PlacementScope, TIER_MEDIUM_CAP_DIRECT_SERVABLE, TIER_PLACEMENT_SCHEMA_VERSION,
            TierDepth, TierMedium, TierPlacementEntry, TierPlacementSnapshotV1,
        };

        let manager = IndexerManager::new(128, 4, None, None).unwrap();
        let owner = InstanceId::new_v4();
        let feature = Feature::Indexer(Default::default());
        manager.on_register(owner, &feature).await.unwrap();

        let cache = CacheManifestId::from_bytes([7; 32]);
        let resource = LogicalResourceId(1);
        let hash = SequenceHash::root(1);
        let epoch = crate::features::indexer::bundle::test_registration_epoch(owner);
        let tier = TierDepth(1);
        manager
            .tier_placements()
            .install_snapshot(
                &TierPlacementSnapshotV1 {
                    v: TIER_PLACEMENT_SCHEMA_VERSION,
                    cache,
                    instance_id: owner,
                    registration_epoch: epoch,
                    snapshot_generation: 1,
                    seq_floor: 0,
                    media: vec![TierMedium {
                        depth: tier,
                        medium: "pinned-host".to_string(),
                        capabilities: TIER_MEDIUM_CAP_DIRECT_SERVABLE,
                    }],
                    manifests: Vec::new(),
                    entries: vec![TierPlacementEntry {
                        scope: PlacementScope::unitary(resource),
                        tier,
                        placement: PhysicalPlacementMode::Whole,
                        generation: 1,
                        keys: KeyRange::Hashes(vec![hash]),
                    }],
                },
                epoch,
            )
            .expect("installs");
        assert!(
            !manager
                .tier_placements()
                .holders(cache, PlacementScope::unitary(resource), hash)
                .is_empty()
        );

        // Re-register: same id, same feature set, new epoch. `on_unregister`
        // does not fire on this path.
        let replacement_epoch = RegistrationEpoch::new();
        let incarnation = crate::registry::RegistryIncarnation::from_u64(2);
        manager
            .stage_registration(
                owner,
                &MutationCredential::generate(),
                replacement_epoch,
                true,
            )
            .unwrap();
        manager
            .commit_registration(
                owner,
                &MutationCredential::generate(),
                replacement_epoch,
                incarnation,
                true,
            )
            .unwrap();
        manager
            .bundle_directory
            .finalize_owner_registration(owner, incarnation)
            .unwrap();
        manager.on_register(owner, &feature).await.unwrap();

        assert!(
            !manager.tier_placements().is_valid(cache, owner),
            "the previous lifetime's projection must not survive a re-registration"
        );
        assert!(
            manager
                .tier_placements()
                .holders(cache, PlacementScope::unitary(resource), hash)
                .is_empty(),
            "and it must not still be answering"
        );
    }

    /// The router split is a security boundary, so it is asserted by routing a
    /// request rather than by reading the code: before R7b both routers were the
    /// same function, and anything added to it silently appeared on the
    /// read-only discovery port too.
    #[tokio::test]
    async fn the_snapshot_mutation_is_reachable_only_on_the_control_router() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode, header::CONTENT_TYPE};
        use kvbm_protocols::tier_protocol::{
            TIER_PLACEMENT_SCHEMA_VERSION, TierPlacementSnapshotV1,
        };
        use tower::ServiceExt as _;

        let manager = Arc::new(IndexerManager::new(128, 4, None, None).unwrap());
        let body = serde_json::to_vec(&TierPlacementSnapshotRequest {
            credential: MutationCredential::generate(),
            snapshot: TierPlacementSnapshotV1 {
                v: TIER_PLACEMENT_SCHEMA_VERSION,
                cache: CacheManifestId::from_bytes([3; 32]),
                instance_id: InstanceId::new_v4(),
                registration_epoch: RegistrationEpoch::new(),
                snapshot_generation: 1,
                seq_floor: 0,
                media: Vec::new(),
                manifests: Vec::new(),
                entries: Vec::new(),
            },
        })
        .unwrap();
        let request = || {
            Request::builder()
                .method("POST")
                .uri(protocol::paths::TIER_PLACEMENT_SNAPSHOT)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(body.clone()))
                .unwrap()
        };

        let public = FeatureManager::public_router(Arc::clone(&manager));
        assert_eq!(
            public.oneshot(request()).await.unwrap().status(),
            StatusCode::NOT_FOUND,
            "a mutation must not exist on the read-only discovery port"
        );

        // On the control port the route exists and rejects on *authority* — an
        // unregistered owner — which is the failure a 404 would have hidden.
        let control = FeatureManager::control_router(manager);
        assert_eq!(
            control.oneshot(request()).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }
}
