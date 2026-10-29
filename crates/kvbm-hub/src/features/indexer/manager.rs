// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the KV indexer feature.
//!
//! Owns [`ManifestIndexes`], attaches the configured ingest and carrier-feed
//! transport during [`velo_hub::Extension::attach`], and exports its own HTTP surface
//! under `/v1/ext/indexer` (the server nests it via
//! `/v1/ext/{NAME}`).

use std::collections::HashSet;
use std::sync::{Arc, OnceLock, RwLock};

use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::{HeaderValue, StatusCode, header::CONTENT_TYPE},
    response::Response,
    routing::{get, post},
};
use dynamo_kv_router::carrier_feed::{CARRIER_FEED_TOPIC, encode_snapshot};
use kvbm_logical::events::CreateKind;
use kvbm_protocols::cache_manifest::RegistrationEpoch;
use tokio::sync::mpsc::unbounded_channel;
use tokio::task::JoinHandle;
use velo_ext::InstanceId;
use velo_hub::store::Removal;
use velo_hub::{AttachContext, Committed, Extension, Rejection, Txn};

use super::bundle::{BundleDirectory, BundleDirectoryError};
use super::feed::{FeedCounters, bind_pub_socket, run_feed_publisher, run_nats_feed_publisher};
use super::index::ManifestIndexes;
use super::ingest::{IngestCounters, IngestSinks, run_ingest_loop};
use super::protocol::{
    self, ByPositionResponse, EventPlane, IndexerConfigResponse, InstancesResponse, QueryRequest,
    QueryResponse, TierPlacementSnapshotRequest, TierPlacementSnapshotResponse,
};
use super::tier_placement::{
    SnapshotInstall, TierPlacementProjection, TierPlacementProjectionError, VeloSnapshotRequester,
};
use super::zmq::{bind_sub_socket, bound_endpoint, port_of};
use crate::features::{FeatureConfigRequirements, KvbmFeature};

/// Default host advertised in `GET /config` endpoints when none is configured.
/// Single-host / loopback deployments work out of the box; multi-host
/// deployments must set an explicit advertise host.
const DEFAULT_ADVERTISE_HOST: &str = "127.0.0.1";
const DEFAULT_BUNDLE_LEASE_TTL_MS: u64 = 30_000;

/// Hub-side KV block index feature manager.
pub struct IndexerManager {
    indexes: Arc<ManifestIndexes>,
    bundle_directory: Arc<BundleDirectory>,
    required_kind: Option<CreateKind>,
    /// ZMQ bind spec (e.g. `tcp://0.0.0.0:0`).
    zmq_bind: String,
    /// Carrier-feed PUB bind spec (e.g. `tcp://0.0.0.0:0`).
    feed_bind: String,
    event_plane: EventPlane,
    nats_server: String,
    nats_subject_prefix: String,
    /// Host advertised in `GET /config` for ingest and carrier-feed endpoints.
    advertise_host: String,
    /// Resolved advertised endpoint (`tcp://host:port`), set during `attach`.
    endpoint: OnceLock<String>,
    /// Resolved advertised carrier-feed endpoint, set during `attach`.
    feed_endpoint: OnceLock<String>,
    /// Ingest task handle (set once spawned during `attach`).
    ingest_task: OnceLock<JoinHandle<()>>,
    /// Carrier-feed task handle (set once spawned during `attach`).
    feed_task: OnceLock<JoinHandle<()>>,
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
    /// Per-reason ingest drop counters.
    ingest_counters: Arc<IngestCounters>,
    /// Carrier-feed publisher counters.
    feed_counters: Arc<FeedCounters>,
    /// The hub authority, set at attach: authenticates mutation credentials.
    authority: OnceLock<velo_hub::Authority>,
}

impl std::fmt::Debug for IndexerManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexerManager")
            .field("max_seq_len", &self.indexes.max_seq_len())
            .field("block_size", &self.indexes.block_size())
            .field("endpoint", &self.endpoint.get())
            .field("feed_endpoint", &self.feed_endpoint.get())
            .field("event_plane", &self.event_plane)
            .finish()
    }
}

impl IndexerManager {
    /// Builds a manager sized for `max_seq_len`/`block_size`. In ZMQ mode,
    /// ingest and carrier-feed sockets use OS-assigned ports by default.
    pub fn new(
        max_seq_len: usize,
        block_size: usize,
        zmq_bind: Option<String>,
        advertise_host: Option<String>,
    ) -> anyhow::Result<Self> {
        let indexes = Arc::new(ManifestIndexes::new(max_seq_len, block_size)?);
        let instances = Arc::new(RwLock::new(HashSet::new()));
        Ok(Self {
            indexes,
            bundle_directory: Arc::new(BundleDirectory::new(DEFAULT_BUNDLE_LEASE_TTL_MS)),
            required_kind: None,
            zmq_bind: zmq_bind.unwrap_or_else(|| "tcp://0.0.0.0:0".to_string()),
            feed_bind: "tcp://0.0.0.0:0".to_string(),
            event_plane: EventPlane::Zmq,
            nats_server: "nats://localhost:4222".to_string(),
            nats_subject_prefix: format!("kvbm.hub.{}", uuid::Uuid::new_v4().simple()),
            advertise_host: advertise_host.unwrap_or_else(|| DEFAULT_ADVERTISE_HOST.to_string()),
            endpoint: OnceLock::new(),
            feed_endpoint: OnceLock::new(),
            ingest_task: OnceLock::new(),
            feed_task: OnceLock::new(),
            tier_placements: Arc::new(TierPlacementProjection::new(Arc::clone(&instances))),
            ingest_counters: Arc::new(IngestCounters::default()),
            feed_counters: Arc::new(FeedCounters::default()),
            authority: OnceLock::new(),
            instances,
        })
    }

    /// Accept only publishers that register with this create kind.
    #[must_use]
    pub fn with_required_kind(mut self, kind: CreateKind) -> Self {
        self.required_kind = Some(kind);
        self
    }

    /// Sets the carrier-feed ZMQ PUB bind spec.
    #[must_use]
    pub fn with_feed_bind(mut self, feed_bind: String) -> Self {
        self.feed_bind = feed_bind;
        self
    }

    /// Selects the event transport and its NATS connection and namespace.
    #[must_use]
    pub fn with_event_plane(
        mut self,
        event_plane: EventPlane,
        nats_server: String,
        nats_subject_prefix: Option<String>,
    ) -> Self {
        self.event_plane = event_plane;
        self.nats_server = nats_server;
        if let Some(prefix) = nats_subject_prefix.filter(|prefix| !prefix.is_empty()) {
            self.nats_subject_prefix = prefix;
        }
        self
    }

    /// Shared tier-placement projection handle (for tests / CT-2a consumers).
    #[must_use]
    pub fn tier_placements(&self) -> &Arc<TierPlacementProjection> {
        &self.tier_placements
    }

    /// Per-reason event-plane ingest drop counters.
    #[must_use]
    pub fn ingest_counters(&self) -> &Arc<IngestCounters> {
        &self.ingest_counters
    }

    /// Carrier-feed publisher counters.
    #[must_use]
    pub fn feed_counters(&self) -> &Arc<FeedCounters> {
        &self.feed_counters
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
        let authority = self.authority.get().ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "hub authority not attached".to_owned(),
        ))?;
        let owner = request.snapshot.instance_id;
        let authorized_epoch =
            super::handlers::authorized_epoch(authority, owner, &request.credential)
                .and_then(|epoch| self.bundle_directory.authorize_epoch(owner, epoch))
                .map_err(|error| {
                    let status = match error {
                        BundleDirectoryError::UnknownOwner { .. }
                        | BundleDirectoryError::UnauthorizedOwner { .. } => {
                            StatusCode::UNAUTHORIZED
                        }
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
        let bindings = self
            .indexes
            .bindings()
            .into_iter()
            .map(
                |(instance, manifest, create_kind)| protocol::InstanceBinding {
                    instance: instance.to_string(),
                    manifest: manifest.to_string(),
                    create_kind,
                },
            )
            .collect();
        InstancesResponse {
            instances,
            bindings,
        }
    }

    /// Shared manifest-scoped indexes, read by embedding processes through this accessor.
    pub fn indexes(&self) -> &Arc<ManifestIndexes> {
        &self.indexes
    }

    /// Resolved advertised ingest endpoint, once `attach` has bound it.
    pub fn endpoint(&self) -> Option<&String> {
        self.endpoint.get()
    }

    /// Resolved advertised carrier-feed endpoint, once `attach` has bound it.
    pub fn feed_endpoint(&self) -> Option<&String> {
        self.feed_endpoint.get()
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
            event_plane: self.event_plane,
            max_seq_len: self.indexes.max_seq_len(),
            block_size: self.indexes.block_size(),
            num_positions: self.indexes.max_seq_len() / self.indexes.block_size(),
            zmq_endpoint: if self.event_plane == EventPlane::Zmq {
                self.endpoint.get().cloned().unwrap_or_default()
            } else {
                String::new()
            },
            feed_endpoint: if self.event_plane == EventPlane::Zmq {
                self.feed_endpoint.get().cloned().unwrap_or_default()
            } else {
                String::new()
            },
            required_kind: self.required_kind,
            nats_subject_prefix: if self.event_plane == EventPlane::Nats {
                self.nats_subject_prefix.clone()
            } else {
                String::new()
            },
        }
    }
}

impl KvbmFeature for IndexerManager {
    fn config_requirements(&self) -> FeatureConfigRequirements {
        // The publisher's page size must match the index's block size or events
        // hash/bucket wrong. `max_seq_len` is NOT a must-match: a larger value
        // simply grows the index (see `prepare`).
        FeatureConfigRequirements {
            block_size: true,
            block_layout: false,
        }
    }

    fn requires_runtime_summary(&self) -> bool {
        // A publisher cannot register without its block_size being checked
        // against the hub's index block size.
        true
    }

    fn authoritative_block_size(&self) -> Option<usize> {
        Some(self.indexes.block_size())
    }
}

impl Extension for IndexerManager {
    const NAME: &'static str = crate::protocol::INDEXER_EXTENSION;
    type Registration = crate::protocol::IndexerFeatureConfig;
    /// Whether the registrant participates.
    type Staged = bool;

    fn descriptor(&self) -> serde_json::Value {
        // Advertise ingest and carrier-feed endpoints with sizing so clients
        // can wire themselves from the aggregate config.
        serde_json::to_value(self.config_response()).unwrap_or(serde_json::Value::Null)
    }

    async fn attach(&self, ctx: &AttachContext<'_>) -> anyhow::Result<()> {
        let (feed_tx, feed_rx) = unbounded_channel();
        let sinks = IngestSinks {
            indexes: Arc::clone(&self.indexes),
            tier_placements: Arc::clone(&self.tier_placements),
            counters: Arc::clone(&self.ingest_counters),
        };
        match self.event_plane {
            EventPlane::Zmq => {
                let sub = bind_sub_socket(&self.zmq_bind)
                    .map_err(|e| anyhow::anyhow!("indexer bind: {e}"))?;
                let bound =
                    bound_endpoint(&sub).map_err(|e| anyhow::anyhow!("indexer endpoint: {e}"))?;
                let port = port_of(&bound).map_err(|e| anyhow::anyhow!("indexer port: {e}"))?;
                let advertised = format!("tcp://{}:{}", self.advertise_host, port);
                let pub_socket = bind_pub_socket(&self.feed_bind)
                    .map_err(|e| anyhow::anyhow!("carrier feed bind: {e}"))?;
                let feed_bound = bound_endpoint(&pub_socket)
                    .map_err(|e| anyhow::anyhow!("carrier feed endpoint: {e}"))?;
                let feed_port =
                    port_of(&feed_bound).map_err(|e| anyhow::anyhow!("carrier feed port: {e}"))?;
                let feed_advertised = format!("tcp://{}:{}", self.advertise_host, feed_port);
                tracing::info!(
                    bound = %bound,
                    advertised = %advertised,
                    feed_bound = %feed_bound,
                    feed_advertised = %feed_advertised,
                    max_seq_len = self.indexes.max_seq_len(),
                    block_size = self.indexes.block_size(),
                    "indexer ingest and carrier feed bound"
                );
                let _ = self.endpoint.set(advertised);
                let _ = self.feed_endpoint.set(feed_advertised);
                self.indexes
                    .set_feed_sink(feed_tx)
                    .map_err(|_| anyhow::anyhow!("carrier feed sink already installed"))?;
                let feed_task = tokio::spawn(run_feed_publisher(
                    pub_socket,
                    feed_rx,
                    ctx.cancellation().clone(),
                    Arc::clone(&self.feed_counters),
                ));
                let _ = self.feed_task.set(feed_task);
                let task = tokio::spawn(run_ingest_loop(sub, sinks, ctx.cancellation().clone()));
                let _ = self.ingest_task.set(task);
            }
            EventPlane::Nats => {
                let client = async_nats::connect(&self.nats_server)
                    .await
                    .map_err(|error| {
                        anyhow::anyhow!("connecting indexer to NATS {}: {error}", self.nats_server)
                    })?;
                let subscriber = client
                    .subscribe(format!("{}.>", self.nats_subject_prefix))
                    .await
                    .map_err(|error| anyhow::anyhow!("subscribing indexer to NATS: {error}"))?;
                let feed_topic =
                    std::str::from_utf8(CARRIER_FEED_TOPIC).expect("feed topic is valid UTF-8");
                let feed_subject =
                    super::ingest::nats_subject(&self.nats_subject_prefix, feed_topic);
                tracing::info!(
                    nats_server = %self.nats_server,
                    subject_prefix = %self.nats_subject_prefix,
                    feed_subject,
                    max_seq_len = self.indexes.max_seq_len(),
                    block_size = self.indexes.block_size(),
                    "indexer NATS transport attached"
                );
                self.indexes
                    .set_feed_sink(feed_tx)
                    .map_err(|_| anyhow::anyhow!("carrier feed sink already installed"))?;
                let feed_task = tokio::spawn(run_nats_feed_publisher(
                    client,
                    feed_subject,
                    feed_rx,
                    ctx.cancellation().clone(),
                    Arc::clone(&self.feed_counters),
                ));
                let _ = self.feed_task.set(feed_task);
                let task = tokio::spawn(super::ingest::run_nats_ingest_loop(
                    subscriber,
                    self.nats_subject_prefix.clone(),
                    sinks,
                    ctx.cancellation().clone(),
                ));
                let _ = self.ingest_task.set(task);
            }
        }

        // Expose the velo-plane block lookup (`QUERY_HANDLER`) when the hub
        // runs with a transport. Discovery-only hubs skip it — clients fall
        // back to the HTTP `POST /query` surface.
        let _ = self.authority.set(ctx.authority().clone());
        if let Some(velo) = ctx.velo() {
            let messenger = velo.messenger();
            messenger
                .register_handler(super::handlers::create_query_handler(Arc::clone(
                    &self.indexes,
                )))
                .map_err(|e| anyhow::anyhow!("indexer query handler: {e}"))?;
            for handler in super::handlers::create_bundle_handlers(
                Arc::clone(&self.bundle_directory),
                ctx.authority().clone(),
            ) {
                messenger
                    .register_handler(handler)
                    .map_err(|error| anyhow::anyhow!("bundle directory handler: {error}"))?;
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
    }

    async fn prepare(
        &self,
        txn: &Txn<'_>,
        payload: Option<crate::protocol::IndexerFeatureConfig>,
    ) -> Result<bool, Rejection> {
        // Bind the cache manifest and create kind before accepting publication.
        let Some(cfg) = payload else {
            return Ok(false);
        };
        if let Some(required) = self.required_kind
            && cfg.create_kind != required
        {
            return Err(Rejection::bad_request(format!(
                "indexer binding: this hub accepts {required:?} publishers only, got {:?}",
                cfg.create_kind
            )));
        }
        self.indexes
            .bind(
                txn.owner.as_u128(),
                &cfg.manifest,
                cfg.create_kind,
                cfg.max_seq_len,
            )
            .map_err(|error| Rejection::bad_request(format!("indexer binding: {error}")))?;
        tracing::debug!(
            instance = %txn.owner,
            max_seq_len = ?cfg.max_seq_len,
            num_positions = self.indexes.max_seq_len() / self.indexes.block_size(),
            "indexer participation staged"
        );
        Ok(true)
    }

    fn commit(&self, participates: bool, registration: &Committed<'_>) {
        let instance_id = registration.owner;
        if !participates {
            self.remove(instance_id);
            return;
        }
        let epoch = RegistrationEpoch::from_uuid(registration.epoch.as_uuid());
        if let Err(error) = self.bundle_directory.commit_owner(instance_id, epoch) {
            tracing::error!(instance = %instance_id, %error, "indexer owner could not be committed");
        }
        // A publisher that re-registers under the same id starts a new epoch:
        // drop the advisory projection and the index state of its previous
        // lifetime. Dropping advisory state is always safe; the publisher
        // pushes a snapshot for the new epoch.
        self.tier_placements.remove_instance(instance_id);
        self.indexes.clear_instance(instance_id.as_u128());
        if let Ok(mut set) = self.instances.write() {
            set.insert(instance_id);
        }
    }

    fn on_removed(&self, removal: Removal) {
        self.remove(removal.id);
    }

    fn control_routes(self: Arc<Self>) -> Router {
        read_routes().merge(control_routes()).with_state(self)
    }

    fn public_routes(self: Arc<Self>) -> Router {
        read_routes().with_state(self)
    }
}

impl IndexerManager {
    fn remove(&self, instance_id: InstanceId) {
        // Publishers stamp events with `velo_id.as_u128()`.
        self.indexes.unbind(instance_id.as_u128());
        self.bundle_directory.remove_owner(instance_id);
        // Advisory placement state is about a process that no longer exists.
        self.tier_placements.remove_instance(instance_id);
        if let Ok(mut set) = self.instances.write() {
            set.remove(&instance_id);
        }
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
        .route(protocol::paths::FEED_SNAPSHOT, get(get_feed_snapshot))
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

async fn get_feed_snapshot(State(mgr): State<Arc<IndexerManager>>) -> Result<Response, StatusCode> {
    let snapshot = mgr.indexes.feed_snapshot();
    let payload = encode_snapshot(&snapshot).map_err(|error| {
        tracing::warn!(%error, "carrier feed snapshot encoding failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let mut response = Response::new(Body::from(payload));
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/msgpack"),
    );
    Ok(response)
}

async fn get_instances(State(mgr): State<Arc<IndexerManager>>) -> Json<InstancesResponse> {
    Json(mgr.instances_response())
}

async fn get_by_position(
    State(mgr): State<Arc<IndexerManager>>,
    Path((manifest, pos)): Path<(String, usize)>,
) -> Result<Json<ByPositionResponse>, (StatusCode, String)> {
    let manifest = manifest
        .parse()
        .map_err(|error: String| (StatusCode::BAD_REQUEST, error))?;
    Ok(Json(mgr.indexes.by_position(manifest, pos)))
}

async fn post_query(
    State(mgr): State<Arc<IndexerManager>>,
    Json(req): Json<QueryRequest>,
) -> Json<QueryResponse> {
    Json(QueryResponse {
        hit: mgr.indexes.query(req.manifest, &req.hashes),
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
    use kvbm_logical::events::KvbmCacheEvents;
    use kvbm_protocols::cache_manifest::{
        CacheManifest, CacheManifestId, ModelIdentity, ResourceRequirement, ResourceRole,
    };

    use super::*;
    use crate::protocol::Feature;
    use crate::server::HubServer;
    use crate::test_hub;
    use velo_hub::protocol::Credential;

    fn test_indexer_feature() -> Feature {
        Feature::Indexer(crate::protocol::IndexerFeatureConfig {
            max_seq_len: None,
            manifest: CacheManifest::new(
                ModelIdentity::new("test-architecture", "test-revision", [41; 32]).unwrap(),
                "test-cache-abi",
                vec![
                    ResourceRequirement::new(LogicalResourceId(1), ResourceRole::PrefixHistory, 4)
                        .unwrap(),
                ],
                std::collections::BTreeMap::new(),
            )
            .unwrap(),
            create_kind: kvbm_logical::events::CreateKind::Block,
        })
    }

    async fn hub(manager: &Arc<IndexerManager>) -> HubServer {
        test_hub::serve(HubServer::builder().extension(Arc::clone(manager))).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rejected_carrier_binding_does_not_register_instance() {
        let manager = Arc::new(IndexerManager::new(128, test_hub::BLOCK_SIZE, None, None).unwrap());
        let server = hub(&manager).await;
        let Feature::Indexer(mut config) = test_indexer_feature() else {
            unreachable!()
        };
        config.create_kind = kvbm_logical::events::CreateKind::Carrier;

        let client = test_hub::client(&server);
        test_hub::register(&client, test_hub::peer(), &[Feature::Indexer(config)])
            .await
            .unwrap_err();
        assert!(manager.instances_response().instances.is_empty());
        assert!(manager.indexes.bindings().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reregister_without_indexer_removes_owner_and_its_authority() {
        let manager = Arc::new(IndexerManager::new(128, test_hub::BLOCK_SIZE, None, None).unwrap());
        let server = hub(&manager).await;
        let client = test_hub::client(&server);
        let peer = test_hub::peer();
        let owner = peer.instance_id();
        test_hub::register(&client, peer.clone(), &[test_indexer_feature()])
            .await
            .unwrap();
        assert_eq!(manager.instances_response().instances.len(), 1);

        test_hub::register(&client, peer, &[]).await.unwrap();

        assert!(manager.instances_response().instances.is_empty());
        assert!(matches!(
            manager
                .bundle_directory
                .authorize_epoch(owner, RegistrationEpoch::new()),
            Err(super::super::bundle::BundleDirectoryError::UnknownOwner { .. })
        ));
    }

    /// A publisher that restarts and re-registers under the *same* instance id
    /// with an unchanged feature set gets no removal, so the registration
    /// commit must drop the advisory projection. Otherwise the dead process
    /// lifetime's Ready set keeps answering `valid` under a brand-new
    /// `RegistrationEpoch`: an unbounded stale success across every restart
    /// that reuses its id.
    ///
    /// Routed through a real hub registration rather than the projection
    /// directly, because the bug was in the lifecycle wiring.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn re_registering_the_same_instance_drops_its_tier_placement_projection() {
        use kvbm_protocols::tier_protocol::{KeyRange, PhysicalPlacementMode};
        use kvbm_protocols::tier_protocol::{
            PlacementScope, TIER_MEDIUM_CAP_DIRECT_SERVABLE, TIER_PLACEMENT_SCHEMA_VERSION,
            TierDepth, TierMedium, TierPlacementEntry, TierPlacementSnapshotV1,
        };

        let manager = Arc::new(IndexerManager::new(128, test_hub::BLOCK_SIZE, None, None).unwrap());
        let server = hub(&manager).await;
        let client = test_hub::client(&server);
        let peer = test_hub::peer();
        let owner = peer.instance_id();
        let feature = test_indexer_feature();
        test_hub::register(&client, peer.clone(), std::slice::from_ref(&feature))
            .await
            .unwrap();
        let Feature::Indexer(config) = &feature else {
            unreachable!()
        };
        let hash = SequenceHash::root(1);
        manager.indexes.apply(KvbmCacheEvents {
            events: kvbm_logical::events::KvCacheEvents::Create(vec![hash]),
            instance_id: owner.as_u128(),
        });
        assert!(
            manager
                .indexes
                .query(config.manifest.id(), &[hash])
                .is_some()
        );

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

        // Re-register: same id, same feature set. No removal fires.
        test_hub::register(&client, peer, std::slice::from_ref(&feature))
            .await
            .unwrap();

        assert!(
            manager
                .indexes
                .query(config.manifest.id(), &[hash])
                .is_none(),
            "registration commit must clear the previous process lifetime's index entries"
        );
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
    /// request rather than by reading the code: anything added to a shared
    /// router silently appears on the read-only discovery port too.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_snapshot_mutation_is_reachable_only_on_the_control_port() {
        use kvbm_protocols::tier_protocol::{
            TIER_PLACEMENT_SCHEMA_VERSION, TierPlacementSnapshotV1,
        };

        let manager = Arc::new(IndexerManager::new(128, test_hub::BLOCK_SIZE, None, None).unwrap());
        let server = hub(&manager).await;
        let request = TierPlacementSnapshotRequest {
            credential: Credential::generate(),
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
        };
        let http = reqwest::Client::new();
        let post = |addr: std::net::SocketAddr| {
            http.post(format!(
                "http://{addr}/v1/ext/{}{}",
                crate::protocol::INDEXER_EXTENSION,
                protocol::paths::TIER_PLACEMENT_SNAPSHOT
            ))
            .json(&request)
            .send()
        };

        assert_eq!(
            post(server.discovery_addr()).await.unwrap().status(),
            reqwest::StatusCode::NOT_FOUND,
            "a mutation must not exist on the read-only discovery port"
        );
        // On the control port the route exists and rejects on *authority* — an
        // unregistered owner — which is the failure a 404 would have hidden.
        assert_eq!(
            post(server.control_addr()).await.unwrap().status(),
            reqwest::StatusCode::UNAUTHORIZED
        );
    }
}
