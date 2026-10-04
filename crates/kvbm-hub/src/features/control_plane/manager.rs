// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `ControlPlaneManager` — bridges hub HTTP control routes to per-leader velo
//! handlers via the typed [`LeaderControlClient`].
//!
//! Each handler:
//! 1. Verifies velo + registry are attached (503 if not).
//! 2. Verifies the target instance is currently registered (404 if not).
//! 3. Builds `LeaderControlClient::new(velo.messenger().clone(), id)` and
//!    calls the typed sub-client.
//! 4. Maps `Result<T, ControlError>` → HTTP status via
//!    [`ControlError::http_status`] + a JSON body of `{ error, kind }`.
//!
//! The hand-rolled "build velo unary → raw payload → decode envelope" pyramid
//! is gone — `LeaderControlClient` already does all of that internally.
//!
//! [`LeaderControlClient`]: kvbm_protocols::control::LeaderControlClient
//! [`ControlError::http_status`]: kvbm_protocols::control::ControlError::http_status

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

use axum::Json;
use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::{get, post};
use futures::future::BoxFuture;
use kvbm_protocols::control::{
    ControlError, DescribeInstanceRequest, InstanceDescription, LeaderControlClient,
    MetricsSnapshotRequest, ModuleId, ResetRequest,
};
use serde::{Deserialize, Serialize};
use velo_ext::{InstanceId, PeerInfo};

use super::module_refresh::{ModuleRefreshRuntime, ModuleTarget};
use crate::features::http::{
    control_error_response, error_response, json_response, ok_response, service_unavailable,
};
use crate::features::p2p::P2pManager;
use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::handlers::{HEARTBEAT_HANDLER, HeartbeatAck, HeartbeatRequest};
use crate::protocol::{
    self, Feature, FeatureKey, MetricsFanoutResponse, MetricsInstanceEntry, ResetFanoutResponse,
    ResetInstanceEntry,
};
use crate::registry::{PeerRegistry, RegistryIncarnation};

/// Per-leader budget for the `/v1/metrics` fanout. A leader that doesn't
/// answer within this window becomes a per-instance `error: "timeout..."`
/// entry; the rest of the fanout still completes. Kept short on purpose —
/// the UI polls this and a sluggish leader shouldn't drag the whole tab.
const METRICS_FANOUT_PER_LEADER: Duration = Duration::from_secs(2);
/// The hub waits this long for each leader reset request.
const RESET_FANOUT_PER_LEADER: Duration = Duration::from_secs(30);

/// Cached `list_modules` result for one instance.
#[derive(Clone, Debug)]
struct ModulesEntry {
    modules: Vec<ModuleId>,
    fetched_at: Instant,
    incarnation: RegistryIncarnation,
}

/// Origin of a cached describe entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DescribeSource {
    /// Pushed by the leader via `POST /v1/instances/{id}/describe`. Steady state.
    Push,
    /// Pulled by the hub via the velo `describe_instance` handler. Fallback
    /// path used after a cold hub restart or when an operator forces a refresh.
    PullFallback,
}

/// Cached describe payload + provenance.
#[derive(Clone, Debug)]
struct DescribeEntry {
    payload: InstanceDescription,
    received_at: Instant,
    source: DescribeSource,
    incarnation: RegistryIncarnation,
}

/// Bridges hub HTTP control routes to per-leader velo handlers and caches
/// per-instance control-plane metadata (currently the enabled module set).
///
/// State is filled during [`FeatureManager::attach`]:
/// - `velo` — needed to send unary calls to a target leader.
/// - `registry` — needed to confirm a target instance is currently registered
///   before issuing the velo call.
/// - `cancel` — child of the hub's shutdown token; the refresh task exits
///   on cancellation.
///
/// On every instance registration the manager's [`FeatureManager::on_register_any`]
/// fans out a `list_modules` query; the result populates `modules_cache`. A
/// periodic refresh task keeps the cache aligned with hot-installed modules.
///
/// When the hub is launched without a velo transport (`velo_port: None`)
/// the control routes return `503 Service Unavailable` and the cache stays
/// empty.
pub struct ControlPlaneManager {
    velo: OnceLock<Arc<velo::Velo>>,
    registry: OnceLock<Arc<dyn PeerRegistry>>,
    /// Per-instance `list_modules` cache. `std::sync::RwLock` is appropriate
    /// here: writes are uncontended, entries are tiny, and the sync trait
    /// method [`FeatureManager::on_unregister`] can drop entries without
    /// touching the async runtime.
    modules_cache: Arc<RwLock<HashMap<InstanceId, ModulesEntry>>>,
    /// Per-instance describe cache. Populated by the leader's HTTP push
    /// (steady state) or by fallback velo pull (`?force=true` or cold cache).
    describe_cache: Arc<RwLock<HashMap<InstanceId, DescribeEntry>>>,
    /// First-register timestamp per instance. Used to surface
    /// `registered_secs_ago` in `503 describe_pending` responses so the UI
    /// can tell whether the leader is "still warming up" or genuinely silent.
    registered_at: Arc<RwLock<HashMap<InstanceId, Instant>>>,
    /// Bounded, cancellation-aware owner for periodic and registration-time
    /// module refresh work.
    module_refresh: OnceLock<ModuleRefreshRuntime>,

    /// Optional reference to the hub's `P2pManager`. Set post-construction
    /// via [`Self::set_p2p_manager`] by the production binary and test
    /// fixtures that need describe-push layout-compat validation.
    ///
    /// When `None` the P2P feature isn't wired (or this is a hub built
    /// without it) and `post_describe` skips the layout-compat check
    /// with a warn-level log. Graceful degradation.
    p2p_manager: OnceLock<Arc<P2pManager>>,
}

impl Default for ControlPlaneManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ControlPlaneManager {
    pub fn new() -> Self {
        Self {
            velo: OnceLock::new(),
            registry: OnceLock::new(),
            modules_cache: Arc::new(RwLock::new(HashMap::new())),
            describe_cache: Arc::new(RwLock::new(HashMap::new())),
            registered_at: Arc::new(RwLock::new(HashMap::new())),
            module_refresh: OnceLock::new(),
            p2p_manager: OnceLock::new(),
        }
    }

    /// Wire the hub's [`P2pManager`] into this manager so that
    /// `post_describe` can validate `layout_compat` against the stored
    /// P2P baseline.
    ///
    /// Call once at construction time (production binary, test fixture).
    /// Subsequent calls after the first are no-ops (OnceLock semantics).
    pub fn set_p2p_manager(&self, p2p: Arc<P2pManager>) {
        let _ = self.p2p_manager.set(p2p);
    }

    /// Snapshot of the cached describe payload for `instance_id`. Returns
    /// `None` if no leader has pushed and no pull has succeeded yet.
    pub fn describe_for(&self, instance_id: InstanceId) -> Option<InstanceDescription> {
        let cache = self.describe_cache.read().ok()?;
        let entry = cache.get(&instance_id)?;
        self.registry
            .get()?
            .is_current(instance_id, entry.incarnation)
            .then(|| entry.payload.clone())
    }

    /// Snapshot of the cached module set for `instance_id`. Returns `None`
    /// when the entry hasn't been fetched yet (treat as "unknown — pass
    /// through to velo", not as "module absent").
    pub fn modules_for(&self, instance_id: InstanceId) -> Option<Vec<ModuleId>> {
        let cache = self.modules_cache.read().ok()?;
        let entry = cache.get(&instance_id)?;
        self.registry
            .get()?
            .is_current(instance_id, entry.incarnation)
            .then(|| entry.modules.clone())
    }

    /// Is `module` known to be enabled on `instance_id`?
    ///
    /// - `None` — cache miss (don't short-circuit, let velo answer).
    /// - `Some(true)` — known enabled.
    /// - `Some(false)` — known absent (Phase D routes return 404
    ///   `module_not_enabled` without touching velo).
    pub fn has_module(&self, instance_id: InstanceId, module: ModuleId) -> Option<bool> {
        self.modules_for(instance_id).map(|v| v.contains(&module))
    }
}

impl std::fmt::Debug for ControlPlaneManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let modules = self.modules_cache.read().map(|m| m.len()).unwrap_or(0);
        let describes = self.describe_cache.read().map(|m| m.len()).unwrap_or(0);
        f.debug_struct("ControlPlaneManager")
            .field("velo_attached", &self.velo.get().is_some())
            .field("registry_attached", &self.registry.get().is_some())
            .field("modules_cached", &modules)
            .field("describes_cached", &describes)
            .finish()
    }
}

impl FeatureManager for ControlPlaneManager {
    fn key(&self) -> FeatureKey {
        FeatureKey::ConnectorControl
    }

    fn attach<'a>(&'a self, ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async move {
            let _ = self.registry.set(ctx.registry.clone());
            if let Some(v) = ctx.velo.clone() {
                let self_id = v.instance_id();
                let messenger = v.messenger().clone();
                let _ = self.velo.set(v);
                let cache = Arc::clone(&self.modules_cache);
                let commit_registry = Arc::clone(&ctx.registry);
                let module_refresh = ModuleRefreshRuntime::new(
                    ctx.cancel.clone(),
                    Arc::clone(&ctx.registry),
                    self_id,
                    messenger,
                    move |target, modules| {
                        commit_modules_if_registered(
                            &cache,
                            Some(&commit_registry),
                            target.instance_id(),
                            target.incarnation(),
                            modules,
                        );
                    },
                );
                module_refresh.spawn_periodic();
                self.module_refresh.set(module_refresh).map_err(|_| {
                    FeatureError::Other(anyhow::anyhow!(
                        "control-plane module refresh runtime was already attached"
                    ))
                })?;
            } else {
                tracing::warn!(
                    "ControlPlaneManager: hub has no velo transport — \
                     control routes will return 503 and modules cache will stay empty"
                );
            }
            Ok(())
        })
    }

    fn on_register<'a>(
        &'a self,
        _instance_id: InstanceId,
        _feature: &'a Feature,
    ) -> BoxFuture<'a, Result<(), FeatureError>> {
        // No client-side `Feature::ConnectorControl` variant exists — this
        // manager only contributes routes. The hub's dispatcher will never
        // call this. Return a key-mismatch error if it ever does, which
        // surfaces as a clear bug.
        Box::pin(async move {
            Err(FeatureError::KeyMismatch {
                manager: FeatureKey::ConnectorControl,
                payload: _feature.key(),
            })
        })
    }

    fn on_register_any<'a>(
        &'a self,
        instance_id: InstanceId,
        _peer: &'a PeerInfo,
        incarnation: RegistryIncarnation,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            // Record the registration timestamp so `503 describe_pending`
            // can surface how long the leader has been registered without
            // pushing — even when velo is absent.
            if let Ok(mut w) = self.registered_at.write() {
                w.insert(instance_id, Instant::now());
            }
            let Some(module_refresh) = self.module_refresh.get() else {
                return;
            };
            // Skip the hub's own self-registration — no control handlers.
            // The HTTP register path won't trigger this for the hub today
            // (hub self-registers via the registry trait), but defensive
            // symmetry with the refresh loop is cheap.
            if self
                .velo
                .get()
                .is_some_and(|velo| instance_id == velo.instance_id())
            {
                return;
            }
            module_refresh.spawn_on_register(ModuleTarget::new(instance_id, incarnation));
        })
    }

    fn on_unregister(&self, instance_id: InstanceId) {
        if let Ok(mut w) = self.modules_cache.write() {
            w.remove(&instance_id);
        }
        if let Ok(mut w) = self.describe_cache.write() {
            w.remove(&instance_id);
        }
        if let Ok(mut w) = self.registered_at.write() {
            w.remove(&instance_id);
        }
    }

    fn control_router(self: Arc<Self>) -> Router {
        routes()
            .route(protocol::paths::RESET_FANOUT, post(reset_fanout))
            .with_state(self)
    }

    fn public_router(self: Arc<Self>) -> Router {
        routes().with_state(self)
    }
}

fn routes() -> Router<Arc<ControlPlaneManager>> {
    use protocol::paths::*;
    Router::new()
        .route(CONNECTOR_HEALTH, get(health_probe))
        .route(INSTANCE_MODULES, get(get_modules))
        .route(INSTANCE_DESCRIBE, get(get_describe).post(post_describe))
        // ---------- Typed control namespace ----------
        .route(CONTROL_CORE_DESCRIBE_INSTANCE, post(core_describe_instance))
        .route(CONTROL_DEV_RESET, post(dev_reset))
        // The `/control/transfer/*` routes moved to `P2pManager` — block copy
        // is a P2P concern, so the transfer surface only exists when the P2P
        // feature is enabled.
        .route(CONTROL_METRICS_SNAPSHOT, post(metrics_snapshot))
        .route(METRICS_FANOUT, get(metrics_fanout))
}

// ---------------------------------------------------------------------------
// Handlers — typed namespace
// ---------------------------------------------------------------------------

/// `POST /control/core/describe_instance` — pull a fresh
/// [`InstanceDescription`] from the leader via velo. Hub also updates the
/// describe cache as a side effect, so this shares the same code path as
/// `GET /describe?force=true` — operators can rely on either to refresh.
async fn core_describe_instance(
    State(mgr): State<Arc<ControlPlaneManager>>,
    Path(instance_id): Path<InstanceId>,
    _body: Option<Json<DescribeInstanceRequest>>,
) -> Response {
    let incarnation = match registered_incarnation(&mgr, instance_id) {
        Ok(incarnation) => incarnation,
        Err(response) => return response,
    };
    let client = match leader_client(&mgr, instance_id) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    match client.core().describe().await {
        Ok(payload) => {
            if let Err(resp) =
                validate_describe_layout(&mgr, instance_id, &payload, "core_describe_instance")
            {
                return *resp;
            }
            commit_describe_if_registered(
                &mgr.describe_cache,
                mgr.registry.get(),
                instance_id,
                incarnation,
                payload.clone(),
                DescribeSource::PullFallback,
            );
            ok_response(&payload)
        }
        Err(err) => control_error_response(instance_id, "describe_instance", err),
    }
}

/// `POST /control/dev/reset` — gated on [`ModuleId::Dev`]. Empty body is
/// equivalent to `ResetRequest::default()` (reset every configured tier).
async fn dev_reset(
    State(mgr): State<Arc<ControlPlaneManager>>,
    Path(instance_id): Path<InstanceId>,
    body: Option<Json<ResetRequest>>,
) -> Response {
    if let Some(resp) = gate(&mgr, instance_id, ModuleId::Dev) {
        return resp;
    }
    let client = match leader_client(&mgr, instance_id) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let req = body.map(|Json(r)| r).unwrap_or_default();
    match client.dev().reset(req).await {
        Ok(resp) => ok_response(&resp),
        Err(err) => control_error_response(instance_id, "reset", err),
    }
}

/// `POST /control/metrics/snapshot` — gated on [`ModuleId::Metrics`].
/// Empty body is equivalent to [`MetricsSnapshotRequest::default`].
async fn metrics_snapshot(
    State(mgr): State<Arc<ControlPlaneManager>>,
    Path(instance_id): Path<InstanceId>,
    body: Option<Json<MetricsSnapshotRequest>>,
) -> Response {
    if let Some(resp) = gate(&mgr, instance_id, ModuleId::Metrics) {
        return resp;
    }
    let client = match leader_client(&mgr, instance_id) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    let _req = body.map(|Json(r)| r).unwrap_or_default();
    match client.metrics().snapshot().await {
        Ok(resp) => ok_response(&resp),
        Err(err) => control_error_response(instance_id, "metrics_snapshot", err),
    }
}

/// `GET /v1/metrics` — fanout across every registered leader that has the
/// `metrics` module enabled. Per-leader failures surface as `{ "error": ... }`
/// entries in the response map rather than failing the whole request.
///
/// Leaders whose cached module set is `Some` but missing `ModuleId::Metrics`
/// are filtered out before the velo call so the request stays cheap with many
/// leaders. Leaders with `None` (cache miss) are queried — they will respond
/// with a `module_not_enabled` `ControlError` if the module is actually
/// absent, which the handler records as the per-instance `error`.
async fn metrics_fanout(State(mgr): State<Arc<ControlPlaneManager>>) -> Response {
    let Some(velo) = mgr.velo.get().cloned() else {
        return service_unavailable("hub has no velo transport configured");
    };
    let Some(registry) = mgr.registry.get().cloned() else {
        return service_unavailable("registry not attached");
    };
    let self_id = velo.instance_id();

    // Collect candidate ids: registered, not the hub, and not known-without-
    // the-metrics-module. Cache misses (`None`) are kept — velo will tell us.
    let candidates: Vec<InstanceId> = registry
        .list()
        .into_iter()
        .map(|p| p.instance_id())
        .filter(|id| *id != self_id)
        .filter(|id| !matches!(mgr.has_module(*id, ModuleId::Metrics), Some(false)))
        .collect();

    let messenger = velo.messenger().clone();
    // Each per-leader call is wrapped in `tokio::time::timeout` so a single
    // slow or hung leader can't stall the whole response. `join_all` only
    // waits for the slowest future — without this, the worst-case latency is
    // the worst-case leader latency, which the UI polls into every 5s.
    let calls = candidates.into_iter().map(|id| {
        let messenger = messenger.clone();
        async move {
            let client = LeaderControlClient::new(messenger, id);
            let outcome =
                tokio::time::timeout(METRICS_FANOUT_PER_LEADER, client.metrics().snapshot()).await;
            (id, outcome)
        }
    });
    let results = futures::future::join_all(calls).await;

    let mut instances: BTreeMap<String, MetricsInstanceEntry> = BTreeMap::new();
    for (id, outcome) in results {
        let entry = match outcome {
            Ok(Ok(snapshot)) => MetricsInstanceEntry {
                snapshot: Some(snapshot),
                error: None,
            },
            Ok(Err(err)) => {
                tracing::debug!(
                    instance = %id, kind = err.kind(), error = %err,
                    "metrics_fanout: leader returned error"
                );
                MetricsInstanceEntry {
                    snapshot: None,
                    error: Some(err.to_string()),
                }
            }
            Err(_elapsed) => {
                tracing::warn!(
                    instance = %id, budget = ?METRICS_FANOUT_PER_LEADER,
                    "metrics_fanout: leader did not respond within budget"
                );
                MetricsInstanceEntry {
                    snapshot: None,
                    error: Some(format!(
                        "timeout after {}s",
                        METRICS_FANOUT_PER_LEADER.as_secs()
                    )),
                }
            }
        };
        instances.insert(id.to_string(), entry);
    }

    let gathered_at_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    ok_response(&MetricsFanoutResponse {
        gathered_at_unix_ms,
        instances,
    })
}

/// `POST /v1/reset` — reset the requested tiers on registered leaders.
async fn reset_fanout(
    State(mgr): State<Arc<ControlPlaneManager>>,
    body: Option<Json<ResetRequest>>,
) -> Response {
    let Some(velo) = mgr.velo.get().cloned() else {
        return service_unavailable("hub has no velo transport configured");
    };
    let Some(registry) = mgr.registry.get().cloned() else {
        return service_unavailable("registry not attached");
    };
    let self_id = velo.instance_id();
    let req = body.map(|Json(request)| request).unwrap_or_default();

    let candidates: Vec<InstanceId> = registry
        .list()
        .into_iter()
        .map(|peer| peer.instance_id())
        .filter(|id| *id != self_id)
        .filter(|id| !matches!(mgr.has_module(*id, ModuleId::Dev), Some(false)))
        .collect();

    let messenger = velo.messenger().clone();
    let calls = candidates.into_iter().map(|id| {
        let messenger = messenger.clone();
        let req = req.clone();
        async move {
            let client = LeaderControlClient::new(messenger, id);
            let outcome =
                tokio::time::timeout(RESET_FANOUT_PER_LEADER, client.dev().reset(req)).await;
            (id, outcome)
        }
    });
    let results = futures::future::join_all(calls).await;

    let mut instances = BTreeMap::new();
    for (id, outcome) in results {
        let entry = match outcome {
            Ok(Ok(response)) => ResetInstanceEntry {
                response: Some(response),
                error: None,
            },
            Ok(Err(err)) => {
                tracing::debug!(
                    instance = %id, kind = err.kind(), error = %err,
                    "reset_fanout: leader returned error"
                );
                ResetInstanceEntry {
                    response: None,
                    error: Some(err.to_string()),
                }
            }
            Err(_elapsed) => {
                tracing::warn!(
                    instance = %id, budget = ?RESET_FANOUT_PER_LEADER,
                    "reset_fanout: leader did not respond within budget"
                );
                ResetInstanceEntry {
                    response: None,
                    error: Some(format!("timeout after {RESET_FANOUT_PER_LEADER:?}")),
                }
            }
        };
        instances.insert(id.to_string(), entry);
    }

    ok_response(&ResetFanoutResponse { instances })
}

/// Module-gating guard for the route handlers.
///
/// Returns `Some(404 module_not_enabled)` only when the modules cache has a
/// confirmed-absent entry for `module` on `instance_id`. Cache misses (i.e.
/// `has_module == None` — pre-discovery or never-populated) pass through
/// (Lesson #3) so an empty cache never short-circuits a legitimate call.
fn gate(mgr: &ControlPlaneManager, instance_id: InstanceId, module: ModuleId) -> Option<Response> {
    match mgr.has_module(instance_id, module) {
        Some(false) => Some(control_error_response(
            instance_id,
            "module_gate",
            ControlError::ModuleNotEnabled(module),
        )),
        _ => None,
    }
}

/// Health probe — not a `LeaderControlClient` call. Sends the hub's own
/// `_kvbm_hub_heartbeat` handler via velo to confirm the leader is reachable
/// at the active-messaging layer. Stays separate from the typed control
/// plane because heartbeat is hub↔peer infrastructure, not a control module.
async fn health_probe(
    State(mgr): State<Arc<ControlPlaneManager>>,
    Path(instance_id): Path<InstanceId>,
) -> Response {
    let Some(velo) = mgr.velo.get() else {
        return service_unavailable("hub has no velo transport configured");
    };
    let Some(registry) = mgr.registry.get() else {
        return service_unavailable("registry not attached");
    };
    if !registry.contains(instance_id) {
        return error_response(StatusCode::NOT_FOUND, "instance not registered");
    }
    let req = HeartbeatRequest { seq: 0 };
    let result: Result<HeartbeatAck, anyhow::Error> = async {
        let ack: HeartbeatAck = velo
            .typed_unary(HEARTBEAT_HANDLER)?
            .payload(&req)?
            .instance(instance_id)
            .send()
            .await?;
        Ok(ack)
    }
    .await;

    match result {
        Ok(ack) => json_response(
            StatusCode::OK,
            serde_json::json!({
                "velo_reachable": true,
                "ack_seq": ack.seq,
                "ack_ok": ack.ok,
            }),
        ),
        Err(e) => {
            tracing::warn!(instance = %instance_id, error = %e, "health probe failed");
            json_response(
                StatusCode::BAD_GATEWAY,
                serde_json::json!({
                    "velo_reachable": false,
                    "error": e.to_string(),
                }),
            )
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct ModulesQuery {
    /// When `true`, bypass the cache and re-fetch via velo even on a hit.
    /// The fresh result replaces the cached entry.
    #[serde(default)]
    force: bool,
}

/// `GET /v1/instances/{id}/modules` — return the cached module set, falling
/// back to an inline single-attempt fetch on miss (or always re-fetching when
/// `?force=true`). Body shape:
///
/// ```json
/// { "modules": ["core", "transfer"], "cached": true, "age_secs": 42 }
/// ```
async fn get_modules(
    State(mgr): State<Arc<ControlPlaneManager>>,
    Path(instance_id): Path<InstanceId>,
    Query(q): Query<ModulesQuery>,
) -> Response {
    // Serve from cache when present and not forced.
    if !q.force
        && let Ok(cache) = mgr.modules_cache.read()
        && let Some(entry) = cache.get(&instance_id)
        && mgr
            .registry
            .get()
            .is_some_and(|registry| registry.is_current(instance_id, entry.incarnation))
    {
        return json_response(
            StatusCode::OK,
            serde_json::json!({
                "modules": &entry.modules,
                "cached": true,
                "age_secs": entry.fetched_at.elapsed().as_secs(),
            }),
        );
    }

    // Miss (or force) — fetch inline (single attempt, no backoff).
    let incarnation = match registered_incarnation(&mgr, instance_id) {
        Ok(incarnation) => incarnation,
        Err(response) => return response,
    };
    let client = match leader_client(&mgr, instance_id) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    match client.list_modules().await {
        Ok(modules) => {
            // Populate the cache so subsequent reads are warm — guarded
            // against the unregister race in the same way as the background
            // tasks. If the instance unregistered while `list_modules` was
            // awaiting, drop the result on the floor and let the next read
            // re-discover.
            commit_modules_if_registered(
                &mgr.modules_cache,
                mgr.registry.get(),
                instance_id,
                incarnation,
                modules.clone(),
            );
            json_response(
                StatusCode::OK,
                serde_json::json!({
                    "modules": modules,
                    "cached": false,
                    "age_secs": 0,
                }),
            )
        }
        Err(err) => control_error_response(instance_id, "list_modules", err),
    }
}

#[derive(Debug, Default, Deserialize)]
struct DescribeQuery {
    /// When `true`, bypass the cache and pull fresh from the leader's velo
    /// `describe_instance` handler. The fresh result replaces the cached
    /// entry with `source = "pull_fallback"`.
    #[serde(default)]
    force: bool,
}

/// `POST /v1/instances/{id}/describe` — leader-initiated push of the typed
/// [`InstanceDescription`]. The hub stores the payload in its describe cache
/// after validating that the instance is still registered (race guard, same
/// pattern as the modules cache).
async fn post_describe(
    State(mgr): State<Arc<ControlPlaneManager>>,
    Path(instance_id): Path<InstanceId>,
    Json(payload): Json<InstanceDescription>,
) -> Response {
    let Some(registry) = mgr.registry.get() else {
        return service_unavailable("registry not attached");
    };
    let Some(incarnation) = registry.current_incarnation(instance_id) else {
        return error_response(StatusCode::NOT_FOUND, "instance not registered");
    };
    // Split-brain detection: if the leader reports a different hub
    // instance_id than the one this hub thinks it is, warn (but accept
    // the push). The leader's view comes from `set_hub_instance_id`,
    // which the connector populates from the value `register_with_hub`
    // returned; a mismatch indicates the leader registered with a hub
    // that has since restarted into a new identity.
    if let (Some(leader_view), Some(velo)) = (payload.hub_instance_id.as_deref(), mgr.velo.get()) {
        let me = velo.instance_id().to_string();
        if leader_view != me {
            tracing::warn!(
                instance = %instance_id,
                leader_view, me,
                "control_plane: leader pushed describe with mismatched hub_instance_id \
                 (split-brain?); accepting payload"
            );
        }
    }
    if let Err(resp) = validate_describe_layout(&mgr, instance_id, &payload, "post_describe") {
        return *resp;
    }
    commit_describe_if_registered(
        &mgr.describe_cache,
        Some(registry),
        instance_id,
        incarnation,
        payload,
        DescribeSource::Push,
    );
    json_response(StatusCode::OK, serde_json::json!({ "stored": true }))
}

/// `GET /v1/instances/{id}/describe` — return the cached [`InstanceDescription`].
///
/// Behaviour:
/// - Cache hit, `force` unset → return cached with `cached: true`,
///   `source` reflecting how the entry landed.
/// - Cache miss, `force` unset → **503** `describe_pending` with body
///   `{ error, kind, registered_secs_ago }`. UI shows a "loading" state.
/// - `force=true` (hit or miss) → pull via the velo `describe_instance`
///   handler, populate the cache with `source = "pull_fallback"`, return
///   the result with `cached: false`.
async fn get_describe(
    State(mgr): State<Arc<ControlPlaneManager>>,
    Path(instance_id): Path<InstanceId>,
    Query(q): Query<DescribeQuery>,
) -> Response {
    if !q.force
        && let Ok(cache) = mgr.describe_cache.read()
        && let Some(entry) = cache.get(&instance_id)
        && mgr
            .registry
            .get()
            .is_some_and(|registry| registry.is_current(instance_id, entry.incarnation))
    {
        let body = serde_json::json!({
            "description": &entry.payload,
            "cached": true,
            "age_secs": entry.received_at.elapsed().as_secs(),
            "source": entry.source,
        });
        return json_response(StatusCode::OK, body);
    }

    if !q.force {
        // Cache miss — surface a structured pending signal.
        let registered_secs_ago = mgr
            .registered_at
            .read()
            .ok()
            .and_then(|m| m.get(&instance_id).map(|t| t.elapsed().as_secs()));
        // Distinguish "instance not registered at all" from "registered but
        // hasn't pushed yet" by checking the registry first.
        let registry_says_known = mgr
            .registry
            .get()
            .map(|r| r.contains(instance_id))
            .unwrap_or(false);
        if !registry_says_known {
            return error_response(StatusCode::NOT_FOUND, "instance not registered");
        }
        return json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            serde_json::json!({
                "error": "describe not yet pushed by leader",
                "kind": "describe_pending",
                "registered_secs_ago": registered_secs_ago,
            }),
        );
    }

    // `force=true` — pull via velo.
    let incarnation = match registered_incarnation(&mgr, instance_id) {
        Ok(incarnation) => incarnation,
        Err(response) => return response,
    };
    let client = match leader_client(&mgr, instance_id) {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    match client.core().describe().await {
        Ok(payload) => {
            if let Err(resp) =
                validate_describe_layout(&mgr, instance_id, &payload, "get_describe_force")
            {
                return *resp;
            }
            commit_describe_if_registered(
                &mgr.describe_cache,
                mgr.registry.get(),
                instance_id,
                incarnation,
                payload.clone(),
                DescribeSource::PullFallback,
            );
            json_response(
                StatusCode::OK,
                serde_json::json!({
                    "description": payload,
                    "cached": false,
                    "age_secs": 0,
                    "source": DescribeSource::PullFallback,
                }),
            )
        }
        Err(err) => control_error_response(instance_id, "describe_instance", err),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// c5: validate a describe payload's `layout_compat` against the
/// `P2pManager` baseline. Common to every code path that lands an
/// `InstanceDescription` in the cache — `post_describe` (leader push),
/// `get_describe` with `force=true` (operator-triggered velo pull),
/// and `core_describe_instance` (typed velo pull endpoint). Without
/// this single source of truth a force-pull would stuff a divergent
/// payload past the c5 gate.
///
/// Returns `Ok(())` to proceed with `commit_describe_if_registered`,
/// or `Err(Response)` carrying the structured HTTP 400 rejection.
///
/// `None` payload → legacy / pre-stamping snapshot → skip (matches
/// the existing `block_size` / `parallelism` optionality contract).
///
/// `p2p_manager` not wired → P2P feature not enabled on this hub →
/// fall through with a warn so standalone hubs are unaffected.
fn validate_describe_layout(
    mgr: &Arc<ControlPlaneManager>,
    instance_id: InstanceId,
    payload: &InstanceDescription,
    call_site: &'static str,
) -> Result<(), Box<Response>> {
    let Some(candidate) = payload.layout_compat.as_ref() else {
        return Ok(());
    };
    let Some(p2p) = mgr.p2p_manager.get() else {
        tracing::warn!(
            instance = %instance_id,
            site = call_site,
            "describe: layout_compat present but P2pManager not wired; \
             skipping validation (P2P feature not enabled on this hub?)"
        );
        return Ok(());
    };
    p2p.check_describe_layout(instance_id, candidate)
        .map_err(|e| Box::new(error_response(StatusCode::BAD_REQUEST, &e.to_string())))
}

/// Insert a freshly fetched `modules` list into the cache **only** if the
/// target is still registered at the moment of insertion.
///
/// The registry check happens *while holding the cache write lock*, so it
/// serialises against [`FeatureManager::on_unregister`] (which removes the
/// entry under the same lock). Three outcomes:
///
/// 1. Eviction won the lock — registry says "absent", we return without
///    inserting. Cache stays clean.
/// 2. We won the lock — registry says "present", we insert. Eviction (if
///    any) blocks on the lock; when it gets it, it removes our entry.
/// 3. Eviction fan-out hasn't been triggered yet — we insert; a subsequent
///    unregister will pick it up via its own write-lock acquisition.
///
/// Without this check, a late `list_modules.await` reply can re-populate the
/// cache for an already-unregistered instance.
fn commit_modules_if_registered(
    cache: &RwLock<HashMap<InstanceId, ModulesEntry>>,
    registry: Option<&Arc<dyn PeerRegistry>>,
    instance_id: InstanceId,
    incarnation: RegistryIncarnation,
    modules: Vec<ModuleId>,
) {
    commit_if_registered(
        cache,
        registry,
        instance_id,
        incarnation,
        "list_modules",
        |_| ModulesEntry {
            modules,
            fetched_at: Instant::now(),
            incarnation,
        },
    );
}

/// Insert a freshly fetched describe payload into the cache, applying the
/// same registry-recheck race guard as the modules cache.
fn commit_describe_if_registered(
    cache: &RwLock<HashMap<InstanceId, DescribeEntry>>,
    registry: Option<&Arc<dyn PeerRegistry>>,
    instance_id: InstanceId,
    incarnation: RegistryIncarnation,
    payload: InstanceDescription,
    source: DescribeSource,
) {
    commit_if_registered(
        cache,
        registry,
        instance_id,
        incarnation,
        "describe",
        |_| DescribeEntry {
            payload,
            received_at: Instant::now(),
            source,
            incarnation,
        },
    );
}

/// Generic "commit a freshly-fetched value into a per-instance cache, but
/// only if the instance is still registered" helper. Both caches use the
/// same race-safe insert pattern (see [`commit_modules_if_registered`]
/// docstring for the original rationale).
///
/// The closure receives `instance_id` so it can capture-by-move (the closures
/// are `FnOnce`).
fn commit_if_registered<V, F>(
    cache: &RwLock<HashMap<InstanceId, V>>,
    registry: Option<&Arc<dyn PeerRegistry>>,
    instance_id: InstanceId,
    incarnation: RegistryIncarnation,
    label: &str,
    build_value: F,
) where
    F: FnOnce(InstanceId) -> V,
{
    let Ok(mut w) = cache.write() else { return };
    // No registry attached → no way to verify; refuse to insert. This only
    // happens during the narrow attach() window; production-time inserts
    // always have a registry.
    let Some(registry) = registry else { return };
    if !registry.is_current(instance_id, incarnation) {
        tracing::debug!(
            instance = %instance_id, %incarnation, what = label,
            "control_plane: late result dropped (registration incarnation changed)"
        );
        return;
    }
    w.insert(instance_id, build_value(instance_id));
}

#[allow(clippy::result_large_err)]
fn registered_incarnation(
    mgr: &ControlPlaneManager,
    instance_id: InstanceId,
) -> Result<RegistryIncarnation, Response> {
    let Some(registry) = mgr.registry.get() else {
        return Err(service_unavailable("registry not attached"));
    };
    registry
        .current_incarnation(instance_id)
        .ok_or_else(|| error_response(StatusCode::NOT_FOUND, "instance not registered"))
}

/// Build a `LeaderControlClient` for `instance_id`, validating velo + registry
/// attachment via the shared [`http::leader_client`](crate::features::http::leader_client)
/// helper. Thin wrapper that supplies this manager's own `OnceLock`s.
#[allow(clippy::result_large_err)]
fn leader_client(
    mgr: &ControlPlaneManager,
    instance_id: InstanceId,
) -> Result<LeaderControlClient, Response> {
    crate::features::http::leader_client(&mgr.velo, &mgr.registry, instance_id)
}
