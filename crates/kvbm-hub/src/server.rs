// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum-based HTTP server for the KVBM hub.
//!
//! Runs two listeners:
//!
//! - **Discovery port** (`1337` default) — serves only the `PeerDiscovery`
//!   HTTP surface. This is the port a velo client's [`HubClient`](crate::HubClient)
//!   hits for peer lookups.
//! - **Control port** (`8337` default) — serves the full control plane
//!   (registration, heartbeat, health) plus mirrored discovery for
//!   convenience.
//!
//! When one or more transports are supplied via
//! [`HubServerBuilder::add_transport`], the hub also participates in velo: it
//! builds an internal `velo::Velo`, self-registers in the registry, and can
//! push active messages (heartbeats, probes) to registered clients.

mod heartbeat;

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use velo_ext::{InstanceId, PeerInfo, WorkerId};

use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::handlers::{HEARTBEAT_HANDLER, HeartbeatAck, HeartbeatRequest};
use crate::protocol::{
    self, ErrorBody, ErrorCode, FeatureDescriptor, FeatureKey, HeartbeatResponse,
    HubConfigResponse, ListInstancesResponse, PeerLookupResponse, PrimaryConfig, ProbeResponse,
};
use crate::registration::{
    RegistrationCredentialError, RegistrationLifecycle, mutation_credential_from_headers,
    register_instance, unregister_instance,
};
use crate::registry::{InMemoryRegistry, PeerRegistry, RegistryError, RegistryIncarnation};
use heartbeat::spawn_heartbeat_task;

/// Default liveness TTL used by the in-memory registry.
pub const DEFAULT_REGISTRATION_TTL: Duration = Duration::from_secs(30);

/// Default reaper tick interval used by the in-memory registry.
pub const DEFAULT_PRUNE_INTERVAL: Duration = Duration::from_secs(10);

/// Default interval between hub-driven heartbeat probes.
pub const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);

/// Default consecutive probe failures before a registered instance is
/// unregistered by the heartbeat task.
pub const DEFAULT_HEARTBEAT_MAX_FAILURES: u32 = 3;

/// Shared hub server state (cheap to clone, all state is inside `Arc`s).
#[derive(Clone)]
pub struct HubServerState {
    registry: Arc<dyn PeerRegistry>,
    velo: Option<Arc<velo::Velo>>,
    managers: Arc<HashMap<FeatureKey, Arc<dyn FeatureManager>>>,
    registration_lifecycle: Arc<RegistrationLifecycle>,
    /// Hub-wide shared config served by `GET /v1/config` and used for
    /// must-match validation at registration.
    primary: Arc<PrimaryConfig>,
    /// Operator-supplied default connector config, served verbatim in
    /// `GET /v1/config`'s `base_config`. Sparse `kv_connector_extra_config` JSON
    /// (`{}` when no `--kvbm` overrides were given).
    base_config: Arc<serde_json::Value>,
}

impl std::fmt::Debug for HubServerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubServerState")
            .field("peers_count", &self.registry.list().len())
            .field("velo_attached", &self.velo.is_some())
            .field("feature_managers", &self.managers.len())
            .finish()
    }
}

impl Default for HubServerState {
    fn default() -> Self {
        Self::new()
    }
}

impl HubServerState {
    /// Create fresh, empty hub state backed by a default
    /// [`InMemoryRegistry`] and no velo participant.
    pub fn new() -> Self {
        let mem: Arc<InMemoryRegistry> = Arc::new(InMemoryRegistry::builder().build());
        let registry: Arc<dyn PeerRegistry> = mem;
        let managers = Arc::new(HashMap::new());
        let registration_lifecycle = RegistrationLifecycle::new(&managers);
        let registrations = registry
            .install_removal_hook(registration_lifecycle.removal_callback())
            .expect("fresh in-memory registry accepts its lifecycle hook");
        registration_lifecycle
            .synchronize_reservations(registrations)
            .expect("fresh registration state accepts an empty reservation snapshot");
        Self {
            registry,
            velo: None,
            managers,
            registration_lifecycle,
            primary: Arc::new(PrimaryConfig::default()),
            base_config: Arc::new(serde_json::json!({})),
        }
    }

    /// Snapshot the currently registered peers.
    pub fn peers(&self) -> Vec<PeerInfo> {
        self.registry.list()
    }

    /// Access the underlying registry (useful for tests / advanced usage).
    pub fn registry(&self) -> &Arc<dyn PeerRegistry> {
        &self.registry
    }

    /// Access the hub's Velo instance, if one was attached.
    pub fn velo(&self) -> Option<&Arc<velo::Velo>> {
        self.velo.as_ref()
    }

    pub(crate) fn managers(&self) -> &HashMap<FeatureKey, Arc<dyn FeatureManager>> {
        &self.managers
    }

    pub(crate) fn registration_lifecycle(&self) -> &RegistrationLifecycle {
        &self.registration_lifecycle
    }

    pub(crate) fn primary_config(&self) -> &PrimaryConfig {
        &self.primary
    }
}

/// Builder for [`HubServer`].
#[derive(Clone)]
pub struct HubServerBuilder {
    bind_addr: IpAddr,
    discovery_port: u16,
    control_port: u16,
    registry: Option<Arc<dyn PeerRegistry>>,
    transports: Vec<Arc<dyn velo::Transport>>,
    registration_ttl: Duration,
    prune_interval: Duration,
    heartbeat_interval: Duration,
    heartbeat_max_failures: u32,
    feature_managers: Vec<Arc<dyn FeatureManager>>,
    primary: PrimaryConfig,
    base_config: serde_json::Value,
}

impl std::fmt::Debug for HubServerBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubServerBuilder")
            .field("bind_addr", &self.bind_addr)
            .field("discovery_port", &self.discovery_port)
            .field("control_port", &self.control_port)
            .field("transports", &self.transports.len())
            .field("registry_injected", &self.registry.is_some())
            .field("registration_ttl", &self.registration_ttl)
            .field("prune_interval", &self.prune_interval)
            .field("feature_managers", &self.feature_managers.len())
            .finish()
    }
}

impl Default for HubServerBuilder {
    fn default() -> Self {
        Self {
            bind_addr: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            discovery_port: protocol::DEFAULT_DISCOVERY_PORT,
            control_port: protocol::DEFAULT_CONTROL_PORT,
            registry: None,
            transports: Vec::new(),
            registration_ttl: DEFAULT_REGISTRATION_TTL,
            prune_interval: DEFAULT_PRUNE_INTERVAL,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            heartbeat_max_failures: DEFAULT_HEARTBEAT_MAX_FAILURES,
            feature_managers: Vec::new(),
            primary: PrimaryConfig::default(),
            base_config: serde_json::json!({}),
        }
    }
}

impl HubServerBuilder {
    /// New builder with default bind `0.0.0.0` and default ports.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind address (default `0.0.0.0`).
    pub fn bind_addr(mut self, addr: IpAddr) -> Self {
        self.bind_addr = addr;
        self
    }

    /// Discovery HTTP port (default `1337`).
    pub fn discovery_port(mut self, port: u16) -> Self {
        self.discovery_port = port;
        self
    }

    /// Control-plane HTTP port (default `8337`).
    pub fn control_port(mut self, port: u16) -> Self {
        self.control_port = port;
        self
    }

    /// Inject a custom `PeerRegistry` backend (etcd, consul, ...). If unset,
    /// the hub uses a default in-memory registry and spawns its own reaper.
    ///
    /// Custom backends are expected to manage their own liveness (e.g. etcd
    /// leases); the hub will not protect any self-entry on them.
    pub fn registry(mut self, r: Arc<dyn PeerRegistry>) -> Self {
        self.registry = Some(r);
        self
    }

    /// Attach a velo transport. When at least one transport is supplied, the
    /// hub builds an internal `velo::Velo` and participates in active
    /// messaging with registered clients.
    pub fn add_transport(mut self, transport: Arc<dyn velo::Transport>) -> Self {
        self.transports.push(transport);
        self
    }

    /// Override the liveness TTL used by the default in-memory registry.
    /// Ignored when a custom registry is injected via [`registry`](Self::registry).
    pub fn registration_ttl(mut self, d: Duration) -> Self {
        self.registration_ttl = d;
        self
    }

    /// Override the reaper tick interval used by the default in-memory
    /// registry. Ignored when a custom registry is injected.
    pub fn prune_interval(mut self, d: Duration) -> Self {
        self.prune_interval = d;
        self
    }

    /// Override the hub-driven heartbeat probe interval. Ignored when no
    /// velo transport is configured.
    pub fn heartbeat_interval(mut self, d: Duration) -> Self {
        self.heartbeat_interval = d;
        self
    }

    /// Override the consecutive-failure threshold before the heartbeat
    /// task unregisters an instance.
    pub fn heartbeat_max_failures(mut self, n: u32) -> Self {
        self.heartbeat_max_failures = n;
        self
    }

    /// Attach a [`FeatureManager`] to the hub. Each manager contributes axum
    /// routes to both listeners and receives register/unregister dispatch
    /// for its [`FeatureKey`]. Duplicate keys cause [`serve`](Self::serve) to
    /// fail.
    pub fn add_feature_manager(mut self, mgr: Arc<dyn FeatureManager>) -> Self {
        self.feature_managers.push(mgr);
        self
    }

    /// Set the hub-wide shared [`PrimaryConfig`] served by `GET /v1/config` and
    /// validated against every registrant's must-match summary. Defaults to
    /// [`PrimaryConfig::default`] (no authoritative fields → validation skipped).
    pub fn primary_config(mut self, primary: PrimaryConfig) -> Self {
        self.primary = primary;
        self
    }

    /// Set the operator-supplied default connector config served verbatim as
    /// `GET /v1/config`'s `base_config`. Expected to be a sparse
    /// `kv_connector_extra_config`-shaped JSON object (the binary builds it from
    /// `--kvbm` / `--kvbm-config` and validates it). Defaults to `{}`.
    pub fn base_kvbm_config(mut self, base_config: serde_json::Value) -> Self {
        self.base_config = base_config;
        self
    }

    /// Bind both listeners and spawn them. Returns a running [`HubServer`].
    pub async fn serve(self) -> Result<HubServer> {
        // Duplicate-key guard — enforced once at startup.
        let mut managers: HashMap<FeatureKey, Arc<dyn FeatureManager>> = HashMap::new();
        for mgr in &self.feature_managers {
            let key = mgr.key();
            if managers.insert(key, Arc::clone(mgr)).is_some() {
                return Err(anyhow::anyhow!(
                    "duplicate FeatureManager registered for key {key:?}"
                ));
            }
        }

        // Reconcile feature-owned authoritative sizing into `primary`. This
        // makes `primary` the guaranteed source of truth for must-match
        // validation: a feature that owns sizing (e.g. KV-index) fills unset
        // primary fields and any explicit primary that disagrees is a config
        // error. Without this, a library hub built with a sizing-owning
        // feature but no `primary_config` would leave the must-match check
        // with nothing to validate against (a registration bypass).
        let mut primary = self.primary;
        for (key, mgr) in managers.iter() {
            let Some(block_size) = mgr.authoritative_block_size() else {
                continue;
            };
            match primary.block_size {
                Some(existing) if existing != block_size => {
                    return Err(anyhow::anyhow!(
                        "primary.block_size ({existing}) conflicts with {key:?} \
                         feature block_size ({block_size})"
                    ));
                }
                _ => primary.block_size = Some(block_size),
            }
        }

        // Keep a concrete handle to `InMemoryRegistry` only long enough to
        // protect the hub's own Velo entry. Lifecycle revocation is installed
        // uniformly through the `PeerRegistry` trait below.
        let (registry, mem_concrete): (Arc<dyn PeerRegistry>, Option<Arc<InMemoryRegistry>>) =
            match self.registry {
                Some(r) => (r, None),
                None => {
                    let mem: Arc<InMemoryRegistry> = Arc::new(
                        InMemoryRegistry::builder()
                            .ttl(self.registration_ttl)
                            .prune_interval(self.prune_interval)
                            .build(),
                    );
                    let dyn_reg: Arc<dyn PeerRegistry> = mem.clone();
                    (dyn_reg, Some(mem))
                }
            };

        // Bind both public sockets before creating Velo, registering the hub,
        // or starting feature work. A bad address is a configuration error and
        // must leave injected registries and managers untouched.
        let discovery_addr = SocketAddr::new(self.bind_addr, self.discovery_port);
        let control_addr = SocketAddr::new(self.bind_addr, self.control_port);
        let discovery_listener = TcpListener::bind(discovery_addr)
            .await
            .with_context(|| format!("binding discovery port {discovery_addr}"))?;
        let control_listener = TcpListener::bind(control_addr)
            .await
            .with_context(|| format!("binding control port {control_addr}"))?;
        let discovery_local = discovery_listener
            .local_addr()
            .context("discovery local_addr")?;
        let control_local = control_listener
            .local_addr()
            .context("control local_addr")?;

        // Build the hub's own Velo if any transports were supplied.
        let (velo, self_registration) = if !self.transports.is_empty() {
            let discovery: Arc<dyn velo::discovery::PeerDiscovery> = registry.clone();
            let mut vb = velo::Velo::builder().discovery(discovery);
            for t in self.transports {
                vb = vb.add_transport(t);
            }
            let v = Arc::new(vb.build().await.context("building hub velo")?);
            // Self-register so clients can discover the hub via
            // `GET /v1/peers/instance/{hub_id}`.
            let incarnation = registry
                .register(v.peer_info())
                .await
                .map_err(|e| anyhow::anyhow!("hub self-register: {e}"))?;
            if let Some(mem) = &mem_concrete {
                mem.protect(v.instance_id());
            }
            let self_registration = Some((v.instance_id(), incarnation));
            (Some(v), self_registration)
        } else {
            (None, None)
        };

        // Create the master shutdown token *before* attaching managers so
        // they can fork child tokens for any background work they spawn
        // during attach (refresh tasks, watchers).
        let cancel = CancellationToken::new();

        let managers = Arc::new(managers);
        let registration_lifecycle = RegistrationLifecycle::new(&managers);

        // Attach every manager now that the registry and (optional) Velo
        // are ready.
        let ctx = HubContext {
            velo: velo.clone(),
            registry: registry.clone(),
            cancel: cancel.child_token(),
        };
        for (key, mgr) in managers.iter() {
            if let Err(error) = mgr.attach(ctx.clone()).await {
                let error = anyhow::anyhow!("FeatureManager({key:?}) attach: {error}");
                return Err(abort_startup(&registry, self_registration, &cancel, error).await);
            }
        }

        // Installing a custom registry's authoritative hook is irreversible.
        // Defer it until every fallible startup step has succeeded so a caller
        // can correct a bind or manager-attach failure and reuse the registry.
        let registrations =
            match registry.install_removal_hook(registration_lifecycle.removal_callback()) {
                Ok(registrations) => registrations,
                Err(error) => {
                    let error = anyhow::anyhow!("installing registry removal hook: {error}");
                    return Err(abort_startup(&registry, self_registration, &cancel, error).await);
                }
            };
        if let Err(error) = registration_lifecycle.synchronize_reservations(registrations) {
            let error = anyhow::anyhow!("reserving existing registry occupants: {error}");
            return Err(abort_startup(&registry, self_registration, &cancel, error).await);
        }

        let reaper_task = registry.clone().spawn_liveness_task(cancel.child_token());

        // Spawn the hub-driven heartbeat task only when velo is configured.
        // Done before `velo` is moved into `HubServerState` below.
        let heartbeat_task = velo.as_ref().map(|v| {
            spawn_heartbeat_task(
                Arc::clone(v),
                registry.clone(),
                v.instance_id(),
                self.heartbeat_interval,
                self.heartbeat_max_failures,
                cancel.child_token(),
            )
        });

        let state = HubServerState {
            registry: registry.clone(),
            velo,
            managers: Arc::clone(&managers),
            registration_lifecycle,
            primary: Arc::new(primary),
            base_config: Arc::new(self.base_config),
        };
        if heartbeat_task.is_none() {
            tracing::info!(
                "hub heartbeat task disabled (no velo transport configured); \
                 instances rely on TTL-based reaping only"
            );
        } else {
            tracing::info!(
                interval_secs = self.heartbeat_interval.as_secs(),
                max_failures = self.heartbeat_max_failures,
                "hub heartbeat task started"
            );
        }

        let mut discovery_router = discovery_router(state.clone());
        let mut control_router = control_router(state.clone());
        for mgr in managers.values() {
            let public = Arc::clone(mgr).public_router();
            let control = Arc::clone(mgr).control_router();
            match mgr.route_prefix() {
                // Feature owns a namespace: nest its relative routes under it.
                Some(seg) => {
                    let base = format!("/v1/features/{seg}");
                    discovery_router = discovery_router.nest(&base, public);
                    control_router = control_router.nest(&base, control);
                }
                // Legacy: manager declares full absolute paths itself.
                None => {
                    discovery_router = discovery_router.merge(public);
                    control_router = control_router.merge(control);
                }
            }
        }
        // Phase E — embedded operator UI mounted on the control listener.
        // Same-origin so the SPA's fetches need no CORS.
        control_router = control_router.merge(crate::web::ui_router());

        let discovery_task = spawn_server(discovery_listener, discovery_router, cancel.clone());
        let control_task = spawn_server(control_listener, control_router, cancel.clone());

        Ok(HubServer {
            state,
            discovery_addr: discovery_local,
            control_addr: control_local,
            cancel,
            discovery_task: Some(discovery_task),
            control_task: Some(control_task),
            reaper_task,
            heartbeat_task,
            self_registration,
        })
    }
}

async fn abort_startup(
    registry: &Arc<dyn PeerRegistry>,
    self_registration: Option<(InstanceId, RegistryIncarnation)>,
    cancel: &CancellationToken,
    startup_error: anyhow::Error,
) -> anyhow::Error {
    cancel.cancel();
    let Some((instance_id, incarnation)) = self_registration else {
        return startup_error;
    };
    match registry.unregister(instance_id, incarnation).await {
        Ok(()) | Err(RegistryError::NotFound(_)) | Err(RegistryError::StaleIncarnation { .. }) => {
            startup_error
        }
        Err(cleanup_error) => anyhow::anyhow!(
            "{startup_error}; additionally failed to remove hub self-registration: {cleanup_error}"
        ),
    }
}

/// A running hub server.
///
/// Drop or call [`shutdown`](Self::shutdown) to cancel both listeners and
/// wait for them to terminate.
pub struct HubServer {
    state: HubServerState,
    discovery_addr: SocketAddr,
    control_addr: SocketAddr,
    cancel: CancellationToken,
    discovery_task: Option<JoinHandle<()>>,
    control_task: Option<JoinHandle<()>>,
    reaper_task: Option<JoinHandle<()>>,
    heartbeat_task: Option<JoinHandle<()>>,
    self_registration: Option<(InstanceId, RegistryIncarnation)>,
}

impl std::fmt::Debug for HubServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubServer")
            .field("discovery_addr", &self.discovery_addr)
            .field("control_addr", &self.control_addr)
            .finish()
    }
}

impl HubServer {
    /// Builder entry point.
    pub fn builder() -> HubServerBuilder {
        HubServerBuilder::new()
    }

    /// Resolved discovery socket address (useful when binding port `0`).
    pub fn discovery_addr(&self) -> SocketAddr {
        self.discovery_addr
    }

    /// Resolved control socket address.
    pub fn control_addr(&self) -> SocketAddr {
        self.control_addr
    }

    /// Shared state handle.
    pub fn state(&self) -> &HubServerState {
        &self.state
    }

    /// Trigger shutdown and await both listeners plus the reaper.
    pub async fn shutdown(mut self) -> Result<()> {
        self.cancel.cancel();
        if let Some(t) = self.discovery_task.take() {
            let _ = t.await;
        }
        if let Some(t) = self.control_task.take() {
            let _ = t.await;
        }
        if let Some(t) = self.reaper_task.take() {
            let _ = t.await;
        }
        if let Some(t) = self.heartbeat_task.take() {
            let _ = t.await;
        }
        self.remove_self_registration().await
    }

    async fn remove_self_registration(&mut self) -> Result<()> {
        let Some((instance_id, incarnation)) = self.self_registration.take() else {
            return Ok(());
        };
        match self
            .state
            .registry
            .unregister(instance_id, incarnation)
            .await
        {
            Ok(())
            | Err(RegistryError::NotFound(_))
            | Err(RegistryError::StaleIncarnation { .. }) => Ok(()),
            Err(error) => Err(anyhow::anyhow!(
                "removing hub self-registration {instance_id}: {error}"
            )),
        }
    }
}

impl Drop for HubServer {
    fn drop(&mut self) {
        self.cancel.cancel();
        let Some((instance_id, incarnation)) = self.self_registration.take() else {
            return;
        };
        let registry = Arc::clone(&self.state.registry);
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = registry.unregister(instance_id, incarnation).await;
            });
        }
    }
}

fn spawn_server(
    listener: TcpListener,
    router: Router,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let serve = axum::serve(listener, router).with_graceful_shutdown(async move {
            cancel.cancelled().await;
        });
        if let Err(e) = serve.await {
            tracing::error!(error = %e, "kvbm-hub listener exited with error");
        }
    })
}

fn discovery_router(state: HubServerState) -> Router {
    Router::new()
        .route(
            protocol::paths::PEERS_BY_INSTANCE,
            get(get_peer_by_instance),
        )
        .route(protocol::paths::PEERS_BY_WORKER, get(get_peer_by_worker))
        .route(protocol::paths::HEALTH, get(health))
        .route(protocol::paths::HUB_CONFIG, get(get_hub_config))
        .with_state(state)
}

fn control_router(state: HubServerState) -> Router {
    Router::new()
        .route(
            protocol::paths::INSTANCES,
            get(list_instances).post(register_instance),
        )
        .route(protocol::paths::INSTANCE_BY_ID, delete(unregister_instance))
        .route(protocol::paths::INSTANCE_HEARTBEAT, post(heartbeat))
        .route(protocol::paths::INSTANCE_PROBE, post(probe_instance))
        // Discovery endpoints are mirrored here for convenience.
        .route(
            protocol::paths::PEERS_BY_INSTANCE,
            get(get_peer_by_instance),
        )
        .route(protocol::paths::PEERS_BY_WORKER, get(get_peer_by_worker))
        .route(protocol::paths::HEALTH, get(health))
        .route(protocol::paths::HUB_CONFIG, get(get_hub_config))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health() -> &'static str {
    "ok"
}

/// `GET /v1/config` — the aggregate config the connector and `kvbmctl` consume.
/// Reports the hub's `primary` config plus one [`FeatureDescriptor`] per
/// attached feature manager (the hub's advertised capability set).
async fn get_hub_config(State(state): State<HubServerState>) -> Json<HubConfigResponse> {
    let primary = (*state.primary).clone();
    let mut features: Vec<FeatureDescriptor> = state
        .managers
        .values()
        .map(|mgr| FeatureDescriptor {
            key: mgr.key(),
            dependencies: mgr.dependencies().to_vec(),
            render_implies: mgr.render_implies().to_vec(),
            config_requirements: mgr.config_requirements(),
            config: mgr.descriptor(&primary),
        })
        .collect();
    // Deterministic order for stable responses / tests.
    features.sort_by(|a, b| a.key.as_str().cmp(b.key.as_str()));
    Json(HubConfigResponse {
        primary,
        features,
        base_config: (*state.base_config).clone(),
    })
}

async fn list_instances(State(state): State<HubServerState>) -> Json<ListInstancesResponse> {
    Json(ListInstancesResponse {
        instances: state.peers(),
    })
}

async fn probe_instance(
    State(state): State<HubServerState>,
    Path(instance_id): Path<InstanceId>,
) -> Result<Json<ProbeResponse>, HubError> {
    if !state.registry.contains(instance_id) {
        return Err(HubError::not_found(format!(
            "instance {instance_id} not registered"
        )));
    }

    let velo = state
        .velo
        .as_ref()
        .ok_or_else(|| HubError::internal("hub velo not configured".to_string()))?;

    let ack: HeartbeatAck = velo
        .typed_unary(HEARTBEAT_HANDLER)
        .map_err(|e| HubError::bad_gateway(format!("probe setup: {e}")))?
        .payload(&HeartbeatRequest { seq: 0 })
        .map_err(|e| HubError::bad_gateway(format!("probe payload: {e}")))?
        .instance(instance_id)
        .send()
        .await
        .map_err(|e| HubError::bad_gateway(format!("probe failed: {e}")))?;

    Ok(Json(ProbeResponse {
        seq: ack.seq,
        ok: ack.ok,
    }))
}

async fn heartbeat(
    State(state): State<HubServerState>,
    Path(instance_id): Path<InstanceId>,
    headers: HeaderMap,
) -> Result<Json<HeartbeatResponse>, HubError> {
    let presented_credential = mutation_credential_from_headers(&headers)
        .map_err(|error| HubError::bad_request(error.to_string()))?;
    let incarnation = state
        .registration_lifecycle
        .credentials()
        .authorize(instance_id, presented_credential.as_ref())
        .map_err(HubError::from_registration_credential)?;
    state
        .registry
        .touch(instance_id, incarnation)
        .await
        .map_err(HubError::from_registry)?;
    Ok(Json(HeartbeatResponse { acknowledged: true }))
}

async fn get_peer_by_instance(
    State(state): State<HubServerState>,
    Path(instance_id): Path<InstanceId>,
) -> Result<Json<PeerLookupResponse>, HubError> {
    state
        .registry
        .discover_by_instance_id(instance_id)
        .await
        .map(|peer_info| Json(PeerLookupResponse { peer_info }))
        .map_err(|_| HubError::not_found(format!("instance {instance_id} not found")))
}

async fn get_peer_by_worker(
    State(state): State<HubServerState>,
    Path(worker_id): Path<u64>,
) -> Result<Json<PeerLookupResponse>, HubError> {
    let wid = WorkerId::from_u64(worker_id);
    state
        .registry
        .discover_by_worker_id(wid)
        .await
        .map(|peer_info| Json(PeerLookupResponse { peer_info }))
        .map_err(|_| HubError::not_found(format!("worker {worker_id} not found")))
}

// ---------------------------------------------------------------------------
// Error plumbing
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct HubError {
    status: StatusCode,
    body: ErrorBody,
}

impl HubError {
    pub(crate) fn from_registry(e: RegistryError) -> Self {
        match e {
            RegistryError::Conflict { .. } => Self::conflict(e.to_string()),
            RegistryError::NotFound(_) => Self::not_found(e.to_string()),
            RegistryError::Backend(err) => Self::internal(format!("registry backend: {err}")),
            RegistryError::RemovalHookAlreadyInstalled => {
                Self::internal("peer-removal hook is already installed".to_string())
            }
            RegistryError::StaleIncarnation { .. } => Self::conflict(e.to_string()),
        }
    }

    pub(crate) fn from_feature(e: FeatureError) -> Self {
        match e {
            FeatureError::InvalidConfig(m) => Self::bad_request(m),
            FeatureError::KeyMismatch { .. } => Self::internal(e.to_string()),
            FeatureError::Other(err) => Self::internal(err.to_string()),
        }
    }

    pub(crate) fn from_registration_credential(e: RegistrationCredentialError) -> Self {
        match e {
            RegistrationCredentialError::Unauthorized { .. } => Self::unauthorized(e.to_string()),
            RegistrationCredentialError::Busy { .. } => Self::conflict(e.to_string()),
            RegistrationCredentialError::NotFound { .. } => Self::not_found(e.to_string()),
            RegistrationCredentialError::StateChanged { .. }
            | RegistrationCredentialError::Unavailable => Self::internal(e.to_string()),
        }
    }

    fn not_found(message: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            body: ErrorBody {
                code: ErrorCode::NotFound,
                message,
            },
        }
    }

    pub(crate) fn bad_request(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorBody {
                code: ErrorCode::BadRequest,
                message,
            },
        }
    }

    fn unauthorized(message: String) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            body: ErrorBody {
                code: ErrorCode::Unauthorized,
                message,
            },
        }
    }

    fn conflict(message: String) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            body: ErrorBody {
                code: ErrorCode::Conflict,
                message,
            },
        }
    }

    pub(crate) fn internal(message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ErrorBody {
                code: ErrorCode::Internal,
                message,
            },
        }
    }

    fn bad_gateway(message: String) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            body: ErrorBody {
                code: ErrorCode::Internal,
                message,
            },
        }
    }
}

impl IntoResponse for HubError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hub_server_state_starts_empty() {
        assert!(HubServerState::new().peers().is_empty());
    }

    #[test]
    fn hub_server_state_default_starts_empty() {
        assert!(HubServerState::default().peers().is_empty());
    }

    #[tokio::test]
    async fn builder_binds_os_assigned_ports() {
        let server = HubServerBuilder::new()
            .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
            .discovery_port(0)
            .control_port(0)
            .serve()
            .await
            .unwrap();
        assert_ne!(server.discovery_addr().port(), 0);
        assert_ne!(server.control_addr().port(), 0);
        assert_ne!(server.discovery_addr().port(), server.control_addr().port());
    }

    #[tokio::test]
    async fn server_entry_point_builder() {
        let server = HubServer::builder()
            .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
            .discovery_port(0)
            .control_port(0)
            .serve()
            .await
            .unwrap();
        assert_eq!(server.state().peers().len(), 0);
        server.shutdown().await.unwrap();
    }
}
