//! The kvbm hub: a velo-hub [`Hub`] serving the kvbm extensions.
//!
//! velo-hub owns discovery, registration, liveness and the two listeners.
//! This module adds the `kvbm` core extension, which serves the aggregate
//! config (`GET /v1/ext/kvbm/config`) and checks each registrant's runtime
//! summary against the features it opts into.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::{Json, Router, extract::State, routing::get};
use velo_hub::store::Removal;
use velo_hub::{
    Committed, Extension, Hub, HubBuilder, RegistrationView, Rejection, ServerConfig, Txn,
};

use crate::features::{FeatureConfigRequirements, KvbmFeature};
use crate::protocol::{
    self, FeatureDescriptor, FeatureKey, HubConfigResponse, PrimaryConfig, RuntimeConfigSummary,
};

/// Default registration lease TTL.
pub const DEFAULT_REGISTRATION_TTL: Duration = Duration::from_secs(30);

/// Default interval between expired-lease sweeps.
pub const DEFAULT_PRUNE_INTERVAL: Duration = Duration::from_secs(10);

/// Default interval between the hub's velo heartbeat probes.
pub const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);

/// Consecutive failed probes before the hub removes a registrant.
pub const DEFAULT_HEARTBEAT_MAX_FAILURES: u32 = 3;

type Install = Box<dyn FnOnce(HubBuilder) -> HubBuilder + Send>;

/// What the core extension knows about one feature extension.
struct FeatureEntry {
    key: FeatureKey,
    dependencies: Vec<FeatureKey>,
    render_implies: Vec<FeatureKey>,
    config_requirements: FeatureConfigRequirements,
    requires_runtime_summary: bool,
    descriptor: Box<dyn Fn() -> serde_json::Value + Send + Sync>,
}

fn keys(labels: &[&str]) -> Vec<FeatureKey> {
    labels
        .iter()
        .filter_map(|label| FeatureKey::from_label(label))
        .collect()
}

/// Builder for [`HubServer`].
pub struct HubServerBuilder {
    server: ServerConfig,
    transports: Vec<Arc<dyn velo::Transport>>,
    primary: PrimaryConfig,
    base_config: serde_json::Value,
    features: Vec<FeatureEntry>,
    authoritative_block_sizes: Vec<(FeatureKey, usize)>,
    installs: Vec<Install>,
}

impl std::fmt::Debug for HubServerBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubServerBuilder")
            .field("server", &self.server)
            .field("transports", &self.transports.len())
            .field("primary", &self.primary)
            .field(
                "features",
                &self.features.iter().map(|f| f.key).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Default for HubServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HubServerBuilder {
    /// A hub on `0.0.0.0` with the default kvbm ports and no velo transport.
    pub fn new() -> Self {
        let server = ServerConfig {
            bind_addr: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            public_port: protocol::DEFAULT_DISCOVERY_PORT,
            control_port: protocol::DEFAULT_CONTROL_PORT,
            ttl: DEFAULT_REGISTRATION_TTL,
            prune_interval: DEFAULT_PRUNE_INTERVAL,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            heartbeat_max_failures: DEFAULT_HEARTBEAT_MAX_FAILURES,
            ..ServerConfig::default()
        };
        Self {
            server,
            transports: Vec::new(),
            primary: PrimaryConfig::default(),
            base_config: serde_json::Value::Object(Default::default()),
            features: Vec::new(),
            authoritative_block_sizes: Vec::new(),
            installs: Vec::new(),
        }
    }

    pub fn bind_addr(mut self, addr: IpAddr) -> Self {
        self.server.bind_addr = addr;
        self
    }

    /// Port of the public (read-only) listener. `0` picks a free port.
    pub fn discovery_port(mut self, port: u16) -> Self {
        self.server.public_port = port;
        self
    }

    /// Port of the control listener. `0` picks a free port.
    pub fn control_port(mut self, port: u16) -> Self {
        self.server.control_port = port;
        self
    }

    /// Give the hub a velo instance on this transport. Without one, the hub
    /// serves HTTP only: no heartbeats and no velo-backed features.
    pub fn add_transport(mut self, transport: Arc<dyn velo::Transport>) -> Self {
        self.transports.push(transport);
        self
    }

    pub fn registration_ttl(mut self, ttl: Duration) -> Self {
        self.server.ttl = ttl;
        self
    }

    pub fn prune_interval(mut self, interval: Duration) -> Self {
        self.server.prune_interval = interval;
        self
    }

    pub fn heartbeat_interval(mut self, interval: Duration) -> Self {
        self.server.heartbeat_interval = interval;
        self
    }

    pub fn heartbeat_max_failures(mut self, failures: u32) -> Self {
        self.server.heartbeat_max_failures = failures;
        self
    }

    /// Hub-wide sizing that must-match features check registrants against.
    pub fn primary_config(mut self, primary: PrimaryConfig) -> Self {
        self.primary = primary;
        self
    }

    /// Base kvbm config served in the aggregate config.
    pub fn base_kvbm_config(mut self, base_config: serde_json::Value) -> Self {
        self.base_config = base_config;
        self
    }

    /// Serve a kvbm feature extension.
    ///
    /// # Panics
    ///
    /// If `E::NAME` is not a [`FeatureKey`] label.
    pub fn extension<E: KvbmFeature>(mut self, feature: Arc<E>) -> Self {
        let key = FeatureKey::from_label(E::NAME)
            .unwrap_or_else(|| panic!("extension {:?} is not a kvbm feature key", E::NAME));
        if let Some(block_size) = feature.authoritative_block_size() {
            self.authoritative_block_sizes.push((key, block_size));
        }
        let described = Arc::clone(&feature);
        self.features.push(FeatureEntry {
            key,
            dependencies: keys(E::registration_requires()),
            render_implies: keys(feature.render_implies()),
            config_requirements: feature.config_requirements(),
            requires_runtime_summary: feature.requires_runtime_summary(),
            descriptor: Box::new(move || described.descriptor()),
        });
        self.installs.push(Box::new(move |builder: HubBuilder| {
            builder.extension(feature)
        }));
        self
    }

    pub async fn serve(self) -> Result<HubServer> {
        let mut primary = self.primary;
        for (key, block_size) in self.authoritative_block_sizes {
            match primary.block_size {
                Some(existing) if existing != block_size => anyhow::bail!(
                    "primary.block_size ({existing}) conflicts with {key:?} \
                     feature block_size ({block_size})"
                ),
                Some(_) => {}
                None => primary.block_size = Some(block_size),
            }
        }
        let mut features = self.features;
        features.sort_by(|a, b| a.key.as_str().cmp(b.key.as_str()));
        let core = KvbmCore {
            primary,
            base_config: self.base_config,
            features,
        };

        let config = velo_hub::HubConfig {
            server: self.server,
            ..Default::default()
        };
        let mut builder = Hub::builder(config).extension(core);
        if !self.transports.is_empty() {
            let mut velo = velo::Velo::builder();
            for transport in self.transports {
                velo = velo.add_transport(transport);
            }
            builder = builder.velo(velo);
        }
        for install in self.installs {
            builder = install(builder);
        }
        let hub = builder.serve().await.context("starting kvbm hub")?;
        Ok(HubServer { hub })
    }
}

/// A running kvbm hub.
pub struct HubServer {
    hub: Hub,
}

impl std::fmt::Debug for HubServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubServer")
            .field("discovery_addr", &self.discovery_addr())
            .field("control_addr", &self.control_addr())
            .finish()
    }
}

impl HubServer {
    pub fn builder() -> HubServerBuilder {
        HubServerBuilder::new()
    }

    /// Address of the public (read-only) listener.
    pub fn discovery_addr(&self) -> SocketAddr {
        self.hub.public_addr()
    }

    pub fn control_addr(&self) -> SocketAddr {
        self.hub.control_addr()
    }

    pub fn hub(&self) -> &Hub {
        &self.hub
    }

    pub async fn shutdown(self) -> Result<()> {
        self.hub.shutdown().await
    }
}

/// The `kvbm` extension: aggregate config and runtime-summary checks.
struct KvbmCore {
    primary: PrimaryConfig,
    base_config: serde_json::Value,
    features: Vec<FeatureEntry>,
}

impl KvbmCore {
    fn config_response(&self) -> HubConfigResponse {
        HubConfigResponse {
            primary: self.primary.clone(),
            features: self
                .features
                .iter()
                .map(|feature| FeatureDescriptor {
                    key: feature.key,
                    dependencies: feature.dependencies.clone(),
                    render_implies: feature.render_implies.clone(),
                    config_requirements: feature.config_requirements,
                    config: (feature.descriptor)(),
                })
                .collect(),
            base_config: self.base_config.clone(),
        }
    }
}

impl Extension for KvbmCore {
    const NAME: &'static str = protocol::KVBM_EXTENSION;
    type Registration = RuntimeConfigSummary;
    type Staged = ();

    fn admit(
        &self,
        view: &RegistrationView<'_>,
        summary: Option<&RuntimeConfigSummary>,
    ) -> Result<(), Rejection> {
        let mut required = FeatureConfigRequirements::default();
        for feature in &self.features {
            let key = feature.key.as_str();
            if !view.extensions.contains_key(key) {
                continue;
            }
            if feature.requires_runtime_summary && summary.is_none() {
                return Err(Rejection::bad_request(format!(
                    "{key} requires a runtime config summary in the {:?} extension",
                    protocol::KVBM_EXTENSION
                )));
            }
            required.block_size |= feature.config_requirements.block_size;
            required.block_layout |= feature.config_requirements.block_layout;
        }
        let Some(summary) = summary else {
            return Ok(());
        };
        if required.block_size
            && let Some(expected) = self.primary.block_size
        {
            check_match("block_size", expected, summary.block_size)?;
        }
        if required.block_layout {
            check_match(
                "block_layout",
                self.primary.block_layout,
                summary.block_layout,
            )?;
        }
        Ok(())
    }

    async fn prepare(
        &self,
        _txn: &Txn<'_>,
        _summary: Option<RuntimeConfigSummary>,
    ) -> Result<(), Rejection> {
        Ok(())
    }

    fn commit(&self, _staged: (), _registration: &Committed<'_>) {}

    fn on_removed(&self, _removal: Removal) {}

    fn public_routes(self: Arc<Self>) -> Router {
        config_routes(self)
    }

    fn control_routes(self: Arc<Self>) -> Router {
        config_routes(self)
    }
}

fn config_routes(core: Arc<KvbmCore>) -> Router {
    Router::new()
        .route(protocol::paths::CONFIG, get(get_config))
        .with_state(core)
}

async fn get_config(State(core): State<Arc<KvbmCore>>) -> Json<HubConfigResponse> {
    Json(core.config_response())
}

fn check_match<T: PartialEq + std::fmt::Debug>(
    field: &str,
    expected: T,
    actual: Option<T>,
) -> Result<(), Rejection> {
    match actual {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(Rejection::bad_request(format!(
            "{field} mismatch: hub requires {expected:?}, registrant declared {actual:?}"
        ))),
        None => Err(Rejection::bad_request(format!(
            "{field} must be declared in the runtime summary (hub requires {expected:?})"
        ))),
    }
}
