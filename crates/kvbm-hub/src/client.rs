//! kvbm client of the hub: a velo-hub [`velo_hub::client::HubClient`] plus
//! the kvbm feature payloads and routes.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use url::Url;
use velo_ext::InstanceId;
use velo_hub::client::{self as hub_client, HubDiscovery};
use velo_hub::protocol::Credential;

use kvbm_protocols::cache_manifest::RegistrationEpoch;

use crate::protocol::{self, Feature, HubConfigResponse, RuntimeConfigSummary};

/// A kvbm client of one hub.
pub struct HubClient {
    inner: hub_client::HubClient,
    http: reqwest::Client,
}

impl std::fmt::Debug for HubClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubClient")
            .field("public_url", self.inner.public_url())
            .field("control_url", self.inner.control_url())
            .finish()
    }
}

/// Builder for [`HubClient`].
#[derive(Debug, Clone)]
pub struct HubClientBuilder {
    inner: hub_client::HubClientBuilder,
    http: Option<reqwest::Client>,
}

impl Default for HubClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HubClientBuilder {
    /// A client of `localhost` on the default kvbm ports.
    pub fn new() -> Self {
        Self {
            inner: hub_client::HubClientBuilder::default()
                .public_port(protocol::DEFAULT_DISCOVERY_PORT)
                .control_port(protocol::DEFAULT_CONTROL_PORT),
            http: None,
        }
    }

    /// Parse `scheme://host[:port]`. The port is the discovery port; the
    /// control port keeps the kvbm default unless set.
    pub fn from_url(url: &str) -> Result<Self> {
        let parsed = Url::parse(url).with_context(|| format!("parsing hub url {url}"))?;
        let inner = hub_client::HubClientBuilder::from_url(url)?
            .public_port(parsed.port().unwrap_or(protocol::DEFAULT_DISCOVERY_PORT))
            .control_port(protocol::DEFAULT_CONTROL_PORT);
        Ok(Self { inner, http: None })
    }

    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.inner = self.inner.host(host);
        self
    }

    pub fn scheme(mut self, scheme: impl Into<String>) -> Self {
        self.inner = self.inner.scheme(scheme);
        self
    }

    pub fn discovery_port(mut self, port: u16) -> Self {
        self.inner = self.inner.public_port(port);
        self
    }

    pub fn control_port(mut self, port: u16) -> Self {
        self.inner = self.inner.control_port(port);
        self
    }

    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.http = Some(client);
        self
    }

    pub fn build(self) -> Result<Arc<HubClient>> {
        let http = self.http.unwrap_or_default();
        let inner = self.inner.http_client(http.clone()).build()?;
        Ok(Arc::new(HubClient { inner, http }))
    }
}

pub(crate) fn feature_payload(feature: &Feature) -> serde_json::Result<serde_json::Value> {
    match feature {
        Feature::P2P(config) => serde_json::to_value(config),
        Feature::ConditionalDisagg(config) => serde_json::to_value(config),
        Feature::Indexer(config) => serde_json::to_value(config),
        Feature::PrefillRouter(config) => serde_json::to_value(config),
    }
}

impl HubClient {
    pub fn public_url(&self) -> &Url {
        self.inner.public_url()
    }

    pub fn control_url(&self) -> &Url {
        self.inner.control_url()
    }

    /// Peer discovery through this hub.
    pub fn discovery(&self) -> HubDiscovery {
        self.inner.discovery()
    }

    /// Register `velo` with `features`, and answer the hub's heartbeat
    /// probes on it. `runtime` is required by the features that check sizing
    /// against the hub. Returns the hub's velo instance id, if it has one.
    pub async fn register(
        &self,
        velo: &velo::Velo,
        features: Vec<Feature>,
        runtime: Option<RuntimeConfigSummary>,
    ) -> Result<Option<InstanceId>> {
        let mut extensions = hub_client::Extensions::new();
        for feature in &features {
            extensions.insert(
                feature.key().as_str().to_string(),
                feature_payload(feature).context("encoding feature payload")?,
            );
        }
        if let Some(runtime) = runtime {
            extensions.insert(
                protocol::KVBM_EXTENSION.to_string(),
                serde_json::to_value(runtime).context("encoding runtime summary")?,
            );
        }
        let response = self
            .inner
            .register(velo, extensions)
            .await
            .context("registering with kvbm hub")?;
        Ok(response.hub_instance_id)
    }

    pub fn hub_velo_id(&self) -> Option<InstanceId> {
        self.inner.registration()?.hub_instance_id
    }

    /// The registration epoch the current credential authorizes.
    pub fn registration_epoch(&self) -> Option<RegistrationEpoch> {
        let registration = self.inner.registration()?;
        Some(RegistrationEpoch::from_uuid(registration.epoch.as_uuid()))
    }

    fn credential(&self) -> Option<Credential> {
        Some(self.inner.registration()?.credential)
    }

    pub async fn unregister(&self) -> Result<()> {
        Ok(self.inner.unregister().await?)
    }

    pub fn is_registered(&self) -> bool {
        self.inner.is_registered()
    }

    pub async fn send_heartbeat(&self) -> Result<()> {
        Ok(self.inner.send_heartbeat().await?)
    }

    /// The hub's aggregate kvbm config.
    pub async fn get_config(&self) -> Result<HubConfigResponse> {
        Ok(self
            .inner
            .ext(protocol::KVBM_EXTENSION)
            .get(protocol::paths::CONFIG)
            .await?)
    }

    /// A lookup client for the hub's indexer, or `None` when the hub does
    /// not serve the indexer.
    pub async fn indexer_lookup_client(
        self: &Arc<Self>,
        messenger: Arc<velo::Messenger>,
    ) -> Result<Option<Arc<crate::features::indexer::IndexerLookupClient>>> {
        // A failed probe (404 when the indexer isn't mounted, or any transport
        // error) means "not available" — not an error for the caller.
        if self
            .inner
            .ext(protocol::INDEXER_EXTENSION)
            .get::<crate::features::indexer::IndexerConfigResponse>("/config")
            .await
            .is_err()
        {
            return Ok(None);
        }
        let hub_id = self.hub_velo_id().ok_or_else(|| {
            anyhow!(
                "indexer enabled but hub velo InstanceId unknown — \
                 register against a velo-enabled hub first"
            )
        })?;
        let credential = self
            .credential()
            .ok_or_else(|| anyhow!("indexer enabled but this client is not registered"))?;
        let registration_epoch = self
            .registration_epoch()
            .ok_or_else(|| anyhow!("indexer enabled but this client is not registered"))?;
        // The snapshot install is a credential-authorized mutation, so the
        // indexer serves it on the control listener only.
        let snapshot_url = self.control_url().join(&format!(
            "/v1/ext/{}{}",
            protocol::INDEXER_EXTENSION,
            crate::features::indexer::protocol::paths::TIER_PLACEMENT_SNAPSHOT
        ))?;
        Ok(Some(crate::features::indexer::IndexerLookupClient::new(
            messenger,
            hub_id,
            credential,
            registration_epoch,
            self.http.clone(),
            snapshot_url,
        )))
    }

    /// `GET path` on the public listener.
    pub async fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = self.public_url().join(path)?;
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .with_context(|| format!("GET {path}"))?;
        parse_json(resp, path).await
    }

    /// `POST path` on the public listener.
    pub async fn post_json<B: serde::Serialize, T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let url = self.public_url().join(path)?;
        let resp = self
            .http
            .post(url)
            .json(body)
            .send()
            .await
            .with_context(|| format!("POST {path}"))?;
        parse_json(resp, path).await
    }

    /// Push this instance's description to the hub's control plane.
    pub async fn push_describe(
        &self,
        instance_id: InstanceId,
        payload: &kvbm_protocols::control::InstanceDescription,
    ) -> Result<()> {
        let path = format!(
            "/v1/ext/{}{}",
            protocol::CONTROL_PLANE_EXTENSION,
            protocol::instance_describe(instance_id)
        );
        let resp = self
            .http
            .post(self.control_url().join(&path)?)
            .json(payload)
            .send()
            .await
            .with_context(|| format!("POST {path}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("push_describe rejected ({status}): {body}");
        }
        Ok(())
    }
}

pub(crate) async fn parse_json<T: serde::de::DeserializeOwned>(
    resp: reqwest::Response,
    path: &str,
) -> Result<T> {
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("{path}: hub answered {status}: {body}");
    }
    resp.json()
        .await
        .with_context(|| format!("{path}: decoding hub response"))
}
