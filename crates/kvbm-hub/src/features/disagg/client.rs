// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-side wrapper around [`HubClient`] for the ConditionalDisagg feature.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use velo::Messenger;
use velo::discovery::PeerDiscovery;
use velo_ext::{InstanceId, PeerInfo};

use super::protocol::ConditionalDisaggInstancesResponse;
use crate::client::HubClient;
use crate::features::prefill_router::{PREFILL_SUBMIT_HANDLER, PrefillDispatchResponse};
use crate::protocol::{
    ConditionalDisaggConfig, ConditionalDisaggRole, Feature, LayoutCompatPayload, P2pConfig,
    PrefillRequest, RuntimeConfigSummary,
};

/// Thin wrapper that registers an instance under the ConditionalDisagg
/// feature and exposes helpers for peer discovery and prefill dispatch.
///
/// Construction requires the [`HubClient`] (HTTP discovery surface) and an
/// [`Arc<Messenger>`] (used to send prefill requests to the hub). Use [`ConditionalDisaggClient::new`]
/// when you have a full [`velo::Velo`], or
/// [`ConditionalDisaggClient::with_messenger`] when only a bare [`Messenger`]
/// is available (e.g. from [`kvbm_engine::runtime::KvbmRuntime`]).
pub struct ConditionalDisaggClient {
    hub: Arc<HubClient>,
    messenger: Arc<Messenger>,
    role: ConditionalDisaggRole,
    /// Hub's velo `InstanceId`, learned on `register()`. Prefill requests
    /// are sent to it.
    hub_velo_id: OnceLock<InstanceId>,
}

impl std::fmt::Debug for ConditionalDisaggClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConditionalDisaggClient")
            .field("role", &self.role)
            .field("hub_velo_id", &self.hub_velo_id.get())
            .finish()
    }
}

impl ConditionalDisaggClient {
    /// Wrap a [`HubClient`] + [`velo::Velo`] and declare a role.
    pub fn new(
        hub: Arc<HubClient>,
        velo: Arc<velo::Velo>,
        role: ConditionalDisaggRole,
    ) -> Arc<Self> {
        Self::with_messenger(hub, velo.messenger().clone(), role)
    }

    /// Wrap a [`HubClient`] + [`Arc<Messenger>`] and declare a role.
    ///
    /// Prefer this constructor when only a bare [`Messenger`] is available.
    /// The [`Messenger`] is the only piece of [`velo::Velo`] this client
    /// actually uses (to send prefill requests to the hub), so a full
    /// [`velo::Velo`] instance is not required.
    pub fn with_messenger(
        hub: Arc<HubClient>,
        messenger: Arc<Messenger>,
        role: ConditionalDisaggRole,
    ) -> Arc<Self> {
        Arc::new(Self {
            hub,
            messenger,
            role,
            hub_velo_id: OnceLock::new(),
        })
    }

    /// Role this participant was built with.
    pub fn role(&self) -> ConditionalDisaggRole {
        self.role
    }

    /// Seed the hub's velo `InstanceId` when registration was performed
    /// elsewhere (e.g. the shared P2P foundation registered `[P2P, CD, ...]` in
    /// one call). Idempotent; needed before prefill requests. No-op on `None`.
    pub fn set_hub_velo_id(&self, hub_velo_id: Option<InstanceId>) {
        if let Some(id) = hub_velo_id {
            let _ = self.hub_velo_id.set(id);
        }
    }

    /// Underlying [`HubClient`] — useful for peer lookups that aren't
    /// feature-scoped.
    pub fn hub(&self) -> &Arc<HubClient> {
        &self.hub
    }

    /// Register the participant with the hub, declaring the CD feature
    /// alongside the mandatory P2P feature carrying `layout_compat`.
    ///
    /// On success, caches the hub's velo `InstanceId` internally so
    /// subsequent prefill requests can target it without re-reading the
    /// response.
    ///
    /// `layout_compat` is mandatory — c2 removed the opt-in bypass path.
    pub async fn register(
        &self,
        velo: &velo::Velo,
        layout_compat: LayoutCompatPayload,
    ) -> Result<Option<InstanceId>> {
        self.register_with(velo, layout_compat, Vec::new(), None)
            .await
    }

    /// Like [`register`](Self::register) but additionally declares
    /// `extra_features` (e.g. `Feature::Indexer`) and an optional must-match
    /// [`RuntimeConfigSummary`], all in one registration. The
    /// mandatory `P2P` + `ConditionalDisagg` features are always prepended.
    pub async fn register_with(
        &self,
        velo: &velo::Velo,
        layout_compat: LayoutCompatPayload,
        extra_features: Vec<Feature>,
        runtime: Option<RuntimeConfigSummary>,
    ) -> Result<Option<InstanceId>> {
        let mut features = vec![
            Feature::P2P(P2pConfig { layout_compat }),
            Feature::ConditionalDisagg(ConditionalDisaggConfig { role: self.role }),
        ];
        features.extend(extra_features);
        let hub_id = self.hub.register(velo, features, runtime).await?;
        if let Some(id) = hub_id {
            let _ = self.hub_velo_id.set(id);
        }
        Ok(hub_id)
    }

    /// Fetch the full CD role split from the hub (uses the discovery port).
    pub async fn list_instances(&self) -> Result<ConditionalDisaggInstancesResponse> {
        let url = self
            .hub
            .public_url()
            .join(&format!(
                "/v1/ext/{}{}",
                crate::protocol::DISAGG_EXTENSION,
                super::protocol::paths::INSTANCES
            ))
            .context("joining CD list path")?;
        let resp = reqwest::get(url)
            .await
            .context("GET /v1/ext/disagg/instances")?;
        if !resp.status().is_success() {
            return Err(anyhow!("CD list endpoint returned {}", resp.status()));
        }
        resp.json::<ConditionalDisaggInstancesResponse>()
            .await
            .context("decoding CD list response")
    }

    /// Poll the CD list endpoint until an instance of `role` is present,
    /// then resolve its [`PeerInfo`] via the hub's `PeerDiscovery` surface.
    pub async fn await_peer_of_role(
        &self,
        role: ConditionalDisaggRole,
        poll: Duration,
        timeout: Duration,
    ) -> Result<PeerInfo> {
        let deadline = Instant::now() + timeout;
        loop {
            let snap = self.list_instances().await?;
            let ids = match role {
                ConditionalDisaggRole::Prefill => &snap.prefill,
                ConditionalDisaggRole::Decode => &snap.decode,
            };
            if let Some(first) = ids.first().copied() {
                return self
                    .hub
                    .discovery()
                    .discover_by_instance_id(first)
                    .await
                    .with_context(|| format!("resolving PeerInfo for {first}"));
            }
            if Instant::now() >= deadline {
                return Err(anyhow!(
                    "timed out waiting for a ConditionalDisagg peer in role {role:?}"
                ));
            }
            tokio::time::sleep(poll).await;
        }
    }

    fn hub_velo_id(&self) -> Result<InstanceId> {
        self.hub_velo_id.get().copied().ok_or_else(|| {
            anyhow!(
                "hub velo instance id unknown — call register() against a hub configured with a velo transport first"
            )
        })
    }

    /// Send a [`PrefillRequest`] to the hub's prefill router. Returns once a
    /// prefill worker has taken it. An error means that no prefill worker
    /// took it, so the caller can prefill locally.
    ///
    /// Only valid for Decode participants — returns an error if this client's
    /// role is not [`ConditionalDisaggRole::Decode`].
    pub async fn push_prefill_request(&self, req: &PrefillRequest) -> Result<()> {
        if self.role != ConditionalDisaggRole::Decode {
            return Err(anyhow!(
                "push_prefill_request is only valid for Decode participants (this client is {:?})",
                self.role
            ));
        }
        let response = self
            .messenger
            .typed_unary::<PrefillDispatchResponse>(PREFILL_SUBMIT_HANDLER)?
            .payload(req)
            .context("encoding PrefillRequest")?
            .instance(self.hub_velo_id()?)
            .send()
            .await
            .context("send the prefill request to the hub")?;
        if response.ok {
            Ok(())
        } else {
            Err(anyhow!(
                "the hub placed no prefill worker: {}",
                response.error.as_deref().unwrap_or("no reason given")
            ))
        }
    }
}
