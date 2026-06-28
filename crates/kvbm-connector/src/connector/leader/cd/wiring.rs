// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub/P2P registration wiring for the connector leader — the additive twin of the
//! legacy `leader::p2p::wire::wire_p2p` (which takes the legacy
//! `ConnectorLeader` and cannot be reused). The pure parts (feature-list
//! assembly, layout-compat payload build) are split from the velo/hub-coupled
//! glue so they stay unit-testable; the glue itself ([`wire_hub`],
//! `spawn_describe_push`) is thin and exercised only against a live hub.

use std::sync::Arc;

use anyhow::{Context, Result};

use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::p2p::session::{
    PeerResolver as EnginePeerResolver, SessionFactory, VeloSessionFactory,
};
use kvbm_engine::worker::SerializedLayout;
use kvbm_hub::protocol::LayoutCompatPayload;
use kvbm_hub::{
    ConditionalDisaggConfig, ConditionalDisaggRole, Feature, HubClient, IndexerFeatureConfig,
    P2pConfig, PrefillBackendAdvertisement, PrefillRouterConfig, VllmHttpEndpoint,
};
use kvbm_physical::layout::LayoutConfig;

use crate::connector::leader::build_hub_client;
use crate::connector::leader::hub_handshake::HubHandshake;
use crate::connector::leader::peer_resolver::HubPeerResolver;
use crate::{InstanceId, KvbmRuntime};

/// The hub-coupled transfer foundation produced by [`wire_hub`]. The caller
/// threads `session_factory` + `peer_resolver` into the engine's disagg ops
/// and holds `hub` alive for the leader's life (its registration guard fires
/// a `DELETE` on drop).
pub(in super::super) struct CdFoundation {
    pub(in super::super) hub: Arc<HubClient>,
    /// Hub's velo `InstanceId` (when the hub runs a velo participant) — the
    /// CD client targets its prefill queue at this id.
    pub(in super::super) hub_velo_id: Option<InstanceId>,
    pub(in super::super) peer_resolver: Arc<HubPeerResolver>,
    pub(in super::super) session_factory: Arc<dyn SessionFactory>,
}

/// Build the `layout_compat` payload the hub's P2P feature requires, from the
/// rank-0 reference layout + worker metadata (faithful port of the legacy
/// `ConnectorLeader::build_layout_compat_payload`).
pub(in super::super) fn build_layout_compat_payload(
    runtime: &Arc<KvbmRuntime>,
    reference_config: &LayoutConfig,
    worker_metadata: &SerializedLayout,
    num_workers: usize,
) -> Result<LayoutCompatPayload> {
    let block_layout_mode = runtime.config().block_layout;
    let parallelism = if reference_config.num_heads.is_none() {
        kvbm_config::ParallelismMode::ReplicatedData
    } else {
        runtime.config().cache.parallelism
    };
    let template = kvbm_engine::leader::parallelism::ParallelismTemplate::from_layout_config(
        reference_config,
        parallelism,
        num_workers,
    )
    .context("building ParallelismTemplate for hub registration payload")?;
    kvbm_engine::leader::layout_compat::build_layout_compat_payload_with_template(
        block_layout_mode,
        worker_metadata,
        Some(&template),
    )
    .context("building layout_compat payload for hub registration")
}

/// Assemble the feature list for the single hub registration, mirroring the
/// legacy assembly exactly: `P2P(layout_compat)` first, then
/// `ConditionalDisagg(role)`, then — for the prefill role only — the optional
/// `PrefillRouter(Http)` advertisement, then `Indexer` when effective.
///
/// `http_url` / `http_model` are the raw `KVBM_VLLM_HTTP_URL` /
/// `KVBM_VLLM_HTTP_MODEL` env reads, injected as arguments so the gating is
/// unit-testable. The advertisement requires both to be set and non-empty AND
/// the hub to offer the prefill-router feature — advertising against a hub
/// without the manager attached fails the whole registration (unknown feature
/// keys are rejected, not ignored).
pub(in super::super) fn assemble_features(
    layout_compat: LayoutCompatPayload,
    cd_role: ConditionalDisaggRole,
    hub_offers_router: bool,
    http_url: Option<String>,
    http_model: Option<String>,
    indexer: Option<IndexerFeatureConfig>,
) -> Vec<Feature> {
    let mut features = vec![
        Feature::P2P(P2pConfig { layout_compat }),
        Feature::ConditionalDisagg(ConditionalDisaggConfig { role: cd_role }),
    ];
    if cd_role == ConditionalDisaggRole::Prefill {
        match (http_url, http_model) {
            (Some(base_url), Some(model))
                if !base_url.is_empty() && !model.is_empty() && hub_offers_router =>
            {
                tracing::info!(
                    base_url,
                    model,
                    "advertising vLLM HTTP endpoint to hub prefill router"
                );
                features.push(Feature::PrefillRouter(PrefillRouterConfig {
                    backend: PrefillBackendAdvertisement::Http(VllmHttpEndpoint {
                        base_url,
                        model,
                    }),
                }));
            }
            (Some(_), Some(_)) if !hub_offers_router => {
                tracing::warn!(
                    "KVBM_VLLM_HTTP_URL/MODEL set but hub does not offer the \
                     prefill-router feature; skipping advertisement"
                );
            }
            (Some(_), None) | (None, Some(_)) => {
                tracing::warn!(
                    "KVBM_VLLM_HTTP_URL and KVBM_VLLM_HTTP_MODEL must both be set to \
                     advertise an HTTP prefill backend; skipping"
                );
            }
            _ => {}
        }
    }
    if let Some(cfg) = indexer {
        features.push(Feature::Indexer(cfg));
    }
    features
}

/// Register `features` with the hub and build the P2P transfer foundation:
/// hub client + velo handlers (heartbeat — installed before registration so
/// the liveness probe never catches us handler-less), the single feature
/// registration, the hub-backed peer resolver, the velo session factory
/// (installed on the engine leader's `transfer` cell), and the describe-push.
pub(in super::super) async fn wire_hub(
    runtime: &Arc<KvbmRuntime>,
    engine_leader: &Arc<InstanceLeader>,
    handshake: &HubHandshake,
    features: Vec<Feature>,
) -> Result<CdFoundation> {
    let velo = runtime
        .velo()
        .context("CD features require a KvbmRuntime built with a Velo (got bare Messenger only)")?
        .clone();

    let hub = build_hub_client(&handshake.url)?;
    hub.register_handlers_messenger(velo.messenger())
        .context("installing hub velo handlers for CD/P2P registration")?;

    let hub_velo_id = hub
        .register_instance_with_features_and_runtime(
            velo.peer_info(),
            features,
            handshake.runtime_summary.clone(),
        )
        .await
        .with_context(|| {
            format!(
                "registering CD/P2P features with kvbm-hub at {}",
                handshake.url
            )
        })?;

    // One hub-backed peer resolver shared by the session factory and the
    // engine's disagg ops, so its registration de-dup works across both
    // paths. MUST be the velo-level resolver: `velo.register_peer` populates
    // the streaming-transport registry `attach_anchor` requires, which a bare
    // `messenger.register_peer` would skip.
    let peer_resolver = HubPeerResolver::new(Arc::clone(&hub), Arc::clone(&velo));
    let session_factory: Arc<dyn SessionFactory> = VeloSessionFactory::with_peer_resolver(
        Arc::clone(&velo),
        Arc::clone(engine_leader),
        runtime.tokio(),
        Arc::clone(&peer_resolver) as Arc<dyn EnginePeerResolver>,
    );
    // Hand the factory to the engine control plane's `transfer` module
    // (registered earlier with an empty cell). Idempotent.
    engine_leader.set_session_factory(Arc::clone(&session_factory));

    spawn_describe_push(runtime, engine_leader, &hub, hub_velo_id);

    Ok(CdFoundation {
        hub,
        hub_velo_id,
        peer_resolver,
        session_factory,
    })
}

/// Push the leader's `InstanceDescription` to the hub (steady-state describe
/// path): inject `hub_instance_id` + `config_blob`, then spawn a task that
/// briefly settles, calls `describe()`, and POSTs the result. Failures fall
/// back to the hub's pull path. Port of the legacy `spawn_describe_push`.
fn spawn_describe_push(
    runtime: &Arc<KvbmRuntime>,
    engine_leader: &Arc<InstanceLeader>,
    hub: &Arc<HubClient>,
    hub_velo_id: Option<InstanceId>,
) {
    if let Some(hub_id) = hub_velo_id {
        engine_leader.set_hub_instance_id(hub_id);
    }
    match serde_json::to_value(runtime.config()) {
        Ok(blob) => {
            engine_leader.set_config_blob(blob);
        }
        Err(e) => tracing::warn!(
            error = %e,
            "failed to serialise KvbmConfig for describe push; continuing without config"
        ),
    }
    let hub = Arc::clone(hub);
    let leader = Arc::clone(engine_leader);
    let instance_id = runtime.messenger().instance_id();
    runtime.tokio().spawn(async move {
        // Brief settle so workers can stamp layouts before the first push.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let describe =
            tokio::time::timeout(std::time::Duration::from_secs(5), leader.describe()).await;
        match describe {
            Ok(Ok(payload)) => {
                if let Err(e) = hub.push_describe(instance_id, &payload).await {
                    tracing::warn!(error = %e, "initial describe push failed; hub can pull via ?force=true");
                } else {
                    tracing::info!(instance = %instance_id, workers = payload.workers.len(), "describe pushed to hub");
                }
            }
            Ok(Err(e)) => tracing::warn!(error = %e, "leader.describe() failed; hub describe stays pending until forced pull"),
            Err(_) => tracing::warn!("leader.describe() timed out after 5s; hub describe stays pending until forced pull"),
        }
    });
}

#[cfg(test)]
mod tests {
    use kvbm_common::KvBlockLayout;
    use kvbm_hub::{BlockLayoutMode, FeatureKey};
    use kvbm_protocols::control::LayoutConfigDescription;

    use super::*;

    fn payload() -> LayoutCompatPayload {
        LayoutCompatPayload {
            mode: BlockLayoutMode::Operational,
            canonical: None,
            per_worker_layout: KvBlockLayout::Universal,
            per_worker_config: LayoutConfigDescription {
                num_blocks: 16,
                num_layers: 2,
                outer_dim: 2,
                page_size: 4,
                inner_dim: 8,
                alignment: 1,
                dtype_width_bytes: 2,
                num_heads: None,
            },
            block_region_sizes: None,
            tp_size: 1,
            pp_size: 1,
        }
    }

    fn keys(features: &[Feature]) -> Vec<FeatureKey> {
        features.iter().map(|f| f.key()).collect()
    }

    fn indexer() -> IndexerFeatureConfig {
        IndexerFeatureConfig {
            max_seq_len: Some(8192),
        }
    }

    #[test]
    fn decode_role_registers_p2p_and_cd_only() {
        // Decode never advertises a prefill-router backend, even with both
        // env values present and the hub offering the feature.
        let features = assemble_features(
            payload(),
            ConditionalDisaggRole::Decode,
            true,
            Some("http://localhost:8000".to_string()),
            Some("qwen".to_string()),
            None,
        );
        assert_eq!(
            keys(&features),
            [FeatureKey::P2P, FeatureKey::ConditionalDisagg]
        );
        assert!(matches!(
            &features[1],
            Feature::ConditionalDisagg(ConditionalDisaggConfig {
                role: ConditionalDisaggRole::Decode
            })
        ));
    }

    #[test]
    fn prefill_role_advertises_router_when_env_and_hub_allow() {
        let features = assemble_features(
            payload(),
            ConditionalDisaggRole::Prefill,
            true,
            Some("http://localhost:8000".to_string()),
            Some("qwen".to_string()),
            None,
        );
        assert_eq!(
            keys(&features),
            [
                FeatureKey::P2P,
                FeatureKey::ConditionalDisagg,
                FeatureKey::PrefillRouter
            ]
        );
        let Feature::PrefillRouter(cfg) = &features[2] else {
            panic!("expected PrefillRouter feature");
        };
        let PrefillBackendAdvertisement::Http(endpoint) = &cfg.backend else {
            panic!("expected an HTTP backend advertisement");
        };
        assert_eq!(endpoint.base_url, "http://localhost:8000");
        assert_eq!(endpoint.model, "qwen");
    }

    #[test]
    fn prefill_router_skipped_without_hub_offering() {
        // Both env values set but the hub doesn't offer the feature:
        // advertising would fail the whole registration, so skip.
        let features = assemble_features(
            payload(),
            ConditionalDisaggRole::Prefill,
            false,
            Some("http://localhost:8000".to_string()),
            Some("qwen".to_string()),
            None,
        );
        assert_eq!(
            keys(&features),
            [FeatureKey::P2P, FeatureKey::ConditionalDisagg]
        );
    }

    #[test]
    fn prefill_router_skipped_on_partial_or_empty_env() {
        for (url, model) in [
            (Some("http://localhost:8000".to_string()), None),
            (None, Some("qwen".to_string())),
            (Some(String::new()), Some("qwen".to_string())),
            (None, None),
        ] {
            let features = assemble_features(
                payload(),
                ConditionalDisaggRole::Prefill,
                true,
                url,
                model,
                None,
            );
            assert_eq!(
                keys(&features),
                [FeatureKey::P2P, FeatureKey::ConditionalDisagg]
            );
        }
    }

    #[test]
    fn indexer_appends_when_effective() {
        let features = assemble_features(
            payload(),
            ConditionalDisaggRole::Decode,
            false,
            None,
            None,
            Some(indexer()),
        );
        assert_eq!(
            keys(&features),
            [
                FeatureKey::P2P,
                FeatureKey::ConditionalDisagg,
                FeatureKey::Indexer
            ]
        );
    }
}
