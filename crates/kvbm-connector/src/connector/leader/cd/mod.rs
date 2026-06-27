// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional-disagg transport wiring for the connector leader.
//!
//! The engine owns the CD decode policy (the `cd_interpose` on its search
//! path) and the per-request lifecycle; this module supplies only the
//! transports the engine's `RemoteOps` needs:
//!
//! * [`plane`] — the production [`kvbm_engine::cd::PrefillPlane`]: completes
//!   the wire `RemotePrefillRequest` (window-truncated token ids + the
//!   provided-prefix hash digest, which never cross the engine seam) from the
//!   leader's slot map and enqueues it on the hub's CD prefill queue.
//! * [`tier`] — the velo tier-signal handler writing the engine-owned
//!   [`kvbm_engine::cd::TierCell`].
//! * [`wiring`] — the hub/P2P registration (feature assembly, peer resolver,
//!   session factory, describe-push), the connector analogue of the legacy
//!   `leader::p2p::wire::wire_p2p`.
//!
//! [`wiring_enabled`] is the single gating decision: CD transports wire up
//! only when BOTH a parsed disagg config and a hub handshake carrying the
//! ConditionalDisagg feature are present; otherwise the engine keeps
//! `RemoteOps::default()` (fully local, byte-equivalent to the pre-CD build).

pub(super) mod plane;
pub(super) mod tier;
pub(super) mod wiring;

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};

use crate::connector::leader::hub_handshake::HubHandshake;

use super::construct::{EngineStack, build_indexer_publisher};
use super::{Construction, Leader};

/// Assemble the production CD transports and return the engine's `RemoteOps`:
/// the decode tier cell (+ its velo handler, installed BEFORE the hub
/// registration so the hub's seed-on-register push is never dropped), the
/// single hub registration with the full feature set, the hub-backed peer
/// resolver + velo session factory, the hub-queue prefill plane, and the
/// translated engine-local config. Also wires the CD-case KV-index publisher
/// once the registration is live, and pins the hub client on the leader.
pub(super) async fn wire_disagg(
    leader: &Arc<Leader>,
    construction: &Construction,
    stack: &EngineStack,
    disagg_cfg: &kvbm_config::DisaggConfig,
    handshake: &HubHandshake,
) -> Result<kvbm_engine::RemoteOps> {
    let runtime = &construction.runtime;
    let velo = runtime
        .velo()
        .context("conditional disagg requires a KvbmRuntime built with a Velo")?
        .clone();

    tracing::info!(
        role = ?disagg_cfg.role,
        hub_url = %handshake.url,
        "registering connector leader with kvbm-hub for conditional disaggregation"
    );

    // Engine-owned breaker tier cell. DECODE ONLY: the velo tier-signal
    // handler must be installed BEFORE the hub registration below — the hub
    // seeds the current tier on `ConditionalDisaggManager::on_register`, and
    // a push racing ahead of the handler would be dropped (defeating the
    // seed-on-join). The prefill role installs no handler: its translated
    // policy is `Never`, so the engine never reads the (default-Calm) cell.
    let tier = Arc::new(kvbm_engine::cd::TierCell::default());
    if disagg_cfg.role == kvbm_config::DisaggregationRole::Decode {
        tier::install_tier_signal_handler(velo.messenger(), Arc::clone(&tier))
            .context("installing CD tier-signal handler on decode (pre-registration)")?;
    }

    // Layout-compat payload from the rank-0 worker metadata (SPMD bring-up
    // populated it during the engine-stack build).
    let (worker_metadata, num_workers) = {
        let workers = construction.workers.lock();
        (workers.metadata.first().cloned(), workers.metadata.len())
    };
    let worker_metadata = worker_metadata.ok_or_else(|| {
        anyhow!(
            "cannot build layout_compat payload for hub registration: worker \
             metadata is empty (engine-stack build must run first)"
        )
    })?;
    let layout_compat = wiring::build_layout_compat_payload(
        runtime,
        &stack.reference_config,
        &worker_metadata,
        num_workers,
    )?;

    let cd_role = match disagg_cfg.role {
        kvbm_config::DisaggregationRole::Prefill => kvbm_hub::ConditionalDisaggRole::Prefill,
        kvbm_config::DisaggregationRole::Decode => kvbm_hub::ConditionalDisaggRole::Decode,
    };
    let indexer =
        handshake
            .has(kvbm_hub::FeatureKey::Indexer)
            .then(|| kvbm_hub::IndexerFeatureConfig {
                max_seq_len: runtime.config().max_seq_len,
            });
    let features = wiring::assemble_features(
        layout_compat,
        cd_role,
        handshake.has(kvbm_hub::FeatureKey::PrefillRouter),
        std::env::var("KVBM_VLLM_HTTP_URL").ok(),
        std::env::var("KVBM_VLLM_HTTP_MODEL").ok(),
        indexer,
    );

    let foundation = wiring::wire_hub(runtime, &stack.instance_leader, handshake, features)
        .await
        .context("CD/P2P foundation wiring failed")?;

    // CD-case KV-index publisher: the registration above included
    // `Feature::Indexer` when effective, so the publisher-implies-registration
    // invariant now holds for this path too.
    if let (Some(endpoint), Some(em)) = (
        handshake.indexer_zmq_endpoint.as_ref(),
        stack.events_manager.as_ref(),
    ) && let Some(publisher) = build_indexer_publisher(runtime, endpoint, em)
    {
        let _ = leader.indexer_publisher.set(publisher);
    }

    // Hub-queue CD client, the plane's enqueue transport. Role-matched: a
    // prefill-role worker constructs the plane too, but its translated policy
    // is `Never` so the engine never dispatches — and if it ever did, the
    // client's hard role guard rejects the push loudly rather than letting a
    // prefill worker feed its own queue.
    let client = kvbm_hub::ConditionalDisaggClient::with_messenger(
        Arc::clone(&foundation.hub),
        velo.messenger().clone(),
        cd_role,
    );
    client.set_hub_velo_id(foundation.hub_velo_id);

    let plane = plane::SlotPrefillPlane::new(
        runtime.messenger().instance_id(),
        stack.reference_config.page_size,
        plane::HubCdEnqueue::new(client),
        Arc::downgrade(leader),
    );

    // Pin the hub registration for the leader's life (RAII `DELETE` on drop).
    let _ = leader.cd_hub_client.set(Arc::clone(&foundation.hub));

    let engine_cfg = kvbm_engine::cd::DisaggConfig::from_connector_config(disagg_cfg);
    Ok(kvbm_engine::RemoteOps::default().with_disagg_transports(
        foundation.session_factory,
        plane,
        tier,
        engine_cfg,
        Some(foundation.peer_resolver),
    ))
}

/// Whether the CD transport wiring should run: requires BOTH a parsed
/// `disagg` config (the role) AND a hub handshake whose effective feature set
/// carries ConditionalDisagg. Mirrors the legacy gate (init.rs hub-handshake
/// filter + the pre-flight that guarantees a role when CD is effective).
pub(super) fn wiring_enabled(
    disagg: Option<&kvbm_config::DisaggConfig>,
    handshake: Option<&HubHandshake>,
) -> bool {
    disagg.is_some() && handshake.is_some_and(|h| h.has(kvbm_hub::FeatureKey::ConditionalDisagg))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn handshake_with(features: &[kvbm_hub::FeatureKey]) -> HubHandshake {
        HubHandshake {
            url: "http://127.0.0.1:1337".to_string(),
            effective: HashSet::from_iter(features.iter().copied()),
            indexer_zmq_endpoint: None,
            runtime_summary: kvbm_hub::RuntimeConfigSummary::default(),
        }
    }

    fn disagg(role: kvbm_config::DisaggregationRole) -> kvbm_config::DisaggConfig {
        kvbm_config::DisaggConfig {
            role,
            ..Default::default()
        }
    }

    #[test]
    fn wiring_requires_both_config_and_cd_feature() {
        let decode = disagg(kvbm_config::DisaggregationRole::Decode);
        let cd_handshake = handshake_with(&[
            kvbm_hub::FeatureKey::P2P,
            kvbm_hub::FeatureKey::ConditionalDisagg,
        ]);

        // Both present ⇒ wire.
        assert!(wiring_enabled(Some(&decode), Some(&cd_handshake)));
        // Prefill role wires too (its policy translates to Never).
        let prefill = disagg(kvbm_config::DisaggregationRole::Prefill);
        assert!(wiring_enabled(Some(&prefill), Some(&cd_handshake)));

        // No disagg config ⇒ local, even with a CD handshake.
        assert!(!wiring_enabled(None, Some(&cd_handshake)));
        // No handshake at all ⇒ local.
        assert!(!wiring_enabled(Some(&decode), None));
        // Handshake without the CD feature (e.g. indexer-only) ⇒ local.
        let indexer_only = handshake_with(&[kvbm_hub::FeatureKey::Indexer]);
        assert!(!wiring_enabled(Some(&decode), Some(&indexer_only)));
        // Nothing ⇒ local.
        assert!(!wiring_enabled(None, None));
    }
}
