// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader-side engine-stack construction for connector.
//!
//! [`build_engine_stack`] is the connector copy of the engine-stack core of the legacy
//! `ConnectorLeader::initialize_async` (lib/kvbm-connector/src/connector/leader/
//! init.rs:160-968), adapted to read the connector [`Construction`] accumulation and to
//! **return** the built `Arc<InstanceLeader>` + `Option<Arc<OffloadEngine>>`
//! rather than stashing them into `self`. Returning them lets the caller clone
//! the leader for the conditional-disagg wiring *before* the factory consumes
//! it into the `LeaderEngine`.
//!
//! Scope: the hubless core — worker layout gather + validate, G2/G3 block
//! counts + host-bypass sentinel, `worker.initialize`, `BlockManager<G2/G3>`,
//! `InstanceLeader` build + `register_handlers`, the `OffloadEngine`
//! (G1→G2 / G2→G3, or bypass G1→G3), worker handler refresh. Plus the hub
//! handshake + `EventsManager` + KV-index publisher / registration bracket
//! (init.rs:442-528) — the publisher always implies a live registration: the
//! indexer-only arm registers here, while the CD/P2P case registers in
//! `Leader::initialize_async`'s CD wiring (see `super::cd`), which builds the
//! CD-case publisher only after that registration succeeds — and the feature
//! brackets: consolidator, control plane + `set_modules`, object_client +
//! remote-search `min_remote_blocks`, G2→G4 object storage. The
//! conditional-disagg transports themselves live in `super::cd`, consumed by
//! `Leader::initialize_async`.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};

use kvbm_common::LogicalResourceId;
use kvbm_config::ParallelismMode;
use kvbm_engine::leader::parallelism::worker_data_placement;
use kvbm_engine::leader::{ConsolidatorParams, InstanceLeader};
use kvbm_engine::object::{ObjectLockManager, create_lock_manager, create_object_client};
use kvbm_engine::offload::{
    ObjectPipelineBuilder, ObjectPresenceFilter, OffloadEngine, PendingTracker, PipelineBuilder,
    S3PresenceChecker, create_policy_from_config,
};
use kvbm_engine::worker::{LeaderLayoutConfig, Worker};
use kvbm_hub::HubClient;
use kvbm_logical::BlockManagerSet;
use kvbm_logical::blocks::{BlockDuplicationPolicy, BlockRegistry};
use kvbm_logical::events::{EventsManager, KvbmCacheEventsPublisher};
use kvbm_logical::manager::{BlockManager, FrequencyTrackingCapacity};
use kvbm_physical::layout::LayoutConfig;
use kvbm_physical::manager::WorkerDataPlacement;

use crate::connector::leader::hub_handshake::{self, HubHandshake};
use crate::connector::leader::hub_indexer;
use crate::{G1, G2, G3, KvbmRuntime};

use super::Construction;

mod resources;

use resources::{
    ResourcePlan, build_collective_bootstrap, logical_tier_block_count, resolve_parallelism,
};

fn local_transfer_placements(
    resource_parallelism: &BTreeMap<LogicalResourceId, ParallelismMode>,
) -> Vec<(LogicalResourceId, WorkerDataPlacement)> {
    resource_parallelism
        .iter()
        .map(|(&resource, &mode)| (resource, worker_data_placement(mode)))
        .collect()
}

/// The leader-side engine stack produced by [`build_engine_stack`]. The caller
/// (`Leader::initialize`) clones `instance_leader` for the CD wiring before
/// moving it into the `LeaderEngine` via `build_local_connector_engine`.
pub(super) struct EngineStack {
    pub(super) instance_leader: Arc<InstanceLeader>,
    pub(super) offloads: Vec<(kvbm_common::LogicalResourceId, Arc<OffloadEngine>)>,
    pub(super) primary_resource: kvbm_common::LogicalResourceId,
    pub(super) admission: resources::ResourceAdmissionPlan,
    /// Rank-0 reference layout (all workers validated equal); the CD wiring
    /// reuses it for the hub `layout_compat` payload and the parallelism
    /// template.
    pub(super) reference_config: LayoutConfig,
    /// Resolved hub handshake, `Some` when a hub is configured. The CD wiring
    /// reads the effective feature set + registration inputs from it — the
    /// stack build consumes everything else it needs, so the handshake must
    /// ride in the result or the caller could never feed the CD gate.
    pub(super) handshake: Option<HubHandshake>,
    /// Block-registration events manager, `Some` when the consolidator or the
    /// KV-index publisher needs it. The CD wiring subscribes the CD-case
    /// indexer publisher from it AFTER its hub registration succeeds.
    pub(super) events_manager: Option<Arc<EventsManager>>,
    /// KV-index publisher, `Some` when the hub's indexer is effective. The caller
    /// must hold it for the leader's life — dropping it aborts the publish task.
    pub(super) indexer_publisher: Option<KvbmCacheEventsPublisher>,
    /// KV-index-only hub registration, `Some` when Indexer is the sole effective
    /// hub feature. RAII `DELETE` on drop, so the caller must hold it alive.
    pub(super) indexer_hub_client: Option<Arc<HubClient>>,
}

/// KV-index-only hub registration (faithful copy of
/// `ConnectorLeader::register_indexer_only`, init.rs:65-91): declare
/// `Feature::Indexer` (+ the must-match runtime summary) so the hub reclaims this
/// instance's index entries on unregister. Returns the [`HubClient`] for the
/// caller to hold alive (the RAII guard must not fire a premature `DELETE`).
async fn register_indexer_only(
    runtime: &Arc<KvbmRuntime>,
    handshake: &HubHandshake,
) -> Result<Arc<HubClient>> {
    let velo = runtime
        .velo()
        .ok_or_else(|| anyhow!("indexer hub registration requires a Velo runtime"))?;
    let hub = super::build_hub_client(&handshake.url)?;
    // Install hub velo handlers (heartbeat) so the hub's liveness probe doesn't
    // unregister us — which would prematurely sweep our index.
    hub.register_handlers_messenger(velo.messenger())
        .context("installing hub velo handlers for indexer registration")?;
    let max_seq_len = runtime.config().max_seq_len;
    hub.register_instance_with_features_and_runtime(
        velo.peer_info(),
        vec![kvbm_hub::Feature::Indexer(kvbm_hub::IndexerFeatureConfig {
            max_seq_len,
        })],
        handshake.runtime_summary.clone(),
    )
    .await
    .context("registering Feature::Indexer with kvbm-hub")?;
    Ok(hub)
}

/// Connect the KV-index ZMQ publisher and wire it to the events manager
/// (init.rs:505-528). Both callers — the indexer-only arm below and the
/// CD-case wiring in `Leader::initialize_async` — must hold a LIVE hub
/// registration before invoking this (the publisher-implies-registration
/// invariant). Connect/build failures degrade to `None` with a warning, never
/// an error: a broken publisher loses index freshness, not correctness.
pub(super) fn build_indexer_publisher(
    runtime: &Arc<KvbmRuntime>,
    endpoint: &str,
    events_manager: &Arc<EventsManager>,
) -> Option<KvbmCacheEventsPublisher> {
    match hub_indexer::ZmqHubPublisher::connect(endpoint) {
        Ok(zmq_pub) => {
            let instance_id = runtime.messenger().instance_id().as_u128();
            match KvbmCacheEventsPublisher::builder()
                .instance_id(instance_id)
                .event_stream(events_manager.subscribe())
                .publisher(Arc::new(zmq_pub))
                .subject(hub_indexer::SUBJECT)
                .build()
            {
                Ok(publisher) => {
                    tracing::info!(endpoint, instance_id, "indexer publisher wired");
                    Some(publisher)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "indexer publisher build failed; skipping");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "indexer PUB connect failed; skipping");
            None
        }
    }
}

/// Build the leader-side engine stack from the registered workers. Async because
/// it awaits the per-worker `get_layout_config`/`initialize` velo round-trips;
/// `Leader::initialize` drives it to completion on the runtime.
pub(super) async fn build_engine_stack(c: &Construction) -> Result<EngineStack> {
    let runtime = &c.runtime;

    // Step 1: gather per-worker layout-config futures (lock held only to clone
    // the futures out), then await them outside the lock.
    let layout_config_futures = {
        let workers = c.workers.lock();
        if workers.connector_clients.is_empty() {
            bail!("No workers registered");
        }
        let mut futures = Vec::with_capacity(workers.connector_clients.len());
        for worker in workers.connector_clients.iter() {
            futures.push(worker.get_layout_config()?);
        }
        futures
    };
    let mut layout_configs = Vec::with_capacity(layout_config_futures.len());
    for (i, future) in layout_config_futures.into_iter().enumerate() {
        let config = future
            .await
            .map_err(|e| anyhow!("Failed to get layout config from worker {i}: {e}"))?;
        layout_configs.push(config);
    }

    // Step 2: validate worker ABI identity and plan every logical resource.
    let cache_identity = c.cache_identity.lock().clone();
    let ResourcePlan {
        config: reference_resources,
        primary: primary_resource,
        primary_layout: reference_config,
        tiers: resource_tiers,
        parallelism: resource_parallelism,
        admission,
    } = ResourcePlan::build(
        runtime,
        &layout_configs,
        *c.manifest.lock(),
        cache_identity.as_ref(),
    )?;

    // Step 3: compute G2/G3 block counts + host-bypass sentinel.
    let primary_tier = resource_tiers
        .get(&primary_resource)
        .expect("primary tier capacity");
    let host_block_count = primary_tier.host_block_count;
    let disk_block_count = primary_tier.disk_block_count;

    // At least one cache tier must produce a non-zero block count, else the
    // leader has nothing to offload to. Fail loudly (mirrors legacy sanity check).
    let host_ok = host_block_count > 0;
    let disk_ok = disk_block_count.is_some_and(|n| n > 0);
    if !host_ok && !disk_ok {
        bail!(
            "KVBM Configuration Error: at least one cache tier must be configured \
             (DYN_KVBM_CPU_CACHE_GB for G2, or DYN_KVBM_DISK_CACHE_GB for G3)."
        );
    }
    let worker_count = c.workers.lock().connector_clients.len();
    let parallelism = resolve_parallelism(runtime.config().cache.parallelism, &reference_config);
    if parallelism != runtime.config().cache.parallelism {
        tracing::info!(
            configured = ?runtime.config().cache.parallelism,
            resolved = ?parallelism,
            "Registered cache has no HeadCount axis; selecting replicated-data placement"
        );
    }
    let logical_disk_block_count = disk_block_count
        .map(|count| logical_tier_block_count(count, parallelism, worker_count))
        .transpose()?;
    let collective = build_collective_bootstrap(
        if resource_parallelism
            .values()
            .any(|mode| *mode == kvbm_config::ParallelismMode::ReplicatedData)
        {
            kvbm_config::ParallelismMode::ReplicatedData
        } else {
            parallelism
        },
        worker_count,
    )?;

    // Host-bypass: disk configured, host not — serve disk hits to GPU directly,
    // no G2 staging. InstanceLeader still requires a G2 manager, so build it with
    // a sentinel block_count of 1 (BlockManager rejects 0; it allocates nothing).
    let bypass_host = runtime.config().cache.bypass_host_cache();
    // Step 4: initialize all workers in parallel, collect their metadata, and
    // configure each transfer client's layout handles.
    let initialize_futures = {
        let workers = c.workers.lock();
        let object_config = runtime.config().object.clone();
        let mut futures = Vec::with_capacity(workers.connector_clients.len());
        for (idx, worker) in workers.connector_clients.iter().enumerate() {
            let leader_config = LeaderLayoutConfig {
                rank: idx,
                worker_count,
                host_block_count,
                disk_block_count,
                resource_tiers: resource_tiers.clone(),
                resource_parallelism: resource_parallelism.clone(),
                object: object_config.clone(),
                parallelism,
                collective: collective.clone(),
            };
            futures.push(worker.initialize(leader_config)?);
        }
        futures
    };
    let mut collected_metadata = Vec::new();
    for (i, future) in initialize_futures.into_iter().enumerate() {
        let worker_layout = future
            .await
            .with_context(|| format!("Failed to initialize worker {i}"))?;
        collected_metadata.push(worker_layout.metadata.clone());
    }

    // Store metadata + configure transfer-client routing and layout handles.
    // `resource_parallelism` is the same authoritative plan sent to workers
    // above, so the leader's SPMD layer cannot drift from worker-side routing.
    let local_transfer_placements = local_transfer_placements(&resource_parallelism);
    {
        let mut workers = c.workers.lock();
        workers.metadata = collected_metadata.clone();
        for (i, (client, metadata)) in workers
            .transfer_clients
            .iter()
            .zip(collected_metadata.iter())
            .enumerate()
        {
            client
                .configure_local_transfer_placements(
                    primary_resource,
                    local_transfer_placements.clone(),
                )
                .with_context(|| {
                    format!("Failed to configure transfer placements for worker {i}")
                })?;
            client
                .configure_layout_handles(metadata)
                .with_context(|| format!("Failed to configure handles for worker {i}"))?;
        }
    }

    // Hub handshake (init.rs:442-472). `hub` absent ⇒ no hub features (normal
    // hub-less work). When present, pull GET /v1/config, resolve the effective
    // feature set, and learn the KV-index ZMQ endpoint.
    let cfg = runtime.config();
    let handshake: Option<HubHandshake> = match cfg.hub.as_ref() {
        Some(hub) => Some(
            hub_handshake::resolve(
                hub,
                reference_config.page_size,
                cfg.block_layout,
                cfg.disagg.as_ref(),
                hub_handshake::WorkerCapabilities::default(),
            )
            .await
            .context("kvbm-hub handshake")?,
        ),
        None => None,
    };
    let indexer_endpoint = handshake
        .as_ref()
        .and_then(|h| h.indexer_zmq_endpoint.clone());

    // Fail fast (before any registration) if remote search is requested but the
    // hub's indexer isn't effective.
    hub_handshake::validate_remote_search_availability(
        cfg.remote_search.as_ref(),
        handshake.as_ref(),
    )?;

    // EventsManager when either the consolidator or the KV-index publisher needs
    // block-registration events — the same Arc wires into the BlockRegistry and
    // every subscriber (init.rs:474-487).
    let events_manager: Option<Arc<EventsManager>> = (c.consolidator_endpoints.is_some()
        || indexer_endpoint.is_some())
    .then(|| Arc::new(EventsManager::builder().build()));

    // KV-index registration + publisher (init.rs:489-528). INVARIANT: the
    // publisher must imply a LIVE hub registration — an unregistered publisher
    // orphans index entries the hub never reclaims on unregister.
    //
    // `register_indexer_only` covers the case where Indexer is the *sole*
    // effective hub feature: register first, then wire the publisher. With
    // P2P / ConditionalDisagg the registration folds into the single CD-wiring
    // POST in `Leader::initialize_async` — which includes `Feature::Indexer`
    // when effective and wires the CD-case publisher only AFTER that
    // registration succeeds. Either way a publisher never exists without a
    // registration.
    let indexer_only = handshake.as_ref().is_some_and(|h| {
        h.has(kvbm_hub::FeatureKey::Indexer)
            && !h.has(kvbm_hub::FeatureKey::P2P)
            && !h.has(kvbm_hub::FeatureKey::ConditionalDisagg)
    });
    let mut indexer_publisher = None;
    let mut indexer_hub_client = None;
    if indexer_only {
        let h = handshake
            .as_ref()
            .expect("indexer_only implies a handshake");
        // Register first, so the publisher never emits without a live registration.
        indexer_hub_client = Some(register_indexer_only(runtime, h).await?);

        if let (Some(endpoint), Some(em)) = (&indexer_endpoint, events_manager.as_ref()) {
            indexer_publisher = build_indexer_publisher(runtime, endpoint, em);
        }
    }

    // Step 5: one independent block namespace and G2 manager per resource.
    let logical_metrics = runtime.observability().logical_aggregator();
    let mut registries = BTreeMap::new();
    let mut g2_manager_set = BlockManagerSet::new();
    for (&resource, layout) in &reference_resources.resources {
        let mut registry_builder = BlockRegistry::builder()
            .frequency_tracker(FrequencyTrackingCapacity::Medium.create_tracker());
        if let Some(em) = events_manager.clone() {
            registry_builder = registry_builder.event_manager(em);
        }
        let registry = registry_builder.build();
        let tier = resource_tiers
            .get(&resource)
            .expect("tier capacity for every resource");
        let resource_parallelism = resolve_parallelism(runtime.config().cache.parallelism, layout);
        let logical_blocks =
            logical_tier_block_count(tier.host_block_count, resource_parallelism, worker_count)?;
        let manager_blocks = if bypass_host {
            logical_blocks.max(1)
        } else {
            logical_blocks
        };
        let manager = Arc::new(
            BlockManager::<G2>::builder()
                .block_count(manager_blocks)
                .block_size(layout.page_size)
                .registry(registry.clone())
                .with_lineage_backend()
                .aggregator(logical_metrics.clone())
                .duplication_policy(BlockDuplicationPolicy::Reject)
                .build()?,
        );
        g2_manager_set.insert(resource, manager)?;
        registries.insert(resource, registry);
    }
    let g2_manager_set = Arc::new(g2_manager_set);
    let primary_registry = registries
        .get(&primary_resource)
        .expect("primary registry")
        .clone();
    let g3_manager: Option<Arc<BlockManager<G3>>> = logical_disk_block_count.map(|count| {
        Arc::new(
            BlockManager::<G3>::builder()
                .block_count(count)
                .block_size(reference_config.page_size)
                .registry(primary_registry.clone())
                .with_lineage_backend()
                .aggregator(logical_metrics.clone())
                .duplication_policy(BlockDuplicationPolicy::Reject)
                .build()
                .expect("Should build G3 manager"),
        )
    });

    // Clone registry + managers for the OffloadEngine (shared state via Arcs).
    let registry_for_offload = Arc::new(
        registries
            .get(&primary_resource)
            .expect("primary registry")
            .clone(),
    );
    let g3_manager_for_offload = g3_manager.clone();

    // Snapshot the InstanceLeader workers (transfer clients) + metadata.
    let (worker_clients, worker_metadata) = {
        let workers = c.workers.lock();
        (workers.transfer_clients.clone(), workers.metadata.clone())
    };
    let num_workers = worker_clients.len();

    // Step 6: InstanceLeader builder.
    let mut leader_builder = InstanceLeader::builder()
        .messenger(runtime.messenger().clone())
        .observability(runtime.observability().clone());
    if let Some(velo) = runtime.velo() {
        leader_builder = leader_builder.velo(velo.clone());
    }
    leader_builder = leader_builder
        .block_layout_mode(runtime.config().block_layout)
        .registry(primary_registry)
        .g2_manager_set(Arc::clone(&g2_manager_set), primary_resource)
        .bypass_host(bypass_host)
        .workers(
            worker_clients
                .into_iter()
                .map(|client| Arc::new(client) as Arc<dyn Worker>)
                .collect(),
        )
        .with_cached_worker_metadata(worker_metadata);
    if let Some(disagg_cfg) = runtime.config().disagg.as_ref() {
        leader_builder = leader_builder.role(disagg_cfg.role);
    }
    let templates = reference_resources
        .resources
        .iter()
        .map(|(&resource, layout)| {
            let mode = resolve_parallelism(runtime.config().cache.parallelism, layout);
            Ok((
                resource,
                kvbm_engine::leader::parallelism::ParallelismTemplate::from_layout_config(
                    layout,
                    mode,
                    num_workers,
                )?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    leader_builder = leader_builder.parallelism_template_set(
        kvbm_engine::leader::parallelism::ParallelismTemplateSet::new(primary_resource, templates)?,
    );
    if let Some(g3_mgr) = g3_manager {
        leader_builder = leader_builder.g3_manager(g3_mgr);
    }

    // Add object_client for G4 search (leader calls has_blocks on S3 directly).
    // Uses rank=None so keys are not prefixed — allows querying all
    // worker-written blocks (init.rs:662-668).
    if let Some(object_config) = &runtime.config().object {
        tracing::debug!("Creating object client for G4 search (no rank prefix)");
        let object_client = create_object_client(object_config, None).await?;
        leader_builder = leader_builder.object_client(object_client);
    }

    // Remote-search block-count threshold (init.rs:670-682). The discovery
    // handle itself is injected post-construction and is not wired for the connector
    // yet — the indexer client only exists once a hub registration
    // completes, and the connector CD wiring registers without installing one.
    if let Some(rs) = runtime
        .config()
        .remote_search
        .as_ref()
        .filter(|r| r.enabled)
    {
        leader_builder =
            leader_builder.min_remote_blocks(rs.min_remote_blocks(reference_config.page_size));
    }

    let leader = leader_builder.build()?;
    leader.register_handlers()?;

    // Start the in-process consolidator if endpoints were provided
    // (init.rs:691-714). Hard-fail: a consolidator config error is a
    // mis-configuration that must surface immediately rather than silently
    // degrade.
    if let Some(endpoints) = c.consolidator_endpoints.as_ref() {
        let em = events_manager
            .clone()
            .expect("events_manager must be Some when consolidator_endpoints is Some");
        let params = ConsolidatorParams {
            vllm_zmq_endpoint: endpoints.vllm_zmq_endpoint.clone(),
            egress_endpoint: endpoints.egress_endpoint.clone(),
            engine_source: endpoints.engine_source,
            events_manager: em,
        };
        tracing::info!(
            egress_endpoint = %endpoints.egress_endpoint,
            has_vllm_zmq = endpoints.vllm_zmq_endpoint.is_some(),
            "Starting in-process consolidator"
        );
        leader
            .with_consolidator(params)
            .await
            .context("failed to start in-process consolidator")?;
        tracing::info!("In-process consolidator started");
    }

    let leader = Arc::new(leader);

    // Register the public leader control plane (init.rs:724-740). `core` +
    // `transfer` are always on; `dev` / `metrics` are opt-in via config. The
    // `transfer` module reads the disagg `SessionFactory` lazily from a cell
    // populated once `Leader::initialize_async`'s CD wiring builds the
    // factory, so registering it before that wiring runs is safe.
    let control_cfg = &runtime.config().control;
    let control_plane = leader
        .register_control_plane(control_cfg.dev, control_cfg.metrics)
        .context("registering leader control plane")?;
    tracing::debug!(
        dev = control_cfg.dev,
        metrics = control_cfg.metrics,
        "Leader control plane registered"
    );

    // Surface the enabled module set on the leader so `describe()` can report
    // it without re-traversing the control plane object.
    leader.set_modules(control_plane.enabled_modules().to_vec());

    // Step 7: OffloadEngine (core pipelines).
    let offload_config = &runtime.config().offload;
    let mut engine_builder = OffloadEngine::builder(leader.clone()).with_runtime(runtime.tokio());

    if bypass_host {
        let g1_to_g3_config = if offload_config.g1_to_g3.policies.is_empty() {
            kvbm_config::TierOffloadConfig {
                policies: vec![kvbm_config::PolicyType::Presence],
                ..Default::default()
            }
        } else {
            offload_config.g1_to_g3.clone()
        };
        let g1_to_g3_pending = Arc::new(PendingTracker::new());
        let g1_to_g3_policy = create_policy_from_config::<G1, G3>(
            &g1_to_g3_config,
            registry_for_offload.clone(),
            Some(g1_to_g3_pending.clone()),
        );
        let g1_to_g3_pipeline = PipelineBuilder::<G1, G3>::new()
            .policy(g1_to_g3_policy)
            .pending_tracker(g1_to_g3_pending)
            .build();
        let g3_mgr = g3_manager_for_offload.clone().ok_or_else(|| {
            anyhow!("Host-bypass mode requires a configured G3 (disk) cache; got none")
        })?;
        engine_builder = engine_builder
            .with_g3_manager(g3_mgr)
            .with_g1_to_g3_pipeline(g1_to_g3_pipeline);
    } else {
        let g1_to_g2_config = if offload_config.g1_to_g2.policies.is_empty() {
            kvbm_config::TierOffloadConfig {
                policies: vec![kvbm_config::PolicyType::Presence],
                ..Default::default()
            }
        } else {
            offload_config.g1_to_g2.clone()
        };
        let g1_to_g2_pending = Arc::new(PendingTracker::new());
        let g1_to_g2_policy = create_policy_from_config::<G1, G2>(
            &g1_to_g2_config,
            registry_for_offload.clone(),
            Some(g1_to_g2_pending.clone()),
        );
        let has_downstream_tier =
            g3_manager_for_offload.is_some() || runtime.config().object.is_some();
        let g1_to_g2_pipeline = PipelineBuilder::<G1, G2>::new()
            .policy(g1_to_g2_policy)
            .pending_tracker(g1_to_g2_pending)
            .auto_chain(has_downstream_tier)
            .build();

        let g2_to_g3_config = if offload_config.g2_to_g3.policies.is_empty() {
            kvbm_config::TierOffloadConfig {
                policies: vec![kvbm_config::PolicyType::Presence],
                ..Default::default()
            }
        } else {
            offload_config.g2_to_g3.clone()
        };
        let g2_to_g3_pending = Arc::new(PendingTracker::new());
        let g2_to_g3_policy = create_policy_from_config::<G2, G3>(
            &g2_to_g3_config,
            registry_for_offload.clone(),
            Some(g2_to_g3_pending.clone()),
        );
        let g2_to_g3_pipeline = PipelineBuilder::<G2, G3>::new()
            .policy(g2_to_g3_policy)
            .pending_tracker(g2_to_g3_pending)
            .build();

        engine_builder = engine_builder.with_g1_to_g2_pipeline(g1_to_g2_pipeline);
        if let Some(g3_mgr) = g3_manager_for_offload {
            engine_builder = engine_builder
                .with_g3_manager(g3_mgr)
                .with_g2_to_g3_pipeline(g2_to_g3_pipeline);
        }
    }

    // Build the G2→G4 object-storage pipeline if configured (init.rs:849-893).
    // Uses the leader's parallel_worker as ObjectBlockOps to fan out to all
    // workers; has_blocks queries S3 with each worker's rank-prefixed keys.
    if let Some(object_config) = &runtime.config().object {
        tracing::debug!("Object storage configured, creating G2→G4 pipeline");

        let instance_id = runtime.messenger().instance_id().to_string();
        let lock_manager: Arc<dyn ObjectLockManager> =
            create_lock_manager(object_config, instance_id).await?;

        if let Some(parallel_worker) = leader.parallel_worker() {
            let object_ops: Arc<dyn kvbm_engine::object::ObjectBlockOps> = parallel_worker;

            let presence_checker = Arc::new(S3PresenceChecker::new(object_ops.clone()));

            let g2_to_g4_pending = Arc::new(PendingTracker::new());
            let presence_filter = Arc::new(
                ObjectPresenceFilter::<G2>::new(presence_checker)
                    .with_pending_tracker(g2_to_g4_pending.clone()),
            );

            let g2_to_g4_config = ObjectPipelineBuilder::<G2>::new()
                .policy(presence_filter)
                .pending_tracker(g2_to_g4_pending)
                .lock_manager(lock_manager)
                .build();

            engine_builder = engine_builder
                .with_object_ops(object_ops)
                .with_g2_to_g4_pipeline(g2_to_g4_config);

            tracing::info!("G2→G4 object storage pipeline configured with presence checking");
        } else {
            tracing::warn!(
                "Object storage configured but no parallel_worker available - G2→G4 pipeline disabled"
            );
        }
    }

    let primary_offload = match engine_builder.build() {
        Ok(offload_engine) => Some(Arc::new(offload_engine)),
        Err(e) => {
            tracing::warn!("Failed to build OffloadEngine: {e}. Continuing without offload.");
            None
        }
    };
    let mut offloads = primary_offload
        .map(|engine| vec![(primary_resource, engine)])
        .unwrap_or_default();
    if !bypass_host {
        for (&resource, registry) in &registries {
            if resource == primary_resource {
                continue;
            }
            let configured = &runtime.config().offload.g1_to_g2;
            let fallback;
            let offload_config = if configured.policies.is_empty() {
                fallback = kvbm_config::TierOffloadConfig {
                    policies: vec![kvbm_config::PolicyType::Presence],
                    ..Default::default()
                };
                &fallback
            } else {
                configured
            };
            let pending = Arc::new(PendingTracker::new());
            let policy = create_policy_from_config::<G1, G2>(
                offload_config,
                Arc::new(registry.clone()),
                Some(Arc::clone(&pending)),
            );
            let pipeline = PipelineBuilder::<G1, G2>::new()
                .resource(resource)
                .policy(policy)
                .pending_tracker(pending)
                .build();
            let capacity = leader.g2_capacity_for(resource).with_context(|| {
                format!("G2 capacity for secondary resource {resource:?} is missing")
            })?;
            let builder = OffloadEngine::builder(leader.clone())
                .with_g2_capacity(capacity)
                .with_runtime(runtime.tokio())
                .with_g1_to_g2_pipeline(pipeline);
            offloads.push((resource, Arc::new(builder.build()?)));
        }
    }

    // Step 8: refresh worker handler lists (workers registered new handlers
    // during init, invalidating the handshake-time cache).
    let worker_instance_ids = {
        let workers = c.workers.lock();
        workers.instance_ids.clone()
    };
    for instance_id in worker_instance_ids.iter() {
        runtime
            .messenger()
            .refresh_handlers(*instance_id)
            .await
            .with_context(|| format!("Failed to refresh handlers for worker {instance_id}"))?;
    }

    Ok(EngineStack {
        instance_leader: leader,
        offloads,
        primary_resource,
        admission,
        reference_config,
        handshake,
        events_manager,
        indexer_publisher,
        indexer_hub_client,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_parallel_resources_configure_independent_local_transfers() {
        let resource = LogicalResourceId(3);
        let placements = local_transfer_placements(&BTreeMap::from([(
            resource,
            ParallelismMode::TensorParallel,
        )]));
        assert_eq!(
            placements,
            vec![(resource, WorkerDataPlacement::TensorSharded)]
        );
    }

    #[test]
    fn mixed_resources_preserve_each_worker_transfer_route() {
        let attention = LogicalResourceId(2);
        let latent = LogicalResourceId(7);
        let placements = local_transfer_placements(&BTreeMap::from([
            (attention, ParallelismMode::TensorParallel),
            (latent, ParallelismMode::ReplicatedData),
        ]));
        assert_eq!(
            placements,
            vec![
                (attention, WorkerDataPlacement::TensorSharded),
                (latent, WorkerDataPlacement::ReplicatedG1StripedLower),
            ]
        );
    }
}
