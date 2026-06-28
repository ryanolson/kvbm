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

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};

use kvbm_common::placement::StripedBlockPlacement;
use kvbm_engine::leader::{ConsolidatorParams, InstanceLeader};
use kvbm_engine::object::{ObjectLockManager, create_lock_manager, create_object_client};
use kvbm_engine::offload::{
    ObjectPipelineBuilder, ObjectPresenceFilter, OffloadEngine, PendingTracker, PipelineBuilder,
    S3PresenceChecker, create_policy_from_config,
};
use kvbm_engine::worker::{CollectiveBootstrap, LeaderLayoutConfig, Worker};
use kvbm_hub::HubClient;
use kvbm_logical::blocks::{BlockDuplicationPolicy, BlockRegistry};
use kvbm_logical::events::{EventsManager, KvbmCacheEventsPublisher};
use kvbm_logical::manager::{BlockManager, FrequencyTrackingCapacity};
use kvbm_physical::layout::LayoutConfig;

use crate::connector::leader::hub_handshake::{self, HubHandshake};
use crate::connector::leader::hub_indexer;
use crate::{G1, G2, G3, KvbmRuntime};

use super::Construction;

/// The leader-side engine stack produced by [`build_engine_stack`]. The caller
/// (`Leader::initialize`) clones `instance_leader` for the CD wiring before
/// moving it into the `LeaderEngine` via `build_local_connector_engine`.
pub(super) struct EngineStack {
    pub(super) instance_leader: Arc<InstanceLeader>,
    pub(super) offload: Option<Arc<OffloadEngine>>,
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

    // Step 2: validate all worker configs match the rank-0 reference.
    let reference_config = layout_configs[0].clone();
    for (i, config) in layout_configs.iter().enumerate().skip(1) {
        if config.num_layers != reference_config.num_layers {
            bail!(
                "Layout config mismatch: worker {i} has {} layers, worker 0 has {}",
                config.num_layers,
                reference_config.num_layers
            );
        }
        if config.outer_dim != reference_config.outer_dim {
            bail!(
                "Layout config mismatch: worker {i} has outer_dim {}, worker 0 has {}",
                config.outer_dim,
                reference_config.outer_dim
            );
        }
        if config.page_size != reference_config.page_size {
            bail!(
                "Layout config mismatch: worker {i} has page_size {}, worker 0 has {}",
                config.page_size,
                reference_config.page_size
            );
        }
        if config.inner_dim != reference_config.inner_dim {
            bail!(
                "Layout config mismatch: worker {i} has inner_dim {}, worker 0 has {}",
                config.inner_dim,
                reference_config.inner_dim
            );
        }
        if config.dtype_width_bytes != reference_config.dtype_width_bytes {
            bail!(
                "Layout config mismatch: worker {i} has dtype_width_bytes {}, worker 0 has {}",
                config.dtype_width_bytes,
                reference_config.dtype_width_bytes
            );
        }
        if config.num_heads != reference_config.num_heads {
            bail!(
                "Layout config mismatch: worker {i} has num_heads {:?}, worker 0 has {:?}",
                config.num_heads,
                reference_config.num_heads
            );
        }
    }

    // Step 3: compute G2/G3 block counts + host-bypass sentinel.
    let bytes_per_block = reference_config.required_bytes() / reference_config.num_blocks;
    let host_block_count = runtime
        .config()
        .cache
        .host
        .compute_num_blocks(bytes_per_block);
    let disk_block_count = runtime
        .config()
        .cache
        .disk
        .as_ref()
        .and_then(|dc| dc.compute_num_blocks(bytes_per_block));

    // At least one cache tier must produce a non-zero block count, else the
    // leader has nothing to offload to. Fail loudly (mirrors legacy sanity check).
    let host_ok = host_block_count.is_some_and(|n| n > 0);
    let disk_ok = disk_block_count.is_some_and(|n| n > 0);
    if !host_ok && !disk_ok {
        bail!(
            "KVBM Configuration Error: at least one cache tier must be configured \
             (DYN_KVBM_CPU_CACHE_GB for G2, or DYN_KVBM_DISK_CACHE_GB for G3)."
        );
    }
    let host_block_count = host_block_count.unwrap_or(0);
    let worker_count = c.workers.lock().connector_clients.len();
    let parallelism = resolve_parallelism(runtime.config().cache.parallelism, &reference_config);
    if parallelism != runtime.config().cache.parallelism {
        tracing::info!(
            configured = ?runtime.config().cache.parallelism,
            resolved = ?parallelism,
            "Registered cache has no HeadCount axis; selecting replicated-data placement"
        );
    }
    let logical_host_block_count =
        logical_tier_block_count(host_block_count, parallelism, worker_count)?;
    let logical_disk_block_count = disk_block_count
        .map(|count| logical_tier_block_count(count, parallelism, worker_count))
        .transpose()?;
    let collective = build_collective_bootstrap(parallelism, worker_count)?;

    // Host-bypass: disk configured, host not — serve disk hits to GPU directly,
    // no G2 staging. InstanceLeader still requires a G2 manager, so build it with
    // a sentinel block_count of 1 (BlockManager rejects 0; it allocates nothing).
    let bypass_host = runtime.config().cache.bypass_host_cache();
    let g2_manager_block_count = if bypass_host {
        logical_host_block_count.max(1)
    } else {
        logical_host_block_count
    };

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

    // Store metadata + configure transfer-client layout handles.
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

    // Step 5: block registry (wired to the EventsManager when present) + G2/G3.
    let mut registry_builder = BlockRegistry::builder()
        .frequency_tracker(FrequencyTrackingCapacity::Medium.create_tracker());
    if let Some(em) = events_manager.clone() {
        registry_builder = registry_builder.event_manager(em);
    }
    let registry = registry_builder.build();
    let logical_metrics = runtime.observability().logical_aggregator();
    let g2_manager = Arc::new(
        BlockManager::<G2>::builder()
            .block_count(g2_manager_block_count)
            .block_size(reference_config.page_size)
            .registry(registry.clone())
            .with_lineage_backend()
            .aggregator(logical_metrics.clone())
            .duplication_policy(BlockDuplicationPolicy::Reject)
            .build()
            .expect("Should build G2 manager"),
    );
    let g3_manager: Option<Arc<BlockManager<G3>>> = logical_disk_block_count.map(|count| {
        Arc::new(
            BlockManager::<G3>::builder()
                .block_count(count)
                .block_size(reference_config.page_size)
                .registry(registry.clone())
                .with_lineage_backend()
                .aggregator(logical_metrics.clone())
                .duplication_policy(BlockDuplicationPolicy::Reject)
                .build()
                .expect("Should build G3 manager"),
        )
    });

    // Clone registry + managers for the OffloadEngine (shared state via Arcs).
    let registry_for_offload = Arc::new(registry.clone());
    let g2_manager_for_offload = g2_manager.clone();
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
        .registry(registry)
        .g2_manager(g2_manager)
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
    let template = kvbm_engine::leader::parallelism::ParallelismTemplate::from_layout_config(
        &reference_config,
        parallelism,
        num_workers,
    )?;
    leader_builder = leader_builder.parallelism_template(template);
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
    let mut engine_builder = OffloadEngine::builder(leader.clone())
        .with_registry(registry_for_offload.clone())
        .with_g2_manager(g2_manager_for_offload)
        .with_runtime(runtime.tokio());

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

    let offload = match engine_builder.build() {
        Ok(offload_engine) => Some(Arc::new(offload_engine)),
        Err(e) => {
            tracing::warn!("Failed to build OffloadEngine: {e}. Continuing without offload.");
            None
        }
    };

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
        offload,
        reference_config,
        handshake,
        events_manager,
        indexer_publisher,
        indexer_hub_client,
    })
}

/// Resolve physical cache distribution from the registered tensor schema.
///
/// A labelled cache without `HeadCount` has no tensor-parallel shard axis.
/// This is the latent/payload layout used by MLA, so all G1 ranks contain the
/// same data and lower tiers must use replicated-data placement. Layouts with
/// an explicit head axis keep the operator-configured mode.
fn resolve_parallelism(
    configured: kvbm_config::ParallelismMode,
    layout: &LayoutConfig,
) -> kvbm_config::ParallelismMode {
    if layout.num_heads.is_none() {
        kvbm_config::ParallelismMode::ReplicatedData
    } else {
        configured
    }
}

fn collective_required(parallelism: kvbm_config::ParallelismMode, worker_count: usize) -> bool {
    parallelism == kvbm_config::ParallelismMode::ReplicatedData && worker_count > 1
}

fn build_collective_bootstrap(
    parallelism: kvbm_config::ParallelismMode,
    worker_count: usize,
) -> Result<Option<CollectiveBootstrap>> {
    if !collective_required(parallelism, worker_count) {
        return Ok(None);
    }

    #[cfg(feature = "nccl")]
    {
        let bootstrap = kvbm_engine::collectives::NcclBootstrap::generate(worker_count)
            .context("generating KVBM NCCL bootstrap for replicated cache workers")?;
        Ok(Some(CollectiveBootstrap::Nccl {
            serialized: bootstrap.serialize(),
        }))
    }

    #[cfg(not(feature = "nccl"))]
    bail!(
        "replicated cache data with {worker_count} workers requires the kvbm-connector `nccl` feature"
    )
}

/// Logical tier capacity represented by equal per-worker physical capacities.
///
/// Tensor-parallel blocks are sharded, so one block consumes one slot on every
/// rank and the logical capacity equals the per-rank capacity. Replicated MLA
/// blocks have one canonical lower-tier copy, so each rank contributes a
/// disjoint stripe and the logical capacity is the aggregate across ranks.
fn logical_tier_block_count(
    per_worker_blocks: usize,
    parallelism: kvbm_config::ParallelismMode,
    worker_count: usize,
) -> Result<usize> {
    if worker_count == 0 {
        bail!("cannot configure cache tiers without workers");
    }
    match parallelism {
        kvbm_config::ParallelismMode::TensorParallel => Ok(per_worker_blocks),
        kvbm_config::ParallelismMode::ReplicatedData => {
            StripedBlockPlacement::new(worker_count)?.global_capacity(per_worker_blocks)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_collective_bootstrap, collective_required, logical_tier_block_count,
        resolve_parallelism,
    };
    use kvbm_config::ParallelismMode;
    use kvbm_physical::layout::LayoutConfig;

    fn layout(num_heads: Option<usize>) -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(128)
            .num_layers(2)
            .outer_dim(if num_heads.is_some() { 2 } else { 1 })
            .page_size(16)
            .inner_dim(512)
            .num_heads(num_heads)
            .build()
            .unwrap()
    }

    #[test]
    fn mla_layout_without_head_axis_selects_replicated_data() {
        assert_eq!(
            resolve_parallelism(ParallelismMode::TensorParallel, &layout(None)),
            ParallelismMode::ReplicatedData
        );
    }

    #[test]
    fn only_multi_worker_replicated_data_requires_a_collective() {
        assert!(!collective_required(ParallelismMode::ReplicatedData, 1));
        assert!(collective_required(ParallelismMode::ReplicatedData, 2));
        assert!(!collective_required(ParallelismMode::TensorParallel, 2));
    }

    #[cfg(feature = "nccl")]
    #[test]
    fn replicated_group_gets_one_decodable_nccl_bootstrap() {
        let collective = build_collective_bootstrap(ParallelismMode::ReplicatedData, 2)
            .unwrap()
            .expect("TP=2 replicated data requires a collective");
        let kvbm_engine::worker::CollectiveBootstrap::Nccl { serialized } = collective;
        let bootstrap = kvbm_engine::collectives::NcclBootstrap::deserialize(&serialized).unwrap();
        assert_eq!(bootstrap.world_size(), 2);
    }

    #[test]
    fn mha_layout_keeps_configured_parallelism() {
        assert_eq!(
            resolve_parallelism(ParallelismMode::TensorParallel, &layout(Some(4))),
            ParallelismMode::TensorParallel
        );
        assert_eq!(
            resolve_parallelism(ParallelismMode::ReplicatedData, &layout(Some(4))),
            ParallelismMode::ReplicatedData
        );
    }

    #[test]
    fn replicated_tiers_aggregate_worker_capacity() {
        assert_eq!(
            logical_tier_block_count(2_000, ParallelismMode::ReplicatedData, 2).unwrap(),
            4_000
        );
    }

    #[test]
    fn tensor_parallel_tiers_keep_per_worker_block_capacity() {
        assert_eq!(
            logical_tier_block_count(2_000, ParallelismMode::TensorParallel, 2).unwrap(),
            2_000
        );
    }

    #[test]
    fn tier_capacity_rejects_empty_groups_and_overflow() {
        assert!(logical_tier_block_count(1, ParallelismMode::ReplicatedData, 0).is_err());
        assert!(logical_tier_block_count(usize::MAX, ParallelismMode::ReplicatedData, 2).is_err());
    }
}
