// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binding-surface assertions + a behavioral lifecycle test.
//!
//! These reproduce the call shapes the PyO3 connector bindings
//! (`src/connector/{leader,worker}`) use against the `kvbm_connector`
//! crate: the engine owns conditional disagg, so the binding's route is
//! always `Direct` (there is no connector-side `Cd`
//! dispatcher route).
//! The `*_binding_surface` fns are never run — type-checking them is the
//! assertion; their dead code is expected. The lifecycle `#[test]` drives the
//! eight [`ConnectorLeaderApi`] methods over a `NoopBlockEngine`.

use std::collections::HashSet;
use std::sync::Arc;

use kvbm_common::{KvBlockLayout, KvDimLayout};

use crate::common::{
    ConsolidatorEndpoints, FinishedStatus, KvConnectorMetadata, Request, SchedulerOutput,
};
use crate::connector::leader::ConnectorLeaderApi;
use crate::{BlockId, EventSource, InstanceId, KvbmRuntime, TensorDescriptor, WorkerAddress};

use super::engine::noop_leader_engine;
use super::leader::Leader;
use super::worker::{ConnectorWorkerInterface, FinishedRequests, Worker};

// ---- trait-bound assertions (mirror the binding's `use` imports) ----

fn assert_leader_api<T: ConnectorLeaderApi>() {}
fn assert_worker_iface<T: ConnectorWorkerInterface>() {}

#[test]
fn trait_bounds_match_binding_imports() {
    // The api rides `Arc<Leader>` (never the bare struct — see the `Leader`
    // doc); the worker interface rides `Worker`.
    assert_leader_api::<Arc<Leader>>();
    assert_worker_iface::<Worker>();
}

// ---- leader surface: the binding's `ApiRoute` retargeted to `Leader` ----

/// Mirror of the leader binding's call surface. The engine owns conditional
/// disagg outright, so there is no connector-side dispatcher and the only
/// route is `Direct`, which holds `&Arc<Leader>` (not
/// `Arc<dyn ConnectorLeaderApi>`) precisely because the trait impl lives on
/// `Arc<Leader>`, never on `Leader`.
enum ApiRoute<'a> {
    Direct(&'a Arc<Leader>),
}

impl ApiRoute<'_> {
    fn create_slot(&self, request: Request) -> anyhow::Result<()> {
        match self {
            ApiRoute::Direct(leader) => leader.create_slot(request),
        }
    }
    fn has_slot(&self, request_id: &str) -> bool {
        match self {
            ApiRoute::Direct(leader) => leader.has_slot(request_id),
        }
    }
    fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> anyhow::Result<()> {
        match self {
            ApiRoute::Direct(leader) => leader.extend_slot_tokens(request_id, tokens),
        }
    }
    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(Option<usize>, bool)> {
        match self {
            ApiRoute::Direct(leader) => {
                leader.get_num_new_matched_tokens(request_id, num_computed_tokens)
            }
        }
    }
    fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> anyhow::Result<()> {
        match self {
            // UFCS against `self: &Arc<Self>`, exactly as the binding calls it.
            ApiRoute::Direct(leader) => {
                Leader::update_state_after_alloc(leader, request_id, block_ids, num_external_tokens)
            }
        }
    }
    fn build_connector_meta(&self, output: SchedulerOutput) -> anyhow::Result<KvConnectorMetadata> {
        match self {
            ApiRoute::Direct(leader) => leader.build_connector_meta(output),
        }
    }
    fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> anyhow::Result<()> {
        match self {
            ApiRoute::Direct(leader) => {
                leader.update_connector_output(finished_sending, finished_recving)
            }
        }
    }
    fn request_finished(&self, request_id: &str) -> FinishedStatus {
        match self {
            ApiRoute::Direct(leader) => leader.request_finished(request_id),
        }
    }
}

/// Compile-only proof of every inherent + `ConnectorLeaderApi` call the leader
/// binding makes. Never invoked; type-checking is the assertion.
#[allow(dead_code)]
fn leader_binding_surface(
    runtime: Arc<KvbmRuntime>,
    instance_id: InstanceId,
    worker_address: WorkerAddress,
) -> anyhow::Result<()> {
    let endpoints = ConsolidatorEndpoints {
        vllm_zmq_endpoint: None,
        egress_endpoint: String::new(),
        // EventSource: FromStr (the binding does `.parse::<EventSource>()`).
        engine_source: "vllm".parse::<EventSource>().map_err(anyhow::Error::msg)?,
    };
    let leader: Arc<Leader> = Arc::new(Leader::new_with_consolidator(runtime, 16, Some(endpoints)));

    // Inherent methods the binding calls directly on `self.inner`.
    let _: usize = leader.block_size();
    let _: usize = leader.get_slot_total_tokens("r")?;
    leader.register_worker(0, instance_id, worker_address)?;
    leader.initialize()?; // self: &Arc<Self>

    // The single direct route; every `ConnectorLeaderApi` method.
    let route = ApiRoute::Direct(&leader);
    route.create_slot(sample_request())?;
    let _: bool = route.has_slot("r");
    route.extend_slot_tokens("r", vec![1])?;
    let _: (Option<usize>, bool) = route.get_num_new_matched_tokens("r", 0)?;
    let block_ids: Vec<BlockId> = Vec::new();
    route.update_state_after_alloc("r", block_ids, 0)?;
    let _: KvConnectorMetadata = route.build_connector_meta(SchedulerOutput::new(0))?;
    route.update_connector_output(HashSet::new(), HashSet::new())?;
    let _: FinishedStatus = route.request_finished("r");
    Ok(())
}

/// Compile-only proof of every inherent + `ConnectorWorkerInterface` call the
/// worker binding makes. Never invoked; type-checking is the assertion.
#[allow(dead_code)]
fn worker_binding_surface(
    runtime: Arc<KvbmRuntime>,
    tensor: Arc<dyn TensorDescriptor>,
    dim_layout: KvDimLayout,
    block_layout: KvBlockLayout,
    metadata: KvConnectorMetadata,
) -> anyhow::Result<()> {
    let worker = Worker::new(runtime);

    // Inherent.
    let _: Vec<u8> = worker.handshake_metadata()?;

    // `ConnectorWorkerInterface` — every method the binding calls.
    worker.register_kv_caches(
        vec![Arc::clone(&tensor)],
        4,
        2,
        dim_layout.clone(),
        block_layout,
    )?;
    worker.register_cross_layers_kv_cache(tensor, 4, 2, dim_layout, block_layout)?;
    if metadata.should_bind() {
        worker.bind_connector_metadata(metadata)?;
    }
    worker.clear_connector_metadata()?;
    worker.start_load_kv()?;
    worker.wait_for_layer_load(0, 0)?;
    worker.save_kv_layer(0, 0)?;
    worker.wait_for_save()?;
    let _: bool = worker.is_initialized();
    worker.shutdown()?;
    let finished: FinishedRequests = worker.get_finished();
    let (_offloading, _onboarding): (HashSet<String>, HashSet<String>) = finished.dissolve();
    let _: HashSet<usize> = worker.get_failed_onboarding();
    Ok(())
}

fn sample_request() -> Request {
    Request::with_token_limits("r", vec![1u32, 2, 3], None, None, None, None, None)
}

// ---- serde wire contract the bindings depend on ----

#[test]
fn kv_connector_metadata_serde_roundtrips_by_name() {
    // Leader binding does `serde_json::to_vec`; worker binding does
    // `serde_json::from_slice`. The field names are the contract.
    let meta = KvConnectorMetadata::new(7);
    let bytes = serde_json::to_vec(&meta).unwrap();
    let back: KvConnectorMetadata = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(back.iteration, 7);
    assert_eq!(meta.should_bind(), back.should_bind());
}

// ---- behavioral: the eight `ConnectorLeaderApi` methods over `NoopBlockEngine` ----

#[test]
fn connector_leader_api_cold_lifecycle_over_noop() {
    let leader: Arc<Leader> = Arc::new(Leader::with_engine(noop_leader_engine(), 4));
    // `&Arc<Leader> -> &dyn ConnectorLeaderApi` doubles as the positive bound
    // assertion and drives every call through the trait, as the binding does.
    let api: &dyn ConnectorLeaderApi = &leader;

    api.create_slot(sample_request()).unwrap();
    assert!(api.has_slot("r"));

    api.extend_slot_tokens("r", vec![4, 5]).unwrap();

    // The Noop engine never matches; the skeleton GNMT reports a zero hit.
    let (matched, load_async) = api.get_num_new_matched_tokens("r", 0).unwrap();
    assert_eq!(matched, Some(0));
    assert!(!load_async);

    let block_ids: Vec<BlockId> = vec![10, 11, 12];
    api.update_state_after_alloc("r", block_ids, 0).unwrap();

    let meta = api.build_connector_meta(SchedulerOutput::new(0)).unwrap();
    assert_eq!(meta.iteration, 0);

    // Noop drains immediately ⇒ `Finished`, slot reaped inline.
    assert_eq!(api.request_finished("r"), FinishedStatus::Finished);
    assert!(!api.has_slot("r"));

    // The terminal sweep is a harmless no-op once the slot is already gone.
    api.update_connector_output(HashSet::new(), HashSet::new())
        .unwrap();
    assert!(!api.has_slot("r"));
}
