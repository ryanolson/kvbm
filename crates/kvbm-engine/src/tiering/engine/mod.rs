// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-side support for the KVBM connector.
//!
//! This module is the canonical home for engine-only machinery that the
//! connector drives across the connector [`kvbm_protocols::connector::LeaderEngine`]
//! seam. It hosts:
//!
//! * [`reconcile`] — the search reconcile core that backs
//!   `get_num_new_matched_tokens` (`OnboardingShard`s, `num_computed_tokens`,
//!   the contiguous walk), with none of the connector-side conditional-disagg
//!   or watchdog glue. Because the core lives inside `kvbm-engine` it refers to
//!   leader types via `crate::leader::...`.
//! * [`local`] — [`LocalConnectorEngine`], the in-process `LeaderEngine` over
//!   an `Arc<InstanceLeader>` (match + onboard + lifecycle).
//! * [`find`] — the unified `find_blocks` / `onboard_blocks` router:
//!   engine-side window derivation, prefill-vs-local routing, the fresh-mint
//!   deferral guard, and the zero-external prefill fall-through (the
//!   source-agnostic seam face).
//! * [`driver`] — the in-process action completion map + lock-release-then-
//!   notify terminal.
//! * [`onboard`] — the off-forward-pass G1←G2 onboard fold.
//! * [`offload`] — the buffer→flush G1→G2 offload submission seam + fold.
//! * [`prefill`] — the prefill-side conditional-disagg cores
//!   (`prefill_accept_core` / `prefill_onboard_by_id` / `prefill_release`, driven
//!   by the `find`/`onboard_blocks` routers): attach to the decode-held session,
//!   pull the provided window into G2, and kick the external suffix G2→G1 at USAA.
//! * [`worker`] — [`WorkerEngine`], the per-TP-rank worker-side GPU pass
//!   runtime (DirectWorker + per-layer CUDA events + the injected completion
//!   delegate).

pub(crate) mod reconcile;

mod config;
mod driver;
mod find;
mod inflight;
mod local;
mod offload;
mod onboard;
mod prefill;
mod worker;

pub use config::{ConnectorEngineConfig, RemoteOps};
pub use worker::{PassOffload, PassOnboard, WorkerEngine, WorkerPassPlan};

use std::sync::Arc;

use kvbm_common::LogicalResourceId;
use kvbm_protocols::connector::{EngineWorkerSink, LeaderEngine, WorkerEngineDriver};

use crate::leader::InstanceLeader;
use crate::offload::OffloadEngine;
use local::{CdRuntime, LocalConnectorEngine};
use offload::{DisabledOffloadSubmit, OffloadEngineSubmit};

/// Build the in-process connector engine the connector drives, returned as BOTH of
/// its seam faces over the same object: the [`LeaderEngine`] the connector's
/// slot lifecycle calls, and the [`WorkerEngineDriver`] the leader's
/// forward-pass flush glue calls (`finish_forward_pass` submits the buffered
/// offloads once every worker's pass-completion event has merged).
///
/// This is the engine crate's construction entry point: the connector passes a
/// worker handshake's `Arc<InstanceLeader>`, the worker-delegate `sink`, a
/// [`ConnectorEngineConfig`] (the layout `block_size` plus the [`RemoteOps`]
/// selection), and — when offload is enabled — a real [`OffloadEngine`]. The
/// connector never names `LocalConnectorEngine` or the offload-submit seam;
/// this factory keeps both crate-internal. `offload: None` yields an
/// onboard-only engine (its offload submit refuses, folding each flush to
/// `Failed(AllBlocks)`).
///
/// `config.remote.search` is the single wiring path for the leader's remote
/// search: when `Some`, construction installs the discovery on the leader
/// *before* building the engine, then arms the engine's `search_remote` knob
/// so its shard finds request the remote path. `None` installs nothing and
/// leaves `search_remote` off.
pub fn build_local_connector_engine(
    leader: Arc<InstanceLeader>,
    sink: Arc<dyn EngineWorkerSink>,
    config: ConnectorEngineConfig,
    offload: Option<Arc<OffloadEngine>>,
) -> (Arc<dyn LeaderEngine>, Arc<dyn WorkerEngineDriver>) {
    let primary_offload = offload.clone();
    let offload_submit: Arc<dyn offload::OffloadSubmit> = match offload {
        Some(offload) => Arc::new(OffloadEngineSubmit::new(offload)),
        None => Arc::new(DisabledOffloadSubmit),
    };
    build_local_connector_engine_inner(leader, sink, config, primary_offload, offload_submit)
}

/// Build one connector engine with independently owned offload pipelines for
/// every logical model resource.
///
/// Legacy offload calls route to `primary_resource`; explicit resource calls
/// fail closed unless the supplied set owns that exact resource.
pub fn build_local_connector_engine_with_resources(
    leader: Arc<InstanceLeader>,
    sink: Arc<dyn EngineWorkerSink>,
    config: ConnectorEngineConfig,
    primary_resource: LogicalResourceId,
    offloads: Vec<(LogicalResourceId, Arc<OffloadEngine>)>,
) -> anyhow::Result<(Arc<dyn LeaderEngine>, Arc<dyn WorkerEngineDriver>)> {
    let submit = OffloadEngineSubmit::from_resources(primary_resource, offloads)?;
    let primary_offload = Some(Arc::clone(submit.primary_engine()));
    Ok(build_local_connector_engine_inner(
        leader,
        sink,
        config,
        primary_offload,
        Arc::new(submit),
    ))
}

fn build_local_connector_engine_inner(
    leader: Arc<InstanceLeader>,
    sink: Arc<dyn EngineWorkerSink>,
    config: ConnectorEngineConfig,
    primary_offload: Option<Arc<OffloadEngine>>,
    offload_submit: Arc<dyn offload::OffloadSubmit>,
) -> (Arc<dyn LeaderEngine>, Arc<dyn WorkerEngineDriver>) {
    let ConnectorEngineConfig { block_size, remote } = config;
    let RemoteOps { search, disagg } = remote;

    // Install the discovery on the leader before constructing the engine — this
    // is the only place the remote-search discovery is wired. `set_remote_discovery`
    // is first-write-wins; a `false` means a discovery was already installed
    // (the engine is being rebuilt over a leader that was already wired, or two
    // engines share one leader). Keep the existing handle and proceed, but warn
    // loudly so the double-wiring is debuggable.
    if let Some(search) = &search
        && !leader.set_remote_discovery(search.discovery.clone())
    {
        tracing::warn!(
            instance_id = %leader.messenger().instance_id(),
            "build_local_connector_engine: remote.search is Some but a remote-search \
             discovery was already installed on this leader; keeping the existing \
             handle (double-wiring?)"
        );
    }

    // Shard finds request the remote path iff remote search is configured.
    let search_remote = search.is_some();

    // Build the conditional-disagg runtime from the sibling `disagg` field (the
    // Disagg path is test-only — no production constructor assembles its
    // transports, so the connector's `RemoteOps::default()` yields `None`).
    let cd =
        disagg.map(|d| CdRuntime::new(d.cfg, d.tier, d.sessions, d.prefill_plane, d.peer_resolver));

    // Prefill OUTPUT capture: hook the CD runtime's register observer onto the
    // offload pipeline's G1→G2 register step, before the engine Arc is
    // consumed into the submit seam. Registration failure (no G1→G2 pipeline)
    // must not fail construction — a decode-only deployment works without it
    // — but a prefill-role engine without it cannot produce remote-prefill
    // output, so warn loudly.
    if let (Some(cd), Some(offload)) = (cd.as_ref(), primary_offload.as_ref()) {
        let observer = Arc::clone(&cd.output);
        if let Err(e) = offload.add_g1_to_g2_register_observer(Arc::new(
            move |blocks: &[kvbm_logical::blocks::ImmutableBlock<crate::G2>]| {
                observer.observe(blocks)
            },
        )) {
            tracing::warn!(
                error = %e,
                "conditional disagg: no G1→G2 offload pipeline to observe; a \
                 decode-only deployment works without it, but a prefill-role \
                 engine cannot produce remote-prefill output"
            );
        }
    }

    let engine = LocalConnectorEngine::with_offload_submit(
        leader,
        sink,
        block_size,
        search_remote,
        offload_submit,
        cd,
    );
    (
        Arc::clone(&engine) as Arc<dyn LeaderEngine>,
        engine as Arc<dyn WorkerEngineDriver>,
    )
}
