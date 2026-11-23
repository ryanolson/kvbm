// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing")]

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::FutureExt;
use kvbm_engine::p2p::session::{MockSessionFactory, SessionFactory};
use kvbm_hub::features::prefill_router::{
    DispatchOutcome, PrefillRouterManager, SelectorConfig, VeloExecutionBackend,
    register_prefill_worker,
};
use kvbm_protocols::disagg::{DISAGG_PROTOCOL_VERSION, RemotePrefillRequest};
use velo::Velo;
use velo::transports::tcp::TcpTransportBuilder;

async fn peer() -> Result<Velo> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let transport = TcpTransportBuilder::new()
        .from_listener(listener)?
        .build()?;
    Velo::builder()
        .add_transport(Arc::new(transport))
        .build()
        .await
}

#[tokio::test]
async fn prefill_validation_hub_opt_in_attaches_once() -> Result<()> {
    let hub = peer().await?;
    let worker = peer().await?;
    hub.register_peer(worker.peer_info())?;
    worker.register_peer(hub.peer_info())?;

    let selector = SelectorConfig {
        per_worker_concurrency: 1,
        block_size: 4,
    };
    let strict = PrefillRouterManager::new(selector);
    let enabled = PrefillRouterManager::with_options(selector, None, true);
    for manager in [&strict, &enabled] {
        manager.selector().add_worker(
            worker.instance_id(),
            VeloExecutionBackend::new(worker.instance_id(), hub.messenger().clone()),
        );
    }

    let (holder, puller) = MockSessionFactory::make_paired();
    let session_id = uuid::Uuid::new_v4();
    let source = holder.open(session_id)?;
    let mut inbox = register_prefill_worker(worker.messenger(), 2, {
        let puller = Arc::clone(&puller);
        move |dispatch| {
            let params = dispatch
                .kv_transfer_params
                .remote_prefill
                .as_ref()
                .context("missing prefill params")?;
            params.validate_for_fpst(&dispatch.request_id, dispatch.token_ids.len())?;
            let endpoint = params.decode_endpoint.clone().context("missing endpoint")?;
            // Stand in for an FPST-enabled prefill peer at its session attach
            // boundary. The mock records each call and resolves immediately.
            puller
                .attach(params.session_id, params.initiator_instance_id, endpoint)
                .now_or_never()
                .expect("mock attach resolves immediately")?;
            Ok(())
        }
    })?;

    let request = RemotePrefillRequest {
        protocol_version: 2,
        request_id: "fpst".into(),
        session_id,
        initiator_instance_id: hub.instance_id(),
        decode_endpoint: source.endpoint(),
        token_ids: vec![0; 5],
        num_provided_tokens: 4,
        request: Default::default(),
        scope_salt: None,
        expected_hash_digest: None,
        bundle: None,
    };

    assert!(matches!(
        strict.router().submit(request.clone()).await,
        DispatchOutcome::Rejected { reason } if reason.contains("UnsupportedProtocolVersion")
    ));
    let mut unknown = request.clone();
    unknown.protocol_version = u16::MAX;
    assert!(matches!(
        enabled.router().submit(unknown).await,
        DispatchOutcome::Rejected { reason } if reason.contains("UnsupportedProtocolVersion")
    ));

    let mut old = request.clone();
    old.protocol_version = DISAGG_PROTOCOL_VERSION;
    assert!(matches!(
        enabled.router().submit(old).await,
        DispatchOutcome::Rejected { reason }
            if reason.contains("protocol version 1") && reason.contains("requires 2")
    ));
    assert!(puller.attach_calls().is_empty());
    assert!(inbox.try_recv().is_err());

    assert_eq!(
        enabled.router().submit(request).await,
        DispatchOutcome::Accepted
    );
    let assignment = inbox.try_recv()?;
    assert_eq!(assignment.request.request_id, "fpst");
    assert_eq!(puller.attach_calls().len(), 1);
    assert_eq!(puller.attach_calls()[0].0, session_id);
    Ok(())
}
