// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use std::sync::Arc;

use kvbm_hub::handlers::{HEARTBEAT_HANDLER, HeartbeatAck, HeartbeatRequest};
use kvbm_hub::protocol::{
    ConditionalDisaggConfig, ConditionalDisaggRole, DISAGG_PROTOCOL_VERSION, ErrorBody, ErrorCode,
    Feature, HeartbeatResponse, LayoutCompatPayload, ListInstancesResponse,
    MUTATION_CREDENTIAL_HEADER, MutationCredential, P2pConfig, PeerLookupResponse, PrefillRequest,
    ProbeResponse, RegisterRequest, RegisterResponse, instance_by_id, instance_heartbeat,
    instance_probe, paths, peers_by_instance, peers_by_worker,
};
use kvbm_hub::{
    ConditionalDisaggClient, ConditionalDisaggInstancesResponse, ConditionalDisaggManager,
    HubClientBuilder, HubServer, InMemoryRegistry, PeerRegistry,
};
use velo::Transport;
use velo::discovery::PeerDiscovery;
use velo::transports::tcp::TcpTransportBuilder;
use velo_ext::{InstanceId, PeerInfo, WorkerAddress};

use dynamo_tokens::TokenBlockSequence;
use kvbm_hub::IndexerManager;
use kvbm_logical::events::{KvCacheEvents, KvbmCacheEvents};
use kvbm_logical::{KvbmSequenceHashProvider, SequenceHash};

// ---- fixtures ---------------------------------------------------------------

async fn start_server() -> HubServer {
    kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .serve()
        .await
        .expect("start test server")
}

fn make_peer() -> PeerInfo {
    let id = InstanceId::new_v4();
    PeerInfo::new(id, WorkerAddress::from_encoded(b"test-addr".to_vec()))
}

fn build_client(server: &HubServer) -> std::sync::Arc<kvbm_hub::HubClient> {
    kvbm_hub::create_client_builder()
        .host("127.0.0.1")
        .discovery_port(server.discovery_addr().port())
        .control_port(server.control_addr().port())
        .build()
        .expect("build client")
}

fn discovery_url(server: &HubServer, path: &str) -> String {
    format!("http://{}{}", server.discovery_addr(), path)
}

fn control_url(server: &HubServer, path: &str) -> String {
    format!("http://{}{}", server.control_addr(), path)
}

/// Build a minimal valid `LayoutCompatPayload` for tests that need to
/// pass the P2P gate without caring about specific shape values.
fn test_layout_compat_payload() -> LayoutCompatPayload {
    use kvbm_common::shape::CanonicalBlockShape;
    use kvbm_common::{BlockLayoutMode, KvBlockLayout};
    use kvbm_protocols::control::LayoutConfigDescription;
    LayoutCompatPayload {
        mode: BlockLayoutMode::Operational,
        canonical: Some(CanonicalBlockShape {
            num_layers_total: 4,
            outer_dim: 2,
            page_size: 16,
            num_heads_total: 8,
            head_dim: 64,
            dtype_width_bytes: 2,
        }),
        per_worker_layout: KvBlockLayout::OperationalNHD,
        per_worker_config: LayoutConfigDescription {
            num_blocks: 16,
            num_layers: 4,
            outer_dim: 2,
            page_size: 16,
            inner_dim: 8 * 64,
            alignment: 256,
            dtype_width_bytes: 2,
            num_heads: Some(8),
        },
        block_region_sizes: None,
        tp_size: 1,
        pp_size: 1,
    }
}

/// Build the standard P2P + CD feature bundle for tests that exercise
/// the post-c2 mandatory-gate path.
fn p2p_cd_features(role: ConditionalDisaggRole) -> Vec<Feature> {
    vec![
        Feature::P2P(P2pConfig {
            layout_compat: test_layout_compat_payload(),
        }),
        Feature::ConditionalDisagg(ConditionalDisaggConfig { role }),
    ]
}

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

async fn register_plain_peer(
    server: &HubServer,
    peer: &PeerInfo,
    current_credential: Option<&MutationCredential>,
) -> RegisterResponse {
    let mut request = http()
        .post(control_url(server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
            runtime: None,
        });
    if let Some(credential) = current_credential {
        request = request.header(MUTATION_CREDENTIAL_HEADER, credential.to_header_value());
    }
    request.send().await.unwrap().json().await.unwrap()
}

async fn heartbeat_status(
    server: &HubServer,
    owner: InstanceId,
    credential: Option<&MutationCredential>,
) -> reqwest::StatusCode {
    let mut request = http().post(control_url(server, &instance_heartbeat(owner)));
    if let Some(credential) = credential {
        request = request.header(MUTATION_CREDENTIAL_HEADER, credential.to_header_value());
    }
    request.send().await.unwrap().status()
}

// ---- handlers module --------------------------------------------------------

#[test]
fn heartbeat_handler_name_is_stable() {
    assert_eq!(HEARTBEAT_HANDLER, "kvbm_hub_heartbeat");
}

#[test]
fn heartbeat_request_default() {
    let req = HeartbeatRequest::default();
    assert_eq!(req.seq, 0);
}

#[test]
fn heartbeat_ack_default() {
    let ack = HeartbeatAck::default();
    assert_eq!(ack.seq, 0);
    assert!(!ack.ok);
}

#[test]
fn heartbeat_request_serde_round_trip() {
    let orig = HeartbeatRequest { seq: 99 };
    let json = serde_json::to_string(&orig).unwrap();
    let back: HeartbeatRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.seq, 99);
}

#[test]
fn heartbeat_ack_serde_round_trip() {
    let orig = HeartbeatAck { seq: 7, ok: true };
    let json = serde_json::to_string(&orig).unwrap();
    let back: HeartbeatAck = serde_json::from_str(&json).unwrap();
    assert_eq!(back.seq, 7);
    assert!(back.ok);
}

#[tokio::test]
async fn create_heartbeat_handler_builds() {
    let server = start_server().await;
    let client = build_client(&server);
    let _handler = kvbm_hub::handlers::create_heartbeat_handler(client);
}

// ---- HTTP-direct server tests -----------------------------------------------

#[tokio::test]
async fn health_check_discovery_port() {
    let server = start_server().await;
    let resp = http()
        .get(discovery_url(&server, paths::HEALTH))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn health_check_control_port() {
    let server = start_server().await;
    let resp = http()
        .get(control_url(&server, paths::HEALTH))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn register_success() {
    let server = start_server().await;
    let peer = make_peer();
    let resp = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
            runtime: None,
        })
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: RegisterResponse = resp.json().await.unwrap();
    assert_eq!(body.instance_id, peer.instance_id());
    assert!(body.mutation_credential.is_some());
}

#[tokio::test]
async fn active_reregister_requires_current_credential_and_rotates_it() {
    let server = start_server().await;
    let peer = make_peer();
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: Vec::new(),
        runtime: None,
    };
    let first: RegisterResponse = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let first_credential = first.mutation_credential.unwrap();

    let unauthenticated = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(unauthenticated.status(), 401);

    let second: RegisterResponse = http()
        .post(control_url(&server, paths::INSTANCES))
        .header(
            kvbm_hub::protocol::MUTATION_CREDENTIAL_HEADER,
            first_credential.to_header_value(),
        )
        .json(&req)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let second_credential = second.mutation_credential.unwrap();
    assert_ne!(first_credential, second_credential);

    let stale = http()
        .post(control_url(&server, paths::INSTANCES))
        .header(
            kvbm_hub::protocol::MUTATION_CREDENTIAL_HEADER,
            first_credential.to_header_value(),
        )
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(stale.status(), 401);
}

#[tokio::test]
async fn unregister_success() {
    let server = start_server().await;
    let peer = make_peer();
    let registration: RegisterResponse = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
            runtime: None,
        })
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let resp = http()
        .delete(control_url(&server, &instance_by_id(peer.instance_id())))
        .header(
            kvbm_hub::protocol::MUTATION_CREDENTIAL_HEADER,
            registration.mutation_credential.unwrap().to_header_value(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 204);
}

#[tokio::test]
async fn one_registration_cannot_unregister_another_owner() {
    let server = start_server().await;
    let attacker = make_peer();
    let victim = make_peer();
    let mut credentials = Vec::new();
    for peer in [&attacker, &victim] {
        let response: RegisterResponse = http()
            .post(control_url(&server, paths::INSTANCES))
            .json(&RegisterRequest {
                peer_info: peer.clone(),
                features: Vec::new(),
                runtime: None,
            })
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        credentials.push(response.mutation_credential.unwrap());
    }

    let spoof = http()
        .delete(control_url(&server, &instance_by_id(victim.instance_id())))
        .header(
            kvbm_hub::protocol::MUTATION_CREDENTIAL_HEADER,
            credentials[0].to_header_value(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(spoof.status(), 401);
    assert!(server.state().registry().contains(victim.instance_id()));
}

#[tokio::test]
async fn unregister_not_found() {
    let server = start_server().await;
    let resp = http()
        .delete(control_url(&server, &instance_by_id(InstanceId::new_v4())))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: ErrorBody = resp.json().await.unwrap();
    assert_eq!(body.code, ErrorCode::NotFound);
}

#[tokio::test]
async fn heartbeat_with_current_credential_is_acknowledged() {
    let server = start_server().await;
    let peer = make_peer();
    let registration = register_plain_peer(&server, &peer, None).await;
    let credential = registration.mutation_credential.unwrap();
    let resp = http()
        .post(control_url(
            &server,
            &instance_heartbeat(peer.instance_id()),
        ))
        .header(MUTATION_CREDENTIAL_HEADER, credential.to_header_value())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: HeartbeatResponse = resp.json().await.unwrap();
    assert!(body.acknowledged);
}

#[tokio::test]
async fn heartbeat_without_credential_is_unauthorized() {
    let server = start_server().await;
    let peer = make_peer();
    register_plain_peer(&server, &peer, None).await;

    assert_eq!(
        heartbeat_status(&server, peer.instance_id(), None).await,
        reqwest::StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn stale_and_cross_owner_credentials_cannot_heartbeat() {
    let server = start_server().await;
    let victim = make_peer();
    let attacker = make_peer();
    let first = register_plain_peer(&server, &victim, None)
        .await
        .mutation_credential
        .unwrap();
    let current = register_plain_peer(&server, &victim, Some(&first))
        .await
        .mutation_credential
        .unwrap();
    let attacker = register_plain_peer(&server, &attacker, None)
        .await
        .mutation_credential
        .unwrap();

    for credential in [&first, &attacker] {
        assert_eq!(
            heartbeat_status(&server, victim.instance_id(), Some(credential)).await,
            reqwest::StatusCode::UNAUTHORIZED
        );
    }
    assert_eq!(
        heartbeat_status(&server, victim.instance_id(), Some(&current)).await,
        reqwest::StatusCode::OK
    );
}

#[tokio::test]
async fn unauthorized_heartbeats_do_not_postpone_expiry() {
    let registry = Arc::new(
        InMemoryRegistry::builder()
            .ttl(Duration::from_secs(1))
            .prune_interval(Duration::from_secs(3600))
            .build(),
    );
    let server = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .registry(Arc::clone(&registry) as Arc<dyn PeerRegistry>)
        .serve()
        .await
        .expect("start expiring test server");
    let victim = make_peer();
    let attacker = make_peer();
    let stale = register_plain_peer(&server, &victim, None)
        .await
        .mutation_credential
        .unwrap();
    let _current = register_plain_peer(&server, &victim, Some(&stale))
        .await
        .mutation_credential
        .unwrap();
    let attacker = register_plain_peer(&server, &attacker, None)
        .await
        .mutation_credential
        .unwrap();

    tokio::time::sleep(Duration::from_millis(250)).await;
    for credential in [None, Some(&stale), Some(&attacker)] {
        assert_eq!(
            heartbeat_status(&server, victim.instance_id(), credential).await,
            reqwest::StatusCode::UNAUTHORIZED
        );
    }
    tokio::time::sleep(Duration::from_millis(850)).await;
    registry.prune_stale();

    assert!(
        !server.state().registry().contains(victim.instance_id()),
        "rejected heartbeats must not refresh the victim's registry lease"
    );
}

#[tokio::test]
async fn heartbeat_unregistered_instance() {
    let server = start_server().await;
    let resp = http()
        .post(control_url(
            &server,
            &instance_heartbeat(InstanceId::new_v4()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn get_peer_by_instance_found() {
    let server = start_server().await;
    let peer = make_peer();
    http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
            runtime: None,
        })
        .send()
        .await
        .unwrap();
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: PeerLookupResponse = resp.json().await.unwrap();
    assert_eq!(body.peer_info.instance_id(), peer.instance_id());
}

#[tokio::test]
async fn get_peer_by_instance_not_found() {
    let server = start_server().await;
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(InstanceId::new_v4()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: ErrorBody = resp.json().await.unwrap();
    assert_eq!(body.code, ErrorCode::NotFound);
}

#[tokio::test]
async fn get_peer_by_worker_found() {
    let server = start_server().await;
    let peer = make_peer();
    http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
            runtime: None,
        })
        .send()
        .await
        .unwrap();
    let resp = http()
        .get(discovery_url(&server, &peers_by_worker(peer.worker_id())))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: PeerLookupResponse = resp.json().await.unwrap();
    assert_eq!(body.peer_info.instance_id(), peer.instance_id());
}

#[tokio::test]
async fn get_peer_by_worker_not_found() {
    let server = start_server().await;
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_worker(InstanceId::new_v4().worker_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: ErrorBody = resp.json().await.unwrap();
    assert_eq!(body.code, ErrorCode::NotFound);
}

#[tokio::test]
async fn control_port_mirrors_discovery_endpoints() {
    let server = start_server().await;
    let peer = make_peer();
    http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
            runtime: None,
        })
        .send()
        .await
        .unwrap();
    let resp = http()
        .get(control_url(&server, &peers_by_instance(peer.instance_id())))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn peers_snapshot_tracks_registrations() {
    let server = start_server().await;
    for _ in 0..3 {
        http()
            .post(control_url(&server, paths::INSTANCES))
            .json(&RegisterRequest {
                peer_info: make_peer(),
                features: Vec::new(),
                runtime: None,
            })
            .send()
            .await
            .unwrap();
    }
    assert_eq!(server.state().peers().len(), 3);
}

#[tokio::test]
async fn peers_snapshot_tracks_unregistrations() {
    let server = start_server().await;
    let a = make_peer();
    let b = make_peer();
    let mut credentials = Vec::new();
    for peer in [&a, &b] {
        let response: RegisterResponse = http()
            .post(control_url(&server, paths::INSTANCES))
            .json(&RegisterRequest {
                peer_info: peer.clone(),
                features: Vec::new(),
                runtime: None,
            })
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        credentials.push(response.mutation_credential.unwrap());
    }
    http()
        .delete(control_url(&server, &instance_by_id(a.instance_id())))
        .header(
            kvbm_hub::protocol::MUTATION_CREDENTIAL_HEADER,
            credentials[0].to_header_value(),
        )
        .send()
        .await
        .unwrap();
    let peers = server.state().peers();
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].instance_id(), b.instance_id());
}

// ---- HubClient tests --------------------------------------------------------

#[tokio::test]
async fn client_builder_requires_host() {
    assert!(HubClientBuilder::new().build().is_err());
}

#[tokio::test]
async fn client_starts_unregistered() {
    let server = start_server().await;
    let client = build_client(&server);
    assert!(!client.is_registered());
}

#[tokio::test]
async fn client_register_sets_is_registered() {
    let server = start_server().await;
    let client = build_client(&server);
    client.register_instance(make_peer()).await.unwrap();
    assert!(client.is_registered());
}

#[tokio::test]
async fn client_register_twice_errors() {
    let server = start_server().await;
    let client = build_client(&server);
    client.register_instance(make_peer()).await.unwrap();
    assert!(client.register_instance(make_peer()).await.is_err());
}

#[tokio::test]
async fn client_heartbeat_while_registered() {
    let server = start_server().await;
    let client = build_client(&server);
    client.register_instance(make_peer()).await.unwrap();
    client.send_heartbeat().await.unwrap();
}

#[tokio::test]
async fn client_heartbeat_before_register_errors() {
    let server = start_server().await;
    let client = build_client(&server);
    assert!(client.send_heartbeat().await.is_err());
}

#[tokio::test]
async fn client_discover_by_instance_id() {
    let server = start_server().await;
    let client = build_client(&server);
    let peer = make_peer();
    client.register_instance(peer.clone()).await.unwrap();
    let found = client
        .discover_by_instance_id(peer.instance_id())
        .await
        .unwrap();
    assert_eq!(found.instance_id(), peer.instance_id());
}

#[tokio::test]
async fn client_discover_by_worker_id() {
    let server = start_server().await;
    let client = build_client(&server);
    let peer = make_peer();
    client.register_instance(peer.clone()).await.unwrap();
    let found = client
        .discover_by_worker_id(peer.worker_id())
        .await
        .unwrap();
    assert_eq!(found.instance_id(), peer.instance_id());
}

#[tokio::test]
async fn client_unregister_removes_from_server() {
    let server = start_server().await;
    let client = build_client(&server);
    let peer = make_peer();
    client.register_instance(peer.clone()).await.unwrap();
    client.unregister().await.unwrap();
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn client_unregister_noop_when_not_registered() {
    let server = start_server().await;
    let client = build_client(&server);
    client.unregister().await.unwrap();
}

#[tokio::test]
async fn client_discover_after_unregister_errors() {
    let server = start_server().await;
    let client = build_client(&server);
    let peer = make_peer();
    client.register_instance(peer.clone()).await.unwrap();
    client.unregister().await.unwrap();
    assert!(
        client
            .discover_by_instance_id(peer.instance_id())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn registration_guard_drop_fires_delete() {
    let server = start_server().await;
    let peer = make_peer();
    {
        let client = build_client(&server);
        client.register_instance(peer.clone()).await.unwrap();
        // Arc<HubClient> drops here → HubRegistrationGuard::drop spawns DELETE
    }
    // Allow the background DELETE task to complete
    tokio::time::sleep(Duration::from_millis(100)).await;
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn registered_client_visible_in_list_then_removed_on_drop() {
    let server = start_server().await;
    let http = http();

    // List is initially empty
    let resp: ListInstancesResponse = http
        .get(control_url(&server, paths::INSTANCES))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(resp.instances.is_empty());

    // Register a client
    let peer = make_peer();
    {
        let client = build_client(&server);
        client.register_instance(peer.clone()).await.unwrap();

        // Instance is now visible via the view API
        let resp: ListInstancesResponse = http
            .get(control_url(&server, paths::INSTANCES))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(resp.instances.len(), 1);
        assert_eq!(resp.instances[0].instance_id(), peer.instance_id());
        // Arc<HubClient> drops here → HubRegistrationGuard::drop spawns DELETE
    }

    // Allow the background DELETE task to complete
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Instance is gone
    let resp: ListInstancesResponse = http
        .get(control_url(&server, paths::INSTANCES))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(resp.instances.is_empty());
}

// ---- Velo probe tests -------------------------------------------------------

fn new_velo_transport() -> Arc<velo::transports::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

async fn new_velo() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_velo_transport())
        .build()
        .await
        .unwrap()
}

async fn start_server_with_transport() -> (HubServer, Arc<velo::transports::tcp::TcpTransport>) {
    let transport = new_velo_transport();
    let server = kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(Arc::clone(&transport) as Arc<dyn Transport>)
        .serve()
        .await
        .expect("start test server");
    (server, transport)
}

async fn wire_mutual_velo(
    server: &HubServer,
    client_velo: &Arc<velo::Velo>,
) -> Arc<kvbm_hub::HubClient> {
    let hub_client = build_client(server);
    hub_client.register_handlers(client_velo).unwrap();
    let hub_id = hub_client
        .register_instance(client_velo.peer_info())
        .await
        .unwrap()
        .expect("hub should return its own instance id when running with a transport");
    let hub_peer = hub_client.discover_by_instance_id(hub_id).await.unwrap();
    client_velo.register_peer(hub_peer).unwrap();
    hub_client
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_probe_happy_path() {
    let (server, _hub_transport) = start_server_with_transport().await;
    let client_velo = new_velo().await;
    let _hub_client = wire_mutual_velo(&server, &client_velo).await;

    tokio::time::sleep(Duration::from_millis(200)).await;

    let instance_id = client_velo.instance_id();
    let resp = http()
        .post(control_url(&server, &instance_probe(instance_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: ProbeResponse = resp.json().await.unwrap();
    assert!(body.ok);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_probe_after_client_velo_shutdown_returns_bad_gateway() {
    let (server, _hub_transport) = start_server_with_transport().await;

    // Build client velo with an explicit transport handle so we can shut it down.
    let client_transport = new_velo_transport();
    let client_velo = velo::Velo::builder()
        .add_transport(Arc::clone(&client_transport) as Arc<dyn Transport>)
        .build()
        .await
        .unwrap();

    let hub_client = wire_mutual_velo(&server, &client_velo).await;
    let instance_id = client_velo.instance_id();

    // Keep hub_client alive so the HTTP registration guard never fires and the
    // instance stays in the registry.
    let _keep_hub_client = Arc::clone(&hub_client);

    // Explicitly shut down the transport — cancels the TCP listener and all
    // connection tasks, making the client unreachable.
    client_transport.shutdown();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let resp = http()
        .post(control_url(&server, &instance_probe(instance_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);
}

// ---- Hub self-registration + hub_instance_id tests --------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn register_response_includes_hub_instance_id_when_velo_configured() {
    let (server, _hub_transport) = start_server_with_transport().await;
    let client_velo = new_velo().await;
    let client = build_client(&server);
    let hub_id = client
        .register_instance(client_velo.peer_info())
        .await
        .unwrap();
    assert!(hub_id.is_some(), "hub_instance_id should be Some");
    // Hub is discoverable immediately.
    let _hub_peer = client
        .discover_by_instance_id(hub_id.unwrap())
        .await
        .unwrap();
}

#[tokio::test]
async fn register_response_hub_instance_id_none_without_transport() {
    let server = start_server().await;
    let client = build_client(&server);
    let hub_id = client.register_instance(make_peer()).await.unwrap();
    assert!(hub_id.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hub_self_registered_in_registry() {
    let (server, hub_transport) = start_server_with_transport().await;
    let hub_velo_id = server.state().velo().expect("hub velo").instance_id();

    // Direct HTTP lookup should resolve the hub's PeerInfo.
    let resp = http()
        .get(discovery_url(&server, &peers_by_instance(hub_velo_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = hub_transport; // keep alive
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hub_self_entry_survives_reaper() {
    // Short TTL; protect() must keep the hub's entry alive.
    let transport = new_velo_transport();
    let server = kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .registration_ttl(Duration::from_millis(50))
        .prune_interval(Duration::from_millis(20))
        .add_transport(Arc::clone(&transport) as Arc<dyn Transport>)
        .serve()
        .await
        .expect("start test server");

    let hub_velo_id = server.state().velo().expect("hub velo").instance_id();

    // Wait well past TTL + multiple prune cycles.
    tokio::time::sleep(Duration::from_millis(300)).await;

    let resp = http()
        .get(discovery_url(&server, &peers_by_instance(hub_velo_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "hub's self-entry should be protected from reaper"
    );
}

// ---- ConditionalDisagg feature tests ---------------------------------------

async fn start_server_with_cd() -> (
    HubServer,
    Arc<velo::transports::tcp::TcpTransport>,
    Arc<ConditionalDisaggManager>,
) {
    let transport = new_velo_transport();
    let cd_manager: Arc<ConditionalDisaggManager> = Arc::new(ConditionalDisaggManager::new());
    let p2p: Arc<kvbm_hub::P2pManager> = Arc::new(kvbm_hub::P2pManager::new());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(Arc::clone(&transport) as Arc<dyn Transport>)
        .add_feature_manager(p2p as Arc<dyn kvbm_hub::FeatureManager>)
        .add_feature_manager(Arc::clone(&cd_manager) as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start test server with CD");
    (server, transport, cd_manager)
}

/// CD-enabled hub without any velo transport. Use this when a test only
/// exercises HTTP registration dispatch and does not need velo peer
/// registration on the hub side — the hub's `velo.register_peer` call
/// would otherwise reject the opaque addresses produced by `make_peer()`.
async fn start_server_with_cd_no_velo() -> (HubServer, Arc<ConditionalDisaggManager>) {
    let cd_manager: Arc<ConditionalDisaggManager> = Arc::new(ConditionalDisaggManager::new());
    let p2p: Arc<kvbm_hub::P2pManager> = Arc::new(kvbm_hub::P2pManager::new());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_feature_manager(p2p as Arc<dyn kvbm_hub::FeatureManager>)
        .add_feature_manager(Arc::clone(&cd_manager) as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start test server with CD");
    (server, cd_manager)
}

#[tokio::test]
async fn feature_register_without_manager_rejects() {
    // Bare server with no managers + P2P+CD features → pre-dispatch
    // accepts (both features present), per-feature dispatch then rejects
    // with "no manager registered for P2P".
    let server = start_server().await;
    let peer = make_peer();
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: p2p_cd_features(ConditionalDisaggRole::Prefill),
        runtime: None,
    };
    let resp = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: ErrorBody = resp.json().await.unwrap();
    assert_eq!(body.code, ErrorCode::BadRequest);

    // Base entry should have been rolled back.
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn failed_authorized_reregister_restores_prior_registration() {
    let (server, cd) = start_server_with_cd_no_velo().await;
    let peer = make_peer();
    let initial: RegisterResponse = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: p2p_cd_features(ConditionalDisaggRole::Prefill),
            runtime: None,
        })
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let credential = initial.mutation_credential.unwrap();

    let rejected = http()
        .post(control_url(&server, paths::INSTANCES))
        .header(
            kvbm_hub::protocol::MUTATION_CREDENTIAL_HEADER,
            credential.to_header_value(),
        )
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: p2p_cd_features(ConditionalDisaggRole::Decode),
            runtime: None,
        })
        .send()
        .await
        .unwrap();
    assert_eq!(rejected.status(), 400);
    assert!(server.state().registry().contains(peer.instance_id()));
    assert_eq!(cd.snapshot().prefill, vec![peer.instance_id()]);
    assert!(cd.snapshot().decode.is_empty());

    let removed = http()
        .delete(control_url(&server, &instance_by_id(peer.instance_id())))
        .header(
            kvbm_hub::protocol::MUTATION_CREDENTIAL_HEADER,
            credential.to_header_value(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(removed.status(), 204);
}

// c2: `feature_cd_register_missing_config_rejects` removed — the type
// system now makes `Feature::ConditionalDisagg(...)` require a config,
// so the missing-config path is unreachable. The cross-feature
// "CD without P2P is rejected" invariant is exercised by
// `cd_layout_compat::cd_register_without_p2p_feature_is_rejected`.

#[tokio::test]
async fn feature_cd_list_empty_on_both_ports() {
    let (server, _cd) = start_server_with_cd_no_velo().await;
    for url in [
        discovery_url(&server, "/v1/features/disagg/instances"),
        control_url(&server, "/v1/features/disagg/instances"),
    ] {
        let body: ConditionalDisaggInstancesResponse =
            http().get(url).send().await.unwrap().json().await.unwrap();
        assert!(body.prefill.is_empty());
        assert!(body.decode.is_empty());
    }
}

#[tokio::test]
async fn register_without_features_field_still_works() {
    // Proves `#[serde(default)]` on RegisterRequest.features is honored for
    // older clients that omit the field entirely.
    let server = start_server().await;
    let peer = make_peer();
    let legacy_json = format!(
        r#"{{"peer_info":{}}}"#,
        serde_json::to_string(&peer).unwrap()
    );
    let resp = http()
        .post(control_url(&server, paths::INSTANCES))
        .header("content-type", "application/json")
        .body(legacy_json)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn feature_cd_role_conflict_on_reregister() {
    let (server, _cd) = start_server_with_cd_no_velo().await;
    let peer = make_peer();
    let first: RegisterResponse = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: p2p_cd_features(ConditionalDisaggRole::Prefill),
            runtime: None,
        })
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let resp = http()
        .post(control_url(&server, paths::INSTANCES))
        .header(
            kvbm_hub::protocol::MUTATION_CREDENTIAL_HEADER,
            first.mutation_credential.unwrap().to_header_value(),
        )
        .json(&RegisterRequest {
            peer_info: peer,
            features: p2p_cd_features(ConditionalDisaggRole::Decode),
            runtime: None,
        })
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn feature_cd_unregister_removes_from_lists() {
    let (server, cd) = start_server_with_cd_no_velo().await;
    let peer = make_peer();
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: p2p_cd_features(ConditionalDisaggRole::Prefill),
        runtime: None,
    };
    let registration: RegisterResponse = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(cd.snapshot().prefill.len(), 1);

    http()
        .delete(control_url(&server, &instance_by_id(peer.instance_id())))
        .header(
            kvbm_hub::protocol::MUTATION_CREDENTIAL_HEADER,
            registration.mutation_credential.unwrap().to_header_value(),
        )
        .send()
        .await
        .unwrap();

    assert!(cd.snapshot().prefill.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn feature_cd_reaper_evicts_from_lists() {
    // No transport — the reaper runs off the in-memory registry ticker and
    // the eviction callback fires into the CD manager regardless of velo.
    let cd_manager: Arc<ConditionalDisaggManager> = Arc::new(ConditionalDisaggManager::new());
    let p2p: Arc<kvbm_hub::P2pManager> = Arc::new(kvbm_hub::P2pManager::new());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .registration_ttl(Duration::from_millis(60))
        .prune_interval(Duration::from_millis(20))
        .add_feature_manager(p2p as Arc<dyn kvbm_hub::FeatureManager>)
        .add_feature_manager(Arc::clone(&cd_manager) as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start test server");

    let peer = make_peer();
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: p2p_cd_features(ConditionalDisaggRole::Prefill),
        runtime: None,
    };
    http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(cd_manager.snapshot().prefill.len(), 1);

    // Wait well past TTL + multiple prune cycles so the reaper evicts the
    // base entry and the eviction callback fans out to the CD manager.
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert!(
        cd_manager.snapshot().prefill.is_empty(),
        "reaper eviction should fan out to the feature manager"
    );
    let fresh = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(
        fresh.status(),
        200,
        "TTL eviction must also remove the stale registration credential"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn feature_cd_prefill_and_decode_register_and_list() {
    let (server, _hub_transport, _cd) = start_server_with_cd().await;

    // Build two velo participants with their own transports + hub clients.
    let p_velo = new_velo().await;
    let d_velo = new_velo().await;

    let p_hub = build_client(&server);
    let d_hub = build_client(&server);
    p_hub.register_handlers(&p_velo).unwrap();
    d_hub.register_handlers(&d_velo).unwrap();

    let p_cd = ConditionalDisaggClient::new(
        Arc::clone(&p_hub),
        Arc::clone(&p_velo),
        ConditionalDisaggRole::Prefill,
    );
    let d_cd = ConditionalDisaggClient::new(
        Arc::clone(&d_hub),
        Arc::clone(&d_velo),
        ConditionalDisaggRole::Decode,
    );

    let p_hub_id = p_cd
        .register(p_velo.peer_info(), test_layout_compat_payload())
        .await
        .unwrap()
        .expect("hub velo id");
    let d_hub_id = d_cd
        .register(d_velo.peer_info(), test_layout_compat_payload())
        .await
        .unwrap()
        .expect("hub velo id");
    assert_eq!(p_hub_id, d_hub_id, "both clients should see the same hub");

    // Give the server a moment to fully settle (listeners + velo peer table).
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Both endpoints must report the same role split, with the correct ids.
    let p_id = p_velo.instance_id();
    let d_id = d_velo.instance_id();
    for url in [
        discovery_url(&server, "/v1/features/disagg/instances"),
        control_url(&server, "/v1/features/disagg/instances"),
    ] {
        let body: ConditionalDisaggInstancesResponse =
            http().get(&url).send().await.unwrap().json().await.unwrap();
        assert_eq!(body.prefill, vec![p_id], "from {url}");
        assert_eq!(body.decode, vec![d_id], "from {url}");
    }

    // Wire the hub's PeerInfo into each participant's velo (needed so velo
    // probes initiated from the hub can route back through the TCP transport).
    let hub_peer = p_hub.discover_by_instance_id(p_hub_id).await.unwrap();
    p_velo.register_peer(hub_peer.clone()).unwrap();
    d_velo.register_peer(hub_peer).unwrap();

    // Prefill side: await the decode peer and register it into prefill's velo.
    let d_peer = p_cd
        .await_peer_of_role(
            ConditionalDisaggRole::Decode,
            Duration::from_millis(50),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
    assert_eq!(d_peer.instance_id(), d_id);
    p_velo.register_peer(d_peer).unwrap();

    // Decode side: symmetric handshake.
    let p_peer = d_cd
        .await_peer_of_role(
            ConditionalDisaggRole::Prefill,
            Duration::from_millis(50),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
    assert_eq!(p_peer.instance_id(), p_id);
    d_velo.register_peer(p_peer).unwrap();

    // Hub can now probe both instances — proves velo handshakes succeeded.
    tokio::time::sleep(Duration::from_millis(200)).await;
    for id in [p_id, d_id] {
        let resp = http()
            .post(control_url(&server, &instance_probe(id)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "probe failed for {id}");
        let body: ProbeResponse = resp.json().await.unwrap();
        assert!(body.ok, "probe returned not-ok for {id}");
    }

    // ---- Prefill queue handshake -------------------------------------------
    //
    // Decode pushes a PrefillRequest to the hub's CD queue; Prefill pulls
    // it back via a timed dequeue. Empty-queue pulls return None.

    // First: Prefill waits on an empty queue and observes the timeout path.
    let empty = p_cd
        .pull_prefill_request(Duration::from_millis(150))
        .await
        .unwrap();
    assert!(
        empty.is_none(),
        "pull on empty queue should return None, got {empty:?}"
    );

    // Decode pushes a request.
    let req = PrefillRequest {
        protocol_version: DISAGG_PROTOCOL_VERSION,
        request_id: "req-golden-1".to_string(),
        session_id: uuid::Uuid::new_v4(),
        initiator_instance_id: d_id,
        decode_endpoint: None,
        token_ids: vec![1, 2, 3],
        num_provided_tokens: 48,
        request: kvbm_protocols::disagg::KvHashingRequestEnvelope::default(),
        expected_hash_digest: None,
        bundle: None,
    };
    d_cd.push_prefill_request(&req).await.unwrap();

    // Prefill pulls and receives it.
    let pulled = p_cd
        .pull_prefill_request(Duration::from_secs(2))
        .await
        .unwrap()
        .expect("prefill should dequeue the request Decode just pushed");
    assert_eq!(pulled, req);

    // Queue is drained — another pull returns None within the timeout window.
    let empty_again = p_cd
        .pull_prefill_request(Duration::from_millis(150))
        .await
        .unwrap();
    assert!(empty_again.is_none());

    // Role guard: Prefill cannot push, Decode cannot pull.
    assert!(
        p_cd.push_prefill_request(&req).await.is_err(),
        "prefill must not be allowed to push"
    );
    assert!(
        d_cd.pull_prefill_request(Duration::from_millis(10))
            .await
            .is_err(),
        "decode must not be allowed to pull"
    );
}

// ---- ConditionalDisagg dispatcher integration -------------------------------

fn init_tracing() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,kvbm_hub=debug")),
            )
            .with_test_writer()
            .try_init();
    });
}

/// Variant of `start_server_with_cd` that installs a `RecordingDispatcher`.
/// Returns the server, transport, manager, and the dispatcher handle so
/// the test can assert which requests the worker handed off.
async fn start_server_with_cd_dispatcher() -> (
    HubServer,
    Arc<velo::transports::tcp::TcpTransport>,
    Arc<ConditionalDisaggManager>,
    Arc<kvbm_hub::RecordingDispatcher>,
) {
    let transport = new_velo_transport();
    let dispatcher = kvbm_hub::RecordingDispatcher::new();
    let cd_manager: Arc<ConditionalDisaggManager> =
        Arc::new(ConditionalDisaggManager::new().with_dispatcher(
            Arc::clone(&dispatcher) as Arc<dyn kvbm_hub::PrefillRequestDispatcher>
        ));
    let p2p: Arc<kvbm_hub::P2pManager> = Arc::new(kvbm_hub::P2pManager::new());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(Arc::clone(&transport) as Arc<dyn Transport>)
        .add_feature_manager(p2p as Arc<dyn kvbm_hub::FeatureManager>)
        .add_feature_manager(Arc::clone(&cd_manager) as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start test server with CD dispatcher");
    (server, transport, cd_manager, dispatcher)
}

/// Hub-side dispatcher worker drains the prefill queue and hands each
/// request to the configured dispatcher implementation. Decode-side
/// `push_prefill_request` should arrive at the recorder within seconds.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dispatcher_worker_drains_queue_and_invokes_dispatcher() {
    init_tracing();
    let (server, _hub_transport, _cd, dispatcher) = start_server_with_cd_dispatcher().await;

    // Decode client + velo so the push has the messenger backing it.
    let d_velo = new_velo().await;
    let d_hub = build_client(&server);
    d_hub.register_handlers(&d_velo).unwrap();
    let d_cd = ConditionalDisaggClient::new(
        Arc::clone(&d_hub),
        Arc::clone(&d_velo),
        ConditionalDisaggRole::Decode,
    );
    let d_hub_id = d_cd
        .register(d_velo.peer_info(), test_layout_compat_payload())
        .await
        .unwrap()
        .expect("hub velo id");
    let hub_peer = d_hub.discover_by_instance_id(d_hub_id).await.unwrap();
    d_velo.register_peer(hub_peer).unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Push two requests; the worker should drain both.
    let req_one = PrefillRequest {
        protocol_version: DISAGG_PROTOCOL_VERSION,
        request_id: "dispatch-test-1".to_string(),
        session_id: uuid::Uuid::new_v4(),
        initiator_instance_id: d_velo.instance_id(),
        decode_endpoint: None,
        token_ids: vec![10, 20, 30],
        num_provided_tokens: 0,
        request: kvbm_protocols::disagg::KvHashingRequestEnvelope::default(),
        expected_hash_digest: None,
        bundle: None,
    };
    let req_two = PrefillRequest {
        request_id: "dispatch-test-2".to_string(),
        ..req_one.clone()
    };
    d_cd.push_prefill_request(&req_one).await.unwrap();
    d_cd.push_prefill_request(&req_two).await.unwrap();

    // Wait for the worker to drain. The pump is a spawned task on the
    // hub's runtime; poll the recorder rather than sleeping a fixed
    // window so the test is fast on a hot loop and still robust under
    // load.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while dispatcher.len() < 2 && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let recorded = dispatcher.recorded();
    assert_eq!(
        recorded.len(),
        2,
        "dispatcher should have received both pushed requests, got {}",
        recorded.len()
    );
    assert_eq!(recorded[0].request_id, "dispatch-test-1");
    assert_eq!(recorded[1].request_id, "dispatch-test-2");
}

/// Hub configured WITHOUT a dispatcher — push must still succeed (the
/// queue handler is still installed) but nothing drains; this is the
/// pre-dispatcher behavior, preserved for backward compat.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_dispatcher_does_not_spawn_worker() {
    let (server, _hub_transport, _cd) = start_server_with_cd().await;

    let d_velo = new_velo().await;
    let d_hub = build_client(&server);
    d_hub.register_handlers(&d_velo).unwrap();
    let d_cd = ConditionalDisaggClient::new(
        Arc::clone(&d_hub),
        Arc::clone(&d_velo),
        ConditionalDisaggRole::Decode,
    );
    let d_hub_id = d_cd
        .register(d_velo.peer_info(), test_layout_compat_payload())
        .await
        .unwrap()
        .expect("hub velo id");
    let hub_peer = d_hub.discover_by_instance_id(d_hub_id).await.unwrap();
    d_velo.register_peer(hub_peer).unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    let req = PrefillRequest {
        protocol_version: DISAGG_PROTOCOL_VERSION,
        request_id: "no-dispatcher".to_string(),
        session_id: uuid::Uuid::new_v4(),
        initiator_instance_id: d_velo.instance_id(),
        decode_endpoint: None,
        token_ids: vec![1],
        num_provided_tokens: 0,
        request: kvbm_protocols::disagg::KvHashingRequestEnvelope::default(),
        expected_hash_digest: None,
        bundle: None,
    };
    // Push succeeds — queue is installed by `attach`, dispatcher absence
    // doesn't change that.
    d_cd.push_prefill_request(&req).await.unwrap();
    // Without a dispatcher there's nothing to drain — a passive consumer
    // (prefill client) can still pull as before.
    let p_velo = new_velo().await;
    let p_hub = build_client(&server);
    p_hub.register_handlers(&p_velo).unwrap();
    let p_cd = ConditionalDisaggClient::new(
        Arc::clone(&p_hub),
        Arc::clone(&p_velo),
        ConditionalDisaggRole::Prefill,
    );
    let p_hub_id = p_cd
        .register(p_velo.peer_info(), test_layout_compat_payload())
        .await
        .unwrap()
        .expect("hub velo id");
    // Wire the hub's PeerInfo into prefill's velo so the queue RPC can
    // round-trip back through the TCP transport.
    let p_hub_peer = p_hub.discover_by_instance_id(p_hub_id).await.unwrap();
    p_velo.register_peer(p_hub_peer).unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let pulled = p_cd
        .pull_prefill_request(Duration::from_secs(2))
        .await
        .unwrap();
    assert!(
        pulled.is_some(),
        "passive pull path should still receive the queued request"
    );
}

// ---- KV indexer velo find_blocks lookup -------------------------------------

/// Block size the indexer test hub + its PLHs agree on.
const IDX_BLOCK_SIZE: usize = 4;

/// Build `n` PLHs at positions `0..n` for a given salt — same recipe the
/// in-crate index unit tests use.
fn idx_plhs(n: usize, salt: u64) -> Vec<SequenceHash> {
    let tokens: Vec<u32> = (0..(IDX_BLOCK_SIZE * n) as u32).collect();
    let seq = TokenBlockSequence::from_slice(&tokens, IDX_BLOCK_SIZE as u32, Some(salt));
    seq.blocks()
        .iter()
        .map(|b| b.kvbm_sequence_hash())
        .collect()
}

/// Hub with a TCP transport **and** an attached `IndexerManager`. Returns the
/// manager handle so the test can seed the index directly.
async fn start_server_with_indexer() -> (
    HubServer,
    Arc<IndexerManager>,
    Arc<velo::transports::tcp::TcpTransport>,
) {
    let transport = new_velo_transport();
    let mgr = Arc::new(IndexerManager::new(64, IDX_BLOCK_SIZE, None, None).unwrap());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(Arc::clone(&transport) as Arc<dyn Transport>)
        .add_feature_manager(Arc::clone(&mgr) as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start test server with indexer");
    (server, mgr, transport)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn indexer_find_blocks_velo_lookup() {
    let (server, mgr, _transport) = start_server_with_indexer().await;
    let client_velo = new_velo().await;
    let hub_client = wire_mutual_velo(&server, &client_velo).await;

    // Seed: this client's instance holds a 3-deep sequence. The index stores
    // holder ids as the velo id's u128 (publishers stamp `velo_id.as_u128()`).
    let holder = client_velo.instance_id();
    let hashes = idx_plhs(3, 4242);
    mgr.index().apply(KvbmCacheEvents {
        events: KvCacheEvents::Create(hashes.clone()),
        instance_id: holder.as_u128(),
    });

    // The gated constructor returns a client because the hub has the indexer.
    let lookup = hub_client
        .indexer_lookup_client(client_velo.messenger().clone())
        .await
        .expect("indexer probe should succeed")
        .expect("indexer is enabled on this hub");

    // Full sequence → deepest hit (position 2), seeded holder present.
    let hit = lookup
        .find_blocks(hashes.clone())
        .await
        .unwrap()
        .expect("deepest hit");
    assert_eq!(hit.matched, hashes[2], "deepest candidate");
    assert!(
        hit.candidates.contains(&holder),
        "candidates should reconstruct the seeded InstanceId"
    );

    // Deepest blocks unknown → falls back to the shallow hit (position 1).
    let unknown = idx_plhs(5, 9999);
    let mut mixed = vec![hashes[0], hashes[1]];
    mixed.extend_from_slice(&unknown[2..]); // positions 2..4 unknown
    let hit = lookup
        .find_blocks(mixed)
        .await
        .unwrap()
        .expect("shallow hit");
    assert_eq!(hit.matched, hashes[1], "shallow fallback");

    // Full miss → None.
    let miss = lookup.find_blocks(idx_plhs(2, 123_456)).await.unwrap();
    assert!(miss.is_none(), "no candidate indexed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn indexer_lookup_client_none_when_indexer_disabled() {
    // A hub with a transport but no IndexerManager.
    let (server, _hub_transport) = start_server_with_transport().await;
    let client_velo = new_velo().await;
    let hub_client = wire_mutual_velo(&server, &client_velo).await;

    let res = hub_client
        .indexer_lookup_client(client_velo.messenger().clone())
        .await
        .expect("probe should not error");
    assert!(res.is_none(), "indexer not enabled ⇒ None");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn indexer_lookup_client_errs_when_hub_has_no_velo() {
    // Indexer mounted, but the hub runs without a transport — so registration
    // returns no hub velo InstanceId and the velo lookup can't be addressed.
    // The probe succeeds (indexer present), so this is the `Err` branch.
    let mgr = Arc::new(IndexerManager::new(64, IDX_BLOCK_SIZE, None, None).unwrap());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_feature_manager(Arc::clone(&mgr) as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start indexer hub without transport");

    let client_velo = new_velo().await;
    let hub_client = build_client(&server);
    let hub_id = hub_client
        .register_instance(client_velo.peer_info())
        .await
        .unwrap();
    assert!(hub_id.is_none(), "discovery-only hub returns no velo id");

    let err = hub_client
        .indexer_lookup_client(client_velo.messenger().clone())
        .await
        .expect_err("indexer enabled but no hub velo ⇒ Err");
    assert!(
        err.to_string().contains("hub velo InstanceId unknown"),
        "unexpected error: {err}"
    );
}

// ---- tier-placement snapshot push (CT-2b) -----------------------------------

/// Register declaring `Feature::Indexer` and wire both velo directions.
///
/// The feature declaration is load-bearing, not decoration: `stage_registration`
/// stores the owner credential only for a *participating* registrant, so a hub
/// client registered without it has no credential the snapshot install can
/// authorize against and every push answers `401 UnknownOwner`.
async fn wire_indexer_participant(
    server: &HubServer,
    client_velo: &Arc<velo::Velo>,
) -> Arc<kvbm_hub::HubClient> {
    let hub_client = build_client(server);
    hub_client.register_handlers(client_velo).unwrap();
    let hub_id = hub_client
        .register_instance_with_features_and_runtime(
            client_velo.peer_info(),
            vec![Feature::Indexer(Default::default())],
            kvbm_hub::protocol::RuntimeConfigSummary {
                block_size: Some(IDX_BLOCK_SIZE),
                block_layout: None,
            },
        )
        .await
        .unwrap()
        .expect("hub should return its own instance id when running with a transport");
    let hub_peer = hub_client.discover_by_instance_id(hub_id).await.unwrap();
    client_velo.register_peer(hub_peer).unwrap();
    hub_client
}

fn tier_snapshot(
    cache: kvbm_protocols::cache_manifest::CacheManifestId,
    instance_id: InstanceId,
    registration_epoch: kvbm_protocols::cache_manifest::RegistrationEpoch,
    generation: u64,
    hash: SequenceHash,
) -> kvbm_protocols::tier_protocol::TierPlacementSnapshotV1 {
    use kvbm_protocols::tier_protocol::{
        KeyRange, PhysicalPlacementMode, PlacementScope, TIER_MEDIUM_CAP_DIRECT_SERVABLE,
        TIER_PLACEMENT_SCHEMA_VERSION, TierDepth, TierMedium, TierPlacementEntry,
        TierPlacementSnapshotV1,
    };
    let tier = TierDepth(1);
    TierPlacementSnapshotV1 {
        v: TIER_PLACEMENT_SCHEMA_VERSION,
        cache,
        instance_id,
        registration_epoch,
        snapshot_generation: generation,
        seq_floor: 0,
        // `validate` rejects an entry whose depth the header omits, so the
        // medium travels with the entry that names its depth.
        media: vec![TierMedium {
            depth: tier,
            medium: "pinned-host".to_string(),
            capabilities: TIER_MEDIUM_CAP_DIRECT_SERVABLE,
        }],
        manifests: Vec::new(),
        entries: vec![TierPlacementEntry {
            scope: PlacementScope::unitary(kvbm_common::LogicalResourceId(1)),
            tier,
            placement: PhysicalPlacementMode::Whole,
            generation: 1,
            keys: KeyRange::Hashes(vec![hash]),
        }],
    }
}

/// The publisher's recovery half, end to end against a live hub: the client
/// pushes a snapshot with a credential it can never read, and the projection
/// goes from "no entry, answers nothing" to valid and answering.
///
/// A re-push of the same generation is asserted too, because that is the
/// steady-state case — R7b §3 pushes every 60 s whether or not anything was
/// lost — and it answers `installed: false`, which a publisher must treat as
/// success or its emission gate never releases.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn indexer_client_pushes_a_tier_placement_snapshot() {
    use kvbm_protocols::cache_manifest::CacheManifestId;
    use kvbm_protocols::tier_protocol::PlacementScope;

    let (server, mgr, _transport) = start_server_with_indexer().await;
    let client_velo = new_velo().await;
    let hub_client = wire_indexer_participant(&server, &client_velo).await;
    let lookup = hub_client
        .indexer_lookup_client(client_velo.messenger().clone())
        .await
        .expect("indexer probe should succeed")
        .expect("indexer is enabled on this hub");

    let cache = CacheManifestId::from_bytes([21; 32]);
    let scope = PlacementScope::unitary(kvbm_common::LogicalResourceId(1));
    let hash = SequenceHash::root(0xC0FF_EE00_0000_0001);
    let instance = client_velo.instance_id();
    let epoch = lookup.registration_epoch();
    assert!(
        !mgr.tier_placements().is_valid(cache, instance),
        "no projection exists before an authorized install"
    );

    let installed = lookup
        .push_tier_placement_snapshot(tier_snapshot(cache, instance, epoch, 1, hash))
        .await
        .expect("an authorized push must be accepted");
    assert!(installed.installed, "first generation installs");
    assert_eq!(installed.installed_generation, 1);
    assert_eq!(installed.seq_floor, 0);

    assert!(mgr.tier_placements().is_valid(cache, instance));
    let holders = mgr.tier_placements().holders(cache, scope, hash);
    assert_eq!(holders.len(), 1, "the pushed key must be answerable");
    assert_eq!(holders[0].instance, instance);

    // The periodic re-push the hub already holds: not an error, and its
    // generation still has to reach `note_snapshot_installed`.
    let repeat = lookup
        .push_tier_placement_snapshot(tier_snapshot(cache, instance, epoch, 1, hash))
        .await
        .expect("a duplicate generation is a success, not an error");
    assert!(!repeat.installed, "already current");
    assert_eq!(repeat.installed_generation, 1);
    assert!(mgr.tier_placements().is_valid(cache, instance));
}

/// The two rejections a publisher must be able to tell apart: its own stale
/// epoch (caught locally, no round trip) and pushing on behalf of an instance
/// whose credential the hub never minted (caught by `authorize_owner_epoch`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tier_placement_snapshot_push_rejects_wrong_epoch_and_foreign_instance() {
    use kvbm_protocols::cache_manifest::{CacheManifestId, RegistrationEpoch};

    let (server, mgr, _transport) = start_server_with_indexer().await;
    let client_velo = new_velo().await;
    let hub_client = wire_indexer_participant(&server, &client_velo).await;
    let lookup = hub_client
        .indexer_lookup_client(client_velo.messenger().clone())
        .await
        .expect("indexer probe should succeed")
        .expect("indexer is enabled on this hub");

    let cache = CacheManifestId::from_bytes([22; 32]);
    let hash = SequenceHash::root(7);
    let instance = client_velo.instance_id();

    let stale = lookup
        .push_tier_placement_snapshot(tier_snapshot(
            cache,
            instance,
            RegistrationEpoch::new(),
            1,
            hash,
        ))
        .await
        .expect_err("a snapshot from a different registration lifecycle");
    assert!(
        stale.to_string().contains("registration epoch"),
        "unexpected error: {stale}"
    );

    let foreign = InstanceId::new_v4();
    let unauthorized = lookup
        .push_tier_placement_snapshot(tier_snapshot(
            cache,
            foreign,
            lookup.registration_epoch(),
            1,
            hash,
        ))
        .await
        .expect_err("this credential does not own that instance");
    assert!(
        unauthorized.to_string().contains("401"),
        "unexpected error: {unauthorized}"
    );
    assert!(!mgr.tier_placements().is_valid(cache, foreign));
    assert!(!mgr.tier_placements().is_valid(cache, instance));
}
