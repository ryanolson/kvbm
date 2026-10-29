// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use std::sync::Arc;

use kvbm_hub::protocol::{
    ConditionalDisaggConfig, ConditionalDisaggRole, DISAGG_PROTOCOL_VERSION, ErrorBody, ErrorCode,
    Feature, LayoutCompatPayload, P2pConfig, PrefillRequest,
};
use kvbm_hub::test_hub;
use kvbm_hub::{
    ConditionalDisaggClient, ConditionalDisaggInstancesResponse, ConditionalDisaggManager,
    HubServer,
};
use velo::Transport;
use velo::discovery::PeerDiscovery;
use velo::transports::tcp::TcpTransportBuilder;
use velo_ext::{InstanceId, PeerInfo, WorkerAddress};
use velo_hub::client::ClientError;

use dynamo_tokens::TokenBlockSequence;
use kvbm_hub::IndexerManager;
use kvbm_logical::events::{KvCacheEvents, KvbmCacheEvents};
use kvbm_logical::{KvbmSequenceHashProvider, SequenceHash};
use kvbm_protocols::cache_manifest::{CacheManifest, ModelIdentity};

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
        .map(Arc::new)
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
    hub_client
        .register(client_velo, vec![], None)
        .await
        .unwrap()
        .expect("hub should return its own instance id when running with a transport");
    hub_client
}

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
        .extension(p2p)
        .extension(Arc::clone(&cd_manager))
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
        .extension(p2p)
        .extension(Arc::clone(&cd_manager))
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
    let resp = test_hub::attempt(
        &server,
        peer.clone(),
        &p2p_cd_features(ConditionalDisaggRole::Prefill),
        None,
    )
    .await;
    assert_eq!(resp.status(), 400);
    let body: ErrorBody = resp.json().await.unwrap();
    assert_eq!(body.code, ErrorCode::BadRequest);

    // Base entry should have been rolled back.
    let resp = http()
        .get(discovery_url(
            &server,
            &format!("/v1/peers/instance/{}", peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
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
        discovery_url(&server, "/v1/ext/disagg/instances"),
        control_url(&server, "/v1/ext/disagg/instances"),
    ] {
        let body: ConditionalDisaggInstancesResponse =
            http().get(url).send().await.unwrap().json().await.unwrap();
        assert!(body.prefill.is_empty());
        assert!(body.decode.is_empty());
    }
}

#[tokio::test]
async fn feature_cd_role_conflict_on_reregister() {
    let (server, _cd) = start_server_with_cd_no_velo().await;
    let peer = make_peer();
    let first = test_hub::attempt(
        &server,
        peer.clone(),
        &p2p_cd_features(ConditionalDisaggRole::Prefill),
        None,
    )
    .await;
    assert_eq!(first.status(), 200);
    let err = first
        .client
        .register_peer(
            peer,
            test_hub::extensions(&p2p_cd_features(ConditionalDisaggRole::Decode), None),
        )
        .await
        .expect_err("a role change on re-register must reject");
    assert!(
        matches!(err, ClientError::Hub { status, .. } if status.as_u16() == 409),
        "{err}"
    );
}

#[tokio::test]
async fn feature_cd_unregister_removes_from_lists() {
    let (server, cd) = start_server_with_cd_no_velo().await;
    let registration = test_hub::attempt(
        &server,
        make_peer(),
        &p2p_cd_features(ConditionalDisaggRole::Prefill),
        None,
    )
    .await;
    assert_eq!(registration.status(), 200);
    assert_eq!(cd.snapshot().prefill.len(), 1);

    registration.client.unregister().await.unwrap();

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
        .extension(p2p)
        .extension(Arc::clone(&cd_manager))
        .serve()
        .await
        .expect("start test server");

    let features = p2p_cd_features(ConditionalDisaggRole::Prefill);
    let _stale = test_hub::attempt(&server, make_peer(), &features, None).await;
    assert_eq!(cd_manager.snapshot().prefill.len(), 1);

    // Wait well past TTL + multiple prune cycles so the reaper evicts the
    // entry and its removal fans out to the CD manager.
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert!(
        cd_manager.snapshot().prefill.is_empty(),
        "reaper eviction should fan out to the feature manager"
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
        .register(&p_velo, test_layout_compat_payload())
        .await
        .unwrap()
        .expect("hub velo id");
    let d_hub_id = d_cd
        .register(&d_velo, test_layout_compat_payload())
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
        discovery_url(&server, "/v1/ext/disagg/instances"),
        control_url(&server, "/v1/ext/disagg/instances"),
    ] {
        let body: ConditionalDisaggInstancesResponse =
            http().get(&url).send().await.unwrap().json().await.unwrap();
        assert_eq!(body.prefill, vec![p_id], "from {url}");
        assert_eq!(body.decode, vec![d_id], "from {url}");
    }

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

    // Role guard: a prefill worker cannot send prefill requests.
    let req = PrefillRequest {
        protocol_version: DISAGG_PROTOCOL_VERSION,
        request_id: "req-golden-1".to_string(),
        session_id: uuid::Uuid::new_v4(),
        initiator_instance_id: d_id,
        decode_endpoint: None,
        token_ids: vec![1, 2, 3],
        num_provided_tokens: 0,
        request: kvbm_protocols::disagg::KvHashingRequestEnvelope::default(),
        expected_hash_digest: None,
        bundle: None,
    };
    assert!(
        p_cd.push_prefill_request(&req).await.is_err(),
        "prefill must not be allowed to push"
    );
}

// ---- Prefill dispatch over velo --------------------------------------------

/// A hub with P2P, CD and the prefill router.
async fn start_server_with_router(per_worker_concurrency: u32) -> HubServer {
    let router = kvbm_hub::PrefillRouterManager::new(kvbm_hub::SelectorConfig {
        per_worker_concurrency,
        block_size: 16,
    });
    kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(new_velo_transport() as Arc<dyn Transport>)
        .extension(Arc::new(kvbm_hub::P2pManager::new()))
        .extension(Arc::new(ConditionalDisaggManager::new()))
        .extension(router)
        .serve()
        .await
        .expect("start test server with the prefill router")
}

fn dispatch_request(id: &str, decode: InstanceId) -> PrefillRequest {
    PrefillRequest {
        protocol_version: DISAGG_PROTOCOL_VERSION,
        request_id: id.to_string(),
        session_id: uuid::Uuid::new_v4(),
        initiator_instance_id: decode,
        decode_endpoint: None,
        token_ids: vec![1, 2, 3],
        num_provided_tokens: 0,
        request: kvbm_protocols::disagg::KvHashingRequestEnvelope::default(),
        expected_hash_digest: None,
        bundle: None,
    }
}

/// A decode client registered with `server`.
async fn decode_client(server: &HubServer) -> (Arc<velo::Velo>, Arc<ConditionalDisaggClient>) {
    let velo = new_velo().await;
    let client = ConditionalDisaggClient::new(
        build_client(server),
        Arc::clone(&velo),
        ConditionalDisaggRole::Decode,
    );
    client
        .register(&velo, test_layout_compat_payload())
        .await
        .unwrap()
        .expect("hub velo id");
    (velo, client)
}

/// A decode worker's request reaches the prefill worker's channel in one
/// trip through the hub, with no poll on either side.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pushed_request_reaches_the_prefill_worker_without_a_poll() {
    let server = start_server_with_router(1).await;

    let p_velo = new_velo().await;
    let mut inbox = kvbm_hub::register_prefill_worker(p_velo.messenger(), 8, |_| Ok(())).unwrap();
    let p_hub = build_client(&server);
    let p_cd = ConditionalDisaggClient::new(
        Arc::clone(&p_hub),
        Arc::clone(&p_velo),
        ConditionalDisaggRole::Prefill,
    );
    let hub_id = p_cd
        .register_with(
            &p_velo,
            test_layout_compat_payload(),
            vec![Feature::PrefillRouter(kvbm_hub::PrefillRouterConfig {
                backend: kvbm_hub::PrefillBackendAdvertisement::Velo {
                    instance_id: p_velo.instance_id(),
                },
            })],
            None,
        )
        .await
        .unwrap()
        .expect("hub velo id");
    let hub_peer = p_hub
        .discovery()
        .discover_by_instance_id(hub_id)
        .await
        .unwrap();
    p_velo.register_peer(hub_peer).unwrap();

    let (d_velo, d_cd) = decode_client(&server).await;

    let mut latencies = Vec::new();
    for n in 0..20 {
        let request = dispatch_request(&format!("r{n}"), d_velo.instance_id());
        let started = std::time::Instant::now();
        d_cd.push_prefill_request(&request).await.unwrap();
        let assignment = tokio::time::timeout(Duration::from_secs(1), inbox.recv())
            .await
            .expect("the request reaches the prefill worker")
            .unwrap();
        latencies.push(started.elapsed());
        assert_eq!(assignment.request.request_id, request.request_id);
        // The done notice frees the only permit, so the next push must
        // wait for it to reach the hub.
        drop(assignment);
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        loop {
            assert!(
                std::time::Instant::now() < deadline,
                "the done notice did not free the prefill worker"
            );
            let counters: serde_json::Value = http()
                .get(control_url(&server, "/v1/ext/prefill_router/counters"))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            if counters["available_permits"] == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    assert!(
        p50 < Duration::from_millis(5),
        "push to prefill worker p50 is {p50:?}: {latencies:?}"
    );
}

/// With no prefill worker, the hub rejects the request at once, so the
/// decode worker can prefill locally.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_push_with_no_prefill_worker_fails_at_once() {
    let server = start_server_with_router(1).await;
    let (d_velo, d_cd) = decode_client(&server).await;

    let started = std::time::Instant::now();
    let error = d_cd
        .push_prefill_request(&dispatch_request("r1", d_velo.instance_id()))
        .await
        .expect_err("no prefill worker is registered");
    assert!(
        started.elapsed() < Duration::from_millis(100),
        "the rejection took {:?}",
        started.elapsed()
    );
    assert!(format!("{error:#}").contains("free capacity"), "{error:#}");
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

fn indexer_manifest() -> CacheManifest {
    CacheManifest::new(
        ModelIdentity::new("test-architecture", "test-revision", [42; 32]).unwrap(),
        "test-cache-abi",
        vec![
            kvbm_protocols::cache_manifest::ResourceRequirement::new(
                kvbm_common::LogicalResourceId(1),
                kvbm_protocols::cache_manifest::ResourceRole::PrefixHistory,
                IDX_BLOCK_SIZE as u32,
            )
            .unwrap(),
        ],
        std::collections::BTreeMap::new(),
    )
    .unwrap()
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
        .extension(Arc::clone(&mgr))
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
    let manifest = indexer_manifest();
    let manifest_id = manifest.id();
    mgr.indexes()
        .bind(
            holder.as_u128(),
            &manifest,
            kvbm_logical::events::CreateKind::Block,
            None,
        )
        .unwrap();
    let hashes = idx_plhs(3, 4242);
    mgr.indexes().apply(KvbmCacheEvents {
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
        .find_blocks(manifest_id, hashes.clone())
        .await
        .unwrap()
        .expect("deepest hit");
    assert_eq!(hit.matched, hashes[2], "deepest candidate");
    assert_eq!(hit.kind, kvbm_logical::events::CreateKind::Block);
    assert!(
        hit.candidates.contains(&holder),
        "candidates should reconstruct the seeded InstanceId"
    );

    // Deepest blocks unknown → falls back to the shallow hit (position 1).
    let unknown = idx_plhs(5, 9999);
    let mut mixed = vec![hashes[0], hashes[1]];
    mixed.extend_from_slice(&unknown[2..]); // positions 2..4 unknown
    let hit = lookup
        .find_blocks(manifest_id, mixed)
        .await
        .unwrap()
        .expect("shallow hit");
    assert_eq!(hit.matched, hashes[1], "shallow fallback");

    // Full miss → None.
    let miss = lookup
        .find_blocks(manifest_id, idx_plhs(2, 123_456))
        .await
        .unwrap();
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
        .extension(Arc::clone(&mgr))
        .serve()
        .await
        .expect("start indexer hub without transport");

    let client_velo = new_velo().await;
    let hub_client = build_client(&server);
    let hub_id = hub_client
        .register(&client_velo, vec![], None)
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
    hub_client
        .register(
            client_velo,
            vec![Feature::Indexer(kvbm_hub::IndexerFeatureConfig {
                max_seq_len: None,
                manifest: indexer_manifest(),
                create_kind: kvbm_logical::events::CreateKind::Block,
            })],
            Some(kvbm_hub::protocol::RuntimeConfigSummary {
                block_size: Some(IDX_BLOCK_SIZE),
                block_layout: None,
            }),
        )
        .await
        .unwrap()
        .expect("hub should return its own instance id when running with a transport");
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
