use std::net::{IpAddr, Ipv4Addr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kvbm_hub::{ControlPlaneManager, HubServer};
use kvbm_protocols::control::{
    ControlError, ControlReply, LIST_MODULES_HANDLER, ListModulesRequest, ListModulesResponse,
    ModuleId, RESET_HANDLER, ResetRequest, ResetResponse, Tier,
};
use velo::Handler;
use velo::transports::tcp::TcpTransportBuilder;

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

async fn start_hub() -> HubServer {
    let transport = new_velo_transport();
    let manager: Arc<ControlPlaneManager> = Arc::new(ControlPlaneManager::new());
    kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(transport as Arc<dyn velo::Transport>)
        .add_feature_manager(manager as Arc<dyn kvbm_hub::FeatureManager>)
        .heartbeat_interval(Duration::from_secs(3600))
        .heartbeat_max_failures(u32::MAX)
        .registration_ttl(Duration::from_secs(3600))
        .serve()
        .await
        .expect("start hub")
}

fn build_client(server: &HubServer) -> Arc<kvbm_hub::HubClient> {
    kvbm_hub::create_client_builder()
        .host(server.discovery_addr().ip().to_string())
        .discovery_port(server.discovery_addr().port())
        .control_port(server.control_addr().port())
        .build()
        .expect("build hub client")
}

fn install_modules(peer: &velo::Velo, modules: Vec<ModuleId>) {
    peer.register_handler(
        Handler::typed_unary_async::<ListModulesRequest, _, _, _>(
            LIST_MODULES_HANDLER,
            move |_ctx| {
                let modules = modules.clone();
                async move { Ok(ControlReply::Ok(ListModulesResponse { modules })) }
            },
        )
        .build(),
    )
    .unwrap();
}

fn install_reset_handler(
    peer: &velo::Velo,
    reply: ControlReply<ResetResponse>,
    received_tiers: Option<Arc<Mutex<Option<Vec<Tier>>>>>,
    calls: Arc<AtomicUsize>,
) {
    peer.register_handler(
        Handler::typed_unary_async::<ResetRequest, ControlReply<ResetResponse>, _, _>(
            RESET_HANDLER,
            move |ctx| {
                let reply = reply.clone();
                let received_tiers = received_tiers.clone();
                let calls = calls.clone();
                let tiers = ctx.input.tiers;
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    if let Some(received_tiers) = received_tiers {
                        *received_tiers.lock().unwrap() = tiers;
                    }
                    Ok(reply)
                }
            },
        )
        .build(),
    )
    .unwrap();
}

fn reset_reply() -> ControlReply<ResetResponse> {
    ControlReply::Ok(ResetResponse {
        reset: vec![Tier::G2],
        failed: vec![],
        skipped_unconfigured: vec![Tier::G3],
    })
}

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

fn fanout_url(server: &HubServer, control: bool) -> String {
    let addr = if control {
        server.control_addr()
    } else {
        server.discovery_addr()
    };
    format!("http://{addr}/v1/reset")
}

fn modules_url(server: &HubServer, id: velo_ext::InstanceId) -> String {
    format!("http://{}/v1/instances/{id}/modules", server.control_addr())
}

async fn wait_for_modules_cached(server: &HubServer, id: velo_ext::InstanceId, deadline: Duration) {
    let started = Instant::now();
    loop {
        let response = http()
            .get(modules_url(server, id))
            .send()
            .await
            .expect("GET modules");
        if response.status().as_u16() == 200 {
            let body: serde_json::Value = response.json().await.expect("modules JSON");
            if body["cached"] == serde_json::json!(true) {
                return;
            }
        }
        assert!(
            started.elapsed() <= deadline,
            "modules cache not populated within {deadline:?}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fanout_passes_tiers_and_returns_each_reset_response() {
    let server = start_hub().await;
    let tiers = vec![Tier::G3, Tier::G1];

    let peer_a = new_velo().await;
    install_modules(&peer_a, vec![ModuleId::Core, ModuleId::Dev]);
    let received_a = Arc::new(Mutex::new(None));
    install_reset_handler(
        &peer_a,
        reset_reply(),
        Some(received_a.clone()),
        Arc::new(AtomicUsize::new(0)),
    );

    let peer_b = new_velo().await;
    install_modules(&peer_b, vec![ModuleId::Core, ModuleId::Dev]);
    let received_b = Arc::new(Mutex::new(None));
    install_reset_handler(
        &peer_b,
        reset_reply(),
        Some(received_b.clone()),
        Arc::new(AtomicUsize::new(0)),
    );

    let client_a = build_client(&server);
    client_a
        .register_instance(peer_a.peer_info())
        .await
        .expect("register a");
    let client_b = build_client(&server);
    client_b
        .register_instance(peer_b.peer_info())
        .await
        .expect("register b");
    wait_for_modules_cached(&server, peer_a.instance_id(), Duration::from_secs(2)).await;
    wait_for_modules_cached(&server, peer_b.instance_id(), Duration::from_secs(2)).await;

    let response = http()
        .post(fanout_url(&server, true))
        .header("content-type", "application/json")
        .body(r#"{"tiers":["g3","g1"]}"#)
        .send()
        .await
        .expect("POST reset fanout");
    assert_eq!(response.status().as_u16(), 200);
    let body: serde_json::Value = response.json().await.expect("reset fanout JSON");
    let instances = body["instances"].as_object().expect("instances object");
    for id in [peer_a.instance_id(), peer_b.instance_id()] {
        assert_eq!(
            instances[&id.to_string()]["response"]["reset"],
            serde_json::json!(["g2"])
        );
        assert!(instances[&id.to_string()].get("error").is_none());
    }
    assert_eq!(*received_a.lock().unwrap(), Some(tiers.clone()));
    assert_eq!(*received_b.lock().unwrap(), Some(tiers));

    server.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fanout_filters_leaders_without_dev_module() {
    let server = start_hub().await;

    let peer_dev = new_velo().await;
    install_modules(&peer_dev, vec![ModuleId::Core, ModuleId::Dev]);
    let dev_calls = Arc::new(AtomicUsize::new(0));
    install_reset_handler(&peer_dev, reset_reply(), None, dev_calls.clone());

    let peer_without_dev = new_velo().await;
    install_modules(&peer_without_dev, vec![ModuleId::Core, ModuleId::Transfer]);
    let filtered_calls = Arc::new(AtomicUsize::new(0));
    install_reset_handler(
        &peer_without_dev,
        reset_reply(),
        None,
        filtered_calls.clone(),
    );

    let dev_client = build_client(&server);
    dev_client
        .register_instance(peer_dev.peer_info())
        .await
        .expect("register dev leader");
    let other_client = build_client(&server);
    other_client
        .register_instance(peer_without_dev.peer_info())
        .await
        .expect("register leader without dev");
    wait_for_modules_cached(&server, peer_dev.instance_id(), Duration::from_secs(2)).await;
    wait_for_modules_cached(
        &server,
        peer_without_dev.instance_id(),
        Duration::from_secs(2),
    )
    .await;

    let response = http()
        .post(fanout_url(&server, true))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("POST reset fanout");
    assert_eq!(response.status().as_u16(), 200);
    let body: serde_json::Value = response.json().await.expect("reset fanout JSON");
    let instances = body["instances"].as_object().expect("instances object");
    assert!(instances.contains_key(&peer_dev.instance_id().to_string()));
    assert!(!instances.contains_key(&peer_without_dev.instance_id().to_string()));
    assert_eq!(dev_calls.load(Ordering::SeqCst), 1);
    assert_eq!(filtered_calls.load(Ordering::SeqCst), 0);

    server.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fanout_returns_leader_error_without_failing_other_results() {
    let server = start_hub().await;

    let peer_ok = new_velo().await;
    install_modules(&peer_ok, vec![ModuleId::Core, ModuleId::Dev]);
    install_reset_handler(&peer_ok, reset_reply(), None, Arc::new(AtomicUsize::new(0)));

    let peer_error = new_velo().await;
    install_modules(&peer_error, vec![ModuleId::Core, ModuleId::Dev]);
    install_reset_handler(
        &peer_error,
        ControlReply::Err(ControlError::Internal("simulated".to_string())),
        None,
        Arc::new(AtomicUsize::new(0)),
    );

    let ok_client = build_client(&server);
    ok_client
        .register_instance(peer_ok.peer_info())
        .await
        .expect("register successful leader");
    let error_client = build_client(&server);
    error_client
        .register_instance(peer_error.peer_info())
        .await
        .expect("register failing leader");
    wait_for_modules_cached(&server, peer_ok.instance_id(), Duration::from_secs(2)).await;
    wait_for_modules_cached(&server, peer_error.instance_id(), Duration::from_secs(2)).await;

    let response = http()
        .post(fanout_url(&server, true))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("POST reset fanout");
    assert_eq!(response.status().as_u16(), 200);
    let body: serde_json::Value = response.json().await.expect("reset fanout JSON");
    let instances = body["instances"].as_object().expect("instances object");
    assert_eq!(
        instances[&peer_ok.instance_id().to_string()]["response"]["reset"],
        serde_json::json!(["g2"])
    );
    assert!(
        instances[&peer_error.instance_id().to_string()]["error"]
            .as_str()
            .is_some_and(|error| error.contains("simulated"))
    );

    server.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reset_fanout_is_control_port_only() {
    let server = start_hub().await;

    let public_response = http()
        .post(fanout_url(&server, false))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("POST reset on public port");
    assert!(
        matches!(public_response.status().as_u16(), 404 | 405),
        "public route returned {}",
        public_response.status()
    );

    let control_response = http()
        .post(fanout_url(&server, true))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("POST reset on control port");
    assert_eq!(control_response.status().as_u16(), 200);

    server.shutdown().await.unwrap();
}
