// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for the KV indexer feature: two mock `PUB` publishers push
//! `KvbmCacheEvents` over ZMQ to the hub's `SUB` ingest socket; the test then
//! asserts the index via the feature's own HTTP surface
//! (`/v1/features/indexer/...`).

use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_tokens::TokenBlockSequence;
use futures::SinkExt;
use kvbm_hub::{
    Feature, FeatureManager, HubServer, IndexerConfigResponse, IndexerFeatureConfig, IndexerManager,
};
use kvbm_logical::events::{KvCacheEvents, KvbmCacheEvents};
use kvbm_logical::{KvbmSequenceHashProvider, SequenceHash};
use kvbm_protocols::cache_manifest::{
    CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};
use serde_json::Value;
use tmq::{Context, Multipart, publish::Publish, publish::publish};
use velo_ext::InstanceId;

const BLOCK_SIZE: u32 = 4;
const MAX_SEQ_LEN: usize = 64;

fn test_manifest() -> CacheManifest {
    CacheManifest::new(
        ModelIdentity::new("test-architecture", "test-revision", [7; 32]).unwrap(),
        "test-cache-abi",
        vec![
            ResourceRequirement::new(
                kvbm_common::LogicalResourceId(1),
                ResourceRole::PrefixHistory,
                BLOCK_SIZE,
            )
            .unwrap(),
        ],
        std::collections::BTreeMap::new(),
    )
    .unwrap()
}

async fn start_hub() -> (HubServer, Arc<IndexerManager>) {
    let manager = Arc::new(
        kvbm_hub::IndexerManager::new(
            MAX_SEQ_LEN,
            BLOCK_SIZE as usize,
            Some("tcp://127.0.0.1:0".to_string()),
            Some("127.0.0.1".to_string()),
        )
        .expect("build indexer manager"),
    );

    let server = kvbm_hub::create_server_builder()
        .bind_addr("127.0.0.1".parse().unwrap())
        .discovery_port(0)
        .control_port(0)
        .heartbeat_interval(Duration::from_secs(3600))
        .heartbeat_max_failures(u32::MAX)
        .registration_ttl(Duration::from_secs(3600))
        .add_feature_manager(Arc::clone(&manager) as Arc<dyn FeatureManager>)
        .serve()
        .await
        .expect("start hub");
    (server, manager)
}

/// Builds `n` PLHs at positions 0..n by laying down `n * BLOCK_SIZE` tokens.
fn plhs(n: usize, salt: u64) -> Vec<SequenceHash> {
    let tokens: Vec<u32> = (0..(BLOCK_SIZE as usize * n) as u32).collect();
    let seq = TokenBlockSequence::from_slice(&tokens, BLOCK_SIZE, Some(salt));
    seq.blocks()
        .iter()
        .map(|b| b.kvbm_sequence_hash())
        .collect()
}

fn connect_pub(endpoint: &str) -> Publish {
    let ctx = Context::new();
    publish(&ctx)
        .set_linger(0)
        .connect(endpoint)
        .expect("connect PUB")
}

async fn send_batch(sock: &mut Publish, events: KvCacheEvents, instance_id: u128) {
    let batch = KvbmCacheEvents {
        events,
        instance_id,
    };
    let payload = rmp_serde::to_vec(&batch).expect("encode batch");
    let frames: Vec<Vec<u8>> = vec![b"kvbm.kv_index".to_vec(), payload];
    sock.send(Multipart::from(frames)).await.expect("PUB send");
}

async fn get_json(http: &reqwest::Client, base: &str, path: &str) -> Value {
    http.get(format!("{base}/v1/features/indexer{path}"))
        .send()
        .await
        .expect("GET")
        .json()
        .await
        .expect("json")
}

fn instances(entry: &Value) -> Vec<String> {
    let mut v: Vec<String> = entry["instances"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    v.sort();
    v
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_instances_publish_index_and_query() {
    let (server, manager) = start_hub().await;
    let base = format!("http://{}", server.discovery_addr());
    let http = reqwest::Client::new();
    let manifest = test_manifest();
    let manifest_id = manifest.id();

    // GET /config: feature present, sizing reported, ZMQ endpoint advertised.
    let cfg: IndexerConfigResponse =
        serde_json::from_value(get_json(&http, &base, "/config").await).expect("config");
    assert_eq!(cfg.block_size, BLOCK_SIZE as usize);
    assert_eq!(cfg.max_seq_len, MAX_SEQ_LEN);
    assert_eq!(cfg.num_positions, MAX_SEQ_LEN / BLOCK_SIZE as usize);
    assert!(
        cfg.zmq_endpoint.starts_with("tcp://127.0.0.1:"),
        "endpoint: {}",
        cfg.zmq_endpoint
    );

    // Two workers holding the same 3-block prefix.
    let instance_a = InstanceId::new_v4();
    let instance_b = InstanceId::new_v4();
    let id_a = instance_a.as_u128();
    let id_b = instance_b.as_u128();
    let indexer = Feature::Indexer(IndexerFeatureConfig {
        max_seq_len: Some(MAX_SEQ_LEN),
        manifest,
        create_kind: kvbm_logical::events::CreateKind::Block,
    });
    manager
        .on_register(instance_a, &indexer)
        .await
        .expect("bind first indexer instance");
    manager
        .on_register(instance_b, &indexer)
        .await
        .expect("bind second indexer instance");
    let registrations = get_json(&http, &base, "/instances").await;
    assert_eq!(registrations["bindings"].as_array().unwrap().len(), 2);
    let binding_instances = registrations["bindings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|binding| {
            binding["instance"]
                .as_str()
                .unwrap()
                .parse::<u128>()
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(
        binding_instances.windows(2).all(|pair| pair[0] <= pair[1]),
        "bindings are sorted by instance"
    );
    assert_eq!(
        registrations["bindings"][0]["manifest"].as_str(),
        Some(manifest_id.to_string().as_str())
    );
    let invalid_manifest_path = http
        .get(format!(
            "{base}/v1/features/indexer/manifests/not-hex/hashes/by_position/0"
        ))
        .send()
        .await
        .expect("bad manifest request");
    assert_eq!(
        invalid_manifest_path.status(),
        reqwest::StatusCode::BAD_REQUEST
    );
    let hashes = plhs(3, 1337);

    let mut pub_a = connect_pub(&cfg.zmq_endpoint);
    let mut pub_b = connect_pub(&cfg.zmq_endpoint);
    // Give the SUB time to register the PUB connections before the first send.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // by_position/0 lists both instances. Resend each poll to defeat ZMQ's
    // slow-joiner (a PUB drops messages sent before the SUB connection
    // completes). Create is idempotent in the index.
    let deadline = Instant::now() + Duration::from_secs(8);
    let body = loop {
        send_batch(&mut pub_a, KvCacheEvents::Create(hashes.clone()), id_a).await;
        send_batch(&mut pub_b, KvCacheEvents::Create(hashes.clone()), id_b).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let body = get_json(
            &http,
            &base,
            &format!("/manifests/{manifest_id}/hashes/by_position/0"),
        )
        .await;
        let ready = body["entries"]
            .as_array()
            .map(|e| !e.is_empty() && instances(&e[0]).len() == 2)
            .unwrap_or(false);
        if ready {
            break body;
        }
        assert!(
            Instant::now() < deadline,
            "timed out indexing creates: {body}"
        );
    };
    assert_eq!(
        instances(&body["entries"][0]),
        vec![id_a.to_string(), id_b.to_string()]
    );

    // POST /query with the full sequence → deepest match (position 2).
    let resp: Value = http
        .post(format!("{base}/v1/features/indexer/query"))
        .json(&serde_json::json!({ "manifest": manifest_id, "hashes": hashes }))
        .send()
        .await
        .expect("POST query")
        .json()
        .await
        .expect("query json");
    let hit = &resp["hit"];
    assert_eq!(hit["position"].as_u64(), Some(2));
    assert_eq!(instances(hit), vec![id_a.to_string(), id_b.to_string()]);

    // Remove instance A's blocks → only B remains at position 0.
    let deadline = Instant::now() + Duration::from_secs(8);
    let body = loop {
        send_batch(&mut pub_a, KvCacheEvents::Remove(hashes.clone()), id_a).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let body = get_json(
            &http,
            &base,
            &format!("/manifests/{manifest_id}/hashes/by_position/0"),
        )
        .await;
        let ready = body["entries"]
            .as_array()
            .and_then(|e| e.first())
            .map(|e0| instances(e0) == vec![id_b.to_string()])
            .unwrap_or(false);
        if ready {
            break body;
        }
        assert!(
            Instant::now() < deadline,
            "timed out indexing remove: {body}"
        );
    };
    assert_eq!(instances(&body["entries"][0]), vec![id_b.to_string()]);

    server.shutdown().await.expect("shutdown");
}
