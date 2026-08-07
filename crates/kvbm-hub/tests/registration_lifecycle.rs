// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use axum::Router;
use futures::future::BoxFuture;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_hub::protocol::{
    Feature, FeatureKey, IndexerFeatureConfig, MUTATION_CREDENTIAL_HEADER, MutationCredential,
    RegisterRequest, RegisterResponse, RuntimeConfigSummary, instance_by_id, instance_heartbeat,
    paths,
};
use kvbm_hub::{
    BundleAdvertisementRecord, BundleInvalidateRequest, BundlePublishRequest,
    BundleQueryMissReason, BundleQueryOutcome, BundleQueryRequest, EvictionCallback, FeatureError,
    FeatureManager, HubContext, HubServer, IndexerManager, PeerRegistry, RegisteredPeer,
    RegistryError, RegistryIncarnation, RegistryRemoval,
};
use kvbm_protocols::cache_manifest::{
    BundleKey, BundleResourceLineage, CacheManifestId, RegistrationEpoch, ResourceRequirement,
    ResourceRole,
};
use velo::Transport;
use velo::discovery::PeerDiscovery;
use velo::transports::tcp::TcpTransportBuilder;
use velo_ext::{InstanceId, PeerInfo, WorkerAddress, WorkerId};

#[derive(Default)]
struct CustomRegistry {
    state: RwLock<CustomRegistryState>,
    fail_next_register: AtomicBool,
    next_register_gate: Mutex<Option<Arc<RegistrationGate>>>,
    next_touch_gate: Mutex<Option<Arc<TouchGate>>>,
}

#[derive(Default)]
struct CustomRegistryState {
    peers: HashMap<InstanceId, RegisteredPeer>,
    last_incarnation: u64,
    removal_callback: Option<EvictionCallback>,
}

impl CustomRegistry {
    fn fail_next_register(&self) {
        self.fail_next_register.store(true, Ordering::Release);
    }

    fn delay_next_register(&self) -> Arc<RegistrationGate> {
        let gate = Arc::new(RegistrationGate::default());
        *self.next_register_gate.lock().unwrap() = Some(Arc::clone(&gate));
        gate
    }

    fn expire(&self, id: InstanceId) {
        let removed = {
            let mut state = self.state.write().unwrap();
            state.peers.remove(&id).map(|registered| {
                (
                    RegistryRemoval::new(id, registered.incarnation()),
                    state.removal_callback.clone(),
                )
            })
        };
        if let Some((removal, Some(callback))) = removed {
            callback(removal);
        }
    }

    fn delay_next_touch(&self) -> Arc<TouchGate> {
        let gate = Arc::new(TouchGate::default());
        *self.next_touch_gate.lock().unwrap() = Some(Arc::clone(&gate));
        gate
    }
}

struct RegistrationGate {
    entered: tokio::sync::Notify,
    release: tokio::sync::Semaphore,
}

impl Default for RegistrationGate {
    fn default() -> Self {
        Self {
            entered: tokio::sync::Notify::new(),
            release: tokio::sync::Semaphore::new(0),
        }
    }
}

struct TouchGate {
    entered: tokio::sync::Notify,
    release: tokio::sync::Semaphore,
}

impl Default for TouchGate {
    fn default() -> Self {
        Self {
            entered: tokio::sync::Notify::new(),
            release: tokio::sync::Semaphore::new(0),
        }
    }
}

impl PeerDiscovery for CustomRegistry {
    fn discover_by_worker_id(&self, worker_id: WorkerId) -> BoxFuture<'_, Result<PeerInfo>> {
        let result = self
            .state
            .read()
            .unwrap()
            .peers
            .values()
            .find(|registered| registered.peer().worker_id() == worker_id)
            .map(|registered| registered.peer().clone())
            .ok_or_else(|| anyhow!("worker {} not registered", worker_id.as_u64()));
        Box::pin(async move { result })
    }

    fn discover_by_instance_id(&self, id: InstanceId) -> BoxFuture<'_, Result<PeerInfo>> {
        let result = self
            .state
            .read()
            .unwrap()
            .peers
            .get(&id)
            .map(|registered| registered.peer().clone())
            .ok_or_else(|| anyhow!("instance {id} not registered"));
        Box::pin(async move { result })
    }
}

#[async_trait]
impl PeerRegistry for CustomRegistry {
    async fn register(&self, peer: PeerInfo) -> Result<RegistryIncarnation, RegistryError> {
        let gate = self.next_register_gate.lock().unwrap().take();
        if let Some(gate) = gate {
            gate.entered.notify_one();
            gate.release
                .acquire()
                .await
                .map_err(|_| RegistryError::Backend(anyhow!("registration gate closed")))?
                .forget();
        }
        if self.fail_next_register.swap(false, Ordering::AcqRel) {
            return Err(RegistryError::Backend(anyhow!("injected register failure")));
        }
        let mut state = self.state.write().unwrap();
        if let Some(existing) = state
            .peers
            .values()
            .find(|current| {
                current.peer().worker_id() == peer.worker_id()
                    && current.peer().instance_id() != peer.instance_id()
            })
            .map(|registered| registered.peer().instance_id())
        {
            return Err(RegistryError::Conflict {
                worker_id: peer.worker_id(),
                existing,
            });
        }
        state.last_incarnation = state
            .last_incarnation
            .checked_add(1)
            .ok_or_else(|| RegistryError::Backend(anyhow!("registry incarnation exhausted")))?;
        let incarnation = RegistryIncarnation::from_u64(state.last_incarnation);
        state
            .peers
            .insert(peer.instance_id(), RegisteredPeer::new(peer, incarnation));
        Ok(incarnation)
    }

    async fn unregister(
        &self,
        id: InstanceId,
        incarnation: RegistryIncarnation,
    ) -> Result<(), RegistryError> {
        let (removal, callback) = {
            let mut state = self.state.write().unwrap();
            let current = state
                .peers
                .get(&id)
                .ok_or(RegistryError::NotFound(id))?
                .incarnation();
            if current != incarnation {
                return Err(RegistryError::StaleIncarnation {
                    instance_id: id,
                    expected: incarnation,
                    current,
                });
            }
            state.peers.remove(&id);
            (
                RegistryRemoval::new(id, incarnation),
                state.removal_callback.clone(),
            )
        };
        if let Some(callback) = callback {
            callback(removal);
        }
        Ok(())
    }

    async fn touch(
        &self,
        id: InstanceId,
        incarnation: RegistryIncarnation,
    ) -> Result<(), RegistryError> {
        let gate = self.next_touch_gate.lock().unwrap().take();
        if let Some(gate) = gate {
            gate.entered.notify_one();
            gate.release
                .acquire()
                .await
                .map_err(|_| RegistryError::Backend(anyhow!("touch gate closed")))?
                .forget();
        }
        let state = self.state.read().unwrap();
        let current = state
            .peers
            .get(&id)
            .ok_or(RegistryError::NotFound(id))?
            .incarnation();
        if current != incarnation {
            return Err(RegistryError::StaleIncarnation {
                instance_id: id,
                expected: incarnation,
                current,
            });
        }
        Ok(())
    }

    fn is_current(&self, id: InstanceId, incarnation: RegistryIncarnation) -> bool {
        self.state
            .read()
            .unwrap()
            .peers
            .get(&id)
            .is_some_and(|registered| registered.incarnation() == incarnation)
    }

    fn current_incarnation(&self, id: InstanceId) -> Option<RegistryIncarnation> {
        self.state
            .read()
            .unwrap()
            .peers
            .get(&id)
            .map(RegisteredPeer::incarnation)
    }

    fn contains(&self, id: InstanceId) -> bool {
        self.state.read().unwrap().peers.contains_key(&id)
    }

    fn list(&self) -> Vec<PeerInfo> {
        self.state
            .read()
            .unwrap()
            .peers
            .values()
            .map(|registered| registered.peer().clone())
            .collect()
    }

    fn registrations(&self) -> Vec<RegisteredPeer> {
        self.state.read().unwrap().peers.values().cloned().collect()
    }

    fn install_removal_hook(
        &self,
        callback: EvictionCallback,
    ) -> Result<Vec<RegisteredPeer>, RegistryError> {
        let mut state = self.state.write().unwrap();
        if state.removal_callback.is_some() {
            return Err(RegistryError::RemovalHookAlreadyInstalled);
        }
        state.removal_callback = Some(callback);
        Ok(state.peers.values().cloned().collect())
    }
}

#[tokio::test]
async fn registry_write_failure_restores_prior_registration_credential() {
    let registry = Arc::new(CustomRegistry::default());
    let server = start_custom_server(Arc::clone(&registry), None, None).await;
    let peer = make_peer();
    let request = register_request(peer.clone());
    let initial = register(&server, &request, None).await;
    let credential = initial.mutation_credential.unwrap();

    registry.fail_next_register();
    let failed = reqwest::Client::new()
        .post(control_url(&server, paths::INSTANCES))
        .header(MUTATION_CREDENTIAL_HEADER, credential.to_header_value())
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(failed.status(), 500);
    assert!(registry.contains(peer.instance_id()));

    let removed = reqwest::Client::new()
        .delete(control_url(&server, &instance_by_id(peer.instance_id())))
        .header(MUTATION_CREDENTIAL_HEADER, credential.to_header_value())
        .send()
        .await
        .unwrap();
    assert_eq!(removed.status(), 204);
}

#[tokio::test]
async fn custom_registry_http_removal_revokes_registration_authority() {
    let registry = Arc::new(CustomRegistry::default());
    let server = start_custom_server(Arc::clone(&registry), None, None).await;
    let peer = make_peer();
    let request = register_request(peer.clone());
    let initial = register(&server, &request, None).await;
    let credential = initial.mutation_credential.unwrap();

    let removed = reqwest::Client::new()
        .delete(control_url(&server, &instance_by_id(peer.instance_id())))
        .header(MUTATION_CREDENTIAL_HEADER, credential.to_header_value())
        .send()
        .await
        .unwrap();
    assert_eq!(removed.status(), 204);

    let fresh = reqwest::Client::new()
        .post(control_url(&server, paths::INSTANCES))
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(fresh.status(), 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn authenticated_heartbeat_cannot_touch_a_replacement_incarnation() {
    let registry = Arc::new(CustomRegistry::default());
    let server = start_custom_server(Arc::clone(&registry), None, None).await;
    let peer = make_peer();
    let request = register_request(peer.clone());
    let initial = register(&server, &request, None).await;
    let stale_credential = initial.mutation_credential.unwrap();
    let gate = registry.delay_next_touch();

    let heartbeat_url = control_url(&server, &instance_heartbeat(peer.instance_id()));
    let heartbeat_credential = stale_credential.clone();
    let heartbeat = tokio::spawn(async move {
        reqwest::Client::new()
            .post(heartbeat_url)
            .header(
                MUTATION_CREDENTIAL_HEADER,
                heartbeat_credential.to_header_value(),
            )
            .send()
            .await
            .unwrap()
    });
    gate.entered.notified().await;

    let replacement = register(&server, &request, Some(&stale_credential)).await;
    let current_credential = replacement.mutation_credential.unwrap();
    gate.release.add_permits(1);

    assert_eq!(
        heartbeat.await.unwrap().status(),
        reqwest::StatusCode::CONFLICT,
        "the stale authorized heartbeat must fail its conditional touch"
    );
    let removed = reqwest::Client::new()
        .delete(control_url(&server, &instance_by_id(peer.instance_id())))
        .header(
            MUTATION_CREDENTIAL_HEADER,
            current_credential.to_header_value(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(removed.status(), reqwest::StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn preloaded_registry_entry_cannot_be_claimed_without_credential() {
    let registry = Arc::new(CustomRegistry::default());
    let peer = make_peer();
    registry.register(peer.clone()).await.unwrap();
    let server = start_custom_server(Arc::clone(&registry), None, None).await;

    let takeover = reqwest::Client::new()
        .post(control_url(&server, paths::INSTANCES))
        .json(&register_request(peer.clone()))
        .send()
        .await
        .unwrap();

    assert_eq!(takeover.status(), reqwest::StatusCode::UNAUTHORIZED);
    assert_eq!(
        registry
            .discover_by_instance_id(peer.instance_id())
            .await
            .unwrap()
            .worker_id(),
        peer.worker_id()
    );
}

#[tokio::test]
async fn hub_self_entry_cannot_be_claimed_as_a_client_registration() {
    let registry = Arc::new(CustomRegistry::default());
    let server = start_custom_server(
        Arc::clone(&registry),
        None,
        Some(new_velo_transport() as Arc<dyn Transport>),
    )
    .await;
    let hub_peer = registry.list().into_iter().next().unwrap();

    let takeover = reqwest::Client::new()
        .post(control_url(&server, paths::INSTANCES))
        .json(&register_request(hub_peer.clone()))
        .send()
        .await
        .unwrap();

    assert_eq!(takeover.status(), reqwest::StatusCode::UNAUTHORIZED);
    assert_eq!(registry.list(), vec![hub_peer]);
}

#[tokio::test]
async fn failed_server_build_does_not_consume_custom_registry_removal_hook() {
    let registry = Arc::new(CustomRegistry::default());
    let occupied = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let occupied_port = occupied.local_addr().unwrap().port();
    let transport = new_velo_transport();

    let failed = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(occupied_port)
        .control_port(0)
        .registry(Arc::clone(&registry) as Arc<dyn PeerRegistry>)
        .add_transport(transport as Arc<dyn Transport>)
        .serve()
        .await;
    assert!(
        failed.is_err(),
        "occupied discovery port must reject startup"
    );
    assert!(
        registry.list().is_empty(),
        "failed startup must roll back the hub's own registry entry"
    );

    let server = start_custom_server(Arc::clone(&registry), None, None).await;
    let peer = make_peer();
    register(&server, &register_request(peer.clone()), None).await;

    registry.expire(peer.instance_id());

    let fresh = reqwest::Client::new()
        .post(control_url(&server, paths::INSTANCES))
        .json(&register_request(peer))
        .send()
        .await
        .unwrap();
    assert_eq!(fresh.status(), 200);
}

#[tokio::test]
async fn failed_manager_attach_cancels_spawned_work_and_leaves_no_hub_entry() {
    let registry = Arc::new(CustomRegistry::default());
    let manager = Arc::new(FailingAttachManager::default());

    let failed = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .registry(Arc::clone(&registry) as Arc<dyn PeerRegistry>)
        .add_transport(new_velo_transport() as Arc<dyn Transport>)
        .add_feature_manager(manager.clone() as Arc<dyn FeatureManager>)
        .serve()
        .await;
    assert!(
        failed.is_err(),
        "injected attach failure must abort startup"
    );
    tokio::time::timeout(Duration::from_secs(1), async {
        while !manager.stopped.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("failed startup did not cancel manager work");
    assert!(
        registry.list().is_empty(),
        "failed attach must not leave a hub self-registration"
    );

    start_custom_server(registry, None, None).await;
}

#[tokio::test]
async fn explicit_shutdown_removes_the_hub_from_a_custom_registry() {
    let registry = Arc::new(CustomRegistry::default());
    let server = start_custom_server(
        Arc::clone(&registry),
        None,
        Some(new_velo_transport() as Arc<dyn Transport>),
    )
    .await;
    assert_eq!(registry.list().len(), 1);

    server.shutdown().await.unwrap();

    assert!(registry.list().is_empty());
}

#[tokio::test]
async fn dropping_the_hub_removes_it_from_a_custom_registry() {
    let registry = Arc::new(CustomRegistry::default());
    let server = start_custom_server(
        Arc::clone(&registry),
        None,
        Some(new_velo_transport() as Arc<dyn Transport>),
    )
    .await;
    assert_eq!(registry.list().len(), 1);

    drop(server);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !registry.list().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping the hub did not remove its custom-registry entry");
}

#[derive(Default)]
struct FailingAttachManager {
    stopped: Arc<AtomicBool>,
}

impl FeatureManager for FailingAttachManager {
    fn key(&self) -> FeatureKey {
        FeatureKey::ConnectorControl
    }

    fn attach<'a>(&'a self, ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        let stopped = Arc::clone(&self.stopped);
        Box::pin(async move {
            tokio::spawn(async move {
                ctx.cancel.cancelled().await;
                stopped.store(true, Ordering::Release);
            });
            Err(FeatureError::Other(anyhow!("injected attach failure")))
        })
    }

    fn on_register<'a>(
        &'a self,
        _instance_id: InstanceId,
        _feature: &'a Feature,
    ) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async { Ok(()) })
    }

    fn on_unregister(&self, _instance_id: InstanceId) {}

    fn control_router(self: Arc<Self>) -> Router {
        Router::new()
    }

    fn public_router(self: Arc<Self>) -> Router {
        Router::new()
    }
}

struct GatedIndexerManager {
    inner: Arc<IndexerManager>,
    next_finalize_gate: Mutex<Option<Arc<RegistrationGate>>>,
}

impl GatedIndexerManager {
    fn new(max_seq_len: usize, block_size: usize) -> Self {
        Self {
            inner: Arc::new(IndexerManager::new(max_seq_len, block_size, None, None).unwrap()),
            next_finalize_gate: Mutex::new(None),
        }
    }

    fn delay_next_finalize(&self) -> Arc<RegistrationGate> {
        let gate = Arc::new(RegistrationGate::default());
        *self.next_finalize_gate.lock().unwrap() = Some(Arc::clone(&gate));
        gate
    }
}

impl FeatureManager for GatedIndexerManager {
    fn key(&self) -> FeatureKey {
        self.inner.key()
    }

    fn config_requirements(&self) -> kvbm_hub::FeatureConfigRequirements {
        self.inner.config_requirements()
    }

    fn authoritative_block_size(&self) -> Option<usize> {
        self.inner.authoritative_block_size()
    }

    fn requires_runtime_summary(&self) -> bool {
        self.inner.requires_runtime_summary()
    }

    fn descriptor(&self, primary: &kvbm_hub::PrimaryConfig) -> serde_json::Value {
        self.inner.descriptor(primary)
    }

    fn route_prefix(&self) -> Option<&'static str> {
        self.inner.route_prefix()
    }

    fn attach<'a>(&'a self, ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        self.inner.attach(ctx)
    }

    fn on_register<'a>(
        &'a self,
        instance_id: InstanceId,
        feature: &'a Feature,
    ) -> BoxFuture<'a, Result<(), FeatureError>> {
        self.inner.on_register(instance_id, feature)
    }

    fn stage_registration(
        &self,
        instance_id: InstanceId,
        credential: &MutationCredential,
        registration_epoch: RegistrationEpoch,
        participates: bool,
    ) -> Result<(), FeatureError> {
        self.inner
            .stage_registration(instance_id, credential, registration_epoch, participates)
    }

    fn commit_registration(
        &self,
        instance_id: InstanceId,
        credential: &MutationCredential,
        registration_epoch: RegistrationEpoch,
        incarnation: RegistryIncarnation,
        participates: bool,
    ) -> Result<(), FeatureError> {
        self.inner.commit_registration(
            instance_id,
            credential,
            registration_epoch,
            incarnation,
            participates,
        )
    }

    fn on_register_any<'a>(
        &'a self,
        instance_id: InstanceId,
        peer: &'a PeerInfo,
        incarnation: RegistryIncarnation,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let gate = self.next_finalize_gate.lock().unwrap().take();
            if let Some(gate) = gate {
                gate.entered.notify_one();
                gate.release.acquire().await.unwrap().forget();
            }
            self.inner
                .on_register_any(instance_id, peer, incarnation)
                .await;
        })
    }

    fn on_unregister(&self, instance_id: InstanceId) {
        self.inner.on_unregister(instance_id);
    }

    fn control_router(self: Arc<Self>) -> Router {
        Arc::clone(&self.inner).control_router()
    }

    fn public_router(self: Arc<Self>) -> Router {
        Arc::clone(&self.inner).public_router()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn indexer_omission_revokes_old_authority_before_registry_write() {
    const BLOCK_SIZE: usize = 4;

    let registry = Arc::new(CustomRegistry::default());
    let indexer = Arc::new(IndexerManager::new(64, BLOCK_SIZE, None, None).unwrap());
    let server = start_custom_server(
        Arc::clone(&registry),
        Some(indexer as Arc<dyn FeatureManager>),
        Some(new_velo_transport() as Arc<dyn Transport>),
    )
    .await;
    let client_velo = new_velo().await;
    let indexed_request = indexer_register_request(client_velo.peer_info(), BLOCK_SIZE);
    let initial = register(&server, &indexed_request, None).await;
    let old_credential = initial.mutation_credential.unwrap();
    let old_epoch = initial.registration_epoch.unwrap();
    let hub_id = initial.hub_instance_id.unwrap();
    let hub_peer = registry.discover_by_instance_id(hub_id).await.unwrap();
    client_velo.register_peer(hub_peer).unwrap();
    let advertisement = bundle_advertisement(client_velo.instance_id(), old_epoch, 7);
    publish_bundle(
        client_velo.messenger(),
        hub_id,
        old_credential.clone(),
        advertisement.clone(),
    )
    .await
    .unwrap();

    let gate = registry.delay_next_register();
    let replacement_url = control_url(&server, paths::INSTANCES);
    let replacement_request = register_request(client_velo.peer_info());
    let replacement_credential = old_credential.clone();
    let replacement = tokio::spawn(async move {
        reqwest::Client::new()
            .post(replacement_url)
            .header(
                MUTATION_CREDENTIAL_HEADER,
                replacement_credential.to_header_value(),
            )
            .json(&replacement_request)
            .send()
            .await
            .unwrap()
    });
    gate.entered.notified().await;

    let stale_publish = publish_bundle(
        client_velo.messenger(),
        hub_id,
        old_credential.clone(),
        bundle_advertisement(client_velo.instance_id(), old_epoch, 8),
    )
    .await;
    let stale_invalidate = invalidate_bundle(
        client_velo.messenger(),
        hub_id,
        old_credential,
        &advertisement,
    )
    .await;
    let staged_query = query_bundle(client_velo.messenger(), hub_id, &advertisement)
        .await
        .unwrap();

    gate.release.add_permits(1);
    assert_eq!(replacement.await.unwrap().status(), reqwest::StatusCode::OK);
    assert!(
        stale_publish.is_err(),
        "the old Indexer credential remained usable while the registry write was pending"
    );
    assert!(
        stale_invalidate.is_err(),
        "the old Indexer credential could invalidate while the registry write was pending"
    );
    assert_eq!(
        staged_query,
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound),
        "a staged Indexer omission remained query-visible"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn indexer_omission_stays_revoked_between_global_commit_and_finalization() {
    const BLOCK_SIZE: usize = 4;

    let registry = Arc::new(CustomRegistry::default());
    let indexer = Arc::new(GatedIndexerManager::new(64, BLOCK_SIZE));
    let server = start_custom_server(
        Arc::clone(&registry),
        Some(Arc::clone(&indexer) as Arc<dyn FeatureManager>),
        Some(new_velo_transport() as Arc<dyn Transport>),
    )
    .await;
    let client_velo = new_velo().await;
    let indexed_request = indexer_register_request(client_velo.peer_info(), BLOCK_SIZE);
    let initial = register(&server, &indexed_request, None).await;
    let old_credential = initial.mutation_credential.unwrap();
    let old_epoch = initial.registration_epoch.unwrap();
    let hub_id = initial.hub_instance_id.unwrap();
    client_velo
        .register_peer(registry.discover_by_instance_id(hub_id).await.unwrap())
        .unwrap();
    let advertisement = bundle_advertisement(client_velo.instance_id(), old_epoch, 7);
    publish_bundle(
        client_velo.messenger(),
        hub_id,
        old_credential.clone(),
        advertisement.clone(),
    )
    .await
    .unwrap();

    let gate = indexer.delay_next_finalize();
    let replacement_url = control_url(&server, paths::INSTANCES);
    let replacement_request = register_request(client_velo.peer_info());
    let replacement_credential = old_credential.clone();
    let replacement = tokio::spawn(async move {
        reqwest::Client::new()
            .post(replacement_url)
            .header(
                MUTATION_CREDENTIAL_HEADER,
                replacement_credential.to_header_value(),
            )
            .json(&replacement_request)
            .send()
            .await
            .unwrap()
    });
    gate.entered.notified().await;

    let stale_publish = publish_bundle(
        client_velo.messenger(),
        hub_id,
        old_credential.clone(),
        bundle_advertisement(client_velo.instance_id(), old_epoch, 8),
    )
    .await;
    let stale_invalidate = invalidate_bundle(
        client_velo.messenger(),
        hub_id,
        old_credential.clone(),
        &advertisement,
    )
    .await;
    let staged_query = query_bundle(client_velo.messenger(), hub_id, &advertisement)
        .await
        .unwrap();

    gate.release.add_permits(1);
    assert_eq!(replacement.await.unwrap().status(), reqwest::StatusCode::OK);
    assert!(
        stale_publish.is_err(),
        "the old Indexer credential revived after the global credential commit"
    );
    assert!(
        stale_invalidate.is_err(),
        "the old Indexer credential invalidated during finalization"
    );
    assert_eq!(
        staged_query,
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
    assert!(
        publish_bundle(
            client_velo.messenger(),
            hub_id,
            old_credential,
            bundle_advertisement(client_velo.instance_id(), old_epoch, 9),
        )
        .await
        .is_err(),
        "the omitted Indexer authority returned after finalization"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn custom_registry_expiry_revokes_indexer_bundle_authority() {
    const BLOCK_SIZE: usize = 4;

    let registry = Arc::new(CustomRegistry::default());
    let indexer = Arc::new(IndexerManager::new(64, BLOCK_SIZE, None, None).unwrap());
    let transport = new_velo_transport();
    let server = start_custom_server(
        Arc::clone(&registry),
        Some(Arc::clone(&indexer) as Arc<dyn FeatureManager>),
        Some(Arc::clone(&transport) as Arc<dyn Transport>),
    )
    .await;
    let client_velo = new_velo().await;
    let request = indexer_register_request(client_velo.peer_info(), BLOCK_SIZE);
    let registration = register(&server, &request, None).await;
    let credential = registration.mutation_credential.unwrap();
    let registration_epoch = registration.registration_epoch.unwrap();
    let hub_id = registration.hub_instance_id.unwrap();
    let hub_peer = registry.discover_by_instance_id(hub_id).await.unwrap();
    client_velo.register_peer(hub_peer).unwrap();
    let owner = client_velo.instance_id();
    publish_bundle(
        client_velo.messenger(),
        hub_id,
        credential.clone(),
        bundle_advertisement(owner, registration_epoch, 1),
    )
    .await
    .unwrap();

    // Force failure after the indexer has rotated its owner credential: Velo
    // rejects this malformed replacement address. The prior bundle authority
    // and peer registration must be restored transactionally.
    let mut invalid_request = request.clone();
    invalid_request.peer_info = PeerInfo::new(
        owner,
        WorkerAddress::from_encoded(b"invalid-velo-address".to_vec()),
    );
    let rejected = reqwest::Client::new()
        .post(control_url(&server, paths::INSTANCES))
        .header(MUTATION_CREDENTIAL_HEADER, credential.to_header_value())
        .json(&invalid_request)
        .send()
        .await
        .unwrap();
    assert_eq!(rejected.status(), 500);
    publish_bundle(
        client_velo.messenger(),
        hub_id,
        credential.clone(),
        bundle_advertisement(owner, registration_epoch, 2),
    )
    .await
    .expect("failed re-registration must restore the prior bundle credential");

    registry.expire(owner);

    assert!(
        publish_bundle(
            client_velo.messenger(),
            hub_id,
            credential,
            bundle_advertisement(owner, registration_epoch, 3),
        )
        .await
        .is_err(),
        "native custom-registry expiry must revoke the owner's bundle credential"
    );
    let fresh = reqwest::Client::new()
        .post(control_url(&server, paths::INSTANCES))
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(
        fresh.status(),
        200,
        "native custom-registry expiry must revoke registration authority"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_indexer_omission_restores_live_tombstone_credential_and_epoch() {
    const BLOCK_SIZE: usize = 4;

    let registry = Arc::new(CustomRegistry::default());
    let indexer = Arc::new(IndexerManager::new(64, BLOCK_SIZE, None, None).unwrap());
    let transport = new_velo_transport();
    let server = start_custom_server(
        Arc::clone(&registry),
        Some(Arc::clone(&indexer) as Arc<dyn FeatureManager>),
        Some(Arc::clone(&transport) as Arc<dyn Transport>),
    )
    .await;
    let client_velo = new_velo().await;
    let request = indexer_register_request(client_velo.peer_info(), BLOCK_SIZE);
    let initial = register(&server, &request, None).await;
    let credential = initial.mutation_credential.unwrap();
    let registration_epoch = initial.registration_epoch.unwrap();
    let hub_id = initial.hub_instance_id.unwrap();
    let hub_peer = registry.discover_by_instance_id(hub_id).await.unwrap();
    client_velo.register_peer(hub_peer).unwrap();
    let owner = client_velo.instance_id();
    let initial_incarnation = registry.current_incarnation(owner).unwrap();

    let retired = bundle_advertisement(owner, registration_epoch, 7);
    publish_bundle(
        client_velo.messenger(),
        hub_id,
        credential.clone(),
        retired.clone(),
    )
    .await
    .unwrap();
    assert!(
        invalidate_bundle(
            client_velo.messenger(),
            hub_id,
            credential.clone(),
            &retired,
        )
        .await
        .unwrap()
    );
    let live = bundle_advertisement(owner, registration_epoch, 9);
    publish_bundle(
        client_velo.messenger(),
        hub_id,
        credential.clone(),
        live.clone(),
    )
    .await
    .unwrap();

    let mut invalid_request = register_request(request.peer_info.clone());
    invalid_request.peer_info = PeerInfo::new(
        owner,
        WorkerAddress::from_encoded(b"invalid-velo-address".to_vec()),
    );
    let rejected = reqwest::Client::new()
        .post(control_url(&server, paths::INSTANCES))
        .header(MUTATION_CREDENTIAL_HEADER, credential.to_header_value())
        .json(&invalid_request)
        .send()
        .await
        .unwrap();
    assert_eq!(rejected.status(), 500);

    let restored_incarnation = registry.current_incarnation(owner).unwrap();
    assert_ne!(restored_incarnation, initial_incarnation);
    assert_eq!(
        registry
            .discover_by_instance_id(owner)
            .await
            .unwrap()
            .address_checksum(),
        request.peer_info.address_checksum(),
        "rollback must restore the prior peer metadata"
    );
    let heartbeat = reqwest::Client::new()
        .post(control_url(&server, &instance_heartbeat(owner)))
        .header(MUTATION_CREDENTIAL_HEADER, credential.to_header_value())
        .send()
        .await
        .unwrap();
    assert_eq!(heartbeat.status(), 200, "prior authority was not restored");

    let BundleQueryOutcome::Hit(hit) = query_bundle(client_velo.messenger(), hub_id, &live)
        .await
        .unwrap()
    else {
        panic!("failed re-registration discarded the prior live advertisement");
    };
    assert_eq!(hit.advertisement.generation, 9);
    assert_eq!(
        hit.advertisement.registration_epoch,
        Some(registration_epoch),
        "rollback must restore the exact prior registration epoch"
    );
    assert!(
        publish_bundle(client_velo.messenger(), hub_id, credential, retired)
            .await
            .is_err(),
        "failed re-registration discarded the prior generation retirement"
    );
    let BundleQueryOutcome::Hit(hit) = query_bundle(client_velo.messenger(), hub_id, &live)
        .await
        .unwrap()
    else {
        panic!("delayed retired publication replaced the restored live advertisement");
    };
    assert_eq!(hit.advertisement.generation, 9);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn successful_reregistration_starts_a_fresh_bundle_generation_domain() {
    const BLOCK_SIZE: usize = 4;

    let registry = Arc::new(CustomRegistry::default());
    let indexer = Arc::new(IndexerManager::new(64, BLOCK_SIZE, None, None).unwrap());
    let transport = new_velo_transport();
    let server = start_custom_server(
        Arc::clone(&registry),
        Some(Arc::clone(&indexer) as Arc<dyn FeatureManager>),
        Some(Arc::clone(&transport) as Arc<dyn Transport>),
    )
    .await;
    let client_velo = new_velo().await;
    let request = indexer_register_request(client_velo.peer_info(), BLOCK_SIZE);
    let initial = register(&server, &request, None).await;
    let old_credential = initial.mutation_credential.unwrap();
    let old_epoch = initial.registration_epoch.unwrap();
    let hub_id = initial.hub_instance_id.unwrap();
    let hub_peer = registry.discover_by_instance_id(hub_id).await.unwrap();
    client_velo.register_peer(hub_peer).unwrap();
    let owner = client_velo.instance_id();
    let initial_incarnation = registry.current_incarnation(owner).unwrap();
    let old_advertisement = bundle_advertisement(owner, old_epoch, 9);

    publish_bundle(
        client_velo.messenger(),
        hub_id,
        old_credential.clone(),
        old_advertisement.clone(),
    )
    .await
    .unwrap();
    let BundleQueryOutcome::Hit(old_hit) =
        query_bundle(client_velo.messenger(), hub_id, &old_advertisement)
            .await
            .unwrap()
    else {
        panic!("initial advertisement must be query-visible");
    };
    assert_eq!(old_hit.advertisement.registration_epoch, Some(old_epoch));

    let mut replacement_request = request.clone();
    replacement_request.peer_info = replacement_peer(&request.peer_info);
    assert_ne!(
        replacement_request.peer_info.address_checksum(),
        request.peer_info.address_checksum(),
        "the regression must exercise a replacement peer as well as a new incarnation"
    );
    let replacement = register(&server, &replacement_request, Some(&old_credential)).await;
    let replacement_credential = replacement.mutation_credential.unwrap();
    let replacement_epoch = replacement.registration_epoch.unwrap();
    assert_ne!(old_epoch, replacement_epoch);
    assert_ne!(
        registry.current_incarnation(owner).unwrap(),
        initial_incarnation,
        "same-ID re-registration must create a fresh registry incarnation"
    );
    assert_eq!(
        query_bundle(client_velo.messenger(), hub_id, &old_advertisement)
            .await
            .unwrap(),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound),
        "the replacement must not inherit the prior incarnation's advertisement"
    );

    let fresh_advertisement = bundle_advertisement(owner, replacement_epoch, 1);
    publish_bundle(
        client_velo.messenger(),
        hub_id,
        replacement_credential,
        fresh_advertisement.clone(),
    )
    .await
    .expect("the replacement incarnation must be allowed to restart at generation one");
    let BundleQueryOutcome::Hit(hit) =
        query_bundle(client_velo.messenger(), hub_id, &fresh_advertisement)
            .await
            .unwrap()
    else {
        panic!("fresh replacement advertisement must be query-visible");
    };
    assert_eq!(hit.advertisement.key, fresh_advertisement.key);
    assert_eq!(hit.advertisement.owner, fresh_advertisement.owner);
    assert_eq!(hit.advertisement.generation, 1);
    assert_eq!(
        hit.advertisement.registration_epoch,
        Some(replacement_epoch)
    );
}

async fn start_custom_server(
    registry: Arc<CustomRegistry>,
    manager: Option<Arc<dyn FeatureManager>>,
    transport: Option<Arc<dyn Transport>>,
) -> HubServer {
    let mut builder = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .registry(registry as Arc<dyn PeerRegistry>);
    if let Some(manager) = manager {
        builder = builder.add_feature_manager(manager);
    }
    if let Some(transport) = transport {
        builder = builder.add_transport(transport);
    }
    builder.serve().await.unwrap()
}

async fn register(
    server: &HubServer,
    request: &RegisterRequest,
    credential: Option<&kvbm_hub::protocol::MutationCredential>,
) -> RegisterResponse {
    let mut request_builder = reqwest::Client::new()
        .post(control_url(server, paths::INSTANCES))
        .json(request);
    if let Some(credential) = credential {
        request_builder =
            request_builder.header(MUTATION_CREDENTIAL_HEADER, credential.to_header_value());
    }
    request_builder.send().await.unwrap().json().await.unwrap()
}

fn register_request(peer_info: PeerInfo) -> RegisterRequest {
    RegisterRequest {
        peer_info,
        features: Vec::new(),
        runtime: None,
    }
}

fn indexer_register_request(peer_info: PeerInfo, block_size: usize) -> RegisterRequest {
    RegisterRequest {
        peer_info,
        features: vec![Feature::Indexer(IndexerFeatureConfig::default())],
        runtime: Some(RuntimeConfigSummary {
            block_size: Some(block_size),
            block_layout: None,
        }),
    }
}

fn bundle_advertisement(
    owner: InstanceId,
    registration_epoch: RegistrationEpoch,
    generation: u64,
) -> BundleAdvertisementRecord {
    let resource = LogicalResourceId(1);
    let hashes = vec![SequenceHash::root(1), SequenceHash::root(1).extend(2)];
    let manifest = CacheManifestId::from_bytes([71; 32]);
    BundleAdvertisementRecord {
        key: BundleKey::from_parts(manifest, hashes[1], 8).unwrap(),
        generation,
        owner,
        registration_epoch: Some(registration_epoch),
        requirements: vec![
            ResourceRequirement::new(resource, ResourceRole::PrefixHistory, 4).unwrap(),
        ],
        lineages: vec![BundleResourceLineage::new(resource, hashes).unwrap()],
        expires_at_unix_ms: u64::MAX,
        placements: Vec::new(),
        stage_cost_hint_us: None,
        advertised_at_unix_ms: None,
    }
}

async fn publish_bundle(
    messenger: &Arc<velo::Messenger>,
    hub_id: InstanceId,
    credential: MutationCredential,
    advertisement: BundleAdvertisementRecord,
) -> Result<()> {
    messenger
        .typed_unary::<()>(kvbm_hub::features::indexer::BUNDLE_PUBLISH_HANDLER)?
        .payload(&BundlePublishRequest {
            credential,
            advertisement,
        })?
        .instance(hub_id)
        .send()
        .await
}

async fn invalidate_bundle(
    messenger: &Arc<velo::Messenger>,
    hub_id: InstanceId,
    credential: MutationCredential,
    advertisement: &BundleAdvertisementRecord,
) -> Result<bool> {
    messenger
        .typed_unary::<bool>(kvbm_hub::features::indexer::BUNDLE_INVALIDATE_HANDLER)?
        .payload(&BundleInvalidateRequest {
            credential,
            key: advertisement.key,
            generation: advertisement.generation,
            owner: advertisement.owner,
            retain_until_unix_ms: u64::MAX,
        })?
        .instance(hub_id)
        .send()
        .await
}

async fn query_bundle(
    messenger: &Arc<velo::Messenger>,
    hub_id: InstanceId,
    advertisement: &BundleAdvertisementRecord,
) -> Result<BundleQueryOutcome> {
    messenger
        .typed_unary::<BundleQueryOutcome>(kvbm_hub::features::indexer::BUNDLE_QUERY_HANDLER)?
        .payload(&BundleQueryRequest {
            manifest: advertisement.key.manifest(),
            requirements: advertisement.requirements.clone(),
            candidates: vec![advertisement.key],
            now_unix_ms: 0,
        })?
        .instance(hub_id)
        .send()
        .await
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
        .unwrap()
}

fn make_peer() -> PeerInfo {
    PeerInfo::new(
        InstanceId::new_v4(),
        WorkerAddress::from_encoded(b"registration-lifecycle-test".to_vec()),
    )
}

fn replacement_peer(peer: &PeerInfo) -> PeerInfo {
    let mut address: HashMap<String, Vec<u8>> =
        rmp_serde::from_slice(peer.worker_address().as_bytes()).unwrap();
    address.insert(
        "registration-lifecycle-test".to_string(),
        b"replacement".to_vec(),
    );
    PeerInfo::new(
        peer.instance_id(),
        WorkerAddress::from_encoded(rmp_serde::to_vec(&address).unwrap()),
    )
}

fn control_url(server: &HubServer, path: &str) -> String {
    format!("http://{}{}", server.control_addr(), path)
}
