// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::disallowed_macros)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, anyhow};
use futures::future::BoxFuture;
use kvbm_common::{BlockId, LogicalLayoutHandle, LogicalResourceId, SequenceHash};
use kvbm_logical::manager::{FrequencyTrackingCapacity, InactiveBackendConfig};
use kvbm_logical::{BlockManager, BlockManagerSet, BlockRegistry};
use kvbm_observability::KvbmObservability;
use kvbm_physical::TransferOptions;
use kvbm_physical::transfer::TransferCompleteNotification;
use kvbm_protocols::cache_manifest::{
    BundleKey, BundleResourceLineage, BundleResourceLineageError, CacheManifest, ModelIdentity,
    ResourceRequirement, ResourceRole,
};
use kvbm_protocols::connector::{
    BundleOffloadPlan, BundleOnboardPlan, CacheScope, EngineWorkerSink, FenceToken,
    FindBlocksOutcome, FindBlocksRequest, LeaderEngine, LoadOutcome, OffloadMode, RequestId,
    ResourceDestination, ResourceOffload, ResourceOnboard, SaveOutcome,
};

use super::super::local::LocalConnectorEngine;
use super::super::offload::{OffloadSubmit, OffloadTransfer};
use super::{BUNDLE_DIRECTORY_TTL_MS, BundleAdmissionConfig, BundleCatalogError};
use crate::leader::{InstanceLeader, RemoteBlockDiscovery, RemoteCandidates};
use crate::object::ObjectBlockOps;
use crate::offload::{ExternalBlock, TransferStatus};
use crate::p2p::StagedPull;
use crate::remote::search::bundle::{
    BundleAdvertisement, BundleDirectoryError, BundleDiscoveryOutcome, BundleDiscoveryQuery,
    BundleInvalidation, BundleMissReason, BundlePullTarget, StagedBundle,
};
use crate::testing::{managers::TestManagerBuilder, messenger::create_messenger_tcp};
use crate::tiering::policy::{
    ResourceComponentBytes, ResourceLineage, ResourcePolicies, ResourcePolicy,
};
use crate::worker::group::ParallelWorkers;
use crate::worker::{
    ConnectRemoteResponse, ImportMetadataResponse, InstanceId, RemoteDescriptor, SerializedLayout,
    SerializedLayoutResponse, Worker, WorkerTransfers,
};
use crate::{G1, G2};

/// One remote pull's announced residency: `resource -> lineage in position
/// order`, the payload shape of a `PulledBundleReadyObserver`.
type PulledLineages = BTreeMap<LogicalResourceId, Vec<SequenceHash>>;

const BLOCK_SIZE: usize = 4;
const RESOURCES: [LogicalResourceId; 3] = [
    LogicalResourceId(10),
    LogicalResourceId(11),
    LogicalResourceId(12),
];

#[derive(Default)]
struct RecordingBundleDirectory {
    advertisements: Mutex<Vec<BundleAdvertisement>>,
    invalidations: Mutex<Vec<BundleInvalidation>>,
}

struct ManualPublicationClock {
    now_unix_ms: AtomicU64,
}

impl ManualPublicationClock {
    fn new(now_unix_ms: u64) -> Self {
        Self {
            now_unix_ms: AtomicU64::new(now_unix_ms),
        }
    }

    fn now_unix_ms(&self) -> u64 {
        self.now_unix_ms.load(Ordering::Acquire)
    }

    fn advance(&self, duration: Duration) {
        self.now_unix_ms.fetch_add(
            u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
            Ordering::AcqRel,
        );
    }
}

#[derive(Default)]
struct ExpiringBundleDirectory {
    current: Mutex<HashMap<BundleKey, BundleAdvertisement>>,
    advertisements: Mutex<Vec<BundleAdvertisement>>,
    advertisement_attempts: AtomicUsize,
    advertisement_failures_remaining: AtomicUsize,
}

impl ExpiringBundleDirectory {
    fn fail_next_advertisements(&self, count: usize) {
        self.advertisement_failures_remaining
            .store(count, Ordering::Release);
    }

    fn advertisement_attempts(&self) -> usize {
        self.advertisement_attempts.load(Ordering::Acquire)
    }

    fn advertisement_count(&self) -> usize {
        self.advertisements.lock().unwrap().len()
    }

    fn advertisement_count_for_generation(&self, generation: u64) -> usize {
        self.advertisements
            .lock()
            .unwrap()
            .iter()
            .filter(|advertisement| advertisement.generation() == generation)
            .count()
    }

    fn visible_generation(&self, key: BundleKey, now_unix_ms: u64) -> Option<u64> {
        self.current
            .lock()
            .unwrap()
            .get(&key)
            .filter(|advertisement| advertisement.expires_at_unix_ms() > now_unix_ms)
            .map(BundleAdvertisement::generation)
    }
}

impl RemoteBlockDiscovery for ExpiringBundleDirectory {
    fn discover(
        &self,
        _hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        Box::pin(async { Ok(None) })
    }

    fn advertise_bundle(
        &self,
        advertisement: BundleAdvertisement,
    ) -> BoxFuture<'static, Result<()>> {
        self.advertisement_attempts.fetch_add(1, Ordering::AcqRel);
        if self
            .advertisement_failures_remaining
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            return Box::pin(async { Err(anyhow!("injected bundle directory failure")) });
        }
        let mut current = self.current.lock().unwrap();
        if current
            .get(&advertisement.key())
            .is_none_or(|installed| installed.generation() <= advertisement.generation())
        {
            current.insert(advertisement.key(), advertisement.clone());
        }
        self.advertisements.lock().unwrap().push(advertisement);
        Box::pin(async { Ok(()) })
    }

    fn invalidate_bundle(
        &self,
        invalidation: BundleInvalidation,
    ) -> BoxFuture<'static, Result<()>> {
        let mut current = self.current.lock().unwrap();
        if current
            .get(&invalidation.key)
            .is_some_and(|installed| installed.generation() == invalidation.generation)
        {
            current.remove(&invalidation.key);
        }
        Box::pin(async { Ok(()) })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirectoryEvent {
    Advertise(BundleKey, u64),
    Invalidate(BundleKey, u64),
}

struct BlockingDirectoryState {
    block_next_advertisement: AtomicBool,
    advertisement_started: tokio::sync::Semaphore,
    release_advertisement: tokio::sync::Semaphore,
    active: Mutex<HashSet<(BundleKey, u64)>>,
    events: Mutex<Vec<DirectoryEvent>>,
}

impl Default for BlockingDirectoryState {
    fn default() -> Self {
        Self {
            block_next_advertisement: AtomicBool::new(false),
            advertisement_started: tokio::sync::Semaphore::new(0),
            release_advertisement: tokio::sync::Semaphore::new(0),
            active: Mutex::new(HashSet::new()),
            events: Mutex::new(Vec::new()),
        }
    }
}

#[derive(Clone, Default)]
struct BlockingBundleDirectory {
    state: Arc<BlockingDirectoryState>,
}

impl BlockingBundleDirectory {
    fn block_next_advertisement(&self) {
        self.state
            .block_next_advertisement
            .store(true, Ordering::Release);
    }

    async fn wait_until_advertisement_starts(&self) {
        self.state
            .advertisement_started
            .acquire()
            .await
            .expect("advertisement-start semaphore closed")
            .forget();
    }

    fn release_advertisement(&self) {
        self.state.release_advertisement.add_permits(1);
    }

    fn active(&self) -> HashSet<(BundleKey, u64)> {
        self.state.active.lock().unwrap().clone()
    }

    fn events(&self) -> Vec<DirectoryEvent> {
        self.state.events.lock().unwrap().clone()
    }
}

impl RemoteBlockDiscovery for BlockingBundleDirectory {
    fn discover(
        &self,
        _hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        Box::pin(async { Ok(None) })
    }

    fn advertise_bundle(
        &self,
        advertisement: BundleAdvertisement,
    ) -> BoxFuture<'static, Result<()>> {
        let state = Arc::clone(&self.state);
        Box::pin(async move {
            if state.block_next_advertisement.swap(false, Ordering::AcqRel) {
                state.advertisement_started.add_permits(1);
                state
                    .release_advertisement
                    .acquire()
                    .await
                    .expect("advertisement-release semaphore closed")
                    .forget();
            }
            let identity = (advertisement.key(), advertisement.generation());
            state.active.lock().unwrap().insert(identity);
            state
                .events
                .lock()
                .unwrap()
                .push(DirectoryEvent::Advertise(identity.0, identity.1));
            Ok(())
        })
    }

    fn invalidate_bundle(
        &self,
        invalidation: BundleInvalidation,
    ) -> BoxFuture<'static, Result<()>> {
        let state = Arc::clone(&self.state);
        Box::pin(async move {
            let identity = (invalidation.key, invalidation.generation);
            state.active.lock().unwrap().remove(&identity);
            state
                .events
                .lock()
                .unwrap()
                .push(DirectoryEvent::Invalidate(identity.0, identity.1));
            Ok(())
        })
    }
}

impl RemoteBlockDiscovery for RecordingBundleDirectory {
    fn discover(
        &self,
        _hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        Box::pin(async { Ok(None) })
    }

    fn discover_bundle(
        &self,
        _query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        Box::pin(async { Ok(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound)) })
    }

    fn advertise_bundle(
        &self,
        advertisement: BundleAdvertisement,
    ) -> BoxFuture<'static, Result<()>> {
        self.advertisements.lock().unwrap().push(advertisement);
        Box::pin(async { Ok(()) })
    }

    fn invalidate_bundle(
        &self,
        invalidation: BundleInvalidation,
    ) -> BoxFuture<'static, Result<()>> {
        self.invalidations.lock().unwrap().push(invalidation);
        Box::pin(async { Ok(()) })
    }
}

struct RegisteringOffloadSubmit {
    managers: BTreeMap<LogicalResourceId, Arc<BlockManager<G2>>>,
}

impl OffloadSubmit for RegisteringOffloadSubmit {
    fn supports_resource(&self, resource: LogicalResourceId) -> bool {
        self.managers.contains_key(&resource)
    }

    fn submit_g1_to_g2(
        &self,
        resource: Option<LogicalResourceId>,
        blocks: Vec<ExternalBlock<G1>>,
        _precondition: Option<velo::EventHandle>,
    ) -> Result<Box<dyn OffloadTransfer>> {
        let resource = resource.ok_or_else(|| anyhow!("bundle child requires a resource"))?;
        let manager = self
            .managers
            .get(&resource)
            .ok_or_else(|| anyhow!("no mock manager for {resource:?}"))?;
        let allocated = manager
            .allocate_blocks(blocks.len())
            .ok_or_else(|| anyhow!("mock G2 manager is full"))?;
        let pins = allocated
            .into_iter()
            .zip(blocks)
            .map(|(block, source)| -> Result<_> {
                let complete = block.stage(source.sequence_hash, manager.block_size())?;
                Ok(manager.register_block(complete))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(pins);
        Ok(Box::new(CompletedTransfer))
    }
}

struct CompletedTransfer;

impl OffloadTransfer for CompletedTransfer {
    fn status(&self) -> TransferStatus {
        TransferStatus::Complete
    }

    fn completed_blocks(&self) -> Vec<BlockId> {
        Vec::new()
    }

    fn failed_blocks(&self) -> Vec<BlockId> {
        Vec::new()
    }

    fn wait_terminal(&self) -> BoxFuture<'static, ()> {
        Box::pin(async {})
    }
}

struct FailedTransfer {
    failed_blocks: Vec<BlockId>,
}

impl OffloadTransfer for FailedTransfer {
    fn status(&self) -> TransferStatus {
        TransferStatus::Failed
    }

    fn completed_blocks(&self) -> Vec<BlockId> {
        Vec::new()
    }

    fn failed_blocks(&self) -> Vec<BlockId> {
        self.failed_blocks.clone()
    }

    fn wait_terminal(&self) -> BoxFuture<'static, ()> {
        Box::pin(async {})
    }
}

struct FailingResourceOffloadSubmit {
    delegate: RegisteringOffloadSubmit,
    failing_resource: LogicalResourceId,
}

impl OffloadSubmit for FailingResourceOffloadSubmit {
    fn supports_resource(&self, resource: LogicalResourceId) -> bool {
        self.delegate.supports_resource(resource)
    }

    fn submit_g1_to_g2(
        &self,
        resource: Option<LogicalResourceId>,
        blocks: Vec<ExternalBlock<G1>>,
        precondition: Option<velo::EventHandle>,
    ) -> Result<Box<dyn OffloadTransfer>> {
        if resource == Some(self.failing_resource) {
            return Ok(Box::new(FailedTransfer {
                failed_blocks: blocks.iter().map(|block| block.block_id).collect(),
            }));
        }
        self.delegate
            .submit_g1_to_g2(resource, blocks, precondition)
    }
}

struct CompletedParallelWorkers;

impl WorkerTransfers for CompletedParallelWorkers {
    fn execute_local_transfer(
        &self,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        Ok(TransferCompleteNotification::completed())
    }

    fn execute_local_transfer_for_resource(
        &self,
        _resource: LogicalResourceId,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        Ok(TransferCompleteNotification::completed())
    }

    fn execute_remote_onboard(
        &self,
        _src: RemoteDescriptor,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote onboard is outside this local transaction test")
    }

    fn execute_remote_offload(
        &self,
        _src: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst: RemoteDescriptor,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote offload is outside this local transaction test")
    }

    fn connect_remote(
        &self,
        _instance_id: InstanceId,
        _metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        Ok(ConnectRemoteResponse::ready())
    }

    fn has_remote_metadata(&self, _instance_id: InstanceId) -> bool {
        false
    }

    fn execute_remote_onboard_for_instance(
        &self,
        _instance_id: InstanceId,
        _remote_logical_type: LogicalLayoutHandle,
        _src_block_ids: Vec<BlockId>,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote instance onboard is outside this local transaction test")
    }
}

impl ObjectBlockOps for CompletedParallelWorkers {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        Box::pin(async move { keys.into_iter().map(|key| (key, None)).collect() })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }
}

impl ParallelWorkers for CompletedParallelWorkers {
    fn export_metadata(&self) -> Result<Vec<SerializedLayoutResponse>> {
        Ok(Vec::new())
    }

    fn import_metadata(
        &self,
        _metadata: Vec<SerializedLayout>,
    ) -> Result<Vec<ImportMetadataResponse>> {
        Ok(Vec::new())
    }

    fn worker_count(&self) -> usize {
        0
    }

    fn workers(&self) -> &[Arc<dyn Worker>] {
        &[]
    }
}

struct HangingParallelWorkers {
    events: velo::EventManager,
    resource: LogicalResourceId,
    started: tokio::sync::Semaphore,
    release: Mutex<Option<velo::Event>>,
    panic_resource: Mutex<Option<LogicalResourceId>>,
    dispatches: Mutex<Vec<LogicalResourceId>>,
    completed: CompletedParallelWorkers,
}

impl HangingParallelWorkers {
    fn new(resource: LogicalResourceId) -> Arc<Self> {
        Arc::new(Self {
            events: velo::EventManager::local(),
            resource,
            started: tokio::sync::Semaphore::new(0),
            release: Mutex::new(None),
            panic_resource: Mutex::new(None),
            dispatches: Mutex::new(Vec::new()),
            completed: CompletedParallelWorkers,
        })
    }

    async fn wait_started(&self) {
        self.started
            .acquire()
            .await
            .expect("hanging transfer semaphore closed")
            .forget();
    }

    fn release(&self) -> Result<()> {
        self.release
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| anyhow!("hanging transfer was not dispatched"))?
            .trigger()?;
        Ok(())
    }

    fn dispatches(&self) -> Vec<LogicalResourceId> {
        self.dispatches.lock().unwrap().clone()
    }

    fn panic_on_dispatch(&self, resource: LogicalResourceId) {
        *self.panic_resource.lock().unwrap() = Some(resource);
    }
}

impl WorkerTransfers for HangingParallelWorkers {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.completed
            .execute_local_transfer(src, dst, src_block_ids, dst_block_ids, options)
    }

    fn execute_local_transfer_for_resource(
        &self,
        resource: LogicalResourceId,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.dispatches.lock().unwrap().push(resource);
        if *self.panic_resource.lock().unwrap() == Some(resource) {
            panic!("injected physical bundle task failure for {resource:?}");
        }
        if resource != self.resource {
            return self.completed.execute_local_transfer_for_resource(
                resource,
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                options,
            );
        }
        let event = self.events.new_event()?;
        let awaiter = event.awaiter()?;
        let mut release = self.release.lock().unwrap();
        anyhow::ensure!(release.is_none(), "hanging transfer dispatched twice");
        *release = Some(event);
        self.started.add_permits(1);
        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.completed
            .execute_remote_onboard(src, dst, dst_block_ids, options)
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst: RemoteDescriptor,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.completed
            .execute_remote_offload(src, src_block_ids, dst, options)
    }

    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        self.completed.connect_remote(instance_id, metadata)
    }

    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool {
        self.completed.has_remote_metadata(instance_id)
    }

    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.completed.execute_remote_onboard_for_instance(
            instance_id,
            remote_logical_type,
            src_block_ids,
            dst,
            dst_block_ids,
            options,
        )
    }
}

impl ObjectBlockOps for HangingParallelWorkers {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        self.completed.has_blocks(keys)
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        self.completed.put_blocks(keys, layout, block_ids)
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        self.completed.get_blocks(keys, layout, block_ids)
    }
}

impl ParallelWorkers for HangingParallelWorkers {
    fn export_metadata(&self) -> Result<Vec<SerializedLayoutResponse>> {
        self.completed.export_metadata()
    }

    fn import_metadata(
        &self,
        metadata: Vec<SerializedLayout>,
    ) -> Result<Vec<ImportMetadataResponse>> {
        self.completed.import_metadata(metadata)
    }

    fn worker_count(&self) -> usize {
        1
    }

    fn workers(&self) -> &[Arc<dyn Worker>] {
        &[]
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_onboard_serializes_collective_admission_across_requests() -> Result<()> {
    let identity = manifest()?.identity();
    let hanging = HangingParallelWorkers::new(RESOURCES[0]);
    let workers: Arc<dyn ParallelWorkers> = hanging.clone();
    let (leader, managers) =
        build_resource_test_leader_for_capacity_with_workers(&identity, 4, workers).await?;
    let engine = LocalConnectorEngine::new(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
    );
    let plan = commit_one_block_bundle(&engine, &identity, &managers)?;

    let first = engine
        .clone()
        .onboard_resources(&"ordered-resource-onboard-a".into(), plan.clone())?;
    hanging.wait_started().await;
    let second = engine
        .clone()
        .onboard_resources(&"ordered-resource-onboard-b".into(), plan)?;
    tokio::task::yield_now().await;
    assert_eq!(
        hanging.dispatches(),
        vec![RESOURCES[0]],
        "another request must not enter the shared collective while the first is draining"
    );

    hanging.release()?;
    hanging.wait_started().await;
    assert_eq!(
        hanging.dispatches(),
        [RESOURCES.as_slice(), &[RESOURCES[0]]].concat(),
        "the second request may start only after every first-request resource drains"
    );
    hanging.release()?;
    wait_until(|| first.is_complete() && second.is_complete()).await;
    assert_eq!(first.outcome(), Some(LoadOutcome::Done));
    assert_eq!(second.outcome(), Some(LoadOutcome::Done));
    assert_eq!(
        hanging.dispatches(),
        [RESOURCES.as_slice(), RESOURCES.as_slice()].concat()
    );
    Ok(())
}

#[derive(Default)]
struct RecordingLoadSink {
    loads: Mutex<Vec<(RequestId, LoadOutcome)>>,
    saves: Mutex<Vec<(RequestId, SaveOutcome)>>,
    fences: Mutex<Vec<FenceToken>>,
}

impl RecordingLoadSink {
    fn loads(&self) -> Vec<(RequestId, LoadOutcome)> {
        self.loads.lock().unwrap().clone()
    }

    fn fences(&self) -> Vec<FenceToken> {
        self.fences.lock().unwrap().clone()
    }

    fn saves(&self) -> Vec<(RequestId, SaveOutcome)> {
        self.saves.lock().unwrap().clone()
    }
}

impl EngineWorkerSink for RecordingLoadSink {
    fn mark_load_finished(&self, request: &RequestId, outcome: LoadOutcome) {
        self.loads.lock().unwrap().push((request.clone(), outcome));
    }

    fn mark_save_finished(&self, request: &RequestId, outcome: SaveOutcome) {
        self.saves.lock().unwrap().push((request.clone(), outcome));
    }

    fn mark_fence_complete(&self, token: FenceToken) {
        self.fences.lock().unwrap().push(token);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_onboard_success_emits_once_after_physical_completion() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for(&identity).await?;
    let sink = Arc::new(RecordingLoadSink::default());
    let engine = LocalConnectorEngine::new(Arc::new(leader), sink.clone(), BLOCK_SIZE, false);
    let plan = commit_one_block_bundle(&engine, &identity, &managers)?;
    let request = "bundle-success-two-phase".to_owned();
    let mut onboard = engine.clone().onboard_resources(&request, plan)?;
    let physical_drain = onboard
        .take_physical_drain_fence()
        .expect("bundle onboard must expose its physical drain fence");

    wait_until(|| sink.loads().len() == 1).await;
    assert!(
        physical_drain.is_complete(),
        "worker visibility must follow the physical-complete transition"
    );
    assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
    assert_eq!(sink.loads(), vec![(request, LoadOutcome::Done)]);
    tokio::task::yield_now().await;
    assert_eq!(sink.loads().len(), 1, "success terminal is exactly once");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_onboard_watchdog_fails_promptly_but_quarantines_until_drain() -> Result<()> {
    let (engine, hanging, sink, plan) = build_hanging_onboard(Duration::from_millis(20)).await?;
    let request = "bundle-watchdog".to_owned();
    let mut onboard = engine.clone().onboard_resources(&request, plan)?;
    let action_id = *onboard.id();
    let physical_drain = onboard
        .take_physical_drain_fence()
        .expect("bundle onboard must expose its physical drain fence");
    hanging.wait_started().await;

    tokio::time::timeout(Duration::from_millis(250), async {
        while !onboard.is_complete() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("bundle onboard watchdog must leave Pending");
    assert!(matches!(
        onboard.outcome(),
        Some(LoadOutcome::FailedPartial { .. })
    ));
    assert!(
        engine
            .actions
            .get(&action_id)
            .is_some_and(|record| record.physical_pending),
        "timed-out transfer must retain its action quarantine"
    );
    assert!(
        !physical_drain.is_complete(),
        "logical timeout must not release the physical drain fence"
    );
    assert!(engine.inflight.lock().unwrap().overlaps(&[hash(1)]));
    assert!(
        sink.loads().is_empty(),
        "the logical handle may fail promptly, but worker-visible failed ids must stay \
         quarantined until the destination transfer physically drains"
    );

    drop(onboard);
    assert!(
        engine.actions.contains_key(&action_id),
        "handle release must not drop a live physical quarantine"
    );
    hanging.release()?;
    wait_until(|| physical_drain.is_complete()).await;
    wait_until(|| sink.loads().len() == 1).await;
    wait_until(|| !engine.actions.contains_key(&action_id)).await;
    assert!(!engine.inflight.lock().unwrap().overlaps(&[hash(1)]));
    assert!(matches!(
        sink.loads()[0].1,
        LoadOutcome::FailedPartial { .. }
    ));
    assert_eq!(
        sink.loads().len(),
        1,
        "drained quarantine must not publish a late success"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_onboard_handle_drop_cancels_promptly_without_late_terminal_mutation() -> Result<()>
{
    let (engine, hanging, sink, plan) = build_hanging_onboard(Duration::from_secs(5)).await?;
    let request = "bundle-drop-cancel".to_owned();
    let mut onboard = engine.clone().onboard_resources(&request, plan)?;
    let action_id = *onboard.id();
    let physical_drain = onboard
        .take_physical_drain_fence()
        .expect("bundle onboard must expose its physical drain fence");
    hanging.wait_started().await;
    drop(onboard);

    wait_until(|| {
        engine.actions.get(&action_id).is_some_and(|record| {
            record.cell.upgrade().is_none()
                && record.physical_pending
                && record
                    .cancel
                    .as_ref()
                    .is_some_and(|cancel| cancel.is_cancelled())
        })
    })
    .await;
    assert!(
        engine
            .actions
            .get(&action_id)
            .is_some_and(|record| record.physical_pending)
    );
    assert!(
        sink.loads().is_empty(),
        "dropping the logical handle must not expose destinations still receiving DMA"
    );
    assert!(!physical_drain.is_complete());

    hanging.release()?;
    wait_until(|| physical_drain.is_complete()).await;
    wait_until(|| sink.loads().len() == 1).await;
    wait_until(|| !engine.actions.contains_key(&action_id)).await;
    assert_eq!(
        sink.loads().len(),
        1,
        "drain must not emit a second terminal"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_onboard_evict_after_logical_timeout_suppresses_deferred_worker_terminal()
-> Result<()> {
    let (engine, hanging, sink, plan) = build_hanging_onboard(Duration::from_millis(20)).await?;
    let request = "bundle-timeout-then-evict".to_owned();
    let mut onboard = engine.clone().onboard_resources(&request, plan)?;
    let physical_drain = onboard
        .take_physical_drain_fence()
        .expect("bundle onboard must expose its physical drain fence");
    hanging.wait_started().await;
    wait_until(|| onboard.is_complete()).await;
    assert!(matches!(
        onboard.outcome(),
        Some(LoadOutcome::FailedPartial { .. })
    ));
    assert!(sink.loads().is_empty());

    let eviction = engine.evict(&request);
    let fence = eviction
        .handle
        .expect("physical pending must remain fenceable after the logical timeout");
    assert!(!fence.is_complete());

    hanging.release()?;
    wait_until(|| physical_drain.is_complete()).await;
    wait_until(|| fence.is_complete()).await;
    assert_eq!(sink.fences().len(), 1);
    assert!(
        sink.loads().is_empty(),
        "the post-timeout eviction fence owns the worker terminal"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_onboard_request_drain_after_logical_timeout_suppresses_deferred_load() -> Result<()>
{
    let (engine, hanging, sink, plan) = build_hanging_onboard(Duration::from_millis(20)).await?;
    let request = "bundle-timeout-then-request-drain".to_owned();
    let mut onboard = engine.clone().onboard_resources(&request, plan)?;
    let physical_drain = onboard
        .take_physical_drain_fence()
        .expect("bundle onboard must expose its physical drain fence");
    hanging.wait_started().await;
    wait_until(|| onboard.is_complete()).await;
    assert!(sink.loads().is_empty());

    engine.offload_drains.insert(request.clone(), ());
    engine
        .take_offload_drain(&request)
        .expect("test request drain registered")
        .commit();
    assert!(sink.saves().is_empty());

    hanging.release()?;
    wait_until(|| physical_drain.is_complete()).await;
    wait_until(|| sink.saves().len() == 1).await;
    assert!(
        sink.loads().is_empty(),
        "the post-timeout request drain folds the load terminal into finished_sending"
    );
    assert_eq!(sink.saves(), vec![(request, SaveOutcome::Done)]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_onboard_join_error_after_logical_timeout_stays_fail_closed() -> Result<()> {
    let (engine, hanging, sink, plan) = build_hanging_onboard(Duration::from_millis(20)).await?;
    let request = "bundle-timeout-then-join-error".to_owned();
    let mut onboard = engine.clone().onboard_resources(&request, plan)?;
    let action_id = *onboard.id();
    let physical_drain = onboard
        .take_physical_drain_fence()
        .expect("bundle onboard must expose its physical drain fence");
    hanging.wait_started().await;
    wait_until(|| onboard.is_complete()).await;

    // Panic the physical task after its hanging notification is released. A
    // JoinError cannot prove that an already-submitted device operation has
    // drained, so the destination quarantine must remain closed indefinitely.
    hanging.panic_on_dispatch(RESOURCES[2]);
    hanging.release()?;
    wait_until(|| hanging.dispatches().contains(&RESOURCES[2])).await;
    tokio::task::yield_now().await;

    assert!(matches!(
        onboard.outcome(),
        Some(LoadOutcome::FailedPartial { .. })
    ));
    assert!(
        engine
            .actions
            .get(&action_id)
            .is_some_and(|record| record.physical_pending),
        "an unprovable physical terminal must retain the destination quarantine"
    );
    assert!(!physical_drain.is_complete());
    assert!(
        sink.loads().is_empty(),
        "a physical JoinError must never expose failed ids for reuse"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_onboard_evict_cancels_promptly_and_holds_fence_until_drain() -> Result<()> {
    let (engine, hanging, sink, plan) = build_hanging_onboard(Duration::from_secs(5)).await?;
    let request = "bundle-evict-cancel".to_owned();
    let mut onboard = engine.clone().onboard_resources(&request, plan)?;
    let action_id = *onboard.id();
    let physical_drain = onboard
        .take_physical_drain_fence()
        .expect("bundle onboard must expose its physical drain fence");
    hanging.wait_started().await;
    let eviction = engine.evict(&request);
    let fence = eviction
        .handle
        .expect("eviction must fence the hanging onboard");

    tokio::time::timeout(Duration::from_millis(250), async {
        while !onboard.is_complete() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("eviction must cancel the bundle driver promptly");
    assert!(matches!(
        onboard.outcome(),
        Some(LoadOutcome::FailedPartial { .. })
    ));
    assert!(
        !fence.is_complete(),
        "eviction fence cannot beat physical drain"
    );
    assert!(
        sink.loads().is_empty(),
        "fenced cancel must suppress load publish"
    );
    assert!(
        engine
            .actions
            .get(&action_id)
            .is_some_and(|record| record.physical_pending)
    );

    drop(onboard);
    hanging.release()?;
    wait_until(|| physical_drain.is_complete()).await;
    wait_until(|| fence.is_complete()).await;
    wait_until(|| !engine.actions.contains_key(&action_id)).await;
    assert_eq!(sink.fences().len(), 1);
    assert!(
        sink.loads().is_empty(),
        "drain must not publish a late load terminal"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_offload_publishes_once_and_onboard_requires_its_exact_lease() -> Result<()> {
    let manifest = manifest()?;
    let identity = manifest.identity();
    let (leader, managers) = build_resource_test_leader().await?;
    let leader = Arc::new(leader);
    let directory = Arc::new(RecordingBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        leader,
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let key = BundleKey::new(&identity, hash(1), BLOCK_SIZE as u64)?;
    let offload = engine.clone().offload_bundle(
        &"bundle-rq".into(),
        BundleOffloadPlan {
            identity: identity.clone(),
            key,
            mode: OffloadMode::Move,
            resources: RESOURCES
                .iter()
                .enumerate()
                .map(|(index, &resource)| ResourceOffload {
                    resource,
                    blocks: vec![(hash(1), 20 + index)],
                })
                .collect(),
        },
    )?;

    assert!(!offload.is_complete());
    assert!(directory.advertisements.lock().unwrap().is_empty());
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&identity, &key)
            .is_none(),
        "buffering cannot publish a partial bundle"
    );
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 0);
    wait_until(|| offload.is_complete()).await;
    assert_eq!(offload.outcome(), Some(SaveOutcome::Done));
    assert_counter(
        engine.leader.observability().unwrap(),
        "kvbm_bundle_txn_total",
        &[("operation", "offload"), ("outcome", "commit")],
        1.0,
    );
    for resource in RESOURCES {
        let resource = resource.0.to_string();
        assert_counter(
            engine.leader.observability().unwrap(),
            "kvbm_bundle_resource_outcome_total",
            &[
                ("operation", "offload_transfer"),
                ("resource", resource.as_str()),
                ("outcome", "complete"),
                ("reason", "complete"),
            ],
            1.0,
        );
        assert_counter(
            engine.leader.observability().unwrap(),
            "kvbm_bundle_resource_planned_bytes_total",
            &[
                ("operation", "offload_transfer"),
                ("resource", resource.as_str()),
            ],
            1024.0,
        );
        assert_counter(
            engine.leader.observability().unwrap(),
            "kvbm_bundle_resource_bytes_total",
            &[
                ("operation", "offload_transfer"),
                ("resource", resource.as_str()),
                ("outcome", "complete"),
            ],
            1024.0,
        );
        assert_histogram_count(
            engine.leader.observability().unwrap(),
            "kvbm_bundle_resource_duration_seconds",
            &[
                ("operation", "offload_transfer"),
                ("resource", resource.as_str()),
                ("outcome", "complete"),
            ],
            1,
        );
        assert_counter(
            engine.leader.observability().unwrap(),
            "kvbm_bundle_transfer_bytes",
            &[
                ("resource", resource.as_str()),
                ("source_tier", "g1"),
                ("destination_tier", "g2"),
            ],
            1024.0,
        );
        assert_histogram_count(
            engine.leader.observability().unwrap(),
            "kvbm_bundle_transfer_seconds",
            &[("resource", resource.as_str()), ("phase", "offload")],
            1,
        );
    }
    wait_until(|| !directory.advertisements.lock().unwrap().is_empty()).await;
    {
        let advertisements = directory.advertisements.lock().unwrap();
        assert_eq!(advertisements.len(), 1);
        assert_eq!(advertisements[0].key(), key);
        assert_eq!(advertisements[0].resources().collect::<Vec<_>>(), RESOURCES);
    }
    assert_eq!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .dependents(RESOURCES[0], hash(1)),
        vec![key],
        "publication installs resource-lineage invalidation before visibility"
    );

    let request = FindBlocksRequest {
        request_id: "bundle-rq".into(),
        cache: CacheScope::Manifest(identity),
        sequence_hashes: Arc::from([hash(1)]),
        num_computed_tokens: 0,
        total_tokens: BLOCK_SIZE + 1,
        transfer_params: None,
        local_prefill_estimate: None,
    };
    let FindBlocksOutcome::Resolved {
        matched_tokens,
        minted: Some(search),
        ..
    } = engine.clone().find_blocks(&request, None)?
    else {
        anyhow::bail!("committed bundle must resolve through manifest search")
    };
    assert_eq!(matched_tokens, BLOCK_SIZE);
    engine.invalidate_resource_blocks(RESOURCES[2], &[hash(1)]);
    wait_until(|| !directory.invalidations.lock().unwrap().is_empty()).await;
    assert_eq!(directory.invalidations.lock().unwrap()[0].key, key);

    let duplicate = engine.clone().onboard_bundle(
        &search,
        vec![
            ResourceDestination {
                resource: RESOURCES[0],
                block_ids: vec![100],
            },
            ResourceDestination {
                resource: RESOURCES[0],
                block_ids: vec![999],
            },
            ResourceDestination {
                resource: RESOURCES[1],
                block_ids: vec![101],
            },
            ResourceDestination {
                resource: RESOURCES[2],
                block_ids: vec![102],
            },
        ],
        matched_tokens,
    );
    assert!(matches!(
        duplicate,
        Err(kvbm_protocols::connector::LeaderEngineError::InvalidBundleTransfer { .. })
    ));

    let onboard = engine.clone().onboard_bundle(
        &search,
        RESOURCES
            .into_iter()
            .enumerate()
            .map(|(index, resource)| ResourceDestination {
                resource,
                block_ids: vec![100 + index],
            })
            .collect(),
        matched_tokens,
    )?;
    wait_until(|| onboard.is_complete()).await;
    assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
    assert!(engine.inflight.lock().unwrap().overlaps(&[hash(1)]));
    drop(onboard);
    assert!(!engine.inflight.lock().unwrap().overlaps(&[hash(1)]));
    Ok(())
}

#[tokio::test]
async fn unadvertisable_lineage_is_rejected_before_catalog_visibility() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader().await?;
    let leader = Arc::new(leader);
    let directory = Arc::new(RecordingBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::new(
        leader,
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
    );
    let key = BundleKey::new(&identity, hash(1), BLOCK_SIZE as u64)?;
    let mut resources = BTreeMap::new();
    let mut lineages = Vec::new();
    for requirement in identity.resources() {
        let resource = requirement.resource();
        let lineage_hash = if resource == RESOURCES[0] {
            resource_hash(77, 7)
        } else {
            key.boundary_hash()
        };
        let manager = &managers[&resource];
        let block = manager
            .allocate_blocks(1)
            .and_then(|mut blocks| blocks.pop())
            .ok_or_else(|| anyhow!("mock G2 manager is full"))?;
        let complete = block.stage(lineage_hash, manager.block_size())?;
        resources.insert(resource, vec![manager.register_block(complete)]);
        lineages.push(ResourceLineage::new(
            resource,
            requirement.role(),
            vec![lineage_hash],
        ));
    }

    let error = engine
        .commit_bundle(identity.clone(), key, 1, resources, lineages)
        .unwrap_err();

    let BundleCatalogError::Advertisement(BundleDirectoryError::ResourceBoundaryMismatch {
        resource,
        boundary_tokens,
    }) = error
    else {
        panic!("unexpected invalid-lineage error: {error}");
    };
    assert_eq!(resource, RESOURCES[0]);
    assert_eq!(boundary_tokens, 4);
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&identity, &key)
            .is_none()
    );
    tokio::task::yield_now().await;
    assert!(directory.advertisements.lock().unwrap().is_empty());
    Ok(())
}

#[tokio::test]
async fn disconnected_chain_cannot_complete_pull_create_or_publish_a_bundle() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader().await?;
    let leader = Arc::new(leader);
    let directory = Arc::new(RecordingBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::new(
        leader,
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
    );
    let announced = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&announced);
    engine.set_pulled_bundle_ready_observer(Arc::new(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
    }));
    let key = BundleKey::new(&identity, hash(2), (2 * BLOCK_SIZE) as u64)?;
    let child_of_another_parent = SequenceHash::root(77).extend(2);
    let lineages = BTreeMap::from([
        (RESOURCES[0], vec![hash(1), child_of_another_parent]),
        (RESOURCES[1], vec![hash(1), hash(2)]),
        (RESOURCES[2], vec![hash(2)]),
    ]);
    let mut staged = Vec::new();
    for (&resource, hashes) in &lineages {
        let manager = Arc::clone(&managers[&resource]);
        let blocks = manager
            .allocate_blocks(hashes.len())
            .ok_or_else(|| anyhow!("mock G2 manager is full"))?
            .into_iter()
            .zip(hashes.iter().copied())
            .map(|(block, hash)| block.stage(hash, manager.block_size()))
            .collect::<Result<Vec<_>, _>>()?;
        staged.push(StagedPull::from_test_parts(
            resource,
            hashes.clone(),
            blocks,
            manager,
        ));
    }
    let bundle = StagedBundle::new(lineages.clone(), staged)?;

    let error = engine
        .commit_pulled_bundle(identity.clone(), key, 1, bundle)
        .await
        .unwrap_err();

    let Some(BundleResourceLineageError::ParentHashMismatch { resource, .. }) =
        error.downcast_ref::<BundleResourceLineageError>()
    else {
        panic!("unexpected disconnected-lineage error: {error:#}");
    };
    assert_eq!(*resource, RESOURCES[0]);
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&identity, &key)
            .is_none(),
        "a disconnected pulled lineage must not create catalog visibility"
    );
    for (&resource, hashes) in &lineages {
        assert!(
            managers[&resource].match_blocks(hashes).is_empty(),
            "a disconnected pulled lineage must not publish resource {resource:?}"
        );
    }
    tokio::task::yield_now().await;
    assert!(directory.advertisements.lock().unwrap().is_empty());
    assert_eq!(
        announced.load(Ordering::SeqCst),
        0,
        "a pull that never published must not announce residency"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn later_local_boundary_onboards_one_fixed_capsule_and_full_histories() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader().await?;
    let engine = LocalConnectorEngine::new(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
    );
    commit_two_block_search_bundle(&engine, &identity, &managers)?;
    let request = FindBlocksRequest {
        request_id: "later-local-boundary".into(),
        cache: CacheScope::Manifest(identity.clone()),
        sequence_hashes: Arc::from([hash(1), hash(2)]),
        num_computed_tokens: 0,
        total_tokens: 2 * BLOCK_SIZE + 1,
        transfer_params: None,
        local_prefill_estimate: None,
    };
    let FindBlocksOutcome::Resolved {
        matched_tokens,
        minted: Some(search),
        ..
    } = engine.clone().find_blocks(&request, None)?
    else {
        anyhow::bail!("two-block bundle did not resolve")
    };
    assert_eq!(matched_tokens, 2 * BLOCK_SIZE);

    let onboard = engine.clone().onboard_bundle(
        &search,
        identity
            .resources()
            .iter()
            .map(|requirement| ResourceDestination {
                resource: requirement.resource(),
                block_ids: match requirement.role() {
                    ResourceRole::PrefixHistory => vec![100, 101],
                    ResourceRole::BoundaryCapsule => vec![200],
                },
            })
            .collect(),
        matched_tokens,
    )?;
    wait_until(|| onboard.is_complete()).await;
    assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_offload_abort_records_one_transaction_and_every_child_outcome() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader().await?;
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
        Arc::new(FailingResourceOffloadSubmit {
            delegate: RegisteringOffloadSubmit { managers },
            failing_resource: RESOURCES[1],
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );

    let offload = engine.clone().offload_bundle(
        &"metric-abort".into(),
        one_block_plan(identity, "metric-abort"),
    )?;
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 0);
    wait_until(|| offload.is_complete()).await;
    assert!(matches!(
        offload.outcome(),
        Some(SaveOutcome::FailedAllBlocks | SaveOutcome::FailedPartial { .. })
    ));
    assert_counter(
        engine.leader.observability().unwrap(),
        "kvbm_bundle_txn_total",
        &[("operation", "offload"), ("outcome", "abort")],
        1.0,
    );
    for resource in RESOURCES {
        let resource_label = resource.0.to_string();
        let (outcome, reason, bytes) = if resource == RESOURCES[1] {
            ("failed", "partial_failure", 0.0)
        } else {
            ("complete", "complete", 1024.0)
        };
        assert_counter(
            engine.leader.observability().unwrap(),
            "kvbm_bundle_resource_outcome_total",
            &[
                ("operation", "offload_transfer"),
                ("resource", resource_label.as_str()),
                ("outcome", outcome),
                ("reason", reason),
            ],
            1.0,
        );
        assert_counter(
            engine.leader.observability().unwrap(),
            "kvbm_bundle_resource_bytes_total",
            &[
                ("operation", "offload_transfer"),
                ("resource", resource_label.as_str()),
                ("outcome", outcome),
            ],
            bytes,
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_bundle_publication_records_abort_instead_of_commit() -> Result<()> {
    let identity = manifest()?.identity();
    let key = BundleKey::new(&identity, hash(1), BLOCK_SIZE as u64)?;
    let (leader, managers) = build_resource_test_leader().await?;
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
        Arc::new(RegisteringOffloadSubmit {
            managers: managers.clone(),
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let newer_resources = managers
        .iter()
        .map(|(&resource, manager)| -> Result<_> {
            let block = manager
                .allocate_blocks(1)
                .and_then(|mut blocks| blocks.pop())
                .ok_or_else(|| anyhow!("test manager has no free block"))?;
            let complete = block.stage(hash(1), manager.block_size())?;
            Ok((resource, vec![manager.register_block(complete)]))
        })
        .collect::<Result<Vec<_>>>()?;
    engine
        .bundle_catalog
        .lock()
        .unwrap()
        .index_mut()
        .commit(&identity, key, 2, newer_resources)?;

    let offload = engine.clone().offload_bundle(
        &"stale-publication".into(),
        one_block_plan(identity, "stale-publication"),
    )?;
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 0);
    wait_until(|| offload.is_complete()).await;

    assert_eq!(offload.outcome(), Some(SaveOutcome::FailedAllBlocks));
    assert_counter(
        engine.leader.observability().unwrap(),
        "kvbm_bundle_txn_total",
        &[("operation", "offload"), ("outcome", "abort")],
        1.0,
    );
    assert_eq!(
        counter_value(
            engine.leader.observability().unwrap(),
            "kvbm_bundle_txn_total",
            &[("operation", "offload"), ("outcome", "commit")],
        ),
        None,
        "a barrier-complete but unpublished bundle is not committed"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tiny_resource_managers_reclaim_an_old_complete_bundle_for_a_newer_one() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for_capacity(&identity, 1).await?;
    let directory = Arc::new(RecordingBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let old_plan = one_block_plan(identity.clone(), "old-complete");
    let old_key = old_plan.key;
    let old = engine
        .clone()
        .offload_bundle(&"old-complete".into(), old_plan)?;
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 0);
    wait_until(|| old.is_complete()).await;
    assert_eq!(old.outcome(), Some(SaveOutcome::Done));
    wait_until(|| directory.advertisements.lock().unwrap().len() == 1).await;

    let new_hash = resource_hash(2, 0);
    let new_key = BundleKey::new(&identity, new_hash, BLOCK_SIZE as u64)?;
    let new_plan = BundleOffloadPlan {
        identity: identity.clone(),
        key: new_key,
        mode: OffloadMode::Move,
        resources: RESOURCES
            .iter()
            .enumerate()
            .map(|(index, &resource)| ResourceOffload {
                resource,
                blocks: vec![(new_hash, 200 + index)],
            })
            .collect(),
    };
    let new = engine
        .clone()
        .offload_bundle(&"new-complete".into(), new_plan)?;
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 1);
    wait_until(|| new.is_complete()).await;

    assert_eq!(new.outcome(), Some(SaveOutcome::Done));
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&identity, &old_key)
            .is_none(),
        "real manager eviction must remove the old complete bundle"
    );
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&identity, &new_key)
            .is_some(),
        "the newer complete bundle must be atomically reacquirable"
    );
    wait_until(|| directory.invalidations.lock().unwrap().len() == 1).await;
    assert_eq!(directory.invalidations.lock().unwrap()[0].key, old_key);
    wait_until(|| directory.advertisements.lock().unwrap().len() == 2).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_slot_managers_readmit_an_identical_bundle_only_with_a_new_generation() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for_capacity(&identity, 1).await?;
    let directory = Arc::new(RecordingBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let first_plan = one_block_plan(identity.clone(), "same-key-generation-1");
    let key = first_plan.key;
    let first = engine
        .clone()
        .offload_bundle(&"same-key-generation-1".into(), first_plan)?;
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 0);
    wait_until(|| first.is_complete()).await;
    assert_eq!(first.outcome(), Some(SaveOutcome::Done));

    let second_plan = one_block_plan(identity.clone(), "same-key-generation-2");
    let second = engine
        .clone()
        .offload_bundle(&"same-key-generation-2".into(), second_plan)?;
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 1);
    wait_until(|| second.is_complete()).await;
    assert_eq!(second.outcome(), Some(SaveOutcome::Done));

    let mut catalog = engine.bundle_catalog.lock().unwrap();
    let lease = catalog
        .lease_exact(&identity, &key)
        .expect("the newer identical bundle must have one exact all-resource lease");
    assert_eq!(lease.generation(), 2);
    let lineages = identity
        .resources()
        .iter()
        .map(|requirement| {
            ResourceLineage::new(
                requirement.resource(),
                requirement.role(),
                vec![key.boundary_hash()],
            )
        })
        .collect();
    assert_eq!(
        catalog.commit(&identity, key, 1, lease.resources(), lineages),
        Err(BundleCatalogError::RetiredGeneration {
            retired: 1,
            attempted: 1,
        }),
        "a delayed old capture must not overwrite the readmitted generation"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_slot_offload_times_out_inflight_advertisement_before_invalidation() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for_capacity(&identity, 1).await?;
    let directory = Arc::new(BlockingBundleDirectory::default());
    directory.block_next_advertisement();
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );

    let old_plan = one_block_plan(identity.clone(), "blocked-advertisement");
    let old_key = old_plan.key;
    let old = engine
        .clone()
        .offload_bundle(&"blocked-advertisement".into(), old_plan)?;
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 0);
    wait_until(|| old.is_complete()).await;
    directory.wait_until_advertisement_starts().await;

    let new_hash = resource_hash(2, 0);
    let new_key = BundleKey::new(&identity, new_hash, BLOCK_SIZE as u64)?;
    let new = engine.clone().offload_bundle(
        &"evict-blocked-advertisement".into(),
        BundleOffloadPlan {
            identity: identity.clone(),
            key: new_key,
            mode: OffloadMode::Move,
            resources: RESOURCES
                .iter()
                .enumerate()
                .map(|(index, &resource)| ResourceOffload {
                    resource,
                    blocks: vec![(new_hash, 300 + index)],
                })
                .collect(),
        },
    )?;
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 1);
    wait_until(|| new.is_complete()).await;

    wait_until(|| directory.events().len() == 2).await;
    assert_eq!(
        directory.active(),
        HashSet::from([(new_key, 2)]),
        "an invalidated generation must not be resurrected by its older advertise task"
    );
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&identity, &old_key)
            .is_none()
    );
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .dependents(RESOURCES[0], hash(1))
            .is_empty()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_pull_orders_invalidation_after_an_inflight_advertisement() -> Result<()> {
    let identity = manifest()?.identity();
    let key = BundleKey::new(&identity, hash(1), BLOCK_SIZE as u64)?;
    let (leader, managers) = build_resource_test_leader_for_capacity(&identity, 1).await?;
    let directory = Arc::new(BlockingBundleDirectory::default());
    directory.block_next_advertisement();
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit {
            managers: managers.clone(),
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let mut staged = Vec::new();
    let lineages = RESOURCES
        .into_iter()
        .map(|resource| (resource, vec![hash(1)]))
        .collect::<BTreeMap<_, _>>();
    for (&resource, manager) in &managers {
        let block = manager
            .allocate_blocks(1)
            .and_then(|mut blocks| blocks.pop())
            .ok_or_else(|| anyhow!("one-slot manager could not allocate pulled block"))?;
        let complete = block.stage(hash(1), manager.block_size())?;
        staged.push(StagedPull::from_test_parts(
            resource,
            vec![hash(1)],
            vec![complete],
            Arc::clone(manager),
        ));
    }
    let bundle = StagedBundle::new(lineages, staged)?;
    let target: Arc<dyn BundlePullTarget> = Arc::clone(&engine) as Arc<dyn BundlePullTarget>;
    let generation = target.reserve_publication_generation()?;
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        target.commit_pulled_bundle(identity.clone(), key, generation, bundle),
    )
    .await
    .expect("one-slot staged publication must not deadlock the eviction observer")?;
    directory.wait_until_advertisement_starts().await;

    let manager = &managers[&RESOURCES[0]];
    let replacement = manager
        .allocate_blocks(1)
        .and_then(|mut blocks| blocks.pop())
        .ok_or_else(|| anyhow!("one-slot manager could not evict pulled block"))?;
    let replacement = replacement.stage(hash(2), manager.block_size())?;
    drop(manager.register_block(replacement));

    directory.release_advertisement();
    wait_until(|| directory.events().len() == 2).await;
    assert!(
        directory.active().is_empty(),
        "remote-pull publication must not re-advertise an evicted generation"
    );
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&identity, &key)
            .is_none()
    );
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .dependents(RESOURCES[0], hash(1))
            .is_empty()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_pull_mints_a_local_publication_generation() -> Result<()> {
    let identity = manifest()?.identity();
    let key = BundleKey::new(&identity, hash(1), BLOCK_SIZE as u64)?;
    let (leader, managers) = build_resource_test_leader_for_capacity(&identity, 1).await?;
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit {
            managers: managers.clone(),
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let mut staged = Vec::new();
    let lineages = RESOURCES
        .into_iter()
        .map(|resource| (resource, vec![hash(1)]))
        .collect::<BTreeMap<_, _>>();
    for (&resource, manager) in &managers {
        let block = manager
            .allocate_blocks(1)
            .and_then(|mut blocks| blocks.pop())
            .ok_or_else(|| anyhow!("one-slot manager could not allocate pulled block"))?;
        let complete = block.stage(hash(1), manager.block_size())?;
        staged.push(StagedPull::from_test_parts(
            resource,
            vec![hash(1)],
            vec![complete],
            Arc::clone(manager),
        ));
    }
    let bundle = StagedBundle::new(lineages, staged)?;
    let target: Arc<dyn BundlePullTarget> = Arc::clone(&engine) as Arc<dyn BundlePullTarget>;
    let generation = target.reserve_publication_generation()?;

    target
        .commit_pulled_bundle(identity.clone(), key, generation, bundle)
        .await?;

    let lease = engine
        .bundle_catalog
        .lock()
        .unwrap()
        .lease_exact(&identity, &key)
        .expect("the complete pulled bundle must be visible");
    assert_eq!(
        lease.generation(),
        1,
        "the puller's owner-local generation must not reuse the source owner's generation"
    );
    Ok(())
}

/// Remote pull is the one route to a resident G2 copy that the offload
/// pipeline's G1→G2 register observer cannot see, so a tier-placement publisher
/// without this seam under-reports its own residency.
///
/// Both arms matter and only one of them is obvious:
///   * a completed pull announces exactly the lineage it published, and
///   * a same-generation re-commit — which `commit_materialized` short-circuits
///     as idempotent, returning `Ok` **without** running the materializer —
///     announces nothing. Firing there would advertise a publication this call
///     did not make, i.e. over-reporting, the one direction the tier stream must
///     never fail in.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_completed_remote_pull_announces_its_residency_exactly_once() -> Result<()> {
    let identity = manifest()?.identity();
    let key = BundleKey::new(&identity, hash(1), BLOCK_SIZE as u64)?;
    let (leader, managers) = build_resource_test_leader_for_capacity(&identity, 2).await?;
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit {
            managers: managers.clone(),
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let observed: Arc<Mutex<Vec<PulledLineages>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&observed);
    engine.set_pulled_bundle_ready_observer(Arc::new(move |lineages| {
        sink.lock().unwrap().push(lineages.clone());
    }));

    let expected = RESOURCES
        .into_iter()
        .map(|resource| (resource, vec![hash(1)]))
        .collect::<BTreeMap<_, _>>();
    let target: Arc<dyn BundlePullTarget> = Arc::clone(&engine) as Arc<dyn BundlePullTarget>;
    let generation = target.reserve_publication_generation()?;
    target
        .commit_pulled_bundle(
            identity.clone(),
            key,
            generation,
            stage_pulled_one_block_bundle(&managers, hash(1))?,
        )
        .await?;

    assert_eq!(
        observed.lock().unwrap().as_slice(),
        std::slice::from_ref(&expected),
        "a completed pull announces the lineage it published"
    );

    // Same key, same generation: idempotent, so nothing is materialized and
    // nothing may be announced.
    target
        .commit_pulled_bundle(
            identity.clone(),
            key,
            generation,
            stage_pulled_one_block_bundle(&managers, hash(1))?,
        )
        .await?;
    assert_eq!(
        observed.lock().unwrap().len(),
        1,
        "an idempotent re-commit publishes nothing, so it announces nothing"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retired_remote_pull_generation_never_materializes_staged_blocks() -> Result<()> {
    let identity = manifest()?.identity();
    let pulled_hash = hash(1);
    let key = BundleKey::new(&identity, pulled_hash, BLOCK_SIZE as u64)?;
    let (leader, managers) = build_resource_test_leader_for_capacity(&identity, 1).await?;
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit {
            managers: managers.clone(),
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let target: Arc<dyn BundlePullTarget> = Arc::clone(&engine) as Arc<dyn BundlePullTarget>;

    target
        .commit_pulled_bundle(
            identity.clone(),
            key,
            1,
            stage_pulled_one_block_bundle(&managers, pulled_hash)?,
        )
        .await?;
    assert_eq!(
        engine.bundle_catalog.lock().unwrap().invalidate_key(key),
        Some(1)
    );

    let late = stage_pulled_one_block_bundle(&managers, pulled_hash)?;
    for manager in managers.values() {
        assert!(
            manager.match_blocks(&[pulled_hash]).is_empty(),
            "staged blocks must remain invisible before the retired commit attempt"
        );
    }
    let error = target
        .commit_pulled_bundle(identity.clone(), key, 1, late)
        .await
        .expect_err("retired generation must reject before materialization");
    assert!(matches!(
        error.downcast_ref::<BundleCatalogError>(),
        Some(BundleCatalogError::RetiredGeneration {
            retired: 1,
            attempted: 1,
        })
    ));
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&identity, &key)
            .is_none()
    );
    for manager in managers.values() {
        assert!(
            manager.match_blocks(&[pulled_hash]).is_empty(),
            "retired commit must never invoke the block materializer"
        );
        assert_eq!(manager.available_blocks(), 1);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn publication_generation_is_shared_by_engines_with_one_owner() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for_capacity(&identity, 2).await?;
    let leader = Arc::new(leader);
    let build_engine = || {
        LocalConnectorEngine::with_offload_submit_and_admission(
            Arc::clone(&leader),
            kvbm_protocols::connector::NoopWorkerSink::new(),
            BLOCK_SIZE,
            true,
            Arc::new(RegisteringOffloadSubmit {
                managers: managers.clone(),
            }),
            None,
            BundleAdmissionConfig::new(
                policies(&identity, 0, None).unwrap(),
                component_bytes(&identity, 1024).unwrap(),
            ),
        )
    };
    let first = build_engine();
    let second = build_engine();
    let first: Arc<dyn BundlePullTarget> = first;
    let second: Arc<dyn BundlePullTarget> = second;

    assert_eq!(first.reserve_publication_generation()?, 1);
    assert_eq!(
        second.reserve_publication_generation()?,
        2,
        "publication generations must share the InstanceLeader owner's lifetime"
    );
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn live_bundle_publication_is_renewed_beyond_the_directory_ttl() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for(&identity).await?;
    let directory = Arc::new(ExpiringBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit {
            managers: managers.clone(),
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let clock = Arc::new(ManualPublicationClock::new(1_000_000));
    let publication_clock = Arc::clone(&clock);
    engine.set_bundle_publication_clock_for_test(Arc::new(move || publication_clock.now_unix_ms()));
    let key = commit_one_block_bundle(&engine, &identity, &managers)?.key;
    yield_until(|| directory.advertisement_count() == 1).await;

    let beyond_ttl = Duration::from_millis(BUNDLE_DIRECTORY_TTL_MS + 1);
    clock.advance(beyond_ttl);
    tokio::time::advance(beyond_ttl).await;
    yield_until(|| directory.advertisement_count() >= 2).await;

    assert_eq!(
        directory.visible_generation(key, clock.now_unix_ms()),
        Some(1),
        "a locally live generation must remain remotely discoverable after its first TTL"
    );
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn transient_initial_publication_failure_is_retried_by_the_refresher() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for(&identity).await?;
    let directory = Arc::new(ExpiringBundleDirectory::default());
    directory.fail_next_advertisements(1);
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit {
            managers: managers.clone(),
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let clock = Arc::new(ManualPublicationClock::new(1_500_000));
    let publication_clock = Arc::clone(&clock);
    engine.set_bundle_publication_clock_for_test(Arc::new(move || publication_clock.now_unix_ms()));
    let key = commit_one_block_bundle(&engine, &identity, &managers)?.key;
    yield_until(|| directory.advertisement_attempts() == 1).await;
    assert_eq!(directory.advertisement_count(), 0);

    let next_refresh = Duration::from_millis(BUNDLE_DIRECTORY_TTL_MS + 1);
    clock.advance(next_refresh);
    tokio::time::advance(next_refresh).await;
    yield_until(|| directory.advertisement_count() == 1).await;

    assert!(directory.advertisement_attempts() >= 2);
    assert_eq!(
        directory.visible_generation(key, clock.now_unix_ms()),
        Some(1)
    );
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn invalidated_bundle_generation_is_not_renewed_or_resurrected() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for(&identity).await?;
    let directory = Arc::new(ExpiringBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit {
            managers: managers.clone(),
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let clock = Arc::new(ManualPublicationClock::new(2_000_000));
    let publication_clock = Arc::clone(&clock);
    engine.set_bundle_publication_clock_for_test(Arc::new(move || publication_clock.now_unix_ms()));
    let key = commit_one_block_bundle(&engine, &identity, &managers)?.key;
    yield_until(|| directory.advertisement_count() == 1).await;

    engine.invalidate_resource_blocks(RESOURCES[0], &[hash(1)]);
    yield_until(|| {
        directory
            .visible_generation(key, clock.now_unix_ms())
            .is_none()
    })
    .await;
    let advertisements_after_invalidation = directory.advertisement_count();

    let two_ttls = Duration::from_millis(BUNDLE_DIRECTORY_TTL_MS.saturating_mul(2));
    clock.advance(two_ttls);
    tokio::time::advance(two_ttls).await;
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }

    assert_eq!(
        directory.advertisement_count(),
        advertisements_after_invalidation,
        "invalidation must cancel renewal for that exact generation"
    );
    assert_eq!(directory.visible_generation(key, clock.now_unix_ms()), None);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn superseded_generation_is_never_renewed_after_the_replacement_publishes() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for(&identity).await?;
    let directory = Arc::new(ExpiringBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit {
            managers: managers.clone(),
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let clock = Arc::new(ManualPublicationClock::new(3_000_000));
    let publication_clock = Arc::clone(&clock);
    engine.set_bundle_publication_clock_for_test(Arc::new(move || publication_clock.now_unix_ms()));
    let key = commit_one_block_bundle(&engine, &identity, &managers)?.key;
    yield_until(|| directory.advertisement_count_for_generation(1) == 1).await;

    let replacement_resources = engine
        .bundle_catalog
        .lock()
        .unwrap()
        .lease_exact(&identity, &key)
        .expect("first generation remains locally live")
        .resources()
        .clone();
    let replacement_lineages = identity
        .resources()
        .iter()
        .map(|requirement| {
            ResourceLineage::new(requirement.resource(), requirement.role(), vec![hash(1)])
        })
        .collect();
    engine.commit_bundle(
        identity,
        key,
        2,
        replacement_resources,
        replacement_lineages,
    )?;
    yield_until(|| directory.advertisement_count_for_generation(2) == 1).await;
    let generation_one_advertisements = directory.advertisement_count_for_generation(1);

    let two_ttls = Duration::from_millis(BUNDLE_DIRECTORY_TTL_MS.saturating_mul(2));
    clock.advance(two_ttls);
    tokio::time::advance(two_ttls).await;
    yield_until(|| directory.advertisement_count_for_generation(2) >= 2).await;

    assert_eq!(
        directory.advertisement_count_for_generation(1),
        generation_one_advertisements,
        "the refresh loop must prune a publication superseded in the catalog"
    );
    assert_eq!(
        directory.visible_generation(key, clock.now_unix_ms()),
        Some(2)
    );
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn dropping_engine_stops_its_single_bundle_publication_refresher() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for(&identity).await?;
    let directory = Arc::new(ExpiringBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit {
            managers: managers.clone(),
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let clock = Arc::new(ManualPublicationClock::new(4_000_000));
    let publication_clock = Arc::clone(&clock);
    engine.set_bundle_publication_clock_for_test(Arc::new(move || publication_clock.now_unix_ms()));
    commit_one_block_bundle(&engine, &identity, &managers)?;
    yield_until(|| directory.advertisement_count() == 1).await;

    let first_ttl = Duration::from_millis(BUNDLE_DIRECTORY_TTL_MS + 1);
    clock.advance(first_ttl);
    tokio::time::advance(first_ttl).await;
    yield_until(|| directory.advertisement_count() >= 2).await;

    let weak_engine = Arc::downgrade(&engine);
    drop(engine);
    yield_until(|| weak_engine.upgrade().is_none()).await;
    let advertisements_after_drop = directory.advertisement_count();
    let later = Duration::from_millis(BUNDLE_DIRECTORY_TTL_MS.saturating_mul(3));
    clock.advance(later);
    tokio::time::advance(later).await;
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }

    assert_eq!(
        directory.advertisement_count(),
        advertisements_after_drop,
        "dropping the engine must cancel its sole weak-owned refresh task"
    );
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn dropping_engine_during_a_blocked_refresh_stops_after_that_bounded_call() -> Result<()> {
    let identity = manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for(&identity).await?;
    let directory = Arc::new(BlockingBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit {
            managers: managers.clone(),
        }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    commit_one_block_bundle(&engine, &identity, &managers)?;
    yield_until(|| directory.events().len() == 1).await;

    directory.block_next_advertisement();
    tokio::time::advance(Duration::from_millis(BUNDLE_DIRECTORY_TTL_MS)).await;
    directory.wait_until_advertisement_starts().await;

    let weak_engine = Arc::downgrade(&engine);
    drop(engine);
    assert!(
        weak_engine.upgrade().is_some(),
        "the in-flight bounded directory call temporarily owns its engine"
    );
    directory.release_advertisement();
    yield_until(|| weak_engine.upgrade().is_none()).await;
    yield_until(|| directory.events().len() == 2).await;
    let events_after_drop = directory.events().len();

    tokio::time::advance(Duration::from_millis(
        BUNDLE_DIRECTORY_TTL_MS.saturating_mul(3),
    ))
    .await;
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        directory.events().len(),
        events_after_drop,
        "the refresher must exit when the last bounded in-flight owner releases"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_offload_rejects_disconnected_and_accepts_projected_mixed_native_histories()
-> Result<()> {
    let identity = mixed_native_manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for(&identity).await?;
    let leader = Arc::new(leader);
    let directory = Arc::new(RecordingBundleDirectory::default());
    assert!(leader.set_remote_discovery(Arc::clone(&directory) as Arc<dyn RemoteBlockDiscovery>));
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        leader,
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let canonical_root = SequenceHash::root(102);
    let canonical_boundary = canonical_root.extend(104);
    let canonical = [canonical_root, canonical_boundary];
    let projected_boundary =
        BundleResourceLineage::project_from_canonical(RESOURCES[1], &canonical, 2)?.hashes()[0];
    assert_ne!(canonical_boundary, projected_boundary);
    let key = BundleKey::new(&identity, canonical_boundary, BLOCK_SIZE as u64)?;

    let disconnected_secondary = engine.clone().offload_bundle(
        &"mixed-native-disconnected".into(),
        BundleOffloadPlan {
            identity: identity.clone(),
            key,
            mode: OffloadMode::Mirror,
            resources: vec![
                ResourceOffload {
                    resource: RESOURCES[0],
                    blocks: vec![(canonical_root, 10), (canonical_boundary, 11)],
                },
                ResourceOffload {
                    resource: RESOURCES[1],
                    blocks: vec![(SequenceHash::root(999), 12)],
                },
                ResourceOffload {
                    resource: RESOURCES[2],
                    blocks: vec![(canonical_boundary, 13)],
                },
            ],
        },
    );
    let Err(kvbm_protocols::connector::LeaderEngineError::InvalidBundleTransfer { reason }) =
        disconnected_secondary
    else {
        panic!("a disconnected secondary history must fail before offload starts");
    };
    assert!(reason.contains("does not reach boundary"));
    assert!(engine.offload_buffer.lock().unwrap().is_empty());

    let offload = engine.clone().offload_bundle(
        &"mixed-native-bundle".into(),
        BundleOffloadPlan {
            identity,
            key,
            mode: OffloadMode::Mirror,
            resources: vec![
                ResourceOffload {
                    resource: RESOURCES[0],
                    blocks: vec![(canonical_root, 20), (canonical_boundary, 21)],
                },
                ResourceOffload {
                    resource: RESOURCES[1],
                    blocks: vec![(projected_boundary, 22)],
                },
                ResourceOffload {
                    resource: RESOURCES[2],
                    blocks: vec![(canonical_boundary, 23)],
                },
            ],
        },
    )?;

    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 0);
    wait_until(|| offload.is_complete()).await;
    assert_eq!(offload.outcome(), Some(SaveOutcome::Done));
    wait_until(|| !directory.advertisements.lock().unwrap().is_empty()).await;
    let advertised = directory.advertisements.lock().unwrap();
    let lineages = advertised[0]
        .lineages()
        .map(|lineage| (lineage.resource(), lineage.hashes().to_vec()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(lineages[&RESOURCES[0]], canonical);
    assert_eq!(lineages[&RESOURCES[1]], vec![projected_boundary]);
    assert_eq!(lineages[&RESOURCES[2]], vec![canonical_boundary]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_offload_admission_accepts_projection_across_plh_position_mode_boundary()
-> Result<()> {
    let identity = mixed_native_manifest()?.identity();
    let (leader, managers) = build_resource_test_leader_for(&identity).await?;
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let canonical = {
        let root = SequenceHash::root(10_000);
        std::iter::once(root)
            .chain((1..514u64).scan(root, |parent, block| {
                *parent = parent.extend(10_000 + block);
                Some(*parent)
            }))
            .collect::<Vec<_>>()
    };
    let projected = BundleResourceLineage::project_from_canonical(RESOURCES[1], &canonical, 2)?;
    assert_eq!(projected.hashes()[255].position(), 255);
    assert_eq!(projected.hashes()[256].position(), 256);
    let boundary_hash = *canonical.last().unwrap();
    let key = BundleKey::new(&identity, boundary_hash, 1_028)?;
    let plan = BundleOffloadPlan {
        identity,
        key,
        mode: OffloadMode::Mirror,
        resources: vec![
            ResourceOffload {
                resource: RESOURCES[0],
                blocks: canonical
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(block, hash)| (hash, block))
                    .collect(),
            },
            ResourceOffload {
                resource: RESOURCES[1],
                blocks: projected
                    .hashes()
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(block, hash)| (hash, 1_000 + block))
                    .collect(),
            },
            ResourceOffload {
                resource: RESOURCES[2],
                blocks: vec![(boundary_hash, 2_000)],
            },
        ],
    };

    let offload = engine
        .clone()
        .offload_bundle(&"plh-mode-boundary".into(), plan)?;

    assert!(!offload.is_complete());
    let buffered = engine.offload_buffer.lock().unwrap();
    assert_eq!(buffered.len(), 3);
    assert_eq!(
        buffered
            .iter()
            .map(|resource| resource.pairs.len())
            .collect::<Vec<_>>(),
        vec![514, 257, 1]
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_offload_rejects_incomplete_or_wrong_lineage_children() -> Result<()> {
    let manifest = manifest()?;
    let identity = manifest.identity();
    let (leader, managers) = build_resource_test_leader().await?;
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );
    let key = BundleKey::new(&identity, hash(2), (2 * BLOCK_SIZE) as u64)?;
    let result = engine.clone().offload_bundle(
        &"incomplete-bundle".into(),
        BundleOffloadPlan {
            identity: identity.clone(),
            key,
            mode: OffloadMode::Move,
            resources: RESOURCES
                .iter()
                .enumerate()
                .map(|(index, &resource)| ResourceOffload {
                    resource,
                    blocks: vec![(hash(2), 20 + index)],
                })
                .collect(),
        },
    );

    assert!(matches!(
        result,
        Err(kvbm_protocols::connector::LeaderEngineError::InvalidBundleTransfer { .. })
    ));
    assert!(engine.offload_buffer.lock().unwrap().is_empty());

    for (request, primary_boundary, capsule_boundary, rejected_resource) in [
        ("wrong-primary", hash(1).extend(99), hash(2), RESOURCES[0]),
        ("wrong-capsule", hash(2), hash(99), RESOURCES[2]),
    ] {
        let wrong_lineage = engine.clone().offload_bundle(
            &request.to_owned(),
            BundleOffloadPlan {
                identity: identity.clone(),
                key,
                mode: OffloadMode::Move,
                resources: vec![
                    ResourceOffload {
                        resource: RESOURCES[0],
                        blocks: vec![(hash(1), 30), (primary_boundary, 40)],
                    },
                    ResourceOffload {
                        resource: RESOURCES[1],
                        blocks: vec![(hash(101), 31), (hash(102), 41)],
                    },
                    ResourceOffload {
                        resource: RESOURCES[2],
                        blocks: vec![(capsule_boundary, 42)],
                    },
                ],
            },
        );
        let Err(kvbm_protocols::connector::LeaderEngineError::InvalidBundleTransfer { reason }) =
            wrong_lineage
        else {
            panic!("noncanonical primary history or capsule must be rejected");
        };
        assert!(reason.contains(&format!("{rejected_resource:?}")));
        assert!(engine.offload_buffer.lock().unwrap().is_empty());
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_same_request_resource_restore_does_not_require_a_bundle_identity() -> Result<()> {
    let (leader, _) = build_resource_test_leader().await?;
    let engine = LocalConnectorEngine::new(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
    );
    let resources = RESOURCES
        .into_iter()
        .enumerate()
        .map(|(index, resource)| ResourceOnboard {
            resource,
            source_block_ids: vec![index],
            destination_block_ids: vec![100 + index],
        })
        .collect();

    let onboard = engine
        .clone()
        .onboard_resource_blocks(&"same-request".to_owned(), resources)?;
    wait_until(|| onboard.is_complete()).await;
    assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resource_policy_role_must_agree_with_the_manifest() -> Result<()> {
    let (leader, managers) = build_resource_test_leader().await?;
    let mut policies = ResourcePolicies::new();
    policies
        .insert(
            RESOURCES[0],
            ResourcePolicy::new(
                ResourceRole::BoundaryCapsule,
                InactiveBackendConfig::Lru,
                InactiveBackendConfig::Lru,
            ),
        )
        .unwrap();
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        true,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(policies, component_bytes(&manifest()?.identity(), 1024)?),
    );
    let identity = manifest()?.identity();
    let key = BundleKey::new(&identity, hash(1), BLOCK_SIZE as u64)?;

    let result = engine.clone().offload_bundle(
        &"policy-role-mismatch".into(),
        BundleOffloadPlan {
            identity,
            key,
            mode: OffloadMode::Move,
            resources: RESOURCES
                .iter()
                .enumerate()
                .map(|(index, &resource)| ResourceOffload {
                    resource,
                    blocks: vec![(hash(1), 50 + index)],
                })
                .collect(),
        },
    );

    assert!(matches!(
        result,
        Err(kvbm_protocols::connector::LeaderEngineError::InvalidBundleTransfer { .. })
    ));
    assert!(engine.offload_buffer.lock().unwrap().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whole_bundle_admission_accepts_high_density_complete_resources() -> Result<()> {
    let identity = manifest()?.identity();
    let plan = one_block_plan(identity.clone(), "admit-complete");
    let (leader, managers) = build_resource_test_leader().await?;
    touch_plan(&managers, &plan, 4);
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 2, None)?,
            component_bytes(&identity, 1024 * 1024)?,
        ),
    );

    let handle = engine
        .clone()
        .offload_bundle(&"admit-complete".into(), plan)?;

    assert!(!handle.is_complete());
    assert_eq!(engine.offload_buffer.lock().unwrap().len(), RESOURCES.len());
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 0);
    wait_until(|| handle.is_complete()).await;
    assert_eq!(handle.outcome(), Some(SaveOutcome::Done));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whole_bundle_admission_allows_first_entry_before_retention_reuse_exists() -> Result<()> {
    let identity = manifest()?.identity();
    let plan = one_block_plan(identity.clone(), "first-entry");
    let (leader, managers) = build_resource_test_leader().await?;
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 1, None)?,
            component_bytes(&identity, 1024 * 1024)?,
        ),
    );

    let handle = engine.clone().offload_bundle(&"first-entry".into(), plan)?;
    assert_eq!(engine.offload_buffer.lock().unwrap().len(), RESOURCES.len());
    kvbm_protocols::connector::WorkerEngineDriver::finish_forward_pass(engine.as_ref(), 0);
    wait_until(|| handle.is_complete()).await;
    assert_eq!(handle.outcome(), Some(SaveOutcome::Done));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whole_bundle_admission_can_enforce_an_explicit_ingress_density_floor() -> Result<()> {
    let identity = manifest()?.identity();
    let plan = one_block_plan(identity.clone(), "reject-ingress-density");
    let (leader, managers) = build_resource_test_leader().await?;
    touch_plan(&managers, &plan, 1);
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies_with_thresholds(&identity, 0, 2, None)?,
            component_bytes(&identity, 1024 * 1024)?,
        ),
    );

    let error = engine
        .clone()
        .offload_bundle(&"reject-ingress-density".into(), plan)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("below_byte_normalized_threshold")
    );
    assert!(engine.offload_buffer.lock().unwrap().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whole_bundle_admission_rejects_incomplete_atomic_resource_without_enqueuing() -> Result<()>
{
    let identity = manifest()?.identity();
    let plan = one_block_plan(identity.clone(), "reject-components");
    let (leader, managers) = build_resource_test_leader().await?;
    touch_plan(&managers, &plan, 4);
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, Some((RESOURCES[0], 2)))?,
            component_bytes(&identity, 1024)?,
        ),
    );

    assert!(
        engine
            .clone()
            .offload_bundle(&"reject-components".into(), plan)
            .is_err()
    );
    assert!(engine.offload_buffer.lock().unwrap().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whole_bundle_admission_rejects_orphan_capsule_without_enqueuing() -> Result<()> {
    let capsule = RESOURCES[2];
    let identity = CacheManifest::new(
        ModelIdentity::new("capsule-only", "revision-a", [7; 32])?,
        "capsule-only-v1",
        vec![ResourceRequirement::new(
            capsule,
            ResourceRole::BoundaryCapsule,
            BLOCK_SIZE as u32,
        )?],
        Default::default(),
    )?
    .identity();
    let plan = one_block_plan(identity.clone(), "reject-orphan");
    let (leader, managers) = build_resource_test_leader().await?;
    touch_plan(&managers, &plan, 4);
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, 1024)?,
        ),
    );

    assert!(
        engine
            .clone()
            .offload_bundle(&"reject-orphan".into(), plan)
            .is_err()
    );
    assert!(engine.offload_buffer.lock().unwrap().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whole_bundle_admission_requires_exact_policy_and_component_coverage() -> Result<()> {
    let identity = manifest()?.identity();
    let plan = one_block_plan(identity.clone(), "reject-coverage");

    for coverage_error in 0..4 {
        let (leader, managers) = build_resource_test_leader().await?;
        touch_plan(&managers, &plan, 4);
        let mut policies = policies(&identity, 0, None)?;
        let mut components = component_bytes(&identity, 1024)?;
        match coverage_error {
            0 => policies = ResourcePolicies::new(),
            1 => components = ResourceComponentBytes::new(),
            2 => policies.insert(
                LogicalResourceId(99),
                ResourcePolicy::new(
                    ResourceRole::PrefixHistory,
                    InactiveBackendConfig::default(),
                    InactiveBackendConfig::default(),
                ),
            )?,
            3 => components.insert(LogicalResourceId(99), [NonZeroU64::new(1024).unwrap()])?,
            _ => unreachable!(),
        }
        let engine = LocalConnectorEngine::with_offload_submit_and_admission(
            Arc::new(leader),
            kvbm_protocols::connector::NoopWorkerSink::new(),
            BLOCK_SIZE,
            false,
            Arc::new(RegisteringOffloadSubmit { managers }),
            None,
            BundleAdmissionConfig::new(policies, components),
        );

        assert!(
            engine
                .clone()
                .offload_bundle(&"reject-coverage".into(), plan.clone())
                .is_err()
        );
        assert!(engine.offload_buffer.lock().unwrap().is_empty());
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whole_bundle_admission_uses_the_coldest_child_as_conservative_reuse() -> Result<()> {
    let identity = manifest()?.identity();
    let plan = two_block_plan(identity.clone());
    let (leader, managers) = build_resource_test_leader().await?;
    touch_plan(&managers, &plan, 1);
    for _ in 0..3 {
        managers[&RESOURCES[0]].block_registry().touch(hash(1));
    }
    let mut policies = ResourcePolicies::new();
    for requirement in identity.resources() {
        let policy = ResourcePolicy::new(
            requirement.role(),
            InactiveBackendConfig::default(),
            InactiveBackendConfig::default(),
        );
        let policy = if requirement.resource() == RESOURCES[0] {
            policy.with_minimum_admission_hits_per_mib(2)
        } else {
            policy
        };
        policies.insert(requirement.resource(), policy)?;
    }
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(policies, component_bytes(&identity, 1024 * 1024)?),
    );

    let error = engine
        .clone()
        .offload_bundle(&"reject-coldest-child".into(), plan)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("below_byte_normalized_threshold")
    );
    assert!(engine.offload_buffer.lock().unwrap().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whole_bundle_admission_rejects_component_byte_scaling_overflow() -> Result<()> {
    let identity = manifest()?.identity();
    let plan = two_block_plan(identity.clone());
    let (leader, managers) = build_resource_test_leader().await?;
    touch_plan(&managers, &plan, 4);
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        Arc::new(leader),
        kvbm_protocols::connector::NoopWorkerSink::new(),
        BLOCK_SIZE,
        false,
        Arc::new(RegisteringOffloadSubmit { managers }),
        None,
        BundleAdmissionConfig::new(
            policies(&identity, 0, None)?,
            component_bytes(&identity, u64::MAX)?,
        ),
    );

    assert!(
        engine
            .clone()
            .offload_bundle(&"reject-overflow".into(), plan)
            .is_err()
    );
    assert!(engine.offload_buffer.lock().unwrap().is_empty());
    Ok(())
}

async fn build_resource_test_leader() -> Result<(
    InstanceLeader,
    BTreeMap<LogicalResourceId, Arc<BlockManager<G2>>>,
)> {
    build_resource_test_leader_for(&manifest()?.identity()).await
}

async fn build_resource_test_leader_for(
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
) -> Result<(
    InstanceLeader,
    BTreeMap<LogicalResourceId, Arc<BlockManager<G2>>>,
)> {
    build_resource_test_leader_for_capacity(identity, 4).await
}

async fn build_resource_test_leader_for_capacity(
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
    block_count: usize,
) -> Result<(
    InstanceLeader,
    BTreeMap<LogicalResourceId, Arc<BlockManager<G2>>>,
)> {
    build_resource_test_leader_for_capacity_with_workers(
        identity,
        block_count,
        Arc::new(CompletedParallelWorkers),
    )
    .await
}

async fn build_resource_test_leader_for_capacity_with_workers(
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
    block_count: usize,
    workers: Arc<dyn ParallelWorkers>,
) -> Result<(
    InstanceLeader,
    BTreeMap<LogicalResourceId, Arc<BlockManager<G2>>>,
)> {
    let messenger = create_messenger_tcp().await?;
    let mut managers = BTreeMap::new();
    let mut set = BlockManagerSet::new();
    for requirement in identity.resources() {
        let resource = requirement.resource();
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(block_count)
                .block_size(usize::try_from(requirement.native_block_tokens().get())?)
                .registry(
                    BlockRegistry::builder()
                        .frequency_tracker(FrequencyTrackingCapacity::Small.create_tracker())
                        .build(),
                )
                .build(),
        );
        set.insert(resource, Arc::clone(&manager))?;
        managers.insert(resource, manager);
    }
    let leader = InstanceLeader::builder()
        .messenger(messenger)
        .registry(BlockRegistry::new())
        .g2_manager_set(Arc::new(set), RESOURCES[0])
        .parallel_worker(workers)
        .observability(Arc::new(KvbmObservability::default()))
        .build()?;
    Ok((leader, managers))
}

async fn build_hanging_onboard(
    watchdog: Duration,
) -> Result<(
    Arc<LocalConnectorEngine>,
    Arc<HangingParallelWorkers>,
    Arc<RecordingLoadSink>,
    BundleOnboardPlan,
)> {
    let identity = manifest()?.identity();
    let hanging = HangingParallelWorkers::new(RESOURCES[1]);
    let workers: Arc<dyn ParallelWorkers> = hanging.clone();
    let (leader, managers) =
        build_resource_test_leader_for_capacity_with_workers(&identity, 4, workers).await?;
    let sink = Arc::new(RecordingLoadSink::default());
    let engine = LocalConnectorEngine::new(Arc::new(leader), sink.clone(), BLOCK_SIZE, false);
    engine.set_bundle_onboard_watchdog_for_test(watchdog);
    let plan = commit_one_block_bundle(&engine, &identity, &managers)?;
    Ok((engine, hanging, sink, plan))
}

fn commit_one_block_bundle(
    engine: &LocalConnectorEngine,
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
    managers: &BTreeMap<LogicalResourceId, Arc<BlockManager<G2>>>,
) -> Result<BundleOnboardPlan> {
    let key = BundleKey::new(identity, hash(1), BLOCK_SIZE as u64)?;
    let mut pins = BTreeMap::new();
    let mut lineages = Vec::new();
    let mut resources = Vec::new();
    for (index, requirement) in identity.resources().iter().enumerate() {
        let resource = requirement.resource();
        let manager = &managers[&resource];
        let block = manager
            .allocate_blocks(1)
            .and_then(|mut blocks| blocks.pop())
            .ok_or_else(|| anyhow!("mock G2 manager is full"))?;
        let complete = block.stage(hash(1), manager.block_size())?;
        let pin = manager.register_block(complete);
        let source_block_id = pin.block_id();
        pins.insert(resource, vec![pin]);
        lineages.push(ResourceLineage::new(
            resource,
            requirement.role(),
            vec![hash(1)],
        ));
        resources.push(ResourceOnboard {
            resource,
            source_block_ids: vec![source_block_id],
            destination_block_ids: vec![100 + index],
        });
    }
    engine.commit_bundle(identity.clone(), key, 1, pins, lineages)?;
    Ok(BundleOnboardPlan {
        identity: identity.clone(),
        key,
        resources,
    })
}

fn commit_two_block_search_bundle(
    engine: &LocalConnectorEngine,
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
    managers: &BTreeMap<LogicalResourceId, Arc<BlockManager<G2>>>,
) -> Result<BundleKey> {
    let key = BundleKey::new(identity, hash(2), (2 * BLOCK_SIZE) as u64)?;
    let mut pins = BTreeMap::new();
    let mut lineages = Vec::new();
    for requirement in identity.resources() {
        let resource = requirement.resource();
        let manager = &managers[&resource];
        let hashes = match requirement.role() {
            ResourceRole::PrefixHistory => vec![hash(1), hash(2)],
            ResourceRole::BoundaryCapsule => vec![hash(2)],
        };
        let completed = manager
            .allocate_blocks(hashes.len())
            .ok_or_else(|| anyhow!("mock G2 manager is full"))?
            .into_iter()
            .zip(hashes.iter().copied())
            .map(|(block, sequence_hash)| block.stage(sequence_hash, manager.block_size()))
            .collect::<Result<Vec<_>, _>>()?;
        pins.insert(resource, manager.register_blocks(completed));
        lineages.push(ResourceLineage::new(resource, requirement.role(), hashes));
    }
    engine.commit_bundle(identity.clone(), key, 1, pins, lineages)?;
    Ok(key)
}

fn stage_pulled_one_block_bundle(
    managers: &BTreeMap<LogicalResourceId, Arc<BlockManager<G2>>>,
    pulled_hash: SequenceHash,
) -> Result<StagedBundle> {
    let lineages = managers
        .keys()
        .map(|resource| (*resource, vec![pulled_hash]))
        .collect::<BTreeMap<_, _>>();
    let mut staged = Vec::with_capacity(managers.len());
    for (&resource, manager) in managers {
        let block = manager
            .allocate_blocks(1)
            .and_then(|mut blocks| blocks.pop())
            .ok_or_else(|| anyhow!("one-slot manager could not allocate staged pull"))?;
        let complete = block.stage(pulled_hash, manager.block_size())?;
        staged.push(StagedPull::from_test_parts(
            resource,
            vec![pulled_hash],
            vec![complete],
            Arc::clone(manager),
        ));
    }
    Ok(StagedBundle::new(lineages, staged)?)
}

fn manifest() -> Result<CacheManifest> {
    Ok(CacheManifest::new(
        ModelIdentity::new("hybrid-cache", "revision-a", [5; 32])?,
        "hybrid-cache-v1",
        vec![
            ResourceRequirement::new(RESOURCES[0], ResourceRole::PrefixHistory, 4)?,
            ResourceRequirement::new(RESOURCES[1], ResourceRole::PrefixHistory, 4)?,
            ResourceRequirement::new(RESOURCES[2], ResourceRole::BoundaryCapsule, 4)?,
        ],
        Default::default(),
    )?)
}

fn mixed_native_manifest() -> Result<CacheManifest> {
    Ok(CacheManifest::new(
        ModelIdentity::new("mixed-native-cache", "revision-a", [6; 32])?,
        "mixed-native-cache-v1",
        vec![
            ResourceRequirement::new(RESOURCES[0], ResourceRole::PrefixHistory, 2)?,
            ResourceRequirement::new(RESOURCES[1], ResourceRole::PrefixHistory, 4)?,
            ResourceRequirement::new(RESOURCES[2], ResourceRole::BoundaryCapsule, 4)?,
        ],
        Default::default(),
    )?)
}

fn one_block_plan(
    identity: kvbm_protocols::cache_manifest::CacheIdentity,
    _request_id: &str,
) -> BundleOffloadPlan {
    let key = BundleKey::new(&identity, hash(1), BLOCK_SIZE as u64).unwrap();
    BundleOffloadPlan {
        identity: identity.clone(),
        key,
        mode: OffloadMode::Move,
        resources: identity
            .resources()
            .iter()
            .map(|requirement| ResourceOffload {
                resource: requirement.resource(),
                blocks: vec![(hash(1), 50 + usize::from(requirement.resource().0))],
            })
            .collect(),
    }
}

fn two_block_plan(identity: kvbm_protocols::cache_manifest::CacheIdentity) -> BundleOffloadPlan {
    let key = BundleKey::new(&identity, hash(2), (2 * BLOCK_SIZE) as u64).unwrap();
    BundleOffloadPlan {
        identity: identity.clone(),
        key,
        mode: OffloadMode::Move,
        resources: identity
            .resources()
            .iter()
            .map(|requirement| ResourceOffload {
                resource: requirement.resource(),
                blocks: match requirement.role() {
                    ResourceRole::PrefixHistory => vec![(hash(1), 51), (hash(2), 52)],
                    ResourceRole::BoundaryCapsule => vec![(hash(2), 53)],
                },
            })
            .collect(),
    }
}

fn policies(
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
    minimum_hits_per_mib: u64,
    atomic_override: Option<(LogicalResourceId, usize)>,
) -> Result<ResourcePolicies> {
    policies_with_thresholds(identity, minimum_hits_per_mib, 0, atomic_override)
}

fn policies_with_thresholds(
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
    minimum_hits_per_mib: u64,
    minimum_admission_hits_per_mib: u64,
    atomic_override: Option<(LogicalResourceId, usize)>,
) -> Result<ResourcePolicies> {
    let mut policies = ResourcePolicies::new();
    for requirement in identity.resources() {
        let mut policy = ResourcePolicy::new(
            requirement.role(),
            InactiveBackendConfig::default(),
            InactiveBackendConfig::default(),
        )
        .with_minimum_hits_per_mib(minimum_hits_per_mib)
        .with_minimum_admission_hits_per_mib(minimum_admission_hits_per_mib);
        if atomic_override.is_some_and(|(resource, _)| resource == requirement.resource()) {
            policy = policy
                .with_atomic_components(NonZeroUsize::new(atomic_override.unwrap().1).unwrap());
        }
        policies.insert(requirement.resource(), policy)?;
    }
    Ok(policies)
}

fn component_bytes(
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
    bytes: u64,
) -> Result<ResourceComponentBytes> {
    let bytes = NonZeroU64::new(bytes).ok_or_else(|| anyhow!("zero component bytes"))?;
    let mut components = ResourceComponentBytes::new();
    for requirement in identity.resources() {
        components.insert(requirement.resource(), [bytes])?;
    }
    Ok(components)
}

fn touch_plan(
    managers: &BTreeMap<LogicalResourceId, Arc<BlockManager<G2>>>,
    plan: &BundleOffloadPlan,
    count: usize,
) {
    for child in &plan.resources {
        for (hash, _) in &child.blocks {
            for _ in 0..count {
                managers[&child.resource].block_registry().touch(*hash);
            }
        }
    }
}

fn hash(index: u64) -> SequenceHash {
    (2..=index).fold(SequenceHash::root(1), |parent, block| parent.extend(block))
}

fn resource_hash(value: u64, position: u64) -> SequenceHash {
    SequenceHash::new(value, None, position)
}

async fn wait_until(done: impl Fn() -> bool) {
    for _ in 0..200 {
        if done() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("bundle action did not reach a terminal state");
}

async fn yield_until(done: impl Fn() -> bool) {
    for _ in 0..200 {
        if done() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("bundle task did not make progress");
}

fn assert_counter(
    observability: &KvbmObservability,
    family_name: &str,
    labels: &[(&str, &str)],
    expected: f64,
) {
    assert_eq!(
        counter_value(observability, family_name, labels),
        Some(expected),
        "unexpected {family_name} value for {labels:?}"
    );
}

fn counter_value(
    observability: &KvbmObservability,
    family_name: &str,
    labels: &[(&str, &str)],
) -> Option<f64> {
    let families = observability.registry().gather();
    let family = families
        .iter()
        .find(|family| family.name() == family_name)?;
    family
        .get_metric()
        .iter()
        .find(|metric| {
            labels.iter().all(|(name, value)| {
                metric
                    .get_label()
                    .iter()
                    .any(|label| label.name() == *name && label.value() == *value)
            })
        })
        .map(|metric| metric.get_counter().value())
}

fn assert_histogram_count(
    observability: &KvbmObservability,
    family_name: &str,
    labels: &[(&str, &str)],
    expected: u64,
) {
    let families = observability.registry().gather();
    let family = families
        .iter()
        .find(|family| family.name() == family_name)
        .unwrap_or_else(|| panic!("missing metric family {family_name}"));
    let metric = family
        .get_metric()
        .iter()
        .find(|metric| {
            labels.iter().all(|(name, value)| {
                metric
                    .get_label()
                    .iter()
                    .any(|label| label.name() == *name && label.value() == *value)
            })
        })
        .unwrap_or_else(|| panic!("missing {family_name} series for {labels:?}"));
    assert_eq!(metric.get_histogram().sample_count(), expected);
}
