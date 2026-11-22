// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared loopback fixture for real complete-bundle remote-pull tests.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use futures::future::BoxFuture;
use kvbm_common::{BlockId, LogicalLayoutHandle, LogicalResourceId, SequenceHash};
use kvbm_logical::{BlockManagerSet, BlockRegistry, KvbmSequenceHashProvider};
use kvbm_physical::{TransferOptions, transfer::TransferCompleteNotification};
use kvbm_protocols::cache_manifest::{
    BundleResourceLineage, CacheIdentity, CacheManifest, ModelIdentity, ResourceRequirement,
    ResourceRole,
};
use velo::transports::tcp::TcpTransportBuilder;

use super::super::{BundleAdvertisement, RemoteBundleCandidate};
use crate::G2;
use crate::leader::InstanceLeader;
use crate::object::ObjectBlockOps;
use crate::p2p::session::{MockSessionFactory, SessionFactory};
use crate::testing::managers::TestManagerBuilder;
use crate::testing::token_blocks::create_token_sequence;
use crate::worker::group::ParallelWorkers;
use crate::worker::{
    ConnectRemoteResponse, ImportMetadataResponse, InstanceId, RemoteDescriptor, SerializedLayout,
    SerializedLayoutResponse, Worker, WorkerTransfers,
};

mod timeout;

pub(crate) use timeout::pull_with_hanging_transfer;

pub(crate) const BLOCK_SIZE: usize = 4;
pub(crate) const RESOURCES: [LogicalResourceId; 3] = [
    LogicalResourceId(10),
    LogicalResourceId(11),
    LogicalResourceId(12),
];

pub(crate) struct LoopbackBundleFixture {
    pub(crate) holder: Arc<InstanceLeader>,
    pub(crate) puller: Arc<InstanceLeader>,
    pub(crate) hashes: Arc<[SequenceHash]>,
    pub(crate) lineages: BTreeMap<LogicalResourceId, Vec<SequenceHash>>,
    pub(crate) identity: CacheIdentity,
}

#[cfg(test)]
struct ResourceManagerFixture {
    managers: Arc<BlockManagerSet<G2>>,
    primary_registry: BlockRegistry,
    canonical_hashes: Arc<[SequenceHash]>,
    lineages: BTreeMap<LogicalResourceId, Vec<SequenceHash>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RecordedOnboard {
    pub(crate) resource: LogicalResourceId,
    pub(crate) source_block_ids: Vec<BlockId>,
    pub(crate) destination_block_ids: Vec<BlockId>,
}

pub(crate) struct RecordingParallelWorkers {
    onboards: Mutex<Vec<RecordedOnboard>>,
    checksum_workers: Vec<Arc<dyn Worker>>,
    checksum_mask: u8,
}

impl Default for RecordingParallelWorkers {
    fn default() -> Self {
        Self::with_checksum_mask(0)
    }
}

impl RecordingParallelWorkers {
    fn with_checksum_mask(checksum_mask: u8) -> Self {
        let checksum_worker = Arc::new(Self {
            onboards: Mutex::new(Vec::new()),
            checksum_workers: Vec::new(),
            checksum_mask,
        });
        Self {
            onboards: Mutex::new(Vec::new()),
            checksum_workers: vec![checksum_worker],
            checksum_mask,
        }
    }

    pub(crate) fn onboards(&self) -> Vec<RecordedOnboard> {
        self.onboards.lock().unwrap().clone()
    }
}

impl WorkerTransfers for RecordingParallelWorkers {
    fn execute_local_transfer(
        &self,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("resource-aware transfer required by bundle loopback fixture")
    }

    fn execute_local_transfer_for_resource(
        &self,
        resource: LogicalResourceId,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        source_block_ids: Arc<[BlockId]>,
        destination_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::ensure!(
            src == LogicalLayoutHandle::G2 && dst == LogicalLayoutHandle::G1,
            "loopback onboard expected G2 to G1, got {src:?} to {dst:?}"
        );
        self.onboards.lock().unwrap().push(RecordedOnboard {
            resource,
            source_block_ids: source_block_ids.to_vec(),
            destination_block_ids: destination_block_ids.to_vec(),
        });
        Ok(TransferCompleteNotification::completed())
    }

    fn execute_remote_onboard(
        &self,
        _src: RemoteDescriptor,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote worker onboard is outside the bundle loopback fixture")
    }

    fn execute_remote_offload(
        &self,
        _src: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst: RemoteDescriptor,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote worker offload is outside the bundle loopback fixture")
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
        anyhow::bail!("remote instance onboard is outside the bundle loopback fixture")
    }
}

impl ObjectBlockOps for RecordingParallelWorkers {
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

impl Worker for RecordingParallelWorkers {
    fn compute_host_payload_digests(
        &self,
        resource: LogicalResourceId,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<Vec<kvbm_physical::transfer::PayloadDigest>>> {
        let checksum_mask = self.checksum_mask;
        Box::pin(async move {
            Ok(block_ids
                .into_iter()
                .map(|_| {
                    let mut bytes = [0_u8; 32];
                    bytes[..2].copy_from_slice(&resource.0.to_le_bytes());
                    bytes[31] = checksum_mask;
                    kvbm_physical::transfer::PayloadDigest::from_bytes(bytes)
                })
                .collect())
        })
    }

    fn g1_handle(&self) -> Option<crate::worker::LayoutHandle> {
        None
    }

    fn g2_handle(&self) -> Option<crate::worker::LayoutHandle> {
        None
    }

    fn g3_handle(&self) -> Option<crate::worker::LayoutHandle> {
        None
    }

    fn export_metadata(&self) -> Result<SerializedLayoutResponse> {
        anyhow::bail!("bundle loopback checksum worker has no physical metadata")
    }

    fn import_metadata(&self, _metadata: SerializedLayout) -> Result<ImportMetadataResponse> {
        anyhow::bail!("bundle loopback checksum worker cannot import physical metadata")
    }
}

impl ParallelWorkers for RecordingParallelWorkers {
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
        self.checksum_workers.len()
    }

    fn workers(&self) -> &[Arc<dyn Worker>] {
        &self.checksum_workers
    }
}

impl LoopbackBundleFixture {
    pub(crate) fn candidate(&self) -> RemoteBundleCandidate {
        self.candidate_with_lease(Duration::from_secs(20))
    }

    pub(crate) fn candidate_with_lease(&self, lease: Duration) -> RemoteBundleCandidate {
        let key = kvbm_protocols::cache_manifest::BundleKey::new(
            &self.identity,
            *self.hashes.last().unwrap(),
            8,
        )
        .unwrap();
        let advertisement = BundleAdvertisement::new(
            self.identity.clone(),
            key,
            4,
            self.holder.messenger().instance_id(),
            self.holder.registration_epoch(),
            unix_time_ms() + 30_000,
            self.lineages.iter().map(|(&resource, hashes)| {
                BundleResourceLineage::new(resource, hashes.clone()).unwrap()
            }),
        )
        .unwrap();
        RemoteBundleCandidate::new(
            advertisement,
            uuid::Uuid::new_v4(),
            unix_time_ms() + u64::try_from(lease.as_millis()).unwrap(),
        )
        .unwrap()
    }
}

pub(crate) async fn build_loopback_bundle_fixture(
    omitted_holder_resource: Option<LogicalResourceId>,
    puller_workers: Option<Arc<dyn ParallelWorkers>>,
) -> Result<LoopbackBundleFixture> {
    build_loopback_bundle_fixture_with_mask(omitted_holder_resource, puller_workers, 0).await
}

#[cfg(test)]
pub(crate) async fn build_corrupt_loopback_bundle_fixture() -> Result<LoopbackBundleFixture> {
    build_loopback_bundle_fixture_with_mask(None, None, 1).await
}

async fn build_loopback_bundle_fixture_with_mask(
    omitted_holder_resource: Option<LogicalResourceId>,
    puller_workers: Option<Arc<dyn ParallelWorkers>>,
    puller_checksum_mask: u8,
) -> Result<LoopbackBundleFixture> {
    let holder_velo = new_velo().await;
    let puller_velo = new_velo().await;
    holder_velo.register_peer(puller_velo.peer_info())?;
    puller_velo.register_peer(holder_velo.peer_info())?;
    let (holder_factory, puller_factory) = MockSessionFactory::make_paired();
    let (holder_managers, holder_registry, hashes) = manager_set(true, omitted_holder_resource);
    let holder = Arc::new(
        InstanceLeader::builder()
            .messenger(holder_velo.messenger().clone())
            .registry(holder_registry)
            .g2_manager_set(holder_managers, RESOURCES[0])
            .parallel_worker(Arc::new(RecordingParallelWorkers::default()))
            .build()?,
    );
    holder.set_session_factory(holder_factory as Arc<dyn SessionFactory>);
    let _control = holder.register_control_plane(false, false)?;
    puller_velo
        .messenger()
        .refresh_handlers(holder_velo.instance_id())
        .await?;

    let (puller_managers, puller_registry, _) = manager_set(false, None);
    let mut puller_builder = InstanceLeader::builder()
        .messenger(puller_velo.messenger().clone())
        .registry(puller_registry)
        .g2_manager_set(puller_managers, RESOURCES[0]);
    puller_builder = puller_builder.parallel_worker(puller_workers.unwrap_or_else(|| {
        Arc::new(RecordingParallelWorkers::with_checksum_mask(
            puller_checksum_mask,
        ))
    }));
    let puller = Arc::new(puller_builder.build()?);
    puller.set_session_factory(puller_factory as Arc<dyn SessionFactory>);

    let lineages = BTreeMap::from([
        (RESOURCES[0], hashes.to_vec()),
        (RESOURCES[1], hashes.to_vec()),
        (RESOURCES[2], vec![hashes[1]]),
    ]);
    Ok(LoopbackBundleFixture {
        holder,
        puller,
        hashes,
        lineages,
        identity: manifest(),
    })
}

#[cfg(test)]
pub(crate) async fn build_mixed_native_loopback_bundle_fixture() -> Result<LoopbackBundleFixture> {
    build_asymmetric_native_loopback_bundle_fixture(BLOCK_SIZE, BLOCK_SIZE * 2).await
}

#[cfg(test)]
pub(crate) async fn build_smaller_native_loopback_bundle_fixture() -> Result<LoopbackBundleFixture>
{
    build_asymmetric_native_loopback_bundle_fixture(BLOCK_SIZE * 2, BLOCK_SIZE).await
}

#[cfg(test)]
async fn build_asymmetric_native_loopback_bundle_fixture(
    primary_block_size: usize,
    secondary_block_size: usize,
) -> Result<LoopbackBundleFixture> {
    let holder_velo = new_velo().await;
    let puller_velo = new_velo().await;
    holder_velo.register_peer(puller_velo.peer_info())?;
    puller_velo.register_peer(holder_velo.peer_info())?;
    let (holder_factory, puller_factory) = MockSessionFactory::make_paired();
    let holder_resources =
        asymmetric_native_manager_set(true, primary_block_size, secondary_block_size);
    let holder = Arc::new(
        InstanceLeader::builder()
            .messenger(holder_velo.messenger().clone())
            .registry(holder_resources.primary_registry)
            .g2_manager_set(holder_resources.managers, RESOURCES[0])
            .parallel_worker(Arc::new(RecordingParallelWorkers::default()))
            .build()?,
    );
    holder.set_session_factory(holder_factory as Arc<dyn SessionFactory>);
    let _control = holder.register_control_plane(false, false)?;
    puller_velo
        .messenger()
        .refresh_handlers(holder_velo.instance_id())
        .await?;

    let puller_resources =
        asymmetric_native_manager_set(false, primary_block_size, secondary_block_size);
    let puller = Arc::new(
        InstanceLeader::builder()
            .messenger(puller_velo.messenger().clone())
            .registry(puller_resources.primary_registry)
            .g2_manager_set(puller_resources.managers, RESOURCES[0])
            .parallel_worker(Arc::new(RecordingParallelWorkers::default()))
            .build()?,
    );
    puller.set_session_factory(puller_factory as Arc<dyn SessionFactory>);

    Ok(LoopbackBundleFixture {
        holder,
        puller,
        hashes: holder_resources.canonical_hashes,
        lineages: holder_resources.lineages,
        identity: mixed_native_manifest(primary_block_size, secondary_block_size),
    })
}

pub(crate) fn manifest() -> CacheIdentity {
    CacheManifest::new(
        ModelIdentity::new("bundle-loopback", "v1", [8; 32]).unwrap(),
        "bundle-loopback-v1",
        vec![
            ResourceRequirement::new(RESOURCES[0], ResourceRole::PrefixHistory, 4).unwrap(),
            ResourceRequirement::new(RESOURCES[1], ResourceRole::PrefixHistory, 4).unwrap(),
            ResourceRequirement::new(RESOURCES[2], ResourceRole::BoundaryCapsule, 4).unwrap(),
        ],
        BTreeMap::new(),
    )
    .unwrap()
    .identity()
}

#[cfg(test)]
fn mixed_native_manifest(primary_block_size: usize, secondary_block_size: usize) -> CacheIdentity {
    CacheManifest::new(
        ModelIdentity::new("bundle-loopback-mixed-native", "v1", [9; 32]).unwrap(),
        "bundle-loopback-mixed-native-v1",
        vec![
            ResourceRequirement::new(
                RESOURCES[0],
                ResourceRole::PrefixHistory,
                u32::try_from(primary_block_size).unwrap(),
            )
            .unwrap(),
            ResourceRequirement::new(
                RESOURCES[1],
                ResourceRole::PrefixHistory,
                u32::try_from(secondary_block_size).unwrap(),
            )
            .unwrap(),
            ResourceRequirement::new(
                RESOURCES[2],
                ResourceRole::BoundaryCapsule,
                u32::try_from(primary_block_size).unwrap(),
            )
            .unwrap(),
        ],
        BTreeMap::new(),
    )
    .unwrap()
    .identity()
}

fn manager_set(
    populated: bool,
    omit: Option<LogicalResourceId>,
) -> (Arc<BlockManagerSet<G2>>, BlockRegistry, Arc<[SequenceHash]>) {
    let sequence = create_token_sequence(2, BLOCK_SIZE, 900);
    let hashes = sequence
        .blocks()
        .iter()
        .map(KvbmSequenceHashProvider::kvbm_sequence_hash)
        .collect::<Arc<[_]>>();
    let mut managers = BlockManagerSet::new();
    let mut primary_registry = None;
    for resource in RESOURCES {
        let registry = BlockRegistry::new();
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(16)
                .block_size(BLOCK_SIZE)
                .registry(registry.clone())
                .build(),
        );
        if populated && omit != Some(resource) {
            let blocks = if resource == RESOURCES[2] {
                vec![sequence.blocks()[1].clone()]
            } else {
                sequence.blocks().to_vec()
            };
            let completed = manager
                .allocate_blocks(blocks.len())
                .unwrap()
                .into_iter()
                .zip(blocks)
                .map(|(block, tokens)| block.complete(&tokens).unwrap())
                .collect();
            manager.register_blocks(completed);
        }
        if resource == RESOURCES[0] {
            primary_registry = Some(registry);
        }
        managers.insert(resource, manager).unwrap();
    }
    (Arc::new(managers), primary_registry.unwrap(), hashes)
}

#[cfg(test)]
fn asymmetric_native_manager_set(
    populated: bool,
    primary_block_size: usize,
    secondary_block_size: usize,
) -> ResourceManagerFixture {
    let canonical_block_size = primary_block_size.min(secondary_block_size);
    let canonical = create_token_sequence(8 / canonical_block_size, canonical_block_size, 1_900);
    let canonical_hashes = canonical
        .blocks()
        .iter()
        .map(KvbmSequenceHashProvider::kvbm_sequence_hash)
        .collect::<Arc<[_]>>();
    let primary_hashes = BundleResourceLineage::project_from_canonical(
        RESOURCES[0],
        &canonical_hashes,
        primary_block_size / canonical_block_size,
    )
    .unwrap()
    .hashes()
    .to_vec();
    let secondary_hashes = BundleResourceLineage::project_from_canonical(
        RESOURCES[1],
        &canonical_hashes,
        secondary_block_size / canonical_block_size,
    )
    .unwrap()
    .hashes()
    .to_vec();
    let lineages = BTreeMap::from([
        (RESOURCES[0], primary_hashes),
        (RESOURCES[1], secondary_hashes),
        (RESOURCES[2], vec![*canonical_hashes.last().unwrap()]),
    ]);
    let mut managers = BlockManagerSet::new();
    let mut primary_registry = None;
    for resource in RESOURCES {
        let registry = BlockRegistry::new();
        let block_size = if resource == RESOURCES[1] {
            secondary_block_size
        } else {
            primary_block_size
        };
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(16)
                .block_size(block_size)
                .registry(registry.clone())
                .build(),
        );
        if populated {
            let hashes = &lineages[&resource];
            let completed = manager
                .allocate_blocks(hashes.len())
                .unwrap()
                .into_iter()
                .zip(hashes.iter().copied())
                .map(|(block, hash)| block.stage(hash, manager.block_size()).unwrap())
                .collect();
            manager.register_blocks(completed);
        }
        if resource == RESOURCES[0] {
            primary_registry = Some(registry);
        }
        managers.insert(resource, manager).unwrap();
    }
    ResourceManagerFixture {
        managers: Arc::new(managers),
        primary_registry: primary_registry.unwrap(),
        canonical_hashes,
        lineages,
    }
}

async fn new_velo() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_transport())
        .build()
        .await
        .map(Arc::new)
        .unwrap()
}

fn new_transport() -> Arc<velo::transports::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

fn unix_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}
