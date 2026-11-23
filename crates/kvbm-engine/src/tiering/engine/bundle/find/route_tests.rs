// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::disallowed_macros)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::Result;
use futures::future::BoxFuture;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::{BlockRegistry, ImmutableBlock};
use kvbm_protocols::cache_manifest::{
    BundleKey, BundleResourceLineage, CacheIdentity, CacheManifest, ModelIdentity,
    RegistrationEpoch, ResourceRequirement, ResourceRole,
};
use kvbm_protocols::connector::{
    ActionId, CacheScope, FindBlocksHandle, FindBlocksOutcome, FindBlocksRequest, LeaderEngine,
    LeaderEngineError, LocalPrefillEstimate, NoopWorkerSink,
};
use kvbm_protocols::disagg::{
    BundlePrefillContext, RemotePrefillParams, SessionEndpoint, TransferParams,
};

use crate::G2;
use crate::InstanceId;
use crate::leader::InstanceLeader;
use crate::leader::{RemoteBlockDiscovery, RemoteCandidates};
use crate::p2p::session::{MockSessionFactory, SessionFactory};
use crate::remote::cd::DisaggConfig;
use crate::remote::cd::budget::TierCell;
use crate::remote::cd::policy::SelectionPolicy;
use crate::remote::cd::wire::{PrefillDispatch, PrefillPlane};
use crate::remote::search::bundle::test_support::{
    BLOCK_SIZE as LOOPBACK_BLOCK_SIZE, RESOURCES as LOOPBACK_RESOURCES, RecordingParallelWorkers,
    build_loopback_bundle_fixture,
};
use crate::remote::search::bundle::{
    BundleAdvertisement, BundleDiscoveryOutcome, BundleDiscoveryQuery, BundleMissReason,
    RemoteBundleCandidate,
};
use crate::testing::managers::TestManagerBuilder;
use crate::testing::messenger::create_messenger_tcp;
use crate::testing::token_blocks::create_token_sequence;
use crate::tiering::engine::inflight::InflightKey;
use crate::tiering::engine::local::{CdRuntime, LocalConnectorEngine};
use crate::tiering::engine::offload::DisabledOffloadSubmit;

const BLOCK_SIZE: usize = 4;
const RESOURCES: [LogicalResourceId; 2] = [LogicalResourceId(90), LogicalResourceId(91)];

struct FindRig {
    engine: Arc<LocalConnectorEngine>,
    hashes: Vec<SequenceHash>,
    held: Vec<ImmutableBlock<G2>>,
}

async fn find_rig(start: u32) -> Result<FindRig> {
    find_rig_with_remote(start, None).await
}

async fn find_rig_with_remote(
    start: u32,
    remote: Option<Arc<dyn RemoteBlockDiscovery>>,
) -> Result<FindRig> {
    find_rig_with_remote_threshold(start, remote, 1).await
}

async fn find_rig_with_remote_threshold(
    start: u32,
    remote: Option<Arc<dyn RemoteBlockDiscovery>>,
    min_remote_blocks: usize,
) -> Result<FindRig> {
    let messenger = create_messenger_tcp().await?;
    let registry = BlockRegistry::new();
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(7)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    let sequence = create_token_sequence(3, BLOCK_SIZE, start);
    let complete = manager
        .allocate_blocks(3)
        .unwrap()
        .into_iter()
        .zip(sequence.blocks())
        .map(|(block, tokens)| block.complete(tokens).unwrap())
        .collect();
    let held = manager.register_blocks(complete);
    let hashes = held.iter().map(ImmutableBlock::sequence_hash).collect();
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry)
            .g2_manager(manager)
            .min_remote_blocks(min_remote_blocks)
            .build()?,
    );
    if let Some(remote) = remote.as_ref() {
        assert!(leader.set_remote_discovery(Arc::clone(remote)));
    }
    Ok(FindRig {
        engine: LocalConnectorEngine::new(
            leader,
            NoopWorkerSink::new(),
            BLOCK_SIZE,
            remote.is_some(),
        ),
        hashes,
        held,
    })
}

struct MissingBundleDirectory {
    queries: Arc<std::sync::Mutex<Vec<BundleDiscoveryQuery>>>,
}

struct FailingThenMissingBundleDirectory {
    queries: Arc<std::sync::Mutex<Vec<BundleDiscoveryQuery>>>,
    first: std::sync::Mutex<Option<RemoteBundleCandidate>>,
}

struct BlockingBundleDirectory {
    started: Arc<AtomicBool>,
    release: Arc<tokio::sync::Notify>,
    candidate: RemoteBundleCandidate,
}

#[derive(Default)]
struct CountingPrefillPlane(AtomicUsize);

impl PrefillPlane for CountingPrefillPlane {
    fn dispatch(&self, _req: PrefillDispatch) -> BoxFuture<'static, Result<()>> {
        self.0.fetch_add(1, Ordering::AcqRel);
        Box::pin(async { Ok(()) })
    }
}

impl RemoteBlockDiscovery for BlockingBundleDirectory {
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
        let started = Arc::clone(&self.started);
        let release = Arc::clone(&self.release);
        let candidate = self.candidate.clone();
        Box::pin(async move {
            let mut released = std::pin::pin!(release.notified());
            // Register before the test observes started and calls notify_waiters.
            released.as_mut().enable();
            started.store(true, Ordering::Release);
            released.await;
            Ok(BundleDiscoveryOutcome::Hit(Box::new(candidate)))
        })
    }
}

impl RemoteBlockDiscovery for FailingThenMissingBundleDirectory {
    fn discover(
        &self,
        _hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        Box::pin(async { Ok(None) })
    }

    fn discover_bundle(
        &self,
        query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        self.queries.lock().unwrap().push(query);
        let outcome = self
            .first
            .lock()
            .unwrap()
            .take()
            .map(Box::new)
            .map(BundleDiscoveryOutcome::Hit)
            .unwrap_or(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound));
        Box::pin(async move { Ok(outcome) })
    }
}

impl RemoteBlockDiscovery for MissingBundleDirectory {
    fn discover(
        &self,
        _hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        Box::pin(async { Ok(None) })
    }

    fn discover_bundle(
        &self,
        query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        self.queries.lock().unwrap().push(query);
        Box::pin(async { Ok(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound)) })
    }
}

fn manifest(revision: &str, resources: &[LogicalResourceId]) -> CacheManifest {
    CacheManifest::new(
        ModelIdentity::new("bundle-router-test", revision, [4; 32]).unwrap(),
        "bundle-router-v1",
        resources
            .iter()
            .enumerate()
            .map(|(index, &resource)| {
                ResourceRequirement::new(
                    resource,
                    if index + 1 == resources.len() {
                        ResourceRole::BoundaryCapsule
                    } else {
                        ResourceRole::PrefixHistory
                    },
                    BLOCK_SIZE as u32,
                )
                .unwrap()
            })
            .collect(),
        Default::default(),
    )
    .unwrap()
}

fn advertised_lineages(
    identity: &CacheIdentity,
    key: BundleKey,
    primary_hashes: &[SequenceHash],
) -> Vec<BundleResourceLineage> {
    identity
        .resources()
        .iter()
        .map(|requirement| {
            let hashes = match requirement.role() {
                ResourceRole::PrefixHistory => {
                    let count = usize::try_from(
                        key.boundary_tokens() / u64::from(requirement.native_block_tokens().get()),
                    )
                    .unwrap();
                    primary_hashes[..count].to_vec()
                }
                ResourceRole::BoundaryCapsule => vec![key.boundary_hash()],
            };
            BundleResourceLineage::new(requirement.resource(), hashes).unwrap()
        })
        .collect()
}

fn find_request(
    request_id: &str,
    identity: CacheIdentity,
    hashes: Vec<SequenceHash>,
) -> FindBlocksRequest {
    FindBlocksRequest {
        request_id: request_id.to_owned(),
        cache: CacheScope::Manifest(identity),
        sequence_hashes: Arc::from(hashes),
        num_computed_tokens: 0,
        total_tokens: 3 * BLOCK_SIZE + 1,
        transfer_params: None,
        local_prefill_estimate: None,
    }
}

fn with_bundle_context(
    mut request: FindBlocksRequest,
    context_identity: &CacheIdentity,
    target: BundleKey,
) -> FindBlocksRequest {
    let mut params = RemotePrefillParams::new(uuid::Uuid::new_v4(), InstanceId::new_v4());
    params.bundle = Some(
        BundlePrefillContext::new(
            context_identity.manifest(),
            context_identity
                .resources()
                .iter()
                .map(ResourceRequirement::resource),
            None,
            target,
            1,
        )
        .unwrap(),
    );
    request.transfer_params = Some(TransferParams::remote_prefill(params));
    request
}

fn commit(rig: &FindRig, identity: &CacheIdentity) -> BundleKey {
    commit_at(rig, identity, 2)
}

fn commit_at(rig: &FindRig, identity: &CacheIdentity, boundary_blocks: usize) -> BundleKey {
    let key = BundleKey::new(
        identity,
        rig.hashes[boundary_blocks - 1],
        (boundary_blocks * BLOCK_SIZE) as u64,
    )
    .unwrap();
    rig.engine
        .bundle_catalog
        .lock()
        .unwrap()
        .index_mut()
        .commit(
            identity,
            key,
            1,
            RESOURCES
                .into_iter()
                .enumerate()
                .map(|(index, resource)| (resource, vec![rig.held[index].clone()])),
        )
        .unwrap();
    key
}

fn resolved(outcome: FindBlocksOutcome) -> (usize, Option<FindBlocksHandle>, bool) {
    match outcome {
        FindBlocksOutcome::Resolved {
            matched_tokens,
            minted,
            release_parked,
        } => (matched_tokens, minted, release_parked),
        other => panic!("expected resolved bundle find, got {other:?}"),
    }
}

#[tokio::test]
async fn bundle_prefill_rejects_a_context_for_another_manifest() -> Result<()> {
    let rig = find_rig(4_250).await?;
    let identity = manifest("request", &RESOURCES).identity();
    let foreign = manifest("foreign", &RESOURCES).identity();
    let target = BundleKey::new(&foreign, rig.hashes[1], (2 * BLOCK_SIZE) as u64)?;
    let request = with_bundle_context(
        find_request("manifest-mismatch", identity, rig.hashes.clone()),
        &foreign,
        target,
    );

    assert!(matches!(
        rig.engine.clone().find_blocks(&request, None),
        Err(LeaderEngineError::InvalidPrefillRequest { .. })
    ));
    Ok(())
}

#[tokio::test]
async fn remote_prefill_rejects_unknown_protocol_before_lifecycle_routing() -> Result<()> {
    let rig = find_rig(4_260).await?;
    let identity = manifest("bad-protocol", &RESOURCES).identity();
    let mut request = find_request("bad-protocol", identity, rig.hashes.clone());
    let mut params = RemotePrefillParams::new(uuid::Uuid::new_v4(), InstanceId::new_v4());
    params.protocol_version = u16::MAX;
    request.transfer_params = Some(TransferParams::remote_prefill(params));

    assert!(matches!(
        rig.engine.clone().find_blocks(&request, None),
        Err(LeaderEngineError::InvalidPrefillRequest { reason })
            if reason.contains("protocol version")
    ));
    Ok(())
}

#[tokio::test]
async fn manifest_remote_prefill_rejects_missing_bundle_before_lifecycle_routing() -> Result<()> {
    let rig = find_rig(4_262).await?;
    let identity = manifest("missing-bundle", &RESOURCES).identity();
    let mut request = find_request("missing-bundle", identity, rig.hashes.clone());
    let mut params = RemotePrefillParams::new(uuid::Uuid::new_v4(), InstanceId::new_v4());
    params.decode_endpoint = Some(SessionEndpoint {
        kind: "test".to_owned(),
        payload: serde_json::Value::Null,
    });
    request.transfer_params = Some(TransferParams::remote_prefill(params));

    assert!(matches!(
        rig.engine.clone().find_blocks(&request, None),
        Err(LeaderEngineError::InvalidPrefillRequest { reason })
            if reason.contains("requires bundle metadata")
    ));
    Ok(())
}

#[tokio::test]
async fn unitary_remote_prefill_rejects_a_full_prompt_window_before_lifecycle_routing() -> Result<()>
{
    let rig = find_rig(4_263).await?;
    let identity = manifest("full-window", &RESOURCES).identity();
    let mut request = find_request("full-window", identity, rig.hashes.clone());
    request.cache = CacheScope::LegacyPrimary;
    request.total_tokens = 2 * BLOCK_SIZE;
    let mut params = RemotePrefillParams::new(uuid::Uuid::new_v4(), InstanceId::new_v4());
    params.decode_endpoint = Some(SessionEndpoint {
        kind: "test".to_owned(),
        payload: serde_json::Value::Null,
    });
    params.num_provided_tokens = request.total_tokens;
    request.transfer_params = Some(TransferParams::remote_prefill(params));

    assert!(matches!(
        rig.engine.clone().find_blocks(&request, None),
        Err(LeaderEngineError::InvalidPrefillRequest { reason })
            if reason.contains("must leave one prompt token")
    ));
    Ok(())
}

#[tokio::test]
async fn remote_prefill_rejects_ragged_provided_window_before_lifecycle_routing() -> Result<()> {
    let rig = find_rig(4_265).await?;
    let identity = manifest("ragged-window", &RESOURCES).identity();
    let mut request = find_request("ragged-window", identity, rig.hashes.clone());
    let mut params = RemotePrefillParams::new(uuid::Uuid::new_v4(), InstanceId::new_v4());
    params.num_provided_tokens = BLOCK_SIZE + 1;
    request.transfer_params = Some(TransferParams::remote_prefill(params));

    assert!(matches!(
        rig.engine.clone().find_blocks(&request, None),
        Err(LeaderEngineError::InvalidPrefillRequest { reason })
            if reason.contains("not aligned")
    ));
    Ok(())
}

#[tokio::test]
async fn bundle_prefill_rejects_a_nonempty_unitary_provided_window() -> Result<()> {
    let rig = find_rig(4_270).await?;
    let identity = manifest("bundle-window", &RESOURCES).identity();
    let target = BundleKey::new(&identity, rig.hashes[1], (2 * BLOCK_SIZE) as u64)?;
    let mut request = with_bundle_context(
        find_request("bundle-window", identity.clone(), rig.hashes.clone()),
        &identity,
        target,
    );
    request
        .transfer_params
        .as_mut()
        .and_then(|transfer| transfer.remote_prefill.as_mut())
        .expect("bundle params")
        .num_provided_tokens = BLOCK_SIZE;

    assert!(matches!(
        rig.engine.clone().find_blocks(&request, None),
        Err(LeaderEngineError::InvalidPrefillRequest { reason })
            if reason.contains("provided window must be zero")
    ));
    Ok(())
}

#[tokio::test]
async fn bundle_prefill_rejects_a_target_outside_the_request_chain() -> Result<()> {
    let rig = find_rig(4_275).await?;
    let identity = manifest("wrong-target", &RESOURCES).identity();
    let target = BundleKey::new(
        &identity,
        SequenceHash::new(99, None, 99),
        (2 * BLOCK_SIZE) as u64,
    )?;
    let request = with_bundle_context(
        find_request("wrong-target", identity.clone(), rig.hashes.clone()),
        &identity,
        target,
    );

    assert!(matches!(
        rig.engine.clone().find_blocks(&request, None),
        Err(LeaderEngineError::InvalidPrefillRequest { .. })
    ));
    Ok(())
}

#[tokio::test]
async fn valid_bundle_prefill_context_uses_the_normal_bundle_find_path() -> Result<()> {
    let rig = find_rig(4_300).await?;
    let identity = manifest("valid-prefill", &RESOURCES).identity();
    let target = commit(&rig, &identity);
    let request = with_bundle_context(
        find_request("valid-prefill", identity.clone(), rig.hashes.clone()),
        &identity,
        target,
    );

    let (matched, minted, _) = resolved(rig.engine.clone().find_blocks(&request, None)?);
    assert_eq!(matched, 2 * BLOCK_SIZE);
    assert!(minted.is_some());
    Ok(())
}

#[tokio::test]
async fn bundle_prefill_never_selects_a_bundle_before_its_initial_boundary() -> Result<()> {
    let rig = find_rig(4_310).await?;
    let identity = manifest("prefill-floor", &RESOURCES).identity();
    let lower = commit_at(&rig, &identity, 1);
    let initial = BundleKey::new(&identity, rig.hashes[1], (2 * BLOCK_SIZE) as u64)?;
    let target = BundleKey::new(&identity, rig.hashes[2], (3 * BLOCK_SIZE) as u64)?;
    let mut request = find_request("prefill-floor", identity.clone(), rig.hashes.clone());
    let mut params = RemotePrefillParams::new(uuid::Uuid::new_v4(), InstanceId::new_v4());
    params.bundle = Some(BundlePrefillContext::new(
        identity.manifest(),
        identity
            .resources()
            .iter()
            .map(ResourceRequirement::resource),
        Some(initial),
        target,
        1,
    )?);
    request.transfer_params = Some(TransferParams::remote_prefill(params));

    let (matched, _, _) = resolved(rig.engine.clone().find_blocks(&request, None)?);
    assert_eq!(lower.boundary_tokens(), BLOCK_SIZE as u64);
    assert_eq!(matched, 0, "a bundle below initial B must not be selected");
    Ok(())
}

#[tokio::test]
async fn bundle_prefill_can_restore_the_exact_initial_bundle_before_computing_target() -> Result<()>
{
    let rig = find_rig(4_315).await?;
    let identity = manifest("prefill-initial", &RESOURCES).identity();
    let initial = commit_at(&rig, &identity, 2);
    let target = BundleKey::new(&identity, rig.hashes[2], (3 * BLOCK_SIZE) as u64)?;
    let mut request = find_request("prefill-initial", identity.clone(), rig.hashes.clone());
    let mut params = RemotePrefillParams::new(uuid::Uuid::new_v4(), InstanceId::new_v4());
    params.bundle = Some(BundlePrefillContext::new(
        identity.manifest(),
        identity
            .resources()
            .iter()
            .map(ResourceRequirement::resource),
        Some(initial),
        target,
        1,
    )?);
    request.transfer_params = Some(TransferParams::remote_prefill(params));

    let (matched, minted, _) = resolved(rig.engine.clone().find_blocks(&request, None)?);
    assert_eq!(matched, 2 * BLOCK_SIZE);
    assert!(minted.is_some());

    let published = commit_at(&rig, &identity, 3);
    assert_eq!(published, target);
    let follow_up = find_request("prefill-published-b2", identity.clone(), rig.hashes.clone());
    let (matched, _, _) = resolved(rig.engine.clone().find_blocks(&follow_up, None)?);
    assert_eq!(
        matched,
        3 * BLOCK_SIZE,
        "published B2 must use the normal bundle path"
    );
    Ok(())
}

#[tokio::test]
async fn bundle_prefill_rejects_an_incomplete_resource_set() -> Result<()> {
    let rig = find_rig(4_325).await?;
    let identity = manifest("incomplete-resources", &RESOURCES).identity();
    let target = BundleKey::new(&identity, rig.hashes[1], (2 * BLOCK_SIZE) as u64)?;
    let mut request = find_request("incomplete-resources", identity.clone(), rig.hashes.clone());
    let mut params = RemotePrefillParams::new(uuid::Uuid::new_v4(), InstanceId::new_v4());
    params.bundle = Some(BundlePrefillContext::new(
        identity.manifest(),
        [RESOURCES[0]],
        None,
        target,
        1,
    )?);
    request.transfer_params = Some(TransferParams::remote_prefill(params));

    assert!(matches!(
        rig.engine.clone().find_blocks(&request, None),
        Err(LeaderEngineError::InvalidPrefillRequest { .. })
    ));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_window_below_remote_threshold_never_queries_the_directory() -> Result<()> {
    let queries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let directory = Arc::new(MissingBundleDirectory {
        queries: Arc::clone(&queries),
    });
    let rig = find_rig_with_remote_threshold(4_290, Some(directory), 4).await?;
    let identity = manifest("below-threshold", &RESOURCES).identity();
    let request = find_request("below-threshold", identity, rig.hashes.clone());

    let (matched, minted, release) = resolved(rig.engine.clone().find_blocks(&request, None)?);

    assert_eq!(matched, 0);
    assert!(minted.is_none());
    assert!(!release);
    tokio::task::yield_now().await;
    assert!(queries.lock().unwrap().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bundle_window_at_remote_threshold_queries_the_directory() -> Result<()> {
    let queries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let directory = Arc::new(MissingBundleDirectory {
        queries: Arc::clone(&queries),
    });
    let rig = find_rig_with_remote_threshold(4_295, Some(directory), 3).await?;
    let identity = manifest("exact-threshold", &RESOURCES).identity();
    let request = find_request("exact-threshold", identity.clone(), rig.hashes.clone());

    let FindBlocksOutcome::Searching { minted: Some(live) } =
        rig.engine.clone().find_blocks(&request, None)?
    else {
        panic!("an exact-threshold bundle window must start remote discovery");
    };
    loop {
        tokio::task::yield_now().await;
        if !matches!(
            rig.engine.clone().find_blocks(&request, Some(&live))?,
            FindBlocksOutcome::Searching { .. }
        ) {
            break;
        }
    }

    let queries = queries.lock().unwrap();
    assert_eq!(queries.len(), 1);
    assert_eq!(queries[0].identity(), &identity);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_bundle_miss_is_async_then_releases_the_search() -> Result<()> {
    let queries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let directory = Arc::new(MissingBundleDirectory {
        queries: Arc::clone(&queries),
    });
    let rig = find_rig_with_remote(4_300, Some(directory)).await?;
    let identity = manifest("remote", &RESOURCES).identity();
    let request = find_request("remote-miss", identity.clone(), rig.hashes.clone());

    let FindBlocksOutcome::Searching { minted: Some(live) } =
        rig.engine.clone().find_blocks(&request, None)?
    else {
        panic!("remote directory lookup must start asynchronously");
    };
    let outcome = loop {
        tokio::task::yield_now().await;
        let outcome = rig.engine.clone().find_blocks(&request, Some(&live))?;
        if !matches!(outcome, FindBlocksOutcome::Searching { .. }) {
            break outcome;
        }
    };
    let (matched, minted, release) = resolved(outcome);
    assert_eq!(matched, 0);
    assert!(minted.is_none());
    assert!(release);
    let queries = queries.lock().unwrap();
    assert_eq!(queries.len(), 1);
    assert_eq!(queries[0].identity(), &identity);
    assert!(!queries[0].candidates().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn real_remote_hit_refreshes_catalog_and_onboards_exact_destinations() -> Result<()> {
    let workers = Arc::new(RecordingParallelWorkers::default());
    let fixture = build_loopback_bundle_fixture(
        None,
        Some(Arc::clone(&workers) as Arc<dyn crate::worker::group::ParallelWorkers>),
    )
    .await?;
    let candidate = fixture.candidate();
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(tokio::sync::Notify::new());
    let directory = Arc::new(BlockingBundleDirectory {
        started: Arc::clone(&started),
        release: Arc::clone(&release),
        candidate,
    });
    assert!(fixture.puller.set_remote_discovery(directory));
    let plane = Arc::new(CountingPrefillPlane::default());
    let sessions: Arc<dyn SessionFactory> = MockSessionFactory::new();
    let cd = CdRuntime::new(
        DisaggConfig {
            selection: SelectionPolicy::Always,
            ..DisaggConfig::default()
        },
        Arc::new(TierCell::default()),
        sessions,
        Arc::clone(&plane) as Arc<dyn PrefillPlane>,
        None,
    );
    let engine = LocalConnectorEngine::with_offload_submit(
        Arc::clone(&fixture.puller),
        NoopWorkerSink::new(),
        LOOPBACK_BLOCK_SIZE,
        true,
        Arc::new(DisabledOffloadSubmit),
        Some(cd),
    );
    let local_seed = BundleKey::new(
        &fixture.identity,
        fixture.hashes[0],
        LOOPBACK_BLOCK_SIZE as u64,
    )?;
    engine.bundle_catalog.lock().unwrap().index_mut().commit(
        &fixture.identity,
        local_seed,
        1,
        fixture
            .identity
            .resources()
            .iter()
            .map(|requirement| (requirement.resource(), Vec::new())),
    )?;
    let request = FindBlocksRequest {
        request_id: "real-remote-hit".to_owned(),
        cache: CacheScope::Manifest(fixture.identity.clone()),
        sequence_hashes: Arc::clone(&fixture.hashes),
        num_computed_tokens: 0,
        total_tokens: 2 * LOOPBACK_BLOCK_SIZE + 1,
        transfer_params: None,
        local_prefill_estimate: Some(LocalPrefillEstimate::from_rate(
            std::time::Duration::from_secs(60),
            1,
        )),
    };

    let FindBlocksOutcome::Searching {
        minted: Some(search),
    } = engine.clone().find_blocks(&request, None)?
    else {
        anyhow::bail!("a local B seed must still search remotely for B2")
    };
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    release.notify_waiters();

    let terminal = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let outcome = engine.clone().find_blocks(&request, Some(&search))?;
            if !matches!(outcome, FindBlocksOutcome::Searching { .. }) {
                return Ok::<_, LeaderEngineError>(outcome);
            }
            tokio::task::yield_now().await;
        }
    })
    .await??;
    let (matched_tokens, minted, release_parked) = resolved(terminal);
    assert_eq!(matched_tokens, 2 * LOOPBACK_BLOCK_SIZE);
    assert!(!release_parked);
    assert_eq!(
        plane.0.load(Ordering::Acquire),
        0,
        "an exact remote B2 hit must suppress conditional-prefill dispatch"
    );
    assert!(
        minted.is_none(),
        "catalog refresh must preserve the original parked search handle"
    );
    let key = BundleKey::new(
        &fixture.identity,
        fixture.hashes[1],
        (2 * LOOPBACK_BLOCK_SIZE) as u64,
    )?;
    assert!(
        engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&fixture.identity, &key)
            .is_some(),
        "the remote hit must refresh the ordinary local bundle catalog"
    );

    let destinations = vec![
        kvbm_protocols::connector::ResourceDestination {
            resource: LOOPBACK_RESOURCES[0],
            block_ids: vec![100, 101],
        },
        kvbm_protocols::connector::ResourceDestination {
            resource: LOOPBACK_RESOURCES[1],
            block_ids: vec![200, 201],
        },
        kvbm_protocols::connector::ResourceDestination {
            resource: LOOPBACK_RESOURCES[2],
            block_ids: vec![300],
        },
    ];
    let onboard = engine
        .clone()
        .onboard_bundle(&search, destinations.clone(), matched_tokens)?;
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !onboard.is_complete() {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    assert_eq!(
        onboard.outcome(),
        Some(kvbm_protocols::connector::LoadOutcome::Done)
    );

    let recorded = workers.onboards();
    assert_eq!(recorded.len(), destinations.len());
    for destination in destinations {
        let transfer = recorded
            .iter()
            .find(|transfer| transfer.resource == destination.resource)
            .expect("every resource must use the common onboard path");
        assert_eq!(transfer.destination_block_ids, destination.block_ids);
        assert_eq!(
            transfer.source_block_ids.len(),
            transfer.destination_block_ids.len()
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_pending_search_cancels_discovery_before_late_publication() -> Result<()> {
    let identity = manifest("drop-cancel", &RESOURCES).identity();
    let sequence = create_token_sequence(3, BLOCK_SIZE, 4_310);
    let hashes = sequence
        .blocks()
        .iter()
        .map(kvbm_logical::KvbmSequenceHashProvider::kvbm_sequence_hash)
        .collect::<Vec<_>>();
    let key = BundleKey::new(&identity, hashes[1], (2 * BLOCK_SIZE) as u64)?;
    let advertisement = BundleAdvertisement::new(
        identity.clone(),
        key,
        1,
        InstanceId::new_v4(),
        RegistrationEpoch::new(),
        unix_time_ms() + 30_000,
        advertised_lineages(&identity, key, &hashes),
    )?;
    let candidate =
        RemoteBundleCandidate::new(advertisement, uuid::Uuid::new_v4(), unix_time_ms() + 20_000)?;
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(tokio::sync::Notify::new());
    let directory = Arc::new(BlockingBundleDirectory {
        started: Arc::clone(&started),
        release: Arc::clone(&release),
        candidate,
    });
    let rig = find_rig_with_remote(4_310, Some(directory)).await?;
    let request = find_request("drop-cancel", identity.clone(), rig.hashes.clone());

    let FindBlocksOutcome::Searching { minted: Some(live) } =
        rig.engine.clone().find_blocks(&request, None)?
    else {
        panic!("remote directory lookup must start asynchronously");
    };
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    drop(live);
    assert!(rig.engine.bundle_searches.is_empty());
    tokio::task::yield_now().await;
    release.notify_waiters();
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    assert!(
        rig.engine
            .bundle_catalog
            .lock()
            .unwrap()
            .lease_exact(&identity, &key)
            .is_none(),
        "a released search must not commit a late directory result"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_deep_remote_pull_fails_closed_without_retrying() -> Result<()> {
    let identity = manifest("remote-retry", &RESOURCES).identity();
    let hashes = create_token_sequence(3, BLOCK_SIZE, 4_325)
        .blocks()
        .iter()
        .map(kvbm_logical::KvbmSequenceHashProvider::kvbm_sequence_hash)
        .collect::<Vec<_>>();
    let deep = BundleKey::new(&identity, hashes[2], 3 * BLOCK_SIZE as u64)?;
    let advertisement = BundleAdvertisement::new(
        identity.clone(),
        deep,
        1,
        InstanceId::new_v4(),
        RegistrationEpoch::new(),
        unix_time_ms() + 30_000,
        advertised_lineages(&identity, deep, &hashes),
    )?;
    let candidate =
        RemoteBundleCandidate::new(advertisement, uuid::Uuid::new_v4(), unix_time_ms() + 20_000)?;
    let queries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let directory = Arc::new(FailingThenMissingBundleDirectory {
        queries: Arc::clone(&queries),
        first: std::sync::Mutex::new(Some(candidate)),
    });
    let rig = find_rig_with_remote(4_325, Some(directory)).await?;
    let request = find_request("remote-retry", identity, rig.hashes.clone());

    let FindBlocksOutcome::Searching { minted: Some(live) } =
        rig.engine.clone().find_blocks(&request, None)?
    else {
        panic!("remote directory lookup must start asynchronously");
    };
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            tokio::task::yield_now().await;
            if !matches!(
                rig.engine.clone().find_blocks(&request, Some(&live))?,
                FindBlocksOutcome::Searching { .. }
            ) {
                return Ok::<(), LeaderEngineError>(());
            }
        }
    })
    .await
    .expect("remote failure should terminate")?;

    let queries = queries.lock().unwrap();
    assert_eq!(queries.len(), 1);
    assert_eq!(queries[0].candidates()[0], deep);
    Ok(())
}

fn unix_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn changed_window_reconciles_to_one_new_common_boundary() -> Result<()> {
    let rig = find_rig(4_350).await?;
    let identity = manifest("v1", &RESOURCES).identity();
    commit_at(&rig, &identity, 2);
    let later = commit_at(&rig, &identity, 3);
    let mut initial = find_request("rq", identity.clone(), rig.hashes.clone());
    initial.total_tokens = 3 * BLOCK_SIZE;

    let (matched, minted, _) = resolved(rig.engine.clone().find_blocks(&initial, None)?);
    assert_eq!(
        matched,
        2 * BLOCK_SIZE,
        "aligned total excludes block three"
    );
    let live = minted.unwrap();

    let mut changed = find_request("rq", identity, rig.hashes.clone());
    changed.num_computed_tokens = 2 * BLOCK_SIZE;
    let (matched, minted, release) =
        resolved(rig.engine.clone().find_blocks(&changed, Some(&live))?);
    assert_eq!(matched, BLOCK_SIZE);
    assert!(minted.is_none());
    assert!(!release);
    assert_eq!(
        rig.engine
            .bundle_searches
            .get(&live.search_id().unwrap())
            .unwrap()
            .lease()
            .unwrap()
            .key(),
        &later
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repoll_keeps_exact_lease_but_cross_request_needs_index_visibility() -> Result<()> {
    let rig = find_rig(4_400).await?;
    let identity = manifest("v1", &RESOURCES).identity();
    let key = commit(&rig, &identity);
    let request = find_request("same-request", identity.clone(), rig.hashes.clone());

    let (matched, minted, _) = resolved(rig.engine.clone().find_blocks(&request, None)?);
    assert_eq!(matched, 2 * BLOCK_SIZE);
    let live = minted.expect("bundle hit mints the existing handle type");
    assert_eq!(rig.engine.bundle_searches.len(), 1);

    rig.engine
        .bundle_catalog
        .lock()
        .unwrap()
        .invalidate_key(key);
    let (matched, minted, release) =
        resolved(rig.engine.clone().find_blocks(&request, Some(&live))?);
    assert_eq!(matched, 2 * BLOCK_SIZE, "the parked lease stays stable");
    assert!(minted.is_none(), "identical repolls reuse the handle");
    assert!(!release);

    let cross = find_request("cross-request", identity, rig.hashes.clone());
    let (matched, minted, _) = resolved(rig.engine.clone().find_blocks(&cross, None)?);
    assert_eq!(matched, 0, "removed bundles are not cross-request hits");
    assert!(minted.is_none());
    drop(live);
    assert!(rig.engine.bundle_searches.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn changed_manifest_or_resource_set_desyncs() -> Result<()> {
    let rig = find_rig(4_500).await?;
    let identity = manifest("v1", &RESOURCES).identity();
    commit(&rig, &identity);
    let request = find_request("rq", identity, rig.hashes.clone());
    let (_, minted, _) = resolved(rig.engine.clone().find_blocks(&request, None)?);
    let live = minted.unwrap();

    for changed in [
        manifest("v2", &RESOURCES).identity(),
        manifest("v1", &[RESOURCES[0], LogicalResourceId(92)]).identity(),
    ] {
        let changed_request = find_request("rq", changed, rig.hashes.clone());
        assert!(matches!(
            rig.engine
                .clone()
                .find_blocks(&changed_request, Some(&live)),
            Err(LeaderEngineError::FindBlocksDesync)
        ));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overlapping_onboard_defers_bundle_find() -> Result<()> {
    let rig = find_rig(4_600).await?;
    let identity = manifest("v1", &RESOURCES).identity();
    commit(&rig, &identity);
    let inflight = InflightKey::Action(ActionId::new());
    rig.engine
        .inflight
        .lock()
        .unwrap()
        .record(inflight, vec![rig.hashes[0]]);

    let request = find_request("rq", identity, rig.hashes.clone());
    assert!(matches!(
        rig.engine.clone().find_blocks(&request, None)?,
        FindBlocksOutcome::Deferred
    ));
    assert!(rig.engine.bundle_searches.is_empty());
    rig.engine.inflight.lock().unwrap().clear(&inflight);
    Ok(())
}
