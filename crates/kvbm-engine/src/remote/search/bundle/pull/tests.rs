// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use futures::future::BoxFuture;
use kvbm_observability::BundleMetrics;
use kvbm_physical::transfer::TransferCompleteNotification;
use kvbm_protocols::cache_manifest::{
    BundleResourceLineage, CacheManifest, CacheManifestId, ModelIdentity, RegistrationEpoch,
    ResourceRole,
};
use kvbm_protocols::control::modules::transfer::TransferSessionCapability;
use kvbm_protocols::disagg::SessionEndpoint;
use velo::{EventHandle, EventManager};

use crate::p2p::StagedPull;
use tokio_util::sync::CancellationToken;

use super::test_support::{
    LoopbackBundleFixture, RESOURCES, build_corrupt_loopback_bundle_fixture,
    build_loopback_bundle_fixture, build_mixed_native_loopback_bundle_fixture,
    build_smaller_native_loopback_bundle_fixture, manifest,
};
use super::*;

async fn leaders(
    omit: Option<LogicalResourceId>,
) -> (
    Arc<InstanceLeader>,
    Arc<InstanceLeader>,
    Arc<[SequenceHash]>,
) {
    let fixture = build_loopback_bundle_fixture(omit, None).await.unwrap();
    (fixture.holder, fixture.puller, fixture.hashes)
}

struct RecordingTarget {
    leader: Arc<InstanceLeader>,
    committed: AtomicBool,
    generations: AtomicUsize,
}

impl BundlePullTarget for RecordingTarget {
    fn instance_leader(&self) -> Arc<InstanceLeader> {
        Arc::clone(&self.leader)
    }

    fn reserve_publication_generation(&self) -> Result<u64> {
        Ok(self.generations.fetch_add(1, Ordering::AcqRel) as u64 + 1)
    }

    fn commit_pulled_bundle(
        &self,
        _identity: CacheIdentity,
        _key: BundleKey,
        _generation: u64,
        bundle: StagedBundle,
    ) -> BoxFuture<'static, Result<()>> {
        let leader = Arc::clone(&self.leader);
        let hidden_before_commit = bundle.lineages().iter().all(|(resource, hashes)| {
            leader
                .g2_manager_for(*resource)
                .is_some_and(|manager| manager.match_blocks(hashes).is_empty())
        });
        let published = bundle.publish().expect("publish staged bundle");
        let complete = published.iter().all(|(resource, blocks)| {
            leader.g2_manager_for(*resource).is_some_and(|manager| {
                let hashes = blocks
                    .iter()
                    .map(|block| block.sequence_hash())
                    .collect::<Vec<_>>();
                manager.match_blocks(&hashes).len() == hashes.len()
            })
        });
        self.committed
            .store(hidden_before_commit && complete, Ordering::Release);
        Box::pin(async move {
            drop(published);
            anyhow::ensure!(
                hidden_before_commit && complete,
                "bundle resources were visible before the transaction committed"
            );
            Ok(())
        })
    }
}

fn candidate(
    holder: &InstanceLeader,
    identity: &CacheIdentity,
    hashes: &[SequenceHash],
) -> RemoteBundleCandidate {
    candidate_with_lease(holder, identity, hashes, Duration::from_secs(20))
}

fn candidate_with_lease(
    holder: &InstanceLeader,
    identity: &CacheIdentity,
    hashes: &[SequenceHash],
    lease: Duration,
) -> RemoteBundleCandidate {
    candidate_with_epoch(holder, identity, hashes, holder.registration_epoch(), lease)
}

fn candidate_with_epoch(
    holder: &InstanceLeader,
    identity: &CacheIdentity,
    hashes: &[SequenceHash],
    registration_epoch: RegistrationEpoch,
    lease: Duration,
) -> RemoteBundleCandidate {
    let key = BundleKey::new(identity, hashes[1], 8).unwrap();
    let lineages = identity.resources().iter().map(|requirement| {
        let hashes = match requirement.role() {
            ResourceRole::PrefixHistory => hashes.to_vec(),
            ResourceRole::BoundaryCapsule => vec![key.boundary_hash()],
        };
        BundleResourceLineage::new(requirement.resource(), hashes).unwrap()
    });
    let advertisement = super::super::BundleAdvertisement::new(
        identity.clone(),
        key,
        4,
        holder.messenger().instance_id(),
        registration_epoch,
        unix_time_ms() + 30_000,
        lineages,
    )
    .unwrap();
    RemoteBundleCandidate::new(
        advertisement,
        uuid::Uuid::new_v4(),
        unix_time_ms() + u64::try_from(lease.as_millis()).unwrap(),
    )
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stale_directory_hit_cannot_pull_from_replacement_owner_lifecycle() -> Result<()> {
    let fixture = build_loopback_bundle_fixture(None, None).await?;
    let registration_a = RegistrationEpoch::new();
    let registration_b = RegistrationEpoch::new();
    assert_ne!(registration_a, registration_b);
    assert!(fixture.holder.set_registration_epoch(registration_b));

    let stale_candidate = candidate_with_epoch(
        &fixture.holder,
        &fixture.identity,
        &fixture.hashes,
        registration_a,
        Duration::from_secs(20),
    );
    let target = Arc::new(RecordingTarget {
        leader: Arc::clone(&fixture.puller),
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });

    let result = pull_remote_bundle(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        stale_candidate,
        fixture.identity.clone(),
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_secs(1),
    )
    .await?;

    assert_eq!(result, BundlePullOutcome::Miss(BundleMissReason::OwnerLost));
    assert!(!target.committed.load(Ordering::Acquire));
    for (&resource, hashes) in &fixture.lineages {
        assert!(
            fixture
                .puller
                .g2_manager_for(resource)
                .unwrap()
                .match_blocks(hashes)
                .is_empty(),
            "stale owner lifecycle leaked destination visibility for {resource:?}"
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum MismatchedOpenKind {
    Sync,
    Async,
}

struct MismatchedHolderTransfer {
    expected_owner: crate::InstanceId,
    mismatched_kind: MismatchedOpenKind,
    opens: AtomicUsize,
    pulls: AtomicUsize,
    closes: AtomicUsize,
}

impl BundleTransfer for MismatchedHolderTransfer {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        _watchdog: Duration,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse, BundleTransferError>> {
        let attempt = self.opens.fetch_add(1, Ordering::AcqRel);
        let expected_owner = self.expected_owner;
        let mismatched_kind = self.mismatched_kind;
        Box::pin(async move {
            let mismatched = attempt == 1;
            let capability = TransferSessionCapability {
                session_id: uuid::Uuid::new_v4(),
                instance_id: if mismatched {
                    uuid::Uuid::new_v4().into()
                } else {
                    expected_owner
                },
                endpoint: SessionEndpoint {
                    kind: "mismatched-holder".to_owned(),
                    payload: serde_json::Value::Null,
                },
                resource,
            };
            if mismatched && matches!(mismatched_kind, MismatchedOpenKind::Async) {
                Ok(OpenTransferSessionResponse::Async { capability })
            } else {
                Ok(OpenTransferSessionResponse::Sync {
                    capability,
                    committed: hashes,
                    breakdown: Default::default(),
                })
            }
        })
    }

    fn pull(
        &self,
        _resource: &OpenedResource,
    ) -> BoxFuture<'_, Result<StagedPull, BundleTransferError>> {
        self.pulls.fetch_add(1, Ordering::AcqRel);
        Box::pin(async {
            unreachable!("a mismatched holder capability must be rejected before pulling")
        })
    }

    fn close(&self, _session_id: uuid::Uuid, _reason: &str) -> BoxFuture<'_, ()> {
        self.closes.fetch_add(1, Ordering::AcqRel);
        Box::pin(async {})
    }
}

#[tokio::test]
async fn sync_open_from_a_different_holder_is_rejected_and_all_sessions_are_closed() -> Result<()> {
    assert_mismatched_holder_is_rejected(MismatchedOpenKind::Sync).await
}

#[tokio::test]
async fn async_open_from_a_different_holder_is_rejected_and_all_sessions_are_closed() -> Result<()>
{
    assert_mismatched_holder_is_rejected(MismatchedOpenKind::Async).await
}

async fn assert_mismatched_holder_is_rejected(kind: MismatchedOpenKind) -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let candidate = candidate(&holder, &identity, &hashes);
    let target = Arc::new(RecordingTarget {
        leader: Arc::clone(&puller),
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });
    let transfer = Arc::new(MismatchedHolderTransfer {
        expected_owner: candidate.advertisement().owner(),
        mismatched_kind: kind,
        opens: AtomicUsize::new(0),
        pulls: AtomicUsize::new(0),
        closes: AtomicUsize::new(0),
    });

    let result = pull_remote_bundle_with_transfer(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate,
        identity,
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_secs(1),
        Arc::clone(&transfer) as Arc<dyn BundleTransfer>,
        BundlePullLimits::for_test(Duration::from_secs(1)),
    )
    .await?;

    assert_eq!(result, BundlePullOutcome::Miss(BundleMissReason::OwnerLost));
    assert_eq!(transfer.opens.load(Ordering::Acquire), 2);
    assert_eq!(transfer.closes.load(Ordering::Acquire), 2);
    assert_eq!(transfer.pulls.load(Ordering::Acquire), 0);
    assert!(!target.committed.load(Ordering::Acquire));
    assert_no_destination_visibility(&puller, &hashes);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn loopback_pulls_every_resource_before_local_commit() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: puller,
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });
    let expected = candidate(&holder, &identity, &hashes).advertisement().key();

    let result = pull_remote_bundle(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate(&holder, &identity, &hashes),
        CacheIdentity::clone(&identity),
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_secs(1),
    )
    .await?;

    assert_eq!(result, BundlePullOutcome::Pulled(expected));
    assert!(target.committed.load(Ordering::Acquire));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn post_dma_payload_corruption_aborts_all_real_loopback_visibility() -> Result<()> {
    let fixture = build_corrupt_loopback_bundle_fixture().await?;
    let target = Arc::new(RecordingTarget {
        leader: Arc::clone(&fixture.puller),
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });

    let result = pull_remote_bundle(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        fixture.candidate(),
        fixture.identity.clone(),
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_secs(1),
    )
    .await?;

    assert_eq!(
        result,
        BundlePullOutcome::Miss(BundleMissReason::ChecksumFailed)
    );
    assert!(!target.committed.load(Ordering::Acquire));
    for (&resource, hashes) in &fixture.lineages {
        assert!(
            fixture
                .puller
                .g2_manager_for(resource)
                .unwrap()
                .match_blocks(hashes)
                .is_empty(),
            "corrupted resource pull leaked visibility for {resource:?}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn larger_native_secondary_pulls_its_advertised_exact_lineage() -> Result<()> {
    assert_mixed_native_pull(build_mixed_native_loopback_bundle_fixture().await?).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn smaller_native_secondary_pulls_its_advertised_exact_lineage() -> Result<()> {
    assert_mixed_native_pull(build_smaller_native_loopback_bundle_fixture().await?).await
}

async fn assert_mixed_native_pull(fixture: LoopbackBundleFixture) -> Result<()> {
    let primary = &fixture.lineages[&RESOURCES[0]];
    let secondary = &fixture.lineages[&RESOURCES[1]];
    assert_ne!(primary.last(), secondary.last());
    assert_eq!(
        fixture
            .holder
            .g2_manager_for(RESOURCES[1])
            .unwrap()
            .match_blocks(secondary)
            .iter()
            .map(|block| block.sequence_hash())
            .collect::<Vec<_>>(),
        *secondary
    );
    assert!(
        fixture
            .holder
            .g2_manager_for(RESOURCES[1])
            .unwrap()
            .match_blocks(primary)
            .is_empty(),
        "the owner must not accidentally satisfy the secondary resource with the primary lineage"
    );

    let target = Arc::new(RecordingTarget {
        leader: Arc::clone(&fixture.puller),
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });
    let expected = fixture.candidate().advertisement().key();
    let result = pull_remote_bundle(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        fixture.candidate(),
        fixture.identity.clone(),
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_secs(1),
    )
    .await?;

    assert_eq!(result, BundlePullOutcome::Pulled(expected));
    assert!(target.committed.load(Ordering::Acquire));
    for (&resource, hashes) in &fixture.lineages {
        assert_eq!(
            fixture
                .puller
                .g2_manager_for(resource)
                .unwrap()
                .match_blocks(hashes)
                .iter()
                .map(|block| block.sequence_hash())
                .collect::<Vec<_>>(),
            *hashes,
            "resource {resource:?} did not restore its exact advertised lineage"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_omitted_remote_resource_aborts_without_destination_visibility() -> Result<()> {
    for omitted in RESOURCES {
        let (holder, puller, hashes) = leaders(Some(omitted)).await;
        let identity = manifest();
        let target = Arc::new(RecordingTarget {
            leader: Arc::clone(&puller),
            committed: AtomicBool::new(false),
            generations: AtomicUsize::new(0),
        });

        let result = pull_remote_bundle(
            Arc::clone(&target) as Arc<dyn BundlePullTarget>,
            candidate(&holder, &identity, &hashes),
            identity,
            CancellationToken::new(),
            tokio::time::Instant::now() + Duration::from_secs(1),
        )
        .await?;

        assert_eq!(
            result,
            BundlePullOutcome::Miss(BundleMissReason::Incomplete)
        );
        assert!(!target.committed.load(Ordering::Acquire));
        for resource in RESOURCES {
            for hash in hashes.iter().copied() {
                assert!(
                    puller
                        .g2_manager_for(resource)
                        .unwrap()
                        .match_blocks(&[hash])
                        .is_empty(),
                    "resource {resource:?} leaked after omitting {omitted:?}"
                );
            }
        }
    }
    Ok(())
}

struct OwnerLossTransfer {
    owner: crate::InstanceId,
    pulls: AtomicUsize,
}

impl BundleTransfer for OwnerLossTransfer {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        _watchdog: Duration,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse, BundleTransferError>> {
        let owner = self.owner;
        Box::pin(async move {
            Ok(OpenTransferSessionResponse::Sync {
                capability: TransferSessionCapability {
                    session_id: uuid::Uuid::new_v4(),
                    instance_id: owner,
                    endpoint: SessionEndpoint {
                        kind: "fault".to_owned(),
                        payload: serde_json::Value::Null,
                    },
                    resource,
                },
                committed: hashes,
                breakdown: Default::default(),
            })
        })
    }

    fn pull(
        &self,
        resource: &OpenedResource,
    ) -> BoxFuture<'_, Result<StagedPull, BundleTransferError>> {
        let attempt = self.pulls.fetch_add(1, Ordering::AcqRel);
        let resource = resource.resource();
        Box::pin(async move {
            if attempt == 0 {
                Err(BundleTransferError::TransferFailed {
                    resource,
                    message: "synthetic first-resource failure".to_owned(),
                })
            } else {
                Err(BundleTransferError::OwnerLost {
                    message: "synthetic owner loss".to_owned(),
                })
            }
        })
    }

    fn close(&self, _session_id: uuid::Uuid, _reason: &str) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

#[tokio::test]
async fn owner_loss_during_pull_aborts_without_local_bundle_commit() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: puller,
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });
    let transfer = Arc::new(OwnerLossTransfer {
        owner: holder.messenger().instance_id(),
        pulls: AtomicUsize::new(0),
    });

    let result = pull_remote_bundle_with_transfer(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate(&holder, &identity, &hashes),
        identity,
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_secs(1),
        Arc::clone(&transfer) as Arc<dyn BundleTransfer>,
        BundlePullLimits::for_test(Duration::from_secs(1)),
    )
    .await;

    assert_eq!(
        result?,
        BundlePullOutcome::Miss(BundleMissReason::OwnerLost)
    );
    assert_eq!(transfer.pulls.load(Ordering::Acquire), RESOURCES.len());
    assert!(!target.committed.load(Ordering::Acquire));
    Ok(())
}

struct PartialFailureTransfer {
    leader: Arc<InstanceLeader>,
    owner: crate::InstanceId,
    failed_resource: LogicalResourceId,
    pulls: AtomicUsize,
    closes: AtomicUsize,
}

impl BundleTransfer for PartialFailureTransfer {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        _watchdog: Duration,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse, BundleTransferError>> {
        let owner = self.owner;
        Box::pin(async move { Ok(sync_open(owner, resource, hashes)) })
    }

    fn pull(
        &self,
        opened: &OpenedResource,
    ) -> BoxFuture<'_, Result<StagedPull, BundleTransferError>> {
        self.pulls.fetch_add(1, Ordering::AcqRel);
        let resource = opened.resource();
        let hashes = opened.hashes().to_vec();
        let manager = self.leader.g2_manager_for(resource).unwrap().clone();
        let failed = resource == self.failed_resource;
        Box::pin(async move {
            if failed {
                return Err(BundleTransferError::TransferFailed {
                    resource,
                    message: "injected resource transfer failure".to_owned(),
                });
            }
            let blocks = manager
                .allocate_blocks(hashes.len())
                .ok_or_else(|| BundleTransferError::TransferFailed {
                    resource,
                    message: "test destination allocation failed".to_owned(),
                })?
                .into_iter()
                .zip(hashes.iter().copied())
                .map(|(block, hash)| {
                    block.stage(hash, manager.block_size()).map_err(|error| {
                        BundleTransferError::TransferFailed {
                            resource,
                            message: error.to_string(),
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(StagedPull::from_test_parts(
                resource, hashes, blocks, manager,
            ))
        })
    }

    fn close(&self, _session_id: uuid::Uuid, _reason: &str) -> BoxFuture<'_, ()> {
        self.closes.fetch_add(1, Ordering::AcqRel);
        Box::pin(async {})
    }
}

#[tokio::test]
async fn one_resource_transfer_failure_rolls_back_successful_stages() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: Arc::clone(&puller),
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });
    let transfer = Arc::new(PartialFailureTransfer {
        leader: Arc::clone(&puller),
        owner: holder.messenger().instance_id(),
        failed_resource: RESOURCES[1],
        pulls: AtomicUsize::new(0),
        closes: AtomicUsize::new(0),
    });

    let result = pull_remote_bundle_with_transfer(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate(&holder, &identity, &hashes),
        identity,
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_secs(1),
        Arc::clone(&transfer) as Arc<dyn BundleTransfer>,
        BundlePullLimits::for_test(Duration::from_secs(1)),
    )
    .await?;

    assert_eq!(
        result,
        BundlePullOutcome::Miss(BundleMissReason::TransferFailed)
    );
    assert_eq!(transfer.pulls.load(Ordering::Acquire), RESOURCES.len());
    assert_eq!(transfer.closes.load(Ordering::Acquire), RESOURCES.len());
    assert!(!target.committed.load(Ordering::Acquire));
    assert_no_destination_visibility(&puller, &hashes);
    Ok(())
}

struct CorruptingTransfer {
    leader: Arc<InstanceLeader>,
    owner: crate::InstanceId,
    corrupted_resource: LogicalResourceId,
    pulls: AtomicUsize,
    closes: AtomicUsize,
}

impl BundleTransfer for CorruptingTransfer {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        _watchdog: Duration,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse, BundleTransferError>> {
        let owner = self.owner;
        Box::pin(async move { Ok(sync_open(owner, resource, hashes)) })
    }

    fn pull(
        &self,
        opened: &OpenedResource,
    ) -> BoxFuture<'_, Result<StagedPull, BundleTransferError>> {
        self.pulls.fetch_add(1, Ordering::AcqRel);
        let resource = opened.resource();
        let hashes = opened.hashes().to_vec();
        let manager = self.leader.g2_manager_for(resource).unwrap().clone();
        let corrupted = resource == self.corrupted_resource;
        Box::pin(async move {
            // Model the transport completing successfully before the receiver's
            // content-integrity verifier detects that one payload was changed.
            if corrupted {
                return Err(BundleTransferError::ChecksumMismatch {
                    resource,
                    message: "injected post-DMA payload corruption".to_owned(),
                });
            }
            let blocks = manager
                .allocate_blocks(hashes.len())
                .ok_or_else(|| BundleTransferError::TransferFailed {
                    resource,
                    message: "test destination allocation failed".to_owned(),
                })?
                .into_iter()
                .zip(hashes.iter().copied())
                .map(|(block, hash)| {
                    block.stage(hash, manager.block_size()).map_err(|error| {
                        BundleTransferError::TransferFailed {
                            resource,
                            message: error.to_string(),
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(StagedPull::from_test_parts(
                resource, hashes, blocks, manager,
            ))
        })
    }

    fn close(&self, _session_id: uuid::Uuid, _reason: &str) -> BoxFuture<'_, ()> {
        self.closes.fetch_add(1, Ordering::AcqRel);
        Box::pin(async {})
    }
}

#[tokio::test]
async fn one_corrupted_resource_aborts_the_complete_bundle_without_visibility() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: Arc::clone(&puller),
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });
    let transfer = Arc::new(CorruptingTransfer {
        leader: Arc::clone(&puller),
        owner: holder.messenger().instance_id(),
        corrupted_resource: RESOURCES[1],
        pulls: AtomicUsize::new(0),
        closes: AtomicUsize::new(0),
    });

    let result = pull_remote_bundle_with_transfer(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate(&holder, &identity, &hashes),
        identity,
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_secs(1),
        Arc::clone(&transfer) as Arc<dyn BundleTransfer>,
        BundlePullLimits::for_test(Duration::from_secs(1)),
    )
    .await?;

    assert_eq!(
        result,
        BundlePullOutcome::Miss(BundleMissReason::ChecksumFailed)
    );
    assert_eq!(transfer.pulls.load(Ordering::Acquire), RESOURCES.len());
    assert_eq!(transfer.closes.load(Ordering::Acquire), RESOURCES.len());
    assert!(!target.committed.load(Ordering::Acquire));
    assert_no_destination_visibility(&puller, &hashes);
    Ok(())
}

struct HangingPullTransfer {
    owner: crate::InstanceId,
    pulls: AtomicUsize,
    closes: AtomicUsize,
}

struct LaunchedNotificationTransfer {
    leader: Arc<InstanceLeader>,
    owner: crate::InstanceId,
    events: Arc<EventManager>,
    pending: Mutex<Vec<EventHandle>>,
    launched: AtomicUsize,
    closes: AtomicUsize,
}

impl LaunchedNotificationTransfer {
    fn release_all(&self) -> Result<()> {
        for handle in self.pending.lock().unwrap().drain(..) {
            self.events.trigger(handle)?;
        }
        Ok(())
    }
}

impl BundleTransfer for LaunchedNotificationTransfer {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        _watchdog: Duration,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse, BundleTransferError>> {
        let owner = self.owner;
        Box::pin(async move { Ok(sync_open(owner, resource, hashes)) })
    }

    fn pull(
        &self,
        opened: &OpenedResource,
    ) -> BoxFuture<'_, Result<StagedPull, BundleTransferError>> {
        let resource = opened.resource();
        let hashes = opened.hashes().to_vec();
        let manager = self.leader.g2_manager_for(resource).unwrap().clone();
        let events = Arc::clone(&self.events);
        Box::pin(async move {
            let destinations = manager.allocate_blocks(hashes.len()).ok_or_else(|| {
                BundleTransferError::TransferFailed {
                    resource,
                    message: "test destination allocation failed".to_owned(),
                }
            })?;
            let event =
                events
                    .new_event()
                    .map_err(|error| BundleTransferError::TransferFailed {
                        resource,
                        message: error.to_string(),
                    })?;
            let handle = event.into_handle();
            let notification =
                TransferCompleteNotification::from_awaiter(events.awaiter(handle).map_err(
                    |error| BundleTransferError::TransferFailed {
                        resource,
                        message: error.to_string(),
                    },
                )?);
            self.pending.lock().unwrap().push(handle);
            self.launched.fetch_add(1, Ordering::AcqRel);

            notification
                .await
                .map_err(|error| BundleTransferError::TransferFailed {
                    resource,
                    message: error.to_string(),
                })?;
            let blocks = destinations
                .into_iter()
                .zip(hashes.iter().copied())
                .map(|(block, hash)| {
                    block.stage(hash, manager.block_size()).map_err(|error| {
                        BundleTransferError::TransferFailed {
                            resource,
                            message: error.to_string(),
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(StagedPull::from_test_parts(
                resource, hashes, blocks, manager,
            ))
        })
    }

    fn close(&self, _session_id: uuid::Uuid, _reason: &str) -> BoxFuture<'_, ()> {
        self.closes.fetch_add(1, Ordering::AcqRel);
        Box::pin(async {})
    }
}

impl BundleTransfer for HangingPullTransfer {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        _watchdog: Duration,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse, BundleTransferError>> {
        let owner = self.owner;
        Box::pin(async move { Ok(sync_open(owner, resource, hashes)) })
    }

    fn pull(
        &self,
        _resource: &OpenedResource,
    ) -> BoxFuture<'_, Result<StagedPull, BundleTransferError>> {
        self.pulls.fetch_add(1, Ordering::AcqRel);
        Box::pin(std::future::pending())
    }

    fn close(&self, _session_id: uuid::Uuid, _reason: &str) -> BoxFuture<'_, ()> {
        self.closes.fetch_add(1, Ordering::AcqRel);
        Box::pin(async {})
    }
}

#[tokio::test]
async fn hung_pull_after_all_opens_times_out_without_closing_inflight_sessions() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: puller,
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });
    let transfer = Arc::new(HangingPullTransfer {
        owner: holder.messenger().instance_id(),
        pulls: AtomicUsize::new(0),
        closes: AtomicUsize::new(0),
    });

    let result = pull_remote_bundle_with_transfer(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate(&holder, &identity, &hashes),
        identity,
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_millis(25),
        Arc::clone(&transfer) as Arc<dyn BundleTransfer>,
        BundlePullLimits::for_test(Duration::from_millis(25)),
    )
    .await?;

    assert_eq!(result, BundlePullOutcome::Miss(BundleMissReason::TimedOut));
    assert_eq!(transfer.pulls.load(Ordering::Acquire), RESOURCES.len());
    assert_eq!(
        transfer.closes.load(Ordering::Acquire),
        0,
        "a hung authorized pull must retain holder-side pins"
    );
    assert!(!target.committed.load(Ordering::Acquire));
    Ok(())
}

#[tokio::test]
async fn lease_expiry_during_pull_is_distinct_from_watchdog_timeout() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: puller,
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });
    let transfer = Arc::new(HangingPullTransfer {
        owner: holder.messenger().instance_id(),
        pulls: AtomicUsize::new(0),
        closes: AtomicUsize::new(0),
    });

    let result = pull_remote_bundle_with_transfer(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate_with_lease(&holder, &identity, &hashes, Duration::from_millis(25)),
        identity,
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_secs(1),
        Arc::clone(&transfer) as Arc<dyn BundleTransfer>,
        BundlePullLimits::for_test(Duration::from_millis(100)),
    )
    .await?;

    assert_eq!(result, BundlePullOutcome::Miss(BundleMissReason::Expired));
    assert_eq!(
        transfer.closes.load(Ordering::Acquire),
        0,
        "lease expiry must not release source pins before physical terminal"
    );
    assert!(!target.committed.load(Ordering::Acquire));
    Ok(())
}

#[tokio::test]
async fn cancellation_during_pull_retains_sessions_and_cannot_commit_late() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: Arc::clone(&puller),
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });
    let transfer = Arc::new(HangingPullTransfer {
        owner: holder.messenger().instance_id(),
        pulls: AtomicUsize::new(0),
        closes: AtomicUsize::new(0),
    });
    let cancel = CancellationToken::new();
    let task = tokio::spawn(pull_remote_bundle_with_transfer(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate(&holder, &identity, &hashes),
        identity,
        cancel.clone(),
        tokio::time::Instant::now() + Duration::from_secs(1),
        Arc::clone(&transfer) as Arc<dyn BundleTransfer>,
        BundlePullLimits::for_test(Duration::from_millis(100)),
    ));
    tokio::time::timeout(Duration::from_secs(1), async {
        while transfer.pulls.load(Ordering::Acquire) != RESOURCES.len() {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    cancel.cancel();

    assert_eq!(
        task.await??,
        BundlePullOutcome::Miss(BundleMissReason::Canceled)
    );
    assert_eq!(
        transfer.closes.load(Ordering::Acquire),
        0,
        "cancellation must not release source pins before physical terminal"
    );
    assert!(!target.committed.load(Ordering::Acquire));
    assert_no_destination_visibility(&puller, &hashes);
    Ok(())
}

#[tokio::test]
async fn launched_pull_timeout_quarantines_destinations_until_notification_drains() -> Result<()> {
    assert_launched_pull_interruption_quarantines(false).await
}

#[tokio::test]
async fn launched_pull_cancel_quarantines_destinations_until_notification_drains() -> Result<()> {
    assert_launched_pull_interruption_quarantines(true).await
}

async fn assert_launched_pull_interruption_quarantines(cancel_pull: bool) -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: Arc::clone(&puller),
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });
    let transfer = Arc::new(LaunchedNotificationTransfer {
        leader: Arc::clone(&puller),
        owner: holder.messenger().instance_id(),
        events: Arc::new(EventManager::local()),
        pending: Mutex::new(Vec::new()),
        launched: AtomicUsize::new(0),
        closes: AtomicUsize::new(0),
    });
    let initial_availability = RESOURCES
        .iter()
        .map(|resource| {
            (
                *resource,
                puller.g2_manager_for(*resource).unwrap().available_blocks(),
            )
        })
        .collect::<Vec<_>>();
    let cancel = CancellationToken::new();
    let deadline = tokio::time::Instant::now()
        + if cancel_pull {
            Duration::from_secs(1)
        } else {
            Duration::from_millis(50)
        };
    let task = tokio::spawn(pull_remote_bundle_with_transfer(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate(&holder, &identity, &hashes),
        identity,
        cancel.clone(),
        deadline,
        Arc::clone(&transfer) as Arc<dyn BundleTransfer>,
        BundlePullLimits::for_test(Duration::from_millis(50)),
    ));
    tokio::time::timeout(Duration::from_secs(1), async {
        while transfer.launched.load(Ordering::Acquire) != RESOURCES.len() {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    if cancel_pull {
        cancel.cancel();
    }

    let outcome = tokio::time::timeout(Duration::from_millis(250), task).await???;
    assert_eq!(
        outcome,
        BundlePullOutcome::Miss(if cancel_pull {
            BundleMissReason::Canceled
        } else {
            BundleMissReason::TimedOut
        })
    );
    assert_eq!(
        transfer.closes.load(Ordering::Acquire),
        0,
        "holder sessions must remain pinned until physical completion drains"
    );
    assert!(!target.committed.load(Ordering::Acquire));

    for (resource, initial) in &initial_availability {
        let manager = puller.g2_manager_for(*resource).unwrap();
        assert!(
            manager.available_blocks() < *initial,
            "interrupted pull must retain resource {resource:?} destination guards"
        );
        assert!(
            manager.allocate_blocks(*initial).is_none(),
            "allocator pressure must not recycle resource {resource:?} destinations before DMA terminal"
        );
    }

    transfer.release_all()?;
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if initial_availability.iter().all(|(resource, initial)| {
                puller.g2_manager_for(*resource).unwrap().available_blocks() == *initial
            }) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await?;
    assert_eq!(transfer.closes.load(Ordering::Acquire), RESOURCES.len());
    Ok(())
}

struct FailingCommitTarget {
    leader: Arc<InstanceLeader>,
    commits: Arc<AtomicUsize>,
}

impl BundlePullTarget for FailingCommitTarget {
    fn instance_leader(&self) -> Arc<InstanceLeader> {
        Arc::clone(&self.leader)
    }

    fn reserve_publication_generation(&self) -> Result<u64> {
        Ok(1)
    }

    fn commit_pulled_bundle(
        &self,
        _identity: CacheIdentity,
        _key: BundleKey,
        _generation: u64,
        _bundle: StagedBundle,
    ) -> BoxFuture<'static, Result<()>> {
        let commits = Arc::clone(&self.commits);
        Box::pin(async move {
            commits.fetch_add(1, Ordering::AcqRel);
            anyhow::bail!("injected catalog commit failure")
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn commit_failure_rolls_back_every_staged_destination() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let commits = Arc::new(AtomicUsize::new(0));
    let target = Arc::new(FailingCommitTarget {
        leader: Arc::clone(&puller),
        commits: Arc::clone(&commits),
    });

    let result = pull_remote_bundle(
        target as Arc<dyn BundlePullTarget>,
        candidate(&holder, &identity, &hashes),
        identity,
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_secs(1),
    )
    .await?;

    assert_eq!(
        result,
        BundlePullOutcome::Miss(BundleMissReason::CommitFailed)
    );
    assert_eq!(commits.load(Ordering::Acquire), 1);
    assert_no_destination_visibility(&puller, &hashes);
    Ok(())
}

fn sync_open(
    owner: crate::InstanceId,
    resource: LogicalResourceId,
    hashes: Vec<SequenceHash>,
) -> OpenTransferSessionResponse {
    OpenTransferSessionResponse::Sync {
        capability: TransferSessionCapability {
            session_id: uuid::Uuid::new_v4(),
            instance_id: owner,
            endpoint: SessionEndpoint {
                kind: "test".to_owned(),
                payload: serde_json::Value::Null,
            },
            resource,
        },
        committed: hashes,
        breakdown: Default::default(),
    }
}

fn assert_no_destination_visibility(leader: &InstanceLeader, hashes: &[SequenceHash]) {
    for resource in RESOURCES {
        for hash in hashes.iter().copied() {
            assert!(
                leader
                    .g2_manager_for(resource)
                    .unwrap()
                    .match_blocks(&[hash])
                    .is_empty(),
                "resource {resource:?} leaked a physically staged block"
            );
        }
    }
}

#[test]
fn timeout_records_one_aborted_transaction_and_bounded_resource_series() {
    let metrics = BundleMetrics::new();
    let registry = prometheus::Registry::new();
    metrics.register(&registry).unwrap();
    PullMetrics::new(Some(metrics), RESOURCES.to_vec())
        .record(&Ok(BundlePullOutcome::Miss(BundleMissReason::TimedOut)));

    let gathered = registry.gather();
    let transaction = gathered
        .iter()
        .find(|family| family.name() == "kvbm_bundle_txn_total")
        .unwrap();
    assert_eq!(transaction.get_metric().len(), 1);
    assert!(has_label(
        &transaction.get_metric()[0],
        "operation",
        "remote_pull"
    ));
    assert!(has_label(&transaction.get_metric()[0], "outcome", "abort"));

    let resources = gathered
        .iter()
        .find(|family| family.name() == "kvbm_bundle_resource_outcome_total")
        .unwrap();
    assert_eq!(resources.get_metric().len(), RESOURCES.len());
    assert!(resources.get_metric().iter().all(|metric| {
        has_label(metric, "operation", "remote_pull")
            && has_label(metric, "outcome", "aborted")
            && has_label(metric, "reason", "timed_out")
    }));
}

#[test]
fn successful_pull_records_actual_bytes_for_each_resource() {
    let metrics = BundleMetrics::new();
    let registry = prometheus::Registry::new();
    metrics.register(&registry).unwrap();
    let mut pull = PullMetrics::new(Some(metrics), RESOURCES.to_vec());
    for (index, resource) in RESOURCES.into_iter().enumerate() {
        pull.observe_transferred_bytes(resource, u64::try_from(index + 1).unwrap() * 128);
    }
    let key = BundleKey::from_parts(
        CacheManifestId::from_bytes([1; 32]),
        SequenceHash::new(1, None, 1),
        4,
    )
    .unwrap();
    pull.record(&Ok(BundlePullOutcome::Pulled(key)));

    let gathered = registry.gather();
    let bytes = gathered
        .iter()
        .find(|family| family.name() == "kvbm_bundle_resource_bytes_total")
        .unwrap();
    for (index, resource) in RESOURCES.into_iter().enumerate() {
        let expected = u64::try_from(index + 1).unwrap() * 128;
        let resource = resource.0.to_string();
        let metric = bytes
            .get_metric()
            .iter()
            .find(|metric| {
                has_label(metric, "resource", &resource)
                    && has_label(metric, "outcome", "committed")
            })
            .expect("resource byte series must exist");
        assert_eq!(metric.get_counter().value() as u64, expected);
    }
}

fn has_label(metric: &prometheus::proto::Metric, name: &str, value: &str) -> bool {
    metric
        .get_label()
        .iter()
        .any(|label| label.name() == name && label.value() == value)
}

struct HangingTransfer {
    closes: AtomicUsize,
}

impl BundleTransfer for HangingTransfer {
    fn open(
        &self,
        _resource: LogicalResourceId,
        _hashes: Vec<SequenceHash>,
        _watchdog: Duration,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse, BundleTransferError>> {
        Box::pin(std::future::pending())
    }

    fn pull(
        &self,
        _resource: &OpenedResource,
    ) -> BoxFuture<'_, Result<StagedPull, BundleTransferError>> {
        Box::pin(std::future::pending())
    }

    fn close(&self, _session_id: uuid::Uuid, _reason: &str) -> BoxFuture<'_, ()> {
        self.closes.fetch_add(1, Ordering::AcqRel);
        Box::pin(async {})
    }
}

#[tokio::test]
async fn hung_open_is_bounded_and_reports_timeout() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let identity = manifest();
    let target = Arc::new(RecordingTarget {
        leader: puller,
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });
    let transfer = Arc::new(HangingTransfer {
        closes: AtomicUsize::new(0),
    });

    let result = tokio::time::timeout(
        Duration::from_millis(250),
        pull_remote_bundle_with_transfer(
            Arc::clone(&target) as Arc<dyn BundlePullTarget>,
            candidate(&holder, &identity, &hashes),
            identity,
            CancellationToken::new(),
            tokio::time::Instant::now() + Duration::from_millis(25),
            transfer as Arc<dyn BundleTransfer>,
            BundlePullLimits::for_test(Duration::from_millis(25)),
        ),
    )
    .await
    .expect("the bundle watchdog must bound a hung open")?;

    assert_eq!(result, BundlePullOutcome::Miss(BundleMissReason::TimedOut));
    assert!(!target.committed.load(Ordering::Acquire));
    Ok(())
}

#[tokio::test]
async fn manifest_mismatch_is_an_incompatible_terminal_miss() -> Result<()> {
    let (holder, puller, hashes) = leaders(None).await;
    let advertised = manifest();
    let expected = CacheManifest::new(
        ModelIdentity::new("bundle-loopback", "different", [7; 32])?,
        "bundle-loopback-v1",
        advertised.resources().to_vec(),
        BTreeMap::new(),
    )?
    .identity();
    let target = Arc::new(RecordingTarget {
        leader: puller,
        committed: AtomicBool::new(false),
        generations: AtomicUsize::new(0),
    });

    let result = pull_remote_bundle(
        Arc::clone(&target) as Arc<dyn BundlePullTarget>,
        candidate(&holder, &advertised, &hashes),
        expected,
        CancellationToken::new(),
        tokio::time::Instant::now() + Duration::from_secs(1),
    )
    .await?;

    assert_eq!(
        result,
        BundlePullOutcome::Miss(BundleMissReason::Incompatible)
    );
    assert_eq!(target.generations.load(Ordering::Acquire), 0);
    assert!(!target.committed.load(Ordering::Acquire));
    Ok(())
}
