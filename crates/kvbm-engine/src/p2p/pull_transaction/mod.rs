// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Puller-local staging for transfer sessions.
//!
//! A staged pull owns destination slots without registering their sequence
//! hashes. Callers may therefore compose several logical-resource pulls and
//! publish them only after the whole transaction succeeds. Dropping the value
//! rolls every slot back through [`CompleteBlock`]'s RAII guard.

mod complete;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::StreamExt;
use kvbm_common::{LogicalResourceId, SequenceHash};
#[cfg(test)]
use kvbm_logical::CompleteBlock;
use kvbm_logical::{BlockManager, ImmutableBlock};
use kvbm_protocols::control::ControlError;
use kvbm_protocols::control::modules::transfer::{
    MatchBreakdown, PullFromSessionRequest, PullFromSessionResponse,
};

use super::PayloadBlock;
use super::session::{AvailabilityDelta, CommitDelta, Session};
use crate::G2;
#[cfg(test)]
use crate::g2_capacity::direct_g2_capacity;
use crate::g2_capacity::{
    G2AllocationKind, G2Capacity, G2StagedAllocation, RequiredStagingPublishedAllocation,
    RequiredStagingStagedAllocation, reserve_compatibility,
};
use crate::leader::InstanceLeader;
use crate::remote::search::bundle::OpenedResource;

/// One resource pulled into private destination slots but not yet registered.
pub(crate) struct StagedPull {
    resource: LogicalResourceId,
    hashes: Vec<SequenceHash>,
    allocation: RequiredStagingStagedAllocation,
    breakdown: MatchBreakdown,
}

enum PullStagingMode {
    CompatibilityChunks,
    CompleteLineage,
}

impl StagedPull {
    pub(crate) fn resource(&self) -> LogicalResourceId {
        self.resource
    }

    pub(crate) fn hashes(&self) -> &[SequenceHash] {
        &self.hashes
    }

    /// Make the staged hashes visible in the resource registry.
    ///
    /// The source capacity can reject a compatibility allocation that it does not own.
    pub(crate) fn publish(
        self,
    ) -> Result<Vec<ImmutableBlock<G2>>, crate::g2_capacity::G2CapacityError> {
        self.allocation.publish()
    }

    pub(crate) fn publish_reversible(
        self,
    ) -> Result<RequiredStagingPublishedAllocation, crate::g2_capacity::G2CapacityError> {
        self.allocation.publish_reversible()
    }

    pub(crate) fn response(&self) -> PullFromSessionResponse {
        PullFromSessionResponse {
            pulled: self.hashes.clone(),
            breakdown: self.breakdown,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_test_parts(
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        blocks: Vec<CompleteBlock<G2>>,
        manager: Arc<BlockManager<G2>>,
    ) -> Self {
        Self::from_test_parts_with_capacity(resource, hashes, blocks, direct_g2_capacity(manager))
    }

    #[cfg(test)]
    pub(crate) fn from_test_parts_with_capacity(
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        blocks: Vec<CompleteBlock<G2>>,
        capacity: Arc<dyn G2Capacity>,
    ) -> Self {
        Self {
            resource,
            allocation: RequiredStagingStagedAllocation::from_compatibility(
                G2StagedAllocation::direct(
                    G2AllocationKind::RequiredStaging,
                    hashes.clone(),
                    blocks,
                )
                .expect("test staged pull has matching hashes and blocks"),
                capacity,
            )
            .expect("test staged pull uses required-staging allocation"),
            hashes,
            breakdown: MatchBreakdown::default(),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_test_required_staging(
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        allocation: RequiredStagingStagedAllocation,
    ) -> Self {
        Self {
            resource,
            hashes,
            allocation,
            breakdown: MatchBreakdown::default(),
        }
    }
}

/// Attach and pull into unregistered destination slots.
pub(crate) async fn stage_from_session(
    leader: &Arc<InstanceLeader>,
    req: PullFromSessionRequest,
) -> Result<StagedPull, ControlError> {
    stage_from_session_with_mode(leader, req, PullStagingMode::CompatibilityChunks).await
}

/// Attach and pull one complete lineage through one G2 reservation.
pub(crate) async fn stage_complete_from_session(
    leader: &Arc<InstanceLeader>,
    opened: &OpenedResource,
) -> Result<StagedPull, ControlError> {
    let capability = opened.capability();
    let req = PullFromSessionRequest {
        session_id: capability.session_id,
        source_instance_id: capability.instance_id,
        endpoint: Some(capability.endpoint.clone()),
        selector: Some(opened.hashes().to_vec()),
        resource: Some(opened.resource()),
        require_payload_integrity: true,
    };
    stage_from_session_with_mode(leader, req, PullStagingMode::CompleteLineage).await
}

async fn stage_from_session_with_mode(
    leader: &Arc<InstanceLeader>,
    req: PullFromSessionRequest,
    mode: PullStagingMode,
) -> Result<StagedPull, ControlError> {
    let (resource, capacity) = resolve_g2_capacity(leader, req.resource)?;
    let endpoint = req.endpoint.ok_or_else(|| {
        ControlError::Internal(
            "endpoint_required: pull_from_session requires an explicit endpoint in v1 \
             (hub-registry resolution is v1.1)"
                .into(),
        )
    })?;
    let factory = leader
        .session_factory_cell()
        .get()
        .ok_or(ControlError::NotInitialized)?
        .clone();

    crate::engine_audit!(
        "transfer_pull_started",
        session_id = %req.session_id,
        source = %req.source_instance_id,
        resource = ?resource,
        selector_present = req.selector.is_some()
    );

    let session = factory
        .attach(req.session_id, req.source_instance_id, endpoint)
        .await
        .map_err(|error| ControlError::Internal(format!("attach: {error:#}")))?;
    let result = match mode {
        PullStagingMode::CompatibilityChunks => {
            stage_attached(
                leader,
                resource,
                capacity,
                Arc::clone(&session),
                req.selector,
                req.require_payload_integrity,
            )
            .await
        }
        PullStagingMode::CompleteLineage => {
            complete::stage_attached(
                leader,
                resource,
                capacity,
                Arc::clone(&session),
                req.selector,
                req.require_payload_integrity,
            )
            .await
        }
    };
    match &result {
        Ok(_) => session.finalize(None),
        Err(error) => session.close(Some(format!("pull failed: {error}"))),
    }
    result
}

async fn stage_attached(
    leader: &Arc<InstanceLeader>,
    resource: LogicalResourceId,
    capacity: Arc<dyn G2Capacity>,
    session: Arc<dyn Session>,
    selector: Option<Vec<SequenceHash>>,
    require_payload_integrity: bool,
) -> Result<StagedPull, ControlError> {
    let committed = drain_committed(&session).await;
    let source_ordinals = source_ordinals(&committed)?;
    let target_hashes = select_hashes(committed, selector)?;
    if target_hashes.is_empty() {
        let allocation = RequiredStagingStagedAllocation::from_compatibility(
            G2StagedAllocation::direct(G2AllocationKind::RequiredStaging, Vec::new(), Vec::new())
                .expect("empty staged pull is valid"),
            capacity,
        )
        .map_err(|error| {
            ControlError::Internal(format!("bind empty required-staging allocation: {error}"))
        })?;
        return Ok(StagedPull {
            resource,
            hashes: Vec::new(),
            allocation,
            breakdown: MatchBreakdown::default(),
        });
    }

    let target_set = target_hashes.iter().copied().collect::<HashSet<_>>();
    let block_size = capacity.block_size();
    let mut pulled_set = HashSet::new();
    let mut staged = Vec::with_capacity(target_set.len());
    let mut availability = session.availability();
    'drain: while let Some(delta) = availability.next().await {
        match delta {
            AvailabilityDelta::Available(blocks) => {
                if require_payload_integrity
                    && blocks.iter().any(|block| target_set.contains(&block.hash))
                {
                    return Err(ControlError::Internal(
                        "payload_checksum_unverified: bundle pull received unverified availability"
                            .to_owned(),
                    ));
                }
                let chunk = blocks
                    .into_iter()
                    .filter(|block| {
                        target_set.contains(&block.hash) && !pulled_set.contains(&block.hash)
                    })
                    .collect::<Vec<_>>();
                let chunk_hashes = chunk.iter().map(|block| block.hash).collect::<Vec<_>>();
                if chunk_hashes.is_empty() {
                    continue;
                }
                let chunk_len = chunk_hashes.len();
                let destinations = reserve_compatibility(
                    capacity.as_ref(),
                    G2AllocationKind::RequiredStaging,
                    chunk_len,
                )
                .map_err(|error| {
                    ControlError::Internal(format!(
                        "pull: failed to reserve {chunk_len} G2 mutable blocks: {error}"
                    ))
                })?;
                let session_for_pull = Arc::clone(&session);
                let hashes_for_pull = chunk_hashes.clone();
                let filled = destinations
                    .transfer_with(move |mutables| {
                        session_for_pull.pull_resource(resource, hashes_for_pull, mutables)
                    })
                    .await
                    .map_err(|error| ControlError::Internal(format!("session.pull: {error:#}")))?;
                if filled.len() != chunk_len {
                    return Err(ControlError::Internal(format!(
                        "pull: session.pull returned {} blocks, expected {chunk_len}",
                        filled.len()
                    )));
                }
                staged.extend(
                    filled
                        .stage_all(&chunk_hashes, block_size)
                        .map_err(|error| {
                            ControlError::Internal(format!("stage pulled block: {error:#}"))
                        })?
                        .into_entries(),
                );
                pulled_set.extend(chunk_hashes);
                if pulled_set.len() == target_set.len() {
                    break 'drain;
                }
            }
            AvailabilityDelta::Verified(records) => {
                let chunk = records
                    .into_iter()
                    .filter(|record| {
                        target_set.contains(&record.block.hash)
                            && !pulled_set.contains(&record.block.hash)
                    })
                    .collect::<Vec<_>>();
                if chunk.is_empty() {
                    continue;
                }
                let chunk_hashes = chunk
                    .iter()
                    .map(|record| record.block.hash)
                    .collect::<Vec<_>>();
                let chunk_len = chunk.len();
                let destinations = reserve_compatibility(
                    capacity.as_ref(),
                    G2AllocationKind::RequiredStaging,
                    chunk_len,
                )
                .map_err(|error| {
                    ControlError::Internal(format!(
                        "pull: failed to reserve {chunk_len} G2 mutable blocks: {error}"
                    ))
                })?;
                let session_for_pull = Arc::clone(&session);
                let hashes_for_pull = chunk_hashes.clone();
                let filled = destinations
                    .transfer_with(move |mutables| {
                        session_for_pull.pull_resource(resource, hashes_for_pull, mutables)
                    })
                    .await
                    .map_err(|error| ControlError::Internal(format!("session.pull: {error:#}")))?;
                if filled.len() != chunk_len {
                    return Err(ControlError::Internal(format!(
                        "pull: session.pull returned {} blocks, expected {chunk_len}",
                        filled.len()
                    )));
                }
                if require_payload_integrity {
                    verify_payloads(leader, resource, &source_ordinals, &chunk, filled.blocks())
                        .await?;
                }
                staged.extend(
                    filled
                        .stage_all(&chunk_hashes, block_size)
                        .map_err(|error| {
                            ControlError::Internal(format!("stage pulled block: {error:#}"))
                        })?
                        .into_entries(),
                );
                pulled_set.extend(chunk_hashes);
                if pulled_set.len() == target_set.len() {
                    break 'drain;
                }
            }
            AvailabilityDelta::Drained => break 'drain,
        }
    }
    drop(availability);

    if pulled_set.len() != target_set.len() {
        return Err(ControlError::Internal(format!(
            "pull: availability drained with {} of {} target hashes pulled",
            pulled_set.len(),
            target_set.len()
        )));
    }
    let staged = order_selected(
        &target_hashes,
        staged
            .into_iter()
            .map(|(hash, block, guard)| (hash, (block, guard)))
            .collect(),
    )?;
    let staged = target_hashes
        .iter()
        .copied()
        .zip(staged)
        .map(|(hash, (block, guard))| (hash, block, guard))
        .collect();

    crate::engine_audit!(
        "transfer_pull_staged",
        session_id = %session.session_id(),
        resource = ?resource,
        pulled = target_hashes.len()
    );
    let allocation = RequiredStagingStagedAllocation::from_compatibility(
        G2StagedAllocation::from_entries(G2AllocationKind::RequiredStaging, staged),
        capacity,
    )
    .map_err(|error| {
        ControlError::Internal(format!("bind required-staging allocation: {error}"))
    })?;
    Ok(StagedPull {
        resource,
        hashes: target_hashes,
        allocation,
        breakdown: MatchBreakdown {
            host_blocks: pulled_set.len(),
            disk_blocks: 0,
            object_blocks: 0,
        },
    })
}

fn source_ordinals(committed: &[SequenceHash]) -> Result<HashMap<SequenceHash, u32>, ControlError> {
    committed
        .iter()
        .copied()
        .enumerate()
        .map(|(ordinal, hash)| {
            u32::try_from(ordinal)
                .map(|ordinal| (hash, ordinal))
                .map_err(|_| {
                    ControlError::Internal(
                        "payload_checksum_order_overflow: committed ordinal exceeds u32".to_owned(),
                    )
                })
        })
        .collect()
}

fn order_selected<T>(
    target_hashes: &[SequenceHash],
    staged: Vec<(SequenceHash, T)>,
) -> Result<Vec<T>, ControlError> {
    let mut by_hash = staged.into_iter().collect::<HashMap<_, _>>();
    let ordered = target_hashes
        .iter()
        .map(|hash| {
            by_hash.remove(hash).ok_or_else(|| {
                ControlError::Internal(format!(
                    "pull: selected hash {hash:?} has no staged destination"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !by_hash.is_empty() {
        return Err(ControlError::Internal(format!(
            "pull: {} staged destination(s) were outside the selector",
            by_hash.len()
        )));
    }
    Ok(ordered)
}

async fn verify_payloads(
    leader: &Arc<InstanceLeader>,
    resource: LogicalResourceId,
    target_ordinals: &std::collections::HashMap<SequenceHash, u32>,
    expected: &[super::session::VerifiedCommittedBlock],
    actual: &[kvbm_logical::MutableBlock<G2>],
) -> Result<(), ControlError> {
    let (expected_checksums, payload_blocks) =
        prepare_payload_verification(resource, target_ordinals, expected, actual)?;
    let actual_checksums = leader
        .payload_checksums(resource, &payload_blocks)
        .await
        .map_err(|error| {
            ControlError::Internal(format!(
                "payload_checksum_unavailable: resource {resource:?}: {error:#}"
            ))
        })?;
    for ((record, expected), actual) in expected
        .iter()
        .zip(expected_checksums)
        .zip(actual_checksums)
    {
        if expected != actual {
            return Err(ControlError::Internal(format!(
                "payload_checksum_mismatch: resource {resource:?} hash {:?}",
                record.block.hash
            )));
        }
    }
    Ok(())
}

fn prepare_payload_verification(
    resource: LogicalResourceId,
    target_ordinals: &std::collections::HashMap<SequenceHash, u32>,
    expected: &[super::session::VerifiedCommittedBlock],
    actual: &[kvbm_logical::MutableBlock<G2>],
) -> Result<(Vec<super::PayloadChecksum>, Vec<PayloadBlock>), ControlError> {
    let mut expected_checksums = Vec::with_capacity(expected.len());
    let mut payload_blocks = Vec::with_capacity(actual.len());
    for (record, block) in expected.iter().zip(actual) {
        let expected_ordinal = target_ordinals
            .get(&record.block.hash)
            .copied()
            .ok_or_else(|| {
                ControlError::Internal(format!(
                    "payload_checksum_unknown_hash: resource {resource:?} hash {:?}",
                    record.block.hash
                ))
            })?;
        if record.ordinal != expected_ordinal {
            return Err(ControlError::Internal(format!(
                "payload_checksum_order_mismatch: resource {resource:?} hash {:?} advertised ordinal {}, expected {expected_ordinal}",
                record.block.hash, record.ordinal
            )));
        }
        expected_checksums.push(record.checksum);
        payload_blocks.push(PayloadBlock {
            hash: record.block.hash,
            block_id: block.block_id(),
            ordinal: expected_ordinal,
        });
    }
    Ok((expected_checksums, payload_blocks))
}

async fn drain_committed(session: &Arc<dyn Session>) -> Vec<SequenceHash> {
    let mut stream = session.commits();
    let mut committed = Vec::new();
    let mut seen = HashSet::new();
    while let Some(delta) = stream.next().await {
        match delta {
            CommitDelta::Added(hashes) => {
                committed.extend(hashes.into_iter().filter(|hash| seen.insert(*hash)));
            }
            CommitDelta::Closed => break,
        }
    }
    committed
}

fn select_hashes(
    committed: Vec<SequenceHash>,
    selector: Option<Vec<SequenceHash>>,
) -> Result<Vec<SequenceHash>, ControlError> {
    let Some(selector) = selector else {
        return Ok(committed);
    };
    let committed = committed.into_iter().collect::<HashSet<_>>();
    let missing = selector
        .iter()
        .filter(|hash| !committed.contains(hash))
        .count();
    if missing != 0 {
        return Err(ControlError::Internal(format!(
            "hashes_not_committed: {missing} hash(es) in selector are not committed"
        )));
    }
    Ok(selector)
}

pub(super) fn resolve_g2_manager(
    leader: &InstanceLeader,
    requested: Option<LogicalResourceId>,
) -> Result<(LogicalResourceId, Arc<BlockManager<G2>>), ControlError> {
    let resource = requested.unwrap_or_else(|| leader.primary_g2_resource());
    let manager = leader.g2_manager_for(resource).cloned().ok_or_else(|| {
        ControlError::Internal(format!(
            "logical_resource_not_found: no G2 manager for resource {resource:?}"
        ))
    })?;
    Ok((resource, manager))
}

fn resolve_g2_capacity(
    leader: &InstanceLeader,
    requested: Option<LogicalResourceId>,
) -> Result<(LogicalResourceId, Arc<dyn G2Capacity>), ControlError> {
    let resource = requested.unwrap_or_else(|| leader.primary_g2_resource());
    let capacity = leader.g2_capacity_for(resource).ok_or_else(|| {
        ControlError::Internal(format!(
            "logical_resource_not_found: no G2 capacity for resource {resource:?}"
        ))
    })?;
    Ok((resource, capacity))
}

#[cfg(test)]
mod integrity_tests {
    use std::sync::Arc;

    use super::*;
    use crate::g2_capacity::DirectG2Capacity;
    use crate::g2_capacity::test_support::RecordingG2Capacity;
    use crate::p2p::PayloadChecksum;
    use crate::p2p::session::{
        CommittedBlock, MockSessionFactory, SessionFactory, VerifiedCommittedBlock,
    };
    use crate::testing::{create_messenger_tcp, managers::TestManagerBuilder};
    use kvbm_logical::blocks::BlockRegistry;

    #[tokio::test]
    async fn staged_pull_uses_the_injected_capacity_for_reserve_and_publish() {
        let registry = BlockRegistry::builder().build();
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(1)
                .block_size(4)
                .registry(registry.clone())
                .build(),
        );
        let capacity = Arc::new(RecordingG2Capacity::new(Arc::clone(&manager)));
        let leader = Arc::new(
            InstanceLeader::builder()
                .messenger(create_messenger_tcp().await.expect("test messenger"))
                .registry(registry)
                .g2_manager(manager)
                .g2_capacity(capacity.clone())
                .build()
                .expect("test leader"),
        );
        let hash = SequenceHash::new(11, None, 0);
        let factory = MockSessionFactory::new();
        factory
            .open(uuid::Uuid::new_v4())
            .expect("test session open");
        let session = factory.last_opened().expect("opened test session");
        session.inject_peer_commit(vec![hash]);
        session.inject_peer_finish_commits();
        session.inject_peer_available(vec![CommittedBlock {
            hash,
            peer_block_id: 0,
        }]);
        session.inject_peer_drained();

        let task = tokio::spawn({
            let leader = Arc::clone(&leader);
            let session: Arc<dyn Session> = session.clone();
            let capacity: Arc<dyn G2Capacity> = capacity.clone();
            async move {
                stage_attached(
                    &leader,
                    LogicalResourceId(0),
                    capacity,
                    session,
                    None,
                    false,
                )
                .await
            }
        });
        session.wait_pull_count(1).await;
        session.resolve_pull(0, Ok(()));

        let staged = task.await.expect("staged pull task").expect("staged pull");
        assert_eq!(
            capacity.allocation_kinds(),
            vec![G2AllocationKind::RequiredStaging]
        );
        assert_eq!(capacity.allocation_count(), 1);
        assert_eq!(capacity.registration_count(), 0);

        let registered = staged.publish().expect("publish staged pull");

        assert_eq!(registered.len(), 1);
        assert_eq!(capacity.registration_count(), 1);
        assert_eq!(capacity.registrations_with_live_lease(), 1);
        assert_eq!(capacity.lease_drop_count(), 1);
    }

    #[tokio::test]
    async fn compatibility_staging_keeps_fragmented_reservations() {
        let registry = BlockRegistry::builder().build();
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(3)
                .block_size(4)
                .registry(registry.clone())
                .build(),
        );
        let capacity = Arc::new(RecordingG2Capacity::new(Arc::clone(&manager)));
        let leader = Arc::new(
            InstanceLeader::builder()
                .messenger(create_messenger_tcp().await.expect("test messenger"))
                .registry(registry)
                .g2_manager(manager.clone())
                .g2_capacity(capacity.clone())
                .build()
                .expect("test leader"),
        );
        let hashes = vec![
            SequenceHash::new(21, None, 0),
            SequenceHash::new(22, Some(21), 1),
            SequenceHash::new(23, Some(22), 2),
        ];
        let factory = MockSessionFactory::new();
        factory
            .open(uuid::Uuid::new_v4())
            .expect("test session open");
        let session = factory.last_opened().expect("opened test session");
        session.inject_peer_commit(hashes.clone());
        session.inject_peer_finish_commits();
        session.inject_peer_available(vec![CommittedBlock {
            hash: hashes[0],
            peer_block_id: 0,
        }]);

        let task = tokio::spawn({
            let leader = Arc::clone(&leader);
            let session: Arc<dyn Session> = session.clone();
            let capacity: Arc<dyn G2Capacity> = capacity.clone();
            async move {
                stage_attached(
                    &leader,
                    LogicalResourceId(0),
                    capacity,
                    session,
                    None,
                    false,
                )
                .await
            }
        });
        session.wait_pull_count(1).await;

        assert_eq!(capacity.allocation_count(), 1);
        assert_eq!(session.pull_calls().len(), 1);
        assert_eq!(session.pull_calls()[0].0, hashes[..1]);
        assert_eq!(manager.available_blocks(), 2);

        session.inject_peer_available(vec![
            CommittedBlock {
                hash: hashes[1],
                peer_block_id: 1,
            },
            CommittedBlock {
                hash: hashes[2],
                peer_block_id: 2,
            },
        ]);
        session.inject_peer_drained();
        session.resolve_pull(0, Ok(()));
        session.wait_pull_count(2).await;

        assert_eq!(capacity.allocation_count(), 2);
        assert_eq!(session.pull_calls().len(), 2);
        assert_eq!(session.pull_calls()[1].0, hashes[1..]);
        assert_eq!(manager.available_blocks(), 0);

        session.resolve_pull(1, Ok(()));
        let staged = task.await.expect("staged pull task").expect("staged pull");

        assert_eq!(staged.hashes(), hashes);
        assert_eq!(capacity.registration_count(), 0);

        let registered = staged.publish().expect("publish staged pull");

        assert_eq!(registered.len(), hashes.len());
        assert_eq!(capacity.registration_count(), 1);
        assert_eq!(manager.match_blocks(&hashes).len(), hashes.len());
        assert_eq!(capacity.lease_drop_count(), 2);
    }

    #[test]
    fn advertised_payload_ordinal_must_match_source_commit_position() {
        let resource = LogicalResourceId(7);
        let hash = SequenceHash::new(11, None, 0);
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(2)
                .block_size(4)
                .build(),
        );
        let capacity = DirectG2Capacity::new(manager);
        let actual = reserve_compatibility(&capacity, G2AllocationKind::RequiredStaging, 1)
            .expect("test allocation");
        let expected = [VerifiedCommittedBlock {
            block: CommittedBlock {
                hash,
                peer_block_id: 0,
            },
            ordinal: 1,
            checksum: PayloadChecksum::from_test_bytes([3; 32]),
        }];
        let ordinals = std::collections::HashMap::from([(hash, 0)]);

        let error = prepare_payload_verification(resource, &ordinals, &expected, actual.blocks())
            .expect_err("wrong advertised order must fail before checksum comparison");
        assert!(
            error
                .to_string()
                .contains("payload_checksum_order_mismatch")
        );
    }

    #[test]
    fn selector_none_preserves_committed_stream_order() {
        let first = SequenceHash::new(1, None, 0);
        let second = SequenceHash::new(2, Some(1), 1);
        assert_eq!(
            select_hashes(vec![second, first], None).unwrap(),
            vec![second, first]
        );
    }

    #[test]
    fn selector_subset_keeps_source_commit_ordinal() {
        let first = SequenceHash::new(1, None, 0);
        let second = SequenceHash::new(2, Some(1), 1);
        let third = SequenceHash::new(3, Some(2), 2);

        let ordinals = source_ordinals(&[first, second, third]).unwrap();

        assert_eq!(ordinals[&third], 2);
    }

    #[test]
    fn reordered_selector_orders_results_without_rewriting_source_ordinals() {
        let first = SequenceHash::new(1, None, 0);
        let second = SequenceHash::new(2, Some(1), 1);
        let third = SequenceHash::new(3, Some(2), 2);
        let selector = vec![third, first];
        let ordinals = source_ordinals(&[first, second, third]).unwrap();

        let ordered = order_selected(&selector, vec![(first, "first"), (third, "third")]).unwrap();

        assert_eq!(ordered, vec!["third", "first"]);
        assert_eq!(ordinals[&third], 2);
        assert_eq!(ordinals[&first], 0);
    }
}
