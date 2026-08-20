// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-reservation staging for a sealed complete resource lineage.

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures::StreamExt;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::control::ControlError;
use kvbm_protocols::control::modules::transfer::MatchBreakdown;

use super::{StagedPull, drain_committed, select_hashes, source_ordinals, verify_payloads};
use crate::g2_capacity::{G2Capacity, reserve_required_staging};
use crate::leader::InstanceLeader;
use crate::p2p::session::{AvailabilityDelta, Session, VerifiedCommittedBlock};

/// Stage one sealed, complete lineage through one destination reservation.
pub(super) async fn stage_attached(
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
        return Err(ControlError::Internal(
            "complete_lineage_empty: a complete pull requires at least one hash".to_owned(),
        ));
    }
    if target_hashes.iter().copied().collect::<HashSet<_>>().len() != target_hashes.len() {
        return Err(ControlError::Internal(
            "complete_lineage_duplicate: a complete pull requires unique hashes".to_owned(),
        ));
    }

    let verified =
        wait_for_availability(&session, &target_hashes, require_payload_integrity).await?;
    let block_size = capacity.block_size();
    let allocation = reserve_required_staging(capacity, target_hashes.len()).map_err(|error| {
        ControlError::Internal(format!(
            "pull: failed to reserve complete G2 lineage of {} blocks: {error}",
            target_hashes.len()
        ))
    })?;
    let leader_for_pull = Arc::clone(leader);
    let session_for_pull = Arc::clone(&session);
    let hashes_for_pull = target_hashes.clone();
    let ordinals_for_pull = source_ordinals;
    let filled = allocation
        .transfer_with(move |mutables| async move {
            let filled = session_for_pull
                .pull_resource(resource, hashes_for_pull.clone(), mutables)
                .await
                .map_err(|error| ControlError::Internal(format!("session.pull: {error:#}")))?;
            if filled.len() != hashes_for_pull.len() {
                return Err(ControlError::Internal(format!(
                    "pull: session.pull returned {} blocks, expected {}",
                    filled.len(),
                    hashes_for_pull.len()
                )));
            }
            if require_payload_integrity {
                verify_payloads(
                    &leader_for_pull,
                    resource,
                    &ordinals_for_pull,
                    &verified,
                    &filled,
                )
                .await?;
            }
            Ok(filled)
        })
        .await?;
    let staged = filled
        .stage_all(&target_hashes, block_size)
        .map_err(|error| {
            ControlError::Internal(format!("stage complete pulled lineage: {error:#}"))
        })?;

    crate::engine_audit!(
        "transfer_pull_staged",
        session_id = %session.session_id(),
        resource = ?resource,
        pulled = target_hashes.len()
    );
    Ok(StagedPull {
        resource,
        hashes: target_hashes.clone(),
        allocation: staged,
        breakdown: MatchBreakdown {
            host_blocks: target_hashes.len(),
            disk_blocks: 0,
            object_blocks: 0,
        },
    })
}

async fn wait_for_availability(
    session: &Arc<dyn Session>,
    target_hashes: &[SequenceHash],
    require_payload_integrity: bool,
) -> Result<Vec<VerifiedCommittedBlock>, ControlError> {
    let target_set = target_hashes.iter().copied().collect::<HashSet<_>>();
    let mut seen = HashSet::with_capacity(target_set.len());
    let mut verified = HashMap::with_capacity(target_set.len());
    let mut availability = session.availability();
    'drain: while let Some(delta) = availability.next().await {
        match delta {
            AvailabilityDelta::Available(blocks) => {
                for block in blocks {
                    if !target_set.contains(&block.hash) || seen.contains(&block.hash) {
                        continue;
                    }
                    if require_payload_integrity {
                        return Err(ControlError::Internal(
                            "payload_checksum_unverified: bundle pull received unverified availability"
                                .to_owned(),
                        ));
                    }
                    seen.insert(block.hash);
                }
            }
            AvailabilityDelta::Verified(records) => {
                for record in records {
                    let hash = record.block.hash;
                    if target_set.contains(&hash) && seen.insert(hash) {
                        verified.insert(hash, record);
                    }
                }
            }
            AvailabilityDelta::Drained => break 'drain,
        }
        if seen.len() == target_set.len() {
            break 'drain;
        }
    }
    drop(availability);

    if seen.len() != target_set.len() {
        return Err(ControlError::Internal(format!(
            "pull: availability drained with {} of {} target hashes ready",
            seen.len(),
            target_set.len()
        )));
    }
    if !require_payload_integrity {
        return Ok(Vec::new());
    }
    target_hashes
        .iter()
        .map(|hash| {
            verified.remove(hash).ok_or_else(|| {
                ControlError::Internal(format!(
                    "payload_checksum_unverified: complete lineage hash {hash:?} lacks verified availability"
                ))
            })
        })
        .collect()
}
