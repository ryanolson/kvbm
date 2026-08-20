// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Requirement-neutral ownership for a required G2 staging transaction.

use std::future::Future;
use std::sync::Arc;

use anyhow::Result as AnyResult;
use kvbm_common::SequenceHash;
use kvbm_logical::blocks::{ImmutableBlock, MutableBlock};

use super::{
    G2Allocation, G2ExactAllocation, G2ExactPublishedRequiredStaging, G2ExactStagedAllocation,
    G2StagedAllocation,
};
#[cfg(test)]
use crate::BlockId;
use crate::G2;
use crate::g2_capacity::{
    G2AllocationKind, G2Capacity, G2CapacityError, G2CapacityRequest, G2CapacityRequirement,
    G2ReclaimCompletion,
};

/// One required-staging allocation with its original capacity contract.
///
/// Compatibility keeps the legacy lease and registration adapter. Exact keeps
/// the source-owned exact registration capability through transfer and staging.
#[must_use = "dropping this allocation releases its destination slots and capacity owner"]
pub(crate) struct RequiredStagingAllocation {
    inner: MutableRequiredStaging,
}

enum MutableRequiredStaging {
    Compatibility {
        allocation: G2Allocation,
        registration_owner: Arc<dyn G2Capacity>,
    },
    Exact(G2ExactAllocation),
}

/// One staged but unregistered required-staging allocation.
///
/// The block identifiers can drive a later physical transfer. Drop rolls every
/// staged slot back without publishing its hashes.
#[must_use = "publish this allocation or drop it to roll back every staged slot"]
pub(crate) struct RequiredStagingStagedAllocation {
    inner: StagedRequiredStaging,
}

/// A registered required-staging allocation that resets itself unless committed.
#[must_use = "commit this registration or drop it to reset every published block"]
pub(crate) struct RequiredStagingPublishedAllocation {
    inner: Option<PublishedRequiredStaging>,
}

enum StagedRequiredStaging {
    Compatibility {
        allocation: G2StagedAllocation,
        registration_owner: Arc<dyn G2Capacity>,
    },
    Exact(G2ExactStagedAllocation),
}

enum PublishedRequiredStaging {
    Compatibility(CompatibilityPublishedAllocation),
    Exact(G2ExactPublishedRequiredStaging),
}

struct CompatibilityPublishedAllocation {
    blocks: Option<Vec<ImmutableBlock<G2>>>,
}

impl RequiredStagingAllocation {
    /// Return the allocation's capacity requirement.
    #[cfg(test)]
    pub(crate) const fn requirement(&self) -> G2CapacityRequirement {
        match &self.inner {
            MutableRequiredStaging::Compatibility { .. } => G2CapacityRequirement::Compatibility,
            MutableRequiredStaging::Exact(_) => G2CapacityRequirement::ExactReclaim,
        }
    }

    /// Return the number of reserved blocks.
    pub(crate) fn len(&self) -> usize {
        match &self.inner {
            MutableRequiredStaging::Compatibility { allocation, .. } => allocation.len(),
            MutableRequiredStaging::Exact(allocation) => allocation.len(),
        }
    }

    /// Transfer mutable blocks and preserve their original capacity owner.
    pub(crate) async fn transfer_with<F, Fut, E>(self, transfer: F) -> Result<Self, E>
    where
        F: FnOnce(Vec<MutableBlock<G2>>) -> Fut,
        Fut: Future<Output = Result<Vec<MutableBlock<G2>>, E>>,
    {
        match self.inner {
            MutableRequiredStaging::Compatibility {
                allocation,
                registration_owner,
            } => allocation
                .transfer_with(transfer)
                .await
                .map(|allocation| Self {
                    inner: MutableRequiredStaging::Compatibility {
                        allocation,
                        registration_owner,
                    },
                }),
            MutableRequiredStaging::Exact(allocation) => allocation
                .transfer_with(transfer)
                .await
                .map(|allocation| Self {
                    inner: MutableRequiredStaging::Exact(allocation),
                }),
        }
    }

    /// Stage every block without publishing it.
    pub(crate) fn stage_all(
        self,
        hashes: &[SequenceHash],
        block_size: usize,
    ) -> AnyResult<RequiredStagingStagedAllocation> {
        match self.inner {
            MutableRequiredStaging::Compatibility {
                allocation,
                registration_owner,
            } => allocation.stage_all(hashes, block_size).map(|allocation| {
                RequiredStagingStagedAllocation {
                    inner: StagedRequiredStaging::Compatibility {
                        allocation,
                        registration_owner,
                    },
                }
            }),
            MutableRequiredStaging::Exact(allocation) => allocation
                .stage_all(hashes, block_size)
                .map(|allocation| RequiredStagingStagedAllocation {
                    inner: StagedRequiredStaging::Exact(allocation),
                }),
        }
    }
}

impl RequiredStagingStagedAllocation {
    /// Bind a compatibility allocation to its original registration owner.
    pub(crate) fn from_compatibility(
        allocation: G2StagedAllocation,
        registration_owner: Arc<dyn G2Capacity>,
    ) -> Result<Self, G2CapacityError> {
        if allocation.kind() != G2AllocationKind::RequiredStaging {
            return Err(G2CapacityError::Rejected(format!(
                "capacity returned {:?} allocation for a RequiredStaging route",
                allocation.kind()
            )));
        }
        Ok(Self {
            inner: StagedRequiredStaging::Compatibility {
                allocation,
                registration_owner,
            },
        })
    }

    /// Return staged block identifiers without publishing them.
    #[cfg(test)]
    pub(crate) fn block_ids(&self) -> Vec<BlockId> {
        match &self.inner {
            StagedRequiredStaging::Compatibility { allocation, .. } => allocation.block_ids(),
            StagedRequiredStaging::Exact(allocation) => allocation.block_ids(),
        }
    }

    /// Publish every staged hash through its original capacity owner.
    pub(crate) fn publish(self) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        self.publish_reversible()
            .map(RequiredStagingPublishedAllocation::commit)
    }

    /// Publish while retaining reset authority for an outer bundle transaction.
    pub(crate) fn publish_reversible(
        self,
    ) -> Result<RequiredStagingPublishedAllocation, G2CapacityError> {
        match self.inner {
            StagedRequiredStaging::Compatibility {
                allocation,
                registration_owner,
            } => registration_owner
                .register_compatibility(allocation)
                .map(|blocks| RequiredStagingPublishedAllocation {
                    inner: Some(PublishedRequiredStaging::Compatibility(
                        CompatibilityPublishedAllocation {
                            blocks: Some(blocks),
                        },
                    )),
                }),
            StagedRequiredStaging::Exact(allocation) => allocation
                .register_required_staging_reversible()
                .map(|allocation| RequiredStagingPublishedAllocation {
                    inner: Some(PublishedRequiredStaging::Exact(allocation)),
                }),
        }
    }
}

impl RequiredStagingPublishedAllocation {
    /// Disarm rollback after every resource registration succeeds.
    pub(crate) fn commit(mut self) -> Vec<ImmutableBlock<G2>> {
        match self
            .inner
            .take()
            .expect("required-staging registration commits once")
        {
            PublishedRequiredStaging::Compatibility(allocation) => allocation.commit(),
            PublishedRequiredStaging::Exact(allocation) => allocation.commit(),
        }
    }
}

impl CompatibilityPublishedAllocation {
    fn commit(mut self) -> Vec<ImmutableBlock<G2>> {
        self.blocks
            .take()
            .expect("compatibility registration commits once")
    }
}

impl Drop for CompatibilityPublishedAllocation {
    fn drop(&mut self) {
        let Some(blocks) = self.blocks.take() else {
            return;
        };
        for block in &blocks {
            block.set_evict_on_reset(true);
        }
        drop(blocks);
    }
}

/// Reserve all blocks for one required-staging transaction.
///
/// Exact reclaim is preferred. Compatibility is used only when the capacity
/// adapter explicitly reports that exact reclaim is unsupported. One pending
/// reclaim receives one bounded completion attempt.
pub(crate) fn reserve_required_staging(
    capacity: Arc<dyn G2Capacity>,
    count: usize,
) -> Result<RequiredStagingAllocation, G2CapacityError> {
    let exact = G2CapacityRequest::exact_reclaim(G2AllocationKind::RequiredStaging, count);
    match capacity.reserve(exact) {
        Ok(decision) => exact_allocation(decision, count),
        Err(G2CapacityError::ExactReclaimUnsupported(request)) if request == exact => {
            reserve_compatibility_required_staging(capacity, count)
        }
        Err(error) => Err(error),
    }
}

fn exact_allocation(
    decision: crate::g2_capacity::G2CapacityDecision,
    expected_count: usize,
) -> Result<RequiredStagingAllocation, G2CapacityError> {
    match decision {
        crate::g2_capacity::G2CapacityDecision::ExactGranted(allocation) => {
            exact_required_staging(allocation, expected_count)
        }
        crate::g2_capacity::G2CapacityDecision::Granted(_) => {
            Err(G2CapacityError::RequirementMismatch {
                expected: G2CapacityRequirement::ExactReclaim,
                actual: G2CapacityRequirement::Compatibility,
            })
        }
        crate::g2_capacity::G2CapacityDecision::PendingReclaim(pending) => {
            match pending.complete() {
                G2ReclaimCompletion::Granted(allocation) => {
                    exact_required_staging(allocation, expected_count)
                }
                G2ReclaimCompletion::PendingReclaim(pending) => {
                    Err(G2CapacityError::PendingReclaim {
                        target_count: pending.plan().target_count(),
                    })
                }
                G2ReclaimCompletion::Rejected(error) => Err(error),
            }
        }
    }
}

fn reserve_compatibility_required_staging(
    capacity: Arc<dyn G2Capacity>,
    count: usize,
) -> Result<RequiredStagingAllocation, G2CapacityError> {
    let request = G2CapacityRequest::compatibility(G2AllocationKind::RequiredStaging, count);
    match capacity.reserve(request)? {
        crate::g2_capacity::G2CapacityDecision::Granted(allocation) => {
            validate_required_staging_grant(allocation.kind(), allocation.len(), count)?;
            Ok(RequiredStagingAllocation {
                inner: MutableRequiredStaging::Compatibility {
                    allocation,
                    registration_owner: capacity,
                },
            })
        }
        crate::g2_capacity::G2CapacityDecision::ExactGranted(_) => {
            Err(G2CapacityError::RequirementMismatch {
                expected: G2CapacityRequirement::Compatibility,
                actual: G2CapacityRequirement::ExactReclaim,
            })
        }
        crate::g2_capacity::G2CapacityDecision::PendingReclaim(pending) => {
            Err(G2CapacityError::PendingReclaim {
                target_count: pending.plan().target_count(),
            })
        }
    }
}

fn exact_required_staging(
    allocation: G2ExactAllocation,
    expected_count: usize,
) -> Result<RequiredStagingAllocation, G2CapacityError> {
    validate_required_staging_grant(allocation.kind(), allocation.len(), expected_count)?;
    Ok(RequiredStagingAllocation {
        inner: MutableRequiredStaging::Exact(allocation),
    })
}

fn validate_required_staging_grant(
    actual_kind: G2AllocationKind,
    actual_count: usize,
    expected_count: usize,
) -> Result<(), G2CapacityError> {
    if actual_kind != G2AllocationKind::RequiredStaging {
        return Err(G2CapacityError::Rejected(format!(
            "capacity returned {actual_kind:?} allocation for a RequiredStaging request"
        )));
    }
    if actual_count != expected_count {
        return Err(G2CapacityError::Rejected(format!(
            "capacity returned {actual_count} blocks for a {expected_count}-block RequiredStaging request"
        )));
    }
    Ok(())
}
