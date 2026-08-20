// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact allocation types and source-owned registration.

use std::future::Future;
use std::sync::Arc;

use anyhow::Result as AnyResult;
use kvbm_common::SequenceHash;
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock, MutableBlock};

use super::storage::{MutableAllocation, StagedAllocation};
use crate::g2_capacity::{
    G2AllocationKind, G2CapacityError, G2CapacityRequest, G2CapacityRequirement, G2LeaseGuard,
};
use crate::{BlockId, G2};

/// Opaque mutable G2 allocation for an exact route.
///
/// Its source adapter supplies the registration owner. The owner remains
/// attached through transfer and staging.
///
/// ```compile_fail
/// use kvbm_engine::g2_capacity::{G2CapacityRequest, G2ExactAllocation};
///
/// fn forge_without_source_owner(request: G2CapacityRequest) {
///     let _ = G2ExactAllocation::new(request, Vec::new());
/// }
/// ```
///
/// External callers cannot take mutable blocks away from the source owner.
///
/// ```compile_fail
/// use kvbm_engine::G2;
/// use kvbm_engine::g2_capacity::G2ExactAllocation;
/// use kvbm_logical::MutableBlock;
///
/// async fn steal(allocation: G2ExactAllocation) {
///     let _: Result<G2ExactAllocation, Vec<MutableBlock<G2>>> = allocation
///         .transfer_with(|blocks| async move { Err(blocks) })
///         .await;
/// }
/// ```
#[must_use = "if you drop this allocation, it releases mutable blocks and its exact registration owner"]
pub struct G2ExactAllocation {
    allocation: MutableAllocation,
    registration_owner: Box<dyn G2ExactRegistrationOwner>,
}

/// Opaque staged G2 allocation for an exact route.
///
/// Call [`Self::register`] to use its source-bound registration owner.
///
/// ```compile_fail
/// use kvbm_engine::g2_capacity::G2ExactStagedAllocation;
///
/// fn bypass_owner(allocation: G2ExactStagedAllocation) {
///     let _ = allocation.register_with(|blocks| blocks);
/// }
/// ```
///
/// ```compile_fail
/// use kvbm_engine::g2_capacity::{G2Capacity, G2ExactStagedAllocation};
///
/// fn use_wrong_adapter(capacity: &dyn G2Capacity, allocation: G2ExactStagedAllocation) {
///     let _ = capacity.register_exact(allocation);
/// }
/// ```
///
/// ```compile_fail
/// use kvbm_engine::g2_capacity::{G2Capacity, G2ExactStagedAllocation};
///
/// fn use_compatibility_path(capacity: &dyn G2Capacity, allocation: G2ExactStagedAllocation) {
///     let _ = capacity.register_compatibility(allocation);
/// }
/// ```
#[must_use = "if you drop these staged blocks, the drop rolls back their slots and cancels the exact owner"]
pub struct G2ExactStagedAllocation {
    allocation: StagedAllocation,
    registration_owner: Box<dyn G2ExactRegistrationOwner>,
}

/// A registered required-staging allocation that can still roll back.
pub(crate) struct G2ExactPublishedRequiredStaging {
    blocks: Option<Vec<ImmutableBlock<G2>>>,
    registration_owner: Option<Box<dyn G2ExactRegistrationOwner>>,
}

/// A compatibility CacheExtension lease for registered cache residency.
///
/// Exact registration retains its owner permit in source state. Exact callers
/// cannot obtain or release that permit.
#[must_use = "if you drop this lease, it releases compatibility CacheExtension capacity"]
pub struct G2CacheExtensionResidencyLease {
    block_count: usize,
    _guards: Vec<Arc<dyn G2LeaseGuard>>,
}

/// Blocks registered by an exact-capacity owner.
pub struct G2RegisteredAllocation {
    kind: G2AllocationKind,
    blocks: Vec<ImmutableBlock<G2>>,
}

/// One-shot registration authority for one exact allocation.
///
/// A source adapter implements this cross-crate capability. The owner holds
/// the exact permit and retains it for CacheExtension residency.
pub trait G2ExactRegistrationOwner: Send {
    /// Register blocks through source adapter state.
    fn register_blocks(
        &mut self,
        blocks: Vec<CompleteBlock<G2>>,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError>;

    /// Reset a successful required-staging registration after bundle failure.
    fn rollback_required_staging(&mut self, blocks: Vec<ImmutableBlock<G2>>);

    /// Retain the exact permit for CacheExtension residency.
    fn retain_cache_extension(self: Box<Self>);
}

impl G2ExactAllocation {
    /// Construct an exact allocation with its source adapter capability.
    ///
    /// The owner is the only exact cancellation unit. This constructor does
    /// not accept a separate exact lease guard.
    pub fn new(
        request: G2CapacityRequest,
        blocks: Vec<MutableBlock<G2>>,
        registration_owner: Box<dyn G2ExactRegistrationOwner>,
    ) -> Result<Self, G2CapacityError> {
        if request.requirement() != G2CapacityRequirement::ExactReclaim {
            return Err(G2CapacityError::RequirementMismatch {
                expected: G2CapacityRequirement::ExactReclaim,
                actual: request.requirement(),
            });
        }
        Ok(Self {
            allocation: MutableAllocation::new(request.kind(), blocks, Arc::new(())),
            registration_owner,
        })
    }

    /// Return the exact registration requirement.
    pub const fn requirement(&self) -> G2CapacityRequirement {
        G2CapacityRequirement::ExactReclaim
    }

    /// Return the allocation policy.
    pub const fn kind(&self) -> G2AllocationKind {
        self.allocation.kind()
    }

    /// Return the number of mutable blocks.
    pub fn len(&self) -> usize {
        self.allocation.len()
    }

    /// Return true when this allocation has no mutable blocks.
    pub fn is_empty(&self) -> bool {
        self.allocation.is_empty()
    }

    /// Return block identifiers without releasing the lease.
    pub(crate) fn block_ids(&self) -> Vec<BlockId> {
        self.allocation.block_ids()
    }

    /// Borrow mutable blocks while this allocation retains its lease.
    pub fn blocks(&self) -> &[MutableBlock<G2>] {
        self.allocation.blocks()
    }

    /// Transfer mutable blocks and preserve the source registration owner.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the internal contract test covers exact transfer ownership"
        )
    )]
    pub(crate) async fn transfer_with<F, Fut, E>(self, transfer: F) -> Result<Self, E>
    where
        F: FnOnce(Vec<MutableBlock<G2>>) -> Fut,
        Fut: Future<Output = Result<Vec<MutableBlock<G2>>, E>>,
    {
        let Self {
            allocation,
            registration_owner,
        } = self;
        allocation
            .transfer_with(transfer)
            .await
            .map(|allocation| Self {
                allocation,
                registration_owner,
            })
    }

    /// Stage every mutable block and preserve the source registration owner.
    pub fn stage_all(
        self,
        hashes: &[SequenceHash],
        block_size: usize,
    ) -> AnyResult<G2ExactStagedAllocation> {
        let selected = vec![true; self.len()];
        self.stage_selected(hashes, block_size, selected)
    }

    /// Stage selected blocks and preserve the source registration owner.
    pub fn stage_selected<I>(
        self,
        hashes: &[SequenceHash],
        block_size: usize,
        selected: I,
    ) -> AnyResult<G2ExactStagedAllocation>
    where
        I: IntoIterator<Item = bool>,
    {
        let Self {
            allocation,
            registration_owner,
        } = self;
        allocation
            .stage_selected(hashes, block_size, selected)
            .map(|allocation| G2ExactStagedAllocation {
                allocation,
                registration_owner,
            })
    }
}

impl G2ExactStagedAllocation {
    /// Return the allocation policy.
    pub const fn kind(&self) -> G2AllocationKind {
        self.allocation.kind()
    }

    /// Return the exact registration requirement.
    pub const fn requirement(&self) -> G2CapacityRequirement {
        G2CapacityRequirement::ExactReclaim
    }

    /// Return staged hashes in registration order.
    pub fn hashes(&self) -> &[SequenceHash] {
        self.allocation.hashes()
    }

    /// Return the number of staged blocks.
    pub fn len(&self) -> usize {
        self.allocation.len()
    }

    /// Return true when this allocation has no staged blocks.
    pub fn is_empty(&self) -> bool {
        self.allocation.is_empty()
    }

    /// Return staged block identifiers without consuming the allocation.
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.allocation.block_ids()
    }

    /// Consume this allocation through its source registration owner.
    pub fn register(self) -> Result<G2RegisteredAllocation, G2CapacityError> {
        let Self {
            allocation,
            mut registration_owner,
        } = self;
        let kind = allocation.kind();
        let blocks = match kind {
            G2AllocationKind::CacheExtension => {
                let blocks = allocation
                    .register_with(|blocks| registration_owner.register_blocks(blocks))?;
                registration_owner.retain_cache_extension();
                blocks
            }
            G2AllocationKind::RequiredStaging => {
                allocation.register_with(|blocks| registration_owner.register_blocks(blocks))?
            }
        };
        Ok(G2RegisteredAllocation { kind, blocks })
    }

    /// Register required-staging blocks while retaining rollback authority.
    pub(crate) fn register_required_staging_reversible(
        self,
    ) -> Result<G2ExactPublishedRequiredStaging, G2CapacityError> {
        let Self {
            allocation,
            mut registration_owner,
        } = self;
        if allocation.kind() != G2AllocationKind::RequiredStaging {
            return Err(G2CapacityError::Rejected(
                "reversible exact registration requires RequiredStaging".to_owned(),
            ));
        }
        let blocks =
            allocation.register_with(|blocks| registration_owner.register_blocks(blocks))?;
        Ok(G2ExactPublishedRequiredStaging {
            blocks: Some(blocks),
            registration_owner: Some(registration_owner),
        })
    }
}

impl G2ExactPublishedRequiredStaging {
    pub(crate) fn commit(mut self) -> Vec<ImmutableBlock<G2>> {
        self.registration_owner.take();
        self.blocks
            .take()
            .expect("exact reversible registration commits once")
    }
}

impl Drop for G2ExactPublishedRequiredStaging {
    fn drop(&mut self) {
        let Some(blocks) = self.blocks.take() else {
            return;
        };
        self.registration_owner
            .as_mut()
            .expect("live exact registration retains its rollback owner")
            .rollback_required_staging(blocks);
    }
}

impl G2CacheExtensionResidencyLease {
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the internal compatibility lease tests construct this lease"
        )
    )]
    pub(super) fn new(block_count: usize, guards: Vec<Arc<dyn G2LeaseGuard>>) -> Self {
        let mut unique_guards: Vec<Arc<dyn G2LeaseGuard>> = Vec::new();
        for guard in guards {
            if !unique_guards
                .iter()
                .any(|existing| Arc::ptr_eq(existing, &guard))
            {
                unique_guards.push(guard);
            }
        }
        Self {
            block_count,
            _guards: unique_guards,
        }
    }

    /// Return the number of registered blocks represented by this lease.
    pub const fn len(&self) -> usize {
        self.block_count
    }

    /// Return true when this lease has no registered blocks.
    pub const fn is_empty(&self) -> bool {
        self.block_count == 0
    }

    #[cfg(test)]
    pub(crate) fn unique_guard_count(&self) -> usize {
        self._guards.len()
    }
}

impl G2RegisteredAllocation {
    /// Return the allocation policy.
    pub const fn kind(&self) -> G2AllocationKind {
        self.kind
    }

    /// Borrow the registered blocks.
    pub fn blocks(&self) -> &[ImmutableBlock<G2>] {
        &self.blocks
    }

    /// Consume this value and return its registered blocks.
    pub fn into_blocks(self) -> Vec<ImmutableBlock<G2>> {
        self.blocks
    }
}
