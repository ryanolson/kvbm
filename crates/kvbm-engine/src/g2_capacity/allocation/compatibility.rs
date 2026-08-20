// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility allocation types.

use std::future::Future;
use std::sync::Arc;

use anyhow::{Result as AnyResult, ensure};
use kvbm_common::SequenceHash;
use kvbm_logical::blocks::{CompleteBlock, MutableBlock};

use super::G2CacheExtensionResidencyLease;
use super::storage::{MutableAllocation, StagedAllocation};
use crate::g2_capacity::{G2AllocationKind, G2CapacityRequirement, G2LeaseGuard};
use crate::{BlockId, G2};

/// Opaque mutable G2 allocation for a compatibility route.
///
/// Direct allocations are internal to the compatibility adapter.
///
/// ```compile_fail
/// use kvbm_engine::g2_capacity::{G2Allocation, G2AllocationKind};
///
/// let _ = G2Allocation::direct(G2AllocationKind::RequiredStaging, Vec::new());
/// ```
///
/// Compatibility allocations cannot rewrap mutable blocks outside this crate.
///
/// ```compile_fail
/// use std::sync::Arc;
/// use kvbm_engine::G2;
/// use kvbm_engine::g2_capacity::{G2Allocation, G2AllocationKind, G2LeaseGuard};
/// use kvbm_logical::MutableBlock;
///
/// fn forge(blocks: Vec<MutableBlock<G2>>) {
///     let _ = G2Allocation::new(
///         G2AllocationKind::RequiredStaging,
///         blocks,
///         Arc::new(()) as Arc<dyn G2LeaseGuard>,
///     );
/// }
/// ```
///
/// External callers cannot take mutable blocks away from the capacity lease.
///
/// ```compile_fail
/// use kvbm_engine::G2;
/// use kvbm_engine::g2_capacity::G2Allocation;
/// use kvbm_logical::MutableBlock;
///
/// async fn steal(allocation: G2Allocation) {
///     let _: Result<G2Allocation, Vec<MutableBlock<G2>>> = allocation
///         .transfer_with(|blocks| async move { Err(blocks) })
///         .await;
/// }
/// ```
#[must_use = "if you drop this allocation, it releases mutable blocks and its capacity lease"]
pub struct G2Allocation {
    allocation: MutableAllocation,
}

/// Opaque staged G2 allocation for a compatibility route.
///
/// External callers cannot register staged blocks through an arbitrary callback.
///
/// ```compile_fail
/// use kvbm_engine::g2_capacity::G2StagedAllocation;
///
/// fn bypass_capacity(allocation: G2StagedAllocation) {
///     let _ = allocation.register_with(|blocks| blocks);
/// }
/// ```
///
/// External callers cannot retain a lease around an arbitrary registration callback.
///
/// ```compile_fail
/// use kvbm_engine::g2_capacity::G2StagedAllocation;
///
/// fn bypass_capacity(allocation: G2StagedAllocation) {
///     let _ = allocation.register_with_cache_residency(|blocks| blocks);
/// }
/// ```
#[must_use = "if you drop these staged blocks, the drop rolls back their slots and releases their leases"]
pub struct G2StagedAllocation {
    allocation: StagedAllocation,
}

impl G2Allocation {
    /// Construct a compatibility allocation with a capacity lease.
    pub(crate) fn new(
        kind: G2AllocationKind,
        blocks: Vec<MutableBlock<G2>>,
        guard: Arc<dyn G2LeaseGuard>,
    ) -> Self {
        Self {
            allocation: MutableAllocation::new(kind, blocks, guard),
        }
    }

    /// Construct a compatibility allocation without an added lease.
    pub(crate) fn direct(kind: G2AllocationKind, blocks: Vec<MutableBlock<G2>>) -> Self {
        Self::new(kind, blocks, Arc::new(()))
    }

    /// Return the compatibility registration requirement.
    pub const fn requirement(&self) -> G2CapacityRequirement {
        G2CapacityRequirement::Compatibility
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

    /// Pass mutable blocks through an asynchronous transfer operation.
    pub(crate) async fn transfer_with<F, Fut, E>(self, transfer: F) -> Result<Self, E>
    where
        F: FnOnce(Vec<MutableBlock<G2>>) -> Fut,
        Fut: Future<Output = Result<Vec<MutableBlock<G2>>, E>>,
    {
        self.allocation
            .transfer_with(transfer)
            .await
            .map(|allocation| Self { allocation })
    }

    /// Stage every mutable block with its sequence hash.
    pub fn stage_all(
        self,
        hashes: &[SequenceHash],
        block_size: usize,
    ) -> AnyResult<G2StagedAllocation> {
        self.allocation
            .stage_all(hashes, block_size)
            .map(|allocation| G2StagedAllocation { allocation })
    }

    /// Stage selected mutable blocks and release the other blocks.
    pub fn stage_selected<I>(
        self,
        hashes: &[SequenceHash],
        block_size: usize,
        selected: I,
    ) -> AnyResult<G2StagedAllocation>
    where
        I: IntoIterator<Item = bool>,
    {
        self.allocation
            .stage_selected(hashes, block_size, selected)
            .map(|allocation| G2StagedAllocation { allocation })
    }
}

impl G2StagedAllocation {
    /// Return the allocation policy.
    pub const fn kind(&self) -> G2AllocationKind {
        self.allocation.kind()
    }

    /// Return the compatibility registration requirement.
    pub const fn requirement(&self) -> G2CapacityRequirement {
        G2CapacityRequirement::Compatibility
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

    /// Run compatibility registration while each lease remains held.
    pub(crate) fn register_with<Output>(
        self,
        register: impl FnOnce(Vec<CompleteBlock<G2>>) -> Output,
    ) -> Output {
        self.allocation.register_with(register)
    }

    /// Register a CacheExtension and retain its compatibility residency lease.
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the internal compatibility lease tests use this registration path"
        )
    )]
    pub(crate) fn register_with_cache_residency<Output>(
        self,
        register: impl FnOnce(Vec<CompleteBlock<G2>>) -> Output,
    ) -> AnyResult<(Output, G2CacheExtensionResidencyLease)> {
        ensure!(
            self.kind() == G2AllocationKind::CacheExtension,
            "only CacheExtension registrations retain a residency lease"
        );
        let block_count = self.len();
        let (output, guards) = self.allocation.register_with_retained_guards(register);
        Ok((
            output,
            G2CacheExtensionResidencyLease::new(block_count, guards),
        ))
    }

    pub(crate) fn direct(
        kind: G2AllocationKind,
        hashes: Vec<SequenceHash>,
        blocks: Vec<CompleteBlock<G2>>,
    ) -> AnyResult<Self> {
        StagedAllocation::direct(kind, hashes, blocks).map(|allocation| Self { allocation })
    }

    pub(crate) fn into_entries(
        self,
    ) -> Vec<(SequenceHash, CompleteBlock<G2>, Arc<dyn G2LeaseGuard>)> {
        self.allocation.into_entries()
    }

    pub(crate) fn from_entries(
        kind: G2AllocationKind,
        entries: Vec<(SequenceHash, CompleteBlock<G2>, Arc<dyn G2LeaseGuard>)>,
    ) -> Self {
        Self {
            allocation: StagedAllocation::from_entries(kind, entries),
        }
    }
}
