// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exclusive store ownership for an inactive lineage pressure action.

use std::sync::Arc;

use crate::blocks::BlockMetadata;
use crate::pools::InactiveCandidate;
use crate::registry::BlockRegistrationHandle;
use crate::{BlockId, SequenceHash};

use super::{BlockSlot, BlockStore, BlockStoreInner, SlotState};

/// Store-level ownership of a complete inactive lineage.
pub(crate) struct StoreInactiveLineageHold<T: BlockMetadata> {
    store: Arc<BlockStore<T>>,
    source_blocks: Vec<(SequenceHash, BlockId)>,
}

impl<T: BlockMetadata> StoreInactiveLineageHold<T> {
    pub(crate) fn source_blocks(&self) -> &[(SequenceHash, BlockId)] {
        &self.source_blocks
    }

    /// Restore support blocks and release only the selected leaf.
    pub(crate) fn commit_victim_release(mut self) -> Option<SequenceHash> {
        let source_blocks = std::mem::take(&mut self.source_blocks);
        self.store.commit_lineage_hold(source_blocks)
    }
}

impl<T: BlockMetadata> Drop for StoreInactiveLineageHold<T> {
    fn drop(&mut self) {
        if !self.source_blocks.is_empty() {
            let source_blocks = std::mem::take(&mut self.source_blocks);
            self.store.restore_lineage_hold(source_blocks);
        }
    }
}

impl<T: BlockMetadata> BlockStore<T> {
    /// Snapshot one complete inactive lineage from its current leaf hash.
    ///
    /// This point lookup resolves the private pool slot, current tenures, and
    /// advisory fields under one store lock. It returns no physical identity
    /// to callers outside the logical manager.
    pub(crate) fn preflight_inactive_lineage_by_hash(
        &self,
        seq_hash: SequenceHash,
    ) -> Option<(InactiveCandidate, Vec<(SequenceHash, BlockId)>)> {
        let inner = self.inner.lock();
        if inner.held_by_hash.contains_key(&seq_hash) {
            return None;
        }
        let block_id = inner.inactive.exact_block_id(seq_hash)?;
        let slot = inner.slots.get(block_id)?;
        let SlotState::Inactive {
            seq_hash: stored_hash,
            ..
        } = &slot.state
        else {
            return None;
        };
        if *stored_hash != seq_hash {
            return None;
        }
        let candidate = InactiveCandidate {
            seq_hash,
            block_id,
            generation: slot.generation,
            inactive_epoch: slot.inactive_epoch,
            features: inner.inactive.advice(seq_hash)?,
        };
        current_inactive_lineage(&inner, candidate).map(|source_blocks| (candidate, source_blocks))
    }

    /// Snapshot one complete inactive lineage without changing pool state.
    pub(crate) fn preflight_inactive_lineage(
        &self,
        candidate: InactiveCandidate,
    ) -> Option<Vec<(SequenceHash, BlockId)>> {
        let inner = self.inner.lock();
        current_inactive_lineage(&inner, candidate)
    }

    pub(crate) fn try_hold_inactive_lineage(
        self: &Arc<Self>,
        candidate: InactiveCandidate,
    ) -> Option<StoreInactiveLineageHold<T>> {
        self.try_hold_inactive_lineage_if_current(candidate, None)
    }

    /// Claim a lineage only when it still equals a preflighted source set.
    pub(crate) fn try_hold_prepared_inactive_lineage(
        self: &Arc<Self>,
        candidate: InactiveCandidate,
        expected_source_blocks: &[(SequenceHash, BlockId)],
    ) -> Option<StoreInactiveLineageHold<T>> {
        self.try_hold_inactive_lineage_if_current(candidate, Some(expected_source_blocks))
    }

    fn try_hold_inactive_lineage_if_current(
        self: &Arc<Self>,
        candidate: InactiveCandidate,
        expected_source_blocks: Option<&[(SequenceHash, BlockId)]>,
    ) -> Option<StoreInactiveLineageHold<T>> {
        let mut inner = self.inner.lock();
        let source_blocks = current_inactive_lineage(&inner, candidate)?;
        if expected_source_blocks.is_some_and(|expected| expected != source_blocks.as_slice()) {
            return None;
        }
        let extracted = inner
            .inactive
            .take_complete_lineage(candidate.seq_hash, candidate.block_id)?;
        debug_assert_eq!(
            extracted, source_blocks,
            "lineage preview changed while the store lock was held"
        );

        for &(seq_hash, block_id) in &source_blocks {
            let handle = take_exact_inactive_handle(&inner.slots[block_id], seq_hash, block_id);
            assert!(
                inner.held_by_hash.insert(seq_hash, block_id).is_none(),
                "inactive lineage hold overlapped an existing held hash"
            );
            inner.slots[block_id].state = SlotState::Held { seq_hash, handle };
        }
        self.metrics
            .dec_inactive_pool_size_by(source_blocks.len() as i64);
        self.metrics
            .inc_held_residency_by(source_blocks.len() as i64);
        drop(inner);

        Some(StoreInactiveLineageHold {
            store: Arc::clone(self),
            source_blocks,
        })
    }

    fn restore_lineage_hold(&self, source_blocks: Vec<(SequenceHash, BlockId)>) {
        let mut inner = self.inner.lock();
        let restored =
            restore_support_blocks(&mut inner, &source_blocks, self.default_reset_on_release);
        self.metrics
            .inc_inactive_pool_size_by(restored.restored_count as i64);
        self.metrics
            .dec_held_residency_by(source_blocks.len() as i64);
        self.metrics
            .inc_reset_pool_size_by(restored.discarded_handles.len() as i64);
        drop(inner);

        for handle in restored.discarded_handles {
            handle.mark_absent::<T>();
        }
    }

    fn commit_lineage_hold(
        &self,
        mut source_blocks: Vec<(SequenceHash, BlockId)>,
    ) -> Option<SequenceHash> {
        let (victim_hash, victim_id) = source_blocks
            .pop()
            .expect("an inactive lineage hold always contains its leaf");
        let mut inner = self.inner.lock();

        let restored =
            restore_support_blocks(&mut inner, &source_blocks, self.default_reset_on_release);
        // An eviction observer receives only a sequence hash. If a newer
        // copy still owns that hash, it must not receive a false removal.
        let victim_replaced = has_newer_registered_copy(&inner, victim_hash, victim_id);
        let victim_handle = take_exact_held_handle(&inner.slots[victim_id], victim_hash, victim_id);
        assert_eq!(
            inner.held_by_hash.remove(&victim_hash),
            Some(victim_id),
            "inactive lineage hold lost its victim ownership fence"
        );
        inner.slots[victim_id].state = SlotState::Reset;
        inner.reset_on_release[victim_id] = self.default_reset_on_release;
        inner.free.push_back(victim_id);

        self.metrics
            .inc_inactive_pool_size_by(restored.restored_count as i64);
        self.metrics
            .dec_held_residency_by((source_blocks.len() + 1) as i64);
        self.metrics
            .inc_reset_pool_size_by((restored.discarded_handles.len() + 1) as i64);
        self.metrics.inc_evictions(1);
        drop(inner);

        for handle in restored.discarded_handles {
            handle.mark_absent::<T>();
        }
        victim_handle.mark_absent::<T>();
        (!victim_replaced).then_some(victim_hash)
    }
}

/// Return the lineage that a hold can claim at this instant.
///
/// Callers hold the store mutex. This check has no mutation side effects.
fn current_inactive_lineage<T: BlockMetadata>(
    inner: &BlockStoreInner<T>,
    candidate: InactiveCandidate,
) -> Option<Vec<(SequenceHash, BlockId)>> {
    if inner.held_by_hash.contains_key(&candidate.seq_hash) {
        return None;
    }
    let slot = inner.slots.get(candidate.block_id)?;
    match &slot.state {
        SlotState::Inactive { seq_hash, .. }
            if *seq_hash == candidate.seq_hash
                && slot.generation == candidate.generation
                && slot.inactive_epoch == candidate.inactive_epoch => {}
        _ => return None,
    }
    let source_blocks = inner
        .inactive
        .complete_lineage(candidate.seq_hash, candidate.block_id)?;
    if source_blocks
        .iter()
        .any(|(seq_hash, _)| inner.held_by_hash.contains_key(seq_hash))
    {
        return None;
    }
    Some(source_blocks)
}

struct RestoreSupportResult {
    restored_count: usize,
    discarded_handles: Vec<BlockRegistrationHandle>,
}

fn restore_support_blocks<T: BlockMetadata>(
    inner: &mut BlockStoreInner<T>,
    source_blocks: &[(SequenceHash, BlockId)],
    default_reset_on_release: bool,
) -> RestoreSupportResult {
    let mut result = RestoreSupportResult {
        restored_count: 0,
        discarded_handles: Vec::new(),
    };
    for &(seq_hash, block_id) in source_blocks {
        let handle = take_exact_held_handle(&inner.slots[block_id], seq_hash, block_id);
        assert_eq!(
            inner.held_by_hash.remove(&seq_hash),
            Some(block_id),
            "inactive lineage hold lost its support ownership fence"
        );
        if has_newer_registered_copy(inner, seq_hash, block_id) {
            inner.slots[block_id].state = SlotState::Reset;
            inner.reset_on_release[block_id] = default_reset_on_release;
            inner.free.push_back(block_id);
            result.discarded_handles.push(handle);
        } else {
            inner.slots[block_id].state = SlotState::Inactive { seq_hash, handle };
            inner.inactive.insert(seq_hash, block_id);
            result.restored_count += 1;
        }
    }
    result
}

fn has_newer_registered_copy<T: BlockMetadata>(
    inner: &BlockStoreInner<T>,
    seq_hash: SequenceHash,
    held_block_id: BlockId,
) -> bool {
    inner
        .active_by_hash
        .get(&seq_hash)
        .is_some_and(|&block_id| block_id != held_block_id)
        || inner.inactive.has(seq_hash)
}

fn take_exact_inactive_handle<T: BlockMetadata>(
    slot: &BlockSlot<T>,
    expected_hash: SequenceHash,
    block_id: BlockId,
) -> BlockRegistrationHandle {
    match &slot.state {
        SlotState::Inactive { seq_hash, handle } if *seq_hash == expected_hash => handle.clone(),
        other => panic!(
            "inactive lineage index returned slot {block_id} for {expected_hash:?}, but slot is {other:?}"
        ),
    }
}

fn take_exact_held_handle<T: BlockMetadata>(
    slot: &BlockSlot<T>,
    expected_hash: SequenceHash,
    block_id: BlockId,
) -> BlockRegistrationHandle {
    match &slot.state {
        SlotState::Held { seq_hash, handle } if *seq_hash == expected_hash => handle.clone(),
        other => panic!(
            "inactive lineage hold lost slot {block_id} for {expected_hash:?}; slot is {other:?}"
        ),
    }
}
