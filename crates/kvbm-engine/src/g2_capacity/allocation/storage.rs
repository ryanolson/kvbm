// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared mutable and staged allocation storage.

use std::future::Future;
use std::sync::Arc;

use anyhow::{Result as AnyResult, ensure};
use kvbm_common::SequenceHash;
use kvbm_logical::blocks::{CompleteBlock, MutableBlock};

use crate::g2_capacity::{G2AllocationKind, G2LeaseGuard};
use crate::{BlockId, G2};

pub(super) struct MutableAllocation {
    kind: G2AllocationKind,
    blocks: Vec<MutableBlock<G2>>,
    guard: Arc<dyn G2LeaseGuard>,
}

pub(super) struct StagedAllocation {
    kind: G2AllocationKind,
    hashes: Vec<SequenceHash>,
    blocks: Vec<CompleteBlock<G2>>,
    guards: Vec<Arc<dyn G2LeaseGuard>>,
}

impl MutableAllocation {
    pub(super) fn new(
        kind: G2AllocationKind,
        blocks: Vec<MutableBlock<G2>>,
        guard: Arc<dyn G2LeaseGuard>,
    ) -> Self {
        Self {
            kind,
            blocks,
            guard,
        }
    }

    pub(super) const fn kind(&self) -> G2AllocationKind {
        self.kind
    }

    pub(super) fn len(&self) -> usize {
        self.blocks.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub(super) fn block_ids(&self) -> Vec<BlockId> {
        self.blocks.iter().map(MutableBlock::block_id).collect()
    }

    pub(super) fn blocks(&self) -> &[MutableBlock<G2>] {
        &self.blocks
    }

    pub(super) async fn transfer_with<F, Fut, E>(self, transfer: F) -> Result<Self, E>
    where
        F: FnOnce(Vec<MutableBlock<G2>>) -> Fut,
        Fut: Future<Output = Result<Vec<MutableBlock<G2>>, E>>,
    {
        let Self {
            kind,
            blocks,
            guard,
        } = self;
        transfer(blocks).await.map(|blocks| Self {
            kind,
            blocks,
            guard,
        })
    }

    pub(super) fn stage_all(
        self,
        hashes: &[SequenceHash],
        block_size: usize,
    ) -> AnyResult<StagedAllocation> {
        let selected = vec![true; self.len()];
        self.stage_selected(hashes, block_size, selected)
    }

    pub(super) fn stage_selected<I>(
        self,
        hashes: &[SequenceHash],
        block_size: usize,
        selected: I,
    ) -> AnyResult<StagedAllocation>
    where
        I: IntoIterator<Item = bool>,
    {
        let Self {
            kind,
            blocks,
            guard,
        } = self;
        let selected = selected.into_iter().collect::<Vec<_>>();
        ensure!(
            blocks.len() == hashes.len(),
            "G2 allocation has {} blocks for {} hashes",
            blocks.len(),
            hashes.len()
        );
        ensure!(
            blocks.len() == selected.len(),
            "G2 allocation has {} blocks for {} selection decisions",
            blocks.len(),
            selected.len()
        );

        let mut staged_hashes = Vec::with_capacity(blocks.len());
        let mut staged_blocks = Vec::with_capacity(blocks.len());
        let mut staged_guards = Vec::with_capacity(blocks.len());
        for ((block, hash), keep) in blocks.into_iter().zip(hashes.iter()).zip(selected) {
            if keep {
                staged_hashes.push(*hash);
                staged_blocks.push(
                    block
                        .stage(*hash, block_size)
                        .map_err(|error| anyhow::anyhow!("stage G2 allocation block: {error:#}"))?,
                );
                staged_guards.push(Arc::clone(&guard));
            }
        }
        drop(guard);
        Ok(StagedAllocation {
            kind,
            hashes: staged_hashes,
            blocks: staged_blocks,
            guards: staged_guards,
        })
    }
}

impl StagedAllocation {
    pub(super) fn set_evict_on_reset(&mut self, value: bool) {
        for block in &mut self.blocks {
            block.set_evict_on_reset(value);
        }
    }

    pub(super) const fn kind(&self) -> G2AllocationKind {
        self.kind
    }

    pub(super) fn hashes(&self) -> &[SequenceHash] {
        &self.hashes
    }

    pub(super) fn len(&self) -> usize {
        self.blocks.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub(super) fn block_ids(&self) -> Vec<BlockId> {
        self.blocks.iter().map(CompleteBlock::block_id).collect()
    }

    pub(super) fn register_with<Output>(
        self,
        register: impl FnOnce(Vec<CompleteBlock<G2>>) -> Output,
    ) -> Output {
        let Self { blocks, guards, .. } = self;
        let result = register(blocks);
        drop(guards);
        result
    }

    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the internal compatibility lease tests retain these guards"
        )
    )]
    pub(super) fn register_with_retained_guards<Output>(
        self,
        register: impl FnOnce(Vec<CompleteBlock<G2>>) -> Output,
    ) -> (Output, Vec<Arc<dyn G2LeaseGuard>>) {
        let Self { blocks, guards, .. } = self;
        let result = register(blocks);
        (result, guards)
    }

    pub(super) fn direct(
        kind: G2AllocationKind,
        hashes: Vec<SequenceHash>,
        blocks: Vec<CompleteBlock<G2>>,
    ) -> AnyResult<Self> {
        ensure!(
            hashes.len() == blocks.len(),
            "G2 staged allocation has {} blocks for {} hashes",
            blocks.len(),
            hashes.len()
        );
        let guard: Arc<dyn G2LeaseGuard> = Arc::new(());
        Ok(Self {
            kind,
            guards: vec![guard; blocks.len()],
            hashes,
            blocks,
        })
    }

    pub(super) fn into_entries(
        self,
    ) -> Vec<(SequenceHash, CompleteBlock<G2>, Arc<dyn G2LeaseGuard>)> {
        let Self {
            hashes,
            blocks,
            guards,
            ..
        } = self;
        hashes
            .into_iter()
            .zip(blocks)
            .zip(guards)
            .map(|((hash, block), guard)| (hash, block, guard))
            .collect()
    }

    pub(super) fn from_entries(
        kind: G2AllocationKind,
        entries: Vec<(SequenceHash, CompleteBlock<G2>, Arc<dyn G2LeaseGuard>)>,
    ) -> Self {
        let mut hashes = Vec::with_capacity(entries.len());
        let mut blocks = Vec::with_capacity(entries.len());
        let mut guards = Vec::with_capacity(entries.len());
        for (hash, block, guard) in entries {
            hashes.push(hash);
            blocks.push(block);
            guards.push(guard);
        }
        Self {
            kind,
            hashes,
            blocks,
            guards,
        }
    }
}
