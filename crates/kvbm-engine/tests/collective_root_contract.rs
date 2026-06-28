// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "collectives")]

use std::ops::Range;
use std::sync::Mutex;

use anyhow::Result;
use kvbm_common::LogicalLayoutHandle;
use kvbm_engine::BlockId;
use kvbm_engine::collectives::CollectiveOps;
use kvbm_physical::transfer::TransferCompleteNotification;

#[derive(Default)]
struct RecordingCollective {
    roots: Mutex<Vec<usize>>,
}

impl CollectiveOps for RecordingCollective {
    fn broadcast(
        &self,
        root_rank: usize,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_block_ids: &[BlockId],
        _dst_block_ids: &[BlockId],
        _layer_range: Option<Range<usize>>,
    ) -> Result<TransferCompleteNotification> {
        self.roots
            .lock()
            .expect("root log poisoned")
            .push(root_rank);
        Ok(TransferCompleteNotification::completed())
    }

    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        2
    }
}

#[test]
fn broadcast_root_is_selected_per_operation() {
    let collective = RecordingCollective::default();
    collective
        .broadcast(
            1,
            LogicalLayoutHandle::G1,
            LogicalLayoutHandle::G1,
            &[7],
            &[11],
            None,
        )
        .expect("broadcast should be accepted");

    assert_eq!(*collective.roots.lock().expect("root log poisoned"), [1]);
}
