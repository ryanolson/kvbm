// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned logical source capability for an exact physical transaction.

use kvbm_logical::InactiveLineageHold;
use kvbm_logical::blocks::BlockMetadata;
use kvbm_protocols::connector::OffloadMode;

use super::state::{PolicyG1G2SourceSettlement, PolicyPhysicalTerminal};

/// One real inactive-lineage hold and its source disposition.
///
/// The exact route derives physical source IDs from this value. Reservation
/// hashes constrain this source but do not establish source authority.
#[must_use = "submit this owned source through its paired bound route"]
pub(super) struct PolicyG1G2Source<T: BlockMetadata> {
    hold: Option<InactiveLineageHold<T>>,
    mode: OffloadMode,
}

impl<T: BlockMetadata> PolicyG1G2Source<T> {
    pub(super) fn new(hold: InactiveLineageHold<T>, mode: OffloadMode) -> Self {
        Self {
            hold: Some(hold),
            mode,
        }
    }

    pub(super) fn source_blocks(&self) -> &[(kvbm_common::SequenceHash, kvbm_common::BlockId)] {
        self.hold
            .as_ref()
            .expect("the physical transaction owns its source hold")
            .source_blocks()
    }

    pub(super) fn settle(mut self, terminal: PolicyPhysicalTerminal) -> PolicyG1G2SourceSettlement {
        let hold = self
            .hold
            .take()
            .expect("the proven source hold is settled once");
        if self.mode == OffloadMode::Move
            && terminal == PolicyPhysicalTerminal::DestinationCommitted
        {
            let notification = hold.commit_victim_release_silent();
            let notification_panicked =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| notification.notify()))
                    .is_err();
            PolicyG1G2SourceSettlement::Committed {
                notification_panicked,
            }
        } else {
            drop(hold);
            PolicyG1G2SourceSettlement::Restored
        }
    }
}

impl<T: BlockMetadata> Drop for PolicyG1G2Source<T> {
    fn drop(&mut self) {
        drop(self.hold.take());
    }
}
