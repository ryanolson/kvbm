// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-aware decisions for a G1 block's lower-tier lifecycle.

use serde::{Deserialize, Serialize};

/// Requested lifecycle for one G1 block.
///
/// This is a policy result, not a transfer command. Executors may satisfy an
/// already-resident `Mirror` or `Move` without copying. A `Move` must complete
/// lower-tier ownership before G1 is released; it never authorizes a blocking
/// copy in the allocator's eviction path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionAction {
    /// Ensure a lower-tier copy while retaining the G1 copy.
    Mirror,
    /// Ensure lower-tier ownership, then release the G1 copy.
    Move,
    /// Release G1 without creating or retaining a lower-tier copy.
    Drop,
}

impl RetentionAction {
    /// Classify a block after model policy decides whether it remains useful.
    pub const fn classify(retain: bool, release_g1: bool) -> Self {
        match (retain, release_g1) {
            (false, _) => Self::Drop,
            (true, false) => Self::Mirror,
            (true, true) => Self::Move,
        }
    }

    /// Whether this action requires a valid lower-tier copy.
    pub const fn requires_lower_tier(self) -> bool {
        matches!(self, Self::Mirror | Self::Move)
    }

    /// Whether G1 may be released after the action is satisfied.
    pub const fn releases_g1(self) -> bool {
        matches!(self, Self::Move | Self::Drop)
    }
}

#[cfg(test)]
mod tests {
    use super::RetentionAction;

    #[test]
    fn model_discard_always_drops() {
        assert_eq!(
            RetentionAction::classify(false, false),
            RetentionAction::Drop
        );
        assert_eq!(
            RetentionAction::classify(false, true),
            RetentionAction::Drop
        );
    }

    #[test]
    fn retained_blocks_mirror_before_release_and_move_when_released() {
        assert_eq!(
            RetentionAction::classify(true, false),
            RetentionAction::Mirror
        );
        assert_eq!(RetentionAction::classify(true, true), RetentionAction::Move);
        assert!(RetentionAction::Mirror.requires_lower_tier());
        assert!(!RetentionAction::Mirror.releases_g1());
        assert!(RetentionAction::Move.requires_lower_tier());
        assert!(RetentionAction::Move.releases_g1());
        assert!(!RetentionAction::Drop.requires_lower_tier());
        assert!(RetentionAction::Drop.releases_g1());
    }
}
