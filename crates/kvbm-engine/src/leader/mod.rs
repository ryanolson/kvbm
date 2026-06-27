// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod accessor;
mod blocks;
pub mod consolidator;
pub mod control;
mod describe_map;
mod instance;
pub mod layout_compat;
mod onboarding;
mod staging;
mod state;
mod types;

// Migration shims: old paths preserved until the legacy delete (P-G).
pub use crate::p2p::dispatch;
pub use crate::p2p::parallelism;
pub use crate::p2p::service as velo;
pub(crate) use crate::p2p::transport;
pub use crate::remote::search;
pub(crate) use crate::remote::search::composer;
pub use crate::remote::search::discovery;

pub use accessor::{BlockAccessor, PolicyContext, TieredBlock};
pub use blocks::BlockHolder;
pub use consolidator::ConsolidatorParams;
pub use control::{ControlModule, ControlPlane, ControlPlaneBuilder};
pub use discovery::{RemoteBlockDiscovery, RemoteCandidates, RemoteDiscoveryHandle};
pub use instance::InstanceLeader;
pub use kvbm_consolidator::{ConsolidatorHandle, EventSource};
pub use onboarding::*;
pub use staging::{StagingResult, stage_g3_to_g2};
pub use state::route_local_to_remote;
pub use transport::MetadataTransport;
pub use types::*;
pub use velo::VeloLeaderService;

use anyhow::Result;

use crate::SequenceHash;

/// Leader trait for distributed block onboarding operations.
pub trait Leader: Send + Sync {
    /// Find matching blocks with default options.
    fn find_matches(&self, sequence_hashes: &[SequenceHash]) -> Result<FindMatchesResult> {
        self.find_matches_with_options(sequence_hashes, FindMatchesOptions::default())
    }

    /// Find matching blocks with custom options.
    fn find_matches_with_options(
        &self,
        sequence_hashes: &[SequenceHash],
        options: FindMatchesOptions,
    ) -> Result<FindMatchesResult>;
}
