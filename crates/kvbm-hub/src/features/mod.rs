// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The kvbm features of the hub.
//!
//! Each feature is a [`velo_hub::Extension`]: it owns its state, mounts its
//! routes under `/v1/ext/{NAME}`, and sees every registration and removal.
//! [`KvbmFeature`] adds the kvbm config each feature advertises in
//! `GET /v1/ext/kvbm/config`.

pub use crate::protocol::FeatureConfigRequirements;

/// Client-side per-feature `kvbmctl` CLI surface. Gated behind the `kvbmctl`
/// feature (it pulls in `clap` usage + the client trait); the default hub build
/// excludes it.
#[cfg(feature = "kvbmctl")]
pub mod cli;
pub mod control_plane;
pub mod disagg;
pub(crate) mod http;
pub mod indexer;
pub mod p2p;
pub mod prefill_router;

/// The kvbm config a feature advertises next to its extension.
pub trait KvbmFeature: velo_hub::Extension {
    /// Extensions a connector should also enable when it enables this one.
    /// A soft co-enable for `kvbmctl render`, not a registration requirement.
    fn render_implies(&self) -> &'static [&'static str] {
        &[]
    }

    /// Primary config fields every registrant of this feature must match.
    fn config_requirements(&self) -> FeatureConfigRequirements {
        FeatureConfigRequirements::default()
    }

    /// Whether a registrant of this feature must send its runtime summary.
    fn requires_runtime_summary(&self) -> bool {
        false
    }

    /// The block size this feature decides, if any.
    fn authoritative_block_size(&self) -> Option<usize> {
        None
    }
}

/// A feature check that failed.
#[derive(Debug, thiserror::Error)]
pub enum FeatureError {
    /// The feature payload is missing or malformed.
    #[error("feature config invalid: {0}")]
    InvalidConfig(String),
    /// Any other failure.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
