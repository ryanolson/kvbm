// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire contract for the leader→worker velo control RPCs.
//!
//! The handler keys + message types here are the single shared contract spoken
//! by the relocated [`super::client::ConnectorWorkerClient`] and the legacy
//! worker velo service. They are deliberately distinct from the connector worker
//! completion path in [`super::velo`], which uses its own
//! `kvbm.connector.worker.*` keys and message types.

use serde::{Deserialize, Serialize};

use crate::BlockId;

/// Leader-driven worker initialization (deferred NIXL registration + layouts).
pub(crate) const INITIALIZE_HANDLER: &str = "kvbm.connector.worker.initialize";
/// Serve the registration-time layout config back to the leader.
pub(crate) const GET_LAYOUT_CONFIG_HANDLER: &str = "kvbm.connector.worker.get_layout_config";
/// Onboarding-complete notification (leader → worker).
pub(crate) const ONBOARD_COMPLETE_HANDLER: &str = "kvbm.connector.worker.onboard_complete";
/// Offloading-complete notification (leader → worker).
pub(crate) const OFFLOAD_COMPLETE_HANDLER: &str = "kvbm.connector.worker.offload_complete";
/// Onboarding-failed notification (leader → worker), naming the failed blocks.
pub(crate) const FAILED_ONBOARD_HANDLER: &str = "kvbm.connector.worker.failed_onboard";

/// Message sent by leader to workers when onboarding completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnboardCompleteMessage {
    pub request_id: String,
}

/// Message sent by leader to workers when offloading completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffloadCompleteMessage {
    pub request_id: String,
}

/// Message sent by leader to workers when onboarding fails for specific blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedOnboardMessage {
    pub request_id: String,
    pub block_ids: Vec<BlockId>,
}
