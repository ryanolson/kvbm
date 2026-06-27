// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Common types shared between the scheduler and connector modules.
//!
//! This module contains types that are used by both the scheduler (G1 block management)
//! and the connector (G2+ offloading), allowing them to communicate without tight coupling.

mod consolidator;
mod finished;
mod metadata;
mod output;
mod request;

pub use consolidator::ConsolidatorEndpoints;
pub use finished::FinishedStatus;
pub use metadata::{IntraPassLoad, IntraPassStore, KvConnectorMetadata};
pub use output::{CachedRequestData, NewRequestData, SchedulerOutput};
pub use request::{Request, RequestMetadata};
