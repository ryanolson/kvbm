// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine shell for the P1 skeleton.
//!
//! The only engine the skeleton ships is [`NoopBlockEngine`] (caches nothing,
//! drains immediately). [`noop_leader_engine`] builds one for the leader's
//! binding-facing constructors that have no injected engine. `NoopBlockEngine`
//! impls the connector [`LeaderEngine`] seam, so it is handed to the leader as an
//! `Arc<dyn LeaderEngine>`.

use std::sync::Arc;

use kvbm_protocols::connector::{LeaderEngine, NoopBlockEngine};

/// Build a standalone [`NoopBlockEngine`] as an `Arc<dyn LeaderEngine>`.
pub fn noop_leader_engine() -> Arc<dyn LeaderEngine> {
    NoopBlockEngine::new() as Arc<dyn LeaderEngine>
}
