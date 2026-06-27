// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discarding action sinks for the standalone P1 skeleton.
//!
//! A real deployment wires the worker's `WorkerEngineDriver` /
//! [`WorkerLeaderActions`] to a live engine and leader. The skeleton's
//! binding-facing worker constructor has no such peer, so it backs the unused
//! side with one of these no-op sinks.

use kvbm_protocols::connector::{FenceToken, WorkerEngineDriver};

use super::worker::WorkerLeaderActions;

/// Drops every worker→engine forward-pass boundary. Backs the binding worker
/// constructor, which has no real engine in the skeleton. `await_fence` is a
/// no-op: the connector worker waits on its own `WorkerCompletionState`, not on
/// the driver (see `worker::state`), so this method is never called by the
/// worker.
pub struct NullEngineSink;

impl WorkerEngineDriver for NullEngineSink {
    fn begin_forward_pass(&self, _iteration: usize) {}
    fn finish_forward_pass(&self, _iteration: usize) {}
    fn await_fence(&self, _token: FenceToken) {}
    fn shutdown(&self) {}
}

/// Drops the worker→leader teardown signal. Backs the binding worker
/// constructor, which has no leader handle in the skeleton.
pub struct NullLeaderSink;

impl WorkerLeaderActions for NullLeaderSink {
    fn shutdown(&self) {}
}
