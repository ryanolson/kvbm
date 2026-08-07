// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared KVBM wire protocols.
//!
//! This crate intentionally contains only serializable protocol data (and,
//! behind the `client` feature, the thin velo client that speaks it). It is
//! shared by the connector, engine, and hub without making any one of those
//! crates depend on another.
//!
//! - [`connector`] — the connector ↔ engine seam: the handle-first
//!   `LeaderEngine` (returning the RAII search/onboard/offload handles), the
//!   `EngineWorkerSink` / `WorkerEngineDriver` engine ↔ worker delegates, the
//!   `FinishedStatus` request-finish / eviction gate, the `LoadOutcome` /
//!   `SaveOutcome` terminal outcomes, and `NoopBlockEngine` (a caches-nothing
//!   stand-in).
//! - [`disagg`] — disaggregation session/request types.
//! - [`control`] — the public leader control plane: the `ControlReply`
//!   envelope, per-module request/response types, the `ModuleId` registry,
//!   and (with `--features client`) the `LeaderControlClient`.
//! - [`tier_protocol`] — the versioned tier-placement event stream (R7b): the
//!   `TierPlacementBatchV1` delta envelope, the `TierPlacementSnapshotV1`
//!   recovery payload, and the publisher-side sequencer. Lives here rather than
//!   in `kvbm-logical/src/events/` because it is built from this crate's
//!   identity types (`CacheManifestId`, `RegistrationEpoch`,
//!   `BundleResourceLineage`); see that module's docs.

pub mod cache_manifest;
pub mod connector;
pub mod control;
pub mod disagg;
pub mod tier_protocol;
