// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill router feature.
//!
//! Owns selection-and-execution of prefill requests across a fleet of
//! registered prefill workers. A decode worker sends each request to the
//! hub as a velo unary; the [`PrefillRouter`] sends it on to a worker
//! advertised through [`PrefillRouterConfig`] at registration, or rejects
//! it at once when no worker is free.
//!
//! Selection is load-aware (lowest `load_net_new`, then lowest `inflight`,
//! then lowest index), capped by a per-worker concurrency limit.
//!
//! Execution sits behind [`PrefillExecutionBackend`] so a worker can be
//! reached over HTTP or a velo unary. [`worker`] is the velo worker side.

pub mod breaker;
pub mod calibration;
pub mod execution;
pub mod manager;
pub mod protocol;
pub mod router;
pub mod selection;
pub mod tier_push;
pub mod worker;

pub use breaker::{BreakerConfig, CircuitBreaker};
pub use calibration::{
    CALIBRATE_HANDLER, CalibrationDefaults, CalibrationRequest, CalibrationResponse,
    CalibrationResults, CalibrationSnapshot, PerformanceModel, RawCalibrationPayload, RawTrace,
    ResolvedCalibrationRequest, ScatterData, analyze as analyze_calibration,
};
pub use execution::{
    DispatchOutcome, HttpExecutionBackend, PrefillExecutionBackend, VeloExecutionBackend,
};
pub use manager::PrefillRouterManager;
pub use protocol::{
    PREFILL_DISPATCH_HANDLER, PREFILL_DONE_HANDLER, PREFILL_SUBMIT_HANDLER,
    PrefillBackendAdvertisement, PrefillDispatchRequest, PrefillDispatchResponse, PrefillDone,
    PrefillRouterConfig, PrefillTargetSummary, TargetsResponse, VllmHttpEndpoint,
};
pub use router::PrefillRouter;
pub use selection::{Selector, SelectorConfig};
pub use tier_push::{DecodeSetProvider, TierBroadcaster};
pub use worker::{PrefillAssignment, PrefillDoneNotice, register_prefill_worker};
