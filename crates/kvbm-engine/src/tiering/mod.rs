// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Piece (a): G1<->G2 local tiering — the offload pipeline plus the
//! seam-facing connector engine (the LeaderEngine / WorkerEngineDriver impls).

#[doc = include_str!("../../docs/offload.md")]
pub mod offload;

/// Engine-side support for the KVBM connector (search reconcile core).
pub(crate) mod engine;
