// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote search-and-pull orchestration.
//!
//! Pipeline: [`discovery`] resolves which remote instances hold uncached blocks
//! → [`plan`] decimates the query and computes pin targets → [`composer`]
//! overlaps G3→G2 staging with the discovery RPC then issues the remote pull.
//!
//! The G4/async search machinery (`g4`) is **parked/unwired** — nothing in the
//! engine calls it yet. It exists to seed the upcoming async remote-search
//! refactor, which will re-enable G4 lookups.

pub(crate) mod composer;
pub mod discovery;
pub(crate) mod plan;

/// Async search machinery (parked/unwired — seeds the upcoming G4 refactor).
#[allow(dead_code)]
mod g4;

pub use g4::{AsyncSearch, G4SearchState};
