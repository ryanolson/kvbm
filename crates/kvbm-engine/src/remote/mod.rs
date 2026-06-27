// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Piece (c): higher-level orchestrations over remote-G2 operations.
//! Today: remote search-and-pull, plus conditional disagg (remote/cd).

pub mod search;

// The decode-side CD decision core + search-time commit are wired into the
// engine's search path, and the connector's CD wiring now assembles the
// production transports (via `RemoteOps::with_disagg_transports`). The allow
// stays because parts of the USAA fan-out remain unbuilt: the per-request
// USAA bookkeeping (remote slots, tier promotion, completion flags), the
// availability-ledger delivery, and the remote-compute-token metric are
// genuinely unused until that work lands.
#[allow(dead_code)]
pub(crate) mod cd;
