// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Outcome of finalizing a request on the leader.

/// Outcome of marking a request slot finished on the leader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishedStatus {
    /// The slot is in inactive state, so the request is finished and can be deleted.
    Finished,

    /// The slot has an active transaction; we must await completion.
    Pending,

    /// The request is not tracked by the leader. There is no slot for the request.
    UntrackedRequest,
}
