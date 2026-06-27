// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `kvbm_audit` tracing target for kvbm-engine.
//!
//! Emits session and worker-side events on the shared
//! `target: "kvbm_audit"` sink the `cd-trace.py` parser reads. Kept as
//! an in-crate macro (rather than a shared dependency) so kvbm-engine
//! takes no extra crate dependency just to emit audit lines.

#[macro_export]
macro_rules! engine_audit {
    ($event:expr, $($field:tt)*) => {
        tracing::info!(
            target: "kvbm_audit",
            event = $event,
            $($field)*
        )
    };
}
