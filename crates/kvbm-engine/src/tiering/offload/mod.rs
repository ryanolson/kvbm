// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offload Engine for asynchronous block transfers between storage tiers.
//!
//! The offload engine provides a policy-based, cancellable pipeline for moving
//! blocks from higher-performance tiers (G1/G2) to lower-cost tiers (G3/G4).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        OffloadEngine                            │
//! │                                                                 │
//! │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐    │
//! │  │G1→G2 Pipeline │────│ G2→G3 Pipeline│    │ G2→G4 Pipeline│    │
//! │  └───────────────┘    └───────────────┘    └───────────────┘    │
//! │         │                     │                     │           │
//! │         └─────────auto_chain──┘                     │           │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! Pipeline stages:
//! ┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐
//! │   Policy    │───▶│ Precondition │───▶│     Batch      │───▶│    Transfer      │
//! │  Evaluator  │    │   Awaiter    │    │   Collector    │    │    Executor      │
//! └─────────────┘    └──────────────┘    └────────────────┘    └──────────────────┘
//!       │                   │                     │                      │
//!       ▼                   ▼                     ▼                      ▼
//!   cancel check       token select          cancel sweep        claim and drain
//! ```
//!
//! # Features
//!
//! - **Policy-based filtering**: Blocks pass through configurable policies
//!   (presence checks, LFU thresholds) before transfer
//! - **Batched transfers**: Blocks are accumulated into batches for efficient
//!   bulk transfers
//! - **Cancellation**: Precommit cancellation with ownership confirmation.
//!   Ambiguous physical completion keeps confirmation pending.
//! - **Pipeline chaining**: G1→G2 completions can automatically feed G2→G3
//!
//! See also: [Developer Guide](../../docs/offload-developer.md) for implementation
//! details and extension rules.
//!
//! # Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use kvbm_engine::{G2, G3};
//! use kvbm_engine::offload::{
//!     OffloadEngine, PipelineBuilder, PresenceAndLFUFilter,
//! };
//!
//! let engine = OffloadEngine::builder(leader.clone())
//!     .with_g3_manager(g3_manager.clone())
//!     .with_g2_to_g3_pipeline(
//!         PipelineBuilder::<G2, G3>::new()
//!             .policy(Arc::new(PresenceAndLFUFilter::with_default_threshold(registry.clone())))
//!             .batch_size(64)
//!             .build()
//!     )
//!     .build()?;
//!
//! let handle = engine.enqueue_g2_to_g3(blocks)?;
//! let mut completion = handle.clone();
//!
//! tokio::select! {
//!     result = completion.wait() => {
//!         println!("Completed: {:?}", result?.completed_blocks);
//!     }
//!     _ = shutdown_signal => {
//!         handle.cancel().wait().await;
//!         println!("Cancellation settled");
//!     }
//! }
//! ```
//!
//! See also: [Developer Guide](../../docs/offload-developer.md)

/// Helper macro to create an NVTX range when the nvtx feature is enabled.
/// The range automatically ends when the returned guard is dropped.
macro_rules! nvtx_range {
    ($name:expr) => {{
        #[cfg(feature = "nvtx")]
        let _range = nvtx::range!($name);
        #[cfg(not(feature = "nvtx"))]
        let _range = ();
        _range
    }};
}

mod batch;
mod cancel;
mod chain_router;
mod container;
mod destination;
mod engine;
mod handle;
mod pending;
mod pipeline;
mod policy;
mod queue;
mod remote_g4;
mod source;

#[cfg(test)]
mod cancel_tests;

// Re-export public API.
pub use cancel::CancelConfirmation;
pub use engine::{OffloadEngine, OffloadEngineBuilder};
pub use handle::{TransferHandle, TransferId, TransferResult, TransferStatus};
pub use pending::PendingTracker;
pub use pipeline::{
    ObjectPipelineBuilder, ObjectPipelineConfig, PipelineBuilder, PipelineConfig, RegisterObserver,
};
pub use policy::{
    AllOfPolicy, AnyOfPolicy, BoxFuture, EvalContext, ObjectLockPresenceFilter,
    ObjectPresenceFilter, OffloadPolicy, PassAllPolicy, PolicyBatchFuture, PolicyFuture,
    PresenceAndLFUFilter, PresenceChecker, PresenceFilter, S3PresenceChecker, async_batch_result,
    async_result, create_policy_from_config, sync_batch_result, sync_result,
};
pub use source::{ExternalBlock, SourceBlock, SourceBlocks};
