// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # KVBM Hub
//!
//! KVBM coordination service built on `velo-hub`. `velo-hub` owns peer
//! discovery, registration, heartbeats and the HTTP listeners. This crate adds
//! the KVBM extensions (P2P, conditional disagg, indexer, prefill router and
//! connector control) and the aggregate KVBM config at
//! [`protocol::paths::HUB_CONFIG`].
//!
//! # Topology
//!
//! A hub server binds two listeners:
//!
//! - **Discovery port** (`1337` default) — read-only peer lookups.
//! - **Control port** (`8337` default) — registration, heartbeat, health.
//!
//! Each extension serves its routes under `/v1/ext/{name}`. The server is
//! also a velo node, so it can message registered clients.
//!
//! # Client usage
//!
//! ```no_run
//! # async fn demo() -> anyhow::Result<()> {
//! use std::sync::Arc;
//!
//! let hub = kvbm_hub::create_client_builder()
//!     .host("name-or-ip")
//!     .discovery_port(1337)
//!     .control_port(8337)
//!     .build()?;
//!
//! let velo = velo::Velo::builder()
//!     .discovery(Arc::new(hub.discovery()))
//!     .build()
//!     .await?;
//!
//! hub.register(&velo, Vec::new(), None).await?;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod config;
pub mod features;
pub mod handlers;
pub mod protocol;
/// `kvbmctl` config rendering. Gated behind the `kvbmctl` feature because it
/// depends on `kvbm-config`, which transitively pulls CUDA (cudarc) — kept out
/// of the default CPU-only hub build.
#[cfg(feature = "kvbmctl")]
pub mod render;
pub mod server;
#[cfg(any(test, feature = "test-support"))]
pub mod test_hub;
pub mod web;

pub use client::{HubClient, HubClientBuilder};
pub use config::{HubConfig, IndexerConfig};
#[cfg(feature = "kvbmctl")]
pub use features::cli::{FeatureCli, feature_clis, hub_arg};
pub use features::control_plane::ControlPlaneManager;
pub use features::disagg::{
    ConditionalDisaggClient, ConditionalDisaggInstancesResponse, ConditionalDisaggManager,
};
pub use features::indexer::{
    ApplyOutcome, BindError, BundleAdvertisementRecord, BundleInvalidateRequest,
    BundleInvalidationRecord, BundlePublishRequest, BundleQueryHit, BundleQueryMissReason,
    BundleQueryOutcome, BundleQueryRequest, EventPlane, FindBlocksHit, IndexerConfigResponse,
    IndexerLookupClient, IndexerManager, InstanceBinding, InstancesResponse, ManifestIndexes,
    QueryRequest, QueryResponse, nats_server_from_env,
};
#[cfg(feature = "kvbmctl")]
pub use features::p2p::cli::{p2p_command, run_p2p};
pub use features::p2p::{P2pClient, P2pManager};
pub use features::prefill_router::{
    BreakerConfig, CALIBRATE_HANDLER, CalibrationDefaults, CalibrationRequest, CalibrationResponse,
    CalibrationResults, CalibrationSnapshot, CircuitBreaker, DecodeSetProvider, DispatchOutcome,
    HttpExecutionBackend, PREFILL_DISPATCH_HANDLER, PerformanceModel, PrefillDispatchRequest,
    PrefillDispatchResponse, PrefillExecutionBackend, PrefillRequestDispatcher, PrefillRouter,
    PrefillRouterManager, RawCalibrationPayload, RawTrace, RecordingDispatcher,
    ResolvedCalibrationRequest, ScatterData, Selector, SelectorConfig, TierBroadcaster,
    VeloExecutionBackend, analyze_calibration,
};
pub use features::{FeatureConfigRequirements, KvbmFeature};
pub use handlers::{TIER_SIGNAL_HANDLER, TierSignal, TierSignalAck};
pub use kvbm_common::BlockLayoutMode;
pub use kvbm_logical::events::CreateKind;
pub use protocol::{
    CD_PREFILL_QUEUE, ConditionalDisaggConfig, ConditionalDisaggRole, DEFAULT_CONTROL_PORT,
    DEFAULT_DISCOVERY_PORT, Feature, FeatureDescriptor, FeatureKey, HubConfigResponse,
    IndexerFeatureConfig, P2pConfig, PrefillBackendAdvertisement, PrefillRequest,
    PrefillRouterConfig, PrimaryConfig, RuntimeConfigSummary, VllmHttpEndpoint,
};
pub use server::{HubServer, HubServerBuilder};

/// Shorthand for [`HubClientBuilder::new`].
pub fn create_client_builder() -> HubClientBuilder {
    HubClientBuilder::new()
}

/// Shorthand for [`HubServerBuilder::new`].
pub fn create_server_builder() -> HubServerBuilder {
    HubServerBuilder::new()
}
