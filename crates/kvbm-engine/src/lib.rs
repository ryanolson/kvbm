// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../docs/architecture.md")]

pub use kvbm_common::{BlockId, LogicalLayoutHandle, SequenceHash};
pub use velo::{InstanceId, PeerInfo, WorkerAddress};

/// GPU/device tier -- HBM KV cache. Fastest access, smallest capacity.
/// Blocks here are actively used by attention kernels.
#[derive(Clone, Copy, Debug)]
pub struct G1;
/// CPU/host tier -- pinned DRAM cache. Microsecond-latency staging area
/// for RDMA transfers and G3/G4 promotion.
#[derive(Clone, Copy, Debug)]
pub struct G2;
/// Disk tier -- NVMe/SSD cache. Millisecond-latency persistent storage
/// for warm blocks.
#[derive(Clone, Copy, Debug)]
pub struct G3;
/// Object store tier -- S3/MinIO. Highest latency but unlimited capacity
/// for cold/archival blocks.
#[derive(Clone, Copy, Debug)]
pub struct G4;

pub mod audit;
#[cfg(feature = "collectives")]
pub mod collectives;

/// The connector's connector engine construction entry points: the leader-side
/// factory plus the worker-side [`WorkerEngine`] and its pass-plan types.
/// `LocalConnectorEngine` itself stays internal.
pub use tiering::engine::{
    ConnectorEngineConfig, PassOffload, PassOnboard, RemoteOps, WorkerEngine, WorkerPassPlan,
    build_local_connector_engine,
};

/// Conditional-disagg transport seam consumed by the connector's CD wiring:
/// the breaker [`cd::TierCell`] the velo tier-signal handler writes, the
/// decode-side [`cd::PrefillPlane`] dispatch seam (+ its [`cd::PrefillDispatch`]
/// payload), and the engine-local [`cd::DisaggConfig`] translated from the
/// connector's parsed config. Curated re-exports — the `remote::cd` internals
/// stay crate-private.
pub mod cd {
    pub use crate::remote::cd::DisaggConfig;
    pub use crate::remote::cd::budget::TierCell;
    pub use crate::remote::cd::wire::{PrefillDispatch, PrefillPlane};
}

#[doc = include_str!("../docs/leader.md")]
pub mod leader;
#[doc = include_str!("../docs/object.md")]
pub mod object;
pub mod p2p;
pub mod tiering;

// `kvbm_engine::offload` keeps the canonical `tiering::offload` reachable at the
// crate root for the mocker + connector construction call sites.
pub use tiering::offload;
pub mod pubsub;
pub mod remote;
#[doc = include_str!("../docs/runtime.md")]
pub mod runtime;
#[doc = include_str!("../docs/worker.md")]
pub mod worker;

#[cfg(feature = "testing")]
pub mod testing;

pub use runtime::{KvbmRuntime, KvbmRuntimeBuilder, RuntimeHandle};
