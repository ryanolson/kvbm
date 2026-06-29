// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use futures::future::{Either, Ready, ready};
use serde::{Deserialize, Serialize};
use std::{
    pin::Pin,
    task::{Context, Poll},
};

pub use crate::worker::{ImportMetadataResponseAwaiter, SerializedResponseAwaiter};
pub use crate::{BlockId, SequenceHash};
pub use kvbm_common::LogicalLayoutHandle;
pub use kvbm_physical::manager::{LayoutHandle, SerializedLayout};

pub struct SerializedLayoutResponse {
    awaiter: Either<Ready<Result<SerializedLayout>>, SerializedResponseAwaiter>,
}

impl SerializedLayoutResponse {
    pub fn ready(layout: SerializedLayout) -> Self {
        Self {
            awaiter: Either::Left(ready(Ok(layout))),
        }
    }

    pub fn from_boxed(awaiter: SerializedResponseAwaiter) -> Self {
        Self {
            awaiter: Either::Right(awaiter),
        }
    }

    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, Either::Right(_))
    }
}

impl std::future::IntoFuture for SerializedLayoutResponse {
    type Output = Result<SerializedLayout>;
    type IntoFuture = Either<Ready<Result<SerializedLayout>>, SerializedResponseAwaiter>;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}

pub struct ImportMetadataResponse {
    awaiter: Either<Ready<Result<Vec<LayoutHandle>>>, ImportMetadataResponseAwaiter>,
}

impl ImportMetadataResponse {
    pub fn ready(handles: Vec<LayoutHandle>) -> Self {
        Self {
            awaiter: Either::Left(ready(Ok(handles))),
        }
    }

    pub fn from_boxed(awaiter: ImportMetadataResponseAwaiter) -> Self {
        Self {
            awaiter: Either::Right(awaiter),
        }
    }

    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, Either::Right(_))
    }
}

impl std::future::IntoFuture for ImportMetadataResponse {
    type Output = Result<Vec<LayoutHandle>>;
    type IntoFuture = Either<Ready<Result<Vec<LayoutHandle>>>, ImportMetadataResponseAwaiter>;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}

/// Response type for `connect_remote` operations.
///
/// This type represents the completion state of a remote metadata import
/// with handle mapping storage. Like other response types, it can be awaited.
///
/// For direct workers, this is typically ready immediately.
/// For replicated workers, this aggregates multiple underlying imports.
pub struct ConnectRemoteResponse {
    awaiter: ConnectRemoteAwaiter,
}

pub enum ConnectRemoteAwaiter {
    Ready(Ready<Result<()>>),
    Event(::velo::EventAwaiter),
}

impl std::future::Future for ConnectRemoteAwaiter {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            Self::Ready(ready) => Pin::new(ready).poll(cx),
            Self::Event(waiter) => Pin::new(waiter).poll(cx),
        }
    }
}

impl ConnectRemoteResponse {
    /// Create a response that is already completed.
    ///
    /// This is used when the connect operation completes synchronously,
    /// such as for DirectWorker with local metadata import.
    pub fn ready() -> Self {
        Self {
            awaiter: ConnectRemoteAwaiter::Ready(ready(Ok(()))),
        }
    }

    /// Create a response from an event waiter.
    ///
    /// This is used when the connect operation requires waiting for
    /// multiple underlying operations to complete (e.g., ReplicatedWorker).
    pub fn from_awaiter(awaiter: ::velo::EventAwaiter) -> Self {
        Self {
            awaiter: ConnectRemoteAwaiter::Event(awaiter),
        }
    }

    /// Check if the response can yield the current task.
    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, ConnectRemoteAwaiter::Event(_))
    }
}

impl std::future::IntoFuture for ConnectRemoteResponse {
    type Output = Result<()>;
    type IntoFuture = ConnectRemoteAwaiter;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}

/// Remote descriptor for transfer operations.
#[derive(Serialize, Deserialize, Clone)]
pub enum RemoteDescriptor {
    Layout {
        handle: LayoutHandle,
        block_ids: Vec<BlockId>,
    },
    Object {
        keys: Vec<SequenceHash>,
    },
}

/// Configuration sent from leader to workers for G2/G3/G4 layout creation.
///
/// This message is sent via Velo RPC during Phase 3 coordination.
/// Workers use this to create additional cache tiers beyond G1 (GPU KV).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CollectiveBootstrap {
    /// KVBM-owned NCCL communicator bootstrap serialized by `NcclBootstrap`.
    Nccl { serialized: Vec<u8> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderLayoutConfig {
    /// Leader provided rank of this worker
    ///
    /// The Connector framework provides us with an ordered list of workers. To ensure
    /// leaders and workers are all in-sync on this information, the leader will send
    /// each worker the rank provided by the Connector framework.
    pub rank: usize,

    /// Number of workers participating in this cache-parallel group.
    #[serde(default = "default_worker_count")]
    pub worker_count: usize,

    /// Number of physical host/pinned blocks contributed by this worker.
    pub host_block_count: usize,

    /// Number of disk blocks for G3 tier (None = no disk tier).
    pub disk_block_count: Option<usize>,

    /// Object storage configuration for G4 tier (None = no object tier).
    ///
    /// When present, workers should instantiate object clients for storing
    /// blocks in external object storage (S3/MinIO).
    #[serde(default)]
    pub object: Option<kvbm_config::ObjectConfig>,

    /// Parallelism mode for this worker.
    ///
    /// In `ReplicatedData`, every worker contributes a disjoint stripe of
    /// canonical lower-tier blocks while G1 remains replicated.
    #[serde(default)]
    pub parallelism: kvbm_config::ParallelismMode,

    /// Optional KVBM-owned collective bootstrap for replicated cache data.
    #[serde(default)]
    pub collective: Option<CollectiveBootstrap>,
}

fn default_worker_count() -> usize {
    1
}

/// Worker's response after configuring additional layouts (G2, G3).
///
/// Returned in response to a `LeaderLayoutConfig` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerLayoutResponse {
    /// Full exported metadata including all registered layouts (G1, G2, G3).
    pub metadata: SerializedLayout,

    /// Which logical layouts were successfully created in this operation.
    pub created_layouts: Vec<LogicalLayoutHandle>,
}

#[cfg(test)]
mod leader_layout_config_tests {
    use super::{CollectiveBootstrap, LeaderLayoutConfig};

    #[test]
    fn collective_bootstrap_round_trips_with_worker_layout_config() {
        let config = LeaderLayoutConfig {
            rank: 1,
            worker_count: 2,
            host_block_count: 128,
            disk_block_count: None,
            object: None,
            parallelism: kvbm_config::ParallelismMode::ReplicatedData,
            collective: Some(CollectiveBootstrap::Nccl {
                serialized: vec![1, 2, 3, 4],
            }),
        };

        let wire = serde_json::to_vec(&config).unwrap();
        let decoded: LeaderLayoutConfig = serde_json::from_slice(&wire).unwrap();
        assert_eq!(decoded.worker_count, 2);
        assert_eq!(decoded.collective, config.collective);
    }
}
