// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic hanging transport for production deadline coverage.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::future::BoxFuture;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::CacheIdentity;
use kvbm_protocols::control::modules::transfer::{
    OpenTransferSessionResponse, TransferSessionCapability,
};
use kvbm_protocols::disagg::SessionEndpoint;
use tokio_util::sync::CancellationToken;

use super::super::deadline::BundlePullLimits;
use super::super::transfer::{BundleTransfer, BundleTransferError};
use super::super::{
    BundlePullOutcome, BundlePullTarget, OpenedResource, pull_remote_bundle_with_transfer,
};
use super::RemoteBundleCandidate;
use crate::InstanceId;

struct HangingBundleTransfer {
    owner: InstanceId,
}

pub(crate) async fn pull_with_hanging_transfer(
    target: Arc<dyn BundlePullTarget>,
    candidate: RemoteBundleCandidate,
    expected_identity: CacheIdentity,
    timeout: Duration,
) -> Result<BundlePullOutcome> {
    let owner = candidate.advertisement().owner();
    pull_remote_bundle_with_transfer(
        target,
        candidate,
        expected_identity,
        CancellationToken::new(),
        tokio::time::Instant::now() + timeout,
        Arc::new(HangingBundleTransfer { owner }),
        BundlePullLimits::PRODUCTION,
    )
    .await
}

impl BundleTransfer for HangingBundleTransfer {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        _watchdog: Duration,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse, BundleTransferError>> {
        let owner = self.owner;
        Box::pin(async move {
            Ok(OpenTransferSessionResponse::Sync {
                capability: TransferSessionCapability {
                    session_id: uuid::Uuid::new_v4(),
                    instance_id: owner,
                    endpoint: SessionEndpoint {
                        kind: "bundle-timeout-fixture".to_owned(),
                        payload: serde_json::Value::Null,
                    },
                    resource,
                },
                committed: hashes,
                breakdown: Default::default(),
            })
        })
    }

    fn pull(
        &self,
        _resource: &OpenedResource,
    ) -> BoxFuture<'_, Result<crate::p2p::StagedPull, BundleTransferError>> {
        Box::pin(std::future::pending())
    }

    fn close(&self, _session_id: uuid::Uuid, _reason: &str) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}
