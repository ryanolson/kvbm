// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Holder-session acquisition, staged physical pull, and bounded cleanup.

use std::sync::Arc;
use std::time::Duration;

use futures::future::{BoxFuture, join_all};
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::RegistrationEpoch;
use kvbm_protocols::control::ControlError;
use kvbm_protocols::control::client::LeaderControlClient;
use kvbm_protocols::control::modules::transfer::{
    CloseTransferSessionRequest, FindMode, OpenTransferSessionRequest, OpenTransferSessionResponse,
    SearchMode, TierSelection,
};

use crate::leader::InstanceLeader;
use crate::p2p::StagedPull;

use super::super::BundleMissReason;
use super::lineage::OpenedResource;

pub(super) trait BundleTransfer: Send + Sync {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        watchdog: Duration,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse, BundleTransferError>>;

    fn pull(
        &self,
        resource: &OpenedResource,
    ) -> BoxFuture<'_, Result<StagedPull, BundleTransferError>>;

    fn close(&self, session_id: uuid::Uuid, reason: &str) -> BoxFuture<'_, ()>;
}

/// Launch a resource pull in a task whose lifetime is independent of its
/// caller's deadline future.
///
/// Dropping Tokio's join handle detaches rather than aborts the task. Once the
/// transfer future has allocated mutable destinations, it therefore continues
/// to own those guards and drain the physical completion notification even if
/// the bundle request has already returned `TimedOut` or `Canceled`. The task
/// also defers holder-session close until after that terminal notification, so
/// source pins cannot be recycled while the transport may still read them.
pub(super) fn spawn_draining_pull(
    transfer: Arc<dyn BundleTransfer>,
    resource: OpenedResource,
    cleanup_timeout: Duration,
) -> tokio::task::JoinHandle<Result<StagedPull, BundleTransferError>> {
    tokio::spawn(async move {
        let session_id = resource.capability().session_id;
        let result = transfer.pull(&resource).await;
        let _ = tokio::time::timeout(
            cleanup_timeout,
            transfer.close(session_id, "bundle resource pull drained"),
        )
        .await;
        result
    })
}

pub(super) struct LeaderBundleTransfer {
    leader: Arc<InstanceLeader>,
    client: LeaderControlClient,
    registration_epoch: RegistrationEpoch,
}

impl LeaderBundleTransfer {
    pub(super) fn new(
        leader: Arc<InstanceLeader>,
        owner: crate::InstanceId,
        registration_epoch: RegistrationEpoch,
    ) -> Self {
        Self {
            client: LeaderControlClient::new(leader.messenger().clone(), owner),
            leader,
            registration_epoch,
        }
    }
}

impl BundleTransfer for LeaderBundleTransfer {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        watchdog: Duration,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse, BundleTransferError>> {
        Box::pin(async move {
            self.client
                .transfer()
                .open_session(OpenTransferSessionRequest {
                    sequence_hashes: hashes,
                    search_mode: SearchMode::Prefix,
                    find_mode: FindMode::Sync,
                    tiers: TierSelection::default(),
                    resource: Some(resource),
                    watchdog_ms: Some(duration_millis(watchdog)),
                    registration_epoch: Some(self.registration_epoch),
                    require_payload_integrity: true,
                })
                .await
                .map_err(|error| classify_control_error(resource, error))
        })
    }

    fn pull(
        &self,
        resource: &OpenedResource,
    ) -> BoxFuture<'_, Result<StagedPull, BundleTransferError>> {
        let leader = Arc::clone(&self.leader);
        let resource = resource.clone();
        Box::pin(async move {
            leader
                .stage_complete_from_session(&resource)
                .await
                .map_err(|error| classify_control_error(resource.resource(), error))
        })
    }

    fn close(&self, session_id: uuid::Uuid, reason: &str) -> BoxFuture<'_, ()> {
        let reason = reason.to_owned();
        Box::pin(async move {
            if let Err(error) = self
                .client
                .transfer()
                .close_session(CloseTransferSessionRequest {
                    session_id,
                    reason: Some(reason),
                })
                .await
            {
                tracing::debug!(
                    %error,
                    %session_id,
                    "bundle session close failed; holder watchdog will reclaim"
                );
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum BundleTransferError {
    #[error("bundle owner was lost: {message}")]
    OwnerLost { message: String },
    #[error("bundle resource {resource:?} transfer failed: {message}")]
    TransferFailed {
        resource: LogicalResourceId,
        message: String,
    },
    #[error("bundle resource {resource:?} payload checksum failed: {message}")]
    ChecksumMismatch {
        resource: LogicalResourceId,
        message: String,
    },
}

impl BundleTransferError {
    pub(super) const fn reason(&self) -> BundleMissReason {
        match self {
            Self::OwnerLost { .. } => BundleMissReason::OwnerLost,
            Self::TransferFailed { .. } => BundleMissReason::TransferFailed,
            Self::ChecksumMismatch { .. } => BundleMissReason::ChecksumFailed,
        }
    }
}

pub(super) async fn close_all(
    transfer: &Arc<dyn BundleTransfer>,
    session_ids: &[uuid::Uuid],
    reason: &str,
    cleanup_timeout: Duration,
) {
    let closes = session_ids.iter().copied().map(|session_id| {
        let transfer = Arc::clone(transfer);
        let reason = reason.to_owned();
        async move {
            let _ =
                tokio::time::timeout(cleanup_timeout, transfer.close(session_id, &reason)).await;
        }
    });
    join_all(closes).await;
}

fn classify_control_error(resource: LogicalResourceId, error: ControlError) -> BundleTransferError {
    let message = error.to_string();
    match &error {
        ControlError::PeerNotFound { .. }
        | ControlError::NotInitialized
        | ControlError::RegistrationEpochMismatch => BundleTransferError::OwnerLost { message },
        ControlError::Internal(reason)
            if reason.contains(" transport:")
                || reason.starts_with("attach:")
                || reason.contains("peer closed") =>
        {
            BundleTransferError::OwnerLost { message }
        }
        ControlError::Internal(reason) if reason.contains("payload_checksum_") => {
            BundleTransferError::ChecksumMismatch { resource, message }
        }
        _ => BundleTransferError::TransferFailed { resource, message },
    }
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().clamp(1, u128::from(u64::MAX)) as u64
}
