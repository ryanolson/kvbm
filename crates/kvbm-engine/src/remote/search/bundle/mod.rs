// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Complete-bundle remote directory contracts and pull orchestration.

use std::collections::BTreeMap;

use kvbm_common::LogicalResourceId;
use kvbm_protocols::cache_manifest::{
    BundleKey, BundleLineageValidationError, BundleResourceLineage, CacheIdentity,
    RegistrationEpoch, ResourceRole, validate_bundle_lineages,
};

use crate::InstanceId;

mod pull;

#[cfg(any(test, feature = "testing"))]
pub(crate) use pull::test_support;
pub(crate) use pull::{BundlePullTarget, OpenedResource, StagedBundle, pull_remote_bundle};

/// One owner's complete, manifest-scoped bundle advertisement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleAdvertisement {
    identity: CacheIdentity,
    key: BundleKey,
    generation: u64,
    owner: InstanceId,
    registration_epoch: RegistrationEpoch,
    expires_at_unix_ms: u64,
    lineages: BTreeMap<LogicalResourceId, BundleResourceLineage>,
}

impl BundleAdvertisement {
    pub fn new(
        identity: CacheIdentity,
        key: BundleKey,
        generation: u64,
        owner: InstanceId,
        registration_epoch: RegistrationEpoch,
        expires_at_unix_ms: u64,
        resource_lineages: impl IntoIterator<Item = BundleResourceLineage>,
    ) -> Result<Self, BundleDirectoryError> {
        if !key.is_compatible_with(&identity) {
            return Err(BundleDirectoryError::ManifestMismatch);
        }
        let resource_lineages = resource_lineages.into_iter().collect::<Vec<_>>();
        Self::validate_lineages(&identity, key, &resource_lineages)?;
        let lineages = resource_lineages
            .into_iter()
            .map(|lineage| (lineage.resource(), lineage))
            .collect();
        Ok(Self {
            identity,
            key,
            generation,
            owner,
            registration_epoch,
            expires_at_unix_ms,
            lineages,
        })
    }

    pub(crate) fn validate_lineages(
        identity: &CacheIdentity,
        key: BundleKey,
        resource_lineages: &[BundleResourceLineage],
    ) -> Result<(), BundleDirectoryError> {
        if !key.is_compatible_with(identity) {
            return Err(BundleDirectoryError::ManifestMismatch);
        }
        validate_bundle_lineages(key, identity.resources(), resource_lineages).map_err(Into::into)
    }

    pub const fn identity(&self) -> &CacheIdentity {
        &self.identity
    }

    pub const fn key(&self) -> BundleKey {
        self.key
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn owner(&self) -> InstanceId {
        self.owner
    }

    pub const fn registration_epoch(&self) -> RegistrationEpoch {
        self.registration_epoch
    }

    pub const fn expires_at_unix_ms(&self) -> u64 {
        self.expires_at_unix_ms
    }

    pub fn resources(&self) -> impl Iterator<Item = LogicalResourceId> + '_ {
        self.lineages.keys().copied()
    }

    pub fn lineages(&self) -> impl Iterator<Item = &BundleResourceLineage> {
        self.lineages.values()
    }

    pub fn matches(&self, query: &BundleDiscoveryQuery) -> bool {
        self.identity == query.identity
            && self.expires_at_unix_ms > query.now_unix_ms
            && query.candidates.contains(&self.key)
    }
}

/// Ordered bundle keys eligible for one remote lookup.
#[derive(Clone, Debug)]
pub struct BundleDiscoveryQuery {
    identity: CacheIdentity,
    candidates: Vec<BundleKey>,
    now_unix_ms: u64,
}

impl BundleDiscoveryQuery {
    pub fn new(identity: CacheIdentity, candidates: Vec<BundleKey>, now_unix_ms: u64) -> Self {
        Self {
            identity,
            candidates,
            now_unix_ms,
        }
    }

    pub const fn identity(&self) -> &CacheIdentity {
        &self.identity
    }

    pub fn candidates(&self) -> &[BundleKey] {
        &self.candidates
    }

    pub const fn now_unix_ms(&self) -> u64 {
        self.now_unix_ms
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BundleDirectoryError {
    #[error("bundle key does not match its cache identity")]
    ManifestMismatch,
    #[error("bundle resources are incomplete: expected {expected:?}, got {actual:?}")]
    IncompleteResources {
        expected: Vec<LogicalResourceId>,
        actual: Vec<LogicalResourceId>,
    },
    #[error("bundle resource {0:?} is duplicated")]
    DuplicateResource(LogicalResourceId),
    #[error(
        "bundle resource {resource:?} has {actual} lineage blocks, expected {expected} for role {role:?} at boundary {boundary_tokens}"
    )]
    InvalidResourceBlockCount {
        resource: LogicalResourceId,
        role: ResourceRole,
        boundary_tokens: u64,
        expected: usize,
        actual: usize,
    },
    #[error("bundle history resource {resource:?} does not reach boundary {boundary_tokens}")]
    ResourceBoundaryMismatch {
        resource: LogicalResourceId,
        boundary_tokens: u64,
    },
    #[error("no bundle prefix history ends at the canonical boundary hash")]
    MissingCanonicalHistoryBoundary,
    #[error("bundle capsule resource {resource:?} does not match the bundle boundary hash")]
    CapsuleBoundaryMismatch { resource: LogicalResourceId },
    #[error("remote bundle lease outlives its advertisement")]
    LeaseOutlivesAdvertisement,
}

impl From<BundleLineageValidationError> for BundleDirectoryError {
    fn from(error: BundleLineageValidationError) -> Self {
        match error {
            BundleLineageValidationError::NoRequirements
            | BundleLineageValidationError::DuplicateRequirement(_)
            | BundleLineageValidationError::UnalignedResourceBoundary { .. } => {
                Self::ManifestMismatch
            }
            BundleLineageValidationError::DuplicateResource(resource) => {
                Self::DuplicateResource(resource)
            }
            BundleLineageValidationError::IncompleteResources { expected, actual } => {
                Self::IncompleteResources { expected, actual }
            }
            BundleLineageValidationError::InvalidResourceBlockCount {
                resource,
                role,
                boundary_tokens,
                expected,
                actual,
            } => Self::InvalidResourceBlockCount {
                resource,
                role,
                boundary_tokens,
                expected,
                actual,
            },
            BundleLineageValidationError::ResourceBoundaryMismatch {
                resource,
                boundary_tokens,
            } => Self::ResourceBoundaryMismatch {
                resource,
                boundary_tokens,
            },
            BundleLineageValidationError::MissingCanonicalHistoryBoundary => {
                Self::MissingCanonicalHistoryBoundary
            }
            BundleLineageValidationError::CapsuleBoundaryMismatch { resource } => {
                Self::CapsuleBoundaryMismatch { resource }
            }
        }
    }
}

/// Stable classification for a complete-bundle directory miss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BundleMissReason {
    NotFound,
    Incompatible,
    Incomplete,
    Expired,
    TimedOut,
    OwnerLost,
    TransferFailed,
    ChecksumFailed,
    CommitFailed,
    Canceled,
}

impl BundleMissReason {
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::NotFound => "not_found",
            Self::Incompatible => "incompatible",
            Self::Incomplete => "incomplete",
            Self::Expired => "expired",
            Self::TimedOut => "timed_out",
            Self::OwnerLost => "owner_lost",
            Self::TransferFailed => "transfer_failed",
            Self::ChecksumFailed => "checksum_failed",
            Self::CommitFailed => "commit_failed",
            Self::Canceled => "canceled",
        }
    }
}

/// Directory lookup outcome that keeps misses distinguishable for metrics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BundleDiscoveryOutcome {
    Hit(Box<RemoteBundleCandidate>),
    Miss(BundleMissReason),
}

/// Terminal result of attempting one directory-issued bundle lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BundlePullOutcome {
    Pulled(BundleKey),
    Miss(BundleMissReason),
}

/// Directory-issued lease for one remote complete-bundle owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteBundleCandidate {
    advertisement: BundleAdvertisement,
    lease_id: uuid::Uuid,
    lease_expires_at_unix_ms: u64,
}

impl RemoteBundleCandidate {
    pub fn new(
        advertisement: BundleAdvertisement,
        lease_id: uuid::Uuid,
        lease_expires_at_unix_ms: u64,
    ) -> Result<Self, BundleDirectoryError> {
        if lease_expires_at_unix_ms > advertisement.expires_at_unix_ms {
            return Err(BundleDirectoryError::LeaseOutlivesAdvertisement);
        }
        Ok(Self {
            advertisement,
            lease_id,
            lease_expires_at_unix_ms,
        })
    }

    pub const fn advertisement(&self) -> &BundleAdvertisement {
        &self.advertisement
    }

    pub const fn lease_id(&self) -> uuid::Uuid {
        self.lease_id
    }

    pub const fn lease_expires_at_unix_ms(&self) -> u64 {
        self.lease_expires_at_unix_ms
    }
}

/// Exact owner-generation invalidation emitted with local bundle eviction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BundleInvalidation {
    pub key: BundleKey,
    pub generation: u64,
    pub owner: InstanceId,
    pub retain_until_unix_ms: u64,
}

pub(crate) fn unix_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests;
