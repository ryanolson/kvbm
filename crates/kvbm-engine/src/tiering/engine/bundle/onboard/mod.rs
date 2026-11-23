// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic multi-resource onboard transaction.

mod driver;

use std::collections::{BTreeMap, BTreeSet};

use kvbm_common::{BlockId, LogicalResourceId};
use kvbm_protocols::cache_manifest::{BundleKey, CacheIdentity};

use super::barrier::{BarrierError, BarrierState, BarrierUpdate, ResourceBarrier};
use super::offload::BundleResourceFailure;

/// Destination reservations and their all-resource visibility barrier.
pub(in crate::tiering::engine) struct BundleOnboard<R> {
    key: BundleKey,
    barrier: ResourceBarrier,
    reservations: Option<BTreeMap<LogicalResourceId, R>>,
}

impl<R> BundleOnboard<R> {
    pub(in crate::tiering::engine) fn new(
        identity: CacheIdentity,
        key: BundleKey,
        reservations: BTreeMap<LogicalResourceId, R>,
    ) -> Result<Self, BundleOnboardError> {
        if !key.is_compatible_with(&identity) {
            return Err(BundleOnboardError::IncompatibleKey);
        }
        validate_resources(&identity, reservations.keys().copied())?;
        Ok(Self {
            key,
            barrier: ResourceBarrier::new(reservations.keys().copied())?,
            reservations: Some(reservations),
        })
    }

    pub(in crate::tiering::engine) fn start(&mut self) -> BundleOnboardState {
        self.barrier.start().into()
    }

    /// Abandon the bundle while keeping the only copy that still needs proof.
    fn into_paused_reservation(self, resource: LogicalResourceId) -> Option<R> {
        self.reservations.and_then(|mut all| all.remove(&resource))
    }

    pub(in crate::tiering::engine) fn complete(
        &mut self,
        resource: LogicalResourceId,
    ) -> Result<OnboardTransition<R>, BundleOnboardError> {
        match self.barrier.succeed(resource)? {
            BarrierUpdate::Pending => Ok(OnboardTransition::Pending),
            BarrierUpdate::Complete => Ok(OnboardTransition::Visible(BundleVisible {
                computed_tokens: self.key.boundary_tokens(),
                reservations: self.reservations.take().unwrap_or_default(),
            })),
            BarrierUpdate::Settled(state) => {
                Ok(OnboardTransition::Settled(BundleOnboardState::from(state)))
            }
            BarrierUpdate::Aborted => Err(BundleOnboardError::InvalidBarrierTransition),
        }
    }

    pub(in crate::tiering::engine) fn fail(
        &mut self,
        resource: LogicalResourceId,
        failed_blocks: Option<Vec<BlockId>>,
    ) -> Result<OnboardTransition<R>, BundleOnboardError> {
        match self.barrier.fail(resource)? {
            BarrierUpdate::Aborted => {
                drop(self.reservations.take());
                Ok(OnboardTransition::Aborted(BundleResourceFailure::new(
                    resource,
                    failed_blocks,
                )))
            }
            BarrierUpdate::Pending => Ok(OnboardTransition::Pending),
            BarrierUpdate::Settled(state) => {
                Ok(OnboardTransition::Settled(BundleOnboardState::from(state)))
            }
            BarrierUpdate::Complete => Err(BundleOnboardError::InvalidBarrierTransition),
        }
    }
}

fn validate_resources(
    identity: &CacheIdentity,
    resources: impl IntoIterator<Item = LogicalResourceId>,
) -> Result<(), BundleOnboardError> {
    let expected = identity
        .resources()
        .iter()
        .map(|requirement| requirement.resource())
        .collect::<BTreeSet<_>>();
    let actual = resources.into_iter().collect::<BTreeSet<_>>();
    if expected == actual {
        Ok(())
    } else {
        Err(BundleOnboardError::ResourceSetMismatch)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::tiering::engine) enum BundleOnboardState {
    Prepared,
    Transferring,
    Visible,
    Aborted,
}

impl From<BarrierState> for BundleOnboardState {
    fn from(state: BarrierState) -> Self {
        match state {
            BarrierState::Prepared => Self::Prepared,
            BarrierState::Transferring => Self::Transferring,
            BarrierState::Complete => Self::Visible,
            BarrierState::Aborted => Self::Aborted,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::tiering::engine) enum OnboardTransition<R> {
    Pending,
    Visible(BundleVisible<R>),
    Aborted(BundleResourceFailure),
    Settled(BundleOnboardState),
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::tiering::engine) struct BundleVisible<R> {
    computed_tokens: u64,
    reservations: BTreeMap<LogicalResourceId, R>,
}

impl<R> BundleVisible<R> {
    pub(in crate::tiering::engine) const fn computed_tokens(&self) -> u64 {
        self.computed_tokens
    }

    #[cfg(test)]
    pub(in crate::tiering::engine) const fn reservations(&self) -> &BTreeMap<LogicalResourceId, R> {
        &self.reservations
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(in crate::tiering::engine) enum BundleOnboardError {
    #[error("bundle key is incompatible with its cache identity")]
    IncompatibleKey,
    #[error("bundle transaction resources do not match its cache identity")]
    ResourceSetMismatch,
    #[error("resource barrier produced an invalid onboard transition")]
    InvalidBarrierTransition,
    #[error(transparent)]
    Barrier(#[from] BarrierError),
}

#[cfg(test)]
mod tests;
