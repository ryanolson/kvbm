// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Complete G2 lineage proof for one validated remote bundle.

use std::collections::BTreeMap;
use std::time::Duration;

use futures::future::BoxFuture;
use kvbm_common::LogicalResourceId;
use kvbm_protocols::cache_manifest::{BundleKey, BundleResourceLineage, CacheIdentity};
use kvbm_protocols::control::modules::transfer::{
    OpenTransferSessionResponse, TransferSessionCapability,
};

use super::super::{BundleAdvertisement, BundleDirectoryError};
use super::transfer::{BundleTransfer, BundleTransferError};

/// One resource lineage minted only after whole-bundle validation.
#[derive(Clone)]
pub(super) struct CompleteG2Lineage {
    lineage: BundleResourceLineage,
}

/// A validated holder capability bound to one complete resource lineage.
#[derive(Clone)]
pub(crate) struct OpenedResource {
    capability: TransferSessionCapability,
    lineage: CompleteG2Lineage,
}

impl CompleteG2Lineage {
    /// Validate the whole resource set before minting any lineage proof.
    pub(super) fn for_bundle(
        identity: &CacheIdentity,
        key: BundleKey,
        lineages: &[BundleResourceLineage],
    ) -> Result<BTreeMap<LogicalResourceId, Self>, BundleDirectoryError> {
        BundleAdvertisement::validate_lineages(identity, key, lineages)?;
        Ok(lineages
            .iter()
            .cloned()
            .map(|lineage| (lineage.resource(), Self { lineage }))
            .collect())
    }

    /// Return the logical resource that owns this complete lineage.
    pub(super) const fn resource(&self) -> LogicalResourceId {
        self.lineage.resource()
    }

    /// Open one transfer session with the complete root-to-leaf lineage.
    pub(super) fn open_full<'a>(
        &'a self,
        transfer: &'a dyn BundleTransfer,
        watchdog: Duration,
    ) -> BoxFuture<'a, Result<OpenTransferSessionResponse, BundleTransferError>> {
        transfer.open(self.resource(), self.lineage.hashes().to_vec(), watchdog)
    }

    /// Bind one holder capability only when its committed lineage is complete.
    pub(super) fn bind_open(
        &self,
        capability: TransferSessionCapability,
        committed: &[kvbm_common::SequenceHash],
    ) -> Option<OpenedResource> {
        if capability.resource != self.resource() || committed != self.lineage.hashes() {
            return None;
        }
        Some(OpenedResource {
            capability,
            lineage: self.clone(),
        })
    }

    pub(super) fn hashes(&self) -> &[kvbm_common::SequenceHash] {
        self.lineage.hashes()
    }
}

impl OpenedResource {
    pub(crate) const fn capability(&self) -> &TransferSessionCapability {
        &self.capability
    }

    pub(crate) const fn resource(&self) -> LogicalResourceId {
        self.lineage.resource()
    }

    pub(crate) fn hashes(&self) -> &[kvbm_common::SequenceHash] {
        self.lineage.hashes()
    }
}

#[cfg(test)]
mod tests;
