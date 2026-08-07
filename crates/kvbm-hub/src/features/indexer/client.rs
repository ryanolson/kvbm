// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-side velo lookup wrapper for the KV indexer feature.
//!
//! Mirrors [`ConditionalDisaggClient`](crate::features::disagg::client::ConditionalDisaggClient):
//! a thin wrapper over an [`Arc<Messenger>`] that knows the indexer
//! [`QUERY_HANDLER`] name and the hub's velo [`InstanceId`], exposing a single
//! per-request [`find_blocks`](IndexerLookupClient::find_blocks) call. Construct
//! it via [`HubClient::indexer_lookup_client`](crate::HubClient::indexer_lookup_client),
//! which gates on the indexer being enabled and supplies the hub's `InstanceId`.
//!
//! # Why the tier-placement snapshot push lives here
//!
//! [`push_tier_placement_snapshot`](IndexerLookupClient::push_tier_placement_snapshot)
//! is the one call in this client that travels HTTP rather than velo, because
//! the hub mounts the install as a credential-authorized mutation on its
//! control port. It belongs here anyway: the credential is registration-scoped
//! authority that
//! [`HubClient`](crate::HubClient) deliberately does not hand out (no accessor,
//! and a fabricated one fails `authorize_owner_epoch` by construction), so a
//! publisher in another crate cannot assemble the request itself. Exposing the
//! *call* instead of the *credential* keeps that property while making the
//! recovery half of R7b §3 reachable.

use std::sync::Arc;

use anyhow::{Context, Result};
use kvbm_logical::SequenceHash;
use kvbm_protocols::cache_manifest::RegistrationEpoch;
use kvbm_protocols::tier_protocol::TierPlacementSnapshotV1;
use url::Url;
use velo::Messenger;
use velo_ext::InstanceId;

use super::protocol::{
    BUNDLE_INVALIDATE_HANDLER, BUNDLE_PUBLISH_HANDLER, BUNDLE_QUERY_HANDLER,
    BundleAdvertisementRecord, BundleInvalidateRequest, BundleInvalidationRecord,
    BundlePublishRequest, BundleQueryOutcome, BundleQueryRequest, FindBlocksHit, QUERY_HANDLER,
    QueryRequest, TierPlacementSnapshotRequest, TierPlacementSnapshotResponse,
};
use crate::protocol::MutationCredential;

/// Velo-plane lookup client for the hub's KV block index.
pub struct IndexerLookupClient {
    messenger: Arc<Messenger>,
    /// Hub's velo `InstanceId` — the target of the lookup unary RPC.
    hub_velo_id: InstanceId,
    mutation_credential: MutationCredential,
    registration_epoch: RegistrationEpoch,
    /// Shared with the owning [`HubClient`](crate::HubClient) — one connection
    /// pool, one set of timeouts.
    http: reqwest::Client,
    /// Fully resolved `POST` target for the tier-placement snapshot install,
    /// built against the hub's **control** base at construction. Resolved once
    /// so the publisher's push path cannot fail on URL joining.
    tier_placement_snapshot_url: Url,
}

impl std::fmt::Debug for IndexerLookupClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexerLookupClient")
            .field("hub_velo_id", &self.hub_velo_id)
            .finish()
    }
}

impl IndexerLookupClient {
    /// Wrap a [`Messenger`] targeting the hub at `hub_velo_id`.
    pub(crate) fn new(
        messenger: Arc<Messenger>,
        hub_velo_id: InstanceId,
        mutation_credential: MutationCredential,
        registration_epoch: RegistrationEpoch,
        http: reqwest::Client,
        tier_placement_snapshot_url: Url,
    ) -> Arc<Self> {
        Arc::new(Self {
            messenger,
            hub_velo_id,
            mutation_credential,
            registration_epoch,
            http,
            tier_placement_snapshot_url,
        })
    }

    /// The registration epoch this client's credential authorizes.
    ///
    /// A publisher stamps it on every tier-placement message
    /// ([`TierPlacementSequencer::new`](kvbm_protocols::tier_protocol::TierPlacementSequencer::new)
    /// takes one), and the hub rejects a snapshot whose body disagrees with the
    /// epoch its credential resolves to. Exposing it here means the publisher
    /// reads the epoch from the same object that will authorize the push,
    /// rather than from a second handle that may have re-registered since.
    #[must_use]
    pub fn registration_epoch(&self) -> RegistrationEpoch {
        self.registration_epoch
    }

    /// The hub's velo `InstanceId` this client targets.
    pub fn hub_velo_id(&self) -> InstanceId {
        self.hub_velo_id
    }

    /// Resolve a candidate block sequence to the deepest indexed block and its
    /// holders, over velo.
    ///
    /// `hashes` are the block-sequence PLHs in position order (low → high). The
    /// hub walks them and returns the deepest one present — so `[x, y, z]` with
    /// `z` missing but `y` indexed yields `Some(hit)` where `hit.matched == y`
    /// and `hit.candidates` are the instances holding `y`. A full miss returns
    /// `Ok(None)`.
    pub async fn find_blocks(&self, hashes: Vec<SequenceHash>) -> Result<Option<FindBlocksHit>> {
        let req = QueryRequest { hashes };
        let hit = self
            .messenger
            .typed_unary::<Option<FindBlocksHit>>(QUERY_HANDLER)?
            .payload(&req)?
            .instance(self.hub_velo_id)
            .send()
            .await?;
        Ok(hit)
    }

    pub async fn publish_bundle(&self, advertisement: BundleAdvertisementRecord) -> Result<()> {
        validate_registration_epoch(advertisement.registration_epoch, self.registration_epoch)?;
        let request = BundlePublishRequest {
            credential: self.mutation_credential.clone(),
            advertisement,
        };
        self.messenger
            .typed_unary::<()>(BUNDLE_PUBLISH_HANDLER)?
            .payload(&request)?
            .instance(self.hub_velo_id)
            .send()
            .await?;
        Ok(())
    }

    pub async fn invalidate_bundle(&self, invalidation: BundleInvalidationRecord) -> Result<bool> {
        let request = BundleInvalidateRequest {
            credential: self.mutation_credential.clone(),
            key: invalidation.key,
            generation: invalidation.generation,
            owner: invalidation.owner,
            retain_until_unix_ms: invalidation.retain_until_unix_ms,
        };
        self.messenger
            .typed_unary::<bool>(BUNDLE_INVALIDATE_HANDLER)?
            .payload(&request)?
            .instance(self.hub_velo_id)
            .send()
            .await
    }

    /// Install this publisher's full tier-placement state on the hub
    /// (`POST /v1/features/indexer/tier-placements/snapshot`).
    ///
    /// This is the reliable, authenticated half of R7b §3: deltas ride the
    /// lossy ZMQ plane and can only *update* a projection, while the hub's map
    /// grows exclusively through an authorized install. A publisher that never
    /// calls this emits deltas no consumer can ever apply.
    ///
    /// Feed the returned
    /// [`TierPlacementSnapshotResponse::installed_generation`] to
    /// [`TierPlacementSequencer::note_snapshot_installed`](kvbm_protocols::tier_protocol::TierPlacementSequencer::note_snapshot_installed)
    /// — including when `installed` is `false`. That is the `AlreadyCurrent`
    /// answer to a periodic push the hub already holds: a success, not an
    /// error, and its generation still has to release the emission gate or the
    /// publisher goes silent.
    ///
    /// The epoch is checked locally first. The hub answers a mismatch with a
    /// `409`, but that costs a round trip to learn something this client
    /// already knows, and a publisher that has silently re-registered under a
    /// new epoch needs a diagnosable error rather than a status code.
    pub async fn push_tier_placement_snapshot(
        &self,
        snapshot: TierPlacementSnapshotV1,
    ) -> Result<TierPlacementSnapshotResponse> {
        anyhow::ensure!(
            snapshot.registration_epoch == self.registration_epoch,
            "tier placement snapshot registration epoch does not match the indexer \
             client registration"
        );
        let request = TierPlacementSnapshotRequest {
            credential: self.mutation_credential.clone(),
            snapshot,
        };
        let resp = self
            .http
            .post(self.tier_placement_snapshot_url.clone())
            .json(&request)
            .send()
            .await
            .with_context(|| format!("POST {}", self.tier_placement_snapshot_url))?;
        crate::client::parse_json(resp).await
    }

    pub async fn find_bundle(&self, request: BundleQueryRequest) -> Result<BundleQueryOutcome> {
        self.messenger
            .typed_unary::<BundleQueryOutcome>(BUNDLE_QUERY_HANDLER)?
            .payload(&request)?
            .instance(self.hub_velo_id)
            .send()
            .await
    }
}

fn validate_registration_epoch(
    advertised: Option<RegistrationEpoch>,
    registered: RegistrationEpoch,
) -> Result<()> {
    anyhow::ensure!(
        advertised == Some(registered),
        "bundle advertisement registration epoch does not match the indexer client registration"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_publication_rejects_missing_or_mismatched_registration_epoch() {
        let registered = RegistrationEpoch::new();

        assert!(validate_registration_epoch(None, registered).is_err());
        assert!(validate_registration_epoch(Some(RegistrationEpoch::new()), registered).is_err());
        assert!(validate_registration_epoch(Some(registered), registered).is_ok());
    }
}
