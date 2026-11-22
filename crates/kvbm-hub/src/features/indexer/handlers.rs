// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side velo active-message handler for the KV indexer lookup.
//!
//! Installed on the hub's own velo in
//! [`IndexerManager::attach`](super::manager::IndexerManager) (when a transport
//! is configured). This is the **client → hub** direction — the reverse of the
//! hub → client heartbeat handler in [`crate::handlers`] — and the velo-plane
//! equivalent of `POST /v1/ext/indexer/query`, keeping the result typed
//! ([`SequenceHash`] + [`InstanceId`]) instead of stringifying it.

use std::sync::Arc;

use uuid::Uuid;
use velo::Handler;
use velo_ext::InstanceId;

use kvbm_protocols::cache_manifest::RegistrationEpoch;
use velo_hub::protocol::Credential;
use velo_hub::{Authority, AuthorityError};

use super::bundle::{BundleDirectory, BundleDirectoryError};
use super::index::ManifestIndexes;
use super::protocol::{
    BUNDLE_INVALIDATE_HANDLER, BUNDLE_PUBLISH_HANDLER, BUNDLE_QUERY_HANDLER,
    BundleInvalidateRequest, BundlePublishRequest, BundleQueryOutcome, BundleQueryRequest,
    FindBlocksHit, QUERY_HANDLER, QueryRequest,
};

/// Build the indexer-lookup velo handler over manifest-scoped indexes.
///
/// Resolves the candidate hashes to the deepest indexed block and its holders
/// via [`ManifestIndexes::query_holders`], reconstructing each holder's
/// [`InstanceId`] from the raw `u128` the index stores (publishers stamp
/// `velo_id.as_u128()`). Request hashes must be ordered by ascending
/// `position()`; unsorted input may return a shallower match. Returns
/// `Ok(None)` on a full miss.
pub fn create_query_handler(indexes: Arc<ManifestIndexes>) -> Handler {
    Handler::typed_unary_async::<QueryRequest, Option<FindBlocksHit>, _, _>(
        QUERY_HANDLER,
        move |ctx| {
            let indexes = Arc::clone(&indexes);
            async move {
                Ok(indexes
                    .query_holders(ctx.input.manifest, &ctx.input.hashes)
                    .map(|(matched, ids, kind)| FindBlocksHit {
                        matched,
                        candidates: ids
                            .into_iter()
                            .map(|u| InstanceId::from(Uuid::from_u128(u)))
                            .collect(),
                        kind,
                    }))
            }
        },
    )
    .build()
}

/// The registration epoch the hub's lease grants `credential` for `owner`.
pub(super) fn authorized_epoch(
    authority: &Authority,
    owner: InstanceId,
    credential: &Credential,
) -> Result<RegistrationEpoch, BundleDirectoryError> {
    let lease = authority
        .authorize(owner, Some(credential))
        .map_err(|error| match error {
            AuthorityError::Unavailable => BundleDirectoryError::Unavailable,
            _ => BundleDirectoryError::UnauthorizedOwner { owner },
        })?;
    Ok(RegistrationEpoch::from_uuid(lease.epoch.as_uuid()))
}

/// Build publish, invalidate, and query handlers for the complete-bundle index.
pub fn create_bundle_handlers(
    directory: Arc<BundleDirectory>,
    authority: Authority,
) -> [Handler; 3] {
    let publish_directory = Arc::clone(&directory);
    let publish_authority = authority.clone();
    let publish = Handler::typed_unary_async::<BundlePublishRequest, (), _, _>(
        BUNDLE_PUBLISH_HANDLER,
        move |ctx| {
            let directory = Arc::clone(&publish_directory);
            let authority = publish_authority.clone();
            async move {
                let request = ctx.input;
                let epoch =
                    authorized_epoch(&authority, request.advertisement.owner, &request.credential)?;
                directory.publish(epoch, request)?;
                Ok(())
            }
        },
    )
    .build();

    let invalidate_directory = Arc::clone(&directory);
    let invalidate = Handler::typed_unary_async::<BundleInvalidateRequest, bool, _, _>(
        BUNDLE_INVALIDATE_HANDLER,
        move |ctx| {
            let directory = Arc::clone(&invalidate_directory);
            let authority = authority.clone();
            async move {
                let request = ctx.input;
                let epoch = authorized_epoch(&authority, request.owner, &request.credential)?;
                Ok(directory.invalidate(epoch, request)?)
            }
        },
    )
    .build();

    let query = Handler::typed_unary_async::<BundleQueryRequest, BundleQueryOutcome, _, _>(
        BUNDLE_QUERY_HANDLER,
        move |ctx| {
            let directory = Arc::clone(&directory);
            async move { Ok(directory.query(ctx.input)) }
        },
    )
    .build();

    [publish, invalidate, query]
}
