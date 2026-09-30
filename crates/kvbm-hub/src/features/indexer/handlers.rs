// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side velo active-message handler for the KV indexer lookup.
//!
//! Installed on the hub's own velo in
//! [`IndexerManager::attach`](super::manager::IndexerManager) (when a transport
//! is configured). This is the **client → hub** direction — the reverse of the
//! hub → client heartbeat handler in [`crate::handlers`] — and the velo-plane
//! equivalent of `POST /v1/features/indexer/query`, keeping the result typed
//! ([`SequenceHash`] + [`InstanceId`]) instead of stringifying it.

use std::sync::Arc;

use uuid::Uuid;
use velo::Handler;
use velo_ext::InstanceId;

use super::bundle::BundleDirectory;
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
/// `velo_id.as_u128()`). Returns `Ok(None)` on a full miss.
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

/// Build publish, invalidate, and query handlers for the complete-bundle index.
pub fn create_bundle_handlers(directory: Arc<BundleDirectory>) -> [Handler; 3] {
    let publish_directory = Arc::clone(&directory);
    let publish = Handler::typed_unary_async::<BundlePublishRequest, (), _, _>(
        BUNDLE_PUBLISH_HANDLER,
        move |ctx| {
            let directory = Arc::clone(&publish_directory);
            async move {
                directory.publish(ctx.input).map_err(anyhow::Error::from)?;
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
            async move { directory.invalidate(ctx.input).map_err(anyhow::Error::from) }
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
