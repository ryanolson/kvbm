//! Hub index protocol adapter for remote discovery.

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use kvbm_engine::remote::search::bundle::{
    BundleAdvertisement, BundleDiscoveryOutcome, BundleDiscoveryQuery, BundleInvalidation,
    BundleMissReason, RemoteBundleCandidate,
};
use kvbm_hub::{
    BundleAdvertisementRecord, BundleInvalidationRecord, BundleQueryHit, BundleQueryMissReason,
    BundleQueryOutcome, BundleQueryRequest, FindBlocksHit, IndexerLookupClient,
};
use kvbm_logical::SequenceHash;

pub(super) trait BlockIndex: Send + Sync {
    fn find_blocks(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<FindBlocksHit>>>;

    fn find_bundle(
        &self,
        _query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        Box::pin(async { Ok(BundleDiscoveryOutcome::Miss(BundleMissReason::NotFound)) })
    }

    fn publish_bundle(
        &self,
        _advertisement: BundleAdvertisement,
    ) -> BoxFuture<'static, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    fn invalidate_bundle(
        &self,
        _invalidation: BundleInvalidation,
    ) -> BoxFuture<'static, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

pub(super) struct HubBlockIndex(pub(super) Arc<IndexerLookupClient>);

impl BlockIndex for HubBlockIndex {
    fn find_blocks(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<FindBlocksHit>>> {
        let index = Arc::clone(&self.0);
        Box::pin(async move {
            index
                .find_blocks(hashes)
                .await
                .context("query KVBM hub block index")
        })
    }

    fn find_bundle(
        &self,
        query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        let index = Arc::clone(&self.0);
        Box::pin(async move {
            let request = BundleQueryRequest {
                manifest: query.identity().manifest(),
                requirements: query.identity().resources().to_vec(),
                candidates: query.candidates().to_vec(),
                now_unix_ms: query.now_unix_ms(),
            };
            match index
                .find_bundle(request)
                .await
                .context("query KVBM hub bundle index")?
            {
                BundleQueryOutcome::Hit(hit) => directory_hit(query, hit)
                    .map(Box::new)
                    .map(BundleDiscoveryOutcome::Hit),
                BundleQueryOutcome::Miss(reason) => {
                    Ok(BundleDiscoveryOutcome::Miss(translate_miss_reason(reason)))
                }
            }
        })
    }

    fn publish_bundle(&self, advertisement: BundleAdvertisement) -> BoxFuture<'static, Result<()>> {
        let index = Arc::clone(&self.0);
        Box::pin(async move {
            index
                .publish_bundle(advertisement_record(&advertisement))
                .await
                .context("publish KVBM bundle advertisement")
        })
    }

    fn invalidate_bundle(
        &self,
        invalidation: BundleInvalidation,
    ) -> BoxFuture<'static, Result<()>> {
        let index = Arc::clone(&self.0);
        Box::pin(async move {
            index
                .invalidate_bundle(BundleInvalidationRecord {
                    key: invalidation.key,
                    generation: invalidation.generation,
                    owner: invalidation.owner,
                    retain_until_unix_ms: invalidation.retain_until_unix_ms,
                })
                .await
                .context("invalidate KVBM bundle advertisement")?;
            Ok(())
        })
    }
}

pub(super) fn advertisement_record(
    advertisement: &BundleAdvertisement,
) -> BundleAdvertisementRecord {
    BundleAdvertisementRecord {
        key: advertisement.key(),
        generation: advertisement.generation(),
        owner: advertisement.owner(),
        registration_epoch: Some(advertisement.registration_epoch()),
        requirements: advertisement.identity().resources().to_vec(),
        lineages: advertisement.lineages().cloned().collect(),
        expires_at_unix_ms: advertisement.expires_at_unix_ms(),
        // Populated by the CT-2 publisher wiring; the hub stamps
        // `advertised_at_unix_ms` itself at publish.
        placements: Vec::new(),
        stage_cost_hint_us: None,
        advertised_at_unix_ms: None,
    }
}

const fn translate_miss_reason(reason: BundleQueryMissReason) -> BundleMissReason {
    match reason {
        BundleQueryMissReason::NotFound => BundleMissReason::NotFound,
        BundleQueryMissReason::Incompatible => BundleMissReason::Incompatible,
        BundleQueryMissReason::Incomplete => BundleMissReason::Incomplete,
        BundleQueryMissReason::Expired => BundleMissReason::Expired,
    }
}

pub(super) fn directory_hit(
    query: BundleDiscoveryQuery,
    hit: BundleQueryHit,
) -> Result<RemoteBundleCandidate> {
    let record = hit.advertisement;
    anyhow::ensure!(
        record.requirements.as_slice() == query.identity().resources(),
        "hub returned incompatible bundle requirements"
    );
    let registration_epoch = record
        .registration_epoch
        .context("hub bundle hit omitted its owner registration epoch")?;
    let advertisement = BundleAdvertisement::new(
        query.identity().clone(),
        record.key,
        record.generation,
        record.owner,
        registration_epoch,
        record.expires_at_unix_ms,
        record.lineages,
    )?;
    anyhow::ensure!(
        advertisement.matches(&query),
        "hub returned an incompatible bundle advertisement"
    );
    RemoteBundleCandidate::new(advertisement, hit.lease_id, hit.lease_expires_at_unix_ms)
        .map_err(anyhow::Error::from)
}
