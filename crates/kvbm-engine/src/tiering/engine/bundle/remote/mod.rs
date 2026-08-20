// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote publication, invalidation, and local commit of pulled bundles.

mod order;
mod publication;
mod timeout;

pub(in crate::tiering::engine) use order::BundleDirectoryOrder;
pub(in crate::tiering::engine) use publication::BundlePublicationRuntime;

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::future::BoxFuture;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::ImmutableBlock;
use kvbm_protocols::cache_manifest::{BundleKey, BundleResourceLineage, CacheIdentity};

use super::{BUNDLE_DIRECTORY_TTL_MS, BundleCatalogError};
use crate::G2;
use crate::leader::InstanceLeader;
use crate::remote::search::bundle::{
    BundleAdvertisement, BundleInvalidation, BundlePullTarget, StagedBundle,
};
use crate::tiering::engine::local::LocalConnectorEngine;
use crate::tiering::policy::ResourceLineage;

use publication::BundlePublication;
use timeout::{DirectoryCallError, run_directory_call};

impl BundlePullTarget for LocalConnectorEngine {
    fn instance_leader(&self) -> Arc<InstanceLeader> {
        Arc::clone(&self.leader)
    }

    fn reserve_publication_generation(&self) -> Result<u64> {
        self.leader
            .reserve_bundle_publication_generation()
            .map_err(anyhow::Error::from)
    }

    fn resource_bytes(&self, resource: LogicalResourceId, logical_blocks: usize) -> Option<u64> {
        self.bundle_admission
            .resource_bytes(resource, logical_blocks)
    }

    fn commit_pulled_bundle(
        &self,
        identity: CacheIdentity,
        key: BundleKey,
        generation: u64,
        bundle: StagedBundle,
    ) -> BoxFuture<'static, Result<()>> {
        let engine = self.weak_self.upgrade();
        Box::pin(async move {
            let engine = engine.ok_or_else(|| anyhow!("bundle pull target was dropped"))?;
            let mut dependency_lineages = Vec::with_capacity(identity.resources().len());
            let mut advertisement_lineages = Vec::with_capacity(identity.resources().len());
            for requirement in identity.resources() {
                let resource = requirement.resource();
                let hashes = bundle
                    .lineages()
                    .get(&resource)
                    .ok_or_else(|| anyhow!("pulled bundle is missing lineage for {resource:?}"))?;
                dependency_lineages.push(ResourceLineage::new(
                    resource,
                    requirement.role(),
                    hashes.clone(),
                ));
                advertisement_lineages.push(BundleResourceLineage::new(resource, hashes.clone())?);
            }
            BundleAdvertisement::validate_lineages(&identity, key, &advertisement_lineages)?;
            let resource_ids = bundle.resource_ids().collect();
            // Snapshot the lineage for the residency observer before the bundle
            // moves into the materializer. Only when one is installed: the
            // clone is per-pull and the default engine has no observer.
            let observed_lineages = engine
                .pulled_bundle_ready
                .get()
                .map(|_| bundle.lineages().clone());
            let mut published = false;
            {
                engine
                    .bundle_catalog
                    .lock()
                    .expect("bundle-catalog mutex poisoned")
                    .commit_materialized(
                        &identity,
                        key,
                        generation,
                        resource_ids,
                        dependency_lineages,
                        || {
                            published = true;
                            bundle.publish().map_err(|error| {
                                BundleCatalogError::Materialization(error.to_string())
                            })
                        },
                    )?;
            }
            // Fired here, not inside the materializer, and only if the
            // materializer actually ran. `commit_materialized` returns `Ok`
            // without publishing on the idempotent arm, and it can still fail
            // ahead of the closure — announcing Ready in either case would be
            // over-reporting residency, which is the one direction this stream
            // must never fail in. Firing outside the closure also keeps a
            // caller-supplied callback off the bundle-catalog lock.
            if let (Some(observer), Some(lineages)) =
                (engine.pulled_bundle_ready.get(), observed_lineages)
                && published
            {
                observer(&lineages);
            }
            engine.queue_directory_update(BundleDirectoryUpdate::Advertise {
                identity,
                key,
                generation,
                lineages: advertisement_lineages,
            });
            Ok(())
        })
    }
}

impl LocalConnectorEngine {
    /// Canonical publication path for local offload and completed remote pull.
    /// Strong pins stay in this frame until the one-lock catalog transaction is
    /// complete; only then is the directory update queued.
    pub(in crate::tiering::engine) fn commit_bundle(
        &self,
        identity: CacheIdentity,
        key: BundleKey,
        generation: u64,
        resources: BTreeMap<LogicalResourceId, Vec<ImmutableBlock<G2>>>,
        lineages: Vec<ResourceLineage>,
    ) -> Result<(), BundleCatalogError> {
        let advertisement_lineages = lineages
            .iter()
            .map(ResourceLineage::advertisement_lineage)
            .collect::<Result<Vec<_>, _>>()?;
        BundleAdvertisement::validate_lineages(&identity, key, &advertisement_lineages)?;
        {
            self.bundle_catalog
                .lock()
                .expect("bundle-catalog mutex poisoned")
                .commit(&identity, key, generation, &resources, lineages)?;
        }
        drop(resources);
        self.queue_directory_update(BundleDirectoryUpdate::Advertise {
            identity,
            key,
            generation,
            lineages: advertisement_lineages,
        });
        Ok(())
    }

    pub(in crate::tiering::engine) fn invalidate_resource_blocks(
        &self,
        resource: LogicalResourceId,
        hashes: &[SequenceHash],
    ) {
        let invalidated = self
            .bundle_catalog
            .lock()
            .expect("bundle-catalog mutex poisoned")
            .invalidate_resource(resource, hashes);
        if invalidated.is_empty() {
            return;
        }
        for event in invalidated {
            tracing::debug!(
                resource = ?event.resource(),
                hash = ?event.hash(),
                boundary_tokens = event.key().boundary_tokens(),
                "resource eviction invalidated a dependent bundle"
            );
            self.queue_directory_update(BundleDirectoryUpdate::Invalidate {
                key: event.key(),
                generation: event.generation(),
            });
        }
    }

    fn queue_directory_update(&self, update: BundleDirectoryUpdate) {
        let engine = self.weak_self.clone();
        self.leader.runtime().spawn(async move {
            let Some(engine) = engine.upgrade() else {
                return;
            };
            let _order = engine.bundle_directory_order.enter(update.key()).await;
            match update {
                BundleDirectoryUpdate::Advertise {
                    identity,
                    key,
                    generation,
                    lineages,
                } => {
                    let is_current = engine
                        .bundle_catalog
                        .lock()
                        .expect("bundle-catalog mutex poisoned")
                        .is_current(&key, generation);
                    if !is_current {
                        engine.bundle_publications.forget(&key, generation);
                        return;
                    }
                    let Some(directory) = engine.leader.remote_discovery() else {
                        return;
                    };
                    let publication = BundlePublication::new(identity, key, generation, lineages);
                    let advertisement = match publication.advertisement(
                        engine.leader.messenger().instance_id(),
                        engine.leader.registration_epoch(),
                        engine
                            .bundle_publications
                            .now_unix_ms()
                            .saturating_add(BUNDLE_DIRECTORY_TTL_MS),
                    ) {
                        Ok(advertisement) => advertisement,
                        Err(error) => {
                            tracing::error!(%error, "committed bundle could not be advertised");
                            return;
                        }
                    };
                    engine.bundle_publications.track(
                        publication,
                        Arc::downgrade(&engine),
                        &engine.leader.runtime(),
                    );
                    if let Err(error) =
                        run_directory_call(directory.advertise_bundle(advertisement)).await
                    {
                        log_directory_call_error("publish", error);
                    }
                }
                BundleDirectoryUpdate::Invalidate { key, generation } => {
                    engine.bundle_publications.forget(&key, generation);
                    let Some(directory) = engine.leader.remote_discovery() else {
                        return;
                    };
                    let invalidation = BundleInvalidation {
                        key,
                        generation,
                        owner: engine.leader.messenger().instance_id(),
                        retain_until_unix_ms: engine
                            .bundle_publications
                            .now_unix_ms()
                            .saturating_add(BUNDLE_DIRECTORY_TTL_MS),
                    };
                    if let Err(error) =
                        run_directory_call(directory.invalidate_bundle(invalidation)).await
                    {
                        log_directory_call_error("invalidation", error);
                    }
                }
            }
        });
    }

    pub(in crate::tiering::engine::bundle::remote) async fn refresh_bundle_publication(
        &self,
        publication: &BundlePublication,
    ) {
        let _order = self.bundle_directory_order.enter(publication.key()).await;
        let is_current = self
            .bundle_catalog
            .lock()
            .expect("bundle-catalog mutex poisoned")
            .is_current(publication.key(), publication.generation());
        if !is_current {
            self.bundle_publications
                .forget(publication.key(), publication.generation());
            return;
        }
        let Some(directory) = self.leader.remote_discovery() else {
            return;
        };
        let advertisement = match publication.advertisement(
            self.leader.messenger().instance_id(),
            self.leader.registration_epoch(),
            self.bundle_publications
                .now_unix_ms()
                .saturating_add(BUNDLE_DIRECTORY_TTL_MS),
        ) {
            Ok(advertisement) => advertisement,
            Err(error) => {
                self.bundle_publications
                    .forget(publication.key(), publication.generation());
                tracing::error!(%error, "committed bundle could not be refreshed");
                return;
            }
        };
        if let Err(error) = run_directory_call(directory.advertise_bundle(advertisement)).await {
            log_directory_call_error("refresh", error);
        }
    }
}

fn log_directory_call_error(operation: &'static str, error: DirectoryCallError) {
    match error {
        DirectoryCallError::Failed(error) => {
            tracing::warn!(%error, operation, "bundle directory update failed");
        }
        DirectoryCallError::TimedOut => {
            tracing::warn!(operation, "bundle directory update timed out");
        }
    }
}

enum BundleDirectoryUpdate {
    Advertise {
        identity: CacheIdentity,
        key: BundleKey,
        generation: u64,
        lineages: Vec<BundleResourceLineage>,
    },
    Invalidate {
        key: BundleKey,
        generation: u64,
    },
}

impl BundleDirectoryUpdate {
    const fn key(&self) -> &BundleKey {
        match self {
            Self::Advertise { key, .. } => key,
            Self::Invalidate { key, .. } => key,
        }
    }
}
