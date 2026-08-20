// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic ownership, lineage, and generation retirement for complete bundles.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::{Duration, Instant};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{BundleKey, BundleResourceLineageError, CacheIdentity};

use super::{BundleIndex, BundleIndexError, BundleLease, BundleResourcePin, CommitDisposition};
use crate::remote::search::bundle::BundleDirectoryError;
use crate::tiering::policy::{BundleDependencyIndex, DependencyError, ResourceLineage};

pub(in crate::tiering::engine) const BUNDLE_DIRECTORY_TTL_MS: u64 = 300_000;

/// One lock-owned catalog containing every visibility and eviction invariant.
pub(in crate::tiering::engine) struct BundleCatalog<P: BundleResourcePin> {
    index: BundleIndex<P>,
    dependencies: BundleDependencyIndex,
    retired_generations: HashMap<BundleKey, RetiredGeneration>,
    retirement_expirations: VecDeque<(BundleKey, RetiredGeneration)>,
}

impl<P: BundleResourcePin> BundleCatalog<P> {
    pub(in crate::tiering::engine) fn new() -> Self {
        Self {
            index: BundleIndex::new(),
            dependencies: BundleDependencyIndex::new(),
            retired_generations: HashMap::new(),
            retirement_expirations: VecDeque::new(),
        }
    }

    /// Publish visibility and reverse lineage as one transaction. The caller
    /// keeps `resources` strongly pinned until this method releases the catalog
    /// lock, so an eviction callback cannot observe a half-published bundle.
    pub(in crate::tiering::engine) fn commit(
        &mut self,
        identity: &CacheIdentity,
        key: BundleKey,
        generation: u64,
        resources: &BTreeMap<LogicalResourceId, P>,
        lineages: Vec<ResourceLineage>,
    ) -> Result<(), BundleCatalogError> {
        self.prune_retired_generations();
        if let Some(retired) = self
            .retired_generations
            .get(&key)
            .filter(|retired| generation <= retired.generation)
        {
            return Err(BundleCatalogError::RetiredGeneration {
                retired: retired.generation,
                attempted: generation,
            });
        }

        let lineage_resources = lineages
            .iter()
            .map(ResourceLineage::resource)
            .collect::<BTreeSet<_>>();
        let pinned_resources = resources.keys().copied().collect::<BTreeSet<_>>();
        if lineage_resources != pinned_resources {
            return Err(BundleCatalogError::LineageResourceSetMismatch {
                pinned: pinned_resources.into_iter().collect(),
                lineage: lineage_resources.into_iter().collect(),
            });
        }

        let Some(bundle) = self
            .index
            .prepare_commit(identity, key, generation, resources)?
        else {
            return Ok(());
        };
        let dependencies = BundleDependencyIndex::prepare(key, lineages)?;

        self.dependencies.install(key, dependencies);
        self.index.install(key, bundle);
        Ok(())
    }

    /// Validate a remote transaction before making any staged physical block
    /// visible, then materialize and install it while the catalog lock remains
    /// held.
    pub(in crate::tiering::engine) fn commit_materialized<F>(
        &mut self,
        identity: &CacheIdentity,
        key: BundleKey,
        generation: u64,
        resource_ids: BTreeSet<LogicalResourceId>,
        lineages: Vec<ResourceLineage>,
        materialize: F,
    ) -> Result<(), BundleCatalogError>
    where
        F: FnOnce() -> Result<BTreeMap<LogicalResourceId, P>, BundleCatalogError>,
    {
        self.prune_retired_generations();
        if let Some(retired) = self
            .retired_generations
            .get(&key)
            .filter(|retired| generation <= retired.generation)
        {
            return Err(BundleCatalogError::RetiredGeneration {
                retired: retired.generation,
                attempted: generation,
            });
        }
        let lineage_resources = lineages
            .iter()
            .map(ResourceLineage::resource)
            .collect::<BTreeSet<_>>();
        if lineage_resources != resource_ids {
            return Err(BundleCatalogError::LineageResourceSetMismatch {
                pinned: resource_ids.into_iter().collect(),
                lineage: lineage_resources.into_iter().collect(),
            });
        }
        let dependencies = BundleDependencyIndex::prepare(key, lineages)?;
        if self
            .index
            .validate_commit(identity, key, generation, &resource_ids)?
            == CommitDisposition::Idempotent
        {
            return Ok(());
        }

        // No fallible catalog operation follows materialization. This closure
        // only registers already-staged CompleteBlocks. BlockManager emits
        // eviction callbacks during allocation, before catalog preflight, and
        // never from register_blocks.
        let resources = materialize()?;
        let bundle = BundleIndex::prepare_validated(generation, &resources);
        self.dependencies.install(key, dependencies);
        self.index.install(key, bundle);
        Ok(())
    }

    pub(in crate::tiering::engine) fn lease_exact(
        &self,
        identity: &CacheIdentity,
        key: &BundleKey,
    ) -> Option<BundleLease<P>> {
        self.index.lease_exact(identity, key)
    }

    pub(in crate::tiering::engine) fn index(&self) -> &BundleIndex<P> {
        &self.index
    }

    #[cfg(test)]
    pub(in crate::tiering::engine) fn index_mut(&mut self) -> &mut BundleIndex<P> {
        &mut self.index
    }

    pub(in crate::tiering::engine) fn is_current(&self, key: &BundleKey, generation: u64) -> bool {
        self.index.generation(key) == Some(generation)
    }

    pub(in crate::tiering::engine) fn invalidate_resource(
        &mut self,
        resource: LogicalResourceId,
        hashes: &[SequenceHash],
    ) -> Vec<InvalidatedBundle> {
        self.prune_retired_generations();
        let mut events = Vec::new();
        for &hash in hashes {
            self.dependencies
                .invalidate(resource, hash, |event| events.push(event));
        }

        events
            .into_iter()
            .filter_map(|event| {
                let generation = self.index.remove(event.key())?;
                self.retire_generation(event.key(), generation);
                Some(InvalidatedBundle {
                    key: event.key(),
                    generation,
                    resource: event.resource(),
                    hash: event.hash(),
                })
            })
            .collect()
    }

    #[cfg(test)]
    pub(in crate::tiering::engine) fn invalidate_key(&mut self, key: BundleKey) -> Option<u64> {
        self.dependencies.untrack(key);
        let generation = self.index.remove(key)?;
        self.retire_generation(key, generation);
        Some(generation)
    }

    #[cfg(test)]
    pub(in crate::tiering::engine) fn dependents(
        &self,
        resource: LogicalResourceId,
        hash: SequenceHash,
    ) -> Vec<BundleKey> {
        self.dependencies.dependents(resource, hash)
    }

    fn retire_generation(&mut self, key: BundleKey, generation: u64) {
        let generation = self
            .retired_generations
            .get(&key)
            .map_or(generation, |retired| retired.generation.max(generation));
        let retirement = RetiredGeneration {
            generation,
            expires_at: Instant::now() + Duration::from_millis(BUNDLE_DIRECTORY_TTL_MS),
        };
        self.retired_generations.insert(key, retirement);
        self.retirement_expirations.push_back((key, retirement));
    }

    fn prune_retired_generations(&mut self) {
        let now = Instant::now();
        while self
            .retirement_expirations
            .front()
            .is_some_and(|(_, retirement)| retirement.expires_at <= now)
        {
            let (key, expired) = self
                .retirement_expirations
                .pop_front()
                .expect("checked non-empty retirement queue");
            if self.retired_generations.get(&key) == Some(&expired) {
                self.retired_generations.remove(&key);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RetiredGeneration {
    generation: u64,
    expires_at: Instant,
}

/// Exact catalog removal emitted after a resource block is reclaimed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::tiering::engine) struct InvalidatedBundle {
    key: BundleKey,
    generation: u64,
    resource: LogicalResourceId,
    hash: SequenceHash,
}

impl InvalidatedBundle {
    pub(in crate::tiering::engine) const fn key(self) -> BundleKey {
        self.key
    }

    pub(in crate::tiering::engine) const fn generation(self) -> u64 {
        self.generation
    }

    pub(in crate::tiering::engine) const fn resource(self) -> LogicalResourceId {
        self.resource
    }

    pub(in crate::tiering::engine) const fn hash(self) -> SequenceHash {
        self.hash
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(in crate::tiering::engine) enum BundleCatalogError {
    #[error(transparent)]
    Index(#[from] BundleIndexError),
    #[error(transparent)]
    Dependency(#[from] DependencyError),
    #[error(transparent)]
    AdvertisementLineage(#[from] BundleResourceLineageError),
    #[error(transparent)]
    Advertisement(#[from] BundleDirectoryError),
    #[error("bundle pins {pinned:?} and lineages {lineage:?} cover different resources")]
    LineageResourceSetMismatch {
        pinned: Vec<LogicalResourceId>,
        lineage: Vec<LogicalResourceId>,
    },
    #[error("bundle generation {attempted} was already retired through generation {retired}")]
    RetiredGeneration { retired: u64, attempted: u64 },
    #[error("bundle materialization failed: {0}")]
    Materialization(String),
}

#[cfg(test)]
mod tests;
