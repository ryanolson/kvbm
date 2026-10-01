// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifest-scoped wrapper over Dynamo's [`LineageIndex`].
//!
//! Each manifest has a grow-only capacity measured in positions
//! (`max_seq_len / block_size`). Block/carrier kind binding and poisoning live
//! here; Dynamo's index owns PLH storage and holder operations.

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use dynamo_kv_router::indexer::lineage::LineageIndex;
use kvbm_logical::SequenceHash;
use kvbm_logical::events::{CreateKind, KvCacheEvents, KvbmCacheEvents};
use kvbm_protocols::cache_manifest::{CacheManifest, CacheManifestId, ResourceRole};
use parking_lot::RwLock;

use super::protocol::{ByPositionResponse, IndexEntry};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyOutcome {
    Applied,
    Unbound,
    KindMismatch,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BindError {
    #[error("carrier registration requires a BoundaryCapsule resource")]
    CarrierWithoutBoundaryCapsule,
    #[error("manifest {manifest} is already indexed as {existing:?}, not {requested:?}")]
    KindConflict {
        manifest: CacheManifestId,
        existing: CreateKind,
        requested: CreateKind,
    },
}

#[derive(Clone, Copy)]
struct IndexerBinding {
    manifest: CacheManifestId,
    kind: CreateKind,
    poisoned: bool,
}

struct ManifestIndex {
    kind: CreateKind,
    index: LineageIndex<u128>,
}

/// Manifest-scoped lineage indexes and the registered instance bindings.
pub struct ManifestIndexes {
    block_size: usize,
    initial_max_seq_len: usize,
    bindings: RwLock<HashMap<u128, IndexerBinding>>,
    indexes: DashMap<CacheManifestId, Arc<ManifestIndex>>,
}

impl ManifestIndexes {
    pub fn new(initial_max_seq_len: usize, block_size: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(block_size > 0, "block_size must be > 0");
        anyhow::ensure!(
            initial_max_seq_len.is_multiple_of(block_size),
            "max_seq_len ({initial_max_seq_len}) must be evenly divisible by block_size ({block_size})"
        );
        Ok(Self {
            block_size,
            initial_max_seq_len,
            bindings: RwLock::new(HashMap::new()),
            indexes: DashMap::new(),
        })
    }

    pub fn bind(
        &self,
        instance: u128,
        manifest: &CacheManifest,
        kind: CreateKind,
        max_seq_len: Option<usize>,
    ) -> Result<(), BindError> {
        if kind == CreateKind::Carrier
            && !manifest
                .resources()
                .iter()
                .any(|resource| resource.role() == ResourceRole::BoundaryCapsule)
        {
            return Err(BindError::CarrierWithoutBoundaryCapsule);
        }

        let manifest_id = manifest.id();
        let mut bindings = self.bindings.write();
        if let Some(index) = self.indexes.get(&manifest_id)
            && index.kind != kind
        {
            return Err(BindError::KindConflict {
                manifest: manifest_id,
                existing: index.kind,
                requested: kind,
            });
        }

        let previous = bindings.get(&instance).copied();
        if let Some(previous) = previous
            && previous.manifest != manifest_id
            && let Some(index) = self.indexes.get(&previous.manifest)
        {
            index.index.remove_holder(instance);
        }

        if !self.indexes.contains_key(&manifest_id) {
            let index = LineageIndex::new((self.initial_max_seq_len / self.block_size) as u64);
            self.indexes.insert(
                manifest_id,
                Arc::new(ManifestIndex {
                    kind,
                    index,
                }),
            );
        }
        if let Some(max_seq_len) = max_seq_len
            && let Some(index) = self.indexes.get(&manifest_id)
        {
            index
                .index
                .grow_to((max_seq_len / self.block_size) as u64);
        }

        bindings.insert(
            instance,
            IndexerBinding {
                manifest: manifest_id,
                kind,
                poisoned: false,
            },
        );
        if let Some(previous) = previous
            && previous.manifest != manifest_id
            && !bindings
                .values()
                .any(|binding| binding.manifest == previous.manifest)
        {
            self.indexes.remove(&previous.manifest);
        }
        Ok(())
    }

    pub fn unbind(&self, instance: u128) {
        let mut bindings = self.bindings.write();
        let Some(binding) = bindings.remove(&instance) else {
            return;
        };
        if let Some(index) = self.indexes.get(&binding.manifest) {
            index.index.remove_holder(instance);
        }
        if !bindings
            .values()
            .any(|other| other.manifest == binding.manifest)
        {
            self.indexes.remove(&binding.manifest);
        }
    }

    pub fn clear_instance(&self, instance: u128) {
        let bindings = self.bindings.read();
        let Some(binding) = bindings.get(&instance) else {
            return;
        };
        if let Some(index) = self.indexes.get(&binding.manifest) {
            index.index.remove_holder(instance);
        }
    }

    pub fn apply(&self, batch: KvbmCacheEvents) -> ApplyOutcome {
        let instance = batch.instance_id;
        let bindings = self.bindings.read();
        let Some(binding) = bindings.get(&instance).copied() else {
            return ApplyOutcome::Unbound;
        };
        if binding.poisoned {
            return ApplyOutcome::KindMismatch;
        }
        let requested_kind = match &batch.events {
            KvCacheEvents::Create(_) => Some(CreateKind::Block),
            KvCacheEvents::CarrierCreate(_) => Some(CreateKind::Carrier),
            KvCacheEvents::Snapshot { kind, .. } => Some(*kind),
            KvCacheEvents::Remove(_) | KvCacheEvents::Shutdown => None,
        };
        if requested_kind.is_some_and(|kind| kind != binding.kind) {
            drop(bindings);
            let mut bindings = self.bindings.write();
            let Some(current) = bindings.get_mut(&instance) else {
                return ApplyOutcome::Unbound;
            };
            if current.manifest != binding.manifest || current.kind != binding.kind {
                return ApplyOutcome::KindMismatch;
            }
            current.poisoned = true;
            if let Some(index) = self.indexes.get(&binding.manifest) {
                index.index.remove_holder(instance);
            }
            tracing::error!(
                instance,
                manifest = %binding.manifest,
                expected = ?binding.kind,
                actual = ?requested_kind,
                "indexer create kind mismatch; binding poisoned"
            );
            return ApplyOutcome::KindMismatch;
        }

        let Some(index) = self.indexes.get(&binding.manifest) else {
            return ApplyOutcome::Unbound;
        };
        match batch.events {
            KvCacheEvents::Create(hashes) | KvCacheEvents::CarrierCreate(hashes) => {
                index.index.insert(instance, &hashes);
            }
            KvCacheEvents::Remove(hashes) => {
                index.index.remove(instance, &hashes);
            }
            KvCacheEvents::Shutdown => index.index.remove_holder(instance),
            KvCacheEvents::Snapshot { hashes, .. } => {
                index.index.replace_holder(instance, &hashes);
            }
        }
        ApplyOutcome::Applied
    }

    pub fn query_holders(
        &self,
        manifest: CacheManifestId,
        hashes: &[SequenceHash],
    ) -> Option<(SequenceHash, Vec<u128>, CreateKind)> {
        let index = self.indexes.get(&manifest)?;
        index
            .index
            .deepest(hashes)
            .map(|hit| (hit.hash, hit.holders, index.kind))
    }

    pub fn query(&self, manifest: CacheManifestId, hashes: &[SequenceHash]) -> Option<IndexEntry> {
        self.query_holders(manifest, hashes)
            .map(|(hash, holders, _)| entry_of_ids(hash, holders))
    }

    pub fn by_position(&self, manifest: CacheManifestId, position: usize) -> ByPositionResponse {
        let Some(index) = self.indexes.get(&manifest) else {
            return ByPositionResponse {
                position,
                entries: Vec::new(),
            };
        };
        let entries = index
            .index
            .entries_at(position as u64)
            .into_iter()
            .map(|hit| entry_of_ids(hit.hash, hit.holders))
            .collect();
        ByPositionResponse { position, entries }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn max_seq_len(&self) -> usize {
        self.indexes
            .iter()
            .map(|index| index.index.max_positions() as usize * self.block_size)
            .max()
            .unwrap_or(self.initial_max_seq_len)
    }

    pub(super) fn bindings(&self) -> Vec<(u128, CacheManifestId, CreateKind)> {
        let mut bindings = self
            .bindings
            .read()
            .iter()
            .map(|(&instance, binding)| (instance, binding.manifest, binding.kind))
            .collect::<Vec<_>>();
        bindings.sort_unstable_by_key(|(instance, _, _)| *instance);
        bindings
    }
}

/// Builds a serializable [`IndexEntry`] from an already-sorted id list.
fn entry_of_ids(hash: SequenceHash, ids: Vec<u128>) -> IndexEntry {
    IndexEntry {
        hash: format!("{hash}"),
        hash_u128: hash.as_u128().to_string(),
        position: hash.position(),
        instances: ids.into_iter().map(|i| i.to_string()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_tokens::TokenBlockSequence;
    use kvbm_logical::KvbmSequenceHashProvider;

    fn manifest(seed: u8, boundary_capsule: bool) -> CacheManifest {
        let resources = if boundary_capsule {
            vec![
                kvbm_protocols::cache_manifest::ResourceRequirement::new(
                    kvbm_common::LogicalResourceId(1),
                    ResourceRole::BoundaryCapsule,
                    4,
                )
                .unwrap(),
            ]
        } else {
            vec![
                kvbm_protocols::cache_manifest::ResourceRequirement::new(
                    kvbm_common::LogicalResourceId(1),
                    ResourceRole::PrefixHistory,
                    4,
                )
                .unwrap(),
            ]
        };
        CacheManifest::new(
            kvbm_protocols::cache_manifest::ModelIdentity::new(
                "test-architecture",
                "test-revision",
                [seed; 32],
            )
            .unwrap(),
            "test-cache-abi",
            resources,
            std::collections::BTreeMap::new(),
        )
        .unwrap()
    }

    /// Builds `n` PLHs at positions 0..n for a given salt by laying down
    /// `n * block_size` tokens.
    fn plhs(block_size: u32, n: usize, salt: u64) -> Vec<SequenceHash> {
        let tokens: Vec<u32> = (0..(block_size as usize * n) as u32).collect();
        let seq = TokenBlockSequence::from_slice(&tokens, block_size, Some(salt));
        seq.blocks()
            .iter()
            .map(|b| b.kvbm_sequence_hash())
            .collect()
    }

    fn create(hashes: Vec<SequenceHash>, instance: u128) -> KvbmCacheEvents {
        KvbmCacheEvents {
            events: KvCacheEvents::Create(hashes),
            instance_id: instance,
        }
    }

    #[test]
    fn new_rejects_non_divisible() {
        assert!(ManifestIndexes::new(10, 4).is_err());
        assert!(ManifestIndexes::new(10, 0).is_err());
        assert!(ManifestIndexes::new(16, 4).is_ok());
    }

    #[test]
    fn position_bucketing_and_by_position() {
        let indexes = ManifestIndexes::new(16, 4).unwrap();
        let manifest = manifest(1, false);
        indexes
            .bind(100, &manifest, CreateKind::Block, None)
            .unwrap();
        assert_eq!(indexes.max_seq_len(), 16);
        let hashes = plhs(4, 3, 1337);
        assert_eq!(
            indexes.apply(create(hashes.clone(), 100)),
            ApplyOutcome::Applied
        );

        // Each PLH lands in its own positional bucket.
        for (pos, h) in hashes.iter().enumerate() {
            assert_eq!(h.position() as usize, pos);
            let resp = indexes.by_position(manifest.id(), pos);
            assert_eq!(resp.position, pos);
            assert_eq!(resp.entries.len(), 1);
            assert_eq!(resp.entries[0].instances, vec!["100".to_string()]);
            assert_eq!(resp.entries[0].hash_u128, h.as_u128().to_string());
        }
        // Empty / out-of-range buckets.
        assert!(indexes.by_position(manifest.id(), 3).entries.is_empty());
        assert!(indexes.by_position(manifest.id(), 99).entries.is_empty());
    }

    #[test]
    fn shared_prefix_lists_both_instances() {
        let indexes = ManifestIndexes::new(16, 4).unwrap();
        let manifest = manifest(2, false);
        indexes
            .bind(1, &manifest, CreateKind::Block, None)
            .unwrap();
        indexes
            .bind(2, &manifest, CreateKind::Block, None)
            .unwrap();
        let hashes = plhs(4, 2, 1337);
        indexes.apply(create(hashes.clone(), 1));
        indexes.apply(create(hashes, 2));

        let resp = indexes.by_position(manifest.id(), 0);
        assert_eq!(resp.entries.len(), 1);
        assert_eq!(
            resp.entries[0].instances,
            vec!["1".to_string(), "2".to_string()]
        );
    }

    #[test]
    fn query_returns_deepest_match() {
        let indexes = ManifestIndexes::new(64, 4).unwrap();
        let manifest = manifest(3, false);
        indexes
            .bind(7, &manifest, CreateKind::Block, None)
            .unwrap();
        // instance 7 holds a 3-deep sequence.
        let hashes = plhs(4, 3, 42);
        indexes.apply(create(hashes.clone(), 7));

        // Query with the full sequence → deepest (position 2).
        let hit = indexes.query(manifest.id(), &hashes).expect("hit");
        assert_eq!(hit.position, 2);
        assert_eq!(hit.instances, vec!["7".to_string()]);

        // Unsorted input still yields the deepest present.
        let mut shuffled = hashes.clone();
        shuffled.reverse();
        assert_eq!(indexes.query(manifest.id(), &shuffled).unwrap().position, 2);

        // A query whose deep blocks are unknown falls back to the shallow hit.
        let unknown = plhs(4, 5, 999);
        let mut mixed = vec![hashes[0], hashes[1]];
        mixed.extend_from_slice(&unknown[2..]); // positions 2..4 unknown
        let hit = indexes.query(manifest.id(), &mixed).expect("shallow hit");
        assert_eq!(hit.position, 1);
    }

    #[test]
    fn query_miss_returns_none() {
        let indexes = ManifestIndexes::new(16, 4).unwrap();
        let manifest = manifest(4, false);
        indexes
            .bind(1, &manifest, CreateKind::Block, None)
            .unwrap();
        let hashes = plhs(4, 2, 1);
        assert!(indexes.query(manifest.id(), &hashes).is_none());
    }

    #[test]
    fn query_holders_returns_deepest_hash_and_sorted_ids() {
        let indexes = ManifestIndexes::new(64, 4).unwrap();
        let manifest = manifest(5, false);
        indexes
            .bind(9, &manifest, CreateKind::Block, None)
            .unwrap();
        indexes
            .bind(2, &manifest, CreateKind::Block, None)
            .unwrap();
        let hashes = plhs(4, 3, 42);
        // Two holders of the shared 3-deep sequence; insert ids out of order.
        indexes.apply(create(hashes.clone(), 9));
        indexes.apply(create(hashes.clone(), 2));

        let (matched, ids, kind) = indexes
            .query_holders(manifest.id(), &hashes)
            .expect("hit");
        assert_eq!(matched, hashes[2], "deepest candidate hash");
        assert_eq!(ids, vec![2u128, 9u128], "holder ids, sorted");
        assert_eq!(kind, CreateKind::Block);

        // Mirrors `query()`: same deepest block, stringified.
        let entry = indexes.query(manifest.id(), &hashes).expect("hit");
        assert_eq!(entry.hash_u128, hashes[2].as_u128().to_string());
        assert_eq!(entry.instances, vec!["2".to_string(), "9".to_string()]);

        // Full miss.
        assert!(
            indexes
                .query_holders(manifest.id(), &plhs(4, 2, 999))
                .is_none()
        );
    }

    #[test]
    fn remove_prunes_entry_when_last_holder_leaves() {
        let indexes = ManifestIndexes::new(16, 4).unwrap();
        let manifest = manifest(6, false);
        indexes
            .bind(1, &manifest, CreateKind::Block, None)
            .unwrap();
        indexes
            .bind(2, &manifest, CreateKind::Block, None)
            .unwrap();
        let hashes = plhs(4, 1, 5);
        indexes.apply(create(hashes.clone(), 1));
        indexes.apply(create(hashes.clone(), 2));

        indexes.apply(KvbmCacheEvents {
            events: KvCacheEvents::Remove(hashes.clone()),
            instance_id: 1,
        });
        assert_eq!(
            indexes.by_position(manifest.id(), 0).entries[0].instances,
            vec!["2".to_string()]
        );

        indexes.apply(KvbmCacheEvents {
            events: KvCacheEvents::Remove(hashes.clone()),
            instance_id: 2,
        });
        assert!(indexes.by_position(manifest.id(), 0).entries.is_empty());
    }

    #[test]
    fn remove_instance_sweeps_all_positions() {
        let indexes = ManifestIndexes::new(16, 4).unwrap();
        let manifest = manifest(7, false);
        indexes
            .bind(1, &manifest, CreateKind::Block, None)
            .unwrap();
        indexes
            .bind(2, &manifest, CreateKind::Block, None)
            .unwrap();
        let hashes = plhs(4, 3, 5);
        indexes.apply(create(hashes.clone(), 1));
        indexes.apply(create(hashes, 2));
        indexes.apply(KvbmCacheEvents {
            events: KvCacheEvents::Shutdown,
            instance_id: 1,
        });
        for pos in 0..3 {
            assert_eq!(
                indexes.by_position(manifest.id(), pos).entries[0].instances,
                vec!["2".to_string()]
            );
        }
        indexes.apply(KvbmCacheEvents {
            events: KvCacheEvents::Shutdown,
            instance_id: 2,
        });
        for pos in 0..3 {
            assert!(indexes.by_position(manifest.id(), pos).entries.is_empty());
        }
    }

    #[test]
    fn out_of_range_create_is_dropped() {
        let indexes = ManifestIndexes::new(8, 4).unwrap(); // 2 positions: 0,1
        let manifest = manifest(8, false);
        indexes
            .bind(1, &manifest, CreateKind::Block, None)
            .unwrap();
        let hashes = plhs(4, 4, 9); // positions 0..3
        indexes.apply(create(hashes, 1));
        assert_eq!(indexes.by_position(manifest.id(), 0).entries.len(), 1);
        assert_eq!(indexes.by_position(manifest.id(), 1).entries.len(), 1);
        assert!(indexes.by_position(manifest.id(), 2).entries.is_empty());
        assert!(indexes.by_position(manifest.id(), 3).entries.is_empty());
    }

    #[test]
    fn grow_raises_capacity_and_never_lowers() {
        let indexes = ManifestIndexes::new(8, 4).unwrap(); // 2 positions
        let index_manifest = manifest(9, false);
        indexes
            .bind(1, &index_manifest, CreateKind::Block, Some(8))
            .unwrap();
        assert_eq!(indexes.max_seq_len(), 8);

        let hashes = plhs(4, 4, 9); // positions 0..3
        indexes.apply(create(hashes.clone(), 1));
        assert!(
            indexes
                .by_position(index_manifest.id(), 2)
                .entries
                .is_empty()
        );
        assert!(
            indexes
                .by_position(index_manifest.id(), 3)
                .entries
                .is_empty()
        );

        // Grow to fit 16 tokens → 4 positions; deeper creates now land.
        indexes
            .bind(1, &index_manifest, CreateKind::Block, Some(16))
            .unwrap();
        assert_eq!(indexes.max_seq_len(), 16);
        indexes.apply(create(hashes.clone(), 1));
        assert_eq!(
            indexes
                .by_position(index_manifest.id(), 3)
                .entries
                .len(),
            1
        );

        // A smaller (or equal) max_seq_len never shrinks capacity.
        indexes
            .bind(1, &index_manifest, CreateKind::Block, Some(8))
            .unwrap();
        assert_eq!(indexes.max_seq_len(), 16);

        // Starting from an empty index (max_seq_len 0) grows on demand.
        let empty = ManifestIndexes::new(0, 4).unwrap();
        let empty_manifest = manifest(10, false);
        empty
            .bind(2, &empty_manifest, CreateKind::Block, None)
            .unwrap();
        assert_eq!(empty.max_seq_len(), 0);
        empty.apply(create(hashes.clone(), 2));
        assert!(empty.by_position(empty_manifest.id(), 0).entries.is_empty());
        empty
            .bind(2, &empty_manifest, CreateKind::Block, Some(16))
            .unwrap();
        assert_eq!(empty.max_seq_len(), 16);
        empty.apply(create(hashes, 2));
        assert_eq!(
            empty
                .by_position(empty_manifest.id(), 3)
                .entries
                .len(),
            1
        );
    }

    #[test]
    fn identical_hashes_are_isolated_by_manifest() {
        let indexes = ManifestIndexes::new(32, 4).unwrap();
        let manifest_a = manifest(1, false);
        let manifest_b = manifest(2, false);
        let hash = plhs(4, 1, 17)[0];
        indexes
            .bind(10, &manifest_a, CreateKind::Block, None)
            .unwrap();
        indexes
            .bind(20, &manifest_b, CreateKind::Block, None)
            .unwrap();
        assert_eq!(indexes.apply(create(vec![hash], 10)), ApplyOutcome::Applied);
        assert_eq!(indexes.apply(create(vec![hash], 20)), ApplyOutcome::Applied);

        assert_eq!(
            indexes.query_holders(manifest_a.id(), &[hash]).unwrap().1,
            vec![10]
        );
        assert_eq!(
            indexes.query_holders(manifest_b.id(), &[hash]).unwrap().1,
            vec![20]
        );
        assert!(indexes.query(manifest(3, false).id(), &[hash]).is_none());
    }

    #[test]
    fn kind_mismatch_clears_and_poisons_until_rebinding() {
        let indexes = ManifestIndexes::new(32, 4).unwrap();
        let manifest = manifest(4, false);
        let hash = plhs(4, 1, 21)[0];
        indexes
            .bind(30, &manifest, CreateKind::Block, None)
            .unwrap();
        assert_eq!(indexes.apply(create(vec![hash], 30)), ApplyOutcome::Applied);
        assert_eq!(
            indexes.apply(KvbmCacheEvents {
                events: KvCacheEvents::CarrierCreate(vec![hash]),
                instance_id: 30,
            }),
            ApplyOutcome::KindMismatch
        );
        assert!(indexes.query(manifest.id(), &[hash]).is_none());
        assert_eq!(
            indexes.apply(create(vec![hash], 30)),
            ApplyOutcome::KindMismatch
        );

        indexes
            .bind(30, &manifest, CreateKind::Block, None)
            .unwrap();
        assert_eq!(indexes.apply(create(vec![hash], 30)), ApplyOutcome::Applied);
        assert_eq!(
            indexes.query_holders(manifest.id(), &[hash]).unwrap().1,
            vec![30]
        );
    }

    #[test]
    fn carrier_requires_boundary_capsule_and_manifest_kind_is_fixed() {
        let indexes = ManifestIndexes::new(32, 4).unwrap();
        let no_capsule = manifest(5, false);
        assert_eq!(
            indexes.bind(40, &no_capsule, CreateKind::Carrier, None),
            Err(BindError::CarrierWithoutBoundaryCapsule)
        );

        let capsule = manifest(6, true);
        indexes.bind(41, &capsule, CreateKind::Block, None).unwrap();
        assert!(matches!(
            indexes.bind(42, &capsule, CreateKind::Carrier, None),
            Err(BindError::KindConflict {
                existing: CreateKind::Block,
                requested: CreateKind::Carrier,
                ..
            })
        ));

        let carrier_manifest = manifest(10, true);
        indexes
            .bind(43, &carrier_manifest, CreateKind::Carrier, None)
            .unwrap();
        let hash = plhs(4, 1, 71)[0];
        assert_eq!(
            indexes.apply(KvbmCacheEvents {
                events: KvCacheEvents::CarrierCreate(vec![hash]),
                instance_id: 43,
            }),
            ApplyOutcome::Applied
        );
        assert_eq!(
            indexes
                .query_holders(carrier_manifest.id(), &[hash])
                .unwrap()
                .2,
            CreateKind::Carrier
        );
        assert_eq!(
            indexes.apply(KvbmCacheEvents {
                events: KvCacheEvents::Create(vec![hash]),
                instance_id: 43,
            }),
            ApplyOutcome::KindMismatch
        );
        assert!(indexes.query(carrier_manifest.id(), &[hash]).is_none());
        assert_eq!(
            indexes.apply(KvbmCacheEvents {
                events: KvCacheEvents::CarrierCreate(vec![hash]),
                instance_id: 43,
            }),
            ApplyOutcome::KindMismatch
        );
        indexes
            .bind(43, &carrier_manifest, CreateKind::Carrier, None)
            .unwrap();
        assert_eq!(
            indexes.apply(KvbmCacheEvents {
                events: KvCacheEvents::CarrierCreate(vec![hash]),
                instance_id: 43,
            }),
            ApplyOutcome::Applied
        );
    }

    #[test]
    fn snapshot_replaces_instance_hashes_and_remove_keeps_event_order() {
        let indexes = ManifestIndexes::new(32, 4).unwrap();
        let manifest = manifest(7, false);
        indexes
            .bind(50, &manifest, CreateKind::Block, None)
            .unwrap();
        let hashes = plhs(4, 3, 44);
        let shallow = hashes[0];
        let deep = hashes[2];
        assert_eq!(indexes.apply(create(vec![deep], 50)), ApplyOutcome::Applied);
        assert_eq!(
            indexes.apply(KvbmCacheEvents {
                events: KvCacheEvents::Snapshot {
                    kind: CreateKind::Block,
                    hashes: vec![shallow],
                },
                instance_id: 50,
            }),
            ApplyOutcome::Applied
        );
        assert!(indexes.query(manifest.id(), &[deep]).is_none());
        assert_eq!(
            indexes
                .query(manifest.id(), &[deep, shallow])
                .unwrap()
                .hash_u128,
            shallow.as_u128().to_string()
        );

        let new_hash = plhs(4, 1, 45)[0];
        assert_eq!(
            indexes.apply(create(vec![new_hash], 50)),
            ApplyOutcome::Applied
        );
        assert_eq!(
            indexes.apply(create(vec![new_hash], 50)),
            ApplyOutcome::Applied
        );
        assert_eq!(
            indexes.apply(KvbmCacheEvents {
                events: KvCacheEvents::Remove(vec![new_hash]),
                instance_id: 50,
            }),
            ApplyOutcome::Applied
        );
        assert!(indexes.query(manifest.id(), &[new_hash]).is_none());
    }

    #[test]
    fn rebinding_migrates_and_unbinding_drops_empty_indexes() {
        let indexes = ManifestIndexes::new(32, 4).unwrap();
        let manifest_a = manifest(8, false);
        let manifest_b = manifest(9, false);
        let hash = plhs(4, 1, 51)[0];
        indexes
            .bind(60, &manifest_a, CreateKind::Block, None)
            .unwrap();
        assert_eq!(indexes.apply(create(vec![hash], 60)), ApplyOutcome::Applied);

        indexes
            .bind(60, &manifest_b, CreateKind::Block, None)
            .unwrap();
        assert!(indexes.query(manifest_a.id(), &[hash]).is_none());
        assert!(indexes.query(manifest_b.id(), &[hash]).is_none());
        assert_eq!(indexes.apply(create(vec![hash], 60)), ApplyOutcome::Applied);
        assert_eq!(indexes.bindings().len(), 1);
        indexes.unbind(60);
        assert!(indexes.query(manifest_b.id(), &[hash]).is_none());
        assert!(indexes.bindings().is_empty());
    }
}
