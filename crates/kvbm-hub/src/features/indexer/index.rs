// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifest-scoped position-bucketed block index.
//!
//! Buckets are a sparse, **grow-only** `DashMap<position, …>`; the capacity
//! (`max_positions = max_seq_len / block_size`) starts from the hub's optional
//! `max_seq_len` and is raised — never lowered — by
//! [`grow_to_max_seq_len`](PositionalIndex::grow_to_max_seq_len), called when a
//! registrant reports a larger `max_seq_len`. Each bucket maps a
//! [`SequenceHash`] (a self-describing PLH) to the set of worker `instance_id`s
//! holding that block. The PLH carries its own `position()`, so ingest needs no
//! out-of-band position data — and queries resolve by walking the candidate
//! hashes and returning the **deepest** one present (PLH lineage guarantees
//! holders of a deep block also hold its ancestors). A create whose position is
//! `>= max_positions` is dropped (bounded against malformed input).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use dashmap::DashMap;
use kvbm_logical::SequenceHash;
use kvbm_logical::events::{CreateKind, KvCacheEvents, KvbmCacheEvents};
use kvbm_protocols::cache_manifest::{CacheManifest, CacheManifestId, ResourceRole};
use parking_lot::RwLock;

use super::protocol::{ByPositionResponse, IndexEntry};

/// One position's block-hash → holding-instances map.
type Bucket = Arc<DashMap<SequenceHash, HashSet<u128>>>;

/// Position-bucketed map of block hash → holding instances.
pub(super) struct PositionalIndex {
    /// Sparse position → bucket map. Buckets are created on demand; missing
    /// positions are simply empty.
    buckets: DashMap<usize, Bucket>,
    block_size: usize,
    /// Current capacity in positions (`max_seq_len / block_size`). Grow-only.
    max_positions: AtomicUsize,
    /// Count of create events whose position exceeded the current capacity.
    dropped_out_of_range: AtomicU64,
}

impl PositionalIndex {
    /// Builds an index with initial capacity for `max_seq_len` tokens at
    /// `block_size` tokens per block. Requires `block_size > 0` and
    /// `max_seq_len % block_size == 0` (`max_seq_len == 0` is allowed — the
    /// index starts empty and grows as registrants report their `max_seq_len`).
    fn new(max_seq_len: usize, block_size: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(block_size > 0, "block_size must be > 0");
        anyhow::ensure!(
            max_seq_len.is_multiple_of(block_size),
            "max_seq_len ({max_seq_len}) must be evenly divisible by block_size ({block_size})"
        );
        Ok(Self {
            buckets: DashMap::new(),
            block_size,
            max_positions: AtomicUsize::new(max_seq_len / block_size),
            dropped_out_of_range: AtomicU64::new(0),
        })
    }

    /// Current number of position buckets (`max_seq_len / block_size`).
    fn num_positions(&self) -> usize {
        self.max_positions.load(Ordering::Relaxed)
    }

    /// Block size (tokens per block) the index was built for.
    #[cfg(test)]
    fn block_size(&self) -> usize {
        self.block_size
    }

    /// Current maximum sequence length (tokens) the index can hold.
    fn max_seq_len(&self) -> usize {
        self.num_positions() * self.block_size
    }

    /// Raise capacity to fit `max_seq_len` tokens. Never lowers it. Called when
    /// a KV-index registrant reports its `max_seq_len` (floored to a whole
    /// number of blocks).
    fn grow_to_max_seq_len(&self, max_seq_len: usize) {
        self.max_positions
            .fetch_max(max_seq_len / self.block_size, Ordering::Relaxed);
    }

    /// Count of create events dropped because their position exceeded the
    /// current capacity.
    #[cfg(test)]
    fn dropped_out_of_range(&self) -> u64 {
        self.dropped_out_of_range.load(Ordering::Relaxed)
    }

    /// Applies one wire batch to the index.
    #[cfg(test)]
    fn apply(&self, batch: KvbmCacheEvents) {
        let instance = batch.instance_id;
        match batch.events {
            KvCacheEvents::Create(hashes) | KvCacheEvents::CarrierCreate(hashes) => {
                for h in hashes {
                    self.insert(h, instance);
                }
            }
            KvCacheEvents::Remove(hashes) => {
                for h in hashes {
                    self.remove(h, instance);
                }
            }
            KvCacheEvents::Shutdown => self.remove_instance(instance),
            KvCacheEvents::Snapshot { hashes, .. } => {
                self.remove_instance(instance);
                for hash in hashes {
                    self.insert(hash, instance);
                }
            }
        }
    }

    /// Clone the bucket `Arc` at `pos` (if any), dropping the outer guard so
    /// inner ops don't hold the outer shard lock.
    fn bucket(&self, pos: usize) -> Option<Bucket> {
        self.buckets.get(&pos).map(|b| Arc::clone(&b))
    }

    fn insert(&self, hash: SequenceHash, instance: u128) {
        let pos = hash.position() as usize;
        if pos >= self.max_positions.load(Ordering::Relaxed) {
            self.dropped_out_of_range.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let bucket = self.buckets.entry(pos).or_default().clone();
        bucket.entry(hash).or_default().insert(instance);
    }

    fn remove(&self, hash: SequenceHash, instance: u128) {
        let Some(bucket) = self.bucket(hash.position() as usize) else {
            return;
        };
        let now_empty = match bucket.get_mut(&hash) {
            Some(mut set) => {
                set.remove(&instance);
                set.is_empty()
            }
            None => false,
        };
        if now_empty {
            bucket.remove_if(&hash, |_, set| set.is_empty());
        }
    }

    /// Removes `instance` from every bucket (used for `Shutdown` and
    /// registry eviction). Empty entries are pruned.
    fn remove_instance(&self, instance: u128) {
        // Snapshot bucket Arcs so we don't hold the outer guard while mutating
        // inner maps.
        let buckets: Vec<Bucket> = self.buckets.iter().map(|b| Arc::clone(&b)).collect();
        for bucket in buckets {
            bucket.retain(|_, set| {
                set.remove(&instance);
                !set.is_empty()
            });
        }
    }

    /// Resolves a candidate block sequence to the deepest indexed block's hash
    /// and the (sorted) raw `u128` ids of the instances holding it.
    ///
    /// Returns the hash with the greatest `position()` among supplied hashes
    /// that are currently held by at least one instance, paired with its holder
    /// ids. Input order does not matter. This is the typed core shared by both
    /// the HTTP [`query`](Self::query) path (which stringifies into an
    /// [`IndexEntry`]) and the velo lookup handler (which keeps the types).
    fn query_holders(&self, hashes: &[SequenceHash]) -> Option<(SequenceHash, Vec<u128>)> {
        let mut best: Option<(SequenceHash, Vec<u128>)> = None;
        for hash in hashes {
            let pos = hash.position();
            let Some(bucket) = self.bucket(pos as usize) else {
                continue;
            };
            let Some(set) = bucket.get(hash) else {
                continue;
            };
            if set.is_empty() {
                continue;
            }
            if best.as_ref().is_none_or(|(h, _)| pos > h.position()) {
                let mut ids: Vec<u128> = set.iter().copied().collect();
                ids.sort_unstable();
                best = Some((*hash, ids));
            }
        }
        best
    }

    /// Resolves a candidate block sequence to the deepest indexed block.
    ///
    /// Returns the entry with the greatest `position()` among supplied hashes
    /// that are currently held by at least one instance. Input order does not
    /// matter.
    #[cfg(test)]
    fn query(&self, hashes: &[SequenceHash]) -> Option<IndexEntry> {
        self.query_holders(hashes)
            .map(|(hash, ids)| entry_of_ids(hash, ids))
    }

    /// Dumps the index bucket at `position`. Out-of-range positions yield an
    /// empty entry list.
    fn by_position(&self, position: usize) -> ByPositionResponse {
        let entries = match self.bucket(position) {
            Some(bucket) => bucket
                .iter()
                .map(|kv| entry_of(*kv.key(), kv.value()))
                .collect(),
            None => Vec::new(),
        };
        ByPositionResponse { position, entries }
    }
}

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
    index: Arc<PositionalIndex>,
    swap: RwLock<()>,
}

/// Manifest-scoped positional indexes and the registered instance bindings.
pub struct ManifestIndexes {
    block_size: usize,
    initial_max_seq_len: usize,
    bindings: RwLock<HashMap<u128, IndexerBinding>>,
    indexes: DashMap<CacheManifestId, Arc<ManifestIndex>>,
}

impl ManifestIndexes {
    pub fn new(initial_max_seq_len: usize, block_size: usize) -> anyhow::Result<Self> {
        PositionalIndex::new(initial_max_seq_len, block_size)?;
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
        {
            if let Some(index) = self.indexes.get(&previous.manifest) {
                let _swap = index.swap.write();
                index.index.remove_instance(instance);
            }
        }

        if !self.indexes.contains_key(&manifest_id) {
            let index = Arc::new(
                PositionalIndex::new(self.initial_max_seq_len, self.block_size)
                    .expect("manifest index configuration was validated at construction"),
            );
            self.indexes.insert(
                manifest_id,
                Arc::new(ManifestIndex {
                    kind,
                    index,
                    swap: RwLock::new(()),
                }),
            );
        }
        if let Some(max_seq_len) = max_seq_len
            && let Some(index) = self.indexes.get(&manifest_id)
        {
            index.index.grow_to_max_seq_len(max_seq_len);
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
            let _swap = index.swap.write();
            index.index.remove_instance(instance);
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
            let _swap = index.swap.write();
            index.index.remove_instance(instance);
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
                let _swap = index.swap.write();
                index.index.remove_instance(instance);
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
                let _swap = index.swap.read();
                for hash in hashes {
                    index.index.insert(hash, instance);
                }
            }
            KvCacheEvents::Remove(hashes) => {
                let _swap = index.swap.read();
                for hash in hashes {
                    index.index.remove(hash, instance);
                }
            }
            KvCacheEvents::Shutdown => {
                let _swap = index.swap.read();
                index.index.remove_instance(instance);
            }
            KvCacheEvents::Snapshot { hashes, .. } => {
                let _swap = index.swap.write();
                index.index.remove_instance(instance);
                for hash in hashes {
                    index.index.insert(hash, instance);
                }
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
        let _swap = index.swap.read();
        index
            .index
            .query_holders(hashes)
            .map(|(hash, holders)| (hash, holders, index.kind))
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
        let _swap = index.swap.read();
        index.index.by_position(position)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn max_seq_len(&self) -> usize {
        self.indexes
            .iter()
            .map(|index| index.index.max_seq_len())
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

/// Builds a serializable [`IndexEntry`] with deterministically sorted
/// instance ids.
fn entry_of(hash: SequenceHash, instances: &HashSet<u128>) -> IndexEntry {
    let mut ids: Vec<u128> = instances.iter().copied().collect();
    ids.sort_unstable();
    entry_of_ids(hash, ids)
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
        assert!(PositionalIndex::new(10, 4).is_err());
        assert!(PositionalIndex::new(10, 0).is_err());
        assert!(PositionalIndex::new(16, 4).is_ok());
    }

    #[test]
    fn position_bucketing_and_by_position() {
        let idx = PositionalIndex::new(16, 4).unwrap();
        assert_eq!(idx.num_positions(), 4);
        let hashes = plhs(4, 3, 1337);
        idx.apply(create(hashes.clone(), 100));

        // Each PLH lands in its own positional bucket.
        for (pos, h) in hashes.iter().enumerate() {
            assert_eq!(h.position() as usize, pos);
            let resp = idx.by_position(pos);
            assert_eq!(resp.entries.len(), 1);
            assert_eq!(resp.entries[0].instances, vec!["100".to_string()]);
            assert_eq!(resp.entries[0].hash_u128, h.as_u128().to_string());
        }
        // Empty / out-of-range buckets.
        assert!(idx.by_position(3).entries.is_empty());
        assert!(idx.by_position(99).entries.is_empty());
    }

    #[test]
    fn shared_prefix_lists_both_instances() {
        let idx = PositionalIndex::new(16, 4).unwrap();
        let hashes = plhs(4, 2, 1337);
        idx.apply(create(hashes.clone(), 1));
        idx.apply(create(hashes.clone(), 2));

        let resp = idx.by_position(0);
        assert_eq!(resp.entries.len(), 1);
        assert_eq!(
            resp.entries[0].instances,
            vec!["1".to_string(), "2".to_string()]
        );
    }

    #[test]
    fn query_returns_deepest_match() {
        let idx = PositionalIndex::new(64, 4).unwrap();
        // instance 7 holds a 3-deep sequence.
        let hashes = plhs(4, 3, 42);
        idx.apply(create(hashes.clone(), 7));

        // Query with the full sequence → deepest (position 2).
        let hit = idx.query(&hashes).expect("hit");
        assert_eq!(hit.position, 2);
        assert_eq!(hit.instances, vec!["7".to_string()]);

        // Unsorted input still yields the deepest present.
        let mut shuffled = hashes.clone();
        shuffled.reverse();
        assert_eq!(idx.query(&shuffled).unwrap().position, 2);

        // A query whose deep blocks are unknown falls back to the shallow hit.
        let unknown = plhs(4, 5, 999);
        let mut mixed = vec![hashes[0], hashes[1]];
        mixed.extend_from_slice(&unknown[2..]); // positions 2..4 unknown
        let hit = idx.query(&mixed).expect("shallow hit");
        assert_eq!(hit.position, 1);
    }

    #[test]
    fn query_miss_returns_none() {
        let idx = PositionalIndex::new(16, 4).unwrap();
        let hashes = plhs(4, 2, 1);
        assert!(idx.query(&hashes).is_none());
    }

    #[test]
    fn query_holders_returns_deepest_hash_and_sorted_ids() {
        let idx = PositionalIndex::new(64, 4).unwrap();
        let hashes = plhs(4, 3, 42);
        // Two holders of the shared 3-deep sequence; insert ids out of order.
        idx.apply(create(hashes.clone(), 9));
        idx.apply(create(hashes.clone(), 2));

        let (matched, ids) = idx.query_holders(&hashes).expect("hit");
        assert_eq!(matched, hashes[2], "deepest candidate hash");
        assert_eq!(ids, vec![2u128, 9u128], "holder ids, sorted");

        // Mirrors `query()`: same deepest block, stringified.
        let entry = idx.query(&hashes).expect("hit");
        assert_eq!(entry.hash_u128, hashes[2].as_u128().to_string());
        assert_eq!(entry.instances, vec!["2".to_string(), "9".to_string()]);

        // Full miss.
        assert!(idx.query_holders(&plhs(4, 2, 999)).is_none());
    }

    #[test]
    fn remove_prunes_entry_when_last_holder_leaves() {
        let idx = PositionalIndex::new(16, 4).unwrap();
        let hashes = plhs(4, 1, 5);
        idx.apply(create(hashes.clone(), 1));
        idx.apply(create(hashes.clone(), 2));

        idx.apply(KvbmCacheEvents {
            events: KvCacheEvents::Remove(hashes.clone()),
            instance_id: 1,
        });
        assert_eq!(
            idx.by_position(0).entries[0].instances,
            vec!["2".to_string()]
        );

        idx.apply(KvbmCacheEvents {
            events: KvCacheEvents::Remove(hashes.clone()),
            instance_id: 2,
        });
        assert!(idx.by_position(0).entries.is_empty());
    }

    #[test]
    fn remove_instance_sweeps_all_positions() {
        let idx = PositionalIndex::new(16, 4).unwrap();
        let hashes = plhs(4, 3, 5);
        idx.apply(create(hashes.clone(), 1));
        idx.apply(create(hashes.clone(), 2));
        idx.remove_instance(1);
        for pos in 0..3 {
            assert_eq!(
                idx.by_position(pos).entries[0].instances,
                vec!["2".to_string()]
            );
        }
        idx.remove_instance(2);
        for pos in 0..3 {
            assert!(idx.by_position(pos).entries.is_empty());
        }
    }

    #[test]
    fn out_of_range_create_is_dropped_with_counter() {
        let idx = PositionalIndex::new(8, 4).unwrap(); // 2 positions: 0,1
        let hashes = plhs(4, 4, 9); // positions 0..3
        idx.apply(create(hashes, 1));
        assert_eq!(idx.dropped_out_of_range(), 2); // positions 2,3 dropped
        assert_eq!(idx.by_position(0).entries.len(), 1);
        assert_eq!(idx.by_position(1).entries.len(), 1);
    }

    #[test]
    fn grow_raises_capacity_and_never_lowers() {
        let idx = PositionalIndex::new(8, 4).unwrap(); // 2 positions
        assert_eq!(idx.num_positions(), 2);

        // Grow to fit 16 tokens → 4 positions; deeper creates now land.
        idx.grow_to_max_seq_len(16);
        assert_eq!(idx.num_positions(), 4);
        assert_eq!(idx.max_seq_len(), 16);
        let hashes = plhs(4, 4, 9); // positions 0..3
        idx.apply(create(hashes.clone(), 1));
        assert_eq!(idx.dropped_out_of_range(), 0);
        assert_eq!(idx.by_position(3).entries.len(), 1);

        // A smaller (or equal) max_seq_len never shrinks capacity.
        idx.grow_to_max_seq_len(8);
        assert_eq!(idx.num_positions(), 4);

        // Starting from an empty index (max_seq_len 0) grows on demand.
        let empty = PositionalIndex::new(0, 4).unwrap();
        assert_eq!(empty.num_positions(), 0);
        empty.apply(create(hashes, 2));
        assert_eq!(empty.dropped_out_of_range(), 4); // all dropped: capacity 0
        empty.grow_to_max_seq_len(16);
        assert_eq!(empty.num_positions(), 4);
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
        assert!(
            indexes
                .query(manifest(3, false).id(), &[hash])
                .is_none()
        );
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
        assert_eq!(indexes.apply(create(vec![hash], 30)), ApplyOutcome::KindMismatch);

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
        indexes
            .bind(41, &capsule, CreateKind::Block, None)
            .unwrap();
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
        assert!(
            indexes
                .query(carrier_manifest.id(), &[hash])
                .is_none()
        );
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
            indexes.query(manifest.id(), &[deep, shallow]).unwrap().hash_u128,
            shallow.as_u128().to_string()
        );

        let new_hash = plhs(4, 1, 45)[0];
        assert_eq!(indexes.apply(create(vec![new_hash], 50)), ApplyOutcome::Applied);
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
