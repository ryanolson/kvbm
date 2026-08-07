// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{
    BundleKey, BundleResourceLineage, CacheManifestId, ResourceRequirement, ResourceRole,
};
use velo_ext::InstanceId;

use super::{BundleDirectory, BundleDirectoryError, test_registration_epoch};
use crate::features::indexer::protocol::{
    BundleAdvertisementRecord, BundleInvalidateRequest, BundlePublishRequest,
    BundleQueryMissReason, BundleQueryOutcome, BundleQueryRequest,
};
use crate::protocol::MutationCredential;

const RESOURCE: LogicalResourceId = LogicalResourceId(10);

fn directory(now: &Arc<AtomicU64>) -> BundleDirectory {
    let now = Arc::clone(now);
    BundleDirectory::with_clock(500, Arc::new(move || now.load(Ordering::Acquire)))
}

fn hashes(boundary_tokens: u64) -> Vec<SequenceHash> {
    let mut hashes = Vec::with_capacity(usize::try_from(boundary_tokens / 4).unwrap());
    let mut current = SequenceHash::root(1);
    for position in 0..boundary_tokens / 4 {
        if position > 0 {
            current = current.extend(position);
        }
        hashes.push(current);
    }
    hashes
}

fn key(manifest: CacheManifestId, boundary_tokens: u64) -> BundleKey {
    BundleKey::from_parts(
        manifest,
        hashes(boundary_tokens).last().copied().unwrap(),
        boundary_tokens,
    )
    .unwrap()
}

fn requirements() -> Vec<ResourceRequirement> {
    vec![ResourceRequirement::new(RESOURCE, ResourceRole::PrefixHistory, 4).unwrap()]
}

fn advertisement(
    owner: InstanceId,
    manifest: CacheManifestId,
    boundary_tokens: u64,
    generation: u64,
    expires_at_unix_ms: u64,
) -> BundlePublishRequest {
    let key = key(manifest, boundary_tokens);
    BundlePublishRequest {
        credential: MutationCredential::for_test_owner(owner),
        advertisement: BundleAdvertisementRecord {
            key,
            generation,
            owner,
            registration_epoch: Some(test_registration_epoch(owner)),
            requirements: requirements(),
            lineages: vec![BundleResourceLineage::new(RESOURCE, hashes(boundary_tokens)).unwrap()],
            expires_at_unix_ms,
            placements: Vec::new(),
            stage_cost_hint_us: None,
            advertised_at_unix_ms: None,
        },
    }
}

fn register_owner(directory: &BundleDirectory, owner: InstanceId) {
    directory
        .register_owner(owner, MutationCredential::for_test_owner(owner))
        .unwrap();
}

fn query(
    manifest: CacheManifestId,
    candidates: Vec<BundleKey>,
    now_unix_ms: u64,
) -> BundleQueryRequest {
    BundleQueryRequest {
        manifest,
        requirements: requirements(),
        candidates,
        now_unix_ms,
    }
}

fn stored_advertisement_count(directory: &BundleDirectory) -> usize {
    directory.state.read().unwrap().advertisements.len()
}

fn scheduled_expiration_count(directory: &BundleDirectory) -> usize {
    directory
        .state
        .read()
        .unwrap()
        .advertisements
        .scheduled_expiration_count()
}

#[test]
fn query_physically_reclaims_expired_advertisements_for_registered_owners() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = directory(&now);
    let expired_owner = InstanceId::new_v4();
    let live_owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([35; 32]);
    let expired_key = key(manifest, 8);
    let live_key = key(manifest, 16);
    register_owner(&directory, expired_owner);
    register_owner(&directory, live_owner);
    directory
        .publish(advertisement(expired_owner, manifest, 8, 1, 10))
        .unwrap();
    directory
        .publish(advertisement(live_owner, manifest, 16, 1, 100))
        .unwrap();

    now.store(11, Ordering::Release);
    assert_eq!(
        directory.query(query(manifest, vec![expired_key], 11)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::Expired)
    );
    let BundleQueryOutcome::Hit(hit) =
        directory.query(query(manifest, vec![expired_key, live_key], 11))
    else {
        panic!("the live fallback advertisement must remain discoverable");
    };

    assert_eq!(hit.advertisement.owner, live_owner);
    assert_eq!(stored_advertisement_count(&directory), 1);
    assert!(
        directory
            .state
            .read()
            .unwrap()
            .owner_credentials
            .contains_key(&expired_owner),
        "expiration must reclaim the advertisement without deregistering its owner"
    );
}

#[test]
fn expiration_classification_survives_unrelated_mutation_pruning() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = directory(&now);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([39; 32]);
    let expired_key = key(manifest, 8);
    register_owner(&directory, owner);
    directory
        .publish(advertisement(owner, manifest, 8, 1, 10))
        .unwrap();

    now.store(11, Ordering::Release);
    directory
        .publish(advertisement(owner, manifest, 16, 1, 100))
        .expect("an unrelated publication should reclaim the expired key");

    assert_eq!(
        directory.query(query(manifest, vec![expired_key], 11)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::Expired),
        "expiry classification must survive mutation-time reclamation"
    );
}

#[test]
fn mutation_time_pruning_bounds_unique_key_advertisements() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = directory(&now);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([36; 32]);
    register_owner(&directory, owner);

    for step in 1..=64 {
        now.store(step, Ordering::Release);
        directory
            .publish(advertisement(owner, manifest, step * 4, 1, step + 1))
            .unwrap();
    }

    assert_eq!(
        stored_advertisement_count(&directory),
        1,
        "normal publication traffic must reclaim expired unique keys"
    );
    assert_eq!(
        scheduled_expiration_count(&directory),
        1,
        "expiration metadata must stay proportional to live advertisements"
    );
    let final_key = key(manifest, 64 * 4);
    let BundleQueryOutcome::Hit(hit) = directory.query(query(manifest, vec![final_key], 64)) else {
        panic!("the final unexpired advertisement must remain discoverable");
    };
    assert_eq!(hit.advertisement.key, final_key);
}

#[test]
fn stale_expiration_does_not_remove_a_refreshed_generation() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = directory(&now);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([37; 32]);
    let bundle_key = key(manifest, 8);
    register_owner(&directory, owner);
    directory
        .publish(advertisement(owner, manifest, 8, 7, 10))
        .unwrap();
    directory
        .publish(advertisement(owner, manifest, 8, 8, 100))
        .unwrap();

    now.store(11, Ordering::Release);
    let BundleQueryOutcome::Hit(hit) = directory.query(query(manifest, vec![bundle_key], 11))
    else {
        panic!("replacing an advertisement must replace its expiration schedule");
    };
    assert_eq!(hit.advertisement.generation, 8);
    assert_eq!(stored_advertisement_count(&directory), 1);
    assert_eq!(scheduled_expiration_count(&directory), 1);
}

#[test]
fn expiration_pruning_preserves_unexpired_generation_tombstones() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = directory(&now);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([38; 32]);
    let retired_key = key(manifest, 8);
    register_owner(&directory, owner);
    directory
        .publish(advertisement(owner, manifest, 8, 7, 100))
        .unwrap();
    directory
        .publish(advertisement(owner, manifest, 16, 1, 10))
        .unwrap();
    assert!(
        directory
            .invalidate(BundleInvalidateRequest {
                credential: MutationCredential::for_test_owner(owner),
                key: retired_key,
                generation: 7,
                owner,
                retain_until_unix_ms: 100,
            })
            .unwrap()
    );

    now.store(11, Ordering::Release);
    directory
        .publish(advertisement(owner, manifest, 24, 1, 100))
        .unwrap();
    assert_eq!(stored_advertisement_count(&directory), 1);
    assert_eq!(
        directory.publish(advertisement(owner, manifest, 8, 7, 200)),
        Err(BundleDirectoryError::StaleGeneration {
            current: 7,
            attempted: 7,
        })
    );
}
