// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::time::Duration;

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{
    BundleKey, BundleResourceLineage, CacheManifestId, RegistrationEpoch, ResourceRequirement,
    ResourceRole,
};
use velo_ext::InstanceId;

use super::generation::RetiredGenerations;
use super::{BundleDirectory, BundleDirectoryError, test_registration_epoch};
use crate::features::indexer::protocol::{
    BundleAdvertisementRecord, BundleInvalidateRequest, BundlePublishRequest,
    BundleQueryMissReason, BundleQueryOutcome, BundleQueryRequest,
};
use crate::protocol::MutationCredential;
use crate::registry::RegistryIncarnation;

const CSA: LogicalResourceId = LogicalResourceId(10);
const HCA: LogicalResourceId = LogicalResourceId(11);
const CAPSULE: LogicalResourceId = LogicalResourceId(12);

fn directory(lease_ttl_ms: u64) -> BundleDirectory {
    BundleDirectory::with_clock(lease_ttl_ms, Arc::new(|| 0))
}

fn directory_with_clock(lease_ttl_ms: u64, now: &Arc<AtomicU64>) -> BundleDirectory {
    let now = Arc::clone(now);
    BundleDirectory::with_clock(lease_ttl_ms, Arc::new(move || now.load(Ordering::Acquire)))
}

fn bounded_directory(
    now: &Arc<AtomicU64>,
    lease_ttl_ms: u64,
    advertisement_ttl_ms: u64,
    advertisements_per_owner: usize,
    advertisements_global: usize,
    absent_retirements_per_owner: usize,
    absent_retirements_global: usize,
) -> BundleDirectory {
    let now = Arc::clone(now);
    BundleDirectory::with_limits(
        lease_ttl_ms,
        advertisement_ttl_ms,
        advertisements_per_owner,
        advertisements_global,
        absent_retirements_per_owner,
        absent_retirements_global,
        Arc::new(move || now.load(Ordering::Acquire)),
    )
}

fn key(manifest: CacheManifestId, boundary: u64) -> BundleKey {
    BundleKey::from_parts(
        manifest,
        csa_hashes(boundary).last().copied().unwrap(),
        boundary,
    )
    .unwrap()
}

fn requirements() -> Vec<ResourceRequirement> {
    vec![
        ResourceRequirement::new(CSA, ResourceRole::PrefixHistory, 4).unwrap(),
        ResourceRequirement::new(HCA, ResourceRole::PrefixHistory, 8).unwrap(),
        ResourceRequirement::new(CAPSULE, ResourceRole::BoundaryCapsule, 4).unwrap(),
    ]
}

fn hash_chain(seed: u64, blocks: usize) -> Vec<SequenceHash> {
    let mut hashes = Vec::with_capacity(blocks);
    let mut current = SequenceHash::root(seed);
    for position in 0..blocks {
        if position > 0 {
            current = current.extend(seed + position as u64);
        }
        hashes.push(current);
    }
    hashes
}

fn csa_hashes(boundary: u64) -> Vec<SequenceHash> {
    hash_chain(1, usize::try_from(boundary / 4).unwrap())
}

fn lineages(
    resources: impl IntoIterator<Item = LogicalResourceId>,
    bundle_key: BundleKey,
) -> Vec<BundleResourceLineage> {
    resources
        .into_iter()
        .map(|resource| {
            if resource == HCA {
                return BundleResourceLineage::project_from_canonical(
                    HCA,
                    &csa_hashes(bundle_key.boundary_tokens()),
                    2,
                )
                .unwrap();
            }
            let hashes = match resource {
                CSA => csa_hashes(bundle_key.boundary_tokens()),
                CAPSULE => vec![bundle_key.boundary_hash()],
                _ => panic!("unexpected test resource {resource:?}"),
            };
            BundleResourceLineage::new(resource, hashes).unwrap()
        })
        .collect()
}

fn advertisement(
    owner: InstanceId,
    manifest: CacheManifestId,
    boundary: u64,
    generation: u64,
    expires_at_unix_ms: u64,
    resources: Vec<LogicalResourceId>,
) -> BundlePublishRequest {
    let key = key(manifest, boundary);
    BundlePublishRequest {
        credential: owner_credential(owner),
        advertisement: BundleAdvertisementRecord {
            key,
            generation,
            owner,
            registration_epoch: Some(test_registration_epoch(owner)),
            requirements: requirements(),
            lineages: lineages(resources, key),
            expires_at_unix_ms,
            placements: Vec::new(),
            stage_cost_hint_us: None,
            advertised_at_unix_ms: None,
        },
    }
}

fn owner_credential(owner: InstanceId) -> MutationCredential {
    MutationCredential::for_test_owner(owner)
}

fn register_owner(directory: &BundleDirectory, owner: InstanceId) {
    directory
        .register_owner(owner, owner_credential(owner))
        .unwrap();
}

fn stage_owner_registration(
    directory: &BundleDirectory,
    owner: InstanceId,
    credential: MutationCredential,
    registration_epoch: RegistrationEpoch,
    incarnation: RegistryIncarnation,
) -> Result<(), BundleDirectoryError> {
    directory.stage_owner_transition(owner, Some(credential), registration_epoch)?;
    directory.bind_owner_registration(owner, registration_epoch, incarnation)
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

fn credential() -> MutationCredential {
    MutationCredential::generate()
}

fn authorize_publish(
    credential: MutationCredential,
    mut request: BundlePublishRequest,
) -> BundlePublishRequest {
    request.credential = credential;
    request
}

#[test]
fn publication_requires_the_registered_owner_epoch() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([54; 32]);
    register_owner(&directory, owner);

    let mut missing = advertisement(owner, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]);
    missing.advertisement.registration_epoch = None;
    assert_eq!(
        directory.publish(missing),
        Err(BundleDirectoryError::RegistrationEpochMismatch { owner })
    );

    let mut replacement = advertisement(owner, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]);
    replacement.advertisement.registration_epoch = Some(RegistrationEpoch::new());
    assert_eq!(
        directory.publish(replacement),
        Err(BundleDirectoryError::RegistrationEpochMismatch { owner })
    );

    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            1,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
}

#[test]
fn cross_owner_credential_cannot_publish_an_advertisement() {
    let directory = directory(500);
    let attacker = InstanceId::new_v4();
    let victim = InstanceId::new_v4();
    let attacker_credential = credential();
    let victim_credential = credential();
    let manifest = CacheManifestId::from_bytes([31; 32]);
    directory
        .register_owner(attacker, attacker_credential.clone())
        .unwrap();
    directory.register_owner(victim, victim_credential).unwrap();

    let result = directory.publish(authorize_publish(
        attacker_credential,
        advertisement(victim, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]),
    ));

    assert_eq!(
        result,
        Err(BundleDirectoryError::UnauthorizedOwner { owner: victim })
    );
    assert_eq!(
        directory.query(query(manifest, vec![key(manifest, 8)], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
}

#[test]
fn cross_owner_credential_cannot_invalidate_an_advertisement() {
    let directory = directory(500);
    let attacker = InstanceId::new_v4();
    let victim = InstanceId::new_v4();
    let attacker_credential = credential();
    let victim_credential = credential();
    let manifest = CacheManifestId::from_bytes([32; 32]);
    let bundle_key = key(manifest, 8);
    directory
        .register_owner(attacker, attacker_credential.clone())
        .unwrap();
    directory
        .register_owner(victim, victim_credential.clone())
        .unwrap();
    directory
        .publish(authorize_publish(
            victim_credential,
            advertisement(victim, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]),
        ))
        .unwrap();

    let result = directory.invalidate(BundleInvalidateRequest {
        credential: attacker_credential,
        key: bundle_key,
        generation: 1,
        owner: victim,
        retain_until_unix_ms: 10_000,
    });

    assert_eq!(
        result,
        Err(BundleDirectoryError::UnauthorizedOwner { owner: victim })
    );
    assert!(matches!(
        directory.query(query(manifest, vec![bundle_key], 1_000)),
        BundleQueryOutcome::Hit(_)
    ));
}

#[test]
fn stale_credential_is_rejected_after_owner_reregistration() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let stale_credential = credential();
    let current_credential = credential();
    let manifest = CacheManifestId::from_bytes([33; 32]);
    directory
        .register_owner(owner, stale_credential.clone())
        .unwrap();
    directory
        .register_owner(owner, current_credential.clone())
        .unwrap();

    assert_eq!(
        directory.publish(authorize_publish(
            stale_credential,
            advertisement(owner, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]),
        )),
        Err(BundleDirectoryError::UnauthorizedOwner { owner })
    );
    directory
        .publish(authorize_publish(
            current_credential,
            advertisement(owner, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]),
        ))
        .unwrap();
}

#[test]
fn staged_owner_rotation_is_hidden_and_rollback_preserves_exact_generation_state() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let previous_credential = credential();
    let replacement_credential = credential();
    let manifest = CacheManifestId::from_bytes([51; 32]);
    let bundle_key = key(manifest, 8);
    directory
        .register_owner(owner, previous_credential.clone())
        .unwrap();
    directory
        .publish(authorize_publish(
            previous_credential.clone(),
            advertisement(owner, manifest, 8, 7, 10_000, vec![CSA, HCA, CAPSULE]),
        ))
        .unwrap();
    assert!(
        directory
            .invalidate(BundleInvalidateRequest {
                credential: previous_credential.clone(),
                key: bundle_key,
                generation: 7,
                owner,
                retain_until_unix_ms: 10_000,
            })
            .unwrap()
    );
    directory
        .publish(authorize_publish(
            previous_credential.clone(),
            advertisement(owner, manifest, 8, 9, 10_000, vec![CSA, HCA, CAPSULE]),
        ))
        .unwrap();

    stage_owner_registration(
        &directory,
        owner,
        replacement_credential,
        test_registration_epoch(owner),
        RegistryIncarnation::from_u64(2),
    )
    .unwrap();
    assert_eq!(
        directory.query(query(manifest, vec![bundle_key], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
    assert_eq!(
        directory.publish(authorize_publish(
            previous_credential.clone(),
            advertisement(owner, manifest, 8, 10, 10_000, vec![CSA, HCA, CAPSULE]),
        )),
        Err(BundleDirectoryError::UnknownOwner { owner })
    );

    stage_owner_registration(
        &directory,
        owner,
        previous_credential.clone(),
        test_registration_epoch(owner),
        RegistryIncarnation::from_u64(3),
    )
    .unwrap();
    let BundleQueryOutcome::Hit(hit) = directory.query(query(manifest, vec![bundle_key], 1_000))
    else {
        panic!("rollback did not restore the preserved live advertisement");
    };
    assert_eq!(hit.advertisement.generation, 9);
    assert_eq!(
        directory.publish(authorize_publish(
            previous_credential,
            advertisement(owner, manifest, 8, 7, 10_000, vec![CSA, HCA, CAPSULE]),
        )),
        Err(BundleDirectoryError::StaleGeneration {
            current: 7,
            attempted: 7,
        })
    );
}

#[test]
fn stale_finalize_cannot_commit_another_owner_incarnation() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let previous_credential = credential();
    let replacement_credential = credential();
    let expected = RegistryIncarnation::from_u64(2);
    let stale = RegistryIncarnation::from_u64(1);
    let manifest = CacheManifestId::from_bytes([52; 32]);
    let bundle_key = key(manifest, 8);
    directory
        .register_owner(owner, previous_credential.clone())
        .unwrap();
    directory
        .publish(authorize_publish(
            previous_credential,
            advertisement(owner, manifest, 8, 9, 10_000, vec![CSA, HCA, CAPSULE]),
        ))
        .unwrap();
    stage_owner_registration(
        &directory,
        owner,
        replacement_credential.clone(),
        test_registration_epoch(owner),
        expected,
    )
    .unwrap();

    assert_eq!(
        directory.finalize_owner_registration(owner, stale),
        Err(BundleDirectoryError::StaleOwnerRegistration {
            owner,
            expected,
            attempted: stale,
        })
    );
    assert_eq!(
        directory.query(query(manifest, vec![bundle_key], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );

    directory
        .finalize_owner_registration(owner, expected)
        .unwrap();
    assert_eq!(
        directory.query(query(manifest, vec![bundle_key], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
    directory
        .publish(authorize_publish(
            replacement_credential,
            advertisement(owner, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]),
        ))
        .unwrap();
}

#[test]
fn valid_same_owner_credential_can_publish_and_invalidate() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let owner_credential = credential();
    let manifest = CacheManifestId::from_bytes([34; 32]);
    let bundle_key = key(manifest, 8);
    directory
        .register_owner(owner, owner_credential.clone())
        .unwrap();
    directory
        .publish(authorize_publish(
            owner_credential.clone(),
            advertisement(owner, manifest, 8, 7, 10_000, vec![CSA, HCA, CAPSULE]),
        ))
        .unwrap();

    assert!(matches!(
        directory.query(query(manifest, vec![bundle_key], 1_000)),
        BundleQueryOutcome::Hit(_)
    ));
    assert!(
        directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential,
                key: bundle_key,
                generation: 7,
                owner,
                retain_until_unix_ms: 10_000,
            })
            .unwrap()
    );
    assert_eq!(
        directory.query(query(manifest, vec![bundle_key], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
}

#[test]
fn unknown_or_removed_owner_is_never_visible() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([1; 32]);
    let publish = advertisement(owner, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]);

    assert!(matches!(
        directory.publish(publish.clone()),
        Err(BundleDirectoryError::UnknownOwner { .. })
    ));
    register_owner(&directory, owner);
    directory.publish(publish).unwrap();
    assert!(matches!(
        directory.query(query(manifest, vec![key(manifest, 8)], 1_000)),
        BundleQueryOutcome::Hit(_)
    ));
    directory.remove_owner(owner);
    assert_eq!(
        directory.query(query(manifest, vec![key(manifest, 8)], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
}

#[test]
fn invalid_query_manifest_mismatch_and_expired_have_distinct_miss_reasons() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([2; 32]);
    register_owner(&directory, owner);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            1,
            999,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();

    assert_eq!(
        directory.query(query(manifest, vec![key(manifest, 8)], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::Expired)
    );
    let mut invalid_query = query(manifest, vec![key(manifest, 8)], 1);
    invalid_query.requirements.push(requirements()[0].clone());
    assert_eq!(
        directory.query(invalid_query),
        BundleQueryOutcome::Miss(BundleQueryMissReason::Incomplete)
    );
    let other = CacheManifestId::from_bytes([3; 32]);
    assert_eq!(
        directory.query(query(other, vec![key(manifest, 8)], 1)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::Incompatible)
    );
}

#[test]
fn duplicate_resources_are_rejected_before_visibility() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([6; 32]);
    register_owner(&directory, owner);
    assert!(
        directory
            .publish(advertisement(
                owner,
                manifest,
                8,
                1,
                10_000,
                vec![CSA, HCA, CAPSULE, CAPSULE],
            ))
            .is_err()
    );

    assert_eq!(
        directory.query(query(manifest, vec![key(manifest, 8)], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
}

#[test]
fn publish_rejects_geometry_that_violates_advertised_requirements() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([16; 32]);
    let bundle_key = key(manifest, 8);
    register_owner(&directory, owner);
    let mut publish = advertisement(owner, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]);
    publish.advertisement.lineages = vec![
        BundleResourceLineage::new(CSA, vec![SequenceHash::root(1)]).unwrap(),
        BundleResourceLineage::new(HCA, hash_chain(101, 1)).unwrap(),
        BundleResourceLineage::new(CAPSULE, vec![bundle_key.boundary_hash()]).unwrap(),
    ];

    assert!(directory.publish(publish).is_err());
    assert_eq!(
        directory.query(query(manifest, vec![bundle_key], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
}

#[test]
fn stored_malformed_lower_order_owner_cannot_mask_valid_same_key_owner() {
    let directory = directory(500);
    let manifest = CacheManifestId::from_bytes([17; 32]);
    let bundle_key = key(manifest, 8);
    let mut owners = [InstanceId::new_v4(), InstanceId::new_v4()];
    owners.sort_by_key(ToString::to_string);
    let [malformed_owner, valid_owner] = owners;
    register_owner(&directory, malformed_owner);
    register_owner(&directory, valid_owner);

    let mut malformed = advertisement(
        malformed_owner,
        manifest,
        8,
        1,
        10_000,
        vec![CSA, HCA, CAPSULE],
    )
    .advertisement;
    malformed.lineages = vec![
        BundleResourceLineage::new(CSA, vec![SequenceHash::root(1)]).unwrap(),
        BundleResourceLineage::new(HCA, hash_chain(101, 1)).unwrap(),
        BundleResourceLineage::new(CAPSULE, vec![bundle_key.boundary_hash()]).unwrap(),
    ];
    directory
        .state
        .write()
        .unwrap()
        .advertisements
        .insert(malformed)
        .unwrap();

    directory
        .publish(advertisement(
            valid_owner,
            manifest,
            8,
            1,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();

    let BundleQueryOutcome::Hit(hit) = directory.query(query(manifest, vec![bundle_key], 1_000))
    else {
        panic!("the valid same-key owner must remain discoverable");
    };
    assert_eq!(hit.advertisement.owner, valid_owner);
}

#[test]
fn incompatible_lower_order_owner_cannot_mask_valid_same_key_owner() {
    let directory = directory(500);
    let manifest = CacheManifestId::from_bytes([17; 32]);
    let bundle_key = key(manifest, 8);
    let mut owners = [InstanceId::new_v4(), InstanceId::new_v4()];
    owners.sort_by_key(ToString::to_string);
    let [malformed_owner, valid_owner] = owners;
    register_owner(&directory, malformed_owner);
    register_owner(&directory, valid_owner);

    let mut malformed = advertisement(
        malformed_owner,
        manifest,
        8,
        1,
        10_000,
        vec![CSA, HCA, CAPSULE],
    );
    malformed.advertisement.requirements = vec![
        ResourceRequirement::new(CSA, ResourceRole::PrefixHistory, 4).unwrap(),
        ResourceRequirement::new(HCA, ResourceRole::BoundaryCapsule, 8).unwrap(),
        ResourceRequirement::new(CAPSULE, ResourceRole::BoundaryCapsule, 4).unwrap(),
    ];
    malformed.advertisement.lineages = vec![
        BundleResourceLineage::new(CSA, csa_hashes(bundle_key.boundary_tokens())).unwrap(),
        BundleResourceLineage::new(HCA, vec![bundle_key.boundary_hash()]).unwrap(),
        BundleResourceLineage::new(CAPSULE, vec![bundle_key.boundary_hash()]).unwrap(),
    ];

    directory
        .publish(malformed)
        .expect("the forged record is self-consistent under its advertised requirements");
    directory
        .publish(advertisement(
            valid_owner,
            manifest,
            8,
            1,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();

    let BundleQueryOutcome::Hit(hit) = directory.query(query(manifest, vec![bundle_key], 1_000))
    else {
        panic!("the valid same-key owner must remain discoverable");
    };
    assert_eq!(hit.advertisement.owner, valid_owner);
}

#[test]
fn stale_generation_is_rejected_and_query_can_retry_earlier_boundary() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([4; 32]);
    register_owner(&directory, owner);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            2,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
    assert!(matches!(
        directory.publish(advertisement(
            owner,
            manifest,
            8,
            1,
            10_000,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::StaleGeneration { .. })
    ));

    let BundleQueryOutcome::Hit(hit) = directory.query(query(
        manifest,
        vec![key(manifest, 16), key(manifest, 8)],
        1_000,
    )) else {
        panic!("expected earlier complete bundle");
    };
    assert_eq!(hit.advertisement.key, key(manifest, 8));
    assert!(hit.lease_expires_at_unix_ms <= 1_500);
    assert!(hit.lease_expires_at_unix_ms <= hit.advertisement.expires_at_unix_ms);
}

#[test]
fn invalidation_requires_the_exact_owner_generation() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([5; 32]);
    let bundle_key = key(manifest, 8);
    register_owner(&directory, owner);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            7,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();

    assert!(
        !directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: bundle_key,
                generation: 6,
                owner,
                retain_until_unix_ms: 10_000,
            })
            .unwrap()
    );
    assert!(
        directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: bundle_key,
                generation: 7,
                owner,
                retain_until_unix_ms: 10_000,
            })
            .unwrap()
    );
    assert_eq!(
        directory.query(query(manifest, vec![bundle_key], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
}

#[test]
fn newer_invalidation_is_an_owner_local_monotonic_high_water() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = directory_with_clock(500, &now);
    let owner = InstanceId::new_v4();
    let other_owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([15; 32]);
    let bundle_key = key(manifest, 8);
    register_owner(&directory, owner);
    register_owner(&directory, other_owner);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            7,
            30_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
    assert!(
        !directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: bundle_key,
                generation: 100,
                owner,
                retain_until_unix_ms: 10_000,
            })
            .unwrap()
    );
    assert_eq!(
        directory.query(query(manifest, vec![bundle_key], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound),
        "generation 7 must not remain visible below a generation-100 invalidation"
    );
    directory
        .publish(advertisement(
            other_owner,
            manifest,
            8,
            1,
            30_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("another owner must not inherit this owner's high-water mark");
    let BundleQueryOutcome::Hit(hit) = directory.query(query(manifest, vec![bundle_key], 1_000))
    else {
        panic!("the other owner's advertisement must remain visible");
    };
    assert_eq!(hit.advertisement.owner, other_owner);
    assert_eq!(hit.advertisement.generation, 1);

    for attempted in [7, 100] {
        assert_eq!(
            directory.publish(advertisement(
                owner,
                manifest,
                8,
                attempted,
                30_000,
                vec![CSA, HCA, CAPSULE],
            )),
            Err(BundleDirectoryError::StaleGeneration {
                current: 100,
                attempted,
            })
        );
    }
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            101,
            30_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("the generation above the retired high-water must remain publishable");

    assert!(
        !directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: bundle_key,
                generation: 90,
                owner,
                retain_until_unix_ms: 20_000,
            })
            .unwrap()
    );
    directory.remove_owner(other_owner);
    let BundleQueryOutcome::Hit(hit) = directory.query(query(manifest, vec![bundle_key], 1_000))
    else {
        panic!("an older invalidation must not remove a newer advertisement");
    };
    assert_eq!(hit.advertisement.owner, owner);
    assert_eq!(hit.advertisement.generation, 101);

    let absent_key = key(manifest, 16);
    assert!(
        !directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: absent_key,
                generation: 100,
                owner,
                retain_until_unix_ms: 400,
            })
            .unwrap()
    );
    assert!(
        !directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: absent_key,
                generation: 90,
                owner,
                retain_until_unix_ms: 450,
            })
            .unwrap()
    );
    now.store(401, Ordering::Release);
    assert!(matches!(
        directory.publish(advertisement(
            owner,
            manifest,
            16,
            100,
            30_000,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::StaleGeneration {
            current: 100,
            attempted: 100,
        })
    ));
    now.store(451, Ordering::Release);
    directory
        .publish(advertisement(
            owner,
            manifest,
            16,
            100,
            30_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("the combined high-water must expire at its bounded longest horizon");
}

#[test]
fn absent_retirements_are_bounded_without_evicting_delayed_replay_protection() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = bounded_directory(&now, 10, 100, 4, 8, 2, 4);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([40; 32]);
    let first = key(manifest, 8);
    let second = key(manifest, 16);
    let live = key(manifest, 24);
    let rejected = key(manifest, 32);
    register_owner(&directory, owner);

    for retired in [first, second] {
        assert!(
            !directory
                .invalidate(BundleInvalidateRequest {
                    credential: owner_credential(owner),
                    key: retired,
                    generation: 7,
                    owner,
                    retain_until_unix_ms: u64::MAX,
                })
                .unwrap()
        );
    }
    assert_eq!(
        directory.invalidate(BundleInvalidateRequest {
            credential: owner_credential(owner),
            key: rejected,
            generation: 7,
            owner,
            retain_until_unix_ms: u64::MAX,
        }),
        Err(BundleDirectoryError::OwnerCapacity { owner, capacity: 2 })
    );
    assert_eq!(directory.state.read().unwrap().retired_generations.len(), 2);
    assert_eq!(
        directory.publish(advertisement(
            owner,
            manifest,
            8,
            7,
            u64::MAX,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::StaleGeneration {
            current: 7,
            attempted: 7,
        }),
        "capacity pressure must not evict an unexpired replay high-water"
    );

    directory
        .publish(advertisement(
            owner,
            manifest,
            24,
            9,
            u64::MAX,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
    assert!(
        directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: live,
                generation: 9,
                owner,
                retain_until_unix_ms: u64::MAX,
            })
            .unwrap(),
        "an exact live invalidation must remain admissible at absent capacity"
    );
    assert_eq!(directory.state.read().unwrap().retired_generations.len(), 3);
    assert_eq!(
        directory.publish(advertisement(
            owner,
            manifest,
            24,
            9,
            u64::MAX,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::StaleGeneration {
            current: 9,
            attempted: 9,
        })
    );

    now.store(11, Ordering::Release);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            7,
            u64::MAX,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("hostile absent retention must expire at the server-owned lease horizon");
}

#[test]
fn live_advertisements_have_server_owned_lifetime_and_per_owner_capacity() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = bounded_directory(&now, 10, 20, 2, 4, 2, 4);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([41; 32]);
    register_owner(&directory, owner);

    for boundary in [8, 16] {
        directory
            .publish(advertisement(
                owner,
                manifest,
                boundary,
                1,
                u64::MAX,
                vec![CSA, HCA, CAPSULE],
            ))
            .unwrap();
    }
    assert_eq!(
        directory.publish(advertisement(
            owner,
            manifest,
            24,
            1,
            u64::MAX,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::OwnerCapacity { owner, capacity: 2 })
    );

    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            2,
            u64::MAX,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("refreshing an existing key must remain possible at capacity");
    let BundleQueryOutcome::Hit(hit) = directory.query(query(manifest, vec![key(manifest, 8)], 0))
    else {
        panic!("refreshed advertisement must remain visible");
    };
    assert_eq!(hit.advertisement.expires_at_unix_ms, 20);

    now.store(21, Ordering::Release);
    directory
        .publish(advertisement(
            owner,
            manifest,
            24,
            1,
            u64::MAX,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("server-expired advertisements must release owner capacity");
    assert_eq!(directory.state.read().unwrap().advertisements.len(), 1);
}

#[test]
fn global_capacity_rejects_new_identities_but_allows_refresh_and_exact_removal() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = bounded_directory(&now, 10, 100, 2, 2, 2, 2);
    let owners = [InstanceId::new_v4(), InstanceId::new_v4()];
    let manifest = CacheManifestId::from_bytes([42; 32]);
    for owner in owners {
        register_owner(&directory, owner);
    }

    directory
        .publish(advertisement(
            owners[0],
            manifest,
            8,
            1,
            100,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
    directory
        .publish(advertisement(
            owners[1],
            manifest,
            16,
            1,
            100,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
    assert_eq!(
        directory.publish(advertisement(
            owners[0],
            manifest,
            24,
            1,
            100,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::GlobalCapacity { capacity: 2 })
    );
    directory
        .publish(advertisement(
            owners[0],
            manifest,
            8,
            2,
            100,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("refresh must remain possible at global advertisement capacity");

    for (owner, boundary) in [(owners[0], 24), (owners[1], 32)] {
        assert!(
            !directory
                .invalidate(BundleInvalidateRequest {
                    credential: owner_credential(owner),
                    key: key(manifest, boundary),
                    generation: 7,
                    owner,
                    retain_until_unix_ms: 10,
                })
                .unwrap()
        );
    }
    assert_eq!(
        directory.invalidate(BundleInvalidateRequest {
            credential: owner_credential(owners[0]),
            key: key(manifest, 40),
            generation: 7,
            owner: owners[0],
            retain_until_unix_ms: 10,
        }),
        Err(BundleDirectoryError::GlobalCapacity { capacity: 4 })
    );
    assert!(
        !directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owners[0]),
                key: key(manifest, 24),
                generation: 8,
                owner: owners[0],
                retain_until_unix_ms: 10,
            })
            .unwrap(),
        "refreshing an existing absent high-water must remain possible at global capacity"
    );
    assert!(
        directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owners[0]),
                key: key(manifest, 8),
                generation: 2,
                owner: owners[0],
                retain_until_unix_ms: 10,
            })
            .unwrap(),
        "exact live removal must remain possible at global absent capacity"
    );
}

#[test]
fn repeated_live_publish_invalidate_cycles_cannot_bypass_total_owner_capacity() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = bounded_directory(&now, 10, 100, 2, 4, 2, 4);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([43; 32]);
    register_owner(&directory, owner);

    for boundary in [8, 16, 24, 32] {
        directory
            .publish(advertisement(
                owner,
                manifest,
                boundary,
                1,
                u64::MAX,
                vec![CSA, HCA, CAPSULE],
            ))
            .unwrap();
        assert!(
            directory
                .invalidate(BundleInvalidateRequest {
                    credential: owner_credential(owner),
                    key: key(manifest, boundary),
                    generation: 1,
                    owner,
                    retain_until_unix_ms: u64::MAX,
                })
                .unwrap()
        );
    }
    assert_eq!(
        directory.state.read().unwrap().retired_generations.len(),
        4,
        "cycling beyond one live-ad capacity window must remain bounded"
    );
    assert_eq!(
        directory.publish(advertisement(
            owner,
            manifest,
            40,
            1,
            u64::MAX,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::OwnerCapacity { owner, capacity: 4 })
    );
    assert_eq!(
        directory.publish(advertisement(
            owner,
            manifest,
            8,
            1,
            u64::MAX,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::StaleGeneration {
            current: 1,
            attempted: 1,
        }),
        "capacity rejection must not evict an earlier replay high-water"
    );
}

#[test]
fn exact_invalidation_leaves_live_advertisement_when_retirement_has_no_slot() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = bounded_directory(&now, 10, 100, 2, 4, 2, 4);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([44; 32]);
    let bundle_key = key(manifest, 8);
    register_owner(&directory, owner);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            7,
            100,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
    directory.state.write().unwrap().retired_generations = RetiredGenerations::new(10, 0, 0, 0, 0);

    assert_eq!(
        directory.invalidate(BundleInvalidateRequest {
            credential: owner_credential(owner),
            key: bundle_key,
            generation: 7,
            owner,
            retain_until_unix_ms: 10,
        }),
        Err(BundleDirectoryError::GlobalCapacity { capacity: 0 })
    );
    assert!(
        matches!(
            directory.query(query(manifest, vec![bundle_key], 0)),
            BundleQueryOutcome::Hit(_)
        ),
        "failed exact invalidation must not remove the live advertisement"
    );
}

#[test]
fn invalidated_generation_cannot_be_republished_before_its_original_expiry() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([9; 32]);
    let bundle_key = key(manifest, 8);
    register_owner(&directory, owner);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            7,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
    assert!(
        directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: bundle_key,
                generation: 7,
                owner,
                retain_until_unix_ms: 10_000,
            })
            .unwrap()
    );

    assert!(matches!(
        directory.publish(advertisement(
            owner,
            manifest,
            8,
            7,
            20_000,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::StaleGeneration {
            current: 7,
            attempted: 7,
        })
    ));
    assert_eq!(
        directory.query(query(manifest, vec![bundle_key], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
}

#[test]
fn invalidation_arriving_before_delayed_publish_still_retires_generation() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = directory_with_clock(500, &now);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([12; 32]);
    let bundle_key = key(manifest, 8);
    register_owner(&directory, owner);

    assert!(
        !directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: bundle_key,
                generation: 7,
                owner,
                retain_until_unix_ms: 10_000,
            })
            .unwrap()
    );
    assert!(matches!(
        directory.publish(advertisement(
            owner,
            manifest,
            8,
            7,
            10_000,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::StaleGeneration {
            current: 7,
            attempted: 7,
        })
    ));
    now.store(501, Ordering::Release);
    assert_eq!(
        directory.query(query(manifest, vec![bundle_key], 501)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            7,
            20_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("an absent tombstone must expire at the server-owned retention horizon");
}

#[test]
fn expired_absent_tombstone_is_pruned_without_a_directory_query() {
    let directory = BundleDirectory::new(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([13; 32]);
    let bundle_key = key(manifest, 8);
    register_owner(&directory, owner);

    assert!(
        !directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: bundle_key,
                generation: 7,
                owner,
                retain_until_unix_ms: 1,
            })
            .unwrap()
    );
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            7,
            u64::MAX,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("mutation-time pruning must not require a consumer query");
}

#[test]
fn invalidated_generation_can_be_republished_after_its_original_expiry() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = directory_with_clock(500, &now);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([11; 32]);
    let bundle_key = key(manifest, 8);
    register_owner(&directory, owner);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            7,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
    assert!(
        directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: bundle_key,
                generation: 7,
                owner,
                retain_until_unix_ms: 10_000,
            })
            .unwrap()
    );

    now.store(10_001, Ordering::Release);
    assert_eq!(
        directory.query(query(manifest, vec![bundle_key], 10_001)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound)
    );
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            7,
            20_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("the retired-generation high-water mark must expire with the original ad");
}

#[test]
fn newer_shorter_lived_generation_preserves_older_retirement_horizon() {
    let now = Arc::new(AtomicU64::new(0));
    let directory = directory_with_clock(500, &now);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([14; 32]);
    let bundle_key = key(manifest, 8);
    register_owner(&directory, owner);
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            7,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
    assert!(
        directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: bundle_key,
                generation: 7,
                owner,
                retain_until_unix_ms: 10_000,
            })
            .unwrap()
    );
    directory
        .publish(advertisement(
            owner,
            manifest,
            8,
            8,
            5_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("a newer generation must remain publishable");
    assert!(
        directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner),
                key: bundle_key,
                generation: 8,
                owner,
                retain_until_unix_ms: 5_000,
            })
            .unwrap()
    );

    now.store(5_001, Ordering::Release);
    assert!(matches!(
        directory.publish(advertisement(
            owner,
            manifest,
            8,
            7,
            10_000,
            vec![CSA, HCA, CAPSULE],
        )),
        Err(BundleDirectoryError::StaleGeneration {
            current: 8,
            attempted: 7,
        })
    ));
}

#[test]
fn reregistered_owner_does_not_inherit_retired_generations() {
    let directory = directory(500);
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([10; 32]);
    let old_credential = credential();
    directory
        .register_owner(owner, old_credential.clone())
        .unwrap();
    directory
        .publish(authorize_publish(
            old_credential.clone(),
            advertisement(owner, manifest, 8, 4, 10_000, vec![CSA, HCA, CAPSULE]),
        ))
        .unwrap();
    directory
        .invalidate(BundleInvalidateRequest {
            credential: old_credential,
            key: key(manifest, 8),
            generation: 4,
            owner,
            retain_until_unix_ms: 10_000,
        })
        .unwrap();

    let replacement_credential = credential();
    directory
        .register_owner(owner, replacement_credential.clone())
        .unwrap();
    directory
        .publish(authorize_publish(
            replacement_credential,
            advertisement(owner, manifest, 8, 1, 20_000, vec![CSA, HCA, CAPSULE]),
        ))
        .expect("a replacement owner must start a fresh generation domain");
}

#[test]
fn generations_are_owner_local_and_invalidation_preserves_other_owners() {
    let directory = directory(500);
    let owner_a = InstanceId::new_v4();
    let owner_b = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([7; 32]);
    let bundle_key = key(manifest, 8);
    register_owner(&directory, owner_a);
    register_owner(&directory, owner_b);

    directory
        .publish(advertisement(
            owner_a,
            manifest,
            8,
            100,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .unwrap();
    directory
        .publish(advertisement(
            owner_b,
            manifest,
            8,
            1,
            10_000,
            vec![CSA, HCA, CAPSULE],
        ))
        .expect("a fresh generation from another owner must not be compared with owner A");

    assert!(
        directory
            .invalidate(BundleInvalidateRequest {
                credential: owner_credential(owner_a),
                key: bundle_key,
                generation: 100,
                owner: owner_a,
                retain_until_unix_ms: 10_000,
            })
            .unwrap()
    );
    let BundleQueryOutcome::Hit(hit) = directory.query(query(manifest, vec![bundle_key], 1_000))
    else {
        panic!("invalidating owner A must preserve owner B's same-key advertisement");
    };
    assert_eq!(hit.advertisement.owner, owner_b);
    assert_eq!(hit.advertisement.generation, 1);
}

#[test]
fn unregister_cannot_interleave_between_publish_validation_and_insert() {
    let owner_checked = Arc::new(Barrier::new(2));
    let release_publish = Arc::new(Barrier::new(2));
    let directory = Arc::new(BundleDirectory::with_publish_after_owner_check(500, {
        let owner_checked = Arc::clone(&owner_checked);
        let release_publish = Arc::clone(&release_publish);
        Arc::new(move || {
            owner_checked.wait();
            release_publish.wait();
        })
    }));
    let owner = InstanceId::new_v4();
    let manifest = CacheManifestId::from_bytes([8; 32]);
    register_owner(&directory, owner);

    let publisher = {
        let directory = Arc::clone(&directory);
        std::thread::spawn(move || {
            directory.publish(advertisement(
                owner,
                manifest,
                8,
                1,
                10_000,
                vec![CSA, HCA, CAPSULE],
            ))
        })
    };
    owner_checked.wait();

    let (owner_churned, churn_complete) = mpsc::channel();
    let churn = {
        let directory = Arc::clone(&directory);
        std::thread::spawn(move || {
            directory.remove_owner(owner);
            register_owner(&directory, owner);
            owner_churned.send(()).unwrap();
        })
    };
    let churn_finished_before_release = churn_complete
        .recv_timeout(Duration::from_millis(100))
        .is_ok();
    release_publish.wait();

    publisher.join().unwrap().unwrap();
    churn.join().unwrap();
    assert!(
        !churn_finished_before_release,
        "owner removal must wait for an in-progress validated publication"
    );
    assert_eq!(
        directory.query(query(manifest, vec![key(manifest, 8)], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound),
        "an advertisement from the removed owner lifecycle must not reappear after registration"
    );
}

#[test]
fn owner_rotation_cannot_interleave_between_publish_authorization_and_insert() {
    let owner_checked = Arc::new(Barrier::new(2));
    let release_publish = Arc::new(Barrier::new(2));
    let should_pause_publish = Arc::new(AtomicBool::new(true));
    let directory = Arc::new(BundleDirectory::with_publish_after_owner_check(500, {
        let owner_checked = Arc::clone(&owner_checked);
        let release_publish = Arc::clone(&release_publish);
        let should_pause_publish = Arc::clone(&should_pause_publish);
        Arc::new(move || {
            if !should_pause_publish.swap(false, Ordering::AcqRel) {
                return;
            }
            owner_checked.wait();
            release_publish.wait();
        })
    }));
    let owner = InstanceId::new_v4();
    let previous_credential = credential();
    let replacement_credential = credential();
    let manifest = CacheManifestId::from_bytes([53; 32]);
    directory
        .register_owner(owner, previous_credential.clone())
        .unwrap();

    let publisher = {
        let directory = Arc::clone(&directory);
        std::thread::spawn(move || {
            directory.publish(authorize_publish(
                previous_credential,
                advertisement(owner, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]),
            ))
        })
    };
    owner_checked.wait();

    let (rotation_staged, stage_complete) = mpsc::channel();
    let rotation = {
        let directory = Arc::clone(&directory);
        let replacement_credential = replacement_credential.clone();
        std::thread::spawn(move || {
            let result = stage_owner_registration(
                &directory,
                owner,
                replacement_credential,
                test_registration_epoch(owner),
                RegistryIncarnation::from_u64(2),
            );
            rotation_staged.send(()).unwrap();
            result
        })
    };
    let rotation_finished_before_release = stage_complete
        .recv_timeout(Duration::from_millis(100))
        .is_ok();
    release_publish.wait();

    publisher.join().unwrap().unwrap();
    rotation.join().unwrap().unwrap();
    assert!(
        !rotation_finished_before_release,
        "owner rotation must wait for an authorized publication to finish"
    );
    assert_eq!(
        directory.query(query(manifest, vec![key(manifest, 8)], 1_000)),
        BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound),
        "the staged rotation must hide the prior lifecycle atomically"
    );
    directory
        .finalize_owner_registration(owner, RegistryIncarnation::from_u64(2))
        .unwrap();
    directory
        .publish(authorize_publish(
            replacement_credential,
            advertisement(owner, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]),
        ))
        .unwrap();
}

/// `requirements` defines the bundle's resource set — "every listed resource is
/// mandatory" — so a placement for a resource outside it is a claim about
/// something this advertisement does not describe. It also feeds `ready_tier()`,
/// which a CT-2a consumer reads as a stage-cost hint, so an unrequired claim is
/// a cost signal derived from a resource the record does not own.
#[test]
fn a_placement_for_an_unrequired_resource_is_refused() {
    use crate::features::indexer::protocol::ReadyPlacement;
    use kvbm_protocols::tier_protocol::{PhysicalPlacementMode, TierDepth};

    let directory = directory(1_000);
    let owner = InstanceId::new_v4();
    register_owner(&directory, owner);
    let manifest = CacheManifestId::from_bytes([5; 32]);

    let mut request = advertisement(owner, manifest, 8, 1, 10_000, vec![CSA, HCA, CAPSULE]);
    request.advertisement.placements = vec![ReadyPlacement {
        resource: LogicalResourceId(99),
        lane: 0,
        tier: TierDepth(1),
        placement: PhysicalPlacementMode::Whole,
    }];
    assert!(matches!(
        directory.publish(request.clone()),
        Err(BundleDirectoryError::UnrequiredPlacement { resource, .. })
            if resource == LogicalResourceId(99)
    ));

    // Additive-safe in both directions: a required resource is accepted, and a
    // publisher that predates R7b sends no placements at all.
    request.advertisement.placements[0].resource = CSA;
    directory.publish(request.clone()).unwrap();
    request.advertisement.generation = 2;
    request.advertisement.placements.clear();
    directory.publish(request).unwrap();
}
