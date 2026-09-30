#![allow(clippy::disallowed_macros)]

use std::collections::BTreeMap;

use kvbm_common::{LogicalResourceId, SequenceHash};

use super::{
    BundleKey, CacheManifest, CacheManifestId, ManifestError, ModelIdentity, ResourceRequirement,
    ResourceRole,
};

fn model() -> ModelIdentity {
    ModelIdentity::new("deepseek_v4", "revision-a", [7; 32]).expect("valid model identity")
}

fn requirement(resource: u16, role: ResourceRole, native_block_tokens: u32) -> ResourceRequirement {
    ResourceRequirement::new(LogicalResourceId(resource), role, native_block_tokens)
        .expect("valid resource requirement")
}

fn manifest(
    resources: Vec<ResourceRequirement>,
    attributes: impl IntoIterator<Item = (&'static str, &'static str)>,
) -> CacheManifest {
    CacheManifest::new(
        model(),
        "rhino-dsv4-cache-v1",
        resources,
        attributes
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
    )
    .expect("valid cache manifest")
}

#[test]
fn manifest_digest_is_canonical_across_input_order() {
    let first = manifest(
        vec![
            requirement(2, ResourceRole::BoundaryCapsule, 256),
            requirement(0, ResourceRole::PrefixHistory, 256),
            requirement(1, ResourceRole::PrefixHistory, 256),
        ],
        [("window", "128"), ("cache_dtype", "fp8")],
    );
    let second = manifest(
        vec![
            requirement(1, ResourceRole::PrefixHistory, 256),
            requirement(2, ResourceRole::BoundaryCapsule, 256),
            requirement(0, ResourceRole::PrefixHistory, 256),
        ],
        [("cache_dtype", "fp8"), ("window", "128")],
    );

    assert_eq!(first.canonical_bytes(), second.canonical_bytes());
    assert_eq!(first.id(), second.id());
    assert_eq!(
        first
            .identity()
            .resources()
            .iter()
            .map(ResourceRequirement::resource)
            .collect::<Vec<_>>(),
        vec![
            LogicalResourceId(0),
            LogicalResourceId(1),
            LogicalResourceId(2)
        ]
    );
}

#[test]
fn manifest_id_parses_displayed_hex() {
    let id = CacheManifestId::from_bytes([0xAB; 32]);
    assert_eq!(id.to_string().parse::<CacheManifestId>().unwrap(), id);
    assert_eq!("AB".repeat(32).parse::<CacheManifestId>().unwrap(), id);
    assert!("ab".parse::<CacheManifestId>().is_err());
    assert!(
        format!("{}g", "0".repeat(63))
            .parse::<CacheManifestId>()
            .is_err()
    );
}

#[test]
fn canonical_history_is_the_finest_resource_with_a_stable_tie_break() {
    let manifest = manifest(
        vec![
            requirement(9, ResourceRole::PrefixHistory, 256),
            requirement(7, ResourceRole::PrefixHistory, 128),
            requirement(3, ResourceRole::PrefixHistory, 128),
            requirement(1, ResourceRole::BoundaryCapsule, 256),
        ],
        [],
    );

    let identity = manifest.identity();
    let canonical = identity
        .canonical_history()
        .expect("manifest has prefix history");

    assert_eq!(canonical.resource(), LogicalResourceId(3));
    assert_eq!(canonical.native_block_tokens().get(), 128);
}

#[test]
fn manifest_json_round_trip_preserves_identity() {
    let original = manifest(
        vec![
            requirement(4, ResourceRole::PrefixHistory, 128),
            requirement(9, ResourceRole::BoundaryCapsule, 256),
        ],
        [("backend", "flashmla"), ("tp", "8")],
    );

    let json = serde_json::to_string(&original).expect("serialize manifest");
    let decoded: CacheManifest = serde_json::from_str(&json).expect("deserialize manifest");

    assert_eq!(decoded, original);
    assert_eq!(decoded.id(), original.id());
    assert_eq!(decoded.identity().alignment_tokens().get(), 256);
}

#[test]
fn manifest_rejects_empty_and_duplicate_requirements() {
    let empty = CacheManifest::new(model(), "abi", Vec::new(), BTreeMap::new());
    assert_eq!(empty, Err(ManifestError::NoResources));

    let duplicate = requirement(3, ResourceRole::PrefixHistory, 256);
    let error = CacheManifest::new(
        model(),
        "abi",
        vec![duplicate.clone(), duplicate],
        BTreeMap::new(),
    )
    .expect_err("duplicate resources must fail");
    assert_eq!(
        error,
        ManifestError::DuplicateResource {
            resource: LogicalResourceId(3)
        }
    );
}

#[test]
fn resource_requirement_rejects_zero_native_block_tokens() {
    assert_eq!(
        ResourceRequirement::new(LogicalResourceId(3), ResourceRole::PrefixHistory, 0,),
        Err(ManifestError::ZeroNativeBlockTokens {
            resource: LogicalResourceId(3)
        })
    );
}

#[test]
fn bundle_key_requires_a_manifest_aligned_nonzero_boundary() {
    let manifest = manifest(
        vec![
            requirement(0, ResourceRole::PrefixHistory, 128),
            requirement(1, ResourceRole::BoundaryCapsule, 256),
        ],
        [],
    );
    let identity = manifest.identity();
    let hash = SequenceHash::new(11, None, 22);

    assert!(BundleKey::new(&identity, hash, 256).is_ok());
    assert!(BundleKey::new(&identity, hash, 0).is_err());
    assert!(BundleKey::new(&identity, hash, 128).is_err());
}

#[test]
fn bundle_key_wire_rejects_an_invalid_hash_mode_without_panicking() {
    let mut invalid_hash = [0u8; 16];
    invalid_hash[0] = 0b1100_0000;
    let bytes = rmp_serde::to_vec_named(&serde_json::json!({
        "manifest": CacheManifestId::from_bytes([1; 32]),
        "boundary_hash": invalid_hash,
        "boundary_tokens": 8,
    }))
    .unwrap();

    let decoded = std::panic::catch_unwind(|| rmp_serde::from_slice::<BundleKey>(&bytes));

    assert!(decoded.is_ok(), "malformed wire data must not panic");
    assert!(
        decoded.unwrap().is_err(),
        "invalid bundle-boundary hash mode must be rejected"
    );
}
