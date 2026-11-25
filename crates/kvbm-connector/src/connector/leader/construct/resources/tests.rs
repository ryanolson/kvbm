use std::collections::BTreeMap;

use kvbm_common::LogicalResourceId;
use kvbm_config::HostCacheConfig;
use kvbm_engine::worker::WorkerCacheConfig;
use kvbm_physical::layout::LayoutConfig;
use kvbm_protocols::cache_manifest::{
    CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};

use super::{resource_tiers_from_config, validate_manifest_geometry};

fn layout(page_size: usize) -> LayoutConfig {
    LayoutConfig::builder()
        .num_blocks(8)
        .num_layers(1)
        .outer_dim(1)
        .page_size(page_size)
        .inner_dim(16)
        .dtype_width_bytes(1)
        .build()
        .unwrap()
}

fn identity() -> kvbm_protocols::cache_manifest::CacheIdentity {
    CacheManifest::new(
        ModelIdentity::new("mixed", "v1", [7; 32]).unwrap(),
        "mixed-v1",
        vec![
            ResourceRequirement::new(LogicalResourceId(3), ResourceRole::BoundaryCapsule, 4)
                .unwrap(),
            ResourceRequirement::new(LogicalResourceId(1), ResourceRole::PrefixHistory, 8).unwrap(),
            ResourceRequirement::new(LogicalResourceId(2), ResourceRole::PrefixHistory, 4).unwrap(),
        ],
        Default::default(),
    )
    .unwrap()
    .identity()
}

fn config(primary: LogicalResourceId) -> WorkerCacheConfig {
    WorkerCacheConfig {
        manifest: Some(identity().manifest()),
        primary,
        resources: BTreeMap::from([
            (LogicalResourceId(3), layout(4)),
            (LogicalResourceId(1), layout(8)),
            (LogicalResourceId(2), layout(4)),
        ]),
    }
}

#[test]
fn manifest_geometry_accepts_order_independent_finest_primary() {
    validate_manifest_geometry(&identity(), &config(LogicalResourceId(2))).unwrap();
}

#[test]
fn manifest_geometry_rejects_coarser_primary_before_runtime_construction() {
    let error = validate_manifest_geometry(&identity(), &config(LogicalResourceId(1)))
        .expect_err("a coarser primary must be rejected");

    assert!(error.to_string().contains("canonical prefix history"));
}

#[test]
fn manifest_geometry_rejects_native_block_size_drift() {
    let mut config = config(LogicalResourceId(2));
    config.resources.insert(LogicalResourceId(2), layout(8));

    let error = validate_manifest_geometry(&identity(), &config)
        .expect_err("manifest and physical native sizes must match");

    assert!(error.to_string().contains("manifest native block size"));
}

#[test]
fn base_capacity_resource_tiers_split_is_equal() {
    let resources = WorkerCacheConfig {
        manifest: None,
        primary: LogicalResourceId(1),
        resources: BTreeMap::from([
            (LogicalResourceId(1), layout(8)),
            (LogicalResourceId(2), layout(8)),
        ]),
    };
    let host = HostCacheConfig {
        num_blocks: Some(16),
        ..Default::default()
    };

    let tiers = resource_tiers_from_config(&resources, &host, None).expect("tier planning");

    assert_eq!(tiers.len(), 2);
    for tier in tiers.values() {
        assert_eq!(
            tier.host_block_count, 8,
            "each resource receives half of the configured host pages"
        );
    }
}
