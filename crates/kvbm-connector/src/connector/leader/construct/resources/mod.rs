// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validation and capacity planning for manifest-scoped worker resources.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use kvbm_common::LogicalResourceId;
use kvbm_config::{DiskCacheConfig, HostCacheConfig};
use kvbm_engine::worker::{ResourceTierConfig, WorkerCacheConfig};
use kvbm_physical::layout::LayoutConfig;
use kvbm_protocols::cache_manifest::CacheManifestId;

use crate::KvbmRuntime;

mod policy;
mod topology;

#[cfg(test)]
mod tests;

pub(in crate::connector::leader) use policy::ResourceAdmissionPlan;
pub(super) use topology::{
    build_collective_bootstrap, logical_tier_block_count, resolve_parallelism,
};

/// Validated rank-invariant resource topology and lower-tier capacities.
pub(super) struct ResourcePlan {
    pub(super) config: WorkerCacheConfig,
    pub(super) primary: LogicalResourceId,
    pub(super) primary_layout: LayoutConfig,
    pub(super) tiers: BTreeMap<LogicalResourceId, ResourceTierConfig>,
    pub(super) parallelism: BTreeMap<LogicalResourceId, kvbm_config::ParallelismMode>,
    pub(super) admission: ResourceAdmissionPlan,
}

impl ResourcePlan {
    pub(super) fn build(
        runtime: &Arc<KvbmRuntime>,
        workers: &[WorkerCacheConfig],
        expected_manifest: Option<CacheManifestId>,
        identity: Option<&kvbm_protocols::cache_manifest::CacheIdentity>,
    ) -> Result<Self> {
        let config = workers
            .first()
            .ok_or_else(|| anyhow!("cannot plan resources without workers"))?
            .clone();
        config.validate()?;
        if config.manifest != expected_manifest {
            bail!(
                "worker cache manifest {:?} does not match leader manifest {:?}",
                config.manifest,
                expected_manifest
            );
        }
        for (rank, worker) in workers.iter().enumerate().skip(1) {
            worker.validate()?;
            if worker != &config {
                bail!("resource layout config mismatch between worker {rank} and worker 0");
            }
        }
        if let Some(identity) = identity {
            validate_manifest_geometry(identity, &config)?;
        }

        let primary = config.primary;
        let primary_layout = config
            .resources
            .get(&primary)
            .expect("validated primary resource")
            .clone();
        let tiers = resource_tiers(runtime, &config)?;
        let parallelism = config
            .resources
            .iter()
            .map(|(&resource, layout)| {
                (
                    resource,
                    resolve_parallelism(runtime.config().cache.parallelism, layout),
                )
            })
            .collect();
        let admission =
            ResourceAdmissionPlan::build(identity, &config.resources, &parallelism, workers.len())?;
        Ok(Self {
            config,
            primary,
            primary_layout,
            tiers,
            parallelism,
            admission,
        })
    }
}

fn validate_manifest_geometry(
    identity: &kvbm_protocols::cache_manifest::CacheIdentity,
    config: &WorkerCacheConfig,
) -> Result<()> {
    ensure!(
        config.manifest == Some(identity.manifest()),
        "worker cache manifest id does not match its registered cache identity"
    );
    ensure!(
        identity.resources().len() == config.resources.len()
            && identity
                .resources()
                .iter()
                .all(|requirement| config.resources.contains_key(&requirement.resource())),
        "cache identity and registered worker layouts have different resource sets"
    );
    let canonical = identity
        .canonical_history()
        .context("cache identity has no canonical prefix history")?;
    ensure!(
        config.primary == canonical.resource(),
        "primary resource {:?} is not canonical prefix history {:?}",
        config.primary,
        canonical.resource()
    );
    let canonical_native = usize::try_from(canonical.native_block_tokens().get())
        .context("canonical prefix history native block size does not fit usize")?;
    for requirement in identity.resources() {
        let resource = requirement.resource();
        let layout = config
            .resources
            .get(&resource)
            .expect("resource-set equality was checked");
        let expected = usize::try_from(requirement.native_block_tokens().get())
            .context("manifest native block size does not fit usize")?;
        ensure!(
            layout.page_size == expected,
            "resource {resource:?} physical block size {} does not match manifest native block size {expected}",
            layout.page_size
        );
        if requirement.role() == kvbm_protocols::cache_manifest::ResourceRole::PrefixHistory {
            ensure!(
                expected.is_multiple_of(canonical_native),
                "prefix history {resource:?} native block size {expected} is not projectable from canonical size {canonical_native}"
            );
        }
    }
    Ok(())
}

fn resource_tiers(
    runtime: &Arc<KvbmRuntime>,
    resources: &WorkerCacheConfig,
) -> Result<BTreeMap<LogicalResourceId, ResourceTierConfig>> {
    resource_tiers_from_config(
        resources,
        &runtime.config().cache.host,
        runtime.config().cache.disk.as_ref(),
    )
}

/// Give every resource an equal share of each configured lower tier.
///
/// The byte estimate comes from [`LayoutConfig::required_bytes`] because the
/// worker wire carries only the layout config. A ragged resource with backing
/// pools of unequal bytes is not representable there; its exact footprint is
/// in its physical layout.
fn resource_tiers_from_config(
    resources: &WorkerCacheConfig,
    host: &HostCacheConfig,
    disk: Option<&DiskCacheConfig>,
) -> Result<BTreeMap<LogicalResourceId, ResourceTierConfig>> {
    let resource_count = resources.resources.len();
    resources
        .resources
        .iter()
        .map(|(&resource, layout)| {
            let bytes_per_block = layout
                .required_bytes()
                .checked_div(layout.num_blocks)
                .ok_or_else(|| anyhow!("resource {resource:?} has zero device blocks"))?;
            let host_block_count =
                host.compute_num_blocks(bytes_per_block).unwrap_or(0) / resource_count;
            let disk_block_count = disk
                .and_then(|disk| disk.compute_num_blocks(bytes_per_block))
                .map(|count| count / resource_count);
            if host_block_count == 0 && disk_block_count.is_none_or(|count| count == 0) {
                bail!("no lower-tier capacity for logical resource {resource:?}");
            }
            Ok((
                resource,
                ResourceTierConfig {
                    host_block_count,
                    disk_block_count,
                },
            ))
        })
        .collect()
}
