// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mixed-resource peer parallelism validation and cache extraction.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use kvbm_common::{LogicalLayoutHandle, LogicalResourceId};
use kvbm_physical::manager::{ParallelismDescriptor, RdmaLayoutDescriptors, WorkerDataPlacement};

use crate::leader::parallelism::{
    ParallelismTemplateSet, validate_remote_metadata, validate_replicated_remote_metadata,
};

/// Rank-ordered peer metadata grouped by logical resource.
pub(super) struct RemoteResourceMetadata {
    primary: LogicalResourceId,
    resources: BTreeMap<LogicalResourceId, RemoteResourceEntry>,
}

struct RemoteResourceEntry {
    descriptors: Vec<ParallelismDescriptor>,
    placement: WorkerDataPlacement,
}

impl RemoteResourceMetadata {
    pub(super) fn iter(
        &self,
    ) -> impl Iterator<
        Item = (
            LogicalResourceId,
            &[ParallelismDescriptor],
            WorkerDataPlacement,
        ),
    > + '_ {
        self.resources
            .iter()
            .map(|(&resource, entry)| (resource, entry.descriptors.as_slice(), entry.placement))
    }
}

pub(super) fn collect_remote_resource_metadata(
    ranks: &[RdmaLayoutDescriptors],
) -> Result<Option<RemoteResourceMetadata>> {
    let resource_rank_count = ranks
        .iter()
        .filter(|rank| rank.resource_parallelism.is_some())
        .count();
    if resource_rank_count == 0 {
        return Ok(None);
    }
    ensure!(
        resource_rank_count == ranks.len(),
        "peer metadata mixes resource-stamped and unstamped ranks"
    );

    let first = ranks[0]
        .resource_parallelism
        .as_ref()
        .expect("resource-stamped rank count was nonzero");
    let primary = first.primary();
    let expected_resources = first.iter().map(|entry| entry.resource).collect::<Vec<_>>();
    let mut resources = expected_resources
        .iter()
        .map(|&resource| {
            let placement = first
                .get(resource)
                .expect("resource originated from first descriptor set")
                .placement;
            (
                resource,
                RemoteResourceEntry {
                    descriptors: Vec::with_capacity(ranks.len()),
                    placement,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    for (rank_index, rank) in ranks.iter().enumerate() {
        let descriptors = rank
            .resource_parallelism
            .as_ref()
            .expect("all resource metadata ranks were checked above");
        let physical_parallelism = rank.parallelism.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "peer rank {rank_index} has resource metadata without a physical-rank descriptor"
            )
        })?;
        ensure!(
            descriptors.primary() == primary,
            "peer rank {rank_index} selects resource {:?} as primary, expected {primary:?}",
            descriptors.primary()
        );
        let actual_resources = descriptors
            .iter()
            .map(|entry| entry.resource)
            .collect::<Vec<_>>();
        ensure!(
            actual_resources == expected_resources,
            "peer rank {rank_index} describes a different logical resource set"
        );

        for descriptor in descriptors.iter() {
            ensure!(
                descriptor.parallelism.rank == physical_parallelism.rank
                    && descriptor.parallelism.tp_size == physical_parallelism.tp_size
                    && descriptor.parallelism.pp_size == physical_parallelism.pp_size,
                "peer resource {:?} descriptor does not use physical rank {} on worker metadata position {rank_index}",
                descriptor.resource,
                physical_parallelism.rank
            );
            let entry = resources
                .get_mut(&descriptor.resource)
                .expect("resource set equality was checked above");
            ensure!(
                descriptor.placement == entry.placement,
                "peer resource {:?} changes placement at rank {rank_index}",
                descriptor.resource
            );
            entry.descriptors.push(descriptor.parallelism.clone());
        }
    }

    Ok(Some(RemoteResourceMetadata { primary, resources }))
}

pub(super) fn validate_remote_resource_metadata(
    local: &ParallelismTemplateSet,
    remote: &RemoteResourceMetadata,
    ranks: &[RdmaLayoutDescriptors],
    required_tier: LogicalLayoutHandle,
) -> Result<()> {
    ensure!(
        remote.primary == local.primary(),
        "peer primary resource {:?} does not match local primary {:?}",
        remote.primary,
        local.primary()
    );
    let local_resources = local
        .iter()
        .map(|(resource, _)| resource)
        .collect::<Vec<_>>();
    let remote_resources = remote.resources.keys().copied().collect::<Vec<_>>();
    ensure!(
        local_resources == remote_resources,
        "peer logical resource set does not match local resource set"
    );

    for (resource, template) in local.iter() {
        let remote_entry = remote
            .resources
            .get(&resource)
            .expect("resource sets were checked above");
        let tier_lists = ranks
            .iter()
            .enumerate()
            .map(|(rank_index, rank)| {
                let layouts = rank.resource_layouts.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "peer rank {rank_index} omits resource layout metadata for {resource:?}"
                    )
                })?;
                let layouts = layouts.get(resource).ok_or_else(|| {
                    anyhow::anyhow!(
                        "peer rank {rank_index} omits physical layouts for {resource:?}"
                    )
                })?;
                Ok(layouts
                    .iter()
                    .map(|layout| layout.logical_type)
                    .collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>>>()?;
        let tier_refs = tier_lists.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let local_placement = template.worker_data_placement();
        ensure!(
            local_placement == remote_entry.placement,
            "resource {resource:?} placement mismatch: local {local_placement:?}, peer {:?}",
            remote_entry.placement
        );

        let validation = match local_placement {
            WorkerDataPlacement::ReplicatedG1StripedLower => validate_replicated_remote_metadata(
                template,
                &remote_entry.descriptors,
                &tier_refs,
                required_tier,
            ),
            WorkerDataPlacement::TensorSharded => validate_remote_metadata(
                template,
                &remote_entry.descriptors,
                &tier_refs,
                required_tier,
            ),
        };
        validation.with_context(|| format!("resource {resource:?} is incompatible"))?;
    }

    Ok(())
}
