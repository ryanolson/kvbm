// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parallelism and physical placement keyed by logical KV resource.

use std::collections::BTreeSet;

use anyhow::{Result, ensure};
use bincode::{Decode, Encode};
use kvbm_common::LogicalResourceId;

use super::{ParallelismDescriptor, WorkerDataPlacement};

/// One resource's per-worker parallelism and physical placement.
#[derive(Debug, Clone, Encode, Decode, PartialEq, Eq)]
pub struct ResourceParallelismDescriptor {
    #[bincode(with_serde)]
    pub resource: LogicalResourceId,
    pub parallelism: ParallelismDescriptor,
    pub placement: WorkerDataPlacement,
}

impl ResourceParallelismDescriptor {
    pub fn new(
        resource: LogicalResourceId,
        parallelism: ParallelismDescriptor,
        placement: WorkerDataPlacement,
    ) -> Self {
        Self {
            resource,
            parallelism,
            placement,
        }
    }
}

/// Deterministic resource map with one selected compatibility primary.
#[derive(Debug, Clone, Encode, Decode, PartialEq, Eq)]
pub struct ResourceParallelismDescriptors {
    #[bincode(with_serde)]
    primary: LogicalResourceId,
    resources: Vec<ResourceParallelismDescriptor>,
}

impl ResourceParallelismDescriptors {
    pub fn new(
        primary: LogicalResourceId,
        mut resources: Vec<ResourceParallelismDescriptor>,
    ) -> Result<Self> {
        resources.sort_by_key(|entry| entry.resource);
        let descriptors = Self { primary, resources };
        descriptors.validate()?;
        Ok(descriptors)
    }

    pub fn primary(&self) -> LogicalResourceId {
        self.primary
    }

    pub fn get(&self, resource: LogicalResourceId) -> Option<&ResourceParallelismDescriptor> {
        self.resources
            .binary_search_by_key(&resource, |entry| entry.resource)
            .ok()
            .map(|index| &self.resources[index])
    }

    pub fn iter(&self) -> impl Iterator<Item = &ResourceParallelismDescriptor> {
        self.resources.iter()
    }

    pub(super) fn validate(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for entry in &self.resources {
            ensure!(
                seen.insert(entry.resource),
                "duplicate logical resource {:?} in parallelism metadata",
                entry.resource
            );
        }
        ensure!(
            self.resources
                .windows(2)
                .all(|pair| pair[0].resource < pair[1].resource),
            "logical resources in parallelism metadata are not strictly ordered"
        );
        ensure!(
            seen.contains(&self.primary),
            "primary logical resource {:?} is absent from parallelism metadata",
            self.primary
        );
        Ok(())
    }
}
