// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parallelism templates owned by logical KV resource.

use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use kvbm_common::LogicalResourceId;

use super::ParallelismTemplate;

/// Resource-keyed templates sharing one physical worker grid.
#[derive(Debug, Clone)]
pub struct ParallelismTemplateSet {
    primary: LogicalResourceId,
    templates: BTreeMap<LogicalResourceId, ParallelismTemplate>,
}

impl ParallelismTemplateSet {
    pub fn new(
        primary: LogicalResourceId,
        templates: Vec<(LogicalResourceId, ParallelismTemplate)>,
    ) -> Result<Self> {
        let expected_len = templates.len();
        let templates = templates.into_iter().collect::<BTreeMap<_, _>>();
        ensure!(
            templates.len() == expected_len,
            "duplicate logical resource in parallelism template set"
        );
        let primary_template = templates.get(&primary).ok_or_else(|| {
            anyhow::anyhow!("primary logical resource {primary:?} has no parallelism template")
        })?;
        let worker_grid = (primary_template.tp_size, primary_template.pp_size);
        ensure!(
            templates
                .values()
                .all(|template| { (template.tp_size, template.pp_size) == worker_grid }),
            "every resource parallelism template must use worker grid {}x{}",
            worker_grid.0,
            worker_grid.1
        );
        Ok(Self { primary, templates })
    }

    pub fn primary(&self) -> LogicalResourceId {
        self.primary
    }

    pub fn get(&self, resource: LogicalResourceId) -> Option<&ParallelismTemplate> {
        self.templates.get(&resource)
    }

    pub fn iter(&self) -> impl Iterator<Item = (LogicalResourceId, &ParallelismTemplate)> + '_ {
        self.templates
            .iter()
            .map(|(&resource, template)| (resource, template))
    }

    pub fn worker_count(&self) -> usize {
        let primary = self
            .get(self.primary)
            .expect("parallelism template set retains its validated primary");
        primary.tp_size * primary.pp_size
    }
}
