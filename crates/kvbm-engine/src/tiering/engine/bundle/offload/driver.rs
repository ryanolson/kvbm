// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use kvbm_common::{BlockId, LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{
    BundleResourceLineage, ResourceRole, validate_bundle_lineages,
};
use kvbm_protocols::connector::{
    ActionId, ActionStatus, BundleOffloadPlan, LeaderEngine, LeaderEngineError, OffloadHandle,
    RequestId,
};

use crate::tiering::engine::driver::ActionRecord;
use crate::tiering::engine::local::LocalConnectorEngine;
use crate::tiering::engine::offload::{
    BufferedOffload, BufferedOffloadCompletion, BundleOffloadRuntime, LocalBundleOffload,
};
use crate::tiering::policy::ResourceLineage;

type SourceBlocksByResource = BTreeMap<LogicalResourceId, Vec<(SequenceHash, BlockId)>>;

impl LocalConnectorEngine {
    pub(in crate::tiering::engine) fn start_bundle_offload(
        self: Arc<Self>,
        req: &RequestId,
        plan: BundleOffloadPlan,
    ) -> Result<OffloadHandle, LeaderEngineError> {
        let sources = self.validate_bundle_offload(&plan)?;
        let planned_bytes = self.admit_bundle(&plan)?;
        let lineages = bundle_lineages(&plan)?;
        let generation = self
            .leader
            .reserve_bundle_publication_generation()
            .map_err(|error| LeaderEngineError::InvalidBundleTransfer {
                reason: error.to_string(),
            })?;
        let mut transaction =
            LocalBundleOffload::new(plan.identity, plan.key, generation, plan.mode, sources)
                .map_err(|error| LeaderEngineError::InvalidBundleTransfer {
                    reason: error.to_string(),
                })?;
        transaction.start();
        let child_count = NonZeroUsize::new(plan.resources.len()).ok_or_else(|| {
            LeaderEngineError::InvalidBundleTransfer {
                reason: "at least one resource is required".to_owned(),
            }
        })?;
        let runtime = Arc::new(BundleOffloadRuntime::new(
            transaction,
            child_count,
            lineages,
        ));

        let action_id = ActionId::new();
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        self.actions.insert(
            action_id,
            ActionRecord::new_save(req.clone(), Arc::downgrade(&cell)),
        );
        self.by_request
            .entry(req.clone())
            .or_default()
            .push(action_id);
        self.offload_drains.insert(req.clone(), ());

        let iteration = self
            .current_iteration
            .load(std::sync::atomic::Ordering::Relaxed);
        self.offload_buffer
            .lock()
            .expect("offload-buffer mutex poisoned")
            .extend(plan.resources.into_iter().map(|child| BufferedOffload {
                action_id,
                request_id: req.clone(),
                resource: Some(child.resource),
                pairs: child.blocks,
                planned_bytes: Some(planned_bytes[&child.resource]),
                iteration,
                completion: BufferedOffloadCompletion::Bundle(Arc::clone(&runtime)),
            }));

        let engine: Arc<dyn LeaderEngine> = self;
        Ok(OffloadHandle::new(action_id, Arc::downgrade(&engine), cell))
    }

    fn validate_bundle_offload(
        &self,
        plan: &BundleOffloadPlan,
    ) -> Result<SourceBlocksByResource, LeaderEngineError> {
        if plan.resources.is_empty() {
            return Err(LeaderEngineError::InvalidBundleTransfer {
                reason: "at least one resource is required".to_owned(),
            });
        }
        if !plan.key.is_compatible_with(&plan.identity) {
            return Err(LeaderEngineError::InvalidBundleTransfer {
                reason: "bundle key is incompatible with its cache identity".to_owned(),
            });
        }
        let primary_resource = self.leader.primary_g2_resource();
        let primary_requirement = plan
            .identity
            .resources()
            .iter()
            .find(|requirement| requirement.resource() == primary_resource)
            .ok_or_else(|| LeaderEngineError::InvalidBundleTransfer {
                reason: format!(
                    "primary resource {primary_resource:?} is absent from the cache identity"
                ),
            })?;
        if primary_requirement.role() != ResourceRole::PrefixHistory {
            return Err(LeaderEngineError::InvalidBundleTransfer {
                reason: format!(
                    "primary resource {primary_resource:?} must be a prefix history, got {:?}",
                    primary_requirement.role()
                ),
            });
        }
        let mut sources = BTreeMap::new();
        for child in &plan.resources {
            if child.blocks.is_empty() {
                return Err(LeaderEngineError::InvalidBundleTransfer {
                    reason: format!("resource {:?} contains no source blocks", child.resource),
                });
            }
            if sources
                .insert(child.resource, child.blocks.clone())
                .is_some()
            {
                return Err(LeaderEngineError::InvalidBundleTransfer {
                    reason: format!("duplicate logical resource {:?}", child.resource),
                });
            }
            if !self.offload_submit.supports_resource(child.resource) {
                return Err(LeaderEngineError::ResourceOffloadNotConfigured {
                    resource: child.resource,
                });
            }
            let requirement = plan
                .identity
                .resources()
                .iter()
                .find(|requirement| requirement.resource() == child.resource)
                .ok_or_else(|| LeaderEngineError::InvalidBundleTransfer {
                    reason: format!(
                        "resource {:?} is absent from the cache identity",
                        child.resource
                    ),
                })?;
            let expected_blocks = match requirement.role() {
                ResourceRole::PrefixHistory => plan
                    .key
                    .boundary_tokens()
                    .checked_div(u64::from(requirement.native_block_tokens().get()))
                    .and_then(|count| usize::try_from(count).ok()),
                ResourceRole::BoundaryCapsule => Some(1),
            }
            .ok_or_else(|| LeaderEngineError::InvalidBundleTransfer {
                reason: format!("resource {:?} block count overflows", child.resource),
            })?;
            if child.blocks.len() != expected_blocks {
                return Err(LeaderEngineError::InvalidBundleTransfer {
                    reason: format!(
                        "resource {:?} requires {expected_blocks} blocks at boundary {}, got {}",
                        child.resource,
                        plan.key.boundary_tokens(),
                        child.blocks.len()
                    ),
                });
            }
            let requires_canonical_boundary = child.resource == primary_resource
                || requirement.role() == ResourceRole::BoundaryCapsule;
            if requires_canonical_boundary
                && child.blocks.last().map(|(hash, _)| *hash) != Some(plan.key.boundary_hash())
            {
                return Err(LeaderEngineError::InvalidBundleTransfer {
                    reason: format!(
                        "primary history or boundary capsule {:?} does not end at the bundle boundary hash",
                        child.resource,
                    ),
                });
            }
        }
        Ok(sources)
    }
}

fn bundle_lineages(plan: &BundleOffloadPlan) -> Result<Vec<ResourceLineage>, LeaderEngineError> {
    let exact = plan
        .resources
        .iter()
        .map(|child| {
            BundleResourceLineage::new(
                child.resource,
                child.blocks.iter().map(|(hash, _)| *hash).collect(),
            )
            .map_err(|error| LeaderEngineError::InvalidBundleTransfer {
                reason: error.to_string(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_bundle_lineages(plan.key, plan.identity.resources(), &exact).map_err(|error| {
        LeaderEngineError::InvalidBundleTransfer {
            reason: error.to_string(),
        }
    })?;

    exact
        .into_iter()
        .map(|lineage| {
            let role = plan
                .identity
                .resources()
                .iter()
                .find(|requirement| requirement.resource() == lineage.resource())
                .map(|requirement| requirement.role())
                .ok_or_else(|| LeaderEngineError::InvalidBundleTransfer {
                    reason: format!(
                        "resource {:?} is absent from the cache identity",
                        lineage.resource()
                    ),
                })?;
            Ok(ResourceLineage::new(
                lineage.resource(),
                role,
                lineage.hashes().to_vec(),
            ))
        })
        .collect()
}
