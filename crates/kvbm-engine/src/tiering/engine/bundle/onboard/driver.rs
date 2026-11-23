// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use kvbm_common::{BlockId, LogicalResourceId, SequenceHash};
use kvbm_logical::ImmutableBlock;
use kvbm_protocols::cache_manifest::{ResourceRequirement, ResourceRole};
use kvbm_protocols::connector::{
    ActionFailure, ActionStatus, BundleOnboardPlan, FindBlocksHandle, LeaderEngineError,
    OnboardHandle, RequestId, ResourceDestination, ResourceOnboard,
};
use tokio_util::sync::CancellationToken;

use super::{BundleOnboard, OnboardTransition};
use crate::G2;
use crate::tiering::engine::local::LocalConnectorEngine;
use crate::tiering::engine::onboard::{
    OnboardDrain, OnboardWatchdog, OnboardWork, dispatch_onboard, include_undispatched_destinations,
};
use crate::tiering::engine::paused::{
    BundleContinuation, LoadContinuation, PausedLoad, ResourceContinuation,
};

type ResourcePins = Vec<ImmutableBlock<G2>>;
type SourceLeases = BTreeMap<LogicalResourceId, ResourcePins>;

struct DispatchFailure {
    resource: LogicalResourceId,
    destination_block_ids: Vec<BlockId>,
}

struct CompletedResource {
    resource: LogicalResourceId,
    destination_block_ids: Vec<BlockId>,
    result: anyhow::Result<()>,
}

enum ResourceStop {
    Proven(Option<DispatchFailure>),
    Unproven {
        resource: LogicalResourceId,
        destination_block_ids: Vec<BlockId>,
        error: anyhow::Error,
    },
}

impl LocalConnectorEngine {
    pub(in crate::tiering::engine) fn start_bundle_onboard(
        self: Arc<Self>,
        req: &RequestId,
        plan: BundleOnboardPlan,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        validate_resource_transfers(self.as_ref(), &plan.resources)?;
        let (source_leases, inflight_hashes) = self.acquire_bundle_sources(&plan)?;
        self.start_bundle_onboard_with_sources(req, plan, source_leases, inflight_hashes, None)
    }

    pub(in crate::tiering::engine) fn start_searched_bundle_onboard(
        self: Arc<Self>,
        handle: &FindBlocksHandle,
        destinations: Vec<ResourceDestination>,
        num_external_tokens: usize,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        let search_id = handle
            .search_id()
            .ok_or(LeaderEngineError::SearchNotMatched)?;
        let state = self
            .bundle_searches
            .get(&search_id)
            .ok_or(LeaderEngineError::SearchNotMatched)?;
        if state.request_id != *handle.request_id() {
            return Err(LeaderEngineError::FindBlocksDesync);
        }
        if state.matched_tokens != num_external_tokens {
            return Err(LeaderEngineError::ExternalTokensMismatch {
                expected: state.matched_tokens,
                got: num_external_tokens,
            });
        }
        let (plan, source_leases, inflight_hashes) = searched_bundle_plan(&state, destinations)?;
        validate_resource_transfers(self.as_ref(), &plan.resources)?;
        let request_id = state.request_id.clone();
        drop(state);
        self.start_bundle_onboard_with_sources(
            &request_id,
            plan,
            source_leases,
            inflight_hashes,
            Some(search_id),
        )
    }

    pub(in crate::tiering::engine) fn start_resource_onboard(
        self: Arc<Self>,
        req: &RequestId,
        resources: Vec<ResourceOnboard>,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        if resources.is_empty() {
            return Err(LeaderEngineError::InvalidResourceOnboard {
                reason: "at least one resource is required".to_owned(),
            });
        }
        validate_resource_transfers(self.as_ref(), &resources)?;
        let admitted = self.admit_action(req, None, false)?;
        let (handle, load) = self.mint_onboard(
            admitted,
            destination_ids(&resources),
            Some(CancellationToken::new()),
            None,
        );
        self.spawn_resource_onboard(load, resources.into_iter());
        Ok(handle)
    }

    pub(in crate::tiering::engine) fn spawn_resource_onboard(
        self: &Arc<Self>,
        load: PausedLoad,
        resources: std::vec::IntoIter<ResourceOnboard>,
    ) {
        self.spawn_onboard_resources(load, resources, None, None);
    }

    fn start_bundle_onboard_with_sources(
        self: Arc<Self>,
        req: &RequestId,
        plan: BundleOnboardPlan,
        source_leases: SourceLeases,
        inflight_hashes: Vec<SequenceHash>,
        search: Option<kvbm_protocols::connector::SearchId>,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        let mut transaction =
            BundleOnboard::new(plan.identity, plan.key, source_leases).map_err(|error| {
                LeaderEngineError::InvalidBundleTransfer {
                    reason: error.to_string(),
                }
            })?;
        transaction.start();

        let admitted = self.admit_action(
            req,
            search.map(crate::tiering::engine::admission::AdmissionOffer::Search),
            false,
        )?;
        let previous = search.map(|id| self.bundle_searches.remove(&id));
        if matches!(previous, Some(None)) {
            drop(admitted);
            self.prune_request_actions(req);
            return Err(LeaderEngineError::SearchNotMatched);
        }
        let cancel = CancellationToken::new();
        let (handle, load) = self.mint_onboard(
            admitted,
            destination_ids(&plan.resources),
            Some(cancel.clone()),
            Some(inflight_hashes),
        );
        drop(previous);
        let watchdog = OnboardWatchdog {
            cancel,
            duration: self.bundle_onboard_watchdog(),
        };
        self.spawn_onboard_resources(
            load,
            plan.resources.into_iter(),
            Some(transaction),
            Some(watchdog),
        );
        Ok(handle)
    }

    fn spawn_onboard_resources(
        self: &Arc<Self>,
        load: PausedLoad,
        resources: std::vec::IntoIter<ResourceOnboard>,
        bundle: Option<BundleOnboard<ResourcePins>>,
        watchdog: Option<OnboardWatchdog>,
    ) {
        let engine = self.clone();
        self.spawn_onboard(
            load,
            async move { engine.run_resource_onboards(resources, bundle).await },
            watchdog,
        );
    }

    fn acquire_bundle_sources(
        &self,
        plan: &BundleOnboardPlan,
    ) -> Result<(SourceLeases, Vec<SequenceHash>), LeaderEngineError> {
        let lease = self
            .bundle_catalog
            .lock()
            .expect("bundle-catalog mutex poisoned")
            .lease_exact(&plan.identity, &plan.key)
            .ok_or_else(|| LeaderEngineError::InvalidBundleTransfer {
                reason: "bundle is not committed in the local index".to_owned(),
            })?;
        let mut leased_resources = lease.into_resources();
        let mut source_leases = BTreeMap::new();
        let mut inflight_hashes = Vec::new();
        for transfer in &plan.resources {
            let pins = leased_resources.remove(&transfer.resource).ok_or_else(|| {
                LeaderEngineError::InvalidBundleTransfer {
                    reason: format!(
                        "bundle lease is missing logical resource {:?}",
                        transfer.resource
                    ),
                }
            })?;
            let leased_ids = pins.iter().map(|pin| pin.block_id()).collect::<Vec<_>>();
            if leased_ids != transfer.source_block_ids {
                return Err(LeaderEngineError::InvalidBundleTransfer {
                    reason: format!(
                        "resource {:?} source ids do not match the committed bundle lease",
                        transfer.resource
                    ),
                });
            }
            inflight_hashes.extend(pins.iter().map(|pin| pin.sequence_hash()));
            source_leases.insert(transfer.resource, pins);
        }
        Ok((source_leases, inflight_hashes))
    }

    async fn run_resource_onboards(
        &self,
        mut resources: std::vec::IntoIter<ResourceOnboard>,
        bundle: Option<BundleOnboard<ResourcePins>>,
    ) -> OnboardWork {
        let (completed, stop) = self
            .execute_resource_onboards(&mut resources, &bundle)
            .await;
        match stop {
            ResourceStop::Proven(failure) => {
                let status = match bundle {
                    Some(transaction) => fold_bundle_onboard(transaction, failure, completed),
                    None => fold_resource_onboard(failure, completed),
                };
                OnboardWork::Proven(include_undispatched_destinations(
                    status,
                    resources.flat_map(|transfer| transfer.destination_block_ids),
                ))
            }
            ResourceStop::Unproven {
                resource,
                destination_block_ids,
                error,
            } => {
                if let Some(bundle) = bundle {
                    let unfilled = destination_ids(resources.as_slice());
                    let continuation = LoadContinuation::Bundle(BundleContinuation {
                        failure: ActionFailure::Resource {
                            resource,
                            block_ids: Some(destination_block_ids),
                        },
                        unfilled,
                        logical_done: false,
                    });
                    return OnboardWork::unproven(
                        error,
                        bundle.into_paused_reservation(resource),
                        Some(continuation),
                    );
                }

                let failure = ActionFailure::Resource {
                    resource,
                    block_ids: Some(destination_block_ids),
                };
                let continuation = LoadContinuation::Resource(ResourceContinuation {
                    failure,
                    remaining: resources,
                });
                OnboardWork::unproven(error, (), Some(continuation))
            }
        }
    }

    async fn execute_resource_onboards(
        &self,
        resources: &mut std::vec::IntoIter<ResourceOnboard>,
        bundle: &Option<BundleOnboard<ResourcePins>>,
    ) -> (Vec<CompletedResource>, ResourceStop) {
        let _admission = self.resource_onboard_admission.lock().await;
        let mut completed = Vec::with_capacity(resources.len());
        for transfer in resources {
            let resource = transfer.resource;
            let destination_block_ids = transfer.destination_block_ids.clone();
            let result = match dispatch_onboard(&self.leader, transfer, bundle).await {
                Ok(OnboardDrain::Proven(result)) => result,
                Ok(OnboardDrain::Unproven { error, .. }) => {
                    return (
                        completed,
                        ResourceStop::Unproven {
                            resource,
                            destination_block_ids,
                            error,
                        },
                    );
                }
                Err(error) => {
                    tracing::error!(%error, ?resource, "resource onboard dispatch failed");
                    return (
                        completed,
                        ResourceStop::Proven(Some(DispatchFailure {
                            resource,
                            destination_block_ids,
                        })),
                    );
                }
            };
            let failed = result.is_err();
            completed.push(CompletedResource {
                resource,
                destination_block_ids,
                result,
            });
            if failed {
                break;
            }
        }
        (completed, ResourceStop::Proven(None))
    }
}

fn destination_ids(resources: &[ResourceOnboard]) -> Vec<BlockId> {
    resources
        .iter()
        .flat_map(|transfer| transfer.destination_block_ids.iter().copied())
        .collect()
}

fn fold_resource_onboard(
    dispatch_failure: Option<DispatchFailure>,
    completed: Vec<CompletedResource>,
) -> ActionStatus {
    if let Some(failure) = dispatch_failure {
        return ActionStatus::Failed(ActionFailure::Resource {
            resource: failure.resource,
            block_ids: Some(failure.destination_block_ids),
        });
    }
    for completed in completed {
        if let Err(error) = completed.result {
            tracing::error!(
                %error,
                resource = ?completed.resource,
                "same-request resource onboard transfer failed"
            );
            return ActionStatus::Failed(ActionFailure::Resource {
                resource: completed.resource,
                block_ids: Some(completed.destination_block_ids),
            });
        }
    }
    ActionStatus::Complete
}

fn searched_bundle_plan(
    state: &crate::tiering::engine::local::BundleSearchState,
    destinations: Vec<ResourceDestination>,
) -> Result<(BundleOnboardPlan, SourceLeases, Vec<SequenceHash>), LeaderEngineError> {
    let mut destination_map = BTreeMap::new();
    for destination in destinations {
        if destination_map
            .insert(destination.resource, destination.block_ids)
            .is_some()
        {
            return Err(invalid_bundle(format!(
                "duplicate destination for {:?}",
                destination.resource
            )));
        }
    }
    let mut destinations = destination_map;
    if destinations.len() != state.identity.resources().len() {
        return Err(invalid_bundle(
            "resource destinations do not match the manifest",
        ));
    }

    let lease = state
        .lease()
        .ok_or_else(|| invalid_bundle("bundle search has no committed lease"))?;
    let boundary = usize::try_from(lease.key().boundary_tokens())
        .map_err(|_| invalid_bundle("bundle boundary does not fit usize"))?;
    let computed = state.computed_tokens;
    let mut resources = Vec::with_capacity(state.identity.resources().len());
    let mut source_leases = BTreeMap::new();
    let mut inflight_hashes = Vec::new();
    for requirement in state.identity.resources() {
        let resource = requirement.resource();
        let destination = destinations
            .remove(&resource)
            .ok_or_else(|| invalid_bundle(format!("missing destination for {resource:?}")))?;
        let source = lease
            .resources()
            .get(&resource)
            .ok_or_else(|| invalid_bundle(format!("search lease is missing {resource:?}")))?;
        let block_range = searched_resource_block_range(
            requirement,
            computed,
            boundary,
            source.len(),
            destination.len(),
        )?;
        let pins = source[block_range.clone()].to_vec();
        let source_block_ids = pins.iter().map(ImmutableBlock::block_id).collect();
        inflight_hashes.extend(pins.iter().map(ImmutableBlock::sequence_hash));
        resources.push(ResourceOnboard {
            resource,
            source_block_ids,
            destination_block_ids: destination[block_range].to_vec(),
        });
        source_leases.insert(resource, pins);
    }
    Ok((
        BundleOnboardPlan {
            identity: state.identity.clone(),
            key: *lease.key(),
            resources,
        },
        source_leases,
        inflight_hashes,
    ))
}

fn searched_resource_block_range(
    requirement: &ResourceRequirement,
    computed: usize,
    boundary: usize,
    source_blocks: usize,
    destination_blocks: usize,
) -> Result<std::ops::Range<usize>, LeaderEngineError> {
    let resource = requirement.resource();
    match requirement.role() {
        ResourceRole::PrefixHistory => {
            let native = requirement.native_block_tokens().get() as usize;
            let first_block = computed / native;
            let end_block = boundary.div_ceil(native);
            if first_block >= end_block
                || source_blocks < end_block
                || destination_blocks < end_block
            {
                return Err(invalid_bundle(format!(
                    "resource {resource:?} cannot cover native block range {first_block}..{end_block}"
                )));
            }
            Ok(first_block..end_block)
        }
        ResourceRole::BoundaryCapsule => {
            if source_blocks != 1 || destination_blocks != 1 {
                return Err(invalid_bundle(format!(
                    "boundary capsule {resource:?} requires exactly one source and one destination; got {source_blocks} and {destination_blocks}"
                )));
            }
            Ok(0..1)
        }
    }
}

fn invalid_bundle(reason: impl Into<String>) -> LeaderEngineError {
    LeaderEngineError::InvalidBundleTransfer {
        reason: reason.into(),
    }
}

fn validate_resource_transfers(
    engine: &LocalConnectorEngine,
    resources: &[ResourceOnboard],
) -> Result<(), LeaderEngineError> {
    if resources.is_empty() {
        return Err(LeaderEngineError::InvalidBundleTransfer {
            reason: "at least one resource is required".to_owned(),
        });
    }
    let mut seen = HashSet::new();
    for transfer in resources {
        if !seen.insert(transfer.resource) {
            return Err(LeaderEngineError::InvalidResourceOnboard {
                reason: format!("duplicate logical resource {:?}", transfer.resource),
            });
        }
        if transfer.source_block_ids.is_empty()
            || transfer.source_block_ids.len() != transfer.destination_block_ids.len()
        {
            return Err(LeaderEngineError::InvalidResourceOnboard {
                reason: format!(
                    "resource {:?} has {} G2 sources and {} G1 destinations",
                    transfer.resource,
                    transfer.source_block_ids.len(),
                    transfer.destination_block_ids.len()
                ),
            });
        }
        if engine.leader.g2_manager_for(transfer.resource).is_none() {
            return Err(LeaderEngineError::ResourceOnboardNotConfigured {
                resource: transfer.resource,
            });
        }
    }
    Ok(())
}

fn fold_bundle_onboard(
    mut transaction: BundleOnboard<ResourcePins>,
    dispatch_failure: Option<DispatchFailure>,
    completed: Vec<CompletedResource>,
) -> ActionStatus {
    let mut outcome = match dispatch_failure.as_ref() {
        Some(failure) => ActionStatus::Failed(ActionFailure::Resource {
            resource: failure.resource,
            block_ids: Some(failure.destination_block_ids.clone()),
        }),
        None => ActionStatus::Complete,
    };
    if let Some(failure) = dispatch_failure {
        let _ = transaction.fail(failure.resource, Some(failure.destination_block_ids));
    }
    for completed in completed {
        let transition = match completed.result {
            Ok(()) => transaction.complete(completed.resource),
            Err(error) => {
                tracing::error!(
                    error = %error,
                    resource = ?completed.resource,
                    "resource onboard transfer failed"
                );
                transaction.fail(
                    completed.resource,
                    Some(completed.destination_block_ids.clone()),
                )
            }
        };
        match transition {
            Ok(OnboardTransition::Visible(visible)) => {
                tracing::trace!(
                    computed_tokens = visible.computed_tokens(),
                    "bundle onboard crossed all-resource visibility barrier"
                );
                outcome = ActionStatus::Complete;
            }
            Ok(OnboardTransition::Aborted(failure)) => {
                outcome = ActionStatus::Failed(ActionFailure::Resource {
                    resource: failure.resource(),
                    block_ids: failure.failed_blocks().map(<[usize]>::to_vec),
                });
            }
            Ok(OnboardTransition::Pending | OnboardTransition::Settled(_)) => {}
            Err(error) => {
                tracing::error!(
                    %error,
                    resource = ?completed.resource,
                    "bundle onboard completion fold failed"
                );
                outcome = ActionStatus::Failed(ActionFailure::Resource {
                    resource: completed.resource,
                    block_ids: Some(completed.destination_block_ids),
                });
            }
        }
    }
    outcome
}
