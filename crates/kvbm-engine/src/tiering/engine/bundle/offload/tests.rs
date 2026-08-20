// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::disallowed_macros)]

use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, Weak};

use anyhow::Result;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::manager::InactiveBackendConfig;
use kvbm_logical::{BlockManagerSet, BlockRegistry};
use kvbm_protocols::cache_manifest::BundleKey;
use kvbm_protocols::connector::{
    ActionFailure, ActionStatus, BundleOffloadPlan, EngineWorkerSink, FenceToken, LeaderEngine,
    LoadOutcome, RequestId, ResourceOffload, SaveOutcome,
};

use super::{BundleOffload, BundleOffloadState, OffloadTransition};
use crate::tiering::engine::bundle::BundleAdmissionConfig;
use crate::tiering::engine::bundle::test_support::{CAPSULE, CSA, HCA, bundle_key, identity};
use crate::tiering::engine::local::LocalConnectorEngine;
use crate::tiering::engine::offload::{BufferedOffload, OffloadSubmit, OffloadTransfer};
use crate::tiering::policy::{ResourceComponentBytes, ResourcePolicies, ResourcePolicy};
use crate::{G1, G2};
use kvbm_protocols::connector::OffloadMode;

#[derive(Default)]
struct RecordingSaveSink {
    saves: Mutex<Vec<(RequestId, SaveOutcome)>>,
}

impl RecordingSaveSink {
    fn saves(&self) -> Vec<(RequestId, SaveOutcome)> {
        self.saves.lock().unwrap().clone()
    }
}

impl EngineWorkerSink for RecordingSaveSink {
    fn mark_load_finished(&self, _request: &RequestId, _outcome: LoadOutcome) {}

    fn mark_save_finished(&self, request: &RequestId, outcome: SaveOutcome) {
        self.saves.lock().unwrap().push((request.clone(), outcome));
    }

    fn mark_fence_complete(&self, _token: FenceToken) {}
}

struct BufferOnlySubmit;

impl OffloadSubmit for BufferOnlySubmit {
    fn supports_resource(&self, _resource: LogicalResourceId) -> bool {
        true
    }

    fn submit_g1_to_g2(
        &self,
        _resource: Option<LogicalResourceId>,
        _blocks: Vec<crate::offload::ExternalBlock<G1>>,
        _precondition: Option<velo::EventHandle>,
    ) -> Result<Box<dyn OffloadTransfer>> {
        unreachable!("this regression drives buffered child terminals directly")
    }
}

async fn bundle_engine() -> Result<(Arc<LocalConnectorEngine>, Arc<RecordingSaveSink>)> {
    let identity = identity();
    let mut managers = BlockManagerSet::new();
    for requirement in identity.resources() {
        let manager = Arc::new(
            crate::testing::managers::TestManagerBuilder::<G2>::new()
                .block_count(2)
                .block_size(usize::try_from(requirement.native_block_tokens().get())?)
                .registry(BlockRegistry::new())
                .build(),
        );
        managers.insert(requirement.resource(), manager)?;
    }
    let leader = Arc::new(
        crate::leader::InstanceLeader::builder()
            .messenger(crate::testing::messenger::create_messenger_tcp().await?)
            .registry(BlockRegistry::new())
            .g2_manager_set(Arc::new(managers), CSA)
            .build()?,
    );
    let mut policies = ResourcePolicies::new();
    let mut component_bytes = ResourceComponentBytes::new();
    for requirement in identity.resources() {
        policies.insert(
            requirement.resource(),
            ResourcePolicy::new(
                requirement.role(),
                InactiveBackendConfig::default(),
                InactiveBackendConfig::default(),
            ),
        )?;
        component_bytes.insert(requirement.resource(), [NonZeroU64::MIN])?;
    }
    let sink = Arc::new(RecordingSaveSink::default());
    let engine = LocalConnectorEngine::with_offload_submit_and_admission(
        leader,
        sink.clone(),
        256,
        true,
        Arc::new(BufferOnlySubmit),
        None,
        BundleAdmissionConfig::new(policies, component_bytes),
    );
    Ok((engine, sink))
}

fn pins() -> (BTreeMap<LogicalResourceId, Arc<()>>, Vec<Weak<()>>) {
    let pins = [CSA, HCA, CAPSULE]
        .into_iter()
        .map(|resource| (resource, Arc::new(())))
        .collect::<BTreeMap<_, _>>();
    let weak = pins.values().map(Arc::downgrade).collect();
    (pins, weak)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_bundle_handle_holds_drain_until_every_child_settles() -> Result<()> {
    let (engine, sink) = bundle_engine().await?;
    let identity = identity();
    let boundary_hash = SequenceHash::root(7);
    let key = BundleKey::new(&identity, boundary_hash, 256)?;
    let request: RequestId = "dropped-bundle".into();
    let handle = engine.clone().offload_bundle(
        &request,
        BundleOffloadPlan {
            identity,
            key,
            mode: OffloadMode::Move,
            resources: [CSA, HCA, CAPSULE]
                .into_iter()
                .enumerate()
                .map(|(index, resource)| ResourceOffload {
                    resource,
                    blocks: vec![(boundary_hash, 20 + index)],
                })
                .collect(),
        },
    )?;
    let action_id = *handle.id();
    let children = {
        let mut buffer = engine
            .offload_buffer
            .lock()
            .expect("offload-buffer mutex poisoned");
        std::mem::take(&mut *buffer)
    };
    assert_eq!(children.len(), 3);

    drop(handle);
    engine
        .take_offload_drain(&request)
        .expect("bundle offload registered a drain")
        .commit();
    assert!(sink.saves().is_empty());

    let child_count = children.len();
    for (index, child) in children.into_iter().enumerate() {
        let BufferedOffload {
            action_id: child_action_id,
            request_id,
            resource,
            pairs,
            completion,
            ..
        } = child;
        assert_eq!(child_action_id, action_id);
        engine.finish_offload_child(
            child_action_id,
            &request_id,
            resource,
            pairs,
            None,
            completion,
            ActionStatus::Failed(ActionFailure::AllBlocks),
        );

        if index + 1 < child_count {
            assert!(
                sink.saves().is_empty(),
                "a partial bundle terminal must retain the request drain"
            );
            assert!(engine.actions.contains_key(&action_id));
        }
    }

    assert_eq!(sink.saves(), vec![(request.clone(), SaveOutcome::Done)]);
    assert!(!engine.actions.contains_key(&action_id));
    assert!(!engine.by_request.contains_key(&request));
    Ok(())
}

#[test]
fn offload_publishes_once_only_after_every_resource_completes() {
    let (sources, _) = pins();
    let mut offload =
        BundleOffload::new(identity(), bundle_key(512), 7, OffloadMode::Mirror, sources).unwrap();
    assert_eq!(offload.start(), BundleOffloadState::Transferring);

    assert!(matches!(
        offload.complete(CSA, Ok(Arc::new(()))).unwrap(),
        OffloadTransition::Pending
    ));
    assert!(matches!(
        offload.complete(CAPSULE, Ok(Arc::new(()))).unwrap(),
        OffloadTransition::Pending
    ));
    let OffloadTransition::Commit(commit) = offload.complete(HCA, Ok(Arc::new(()))).unwrap() else {
        panic!("the final resource must publish one commit");
    };
    assert_eq!(commit.resources().len(), 3);
    assert_eq!(commit.key(), &bundle_key(512));
    assert_eq!(commit.generation(), 7);
    assert_eq!(commit.retained_sources().unwrap().len(), 3);

    assert_eq!(
        offload.complete(HCA, Ok(Arc::new(()))).unwrap(),
        OffloadTransition::Settled(BundleOffloadState::Committed),
        "a duplicate completion cannot publish twice"
    );
}

#[test]
fn failed_resource_aborts_staging_and_retains_every_source_pin() {
    let (sources, source_weak) = pins();
    let staged = Arc::new(());
    let staged_weak = Arc::downgrade(&staged);
    let mut offload =
        BundleOffload::new(identity(), bundle_key(512), 9, OffloadMode::Move, sources).unwrap();
    offload.start();
    assert!(matches!(
        offload.complete(CSA, Ok(staged)).unwrap(),
        OffloadTransition::Pending
    ));

    let OffloadTransition::Abort(abort) = offload.fail(HCA, Some(vec![4, 8])).unwrap() else {
        panic!("one failed child must abort the bundle");
    };
    assert_eq!(abort.failure().resource(), HCA);
    assert_eq!(abort.failure().failed_blocks(), Some(&[4, 8][..]));
    assert!(
        staged_weak.upgrade().is_none(),
        "staged pins must be dropped"
    );
    assert!(source_weak.iter().all(|pin| pin.upgrade().is_some()));
    assert_eq!(abort.retained_sources().len(), 3);
}

#[test]
fn move_releases_sources_only_after_commit_while_mirror_retains_them() {
    let (move_sources, move_weak) = pins();
    let mut moving = BundleOffload::new(
        identity(),
        bundle_key(256),
        1,
        OffloadMode::Move,
        move_sources,
    )
    .unwrap();
    moving.start();
    moving.complete(CSA, Ok(Arc::new(()))).unwrap();
    moving.complete(HCA, Ok(Arc::new(()))).unwrap();
    assert!(move_weak.iter().all(|pin| pin.upgrade().is_some()));
    let OffloadTransition::Commit(commit) = moving.complete(CAPSULE, Ok(Arc::new(()))).unwrap()
    else {
        panic!("all resources should commit");
    };
    assert!(commit.retained_sources().is_none());
    assert!(move_weak.iter().all(|pin| pin.upgrade().is_none()));

    let (mirror_sources, mirror_weak) = pins();
    let mut mirroring = BundleOffload::new(
        identity(),
        bundle_key(256),
        2,
        OffloadMode::Mirror,
        mirror_sources,
    )
    .unwrap();
    mirroring.start();
    mirroring.complete(CSA, Ok(Arc::new(()))).unwrap();
    mirroring.complete(HCA, Ok(Arc::new(()))).unwrap();
    let OffloadTransition::Commit(commit) = mirroring.complete(CAPSULE, Ok(Arc::new(()))).unwrap()
    else {
        panic!("all resources should commit");
    };
    assert!(mirror_weak.iter().all(|pin| pin.upgrade().is_some()));
    drop(commit);
    assert!(mirror_weak.iter().all(|pin| pin.upgrade().is_none()));
}
