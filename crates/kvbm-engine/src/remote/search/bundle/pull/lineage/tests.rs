// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

use futures::executor::block_on;
use futures::future::BoxFuture;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{
    BundleKey, BundleResourceLineage, CacheIdentity, CacheManifest, ModelIdentity,
    ResourceRequirement, ResourceRole,
};
use kvbm_protocols::control::modules::transfer::{
    OpenTransferSessionResponse, TransferSessionCapability,
};
use kvbm_protocols::disagg::SessionEndpoint;

use super::super::super::BundleDirectoryError;
use super::super::transfer::{BundleTransfer, BundleTransferError};
use super::{CompleteG2Lineage, OpenedResource};
use crate::p2p::StagedPull;

const HISTORY: LogicalResourceId = LogicalResourceId(10);
const CAPSULE: LogicalResourceId = LogicalResourceId(11);

#[test]
fn complete_g2_lineage_requires_every_bundle_resource() {
    let (identity, key, lineages) = valid_bundle();

    let error = match CompleteG2Lineage::for_bundle(&identity, key, &lineages[..1]) {
        Ok(_) => panic!("a missing resource must not mint a complete G2 lineage"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        BundleDirectoryError::IncompleteResources {
            expected: vec![HISTORY, CAPSULE],
            actual: vec![HISTORY],
        }
    );
}

#[test]
fn complete_g2_lineage_rejects_a_caller_truncated_contract() {
    let (_, key, lineages) = valid_bundle();
    let truncated_identity = identity(vec![
        ResourceRequirement::new(HISTORY, ResourceRole::PrefixHistory, 4).unwrap(),
    ]);

    let error = match CompleteG2Lineage::for_bundle(&truncated_identity, key, &lineages[..1]) {
        Ok(_) => panic!("caller-selected requirements must not redefine bundle completeness"),
        Err(error) => error,
    };

    assert_eq!(error, BundleDirectoryError::ManifestMismatch);
}

#[test]
fn complete_g2_lineage_rejects_a_truncated_history() {
    let (identity, key, mut lineages) = valid_bundle();
    lineages[0] = BundleResourceLineage::new(HISTORY, vec![hash(0)]).unwrap();

    let error = match CompleteG2Lineage::for_bundle(&identity, key, &lineages) {
        Ok(_) => panic!("a truncated history must not mint a complete G2 lineage"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        BundleDirectoryError::InvalidResourceBlockCount {
            resource: HISTORY,
            role: ResourceRole::PrefixHistory,
            boundary_tokens: 8,
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn complete_g2_lineage_preserves_root_to_leaf_order() {
    let (identity, key, lineages) = valid_bundle();

    let complete = CompleteG2Lineage::for_bundle(&identity, key, &lineages).expect("valid bundle");

    assert_eq!(complete.len(), 2);
    assert_eq!(complete[&HISTORY].resource(), HISTORY);
    assert_eq!(complete[&HISTORY].hashes(), &[hash(0), hash(1)]);
    assert_eq!(complete[&CAPSULE].resource(), CAPSULE);
    assert_eq!(complete[&CAPSULE].hashes(), &[hash(1)]);
}

#[test]
fn full_lineage_transfer_adapter_never_opens_a_prefix() {
    let (identity, key, lineages) = valid_bundle();
    let complete = CompleteG2Lineage::for_bundle(&identity, key, &lineages).unwrap();
    let transfer = RecordingOpen::default();

    block_on(complete[&HISTORY].open_full(&transfer, Duration::from_millis(1)))
        .expect("record full lineage open");

    assert_eq!(
        transfer.opened.lock().unwrap().as_slice(),
        &[(HISTORY, vec![hash(0), hash(1)])]
    );
}

#[test]
fn opened_resource_retains_the_validated_complete_lineage() {
    let (identity, key, lineages) = valid_bundle();
    let complete = CompleteG2Lineage::for_bundle(&identity, key, &lineages).unwrap();
    let capability = TransferSessionCapability {
        session_id: uuid::Uuid::new_v4(),
        instance_id: uuid::Uuid::new_v4().into(),
        endpoint: SessionEndpoint {
            kind: "complete-lineage-test".to_owned(),
            payload: serde_json::Value::Null,
        },
        resource: HISTORY,
    };

    let opened = complete[&HISTORY]
        .bind_open(capability.clone(), &[hash(0), hash(1)])
        .expect("bind the complete committed lineage");

    assert_eq!(opened.capability(), &capability);
    assert_eq!(opened.resource(), HISTORY);
    assert_eq!(opened.hashes(), &[hash(0), hash(1)]);
}

#[test]
fn opened_resource_rejects_an_incomplete_or_mismatched_open() {
    let (identity, key, lineages) = valid_bundle();
    let complete = CompleteG2Lineage::for_bundle(&identity, key, &lineages).unwrap();
    let mut capability = TransferSessionCapability {
        session_id: uuid::Uuid::new_v4(),
        instance_id: uuid::Uuid::new_v4().into(),
        endpoint: SessionEndpoint {
            kind: "complete-lineage-test".to_owned(),
            payload: serde_json::Value::Null,
        },
        resource: HISTORY,
    };

    assert!(
        complete[&HISTORY]
            .bind_open(capability.clone(), &[hash(0)])
            .is_none()
    );

    capability.resource = CAPSULE;
    assert!(
        complete[&HISTORY]
            .bind_open(capability, &[hash(0), hash(1)])
            .is_none()
    );
}

fn valid_bundle() -> (CacheIdentity, BundleKey, Vec<BundleResourceLineage>) {
    let requirements = vec![
        ResourceRequirement::new(HISTORY, ResourceRole::PrefixHistory, 4).unwrap(),
        ResourceRequirement::new(CAPSULE, ResourceRole::BoundaryCapsule, 8).unwrap(),
    ];
    let identity = identity(requirements);
    let key = BundleKey::new(&identity, hash(1), 8).unwrap();
    let lineages = vec![
        BundleResourceLineage::new(HISTORY, vec![hash(0), hash(1)]).unwrap(),
        BundleResourceLineage::new(CAPSULE, vec![hash(1)]).unwrap(),
    ];
    (identity, key, lineages)
}

fn identity(requirements: Vec<ResourceRequirement>) -> CacheIdentity {
    CacheManifest::new(
        ModelIdentity::new("complete-g2-lineage-test", "v1", [4; 32]).unwrap(),
        "complete-g2-lineage-v1",
        requirements,
        BTreeMap::new(),
    )
    .unwrap()
    .identity()
}

fn hash(position: u64) -> SequenceHash {
    (1..=position).fold(SequenceHash::root(1), |parent, block| {
        parent.extend(block + 1)
    })
}

#[derive(Default)]
struct RecordingOpen {
    opened: Mutex<Vec<(LogicalResourceId, Vec<SequenceHash>)>>,
}

impl BundleTransfer for RecordingOpen {
    fn open(
        &self,
        resource: LogicalResourceId,
        hashes: Vec<SequenceHash>,
        _watchdog: Duration,
    ) -> BoxFuture<'_, Result<OpenTransferSessionResponse, BundleTransferError>> {
        self.opened.lock().unwrap().push((resource, hashes));
        Box::pin(async { Ok(OpenTransferSessionResponse::NoBlocksFound) })
    }

    fn pull(
        &self,
        _resource: &OpenedResource,
    ) -> BoxFuture<'_, Result<StagedPull, BundleTransferError>> {
        Box::pin(async { panic!("lineage open test does not pull") })
    }

    fn close(&self, _session_id: uuid::Uuid, _reason: &str) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}
