// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::BlockManager;
use kvbm_logical::blocks::{CompleteBlock, ImmutableBlock};

use super::StagedBundle;
use crate::G2;
use crate::g2_capacity::{
    G2Capacity, G2CapacityDecision, G2CapacityError, G2CapacityRequest, G2CapacityRequirement,
    G2ExactAllocation, G2ExactRegistrationOwner, G2StagedAllocation, direct_g2_capacity,
    reserve_required_staging,
};
use crate::p2p::StagedPull;
use crate::testing::managers::TestManagerBuilder;

struct RejectRegistrationCapacity {
    manager: Arc<BlockManager<G2>>,
}

struct ExactTestCapacity {
    manager: Arc<BlockManager<G2>>,
    fail_registration: bool,
    rollbacks: Arc<AtomicUsize>,
}

struct ExactTestOwner {
    manager: Arc<BlockManager<G2>>,
    fail_registration: bool,
    rollbacks: Arc<AtomicUsize>,
}

impl G2ExactRegistrationOwner for ExactTestOwner {
    fn register_blocks(
        &mut self,
        blocks: Vec<CompleteBlock<G2>>,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        if self.fail_registration {
            return Err(G2CapacityError::Rejected(
                "forced exact registration failure".to_owned(),
            ));
        }
        Ok(self.manager.register_blocks(blocks))
    }

    fn rollback_required_staging(&mut self, blocks: Vec<ImmutableBlock<G2>>) {
        self.rollbacks.fetch_add(1, Ordering::Relaxed);
        self.manager.release_blocks(blocks, Some(true));
    }

    fn retain_cache_extension(self: Box<Self>) {}
}

impl G2Capacity for ExactTestCapacity {
    fn reserve(&self, request: G2CapacityRequest) -> Result<G2CapacityDecision, G2CapacityError> {
        if request.requirement() != G2CapacityRequirement::ExactReclaim {
            return Err(G2CapacityError::RequirementMismatch {
                expected: G2CapacityRequirement::ExactReclaim,
                actual: request.requirement(),
            });
        }
        let blocks = self
            .manager
            .allocate_blocks(request.count())
            .ok_or(G2CapacityError::Unavailable(request))?;
        G2ExactAllocation::new(
            request,
            blocks,
            Box::new(ExactTestOwner {
                manager: Arc::clone(&self.manager),
                fail_registration: self.fail_registration,
                rollbacks: Arc::clone(&self.rollbacks),
            }),
        )
        .map(G2CapacityDecision::ExactGranted)
    }

    fn block_size(&self) -> usize {
        self.manager.block_size()
    }

    fn manager_id(&self) -> kvbm_logical::ManagerId {
        self.manager.id()
    }

    fn register_compatibility(
        &self,
        _allocation: G2StagedAllocation,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        Err(G2CapacityError::RequirementMismatch {
            expected: G2CapacityRequirement::ExactReclaim,
            actual: G2CapacityRequirement::Compatibility,
        })
    }

    fn match_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.manager.match_blocks(hashes)
    }

    fn match_inactive_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.manager.match_inactive_blocks(hashes)
    }

    fn has_any_registered_hashes(&self, hashes: &[SequenceHash]) -> bool {
        self.manager.has_any_registered_hashes(hashes)
    }

    fn scan_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<G2>> {
        self.manager.scan_matches(hashes, touch)
    }
}

impl G2Capacity for RejectRegistrationCapacity {
    fn reserve(&self, _request: G2CapacityRequest) -> Result<G2CapacityDecision, G2CapacityError> {
        Err(G2CapacityError::Rejected(
            "test capacity does not reserve".to_owned(),
        ))
    }

    fn block_size(&self) -> usize {
        self.manager.block_size()
    }

    fn manager_id(&self) -> kvbm_logical::ManagerId {
        self.manager.id()
    }

    fn register_compatibility(
        &self,
        _allocation: G2StagedAllocation,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        Err(G2CapacityError::Rejected(
            "forced second resource registration failure".to_owned(),
        ))
    }

    fn match_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.manager.match_blocks(hashes)
    }

    fn match_inactive_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.manager.match_inactive_blocks(hashes)
    }

    fn has_any_registered_hashes(&self, hashes: &[SequenceHash]) -> bool {
        self.manager.has_any_registered_hashes(hashes)
    }

    fn scan_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<G2>> {
        self.manager.scan_matches(hashes, touch)
    }
}

#[test]
fn later_registration_failure_resets_every_prior_resource() {
    let first_resource = LogicalResourceId(10);
    let second_resource = LogicalResourceId(20);
    let first_hash = SequenceHash::root(10);
    let second_hash = SequenceHash::root(20);
    let first_manager = manager();
    let second_manager = manager();
    let first = staged_resource(
        first_resource,
        first_hash,
        Arc::clone(&first_manager),
        direct_g2_capacity(Arc::clone(&first_manager)),
    );
    let second = staged_resource(
        second_resource,
        second_hash,
        Arc::clone(&second_manager),
        Arc::new(RejectRegistrationCapacity {
            manager: Arc::clone(&second_manager),
        }),
    );
    let lineages = BTreeMap::from([
        (first_resource, vec![first_hash]),
        (second_resource, vec![second_hash]),
    ]);
    let bundle = StagedBundle::new(lineages, [first, second]).expect("valid staged bundle");

    let error = bundle
        .publish()
        .expect_err("the second resource registration must fail");

    assert!(
        error
            .to_string()
            .contains("forced second resource registration failure")
    );
    assert!(first_manager.match_blocks(&[first_hash]).is_empty());
    assert!(second_manager.match_blocks(&[second_hash]).is_empty());
    assert_eq!(first_manager.reset_len(), 1);
    assert_eq!(second_manager.reset_len(), 1);
}

#[test]
fn later_exact_registration_failure_invokes_the_first_source_rollback() {
    let first_resource = LogicalResourceId(10);
    let second_resource = LogicalResourceId(20);
    let first_hash = SequenceHash::root(10);
    let second_hash = SequenceHash::root(20);
    let first_manager = manager();
    let second_manager = manager();
    let first_rollbacks = Arc::new(AtomicUsize::new(0));
    let second_rollbacks = Arc::new(AtomicUsize::new(0));
    let first_capacity: Arc<dyn G2Capacity> = Arc::new(ExactTestCapacity {
        manager: Arc::clone(&first_manager),
        fail_registration: false,
        rollbacks: Arc::clone(&first_rollbacks),
    });
    let second_capacity: Arc<dyn G2Capacity> = Arc::new(ExactTestCapacity {
        manager: Arc::clone(&second_manager),
        fail_registration: true,
        rollbacks: Arc::clone(&second_rollbacks),
    });
    let first = staged_exact_resource(first_resource, first_hash, first_capacity);
    let second = staged_exact_resource(second_resource, second_hash, second_capacity);
    let lineages = BTreeMap::from([
        (first_resource, vec![first_hash]),
        (second_resource, vec![second_hash]),
    ]);
    let bundle = StagedBundle::new(lineages, [first, second]).expect("valid staged bundle");

    let error = bundle
        .publish()
        .expect_err("the second exact registration must fail");

    assert!(
        error
            .to_string()
            .contains("forced exact registration failure")
    );
    assert_eq!(first_rollbacks.load(Ordering::Relaxed), 1);
    assert_eq!(second_rollbacks.load(Ordering::Relaxed), 0);
    assert!(first_manager.match_blocks(&[first_hash]).is_empty());
    assert!(second_manager.match_blocks(&[second_hash]).is_empty());
    assert_eq!(first_manager.reset_len(), 1);
    assert_eq!(second_manager.reset_len(), 1);
}

fn manager() -> Arc<BlockManager<G2>> {
    Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .build(),
    )
}

fn staged_resource(
    resource: LogicalResourceId,
    hash: SequenceHash,
    manager: Arc<BlockManager<G2>>,
    capacity: Arc<dyn G2Capacity>,
) -> StagedPull {
    let blocks = manager
        .allocate_blocks(1)
        .expect("test destination")
        .into_iter()
        .map(|block| {
            block
                .stage(hash, manager.block_size())
                .expect("stage test block")
        })
        .collect();
    StagedPull::from_test_parts_with_capacity(resource, vec![hash], blocks, capacity)
}

fn staged_exact_resource(
    resource: LogicalResourceId,
    hash: SequenceHash,
    capacity: Arc<dyn G2Capacity>,
) -> StagedPull {
    let block_size = capacity.block_size();
    let staged = reserve_required_staging(capacity, 1)
        .expect("reserve exact test staging")
        .stage_all(&[hash], block_size)
        .expect("stage exact test block");
    StagedPull::from_test_required_staging(resource, vec![hash], staged)
}
