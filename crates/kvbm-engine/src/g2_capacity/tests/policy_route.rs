// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use kvbm_common::tokens::TokenBlockSequence;
use kvbm_common::{LogicalLayoutHandle, LogicalResourceId};
use kvbm_logical::manager::{BlockManager, ScorerParams};
use kvbm_logical::{BlockManagerSet, BlockRegistry};
use kvbm_physical::transfer::{TransferDrainOutcome, TransferOptions};
use kvbm_protocols::connector::OffloadMode;
use tokio::sync::Barrier;

use super::{ExactRegistrationCapacity, G2Capacity, SequenceHash};
use crate::g2_capacity::{
    G2AllocationKind, G2CapacityRequest, PolicyCancelDisposition, PolicyG1G2BoundRoute,
    PolicyG1G2Installation, PolicyG1G2Route, PolicyG1G2SourceSettlement, PolicyG1G2SubmitError,
    PolicyG1G2TransferExecutor, PolicyG1G2TransferReceipt, PolicyPhysicalTerminal,
};
use crate::leader::InstanceLeader;
use crate::testing::managers::TestManagerBuilder;
use crate::testing::messenger::create_messenger_tcp;
use crate::{BlockId, G1, G2};

mod session_staging;

#[derive(Default)]
struct ImmediateTransfer {
    calls: AtomicUsize,
    records: Mutex<Vec<TransferRecord>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TransferRecord {
    resource: LogicalResourceId,
    src: LogicalLayoutHandle,
    dst: LogicalLayoutHandle,
    src_blocks: Vec<BlockId>,
    dst_blocks: Vec<BlockId>,
}

struct OutcomeTransfer {
    outcome: Mutex<Option<TransferDrainOutcome>>,
}

struct GatedTransfer {
    started: Arc<Barrier>,
    release: Arc<Barrier>,
}

struct PanickingTransfer;

struct DispatchErrorTransfer;

impl ImmediateTransfer {
    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    fn records(&self) -> Vec<TransferRecord> {
        self.records.lock().expect("transfer records lock").clone()
    }
}

impl PolicyG1G2TransferExecutor for ImmediateTransfer {
    fn execute(
        &self,
        resource: LogicalResourceId,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_blocks: Vec<BlockId>,
        dst_blocks: Vec<BlockId>,
        _options: TransferOptions,
    ) -> Result<PolicyG1G2TransferReceipt> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.records
            .lock()
            .expect("transfer records lock")
            .push(TransferRecord {
                resource,
                src,
                dst,
                src_blocks,
                dst_blocks,
            });
        Ok(PolicyG1G2TransferReceipt::new(Box::pin(async {
            TransferDrainOutcome::Completed
        })))
    }
}

impl PolicyG1G2TransferExecutor for OutcomeTransfer {
    fn execute(
        &self,
        _resource: LogicalResourceId,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_blocks: Vec<BlockId>,
        _dst_blocks: Vec<BlockId>,
        _options: TransferOptions,
    ) -> Result<PolicyG1G2TransferReceipt> {
        let outcome = self
            .outcome
            .lock()
            .expect("outcome lock")
            .take()
            .expect("one transfer outcome");
        Ok(PolicyG1G2TransferReceipt::new(Box::pin(
            async move { outcome },
        )))
    }
}

impl PolicyG1G2TransferExecutor for GatedTransfer {
    fn execute(
        &self,
        _resource: LogicalResourceId,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_blocks: Vec<BlockId>,
        _dst_blocks: Vec<BlockId>,
        _options: TransferOptions,
    ) -> Result<PolicyG1G2TransferReceipt> {
        let started = Arc::clone(&self.started);
        let release = Arc::clone(&self.release);
        Ok(PolicyG1G2TransferReceipt::new(Box::pin(async move {
            started.wait().await;
            release.wait().await;
            TransferDrainOutcome::Completed
        })))
    }
}

impl PolicyG1G2TransferExecutor for PanickingTransfer {
    fn execute(
        &self,
        _resource: LogicalResourceId,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_blocks: Vec<BlockId>,
        _dst_blocks: Vec<BlockId>,
        _options: TransferOptions,
    ) -> Result<PolicyG1G2TransferReceipt> {
        Ok(PolicyG1G2TransferReceipt::new(Box::pin(async {
            panic!("injected committed transfer panic")
        })))
    }
}

impl PolicyG1G2TransferExecutor for DispatchErrorTransfer {
    fn execute(
        &self,
        _resource: LogicalResourceId,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_blocks: Vec<BlockId>,
        _dst_blocks: Vec<BlockId>,
        _options: TransferOptions,
    ) -> Result<PolicyG1G2TransferReceipt> {
        Err(anyhow::anyhow!("injected synchronous dispatch error"))
    }
}

#[tokio::test]
async fn dedicated_exact_capacity_preserves_leader_compatibility_capacity() {
    let resource = LogicalResourceId(77);
    let registry = BlockRegistry::new();
    let destination = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .registry(registry.clone())
            .build(),
    );
    let mut managers = BlockManagerSet::new();
    managers
        .insert(resource, Arc::clone(&destination))
        .expect("insert destination manager");
    let leader = InstanceLeader::builder()
        .messenger(create_messenger_tcp().await.expect("build test messenger"))
        .registry(registry)
        .g2_manager_set(Arc::new(managers), resource)
        .build()
        .expect("build leader with default compatibility capacity");
    let exact = Arc::new(ExactRegistrationCapacity::new(destination));

    // SAFETY: The supplied capacity owns the leader's G2 manager for this resource.
    let installation = unsafe {
        PolicyG1G2Route::new_with_capacity(
            leader.clone(),
            resource,
            Arc::clone(&exact) as Arc<dyn G2Capacity>,
        )
    }
    .expect("build the dedicated exact route");

    let compatibility = leader
        .g2_capacity_for(resource)
        .expect("retain the leader compatibility capacity")
        .reserve(G2CapacityRequest::compatibility(
            G2AllocationKind::RequiredStaging,
            1,
        ))
        .expect("the compatibility route remains usable");
    assert!(matches!(
        compatibility,
        super::G2CapacityDecision::Granted(_)
    ));
    drop(compatibility);

    let source = source_lineage(1);
    let route = installation
        .validate(resource, &source.manager)
        .unwrap_or_else(|_| panic!("bind the dedicated exact route"))
        .bind();
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve dedicated exact capacity");

    assert_eq!(
        exact.requests(),
        vec![G2CapacityRequest::exact_reclaim(
            G2AllocationKind::CacheExtension,
            1,
        )]
    );
    drop(reservation);
}

#[tokio::test]
async fn dedicated_exact_capacity_rejects_a_foreign_g2_manager() {
    let resource = LogicalResourceId(78);
    let registry = BlockRegistry::new();
    let destination = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .registry(registry.clone())
            .build(),
    );
    let foreign = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .registry(registry.clone())
            .build(),
    );
    let mut managers = BlockManagerSet::new();
    managers
        .insert(resource, destination)
        .expect("insert destination manager");
    let leader = InstanceLeader::builder()
        .messenger(create_messenger_tcp().await.expect("build test messenger"))
        .registry(registry)
        .g2_manager_set(Arc::new(managers), resource)
        .build()
        .expect("build leader with default compatibility capacity");

    // SAFETY: This test deliberately violates the capacity-manager identity contract.
    let error = unsafe {
        PolicyG1G2Route::new_with_capacity(
            leader,
            resource,
            Arc::new(ExactRegistrationCapacity::new(foreign)),
        )
    }
    .expect_err("reject capacity for another G2 manager");

    assert!(
        error
            .to_string()
            .contains("does not own the leader G2 manager")
    );
}

#[test]
fn foreign_resource_fails_before_capacity_or_source_mutation() {
    let (capacity, _) = capacity(1);
    let source = source_lineage(1);
    let installation = test_installation(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(7),
        Arc::new(ImmediateTransfer::default()),
    );

    let error = match installation.validate(LogicalResourceId(8), &source.manager) {
        Ok(_) => panic!("reject a foreign source resource"),
        Err((error, _installation)) => error,
    };

    assert!(error.to_string().contains("does not match source resource"));
    assert!(capacity.requests().is_empty());
    assert_eq!(source.manager.inactive_len(), 1);
}

#[tokio::test]
async fn completed_move_derives_source_ids_and_commits_inside_the_engine() {
    let (capacity, destination) = capacity(2);
    let source = source_lineage(2);
    let expected_source = source.blocks.clone();
    let hashes = source.hashes();
    let transfer = Arc::new(ImmediateTransfer::default());
    let route = bound_route(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(7),
        Arc::clone(&transfer) as Arc<dyn PolicyG1G2TransferExecutor>,
        &source.manager,
    );
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve exact G2 capacity");
    let owned = source.take();

    let completion = route
        .submit(reservation, owned, OffloadMode::Move)
        .expect("submit the owned exact source")
        .wait()
        .await
        .expect("prove physical drain");

    assert_eq!(
        completion.physical().terminal(),
        PolicyPhysicalTerminal::DestinationCommitted
    );
    assert_eq!(
        completion.source(),
        PolicyG1G2SourceSettlement::Committed {
            notification_panicked: false,
        }
    );
    assert_eq!(
        capacity.requests(),
        vec![G2CapacityRequest::exact_reclaim(
            G2AllocationKind::CacheExtension,
            2,
        )]
    );
    assert_eq!(capacity.exact_registration_count(), 1);
    assert_eq!(destination.match_blocks(&hashes).len(), 2);
    assert_eq!(source.manager.inactive_len(), 1);
    assert_eq!(
        transfer.records()[0],
        TransferRecord {
            resource: LogicalResourceId(7),
            src: LogicalLayoutHandle::G1,
            dst: LogicalLayoutHandle::G2,
            src_blocks: expected_source.iter().map(|(_, block)| *block).collect(),
            dst_blocks: vec![0, 1],
        }
    );
}

#[tokio::test]
async fn completed_mirror_restores_the_full_source_lineage() {
    let (capacity, _) = capacity(2);
    let source = source_lineage(2);
    let route = bound_route(
        capacity,
        LogicalResourceId(70),
        Arc::new(ImmediateTransfer::default()),
        &source.manager,
    );
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve exact G2 capacity");

    let completion = route
        .submit(reservation, source.take(), OffloadMode::Mirror)
        .expect("submit the mirrored source")
        .wait()
        .await
        .expect("prove mirror drain");

    assert_eq!(
        completion.physical().terminal(),
        PolicyPhysicalTerminal::DestinationCommitted
    );
    assert_eq!(completion.source(), PolicyG1G2SourceSettlement::Restored);
    assert_eq!(source.manager.inactive_len(), 2);
}

#[tokio::test]
async fn cancellation_before_commit_restores_source_without_dispatch() {
    let (capacity, destination) = capacity(1);
    let source = source_lineage(1);
    let transfer = Arc::new(ImmediateTransfer::default());
    let route = bound_route(
        capacity,
        LogicalResourceId(71),
        Arc::clone(&transfer) as Arc<dyn PolicyG1G2TransferExecutor>,
        &source.manager,
    );
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve exact G2 capacity");
    let cancellation = reservation.cancellation();
    assert_eq!(
        cancellation.cancel(),
        PolicyCancelDisposition::CancelledBeforeCommit
    );

    let completion = route
        .submit(reservation, source.take(), OffloadMode::Move)
        .expect("submit the cancelled source")
        .wait()
        .await
        .expect("prove the cancellation terminal");

    assert_eq!(
        completion.physical().terminal(),
        PolicyPhysicalTerminal::CancelledBeforeCommit
    );
    assert_eq!(completion.source(), PolicyG1G2SourceSettlement::Restored);
    assert_eq!(transfer.calls(), 0);
    assert_eq!(source.manager.inactive_len(), 1);
    assert_eq!(destination.available_blocks(), 1);
    assert_eq!(
        cancellation.cancel(),
        PolicyCancelDisposition::AlreadyTerminal
    );
}

#[tokio::test]
async fn synchronous_dispatch_error_restores_source_and_releases_capacity() {
    let (capacity, destination) = capacity(1);
    let source = source_lineage(1);
    let route = bound_route(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(72),
        Arc::new(DispatchErrorTransfer),
        &source.manager,
    );
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve exact G2 capacity");

    let completion = route
        .submit(reservation, source.take(), OffloadMode::Move)
        .expect("submit the source before dispatch")
        .wait()
        .await
        .expect("return a safe dispatch failure");

    assert_eq!(
        completion.physical().terminal(),
        PolicyPhysicalTerminal::Failed
    );
    assert!(
        completion
            .physical()
            .failure()
            .is_some_and(|failure| failure.contains("synchronous dispatch error"))
    );
    assert_eq!(completion.source(), PolicyG1G2SourceSettlement::Restored);
    assert_eq!(source.manager.inactive_len(), 1);
    assert_eq!(destination.available_blocks(), 1);
    assert_eq!(capacity.exact_permit_drop_count(), 1);
}

#[tokio::test]
async fn foreign_route_reservation_fails_before_dispatch_and_restores_source() {
    let (capacity, destination) = capacity(1);
    let source = source_lineage(1);
    let first = bound_route(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(73),
        Arc::new(ImmediateTransfer::default()),
        &source.manager,
    );
    let second_transfer = Arc::new(ImmediateTransfer::default());
    let second = bound_route(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(73),
        Arc::clone(&second_transfer) as Arc<dyn PolicyG1G2TransferExecutor>,
        &source.manager,
    );
    let reservation = first
        .reserve(&source.hashes())
        .expect("reserve on the first route");

    let completion = second
        .submit(reservation, source.take(), OffloadMode::Move)
        .expect("submit for an asynchronous binding rejection")
        .wait()
        .await
        .expect("prove the binding rejection terminal");

    assert_eq!(
        completion.physical().terminal(),
        PolicyPhysicalTerminal::Failed
    );
    assert!(
        completion
            .physical()
            .failure()
            .is_some_and(|failure| failure.contains("another bound route"))
    );
    assert_eq!(completion.source(), PolicyG1G2SourceSettlement::Restored);
    assert_eq!(second_transfer.calls(), 0);
    assert_eq!(source.manager.inactive_len(), 1);
    assert_eq!(destination.available_blocks(), 1);
}

#[tokio::test]
async fn drained_error_releases_destination_before_terminal_publication() {
    let (capacity, destination) = capacity(1);
    let source = source_lineage(1);
    let route = bound_route(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(8),
        Arc::new(OutcomeTransfer {
            outcome: Mutex::new(Some(TransferDrainOutcome::DrainedWithError(
                anyhow::anyhow!("injected drained failure"),
            ))),
        }),
        &source.manager,
    );
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve exact G2 capacity");
    let cancellation = reservation.cancellation();
    let state_during_release = Arc::new(Mutex::new(None));
    capacity.observe_exact_permit_drop(Arc::new({
        let cancellation = cancellation.clone();
        let state_during_release = Arc::clone(&state_during_release);
        move || {
            *state_during_release.lock().expect("release state lock") = Some(cancellation.cancel());
        }
    }));
    let completion = route
        .submit(reservation, source.take(), OffloadMode::Move)
        .expect("submit the owned exact source")
        .wait()
        .await
        .expect("return the proven drained failure");

    assert_eq!(
        completion.physical().terminal(),
        PolicyPhysicalTerminal::Failed
    );
    assert_eq!(completion.source(), PolicyG1G2SourceSettlement::Restored);
    assert_eq!(source.manager.inactive_len(), 1);
    assert_eq!(destination.available_blocks(), 1);
    assert_eq!(capacity.exact_permit_drop_count(), 1);
    assert_eq!(
        *state_during_release.lock().expect("release state lock"),
        Some(PolicyCancelDisposition::CommittedTransferDraining),
        "the destination permit drops before cancellation becomes terminal"
    );
    assert_eq!(
        cancellation.cancel(),
        PolicyCancelDisposition::AlreadyTerminal
    );
}

#[tokio::test]
async fn unproven_receipt_retains_source_lease_destination_pins_and_cancellation() {
    let (capacity, destination) = capacity(1);
    let source = source_lineage(1);
    let route = bound_route(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(9),
        Arc::new(OutcomeTransfer {
            outcome: Mutex::new(Some(TransferDrainOutcome::Unproven(anyhow::anyhow!(
                "injected ambiguous receipt"
            )))),
        }),
        &source.manager,
    );
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve exact G2 capacity");
    let cancellation = reservation.cancellation();
    let execution = route
        .submit(reservation, source.take(), OffloadMode::Move)
        .expect("submit the owned exact source");

    assert!(execution.wait().await.is_err());
    assert_eq!(source.manager.inactive_len(), 0);
    assert_eq!(destination.available_blocks(), 0);
    assert_eq!(capacity.exact_permit_drop_count(), 0);
    assert_eq!(
        cancellation.cancel(),
        PolicyCancelDisposition::CommittedTransferDraining
    );
}

#[tokio::test]
async fn committed_panic_retains_source_and_destination_pins() {
    let (capacity, destination) = capacity(1);
    let source = source_lineage(1);
    let route = bound_route(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(10),
        Arc::new(PanickingTransfer),
        &source.manager,
    );
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve exact G2 capacity");
    let cancellation = reservation.cancellation();

    assert!(
        route
            .submit(reservation, source.take(), OffloadMode::Move)
            .expect("submit the owned exact source")
            .wait()
            .await
            .is_err()
    );
    assert_eq!(source.manager.inactive_len(), 0);
    assert_eq!(destination.available_blocks(), 0);
    assert_eq!(capacity.exact_permit_drop_count(), 0);
    assert_eq!(
        cancellation.cancel(),
        PolicyCancelDisposition::CommittedTransferDraining
    );
}

#[tokio::test]
async fn dropping_execution_does_not_end_source_before_proven_drain() {
    let (capacity, _) = capacity(1);
    let source = source_lineage(1);
    let started = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let route = bound_route(
        capacity,
        LogicalResourceId(11),
        Arc::new(GatedTransfer {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        }),
        &source.manager,
    );
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve exact G2 capacity");
    let cancellation = reservation.cancellation();
    let execution = route
        .submit(reservation, source.take(), OffloadMode::Mirror)
        .expect("submit the owned exact source");

    started.wait().await;
    drop(execution);
    assert_eq!(source.manager.inactive_len(), 0);
    release.wait().await;
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while cancellation.cancel() != PolicyCancelDisposition::AlreadyTerminal {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the detached exact transaction must settle");
    assert_eq!(source.manager.inactive_len(), 1);
}

#[test]
fn a_foreign_source_manager_is_rejected_before_dispatch() {
    let (capacity, destination) = capacity(1);
    let expected_source = source_lineage(1);
    let foreign_source = source_lineage(1);
    let transfer = Arc::new(ImmediateTransfer::default());
    let route = bound_route(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(12),
        Arc::clone(&transfer) as Arc<dyn PolicyG1G2TransferExecutor>,
        &expected_source.manager,
    );
    let reservation = route
        .reserve(&expected_source.hashes())
        .expect("reserve exact G2 capacity");
    let cancellation = reservation.cancellation();

    let error = match route.submit(reservation, foreign_source.take(), OffloadMode::Move) {
        Ok(_) => panic!("reject a source from another manager"),
        Err(error) => error,
    };

    assert_eq!(error, PolicyG1G2SubmitError::ForeignSourceManager);
    assert_eq!(transfer.calls(), 0);
    assert_eq!(foreign_source.manager.inactive_len(), 1);
    assert_eq!(destination.available_blocks(), 1);
    assert_eq!(capacity.exact_permit_drop_count(), 1);
    assert_eq!(
        cancellation.cancel(),
        PolicyCancelDisposition::AlreadyTerminal
    );
}

#[tokio::test]
async fn abandoned_task_owner_publishes_a_restored_terminal() {
    let (capacity, destination) = capacity(1);
    let source = source_lineage(1);
    let transfer = Arc::new(ImmediateTransfer::default());
    let route = bound_route(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(75),
        Arc::clone(&transfer) as Arc<dyn PolicyG1G2TransferExecutor>,
        &source.manager,
    );
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve exact G2 capacity");
    let cancellation = reservation.cancellation();

    let execution = route
        .abandon_before_task_start_for_test(reservation, source.take(), OffloadMode::Move)
        .expect("simulate an accepted task that never starts");

    assert_eq!(transfer.calls(), 0);
    assert_eq!(source.manager.inactive_len(), 1);
    assert!(execution.is_finished());
    let completion = execution
        .wait()
        .await
        .expect("the abandoned task publishes a defined terminal");
    assert_eq!(
        completion.physical().terminal(),
        PolicyPhysicalTerminal::Failed
    );
    assert_eq!(completion.source(), PolicyG1G2SourceSettlement::Restored);
    assert_eq!(destination.available_blocks(), 1);
    assert_eq!(capacity.exact_permit_drop_count(), 1);
    assert_eq!(
        cancellation.cancel(),
        PolicyCancelDisposition::AlreadyTerminal
    );
}

#[test]
fn tokio_no_threads_spawn_panic_restores_the_pre_start_transaction() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .max_blocking_threads(1)
        .thread_stack_size(usize::MAX)
        .build()
        .expect("build a runtime whose blocking worker cannot start");
    let (capacity, destination) = capacity(1);
    let source = source_lineage(1);
    let transfer = Arc::new(ImmediateTransfer::default());
    let route = bound_route_with_runtime(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(76),
        Arc::clone(&transfer) as Arc<dyn PolicyG1G2TransferExecutor>,
        &source.manager,
        runtime.handle().clone(),
    );
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve exact G2 capacity");
    let cancellation = reservation.cancellation();

    let execution = route
        .submit(reservation, source.take(), OffloadMode::Move)
        .expect("convert the Tokio spawn panic into a defined execution");

    assert_eq!(transfer.calls(), 0);
    assert_eq!(source.manager.inactive_len(), 1);
    assert!(execution.is_finished());
    let completion = futures::executor::block_on(execution.wait())
        .expect("the rejected spawn publishes a defined terminal");
    assert_eq!(
        completion.physical().terminal(),
        PolicyPhysicalTerminal::Failed
    );
    assert_eq!(completion.source(), PolicyG1G2SourceSettlement::Restored);
    assert!(
        completion
            .physical()
            .failure()
            .is_some_and(|failure| failure.contains("ended before it started"))
    );
    assert_eq!(destination.available_blocks(), 1);
    assert_eq!(capacity.exact_permit_drop_count(), 1);
    assert_eq!(
        cancellation.cancel(),
        PolicyCancelDisposition::AlreadyTerminal
    );
    drop(runtime);
}

#[test]
fn no_runtime_restores_source_before_the_reservation_becomes_terminal() {
    let (capacity, destination) = capacity(1);
    let source = source_lineage(1);
    let route = bound_route(
        Arc::clone(&capacity) as Arc<dyn G2Capacity>,
        LogicalResourceId(74),
        Arc::new(ImmediateTransfer::default()),
        &source.manager,
    );
    let reservation = route
        .reserve(&source.hashes())
        .expect("reserve exact G2 capacity");
    let cancellation = reservation.cancellation();

    let result = route.submit(reservation, source.take(), OffloadMode::Move);

    assert!(matches!(result, Err(PolicyG1G2SubmitError::NoRuntime)));
    assert_eq!(source.manager.inactive_len(), 1);
    assert_eq!(destination.available_blocks(), 1);
    assert_eq!(capacity.exact_permit_drop_count(), 1);
    assert_eq!(
        cancellation.cancel(),
        PolicyCancelDisposition::AlreadyTerminal
    );
}

fn test_installation(
    capacity: Arc<dyn G2Capacity>,
    resource: LogicalResourceId,
    transfer: Arc<dyn PolicyG1G2TransferExecutor>,
) -> PolicyG1G2Installation {
    PolicyG1G2Route::from_test_parts(capacity, resource, transfer)
}

fn bound_route(
    capacity: Arc<dyn G2Capacity>,
    resource: LogicalResourceId,
    transfer: Arc<dyn PolicyG1G2TransferExecutor>,
    source_manager: &BlockManager<G1>,
) -> PolicyG1G2BoundRoute<G1> {
    let installation = test_installation(capacity, resource, transfer);
    let validated = match installation.validate(resource, source_manager) {
        Ok(validated) => validated,
        Err(_) => panic!("validate the paired exact installation"),
    };
    validated.bind()
}

fn bound_route_with_runtime(
    capacity: Arc<dyn G2Capacity>,
    resource: LogicalResourceId,
    transfer: Arc<dyn PolicyG1G2TransferExecutor>,
    source_manager: &BlockManager<G1>,
    runtime: tokio::runtime::Handle,
) -> PolicyG1G2BoundRoute<G1> {
    let installation =
        PolicyG1G2Route::from_test_parts_with_runtime(capacity, resource, transfer, runtime);
    let validated = match installation.validate(resource, source_manager) {
        Ok(validated) => validated,
        Err(_) => panic!("validate the paired exact installation"),
    };
    validated.bind()
}

fn capacity(block_count: usize) -> (Arc<ExactRegistrationCapacity>, Arc<BlockManager<G2>>) {
    let manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(block_count)
            .block_size(4)
            .build(),
    );
    (
        Arc::new(ExactRegistrationCapacity::new(Arc::clone(&manager))),
        manager,
    )
}

struct SourceLineage {
    manager: Arc<BlockManager<G1>>,
    blocks: Vec<(SequenceHash, BlockId)>,
}

impl SourceLineage {
    fn hashes(&self) -> Vec<SequenceHash> {
        self.blocks.iter().map(|(hash, _)| *hash).collect()
    }

    fn take(&self) -> kvbm_logical::InactiveLineageHold<G1> {
        let candidate = self
            .manager
            .inactive_candidates(1)
            .into_iter()
            .next()
            .expect("the source lineage has one leaf candidate");
        let hold = self
            .manager
            .try_hold_inactive_lineage(candidate)
            .expect("hold the complete source lineage");
        assert_eq!(hold.source_blocks(), self.blocks.as_slice());
        hold
    }
}

fn source_lineage(count: usize) -> SourceLineage {
    let manager = Arc::new(
        BlockManager::<G1>::builder()
            .block_count(count + 2)
            .block_size(1)
            .registry(BlockRegistry::new())
            .with_valued_lineage_backend(ScorerParams::default())
            .build()
            .expect("build a valued source manager"),
    );
    let tokens = (1..=count as u32).collect::<Vec<_>>();
    let sequence = TokenBlockSequence::from_slice(&tokens, 1, Some(0x5eed));
    let blocks = sequence
        .blocks()
        .iter()
        .map(|token_block| {
            let mutable = manager
                .allocate_blocks(1)
                .expect("allocate one source block")
                .pop()
                .expect("one source block");
            let complete = mutable
                .complete(token_block)
                .expect("complete one source block");
            manager.register_block(complete)
        })
        .collect::<Vec<_>>();
    let expected = blocks
        .iter()
        .map(|block| (block.sequence_hash(), block.block_id()))
        .collect();
    drop(blocks);
    SourceLineage {
        manager,
        blocks: expected,
    }
}
