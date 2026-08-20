// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact physical transaction and panic supervision.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use kvbm_common::{KvbmTransferRoute, LogicalLayoutHandle};
use kvbm_logical::blocks::BlockMetadata;
use kvbm_physical::transfer::{TransferDrainOutcome, TransferOptions};
use tokio::sync::oneshot;

use super::source::PolicyG1G2Source;
use super::state::{PolicyG1G2Completion, PolicyG1G2Reservation, PolicyPhysicalCompletion};
use super::{PolicyG1G2RouteCore, RouteBinding};

pub(super) struct PolicyG1G2Transaction<T: BlockMetadata> {
    route: TransactionRoute,
    reservation: PolicyG1G2Reservation,
    source: Option<PolicyG1G2Source<T>>,
    phase: PolicyTransactionPhase,
}

/// Own the complete transaction until the blocking task starts.
///
/// Drop restores the source before the armed reservation marks its terminal.
pub(super) struct PolicyG1G2TransactionOwner<T: BlockMetadata> {
    transaction: Arc<Mutex<Option<PolicyG1G2Transaction<T>>>>,
    publisher: Arc<PolicyG1G2CompletionPublisher>,
    restore_on_drop: bool,
}

struct PolicyG1G2CompletionPublisher {
    sender: Mutex<Option<oneshot::Sender<PolicyG1G2Completion>>>,
    finished: Arc<AtomicBool>,
}

pub(super) struct TransactionRoute {
    pub(super) core: Arc<PolicyG1G2RouteCore>,
    pub(super) binding: Arc<RouteBinding>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PolicyTransactionPhase {
    Open,
    CommittedUndrained,
    PhysicallyDrained,
}

impl<T: BlockMetadata> PolicyG1G2Transaction<T> {
    pub(super) fn new(
        route: TransactionRoute,
        reservation: PolicyG1G2Reservation,
        source: PolicyG1G2Source<T>,
    ) -> Self {
        Self {
            route,
            reservation,
            source: Some(source),
            phase: PolicyTransactionPhase::Open,
        }
    }

    fn start(&mut self) {
        self.reservation.disarm_terminal_on_drop();
    }

    fn abandon(mut self) -> PolicyG1G2Completion {
        let cancellation = self.reservation.cancellation();
        let physical = self
            .reservation
            .uncommitted_failure("the exact blocking task ended before it started");
        let source = self
            .source
            .take()
            .expect("the abandoned transaction restores its logical source");
        let settlement = source.settle(physical.terminal());
        cancellation.mark_terminal();
        drop(self);
        PolicyG1G2Completion::new(physical, settlement)
    }

    fn execute(&mut self) -> Option<PolicyPhysicalCompletion> {
        if !Arc::ptr_eq(&self.route.binding, &self.reservation.binding) {
            return Some(
                self.reservation
                    .uncommitted_failure("the exact G2 reservation belongs to another bound route"),
            );
        }
        let Some(allocation) = self.reservation.allocation.as_ref() else {
            return Some(
                self.reservation
                    .uncommitted_failure("the exact G2 reservation is empty"),
            );
        };
        let source_blocks = self
            .source
            .as_ref()
            .expect("the transaction owns its logical source")
            .source_blocks();
        if source_blocks.len() != allocation.len() {
            return Some(self.reservation.uncommitted_failure(format!(
                "the owned source has {} blocks for a {}-block G2 reservation",
                source_blocks.len(),
                allocation.len()
            )));
        }
        if !self.reservation.cancellation.claim_commit() {
            drop(self.reservation.allocation.take());
            return Some(PolicyPhysicalCompletion::cancelled());
        }
        self.phase = PolicyTransactionPhase::CommittedUndrained;

        let hashes = source_blocks
            .iter()
            .map(|(hash, _)| *hash)
            .collect::<Vec<_>>();
        let src_blocks = source_blocks
            .iter()
            .map(|(_, block)| *block)
            .collect::<Vec<_>>();
        let dst_blocks = allocation.block_ids();
        let options = match TransferOptions::builder()
            .metric_route(KvbmTransferRoute::OffloadD2H)
            .build()
        {
            Ok(options) => options,
            Err(error) => {
                self.phase = PolicyTransactionPhase::PhysicallyDrained;
                return Some(PolicyPhysicalCompletion::failed(format!(
                    "policy transfer options failed: {error}"
                )));
            }
        };
        let receipt = match self.route.core.transfer.execute(
            self.route.core.resource,
            LogicalLayoutHandle::G1,
            LogicalLayoutHandle::G2,
            src_blocks,
            dst_blocks,
            options,
        ) {
            Ok(receipt) => receipt,
            Err(error) => {
                self.phase = PolicyTransactionPhase::PhysicallyDrained;
                return Some(PolicyPhysicalCompletion::failed(format!(
                    "physical G1-to-G2 dispatch failed: {error:#}"
                )));
            }
        };
        match receipt.drain() {
            TransferDrainOutcome::Completed => {}
            TransferDrainOutcome::DrainedWithError(error) => {
                self.phase = PolicyTransactionPhase::PhysicallyDrained;
                return Some(PolicyPhysicalCompletion::failed(format!(
                    "physical G1-to-G2 transfer drained with an error: {error:#}"
                )));
            }
            TransferDrainOutcome::Unproven(error) => {
                tracing::error!(
                    error = %error,
                    "the exact G1-to-G2 receipt did not prove drain; retaining source and destination pins"
                );
                return None;
            }
        }
        self.phase = PolicyTransactionPhase::PhysicallyDrained;

        let allocation = self
            .reservation
            .allocation
            .take()
            .expect("the exact allocation remains pinned through physical drain");
        let staged = match allocation.stage_all(&hashes, self.route.core.capacity.block_size()) {
            Ok(staged) => staged,
            Err(error) => {
                return Some(PolicyPhysicalCompletion::failed(format!(
                    "G2 transfer staging failed: {error:#}"
                )));
            }
        };
        let registered = match staged.register() {
            Ok(registered) => registered,
            Err(error) => {
                return Some(PolicyPhysicalCompletion::failed(format!(
                    "exact G2 registration failed: {error}"
                )));
            }
        };
        drop(registered);
        Some(PolicyPhysicalCompletion::destination_committed())
    }

    fn into_completion(mut self, physical: PolicyPhysicalCompletion) -> PolicyG1G2Completion {
        debug_assert_ne!(self.phase, PolicyTransactionPhase::CommittedUndrained);
        let cancellation = self.reservation.cancellation();
        drop(self.reservation.allocation.take());
        let source = self
            .source
            .take()
            .expect("the proven transaction settles its logical source");
        let settlement = source.settle(physical.terminal());
        cancellation.mark_terminal();
        drop(self);
        PolicyG1G2Completion::new(physical, settlement)
    }
}

impl<T: BlockMetadata> PolicyG1G2TransactionOwner<T> {
    pub(super) fn new(
        transaction: PolicyG1G2Transaction<T>,
        sender: oneshot::Sender<PolicyG1G2Completion>,
        finished: Arc<AtomicBool>,
    ) -> Self {
        Self {
            transaction: Arc::new(Mutex::new(Some(transaction))),
            publisher: Arc::new(PolicyG1G2CompletionPublisher {
                sender: Mutex::new(Some(sender)),
                finished,
            }),
            restore_on_drop: true,
        }
    }

    pub(super) fn task_owner(&self) -> Self {
        Self {
            transaction: Arc::clone(&self.transaction),
            publisher: Arc::clone(&self.publisher),
            restore_on_drop: true,
        }
    }

    pub(super) fn commit_to_task(mut self) {
        self.restore_on_drop = false;
    }

    pub(super) fn supervise(mut self) {
        let mut transaction = self
            .transaction
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .expect("the pre-start owner holds one exact transaction");
        self.restore_on_drop = false;
        transaction.start();
        let Some(completion) = supervise_policy_transaction(transaction) else {
            tracing::error!("the exact G1-to-G2 supervisor retained source and destination pins");
            return;
        };
        self.publisher.publish(completion);
    }
}

impl<T: BlockMetadata> Drop for PolicyG1G2TransactionOwner<T> {
    fn drop(&mut self) {
        if !self.restore_on_drop {
            return;
        }
        let transaction = self
            .transaction
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(transaction) = transaction {
            self.publisher.publish(transaction.abandon());
        }
    }
}

impl PolicyG1G2CompletionPublisher {
    fn publish(&self, completion: PolicyG1G2Completion) {
        self.finished.store(true, Ordering::Release);
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(sender) = sender {
            let _ = sender.send(completion);
        }
    }
}

pub(super) fn supervise_policy_transaction<T: BlockMetadata>(
    mut transaction: PolicyG1G2Transaction<T>,
) -> Option<PolicyG1G2Completion> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| transaction.execute()));
    match result {
        Ok(Some(completion)) => Some(transaction.into_completion(completion)),
        Ok(None) => {
            std::mem::forget(transaction);
            None
        }
        Err(_) if transaction.phase != PolicyTransactionPhase::CommittedUndrained => Some(
            transaction.into_completion(PolicyPhysicalCompletion::failed(
                "the exact physical supervisor panicked after a safe terminal",
            )),
        ),
        Err(_) => {
            std::mem::forget(transaction);
            None
        }
    }
}
