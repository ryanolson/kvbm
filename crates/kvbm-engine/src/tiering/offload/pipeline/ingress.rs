// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared ingress and cancellation watcher for offload pipelines.

use std::collections::HashSet;
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::Mutex as ParkingMutex;
use tokio::sync::watch;

use kvbm_logical::blocks::BlockMetadata;

use super::super::cancel::{CancellationToken, CancellationUnit};
use super::super::container::OffloadContainer;
use super::super::handle::{TransferId, TransferState, TransferStatus};
use super::super::queue::CancellableQueue;
use super::super::source::SourceBlocks;
use super::shutdown::PRECOMMIT_SHUTDOWN_ERROR;

/// Cloneable ingress for a pipeline that receives auto-chain containers.
///
/// It uses the same cancellation queues and token watcher as direct enqueue.
#[derive(Clone)]
pub(crate) struct PipelineIngress<T: BlockMetadata> {
    pub(super) eval_queue: Arc<CancellableQueue<OffloadContainer<T>>>,
    pub(super) cancellation_queues: Vec<Arc<CancellableQueue<OffloadContainer<T>>>>,
    pub(super) transfers: Arc<DashMap<TransferId, Arc<std::sync::Mutex<TransferState>>>>,
    pub(super) registration_gate: Arc<ParkingMutex<()>>,
    pub(super) cancel_tx: watch::Sender<HashSet<TransferId>>,
    pub(super) runtime: tokio::runtime::Handle,
}

impl<T: BlockMetadata> PipelineIngress<T> {
    pub(crate) fn enqueue(
        &self,
        transfer_id: TransferId,
        source: SourceBlocks<T>,
        state: Arc<std::sync::Mutex<TransferState>>,
    ) -> bool {
        let precondition = state.lock().unwrap().precondition;
        let container =
            OffloadContainer::new(transfer_id, source, Arc::clone(&state), precondition);
        self.enqueue_container(transfer_id, state, container)
    }

    pub(crate) fn enqueue_chained(
        &self,
        transfer_id: TransferId,
        source: SourceBlocks<T>,
        state: Arc<std::sync::Mutex<TransferState>>,
        cancellation: CancellationUnit,
    ) -> bool {
        let precondition = state.lock().unwrap().precondition;
        let container = OffloadContainer::with_cancellation(
            transfer_id,
            source,
            Arc::clone(&state),
            precondition,
            cancellation,
        );
        self.enqueue_container(transfer_id, state, container)
    }

    fn enqueue_container(
        &self,
        transfer_id: TransferId,
        state: Arc<std::sync::Mutex<TransferState>>,
        container: OffloadContainer<T>,
    ) -> bool {
        let cancel_token = container.cancellation_token();
        {
            let _registration = self.registration_gate.lock();
            if self.transfers.contains_key(&transfer_id) {
                container.fail(format!("duplicate transfer ID {transfer_id}"));
                return false;
            }
            if let Err(container) = self.eval_queue.push_or_return(transfer_id, container) {
                let error = if self.eval_queue.is_closed() {
                    PRECOMMIT_SHUTDOWN_ERROR.to_string()
                } else {
                    format!("transfer queue rejected {transfer_id}")
                };
                container.fail(error);
                return false;
            }
            self.transfers.insert(transfer_id, Arc::clone(&state));
        }

        let queues = self.cancellation_queues.clone();
        let cancel_tx = self.cancel_tx.clone();
        let mut status_rx = state.lock().unwrap().subscribe_status();
        let transfers = Arc::clone(&self.transfers);
        let registration_gate = Arc::clone(&self.registration_gate);
        self.runtime.spawn(async move {
            if wait_for_cancellation_or_terminal(&cancel_token, &mut status_rx).await {
                state.lock().unwrap().begin_cancellation();
                if cancel_token.is_precommit_cancelled() {
                    let _registration = registration_gate.lock();
                    let owns_registration = transfers
                        .get(&transfer_id)
                        .is_some_and(|current| Arc::ptr_eq(current.value(), &state));
                    if owns_registration {
                        for queue in &queues {
                            queue.mark_cancelled(transfer_id);
                        }
                        cancel_tx.send_modify(|set| {
                            set.insert(transfer_id);
                        });
                    }
                }
            }

            wait_for_terminal(&mut status_rx).await;
            let _registration = registration_gate.lock();
            let owns_registration = transfers
                .get(&transfer_id)
                .is_some_and(|current| Arc::ptr_eq(current.value(), &state));
            if owns_registration {
                for queue in &queues {
                    queue.clear_cancelled(transfer_id);
                }
                cancel_tx.send_modify(|set| {
                    set.remove(&transfer_id);
                });
                transfers.remove_if(&transfer_id, |_, current| Arc::ptr_eq(current, &state));
            }
        });
        true
    }
}

/// Wait for cancellation, or stop the watcher when the transfer terminates.
async fn wait_for_cancellation_or_terminal(
    cancel_token: &CancellationToken,
    status_rx: &mut watch::Receiver<TransferStatus>,
) -> bool {
    tokio::select! {
        _ = cancel_token.wait_requested() => true,
        _ = wait_for_terminal(status_rx) => false,
    }
}

async fn wait_for_terminal(status_rx: &mut watch::Receiver<TransferStatus>) {
    while !status_rx.borrow().is_terminal() {
        if status_rx.changed().await.is_err() {
            return;
        }
    }
}
