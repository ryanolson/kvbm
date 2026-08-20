// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote G4 route ownership and completion handling.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::leader::InstanceLeader;
use crate::worker::RemoteDescriptor;
use crate::{BlockId, G2, SequenceHash};
use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_physical::transfer::{
    TransferCompleteNotification, TransferDrainOutcome, TransferOptions,
};

use super::cancel::CancellationUnit;
use super::handle::{TransferId, TransferState, fail_transfer_unit, settle_transfer_unit};

/// One remote G2-to-G4 route that owns source guards until physical drain.
pub(super) struct RemoteG4OffloadRequest {
    transfer_id: TransferId,
    keys: Vec<SequenceHash>,
    block_ids: Vec<BlockId>,
    source_guards: Vec<ImmutableBlock<G2>>,
    state: Arc<std::sync::Mutex<TransferState>>,
    cancellation: CancellationUnit,
}

/// Owns a request until a receipt proves physical drain.
struct RemoteG4DrainOwnership {
    request: Option<RemoteG4OffloadRequest>,
}

impl RemoteG4OffloadRequest {
    /// Convert source guards into one remote route after fan-out selected it.
    pub(super) fn from_blocks(
        transfer_id: TransferId,
        source_guards: Vec<ImmutableBlock<G2>>,
        state: Arc<std::sync::Mutex<TransferState>>,
        cancellation: CancellationUnit,
    ) -> Self {
        let keys = source_guards
            .iter()
            .map(|block| block.sequence_hash())
            .collect();
        let block_ids = source_guards.iter().map(|block| block.block_id()).collect();
        Self {
            transfer_id,
            keys,
            block_ids,
            source_guards,
            state,
            cancellation,
        }
    }

    pub(super) fn fail_before_dispatch(self, error: String) {
        self.release_and_fail(error);
    }

    fn release_and_fail(self, error: String) {
        let Self {
            source_guards,
            state,
            cancellation,
            ..
        } = self;
        drop(source_guards);
        fail_transfer_unit(cancellation, state, error);
    }

    fn settle_after_proven_drain(self) {
        let Self {
            source_guards,
            state,
            cancellation,
            ..
        } = self;
        drop(source_guards);
        settle_transfer_unit(cancellation, state);
    }

    /// Retain source guards and route ownership when physical drain is unknown.
    fn retain_unproven(self, error: String) {
        let Self {
            source_guards,
            state,
            cancellation,
            ..
        } = self;
        state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record_error(error);

        // No completion source remains after an ambiguous error or runtime
        // abort. This intentional leak prevents G2 reuse and terminal status
        // while worker upload can still access the source.
        std::mem::forget(source_guards);
        std::mem::forget(cancellation);
    }
}

impl RemoteG4DrainOwnership {
    fn new(request: RemoteG4OffloadRequest) -> Self {
        Self {
            request: Some(request),
        }
    }

    fn request(&self) -> &RemoteG4OffloadRequest {
        self.request
            .as_ref()
            .expect("remote G4 drain owns its request")
    }

    fn settle_success(mut self) {
        self.request
            .take()
            .expect("remote G4 drain owns its request")
            .settle_after_proven_drain();
    }

    fn fail_after_proven_drain(mut self, error: String) {
        self.request
            .take()
            .expect("remote G4 drain owns its request")
            .release_and_fail(error);
    }

    fn fail_definitive_dispatch(mut self, error: String) {
        self.request
            .take()
            .expect("remote G4 drain owns its request")
            .fail_before_dispatch(error);
    }

    fn retain_unproven(mut self, error: String) {
        self.request
            .take()
            .expect("remote G4 drain owns its request")
            .retain_unproven(error);
    }
}

impl Drop for RemoteG4DrainOwnership {
    fn drop(&mut self) {
        if let Some(request) = self.request.take() {
            request.retain_unproven(
                "remote G4 drain ended without proven physical completion".to_string(),
            );
        }
    }
}

async fn await_remote_g4_proven_drain(
    ownership: RemoteG4DrainOwnership,
    notification: TransferCompleteNotification,
) -> Result<(), String> {
    match notification.await_drain().await {
        TransferDrainOutcome::Completed => {
            ownership.settle_success();
            Ok(())
        }
        TransferDrainOutcome::DrainedWithError(error) => {
            let error = format!("remote G4 drained after a dispatch failure: {error}");
            ownership.fail_after_proven_drain(error.clone());
            Err(error)
        }
        TransferDrainOutcome::Unproven(error) => {
            let error = format!("remote G4 completion is ambiguous: {error}");
            ownership.retain_unproven(error.clone());
            Err(error)
        }
    }
}

/// Process remote G4 requests through worker object storage operations.
pub(super) async fn run(
    mut rx: mpsc::Receiver<RemoteG4OffloadRequest>,
    leader: Arc<InstanceLeader>,
) {
    tracing::info!("Remote G4 offload task started");

    while let Some(request) = rx.recv().await {
        assert!(
            request.cancellation.claim_commitment(),
            "remote G4 belongs to one committed logical transfer"
        );
        let transfer_id = request.transfer_id;
        let num_blocks = request.keys.len();
        tracing::debug!(
            %request.transfer_id,
            num_blocks,
            "Processing remote G4 offload request"
        );

        let ownership = RemoteG4DrainOwnership::new(request);
        let result = leader.execute_remote_offload(
            LogicalLayoutHandle::G2,
            RemoteDescriptor::Object {
                keys: ownership.request().keys.clone(),
            },
            ownership.request().block_ids.clone(),
            TransferOptions::default(),
        );

        match result {
            Ok(notification) => {
                // A child task survives cancellation of this intake task. Its
                // drop owner retains guards if the runtime aborts the child.
                let drain = tokio::spawn(await_remote_g4_proven_drain(ownership, notification));
                match drain.await {
                    Ok(Ok(())) => {
                        tracing::info!(
                            %transfer_id,
                            num_blocks,
                            "Remote G4 offload completed successfully"
                        );
                    }
                    Ok(Err(error)) => {
                        tracing::warn!(
                            %transfer_id,
                            num_blocks,
                            %error,
                            "Remote G4 offload completion failed"
                        );
                    }
                    Err(error) => {
                        tracing::warn!(
                            %transfer_id,
                            num_blocks,
                            %error,
                            "Remote G4 drain task stopped without proof"
                        );
                    }
                }
            }
            Err(error) => {
                let error = format!("failed to dispatch remote G4 offload: {error}");
                ownership.fail_definitive_dispatch(error.clone());
                tracing::warn!(
                    %transfer_id,
                    num_blocks,
                    %error,
                    "Remote G4 dispatch failed before outstanding work"
                );
            }
        }
    }

    tracing::info!("Remote G4 offload task shutting down");
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::anyhow;
    use tokio::sync::mpsc;

    use super::*;
    use crate::testing::{TestManagerBuilder, TestRegistryBuilder, create_messenger_tcp};
    use crate::tiering::offload::TransferStatus;
    use crate::tiering::offload::pipeline::ChainOutput;

    async fn test_leader() -> Arc<InstanceLeader> {
        let messenger = create_messenger_tcp().await.expect("create messenger");
        let registry = Arc::new(TestRegistryBuilder::new().build());
        let g2_manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(4)
                .block_size(4)
                .registry(registry.as_ref().clone())
                .build(),
        );
        Arc::new(
            InstanceLeader::builder()
                .messenger(messenger)
                .registry(registry.as_ref().clone())
                .g2_manager(g2_manager)
                .workers(Vec::new())
                .build()
                .expect("build test leader"),
        )
    }

    fn remote_request_with_guard() -> (
        RemoteG4OffloadRequest,
        ImmutableBlock<G2>,
        super::super::handle::TransferHandle,
    ) {
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(1)
                .block_size(4)
                .build(),
        );
        let token_block = crate::testing::create_sequential_block(0, manager.block_size());
        let sequence_hash = crate::testing::populate_manager_with_blocks(
            manager.as_ref(),
            std::slice::from_ref(&token_block),
        )
        .expect("populate source manager")[0];
        let block = manager
            .match_blocks(&[sequence_hash])
            .pop()
            .expect("match source block");
        let observer = block.clone();
        let block_id = block.block_id();
        let transfer_id = TransferId::new();
        let (mut state, handle) = TransferState::new(transfer_id, vec![block_id]);
        let token = state.cancellation_token();
        state.add_passed([block_id]);
        state.mark_committed_blocks([block_id]);
        let state = Arc::new(std::sync::Mutex::new(state));
        let cancellation = token
            .root_unit()
            .expect("create root cancellation unit")
            .bind_state(Arc::clone(&state));
        assert!(cancellation.claim_commitment());

        (
            RemoteG4OffloadRequest {
                transfer_id,
                keys: vec![sequence_hash],
                block_ids: vec![block_id],
                source_guards: vec![block],
                state,
                cancellation,
            },
            observer,
            handle,
        )
    }

    #[tokio::test]
    async fn aborted_remote_drain_retains_guards_and_cancellation() {
        let (request, observer, handle) = remote_request_with_guard();
        let events = Arc::new(velo::EventManager::local());
        let physical_event = events.new_event().expect("create physical event");
        let notification = TransferCompleteNotification::from_awaiter(
            events
                .awaiter(physical_event.handle())
                .expect("create physical completion awaiter"),
        );
        let mut drain = tokio::spawn(await_remote_g4_proven_drain(
            RemoteG4DrainOwnership::new(request),
            notification,
        ));

        assert!(
            tokio::time::timeout(Duration::from_millis(25), &mut drain)
                .await
                .is_err(),
            "physical work must remain active before task abort",
        );
        drain.abort();
        assert!(
            drain
                .await
                .expect_err("abort the remote drain")
                .is_cancelled()
        );

        assert_eq!(observer.use_count(), 2);
        assert!(
            tokio::time::timeout(Duration::from_millis(25), handle.cancel().wait())
                .await
                .is_err(),
            "an unproven drain must retain its cancellation unit",
        );
    }

    #[tokio::test]
    async fn ambiguous_remote_notification_retains_guards_and_cancellation() {
        let (request, observer, handle) = remote_request_with_guard();
        let events = Arc::new(velo::EventManager::local());
        let physical_event = events.new_event().expect("create physical event");
        let notification = TransferCompleteNotification::from_awaiter(
            events
                .awaiter(physical_event.handle())
                .expect("create physical completion awaiter"),
        );
        let drain = tokio::spawn(await_remote_g4_proven_drain(
            RemoteG4DrainOwnership::new(request),
            notification,
        ));

        physical_event
            .poison("worker response did not prove upload drain")
            .expect("poison the physical notification");
        let error = drain
            .await
            .expect("remote drain task returns")
            .expect_err("ambiguous notification must not settle");
        assert!(error.contains("completion is ambiguous"));
        assert_eq!(observer.use_count(), 2);
        assert!(
            tokio::time::timeout(Duration::from_millis(25), handle.cancel().wait())
                .await
                .is_err(),
            "ambiguous completion must retain its cancellation unit",
        );
    }

    #[tokio::test]
    async fn proven_remote_drain_releases_guards_and_settles() {
        let (request, observer, mut handle) = remote_request_with_guard();
        let events = Arc::new(velo::EventManager::local());
        let physical_event = events.new_event().expect("create physical event");
        let notification = TransferCompleteNotification::from_awaiter(
            events
                .awaiter(physical_event.handle())
                .expect("create physical completion awaiter"),
        );
        let drain = tokio::spawn(await_remote_g4_proven_drain(
            RemoteG4DrainOwnership::new(request),
            notification,
        ));

        physical_event
            .trigger()
            .expect("complete the physical notification");
        drain
            .await
            .expect("remote drain task returns")
            .expect("proven drain succeeds");

        assert_eq!(observer.use_count(), 1);
        let result = handle
            .wait()
            .await
            .expect("route publishes terminal success");
        assert_eq!(result.status, TransferStatus::Complete);
        tokio::time::timeout(Duration::from_millis(100), handle.cancel().wait())
            .await
            .expect("proven drain releases its cancellation unit");
    }

    #[tokio::test]
    async fn dispatch_error_after_proven_child_drain_releases_and_fails_route() {
        let (request, observer, mut handle) = remote_request_with_guard();
        let events = Arc::new(velo::EventManager::local());
        let notification = TransferCompleteNotification::aggregate_results(
            vec![
                Ok(TransferCompleteNotification::completed()),
                Err(anyhow!("second worker rejected the dispatch")),
            ],
            &events,
            &tokio::runtime::Handle::current(),
        )
        .expect("launched worker returns a completion receipt");

        let error =
            await_remote_g4_proven_drain(RemoteG4DrainOwnership::new(request), notification)
                .await
                .expect_err("dispatch failure reaches the logical route");
        assert!(error.contains("drained after a dispatch failure"));
        assert_eq!(observer.use_count(), 1);
        let result = handle
            .wait()
            .await
            .expect("proven drain publishes a terminal failure");
        assert_eq!(result.status, TransferStatus::Failed);
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("second worker rejected the dispatch"))
        );
        tokio::time::timeout(Duration::from_millis(100), handle.cancel().wait())
            .await
            .expect("proven drain releases its cancellation unit");
    }

    #[tokio::test]
    async fn child_notification_error_keeps_remote_drain_unproven() {
        let (request, observer, handle) = remote_request_with_guard();
        let events = Arc::new(velo::EventManager::local());
        let event = events.new_event().expect("create worker completion event");
        let child = TransferCompleteNotification::from_awaiter(
            events
                .awaiter(event.handle())
                .expect("create worker completion awaiter"),
        );
        let notification = TransferCompleteNotification::aggregate_results(
            vec![
                Ok(child),
                Err(anyhow!("second worker rejected the dispatch")),
            ],
            &events,
            &tokio::runtime::Handle::current(),
        )
        .expect("launched worker returns a completion receipt");
        event
            .poison("worker completion did not prove drain")
            .expect("poison child receipt");

        let error =
            await_remote_g4_proven_drain(RemoteG4DrainOwnership::new(request), notification)
                .await
                .expect_err("unproven child completion must not settle");
        assert!(error.contains("completion is ambiguous"));
        assert_eq!(observer.use_count(), 2);
        assert!(
            tokio::time::timeout(Duration::from_millis(25), handle.cancel().wait())
                .await
                .is_err(),
            "unproven child completion retains its cancellation unit",
        );
    }

    #[tokio::test]
    async fn synchronous_remote_dispatch_error_releases_and_fails_route() {
        let leader = test_leader().await;
        let (request, observer, mut handle) = remote_request_with_guard();
        let (tx, rx) = mpsc::channel(1);
        let remote_task = tokio::spawn(run(rx, leader));

        tx.send(request).await.expect("send remote G4 request");
        let result = tokio::time::timeout(Duration::from_millis(100), handle.wait())
            .await
            .expect("synchronous dispatch error publishes a terminal result")
            .expect("transfer handle returns a result");

        assert_eq!(result.status, TransferStatus::Failed);
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("No parallel worker configured"))
        );
        assert_eq!(observer.use_count(), 1);
        tokio::time::timeout(Duration::from_millis(100), handle.cancel().wait())
            .await
            .expect("pre-dispatch failure releases its cancellation unit");

        drop(tx);
        remote_task.await.expect("remote G4 task exits");
    }

    #[tokio::test]
    async fn remote_g4_request_holds_the_shared_cancellation_unit() {
        let transfer_id = TransferId::new();
        let (state, handle) = TransferState::new(transfer_id, vec![7]);
        let state = Arc::new(std::sync::Mutex::new(state));
        let token = state.lock().unwrap().cancellation_token();
        let root = token.root_unit().expect("create root cancellation unit");
        assert!(root.claim_commitment());
        let cancellation = root
            .fan_out(1)
            .pop()
            .expect("remote G4 owns one route unit");
        let request = RemoteG4OffloadRequest {
            transfer_id,
            keys: vec![SequenceHash::new(7, None, 0)],
            block_ids: vec![7],
            source_guards: Vec::new(),
            state: Arc::clone(&state),
            cancellation,
        };

        assert!(request.cancellation.claim_commitment());
        assert!(Arc::ptr_eq(&request.state, &state));
        assert!(
            tokio::time::timeout(Duration::from_millis(25), handle.cancel().wait(),)
                .await
                .is_err(),
            "the remote request must keep cancellation pending",
        );

        drop(request);
        tokio::time::timeout(Duration::from_millis(100), handle.cancel().wait())
            .await
            .expect("remote request release confirms cancellation");
    }

    #[tokio::test]
    async fn remote_chain_route_defers_terminal_failure_until_request_settlement() {
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(1)
                .block_size(4)
                .build(),
        );
        let token_block = crate::testing::create_sequential_block(0, manager.block_size());
        let sequence_hash = crate::testing::populate_manager_with_blocks(
            manager.as_ref(),
            std::slice::from_ref(&token_block),
        )
        .expect("populate source manager")[0];
        let block = manager
            .match_blocks(&[sequence_hash])
            .pop()
            .expect("match source block");
        let block_id = block.block_id();
        let transfer_id = TransferId::new();
        let (mut state, mut handle) = TransferState::new(transfer_id, vec![block_id]);
        let token = state.cancellation_token();
        let cancellation = token.root_unit().expect("create root cancellation unit");
        assert!(cancellation.claim_commitment());
        state.add_passed([block_id]);
        state.mark_committed_blocks([block_id]);
        state.mark_completed([block_id]);
        let state = Arc::new(std::sync::Mutex::new(state));
        let (chain_tx, chain_rx) = mpsc::channel(1);
        let (remote_tx, mut remote_rx) = mpsc::channel(1);
        let router = tokio::spawn(super::super::chain_router::run(
            chain_rx,
            None,
            None,
            Some(remote_tx),
        ));

        chain_tx
            .send(ChainOutput {
                transfer_id,
                blocks: vec![block],
                state: Arc::clone(&state),
                cancellation,
            })
            .await
            .expect("send source chain output");
        let request = remote_rx.recv().await.expect("receive remote G4 request");

        assert!(
            tokio::time::timeout(Duration::from_millis(25), handle.wait())
                .await
                .is_err(),
            "the held remote request keeps the logical transfer active",
        );
        request.fail_before_dispatch("injected remote route failure".to_string());
        let result = handle
            .wait()
            .await
            .expect("remote failure publishes a result");
        assert_eq!(result.status, TransferStatus::Failed);

        drop(chain_tx);
        router
            .await
            .expect("chain router stops after channel close");
    }

    #[tokio::test]
    async fn remote_g4_request_retains_source_guards_until_request_drop() {
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(1)
                .block_size(4)
                .build(),
        );
        let token_block = crate::testing::create_sequential_block(0, manager.block_size());
        let sequence_hash = crate::testing::populate_manager_with_blocks(
            manager.as_ref(),
            std::slice::from_ref(&token_block),
        )
        .expect("populate source manager")[0];
        let block = manager
            .match_blocks(&[sequence_hash])
            .pop()
            .expect("match source block");
        let observer = block.clone();
        let block_id = block.block_id();
        let transfer_id = TransferId::new();
        let (state, _handle) = TransferState::new(transfer_id, vec![block_id]);
        let token = state.cancellation_token();
        let cancellation = token.root_unit().expect("create root cancellation unit");
        assert!(cancellation.claim_commitment());
        let state = Arc::new(std::sync::Mutex::new(state));
        let (chain_tx, chain_rx) = mpsc::channel(1);
        let (remote_tx, mut remote_rx) = mpsc::channel(1);
        let router = tokio::spawn(super::super::chain_router::run(
            chain_rx,
            None,
            None,
            Some(remote_tx),
        ));

        chain_tx
            .send(ChainOutput {
                transfer_id,
                blocks: vec![block],
                state,
                cancellation,
            })
            .await
            .expect("send source chain output");
        let request = remote_rx.recv().await.expect("receive remote G4 request");

        assert_eq!(observer.use_count(), 2);
        drop(request);
        assert_eq!(observer.use_count(), 1);

        drop(chain_tx);
        router.await.expect("chain router exits");
    }
}
