// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test workers and checkpoints shared by the engine's onboard tests.

#![allow(clippy::disallowed_macros)]

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::time::Duration;

use anyhow::{Result, anyhow};
use futures::future::BoxFuture;
use kvbm_common::{BlockId, LogicalLayoutHandle, LogicalResourceId, SequenceHash};
use kvbm_logical::{BlockManager, BlockManagerSet, BlockRegistry};
use kvbm_observability::KvbmObservability;
use kvbm_physical::TransferOptions;
use kvbm_physical::transfer::{TransferCompleteNotification, TransferDrainOutcome};
use kvbm_protocols::connector::ActionId;

use crate::G2;
use crate::leader::InstanceLeader;
use crate::object::ObjectBlockOps;
use crate::testing::{managers::TestManagerBuilder, messenger::create_messenger_tcp};
use crate::worker::group::ParallelWorkers;
use crate::worker::{
    ConnectRemoteResponse, ImportMetadataResponse, InstanceId, RemoteDescriptor, SerializedLayout,
    SerializedLayoutResponse, Worker, WorkerTransfers,
};

/// Unconsumed driver completions by action ID. A resumed driver reports
/// another completion under the same ID.
static FINISHED_ONBOARD_DRIVERS: LazyLock<Mutex<HashMap<ActionId, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Record that the onboard driver task for `action_id` has done all its work.
/// The onboard tasks call this as their last statement.
pub(super) fn note_onboard_driver_finished(action_id: ActionId) {
    *FINISHED_ONBOARD_DRIVERS
        .lock()
        .unwrap()
        .entry(action_id)
        .or_default() += 1;
}

/// Wait for and consume the next completion reported for `action_id`.
///
/// The driver writes the handle's cell before it calls the worker sink, so a
/// test that waits only for a terminal on the handle can read the sink too
/// early. Reports stay queued until observed, so two waits on a resumed
/// action see two distinct driver runs.
pub(super) async fn wait_onboard_driver_finished(action_id: ActionId) {
    for _ in 0..500 {
        {
            let mut finished = FINISHED_ONBOARD_DRIVERS.lock().unwrap();
            if let Some(count) = finished.get_mut(&action_id) {
                *count -= 1;
                if *count == 0 {
                    finished.remove(&action_id);
                }
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the onboard driver task for {action_id:?} did not return");
}

#[tokio::test(start_paused = true)]
async fn onboard_driver_wait_consumes_one_completion() {
    let action = ActionId::new();
    note_onboard_driver_finished(action);
    wait_onboard_driver_finished(action).await;
    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            wait_onboard_driver_finished(action)
        )
        .await
        .is_err(),
        "a resumed driver must report a new completion",
    );

    note_onboard_driver_finished(action);
    note_onboard_driver_finished(action);
    wait_onboard_driver_finished(action).await;
    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            wait_onboard_driver_finished(action)
        )
        .await
        .is_ok(),
        "both reported completions must survive until their waits",
    );
}

/// Build a leader with one G2 manager of `g2_blocks` blocks for `resource`
/// and the given workers.
pub(super) async fn build_leader_with_workers(
    resource: LogicalResourceId,
    g2_blocks: usize,
    block_size: usize,
    workers: Arc<dyn ParallelWorkers>,
) -> Result<(InstanceLeader, Arc<BlockManager<G2>>)> {
    let messenger = create_messenger_tcp().await?;
    build_leader_with_messenger(messenger, resource, g2_blocks, block_size, workers)
}

pub(super) fn build_leader_with_messenger(
    messenger: Arc<velo::Messenger>,
    resource: LogicalResourceId,
    g2_blocks: usize,
    block_size: usize,
    workers: Arc<dyn ParallelWorkers>,
) -> Result<(InstanceLeader, Arc<BlockManager<G2>>)> {
    let (leader, mut managers) =
        build_leader_with_resources(messenger, &[resource], g2_blocks, block_size, workers)?;
    Ok((leader, managers.remove(0)))
}

pub(super) fn build_leader_with_resources(
    messenger: Arc<velo::Messenger>,
    resources: &[LogicalResourceId],
    g2_blocks: usize,
    block_size: usize,
    workers: Arc<dyn ParallelWorkers>,
) -> Result<(InstanceLeader, Vec<Arc<BlockManager<G2>>>)> {
    let registry = BlockRegistry::builder().build();
    let mut set = BlockManagerSet::new();
    let mut managers = Vec::with_capacity(resources.len());
    for &resource in resources {
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(g2_blocks)
                .block_size(block_size)
                .registry(registry.clone())
                .build(),
        );
        set.insert(resource, manager.clone())?;
        managers.push(manager);
    }
    let leader = InstanceLeader::builder()
        .messenger(messenger)
        .registry(registry)
        .g2_manager_set(Arc::new(set), resources[0])
        .parallel_worker(workers)
        .observability(Arc::new(KvbmObservability::default()))
        .build()?;
    Ok((leader, managers))
}

pub(super) struct CompletedParallelWorkers;

impl WorkerTransfers for CompletedParallelWorkers {
    fn execute_local_transfer(
        &self,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        Ok(TransferCompleteNotification::completed())
    }

    fn execute_local_transfer_for_resource(
        &self,
        _resource: LogicalResourceId,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        Ok(TransferCompleteNotification::completed())
    }

    fn execute_remote_onboard(
        &self,
        _src: RemoteDescriptor,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote onboard is outside this local transaction test")
    }

    fn execute_remote_offload(
        &self,
        _src: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst: RemoteDescriptor,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote offload is outside this local transaction test")
    }

    fn connect_remote(
        &self,
        _instance_id: InstanceId,
        _metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        Ok(ConnectRemoteResponse::ready())
    }

    fn has_remote_metadata(&self, _instance_id: InstanceId) -> bool {
        false
    }

    fn execute_remote_onboard_for_instance(
        &self,
        _instance_id: InstanceId,
        _remote_logical_type: LogicalLayoutHandle,
        _src_block_ids: Vec<BlockId>,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote instance onboard is outside this local transaction test")
    }
}

impl ObjectBlockOps for CompletedParallelWorkers {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        Box::pin(async move { keys.into_iter().map(|key| (key, None)).collect() })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }
}

impl ParallelWorkers for CompletedParallelWorkers {
    fn export_metadata(&self) -> Result<Vec<SerializedLayoutResponse>> {
        Ok(Vec::new())
    }

    fn import_metadata(
        &self,
        _metadata: Vec<SerializedLayout>,
    ) -> Result<Vec<ImportMetadataResponse>> {
        Ok(Vec::new())
    }

    fn worker_count(&self) -> usize {
        0
    }

    fn workers(&self) -> &[Arc<dyn Worker>] {
        &[]
    }
}

/// A worker whose transfer for one resource is accepted and then held.
///
/// The worker signals that the copy started, then keeps the copy running
/// until the test decides how it ends. The running copy is a guard
/// ([`RunningCopy`]) separate from the receipt, so a test can resolve the
/// receipt while the copy is still running.
pub(super) struct HangingParallelWorkers {
    resource: LogicalResourceId,
    started: tokio::sync::Semaphore,
    /// The held transfer's reply to the leader.
    reply: Mutex<Option<tokio::sync::oneshot::Sender<TransferDrainOutcome>>>,
    /// The running copy, until the test takes it.
    running: Mutex<Option<RunningCopy>>,
    /// Sees the running copy wherever it is held.
    copy: Mutex<Weak<()>>,
    /// The outcome the worker sent for the held transfer.
    delivered: Mutex<Option<DeliveredOutcome>>,
    panic_resource: Mutex<Option<LogicalResourceId>>,
    dispatches: Mutex<Vec<LogicalResourceId>>,
    completed: CompletedParallelWorkers,
}

/// The kind of outcome a [`HangingParallelWorkers`] sent to the leader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeliveredOutcome {
    Completed,
    DrainedWithError,
    Unproven,
}

/// The held copy. While a `RunningCopy` exists the copy may still write its
/// destination blocks; dropping it ends the copy.
pub(super) struct RunningCopy(#[allow(dead_code)] Arc<()>);

impl HangingParallelWorkers {
    pub(super) fn new(resource: LogicalResourceId) -> Arc<Self> {
        Arc::new(Self {
            resource,
            started: tokio::sync::Semaphore::new(0),
            reply: Mutex::new(None),
            running: Mutex::new(None),
            copy: Mutex::new(Weak::new()),
            delivered: Mutex::new(None),
            panic_resource: Mutex::new(None),
            dispatches: Mutex::new(Vec::new()),
            completed: CompletedParallelWorkers,
        })
    }

    /// Wait until the held transfer is dispatched. Panics after five seconds,
    /// so a failure before dispatch does not hang the test.
    pub(super) async fn wait_started(&self) {
        tokio::time::timeout(Duration::from_secs(5), self.started.acquire())
            .await
            .expect("the hanging transfer was not dispatched within five seconds")
            .expect("hanging transfer semaphore closed")
            .forget();
    }

    /// The outcome the worker sent for the held transfer, if it sent one.
    pub(super) fn delivered(&self) -> Option<DeliveredOutcome> {
        *self.delivered.lock().unwrap()
    }

    /// Take the running copy, so the test owns when it ends.
    pub(super) fn take_running_copy(&self) -> Result<RunningCopy> {
        self.running
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| anyhow!("no running copy to take"))
    }

    /// `true` while the held copy has not ended.
    pub(super) fn copy_running(&self) -> bool {
        self.copy.lock().unwrap().strong_count() > 0
    }

    fn reply(&self, outcome: TransferDrainOutcome) -> Result<()> {
        let kind = match &outcome {
            TransferDrainOutcome::Completed => DeliveredOutcome::Completed,
            TransferDrainOutcome::DrainedWithError(_) => DeliveredOutcome::DrainedWithError,
            TransferDrainOutcome::Unproven(_) => DeliveredOutcome::Unproven,
        };
        self.reply
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| anyhow!("hanging transfer was not dispatched"))?
            .send(outcome)
            .map_err(|_| anyhow!("the leader dropped the hanging transfer receipt"))?;
        *self.delivered.lock().unwrap() = Some(kind);
        Ok(())
    }

    /// Finish the copy, then report success.
    pub(super) fn release(&self) -> Result<()> {
        drop(self.running.lock().unwrap().take());
        self.reply(TransferDrainOutcome::Completed)
    }

    /// Lose the worker's reply while `copy` is still running. The receipt
    /// resolves `Unproven`, as it does when the RPC to a worker fails after
    /// the worker accepted the request.
    pub(super) fn lose_reply(&self, _copy: &RunningCopy) -> Result<()> {
        anyhow::ensure!(self.copy_running(), "the copy must still be running");
        self.reply(TransferDrainOutcome::Unproven(anyhow!(
            "injected lost reply: the worker copy may still be running"
        )))
    }

    /// End the copy, then report that it drained with an error.
    pub(super) fn fail_drained(&self, copy: RunningCopy) -> Result<()> {
        drop(copy);
        anyhow::ensure!(!self.copy_running(), "the copy must have ended");
        self.reply(TransferDrainOutcome::DrainedWithError(anyhow!(
            "injected transfer failure after the copy drained"
        )))
    }

    pub(super) fn dispatches(&self) -> Vec<LogicalResourceId> {
        self.dispatches.lock().unwrap().clone()
    }

    pub(super) fn panic_on_dispatch(&self, resource: LogicalResourceId) {
        *self.panic_resource.lock().unwrap() = Some(resource);
    }
}

impl WorkerTransfers for HangingParallelWorkers {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.completed
            .execute_local_transfer(src, dst, src_block_ids, dst_block_ids, options)
    }

    fn execute_local_transfer_for_resource(
        &self,
        resource: LogicalResourceId,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.dispatches.lock().unwrap().push(resource);
        if *self.panic_resource.lock().unwrap() == Some(resource) {
            panic!("injected physical bundle task failure for {resource:?}");
        }
        if resource != self.resource {
            return self.completed.execute_local_transfer_for_resource(
                resource,
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                options,
            );
        }
        let (reply, outcome) = tokio::sync::oneshot::channel();
        let mut held = self.reply.lock().unwrap();
        anyhow::ensure!(held.is_none(), "hanging transfer dispatched twice");
        *held = Some(reply);
        let copy = Arc::new(());
        *self.copy.lock().unwrap() = Arc::downgrade(&copy);
        *self.running.lock().unwrap() = Some(RunningCopy(copy));
        self.started.add_permits(1);
        Ok(TransferCompleteNotification::from_drain_future(
            async move {
                outcome.await.unwrap_or_else(|_| {
                    TransferDrainOutcome::Unproven(anyhow!("hanging worker dropped its reply"))
                })
            },
        ))
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.completed
            .execute_remote_onboard(src, dst, dst_block_ids, options)
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst: RemoteDescriptor,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.completed
            .execute_remote_offload(src, src_block_ids, dst, options)
    }

    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        self.completed.connect_remote(instance_id, metadata)
    }

    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool {
        self.completed.has_remote_metadata(instance_id)
    }

    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.completed.execute_remote_onboard_for_instance(
            instance_id,
            remote_logical_type,
            src_block_ids,
            dst,
            dst_block_ids,
            options,
        )
    }
}

impl ObjectBlockOps for HangingParallelWorkers {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        self.completed.has_blocks(keys)
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        self.completed.put_blocks(keys, layout, block_ids)
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        self.completed.get_blocks(keys, layout, block_ids)
    }
}

impl ParallelWorkers for HangingParallelWorkers {
    fn export_metadata(&self) -> Result<Vec<SerializedLayoutResponse>> {
        self.completed.export_metadata()
    }

    fn import_metadata(
        &self,
        metadata: Vec<SerializedLayout>,
    ) -> Result<Vec<ImportMetadataResponse>> {
        self.completed.import_metadata(metadata)
    }

    fn worker_count(&self) -> usize {
        1
    }

    fn workers(&self) -> &[Arc<dyn Worker>] {
        &[]
    }
}
