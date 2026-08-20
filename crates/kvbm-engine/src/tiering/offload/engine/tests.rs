// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use futures::future::BoxFuture;
use kvbm_common::LogicalResourceId;
use kvbm_logical::BlockManagerSet;
use kvbm_physical::manager::SerializedLayout;
use kvbm_physical::transfer::{TransferCompleteNotification, TransferOptions};

use super::super::container::OffloadContainer;
use super::super::handle::TransferStatus;
use super::super::pipeline::{ObjectPipelineBuilder, PipelineBuilder, PreconditionAwaiter};
use super::super::queue::CancellableQueue;
use super::super::source::ExternalBlock;
use super::*;
use crate::SequenceHash;
use crate::testing::{TestManagerBuilder, TestRegistryBuilder, create_messenger_tcp};
use crate::worker::group::ParallelWorkers;
use crate::worker::{
    ConnectRemoteResponse, ImportMetadataResponse, RemoteDescriptor, SerializedLayoutResponse,
    Worker, WorkerTransfers,
};

#[derive(Default)]
struct ResourceRecordingWorkers {
    resources: Mutex<Vec<LogicalResourceId>>,
}

impl ResourceRecordingWorkers {
    fn resources(&self) -> Vec<LogicalResourceId> {
        self.resources.lock().unwrap().clone()
    }
}

impl WorkerTransfers for ResourceRecordingWorkers {
    fn execute_local_transfer(
        &self,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("the resource-aware transfer path is required")
    }

    fn execute_local_transfer_for_resource(
        &self,
        resource: LogicalResourceId,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        self.resources.lock().unwrap().push(resource);
        Ok(TransferCompleteNotification::completed())
    }

    fn execute_remote_onboard(
        &self,
        _src: RemoteDescriptor,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote onboard is outside this test")
    }

    fn execute_remote_offload(
        &self,
        _src: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst: RemoteDescriptor,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote offload is outside this test")
    }

    fn connect_remote(
        &self,
        _instance_id: crate::InstanceId,
        _metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        Ok(ConnectRemoteResponse::ready())
    }

    fn has_remote_metadata(&self, _instance_id: crate::InstanceId) -> bool {
        false
    }

    fn execute_remote_onboard_for_instance(
        &self,
        _instance_id: crate::InstanceId,
        _remote_logical_type: LogicalLayoutHandle,
        _src_block_ids: Vec<BlockId>,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("remote instance onboard is outside this test")
    }
}

impl ObjectBlockOps for ResourceRecordingWorkers {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        Box::pin(async move { keys.into_iter().map(|hash| (hash, None)).collect() })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _src_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _dst_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }
}

impl ParallelWorkers for ResourceRecordingWorkers {
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

#[derive(Default)]
struct GatedObjectBlockOps {
    entered: AtomicBool,
    calls: AtomicUsize,
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

impl GatedObjectBlockOps {
    async fn wait_until_started(&self) {
        while !self.entered.load(Ordering::SeqCst) {
            self.started.notified().await;
        }
    }

    async fn wait_for_calls(&self, expected: usize) {
        while self.calls.load(Ordering::SeqCst) < expected {
            self.started.notified().await;
        }
    }
}

impl ObjectBlockOps for GatedObjectBlockOps {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        Box::pin(async move { keys.into_iter().map(|hash| (hash, None)).collect() })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _src_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.store(true, Ordering::SeqCst);
        self.started.notify_waiters();
        let release = Arc::clone(&self.release).notified_owned();
        Box::pin(async move {
            release.await;
            keys.into_iter().map(Ok).collect()
        })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _dst_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Ok).collect() })
    }
}

async fn test_engine() -> OffloadEngine {
    let messenger = create_messenger_tcp().await.expect("create messenger");
    let registry = Arc::new(TestRegistryBuilder::new().build());
    let g2_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(4)
            .block_size(4)
            .registry(registry.as_ref().clone())
            .build(),
    );
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry.as_ref().clone())
            .g2_manager(g2_manager)
            .workers(Vec::new())
            .build()
            .expect("build test leader"),
    );
    OffloadEngine::builder(leader)
        .build()
        .expect("build offload engine")
}

#[tokio::test]
async fn builder_does_not_require_an_engine_registry() {
    let messenger = create_messenger_tcp().await.expect("create messenger");
    let registry = Arc::new(TestRegistryBuilder::new().build());
    let g2_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .registry(registry.as_ref().clone())
            .build(),
    );
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry.as_ref().clone())
            .g2_manager(g2_manager)
            .workers(Vec::new())
            .build()
            .expect("build test leader"),
    );

    OffloadEngine::builder(leader)
        .build()
        .expect("the leader owns the registry dependency");
}

#[tokio::test]
async fn builder_rejects_capacity_for_a_different_selected_g2_manager() {
    let messenger = create_messenger_tcp().await.expect("create messenger");
    let primary_resource = LogicalResourceId(2);
    let secondary_resource = LogicalResourceId(7);
    let primary_registry = Arc::new(TestRegistryBuilder::new().build());
    let secondary_registry = Arc::new(TestRegistryBuilder::new().build());
    let primary_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .registry(primary_registry.as_ref().clone())
            .build(),
    );
    let secondary_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .registry(secondary_registry.as_ref().clone())
            .build(),
    );
    let mut managers = BlockManagerSet::new();
    managers
        .insert(primary_resource, Arc::clone(&primary_manager))
        .expect("insert primary manager");
    managers
        .insert(secondary_resource, Arc::clone(&secondary_manager))
        .expect("insert secondary manager");
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(primary_registry.as_ref().clone())
            .g2_manager_set(Arc::new(managers), primary_resource)
            .workers(Vec::new())
            .build()
            .expect("build resource-aware leader"),
    );

    let result = OffloadEngine::builder(leader)
        .with_g2_capacity(crate::g2_capacity::direct_g2_capacity(primary_manager))
        .with_g1_to_g2_pipeline(
            PipelineBuilder::<G1, G2>::new()
                .resource(secondary_resource)
                .build(),
        )
        .build();

    let error = match result {
        Ok(_) => panic!("mismatched capacity must be rejected"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("G2 capacity manager does not match selected resource"),
        "unexpected error: {error:#}",
    );
}

#[tokio::test]
async fn builder_accepts_capacity_for_the_selected_g2_manager() {
    let messenger = create_messenger_tcp().await.expect("create messenger");
    let primary_resource = LogicalResourceId(2);
    let secondary_resource = LogicalResourceId(7);
    let primary_registry = Arc::new(TestRegistryBuilder::new().build());
    let secondary_registry = Arc::new(TestRegistryBuilder::new().build());
    let primary_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .registry(primary_registry.as_ref().clone())
            .build(),
    );
    let secondary_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .registry(secondary_registry.as_ref().clone())
            .build(),
    );
    let mut managers = BlockManagerSet::new();
    managers
        .insert(primary_resource, primary_manager)
        .expect("insert primary manager");
    managers
        .insert(secondary_resource, Arc::clone(&secondary_manager))
        .expect("insert secondary manager");
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(primary_registry.as_ref().clone())
            .g2_manager_set(Arc::new(managers), primary_resource)
            .workers(Vec::new())
            .build()
            .expect("build resource-aware leader"),
    );

    OffloadEngine::builder(leader)
        .with_g2_capacity(crate::g2_capacity::direct_g2_capacity(secondary_manager))
        .with_g1_to_g2_pipeline(
            PipelineBuilder::<G1, G2>::new()
                .resource(secondary_resource)
                .build(),
        )
        .build()
        .expect("capacity belongs to the selected resource manager");
}

#[tokio::test]
async fn builder_rejects_zero_flush_interval_for_a_block_pipeline() {
    let messenger = create_messenger_tcp().await.expect("create messenger");
    let registry = Arc::new(TestRegistryBuilder::new().build());
    let g2_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .registry(registry.as_ref().clone())
            .build(),
    );
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry.as_ref().clone())
            .g2_manager(g2_manager)
            .workers(Vec::new())
            .build()
            .expect("build test leader"),
    );

    let result = OffloadEngine::builder(leader)
        .with_g1_to_g2_pipeline(
            PipelineBuilder::<G1, G2>::new()
                .flush_interval(Duration::ZERO)
                .build(),
        )
        .build();
    let error = match result {
        Ok(_) => panic!("zero flush interval must be rejected"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("flush interval"));
}

#[tokio::test]
async fn builder_rejects_zero_sweep_interval_for_an_object_pipeline() {
    let messenger = create_messenger_tcp().await.expect("create messenger");
    let registry = Arc::new(TestRegistryBuilder::new().build());
    let g2_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .registry(registry.as_ref().clone())
            .build(),
    );
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry.as_ref().clone())
            .g2_manager(g2_manager)
            .workers(Vec::new())
            .build()
            .expect("build test leader"),
    );

    let result = OffloadEngine::builder(leader)
        .with_object_ops(Arc::new(GatedObjectBlockOps::default()))
        .with_g2_to_g4_pipeline(
            ObjectPipelineBuilder::<G2>::new()
                .sweep_interval(Duration::ZERO)
                .build(),
        )
        .build();
    let error = match result {
        Ok(_) => panic!("zero sweep interval must be rejected"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("sweep interval"));
}

#[tokio::test]
async fn engine_drop_releases_precommit_batches_behind_a_gated_object_transfer() {
    let messenger = create_messenger_tcp().await.expect("create messenger");
    let registry = Arc::new(TestRegistryBuilder::new().build());
    let g2_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(16)
            .block_size(4)
            .registry(registry.as_ref().clone())
            .build(),
    );
    let token_sequence = crate::testing::create_token_sequence(11, g2_manager.block_size(), 0);
    let hashes =
        crate::testing::populate_manager_with_blocks(g2_manager.as_ref(), token_sequence.blocks())
            .expect("populate G2 source blocks");
    let mut sources_and_observers: Vec<_> = hashes
        .iter()
        .map(|hash| {
            let source = g2_manager
                .match_blocks(&[*hash])
                .pop()
                .expect("match source block");
            let observer = source.clone();
            (*hash, Some(source), observer)
        })
        .collect();
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry.as_ref().clone())
            .g2_manager(Arc::clone(&g2_manager))
            .workers(Vec::new())
            .build()
            .expect("build test leader"),
    );
    let leader_observer = Arc::downgrade(&leader);
    let pending_tracker = Arc::new(super::super::pending::PendingTracker::new());
    let object_ops = Arc::new(GatedObjectBlockOps::default());
    let engine = OffloadEngine::builder(Arc::clone(&leader))
        .with_object_ops(object_ops.clone())
        .with_g2_to_g4_pipeline(
            ObjectPipelineBuilder::<G2>::new()
                .batch_size(1)
                .min_batch_size(1)
                .max_concurrent_transfers(1)
                .pending_tracker(Arc::clone(&pending_tracker))
                .build(),
        )
        .build()
        .expect("build object offload engine");

    let mut handles = Vec::new();
    for (_, source, _) in &mut sources_and_observers {
        handles.push(
            engine
                .enqueue_g2_to_g4(SourceBlocks::Strong(vec![
                    source.take().expect("source block moves into the pipeline"),
                ]))
                .expect("enqueue source block"),
        );
    }
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        object_ops.wait_until_started(),
    )
    .await
    .expect("first object transfer must commit");
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while pending_tracker.len() != 11 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("all batches must reach the full executor boundary");
    assert_eq!(object_ops.calls.load(Ordering::SeqCst), 1);

    drop(engine);
    drop(leader);

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while leader_observer.upgrade().is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("precommit stage tasks must release the leader");
    assert_eq!(object_ops.calls.load(Ordering::SeqCst), 1);

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while pending_tracker.len() != 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("only the committed batches retain pending guards");
    for (hash, _, observer) in &sources_and_observers {
        if pending_tracker.is_pending(hash) {
            assert!(
                observer.use_count() > 1,
                "committed source guard stays live"
            );
        } else {
            assert_eq!(
                observer.use_count(),
                1,
                "precommit source guard must release on engine drop",
            );
        }
    }

    object_ops.release.notify_one();
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        object_ops.wait_for_calls(2),
    )
    .await
    .expect("second committed batch must continue after the first drain");
    object_ops.release.notify_one();
    for handle in &mut handles {
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), handle.wait()).await;
    }
}

#[tokio::test]
async fn dropping_engine_releases_idle_pipeline_stage_tasks() {
    let messenger = create_messenger_tcp().await.expect("create messenger");
    let registry = Arc::new(TestRegistryBuilder::new().build());
    let g2_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .registry(registry.as_ref().clone())
            .build(),
    );
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry.as_ref().clone())
            .g2_manager(g2_manager)
            .workers(Vec::new())
            .build()
            .expect("build test leader"),
    );
    let leader_observer = Arc::downgrade(&leader);
    let engine = OffloadEngine::builder(Arc::clone(&leader))
        .with_g1_to_g2_pipeline(PipelineBuilder::<G1, G2>::new().build())
        .build()
        .expect("build offload engine with one pipeline");

    drop(engine);
    drop(leader);

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while leader_observer.upgrade().is_some() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("pipeline stage tasks must terminate after engine drop");
}

#[tokio::test]
async fn dropping_engine_preserves_committed_object_transfer() {
    let messenger = create_messenger_tcp().await.expect("create messenger");
    let registry = Arc::new(TestRegistryBuilder::new().build());
    let g2_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(1)
            .block_size(4)
            .registry(registry.as_ref().clone())
            .build(),
    );
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry.as_ref().clone())
            .g2_manager(g2_manager)
            .workers(Vec::new())
            .build()
            .expect("build test leader"),
    );
    let object_ops = Arc::new(GatedObjectBlockOps::default());
    let engine = OffloadEngine::builder(leader)
        .with_object_ops(object_ops.clone())
        .with_g2_to_g4_pipeline(
            ObjectPipelineBuilder::<G2>::new()
                .batch_size(1)
                .min_batch_size(1)
                .build(),
        )
        .build()
        .expect("object operations are sufficient for a local G4 pipeline");
    let mut handle = engine
        .enqueue_g2_to_g4(SourceBlocks::External(vec![ExternalBlock::new(
            0,
            SequenceHash::new(31, None, 0),
        )]))
        .expect("enqueue object transfer");
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        object_ops.wait_until_started(),
    )
    .await
    .expect("object transfer must cross the commitment boundary");

    drop(engine);
    object_ops.release.notify_one();

    let result = tokio::time::timeout(std::time::Duration::from_secs(1), handle.wait())
        .await
        .expect("committed object transfer must survive engine drop")
        .expect("committed object transfer must publish a result");
    assert_eq!(result.status, TransferStatus::Complete);
}

#[tokio::test]
async fn two_resource_engines_dispatch_through_their_own_capacity_and_route() {
    let messenger = create_messenger_tcp().await.expect("create messenger");
    let primary = LogicalResourceId(2);
    let secondary = LogicalResourceId(7);
    let primary_registry = Arc::new(TestRegistryBuilder::new().build());
    let secondary_registry = Arc::new(TestRegistryBuilder::new().build());
    let primary_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .registry(primary_registry.as_ref().clone())
            .build(),
    );
    let secondary_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(2)
            .block_size(4)
            .registry(secondary_registry.as_ref().clone())
            .build(),
    );
    let mut managers = BlockManagerSet::new();
    managers
        .insert(primary, Arc::clone(&primary_manager))
        .expect("insert primary manager");
    managers
        .insert(secondary, Arc::clone(&secondary_manager))
        .expect("insert secondary manager");
    let workers = Arc::new(ResourceRecordingWorkers::default());
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(primary_registry.as_ref().clone())
            .g2_manager_set(Arc::new(managers), primary)
            .parallel_worker(workers.clone())
            .build()
            .expect("build two-resource leader"),
    );

    for (resource, hash_value) in [(primary, 11), (secondary, 22)] {
        let capacity = leader
            .g2_capacity_for(resource)
            .expect("resolve the resource capacity");
        let config = PipelineBuilder::<G1, G2>::new()
            .resource(resource)
            .batch_size(1)
            .min_batch_size(1)
            .build();
        let engine = OffloadEngine::builder(Arc::clone(&leader))
            .with_g2_capacity(capacity)
            .with_g1_to_g2_pipeline(config)
            .build()
            .expect("build one resource offload engine");
        let hash = SequenceHash::new(hash_value, None, hash_value);
        let mut handle = engine
            .enqueue_g1_to_g2(SourceBlocks::External(vec![ExternalBlock::new(0, hash)]))
            .expect("enqueue one resource transfer");
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), handle.wait())
            .await
            .expect("resource transfer timed out")
            .expect("resource transfer returned no result");
        assert_eq!(result.status, TransferStatus::Complete);
    }

    assert_eq!(workers.resources(), vec![primary, secondary]);
}

#[test]
fn test_transfer_id_generation() {
    assert_ne!(TransferId::new(), TransferId::new());
}

#[tokio::test]
async fn terminal_transfer_cleanup_keeps_only_live_engine_state() {
    let engine = test_engine().await;
    let live_source = SourceBlocks::External(vec![super::super::source::ExternalBlock::<G2>::new(
        1,
        SequenceHash::new(1, None, 0),
    )]);
    let terminal_source =
        SourceBlocks::External(vec![super::super::source::ExternalBlock::<G2>::new(
            2,
            SequenceHash::new(2, None, 0),
        )]);
    let (_live_id, live_state, _live_handle) = engine.create_transfer(&live_source);
    let (_terminal_id, terminal_state, mut terminal_handle) =
        engine.create_transfer(&terminal_source);

    terminal_state.lock().unwrap().set_complete();
    terminal_handle
        .wait()
        .await
        .expect("terminal transfer publishes its result");

    assert_eq!(
        engine.active_transfer_count(),
        1,
        "terminal state must disappear while live state remains tracked",
    );
    drop(live_state);
    assert_eq!(engine.active_transfer_count(), 0);
}

#[tokio::test]
async fn aborted_precondition_task_terminalizes_and_prunes_registered_transfer() {
    let engine = test_engine().await;
    let event = engine
        .leader
        .messenger()
        .events()
        .new_event()
        .expect("create pending precondition");
    let source = SourceBlocks::External(vec![super::super::source::ExternalBlock::<G2>::new(
        3,
        SequenceHash::new(3, None, 0),
    )]);
    let (transfer_id, state, mut handle) = engine.create_transfer(&source);
    let container = OffloadContainer::new(transfer_id, source, state, Some(event.handle()));
    let input = Arc::new(CancellableQueue::new());
    let output = Arc::new(CancellableQueue::new());
    let awaiter =
        PreconditionAwaiter::new(Arc::clone(&input), output, Arc::clone(&engine.leader), 1);
    let task = tokio::spawn(awaiter.run());

    assert!(input.push(transfer_id, container));
    tokio::time::timeout(std::time::Duration::from_millis(250), async {
        while !input.is_empty_approx() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("precondition task owns the registered transfer");
    task.abort();
    task.await.expect_err("abort the precondition awaiter");

    let result = tokio::time::timeout(std::time::Duration::from_millis(250), handle.wait())
        .await
        .expect("aborted container publishes a terminal result")
        .expect("transfer handle returns a result");
    assert_eq!(result.status, TransferStatus::Failed);
    assert_eq!(engine.active_transfer_count(), 0);
}

#[tokio::test]
async fn builder_rejects_local_and_remote_g4_modes_together() {
    let messenger = create_messenger_tcp().await.expect("create messenger");
    let registry = Arc::new(TestRegistryBuilder::new().build());
    let g2_manager = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(4)
            .block_size(4)
            .registry(registry.as_ref().clone())
            .build(),
    );
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry.as_ref().clone())
            .g2_manager(g2_manager)
            .workers(Vec::new())
            .build()
            .expect("build test leader"),
    );

    let result = OffloadEngine::builder(leader)
        .with_g2_to_g4_pipeline(ObjectPipelineConfig::<G2>::default())
        .with_enable_remote_g4(true)
        .build();
    let error = match result {
        Ok(_) => panic!("builder accepted both G2-to-G4 modes"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("local and remote G2-to-G4 offload modes are mutually exclusive")
    );
}
