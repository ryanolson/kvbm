// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime construction and shared ingress for offload pipelines.

use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use parking_lot::Mutex as ParkingMutex;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::blocks::{BlockMetadata, ImmutableBlock};

use crate::leader::InstanceLeader;
use crate::object::ObjectBlockOps;

use super::super::batch::{BatchCollector, BatchOutputRx};
use super::super::cancel::CancellationUnit;
use super::super::container::OffloadContainer;
use super::super::destination::PipelineDestination;
use super::super::handle::{TransferId, TransferState};
use super::super::pending::PendingTracker;
use super::super::queue::CancellableQueue;
use super::super::source::SourceBlocks;
use super::block_executor::BlockTransferExecutor;
use super::config::{ObjectPipelineConfig, PipelineBaseConfig, PipelineConfig};
use super::ingress::PipelineIngress;
use super::object_executor::ObjectTransferExecutor;
use super::shutdown::{ExecutorShutdown, PRECOMMIT_SHUTDOWN_ERROR, PipelineCommitmentGate};
use super::{PolicyEvaluator, PreconditionAwaiter, cancel_sweeper};

const EXECUTOR_INPUT_CAPACITY: usize = 8;

/// One registered batch for a downstream pipeline.
pub(crate) struct ChainOutput<T: BlockMetadata> {
    pub transfer_id: TransferId,
    pub blocks: Vec<ImmutableBlock<T>>,
    #[allow(dead_code)]
    pub(crate) state: Arc<std::sync::Mutex<TransferState>>,
    pub(crate) cancellation: CancellationUnit,
}

/// Receiver for registered downstream blocks.
pub(crate) type ChainOutputRx<T> = mpsc::Receiver<ChainOutput<T>>;

/// Callback for one registered destination batch.
pub type RegisterObserver<Dst> = Arc<dyn Fn(&[ImmutableBlock<Dst>]) + Send + Sync + 'static>;

pub(super) type RegisterObservers<Dst> = ParkingMutex<Vec<RegisterObserver<Dst>>>;

/// Common stage owner for block and object pipelines.
struct PipelineRuntime<Src: BlockMetadata> {
    eval_queue: Arc<CancellableQueue<OffloadContainer<Src>>>,
    cancel_tx: watch::Sender<HashSet<TransferId>>,
    cancellation_queues: Vec<Arc<CancellableQueue<OffloadContainer<Src>>>>,
    transfers: Arc<DashMap<TransferId, Arc<std::sync::Mutex<TransferState>>>>,
    registration_gate: Arc<ParkingMutex<()>>,
    runtime: tokio::runtime::Handle,
    shutdown_tx: watch::Sender<bool>,
    commitment_gate: Arc<PipelineCommitmentGate>,
    task_handles: Vec<JoinHandle<()>>,
}

impl<Src: BlockMetadata> PipelineRuntime<Src> {
    fn start(
        config: &PipelineBaseConfig<Src>,
        leader: Arc<InstanceLeader>,
        runtime: tokio::runtime::Handle,
    ) -> (Self, BatchOutputRx<Src>) {
        let eval_queue = Arc::new(CancellableQueue::new());
        let precondition_queue = Arc::new(CancellableQueue::new());
        let batch_queue = Arc::new(CancellableQueue::new());
        let cancellation_queues = vec![
            Arc::clone(&eval_queue),
            Arc::clone(&precondition_queue),
            Arc::clone(&batch_queue),
        ];
        let (cancel_tx, cancel_rx) = watch::channel(HashSet::new());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let commitment_gate = Arc::new(PipelineCommitmentGate::new());
        let (batch_tx, batch_rx) = mpsc::channel(EXECUTOR_INPUT_CAPACITY);
        let pending_tracker = config
            .pending_tracker
            .clone()
            .unwrap_or_else(|| Arc::new(PendingTracker::new()));

        let evaluator = PolicyEvaluator {
            policies: config.policies.clone(),
            timeout: config.policy_timeout,
            input_queue: Arc::clone(&eval_queue),
            output_queue: Arc::clone(&precondition_queue),
            cancel_rx: cancel_rx.clone(),
            pending_tracker: Arc::clone(&pending_tracker),
        };
        let evaluator_task = runtime.spawn(evaluator.run());

        let awaiter = PreconditionAwaiter::new(
            Arc::clone(&precondition_queue),
            Arc::clone(&batch_queue),
            leader,
            config.max_concurrent_precondition_awaits.max(1),
        );
        let precondition_task = runtime.spawn(awaiter.run());

        let collector = BatchCollector::new(
            config.batch_config.clone(),
            Arc::clone(&batch_queue),
            batch_tx,
            cancel_rx.clone(),
            shutdown_rx.clone(),
        );
        let collector_task = runtime.spawn(collector.run());

        let sweeper_queues = cancellation_queues.clone();
        let sweep_interval = config.sweep_interval;
        let sweeper_task = runtime.spawn(async move {
            cancel_sweeper(sweeper_queues, cancel_rx, shutdown_rx, sweep_interval).await;
        });

        (
            Self {
                eval_queue,
                cancel_tx,
                cancellation_queues,
                transfers: Arc::new(DashMap::new()),
                registration_gate: Arc::new(ParkingMutex::new(())),
                runtime,
                shutdown_tx,
                commitment_gate,
                task_handles: vec![
                    evaluator_task,
                    precondition_task,
                    collector_task,
                    sweeper_task,
                ],
            },
            batch_rx,
        )
    }

    fn attach(&mut self, task: JoinHandle<()>) {
        self.task_handles.push(task);
    }

    fn enqueue(
        &self,
        transfer_id: TransferId,
        source: SourceBlocks<Src>,
        state: Arc<std::sync::Mutex<TransferState>>,
    ) -> bool {
        self.ingress().enqueue(transfer_id, source, state)
    }

    fn ingress(&self) -> PipelineIngress<Src> {
        PipelineIngress {
            eval_queue: Arc::clone(&self.eval_queue),
            cancellation_queues: self.cancellation_queues.clone(),
            transfers: Arc::clone(&self.transfers),
            registration_gate: Arc::clone(&self.registration_gate),
            cancel_tx: self.cancel_tx.clone(),
            runtime: self.runtime.clone(),
        }
    }

    fn shutdown_receiver(&self) -> watch::Receiver<bool> {
        self.shutdown_tx.subscribe()
    }

    fn commitment_gate(&self) -> Arc<PipelineCommitmentGate> {
        Arc::clone(&self.commitment_gate)
    }
}

impl<Src: BlockMetadata> Drop for PipelineRuntime<Src> {
    fn drop(&mut self) {
        let queued_containers = self
            .commitment_gate
            .publish_shutdown(&self.shutdown_tx, || {
                let mut queued_containers = Vec::new();
                for queue in &self.cancellation_queues {
                    queued_containers.extend(queue.close_and_drain());
                }
                queued_containers
            });
        for container in queued_containers {
            container.fail(PRECOMMIT_SHUTDOWN_ERROR.to_string());
        }
    }
}

/// A running block destination pipeline.
pub(crate) struct Pipeline<Src: BlockMetadata, Dst: BlockMetadata> {
    runtime: PipelineRuntime<Src>,
    chain_rx: Option<ChainOutputRx<Dst>>,
    register_observers: Arc<RegisterObservers<Dst>>,
    auto_chain: bool,
    destination: PhantomData<Dst>,
}

impl<Src: BlockMetadata, Dst: BlockMetadata> Pipeline<Src, Dst> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        config: PipelineConfig<Src, Dst>,
        destination: impl Into<PipelineDestination<Dst>>,
        leader: Arc<InstanceLeader>,
        src_layout: LogicalLayoutHandle,
        dst_layout: LogicalLayoutHandle,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self> {
        config.validate()?;
        let (base, options) = config.into_parts();
        let (chain_tx, chain_rx) = if options.auto_chain {
            let (sender, receiver) = mpsc::channel(64);
            (Some(sender), Some(receiver))
        } else {
            (None, None)
        };
        let register_observers = Arc::new(ParkingMutex::new(Vec::new()));
        let (mut pipeline_runtime, batch_rx) =
            PipelineRuntime::start(&base, Arc::clone(&leader), runtime.clone());
        let executor = BlockTransferExecutor {
            input_rx: batch_rx,
            leader,
            destination: destination.into().into_inner(),
            resource: options.resource,
            src_layout,
            dst_layout,
            skip_transfers: base.skip_transfers,
            max_concurrent_transfers: base.max_concurrent_transfers,
            chain_tx,
            register_observers: Arc::clone(&register_observers),
            shutdown: ExecutorShutdown::new(pipeline_runtime.shutdown_receiver()),
            commitment_gate: pipeline_runtime.commitment_gate(),
            _src_marker: PhantomData::<Src>,
        };
        pipeline_runtime.attach(runtime.spawn(executor.run()));

        Ok(Self {
            runtime: pipeline_runtime,
            chain_rx,
            register_observers,
            auto_chain: options.auto_chain,
            destination: PhantomData,
        })
    }

    pub(crate) fn enqueue(
        &self,
        transfer_id: TransferId,
        source: SourceBlocks<Src>,
        state: Arc<std::sync::Mutex<TransferState>>,
    ) -> bool {
        self.runtime.enqueue(transfer_id, source, state)
    }

    pub(crate) fn ingress(&self) -> PipelineIngress<Src> {
        self.runtime.ingress()
    }

    /// Return true if this pipeline sends registered blocks downstream.
    pub(crate) fn auto_chain(&self) -> bool {
        self.auto_chain
    }

    /// Take the downstream receiver.
    pub(crate) fn take_chain_rx(&mut self) -> Option<ChainOutputRx<Dst>> {
        self.chain_rx.take()
    }

    /// Add a callback for destination registration.
    pub(crate) fn add_register_observer(&self, observer: RegisterObserver<Dst>) {
        self.register_observers.lock().push(observer);
    }
}

/// A running object destination pipeline.
pub(crate) struct ObjectPipeline<Src: BlockMetadata> {
    runtime: PipelineRuntime<Src>,
}

impl<Src: BlockMetadata> ObjectPipeline<Src> {
    pub(crate) fn new(
        config: ObjectPipelineConfig<Src>,
        object_ops: Arc<dyn ObjectBlockOps>,
        src_layout: LogicalLayoutHandle,
        leader: Arc<InstanceLeader>,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self> {
        config.validate()?;
        let (base, options) = config.into_parts();
        let (mut pipeline_runtime, batch_rx) =
            PipelineRuntime::start(&base, leader, runtime.clone());
        let executor = ObjectTransferExecutor::new_with_commitment_gate(
            batch_rx,
            object_ops,
            src_layout,
            base.skip_transfers,
            base.max_concurrent_transfers,
            options.lock_manager,
            pipeline_runtime.shutdown_receiver(),
            pipeline_runtime.commitment_gate(),
        );
        pipeline_runtime.attach(runtime.spawn(executor.run()));
        Ok(Self {
            runtime: pipeline_runtime,
        })
    }

    pub(crate) fn enqueue(
        &self,
        transfer_id: TransferId,
        source: SourceBlocks<Src>,
        state: Arc<std::sync::Mutex<TransferState>>,
    ) -> bool {
        self.runtime.enqueue(transfer_id, source, state)
    }

    pub(crate) fn ingress(&self) -> PipelineIngress<Src> {
        self.runtime.ingress()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use super::super::super::source::{ExternalBlock, SourceBlocks};
    use super::super::shutdown::PRECOMMIT_SHUTDOWN_ERROR;
    use super::*;
    use crate::SequenceHash;
    use crate::offload::handle::TransferStatus;

    #[tokio::test]
    async fn drop_publishes_shutdown_before_a_blocked_downstream_queue_close() {
        let eval_queue = Arc::new(CancellableQueue::new());
        let downstream_queue = Arc::new(CancellableQueue::new());
        let cancellation_queues = vec![Arc::clone(&eval_queue), Arc::clone(&downstream_queue)];
        let (cancel_tx, _cancel_rx) = watch::channel(HashSet::new());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let runtime = PipelineRuntime::<()> {
            eval_queue: Arc::clone(&eval_queue),
            cancel_tx,
            cancellation_queues,
            transfers: Arc::new(DashMap::new()),
            registration_gate: Arc::new(ParkingMutex::new(())),
            runtime: tokio::runtime::Handle::current(),
            shutdown_tx,
            commitment_gate: Arc::new(PipelineCommitmentGate::new()),
            task_handles: Vec::new(),
        };
        let admission_guard = downstream_queue.lock_admission_for_test();
        let (drop_started_tx, drop_started_rx) = mpsc::channel();
        let (drop_finished_tx, drop_finished_rx) = mpsc::channel();

        let dropper = std::thread::spawn(move || {
            drop_started_tx.send(()).expect("test starts runtime drop");
            drop(runtime);
            drop_finished_tx
                .send(())
                .expect("test observes runtime drop");
        });
        drop_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("runtime drop starts");

        let deadline = Instant::now() + Duration::from_secs(1);
        while !eval_queue.is_closed() && Instant::now() < deadline {
            std::thread::yield_now();
        }
        let earlier_queue_closed = eval_queue.is_closed();
        let shutdown_published = *shutdown_rx.borrow();
        let drop_finished_early = drop_finished_rx
            .recv_timeout(Duration::from_millis(25))
            .is_ok();

        drop(admission_guard);
        if !drop_finished_early {
            drop_finished_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("runtime drop finishes after admission releases");
        }
        dropper.join().expect("runtime drop thread exits");
        assert!(
            earlier_queue_closed,
            "runtime drop closes the earlier queue before it blocks downstream"
        );
        assert!(
            shutdown_published,
            "shutdown must publish before a later queue close can block"
        );
        assert!(
            !drop_finished_early,
            "the downstream admission guard blocks its close"
        );
    }

    #[tokio::test]
    async fn drop_drains_later_queue_after_an_earlier_close_blocks() {
        let eval_queue = Arc::new(CancellableQueue::new());
        let later_queue = Arc::new(CancellableQueue::new());
        let cancellation_queues = vec![Arc::clone(&eval_queue), Arc::clone(&later_queue)];
        let (cancel_tx, _cancel_rx) = watch::channel(HashSet::new());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let runtime = PipelineRuntime::<()> {
            eval_queue: Arc::clone(&eval_queue),
            cancel_tx,
            cancellation_queues,
            transfers: Arc::new(DashMap::new()),
            registration_gate: Arc::new(ParkingMutex::new(())),
            runtime: tokio::runtime::Handle::current(),
            shutdown_tx,
            commitment_gate: Arc::new(PipelineCommitmentGate::new()),
            task_handles: Vec::new(),
        };
        let earlier_admission_guard = eval_queue.lock_admission_for_test();
        let (drop_started_tx, drop_started_rx) = mpsc::channel();
        let (drop_finished_tx, drop_finished_rx) = mpsc::channel();

        let dropper = std::thread::spawn(move || {
            drop_started_tx.send(()).expect("test starts runtime drop");
            drop(runtime);
            drop_finished_tx
                .send(())
                .expect("test observes runtime drop");
        });
        drop_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("runtime drop starts");

        let deadline = Instant::now() + Duration::from_secs(1);
        while !*shutdown_rx.borrow() && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(
            *shutdown_rx.borrow(),
            "shutdown publishes before the earlier queue admission lock releases"
        );

        let transfer_id = TransferId::new();
        let (state, mut handle) = TransferState::new(transfer_id, vec![77]);
        let container = OffloadContainer::new(
            transfer_id,
            SourceBlocks::External(vec![ExternalBlock::<()>::new(
                77,
                SequenceHash::new(77, None, 0),
            )]),
            Arc::new(std::sync::Mutex::new(state)),
            None,
        );
        assert!(
            later_queue.push_or_return(transfer_id, container).is_ok(),
            "the later queue remains open while the earlier close blocks"
        );
        assert_eq!(later_queue.len_approx(), 1);

        drop(earlier_admission_guard);
        drop_finished_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("runtime drop finishes after the earlier close releases");
        dropper.join().expect("runtime drop thread exits");

        let result = tokio::time::timeout(Duration::from_millis(250), handle.wait())
            .await
            .expect("runtime drop settles late queued work")
            .expect("runtime drop publishes a transfer result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(result.error.as_deref(), Some(PRECOMMIT_SHUTDOWN_ERROR));
        assert!(later_queue.is_empty_approx());
        assert!(later_queue.pop().is_none());
    }

    #[tokio::test]
    async fn drop_drains_a_committed_chained_container_from_the_later_queue() {
        let eval_queue = Arc::new(CancellableQueue::new());
        let later_queue = Arc::new(CancellableQueue::new());
        let (cancel_tx, _cancel_rx) = watch::channel(HashSet::new());
        let (shutdown_tx, _shutdown_rx) = watch::channel(false);
        let runtime = PipelineRuntime::<()> {
            eval_queue: Arc::clone(&eval_queue),
            cancel_tx,
            cancellation_queues: vec![eval_queue, Arc::clone(&later_queue)],
            transfers: Arc::new(DashMap::new()),
            registration_gate: Arc::new(ParkingMutex::new(())),
            runtime: tokio::runtime::Handle::current(),
            shutdown_tx,
            commitment_gate: Arc::new(PipelineCommitmentGate::new()),
            task_handles: Vec::new(),
        };
        let transfer_id = TransferId::new();
        let (mut state, mut handle) = TransferState::new(transfer_id, vec![78]);
        let cancellation_token = state.cancellation_token();
        let upstream = cancellation_token
            .root_unit()
            .expect("the upstream route owns its root unit");
        assert!(upstream.claim_commitment());
        state.mark_committed();
        let state = Arc::new(std::sync::Mutex::new(state));
        let mut children = upstream.fan_out(1);
        let container = OffloadContainer::with_cancellation(
            transfer_id,
            SourceBlocks::External(vec![ExternalBlock::<()>::new(
                78,
                SequenceHash::new(78, None, 0),
            )]),
            state,
            None,
            children
                .pop()
                .expect("the downstream route owns one child unit"),
        );
        assert!(later_queue.push_or_return(transfer_id, container).is_ok());

        drop(runtime);

        tokio::time::timeout(Duration::from_millis(250), handle.cancel().wait())
            .await
            .expect("runtime drain settles the chained child unit");
        let result = handle
            .wait()
            .await
            .expect("runtime drain publishes a transfer result");
        assert_eq!(result.status, TransferStatus::Failed);
        assert_eq!(result.error.as_deref(), Some(PRECOMMIT_SHUTDOWN_ERROR));
        assert!(later_queue.is_empty_approx());
    }
}
