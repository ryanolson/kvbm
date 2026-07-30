// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replicated data worker for MLA (Multi-head Latent Attention) scenarios.
//!
//! In MLA architectures, G1 KV blocks are replicated across all workers rather
//! than sharded. Lower tiers are striped across the worker group so each logical
//! block has exactly one lower-tier owner.
//!
//! # Architecture
//!
//! ```text
//! Global G2 block N ──→ owner = N % world_size, local = N / world_size
//! Owner G2          ──→ owner G1 ───broadcast(root=owner)──→ every G1 replica
//! ```
//!
//! # Transfer Semantics
//!
//! | Operation | Behavior |
//! |-----------|----------|
//! | G2 → G1 (onboard) | Each G2 owner transfers its batch, then broadcasts from that owner |
//! | G1 → G2 (offload) | Each rank writes only the global G2 blocks it owns |
//! | G2 ↔ G3 | Not yet supported by this worker |
//! | G1 → G1 (local) | All ranks execute (data is replicated) |

mod planner;

use planner::ReplicatedTransferPlanner;

use super::*;

use crate::KvbmRuntime;
use crate::collectives::CollectiveOps;
use anyhow::{Context, Result, bail, ensure};

use std::sync::Arc;

type CollectiveJob = Box<dyn FnOnce() + Send + 'static>;

/// Rank-local executor for synchronous collective dispatch.
///
/// Each co-located rank owns a dedicated OS thread, so collective entry never
/// depends on Tokio's async-worker or blocking-pool limits. The channel also
/// preserves dispatch order for this rank.
#[derive(Clone)]
struct CollectiveExecutor {
    jobs: std::sync::mpsc::Sender<CollectiveJob>,
}

impl CollectiveExecutor {
    fn new(rank: usize) -> Result<Self> {
        let (jobs, receiver) = std::sync::mpsc::channel::<CollectiveJob>();
        std::thread::Builder::new()
            .name(format!("kvbm-collective-rank-{rank}"))
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_err() {
                        tracing::error!(rank, "KVBM collective dispatch job panicked");
                    }
                }
            })
            .context("failed to spawn rank-local collective executor")?;
        Ok(Self { jobs })
    }

    async fn dispatch<Dispatch>(&self, dispatch: Dispatch) -> Result<TransferCompleteNotification>
    where
        Dispatch: FnOnce() -> Result<TransferCompleteNotification> + Send + 'static,
    {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        self.jobs
            .send(Box::new(move || {
                let _ = result_tx.send(dispatch());
            }))
            .map_err(|_| anyhow::anyhow!("rank-local collective executor stopped"))?;
        result_rx
            .await
            .context("rank-local collective dispatch job terminated")?
    }
}

/// Replicated data worker for MLA scenarios.
///
/// G1 is replicated on every rank while G2 is striped across ranks. When loading
/// data to G1, each G2 owner transfers its local batch and broadcasts that batch
/// to the same G1 block IDs on every other rank.
///
/// # Requirements
///
/// - Every worker must have equal-sized local G2 storage
/// - Replicated G1 allocators must assign the same destination IDs on every rank
/// - A [`CollectiveOps`] implementation must be provided for broadcasting
///
/// # Trait Implementations
///
/// - [`WorkerTransfers`]: Specialized routing based on source/destination tiers
pub struct ReplicatedDataWorker {
    inner: Arc<PhysicalWorker>,
    runtime: Arc<KvbmRuntime>,
    collective: Arc<dyn CollectiveOps>,
    collective_executor: CollectiveExecutor,
    planner: ReplicatedTransferPlanner,
    rank: usize,
}

impl ReplicatedDataWorker {
    /// Create a new ReplicatedDataWorker.
    ///
    /// # Arguments
    /// * `worker` - The rank-local physical worker with G1 and G2 layouts
    /// * `runtime` - Runtime used to sequence owner copies and collectives
    /// * `collective` - The collective ops implementation for broadcasting
    pub fn new(
        worker: Arc<PhysicalWorker>,
        runtime: Arc<KvbmRuntime>,
        collective: Arc<dyn CollectiveOps>,
    ) -> Result<Self> {
        let rank = worker
            .rank()
            .context("replicated data worker requires a physical worker rank")?;
        ensure!(
            rank == collective.rank(),
            "physical worker rank {rank} does not match collective rank {}",
            collective.rank()
        );
        let planner = ReplicatedTransferPlanner::new(collective.world_size())?;
        let collective_executor = CollectiveExecutor::new(rank)?;

        Ok(Self {
            inner: worker,
            runtime,
            collective,
            collective_executor,
            planner,
            rank,
        })
    }

    /// Get access to the underlying SpmdWorker.
    pub fn inner(&self) -> &PhysicalWorker {
        &self.inner
    }

    /// Get the rank of the underlying worker.
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl WorkerTransfers for ReplicatedDataWorker {
    fn local_onboard_requires_serialization(&self, _resource: Option<LogicalResourceId>) -> bool {
        true
    }

    fn abort_local_collectives(&self, reason: String) -> Result<TransferCompleteNotification> {
        self.collective.abort(&reason)?;
        Ok(TransferCompleteNotification::completed())
    }

    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let resource = self
            .inner
            .resource_handles()
            .map(ResourceLayoutHandles::primary)
            .unwrap_or_default();
        self.execute_local_transfer_for_resource(
            resource,
            src,
            dst,
            src_block_ids,
            dst_block_ids,
            options,
        )
    }

    fn execute_local_transfer_for_resource(
        &self,
        resource: LogicalResourceId,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        match (src, dst) {
            (LogicalLayoutHandle::G1, LogicalLayoutHandle::G1) => {
                self.inner.execute_local_transfer_for_resource(
                    resource,
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    options,
                )
            }
            (LogicalLayoutHandle::G1, LogicalLayoutHandle::G2) => {
                let plan = self.planner.plan_offload(
                    self.rank(),
                    src_block_ids.as_ref(),
                    dst_block_ids.as_ref(),
                )?;
                if plan.g1_block_ids().is_empty() {
                    return Ok(TransferCompleteNotification::completed());
                }

                self.inner.execute_local_transfer_for_resource(
                    resource,
                    src,
                    dst,
                    Arc::from(plan.g1_block_ids()),
                    Arc::from(plan.local_g2_block_ids()),
                    options,
                )
            }
            (LogicalLayoutHandle::G2, LogicalLayoutHandle::G1) => {
                let plans = self
                    .planner
                    .plan_onboard(src_block_ids.as_ref(), dst_block_ids.as_ref())?;
                if plans.is_empty() {
                    return Ok(TransferCompleteNotification::completed());
                }

                let event_system = self.runtime.event_system();
                let event = event_system.new_event()?;
                let awaiter = event_system.awaiter(event.handle())?;
                let inner = Arc::clone(&self.inner);
                let collective = Arc::clone(&self.collective);
                let collective_executor = self.collective_executor.clone();
                let rank = self.rank();
                let layer_range = options.layer_range.clone();

                self.runtime.tokio().spawn(async move {
                    let result = execute_onboard_plans(
                        inner,
                        collective,
                        collective_executor,
                        rank,
                        resource,
                        plans,
                        options,
                        layer_range,
                    )
                    .await;
                    match result {
                        Ok(()) => {
                            let _ = event.trigger();
                        }
                        Err(error) => {
                            let _ = event.poison(error.to_string());
                        }
                    }
                });

                Ok(TransferCompleteNotification::from_awaiter(awaiter))
            }
            _ => bail!(
                "replicated data worker does not yet support local transfer {src:?} -> {dst:?}"
            ),
        }
    }

    #[expect(unused_variables)]
    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!("replicated data worker remote onboard is not yet implemented")
    }

    #[expect(unused_variables)]
    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst: RemoteDescriptor,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!("replicated data worker remote offload is not yet implemented")
    }

    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        // Use the shared implementation
        self.inner.connect_remote(instance_id, metadata)
    }

    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool {
        self.inner.has_remote_metadata(instance_id)
    }

    #[expect(unused_variables)]
    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        bail!("replicated data worker instance remote onboard is not yet implemented")
    }

    fn execute_remote_pull_plan(
        &self,
        plan: crate::leader::dispatch::WorkerPullPlan,
    ) -> Result<TransferCompleteNotification> {
        ensure!(
            plan.source_layout == LogicalLayoutHandle::G2
                && plan.dst_layout == LogicalLayoutHandle::G2,
            "replicated remote pull must land once in striped G2; got {:?} -> {:?}",
            plan.source_layout,
            plan.dst_layout
        );
        self.inner.execute_remote_pull_plan(plan)
    }
}

impl Worker for ReplicatedDataWorker {
    fn compute_host_payload_digests(
        &self,
        resource: LogicalResourceId,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Result<Vec<kvbm_physical::transfer::PayloadDigest>>> {
        Worker::compute_host_payload_digests(self.inner.as_ref(), resource, block_ids)
    }

    fn g1_handle(&self) -> Option<LayoutHandle> {
        self.inner.g1_handle()
    }

    fn g2_handle(&self) -> Option<LayoutHandle> {
        self.inner.g2_handle()
    }

    fn g3_handle(&self) -> Option<LayoutHandle> {
        self.inner.g3_handle()
    }

    fn export_metadata(&self) -> Result<SerializedLayoutResponse> {
        Worker::export_metadata(self.inner.as_ref())
    }

    fn import_metadata(&self, metadata: SerializedLayout) -> Result<ImportMetadataResponse> {
        Worker::import_metadata(self.inner.as_ref(), metadata)
    }
}

impl ObjectBlockOps for ReplicatedDataWorker {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        ObjectBlockOps::has_blocks(self.inner.as_ref(), keys)
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        src_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        ObjectBlockOps::put_blocks(self.inner.as_ref(), keys, src_layout, block_ids)
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        dst_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        ObjectBlockOps::get_blocks(self.inner.as_ref(), keys, dst_layout, block_ids)
    }
}

async fn execute_onboard_plans(
    inner: Arc<PhysicalWorker>,
    collective: Arc<dyn CollectiveOps>,
    collective_executor: CollectiveExecutor,
    rank: usize,
    resource: LogicalResourceId,
    plans: Vec<planner::ReplicaOnboardPlan>,
    options: kvbm_physical::transfer::TransferOptions,
    layer_range: Option<std::ops::Range<usize>>,
) -> Result<()> {
    drain_all_replica_steps(plans, move |plan| {
        let inner = Arc::clone(&inner);
        let collective = Arc::clone(&collective);
        let collective_executor = collective_executor.clone();
        let options = options.clone();
        let layer_range = layer_range.clone();
        async move {
            let g1_block_ids: Arc<[BlockId]> = Arc::from(plan.g1_block_ids());
            let broadcast_block_ids = Arc::clone(&g1_block_ids);
            let root_rank = plan.root_rank();
            execute_owner_copy_then_broadcast(
                collective_executor,
                rank == root_rank,
                move || {
                    inner.execute_local_transfer_for_resource(
                        resource,
                        LogicalLayoutHandle::G2,
                        LogicalLayoutHandle::G1,
                        Arc::from(plan.local_g2_block_ids()),
                        Arc::clone(&g1_block_ids),
                        options,
                    )
                },
                move || {
                    collective.broadcast_for_resource(
                        resource,
                        root_rank,
                        LogicalLayoutHandle::G1,
                        LogicalLayoutHandle::G1,
                        broadcast_block_ids.as_ref(),
                        broadcast_block_ids.as_ref(),
                        layer_range,
                    )
                },
            )
            .await
            .with_context(|| format!("replicated onboard step from rank {root_rank} failed"))
        }
    })
    .await
}

/// Drain every replica step before returning any earlier failure so all ranks
/// preserve the same collective sequence across a multi-plan onboard.
async fn drain_all_replica_steps<Steps, Step, StepFuture>(
    steps: Steps,
    mut execute: Step,
) -> Result<()>
where
    Steps: IntoIterator,
    Step: FnMut(Steps::Item) -> StepFuture,
    StepFuture: std::future::Future<Output = Result<()>>,
{
    let mut errors = Vec::new();
    for step in steps {
        if let Err(error) = execute(step).await {
            errors.push(format!("{error:#}"));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        bail!(errors.join("; "))
    }
}

/// Preserve the collective sequence even when the root's local copy fails.
///
/// Every rank must enter the broadcast in the same order. The root therefore
/// retains its copy failure, drains the broadcast alongside its peers, and only
/// then reports the combined terminal error.
async fn execute_owner_copy_then_broadcast<OwnerCopy, Broadcast>(
    collective_executor: CollectiveExecutor,
    is_root: bool,
    owner_copy: OwnerCopy,
    broadcast: Broadcast,
) -> Result<()>
where
    OwnerCopy: FnOnce() -> Result<TransferCompleteNotification>,
    Broadcast: FnOnce() -> Result<TransferCompleteNotification> + Send + 'static,
{
    let mut errors = Vec::new();
    if is_root {
        drain_replica_phase("owner G2 to G1 copy", owner_copy(), &mut errors).await;
    }
    let broadcast = collective_executor.dispatch(broadcast).await;
    drain_replica_phase("replicated G1 broadcast", broadcast, &mut errors).await;

    if errors.is_empty() {
        Ok(())
    } else {
        bail!(errors.join("; "))
    }
}

async fn drain_replica_phase(
    phase: &str,
    notification: Result<TransferCompleteNotification>,
    errors: &mut Vec<String>,
) {
    match notification {
        Ok(notification) => {
            if let Err(error) = notification.await {
                errors.push(format!("{phase} completion failed: {error}"));
            }
        }
        Err(error) => errors.push(format!("{phase} dispatch failed: {error}")),
    }
}

#[cfg(test)]
mod trait_tests {
    use kvbm_config::KvbmConfig;
    use kvbm_memory::StorageKind;
    use kvbm_physical::testing::{create_fc_layout, create_test_agent, create_transfer_manager};
    use kvbm_physical::transfer::{FillPattern, fill_blocks};

    use super::*;
    use crate::collectives::StubCollectiveOps;
    use crate::testing::create_messenger_tcp;

    fn assert_worker<T: Worker>() {}

    #[test]
    fn replicated_data_policy_is_a_complete_worker() {
        assert_worker::<ReplicatedDataWorker>();
    }

    #[tokio::test]
    async fn replicated_worker_forwards_host_payload_digests_to_physical_worker() {
        let agent = create_test_agent(&format!("replicated-digest-{}", uuid::Uuid::new_v4()));
        let layout = create_fc_layout(agent.clone(), StorageKind::System, 2);
        fill_blocks(&layout, &[0], FillPattern::Constant(53)).unwrap();
        let manager = create_transfer_manager(agent, None).unwrap();
        let g2 = manager.register_layout(layout).unwrap();
        let inner = Arc::new(
            PhysicalWorker::builder()
                .manager(manager)
                .g2_handle(g2)
                .rank(0)
                .build()
                .unwrap(),
        );

        let messenger = create_messenger_tcp().await.unwrap();
        let runtime = Arc::new(
            KvbmRuntime::builder(KvbmConfig::default())
                .with_runtime_handle(tokio::runtime::Handle::current())
                .with_messenger(messenger)
                .build_leader()
                .await
                .unwrap(),
        );
        let collective = Arc::new(StubCollectiveOps::single_worker(
            runtime.event_system().as_ref().clone(),
        ));
        let worker = ReplicatedDataWorker::new(Arc::clone(&inner), runtime, collective).unwrap();

        let expected = inner
            .compute_host_payload_digests(LogicalResourceId::default(), vec![0])
            .await
            .unwrap();
        let actual = worker
            .compute_host_payload_digests(LogicalResourceId::default(), vec![0])
            .await
            .unwrap();

        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod drain_tests {
    use std::sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    use std::time::Duration;

    use anyhow::{Result, anyhow};
    use velo::EventManager;

    use super::{
        CollectiveExecutor, TransferCompleteNotification, drain_all_replica_steps,
        execute_owner_copy_then_broadcast,
    };

    #[tokio::test]
    async fn root_copy_dispatch_error_still_enters_and_drains_broadcast() -> Result<()> {
        let events = Arc::new(EventManager::local());
        let broadcast_event = events.new_event()?;
        let broadcast_notification =
            TransferCompleteNotification::from_awaiter(events.awaiter(broadcast_event.handle())?);
        let broadcast_entered = Arc::new(AtomicBool::new(false));
        let collective_executor = CollectiveExecutor::new(0)?;

        let mut completion = Box::pin(execute_owner_copy_then_broadcast(
            collective_executor,
            true,
            || Err(anyhow!("injected owner copy dispatch failure")),
            {
                let broadcast_entered = Arc::clone(&broadcast_entered);
                move || {
                    broadcast_entered.store(true, Ordering::SeqCst);
                    Ok(broadcast_notification)
                }
            },
        ));

        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                tokio::select! {
                    result = &mut completion => {
                        panic!("root completed before its collective drained: {result:?}");
                    }
                    () = tokio::task::yield_now() => {
                        if broadcast_entered.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                }
            }
        })
        .await
        .expect("rank-local executor did not enter the broadcast");
        assert!(
            broadcast_entered.load(Ordering::SeqCst),
            "a root copy dispatch failure must not skip the collective"
        );

        broadcast_event.trigger()?;
        let failure = tokio::time::timeout(Duration::from_secs(1), completion)
            .await?
            .expect_err("the retained root copy failure must fail the replica step");
        assert!(
            failure
                .to_string()
                .contains("injected owner copy dispatch failure")
        );
        Ok(())
    }

    #[tokio::test]
    async fn first_plan_failure_still_enters_and_drains_every_later_broadcast() -> Result<()> {
        let events = Arc::new(EventManager::local());
        let second_event = events.new_event()?;
        let second_notification =
            TransferCompleteNotification::from_awaiter(events.awaiter(second_event.handle())?);
        let delayed = Arc::new(Mutex::new(Some(second_notification)));
        let broadcasts = Arc::new(AtomicUsize::new(0));
        let collective_executor = CollectiveExecutor::new(0)?;

        let completion = tokio::spawn(drain_all_replica_steps(0..2, {
            let delayed = Arc::clone(&delayed);
            let broadcasts = Arc::clone(&broadcasts);
            let collective_executor = collective_executor.clone();
            move |step| {
                let delayed = Arc::clone(&delayed);
                let broadcasts = Arc::clone(&broadcasts);
                let collective_executor = collective_executor.clone();
                async move {
                    execute_owner_copy_then_broadcast(
                        collective_executor,
                        true,
                        move || {
                            if step == 0 {
                                Err(anyhow!("first-plan owner copy failed"))
                            } else {
                                Ok(TransferCompleteNotification::completed())
                            }
                        },
                        move || {
                            broadcasts.fetch_add(1, Ordering::SeqCst);
                            if step == 1 {
                                Ok(delayed.lock().unwrap().take().unwrap())
                            } else {
                                Ok(TransferCompleteNotification::completed())
                            }
                        },
                    )
                    .await
                }
            }
        }));

        tokio::time::timeout(Duration::from_secs(1), async {
            while broadcasts.load(Ordering::SeqCst) != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await?;
        assert!(
            !completion.is_finished(),
            "second broadcast must still drain"
        );
        second_event.trigger()?;
        let failure = tokio::time::timeout(Duration::from_secs(1), completion)
            .await??
            .expect_err("first plan failure must surface after every plan drains");
        assert!(failure.to_string().contains("first-plan owner copy failed"));
        Ok(())
    }

    #[test]
    fn one_async_and_one_blocking_runtime_thread_allows_every_rank_to_enter_collective()
    -> Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(1)
            .enable_all()
            .build()?;
        runtime.block_on(async {
            const RANKS: usize = 4;
            let rendezvous = Arc::new((Mutex::new(0usize), Condvar::new()));
            let mut ranks = Vec::with_capacity(RANKS);

            for rank in 0..RANKS {
                let rendezvous = Arc::clone(&rendezvous);
                let collective_executor = CollectiveExecutor::new(rank)?;
                ranks.push(tokio::spawn(execute_owner_copy_then_broadcast(
                    collective_executor,
                    false,
                    || Ok(TransferCompleteNotification::completed()),
                    move || {
                        let (arrivals, ready) = rendezvous.as_ref();
                        let mut arrivals = arrivals.lock().unwrap();
                        *arrivals += 1;
                        if *arrivals == RANKS {
                            ready.notify_all();
                        } else {
                            let (observed, timeout) = ready
                                .wait_timeout_while(arrivals, Duration::from_secs(5), |count| {
                                    *count != RANKS
                                })
                                .unwrap();
                            arrivals = observed;
                            if timeout.timed_out() && *arrivals != RANKS {
                                return Err(anyhow!(
                                    "peer ranks were starved before collective entry"
                                ));
                            }
                        }
                        Ok(TransferCompleteNotification::completed())
                    },
                )));
            }

            for rank in ranks {
                rank.await??;
            }
            Result::<()>::Ok(())
        })
    }
}
