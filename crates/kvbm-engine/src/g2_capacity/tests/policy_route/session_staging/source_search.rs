use super::*;
use crate::G3;
use crate::object::ObjectBlockOps;
use crate::p2p::g1_source::G1SessionSource;
use crate::worker::group::ParallelWorkers;
use crate::worker::{
    ConnectRemoteResponse, ImportMetadataResponse, InstanceId, RemoteDescriptor, SerializedLayout,
    SerializedLayoutResponse, TransferCompleteNotification, Worker, WorkerTransfers,
};
use kvbm_protocols::control::modules::transfer::{
    FindMode, OpenTransferSessionRequest, OpenTransferSessionResponse, SearchMode, TierSelection,
};

struct Holder {
    fixture: Fixture,
    leader: Arc<InstanceLeader>,
    source: Arc<G1SessionSource>,
    sessions: Arc<MockSessionFactory>,
}

/// Select the device tier. Every tier beyond G2 is opt-in, so a holder
/// search reaches G1 only for a caller that intends to pull.
fn device_tier() -> TierSelection {
    TierSelection {
        g1: true,
        ..Default::default()
    }
}

impl Holder {
    async fn new(count: usize) -> Result<Self> {
        Self::with_transfer(count, Arc::new(ImmediateTransfer::default())).await
    }

    async fn with_transfer(
        count: usize,
        transfer: Arc<dyn PolicyG1G2TransferExecutor>,
    ) -> Result<Self> {
        Self::build(count, transfer, None, None).await
    }

    async fn with_disk(
        count: usize,
        transfer: Arc<dyn PolicyG1G2TransferExecutor>,
        g3: Arc<BlockManager<G3>>,
    ) -> Result<Self> {
        Self::build(count, transfer, Some(g3), None).await
    }

    /// Like [`Self::with_disk`], with a working `ParallelWorkers` so a
    /// genuine G3 hit stages all the way to a registered G2 block instead of
    /// tripping the leader's missing-worker fail-fast.
    async fn with_disk_and_worker(
        count: usize,
        transfer: Arc<dyn PolicyG1G2TransferExecutor>,
        g3: Arc<BlockManager<G3>>,
        parallel_worker: Arc<dyn ParallelWorkers>,
    ) -> Result<Self> {
        Self::build(count, transfer, Some(g3), Some(parallel_worker)).await
    }

    async fn build(
        count: usize,
        transfer: Arc<dyn PolicyG1G2TransferExecutor>,
        g3: Option<Arc<BlockManager<G3>>>,
        parallel_worker: Option<Arc<dyn ParallelWorkers>>,
    ) -> Result<Self> {
        let fixture = Fixture::new(count, count + 2)?;
        let source = fixture.route(transfer).into_session_source(&fixture.g1)?;
        let mut managers = BlockManagerSet::new();
        managers.insert(RESOURCE, fixture.g2.clone())?;
        let mut builder = InstanceLeader::builder()
            .messenger(create_messenger_tcp().await?)
            .registry(BlockRegistry::new())
            .g2_manager_set(Arc::new(managers), RESOURCE);
        if let Some(g3) = g3 {
            builder = builder.g3_manager(g3);
        }
        if let Some(parallel_worker) = parallel_worker {
            builder = builder.parallel_worker(parallel_worker);
        }
        let leader = Arc::new(builder.build()?);
        let sessions = MockSessionFactory::new();
        ensure!(leader.set_session_factory(sessions.clone()));
        leader.install_g1_sources(std::slice::from_ref(&source))?;
        Ok(Self {
            fixture,
            leader,
            source,
            sessions,
        })
    }

    async fn open(&self, mode: SearchMode, tiers: TierSelection) -> Result<Vec<SequenceHash>> {
        match self.response(mode, tiers).await? {
            OpenTransferSessionResponse::Sync { committed, .. } => Ok(committed),
            OpenTransferSessionResponse::NoBlocksFound => Ok(Vec::new()),
            OpenTransferSessionResponse::Async { .. } => {
                anyhow::bail!("unexpected asynchronous result")
            }
        }
    }

    async fn response(
        &self,
        mode: SearchMode,
        tiers: TierSelection,
    ) -> Result<OpenTransferSessionResponse> {
        Ok(self
            .leader
            .open_transfer_session(OpenTransferSessionRequest {
                sequence_hashes: self.fixture.hashes.clone(),
                resource: Some(RESOURCE),
                find_mode: FindMode::Sync,
                search_mode: mode,
                tiers,
                ..Default::default()
            })
            .await?)
    }

    fn session(&self) -> Result<Arc<MockSession>> {
        self.sessions.last_opened().context("opened holder session")
    }

    /// Wait for the populator to finish every availability batch.
    ///
    /// Publication is tiered, so a wait on the first batch would read a
    /// partial set and race the batches that follow it.
    async fn available(&self) -> Result<Arc<MockSession>> {
        let session = self.session()?;
        wait_for(|| session.finish_availability_called()).await?;
        Ok(session)
    }
}

/// Local-only `ParallelWorkers`. `execute_local_transfer` always reports
/// success, so `stage_g3_to_g2` can register a real G2 block for a genuine
/// G3 hit without a live worker. No search fixture that needs this stub
/// drives a remote or object-store path, so those methods bail or report
/// absence.
#[derive(Default)]
struct StubParallelWorkers;

impl WorkerTransfers for StubParallelWorkers {
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

    fn execute_remote_onboard(
        &self,
        _src: RemoteDescriptor,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("stub: execute_remote_onboard not implemented")
    }

    fn execute_remote_offload(
        &self,
        _src: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst: RemoteDescriptor,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        anyhow::bail!("stub: execute_remote_offload not implemented")
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
        anyhow::bail!("stub: execute_remote_onboard_for_instance not implemented")
    }
}

impl ObjectBlockOps for StubParallelWorkers {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        Box::pin(async move { keys.into_iter().map(|k| (k, None)).collect() })
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

impl ParallelWorkers for StubParallelWorkers {
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

#[tokio::test]
async fn default_tiers_keep_the_holder_g2_only() -> Result<()> {
    let transfer = Arc::new(ImmediateTransfer::default());
    let holder = Holder::with_transfer(3, transfer.clone()).await?;
    ensure!(
        holder
            .fixture
            .g2
            .match_blocks(&holder.fixture.hashes)
            .is_empty()
    );
    ensure!(
        holder
            .open(SearchMode::Prefix, TierSelection::default())
            .await?
            .is_empty(),
        "a caller that does not select G1 must see the G2-only result"
    );
    ensure!(
        transfer.calls() == 0,
        "an unselected tier must not start a device copy"
    );
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == holder.fixture.hashes);
    Ok(())
}

#[tokio::test]
async fn prefix_search_keeps_the_single_lock_g2_walk() -> Result<()> {
    let holder = Holder::new(3).await?;
    let hashes = holder.fixture.hashes.clone();
    drop(retained(&holder.fixture.g2, &[hashes[0]])?);
    let before = holder.fixture.g2.metrics().snapshot();
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == hashes);
    holder.available().await?;
    let after = holder.fixture.g2.metrics().snapshot();
    ensure!(
        after.match_hashes_requested == before.match_hashes_requested + 3,
        "the prefix walk must reach G2 through one match_prefix per run"
    );
    // Staging dedups its two pinned G1 hashes against G2 with one scatter
    // scan. The search itself must add none.
    ensure!(
        after.scan_hashes_requested == before.scan_hashes_requested + 2,
        "a Prefix search must not scatter-scan G2"
    );
    Ok(())
}

#[tokio::test]
async fn prefix_search_pins_only_the_served_prefix() -> Result<()> {
    let mut holder = Holder::new(4).await?;
    let absent = holder.fixture.pins.remove(1);
    absent.set_evict_on_reset(true);
    drop(absent);
    // Leave the blocks past the gap registered but inactive.
    holder.fixture.pins.truncate(1);
    let hashes = holder.fixture.hashes.clone();
    let before = holder.fixture.g1.metrics().snapshot();
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == vec![hashes[0]]);
    let after = holder.fixture.g1.metrics().snapshot();
    ensure!(
        after.scan_hashes_requested == before.scan_hashes_requested,
        "a Prefix search must not scatter-scan G1"
    );
    ensure!(
        after.inactive_pool_size == before.inactive_pool_size,
        "the walk must not promote a block past the prefix gap"
    );
    Ok(())
}

#[tokio::test]
async fn scatter_prefers_a_device_hit_over_a_disk_hit() -> Result<()> {
    let transfer = Arc::new(ImmediateTransfer::default());
    let g3 = Arc::new(
        TestManagerBuilder::<G3>::new()
            .block_count(2)
            .block_size(4)
            .build(),
    );
    let holder = Holder::with_disk(1, transfer.clone(), Arc::clone(&g3)).await?;
    let hashes = holder.fixture.hashes.clone();
    drop(retained(&g3, &hashes)?);

    // Control: the disk tier holds the hash and the scatter walk reaches
    // it. This holder has no parallel worker, so a G3 selection fails the
    // open before any session exists.
    let disk_only = holder
        .response(
            SearchMode::Scatter,
            TierSelection {
                g3: true,
                ..Default::default()
            },
        )
        .await;
    let error = format!(
        "{:#}",
        disk_only
            .err()
            .context("a G3 hit without a parallel worker must fail the open")?
    );
    ensure!(error.contains("g3_requires_parallel_worker"), "{error}");

    let before = g3.metrics().snapshot();
    match holder
        .response(
            SearchMode::Scatter,
            TierSelection {
                g1: true,
                g3: true,
                g4: false,
            },
        )
        .await?
    {
        OpenTransferSessionResponse::Sync {
            committed,
            breakdown,
            ..
        } => {
            ensure!(committed == hashes);
            ensure!(breakdown.device_blocks == 1);
            ensure!(breakdown.disk_blocks == 0);
            ensure!(breakdown.host_blocks == 0);
        }
        _ => anyhow::bail!("a device hit must open a synchronous session"),
    }
    ensure!(
        g3.metrics().snapshot().scan_hashes_requested == before.scan_hashes_requested,
        "G1 supplied every miss, so the disk tier must stay unread"
    );
    let session = holder.available().await?;
    ensure!(session.make_available_calls() == vec![hashes]);
    ensure!(transfer.records()[0].src == LogicalLayoutHandle::G1);
    Ok(())
}

#[tokio::test]
async fn disabling_the_source_mid_copy_does_not_cancel_it() -> Result<()> {
    let transfer = gated_transfer();
    let holder = Holder::with_transfer(1, transfer.clone()).await?;
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == holder.fixture.hashes);
    transfer.started.wait().await;
    holder.source.set_enabled(false);
    transfer.release.wait().await;
    let session = holder.available().await?;
    ensure!(session.make_available_calls() == vec![holder.fixture.hashes.clone()]);
    Ok(())
}

#[tokio::test]
async fn holder_finds_g1_only_blocks_and_stages_temporary_g2() -> Result<()> {
    let transfer = Arc::new(ImmediateTransfer::default());
    let holder = Holder::with_transfer(3, transfer.clone()).await?;
    ensure!(
        holder
            .fixture
            .g2
            .match_blocks(&holder.fixture.hashes)
            .is_empty()
    );
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == holder.fixture.hashes);
    let session = holder.available().await?;
    ensure!(session.make_available_calls() == vec![holder.fixture.hashes.clone()]);
    ensure!(transfer.records()[0].src_blocks.len() == 3);
    ensure!(holder.fixture.g1.match_blocks(&holder.fixture.hashes).len() == 3);
    Ok(())
}

#[tokio::test]
async fn mixed_g1_g2_prefix_keeps_request_order_and_copies_only_misses() -> Result<()> {
    let transfer = Arc::new(ImmediateTransfer::default());
    let holder = Holder::with_transfer(3, transfer.clone()).await?;
    let hashes = &holder.fixture.hashes;
    drop(retained(&holder.fixture.g2, &[hashes[0], hashes[2]])?);
    let hashes = hashes.clone();
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == hashes);
    let session = holder.available().await?;
    // The resident G2 prefix publishes first. The device batch follows it
    // and carries the retained G2 copy of the last hash, which staging
    // reuses instead of copying.
    ensure!(session.make_available_calls() == vec![vec![hashes[0]], vec![hashes[1], hashes[2]]]);
    let records = transfer.records();
    ensure!(records.len() == 1);
    ensure!(records[0].src_blocks == vec![holder.fixture.pins[1].block_id()]);
    Ok(())
}

/// Resident G2 hits reach the puller before the device copy lands.
///
/// One batch for the whole session would charge every hit the latency of
/// the slowest tier the search touched.
#[tokio::test]
async fn resident_g2_hits_publish_before_the_g1_copy_lands() -> Result<()> {
    let transfer = gated_transfer();
    let holder = Holder::with_transfer(2, transfer.clone()).await?;
    let hashes = holder.fixture.hashes.clone();
    let primary = retained(&holder.fixture.g2, &hashes[..1])?;
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == hashes);
    transfer.started.wait().await;
    let published_during_the_copy = holder.session()?.make_available_calls();
    // Release the gate before the assertions so the copy drains and a
    // failed assertion leaves no parked stager behind.
    transfer.release.wait().await;
    let session = holder.available().await?;
    ensure!(
        published_during_the_copy == vec![vec![hashes[0]]],
        "the resident G2 hit must publish while the copy runs"
    );
    ensure!(session.make_available_calls() == vec![vec![hashes[0]], vec![hashes[1]]]);
    drop(primary);
    Ok(())
}

/// The commit stream must close before the G1 copy lands.
///
/// The holder commits the full set up front, so no later commit exists to
/// wait for. A puller attached to `drain_committed` must not pay the device
/// copy latency for a set the holder already knows in full.
#[tokio::test]
async fn commit_stream_closes_before_the_g1_copy_lands() -> Result<()> {
    let transfer = gated_transfer();
    let holder = Holder::with_transfer(2, transfer.clone()).await?;
    let hashes = holder.fixture.hashes.clone();
    let primary = retained(&holder.fixture.g2, &hashes[..1])?;
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == hashes);
    transfer.started.wait().await;
    let closed_during_the_copy = holder.session()?.finish_commits_called();
    // Release the gate before the assertions so the copy drains and a
    // failed assertion leaves no parked stager behind.
    transfer.release.wait().await;
    ensure!(
        closed_during_the_copy,
        "the commit stream must close before the device copy lands"
    );
    holder.available().await?;
    drop(primary);
    Ok(())
}

/// A failed device copy closes the session and names the staging step.
///
/// The batch the holder already published stays published, and the copy
/// releases its G1 pins and its G2 reservation.
#[tokio::test]
async fn g1_staging_failure_closes_the_session_with_the_stage_error() -> Result<()> {
    let transfer = Arc::new(OutcomeTransfer {
        outcome: Mutex::new(Some(TransferDrainOutcome::DrainedWithError(
            anyhow::anyhow!("injected staging failure"),
        ))),
    });
    let holder = Holder::with_transfer(2, transfer).await?;
    let hashes = holder.fixture.hashes.clone();
    let primary = retained(&holder.fixture.g2, &hashes[..1])?;
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == hashes);
    let session = holder.session()?;
    wait_for(|| session.closed_reason().is_some()).await?;
    ensure!(
        session
            .closed_reason()
            .flatten()
            .is_some_and(|reason| reason.contains("stage G1 blocks")),
        "the close reason must name the staging step"
    );
    ensure!(session.make_available_calls() == vec![vec![hashes[0]]]);
    // Four G2 blocks, one held by the retained primary. The drained
    // failure returned the staging destination to the pool.
    ensure!(holder.fixture.g2.available_blocks() == 3);
    ensure!(holder.fixture.g1.match_blocks(&hashes).len() == 2);
    drop(primary);
    Ok(())
}

/// A cross-tier hole must cost one store lock per tier, not one per tier
/// per cursor visit.
///
/// G2 holds only hashes[0] and G1 holds hashes[0] and hashes[2], not
/// hashes[1]. The walk asks G2 once over the full request (a hit run of
/// one, stopping at hashes[1]), then asks G1 at that same cursor. G1 also
/// stops at hashes[1], so the walk must end there without a second round
/// that re-asks either tier at a cursor it has already declared a miss.
#[tokio::test]
async fn prefix_walk_asks_each_tier_once_per_cursor() -> Result<()> {
    let mut holder = Holder::new(3).await?;
    let absent = holder.fixture.pins.remove(1);
    absent.set_evict_on_reset(true);
    drop(absent);
    let hashes = holder.fixture.hashes.clone();
    drop(retained(&holder.fixture.g2, &[hashes[0]])?);
    let g2_before = holder.fixture.g2.metrics().snapshot();
    let g1_before = holder.fixture.g1.metrics().snapshot();
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == vec![hashes[0]]);
    holder.available().await?;
    let g2_after = holder.fixture.g2.metrics().snapshot();
    let g1_after = holder.fixture.g1.metrics().snapshot();
    ensure!(
        g2_after.match_hashes_requested == g2_before.match_hashes_requested + 3,
        "the walk must ask G2 only at a cursor it has not already stopped at"
    );
    ensure!(
        g1_after.match_hashes_requested == g1_before.match_hashes_requested + 2,
        "the walk must ask G1 only at a cursor it has not already stopped at"
    );
    Ok(())
}

/// A G1 run that ends mid-request must not end the walk: G2 can hold the
/// hash right after it, and the walk must resume there and join the two
/// runs into one committed prefix.
///
/// G2 holds hashes[0] and hashes[2], not hashes[1]. G1 holds hashes[1]
/// only. The walk's first G2 call serves hashes[0] and stops at hashes[1];
/// G1 then serves hashes[1] and stops at hashes[2]; the walk must ask G2
/// again at that cursor and pick up hashes[2] instead of stopping where G1
/// did.
#[tokio::test]
async fn prefix_walk_resumes_g2_after_a_g1_run() -> Result<()> {
    let mut holder = Holder::new(3).await?;
    let hashes = holder.fixture.hashes.clone();
    drop(retained(&holder.fixture.g2, &[hashes[0], hashes[2]])?);
    let absent2 = holder.fixture.pins.remove(2);
    absent2.set_evict_on_reset(true);
    drop(absent2);
    let absent0 = holder.fixture.pins.remove(0);
    absent0.set_evict_on_reset(true);
    drop(absent0);
    let registry = holder.fixture.g2.block_registry();
    let counts_before: Vec<u32> = hashes.iter().map(|hash| registry.count(*hash)).collect();
    match holder.response(SearchMode::Prefix, device_tier()).await? {
        OpenTransferSessionResponse::Sync {
            committed,
            breakdown,
            ..
        } => {
            ensure!(committed == hashes);
            ensure!(breakdown.host_blocks == 2);
            ensure!(breakdown.device_blocks == 1);
        }
        _ => anyhow::bail!("a mixed G2/G1 prefix must open a synchronous session"),
    }
    let counts_after: Vec<u32> = hashes.iter().map(|hash| registry.count(*hash)).collect();
    ensure!(
        counts_after == counts_before,
        "a remote prefix walk must not touch the G2 frequency sketch on the resumed \
         run after a G1 hit, the same as the run before it"
    );
    let session = holder.available().await?;
    ensure!(
        session.make_available_calls() == vec![vec![hashes[0], hashes[2]], vec![hashes[1]]],
        "stage_phase must publish one batch per tier, not one per prefix run"
    );
    Ok(())
}

#[tokio::test]
async fn prefix_stops_at_a_cross_tier_hole_but_scatter_keeps_later_hits() -> Result<()> {
    let mut holder = Holder::new(3).await?;
    let absent = holder.fixture.pins.remove(1);
    absent.set_evict_on_reset(true);
    drop(absent);
    let hashes = &holder.fixture.hashes;
    drop(retained(&holder.fixture.g2, &[hashes[0]])?);
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == vec![hashes[0]]);
    match holder.response(SearchMode::Scatter, device_tier()).await? {
        OpenTransferSessionResponse::Sync {
            committed,
            breakdown,
            ..
        } => {
            ensure!(committed == vec![hashes[0], hashes[2]]);
            ensure!(breakdown.host_blocks == 1);
            ensure!(breakdown.device_blocks == 1);
        }
        _ => anyhow::bail!("a mixed G2/G1 scatter must open a synchronous session"),
    }
    Ok(())
}

/// G3 is a bottom tier: it must see only the hashes G1 could not also
/// supply, not the whole G2-miss set. Handing it the whole miss set would
/// waste a disk lookup on a hash the device tier already serves.
///
/// G2 holds hashes[0], G1 holds hashes[1] only, G3 holds hashes[2].
#[tokio::test]
async fn scatter_hands_g3_only_the_hashes_g1_lacks() -> Result<()> {
    let transfer = Arc::new(ImmediateTransfer::default());
    let g3 = Arc::new(
        TestManagerBuilder::<G3>::new()
            .block_count(2)
            .block_size(4)
            .build(),
    );
    let mut holder = Holder::with_disk_and_worker(
        3,
        transfer.clone(),
        Arc::clone(&g3),
        Arc::new(StubParallelWorkers),
    )
    .await?;
    let hashes = holder.fixture.hashes.clone();
    drop(retained(&holder.fixture.g2, &[hashes[0]])?);
    let absent2 = holder.fixture.pins.remove(2);
    absent2.set_evict_on_reset(true);
    drop(absent2);
    let absent0 = holder.fixture.pins.remove(0);
    absent0.set_evict_on_reset(true);
    drop(absent0);
    drop(retained(&g3, &[hashes[2]])?);

    let before = g3.metrics().snapshot();
    match holder
        .response(
            SearchMode::Scatter,
            TierSelection {
                g1: true,
                g3: true,
                g4: false,
            },
        )
        .await?
    {
        OpenTransferSessionResponse::Sync {
            committed,
            breakdown,
            ..
        } => {
            ensure!(committed == hashes);
            ensure!(breakdown.host_blocks == 1);
            ensure!(breakdown.device_blocks == 1);
            ensure!(breakdown.disk_blocks == 1);
        }
        _ => anyhow::bail!("a G1+G3 scatter must open a synchronous session"),
    }
    ensure!(
        g3.metrics().snapshot().scan_hashes_requested == before.scan_hashes_requested + 1,
        "G3 must scan only the hash G1 could not also supply"
    );
    holder.available().await?;
    Ok(())
}

#[tokio::test]
async fn dropping_source_authority_removes_it_from_holder_search() -> Result<()> {
    let holder = Holder::new(1).await?;
    let Holder {
        source,
        fixture,
        leader,
        ..
    } = holder;
    drop(source);
    let response = leader
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: fixture.hashes.clone(),
            resource: Some(RESOURCE),
            find_mode: FindMode::Sync,
            tiers: device_tier(),
            ..Default::default()
        })
        .await?;
    ensure!(matches!(
        response,
        OpenTransferSessionResponse::NoBlocksFound
    ));
    Ok(())
}

#[tokio::test]
async fn source_installation_rejects_another_destination_manager() -> Result<()> {
    let holder = Holder::new(1).await?;
    let foreign = Fixture::new(1, 1)?;
    let source = foreign
        .route(Arc::new(ImmediateTransfer::default()))
        .into_session_source(&foreign.g1)?;
    ensure!(holder.leader.install_g1_sources(&[source]).is_err());
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == holder.fixture.hashes);
    Ok(())
}

/// Two sources for the same resource inside one `install_g1_sources` call
/// must be rejected before either reaches the registry, even though each
/// one alone would bind cleanly. The registry's per-call `HashSet` only
/// catches this when both land in the same slice: two separate calls with
/// the same `Arc` instead hit the ptr-equality re-install path, which is
/// covered elsewhere.
#[tokio::test]
async fn installing_one_resource_twice_in_one_call_is_rejected() -> Result<()> {
    let holder = Holder::new(1).await?;
    let error = match holder
        .leader
        .install_g1_sources(&[holder.source.clone(), holder.source.clone()])
    {
        Ok(()) => anyhow::bail!("a repeated resource in one call must be rejected"),
        Err(error) => error,
    };
    ensure!(
        format!("{error:#}").contains("duplicate G1 source resource"),
        "{error:#}"
    );
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == holder.fixture.hashes);
    Ok(())
}

#[tokio::test]
async fn retired_source_binding_cannot_reassign_the_physical_pool() -> Result<()> {
    let holder = Holder::new(1).await?;
    let foreign = source(1)?;
    let replacement = bound_route(
        holder.fixture.capacity.clone(),
        RESOURCE,
        Arc::new(ImmediateTransfer::default()),
        &foreign.manager,
    )
    .into_session_source(&foreign.manager)?;
    drop(holder.source);
    ensure!(
        holder.leader.install_g1_sources(&[replacement]).is_err(),
        "a physical resource must keep its original logical binding"
    );
    Ok(())
}

#[tokio::test]
async fn source_binding_rejects_another_g1_manager() -> Result<()> {
    let fixture = Fixture::new(1, 1)?;
    let foreign = source(1)?;
    ensure!(
        fixture
            .route(Arc::new(ImmediateTransfer::default()))
            .into_session_source(&foreign.manager)
            .is_err()
    );
    Ok(())
}
