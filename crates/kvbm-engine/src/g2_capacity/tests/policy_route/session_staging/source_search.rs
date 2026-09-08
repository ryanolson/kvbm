use super::*;
use crate::G3;
use crate::p2p::g1_source::G1SessionSource;
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
        Self::build(count, transfer, None).await
    }

    async fn with_disk(
        count: usize,
        transfer: Arc<dyn PolicyG1G2TransferExecutor>,
        g3: Arc<BlockManager<G3>>,
    ) -> Result<Self> {
        Self::build(count, transfer, Some(g3)).await
    }

    async fn build(
        count: usize,
        transfer: Arc<dyn PolicyG1G2TransferExecutor>,
        g3: Option<Arc<BlockManager<G3>>>,
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
    // Release the copy before the assertions. A parked gate outlives a
    // failed test and blocks the runtime drop instead of failing it.
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
    // Release the copy before the assertions. A parked gate outlives a
    // failed test and blocks the runtime drop instead of failing it.
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

#[tokio::test]
async fn prefix_stops_at_a_cross_tier_hole_but_scatter_keeps_later_hits() -> Result<()> {
    let mut holder = Holder::new(3).await?;
    let absent = holder.fixture.pins.remove(1);
    absent.set_evict_on_reset(true);
    drop(absent);
    let hashes = &holder.fixture.hashes;
    drop(retained(&holder.fixture.g2, &[hashes[0]])?);
    ensure!(holder.open(SearchMode::Prefix, device_tier()).await? == vec![hashes[0]]);
    ensure!(holder.open(SearchMode::Scatter, device_tier()).await? == vec![hashes[0], hashes[2]]);
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
