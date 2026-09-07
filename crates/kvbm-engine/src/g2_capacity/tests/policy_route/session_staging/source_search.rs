use super::*;
use crate::p2p::g1_source::G1SessionSource;
use kvbm_protocols::control::modules::transfer::{
    FindMode, OpenTransferSessionRequest, OpenTransferSessionResponse, SearchMode,
};

struct Holder {
    fixture: Fixture,
    leader: Arc<InstanceLeader>,
    source: Arc<G1SessionSource>,
    transfer: Arc<ImmediateTransfer>,
    sessions: Arc<MockSessionFactory>,
}

impl Holder {
    async fn new(count: usize) -> Result<Self> {
        let fixture = Fixture::new(count, count + 2)?;
        let transfer = Arc::new(ImmediateTransfer::default());
        let source = fixture
            .route(transfer.clone())
            .into_session_source(&fixture.g1)?;
        let mut managers = BlockManagerSet::new();
        managers.insert(RESOURCE, fixture.g2.clone())?;
        let leader = Arc::new(
            InstanceLeader::builder()
                .messenger(create_messenger_tcp().await?)
                .registry(BlockRegistry::new())
                .g2_manager_set(Arc::new(managers), RESOURCE)
                .build()?,
        );
        let sessions = MockSessionFactory::new();
        ensure!(leader.set_session_factory(sessions.clone()));
        leader.install_g1_sources(std::slice::from_ref(&source))?;
        Ok(Self {
            fixture,
            leader,
            source,
            transfer,
            sessions,
        })
    }

    async fn open(&self, mode: SearchMode) -> Result<Vec<SequenceHash>> {
        match self
            .leader
            .open_transfer_session(OpenTransferSessionRequest {
                sequence_hashes: self.fixture.hashes.clone(),
                resource: Some(RESOURCE),
                find_mode: FindMode::Sync,
                search_mode: mode,
                ..Default::default()
            })
            .await?
        {
            OpenTransferSessionResponse::Sync { committed, .. } => Ok(committed),
            OpenTransferSessionResponse::NoBlocksFound => Ok(Vec::new()),
            OpenTransferSessionResponse::Async { .. } => {
                anyhow::bail!("unexpected asynchronous result")
            }
        }
    }

    async fn available(&self) -> Result<Arc<MockSession>> {
        let session = self
            .sessions
            .last_opened()
            .context("opened holder session")?;
        wait_for(|| !session.make_available_calls().is_empty()).await?;
        Ok(session)
    }
}

#[tokio::test]
async fn holder_finds_g1_only_blocks_and_stages_temporary_g2() -> Result<()> {
    let holder = Holder::new(3).await?;
    ensure!(
        holder
            .fixture
            .g2
            .match_blocks(&holder.fixture.hashes)
            .is_empty()
    );
    ensure!(holder.open(SearchMode::Prefix).await? == holder.fixture.hashes);
    let session = holder.available().await?;
    ensure!(session.make_available_calls() == vec![holder.fixture.hashes.clone()]);
    ensure!(holder.transfer.records()[0].src_blocks.len() == 3);
    ensure!(holder.fixture.g1.match_blocks(&holder.fixture.hashes).len() == 3);
    Ok(())
}

#[tokio::test]
async fn mixed_g1_g2_prefix_keeps_request_order_and_copies_only_misses() -> Result<()> {
    let holder = Holder::new(3).await?;
    let hashes = &holder.fixture.hashes;
    drop(retained(&holder.fixture.g2, &[hashes[0], hashes[2]])?);
    ensure!(holder.open(SearchMode::Prefix).await? == *hashes);
    let session = holder.available().await?;
    ensure!(session.make_available_calls() == vec![hashes.clone()]);
    let records = holder.transfer.records();
    ensure!(records.len() == 1);
    ensure!(records[0].src_blocks == vec![holder.fixture.pins[1].block_id()]);
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
    ensure!(holder.open(SearchMode::Prefix).await? == vec![hashes[0]]);
    ensure!(holder.open(SearchMode::Scatter).await? == vec![hashes[0], hashes[2]]);
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
    ensure!(holder.open(SearchMode::Prefix).await? == holder.fixture.hashes);
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
