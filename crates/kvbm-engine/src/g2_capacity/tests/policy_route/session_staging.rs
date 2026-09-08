use std::time::Duration;

use anyhow::{Context, ensure};
use futures::future::BoxFuture;
use kvbm_logical::ImmutableBlock;

use super::*;
use crate::g2_capacity::test_support::RecordingG2Capacity;
use crate::p2p::session::{MockSession, MockSessionFactory, Session, SessionFactory};

#[cfg(feature = "testing-nixl")]
mod device;
mod failures;
mod source_search;

const RESOURCE: LogicalResourceId = LogicalResourceId(7);

struct Fixture {
    g1: Arc<BlockManager<G1>>,
    g2: Arc<BlockManager<G2>>,
    pins: Vec<ImmutableBlock<G1>>,
    hashes: Vec<SequenceHash>,
    capacity: Arc<ExactRegistrationCapacity>,
    holder: Arc<MockSession>,
}

impl Fixture {
    fn new(source_count: usize, destination_count: usize) -> Result<Self> {
        let Source { manager: g1, pins } = source(source_count)?;
        let hashes = pins
            .iter()
            .map(ImmutableBlock::sequence_hash)
            .collect::<Vec<_>>();
        let (capacity, g2) = capacity(destination_count);
        let factory = MockSessionFactory::new();
        factory.open(uuid::Uuid::new_v4())?;
        let holder = factory.last_opened().context("holder session")?;
        holder.commit(hashes.clone())?;
        Ok(Self {
            g1,
            g2,
            pins,
            hashes,
            capacity,
            holder,
        })
    }

    fn route(&self, transfer: Arc<dyn PolicyG1G2TransferExecutor>) -> PolicyG1G2BoundRoute<G1> {
        bound_route(self.capacity.clone(), RESOURCE, transfer, &self.g1)
    }

    fn stage(
        &mut self,
        transfer: Arc<dyn PolicyG1G2TransferExecutor>,
    ) -> BoxFuture<'static, Result<()>> {
        self.submit(&self.route(transfer))
    }

    fn submit(&mut self, route: &PolicyG1G2BoundRoute<G1>) -> BoxFuture<'static, Result<()>> {
        let staged = route.stage_to_g2(std::mem::take(&mut self.pins));
        let holder = self.holder.clone();
        Box::pin(async move { holder.make_available(staged.await?) })
    }

    fn check_ownership(
        &self,
        source_free: usize,
        destination_free: usize,
        permit_drops: usize,
    ) -> Result<()> {
        ensure!(
            self.g1.available_blocks() == source_free,
            "G1 ownership differs"
        );
        ensure!(
            self.g2.available_blocks() == destination_free,
            "G2 ownership differs"
        );
        ensure!(
            self.capacity.exact_permit_drop_count() == permit_drops,
            "exact permit ownership differs"
        );
        Ok(())
    }

    fn release(self) -> Arc<BlockManager<G2>> {
        self.holder.close(None);
        self.g2
    }
}

struct Source {
    manager: Arc<BlockManager<G1>>,
    pins: Vec<ImmutableBlock<G1>>,
}

fn source(count: usize) -> Result<Source> {
    let manager = Arc::new(
        TestManagerBuilder::<G1>::new()
            .block_count(count)
            .block_size(4)
            .build(),
    );
    let pins = retained(&manager, &chain(count, 80))?;
    Ok(Source { manager, pins })
}

/// Build one root-to-leaf lineage.
///
/// A production source always holds a chained prefix. Unrelated roots let a
/// fixture pass a destination rule that only a real lineage can reach.
fn chain(len: usize, seed: u64) -> Vec<SequenceHash> {
    let mut hashes = Vec::with_capacity(len);
    if len == 0 {
        return hashes;
    }
    let mut hash = SequenceHash::root(seed);
    hashes.push(hash);
    for offset in 1..len {
        hash = hash.extend(seed + offset as u64);
        hashes.push(hash);
    }
    hashes
}

fn retained<T: kvbm_logical::blocks::BlockMetadata + Sync>(
    manager: &BlockManager<T>,
    hashes: &[SequenceHash],
) -> Result<Vec<ImmutableBlock<T>>> {
    let blocks = manager
        .allocate_blocks(hashes.len())
        .context("allocate retained blocks")?
        .into_iter()
        .zip(hashes)
        .map(|(block, hash)| {
            block
                .stage(*hash, manager.block_size())
                .map_err(|error| anyhow::anyhow!("{error}"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(manager.register_blocks(blocks))
}

fn gated_transfer() -> Arc<GatedTransfer> {
    Arc::new(GatedTransfer {
        started: Arc::new(Barrier::new(2)),
        release: Arc::new(Barrier::new(2)),
    })
}

async fn wait_for(mut condition: impl FnMut() -> bool) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(2), async {
        while !condition() {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    Ok(())
}

#[tokio::test]
async fn sealed_commits_accept_staged_availability() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    f.holder.finish_commits()?;
    f.stage(Arc::new(ImmediateTransfer::default())).await?;
    ensure!(f.holder.make_available_calls() == vec![f.hashes.clone()]);
    Ok(())
}

#[tokio::test]
async fn exact_staging_publishes_temporary_blocks_without_evicting_g1() -> Result<()> {
    let mut f = Fixture::new(2, 2)?;
    let source_ids = f
        .pins
        .iter()
        .map(ImmutableBlock::block_id)
        .collect::<Vec<_>>();
    let transfer = Arc::new(ImmediateTransfer::default());
    f.stage(transfer.clone()).await?;

    ensure!(f.holder.make_available_calls() == vec![f.hashes.clone()]);
    f.check_ownership(2, 0, 1)?;
    ensure!(
        f.g1.match_blocks(&f.hashes).len() == 2,
        "transport must not evict G1"
    );
    ensure!(
        f.capacity.requests()
            == vec![G2CapacityRequest::exact_reclaim(
                G2AllocationKind::RequiredStaging,
                2
            )]
    );
    ensure!(f.capacity.exact_registration_count() == 1);
    let records = transfer.records();
    ensure!(records.len() == 1);
    ensure!(records[0].resource == RESOURCE);
    ensure!(records[0].src == LogicalLayoutHandle::G1);
    ensure!(records[0].dst == LogicalLayoutHandle::G2);
    ensure!(records[0].src_blocks == source_ids);
    let hashes = f.hashes.clone();
    let g2 = f.release();
    ensure!(g2.available_blocks() == 2);
    ensure!(
        g2.match_blocks(&hashes).is_empty(),
        "temporary primaries must reset"
    );
    Ok(())
}

#[tokio::test]
async fn delayed_copy_keeps_sources_destinations_and_exact_owner_until_drain() -> Result<()> {
    let mut f = Fixture::new(2, 2)?;
    let transfer = gated_transfer();
    let completion = f.stage(transfer.clone());
    transfer.started.wait().await;
    f.check_ownership(0, 0, 0)?;
    ensure!(f.holder.make_available_calls().is_empty());
    transfer.release.wait().await;
    completion.await?;
    f.check_ownership(2, 0, 1)?;
    ensure!(f.holder.make_available_calls().len() == 1);
    Ok(())
}

/// A parked copy must leave the blocking pool free.
///
/// The pool is the only place `spawn_blocking` can run. A stager that
/// parks there starves every offload supervisor and every other blocking
/// task the runtime owns for the whole length of one PCIe copy.
#[test]
fn staging_does_not_occupy_the_blocking_pool() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()?;
    let handle = runtime.handle().clone();
    runtime.block_on(async move {
        let mut f = Fixture::new(1, 1)?;
        let transfer = gated_transfer();
        let route = bound_route_with_runtime(
            f.capacity.clone(),
            RESOURCE,
            transfer.clone(),
            &f.g1,
            handle,
        );
        let completion = f.submit(&route);
        transfer.started.wait().await;
        let free = tokio::time::timeout(
            Duration::from_millis(500),
            tokio::task::spawn_blocking(|| ()),
        )
        .await
        .is_ok();
        // Release the gate before the assertions so the copy drains and a
        // failed assertion leaves no parked stager behind.
        transfer.release.wait().await;
        completion.await?;
        ensure!(free, "the parked stager occupied the only blocking thread");
        Ok(())
    })
}

async fn interrupted_copy(timeout: bool) -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    let transfer = gated_transfer();
    let completion = f.stage(transfer.clone());
    transfer.started.wait().await;
    if timeout {
        ensure!(
            tokio::time::timeout(Duration::from_millis(10), completion)
                .await
                .is_err()
        );
    } else {
        drop(completion);
    }
    f.check_ownership(0, 0, 0)?;
    transfer.release.wait().await;
    wait_for(|| f.capacity.exact_permit_drop_count() == 1 && f.g2.available_blocks() == 1).await?;
    f.check_ownership(1, 1, 1)?;
    // A drained copy returns its source pins to the inactive pool. The Drop
    // leak path would forget them and leave no G1 match.
    ensure!(f.g1.match_blocks(&f.hashes).len() == 1);
    ensure!(f.g2.match_blocks(&f.hashes).is_empty());
    ensure!(f.holder.make_available_calls().is_empty());
    Ok(())
}

#[tokio::test]
async fn dropped_future_does_not_release_live_copy_or_publish_later() -> Result<()> {
    interrupted_copy(false).await
}

#[tokio::test]
async fn timeout_does_not_release_live_copy_or_publish_later() -> Result<()> {
    interrupted_copy(true).await
}

#[tokio::test]
async fn retained_g2_hits_need_no_allocation_or_copy() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    let hash = f.hashes[0];
    drop(retained(&f.g2, &[hash])?);
    let transfer = Arc::new(ImmediateTransfer::default());
    f.stage(transfer.clone()).await?;
    ensure!(f.capacity.requests().is_empty());
    ensure!(transfer.calls() == 0);
    ensure!(f.release().match_blocks(&[hash]).len() == 1);
    Ok(())
}

#[tokio::test]
async fn registration_collision_preserves_the_retained_primary() -> Result<()> {
    let mut f = Fixture::new(1, 2)?;
    let hash = f.hashes[0];
    let transfer = gated_transfer();
    let completion = f.stage(transfer.clone());
    transfer.started.wait().await;
    let primary = retained(&f.g2, &[hash])?;
    let primary_id = primary[0].block_id();
    transfer.release.wait().await;
    completion.await?;
    drop(primary);
    let matched = f.release().match_blocks(&[hash]);
    ensure!(matched.len() == 1);
    ensure!(matched[0].block_id() == primary_id);
    Ok(())
}

#[tokio::test]
async fn explicit_adoption_preserves_a_temporary_primary() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    let hash = f.hashes[0];
    f.stage(Arc::new(ImmediateTransfer::default())).await?;
    for block in f.g2.match_blocks(&[hash]) {
        block.set_evict_on_reset(false);
    }
    ensure!(f.release().match_blocks(&[hash]).len() == 1);
    Ok(())
}

#[tokio::test]
async fn capacity_exhaustion_publishes_no_partial_batch() -> Result<()> {
    let mut f = Fixture::new(2, 1)?;
    let hash = f.hashes[0];
    let retained = retained(&f.g2, &[hash])?;
    let transfer = Arc::new(ImmediateTransfer::default());
    ensure!(f.stage(transfer.clone()).await.is_err());
    ensure!(transfer.calls() == 0);
    ensure!(f.holder.make_available_calls().is_empty());
    drop(retained);
    ensure!(f.g2.match_blocks(&[hash]).len() == 1);
    Ok(())
}

#[tokio::test]
async fn foreign_source_manager_fails_before_allocation() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    f.pins = source(1)?.pins;
    let transfer = Arc::new(ImmediateTransfer::default());
    ensure!(f.stage(transfer.clone()).await.is_err());
    ensure!(f.capacity.requests().is_empty());
    ensure!(transfer.calls() == 0);
    ensure!(f.holder.make_available_calls().is_empty());
    Ok(())
}

#[tokio::test]
async fn drained_copy_failure_releases_the_whole_batch_without_publication() -> Result<()> {
    let mut f = Fixture::new(2, 2)?;
    let transfer = Arc::new(OutcomeTransfer {
        outcome: Mutex::new(Some(TransferDrainOutcome::DrainedWithError(
            anyhow::anyhow!("injected drained failure"),
        ))),
    });
    ensure!(f.stage(transfer).await.is_err());
    ensure!(f.holder.make_available_calls().is_empty());
    f.check_ownership(2, 2, 1)?;
    ensure!(f.capacity.exact_registration_count() == 0);
    Ok(())
}

#[tokio::test]
async fn unproven_copy_failure_quarantines_pins_and_exact_capacity() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    let transfer = Arc::new(OutcomeTransfer {
        outcome: Mutex::new(Some(TransferDrainOutcome::Unproven(anyhow::anyhow!(
            "injected uncertain drain"
        )))),
    });
    ensure!(f.stage(transfer).await.is_err());
    ensure!(f.holder.make_available_calls().is_empty());
    f.check_ownership(0, 0, 0)
}

#[tokio::test]
async fn compatibility_capacity_keeps_its_lease_and_resets_temporary_blocks() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    let hash = f.hashes[0];
    let capacity = Arc::new(RecordingG2Capacity::new(f.g2.clone()));
    let route = bound_route(
        capacity.clone(),
        RESOURCE,
        Arc::new(ImmediateTransfer::default()),
        &f.g1,
    );
    f.submit(&route).await?;
    ensure!(capacity.allocation_kinds() == vec![G2AllocationKind::RequiredStaging]);
    ensure!(capacity.registrations_with_live_lease() == 1);
    ensure!(capacity.lease_drop_count() == 1);
    ensure!(f.release().match_blocks(&[hash]).is_empty());
    Ok(())
}
