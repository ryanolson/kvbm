use super::*;
use crate::g2_capacity::{
    DirectG2Capacity, G2CapacityDecision, G2CapacityError, G2StagedAllocation,
};

struct RejectRegistration(DirectG2Capacity);

impl G2Capacity for RejectRegistration {
    fn reserve(&self, request: G2CapacityRequest) -> Result<G2CapacityDecision, G2CapacityError> {
        self.0.reserve(request)
    }

    fn block_size(&self) -> usize {
        self.0.block_size()
    }

    fn manager_id(&self) -> kvbm_logical::ManagerId {
        self.0.manager_id()
    }

    fn register_compatibility(
        &self,
        _allocation: G2StagedAllocation,
    ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
        Err(G2CapacityError::Rejected(
            "injected registration failure".to_owned(),
        ))
    }

    fn match_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
        self.0.match_blocks(hashes)
    }

    fn scan_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> std::collections::HashMap<SequenceHash, ImmutableBlock<G2>> {
        self.0.scan_matches(hashes, touch)
    }
}

#[tokio::test]
async fn failed_registration_preserves_hits_and_publishes_no_batch() -> Result<()> {
    let mut f = Fixture::new(2, 2)?;
    drop(retained(&f.g2, &f.hashes[..1])?);
    let transfer = Arc::new(ImmediateTransfer::default());
    let route = bound_route(
        Arc::new(RejectRegistration(DirectG2Capacity::new(f.g2.clone()))),
        RESOURCE,
        transfer.clone(),
        &f.g1,
    );
    ensure!(
        f.submit(&route, TransferCompleteNotification::completed())
            .await
            .is_err()
    );
    ensure!(transfer.records()[0].src_blocks.len() == 1);
    ensure!(f.holder.make_available_calls().is_empty());
    ensure!(f.g2.available_blocks() == 2);
    ensure!(f.g2.match_blocks(&f.hashes).len() == 1);
    ensure!(f.g2.match_blocks(&f.hashes[1..]).is_empty());
    Ok(())
}

#[tokio::test]
async fn copy_dispatch_failure_releases_capacity_without_publication() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    ensure!(f.stage(Arc::new(DispatchErrorTransfer)).await.is_err());
    ensure!(f.holder.make_available_calls().is_empty());
    f.check_ownership(1, 1, 1)
}

#[tokio::test]
async fn dispatch_panic_retains_source_and_destination_without_publication() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    ensure!(f.stage(Arc::new(PanickingTransfer)).await.is_err());
    ensure!(f.holder.make_available_calls().is_empty());
    f.check_ownership(0, 0, 0)
}

#[tokio::test]
async fn canceled_source_fence_retains_g1_until_writes_finish() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    let transfer = Arc::new(ImmediateTransfer::default());
    let events = velo::EventManager::local();
    let written = events.new_event()?;
    let ready = TransferCompleteNotification::from_awaiter(events.awaiter(written.handle())?);
    drop(f.submit(&f.route(transfer.clone()), ready));
    f.check_ownership(0, 1, 0)?;
    ensure!(f.capacity.requests().is_empty());
    written.trigger()?;
    wait_for(|| f.g1.available_blocks() == 1).await?;
    ensure!(transfer.calls() == 0);
    ensure!(f.holder.make_available_calls().is_empty());
    Ok(())
}

#[tokio::test]
async fn unproven_source_fence_retains_g1_without_copy_or_allocation() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    let transfer = Arc::new(ImmediateTransfer::default());
    let events = velo::EventManager::local();
    let written = events.new_event()?;
    let ready = TransferCompleteNotification::from_awaiter(events.awaiter(written.handle())?);
    let completion = f.submit(&f.route(transfer.clone()), ready);
    written.poison("source writes did not prove completion")?;
    ensure!(completion.await.is_err());
    f.check_ownership(0, 1, 0)?;
    ensure!(f.capacity.requests().is_empty());
    ensure!(transfer.calls() == 0);
    ensure!(f.holder.make_available_calls().is_empty());
    Ok(())
}

#[tokio::test]
async fn closed_session_rejects_publication_and_releases_temporary_blocks() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    f.holder.finish_availability()?;
    ensure!(
        f.stage(Arc::new(ImmediateTransfer::default()))
            .await
            .is_err()
    );
    ensure!(f.holder.make_available_calls().is_empty());
    ensure!(f.g2.available_blocks() == 1);
    ensure!(f.g2.match_blocks(&f.hashes).is_empty());
    Ok(())
}

#[tokio::test]
async fn unequal_block_sizes_fail_before_copy_or_allocation() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    let g1 = source_lineage(1);
    f.pins = g1.manager.match_blocks(&g1.hashes());
    f.g1 = g1.manager;
    let transfer = Arc::new(ImmediateTransfer::default());
    ensure!(f.stage(transfer.clone()).await.is_err());
    ensure!(f.capacity.requests().is_empty());
    ensure!(transfer.calls() == 0);
    Ok(())
}

#[tokio::test]
async fn repeated_source_hashes_fail_before_copy_or_allocation() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    f.pins.push(f.pins[0].clone());
    let transfer = Arc::new(ImmediateTransfer::default());
    ensure!(f.stage(transfer.clone()).await.is_err());
    ensure!(f.capacity.requests().is_empty());
    ensure!(transfer.calls() == 0);
    Ok(())
}

#[tokio::test]
async fn reallocated_temporary_slot_restores_manager_retention() -> Result<()> {
    let mut f = Fixture::new(1, 1)?;
    let hash = f.hashes[0];
    f.stage(Arc::new(ImmediateTransfer::default())).await?;
    let g2 = f.release();
    ensure!(g2.match_blocks(&[hash]).is_empty());
    let next_hash = SequenceHash::new(1000, None, 0);
    drop(retained(&g2, &[next_hash])?);
    ensure!(g2.match_blocks(&[next_hash]).len() == 1);
    Ok(())
}
