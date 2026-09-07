use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use futures::future::BoxFuture;
use kvbm_common::{LogicalLayoutHandle, SequenceHash};
use kvbm_logical::{ImmutableBlock, LifecyclePinRef};
use kvbm_physical::transfer::{
    TransferCompleteNotification, TransferDrainOutcome, TransferOptions,
};
use tokio::sync::oneshot;

use super::{PolicyG1G2BoundRoute, PolicyG1G2RouteCore, PolicyG1SourceMetadata};
use crate::G2;
use crate::g2_capacity::{RequiredStagingStagedAllocation, reserve_required_staging};
use crate::p2p::session::Session;

type StagedBlocks = Vec<ImmutableBlock<G2>>;

struct SessionStagingCopy {
    route: Arc<PolicyG1G2RouteCore>,
    source: Vec<LifecyclePinRef>,
    destination: Option<RequiredStagingStagedAllocation>,
    undrained: bool,
}

impl<T: PolicyG1SourceMetadata> PolicyG1G2BoundRoute<T> {
    pub fn stage_for_session(
        &self,
        source: Vec<ImmutableBlock<T>>,
        writes_complete: TransferCompleteNotification,
        session: Arc<dyn Session>,
    ) -> BoxFuture<'static, Result<()>> {
        let pins = source.iter().map(ImmutableBlock::pin).collect::<Vec<_>>();
        let validation = (|| {
            ensure!(
                pins.iter()
                    .all(|pin| pin.manager_id() == self.source_manager_id),
                "session staging source belongs to another G1 manager"
            );
            ensure!(
                self.source_block_size == self.core.capacity.block_size(),
                "session staging requires equal G1 and G2 block sizes"
            );
            ensure!(
                pins.iter()
                    .map(LifecyclePinRef::sequence_hash)
                    .collect::<HashSet<_>>()
                    .len()
                    == pins.len(),
                "session staging source contains duplicate hashes"
            );
            self.core
                .runtime
                .clone()
                .context("session staging requires a runtime")
        })();
        let runtime = match validation {
            Ok(runtime) => runtime,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let (sender, receiver) = oneshot::channel();
        let mut copy = SessionStagingCopy {
            route: Arc::clone(&self.core),
            source: pins,
            destination: None,
            undrained: true,
        };
        let spawned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.spawn_blocking(move || {
                let result = copy.execute(writes_complete, &sender);
                drop(copy);
                let _ = sender.send(result);
            })
        }));
        if spawned.is_err() {
            return Box::pin(async { Err(anyhow!("runtime rejected session staging task")) });
        }
        Box::pin(async move {
            let blocks = receiver
                .await
                .context("session staging task ended without a result")??;
            if !blocks.is_empty() {
                session.make_available(blocks)?;
            }
            Ok(())
        })
    }
}

impl SessionStagingCopy {
    fn execute(
        &mut self,
        writes_complete: TransferCompleteNotification,
        sender: &oneshot::Sender<Result<StagedBlocks>>,
    ) -> Result<StagedBlocks> {
        self.settle(futures::executor::block_on(writes_complete.await_drain()))?;
        ensure!(
            !sender.is_closed(),
            "session staging was canceled before dispatch"
        );
        let hashes = self
            .source
            .iter()
            .map(LifecyclePinRef::sequence_hash)
            .collect::<Vec<_>>();
        let mut available = self.route.capacity.scan_matches(&hashes, false);
        self.validate_available(&available)?;
        let missing = self
            .source
            .iter()
            .filter(|pin| !available.contains_key(&pin.sequence_hash()))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            let missing_hashes = missing
                .iter()
                .map(|pin| pin.sequence_hash())
                .collect::<Vec<_>>();
            let source_ids = missing.iter().map(|pin| pin.block_id()).collect();
            let allocation =
                reserve_required_staging(Arc::clone(&self.route.capacity), missing.len())?
                    .stage_all(&missing_hashes, self.route.capacity.block_size())?;
            let destination_ids = allocation.block_ids();
            self.destination = Some(allocation);
            ensure!(
                !sender.is_closed(),
                "session staging was canceled before dispatch"
            );
            self.undrained = true;
            let receipt = match self.route.transfer.execute(
                self.route.resource,
                LogicalLayoutHandle::G1,
                LogicalLayoutHandle::G2,
                source_ids,
                destination_ids,
                TransferOptions::default(),
            ) {
                Ok(receipt) => receipt,
                Err(error) => {
                    self.undrained = false;
                    return Err(error.context("session staging dispatch failed"));
                }
            };
            self.settle(receipt.drain())?;
            self.source.clear();
            ensure!(
                !sender.is_closed(),
                "session staging was canceled during the copy"
            );
            let blocks = self
                .destination
                .take()
                .context("session staging lost its allocation")?
                .publish_temporary()?;
            ensure!(
                blocks.len() == missing_hashes.len(),
                "session staging registration returned an incomplete batch"
            );
            for (hash, block) in missing_hashes.into_iter().zip(blocks) {
                ensure!(
                    block.sequence_hash() == hash,
                    "session staging registration changed a hash"
                );
                available.insert(hash, block);
            }
            self.validate_available(&available)?;
        }
        hashes
            .into_iter()
            .map(|hash| {
                available
                    .remove(&hash)
                    .context("session staging omitted a hash")
            })
            .collect()
    }

    fn validate_available(&self, blocks: &HashMap<SequenceHash, ImmutableBlock<G2>>) -> Result<()> {
        ensure!(
            blocks
                .iter()
                .all(|(hash, block)| *hash == block.sequence_hash()
                    && block.pin().manager_id() == self.route.capacity.manager_id()),
            "session staging G2 blocks belong to another hash or manager"
        );
        Ok(())
    }

    fn settle(&mut self, outcome: TransferDrainOutcome) -> Result<()> {
        match outcome {
            TransferDrainOutcome::Completed => {
                self.undrained = false;
                Ok(())
            }
            TransferDrainOutcome::DrainedWithError(error) => {
                self.undrained = false;
                Err(error)
            }
            TransferDrainOutcome::Unproven(error) => Err(error),
        }
    }
}

impl Drop for SessionStagingCopy {
    fn drop(&mut self) {
        if self.undrained {
            tracing::error!(resource = ?self.route.resource, "session staging has no physical drain proof and retains its pins and capacity");
            std::mem::forget((
                std::mem::take(&mut self.source),
                self.destination.take(),
                Arc::clone(&self.route),
            ));
        }
    }
}
