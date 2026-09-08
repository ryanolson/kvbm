use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use futures::future::BoxFuture;
use kvbm_common::{LogicalLayoutHandle, SequenceHash};
use kvbm_logical::{ImmutableBlock, LifecyclePinRef};
use kvbm_physical::transfer::{TransferDrainOutcome, TransferOptions};
use tokio::sync::oneshot;

use super::{PolicyG1G2BoundRoute, PolicyG1G2RouteCore, PolicyG1SourceMetadata};
use crate::G2;
use crate::g2_capacity::{RequiredStagingStagedAllocation, reserve_required_staging};

type StagedBlocks = Vec<ImmutableBlock<G2>>;

struct SessionStagingCopy {
    route: Arc<PolicyG1G2RouteCore>,
    source: Vec<LifecyclePinRef>,
    destination: Option<RequiredStagingStagedAllocation>,
    undrained: bool,
}

impl<T: PolicyG1SourceMetadata> PolicyG1G2BoundRoute<T> {
    pub fn into_session_source(
        self,
        manager: &Arc<kvbm_logical::BlockManager<T>>,
    ) -> Result<Arc<crate::p2p::g1_source::G1SessionSource>>
    where
        T: Send + 'static,
    {
        ensure!(
            manager.id() == self.source_manager_id,
            "session source belongs to another G1 manager"
        );
        Ok(crate::p2p::g1_source::G1SessionSource::new(
            self.core.resource,
            self.core.capacity.manager_id(),
            Arc::downgrade(manager),
            self,
        ))
    }

    /// Copy the registered G1 blocks of `source` into temporary G2 blocks.
    ///
    /// The copy carries no source-write fence. Every pin names a registered
    /// block, and the installer's contract states that a registered block
    /// holds completed writes. If the copy carried a fence, the fence waits
    /// on a receipt that the caller cannot fail to satisfy. The fence
    /// parameter then hides the real rule at the registration seam. The
    /// Rhino half of the same rule lives on
    /// `KvRuntime::register_request_blocks`.
    pub(crate) fn stage_to_g2(
        &self,
        source: Vec<ImmutableBlock<T>>,
    ) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>> {
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
            // No DMA runs before the dispatch site re-arms this flag, so a
            // task that ends early releases its pins and its capacity.
            undrained: false,
        };
        // tokio 1.48.0 Handle::spawn has no panic path: it reaches
        // spawn_named -> Inner::spawn, and the only spawn-time panic in this
        // tokio version is spawn_blocking's SpawnError::NoThreads
        // (tokio/src/runtime/blocking/pool.rs:324), which this call never
        // takes.
        runtime.spawn(async move {
            let result = copy.execute(&sender).await;
            drop(copy);
            let _ = sender.send(result);
        });
        Box::pin(async move {
            receiver
                .await
                .context("session staging task ended without a result")?
        })
    }
}

impl SessionStagingCopy {
    async fn execute(
        &mut self,
        sender: &oneshot::Sender<Result<StagedBlocks>>,
    ) -> Result<StagedBlocks> {
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
            self.settle(receipt.into_completion().await)?;
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

/// Leak the pins, the destination allocation, and the route on an unproven
/// drain.
///
/// The leak is the fail-closed answer to a DMA that can still be live. A
/// release returns the G1 pages and the G2 slot to the free pool while the
/// engine can still write them, which corrupts another request. The route
/// `Arc` joins the leak because it keeps the transfer executor alive: the
/// leader, the workers, and the registered layouts must outlive a copy that
/// can still run. The forgotten allocation already holds its slot owner.
///
/// A runtime shutdown drops the staging task at its await point, which
/// reaches this path with `undrained` set. The copy is then exactly as
/// unproven as a panic or an uncertain drain, so it gets the same answer.
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
