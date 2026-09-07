use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use anyhow::{Result, ensure};
use futures::future::BoxFuture;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::{BlockManager, ImmutableBlock, ManagerId};
use kvbm_physical::transfer::TransferCompleteNotification;
use parking_lot::RwLock;

use crate::G2;
use crate::g2_capacity::{PolicyG1G2BoundRoute, PolicyG1SourceMetadata};
use crate::leader::InstanceLeader;

pub struct G1SessionSource {
    resource: LogicalResourceId,
    destination_manager: ManagerId,
    enabled: AtomicBool,
    lookup: Box<dyn SourceLookup>,
}

trait SourceLookup: Send + Sync {
    fn pin(&self, hashes: &[SequenceHash]) -> Option<Box<dyn PinnedG1Source>>;
}

struct TypedSource<T: PolicyG1SourceMetadata> {
    manager: Weak<BlockManager<T>>,
    route: Arc<PolicyG1G2BoundRoute<T>>,
}

pub(crate) trait PinnedG1Source: Send {
    fn hashes(&self) -> Vec<SequenceHash>;
    fn retain(&mut self, hashes: &HashSet<SequenceHash>);
    fn stage(self: Box<Self>) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>>;
}

struct TypedPins<T: PolicyG1SourceMetadata> {
    blocks: Vec<ImmutableBlock<T>>,
    route: Arc<PolicyG1G2BoundRoute<T>>,
}

impl G1SessionSource {
    pub(crate) fn new<T: PolicyG1SourceMetadata + Send + 'static>(
        resource: LogicalResourceId,
        destination_manager: ManagerId,
        manager: Weak<BlockManager<T>>,
        route: PolicyG1G2BoundRoute<T>,
    ) -> Arc<Self> {
        Arc::new(Self {
            resource,
            destination_manager,
            enabled: AtomicBool::new(true),
            lookup: Box::new(TypedSource {
                manager,
                route: Arc::new(route),
            }),
        })
    }

    pub fn resource(&self) -> LogicalResourceId {
        self.resource
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    pub(crate) fn pin(&self, hashes: &[SequenceHash]) -> Option<Box<dyn PinnedG1Source>> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }
        self.lookup.pin(hashes)
    }
}

impl<T: PolicyG1SourceMetadata + Send + 'static> SourceLookup for TypedSource<T> {
    fn pin(&self, hashes: &[SequenceHash]) -> Option<Box<dyn PinnedG1Source>> {
        let manager = self.manager.upgrade()?;
        let mut matches = manager.scan_matches(hashes, false);
        let blocks = hashes
            .iter()
            .filter_map(|hash| matches.remove(hash))
            .collect::<Vec<_>>();
        if blocks.is_empty() {
            return None;
        }
        Some(Box::new(TypedPins {
            blocks,
            route: Arc::clone(&self.route),
        }))
    }
}

impl<T: PolicyG1SourceMetadata + Send + 'static> PinnedG1Source for TypedPins<T> {
    fn hashes(&self) -> Vec<SequenceHash> {
        self.blocks
            .iter()
            .map(ImmutableBlock::sequence_hash)
            .collect()
    }

    fn retain(&mut self, hashes: &HashSet<SequenceHash>) {
        self.blocks
            .retain(|block| hashes.contains(&block.sequence_hash()));
    }

    fn stage(self: Box<Self>) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>> {
        self.route
            .stage_to_g2(self.blocks, TransferCompleteNotification::completed())
    }
}

#[derive(Default)]
pub(crate) struct G1SourceRegistry {
    sources: RwLock<HashMap<LogicalResourceId, Weak<G1SessionSource>>>,
}

impl G1SourceRegistry {
    pub(crate) fn install(
        &self,
        leader: &InstanceLeader,
        sources: &[Arc<G1SessionSource>],
    ) -> Result<()> {
        let mut resources = HashSet::new();
        for source in sources {
            ensure!(
                resources.insert(source.resource),
                "duplicate G1 source resource {:?}",
                source.resource
            );
            let capacity = leader.g2_capacity_for(source.resource).ok_or_else(|| {
                anyhow::anyhow!("G1 source has no G2 resource {:?}", source.resource)
            })?;
            ensure!(
                capacity.manager_id() == source.destination_manager,
                "G1 source belongs to another destination manager"
            );
        }
        let mut installed = self.sources.write();
        for source in sources {
            if let Some(current) = installed.get(&source.resource) {
                ensure!(
                    current.ptr_eq(&Arc::downgrade(source)),
                    "G1 source resource {:?} already has a logical binding",
                    source.resource
                );
            }
        }
        for source in sources {
            installed.insert(source.resource, Arc::downgrade(source));
        }
        Ok(())
    }

    pub(crate) fn get(&self, resource: LogicalResourceId) -> Option<Arc<G1SessionSource>> {
        self.sources.read().get(&resource).and_then(Weak::upgrade)
    }
}
