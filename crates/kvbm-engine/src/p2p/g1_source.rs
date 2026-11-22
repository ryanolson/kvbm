use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use anyhow::{Result, ensure};
use futures::future::BoxFuture;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::{BlockManager, ImmutableBlock, ManagerId};
use parking_lot::RwLock;

use crate::G2;
use crate::g2_capacity::{PolicyG1G2BoundRoute, PolicyG1SourceMetadata};
use crate::leader::InstanceLeader;

/// One holder-side source of G1 (device) blocks for remote search.
///
/// The logical manager installs one instance per resource and per lane. The
/// lane is the execution lane of the G1 manager (`BlockManager::lane`), and
/// the leader keeps the sources of one resource apart by it. A
/// holder search opens a pin set through `G1SessionSource::pins`, then
/// stages the pinned blocks into G2 through the certified
/// [`crate::g2_capacity::PolicyG1G2BoundRoute`].
/// [`G1SessionSource::set_enabled`] revokes new pins. It does not cancel
/// a copy that already holds its pins.
pub struct G1SessionSource {
    resource: LogicalResourceId,
    lane: u8,
    destination_manager: ManagerId,
    enabled: AtomicBool,
    lookup: Box<dyn SourceLookup>,
}

trait SourceLookup: Send + Sync {
    fn pins(&self) -> Box<dyn PinnedG1Source>;
}

struct TypedSource<T: PolicyG1SourceMetadata> {
    manager: Weak<BlockManager<T>>,
    route: Arc<PolicyG1G2BoundRoute<T>>,
}

/// The G1 blocks one holder search pinned, and the copy that stages them
/// into G2.
///
/// The search walks the tiers from a single prefix cursor, so a G1 run
/// and a G2 run alternate and the pinned set grows over several calls.
/// The set stages as one batch, which reserves G2 capacity once per
/// session instead of once per run.
///
/// Every pinned block is a block the holder committed, so a set that
/// never grew carries no copy and the caller drops it.
pub(crate) trait PinnedG1Source: Send {
    /// Pin the leading run of `hashes` that this tier holds and report
    /// how many blocks it added. The count advances the caller's
    /// cross-tier prefix cursor.
    fn pin_prefix(&mut self, hashes: &[SequenceHash]) -> usize;

    /// Pin every hash this tier holds, ignoring gaps.
    fn pin(&mut self, hashes: &[SequenceHash]);

    fn len(&self) -> usize;

    fn hashes(&self) -> Vec<SequenceHash>;

    fn stage(self: Box<Self>) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>>;
}

struct TypedPins<T: PolicyG1SourceMetadata> {
    manager: Weak<BlockManager<T>>,
    blocks: Vec<ImmutableBlock<T>>,
    route: Arc<PolicyG1G2BoundRoute<T>>,
}

impl G1SessionSource {
    pub(crate) fn new<T: PolicyG1SourceMetadata + Send + 'static>(
        resource: LogicalResourceId,
        lane: u8,
        destination_manager: ManagerId,
        manager: Weak<BlockManager<T>>,
        route: PolicyG1G2BoundRoute<T>,
    ) -> Arc<Self> {
        Arc::new(Self {
            resource,
            lane,
            destination_manager,
            enabled: AtomicBool::new(true),
            lookup: Box::new(TypedSource {
                manager,
                route: Arc::new(route),
            }),
        })
    }

    /// The execution lane of the G1 manager behind this source.
    pub fn lane(&self) -> u8 {
        self.lane
    }

    /// Gate new pins. A copy that already holds its pins runs to
    /// completion: revoking a source must not abandon a live DMA into G2
    /// slots the destination manager still owns.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Open an empty pin set for one holder search, or `None` when the
    /// logical owner disabled this source. A set whose G1 manager is gone
    /// pins nothing, so a retired manager reads the same as a disabled
    /// source.
    pub(crate) fn pins(&self) -> Option<Box<dyn PinnedG1Source>> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }
        Some(self.lookup.pins())
    }
}

impl<T: PolicyG1SourceMetadata + Send + 'static> SourceLookup for TypedSource<T> {
    fn pins(&self) -> Box<dyn PinnedG1Source> {
        Box::new(TypedPins {
            manager: self.manager.clone(),
            blocks: Vec::new(),
            route: Arc::clone(&self.route),
        })
    }
}

impl<T: PolicyG1SourceMetadata + Send + 'static> PinnedG1Source for TypedPins<T> {
    fn pin_prefix(&mut self, hashes: &[SequenceHash]) -> usize {
        let Some(manager) = self.manager.upgrade() else {
            return 0;
        };
        // touch = false: a remote request must not count as a local hit.
        // If the walk counted it, the block this node no longer reads
        // stays ahead of one its own requests still need.
        let run = manager.match_prefix(hashes, false);
        let pinned = run.len();
        self.blocks.extend(run);
        pinned
    }

    fn pin(&mut self, hashes: &[SequenceHash]) {
        let Some(manager) = self.manager.upgrade() else {
            return;
        };
        // touch = false for the same reason as the prefix walk.
        let mut matches = manager.scan_matches(hashes, false);
        self.blocks
            .extend(hashes.iter().filter_map(|hash| matches.remove(hash)));
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }

    fn hashes(&self) -> Vec<SequenceHash> {
        self.blocks
            .iter()
            .map(ImmutableBlock::sequence_hash)
            .collect()
    }

    fn stage(self: Box<Self>) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>> {
        self.route.stage_to_g2(self.blocks)
    }
}

/// The pin sets of the lanes of one resource for one holder search.
///
/// Each lane has its own device pool, so a block of the request can be on
/// any lane. Each lane stages its own pins into the G2 pool that all lanes
/// share, so the search treats the lanes as one G1 tier.
struct LanePins {
    lanes: Vec<Box<dyn PinnedG1Source>>,
}

impl PinnedG1Source for LanePins {
    fn pin_prefix(&mut self, hashes: &[SequenceHash]) -> usize {
        // Each pass asks each lane for the run at the cursor. The walk ends
        // after a pass in which no lane holds the hash at the cursor.
        let mut cursor = 0;
        loop {
            let start = cursor;
            for lane in &mut self.lanes {
                cursor += lane.pin_prefix(&hashes[cursor..]);
            }
            if cursor == start || cursor == hashes.len() {
                return cursor;
            }
        }
    }

    fn pin(&mut self, hashes: &[SequenceHash]) {
        // A hash that an earlier lane pinned needs no second copy.
        let mut remaining = hashes.to_vec();
        for lane in &mut self.lanes {
            if remaining.is_empty() {
                return;
            }
            let before: HashSet<SequenceHash> = lane.hashes().into_iter().collect();
            lane.pin(&remaining);
            let pinned: HashSet<SequenceHash> = lane
                .hashes()
                .into_iter()
                .filter(|hash| !before.contains(hash))
                .collect();
            remaining.retain(|hash| !pinned.contains(hash));
        }
    }

    fn len(&self) -> usize {
        self.lanes.iter().map(|lane| lane.len()).sum()
    }

    fn hashes(&self) -> Vec<SequenceHash> {
        self.lanes.iter().flat_map(|lane| lane.hashes()).collect()
    }

    fn stage(self: Box<Self>) -> BoxFuture<'static, Result<Vec<ImmutableBlock<G2>>>> {
        let copies: Vec<_> = self
            .lanes
            .into_iter()
            .filter(|lane| lane.len() > 0)
            .map(|lane| lane.stage())
            .collect();
        Box::pin(async move {
            let mut staged = Vec::new();
            for blocks in futures::future::join_all(copies).await {
                staged.extend(blocks?);
            }
            Ok(staged)
        })
    }
}

#[derive(Default)]
pub(crate) struct G1SourceRegistry {
    sources: RwLock<HashMap<(LogicalResourceId, u8), Weak<G1SessionSource>>>,
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
                resources.insert((source.resource, source.lane)),
                "duplicate G1 source resource {:?} in lane {}",
                source.resource,
                source.lane
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
            if let Some(current) = installed.get(&(source.resource, source.lane)) {
                // The map keeps its own `Weak`, which reserves the
                // allocation for as long as the entry lives. A later
                // `Arc` therefore never lands on the retired address, so
                // pointer equality cannot mistake a replacement source
                // for the original binding.
                ensure!(
                    current.ptr_eq(&Arc::downgrade(source)),
                    "G1 source resource {:?} in lane {} already has a logical binding",
                    source.resource,
                    source.lane
                );
            }
        }
        for source in sources {
            installed.insert((source.resource, source.lane), Arc::downgrade(source));
        }
        Ok(())
    }

    /// The source of `resource` in one lane.
    pub(crate) fn get(
        &self,
        resource: LogicalResourceId,
        lane: u8,
    ) -> Option<Arc<G1SessionSource>> {
        self.sources
            .read()
            .get(&(resource, lane))
            .and_then(Weak::upgrade)
    }

    /// Open an empty pin set over the enabled sources of `resource`, in lane
    /// order. `None` when no lane has an enabled source. One lane gives the
    /// pin set of its source as it is.
    pub(crate) fn pins(&self, resource: LogicalResourceId) -> Option<Box<dyn PinnedG1Source>> {
        let mut sources: Vec<Arc<G1SessionSource>> = self
            .sources
            .read()
            .iter()
            .filter(|((source_resource, _), _)| *source_resource == resource)
            .filter_map(|(_, source)| source.upgrade())
            .collect();
        sources.sort_by_key(|source| source.lane);
        let mut lanes: Vec<Box<dyn PinnedG1Source>> =
            sources.iter().filter_map(|source| source.pins()).collect();
        match lanes.len() {
            0 => None,
            1 => lanes.pop(),
            _ => Some(Box::new(LanePins { lanes })),
        }
    }
}
