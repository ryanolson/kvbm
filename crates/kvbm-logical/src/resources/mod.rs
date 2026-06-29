// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed logical block-manager ownership for multi-resource models.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{BlockManager, BlockMetadata};
pub use kvbm_common::LogicalResourceId;

/// KVBM-owned logical managers keyed by model resource identity.
///
/// Every manager retains its normal typed RAII lifecycle. The set only owns
/// routing and prevents two pools from claiming the same resource ID.
pub struct BlockManagerSet<T: BlockMetadata> {
    managers: BTreeMap<LogicalResourceId, Arc<BlockManager<T>>>,
}

impl<T: BlockMetadata + Sync> BlockManagerSet<T> {
    pub fn new() -> Self {
        Self {
            managers: BTreeMap::new(),
        }
    }

    pub fn insert(
        &mut self,
        resource: LogicalResourceId,
        manager: Arc<BlockManager<T>>,
    ) -> Result<(), DuplicateLogicalResource> {
        if self.managers.contains_key(&resource) {
            return Err(DuplicateLogicalResource { resource });
        }
        self.managers.insert(resource, manager);
        Ok(())
    }

    pub fn get(&self, resource: LogicalResourceId) -> Option<&Arc<BlockManager<T>>> {
        self.managers.get(&resource)
    }

    pub fn iter(&self) -> impl Iterator<Item = (LogicalResourceId, &Arc<BlockManager<T>>)> + '_ {
        self.managers
            .iter()
            .map(|(&resource, manager)| (resource, manager))
    }

    pub fn len(&self) -> usize {
        self.managers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.managers.is_empty()
    }
}

impl<T: BlockMetadata + Sync> Default for BlockManagerSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("logical resource {resource:?} already has a block manager")]
pub struct DuplicateLogicalResource {
    pub resource: LogicalResourceId,
}

#[cfg(test)]
mod tests {
    use super::{BlockManagerSet, LogicalResourceId};
    use crate::{BlockManager, BlockRegistry};
    use std::sync::Arc;

    #[derive(Clone)]
    struct G2;

    fn manager(blocks: usize) -> Arc<BlockManager<G2>> {
        Arc::new(
            BlockManager::builder()
                .block_count(blocks)
                .block_size(4)
                .registry(BlockRegistry::new())
                .with_lru_backend()
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn owns_distinct_managers_in_resource_order() {
        let first = manager(4);
        let second = manager(8);
        let mut set = BlockManagerSet::new();
        set.insert(LogicalResourceId(9), Arc::clone(&first))
            .unwrap();
        set.insert(LogicalResourceId(2), Arc::clone(&second))
            .unwrap();

        assert_eq!(set.len(), 2);
        assert_eq!(
            set.iter().map(|(resource, _)| resource).collect::<Vec<_>>(),
            vec![LogicalResourceId(2), LogicalResourceId(9)]
        );
        assert_eq!(set.get(LogicalResourceId(9)).unwrap().id(), first.id());
        assert_eq!(set.get(LogicalResourceId(2)).unwrap().id(), second.id());
    }

    #[test]
    fn rejects_duplicate_resource_ownership() {
        let mut set = BlockManagerSet::new();
        set.insert(LogicalResourceId(3), manager(4)).unwrap();
        let error = set.insert(LogicalResourceId(3), manager(8)).unwrap_err();
        assert_eq!(error.resource, LogicalResourceId(3));
    }
}
