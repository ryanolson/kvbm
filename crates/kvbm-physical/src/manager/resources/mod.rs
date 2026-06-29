// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Physical layout handles owned by logical KV resource.

use std::collections::BTreeMap;

use anyhow::{Result, ensure};
use kvbm_common::{LogicalLayoutHandle, LogicalResourceId};

use super::LayoutHandle;

/// Physical tier handles keyed by logical KV resource.
#[derive(Clone, Debug)]
pub struct ResourceLayoutHandles {
    primary: LogicalResourceId,
    resources: BTreeMap<LogicalResourceId, TierLayoutHandles>,
}

impl ResourceLayoutHandles {
    pub fn new(
        primary: LogicalResourceId,
        resources: Vec<(LogicalResourceId, TierLayoutHandles)>,
    ) -> Result<Self> {
        let expected_len = resources.len();
        let resources = resources.into_iter().collect::<BTreeMap<_, _>>();
        ensure!(
            resources.len() == expected_len,
            "duplicate logical resource in physical layout handles"
        );
        ensure!(
            resources.contains_key(&primary),
            "primary logical resource {primary:?} is absent from physical layout handles"
        );
        Ok(Self { primary, resources })
    }

    pub fn primary(&self) -> LogicalResourceId {
        self.primary
    }

    pub fn get(&self, resource: LogicalResourceId) -> Option<&TierLayoutHandles> {
        self.resources.get(&resource)
    }

    pub fn handle(
        &self,
        resource: LogicalResourceId,
        tier: LogicalLayoutHandle,
    ) -> Option<LayoutHandle> {
        self.get(resource).and_then(|handles| handles.handle(tier))
    }

    pub fn iter(&self) -> impl Iterator<Item = (LogicalResourceId, &TierLayoutHandles)> + '_ {
        self.resources
            .iter()
            .map(|(&resource, handles)| (resource, handles))
    }
}

/// G1/G2/G3 physical handles for one logical KV resource.
#[derive(Clone, Copy, Debug, Default)]
pub struct TierLayoutHandles {
    g1: Option<LayoutHandle>,
    g2: Option<LayoutHandle>,
    g3: Option<LayoutHandle>,
}

impl TierLayoutHandles {
    pub fn new(
        g1: Option<LayoutHandle>,
        g2: Option<LayoutHandle>,
        g3: Option<LayoutHandle>,
    ) -> Self {
        Self { g1, g2, g3 }
    }

    pub fn handle(&self, tier: LogicalLayoutHandle) -> Option<LayoutHandle> {
        match tier {
            LogicalLayoutHandle::G1 => self.g1,
            LogicalLayoutHandle::G2 => self.g2,
            LogicalLayoutHandle::G3 => self.g3,
            LogicalLayoutHandle::G4 => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_resource_tiers_and_preserves_resource_order() {
        let first = TierLayoutHandles::new(
            Some(LayoutHandle::new(4, 1)),
            Some(LayoutHandle::new(4, 2)),
            None,
        );
        let second = TierLayoutHandles::new(
            Some(LayoutHandle::new(4, 3)),
            Some(LayoutHandle::new(4, 4)),
            Some(LayoutHandle::new(4, 5)),
        );
        let handles = ResourceLayoutHandles::new(
            LogicalResourceId(7),
            vec![
                (LogicalResourceId(7), first),
                (LogicalResourceId(1), second),
            ],
        )
        .unwrap();

        assert_eq!(handles.primary(), LogicalResourceId(7));
        assert_eq!(
            handles.handle(LogicalResourceId(7), LogicalLayoutHandle::G2),
            Some(LayoutHandle::new(4, 2))
        );
        assert_eq!(
            handles.handle(LogicalResourceId(1), LogicalLayoutHandle::G3),
            Some(LayoutHandle::new(4, 5))
        );
        assert_eq!(
            handles
                .iter()
                .map(|(resource, _)| resource)
                .collect::<Vec<_>>(),
            vec![LogicalResourceId(1), LogicalResourceId(7)]
        );
    }

    #[test]
    fn rejects_duplicate_or_missing_primary_resource() {
        let tiers = TierLayoutHandles::new(None, Some(LayoutHandle::new(4, 2)), None);
        assert!(
            ResourceLayoutHandles::new(
                LogicalResourceId(1),
                vec![(LogicalResourceId(1), tiers), (LogicalResourceId(1), tiers),],
            )
            .is_err()
        );
        assert!(
            ResourceLayoutHandles::new(LogicalResourceId(9), vec![(LogicalResourceId(1), tiers)],)
                .is_err()
        );
    }
}
