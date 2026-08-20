// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Certified installation capability for one exact G1-to-G2 route.

use std::marker::PhantomData;
use std::sync::Arc;

use kvbm_logical::ManagerId;
use kvbm_logical::blocks::BlockMetadata;

use super::PolicyG1G2Route;

/// Metadata that represents a physical G1 source for the exact policy route.
///
/// # Safety
///
/// The metadata type must map only to a physical G1 block manager. The exact
/// transaction hard-codes the G1 physical layout for every implementation.
pub unsafe trait PolicyG1SourceMetadata: BlockMetadata {}

// SAFETY: `crate::G1` is the engine's physical GPU-tier metadata marker.
unsafe impl PolicyG1SourceMetadata for crate::G1 {}

/// One certified physical route and its linear logical half.
///
/// Only the unsafe [`PolicyG1G2Route`] constructors can mint this capability.
/// Safe code can move it to the trusted logical factory, but cannot split or
/// clone it.
#[must_use = "install this certified route or return it to its one-shot cell"]
pub struct PolicyG1G2Installation {
    pub(super) route: PolicyG1G2Route,
    pub(super) manager_identity: ExactG1G2ManagerIdentity,
}

/// One installation validated against a typed G1 manager.
///
/// This intermediate capability keeps validation before the reversible route
/// claim. Its final bind is infallible.
#[doc(hidden)]
pub struct PolicyG1G2ValidatedInstallation<T: PolicyG1SourceMetadata> {
    pub(super) installation: PolicyG1G2Installation,
    pub(super) manager_id: ManagerId,
    pub(super) source_type: PhantomData<fn() -> T>,
}

pub(super) struct ExactG1G2RouteIdentity {
    core: Arc<IdentityCore>,
}

pub(super) struct ExactG1G2ManagerIdentity {
    core: Arc<IdentityCore>,
}

struct IdentityCore;

pub(super) fn new_identity_pair() -> (ExactG1G2RouteIdentity, ExactG1G2ManagerIdentity) {
    let core = Arc::new(IdentityCore);
    (
        ExactG1G2RouteIdentity {
            core: Arc::clone(&core),
        },
        ExactG1G2ManagerIdentity { core },
    )
}

impl ExactG1G2RouteIdentity {
    pub(super) fn matches_manager(&self, manager: &ExactG1G2ManagerIdentity) -> bool {
        Arc::ptr_eq(&self.core, &manager.core)
    }
}

impl<T: PolicyG1SourceMetadata> PolicyG1G2ValidatedInstallation<T> {
    /// Recover the complete installation before its logical claim starts.
    pub fn into_installation(self) -> PolicyG1G2Installation {
        self.installation
    }
}

impl std::fmt::Debug for PolicyG1G2Installation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("PolicyG1G2Installation(..)")
    }
}
