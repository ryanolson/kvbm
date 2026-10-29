// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the ConditionalDisagg feature.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use axum::{Json, Router, extract::State, routing::get};
use parking_lot::RwLock;
use velo_ext::InstanceId;

use super::protocol::ConditionalDisaggInstancesResponse;
use crate::features::prefill_router::{DecodeSetProvider, TierBroadcaster};
use velo_hub::store::Removal;
use velo_hub::{AttachContext, Committed, Extension, Rejection, Txn};

use crate::features::KvbmFeature;
use crate::protocol::{self, ConditionalDisaggConfig, ConditionalDisaggRole};

/// Tracks which instances participate in ConditionalDisagg and under what role.
///
/// State is kept behind a single `RwLock` — lookups are O(1) via the
/// `by_instance` map, and role-filtered listings iterate the matching set.
pub struct ConditionalDisaggManager {
    inner: RwLock<CdInner>,
    velo: OnceLock<velo::WeakVelo>,
    /// P2 CD-breaker tier-push broadcaster, late-bound by the binary when the
    /// prefill-router breaker is enabled. `on_register` of a NEW decode pushes
    /// the current tier to it so a freshly-joined decode is seeded (not left
    /// at the default Calm while the fleet is HOT). `None` ⇒ breaker disabled
    /// ⇒ no push (inert == prior behavior).
    tier_broadcaster: OnceLock<Arc<TierBroadcaster>>,
}

struct CdInner {
    prefill: HashSet<InstanceId>,
    decode: HashSet<InstanceId>,
    by_instance: HashMap<InstanceId, ConditionalDisaggRole>,
}

impl std::fmt::Debug for ConditionalDisaggManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        f.debug_struct("ConditionalDisaggManager")
            .field("prefill_count", &inner.prefill.len())
            .field("decode_count", &inner.decode.len())
            .field("velo_attached", &self.velo.get().is_some())
            .finish()
    }
}

impl Default for ConditionalDisaggManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ConditionalDisaggManager {
    /// Create an empty manager with no attached Velo.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(CdInner {
                prefill: HashSet::new(),
                decode: HashSet::new(),
                by_instance: HashMap::new(),
            }),
            velo: OnceLock::new(),
            tier_broadcaster: OnceLock::new(),
        }
    }

    /// Late-bind the P2 CD-breaker [`TierBroadcaster`]. Called by the binary
    /// after both the CD manager and the prefill-router manager (which owns the
    /// breaker + broadcaster) have been constructed. Idempotent; first writer
    /// wins. Wiring this is what lets a NEW decode registration seed its tier.
    pub fn set_tier_broadcaster(&self, broadcaster: Arc<TierBroadcaster>) {
        let _ = self.tier_broadcaster.set(broadcaster);
    }

    /// Current snapshot of the role split, sorted deterministically.
    pub fn snapshot(&self) -> ConditionalDisaggInstancesResponse {
        let inner = self.inner.read();
        ConditionalDisaggInstancesResponse {
            prefill: inner.prefill.iter().copied().collect(),
            decode: inner.decode.iter().copied().collect(),
        }
    }

    /// Hub Velo handle stashed during [`velo_hub::Extension::attach`], if any.
    pub fn velo_handle(&self) -> Option<velo::Velo> {
        self.velo.get().and_then(velo::WeakVelo::upgrade)
    }

    fn remove(&self, owner: InstanceId) {
        let mut inner = self.inner.write();
        if let Some(role) = inner.by_instance.remove(&owner) {
            remove_role(&mut inner, owner, role);
        }
    }
}

/// The CD manager owns the decode/prefill role split, so it is the source of
/// truth for the tier-push fan-out target. Returning ONLY the decode set keeps
/// prefill instances off the push path (the MUST-FIX from the design).
impl DecodeSetProvider for ConditionalDisaggManager {
    fn decode_instances(&self) -> Vec<InstanceId> {
        self.inner.read().decode.iter().copied().collect()
    }
}

fn insert_role(inner: &mut CdInner, id: InstanceId, role: ConditionalDisaggRole) {
    match role {
        ConditionalDisaggRole::Prefill => {
            inner.prefill.insert(id);
        }
        ConditionalDisaggRole::Decode => {
            inner.decode.insert(id);
        }
    }
}

fn remove_role(inner: &mut CdInner, id: InstanceId, role: ConditionalDisaggRole) {
    match role {
        ConditionalDisaggRole::Prefill => {
            inner.prefill.remove(&id);
        }
        ConditionalDisaggRole::Decode => {
            inner.decode.remove(&id);
        }
    }
}

impl KvbmFeature for ConditionalDisaggManager {
    fn render_implies(&self) -> &'static [&'static str] {
        // A disagg connector should also join the prefill router, so the
        // prefill side advertises its execution backend. A soft co-enable:
        // a decode sends no router payload, so a hard requirement would
        // reject it.
        &[protocol::PREFILL_ROUTER_EXTENSION]
    }
}

impl Extension for ConditionalDisaggManager {
    const NAME: &'static str = protocol::DISAGG_EXTENSION;
    type Registration = ConditionalDisaggConfig;
    /// The role to record, and whether the owner is new to it.
    type Staged = Option<(ConditionalDisaggRole, bool)>;

    /// CD is a specialisation of P2P: `layout_compat` lives in the P2P
    /// payload, so every CD registrant must also register P2P.
    fn registration_requires() -> &'static [&'static str] {
        &[protocol::P2P_EXTENSION]
    }

    async fn attach(&self, ctx: &AttachContext<'_>) -> anyhow::Result<()> {
        if let Some(velo) = ctx.velo() {
            let _ = self.velo.set(velo.downgrade());
        }
        Ok(())
    }

    async fn prepare(
        &self,
        txn: &Txn<'_>,
        payload: Option<ConditionalDisaggConfig>,
    ) -> Result<Self::Staged, Rejection> {
        let Some(cfg) = payload else {
            return Ok(None);
        };
        let owner = txn.owner;
        match self.inner.read().by_instance.get(&owner).copied() {
            Some(prior) if prior != cfg.role => Err(Rejection::conflict(format!(
                "instance {owner} already registered as {prior:?}, cannot switch to {:?}",
                cfg.role
            ))),
            prior => Ok(Some((cfg.role, prior.is_none()))),
        }
    }

    fn commit(&self, staged: Self::Staged, registration: &Committed<'_>) {
        let owner = registration.owner;
        let Some((role, new)) = staged else {
            if !registration.opted_in(Self::NAME) {
                self.remove(owner);
            }
            return;
        };
        if !new {
            return;
        }
        {
            let mut inner = self.inner.write();
            inner.by_instance.insert(owner, role);
            insert_role(&mut inner, owner, role);
        }
        // A new decode starts at the Calm tier, so seed it with the current
        // breaker tier. A prefill never gets the tier.
        if role == ConditionalDisaggRole::Decode
            && let Some(b) = self.tier_broadcaster.get()
        {
            b.push_current_to(owner);
        }
    }

    fn on_removed(&self, removal: Removal) {
        self.remove(removal.id);
    }

    fn control_routes(self: Arc<Self>) -> Router {
        routes(self)
    }

    fn public_routes(self: Arc<Self>) -> Router {
        routes(self)
    }
}

fn routes(manager: Arc<ConditionalDisaggManager>) -> Router {
    Router::new()
        .route(super::protocol::paths::INSTANCES, get(list_instances))
        .with_state(manager)
}

async fn list_instances(
    State(mgr): State<Arc<ConditionalDisaggManager>>,
) -> Json<ConditionalDisaggInstancesResponse> {
    Json(mgr.snapshot())
}
