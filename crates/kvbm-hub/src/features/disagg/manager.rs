// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side manager for the ConditionalDisagg feature.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::{Json, Router, extract::State, routing::get};
use parking_lot::RwLock;
use tokio::task::JoinHandle;
use velo::queue::NextOptions;
use velo::queue::backends::messenger::{MessengerQueueBackend, MessengerQueueConfig};
use velo_ext::InstanceId;

use super::protocol::ConditionalDisaggInstancesResponse;
use crate::features::prefill_router::{
    DecodeSetProvider, DispatchOutcome, PrefillRequestDispatcher, TierBroadcaster,
};
use velo_hub::store::Removal;
use velo_hub::{AttachContext, Committed, Extension, Rejection, Txn};

use crate::features::{FeatureError, KvbmFeature};
use crate::protocol::{self, ConditionalDisaggConfig, ConditionalDisaggRole, PrefillRequest};

/// Tracks which instances participate in ConditionalDisagg and under what role.
///
/// State is kept behind a single `RwLock` — lookups are O(1) via the
/// `by_instance` map, and role-filtered listings iterate the matching set.
pub struct ConditionalDisaggManager {
    inner: RwLock<CdInner>,
    velo: OnceLock<velo::WeakVelo>,
    /// Hub-local queue backend owning the CD prefill queue. Lazily created
    /// during [`velo_hub::Extension::attach`] when the hub has a Velo instance —
    /// `None` when the hub is discovery-only.
    queue_backend: OnceLock<Arc<MessengerQueueBackend>>,
    /// Optional bound on the prefill queue depth. `None` = unbounded.
    queue_capacity: Option<usize>,
    /// Optional dispatcher for the prefill queue. When set,
    /// [`velo_hub::Extension::attach`] spawns a background worker that
    /// drains the queue and hands each request to this dispatcher.
    dispatcher: Option<Arc<dyn PrefillRequestDispatcher>>,
    /// Worker task handle (set once spawned during `attach`).
    dispatcher_task: OnceLock<JoinHandle<()>>,
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
    /// Create an empty manager with no attached Velo and an unbounded
    /// prefill queue.
    pub fn new() -> Self {
        Self::with_queue_capacity(None)
    }

    /// Create a manager with an explicit capacity bound on the prefill queue.
    pub fn with_queue_capacity(capacity: Option<usize>) -> Self {
        Self {
            inner: RwLock::new(CdInner {
                prefill: HashSet::new(),
                decode: HashSet::new(),
                by_instance: HashMap::new(),
            }),
            velo: OnceLock::new(),
            queue_backend: OnceLock::new(),
            queue_capacity: capacity,
            dispatcher: None,
            dispatcher_task: OnceLock::new(),
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

    /// Builder: install a [`PrefillRequestDispatcher`]. When set, the
    /// hub spawns a background worker (in [`velo_hub::Extension::attach`])
    /// that drains the prefill queue and hands each item to the
    /// dispatcher.
    pub fn with_dispatcher(mut self, dispatcher: Arc<dyn PrefillRequestDispatcher>) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    /// Install a [`PrefillRequestDispatcher`] AFTER
    /// [`velo_hub::Extension::attach`] has run. Intended for late-binding:
    /// e.g. the binary attaches the disagg manager first, then attaches
    /// the [`PrefillRouterManager`](crate::PrefillRouterManager), then
    /// installs the router as this manager's dispatcher by calling
    /// `start_dispatcher(router_manager.dispatcher())`.
    ///
    /// Returns an error if no Velo / queue backend was attached (the
    /// hub was started without Velo) or if a dispatcher loop is already
    /// running.
    pub fn start_dispatcher(
        self: &Arc<Self>,
        dispatcher: Arc<dyn PrefillRequestDispatcher>,
    ) -> Result<(), FeatureError> {
        let backend = self.queue_backend.get().cloned().ok_or_else(|| {
            FeatureError::Other(anyhow::anyhow!(
                "start_dispatcher: no queue backend (was the hub started without Velo?)"
            ))
        })?;
        if self.dispatcher_task.get().is_some() {
            return Err(FeatureError::Other(anyhow::anyhow!(
                "start_dispatcher: a dispatcher loop is already running"
            )));
        }
        let task = tokio::spawn(prefill_dispatcher_loop(backend, dispatcher));
        if self.dispatcher_task.set(task).is_err() {
            return Err(FeatureError::Other(anyhow::anyhow!(
                "start_dispatcher: lost race to install dispatcher loop"
            )));
        }
        tracing::info!("CD prefill dispatcher worker started (late-bound)");
        Ok(())
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

    /// Hub-local queue backend for the CD prefill queue, if the hub was
    /// configured with a Velo instance.
    pub fn queue_backend(&self) -> Option<&Arc<MessengerQueueBackend>> {
        self.queue_backend.get()
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
        let Some(velo) = ctx.velo() else {
            // A discovery-only hub has no queue; the CD list routes still work.
            return Ok(());
        };
        let backend = Arc::new(MessengerQueueBackend::new(
            velo.messenger().clone(),
            velo.instance_id(),
            MessengerQueueConfig {
                capacity: self.queue_capacity,
            },
        ));
        // The first receiver registers the `velo.queue.rpc` handler on the
        // hub's Velo. Without it, the first remote enqueue finds no handler.
        // The queue service outlives the dropped receiver while the backend
        // is held.
        velo::queue::receiver::<Vec<u8>>(backend.as_ref(), protocol::CD_PREFILL_QUEUE)
            .await
            .map_err(|e| anyhow::anyhow!("CD queue init: {e}"))?;
        let _ = self.queue_backend.set(Arc::clone(&backend));
        let _ = self.velo.set(velo.downgrade());
        if let Some(dispatcher) = self.dispatcher.clone() {
            let task = tokio::spawn(prefill_dispatcher_loop(Arc::clone(&backend), dispatcher));
            let _ = self.dispatcher_task.set(task);
            tracing::info!("CD prefill dispatcher worker started");
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

/// Long-running task that drains the CD prefill queue and hands each
/// dequeued [`PrefillRequest`] to the configured dispatcher.
///
/// The loop terminates if the queue receiver fails to be (re)created —
/// that signals the underlying messenger backend has shut down. Per-
/// iteration `next_with_options` errors and dispatcher errors are
/// logged and skipped; we never want one bad request to take down the
/// pump.
async fn prefill_dispatcher_loop(
    backend: Arc<MessengerQueueBackend>,
    dispatcher: Arc<dyn PrefillRequestDispatcher>,
) {
    // Long-poll window. The receiver returns as soon as it has a full
    // batch OR the timeout fires — for the dispatcher's purposes we
    // want each request handed off ASAP, so use batch_size=1 (return on
    // first item) with a long idle timeout so wakeups are cheap when
    // nothing's flowing.
    const POLL_TIMEOUT: Duration = Duration::from_secs(30);
    const BATCH_SIZE: usize = 1;

    loop {
        // Re-create the receiver each iteration. The backend caches the
        // underlying handler; this is cheap.
        let receiver = match velo::queue::receiver::<Vec<u8>>(
            backend.as_ref(),
            protocol::CD_PREFILL_QUEUE,
        )
        .await
        {
            Ok(r) => r,
            Err(err) => {
                tracing::error!(error = %err, "CD dispatcher: receiver build failed; shutting down loop");
                return;
            }
        };

        let batch = match receiver
            .next_with_options(
                NextOptions::new()
                    .batch_size(BATCH_SIZE)
                    .timeout(POLL_TIMEOUT),
            )
            .await
        {
            Ok(b) => b,
            Err(err) => {
                tracing::warn!(error = %err, "CD dispatcher: dequeue failed; retrying");
                continue;
            }
        };

        if batch.is_empty() {
            // Idle window — long-poll timed out. Loop back.
            continue;
        }

        for bytes in batch {
            let req: PrefillRequest = match serde_json::from_slice(&bytes) {
                Ok(r) => r,
                Err(err) => {
                    tracing::error!(
                        error = %err,
                        bytes_len = bytes.len(),
                        "CD dispatcher: undecodable PrefillRequest; dropping"
                    );
                    continue;
                }
            };
            let request_id = req.request_id.clone();
            tracing::info!(
                request_id = request_id,
                session_id = %req.session_id,
                initiator = %req.initiator_instance_id,
                "CD dispatcher: dispatching PrefillRequest"
            );
            match dispatcher.dispatch(req).await {
                Ok(DispatchOutcome::Accepted) => {
                    tracing::info!(request_id, "CD dispatcher: accepted");
                }
                Ok(DispatchOutcome::Rejected { reason }) => {
                    tracing::warn!(request_id, reason, "CD dispatcher: rejected");
                }
                Err(err) => {
                    tracing::error!(request_id, error = %err, "CD dispatcher: error");
                }
            }
        }
    }
}
