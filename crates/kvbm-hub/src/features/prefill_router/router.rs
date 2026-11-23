// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Places each prefill request on a worker that the load-aware
//! [`Selector`] picks, and holds that worker's capacity until the request
//! ends.
//!
//! A decode worker sends a request to [`PREFILL_SUBMIT_HANDLER`] on the hub.
//! [`PrefillRouter::submit`] picks the free worker with the least load and
//! sends the request to it. A velo worker acks when it has taken the request,
//! and sends [`PREFILL_DONE_HANDLER`] when the request ends; the charge is
//! held until then. When no worker is free, or the chosen workers refuse the
//! request, the decode worker gets a rejection at once and can prefill
//! locally. The hub holds no request.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use anyhow::Result;
use kvbm_protocols::disagg::FPST_PROTOCOL_VERSION;
use parking_lot::Mutex;
use velo::{Context, Handler, Messenger};
use velo_ext::{InstanceId, WorkerId};

use super::execution::DispatchOutcome;
use super::protocol::{
    PREFILL_DONE_HANDLER, PREFILL_SUBMIT_HANDLER, PrefillDispatchResponse, PrefillDone,
};
use super::selection::{PickedSlot, Selector};
use crate::protocol::PrefillRequest;

/// The number of workers that a request is sent to before it is rejected.
const MAX_ATTEMPTS: usize = 2;

pub struct PrefillRouter {
    selector: Arc<Selector>,
    allow_fpst: bool,
    /// The charges of the requests that a velo worker took and has not
    /// finished, by request id.
    running: Mutex<HashMap<String, PickedSlot>>,
}

impl std::fmt::Debug for PrefillRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillRouter")
            .field("selector", &self.selector)
            .field("running", &self.running.lock().len())
            .finish()
    }
}

impl PrefillRouter {
    /// Build a router for ordinary remote prefill.
    pub fn new(selector: Arc<Selector>) -> Arc<Self> {
        Self::with_fpst(selector, false)
    }

    pub(super) fn with_fpst(selector: Arc<Selector>, allow_fpst: bool) -> Arc<Self> {
        Arc::new(Self {
            selector,
            allow_fpst,
            running: Mutex::new(HashMap::new()),
        })
    }

    /// Underlying selector (for the manager's HTTP introspection
    /// endpoints).
    pub fn selector(&self) -> &Arc<Selector> {
        &self.selector
    }

    /// Send `request` to the free worker with the least load. If that worker
    /// does not take it, try one other free worker.
    pub async fn submit(&self, request: PrefillRequest) -> DispatchOutcome {
        let validation = if self.allow_fpst && request.protocol_version == FPST_PROTOCOL_VERSION {
            request
                .remote_prefill_params()
                .validate_for_fpst(&request.request_id, request.token_ids.len())
        } else {
            request.validate()
        };
        if let Err(error) = validation {
            return DispatchOutcome::Rejected {
                reason: format!("invalid remote prefill request: {error:?}"),
            };
        }
        let net_new = self.selector.config().net_new(&request);
        let mut tried = Vec::new();
        let mut reason = String::from("no prefill worker has free capacity");
        while tried.len() < MAX_ATTEMPTS {
            let Some(picked) = self.selector.try_pick(net_new, &tried) else {
                break;
            };
            let instance_id = picked.slot.instance_id;
            let backend = Arc::clone(&picked.slot.backend);
            tracing::debug!(
                %instance_id,
                backend = backend.label(),
                request_id = %request.request_id,
                net_new,
                "PrefillRouter: dispatching"
            );
            // Request IDs are local to each decode worker; the session ID
            // is unique across them.
            let session = request.session_id.to_string();
            if !backend.reports_done() {
                tokio::spawn(async move {
                    let _picked = picked;
                    log_terminal(backend.label(), &request, backend.execute(&request).await);
                });
                return DispatchOutcome::Accepted;
            }
            {
                let mut running = self.running.lock();
                if running.contains_key(&session) {
                    return DispatchOutcome::Rejected {
                        reason: format!(
                            "request {} is already running on a prefill worker",
                            request.request_id
                        ),
                    };
                }
                running.insert(session.clone(), picked);
            }
            match backend.execute(&request).await {
                Ok(DispatchOutcome::Accepted) => return DispatchOutcome::Accepted,
                Ok(DispatchOutcome::Rejected { reason: refused }) => reason = refused,
                // The worker may run the request, so no other worker gets it.
                // Its charge stays until its done notice or until it leaves.
                Ok(DispatchOutcome::Unknown { reason }) => {
                    tracing::warn!(
                        %instance_id,
                        request_id = %request.request_id,
                        %reason,
                        "PrefillRouter: no reply from the worker"
                    );
                    return DispatchOutcome::Rejected { reason };
                }
                Err(error) => reason = format!("{error:#}"),
            }
            self.running.lock().remove(&session);
            tracing::warn!(
                %instance_id,
                request_id = %request.request_id,
                %reason,
                "PrefillRouter: worker did not take the request"
            );
            tried.push(instance_id);
        }
        DispatchOutcome::Rejected { reason }
    }

    /// Free the charge of the session `session_id` when the worker that runs
    /// it reports that it ended. A notice from another sender is ignored.
    pub fn done(&self, session_id: &str, sender: WorkerId) {
        let mut running = self.running.lock();
        match running.get(session_id) {
            Some(picked) if picked.slot.instance_id.worker_id() == sender => {
                running.remove(session_id);
            }
            Some(_) => tracing::warn!(
                session_id,
                "PrefillRouter: done notice from a worker that does not run the request"
            ),
            None => {}
        }
    }

    /// Free the charges of every request on a worker that left the fleet.
    pub fn forget_worker(&self, instance_id: InstanceId) {
        self.running
            .lock()
            .retain(|_, picked| picked.slot.instance_id != instance_id);
    }

    /// Install [`PREFILL_SUBMIT_HANDLER`] and [`PREFILL_DONE_HANDLER`] on the
    /// hub's messenger.
    pub fn register_handlers(self: &Arc<Self>, messenger: &Messenger) -> Result<()> {
        let router = Arc::downgrade(self);
        let submit = Handler::typed_unary_async::<PrefillRequest, PrefillDispatchResponse, _, _>(
            PREFILL_SUBMIT_HANDLER,
            move |ctx| {
                let router = Weak::clone(&router);
                async move {
                    let outcome = match router.upgrade() {
                        Some(router) => router.submit(ctx.input).await,
                        None => DispatchOutcome::Rejected {
                            reason: "the prefill router stopped".into(),
                        },
                    };
                    Ok(match outcome {
                        DispatchOutcome::Accepted => PrefillDispatchResponse {
                            ok: true,
                            error: None,
                        },
                        DispatchOutcome::Rejected { reason }
                        | DispatchOutcome::Unknown { reason } => PrefillDispatchResponse {
                            ok: false,
                            error: Some(reason),
                        },
                    })
                }
            },
        )
        .build();
        let router = Arc::downgrade(self);
        let done = Handler::am_handler(PREFILL_DONE_HANDLER, move |ctx: Context| {
            let done: PrefillDone = serde_json::from_slice(&ctx.payload)?;
            if let Some(router) = router.upgrade() {
                router.done(&done.session_id, ctx.sender_worker_id());
            }
            Ok(())
        })
        .build();
        for handler in [submit, done] {
            let name = handler.name().to_owned();
            messenger
                .register_handler(handler)
                .map_err(|error| anyhow::anyhow!("register {name}: {error}"))?;
        }
        Ok(())
    }
}

fn log_terminal(label: &str, request: &PrefillRequest, outcome: Result<DispatchOutcome>) {
    let request_id = &request.request_id;
    match outcome {
        Ok(DispatchOutcome::Accepted) => {
            tracing::debug!(backend = label, request_id, "PrefillRouter: completed");
        }
        Ok(DispatchOutcome::Rejected { reason } | DispatchOutcome::Unknown { reason }) => {
            tracing::warn!(
                backend = label,
                request_id,
                reason,
                "PrefillRouter: rejected"
            );
        }
        Err(error) => {
            tracing::error!(backend = label, request_id, error = %error, "PrefillRouter: error");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::prefill_router::execution::{
        PrefillExecutionBackend, VeloExecutionBackend,
    };
    use crate::features::prefill_router::protocol::{
        PREFILL_DISPATCH_HANDLER, PrefillDispatchRequest,
    };
    use crate::features::prefill_router::selection::SelectorConfig;
    use crate::features::prefill_router::worker::{PrefillAssignment, register_prefill_worker};
    use async_trait::async_trait;
    use kvbm_protocols::disagg::DISAGG_PROTOCOL_VERSION;
    use std::time::Duration;
    use tokio::sync::{Notify, mpsc};
    use velo::transports::tcp::TcpTransportBuilder;

    /// A backend that returns only when `release` fires, like the HTTP one.
    struct TerminalBackend {
        id: InstanceId,
        release: Arc<Notify>,
    }

    #[async_trait]
    impl PrefillExecutionBackend for TerminalBackend {
        fn instance_id(&self) -> InstanceId {
            self.id
        }
        fn label(&self) -> &'static str {
            "terminal"
        }
        fn reports_done(&self) -> bool {
            false
        }
        async fn execute(&self, _req: &PrefillRequest) -> Result<DispatchOutcome> {
            self.release.notified().await;
            Ok(DispatchOutcome::Accepted)
        }
    }

    fn make_request(id: &str) -> PrefillRequest {
        use kvbm_protocols::disagg::KvHashingRequestEnvelope;
        PrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: id.to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: InstanceId::new_v4(),
            decode_endpoint: None,
            token_ids: vec![0u32; 64],
            num_provided_tokens: 0,
            request: KvHashingRequestEnvelope::default(),
            scope_salt: None,
            expected_hash_digest: None,
            bundle: None,
        }
    }

    fn selector(per_worker_concurrency: u32) -> Arc<Selector> {
        Selector::new(SelectorConfig {
            per_worker_concurrency,
            block_size: 16,
        })
    }

    async fn new_velo() -> velo::Velo {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let transport = TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap();
        velo::Velo::builder()
            .add_transport(Arc::new(transport))
            .build()
            .await
            .unwrap()
    }

    /// A hub with the router handlers, and `workers` velo peers of it.
    struct Fleet {
        hub: velo::Velo,
        router: Arc<PrefillRouter>,
        workers: Vec<velo::Velo>,
    }

    impl Fleet {
        async fn new(per_worker_concurrency: u32, workers: usize) -> Self {
            let hub = new_velo().await;
            let router = PrefillRouter::new(selector(per_worker_concurrency));
            router.register_handlers(hub.messenger()).unwrap();
            let mut peers = Vec::new();
            for _ in 0..workers {
                let worker = new_velo().await;
                hub.register_peer(worker.peer_info()).unwrap();
                worker.register_peer(hub.peer_info()).unwrap();
                router.selector().add_worker(
                    worker.instance_id(),
                    VeloExecutionBackend::new(worker.instance_id(), hub.messenger().clone()),
                );
                peers.push(worker);
            }
            Self {
                hub,
                router,
                workers: peers,
            }
        }

        fn inbox(&self, worker: usize) -> mpsc::Receiver<PrefillAssignment> {
            register_prefill_worker(self.workers[worker].messenger(), 8, |_| Ok(())).unwrap()
        }
    }

    async fn eventually(mut check: impl FnMut() -> bool) -> bool {
        for _ in 0..200 {
            if check() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        check()
    }

    /// A worker that took the frame but never replied may run the request.
    /// The router must not give it to a second worker, or two workers would
    /// serve one session, and the silent worker keeps its charge.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_worker_that_never_replies_gets_no_retry_and_keeps_its_charge() {
        let fleet = Fleet::new(1, 2).await;
        let received = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for worker in &fleet.workers {
            let received = Arc::clone(&received);
            let handler = Handler::typed_unary_async::<
                PrefillDispatchRequest,
                PrefillDispatchResponse,
                _,
                _,
            >(PREFILL_DISPATCH_HANDLER, move |_ctx| {
                received.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async move { std::future::pending().await }
            })
            .build();
            worker.messenger().register_handler(handler).unwrap();
        }

        let outcome = fleet.router.submit(make_request("r1")).await;
        assert!(
            matches!(outcome, DispatchOutcome::Rejected { .. }),
            "{outcome:?}"
        );
        assert_eq!(received.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(fleet.router.selector().available_permits(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_charge_lasts_until_the_worker_reports_done() {
        let fleet = Fleet::new(1, 1).await;
        let mut inbox = fleet.inbox(0);

        let outcome = fleet.router.submit(make_request("r1")).await;
        assert_eq!(outcome, DispatchOutcome::Accepted);
        let assignment = inbox.recv().await.expect("the worker takes the request");
        assert_eq!(assignment.request.request_id, "r1");
        assert_eq!(fleet.router.selector().available_permits(), 0);

        drop(assignment);
        assert!(eventually(|| fleet.router.selector().available_permits() == 1).await);
        assert_eq!(fleet.router.selector().snapshot()[0].counters().inflight, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_busy_worker_sends_the_next_request_to_the_free_one() {
        let fleet = Fleet::new(1, 2).await;
        let mut first = fleet.inbox(0);
        let mut second = fleet.inbox(1);

        assert_eq!(
            fleet.router.submit(make_request("r1")).await,
            DispatchOutcome::Accepted
        );
        assert_eq!(
            fleet.router.submit(make_request("r2")).await,
            DispatchOutcome::Accepted
        );
        let one = first.recv().await.unwrap();
        let two = second.recv().await.unwrap();
        let mut ids = [
            one.request.request_id.clone(),
            two.request.request_id.clone(),
        ];
        ids.sort();
        assert_eq!(ids, ["r1".to_string(), "r2".to_string()]);

        match fleet.router.submit(make_request("r3")).await {
            DispatchOutcome::Rejected { reason } => assert!(reason.contains("free capacity")),
            outcome => panic!("both workers are busy, got {outcome:?}"),
        }

        drop(one);
        assert!(eventually(|| fleet.router.selector().available_permits() == 1).await);
        assert_eq!(
            fleet.router.submit(make_request("r3")).await,
            DispatchOutcome::Accepted
        );
    }

    #[tokio::test]
    async fn no_worker_rejects_at_once() {
        let router = PrefillRouter::new(selector(1));
        match router.submit(make_request("r1")).await {
            DispatchOutcome::Rejected { reason } => assert!(reason.contains("free capacity")),
            outcome => panic!("expected a rejection, got {outcome:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_worker_that_cannot_take_the_request_frees_its_charge_and_the_next_one_gets_it() {
        // Worker 0 has no dispatch handler, as a worker that is going away.
        let fleet = Fleet::new(1, 2).await;
        let mut alive = fleet.inbox(1);
        // Make the dead worker the first pick.
        fleet.router.selector().snapshot()[1]
            .counters
            .lock()
            .load_net_new = 1_000;

        assert_eq!(
            fleet.router.submit(make_request("r1")).await,
            DispatchOutcome::Accepted
        );
        assert_eq!(alive.recv().await.unwrap().request.request_id, "r1");
        let dead = &fleet.router.selector().snapshot()[0];
        assert_eq!(dead.instance_id, fleet.workers[0].instance_id());
        assert_eq!(dead.counters().inflight, 0);
        assert_eq!(dead.permits.available_permits(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_request_that_no_worker_takes_is_rejected() {
        let fleet = Fleet::new(1, 1).await;
        assert!(matches!(
            fleet.router.submit(make_request("r1")).await,
            DispatchOutcome::Rejected { .. }
        ));
        assert_eq!(fleet.router.selector().available_permits(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_session_runs_once() {
        let fleet = Fleet::new(2, 1).await;
        let mut inbox = fleet.inbox(0);
        let request = make_request("r1");
        assert_eq!(
            fleet.router.submit(request.clone()).await,
            DispatchOutcome::Accepted
        );
        assert!(matches!(
            fleet.router.submit(request).await,
            DispatchOutcome::Rejected { .. }
        ));
        let _held = inbox.recv().await.unwrap();
        assert!(inbox.try_recv().is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_worker_that_leaves_frees_its_charges() {
        let fleet = Fleet::new(1, 1).await;
        let mut inbox = fleet.inbox(0);
        fleet.router.submit(make_request("r1")).await;
        let _held = inbox.recv().await.unwrap();
        let slot = Arc::clone(&fleet.router.selector().snapshot()[0]);
        assert_eq!(slot.counters().inflight, 1);

        fleet.router.forget_worker(slot.instance_id);
        assert_eq!(slot.counters().inflight, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_done_notice_from_another_sender_is_ignored() {
        let fleet = Fleet::new(1, 1).await;
        let mut inbox = fleet.inbox(0);
        let request = make_request("r1");
        let session = request.session_id.to_string();
        fleet.router.submit(request).await;
        let _held = inbox.recv().await.unwrap();

        fleet
            .router
            .done(&session, fleet.hub.instance_id().worker_id());
        assert_eq!(fleet.router.selector().available_permits(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn two_decode_workers_may_use_the_same_request_id() {
        let fleet = Fleet::new(2, 1).await;
        let mut inbox = fleet.inbox(0);
        for _ in 0..2 {
            assert_eq!(
                fleet.router.submit(make_request("r1")).await,
                DispatchOutcome::Accepted
            );
        }
        let _first = inbox.recv().await.unwrap();
        let _second = inbox.recv().await.unwrap();
    }

    #[tokio::test]
    async fn a_backend_without_done_holds_the_charge_until_execute_returns() {
        let sel = selector(1);
        let release = Arc::new(Notify::new());
        sel.add_worker(
            InstanceId::new_v4(),
            Arc::new(TerminalBackend {
                id: InstanceId::new_v4(),
                release: Arc::clone(&release),
            }),
        );
        let router = PrefillRouter::new(Arc::clone(&sel));

        assert_eq!(
            router.submit(make_request("r1")).await,
            DispatchOutcome::Accepted
        );
        assert_eq!(sel.available_permits(), 0);
        release.notify_one();
        assert!(eventually(|| sel.available_permits() == 1).await);
    }
}
