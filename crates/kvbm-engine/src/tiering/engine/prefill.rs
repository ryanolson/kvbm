// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prefill-side conditional disaggregation: the inbound half of the CD split.
//!
//! A prefill worker receives a dispatched vLLM request whose
//! `kv_transfer_params` carry `RemotePrefillParams`; the `find_blocks` router's
//! prefill arm drives [`LocalConnectorEngine::prefill_accept_core`]. The engine
//! latches a per-request lifecycle, attaches to the decode-held session, drains
//! the decode-committed window into local G2, and at USAA
//! ([`LocalConnectorEngine::prefill_onboard_by_id`], driven by the
//! `onboard_blocks` router) copies the EXTERNAL suffix of the pulled window
//! into the vLLM-allocated G1 blocks, signaling completion through the same
//! load-action terminal the decode onboard paths use. Ported from the legacy
//! prefill coordinator
//! (`ensure_started` / `run_setup` / `on_usaa` / `kick_onboard` /
//! `cleanup_failed_request`).
//!
//! The prefill side never seals the decode side: it consumes
//! `commits()`/`availability()` and pulls; it does not call
//! `finish_commits`/`finish_availability` on its own planes — the session
//! stays parked on the lifecycle (open for the output direction) until the
//! RAII release finalizes it.
//!
//! The OUTPUT direction: as the worker computes the rest of the prompt, its
//! G1 blocks offload to G2 and the offload pipeline's register observer
//! (`remote::cd::output`) matches them against the lifecycle's expected
//! output set — the full PLH chain minus the provided window, tracked at
//! accept — and publishes them into the parked session for the decode drain.
//! Blocks already G2-resident at accept publish immediately (they never
//! re-register, so the observer alone would wait on them forever). The RAII
//! release defers the session finalize until that output has drained,
//! bounded by a watchdog.
//!
//! Every await after the session is parked is unblockable by session
//! teardown: the attach is followed by a claimed-check (a release that fires
//! before attach completes closes the session immediately after the store
//! AND resolves any kick USAA latched in the meantime — the release itself
//! found nothing parked and skipped the gate, so that action would otherwise
//! never reach its terminal), the commit/availability drains end on the
//! session close terminators, and
//! an in-flight `session.pull` fails via the pending-pull drain —
//! `VeloSession` guarantees the last two by construction and `MockSession`
//! models them. The two pre-attach awaits (`PeerResolver::resolve_and_register`
//! and `SessionFactory::attach`) have no teardown-driven unblock and must be
//! deadline-bounded by the production resolver/factory implementations.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use futures::StreamExt;

use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::TransferOptions;
use kvbm_protocols::connector::{
    AcceptId, ActionFailure, ActionId, ActionStatus, LeaderEngine, LeaderEngineError, OnboardHandle,
};
use kvbm_protocols::connector::{BlockId, RequestId, SequenceHash};
use kvbm_protocols::disagg::{RemotePrefillParams, digest_provided_hashes};

use super::driver::ActionRecord;
use super::inflight::InflightKey;
use super::local::LocalConnectorEngine;
use super::onboard;
use crate::p2p::session::{AvailabilityDelta, CommitDelta};
use crate::remote::cd::prefill::{ParkOutcome, PrefillKick, PrefillRequestState};

/// Engine-internal outcome of [`LocalConnectorEngine::prefill_accept_core`]:
/// the accept facts without a minted seam handle. The `find_blocks` router
/// mints a pure-RAII [`kvbm_protocols::connector::FindBlocksHandle`] over a
/// first latch and reads the external-token count off the outcome.
pub(super) enum PrefillAcceptCore {
    /// First latch of the lifecycle (or a fresh latch after an eviction
    /// released the prior generation). The engine-stored external-token cell is
    /// owned by the per-request state; the count rides `external_tokens`.
    Accepted {
        accept_id: AcceptId,
        external_tokens: usize,
    },
    /// Idempotent re-poll against the already-latched lifecycle; the stored
    /// cell was refreshed. Never a second latch.
    Refreshed { external_tokens: usize },
}

impl LocalConnectorEngine {
    /// The accept core. First call latches the lifecycle and spawns the pull
    /// pipeline; an idempotent re-poll recomputes
    /// `num_provided_tokens - num_computed_tokens` (saturating), stores it
    /// `Release` into the shared cell, and returns `Refreshed` — the pipeline
    /// is spawned only once and no second lifecycle is ever latched.
    ///
    /// Borrowed inputs by design: a re-poll returns before ANY copy is made
    /// (the window/output snapshots and the params clone are once-per-latch,
    /// never per-poll).
    pub(super) fn prefill_accept_core(
        self: Arc<Self>,
        request_id: &RequestId,
        params: &RemotePrefillParams,
        sequence_hashes: &[SequenceHash],
        num_computed_tokens: usize,
    ) -> Result<PrefillAcceptCore, LeaderEngineError> {
        let Some(cd) = self.cd.as_ref() else {
            // A dispatched prefill landing on a non-CD engine is a deployment
            // misconfiguration — refuse loudly instead of recomputing from
            // zero silently.
            return Err(LeaderEngineError::DisaggNotConfigured);
        };

        let external_tokens = params
            .num_provided_tokens
            .saturating_sub(num_computed_tokens);

        // A latch already holds for this rid. When the re-dispatch targets the
        // SAME session it is an idempotent re-poll: refresh the shared cell the
        // held handle reads (vLLM may re-poll with a different
        // `num_computed_tokens` between GNMT and USAA) and answer Refreshed.
        //
        // A DIFFERENT session means the decode side recompute-rescheduled this
        // rid onto a fresh session (its computed offset moved). The latched
        // lifecycle is abandoned — answering Refreshed against its dead session
        // would leave the fresh session unattached and wedge decode until its
        // watchdog. Tear the abandoned generation down through the same path
        // the RAII release drives, then fall through to latch + attach the
        // fresh one.
        if let Some(state) = cd.prefill.get(request_id) {
            if state.session_id() == params.session_id {
                state.store_external_tokens(external_tokens);
                return Ok(PrefillAcceptCore::Refreshed { external_tokens });
            }
            self.prefill_release(request_id, state.accept_id());
        }

        // The provided window `[0, DNPT/block_size)` — conservative over-pull:
        // the whole decode-committed window lands in local G2 (cache-warming),
        // while only the external suffix is copied to G1 at USAA.
        let dnpt_blocks = params.num_provided_tokens / self.block_size;
        if dnpt_blocks > sequence_hashes.len() {
            return Err(LeaderEngineError::InvalidPrefillRequest {
                reason: format!(
                    "provided window [0, {dnpt_blocks}) exceeds the request's hashed \
                     block count {}",
                    sequence_hashes.len()
                ),
            });
        }
        let expected_hashes: Vec<SequenceHash> = sequence_hashes[..dnpt_blocks].to_vec();

        // Defense-in-depth: when decode shipped a digest of its provided
        // slice, assert the locally-recomputed window matches bit-for-bit —
        // a mismatch means the two sides disagree on hashing inputs and the
        // RDMA pull would stall on unrelated keys.
        if let Some(expected_digest) = params.expected_hash_digest {
            let local_digest = digest_provided_hashes(&expected_hashes);
            if local_digest != expected_digest {
                return Err(LeaderEngineError::InvalidPrefillRequest {
                    reason: format!(
                        "hash digest divergence: decode=0x{expected_digest:016x} \
                         prefill=0x{local_digest:016x}"
                    ),
                });
            }
        }

        let accept_id = AcceptId::new();
        let cell = Arc::new(AtomicUsize::new(external_tokens));
        let state = Arc::new(PrefillRequestState::new(
            accept_id,
            expected_hashes,
            Arc::clone(&cell),
            params.session_id,
        ));
        if cd
            .prefill
            .insert(request_id.clone(), Arc::clone(&state))
            .is_err()
        {
            // Lost an insert race for the same rid — answer as the idempotent
            // re-poll against whoever won.
            if let Some(existing) = cd.prefill.get(request_id) {
                existing.store_external_tokens(external_tokens);
            }
            return Ok(PrefillAcceptCore::Refreshed { external_tokens });
        }

        // The OUTPUT direction: everything past the pulled window is owed back
        // to the decode peer — the full PLH chain minus the provided window.
        // Zero-external lifecycles still owe output. Track FIRST, then sweep
        // for already-in-G2 residents: a block registering concurrently
        // between the two is caught by the register observer — a duplicate
        // publish is benign (the decode drain dedups: commit accumulation
        // set, per-delta slot dedup, filled-set filter), but a MISSED block
        // wedges the decode side until its watchdog under-delivers.
        let expected_outputs: Vec<SequenceHash> = sequence_hashes[dnpt_blocks..].to_vec();
        cd.output.track(
            request_id.clone(),
            accept_id,
            expected_outputs.iter().copied().collect(),
            Arc::downgrade(&state),
        );
        let mut resident = self
            .leader
            .g2_manager()
            .scan_matches(&expected_outputs, true);
        if !resident.is_empty() {
            let mut found_hashes = Vec::with_capacity(resident.len());
            let mut found_blocks = Vec::with_capacity(resident.len());
            for hash in &expected_outputs {
                if let Some(block) = resident.remove(hash) {
                    found_hashes.push(*hash);
                    found_blocks.push(block);
                }
            }
            // The resident pins publish here (they will never re-register, so
            // the observer would wait on them forever); they buffer pre-park
            // and drain when the pipeline parks the attached session.
            cd.output
                .untrack_hashes(request_id, accept_id, &found_hashes);
            state.commit_output(found_blocks);
        }

        // Spawn the pull pipeline. It owns the ORIGINATING state Arc — every
        // terminal/teardown it fires binds to this lifecycle, never a map
        // re-fetch (the cross-lifecycle hazard). The params clone happens
        // here, once per latch.
        let task_engine = Arc::clone(&self);
        let task_rid = request_id.clone();
        let task_state = Arc::clone(&state);
        let task_params = params.clone();
        self.leader.runtime().spawn(async move {
            let result = Arc::clone(&task_engine)
                .run_prefill_pipeline(task_rid.clone(), task_params, Arc::clone(&task_state))
                .await;
            if let Err(e) = result {
                task_engine.prefill_pipeline_failed(
                    &task_rid,
                    &task_state,
                    format!("prefill pipeline failed: {e:#}"),
                );
            }
        });

        Ok(PrefillAcceptCore::Accepted {
            accept_id,
            external_tokens,
        })
    }

    /// The USAA intercept core, keyed by `(request_id, accept_id)` (the unified
    /// `onboard_blocks` router routes here off its opaque handle). Slices the
    /// external G1 SUFFIX from the full vLLM allocation, mints the load action,
    /// and runs the two-phase kick handshake: latch the kick THEN check
    /// `pulls_complete` (the pipeline sets `pulls_complete` THEN takes the
    /// latch — either ordering fires exactly one kick). A stashed pre-USAA
    /// pipeline failure replays here instead: the minted action resolves
    /// immediately Failed over the EXTERNAL slice only (never the
    /// locally-computed prefix — reporting it would force a recompute from
    /// token zero) and the state is removed.
    ///
    /// ONE onboard per accept generation: a re-entrant call against a live
    /// generation is refused with
    /// [`LeaderEngineError::OnboardAlreadyInFlight`]. Without the refusal a
    /// second onboard could never reach a terminal — pre-pulls-complete it
    /// would replace the parked kick (the first action forever `Pending`);
    /// post-complete its fired kick would lose the `claim_kick` CAS without
    /// `finish_load_action` (the second action forever `Pending`). The
    /// refusal is also what makes the deferral-guard record below
    /// once-per-generation by construction.
    pub(super) fn prefill_onboard_by_id(
        self: Arc<Self>,
        request_id: &RequestId,
        accept_id: AcceptId,
        block_ids: &[BlockId],
    ) -> Result<OnboardHandle, LeaderEngineError> {
        let Some(cd) = self.cd.as_ref() else {
            return Err(LeaderEngineError::DisaggNotConfigured);
        };
        let request_id = request_id.clone();
        let Some(state) = cd.prefill.get(&request_id) else {
            return Err(LeaderEngineError::PrefillSessionStale);
        };
        if state.accept_id() != accept_id {
            // The map holds a FRESH generation (evict + re-accept of the same
            // rid); the stale handle must not onboard against it.
            return Err(LeaderEngineError::PrefillSessionStale);
        }
        // vLLM lays out `block_ids` as `[locally-computed prefix | external]`.
        // The external width comes from the engine-stored cell (the same
        // value the handle reports); the correct slice is the SUFFIX —
        // a prefix slice would overwrite vLLM's already-computed blocks AND
        // miss the last external block. Validated BEFORE the one-onboard
        // claim: a refused caller must be able to retry with corrected ids —
        // consuming the claim first would wedge the generation permanently.
        let external_blocks = state.external_tokens() / self.block_size;
        if block_ids.len() < external_blocks {
            return Err(LeaderEngineError::InvalidPrefillRequest {
                reason: format!(
                    "block_ids len {} < external blocks {external_blocks}",
                    block_ids.len()
                ),
            });
        }
        let external_g1: Vec<BlockId> = block_ids[block_ids.len() - external_blocks..].to_vec();

        if !state.claim_usaa() {
            return Err(LeaderEngineError::OnboardAlreadyInFlight);
        }

        // Mint the action BEFORE consulting the gate so the replay arm can
        // resolve this same action (the gate keeps the failure-stash check and
        // the kick latch atomic — a pipeline failure can never slip between
        // them). Register in `actions`/`by_request` before the handle returns
        // so eviction fences and drain emissions gate this load.
        let action_id = ActionId::new();
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        self.actions.insert(
            action_id,
            ActionRecord::new(request_id.clone(), Arc::downgrade(&cell)),
        );
        self.by_request
            .entry(request_id.clone())
            .or_default()
            .push(action_id);

        // Record the in-flight onboard's hashes for the deferral guard — the
        // EXTERNAL suffix of the decode-provided window (`expected_hashes`),
        // exactly the `external_blocks` hashes the kick copies into the G1
        // suffix. Keyed by the accept generation: the clear lands at the
        // lifecycle release (`prefill_release`), which fires even on the
        // failure paths that remove the engine-side state before the
        // connector's handle drops (pre-USAA replay, pipeline failure, kick
        // failure).
        let expected = state.expected_hashes();
        let suffix_start = expected.len().saturating_sub(external_blocks);
        let guard_window = expected[suffix_start..].to_vec();
        self.inflight
            .lock()
            .expect("inflight-guard mutex poisoned")
            .record(InflightKey::Prefill(accept_id), guard_window);

        let latch = state.latch_kick(PrefillKick {
            action_id,
            external_g1: external_g1.clone(),
        });

        if let Err(reason) = latch {
            // Pre-USAA failure replay: the immediately-Failed action covers
            // the external slice only; the state was kept latched for exactly
            // this replay and is removed here (the failure path already
            // closed the session — the take below is a defensive no-op).
            tracing::warn!(%request_id, %reason, "prefill onboard: replaying pre-USAA failure stash");
            self.finish_load_action(
                action_id,
                &request_id,
                ActionStatus::Failed(ActionFailure::Partial {
                    block_ids: external_g1.clone(),
                }),
                external_g1.clone(),
            );
            if let Some(session) = state.take_session() {
                session.close(Some(reason));
            }
            // A fall-through binding acquired before the external count
            // flipped positive would otherwise strand its `searches` entry.
            if let Some(sid) = state.take_local_search() {
                self.release_search(&sid);
            }
            cd.output.untrack(&request_id, state.accept_id());
            cd.prefill.release_if_matches(&request_id, &state);
            let me: Arc<dyn LeaderEngine> = self;
            return Ok(OnboardHandle::new(
                action_id,
                Arc::downgrade(&me),
                cell,
                external_g1,
            ));
        }

        // Two-phase handshake, USAA side: the kick is latched; if the pipeline
        // already finished, fire it now (the take is the exactly-once point;
        // the kick's own CAS backstops).
        if state.pulls_complete()
            && let Some(kick) = state.take_pending_kick()
        {
            let kick_engine = Arc::clone(&self);
            let kick_state = Arc::clone(&state);
            let kick_rid = request_id.clone();
            self.leader.runtime().spawn(async move {
                kick_engine
                    .run_prefill_kick(kick_rid, kick_state, kick)
                    .await;
            });
        }

        let me: Arc<dyn LeaderEngine> = self;
        Ok(OnboardHandle::new(
            action_id,
            Arc::downgrade(&me),
            cell,
            external_g1,
        ))
    }

    /// `release_prefill_session` body — the RAII teardown target. Generation-
    /// guarded: an `accept_id` mismatch means the map holds a FRESH lifecycle
    /// a stale handle drop must not touch. The cleanup-claim CAS winner owns
    /// the session decision — close while the pipeline is incomplete (this
    /// unblocks every pipeline await) or failed; for a completed unfailed
    /// lifecycle the finalize is DEFERRED until the output direction drains
    /// (the prefill may still be offloading the computed suffix the decode
    /// peer is waiting on). The drain task owns the ORIGINATING state `Arc`
    /// for its whole life — the map entry is removed inline FIRST so a
    /// re-accept of the same rid cannot collide with the accept latch while
    /// the drain still polls; generation-scoped observer state keeps the old
    /// drain and a fresh same-rid lifecycle independent. On the close path,
    /// state removal stays outside the claim: idempotent and ptr-guarded, so
    /// a pre-USAA-failure-retained state still frees when the request ends.
    ///
    /// The finalize-vs-close decision is made on the state visible at release
    /// time: a release racing a kick whose G2→G1 transfer is still in flight
    /// (reachable only through an evict of a request parked in
    /// WAITING_FOR_REMOTE_KVS) reads `pulls_complete && !failed` and
    /// finalizes; if that transfer then fails, the peer has already seen a
    /// cooperative `Finished`. The vLLM-facing terminal is still `Failed`
    /// over the external slice and the eviction fence holds the action, so
    /// the divergence is peer-signaling only.
    pub(super) fn prefill_release(&self, request_id: &RequestId, accept_id: AcceptId) {
        // Clear this generation's in-flight onboard deferral record FIRST,
        // before any map/generation guard: the failure terminals (pre-USAA
        // replay, pipeline failure, kick failure) remove the engine-side state
        // while the connector still parks the handle, and the eventual handle
        // drop landing here must still clear. Idempotent; keyed by the
        // releasing handle's OWN generation, so a stale release never touches
        // a fresh re-latch's entry.
        self.inflight
            .lock()
            .expect("inflight-guard mutex poisoned")
            .clear(&InflightKey::Prefill(accept_id));
        let Some(cd) = self.cd.as_ref() else {
            return;
        };
        let Some(state) = cd.prefill.get(request_id) else {
            return;
        };
        if state.accept_id() != accept_id {
            return;
        }
        // One teardown for both lifecycles: the zero-external fall-through
        // binds a local search INSIDE this generation; release it with the
        // generation. Take-once, so a racing teardown cannot double-release;
        // a binding already consumed by the zero-stored onboard delegation has
        // no `searches` entry left and the release no-ops.
        if let Some(sid) = state.take_local_search() {
            self.release_search(&sid);
        }
        if state.claim_cleanup() {
            if state.pulls_complete() && state.pending_failure().is_none() {
                cd.prefill.release_if_matches(request_id, &state);
                let observer = Arc::clone(&cd.output);
                let poll = cd.output_drain_poll();
                let watchdog = cd.output_drain_watchdog();
                let rid = request_id.clone();
                self.leader.runtime().spawn(async move {
                    let deadline = tokio::time::Instant::now() + watchdog;
                    while observer.has_pending(&rid, accept_id) {
                        if tokio::time::Instant::now() >= deadline {
                            tracing::warn!(
                                request_id = %rid,
                                residual = observer.residual(&rid, accept_id),
                                "prefill release: output residual not drained within \
                                 watchdog; forcing session finalize"
                            );
                            break;
                        }
                        tokio::time::sleep(poll).await;
                    }
                    // Untrack BEFORE the finalize becomes observable: a block
                    // registering now must not match the abandoned residual
                    // and dispatch into the session mid-finalize.
                    observer.untrack(&rid, accept_id);
                    if let Some(session) = state.take_session() {
                        session.finalize(None);
                    }
                });
                return;
            }
            if let Some(session) = state.take_session() {
                session.close(Some("prefill request released".to_string()));
            }
        }
        cd.output.untrack(request_id, accept_id);
        cd.prefill.release_if_matches(request_id, &state);
    }

    /// The pull pipeline: optional peer resolve, attach, drain the decode's
    /// commit stream against the expected window, drain availability pulling
    /// maximal contiguous runs into registered G2, then run the pipeline half
    /// of the kick handshake. Owns the ORIGINATING state Arc for its whole
    /// life; `Err` routes to [`Self::prefill_pipeline_failed`].
    async fn run_prefill_pipeline(
        self: Arc<Self>,
        request_id: RequestId,
        params: RemotePrefillParams,
        state: Arc<PrefillRequestState>,
    ) -> Result<()> {
        let cd = self
            .cd
            .as_ref()
            .expect("prefill pipeline spawns only on a CD-configured engine");

        let endpoint = params.decode_endpoint.clone().ok_or_else(|| {
            anyhow!("RemotePrefillParams.decode_endpoint missing for {request_id}")
        })?;

        if let Some(resolver) = &cd.peer_resolver {
            resolver
                .resolve_and_register(params.initiator_instance_id)
                .await
                .map_err(|e| {
                    anyhow!(
                        "resolve+register decode peer {}: {e:#}",
                        params.initiator_instance_id
                    )
                })?;
        }

        let session = cd
            .sessions
            .attach(params.session_id, params.initiator_instance_id, endpoint)
            .await?;

        // Park the session unless a release already claimed this lifecycle
        // while the attach was in flight — its `take_session` found nothing,
        // so this task owns the close. The park also drains any output the
        // observer buffered pre-attach; a failed drain fails the pipeline
        // (the session IS parked, so the failure path closes it).
        match state.park_session(Arc::clone(&session)) {
            ParkOutcome::Parked => {}
            ParkOutcome::Refused => {
                session.close(Some(
                    "prefill request released before attach completed".to_string(),
                ));
                // A kick latched before that release has no other owner left:
                // the release skipped the gate (nothing was parked to close)
                // and this pipeline exits cleanly, so neither the failure path
                // nor the pipeline tail will ever consume it. Resolve its
                // action through the load terminal here — Failed over the
                // external slice, same shape as the failure paths — so the
                // worker's wait and any eviction fence riding the action see
                // the terminal.
                if let Some(kick) = state.take_pending_kick() {
                    tracing::warn!(
                        %request_id,
                        "prefill release won the pre-attach race; failing the latched kick"
                    );
                    self.finish_load_action(
                        kick.action_id,
                        &request_id,
                        ActionStatus::Failed(ActionFailure::Partial {
                            block_ids: kick.external_g1.clone(),
                        }),
                        kick.external_g1,
                    );
                }
                return Ok(());
            }
            ParkOutcome::PublishFailed(e) => return Err(e),
        }

        let expected = state.expected_hashes().to_vec();
        let expected_count = expected.len();
        if expected_count > 0 {
            let expected_set: HashSet<SequenceHash> = expected.iter().copied().collect();

            // 1. Drain commits until every expected window hash is committed.
            //    A hash outside the window means the two sides disagree on the
            //    provided slice; a `Closed` before the full count is
            //    under-delivery. Both fail the pipeline rather than hang.
            let mut seen: HashSet<SequenceHash> = HashSet::new();
            let mut commits = session.commits();
            while let Some(delta) = commits.next().await {
                match delta {
                    CommitDelta::Added(hashes) => {
                        for h in hashes {
                            if !expected_set.contains(&h) {
                                bail!(
                                    "prefill pipeline: committed hash {h:?} is not in the \
                                     expected provided window"
                                );
                            }
                            seen.insert(h);
                        }
                        if seen.len() >= expected_count {
                            break;
                        }
                    }
                    CommitDelta::Closed => {
                        if seen.len() < expected_count {
                            bail!(
                                "prefill pipeline: commits closed before all expected hashes \
                                 arrived (got {} of {expected_count})",
                                seen.len()
                            );
                        }
                        break;
                    }
                }
            }
            drop(commits);

            // 2. Drain availability, pulling maximal contiguous runs (by
            //    absolute expected-hash index) into registered G2. Arrival
            //    order is not positional order — sparse/coalesced deltas are
            //    valid — so each delta is filtered, indexed, sorted, and
            //    regrouped before pulling.
            let slot_of: HashMap<SequenceHash, usize> =
                expected.iter().enumerate().map(|(i, h)| (*h, i)).collect();
            let mut filled: HashSet<SequenceHash> = HashSet::new();
            let mut avail = session.availability();
            while let Some(delta) = avail.next().await {
                match delta {
                    AvailabilityDelta::Available(blocks) => {
                        for b in &blocks {
                            if !slot_of.contains_key(&b.hash) {
                                bail!(
                                    "prefill pipeline: availability carried hash {:?} not in \
                                     the expected provided window",
                                    b.hash
                                );
                            }
                        }
                        let mut indexed: Vec<(usize, SequenceHash)> = blocks
                            .into_iter()
                            .filter(|b| !filled.contains(&b.hash))
                            .map(|b| (slot_of[&b.hash], b.hash))
                            .collect();
                        if indexed.is_empty() {
                            continue;
                        }
                        indexed.sort_by_key(|(slot, _)| *slot);
                        // Within-delta duplicates (a peer double-publish, or
                        // the replay coalescer merging pre-subscribe deltas)
                        // would land the same slot twice — the `filled`
                        // filter only dedups across deltas. See the decode
                        // drain in `onboard::run_remote_onboard`.
                        indexed.dedup_by_key(|(slot, _)| *slot);

                        for run in onboard::group_contiguous_runs(indexed) {
                            let hashes: Vec<SequenceHash> = run.iter().map(|(_, h)| *h).collect();
                            let registered = onboard::pull_run_into_g2(
                                &self.leader,
                                &session,
                                hashes,
                                self.block_size,
                            )
                            .await?;
                            for ((slot, hash), block) in run.iter().zip(registered) {
                                state.push_registered(*slot, block);
                                filled.insert(*hash);
                            }
                        }
                        if filled.len() == expected_count {
                            break;
                        }
                    }
                    AvailabilityDelta::Drained => break,
                }
            }
            drop(avail);

            if filled.len() != expected_count {
                bail!(
                    "prefill pipeline: availability drained with {} of {expected_count} \
                     window hashes filled",
                    filled.len()
                );
            }
        }

        // Two-phase handshake, pipeline side: publish completion FIRST, then
        // consume any kick USAA already latched. The USAA path latches first
        // and loads `pulls_complete` second, so whichever interleaving the two
        // race into, at least one side fires the kick — and the take (plus the
        // kick's CAS) keeps it to exactly one.
        state.mark_pulls_complete();
        if let Some(kick) = state.take_pending_kick() {
            Arc::clone(&self)
                .run_prefill_kick(request_id, Arc::clone(&state), kick)
                .await;
        }
        Ok(())
    }

    /// Pipeline failure terminal. Post-USAA (a kick was latched): resolve the
    /// latched load action Failed over the external slice, close the session,
    /// remove the state ptr-guarded. Pre-USAA: stash the reason and close the
    /// session, KEEPING the state latched so `prefill_onboard_by_id` can replay
    /// the failure against the real G1 ids — never an empty failed set, which
    /// vLLM would treat as a successful async load.
    fn prefill_pipeline_failed(
        self: &Arc<Self>,
        request_id: &RequestId,
        state: &Arc<PrefillRequestState>,
        reason: String,
    ) {
        tracing::error!(%request_id, %reason, "prefill pipeline failed");
        match state.fail_pipeline(reason.clone()) {
            Some(kick) => {
                // The terminal always fires — the worker is waiting on this
                // action regardless of who started the teardown.
                self.finish_load_action(
                    kick.action_id,
                    request_id,
                    ActionStatus::Failed(ActionFailure::Partial {
                        block_ids: kick.external_g1.clone(),
                    }),
                    kick.external_g1,
                );
                if state.claim_cleanup()
                    && let Some(session) = state.take_session()
                {
                    session.close(Some(reason));
                }
                if let Some(sid) = state.take_local_search() {
                    self.release_search(&sid);
                }
                if let Some(cd) = self.cd.as_ref() {
                    cd.output.untrack(request_id, state.accept_id());
                    cd.prefill.release_if_matches(request_id, state);
                }
            }
            None => {
                // Pre-USAA: `fail_pipeline` stashed the reason. The cleanup
                // claim serializes against a concurrent release; the loser
                // returns and leaves the session to the winner.
                if state.claim_cleanup()
                    && let Some(session) = state.take_session()
                {
                    session.close(Some(reason));
                }
            }
        }
    }

    /// The USAA kick: sort the registered G2 pins by absolute position, take
    /// the external-count SUFFIX, and copy it G2→G1 onto the external G1
    /// slice with positional pairing. Complete leaves the session parked and
    /// the state latched (the output direction stays open until release);
    /// failure resolves the action Failed over the external slice, closes the
    /// session, and removes the state ptr-guarded.
    async fn run_prefill_kick(
        self: Arc<Self>,
        request_id: RequestId,
        state: Arc<PrefillRequestState>,
        kick: PrefillKick,
    ) {
        if !state.claim_kick() {
            return;
        }
        let PrefillKick {
            action_id,
            external_g1,
        } = kick;
        let external_blocks = external_g1.len();
        if external_blocks == 0 {
            // Nothing external to move (zero-external CD) — immediately
            // terminal, no transfer.
            self.finish_load_action(action_id, &request_id, ActionStatus::Complete, external_g1);
            return;
        }

        let outcome = match state.registered_suffix(external_blocks) {
            Some(suffix) => {
                let g2_block_ids: Vec<BlockId> = suffix.iter().map(|b| b.block_id()).collect();
                // The cloned pins stay alive across the copy (their twins are
                // also parked on the state, but the copy must not depend on
                // that parking outliving it).
                let _hold = suffix;
                match self.leader.execute_local_transfer(
                    LogicalLayoutHandle::G2,
                    LogicalLayoutHandle::G1,
                    g2_block_ids,
                    external_g1.clone(),
                    TransferOptions::default(),
                ) {
                    Ok(notification) => match notification.await {
                        Ok(()) => ActionStatus::Complete,
                        Err(e) => {
                            tracing::error!(error = %e, %request_id, "prefill kick: G2->G1 transfer failed");
                            ActionStatus::Failed(ActionFailure::AllBlocks)
                        }
                    },
                    Err(e) => {
                        tracing::error!(error = %e, %request_id, "prefill kick: G2->G1 dispatch failed");
                        ActionStatus::Failed(ActionFailure::AllBlocks)
                    }
                }
            }
            None => {
                tracing::error!(
                    %request_id,
                    external_blocks,
                    "prefill kick: fewer registered G2 blocks than the external slice"
                );
                ActionStatus::Failed(ActionFailure::AllBlocks)
            }
        };

        let failed = !matches!(outcome, ActionStatus::Complete);
        if failed {
            // The stash must land BEFORE the terminal becomes observable: the
            // terminal is what lets the connector reap the slot and drop the
            // session handle, and that RAII release decides finalize-vs-close
            // by reading the stash — stashing after the terminal admits a
            // window where the failed lifecycle is finalized (a clean
            // cooperative seal) to the decode peer.
            state.stash_failure("prefill onboard kick failed".to_string());
        }
        // `finish_load_action` resolves a total failure to these concrete
        // external dest ids before the cell write and sink notify.
        self.finish_load_action(action_id, &request_id, outcome, external_g1);

        if failed {
            if state.claim_cleanup()
                && let Some(session) = state.take_session()
            {
                session.close(Some("prefill onboard kick failed".to_string()));
            }
            if let Some(sid) = state.take_local_search() {
                self.release_search(&sid);
            }
            if let Some(cd) = self.cd.as_ref() {
                cd.output.untrack(&request_id, state.accept_id());
                cd.prefill.release_if_matches(&request_id, &state);
            }
        }
        // On Complete: the session stays parked and the state latched — the
        // computed-suffix output direction publishes over this same session
        // later; the RAII release owns the finalize.
    }
}
