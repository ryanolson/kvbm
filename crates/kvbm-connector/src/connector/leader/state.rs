// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private slot-lifecycle state for [`super::Leader`].
//!
//! Owns the slot map plus the lifecycle operations (gnmt, allocate,
//! request_finished, on_evicted, build_connector_meta, update_connector_output).
//! Pure logic, single-owner; the wrapping `Leader` puts it behind a `Mutex` so
//! the public API takes `&self`.
//!
//! GNMT and USAA are pure ADAPTERS over the engine's unified seam: `gnmt`
//! translates slot facts into a `FindBlocksRequest`, calls
//! `LeaderEngine::find_blocks`, and maps the outcome onto vLLM's
//! `(Option<usize>, bool)` contract; `allocate` guards the slot invariants and
//! calls `LeaderEngine::onboard_blocks`. All match routing — window derivation,
//! fresh-vs-refresh, prefill-vs-local, the in-flight onboard deferral guard,
//! external-token validation — lives engine-side.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use kvbm_common::{BlockId, SequenceHash};
use kvbm_protocols::connector::{
    EvictionFence, FindBlocksOutcome, FindBlocksRequest, FinishedStatus as EngineFinishedStatus,
    LeaderEngine, OnboardHandle,
};
use prometheus::IntCounter;

use crate::common::{
    CachedRequestData, FinishedStatus as ApiFinishedStatus, KvConnectorMetadata, Request,
    SchedulerOutput,
};

use super::super::metadata::ConnectorMetadata;
use super::Error;
use super::slot::RequestSlot;

/// Result of the per-step scheduler-output walk.
pub(crate) struct WalkOutcome {
    /// The wire plan for this step (intra-pass fields `None`; completion
    /// events embedded by the wrapping `Leader` when the flush glue is armed).
    pub(crate) metadata: KvConnectorMetadata,
    /// Whether any offload action was handed to the engine this step — the
    /// signal that the leader must arm the forward-pass flush trigger.
    pub(crate) scheduled_offloads: bool,
}

pub(crate) struct LeaderState {
    slots: HashMap<String, RequestSlot>,
    engine: Arc<dyn LeaderEngine>,
    /// Token-block granularity: tokens per G1 block.
    block_size: usize,
    /// Requests evicted this metadata cycle — dedups the wire's
    /// `evicted_requests` list to one entry per request per step. NOT the
    /// fence gate: a fence is minted on the first eviction of the cycle and
    /// again on any re-eviction that finds live visible handles (see
    /// [`Self::on_evicted`]).
    pending_evictions: HashSet<String>,
    /// Eviction fences minted this metadata cycle, drained into the per-step
    /// payload by [`Self::build_metadata`]. A request may contribute more
    /// than one fence in a cycle (same-cycle restore + re-evict); tokens are
    /// per-(generation, worker) UUIDs, so the worker awaits the union.
    pending_fences: Vec<EvictionFence>,
    /// Requests that finished with work still in flight (the
    /// [`Self::request_finished`] `Pending` decision). The emission was handed
    /// to the engine in-call (drain commit); this worklist only drives the
    /// handle-gated reap in [`Self::update_connector_output`].
    finishing: HashSet<String>,
    iteration: u64,
    /// Prometheus counter for GNMT-matched tokens. `None` in unit-test
    /// contexts that construct `LeaderState` without a live runtime (the
    /// increment is skipped). Incremented by the engine-reported
    /// token-granular `matched_tokens` exactly once per minted lifecycle
    /// (re-polls never double-count; the once-flag lives on the slot). Counts
    /// dispatched-prefill external tokens too — see REFACTOR.md, "CD seam
    /// correction".
    matched_tokens: Option<IntCounter>,
}

impl LeaderState {
    pub(crate) fn new(
        engine: Arc<dyn LeaderEngine>,
        block_size: usize,
        matched_tokens: Option<IntCounter>,
    ) -> Self {
        Self {
            slots: HashMap::new(),
            engine,
            block_size,
            pending_evictions: HashSet::new(),
            pending_fences: Vec::new(),
            finishing: HashSet::new(),
            iteration: 0,
            matched_tokens,
        }
    }

    /// Whether any slot holds an engine-minted handle (`proposal`/`onboard`/
    /// `offload`, or a drain-holder). [`super::Leader::install_engine`] refuses
    /// to swap the engine while any exists: handles carry a `Weak` to the engine
    /// that minted them, so swapping would orphan their RAII release.
    ///
    /// Handle-FREE slots are fine — a benign pre-init `NoopEngine` GNMT
    /// (REFACTOR.md:525, zero `Resolved` → no handle) or a `create_slot` leaves
    /// the slot handle-free, so install stays recoverable on that documented
    /// path. Since the placeholder never mints, this is always satisfied at the
    /// legitimate install point and only trips a genuine swap-under-live-handles.
    pub(crate) fn has_live_handles(&self) -> bool {
        self.slots.values().any(|s| {
            s.proposal.is_some()
                || s.onboard.is_some()
                || !s.offloads.is_empty()
                || !s.drain_holders.is_empty()
        })
    }

    /// Swap in the real engine built by deferred construction, replacing the
    /// placeholder the binding constructors start on. The caller
    /// ([`super::Leader::install_engine`]) enforces both single-shot and the
    /// no-live-handles precondition, so this swap orphans no in-flight handle.
    pub(crate) fn install_engine(&mut self, engine: Arc<dyn LeaderEngine>) {
        self.engine = engine;
    }

    pub(crate) fn contains(&self, request_id: &str) -> bool {
        self.slots.contains_key(request_id)
    }

    /// Borrow a slot. Production consumer: the CD prefill plane's dispatch
    /// snapshot (token ids + PLH chain for the wire request); tests read
    /// slot internals through it as well.
    pub(crate) fn get(&self, request_id: &str) -> Option<&RequestSlot> {
        self.slots.get(request_id)
    }

    #[cfg(test)]
    pub(crate) fn get_mut(&mut self, request_id: &str) -> Option<&mut RequestSlot> {
        self.slots.get_mut(request_id)
    }

    /// The full GNMT entry — TRANSLATION ONLY. Builds a [`FindBlocksRequest`]
    /// from slot facts (the slot's cached hash-chain `Arc` — a re-poll bumps
    /// a refcount, never copies hashes — this poll's computed count, total
    /// tokens, and the slot's parsed transfer params passed through whole),
    /// hands the engine the parked lifecycle handle so it can route
    /// fresh-vs-refresh, and maps the outcome onto vLLM's
    /// `(matched_tokens, load_async)` contract:
    ///
    /// - `Deferred` / `Searching` → `(None, false)` (vLLM parks the request
    ///   and re-polls next step); a `Searching` mint parks on the slot;
    /// - `Resolved { matched_tokens: 0 }` → `(Some(0), false)`;
    /// - `Resolved { matched_tokens: n }` → `(Some(n), true)` — the async
    ///   external load vLLM commits to at USAA time
    ///   (`WAITING_FOR_REMOTE_KVS`).
    ///
    /// A MISSING slot is a runtime logic error ([`Error::SlotNotFound`]):
    /// GNMT is the connector's first touch of a request and the binding
    /// layer creates the slot from the vLLM Request at the top of every
    /// `get_num_new_matched_tokens` call, BEFORE this runs — a poll with no
    /// slot means that ordering broke, and answering it would hand vLLM a
    /// zero derived from a request whose tokens were never seen.
    ///
    /// A `Resolved` mint parks; `release_parked` drops the parked handle,
    /// Issue-A gated on the in-flight onboard (the load still reads the
    /// engine-pinned source, so the pin may only drop once the onboard is
    /// terminal). No window math and no source routing happen here — the
    /// engine owns both.
    pub(crate) fn gnmt(
        &mut self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool), Error> {
        let Some(slot) = self.slots.get_mut(request_id) else {
            return Err(Error::SlotNotFound(request_id.to_string()));
        };
        let request = FindBlocksRequest {
            request_id: request_id.to_string(),
            sequence_hashes: slot.sequence_hashes(),
            num_computed_tokens,
            total_tokens: slot.total_tokens(),
            transfer_params: slot.transfer_params.clone(),
        };
        let outcome = Arc::clone(&self.engine)
            .find_blocks(&request, slot.proposal.as_ref())
            .map_err(|source| Error::FindBlocksRejected {
                request_id: request_id.to_string(),
                source,
            })?;

        match outcome {
            FindBlocksOutcome::Deferred => Ok((None, false)),
            FindBlocksOutcome::Searching { minted } => {
                if let Some(handle) = minted {
                    slot.proposal = Some(handle);
                    // Fresh lifecycle: re-arm the once-flag so its eventual
                    // hit is counted exactly once.
                    slot.reset_matched_tokens_reported();
                }
                Ok((None, false))
            }
            FindBlocksOutcome::Resolved {
                matched_tokens,
                minted,
                release_parked,
            } => {
                if let Some(handle) = minted {
                    slot.proposal = Some(handle);
                    slot.reset_matched_tokens_reported();
                }
                // Engine instruction: the parked lifecycle has no further use
                // (zero-refine / Lost / empty window). Issue-A gate (same as
                // the recv-side release): the handle may only drop once no
                // in-flight onboard reads its pinned source. A pending load
                // keeps it parked; `finished_recving` or the reap frees it at
                // the load's terminal.
                if release_parked && slot.onboard.as_ref().is_none_or(OnboardHandle::is_complete) {
                    slot.proposal = None;
                }
                if matched_tokens == 0 {
                    Ok((Some(0), false))
                } else {
                    let first_report = slot.mark_matched_tokens_reported();
                    if first_report && let Some(counter) = &self.matched_tokens {
                        counter.inc_by(matched_tokens as u64);
                    }
                    Ok((Some(matched_tokens), true))
                }
            }
        }
    }

    /// Register a slot from a vLLM `create_slot` Request.
    pub(crate) fn create_slot(&mut self, request: Request) -> Result<(), Error> {
        if self.slots.contains_key(&request.request_id) {
            return Err(Error::SlotAlreadyExists(request.request_id.clone()));
        }
        let request_id = request.request_id.clone();
        self.slots.insert(
            request_id,
            RequestSlot::from_request(request, self.block_size as u32),
        );
        Ok(())
    }

    /// USAA core. Hands vLLM's G1 block ids to the slot and, when vLLM commits
    /// to the async external load it was promised at GNMT time
    /// (`num_external_tokens > 0`), issues the engine onboard in-call through
    /// the ONE unified verb: `dest` is the FULL allocated list — the engine
    /// routes off the parked lifecycle's kind, slices the external window
    /// itself, and validates the committed count against its stored promise.
    /// The minted handle parks on the slot; its terminal surfaces worker-side
    /// as `finished_recving` and is released by
    /// [`Self::update_connector_output`].
    pub(crate) fn allocate(
        &mut self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<(), Error> {
        let slot = self
            .slots
            .get_mut(request_id)
            .ok_or_else(|| Error::SlotNotFound(request_id.to_string()))?;
        slot.set_block_ids(block_ids);
        if num_external_tokens == 0 {
            return Ok(());
        }
        // Per-slot double-park guard. Runtime guard, not a debug_assert:
        // re-entering onboard would replace `slot.onboard` and detach the
        // leader's view of the live load. (The engine separately refuses
        // re-entry per latched generation; this guards the slot's own view.)
        if slot.onboard.is_some() {
            return Err(Error::OnboardAlreadyInFlight(request_id.to_string()));
        }
        // vLLM only commits external tokens for a request it was promised a
        // hit on, so a missing parked lifecycle is a broken GNMT↔USAA
        // contract — fail loud rather than leave the request parked in
        // WAITING_FOR_REMOTE_KVS waiting on a load nobody started.
        let Some(proposal) = slot.proposal.as_ref() else {
            return Err(Error::ExternalLoadWithoutSearch(request_id.to_string()));
        };
        let onboard = Arc::clone(&self.engine)
            .onboard_blocks(proposal, &slot.block_ids, num_external_tokens)
            .map_err(|source| Error::OnboardRejected {
                request_id: request_id.to_string(),
                source,
            })?;
        slot.onboard = Some(onboard);
        Ok(())
    }

    pub(crate) fn extend_tokens(
        &mut self,
        request_id: &str,
        tokens: Vec<u32>,
    ) -> Result<(), Error> {
        let slot = self
            .slots
            .get_mut(request_id)
            .ok_or_else(|| Error::SlotNotFound(request_id.to_string()))?;
        slot.extend_tokens(tokens).map_err(|e| Error::TokenExtend {
            request_id: request_id.to_string(),
            message: e.to_string(),
        })
    }

    pub(crate) fn total_tokens(&self, request_id: &str) -> Result<usize, Error> {
        self.slots
            .get(request_id)
            .map(|s| s.total_tokens())
            .ok_or_else(|| Error::SlotNotFound(request_id.to_string()))
    }

    /// vLLM's `request_finished` (terminal). Inspects the slot's in-flight
    /// handles and maps to the 3-variant API status per REFACTOR.md §4 (D
    /// refinement) — no blocking or remote engine call (`is_complete()` is a
    /// local cell read; the drain take/commit are local map ops):
    ///
    /// - no slot ⇒ `UntrackedRequest` (benign: vLLM finishes WAITING-aborted
    ///   requests that never got a slot);
    /// - slot with no in-flight onboard/offload ⇒ reap **inline** and report
    ///   `Finished`;
    /// - slot with an in-flight onboard or offload ⇒ hand the engine the
    ///   terminal coordination IN-CALL — `take_offload_drain` + `commit` (commit
    ///   ARMS the engine's emit-on-last-terminal; the engine fires the single
    ///   `finished_sending` when the last pending-at-commit action drains,
    ///   immediately if none are) — keep the slot, report `Pending`. The
    ///   handle-gated reap is deferred to [`Self::update_connector_output`].
    ///   A request with only an in-flight onboard and no registered offloads
    ///   has no drain; its load terminal surfaces via `finished_recving`, which
    ///   vLLM also frees a finished request's blocks on.
    pub(crate) fn request_finished(&mut self, request_id: &str) -> ApiFinishedStatus {
        let Some(slot) = self.slots.get(request_id) else {
            return ApiFinishedStatus::UntrackedRequest;
        };
        // Do NOT null `slot.proposal` here: an in-flight onboard is still
        // reading the lifecycle's pinned source, and the release happens only
        // at reap (RAII) — never before the handle reading it is terminal
        // (REFACTOR.md §4, "no early search null" / issue A).
        let pending = slot.onboard.as_ref().is_some_and(|h| !h.is_complete())
            || slot.offloads.iter().any(|h| !h.is_complete());
        if pending {
            // D: the engine owns the emission from here. Consume-once: a drain
            // exists iff offloads were registered and it was not taken before.
            if let Some(drain) = self.engine.take_offload_drain(&request_id.to_string()) {
                drain.commit();
            }
            self.finishing.insert(request_id.to_string());
            ApiFinishedStatus::Pending
        } else {
            // Scrub an unconsumed drain registration WITHOUT committing: vLLM
            // frees the blocks on this `Finished` answer immediately, and a
            // later `finished_sending` for an already-freed request asserts in
            // its scheduler. Dropping the un-committed drain emits nothing.
            drop(self.engine.take_offload_drain(&request_id.to_string()));
            if slot.drain_holder_draining() {
                // Evicted-then-finished (the A×E finish-during-drain twin):
                // the VISIBLE handles are empty, but the drain-holder's onboard
                // is still reading the held lifecycle pin. The directive answer
                // is still `Finished` — the reap alone defers to the sweep.
                self.finishing.insert(request_id.to_string());
            } else {
                // Immediate reap: dropping the slot releases its RAII handles
                // (the parked `proposal` via its kind-routed release, tracked
                // actions via `release_action`). Nothing in flight is reading
                // a pin — `pending` was false and the drain-holders are not
                // draining. The handle drops here fire the lifecycle
                // releases, which also clear the engine's deferral-guard
                // entries.
                self.slots.remove(request_id);
            }
            ApiFinishedStatus::Finished
        }
    }

    /// vLLM-side preemption (non-terminal). The slot survives and is reset to a
    /// fresh-GNMT shape; the engine publish is deduped to at most once per
    /// metadata cycle so the worker fences exactly once.
    ///
    /// Two distinct hazards close here:
    /// - **The onboard source pin (A×E).** The in-flight onboard is still
    ///   reading the lifecycle's pinned G2/G3 source; dropping `proposal`
    ///   synchronously would free that source mid-drain. So the parked
    ///   `proposal` (+ the onboard reading it) are *moved* into
    ///   `slot.drain_holders`, keeping the pin alive. The RAII release is
    ///   deferred to [`Self::update_connector_output`], which drops the holder
    ///   once the onboard handle is terminal.
    /// - **Restore = fresh GNMT.** The slot's *visible* match state is reset, so
    ///   vLLM's reuse of the same `req_id` re-enters `find_blocks` on the
    ///   fresh-mint arm (visible `proposal == None`), not the refresh arm.
    pub(crate) fn on_evicted(&mut self, request_id: &str) -> Result<EngineFinishedStatus, Error> {
        // Fence decision BEFORE any handle view detaches. The engine's `evict`
        // captures every action still in flight for this request (the onboard
        // reading the lifecycle pin AND the offload actions reading the G1
        // blocks vLLM is about to recycle) into the per-worker fence — and
        // internally tears down a dispatched prefill's generation. A handle
        // dropped before the fence is minted fires `release_action`, pruning
        // the action record the capture reads — the in-flight G1 read would
        // escape the fence and the worker would proceed mid-copy. Once
        // fence-armed, a later handle drop defers record removal until the
        // driver terminal.
        //
        // A fence is minted on the FIRST eviction of a metadata cycle AND again
        // whenever the slot holds live visible handles: a same-cycle restore
        // re-arms proposal/onboard/offloads, and those new in-flight actions
        // need their own capture — `pending_evictions` alone would detach them
        // unfenced. Multiple fences per request per cycle are safe (per-token
        // UUIDs; the worker awaits the union for its rank). Drain-held onboards
        // don't re-fence: their generation's fence already captured them.
        let must_fence = {
            let slot = self
                .slots
                .get(request_id)
                .ok_or_else(|| Error::SlotNotFound(request_id.to_string()))?;
            slot.proposal.is_some() || slot.onboard.is_some() || !slot.offloads.is_empty()
        };
        let first_this_cycle = self.pending_evictions.insert(request_id.to_string());
        let mut fence_handle = None;
        if first_this_cycle || must_fence {
            let outcome = self.engine.evict(&request_id.to_string());
            // An EMPTY fence (nothing was in flight engine-side) is wire
            // noise: the workers would await zero tokens. Only a fence that
            // carries tokens rides the envelope; `evicted_requests` records
            // the rid either way (`pending_evictions` above).
            if !outcome.fence.per_worker.is_empty() {
                self.pending_fences.push(outcome.fence);
            }
            fence_handle = outcome.handle;
        }

        let slot = self
            .slots
            .get_mut(request_id)
            .expect("presence checked by the fence decision above");

        // A×E hazard: move the parked lifecycle (+ the in-flight onboard
        // reading its pinned source, + the eviction's leader-side fence
        // handle) into a NEW drain-holder entry so the pin drains rather than
        // drops. PUSH, never overwrite — a restored request re-evicted while
        // an older generation's holder still drains must keep that holder.
        // The push is UNIFORM, kind-blind: a prefill-kind handle riding a
        // holder is safe because `evict` already released the generation
        // engine-side, so the holder's eventual drop no-ops on the generation
        // guard. The RAII release is deferred to `update_connector_output`.
        if let Some(proposal) = slot.proposal.take() {
            let onboard = slot.onboard.take();
            slot.drain_holders.push((proposal, onboard, fence_handle));
        } else if let Some(handle) = fence_handle {
            // Holderless eviction (no proposal was parked, but the engine
            // armed in-flight work): keep the leader's observational view of
            // the drain on the slot; the sweep drops it once complete.
            slot.fence_holders.push(handle);
        }

        // Reset the VISIBLE match state so the next GNMT runs fresh. The
        // `sequence` is kept — the scheduler resyncs it. The `offloads` are
        // dropped (view-detach: they read G1, which the fence covers); dropping
        // them is what makes a stale pre-eviction save terminal flip an
        // already-gone handle instead of emitting `finished_sending` for the
        // restored generation.
        //
        // An onboard still visible HERE has no lifecycle pin to ride a
        // drain-holder (the engine consumed it at onboard time and a re-poll
        // resolved `release_parked`). Dropping the handle is the pre-existing
        // view-detach — the fence minted above already captured the in-flight
        // action; the eventual holder release clears the engine's
        // deferral-guard entry.
        slot.onboard = None;
        slot.offloads.clear();
        slot.block_ids.clear();
        slot.evaluated_tokens = 0;

        Ok(EngineFinishedStatus::Finished)
    }

    /// vLLM-side preemption fan-in: the scheduler's `preempted_req_ids` for
    /// this step, applied through [`Self::on_evicted`] one rid at a time.
    /// Runs BEFORE the scheduled-request walk so the minted fences are staged
    /// in `pending_fences` when `build_metadata` drains them into this same
    /// step's envelope.
    ///
    /// An unknown rid is tolerated with a warning, never an error: vLLM can
    /// preempt a request the connector already finished tracking (e.g. one
    /// aborted after the connector reaped its slot), and failing the step for
    /// it would take down requests that are still healthy.
    pub(crate) fn on_preempted(&mut self, request_ids: &[String]) {
        for rid in request_ids {
            if let Err(error) = self.on_evicted(rid) {
                tracing::warn!(
                    request_id = %rid,
                    %error,
                    "preempted request unknown to the connector; skipping"
                );
            }
        }
    }

    /// Advance the iteration counter and drain the pending evictions + fences
    /// into a per-step payload.
    pub(crate) fn build_metadata(&mut self) -> ConnectorMetadata {
        self.iteration = self.iteration.saturating_add(1);
        ConnectorMetadata {
            iteration: self.iteration,
            evicted_requests: std::mem::take(&mut self.pending_evictions)
                .into_iter()
                .collect(),
            fences: std::mem::take(&mut self.pending_fences),
        }
    }

    /// Per-step scheduler-output walk. Syncs the block-id list for every
    /// scheduled request, advances each slot's `evaluated_tokens` save cursor
    /// (capped at hash-complete × allocated blocks), and hands completed novel
    /// blocks to the engine's offload pipeline as `(SequenceHash, BlockId)`
    /// pairs. The offload pipeline's presence filter deduplicates already-cached
    /// blocks, so the cursor always starts at 0 on a fresh slot.
    ///
    /// Intra-pass fields are left `None`: engine-owned inter-pass onboard runs
    /// on the leader runtime; intra-pass load would need new seam surface.
    /// `scheduled_offloads` tells the wrapping `Leader` to arm the
    /// forward-pass flush trigger (events embedded there — the walk itself
    /// stays velo-free and unit-testable).
    pub(crate) fn build_connector_meta(&mut self, output: &SchedulerOutput) -> WalkOutcome {
        // Preemptions land BEFORE the scheduled-request walk (and before the
        // empty-frame early return): the fences they mint must be staged for
        // THIS step's envelope, and a preempted request must be detached
        // before the walk could advance its save cursor.
        self.on_preempted(&output.preempted_req_ids);
        if output.total_num_scheduled_tokens == 0 {
            return WalkOutcome {
                metadata: KvConnectorMetadata::new(output.iteration),
                scheduled_offloads: false,
            };
        }
        let mut scheduled_offloads = false;

        for new_req in &output.scheduled_new_reqs {
            let Some(slot) = self.slots.get_mut(&new_req.req_id) else {
                tracing::debug!(
                    request_id = %new_req.req_id,
                    "build_connector_meta: skipping unknown new request"
                );
                continue;
            };
            sync_full_block_list(slot, &new_req.block_ids);
            let scheduled = output
                .num_scheduled_tokens
                .get(&new_req.req_id)
                .copied()
                .unwrap_or(0);
            let target = new_req.num_computed_tokens + scheduled;
            let req_id = new_req.req_id.clone();
            scheduled_offloads |= self.offload_step(&req_id, target);
        }

        for cached in &output.scheduled_cached_reqs {
            let target = self.compute_cached_target(cached, &output.num_scheduled_tokens);
            let req_id = cached.req_id.clone();
            scheduled_offloads |= self.offload_step(&req_id, target);
        }

        WalkOutcome {
            metadata: KvConnectorMetadata::new(output.iteration),
            scheduled_offloads,
        }
    }

    /// Compute the target token count for a cached request and sync its block
    /// list, returning the target so the borrow of `cached` ends before
    /// `offload_step` borrows `self` mutably.
    fn compute_cached_target(
        &mut self,
        cached: &CachedRequestData,
        num_scheduled_tokens: &HashMap<String, usize>,
    ) -> usize {
        let scheduled = num_scheduled_tokens
            .get(&cached.req_id)
            .copied()
            .unwrap_or(0);

        let Some(slot) = self.slots.get_mut(&cached.req_id) else {
            tracing::debug!(
                request_id = %cached.req_id,
                "build_connector_meta: skipping unknown cached request"
            );
            return 0;
        };

        if cached.resumed {
            // A resumed request carries the FULL token list; extend the
            // sequence by the suffix beyond what the slot already holds.
            if let Some(all) = &cached.all_token_ids {
                let have = slot.total_tokens();
                if all.len() > have
                    && let Err(e) = slot.extend_tokens(all[have..].to_vec())
                {
                    tracing::warn!(
                        request_id = %cached.req_id,
                        error = %e,
                        "failed to extend resumed slot tokens; skipping"
                    );
                    return 0;
                }
            }
            // For a resumed request, new_block_ids is the full fresh list.
            sync_full_block_list(slot, &cached.new_block_ids);
            // Cursor was reset to 0 at eviction.
            cached.num_computed_tokens + scheduled
        } else {
            // Plain decode/chunk growth: new_block_ids are purely the delta.
            slot.block_ids.extend_from_slice(&cached.new_block_ids);
            slot.evaluated_tokens + scheduled
        }
    }

    /// Advance a slot's `evaluated_tokens` save cursor to `desired_tokens` and
    /// submit any newly-complete novel blocks to the engine's offload pipeline.
    /// Returns whether an offload action was handed to the engine.
    ///
    /// Cursor advancement rule:
    /// - While `evaluated_tokens < assigned_tokens`, the cursor is capped at
    ///   `assigned_tokens` (the offload can only proceed for complete, allocated
    ///   blocks; we don't advance past them until they are committed).
    /// - Once `evaluated_tokens >= assigned_tokens`, the cursor advances
    ///   freely to `desired_tokens` — the scheduler may schedule tokens that
    ///   don't yet complete a block, and we track that progress so a later
    ///   step that fills the block picks up from the right place.
    ///
    /// Caller guarantees the request has a slot (private helper). The engine
    /// call is sync and cheap; borrows are structured to avoid holding the slot
    /// reference across the engine call.
    fn offload_step(&mut self, request_id: &str, desired_tokens: usize) -> bool {
        // Phase 1: read what we need and compute pairs under one get_mut.
        let maybe_pairs = {
            let Some(slot) = self.slots.get_mut(request_id) else {
                return false;
            };
            if desired_tokens <= slot.evaluated_tokens {
                return false; // no forward progress
            }
            let assigned_tokens = slot.assigned_blocks() * self.block_size;
            // Advance the cursor: capped at assigned_tokens unless the cursor
            // is already at or past the assignment boundary (soft decode tracking).
            let new_eval = if slot.evaluated_tokens >= assigned_tokens {
                desired_tokens // already at boundary — track freely
            } else {
                desired_tokens.min(assigned_tokens)
            };
            let from_block = slot.evaluated_tokens / self.block_size;
            let to_block = new_eval.min(assigned_tokens) / self.block_size; // exclusive
            slot.evaluated_tokens = new_eval;

            if to_block <= from_block {
                return false; // cursor advanced but no new complete block crossed
            }

            let pairs: Vec<(SequenceHash, BlockId)> = (from_block..to_block)
                .map(|i| (slot.sequence_hash(i), slot.block_ids[i]))
                .collect();
            pairs
        };

        // Phase 2: call engine (no slot borrow held).
        match Arc::clone(&self.engine).offload(&request_id.to_string(), maybe_pairs) {
            Ok(handle) => {
                // Phase 3: push handle under a second get_mut.
                if let Some(slot) = self.slots.get_mut(request_id) {
                    slot.offloads.push(handle);
                }
                true
            }
            Err(e) => {
                tracing::warn!(
                    request_id,
                    error = %e,
                    "engine rejected offload; skipping step"
                );
                false
            }
        }
    }

    /// vLLM's `update_connector_output`. Runs the deferred, HANDLE-gated reap
    /// for the `Pending` case of [`Self::request_finished`]: the emission was
    /// handed to the engine at finish time, so this sweep only reaps each
    /// finishing slot once its onboard + every offload handle reports terminal.
    /// The vLLM sets are the wake-up cadence, never the gate.
    pub(crate) fn update_connector_output(
        &mut self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) {
        // The emission (engine-armed at request_finished) produced this set;
        // nothing here keys off it — handles are the gate.
        let _ = finished_sending;

        let finishing = std::mem::take(&mut self.finishing);
        for req in finishing {
            match self.slots.get(&req) {
                // Already gone (defensive; reap is sweep-only for Pending slots).
                None => {}
                Some(slot) => {
                    // Visible handles terminal AND the eviction drain-holder
                    // (if any) fully drained — reaping drops the holder, which
                    // releases the held lifecycle pin (A×E).
                    let terminal = slot.onboard.as_ref().is_none_or(OnboardHandle::is_complete)
                        && slot.offloads.iter().all(|h| h.is_complete())
                        && !slot.drain_holder_draining();
                    if terminal {
                        // Dropping the slot releases the RAII handles — the
                        // parked `proposal` via its kind-routed release (only
                        // now is nothing reading it), tracked actions via
                        // `release_action`; the lifecycle releases also
                        // clear the engine's deferral-guard entries.
                        self.slots.remove(&req);
                    } else {
                        // Still draining: keep it on the worklist.
                        self.finishing.insert(req);
                    }
                }
            }
        }

        // Recv-side release. vLLM reporting a request's async load done is the
        // cadence; the handle is the authority — only a TERMINAL onboard is
        // dropped (a pending one stays parked and resolves on a later sweep or
        // at finish). Dropping the onboard also drops `proposal`: the engine
        // consumed the lifecycle's pinned state at onboard time, and with the
        // load terminal nothing reads the source any longer — holding the pin
        // would keep its G2/G3 blocks resident for no reader. The slot itself
        // stays (the request is still decoding); a finishing request was
        // already reaped by the sweep above on the same gates.
        for req in &finished_recving {
            if let Some(slot) = self.slots.get_mut(req)
                && slot
                    .onboard
                    .as_ref()
                    .is_some_and(OnboardHandle::is_complete)
            {
                slot.onboard = None;
                slot.proposal = None;
            }
        }

        // Poll the eviction drain-holders: each eviction moved the parked
        // `proposal` (+ in-flight `onboard`, + its leader-side fence handle)
        // into its own entry so the source pin drains rather than drops. Drop
        // each entry whose drain completed — the engine fence when one was
        // minted (it covers the held onboard AND the view-detached offloads),
        // else the held onboard's terminal — firing the kind-routed RAII
        // release that frees that generation's now-unneeded pin;
        // still-draining entries stay. The holderless fence handles sweep on
        // the same gate: a completed fence has nothing left to observe.
        for slot in self.slots.values_mut() {
            slot.drain_holders.retain(super::slot::holder_draining);
            slot.fence_holders.retain(|fence| !fence.is_complete());
        }
    }
}

/// Sync a slot's `block_ids` with a new authoritative list. The existing
/// prefix must agree with the new list (debug-checked); the tail is appended.
fn sync_full_block_list(slot: &mut RequestSlot, ids: &[BlockId]) {
    if ids.len() > slot.block_ids.len() {
        debug_assert_eq!(
            ids[..slot.block_ids.len()],
            slot.block_ids[..],
            "block id prefix mismatch when syncing slot {} — engine and vLLM disagree",
            slot.request_id,
        );
        slot.block_ids
            .extend_from_slice(&ids[slot.block_ids.len()..]);
    }
}
