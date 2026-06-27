// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The unified `find_blocks` / `onboard_blocks` router.
//!
//! The connector drives exactly two match-side verbs and the engine routes
//! everything else. The seam stays SOURCE-AGNOSTIC: internally the engine
//! stages `{local, remote}` G2 → G1 where the remote G2 may come from a remote
//! search OR a dispatched remote prefill — nothing search-vs-prefill-shaped
//! crosses back. [`route_find_blocks`](LocalConnectorEngine::route_find_blocks)
//! returns only adapter facts (token-granular matched count, the parked-handle
//! bookkeeping);
//! [`route_onboard_blocks`](LocalConnectorEngine::route_onboard_blocks) returns
//! the one `OnboardHandle`.
//!
//! ## `find_blocks` routing
//!
//! 1. **Prefill arm** — `req.transfer_params` carry `remote_prefill`: the
//!    accept core latches (first poll) or refreshes (re-poll) the lifecycle;
//!    the external-token count passes through token-granular. A latch with no
//!    caller handle, or a kind mismatch between the live handle and the
//!    request, is a loud [`LeaderEngineError::FindBlocksDesync`]. A
//!    ZERO-external lifecycle falls through to the local arm with the search
//!    bound INSIDE the lifecycle (`PrefillRequestState::bind_local_search`) —
//!    the connector still parks only the one prefill handle. The WHOLE prefill
//!    arm (including the fall-through's internal mint and refresh) is EXEMPT
//!    from the deferral guard: the disagg dispatcher placed the request here;
//!    it must never park.
//! 2. **Window derivation** — engine-side, from the shared hash chain plus the
//!    counts, as an index RANGE over the request's `Arc` (no per-poll window
//!    copy; see [`derive_window`]). The derived view also carries the computed
//!    prefix slice `[0, computed)` so the CD interpose can name the prefix
//!    hashes a remote commit must serve. An EMPTY window short-circuits to a
//!    synchronous zero `Resolved` without touching a live search; on the
//!    pure-local path it carries `release_parked` so the connector drops its
//!    pin (gated connector-side on the in-flight onboard).
//! 3. **Refresh arm** — `live` is a Search-kind handle: the deferral guard
//!    runs first — a window overlapping ANOTHER lifecycle's in-flight onboard
//!    resolves `Deferred` while the live search keeps running untouched.
//!    Otherwise reconcile in place (never a second mint; a pure re-poll skips
//!    the merge via the set-equality heuristic — see
//!    `LocalConnectorEngine::local_refresh`). A zero-refine or `Lost`
//!    resolves zero with `release_parked`; `Pending` maps to `Searching`.
//! 4. **Fresh arm** — the deferral guard again: a fresh window overlapping an
//!    in-flight onboard resolves `Deferred` with no side effect. Otherwise the
//!    local search runs; a latch mints the one Search-kind handle.
//!
//! Self-deferral is impossible by construction on BOTH guarded arms: a
//! lifecycle records into the guard only at its USAA-time onboard mint, and
//! vLLM never re-calls GNMT for a request after USAA (the `num_computed == 0`
//! guard in its scheduler) — except preemption-restore, which resets to a
//! FRESH poll whose deferral on the prior generation's still-draining load is
//! intended behavior (the old load must drain first).
//!
//! ## `onboard_blocks` routing
//!
//! Kind-routed off the opaque handle: a Search-kind handle onboards the
//! matched local span (with a log-level committed-vs-promised token check the
//! connector never had); a Prefill-kind handle with stored external work
//! validates the committed count against the engine's stored promise
//! ([`LeaderEngineError::ExternalTokensMismatch`]) and runs the USAA kick; a
//! zero-stored prefill delegates to its internally-bound local search. One
//! onboard per latched generation —
//! [`LeaderEngineError::OnboardAlreadyInFlight`] refuses re-entry.

use std::ops::Range;
use std::sync::Arc;

use kvbm_protocols::connector::{BlockId, RequestId, SequenceHash};
use kvbm_protocols::connector::{
    FindBlocksHandle, FindBlocksOutcome, FindBlocksRequest, LeaderEngine, LeaderEngineError,
    OnboardHandle, SearchId,
};
use kvbm_protocols::disagg::RemotePrefillParams;

use super::local::{LocalConnectorEngine, LocalSearchOutcome, MatchStatus, MatchWindow};
use super::prefill::PrefillAcceptCore;
use crate::remote::cd::prefill::PrefillRequestState;

/// The engine-side projection of one poll's eligible match window — an index
/// RANGE over the request's shared hash chain, never an owned copy.
pub(super) struct DerivedWindow {
    /// vLLM's computed prefix in blocks (`num_computed_tokens / block_size`).
    pub(super) computed_blocks: usize,
    /// The eligible suffix range `computed .. eligible` into
    /// `req.sequence_hashes`.
    pub(super) range: Range<usize>,
}

impl DerivedWindow {
    pub(super) fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Borrow the window slice out of the request's shared chain.
    pub(super) fn slice<'a>(&self, req: &'a FindBlocksRequest) -> &'a [SequenceHash] {
        &req.sequence_hashes[self.range.clone()]
    }

    /// The window as the match cores' borrowed input shape. The computed
    /// prefix `[0, computed)` rides along (clamped to the hashed chain) so the
    /// CD interpose can name the prefix hashes a remote commit must serve —
    /// the suffix window alone cannot express them.
    pub(super) fn view<'a>(&self, req: &'a FindBlocksRequest) -> MatchWindow<'a> {
        let prefix_end = self.computed_blocks.min(req.sequence_hashes.len());
        MatchWindow {
            request_id: &req.request_id,
            prefix_plhs: &req.sequence_hashes[..prefix_end],
            block_plhs: self.slice(req),
            computed_blocks: self.computed_blocks as u32,
        }
    }
}

/// Derive the eligible match window from the request's full hash chain and
/// counts. Applies vLLM's last-must-recompute exclusion: the request must
/// compute at least one token locally, so a block-aligned sequence excludes its
/// final full block and a ragged one already cannot match its partial tail —
/// `(total_tokens - 1) / block_size` expresses both. The bound is clamped to
/// the hashed chain defensively (a well-formed request never exceeds it), and a
/// computed prefix at or past the eligible bound yields an empty window.
pub(super) fn derive_window(req: &FindBlocksRequest, block_size: usize) -> DerivedWindow {
    let computed_blocks = req.num_computed_tokens / block_size;
    let eligible_blocks =
        (req.total_tokens.saturating_sub(1) / block_size).min(req.sequence_hashes.len());
    DerivedWindow {
        computed_blocks,
        range: computed_blocks.min(eligible_blocks)..eligible_blocks,
    }
}

impl LocalConnectorEngine {
    /// `find_blocks` body — see the module docs for the routing table.
    pub(super) fn route_find_blocks(
        self: &Arc<Self>,
        req: &FindBlocksRequest,
        live: Option<&FindBlocksHandle>,
    ) -> Result<FindBlocksOutcome, LeaderEngineError> {
        if let Some(params) = req
            .transfer_params
            .as_ref()
            .and_then(|t| t.remote_prefill.as_ref())
        {
            return self.find_blocks_prefill(req, params, live);
        }
        // A Prefill-kind live handle on a request with no decode params means
        // engine and connector disagree about the latched lifecycle's kind.
        if live.is_some_and(|h| h.prefill_accept_id().is_some()) {
            return Err(LeaderEngineError::FindBlocksDesync);
        }

        let derived = derive_window(req, self.block_size);
        if derived.is_empty() {
            // The local prefix covers everything eligible — nothing external is
            // fetchable, so the answer is a synchronous zero and the parked pin
            // has no further use. NEVER touches a live search (a zero-width
            // refresh has nothing to reconcile); the connector executes the
            // drop, gated on its in-flight onboard.
            return Ok(FindBlocksOutcome::Resolved {
                matched_tokens: 0,
                minted: None,
                release_parked: true,
            });
        }

        // The in-flight onboard deferral guard runs on BOTH local arms (fresh
        // AND refresh): a window intersecting hashes another lifecycle is
        // loading into G1 is deferred with no engine side effect — the caller
        // re-polls and the entry clears at that lifecycle's release.
        // Self-deferral cannot happen here: a lifecycle records only at its
        // USAA-time onboard mint and GNMT is never re-called post-USAA, so an
        // overlap is always ANOTHER lifecycle's load (or, on the fresh arm, a
        // preemption-restored request's own PRIOR generation still draining in
        // its holder — intended: the old load must drain first).
        if self
            .inflight
            .lock()
            .expect("inflight-guard mutex poisoned")
            .overlaps(derived.slice(req))
        {
            return Ok(FindBlocksOutcome::Deferred);
        }
        let window = derived.view(req);

        // REFRESH arm: reconcile in place, never a second mint; a deferral
        // above leaves the live search running untouched.
        if let Some(handle) = live {
            let sid = handle
                .search_id()
                .expect("prefill-kind live handles were routed above");
            return Ok(match self.local_refresh(sid, &window) {
                MatchStatus::Matched { hit_blocks } if hit_blocks > 0 => {
                    FindBlocksOutcome::Resolved {
                        matched_tokens: hit_blocks as usize * self.block_size,
                        minted: None,
                        release_parked: false,
                    }
                }
                // Zero-refine / Lost: the external prefix is gone; the parked
                // pin is released (connector-side, Issue-A-gated) so the next
                // poll fresh-mints instead of refining a dead latch.
                MatchStatus::Matched { .. } | MatchStatus::Lost => FindBlocksOutcome::Resolved {
                    matched_tokens: 0,
                    minted: None,
                    release_parked: true,
                },
                MatchStatus::Pending => FindBlocksOutcome::Searching { minted: None },
            });
        }

        // FRESH-mint arm.
        Ok(match self.local_search(&window) {
            LocalSearchOutcome::NoMatch => FindBlocksOutcome::Resolved {
                matched_tokens: 0,
                minted: None,
                release_parked: false,
            },
            LocalSearchOutcome::Latched { search_id, status } => {
                let me: Arc<dyn LeaderEngine> = Arc::clone(self) as Arc<dyn LeaderEngine>;
                let minted = Some(FindBlocksHandle::search(
                    req.request_id.clone(),
                    search_id,
                    Arc::downgrade(&me),
                ));
                match status {
                    MatchStatus::Matched { hit_blocks } => FindBlocksOutcome::Resolved {
                        matched_tokens: hit_blocks as usize * self.block_size,
                        minted,
                        release_parked: false,
                    },
                    MatchStatus::Pending | MatchStatus::Lost => {
                        FindBlocksOutcome::Searching { minted }
                    }
                }
            }
        })
    }

    /// The dispatched-remote-prefill arm of [`Self::route_find_blocks`].
    fn find_blocks_prefill(
        self: &Arc<Self>,
        req: &FindBlocksRequest,
        params: &RemotePrefillParams,
        live: Option<&FindBlocksHandle>,
    ) -> Result<FindBlocksOutcome, LeaderEngineError> {
        let Some(cd) = self.cd.as_ref() else {
            // A dispatched prefill landing on a non-CD engine is a deployment
            // misconfiguration — refuse loudly (same as the accept core).
            return Err(LeaderEngineError::DisaggNotConfigured);
        };
        let rid = &req.request_id;
        // A Search-kind live handle on a dispatched-prefill request is a kind
        // disagreement; refuse BEFORE the accept can latch a lifecycle whose
        // handle nobody would hold.
        if live.is_some_and(|h| h.search_id().is_some()) {
            return Err(LeaderEngineError::FindBlocksDesync);
        }
        // A latched lifecycle with no caller handle means the connector lost
        // the RAII release — fail loud rather than run a session nobody can
        // tear down.
        if live.is_none() && cd.prefill.get(rid).is_some() {
            return Err(LeaderEngineError::FindBlocksDesync);
        }

        let (external, minted) = match Arc::clone(self).prefill_accept_core(
            rid,
            params,
            &req.sequence_hashes,
            req.num_computed_tokens,
        )? {
            PrefillAcceptCore::Accepted {
                accept_id,
                external_tokens,
            } => {
                // First latch — or a fresh latch after an eviction released the
                // prior generation. A stale handle the connector still parks is
                // simply replaced; its eventual drop no-ops on the AcceptId
                // guard.
                let me: Arc<dyn LeaderEngine> = Arc::clone(self) as Arc<dyn LeaderEngine>;
                let handle = FindBlocksHandle::prefill(rid.clone(), accept_id, Arc::downgrade(&me));
                (external_tokens, Some(handle))
            }
            PrefillAcceptCore::Refreshed { external_tokens } => {
                if live.is_none() {
                    // An accept-insert race latched between the check above and
                    // the core — still a lifecycle with no caller handle.
                    return Err(LeaderEngineError::FindBlocksDesync);
                }
                (external_tokens, None)
            }
        };
        if external > 0 {
            // Already a TOKEN count (the engine stores
            // `num_provided_tokens - num_computed_tokens`, saturating) —
            // source-agnostic passthrough.
            return Ok(FindBlocksOutcome::Resolved {
                matched_tokens: external,
                minted,
                release_parked: false,
            });
        }

        // ZERO-external fall-through: everything decode provided is already
        // computed locally, but the parked lifecycle keeps the cache-warming
        // pull (and the output direction) alive, so the local search binds
        // INSIDE it rather than minting a second connector-visible handle.
        let Some(state) = cd.prefill.get(rid) else {
            // The lifecycle vanished between the accept and the fall-through
            // (a racing evict); answer zero and let the next poll re-latch.
            return Ok(FindBlocksOutcome::Resolved {
                matched_tokens: 0,
                minted,
                release_parked: false,
            });
        };
        Ok(self.prefill_fall_through(req, &state, minted))
    }

    /// The zero-external local fall-through, bound inside the prefill
    /// lifecycle. Mirrors the pure-local fresh/refresh/empty-window arms with
    /// two deliberate differences: the internal mint is EXEMPT from the
    /// deferral guard, and the outcome never carries `release_parked` (the
    /// connector's parked handle is the PREFILL lifecycle, which must stay
    /// alive; the internal pin is released engine-side instead).
    fn prefill_fall_through(
        self: &Arc<Self>,
        req: &FindBlocksRequest,
        state: &Arc<PrefillRequestState>,
        minted: Option<FindBlocksHandle>,
    ) -> FindBlocksOutcome {
        let derived = derive_window(req, self.block_size);
        if derived.is_empty() {
            // Refine-to-zero analogue on the INTERNAL binding only.
            if let Some(sid) = state.take_local_search() {
                self.release_search(&sid);
            }
            return FindBlocksOutcome::Resolved {
                matched_tokens: 0,
                minted,
                release_parked: false,
            };
        }
        let window = derived.view(req);

        if let Some(sid) = state.local_search_id() {
            return match self.local_refresh(sid, &window) {
                MatchStatus::Matched { hit_blocks } if hit_blocks > 0 => {
                    FindBlocksOutcome::Resolved {
                        matched_tokens: hit_blocks as usize * self.block_size,
                        minted,
                        release_parked: false,
                    }
                }
                MatchStatus::Matched { .. } | MatchStatus::Lost => {
                    // Zero-refine / Lost: release the dead internal pin so the
                    // next poll fresh-mints a new one.
                    if let Some(sid) = state.take_local_search() {
                        self.release_search(&sid);
                    }
                    FindBlocksOutcome::Resolved {
                        matched_tokens: 0,
                        minted,
                        release_parked: false,
                    }
                }
                MatchStatus::Pending => FindBlocksOutcome::Searching { minted },
            };
        }

        // Fresh INTERNAL mint — exempt from the deferral guard.
        match self.local_search(&window) {
            LocalSearchOutcome::NoMatch => FindBlocksOutcome::Resolved {
                matched_tokens: 0,
                minted,
                release_parked: false,
            },
            LocalSearchOutcome::Latched { search_id, status } => {
                state.bind_local_search(search_id);
                match status {
                    MatchStatus::Matched { hit_blocks } => FindBlocksOutcome::Resolved {
                        matched_tokens: hit_blocks as usize * self.block_size,
                        minted,
                        release_parked: false,
                    },
                    MatchStatus::Pending | MatchStatus::Lost => {
                        FindBlocksOutcome::Searching { minted }
                    }
                }
            }
        }
    }

    /// `onboard_blocks` body — kind-routed off the opaque handle (see the
    /// module docs).
    pub(super) fn route_onboard_blocks(
        self: &Arc<Self>,
        handle: &FindBlocksHandle,
        dest: &[BlockId],
        num_external_tokens: usize,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        // LOCAL arm: the committed-vs-promised check the connector never had,
        // log-level (the local promise is engine-derived, so a divergence is
        // diagnostic — the slice math below keys off the engine's own state).
        if let Some(search_id) = handle.search_id() {
            self.warn_on_promise_divergence(search_id, handle.request_id(), num_external_tokens);
            return Arc::clone(self).local_onboard(search_id, dest);
        }

        // PREFILL arm.
        let accept_id = handle
            .prefill_accept_id()
            .expect("a FindBlocksHandle is either Search- or Prefill-kind");
        let request_id = handle.request_id();
        let Some(cd) = self.cd.as_ref() else {
            return Err(LeaderEngineError::DisaggNotConfigured);
        };
        let Some(state) = cd.prefill.get(request_id) else {
            return Err(LeaderEngineError::PrefillSessionStale);
        };
        if state.accept_id() != accept_id {
            return Err(LeaderEngineError::PrefillSessionStale);
        }
        let stored = state.external_tokens();
        if stored == 0 {
            // Zero-stored delegation: the lifecycle's internally-bound local
            // search owns this onboard (the prefill session only cache-warms).
            // No binding means no matched search to onboard from. Re-entry is
            // naturally refused: the first onboard consumes the latch, so a
            // second delegation finds no live search.
            let Some(sid) = state.local_search_id() else {
                return Err(LeaderEngineError::SearchNotMatched);
            };
            self.warn_on_promise_divergence(sid, request_id, num_external_tokens);
            return Arc::clone(self).local_onboard(sid, dest);
        }
        // vLLM commits exactly what `find_blocks` promised; the promise is the
        // engine-stored cell this same generation refreshes. A divergence means
        // the suffix arithmetic in the kick would slice the wrong blocks.
        if num_external_tokens != stored {
            return Err(LeaderEngineError::ExternalTokensMismatch {
                expected: stored,
                got: num_external_tokens,
            });
        }
        Arc::clone(self).prefill_onboard_by_id(request_id, accept_id, dest)
    }

    /// Log-level committed-vs-promised divergence check, shared by both
    /// local-onboard entry points (the Search-kind arm and the zero-stored
    /// prefill delegation): the local promise is engine-derived, so a
    /// divergence is diagnostic — the onboard slice math keys off the
    /// engine's own state, never the committed count.
    fn warn_on_promise_divergence(
        &self,
        search_id: SearchId,
        request_id: &RequestId,
        num_external_tokens: usize,
    ) {
        if let Some(entry) = self.searches.get(&search_id)
            && let MatchStatus::Matched { hit_blocks } =
                *entry.status.lock().expect("search-status mutex poisoned")
        {
            let promised = hit_blocks as usize * self.block_size;
            if promised != num_external_tokens {
                tracing::warn!(
                    %request_id,
                    promised,
                    committed = num_external_tokens,
                    "onboard_blocks: committed external tokens diverge from the matched promise"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BS: usize = 4;

    fn h(i: u64) -> SequenceHash {
        SequenceHash::new(i, None, i)
    }

    fn chain(n: u64) -> Vec<SequenceHash> {
        (0..n).map(h).collect()
    }

    fn req(
        sequence_hashes: Vec<SequenceHash>,
        num_computed_tokens: usize,
        total_tokens: usize,
    ) -> FindBlocksRequest {
        FindBlocksRequest {
            request_id: "rq".to_string(),
            sequence_hashes: Arc::from(sequence_hashes),
            num_computed_tokens,
            total_tokens,
            transfer_params: None,
        }
    }

    /// The derived window's hashes (resolved through the borrow, since the
    /// derivation itself is an index range — never a copy).
    fn window_of(r: &FindBlocksRequest) -> Vec<SequenceHash> {
        derive_window(r, BS).slice(r).to_vec()
    }

    /// A block-aligned sequence excludes its final FULL block: the request must
    /// recompute at least one token locally, so `total = 3·BS` leaves only the
    /// first two blocks eligible. (`total / BS` instead of `(total-1) / BS`
    /// would wrongly include the third.)
    #[test]
    fn window_block_aligned_excludes_final_full_block() {
        let r = req(chain(3), 0, 3 * BS);
        assert_eq!(derive_window(&r, BS).computed_blocks, 0);
        assert_eq!(window_of(&r), vec![h(0), h(1)]);
    }

    /// A ragged sequence's partial tail is already un-matchable, so every
    /// complete block stays eligible.
    #[test]
    fn window_ragged_keeps_all_complete_blocks() {
        let r = req(chain(3), 0, 3 * BS + 2);
        assert_eq!(window_of(&r), vec![h(0), h(1), h(2)]);
    }

    /// The computed prefix offsets the window start (block-granular).
    #[test]
    fn window_computed_prefix_offsets_start() {
        let r = req(chain(4), BS, 4 * BS + 1);
        assert_eq!(derive_window(&r, BS).computed_blocks, 1);
        assert_eq!(window_of(&r), vec![h(1), h(2), h(3)]);
    }

    /// A computed prefix at (or past) the eligible bound yields an empty
    /// window — the empty-window short-circuit's input shape.
    #[test]
    fn window_computed_covering_eligible_is_empty() {
        let r = req(chain(3), 2 * BS, 3 * BS);
        assert!(derive_window(&r, BS).is_empty());
        // Past the bound (mid-decode re-poll shapes) is empty too, not a panic.
        let r = req(chain(3), 5 * BS, 3 * BS);
        let derived = derive_window(&r, BS);
        assert!(derived.is_empty());
        assert!(
            derived.slice(&r).is_empty(),
            "the slice borrow never panics"
        );
    }

    /// Zero totals derive an empty window.
    #[test]
    fn window_zero_total_tokens_is_empty() {
        let r = req(Vec::new(), 0, 0);
        let derived = derive_window(&r, BS);
        assert_eq!(derived.computed_blocks, 0);
        assert!(derived.is_empty());
    }

    /// The eligible bound clamps to the hashed chain (a malformed
    /// `total_tokens` never slices past the hashes that exist).
    #[test]
    fn window_eligible_clamps_to_chain_length() {
        let r = req(chain(2), 0, 10 * BS);
        assert_eq!(window_of(&r), vec![h(0), h(1)]);
    }
}
