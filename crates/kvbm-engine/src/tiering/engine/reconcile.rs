// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-side search reconciliation for `get_num_new_matched_tokens`.
//!
//! This is the engine-side reconcile core. It owns the match-state
//! mechanics (`OnboardingShard`s, `num_computed_tokens`, the contiguous
//! walk) and holds **no** connector-side conditional-disagg or
//! watchdog/timeout glue: there is no CD payload, CD-only constructor, or
//! `Instant` watchdog field here.
//!
//! # Coordinate system
//!
//! Every function in this module operates in **absolute block-index
//! coordinates over the full `sequence_hashes` vector** of the logical
//! sequence. `start_block`, `end_block`, `num_computed_tokens / block_size`,
//! and `compute_last_block_index(total_tokens, ..)` are all indices into
//! that one vector, measured from its origin (block 0). A caller holding
//! suffix-relative data (offsets relative to the as-yet-unmatched suffix)
//! **must rebase those offsets to absolute whole-sequence coordinates before
//! calling in** — these functions do no such rebasing themselves.
//!
//! # Reconciliation
//!
//! vLLM may re-poll `get_num_new_matched_tokens` for the same request between
//! forward passes with a different `num_computed_tokens`. Between polls the
//! scheduler may:
//!   * absorb more G1 blocks (`new > old`),
//!   * evict previously-computed G1 blocks (`new < old`), or
//!   * restore a prefix from eviction and thereby grow the visible sequence.
//!
//! Rather than discarding the saved search on mismatch, we reconcile a list of
//! `OnboardingShard`s covering contiguous block-index ranges of the sequence.
//! Each shard owns its own `FindMatchesResult`. On a mismatch we:
//!   * Case B (`new > old`): update `num_computed_tokens`; the completion walk
//!     masks off the redundant prefix via `effective_start`.
//!   * Case C (`new < old`): prepend a new shard covering
//!     `[new/bs .. shards[0].start_block)`.
//!   * Case D (`total_tokens` grew, indicating eviction restore): append a new
//!     shard for the new upper range.
//!
//! On completion, the outcome is computed by walking shards in order and
//! short-circuiting at the first hole (first-hole first-match semantics).

// P2.0: the reconcile core has no production caller yet — the first one
// arrives in P2.1 (`update_pin`). Both the lib and lib-test clippy targets
// flag these symbols as dead even though the ported unit tests exercise the
// public entry points, because (a) the lib target strips `cfg(test)`, and
// (b) `total_tokens_at_start` is read only by the derived `Debug`, which
// clippy 1.93 intentionally ignores for dead-code. A single module-level
// allow covers only the genuine "no production caller until P2.1" gap; the
// engine-match-state surface itself is copied whole from the legacy reconcile
// core (shards, num_computed_tokens, matched_span, aggregate_breakdown,
// release_all), with none of the connector-side CD / watchdog glue.
#![allow(dead_code)]

use crate::leader::{
    FindMatchesOptions, FindMatchesResult, InstanceLeader, Leader, MatchBreakdown,
    OnboardingStatus, StagingMode,
};
use anyhow::{Result, anyhow};
use kvbm_common::SequenceHash;

// ============================================================================
// Match-state types (engine-only half of the legacy OnboardingState)
// ============================================================================

/// A single contiguous sub-range of the logical sequence being searched.
///
/// Multiple shards exist when the search has been reconciled against a changing
/// `num_computed_tokens` or `total_tokens`: for example, when vLLM evicts G1 blocks
/// between polls we prepend a new prefix shard, and when tokens are restored from
/// eviction we append a new upper shard. On completion we walk shards in order
/// and unify their match counts using first-hole semantics.
#[derive(Debug)]
pub struct OnboardingShard {
    /// Block index in the logical sequence where this shard's search starts (inclusive).
    pub start_block: usize,

    /// Number of sequence hashes this shard queried. The shard covers block
    /// indices `[start_block .. start_block + num_queried_blocks)`.
    pub num_queried_blocks: usize,

    /// The find session that owns the matched blocks via RAII.
    pub find_session: FindMatchesResult,
}

impl OnboardingShard {
    /// Exclusive end block index of this shard.
    pub fn end_block(&self) -> usize {
        self.start_block + self.num_queried_blocks
    }

    /// Best-effort release of the underlying session.
    ///
    /// For `Ready` variants this is a no-op (blocks drop via RAII). For
    /// `AsyncSession` variants this calls `release_session` on the leader so
    /// that server-side session state is freed.
    pub fn release(&self, leader: &InstanceLeader) {
        if let Some(session_id) = self.find_session.session_id() {
            leader.release_session(session_id);
        }
    }
}

/// Engine match state for an onboarding search.
///
/// `shards` is a list of contiguous `OnboardingShard`s covering some block-index
/// range of the logical sequence; shards are reconciled and added when
/// `num_computed_tokens` or `total_tokens` changes between calls to
/// `get_num_new_matched_tokens` (see [`reconcile_state`]).
#[derive(Debug)]
pub struct OnboardingState {
    /// The number of tokens that match tokens already in the G1 storage,
    /// as last reported by vLLM. May be updated on retries.
    pub num_computed_tokens: usize,

    /// The `total_tokens` captured when the earliest shard was issued. Used
    /// to detect when the logical sequence has grown (eviction restore).
    pub total_tokens_at_start: usize,

    /// Shards sorted by `start_block` ascending. Invariant: contiguous and
    /// non-overlapping, i.e. `shards[i+1].start_block == shards[i].end_block()`.
    pub shards: Vec<OnboardingShard>,
}

impl OnboardingState {
    /// Build a new state from a single initial shard.
    pub fn new(
        num_computed_tokens: usize,
        total_tokens_at_start: usize,
        initial_shard: OnboardingShard,
    ) -> Self {
        let state = Self {
            num_computed_tokens,
            total_tokens_at_start,
            shards: vec![initial_shard],
        };
        state.debug_assert_contiguous();
        state
    }

    /// Total number of blocks queried across all shards (for metrics).
    pub fn total_query_blocks(&self) -> usize {
        self.shards.iter().map(|s| s.num_queried_blocks).sum()
    }

    /// Sum of match breakdowns across all shards.
    pub fn aggregate_breakdown(&self) -> MatchBreakdown {
        self.shards
            .iter()
            .map(|s| s.find_session.match_breakdown())
            .fold(MatchBreakdown::default(), |acc, b| MatchBreakdown {
                host_blocks: acc.host_blocks + b.host_blocks,
                disk_blocks: acc.disk_blocks + b.disk_blocks,
                object_blocks: acc.object_blocks + b.object_blocks,
            })
    }

    /// Return `true` iff every shard has reached a terminal state.
    pub fn all_shards_terminal(&self) -> bool {
        self.shards.iter().all(shard_is_terminal)
    }

    /// Compute the `(effective_start, final_end)` block-index span covered by
    /// the contiguous match so far.
    ///
    /// `effective_start` is the greater of the earliest shard's start and the
    /// current `num_computed_tokens / bs`. `final_end` is the first-hole
    /// boundary walking contiguously from `shards[0].start_block`.
    ///
    /// Precondition: all shards are terminal (`all_shards_terminal()` is true).
    pub fn matched_span(&self, block_size: usize) -> (usize, usize) {
        debug_assert!(!self.shards.is_empty());
        debug_assert!(self.all_shards_terminal());

        let mut running_end = self.shards[0].start_block;
        let mut final_end = running_end;
        for shard in &self.shards {
            debug_assert_eq!(shard.start_block, running_end);
            let matched = shard_terminal_matched_count(shard);
            if matched < shard.num_queried_blocks {
                final_end = running_end + matched;
                break;
            }
            running_end += shard.num_queried_blocks;
            final_end = running_end;
        }

        let new_computed_blocks = self.num_computed_tokens / block_size;
        let effective_start = self.shards[0].start_block.max(new_computed_blocks);
        (effective_start, final_end)
    }

    /// Check invariants on shard list (contiguous, non-overlapping, sorted).
    pub(crate) fn debug_assert_contiguous(&self) {
        if cfg!(debug_assertions) {
            for pair in self.shards.windows(2) {
                debug_assert_eq!(
                    pair[1].start_block,
                    pair[0].end_block(),
                    "OnboardingState shards must be contiguous: {:?}",
                    self.shards
                        .iter()
                        .map(|s| (s.start_block, s.num_queried_blocks))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    /// Release sessions for every shard (best-effort cleanup).
    pub fn release_all(&self, leader: &InstanceLeader) {
        for shard in &self.shards {
            shard.release(leader);
        }
    }
}

/// True if the find session of this shard has reached a terminal state.
pub fn shard_is_terminal(shard: &OnboardingShard) -> bool {
    match &shard.find_session {
        FindMatchesResult::Ready(_) => true,
        FindMatchesResult::AsyncSession(s) => matches!(
            s.status(),
            OnboardingStatus::Complete { .. }
                | OnboardingStatus::Holding { .. }
                | OnboardingStatus::Prepared { .. }
        ),
    }
}

/// Return the matched block count for a terminal shard.
///
/// Panics if called on a non-terminal shard; callers must gate on
/// [`OnboardingState::all_shards_terminal`] first.
///
/// **Drain-idempotent**: both Ready and AsyncSession variants return the
/// shard's matched count from a source captured at terminal-state time
/// (Ready: `match_breakdown`, set at construction; AsyncSession:
/// `OnboardingStatus::Complete.matched_blocks`, set by the staging path).
/// This is required so [`OnboardingState::matched_span`] returns the same
/// value before and after `take_g2_blocks` / `take_g3_blocks` has been called
/// on a shard. Reading `r.total_count()` (live Vec length) would shrink
/// `matched_span.final_end` post-drain.
pub fn shard_terminal_matched_count(shard: &OnboardingShard) -> usize {
    match &shard.find_session {
        // Ready: sum the per-tier breakdown captured at construction. This
        // covers bypass-mode hits (host_blocks + disk_blocks) and the
        // non-bypass case (host_blocks only). Reading `r.total_count()`
        // here would drop to 0 after `take_g2_blocks`/`take_g3_blocks`.
        FindMatchesResult::Ready(r) => {
            let b = r.match_breakdown();
            b.host_blocks + b.disk_blocks + b.object_blocks
        }
        FindMatchesResult::AsyncSession(s) => match s.status() {
            OnboardingStatus::Complete { matched_blocks } => matched_blocks,
            // Holding / Prepared are not currently produced on this path; treat the
            // session as if its g2_count() is authoritative.
            OnboardingStatus::Holding { .. } | OnboardingStatus::Prepared { .. } => {
                s.get_blocks_count().unwrap_or(0)
            }
            OnboardingStatus::Searching
            | OnboardingStatus::Preparing { .. }
            | OnboardingStatus::Staging { .. } => {
                debug_assert!(
                    false,
                    "shard_terminal_matched_count called on non-terminal shard"
                );
                0
            }
        },
    }
}

/// Outcome of checking for matched tokens - used as guard pattern
/// to ensure state transitions are handled on all return paths.
///
/// Retained whole here (rather than collapsed into a pin-flavored outcome):
/// the P2.1 `update_pin` translation is responsible for mapping this into its
/// own type. `compute_outcome` only ever produces `InProgress` / `Found`.
pub enum MatchCheckOutcome {
    /// Still searching/staging - stay in PreparingToOnboard
    InProgress,
    /// No match possible (not enough tokens, or search found 0) - transition to Inactive.
    ///
    /// Not produced by the reconcile core (`compute_outcome` reports
    /// `Found { matched_tokens: 0 }` for the no-usable-match case); kept to
    /// preserve the enum's contract for the P2.1 caller. The ported tests
    /// only match on it, so it is never constructed in this module.
    NoMatch,
    /// Found matches - transition to Onboarding
    Found { matched_tokens: usize },
}

// ============================================================================
// Reconcile core
// ============================================================================

/// Compute the exclusive upper-block-index of the search range for a given
/// `total_tokens`. Matches the invariant the original `process_match` used:
/// if the token count lands exactly on a block boundary, the last full block
/// is excluded from the search.
fn compute_last_block_index(total_tokens: usize, block_size: usize) -> usize {
    if total_tokens.is_multiple_of(block_size) {
        (total_tokens / block_size).saturating_sub(1)
    } else {
        total_tokens / block_size
    }
}

/// Issue a new find_matches for the given block-index range and return a shard.
///
/// `sequence_hashes` is the slot's full sequence-hash vector; we slice
/// `[start_block .. end_block_exclusive)`. `pub(crate)` so the engine's
/// initial-search path (`tiering::engine`) issues its first shard
/// through the same find policy (`search_remote` + `StagingMode::Full`) the
/// reconcile walk uses, rather than duplicating the options. `search_remote`
/// is the engine's config-driven knob (`remote.search.is_some()` → `true`,
/// `remote.search == None` → `false`); it is forwarded verbatim into the
/// [`FindMatchesOptions`] every shard issues.
pub(crate) fn issue_shard(
    leader: &dyn Leader,
    sequence_hashes: &[SequenceHash],
    start_block: usize,
    end_block_exclusive: usize,
    search_remote: bool,
) -> Result<OnboardingShard> {
    debug_assert!(start_block < end_block_exclusive);
    debug_assert!(end_block_exclusive <= sequence_hashes.len());

    let slice = &sequence_hashes[start_block..end_block_exclusive];
    let options = FindMatchesOptions {
        search_remote,
        staging_mode: StagingMode::Full,
    };

    tracing::debug!(
        start_block,
        end_block_exclusive,
        num_hashes = slice.len(),
        "issuing new shard find_matches_with_options"
    );

    let find_session = leader
        .find_matches_with_options(slice, options)
        .map_err(|e| {
            tracing::error!("Failed to start find operation: {}", e);
            anyhow!("Failed to start find operation: {}", e)
        })?;

    Ok(OnboardingShard {
        start_block,
        num_queried_blocks: end_block_exclusive - start_block,
        find_session,
    })
}

/// Reconcile the stored onboarding state against the latest
/// `num_computed_tokens` and `total_tokens`, issuing new shards as needed,
/// and return the current match outcome.
///
/// Cases (let `old = state.num_computed_tokens`, `new = num_computed_tokens`,
/// `bs = block_size`):
///   * **A** — `new == old` and `total_tokens` unchanged: no-op.
///   * **B** — `new > old`: update `state.num_computed_tokens = new`. Shards
///     are untouched; the completion walk masks the prefix via `effective_start`.
///   * **C** — `new < old`: prepend a new shard covering
///     `[new/bs .. shards[0].start_block)`.
///   * **D** — `current_last_block_index > shards.last().end_block()`: append
///     a new shard for `[shards.last().end_block() .. current_last_block_index)`.
///
/// B+D and C+D may apply in the same call.
pub(crate) fn reconcile_state(
    state: &mut OnboardingState,
    num_computed_tokens: usize,
    total_tokens: usize,
    block_size: usize,
    sequence_hashes: &[SequenceHash],
    leader: &dyn Leader,
    search_remote: bool,
) -> Result<()> {
    debug_assert!(!state.shards.is_empty());
    state.debug_assert_contiguous();

    let old = state.num_computed_tokens;
    let new = num_computed_tokens;

    // Contract asserts (unchanged from legacy behavior): vLLM always passes
    // block-aligned values. A mismatch here is a caller-side contract breach.
    assert!(
        new.is_multiple_of(block_size),
        "num_computed_tokens {} must be a multiple of block_size {}",
        new,
        block_size
    );

    // Case B: new > old ----------------------------------------------------
    if new > old {
        let delta = new - old;
        assert!(
            delta.is_multiple_of(block_size),
            "num_computed_tokens delta {} -> {} must be a multiple of block_size {}",
            old,
            new,
            block_size,
        );
        tracing::debug!(
            old,
            new,
            "num_computed_tokens increased; masking prefix via effective_start"
        );
        state.num_computed_tokens = new;
    }

    // Case C: new < old ----------------------------------------------------
    if new < old {
        let delta = old - new;
        assert!(
            delta.is_multiple_of(block_size),
            "num_computed_tokens delta {} -> {} must be a multiple of block_size {}",
            old,
            new,
            block_size,
        );
        let new_start_block = new / block_size;
        let current_head_start = state.shards[0].start_block;
        debug_assert!(new_start_block <= current_head_start);

        if new_start_block < current_head_start {
            tracing::debug!(
                old,
                new,
                new_start_block,
                current_head_start,
                "num_computed_tokens decreased; prepending prefix shard"
            );
            let new_shard = issue_shard(
                leader,
                sequence_hashes,
                new_start_block,
                current_head_start,
                search_remote,
            )?;
            state.shards.insert(0, new_shard);
        }
        state.num_computed_tokens = new;
    }

    // Case D: total_tokens grew -------------------------------------------
    let current_last_block_index = compute_last_block_index(total_tokens, block_size);
    let existing_end = state.shards.last().unwrap().end_block();
    if current_last_block_index > existing_end {
        tracing::debug!(
            existing_end,
            current_last_block_index,
            "total_tokens grew (eviction restore); appending suffix shard"
        );
        let new_shard = issue_shard(
            leader,
            sequence_hashes,
            existing_end,
            current_last_block_index,
            search_remote,
        )?;
        state.shards.push(new_shard);
    }

    state.debug_assert_contiguous();
    Ok(())
}

/// Compute the outcome from a reconciled onboarding state.
pub(crate) fn compute_outcome(state: &OnboardingState, block_size: usize) -> MatchCheckOutcome {
    // Step 1: any shard still working? Return InProgress.
    if !state.all_shards_terminal() {
        tracing::trace!("Find operation still in progress (some shards non-terminal)");
        return MatchCheckOutcome::InProgress;
    }

    // Step 2: walk contiguously with first-hole short-circuit.
    let (effective_start, final_end) = state.matched_span(block_size);
    let matched_tokens = final_end.saturating_sub(effective_start) * block_size;

    tracing::debug!(
        effective_start,
        final_end,
        matched_tokens,
        num_shards = state.shards.len(),
        "Find completed (walk)"
    );

    if matched_tokens == 0 {
        // No external blocks usable — either the prefix had a hole, or the
        // `effective_start` ate the whole range. Flow to Inactive via Found{0}.
        MatchCheckOutcome::Found { matched_tokens: 0 }
    } else {
        MatchCheckOutcome::Found { matched_tokens }
    }
}

// ============================================================================
// Reconciliation unit tests
// ============================================================================
//
// These tests exercise `reconcile_state` and `compute_outcome` directly via a
// minimal `TestLeader` stub of the `Leader` trait. They cover Cases A/B/C/D
// from the design and the multi-shard walk + first-hole semantics. Async
// shard variants construct `AsyncSessionResult`s directly from the public
// `::new` constructor with pre-filled status/blocks, so the tests don't
// require the kvbm-engine `testing` feature.
//
// All coordinates below are ABSOLUTE block indices over the full
// `sequence_hashes` vector (see module docs).

#[cfg(test)]
mod reconcile_tests {
    use super::*;
    use crate::leader::{
        AsyncSessionResult, FindMatchesResult, MatchBreakdown, OnboardingStatus, ReadyResult,
        SessionId,
    };
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;
    use tokio::sync::{Mutex as TokioMutex, watch};
    use uuid::Uuid;

    const BS: usize = 4;

    /// A stub `Leader` that returns canned `FindMatchesResult`s in queue
    /// order. Each call consumes the next item from `responses`. If the queue
    /// is exhausted, the call panics — tests should provide exactly the
    /// number of canned results they expect.
    struct TestLeader {
        responses: StdMutex<Vec<FindMatchesResult>>,
        calls: StdMutex<Vec<usize>>,
    }

    impl TestLeader {
        fn new(responses: Vec<FindMatchesResult>) -> Self {
            Self {
                responses: StdMutex::new(responses),
                calls: StdMutex::new(Vec::new()),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    impl Leader for TestLeader {
        fn find_matches_with_options(
            &self,
            sequence_hashes: &[SequenceHash],
            _options: FindMatchesOptions,
        ) -> Result<FindMatchesResult> {
            self.calls.lock().unwrap().push(sequence_hashes.len());
            let mut q = self.responses.lock().unwrap();
            assert!(
                !q.is_empty(),
                "TestLeader: unexpected find_matches_with_options call ({} hashes)",
                sequence_hashes.len()
            );
            Ok(q.remove(0))
        }
    }

    /// Build a vector of dummy `SequenceHash` values. The tests don't depend
    /// on actual hash content, only on slice indices.
    fn dummy_hashes(n: usize) -> Vec<SequenceHash> {
        (0..n as u64)
            .map(|i| SequenceHash::new(i, None, i))
            .collect()
    }

    /// Construct a Ready shard result with `g2_count` empty G2 placeholders.
    /// Since `ImmutableBlock<G2>` cannot be constructed outside kvbm-logical,
    /// this only works for `g2_count == 0`. Callers that need a non-zero
    /// match count should use `complete_async(matched)` instead.
    fn ready_zero() -> FindMatchesResult {
        FindMatchesResult::Ready(ReadyResult::new(vec![], MatchBreakdown::default()))
    }

    /// Construct a terminal AsyncSession with `matched_blocks` reported via
    /// the watch channel. The blocks vec is empty (only the count matters
    /// for reconciliation/walk logic).
    fn complete_async(matched: usize) -> FindMatchesResult {
        // Drop the sender so the channel stays at its latest value; receivers
        // will still observe `Complete { matched_blocks: matched }`.
        let (status_tx, status_rx) = watch::channel(OnboardingStatus::Complete {
            matched_blocks: matched,
        });
        drop(status_tx);
        FindMatchesResult::AsyncSession(AsyncSessionResult::new(
            SessionId::from(Uuid::nil()),
            status_rx,
            Arc::new(TokioMutex::new(Some(Vec::new()))),
            Arc::new(TokioMutex::new(MatchBreakdown::default())),
        ))
    }

    /// Construct a pending AsyncSession (status = Searching). Returns the
    /// session and the watch sender so tests can transition it later.
    fn pending_async() -> (FindMatchesResult, watch::Sender<OnboardingStatus>) {
        let (status_tx, status_rx) = watch::channel(OnboardingStatus::Searching);
        let session = AsyncSessionResult::new(
            SessionId::from(Uuid::nil()),
            status_rx,
            Arc::new(TokioMutex::new(None)),
            Arc::new(TokioMutex::new(MatchBreakdown::default())),
        );
        (FindMatchesResult::AsyncSession(session), status_tx)
    }

    /// Build a fresh `OnboardingState` with a single shard.
    fn make_state(
        num_computed_tokens: usize,
        total_tokens_at_start: usize,
        start_block: usize,
        num_queried_blocks: usize,
        find_session: FindMatchesResult,
    ) -> OnboardingState {
        OnboardingState::new(
            num_computed_tokens,
            total_tokens_at_start,
            OnboardingShard {
                start_block,
                num_queried_blocks,
                find_session,
            },
        )
    }

    // ------------------------------------------------------------------
    // Case A — no change
    // ------------------------------------------------------------------

    #[test]
    fn case_a_unchanged_in_progress() {
        // single shard, async pending. Expect InProgress, shards untouched.
        let (pending, _tx) = pending_async();
        let mut state = make_state(40, 64, 10, 5, pending);
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        reconcile_state(&mut state, 40, 64, BS, &hashes, &leader, true).unwrap();
        assert_eq!(leader.call_count(), 0);
        assert_eq!(state.shards.len(), 1);
        assert!(matches!(
            compute_outcome(&state, BS),
            MatchCheckOutcome::InProgress
        ));
    }

    #[test]
    fn case_a_unchanged_terminal_full_match() {
        // single shard, async complete with all 5 blocks. Expect Found{5*BS}.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        reconcile_state(&mut state, 40, 64, BS, &hashes, &leader, true).unwrap();
        assert_eq!(leader.call_count(), 0);
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 5 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    // ------------------------------------------------------------------
    // Case B — new > old (mask via effective_start)
    // ------------------------------------------------------------------

    #[test]
    fn case_b_mask_partial() {
        // Shard at start_block=10, num_queried=5, all 5 matched.
        // num_computed grows from 40 (=10*BS) to 48 (=12*BS): mask 2 blocks.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        reconcile_state(&mut state, 48, 64, BS, &hashes, &leader, true).unwrap();
        assert_eq!(leader.call_count(), 0);
        assert_eq!(state.num_computed_tokens, 48);
        assert_eq!(state.shards.len(), 1);

        // effective_start = max(10, 12) = 12; final_end = 15; matched = 3*BS
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 3 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_b_mask_consumes_all() {
        // Mask covers the entire matched range -> 0 effective external tokens.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        reconcile_state(&mut state, 60, 64, BS, &hashes, &leader, true).unwrap();
        // effective_start = 15, final_end = 15 -> 0
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 0),
            other => panic!("expected Found{{0}}, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    #[should_panic(expected = "must be a multiple of block_size")]
    fn case_b_unaligned_delta_panics() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        // 42 is not a multiple of BS=4 -> the top-level alignment assert fires.
        reconcile_state(&mut state, 42, 64, BS, &hashes, &leader, true).unwrap();
    }

    // ------------------------------------------------------------------
    // Case C — new < old (prepend prefix shard)
    // ------------------------------------------------------------------

    #[test]
    fn case_c_prepend_full_match() {
        // shard at start_block=10, num=5, complete with 5 matches.
        // num_computed drops from 40 to 32 -> prepend shard [8..10) of size 2.
        // Test leader returns a fully-matched (2/2) async response for the prefix.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![complete_async(2)]);

        reconcile_state(&mut state, 32, 64, BS, &hashes, &leader, true).unwrap();
        assert_eq!(leader.call_count(), 1);
        assert_eq!(state.num_computed_tokens, 32);
        assert_eq!(state.shards.len(), 2);
        assert_eq!(state.shards[0].start_block, 8);
        assert_eq!(state.shards[0].num_queried_blocks, 2);
        assert_eq!(state.shards[1].start_block, 10);

        // effective_start = max(8, 8) = 8; running_end walks 8 -> 10 -> 15.
        // matched = (15 - 8)*BS = 28.
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 7 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_c_prefix_hole_zero_match() {
        // Prefix shard returns a hole (1 of 2 matched). Walk short-circuits at
        // running_end + 1 = 9. effective_start = 8. final_end = 9. matched = 1*BS.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![complete_async(1)]);

        reconcile_state(&mut state, 32, 64, BS, &hashes, &leader, true).unwrap();
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, BS),
            other => panic!("expected Found{{BS}}, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_c_prefix_zero_match() {
        // Prefix shard returns 0 matches -> full hole at the very start.
        // running_end starts at 8, matched=0 -> final_end=8. effective_start=8.
        // matched = 0.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![complete_async(0)]);

        reconcile_state(&mut state, 32, 64, BS, &hashes, &leader, true).unwrap();
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 0),
            other => panic!("expected Found{{0}}, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_c_prefix_pending_keeps_in_progress() {
        // Old suffix terminal, new prefix in progress -> InProgress.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let (pending, _tx) = pending_async();
        let leader = TestLeader::new(vec![pending]);

        reconcile_state(&mut state, 32, 64, BS, &hashes, &leader, true).unwrap();
        assert!(matches!(
            compute_outcome(&state, BS),
            MatchCheckOutcome::InProgress
        ));
    }

    #[test]
    fn case_c_full_preemption_no_special_logging() {
        // num_computed_tokens drops to 0. Should be handled identically to any
        // other Case C — prepend a shard for the full prefix.
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![complete_async(10)]);

        reconcile_state(&mut state, 0, 64, BS, &hashes, &leader, true).unwrap();
        assert_eq!(state.num_computed_tokens, 0);
        assert_eq!(state.shards.len(), 2);
        assert_eq!(state.shards[0].start_block, 0);
        assert_eq!(state.shards[0].num_queried_blocks, 10);
        // effective_start = max(0, 0) = 0; full match across both shards = 15 * BS
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 15 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    #[should_panic(expected = "must be a multiple of block_size")]
    fn case_c_unaligned_delta_panics() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);
        // 38 is not a multiple of BS=4.
        reconcile_state(&mut state, 38, 64, BS, &hashes, &leader, true).unwrap();
    }

    // ------------------------------------------------------------------
    // Case D — total_tokens grew (eviction restore)
    // ------------------------------------------------------------------

    #[test]
    fn case_d_appends_upper_shard() {
        // Original total_tokens=64 -> last_block_index=15; shard covers [10..15).
        // total_tokens grows to 96 -> last_block_index=23. Append [15..23).
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        let leader = TestLeader::new(vec![complete_async(8)]);

        reconcile_state(&mut state, 40, 96, BS, &hashes, &leader, true).unwrap();
        assert_eq!(leader.call_count(), 1);
        assert_eq!(state.shards.len(), 2);
        assert_eq!(state.shards[1].start_block, 15);
        assert_eq!(state.shards[1].num_queried_blocks, 8);
        // walk: 10 -> 15 (shard0 full) -> 23 (shard1 full). effective_start=10. matched=13*BS.
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 13 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_d_pending_upper_shard() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        let (pending, _tx) = pending_async();
        let leader = TestLeader::new(vec![pending]);

        reconcile_state(&mut state, 40, 96, BS, &hashes, &leader, true).unwrap();
        assert!(matches!(
            compute_outcome(&state, BS),
            MatchCheckOutcome::InProgress
        ));
    }

    // ------------------------------------------------------------------
    // Combinations: B+D, C+D
    // ------------------------------------------------------------------

    #[test]
    fn case_b_plus_d() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        let leader = TestLeader::new(vec![complete_async(8)]);

        // num grows by 8 (mask 2), total grows -> append shard [15..23) full
        reconcile_state(&mut state, 48, 96, BS, &hashes, &leader, true).unwrap();
        // effective_start = max(10, 12) = 12; final_end = 23 -> matched = 11*BS
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 11 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn case_c_plus_d() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        // First call: prefix [8..10) (Case C), then second: upper [15..23) (Case D).
        let leader = TestLeader::new(vec![complete_async(2), complete_async(8)]);

        reconcile_state(&mut state, 32, 96, BS, &hashes, &leader, true).unwrap();
        assert_eq!(leader.call_count(), 2);
        assert_eq!(state.shards.len(), 3);
        // walk: 8 -> 10 -> 15 -> 23. effective_start=8. matched = 15*BS.
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 15 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    // ------------------------------------------------------------------
    // Walk semantics edge cases
    // ------------------------------------------------------------------

    #[test]
    fn walk_short_circuits_on_first_hole() {
        // Two terminal shards; first has a hole (matched < num_queried).
        // Second shard's match count is irrelevant past the hole.
        let mut state = make_state(40, 64, 10, 5, complete_async(3));
        // Pre-attach a second shard manually for this test.
        state.shards.push(OnboardingShard {
            start_block: 15,
            num_queried_blocks: 3,
            find_session: complete_async(3),
        });
        // walk: 10 -> 13 (hole). effective_start = 10. matched = 3*BS.
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 3 * BS),
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn walk_one_pending_shard_returns_in_progress() {
        // Two shards: first terminal (full), second pending. Step-1 returns
        // InProgress (no partial reporting).
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let (pending, _tx) = pending_async();
        state.shards.push(OnboardingShard {
            start_block: 15,
            num_queried_blocks: 3,
            find_session: pending,
        });
        assert!(matches!(
            compute_outcome(&state, BS),
            MatchCheckOutcome::InProgress
        ));
    }

    // ------------------------------------------------------------------
    // Invariants and metric plumbing
    // ------------------------------------------------------------------

    #[test]
    fn shard_contiguity_invariant_after_reconcile() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        let leader = TestLeader::new(vec![complete_async(2), complete_async(8)]);

        reconcile_state(&mut state, 32, 96, BS, &hashes, &leader, true).unwrap();
        // debug_assert_contiguous would have panicked if the invariant was
        // violated; call it explicitly to be sure.
        state.debug_assert_contiguous();
        assert_eq!(state.shards[0].end_block(), state.shards[1].start_block);
        assert_eq!(state.shards[1].end_block(), state.shards[2].start_block);
    }

    #[test]
    fn total_query_blocks_sums_across_shards() {
        let mut state = make_state(40, 64, 10, 5, complete_async(5));
        let hashes = dummy_hashes(30);
        let leader = TestLeader::new(vec![complete_async(2), complete_async(8)]);

        reconcile_state(&mut state, 32, 96, BS, &hashes, &leader, true).unwrap();
        assert_eq!(state.total_query_blocks(), 2 + 5 + 8);
    }

    /// Direct repro of the scenario reported in #5285: at bs=16, vLLM polled
    /// `get_num_new_matched_tokens` first with 21536 and then with 21520 (a
    /// 16-token == 1-block decrease, i.e. one G1 block was evicted between
    /// scheduler passes). The legacy code panicked in debug here. With
    /// reconciliation, we simply prepend a one-block prefix shard and report
    /// a (potentially) extended match without losing the suffix work.
    #[test]
    fn issue_5285_repro_one_block_eviction() {
        const REAL_BS: usize = 16;
        // total_tokens beyond the first poll's range; pick something that gives
        // headroom and a stable last_block_index.
        let total_tokens = 24_000_usize;
        let last_block_index = total_tokens / REAL_BS;
        // First poll: num_computed_tokens=21536 -> shard at 21536/16 = 1346,
        // covering [1346 .. last_block_index). Suppose all blocks matched.
        let num_blocks_first = last_block_index - 1346;
        let mut state = OnboardingState::new(
            21536,
            total_tokens,
            OnboardingShard {
                start_block: 1346,
                num_queried_blocks: num_blocks_first,
                find_session: complete_async(num_blocks_first),
            },
        );
        let hashes = dummy_hashes(last_block_index);

        // Second poll: 21520 = 1345 * 16. We expect a new prefix shard
        // [1345..1346) of size 1.
        let leader = TestLeader::new(vec![complete_async(1)]);
        reconcile_state(
            &mut state,
            21520,
            total_tokens,
            REAL_BS,
            &hashes,
            &leader,
            true,
        )
        .unwrap();

        assert_eq!(leader.call_count(), 1);
        assert_eq!(state.shards.len(), 2);
        assert_eq!(state.shards[0].start_block, 1345);
        assert_eq!(state.shards[0].num_queried_blocks, 1);

        // effective_start = max(1345, 21520/16=1345) = 1345
        // final_end = last_block_index
        // matched = (last_block_index - 1345) * 16
        let expected = (last_block_index - 1345) * REAL_BS;
        match compute_outcome(&state, REAL_BS) {
            MatchCheckOutcome::Found { matched_tokens } => {
                assert_eq!(matched_tokens, expected)
            }
            other => panic!("expected Found, got {:?}", other_repr(&other)),
        }
    }

    #[test]
    fn ready_zero_match_returns_found_zero() {
        // A Ready variant with no matched blocks -> Found{0}. (This is the
        // legacy single-shard happy path with an empty result.)
        let mut state = make_state(40, 64, 10, 5, ready_zero());
        let hashes = dummy_hashes(20);
        let leader = TestLeader::new(vec![]);

        reconcile_state(&mut state, 40, 64, BS, &hashes, &leader, true).unwrap();
        match compute_outcome(&state, BS) {
            MatchCheckOutcome::Found { matched_tokens } => assert_eq!(matched_tokens, 0),
            other => panic!("expected Found{{0}}, got {:?}", other_repr(&other)),
        }
    }

    fn other_repr(o: &MatchCheckOutcome) -> &'static str {
        match o {
            MatchCheckOutcome::InProgress => "InProgress",
            MatchCheckOutcome::NoMatch => "NoMatch",
            MatchCheckOutcome::Found { .. } => "Found",
        }
    }

    /// `issue_shard` forwards the `search_remote` flag verbatim into the
    /// `FindMatchesOptions` the leader receives. A stub leader records the flag
    /// of each find; the engine's config-driven knob must reach the leader
    /// unchanged for both values.
    #[test]
    fn issue_shard_forwards_search_remote_flag() {
        /// Records the `search_remote` field of every find it is asked to run.
        struct RecordingLeader {
            seen: StdMutex<Vec<bool>>,
        }
        impl Leader for RecordingLeader {
            fn find_matches_with_options(
                &self,
                _sequence_hashes: &[SequenceHash],
                options: FindMatchesOptions,
            ) -> Result<FindMatchesResult> {
                self.seen.lock().unwrap().push(options.search_remote);
                Ok(ready_zero())
            }
        }

        let hashes = dummy_hashes(4);
        let leader = RecordingLeader {
            seen: StdMutex::new(Vec::new()),
        };

        issue_shard(&leader, &hashes, 0, 2, false).unwrap();
        issue_shard(&leader, &hashes, 0, 2, true).unwrap();

        assert_eq!(*leader.seen.lock().unwrap(), vec![false, true]);
    }
}
