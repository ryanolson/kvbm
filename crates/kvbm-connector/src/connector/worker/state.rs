// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side completion + eviction-fence state for connector.
//!
//! The engine PUSHES per-request terminals into this state OFF the forward-pass
//! thread (via the injected [`kvbm_protocols::connector::EngineWorkerSink`],
//! whose connector-side impl is [`WorkerSink`]); the worker's forward-pass
//! thread DRAINS them ([`WorkerCompletionState::drain_finished`] /
//! [`WorkerCompletionState::drain_failed`]) and BLOCKS on
//! [`WorkerCompletionState::await_fences`] until the engine has drained every
//! action captured at eviction time. The per-step block point is the
//! iteration-idempotent [`WorkerCompletionState::ensure_fences_awaited`]
//! funnel: both worker hooks that can carry a step's fences route through it,
//! so the step awaits once regardless of which hook runs first.
//!
//! ## `await_fence` home (decision)
//!
//! REFACTOR.md §5 offers two homes for the forward-pass fence wait: the engine's
//! [`kvbm_protocols::connector::WorkerEngineDriver::await_fence`], or the
//! worker's own completion state. We wait on the worker's OWN
//! `WorkerCompletionState` here. The engine's `mark_fence_complete` push lands
//! in this same struct's `completed_fences` set + [`Condvar`], so the
//! `await_fences` waiter wakes with NO leader/engine round-trip — exactly §5's
//! "the `await_fence` waiter reads its local `WorkerCompletionState`, no leader
//! round-trip" clarification. `WorkerEngineDriver::await_fence` is therefore
//! intentionally NOT called by the connector worker (the worker holds the driver
//! only for the forward-pass boundaries + shutdown).
//!
//! ## Off-forward-pass-thread guard
//!
//! `await_fences` blocks the model-runner (forward-pass) thread. If a completion
//! were ever driven ON that same thread it could only run after `await_fences`
//! returned → self-deadlock. The worker records the forward-pass thread id (see
//! [`WorkerCompletionState::mark_forward_pass_thread`]); every `record_*`
//! completion entry `debug_assert`s it is NOT running on that thread, turning the
//! footgun into a loud debug failure rather than a silent hang.

use std::collections::HashSet;
use std::sync::Arc;
use std::thread::ThreadId;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use kvbm_protocols::connector::{
    EngineWorkerSink, FenceToken, LoadOutcome, RequestId, SaveOutcome,
};

use super::FinishedRequests;

/// Mutex-guarded interior of [`WorkerCompletionState`]. One lock guards every
/// set so the load-failure pairing (`finished_recving` + `failed_onboarding`)
/// and the fence insert + predicate check are each atomic.
#[derive(Default)]
struct Inner {
    /// Requests whose offload (save) reached its single per-request terminal.
    /// Drained into `FinishedRequests::offloading`.
    finished_sending: HashSet<RequestId>,
    /// Requests whose onboard (load) reached a terminal — success OR failure.
    /// A failed load still lands here so vLLM leaves `WAITING_FOR_REMOTE_KVS`.
    /// Drained into `FinishedRequests::onboarding`.
    finished_recving: HashSet<RequestId>,
    /// G1 block ids that failed to load. NOTE: this is block ids (`usize`), not
    /// request ids — it backs the binding's
    /// `ConnectorWorkerInterface::get_failed_onboarding() -> HashSet<usize>`
    /// (vLLM's `get_block_ids_with_load_errors`). Paired with the request's
    /// entry in `finished_recving` under one lock (legacy `mark_failed_onboarding`
    /// contract): vLLM needs the request surfaced in the same pass as its errors.
    failed_onboarding: HashSet<usize>,
    /// Eviction-fence tokens the engine has reported drained. Read by the
    /// `await_fences` predicate and CONSUMED by it: a satisfied await removes
    /// the tokens it waited on, so the set never accumulates per-(generation,
    /// worker) UUIDs across the process lifetime.
    completed_fences: HashSet<FenceToken>,
    /// The iteration whose fences [`WorkerCompletionState::ensure_fences_awaited`]
    /// already awaited+took — the second caller in the same step no-ops on it.
    fenced_iteration: Option<u64>,
    /// The model-runner (forward-pass) thread, recorded by
    /// `mark_forward_pass_thread`. Arms the off-thread completion guard.
    forward_pass_thread: Option<ThreadId>,
}

/// Worker-owned completion + fence state. Shared as `Arc<WorkerCompletionState>`
/// between the worker (drains / awaits) and the injected [`WorkerSink`] (the
/// engine's push face).
pub struct WorkerCompletionState {
    inner: Mutex<Inner>,
    /// Woken by `record_fence_complete`; waited on by `await_fences`.
    fence_cv: Condvar,
}

impl Default for WorkerCompletionState {
    fn default() -> Self {
        Self::new()
    }
}

/// How long the forward-pass fence wait blocks between diagnostic warns. Purely
/// observational: crossing it emits a warn naming the still-pending tokens and
/// then KEEPS WAITING — it never breaks the barrier (see
/// [`WorkerCompletionState::ensure_fences_awaited_with_interval`]).
const FENCE_DIAGNOSTIC_INTERVAL: Duration = Duration::from_secs(30);

impl WorkerCompletionState {
    /// Fresh, empty state.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            fence_cv: Condvar::new(),
        }
    }

    /// Record the calling thread as the forward-pass (model-runner) thread, so
    /// the off-thread guard in the `record_*` path can fire if a completion is
    /// ever driven ON the thread that `await_fences` blocks. Idempotent: the
    /// model-runner thread is stable, so re-recording overwrites with the same
    /// id. Called from the worker's per-step forward-pass entry.
    pub fn mark_forward_pass_thread(&self) {
        self.inner.lock().forward_pass_thread = Some(std::thread::current().id());
    }

    /// Off-forward-pass-thread guard. Completions are engine-driven and MUST run
    /// off the forward-pass thread (REFACTOR.md §3); running one on that thread
    /// self-deadlocks `await_fences`. Debug-only; inert until the forward-pass
    /// thread is recorded.
    fn assert_off_forward_pass_thread(inner: &Inner) {
        if let Some(fp) = inner.forward_pass_thread {
            debug_assert!(
                fp != std::thread::current().id(),
                "engine→worker completion fired on the forward-pass thread; \
                 completions must be driven off-thread or await_fences self-deadlocks",
            );
        }
    }

    /// Onboard (load) terminal for `req`. Always surfaces `req` in
    /// `finished_recving` (success or failure — vLLM must see it to leave
    /// `WAITING_FOR_REMOTE_KVS`); on `FailedPartial`, also records the failed G1
    /// block ids (a total failure arrives already resolved to the load's full
    /// dest set — `LoadOutcome` has no id-less failure). One lock keeps the
    /// recving + failed inserts paired.
    pub fn record_load_finished(&self, req: &RequestId, outcome: LoadOutcome) {
        let mut inner = self.inner.lock();
        Self::assert_off_forward_pass_thread(&inner);
        inner.finished_recving.insert(req.clone());
        if let LoadOutcome::FailedPartial { block_ids } = outcome {
            inner.failed_onboarding.extend(block_ids);
        }
    }

    /// Offload (save) terminal for `req` → `finished_sending`. Fired exactly once
    /// per request (leader-gated by `RequestOffloadDrain`, never the per-action
    /// driver — REFACTOR.md §3/B), so a plain insert is correct.
    pub fn record_save_finished(&self, req: &RequestId, _outcome: SaveOutcome) {
        let mut inner = self.inner.lock();
        Self::assert_off_forward_pass_thread(&inner);
        inner.finished_sending.insert(req.clone());
    }

    /// Mark `token`'s eviction drain complete and wake any `await_fences` waiter.
    pub fn record_fence_complete(&self, token: FenceToken) {
        let mut inner = self.inner.lock();
        Self::assert_off_forward_pass_thread(&inner);
        inner.completed_fences.insert(token);
        drop(inner);
        self.fence_cv.notify_all();
    }

    /// Drain (consume-once) the finished sets. `offloading` = `finished_sending`,
    /// `onboarding` = `finished_recving`, matching the binding's
    /// `(sending, recving)` field order. A second call returns empty sets.
    pub fn drain_finished(&self) -> FinishedRequests {
        let mut inner = self.inner.lock();
        FinishedRequests {
            offloading: std::mem::take(&mut inner.finished_sending),
            onboarding: std::mem::take(&mut inner.finished_recving),
        }
    }

    /// Drain (consume-once) the failed-load G1 block ids.
    pub fn drain_failed(&self) -> HashSet<usize> {
        std::mem::take(&mut self.inner.lock().failed_onboarding)
    }

    /// Block the calling (forward-pass) thread until EVERY token in `tokens` is
    /// complete, then TAKE those tokens out of `completed_fences`. Unbounded BY
    /// DESIGN: the eviction barrier must hold until the engine drains the
    /// captured actions (a never-completing token is a wiring bug, not a
    /// timeout case). Returns immediately when `tokens` is empty.
    ///
    /// Take-on-await is the leak fix: tokens are per-(generation, worker)
    /// UUIDs minted fresh on every eviction, so a token left behind after its
    /// await is garbage forever. Each delivered token is awaited (and so
    /// taken) exactly once — see [`Self::ensure_fences_awaited`].
    pub fn await_fences(&self, tokens: &[FenceToken]) {
        if tokens.is_empty() {
            return;
        }
        let mut inner = self.inner.lock();
        while !tokens.iter().all(|t| inner.completed_fences.contains(t)) {
            self.fence_cv.wait(&mut inner);
        }
        Self::take_fences(&mut inner, tokens);
    }

    /// Iteration-idempotent funnel over [`Self::await_fences`] — the SINGLE
    /// block point for a step's fences, callable from either worker hook.
    /// vLLM drives `handle_preemptions` AND the metadata bind every step (in
    /// runner-dependent order), both carrying the same sealed envelope: the
    /// FIRST caller for `iteration` awaits+takes this rank's tokens and
    /// records the iteration; the second caller no-ops on the recorded
    /// iteration. That pairing is the leak invariant: every delivered token
    /// is awaited+taken exactly once, because one of the two callers always
    /// runs and the other always funnels into the same recorded step.
    pub fn ensure_fences_awaited(&self, iteration: u64, tokens: &[FenceToken]) {
        self.ensure_fences_awaited_with_interval(iteration, tokens, FENCE_DIAGNOSTIC_INTERVAL);
    }

    /// Diagnostic-interval-parameterized body of [`Self::ensure_fences_awaited`]
    /// (the `interval` argument lets tests drive the warn path with a short
    /// period). The wait remains effectively UNBOUNDED: `interval` only bounds
    /// each INNER `wait_for` so that, after the threshold, a `warn!` naming the
    /// still-pending tokens fires and the loop goes BACK to waiting. The sole
    /// loop-exit is the all-tokens-present predicate; there is no break, no
    /// early return, and no proceed-on-timeout — the eviction fence gates G1
    /// REUSE, so proceeding past an incomplete fence would let the forward pass
    /// overwrite a block whose transfer is still draining (memory corruption).
    fn ensure_fences_awaited_with_interval(
        &self,
        iteration: u64,
        tokens: &[FenceToken],
        interval: Duration,
    ) {
        let mut inner = self.inner.lock();
        if inner.fenced_iteration == Some(iteration) {
            return;
        }
        // Record BEFORE blocking: both callers run sequentially on the
        // forward-pass thread, so nothing can race the wait — and a re-entry
        // for the same step must see the claim even mid-wait.
        inner.fenced_iteration = Some(iteration);
        let start = std::time::Instant::now();
        let mut warned = false;
        while !tokens.iter().all(|t| inner.completed_fences.contains(t)) {
            if self.fence_cv.wait_for(&mut inner, interval).timed_out()
                && !tokens.iter().all(|t| inner.completed_fences.contains(t))
            {
                let pending: Vec<FenceToken> = tokens
                    .iter()
                    .filter(|t| !inner.completed_fences.contains(t))
                    .copied()
                    .collect();
                tracing::warn!(
                    iteration,
                    elapsed_secs = start.elapsed().as_secs(),
                    pending_count = pending.len(),
                    ?pending,
                    "fence-await still blocked on undrained eviction fences; the \
                     forward pass is held BY DESIGN until every fence drains (G1 \
                     reuse barrier) — a token stuck here indicates an undelivered \
                     engine fence completion",
                );
                warned = true;
            }
        }
        if warned {
            tracing::info!(
                iteration,
                elapsed_secs = start.elapsed().as_secs(),
                "fence-await unblocked; all eviction fences drained",
            );
        }
        Self::take_fences(&mut inner, tokens);
    }

    /// Consume awaited tokens out of `completed_fences` (the take half of
    /// take-on-await). Caller holds the lock and has seen every token present.
    fn take_fences(inner: &mut Inner, tokens: &[FenceToken]) {
        for token in tokens {
            inner.completed_fences.remove(token);
        }
    }

    /// Bounded variant of [`await_fences`](Self::await_fences) for deterministic
    /// tests: returns `true` once every token is complete (taking them, like
    /// the unbounded variant), or `false` if `timeout` elapses first (so a
    /// test fails on timeout instead of hanging). Test-only — production
    /// blocks unboundedly on the barrier.
    #[cfg(test)]
    pub fn try_await_fences(&self, tokens: &[FenceToken], timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let mut inner = self.inner.lock();
        loop {
            if tokens.iter().all(|t| inner.completed_fences.contains(t)) {
                Self::take_fences(&mut inner, tokens);
                return true;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }
            self.fence_cv.wait_for(&mut inner, deadline - now);
        }
    }

    /// Test-only read: is `token` still sitting in `completed_fences`? The
    /// take-on-await reproducers pin the leak fix on this.
    #[cfg(test)]
    pub fn fence_recorded(&self, token: &FenceToken) -> bool {
        self.inner.lock().completed_fences.contains(token)
    }
}

/// Connector-side impl of the engine→worker completion push. Holds the worker's
/// `Arc<WorkerCompletionState>` and forwards each engine terminal to the matching
/// `record_*`. The worker hands one of these (as `Arc<dyn EngineWorkerSink>`) to
/// the engine at construction (P-D1b); the engine drives it OFF the forward-pass
/// thread.
pub struct WorkerSink {
    state: Arc<WorkerCompletionState>,
}

impl WorkerSink {
    /// Wrap a shared [`WorkerCompletionState`] as the engine's push face.
    pub fn new(state: Arc<WorkerCompletionState>) -> Self {
        Self { state }
    }
}

impl EngineWorkerSink for WorkerSink {
    fn mark_load_finished(&self, req: &RequestId, outcome: LoadOutcome) {
        self.state.record_load_finished(req, outcome);
    }

    fn mark_save_finished(&self, req: &RequestId, outcome: SaveOutcome) {
        self.state.record_save_finished(req, outcome);
    }

    fn mark_fence_complete(&self, token: FenceToken) {
        self.state.record_fence_complete(token);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    // (a) Reproducer-first: await_fences must BLOCK until record_fence_complete.
    // A pre-completion bounded wait times out (proves it blocks); a completer
    // thread then fires the token after a bounded delay and the wait returns
    // only after — within a generous timeout, so a regression fails (never hangs).
    #[test]
    fn await_fences_blocks_until_fence_complete() {
        let state = Arc::new(WorkerCompletionState::new());
        let token = FenceToken::new(0);

        // Not yet complete: a bounded wait must report timeout (still blocked).
        assert!(
            !state.try_await_fences(&[token], Duration::from_millis(50)),
            "await must block while the fence is incomplete",
        );

        // Completer fires after a bounded delay on another thread.
        let completer_state = Arc::clone(&state);
        let start = Instant::now();
        let completer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(25));
            completer_state.record_fence_complete(token);
        });

        // The wait returns only after completion, and well within the timeout.
        assert!(
            state.try_await_fences(&[token], Duration::from_secs(2)),
            "await must return once the fence completes",
        );
        assert!(
            start.elapsed() >= Duration::from_millis(20),
            "await returned before the completer could have fired",
        );
        completer.join().unwrap();
    }

    // await_fences with no tokens is an immediate no-op (the common no-eviction
    // step must not block the forward pass).
    #[test]
    fn await_fences_empty_returns_immediately() {
        let state = WorkerCompletionState::new();
        assert!(state.try_await_fences(&[], Duration::from_millis(0)));
        state.await_fences(&[]); // unbounded variant must also return at once.
    }

    // A fence completes only its own token: an await on a still-pending token
    // does not spuriously return when an unrelated token completes.
    #[test]
    fn await_fences_waits_for_all_tokens() {
        let state = Arc::new(WorkerCompletionState::new());
        let a = FenceToken::new(0);
        let b = FenceToken::new(0);
        state.record_fence_complete(a);
        // `a` done, `b` outstanding → must still block.
        assert!(!state.try_await_fences(&[a, b], Duration::from_millis(50)));
        state.record_fence_complete(b);
        assert!(state.try_await_fences(&[a, b], Duration::from_millis(50)));
    }

    // Reproducer (the unawaited-token leak): tokens are per-(generation,
    // worker) UUIDs minted fresh per eviction, so anything left in
    // `completed_fences` after its await is garbage forever. A satisfied
    // await must TAKE its tokens. Pre-fix, the set retained them.
    #[test]
    fn await_fences_takes_completed_tokens() {
        let state = WorkerCompletionState::new();
        let token = FenceToken::new(0);
        state.record_fence_complete(token);
        assert!(state.fence_recorded(&token));

        state.await_fences(&[token]);
        assert!(
            !state.fence_recorded(&token),
            "take-on-await must drain the awaited token from completed_fences"
        );
    }

    // The diagnostic interval is observational ONLY: a short interval makes the
    // warn path fire, but the wait MUST NOT break on timeout. A completer fires
    // the fence well after the interval; the call must keep blocking until then.
    #[test]
    fn ensure_fences_awaited_keeps_blocking_past_diagnostic_interval() {
        let state = Arc::new(WorkerCompletionState::new());
        let token = FenceToken::new(0);

        let completer_state = Arc::clone(&state);
        let completer = std::thread::spawn(move || {
            // >> the 20ms test interval, so at least one diagnostic timeout fires.
            std::thread::sleep(Duration::from_millis(100));
            completer_state.record_fence_complete(token);
        });

        let start = Instant::now();
        state.ensure_fences_awaited_with_interval(1, &[token], Duration::from_millis(20));
        // Returned only AFTER the fence actually landed — proves the diagnostic
        // timeout did not break the wait. Lower-bound only (gated by the
        // completer), so non-flaky.
        assert!(
            start.elapsed() >= Duration::from_millis(90),
            "fence-await returned before the fence completed; diagnostic timeout must NOT break the wait",
        );
        assert!(
            !state.fence_recorded(&token),
            "the awaited token must be taken on completion",
        );
        completer.join().unwrap();
    }

    // The iteration-idempotent funnel: the FIRST caller for an iteration
    // awaits+takes; a SECOND caller with the same iteration no-ops even though
    // the tokens are gone (re-awaiting taken tokens would block forever); a
    // NEW iteration awaits again.
    #[test]
    fn ensure_fences_awaited_is_iteration_idempotent() {
        let state = Arc::new(WorkerCompletionState::new());
        let token = FenceToken::new(0);
        state.record_fence_complete(token);

        // First caller: awaits and takes.
        state.ensure_fences_awaited(3, &[token]);
        assert!(!state.fence_recorded(&token), "first caller took the token");

        // Second caller, same iteration, token GONE: must return without
        // blocking. Run it bounded so a regression fails instead of hanging.
        let second = Arc::clone(&state);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            second.ensure_fences_awaited(3, &[token]);
            tx.send(()).ok();
        });
        assert!(
            rx.recv_timeout(Duration::from_secs(2)).is_ok(),
            "the second caller for an already-awaited iteration must no-op"
        );

        // A fresh iteration is a fresh await: it blocks until ITS token lands,
        // then takes it.
        let t2 = FenceToken::new(0);
        state.record_fence_complete(t2);
        state.ensure_fences_awaited(4, &[t2]);
        assert!(!state.fence_recorded(&t2));
    }
}
