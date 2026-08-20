// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cancellation protocol for offload transfers.
//!
//! The cancellation protocol ensures clean release of all blocks with confirmation
//! that no outstanding operations remain:
//!
//! 1. `cancel()` selects precommit cancellation or committed drain.
//! 2. Each container carries one non-clone work unit.
//! 3. The chain router atomically replaces one unit with its route units.
//! 4. Executors release source guards before terminal status.
//! 5. The final unit release confirms cancellation.

use std::sync::{Arc, Mutex};

use tokio::sync::watch;

use super::handle::TransferState;

const ABANDONED_COMMITTED_ROUTE_ERROR: &str = "committed offload route dropped before settlement";

/// State of a cancellation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CancelState {
    /// Transfer is active, not cancelled
    Active,
    /// Cancel requested, waiting for checkpoint
    Requested,
    /// Draining in-flight operations
    Draining {
        /// Number of logical work units remaining
        in_flight: usize,
    },
    /// All operations complete, blocks released, confirmed
    Confirmed,
}

impl CancelState {
    /// Check if cancellation has been requested (including draining/confirmed states).
    #[cfg(test)]
    pub(crate) fn is_cancelled(&self) -> bool {
        !matches!(self, CancelState::Active)
    }

    /// Check if the state is draining.
    #[cfg(test)]
    pub(crate) fn is_draining(&self) -> bool {
        matches!(self, CancelState::Draining { .. })
    }

    /// Check if cancellation is fully confirmed.
    pub fn is_confirmed(&self) -> bool {
        matches!(self, CancelState::Confirmed)
    }
}

/// Token for requesting and tracking cancellation.
///
/// The token is shared between the `TransferHandle` (user-facing) and the
/// pipeline stages (internal). When `request()` is called, stages will
/// check at safe points and transition through draining to confirmed.
#[derive(Clone)]
pub(crate) struct CancellationToken {
    /// Sender for cancellation requests
    request_tx: Arc<watch::Sender<bool>>,
    /// Serializes cancellation requests with the physical commitment claim.
    commitment_gate: Arc<Mutex<CommitmentGate>>,
    /// Sends terminal confirmation after the last work unit releases.
    state_tx: Arc<watch::Sender<CancelState>>,
    /// Receiver for cancel state updates
    state_rx: watch::Receiver<CancelState>,
}

/// Shared state for the pre-commit winner selection.
struct CommitmentGate {
    phase: CommitmentPhase,
    cancel_requested: bool,
    root_issued: bool,
    outstanding_units: usize,
}

#[derive(Default)]
enum CommitmentPhase {
    #[default]
    Open,
    Committed,
    Cancelled,
}

/// One retained unit of work for a logical transfer.
///
/// A container owns this lease before commitment. A resolved batch or chain
/// handoff owns it after commitment. Cancellation confirms after all leases
/// release.
pub(crate) struct CancellationUnit {
    token: CancellationToken,
    state: Option<Arc<Mutex<TransferState>>>,
    active: bool,
    settled: bool,
}

impl CancellationToken {
    /// Create a new cancellation token.
    pub(crate) fn new() -> Self {
        let (request_tx, _) = watch::channel(false);
        let (state_tx, state_rx) = watch::channel(CancelState::Active);

        CancellationToken {
            request_tx: Arc::new(request_tx),
            commitment_gate: Arc::new(Mutex::new(CommitmentGate {
                phase: CommitmentPhase::Open,
                cancel_requested: false,
                root_issued: false,
                outstanding_units: 0,
            })),
            state_tx: Arc::new(state_tx.clone()),
            state_rx,
        }
    }

    /// Request cancellation.
    ///
    /// This signals all pipeline stages to stop processing at the next safe point.
    /// Returns immediately - use `wait_confirmed()` to await full confirmation.
    pub fn request(&self) {
        let mut gate = self
            .commitment_gate
            .lock()
            .expect("cancellation commitment gate lock poisoned");
        gate.cancel_requested = true;
        if matches!(gate.phase, CommitmentPhase::Open) {
            gate.phase = CommitmentPhase::Cancelled;
        }
        self.request_tx.send_replace(true);
        self.publish_state(&gate);
    }

    /// Atomically claim the physical commitment boundary.
    ///
    /// A cancellation request that wins this gate drops the container before
    /// weak blocks upgrade. A commitment claim that wins makes later requests
    /// drain physical work instead.
    fn claim_commitment(&self) -> bool {
        let mut gate = self
            .commitment_gate
            .lock()
            .expect("cancellation commitment gate lock poisoned");
        assert!(
            gate.outstanding_units > 0,
            "commitment requires an owned cancellation unit"
        );
        let committed = match gate.phase {
            CommitmentPhase::Open => {
                gate.phase = CommitmentPhase::Committed;
                true
            }
            CommitmentPhase::Committed => true,
            CommitmentPhase::Cancelled => false,
        };
        self.publish_state(&gate);
        committed
    }

    /// Create the sole root unit before the transfer becomes observable.
    pub(crate) fn root_unit(&self) -> Option<CancellationUnit> {
        let mut gate = self
            .commitment_gate
            .lock()
            .expect("cancellation commitment gate lock poisoned");
        if !matches!(gate.phase, CommitmentPhase::Open) || gate.root_issued {
            return None;
        }
        gate.root_issued = true;
        gate.outstanding_units += 1;
        self.publish_state(&gate);
        Some(CancellationUnit {
            token: self.clone(),
            state: None,
            active: true,
            settled: false,
        })
    }

    /// Report whether a request won before the physical commitment claim.
    pub(crate) fn is_precommit_cancelled(&self) -> bool {
        matches!(
            self.commitment_gate
                .lock()
                .expect("cancellation commitment gate lock poisoned")
                .phase,
            CommitmentPhase::Cancelled
        )
    }

    fn release_unit(&self) {
        let mut gate = self
            .commitment_gate
            .lock()
            .expect("cancellation commitment gate lock poisoned");
        assert!(
            gate.outstanding_units > 0,
            "cancellation unit count underflow"
        );
        gate.outstanding_units -= 1;
        self.publish_state(&gate);
    }

    fn publish_state(&self, gate: &CommitmentGate) {
        let state = if !gate.cancel_requested {
            CancelState::Active
        } else if gate.outstanding_units == 0 {
            CancelState::Confirmed
        } else {
            match gate.phase {
                CommitmentPhase::Open | CommitmentPhase::Cancelled => CancelState::Requested,
                CommitmentPhase::Committed => CancelState::Draining {
                    in_flight: gate.outstanding_units,
                },
            }
        };
        self.state_tx.send_replace(state);
    }

    /// Check if cancellation has been requested.
    pub fn is_requested(&self) -> bool {
        *self.request_tx.borrow()
    }

    /// Wait for a cancellation request.
    ///
    /// Each caller receives an independent watch receiver. This lets a
    /// pipeline stage use this future in `tokio::select!` without consuming
    /// another stage's notification.
    pub async fn wait_requested(&self) {
        let mut request_rx = self.request_tx.subscribe();
        while !*request_rx.borrow() {
            if request_rx.changed().await.is_err() {
                return;
            }
        }
    }

    /// Wait until a cancellation request wins before commitment.
    pub(crate) async fn wait_precommit_cancelled(&self) {
        let mut state_rx = self.state_rx.clone();
        loop {
            if self.is_precommit_cancelled() {
                return;
            }
            if state_rx.changed().await.is_err() {
                return;
            }
        }
    }

    /// Get the current cancellation state.
    #[cfg(test)]
    pub(crate) fn state(&self) -> CancelState {
        *self.state_rx.borrow()
    }

    /// Check if cancellation is fully confirmed.
    #[cfg(test)]
    pub(crate) fn is_confirmed(&self) -> bool {
        self.state().is_confirmed()
    }

    /// Create a future that resolves when cancellation is confirmed.
    ///
    /// This is the primary way to await clean release of all blocks.
    pub fn wait_confirmed(&self) -> CancelConfirmation {
        CancelConfirmation {
            state_rx: self.state_rx.clone(),
        }
    }
}

impl CancellationUnit {
    /// Attach the transfer state that owns this work unit.
    pub(crate) fn bind_state(mut self, state: Arc<Mutex<TransferState>>) -> Self {
        if let Some(bound) = &self.state {
            assert!(
                Arc::ptr_eq(bound, &state),
                "a cancellation unit cannot change transfer ownership"
            );
        } else {
            self.state = Some(state);
        }
        self
    }

    pub(crate) fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    pub(crate) fn claim_commitment(&self) -> bool {
        self.token.claim_commitment()
    }

    pub(crate) fn is_committed(&self) -> bool {
        matches!(
            self.token
                .commitment_gate
                .lock()
                .expect("cancellation commitment gate lock poisoned")
                .phase,
            CommitmentPhase::Committed
        )
    }

    /// Release this unit and run the terminal action only for the final unit.
    ///
    /// The final unit remains live during the action. A concurrent cancellation
    /// request therefore observes draining work until terminal state is visible.
    pub(crate) fn settle(mut self, on_last: impl FnOnce()) {
        let is_last = self.release_nonfinal_unit();

        if is_last {
            on_last();
        }
        self.settled = true;
    }

    /// Atomically replace this route unit with one unit for each target.
    pub(crate) fn fan_out(mut self, target_count: usize) -> Vec<Self> {
        assert!(target_count > 0, "zero-route work must settle directly");
        let token = self.token.clone();
        {
            let mut gate = token
                .commitment_gate
                .lock()
                .expect("cancellation commitment gate lock poisoned");
            assert!(
                matches!(gate.phase, CommitmentPhase::Committed),
                "only committed work can fan out"
            );
            assert!(
                gate.outstanding_units > 0,
                "cancellation unit count underflow"
            );
            gate.outstanding_units = gate.outstanding_units - 1 + target_count;
            token.publish_state(&gate);
        }
        self.active = false;
        let state = self.state.clone();
        (0..target_count)
            .map(|_| Self {
                token: token.clone(),
                state: state.clone(),
                active: true,
                settled: false,
            })
            .collect()
    }

    /// Release a non-final unit, or retain the final unit through settlement.
    fn release_nonfinal_unit(&mut self) -> bool {
        let mut gate = self
            .token
            .commitment_gate
            .lock()
            .expect("cancellation commitment gate lock poisoned");
        assert!(
            gate.outstanding_units > 0,
            "cancellation unit count underflow"
        );
        if gate.outstanding_units == 1 {
            true
        } else {
            gate.outstanding_units -= 1;
            self.token.publish_state(&gate);
            self.active = false;
            false
        }
    }

    /// Convert an abnormal committed drop into one failed route settlement.
    fn fail_abandoned_route(&mut self) {
        let state = Arc::clone(
            self.state
                .as_ref()
                .expect("a bound cancellation unit owns transfer state"),
        );
        state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record_error(ABANDONED_COMMITTED_ROUTE_ERROR.to_string());

        if self.release_nonfinal_unit() {
            state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .finish_logical_operation();
            self.token.release_unit();
            self.active = false;
        }
    }
}

impl Drop for CancellationUnit {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if !self.settled && self.state.is_some() && self.is_committed() {
            self.fail_abandoned_route();
        } else {
            self.token.release_unit();
            self.active = false;
        }
    }
}

/// Future that resolves when cancellation is fully confirmed.
///
/// Obtained via `CancellationToken::wait_confirmed()` or `TransferHandle::cancel()`.
pub struct CancelConfirmation {
    state_rx: watch::Receiver<CancelState>,
}

impl CancelConfirmation {
    /// Wait for confirmation (async).
    ///
    /// This is the recommended way to await cancellation confirmation.
    pub async fn wait(mut self) {
        loop {
            // Check current state
            if self.state_rx.borrow().is_confirmed() {
                return;
            }

            // Wait for state change
            if self.state_rx.changed().await.is_err() {
                // Channel closed, treat as confirmed
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_state_transitions() {
        let state = CancelState::Active;
        assert!(!state.is_cancelled());
        assert!(!state.is_draining());
        assert!(!state.is_confirmed());

        let state = CancelState::Requested;
        assert!(state.is_cancelled());
        assert!(!state.is_draining());
        assert!(!state.is_confirmed());

        let state = CancelState::Draining { in_flight: 5 };
        assert!(state.is_cancelled());
        assert!(state.is_draining());
        assert!(!state.is_confirmed());

        let state = CancelState::Confirmed;
        assert!(state.is_cancelled());
        assert!(!state.is_draining());
        assert!(state.is_confirmed());
    }

    #[test]
    fn test_cancellation_token_request() {
        let token = CancellationToken::new();

        assert!(!token.is_requested());
        assert_eq!(token.state(), CancelState::Active);

        token.request();

        assert!(token.is_requested());
    }

    #[test]
    fn commitment_claim_applies_to_later_chain_stages() {
        let token = CancellationToken::new();
        let unit = token.root_unit().expect("create root unit");

        assert!(unit.claim_commitment());
        token.request();

        assert!(
            unit.claim_commitment(),
            "a post-commit cancellation cannot split later chain stages"
        );
    }

    #[tokio::test]
    async fn test_cancel_confirmation_immediate() {
        let token = CancellationToken::new();
        token.request();

        // Should resolve immediately
        token.wait_confirmed().wait().await;
        assert!(token.is_confirmed());
    }

    #[tokio::test]
    async fn test_cancel_confirmation_delayed() {
        let token = CancellationToken::new();
        let unit = token.root_unit().expect("create root unit");

        token.request();
        let confirmation = token.wait_confirmed();

        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            drop(unit);
        });

        // Wait for confirmation
        tokio::time::timeout(tokio::time::Duration::from_millis(100), confirmation.wait())
            .await
            .expect("Should complete within timeout");

        assert!(token.is_confirmed());
    }

    #[tokio::test]
    async fn test_wait_requested_wakes_a_token_clone() {
        let token = CancellationToken::new();
        let waiter = token.clone();

        let task = tokio::spawn(async move {
            waiter.wait_requested().await;
        });

        token.request();
        let _ = tokio::time::timeout(tokio::time::Duration::from_millis(50), task)
            .await
            .expect("cancellation wait must wake");
    }

    /// Test that confirmation does NOT resolve while in-flight > 0.
    /// This is a critical invariant: cancellation only completes after draining.
    #[tokio::test]
    async fn test_confirmation_blocked_during_draining() {
        let token = CancellationToken::new();
        let unit = token.root_unit().expect("create root unit");
        assert!(unit.claim_commitment());

        token.request();

        // Confirmation should NOT resolve while draining
        let confirmation = token.wait_confirmed();
        let result =
            tokio::time::timeout(tokio::time::Duration::from_millis(30), confirmation.wait()).await;
        assert!(result.is_err(), "Should timeout while in_flight > 0");

        // Still draining
        assert_eq!(token.state(), CancelState::Draining { in_flight: 1 });
    }

    /// Test that the last unit release transitions directly to Confirmed.
    #[test]
    fn test_draining_zero_confirms() {
        let token = CancellationToken::new();
        let unit = token.root_unit().expect("create root unit");
        assert!(unit.claim_commitment());

        token.request();
        assert_eq!(token.state(), CancelState::Draining { in_flight: 1 });

        drop(unit);
        assert_eq!(token.state(), CancelState::Confirmed);
    }

    /// Test the full draining sequence: Requested → Draining(n) → ... → Confirmed.
    #[test]
    fn test_full_draining_sequence() {
        let token = CancellationToken::new();
        let root = token.root_unit().expect("create root unit");
        assert!(root.claim_commitment());
        let mut units = root.fan_out(3);

        // Start active
        assert_eq!(token.state(), CancelState::Active);

        // Request
        token.request();
        assert!(token.is_requested());
        assert_eq!(token.state(), CancelState::Draining { in_flight: 3 });

        drop(units.pop());
        assert_eq!(token.state(), CancelState::Draining { in_flight: 2 });

        drop(units.pop());
        assert_eq!(token.state(), CancelState::Draining { in_flight: 1 });

        drop(units.pop());
        assert!(token.is_confirmed());
    }

    #[test]
    fn confirmed_transfer_cannot_create_new_work() {
        let token = CancellationToken::new();
        let root = token.root_unit().expect("create root unit");
        token.request();
        drop(root);

        assert!(token.is_confirmed());
        assert!(token.root_unit().is_none());
    }

    #[test]
    fn released_root_cannot_be_reissued() {
        let token = CancellationToken::new();
        let root = token.root_unit().expect("create root unit");
        drop(root);

        assert!(token.root_unit().is_none());
    }

    #[test]
    fn concurrent_request_and_fan_out_have_no_confirmation_gap() {
        let token = CancellationToken::new();
        let root = token.root_unit().expect("create root unit");
        assert!(root.claim_commitment());
        let start = Arc::new(std::sync::Barrier::new(3));
        let units = std::thread::scope(|scope| {
            let request_token = token.clone();
            let request_start = Arc::clone(&start);
            let request = scope.spawn(move || {
                request_start.wait();
                request_token.request();
            });
            let fan_start = Arc::clone(&start);
            let fan_out = scope.spawn(move || {
                fan_start.wait();
                root.fan_out(3)
            });
            start.wait();
            request.join().expect("request thread completes");
            fan_out.join().expect("fan-out thread completes")
        });

        assert_eq!(token.state(), CancelState::Draining { in_flight: 3 });
        assert!(!token.is_confirmed());
        drop(units);
        assert!(token.is_confirmed());
    }
}
