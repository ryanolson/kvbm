// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reservation, cancellation, and proven-completion state.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use tokio::sync::oneshot;

use super::RouteBinding;
use crate::g2_capacity::G2ExactAllocation;

/// One opaque exact reservation from a bound policy route.
#[must_use = "submit this reservation through its bound route or drop it"]
pub struct PolicyG1G2Reservation {
    pub(super) allocation: Option<G2ExactAllocation>,
    pub(super) cancellation: PolicyG1G2CancelHandle,
    pub(super) binding: Arc<RouteBinding>,
    mark_terminal_on_drop: bool,
}

/// Cancellation control for one policy transfer.
#[derive(Clone)]
pub struct PolicyG1G2CancelHandle {
    state: Arc<Mutex<CommitState>>,
}

/// One physical transfer that owns its real logical source.
#[must_use = "await the result before source settlement"]
pub struct PolicyG1G2Execution {
    cancellation: PolicyG1G2CancelHandle,
    completion: Option<oneshot::Receiver<PolicyG1G2Completion>>,
    finished: Arc<AtomicBool>,
}

/// One proven physical result after the engine settled its logical source.
#[must_use = "inspect the physical and source results"]
pub struct PolicyG1G2Completion {
    physical: PolicyPhysicalCompletion,
    source: PolicyG1G2SourceSettlement,
}

/// A synchronous physical submission error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyG1G2SubmitError {
    /// Submission needs an active Tokio runtime.
    NoRuntime,
    /// The owned source belongs to another logical manager.
    ForeignSourceManager,
}

/// The background physical transaction ended without proven drain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolicyG1G2ExecutionError;

/// The result of a cancellation request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyCancelDisposition {
    CancelledBeforeCommit,
    CommittedTransferDraining,
    AlreadyTerminal,
}

/// The stable terminal class for one policy transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyPhysicalTerminal {
    DestinationCommitted,
    CancelledBeforeCommit,
    Failed,
}

/// The logical source result after one proven physical terminal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyG1G2SourceSettlement {
    Restored,
    Committed { notification_panicked: bool },
}

/// One terminal result from the exact physical route.
#[derive(Debug)]
pub struct PolicyPhysicalCompletion {
    terminal: PolicyPhysicalTerminal,
    failure: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommitState {
    Open,
    CancelledBeforeCommit,
    Committed,
    Terminal,
}

impl PolicyG1G2Reservation {
    pub(super) fn new(allocation: G2ExactAllocation, binding: Arc<RouteBinding>) -> Self {
        Self {
            allocation: Some(allocation),
            cancellation: PolicyG1G2CancelHandle {
                state: Arc::new(Mutex::new(CommitState::Open)),
            },
            binding,
            mark_terminal_on_drop: true,
        }
    }

    pub fn cancellation(&self) -> PolicyG1G2CancelHandle {
        self.cancellation.clone()
    }

    pub fn len(&self) -> usize {
        self.allocation.as_ref().map_or(0, G2ExactAllocation::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn uncommitted_failure(
        &mut self,
        failure: impl Into<String>,
    ) -> PolicyPhysicalCompletion {
        drop(self.allocation.take());
        if self.cancellation.settle_uncommitted_failure() {
            PolicyPhysicalCompletion::failed(failure)
        } else {
            PolicyPhysicalCompletion::cancelled()
        }
    }

    pub(super) fn disarm_terminal_on_drop(&mut self) {
        self.mark_terminal_on_drop = false;
    }
}

impl std::fmt::Debug for PolicyG1G2Reservation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PolicyG1G2Reservation")
            .field("blocks", &self.len())
            .finish_non_exhaustive()
    }
}

impl Drop for PolicyG1G2Reservation {
    fn drop(&mut self) {
        if self.allocation.is_some() {
            drop(self.allocation.take());
            if self.mark_terminal_on_drop {
                self.cancellation.mark_terminal();
            }
        }
    }
}

impl PolicyG1G2CancelHandle {
    pub fn cancel(&self) -> PolicyCancelDisposition {
        let mut state = self.state.lock();
        match *state {
            CommitState::Open => {
                *state = CommitState::CancelledBeforeCommit;
                PolicyCancelDisposition::CancelledBeforeCommit
            }
            CommitState::CancelledBeforeCommit => PolicyCancelDisposition::CancelledBeforeCommit,
            CommitState::Committed => PolicyCancelDisposition::CommittedTransferDraining,
            CommitState::Terminal => PolicyCancelDisposition::AlreadyTerminal,
        }
    }

    pub(super) fn claim_commit(&self) -> bool {
        let mut state = self.state.lock();
        match *state {
            CommitState::Open => {
                *state = CommitState::Committed;
                true
            }
            CommitState::CancelledBeforeCommit => false,
            CommitState::Committed | CommitState::Terminal => false,
        }
    }

    pub(super) fn mark_terminal(&self) {
        *self.state.lock() = CommitState::Terminal;
    }

    fn settle_uncommitted_failure(&self) -> bool {
        let mut state = self.state.lock();
        match *state {
            CommitState::Open => {
                *state = CommitState::Committed;
                true
            }
            CommitState::CancelledBeforeCommit => false,
            CommitState::Committed | CommitState::Terminal => false,
        }
    }
}

impl PolicyG1G2Execution {
    pub(super) fn new(
        cancellation: PolicyG1G2CancelHandle,
        completion: oneshot::Receiver<PolicyG1G2Completion>,
        finished: Arc<AtomicBool>,
    ) -> Self {
        Self {
            cancellation,
            completion: Some(completion),
            finished,
        }
    }

    pub fn cancellation(&self) -> PolicyG1G2CancelHandle {
        self.cancellation.clone()
    }

    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    pub async fn wait(mut self) -> Result<PolicyG1G2Completion, PolicyG1G2ExecutionError> {
        let receiver = self
            .completion
            .take()
            .expect("the exact execution completion receiver is consumed once");
        receiver.await.map_err(|_| PolicyG1G2ExecutionError)
    }
}

impl Drop for PolicyG1G2Execution {
    fn drop(&mut self) {
        let _ = self.cancellation.cancel();
    }
}

impl PolicyG1G2Completion {
    pub(super) fn new(
        physical: PolicyPhysicalCompletion,
        source: PolicyG1G2SourceSettlement,
    ) -> Self {
        Self { physical, source }
    }

    pub const fn physical(&self) -> &PolicyPhysicalCompletion {
        &self.physical
    }

    pub const fn source(&self) -> PolicyG1G2SourceSettlement {
        self.source
    }
}

impl PolicyPhysicalCompletion {
    pub const fn terminal(&self) -> PolicyPhysicalTerminal {
        self.terminal
    }

    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    pub(super) fn destination_committed() -> Self {
        Self {
            terminal: PolicyPhysicalTerminal::DestinationCommitted,
            failure: None,
        }
    }

    pub(super) fn cancelled() -> Self {
        Self {
            terminal: PolicyPhysicalTerminal::CancelledBeforeCommit,
            failure: None,
        }
    }

    pub(super) fn failed(failure: impl Into<String>) -> Self {
        Self {
            terminal: PolicyPhysicalTerminal::Failed,
            failure: Some(failure.into()),
        }
    }
}

impl std::fmt::Display for PolicyG1G2SubmitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoRuntime => {
                formatter.write_str("policy G1-to-G2 submission needs a Tokio runtime")
            }
            Self::ForeignSourceManager => {
                formatter.write_str("the exact source belongs to another logical manager")
            }
        }
    }
}

impl std::error::Error for PolicyG1G2SubmitError {}

impl std::fmt::Display for PolicyG1G2ExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("the policy G1-to-G2 task ended without proven physical drain")
    }
}

impl std::error::Error for PolicyG1G2ExecutionError {}
