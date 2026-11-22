// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The unresolved participants of an `Unproven` transfer.
//!
//! A copy that runs on several workers ends `Unproven` when any part of it is
//! unproven. The aggregate receipt keeps every part's typed error (see
//! [`AggregateTransferFailure`]), so each unproven part's
//! [`LocalTransferRecheck`] survives. [`UnresolvedTransfer`] lists those parts
//! and settles the transfer only when every one of them is resolved.

use std::time::Duration;

use ::velo::Messenger;
use kvbm_physical::transfer::{
    AggregateTransferFailure, TransferDrainOutcome, TransferPartFailure,
};

use super::check::LocalTransferRecheck;

/// One part of a transfer that has no drain proof yet.
#[derive(Debug)]
pub struct UnresolvedParticipant {
    recheck: Option<LocalTransferRecheck>,
    error: anyhow::Error,
}

impl UnresolvedParticipant {
    /// The capability to check this part again. `None` when no check can
    /// resolve it, for example an in-process worker whose own drain proof
    /// failed.
    pub fn recheck(&self) -> Option<&LocalTransferRecheck> {
        self.recheck.as_ref()
    }

    /// The error of the last outcome of this part.
    pub fn error(&self) -> &anyhow::Error {
        &self.error
    }
}

/// The parts of an [`TransferDrainOutcome::Unproven`] transfer, split into the
/// parts that are still unresolved and the failures already proven drained.
///
/// It settles only when every unresolved participant is resolved. A
/// participant without a re-check capability never resolves, so the transfer
/// stays `Unproven`.
#[derive(Debug)]
pub struct UnresolvedTransfer {
    unresolved: Vec<UnresolvedParticipant>,
    drained: Vec<anyhow::Error>,
}

impl UnresolvedTransfer {
    /// Split the error of an `Unproven` outcome into its participants.
    ///
    /// Nested aggregate failures are walked to their leaves. A leaf is one
    /// participant, with the [`LocalTransferRecheck`] its error carries, if
    /// any.
    pub fn from_unproven(error: anyhow::Error) -> Self {
        let mut transfer = Self {
            unresolved: Vec::new(),
            drained: Vec::new(),
        };
        transfer.collect(error);
        transfer
    }

    fn collect(&mut self, error: anyhow::Error) {
        match error.downcast::<AggregateTransferFailure>() {
            Ok(aggregate) if aggregate.is_unproven() => {
                for part in aggregate.into_parts() {
                    match part {
                        TransferPartFailure::DrainedWithError(error) => self.drained.push(error),
                        TransferPartFailure::Unproven(error) => self.collect(error),
                        // A part of an unknown kind has no proof.
                        other => self.unresolved.push(UnresolvedParticipant {
                            recheck: None,
                            error: other.into_error(),
                        }),
                    }
                }
            }
            // An unproven part whose aggregate claims every part drained is
            // inconsistent. Keep it unresolved, with nothing to check.
            Ok(aggregate) => self.unresolved.push(UnresolvedParticipant {
                recheck: None,
                error: anyhow::Error::new(aggregate),
            }),
            Err(error) => {
                let recheck = error.downcast_ref::<LocalTransferRecheck>().cloned();
                self.unresolved
                    .push(UnresolvedParticipant { recheck, error });
            }
        }
    }

    /// The participants that have no drain proof yet.
    pub fn participants(&self) -> &[UnresolvedParticipant] {
        &self.unresolved
    }

    /// The re-check capabilities of the unresolved participants.
    pub fn rechecks(&self) -> impl Iterator<Item = &LocalTransferRecheck> {
        self.unresolved
            .iter()
            .filter_map(UnresolvedParticipant::recheck)
    }

    /// Whether every participant is resolved.
    pub fn is_resolved(&self) -> bool {
        self.unresolved.is_empty()
    }

    /// Check every unresolved participant that has a capability once, at the
    /// same time, each under `deadline`, and record the answers.
    ///
    /// A participant leaves the unresolved list only on a proof from its
    /// worker: "completed", or "drained with error", which is kept.
    pub async fn recheck(&mut self, messenger: &Messenger, deadline: Duration) {
        let outcomes = futures::future::join_all(self.unresolved.iter().map(|participant| {
            let recheck = participant.recheck.clone();
            async move {
                match recheck {
                    Some(recheck) => Some(recheck.check(messenger, deadline).await),
                    None => None,
                }
            }
        }))
        .await;
        let participants = std::mem::take(&mut self.unresolved);
        for (mut participant, outcome) in participants.into_iter().zip(outcomes) {
            match outcome {
                Some(TransferDrainOutcome::Completed) => {}
                Some(TransferDrainOutcome::DrainedWithError(error)) => self.drained.push(error),
                Some(TransferDrainOutcome::Unproven(error)) => {
                    participant.error = error;
                    self.unresolved.push(participant);
                }
                None => self.unresolved.push(participant),
            }
        }
    }

    /// The outcome of the transfer now.
    ///
    /// `Unproven` while any participant is unresolved; its error keeps every
    /// unresolved participant's capability. Otherwise `DrainedWithError` if
    /// any part failed, and `Completed` if none did.
    pub fn into_outcome(mut self) -> TransferDrainOutcome {
        // One part keeps its own error, so its capability stays at the top.
        match (self.drained.len(), self.unresolved.len()) {
            (0, 0) => return TransferDrainOutcome::Completed,
            (1, 0) => return TransferDrainOutcome::DrainedWithError(self.drained.remove(0)),
            (0, 1) => return TransferDrainOutcome::Unproven(self.unresolved.remove(0).error),
            _ => {}
        }
        let parts = self
            .drained
            .into_iter()
            .map(TransferPartFailure::DrainedWithError)
            .chain(
                self.unresolved
                    .into_iter()
                    .map(|participant| TransferPartFailure::Unproven(participant.error)),
            )
            .collect();
        AggregateTransferFailure::outcome(None, parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capability(worker_id: u64) -> LocalTransferRecheck {
        LocalTransferRecheck {
            instance: ::velo::InstanceId::new_v4(),
            worker_id: Some(worker_id),
            incarnation: Some(uuid::Uuid::new_v4()),
            transfer_id: uuid::Uuid::new_v4(),
        }
    }

    fn unproven_on(recheck: &LocalTransferRecheck) -> anyhow::Error {
        anyhow::anyhow!("lost reply").context(recheck.clone())
    }

    fn aggregate(parts: Vec<TransferPartFailure>) -> anyhow::Error {
        anyhow::Error::new(AggregateTransferFailure::new(None, parts))
    }

    #[test]
    fn a_nested_aggregate_lists_every_leaf_capability() {
        let (first, second, third) = (capability(1), capability(2), capability(3));
        let inner = aggregate(vec![
            TransferPartFailure::Unproven(unproven_on(&first)),
            TransferPartFailure::Unproven(unproven_on(&second)),
        ])
        .context("an outer step wrapped the inner aggregate");
        let outer = aggregate(vec![
            TransferPartFailure::DrainedWithError(anyhow::anyhow!("dispatch failed")),
            TransferPartFailure::Unproven(inner),
            TransferPartFailure::Unproven(unproven_on(&third)),
        ]);

        let transfer = UnresolvedTransfer::from_unproven(outer);

        let found: Vec<_> = transfer.rechecks().cloned().collect();
        assert_eq!(found, vec![first, second, third]);
        assert_eq!(transfer.drained.len(), 1);
    }

    #[test]
    fn a_participant_without_a_capability_stays_unresolved() {
        let recheck = capability(1);
        let transfer = UnresolvedTransfer::from_unproven(aggregate(vec![
            TransferPartFailure::Unproven(unproven_on(&recheck)),
            TransferPartFailure::Unproven(anyhow::anyhow!("in-process drain failed")),
        ]));

        assert_eq!(transfer.participants().len(), 2);
        assert_eq!(transfer.rechecks().count(), 1);
        assert!(transfer.participants()[1].recheck().is_none());
        assert!(matches!(
            transfer.into_outcome(),
            TransferDrainOutcome::Unproven(_)
        ));
    }

    #[test]
    fn an_unproven_part_whose_aggregate_claims_drain_stays_unresolved() {
        let inconsistent = aggregate(vec![TransferPartFailure::DrainedWithError(
            anyhow::anyhow!("drained"),
        )]);
        let transfer =
            UnresolvedTransfer::from_unproven(aggregate(vec![TransferPartFailure::Unproven(
                inconsistent,
            )]));

        assert_eq!(transfer.participants().len(), 1);
        assert!(!transfer.is_resolved());
    }

    #[test]
    fn a_settled_transfer_keeps_its_drained_failures() {
        let transfer = UnresolvedTransfer {
            unresolved: Vec::new(),
            drained: vec![anyhow::anyhow!("first"), anyhow::anyhow!("second")],
        };
        match transfer.into_outcome() {
            TransferDrainOutcome::DrainedWithError(error) => {
                assert_eq!(error.to_string(), "first; second");
            }
            TransferDrainOutcome::Completed => panic!("two parts failed"),
            TransferDrainOutcome::Unproven(error) => panic!("no part is unresolved: {error:#}"),
        }
    }
}
