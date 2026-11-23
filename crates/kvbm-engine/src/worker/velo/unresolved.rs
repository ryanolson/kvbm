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

use ::velo::{InstanceId, Messenger};
use kvbm_physical::transfer::{
    AggregateTransferFailure, TransferDrainOutcome, TransferPartFailure,
};

use super::check::LocalTransferRecheck;

/// One part of a transfer that has no drain proof yet.
#[derive(Debug)]
pub struct UnresolvedParticipant {
    recheck: Option<LocalTransferRecheck>,
    /// [`TransferPartFailure::Unproven`], or a kind this crate does not know,
    /// kept whole so a re-fold gives it back unchanged.
    part: TransferPartFailure,
}

impl UnresolvedParticipant {
    /// The capability to check this part again. `None` when no check can
    /// resolve it: an in-process worker whose own drain proof failed, or a
    /// part of a kind this crate does not know.
    pub fn recheck(&self) -> Option<&LocalTransferRecheck> {
        self.recheck.as_ref()
    }

    /// The error of the last outcome of this part.
    pub fn error(&self) -> &anyhow::Error {
        self.part.error()
    }

    fn on_worker(&self, instance: InstanceId, worker_id: Option<u64>) -> bool {
        self.recheck
            .as_ref()
            .is_some_and(|recheck| recheck.instance == instance && recheck.worker_id == worker_id)
    }
}

/// Bind the new error of an unresolved participant to its capability.
///
/// A re-fold reads a participant's identity from the outermost
/// [`LocalTransferRecheck`] of its error, and walks any
/// [`AggregateTransferFailure`] it finds. An error whose outermost capability
/// is `recheck` and that holds no aggregate already has the right identity,
/// so it is kept as it is: that is the usual answer of a check. Any other
/// error is wrapped in an opaque [`RecheckDiagnostic`] under `recheck`.
fn bind(recheck: &LocalTransferRecheck, error: anyhow::Error) -> anyhow::Error {
    let same_identity = error.downcast_ref::<LocalTransferRecheck>() == Some(recheck)
        && error.downcast_ref::<AggregateTransferFailure>().is_none();
    if same_identity {
        error
    } else {
        anyhow::Error::new(RecheckDiagnostic(error)).context(recheck.clone())
    }
}

/// The new error of an unresolved participant, opaque to classification: a
/// downcast cannot reach a capability or an aggregate inside it. Its text is
/// the full text of that error.
#[derive(Debug)]
struct RecheckDiagnostic(anyhow::Error);

impl std::fmt::Display for RecheckDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#}", self.0)
    }
}

impl std::error::Error for RecheckDiagnostic {}

/// The parts of an [`TransferDrainOutcome::Unproven`] transfer, split into the
/// parts that are still unresolved and the failures already proven drained.
///
/// It settles only when every unresolved participant is resolved. A
/// participant without a re-check capability never resolves, so the transfer
/// stays `Unproven`.
#[derive(Debug)]
pub struct UnresolvedTransfer {
    /// The label of the top aggregate, kept for the text of a re-fold.
    label: Option<String>,
    unresolved: Vec<UnresolvedParticipant>,
    drained: Vec<anyhow::Error>,
}

impl UnresolvedTransfer {
    /// Split the error of an `Unproven` outcome into its participants.
    ///
    /// Nested aggregate failures are walked to their leaves, including those
    /// inside a part marked drained. A leaf is proven only when it is an
    /// explicit drained error that holds no unproven aggregate. Every other
    /// leaf is one participant, with the [`LocalTransferRecheck`] its error
    /// carries, if any. A part of a kind this crate does not know is one
    /// participant with no capability, kept whole.
    pub fn from_unproven(error: anyhow::Error) -> Self {
        let label = error
            .downcast_ref::<AggregateTransferFailure>()
            .filter(|aggregate| aggregate.is_unproven())
            .and_then(|aggregate| aggregate.label().map(str::to_owned));
        let mut transfer = Self {
            label,
            unresolved: Vec::new(),
            drained: Vec::new(),
        };
        transfer.collect_unproven(error);
        transfer
    }

    /// Collect an error the caller reported as unproven.
    fn collect_unproven(&mut self, error: anyhow::Error) {
        match error.downcast::<AggregateTransferFailure>() {
            Ok(aggregate) if aggregate.is_unproven() => self.collect_parts(aggregate),
            // An unproven error whose aggregate proves every part drained is
            // inconsistent. Keep it unresolved, with nothing to check.
            Ok(aggregate) => self.unresolved.push(UnresolvedParticipant {
                recheck: None,
                part: TransferPartFailure::Unproven(anyhow::Error::new(aggregate)),
            }),
            Err(error) => {
                let recheck = error.downcast_ref::<LocalTransferRecheck>().cloned();
                self.unresolved.push(UnresolvedParticipant {
                    recheck,
                    part: TransferPartFailure::Unproven(error),
                });
            }
        }
    }

    fn collect_parts(&mut self, aggregate: AggregateTransferFailure) {
        for part in aggregate.into_parts() {
            match part {
                // `is_unproven` looks inside for a nested unproven aggregate.
                TransferPartFailure::DrainedWithError(error)
                    if error
                        .downcast_ref::<AggregateTransferFailure>()
                        .is_some_and(AggregateTransferFailure::is_unproven) =>
                {
                    self.collect_unproven(error);
                }
                TransferPartFailure::DrainedWithError(error) => self.drained.push(error),
                TransferPartFailure::Unproven(error) => self.collect_unproven(error),
                // A kind this crate does not know is opaque and unproven.
                other => self.unresolved.push(UnresolvedParticipant {
                    recheck: None,
                    part: other,
                }),
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

    /// Apply the outcome of one check of `recheck` to its participant, and
    /// leave every other participant and every drained failure as it is.
    ///
    /// "Completed" resolves the participant. "Drained with error" resolves it
    /// and keeps the error, unless the error holds an unproven aggregate. Any
    /// other outcome keeps it unresolved, with the new error as its
    /// diagnostic. The capability stays the participant's identity: the new
    /// error is bound to it, and any capability or aggregate inside the new
    /// error is hidden, so a re-fold of [`Self::into_outcome`] finds this
    /// participant with this capability and nothing else. Returns `false`, and changes nothing, when no unresolved
    /// participant has this capability.
    pub fn apply(&mut self, recheck: &LocalTransferRecheck, outcome: TransferDrainOutcome) -> bool {
        let Some(index) = self
            .unresolved
            .iter()
            .position(|participant| participant.recheck.as_ref() == Some(recheck))
        else {
            return false;
        };
        match outcome {
            TransferDrainOutcome::Completed => {
                self.unresolved.remove(index);
            }
            // A drained outcome that holds an unproven aggregate proves
            // nothing.
            TransferDrainOutcome::DrainedWithError(error)
                if error
                    .downcast_ref::<AggregateTransferFailure>()
                    .is_some_and(AggregateTransferFailure::is_unproven) =>
            {
                self.unresolved[index].part = TransferPartFailure::Unproven(bind(recheck, error));
            }
            TransferDrainOutcome::DrainedWithError(error) => {
                self.unresolved.remove(index);
                self.drained.push(error);
            }
            TransferDrainOutcome::Unproven(error) => {
                self.unresolved[index].part = TransferPartFailure::Unproven(bind(recheck, error));
            }
        }
        true
    }

    /// Check every unresolved participant that has a capability once, at the
    /// same time, each under `deadline`, and apply the answers.
    pub async fn recheck(&mut self, messenger: &Messenger, deadline: Duration) {
        let targets: Vec<_> = self.rechecks().cloned().collect();
        self.check_and_apply(targets, messenger, deadline).await;
    }

    /// Check, once each, only the unresolved participants on one worker:
    /// velo instance `instance` and worker id `worker_id`. No other worker
    /// is asked, and no other participant changes.
    pub async fn recheck_worker(
        &mut self,
        messenger: &Messenger,
        instance: InstanceId,
        worker_id: Option<u64>,
        deadline: Duration,
    ) {
        let targets: Vec<_> = self
            .unresolved
            .iter()
            .filter(|participant| participant.on_worker(instance, worker_id))
            .filter_map(|participant| participant.recheck.clone())
            .collect();
        self.check_and_apply(targets, messenger, deadline).await;
    }

    async fn check_and_apply(
        &mut self,
        targets: Vec<LocalTransferRecheck>,
        messenger: &Messenger,
        deadline: Duration,
    ) {
        let outcomes = futures::future::join_all(
            targets
                .iter()
                .map(|recheck| recheck.check(messenger, deadline)),
        )
        .await;
        for (recheck, outcome) in targets.iter().zip(outcomes) {
            self.apply(recheck, outcome);
        }
    }

    /// The outcome of the transfer now.
    ///
    /// `Unproven` while any participant is unresolved; its error keeps every
    /// unresolved participant's capability, and the top label. Otherwise
    /// `DrainedWithError` if any part failed, and `Completed` if none did.
    pub fn into_outcome(mut self) -> TransferDrainOutcome {
        // One unlabelled part keeps its own error, so its capability stays at
        // the top. A part of an unknown kind stays inside an aggregate, so a
        // re-fold still sees its kind.
        if self.label.is_none() {
            match (self.drained.len(), self.unresolved.len()) {
                (0, 0) => return TransferDrainOutcome::Completed,
                (1, 0) => return TransferDrainOutcome::DrainedWithError(self.drained.remove(0)),
                (0, 1) if matches!(self.unresolved[0].part, TransferPartFailure::Unproven(_)) => {
                    return TransferDrainOutcome::Unproven(
                        self.unresolved.remove(0).part.into_error(),
                    );
                }
                _ => {}
            }
        }
        let parts = self
            .drained
            .into_iter()
            .map(TransferPartFailure::DrainedWithError)
            .chain(
                self.unresolved
                    .into_iter()
                    .map(|participant| participant.part),
            )
            .collect();
        AggregateTransferFailure::outcome(self.label, parts)
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
            label: None,
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

    /// A child labelled drained still holds an unproven leaf. The leaf stays
    /// unresolved whatever label its parent carries.
    #[test]
    fn an_unproven_leaf_under_a_drained_labelled_child_stays_unresolved() {
        let recheck = capability(1);
        let grandchild = aggregate(vec![TransferPartFailure::Unproven(unproven_on(&recheck))]);
        let child = aggregate(vec![
            TransferPartFailure::DrainedWithError(grandchild),
            TransferPartFailure::DrainedWithError(anyhow::anyhow!("a drained part")),
        ]);
        let outer = aggregate(vec![TransferPartFailure::DrainedWithError(child)]);

        let transfer = UnresolvedTransfer::from_unproven(outer);

        assert!(!transfer.is_resolved());
        assert_eq!(
            transfer.rechecks().cloned().collect::<Vec<_>>(),
            vec![recheck]
        );
        assert!(matches!(
            transfer.into_outcome(),
            TransferDrainOutcome::Unproven(_)
        ));
    }

    /// The top label survives a re-fold, so the text matches the fold's.
    #[test]
    fn a_refold_keeps_the_top_label() {
        let (first, second) = (capability(1), capability(2));
        let labelled = anyhow::Error::new(AggregateTransferFailure::new(
            Some("replicated onboard rank group failed".to_owned()),
            vec![
                TransferPartFailure::Unproven(unproven_on(&first)),
                TransferPartFailure::Unproven(unproven_on(&second)),
            ],
        ));
        let text = labelled.to_string();

        let transfer = UnresolvedTransfer::from_unproven(labelled);
        let TransferDrainOutcome::Unproven(error) = transfer.into_outcome() else {
            panic!("both participants are unresolved");
        };
        assert_eq!(error.to_string(), text);
    }

    /// One worker's proof resolves only its own participant. The others and
    /// the drained failures stay, and a capability not in the transfer
    /// changes nothing.
    #[test]
    fn one_proof_resolves_only_its_participant() {
        let (first, second, stranger) = (capability(1), capability(2), capability(3));
        let mut transfer = UnresolvedTransfer::from_unproven(aggregate(vec![
            TransferPartFailure::DrainedWithError(anyhow::anyhow!("a drained part")),
            TransferPartFailure::Unproven(unproven_on(&first)),
            TransferPartFailure::Unproven(unproven_on(&second)),
        ]));

        assert!(!transfer.apply(&stranger, TransferDrainOutcome::Completed));
        assert_eq!(transfer.participants().len(), 2);

        assert!(transfer.apply(&first, TransferDrainOutcome::Completed));
        assert_eq!(
            transfer.rechecks().cloned().collect::<Vec<_>>(),
            vec![second.clone()]
        );

        assert!(transfer.apply(
            &second,
            second.unproven(
                super::super::check::CheckEvidence::default(),
                "still unknown"
            )
        ));
        assert_eq!(
            transfer.rechecks().cloned().collect::<Vec<_>>(),
            vec![second.clone()]
        );

        let claims_drain = aggregate(vec![TransferPartFailure::Unproven(anyhow::anyhow!("lost"))]);
        assert!(transfer.apply(
            &second,
            TransferDrainOutcome::DrainedWithError(claims_drain)
        ));
        assert_eq!(
            transfer.participants().len(),
            1,
            "a drained outcome holding an unproven aggregate proves nothing"
        );

        assert!(transfer.apply(&second, TransferDrainOutcome::Completed));
        assert!(transfer.is_resolved());
        match transfer.into_outcome() {
            TransferDrainOutcome::DrainedWithError(error) => {
                assert_eq!(error.to_string(), "a drained part");
            }
            TransferDrainOutcome::Completed => panic!("the drained failure must be kept"),
            TransferDrainOutcome::Unproven(error) => {
                panic!("every participant resolved: {error:#}")
            }
        }
    }

    #[test]
    fn a_contradictory_drain_update_keeps_the_original_capability_after_refolding() {
        let worker = capability(1);
        let mut transfer = UnresolvedTransfer::from_unproven(unproven_on(&worker));
        let claims_drain = aggregate(vec![TransferPartFailure::Unproven(anyhow::anyhow!(
            "drain is still unproven"
        ))]);

        assert!(transfer.apply(
            &worker,
            TransferDrainOutcome::DrainedWithError(claims_drain)
        ));
        assert_eq!(transfer.rechecks().collect::<Vec<_>>(), vec![&worker]);

        let TransferDrainOutcome::Unproven(error) = transfer.into_outcome() else {
            panic!("the update did not prove drain");
        };
        let transfer = UnresolvedTransfer::from_unproven(error);
        assert_eq!(transfer.rechecks().collect::<Vec<_>>(), vec![&worker]);
    }

    #[test]
    fn a_failed_update_cannot_change_which_transfer_resolves_the_participant() {
        let original = capability(1);
        let mut different_transfer = original.clone();
        different_transfer.transfer_id = uuid::Uuid::new_v4();
        let mut transfer = UnresolvedTransfer::from_unproven(unproven_on(&original));

        assert!(!transfer.apply(&different_transfer, TransferDrainOutcome::Completed));
        assert!(transfer.apply(
            &original,
            TransferDrainOutcome::Unproven(unproven_on(&different_transfer)),
        ));

        let TransferDrainOutcome::Unproven(error) = transfer.into_outcome() else {
            panic!("the update did not prove drain");
        };
        let mut transfer = UnresolvedTransfer::from_unproven(error);
        assert!(!transfer.apply(&different_transfer, TransferDrainOutcome::Completed));
        assert!(transfer.apply(&original, TransferDrainOutcome::Completed));
        assert!(!transfer.apply(&original, TransferDrainOutcome::Completed));
    }

    /// A new error that holds other capabilities, inside an aggregate, stays
    /// one participant with the original capability after a re-fold.
    #[test]
    fn a_replacement_holding_other_capabilities_keeps_the_original_identity() {
        let (original, other_a, other_b) = (capability(1), capability(2), capability(3));
        let mut transfer = UnresolvedTransfer::from_unproven(unproven_on(&original));
        let replacement = aggregate(vec![
            TransferPartFailure::Unproven(unproven_on(&other_a)),
            TransferPartFailure::Unproven(unproven_on(&other_b)),
        ]);

        assert!(transfer.apply(&original, TransferDrainOutcome::Unproven(replacement)));

        let TransferDrainOutcome::Unproven(error) = transfer.into_outcome() else {
            panic!("the update did not prove drain");
        };
        let text = format!("{error:#}");
        assert!(text.contains("lost reply"), "{text}");
        let transfer = UnresolvedTransfer::from_unproven(error);
        assert_eq!(transfer.participants().len(), 1);
        assert_eq!(transfer.rechecks().collect::<Vec<_>>(), vec![&original]);
    }

    /// The usual answer of a check already carries the capability, so it is
    /// kept as it is, and its text names the capability once.
    #[test]
    fn a_check_answer_keeps_its_own_error() {
        let original = capability(1);
        let mut transfer = UnresolvedTransfer::from_unproven(unproven_on(&original));

        assert!(transfer.apply(
            &original,
            original.unproven(
                super::super::check::CheckEvidence::default(),
                "still unknown"
            )
        ));

        let text = format!("{:#}", transfer.participants()[0].error());
        assert_eq!(text, format!("{original}: still unknown"));
    }
}
