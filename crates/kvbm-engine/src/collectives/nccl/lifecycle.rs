// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Mutex;

use anyhow::Result;

use super::super::nccl_ffi::{NCCL_IN_PROGRESS, NCCL_SUCCESS, NcclResult};

#[derive(Debug)]
enum CommunicatorState {
    Active,
    /// A nonblocking group submission has returned `ncclInProgress`.
    /// Only async-state polling or abort is legal until it completes.
    Submitting {
        deferred_failure: Option<String>,
    },
    /// A group transaction failed after opening. No later operation may use
    /// the communicator before fatal handling takes native abort ownership.
    Failed {
        reason: String,
    },
    Aborting {
        reason: String,
    },
    Aborted {
        reason: String,
        error: Option<String>,
    },
}

#[derive(Debug)]
pub(super) struct CommunicatorLifecycle {
    state: Mutex<CommunicatorState>,
}

impl CommunicatorLifecycle {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(CommunicatorState::Active),
        }
    }

    pub(super) fn ensure_active(&self) -> Result<()> {
        match &*self.state.lock().unwrap() {
            CommunicatorState::Active => Ok(()),
            CommunicatorState::Submitting { .. } => {
                anyhow::bail!("NCCL communicator already has a submission in progress")
            }
            CommunicatorState::Failed { reason } => {
                anyhow::bail!("NCCL communicator failed: {reason}")
            }
            CommunicatorState::Aborting { reason } | CommunicatorState::Aborted { reason, .. } => {
                anyhow::bail!("NCCL communicator is aborted: {reason}")
            }
        }
    }

    /// Execute one complete NCCL group transaction with exclusive native
    /// admission. The closure must balance every successful `ncclGroupStart`
    /// with `ncclGroupEnd`, including its error paths.
    pub(super) fn submit_group(
        &self,
        submission: impl FnOnce() -> Result<GroupSubmission>,
    ) -> Result<GroupSubmission> {
        let mut state = self.state.lock().unwrap();
        match &*state {
            CommunicatorState::Active => {}
            CommunicatorState::Submitting { .. } => {
                anyhow::bail!("NCCL communicator already has a submission in progress")
            }
            CommunicatorState::Failed { reason } => {
                anyhow::bail!("NCCL communicator failed: {reason}")
            }
            CommunicatorState::Aborting { reason } | CommunicatorState::Aborted { reason, .. } => {
                anyhow::bail!("NCCL communicator is aborted: {reason}")
            }
        }

        let submission = match submission() {
            Ok(submission) => submission,
            Err(error) => {
                *state = CommunicatorState::Failed {
                    reason: format!("{error:#}"),
                };
                return Err(error);
            }
        };
        let deferred_failure = submission
            .deferred_error
            .as_ref()
            .map(|error| format!("{error:#}"));
        if submission.completion == NCCL_IN_PROGRESS {
            *state = CommunicatorState::Submitting { deferred_failure };
        } else if let Some(reason) = deferred_failure {
            *state = CommunicatorState::Failed { reason };
        }
        Ok(submission)
    }

    /// Poll the only operation allowed while a nonblocking group submission
    /// is outstanding. The mutex is released between polls so abort can take
    /// exclusive native ownership.
    pub(super) fn poll_submission(
        &self,
        poll: impl FnOnce() -> Result<NcclResult>,
    ) -> Result<NcclResult> {
        let mut state = self.state.lock().unwrap();
        let deferred_failure = match &*state {
            CommunicatorState::Submitting { deferred_failure } => deferred_failure.clone(),
            CommunicatorState::Active => {
                anyhow::bail!("NCCL communicator has no submission in progress")
            }
            CommunicatorState::Failed { reason } => {
                anyhow::bail!("NCCL communicator failed: {reason}")
            }
            CommunicatorState::Aborting { reason } | CommunicatorState::Aborted { reason, .. } => {
                anyhow::bail!("NCCL communicator is aborted: {reason}")
            }
        };

        let result = poll()?;
        if result == NCCL_SUCCESS {
            *state = match deferred_failure {
                Some(reason) => CommunicatorState::Failed { reason },
                None => CommunicatorState::Active,
            };
        }
        Ok(result)
    }

    /// Elect one abort caller while holding exclusive native-call admission.
    /// Other callers wait on the mutex and then observe the stored result.
    pub(super) fn abort_with(
        &self,
        reason: &str,
        abort: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        match &*state {
            CommunicatorState::Active
            | CommunicatorState::Submitting { .. }
            | CommunicatorState::Failed { .. } => {
                *state = CommunicatorState::Aborting {
                    reason: reason.to_owned(),
                };
                let result = abort();
                let error = result.as_ref().err().map(|error| format!("{error:#}"));
                *state = CommunicatorState::Aborted {
                    reason: reason.to_owned(),
                    error,
                };
                result
            }
            CommunicatorState::Aborting { .. } => {
                unreachable!("abort state is protected for the complete native abort call")
            }
            CommunicatorState::Aborted { error, .. } => stored_abort_result(error),
        }
    }

    pub(super) fn communicator_released(&self) -> bool {
        matches!(
            &*self.state.lock().unwrap(),
            CommunicatorState::Aborted { .. }
        )
    }
}

#[derive(Debug)]
pub(super) struct GroupSubmission {
    pub(super) completion: NcclResult,
    pub(super) deferred_error: Option<anyhow::Error>,
}

fn stored_abort_result(error: &Option<String>) -> Result<()> {
    match error {
        Some(error) => anyhow::bail!(error.clone()),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier, mpsc};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn complete_group_holds_admission_and_abort_result_is_replayed() {
        let lifecycle = Arc::new(CommunicatorLifecycle::new());
        let native_entered = Arc::new(Barrier::new(2));
        let release_native = Arc::new(Barrier::new(2));
        let native = {
            let lifecycle = Arc::clone(&lifecycle);
            let native_entered = Arc::clone(&native_entered);
            let release_native = Arc::clone(&release_native);
            thread::spawn(move || {
                lifecycle.submit_group(|| {
                    native_entered.wait();
                    release_native.wait();
                    Ok(GroupSubmission {
                        completion: NCCL_SUCCESS,
                        deferred_error: None,
                    })
                })
            })
        };
        native_entered.wait();
        assert!(
            lifecycle.state.try_lock().is_err(),
            "a complete NCCL group transaction must retain native admission"
        );

        let release_abort = Arc::new(Barrier::new(2));
        let (abort_entered, abort_entered_rx) = mpsc::channel();
        let (abort_attempted, abort_attempted_rx) = mpsc::channel();
        let first_abort = {
            let lifecycle = Arc::clone(&lifecycle);
            let release_abort = Arc::clone(&release_abort);
            thread::spawn(move || {
                abort_attempted.send(()).unwrap();
                lifecycle.abort_with("rank RPC failed", || {
                    abort_entered.send(()).unwrap();
                    release_abort.wait();
                    anyhow::bail!("injected ncclCommAbort failure")
                })
            })
        };
        abort_attempted_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("abort thread did not reach lifecycle admission");

        release_native.wait();
        let submission = native.join().unwrap().unwrap();
        assert_eq!(submission.completion, NCCL_SUCCESS);
        abort_entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("abort did not acquire native-call ownership");

        let repeated_abort = {
            let lifecycle = Arc::clone(&lifecycle);
            thread::spawn(move || {
                lifecycle.abort_with("different reason", || -> Result<()> {
                    panic!("a repeated abort must not invoke the native abort twice")
                })
            })
        };
        release_abort.wait();

        let first = first_abort
            .join()
            .unwrap()
            .expect_err("the injected native abort must fail");
        let repeated = repeated_abort
            .join()
            .unwrap()
            .expect_err("a concurrent abort caller must observe the elected abort failure");
        assert_eq!(repeated.to_string(), first.to_string());
        assert!(lifecycle.communicator_released());
        let rejected = lifecycle
            .ensure_active()
            .expect_err("abort admission must reject every later native call");
        assert!(rejected.to_string().contains("rank RPC failed"));
    }

    #[test]
    fn successful_abort_is_idempotent_and_marks_the_handle_released() {
        let lifecycle = CommunicatorLifecycle::new();
        lifecycle.abort_with("fatal failure", || Ok(())).unwrap();
        lifecycle
            .abort_with("ignored repeat", || -> Result<()> {
                panic!("a completed abort must not invoke the native abort twice")
            })
            .unwrap();
        assert!(lifecycle.communicator_released());
        assert!(lifecycle.ensure_active().is_err());
    }

    #[test]
    fn submission_errors_fail_closed_and_in_progress_allows_only_poll_or_abort() {
        let direct_failure = CommunicatorLifecycle::new();
        direct_failure
            .submit_group(|| anyhow::bail!("injected ncclGroupEnd failure"))
            .expect_err("the submission must fail");
        assert!(
            direct_failure
                .submit_group(|| { panic!("a failed communicator must reject later submissions") })
                .unwrap_err()
                .to_string()
                .contains("injected ncclGroupEnd failure")
        );
        direct_failure.abort_with("clean up", || Ok(())).unwrap();

        let lifecycle = CommunicatorLifecycle::new();
        lifecycle
            .submit_group(|| {
                Ok(GroupSubmission {
                    completion: NCCL_IN_PROGRESS,
                    deferred_error: None,
                })
            })
            .unwrap();
        assert!(lifecycle.ensure_active().is_err());
        assert_eq!(
            lifecycle.poll_submission(|| Ok(NCCL_IN_PROGRESS)).unwrap(),
            NCCL_IN_PROGRESS
        );
        assert_eq!(
            lifecycle.poll_submission(|| Ok(NCCL_SUCCESS)).unwrap(),
            NCCL_SUCCESS
        );
        lifecycle.ensure_active().unwrap();

        lifecycle
            .submit_group(|| {
                Ok(GroupSubmission {
                    completion: NCCL_IN_PROGRESS,
                    deferred_error: None,
                })
            })
            .unwrap();
        lifecycle
            .abort_with("cancel submission", || Ok(()))
            .unwrap();
        assert!(lifecycle.poll_submission(|| Ok(NCCL_SUCCESS)).is_err());

        let failed = CommunicatorLifecycle::new();
        failed
            .submit_group(|| {
                Ok(GroupSubmission {
                    completion: NCCL_IN_PROGRESS,
                    deferred_error: Some(anyhow::anyhow!("injected grouped operation failure")),
                })
            })
            .unwrap();
        assert_eq!(
            failed.poll_submission(|| Ok(NCCL_SUCCESS)).unwrap(),
            NCCL_SUCCESS
        );
        assert!(
            failed
                .ensure_active()
                .expect_err("a deferred group failure must remain fail-closed")
                .to_string()
                .contains("injected grouped operation failure")
        );
        failed
            .abort_with("clean up failed group", || Ok(()))
            .unwrap();
    }
}
