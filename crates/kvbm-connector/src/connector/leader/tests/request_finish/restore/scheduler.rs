// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Stdio;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use super::{FinishedRequests, FinishedStatus, Leader, REQUEST, Value, WAIT, json};
use anyhow::{Context, Result, ensure};

pub(super) struct SchedulerCallbacks {
    pinned: Option<PinnedScheduler>,
    scheduler_assertions: Vec<u64>,
}

impl SchedulerCallbacks {
    pub(super) fn new() -> Result<Self> {
        let pinned = std::env::var_os("KVBM_PINNED_VLLM_SCHEDULER")
            .map(PinnedScheduler::spawn)
            .transpose()?;
        Ok(Self {
            pinned,
            scheduler_assertions: Vec::new(),
        })
    }

    pub(super) async fn abort(
        &mut self,
        leader: &Leader,
        waiting_for_remote: bool,
    ) -> Result<bool> {
        self.finish_request(leader, "abort", waiting_for_remote)
            .await
    }

    pub(super) async fn stop_preempted(&mut self, leader: &Leader) -> Result<bool> {
        self.finish_request(leader, "stop", false).await
    }

    async fn finish_request(
        &mut self,
        leader: &Leader,
        command: &str,
        waiting_for_remote: bool,
    ) -> Result<bool> {
        if let Some(scheduler) = &mut self.pinned {
            scheduler
                .send(json!({
                    "command": command,
                    "request": REQUEST,
                    "waiting_for_remote": waiting_for_remote,
                }))
                .await?;
            assert_eq!(
                scheduler.receive().await?,
                json!({"callback": "request_finished", "request": REQUEST})
            );
        }
        let pending = leader.request_finished(REQUEST) == FinishedStatus::Pending;
        if let Some(scheduler) = &mut self.pinned {
            scheduler.send(json!({"pending": pending})).await?;
            assert_eq!(scheduler.receive().await?, json!({"retained": true}));
        }
        Ok(pending)
    }

    pub(super) async fn deliver(
        &mut self,
        leader: &Leader,
        output: &FinishedRequests,
    ) -> Result<()> {
        let sets = json!({
            "finished_sending": output.offloading,
            "finished_recving": output.onboarding,
        });
        if let Some(scheduler) = &mut self.pinned {
            scheduler
                .send(json!({"command": "deliver", "output": sets}))
                .await?;
            assert_eq!(
                scheduler.receive().await?,
                json!({"callback": "update_connector_output", "output": sets})
            );
        }
        leader.update_connector_output(output.offloading.clone(), output.onboarding.clone())?;
        if let Some(scheduler) = &mut self.pinned {
            scheduler.send(json!({"updated": true})).await?;
            let result = scheduler.receive().await?;
            if let Some(line) = result["assertion"].as_u64() {
                self.scheduler_assertions.push(line);
            }
        }
        Ok(())
    }

    pub(super) async fn assert_one_free(mut self) -> Result<()> {
        if let Some(scheduler) = &mut self.pinned {
            scheduler.send(json!({"command": "end"})).await?;
            let result = scheduler.receive().await?;
            assert_eq!(result, json!({"frees": [REQUEST]}));
            ensure!(
                tokio::time::timeout(WAIT, scheduler.child.wait())
                    .await??
                    .success()
            );
            assert!(
                self.scheduler_assertions.is_empty(),
                "pinned vLLM scheduler assertions at {:?}",
                self.scheduler_assertions
            );
        }
        Ok(())
    }
}

struct PinnedScheduler {
    child: Child,
    commands: ChildStdin,
    results: BufReader<ChildStdout>,
}

impl PinnedScheduler {
    fn spawn(source: std::ffi::OsString) -> Result<Self> {
        let mut child = Command::new("python3")
            .args(["-B", "-u", "-c", include_str!("pinned_scheduler.py")])
            .arg(source)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;
        let commands = child.stdin.take().context("scheduler command pipe")?;
        let results = BufReader::new(child.stdout.take().context("scheduler result pipe")?);
        Ok(Self {
            child,
            commands,
            results,
        })
    }

    async fn send(&mut self, message: Value) -> Result<()> {
        let line = format!("{message}\n");
        tokio::time::timeout(WAIT, self.commands.write_all(line.as_bytes())).await??;
        Ok(())
    }

    async fn receive(&mut self) -> Result<Value> {
        let mut line = String::new();
        let received = tokio::time::timeout(WAIT, self.results.read_line(&mut line)).await??;
        ensure!(received != 0, "pinned scheduler closed the result pipe");
        Ok(serde_json::from_str(&line)?)
    }
}
