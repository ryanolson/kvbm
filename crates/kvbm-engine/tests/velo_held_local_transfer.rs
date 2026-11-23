// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing")]

use std::time::Duration;

use anyhow::{Context, Result};
use kvbm_engine::LogicalLayoutHandle;
use kvbm_engine::testing::create_messenger_pair_tcp;
use kvbm_engine::worker::velo::{
    LocalTransferCheckTiming, LocalTransferRecheck, VeloWorkerClient, serve_held_local_transfers,
};
use kvbm_engine::worker::{LogicalResourceId, WorkerTransfers};
use kvbm_physical::transfer::{
    TransferCompleteNotification, TransferDrainOutcome, TransferOptions,
};
use parking_lot::Mutex;
use tokio::sync::oneshot;
use tokio::time::{sleep, timeout};
use uuid::Uuid;

const WORKER: u64 = 1;
const RESOURCE: LogicalResourceId = LogicalResourceId(3);
const DEADLINE: Duration = Duration::from_secs(5);
const COPY_ERROR: &str = "held copy failed after draining";

async fn lose_reply_then_drain(outcome: TransferDrainOutcome) -> Result<TransferDrainOutcome> {
    let pair = create_messenger_pair_tcp().await?;
    let (on_dispatch, dispatched) = oneshot::channel();
    let on_dispatch = Mutex::new(Some(on_dispatch));
    let transfers = serve_held_local_transfers(
        &pair.messenger_b,
        WORKER,
        false,
        move |resource, src, dst| {
            let (finish, held) = oneshot::channel();
            let on_dispatch = on_dispatch.lock().take().expect("one dispatch");
            assert!(
                on_dispatch.send((resource, src, dst, finish)).is_ok(),
                "the caller is waiting for dispatch"
            );
            Ok(TransferCompleteNotification::from_drain_future(async {
                held.await
                    .unwrap_or_else(|error| TransferDrainOutcome::Unproven(error.into()))
            }))
        },
    )?;
    let lose_reply = transfers.gate_reply_loss();
    let worker = pair.messenger_b.instance_id();
    let client = VeloWorkerClient::new_for_worker(pair.messenger_a.clone(), worker, WORKER)
        .with_local_transfer_check(LocalTransferCheckTiming {
            wait: Duration::ZERO,
            deadline: DEADLINE,
        });
    let receipt = client.execute_local_transfer_for_resource(
        RESOURCE,
        LogicalLayoutHandle::G2,
        LogicalLayoutHandle::G1,
        [7].into(),
        [13].into(),
        TransferOptions::default(),
    )?;

    let (resource, src, dst, finish) = timeout(DEADLINE, dispatched).await??;
    assert_eq!(
        (resource, src.as_slice(), dst.as_slice()),
        (RESOURCE, &[7][..], &[13][..])
    );
    lose_reply.send(()).expect("the transfer RPC is waiting");
    let TransferDrainOutcome::Unproven(error) = timeout(DEADLINE, receipt.await_drain()).await?
    else {
        anyhow::bail!("the live RPC failed while its copy was held");
    };
    assert_eq!(
        error.root_cause().to_string(),
        "the worker reports the copy still active",
        "the lost reply must be checked against the production worker record"
    );
    let recheck = error
        .downcast_ref::<LocalTransferRecheck>()
        .context("the lost reply must carry a recheck")?;
    assert_eq!(recheck.instance, worker);
    assert_eq!(recheck.worker_id, Some(WORKER));
    assert!(recheck.incarnation.is_some());

    assert!(finish.send(outcome).is_ok(), "the copy is still held");
    let resolved = timeout(DEADLINE, async {
        loop {
            let outcome = recheck.check(&pair.messenger_a, DEADLINE).await;
            if !matches!(outcome, TransferDrainOutcome::Unproven(_)) {
                return outcome;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .context("the worker did not record the drain")?;

    let wrong_id = LocalTransferRecheck {
        transfer_id: Uuid::nil(),
        ..recheck.clone()
    };
    let wrong_incarnation = LocalTransferRecheck {
        incarnation: Some(Uuid::nil()),
        ..recheck.clone()
    };
    for wrong in [wrong_id, wrong_incarnation] {
        assert!(matches!(
            wrong.check(&pair.messenger_a, DEADLINE).await,
            TransferDrainOutcome::Unproven(_)
        ));
    }
    Ok(resolved)
}

#[tokio::test]
async fn completed_held_copy_resolves_lost_reply() -> Result<()> {
    assert!(matches!(
        lose_reply_then_drain(TransferDrainOutcome::Completed).await?,
        TransferDrainOutcome::Completed
    ));
    Ok(())
}

#[tokio::test]
async fn drained_error_on_held_copy_resolves_lost_reply() -> Result<()> {
    let TransferDrainOutcome::DrainedWithError(error) = lose_reply_then_drain(
        TransferDrainOutcome::DrainedWithError(anyhow::anyhow!(COPY_ERROR)),
    )
    .await?
    else {
        anyhow::bail!("the recheck must return a drained error");
    };
    assert_eq!(error.to_string(), COPY_ERROR);
    Ok(())
}
