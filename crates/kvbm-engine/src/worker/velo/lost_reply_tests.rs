// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A local transfer whose worker reply is lost.
//!
//! The client waits, then checks once with the worker that took the copy.
//! Only a record from the same worker incarnation decides the outcome. Every
//! other answer, and no answer, leaves the receipt `Unproven` with a
//! [`LocalTransferRecheck`] attached.
//!
//! The worker in these tests runs the production local-transfer handlers
//! ([`register_local_transfer_handlers`]) over a stub copy, because the real
//! service needs a CUDA and UCX worker. The fault seams are the `#[cfg(test)]`
//! hooks on [`LocalTransferRecords`].

#![allow(clippy::disallowed_macros)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::check::{CheckReply, IncarnationReply, LocalTransferRecords, Record};
use super::service::{LocalTransferDispatch, register_local_transfer_handlers};
use super::*;
use crate::testing::messenger::{MessengerPair, create_messenger_pair_tcp, create_messenger_tcp};

const WORKER: u64 = 1;
const TIMING: LocalTransferCheckTiming = LocalTransferCheckTiming {
    wait: Duration::from_millis(100),
    deadline: Duration::from_millis(400),
};

/// A stub copy that ends at once with the outcome `outcome` makes.
fn copy_ending(outcome: fn() -> TransferDrainOutcome) -> LocalTransferDispatch {
    Arc::new(move |_message| {
        Ok(TransferCompleteNotification::from_drain_future(
            std::future::ready(outcome()),
        ))
    })
}

fn serve(
    messenger: &Messenger,
    records: &Arc<LocalTransferRecords>,
    dispatch: LocalTransferDispatch,
) -> Result<()> {
    register_local_transfer_handlers(messenger, Some(WORKER), Arc::clone(records), dispatch)
}

fn client(pair: &MessengerPair) -> VeloWorkerClient {
    VeloWorkerClient::new_for_worker(
        Arc::clone(&pair.messenger_a),
        pair.messenger_b.instance_id(),
        WORKER,
    )
    .with_local_transfer_check(TIMING)
}

fn transfer(client: &VeloWorkerClient) -> Result<TransferCompleteNotification> {
    client.execute_local_transfer(
        LogicalLayoutHandle::G2,
        LogicalLayoutHandle::G1,
        Arc::<[BlockId]>::from(vec![0]),
        Arc::<[BlockId]>::from(vec![0]),
        TransferOptions::default(),
    )
}

fn checks(records: &LocalTransferRecords) -> usize {
    records.hooks.checks.load(Ordering::Acquire)
}

/// The re-check capability of an `Unproven` outcome. Panics on any other
/// outcome, or when the capability is missing.
fn recheck_of(outcome: &TransferDrainOutcome) -> LocalTransferRecheck {
    match outcome {
        TransferDrainOutcome::Unproven(error) => error
            .downcast_ref::<LocalTransferRecheck>()
            .cloned()
            .unwrap_or_else(|| panic!("an unproven lost reply must carry its re-check: {error:#}")),
        TransferDrainOutcome::Completed => panic!("expected Unproven, got Completed"),
        TransferDrainOutcome::DrainedWithError(error) => {
            panic!("expected Unproven, got DrainedWithError: {error:#}")
        }
    }
}

/// Assert the capability names the worker that took the copy, under the
/// incarnation of `records`.
fn assert_names_the_worker(
    recheck: &LocalTransferRecheck,
    pair: &MessengerPair,
    records: &LocalTransferRecords,
) {
    assert_eq!(recheck.instance, pair.messenger_b.instance_id());
    assert_eq!(recheck.worker_id, Some(WORKER));
    assert_eq!(recheck.incarnation, Some(records.incarnation()));
}

#[tokio::test]
async fn a_reply_lost_after_the_copy_completed_resolves_completed_after_one_check() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let records = LocalTransferRecords::new();
    serve(
        &pair.messenger_b,
        &records,
        copy_ending(|| TransferDrainOutcome::Completed),
    )?;
    records.hooks.lose_next_reply.store(true, Ordering::Release);

    let outcome = transfer(&client(&pair))?.await_drain().await;

    assert!(
        matches!(outcome, TransferDrainOutcome::Completed),
        "the worker recorded the copy as completed, so the receipt must be Completed, got {}",
        describe(&outcome)
    );
    assert_eq!(checks(&records), 1, "exactly one check");
    Ok(())
}

#[tokio::test]
async fn a_reply_lost_after_a_drained_failure_resolves_drained_with_error() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let records = LocalTransferRecords::new();
    serve(
        &pair.messenger_b,
        &records,
        copy_ending(|| {
            TransferDrainOutcome::DrainedWithError(anyhow::anyhow!("injected drained failure"))
        }),
    )?;
    records.hooks.lose_next_reply.store(true, Ordering::Release);

    let outcome = transfer(&client(&pair))?.await_drain().await;

    match &outcome {
        TransferDrainOutcome::DrainedWithError(error) => {
            assert!(format!("{error:#}").contains("injected drained failure"));
        }
        other => panic!(
            "the worker proved a drained failure, so the receipt must be DrainedWithError, got {}",
            describe(other)
        ),
    }
    assert_eq!(checks(&records), 1, "exactly one check");
    Ok(())
}

#[tokio::test]
async fn a_check_that_arrives_before_the_copy_is_recorded_stays_unproven() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let records = LocalTransferRecords::new();
    serve(
        &pair.messenger_b,
        &records,
        copy_ending(|| TransferDrainOutcome::Completed),
    )?;
    let (open, gate) = tokio::sync::oneshot::channel();
    *records.hooks.defer_next.lock() = Some(gate);

    let outcome = transfer(&client(&pair))?.await_drain().await;

    let recheck = recheck_of(&outcome);
    assert_names_the_worker(&recheck, &pair, &records);
    assert_eq!(checks(&records), 1, "exactly one check");
    // The copy runs after the check. Its record appears under the id the
    // capability names, but the receipt already holds the unknown outcome.
    open.send(()).expect("the deferred transfer is waiting");
    wait_for_record(&records, recheck.transfer_id, Record::Completed).await;
    Ok(())
}

#[tokio::test]
async fn a_check_after_the_record_was_evicted_stays_unproven() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let records = LocalTransferRecords::new();
    serve(
        &pair.messenger_b,
        &records,
        copy_ending(|| TransferDrainOutcome::Completed),
    )?;
    records
        .hooks
        .evict_after_next
        .store(true, Ordering::Release);
    records.hooks.lose_next_reply.store(true, Ordering::Release);

    let outcome = transfer(&client(&pair))?.await_drain().await;

    let recheck = recheck_of(&outcome);
    assert_names_the_worker(&recheck, &pair, &records);
    assert_eq!(checks(&records), 1, "exactly one check");
    assert_eq!(records.check(recheck.transfer_id).record, Record::Unknown);
    Ok(())
}

#[tokio::test]
async fn a_silent_worker_leaves_the_transfer_unproven_within_the_wait_and_deadline() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let incarnation = uuid::Uuid::new_v4();
    let silent_checks = Arc::new(AtomicUsize::new(0));
    serve_silent_worker(&pair.messenger_b, incarnation, &silent_checks)?;

    let client = client(&pair);
    let started = Instant::now();
    let outcome = transfer(&client)?.await_drain().await;
    let elapsed = started.elapsed();

    let recheck = recheck_of(&outcome);
    assert_eq!(recheck.instance, pair.messenger_b.instance_id());
    assert_eq!(recheck.incarnation, Some(incarnation));
    assert_eq!(
        silent_checks.load(Ordering::Acquire),
        1,
        "exactly one check"
    );
    assert!(
        elapsed >= TIMING.wait,
        "the client must wait before it checks, but resolved after {elapsed:?}"
    );
    assert!(
        elapsed < TIMING.wait + TIMING.deadline + Duration::from_millis(500),
        "the check must give up at its deadline, but resolved after {elapsed:?}"
    );
    Ok(())
}

#[tokio::test]
async fn the_same_incarnation_reconnecting_without_the_record_stays_unproven() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let records = LocalTransferRecords::new();
    let (finish, copy_done) = tokio::sync::oneshot::channel::<()>();
    let copy_done = Arc::new(parking_lot::Mutex::new(Some(copy_done)));
    let dispatch: LocalTransferDispatch = Arc::new(move |_message| {
        let copy_done = copy_done.lock().take().expect("one copy");
        Ok(TransferCompleteNotification::from_drain_future(
            async move {
                let _ = copy_done.await;
                TransferDrainOutcome::Completed
            },
        ))
    });
    serve(&pair.messenger_b, &records, dispatch)?;
    records.hooks.lose_next_reply.store(true, Ordering::Release);

    let receipt = transfer(&client(&pair))?;
    let transfer_id = wait_for_active(&records).await;
    // The worker's service comes back with the same incarnation and none of
    // its records, and replaces its handlers. The copy then finishes, and its
    // reply is lost.
    let reconnected = LocalTransferRecords::with_incarnation(records.incarnation());
    serve(
        &pair.messenger_b,
        &reconnected,
        copy_ending(|| TransferDrainOutcome::Completed),
    )?;
    finish.send(()).expect("the copy is running");
    let outcome = receipt.await_drain().await;

    let recheck = recheck_of(&outcome);
    assert_names_the_worker(&recheck, &pair, &records);
    assert_eq!(recheck.transfer_id, transfer_id);
    assert_eq!(checks(&reconnected), 1, "exactly one check");
    Ok(())
}

#[tokio::test]
async fn a_replacement_worker_at_a_new_instance_proves_nothing() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let records = LocalTransferRecords::new();
    serve(
        &pair.messenger_b,
        &records,
        copy_ending(|| TransferDrainOutcome::Completed),
    )?;
    // The original worker stays alive but stops answering checks.
    let silent_checks = Arc::new(AtomicUsize::new(0));
    register_silent_check(&pair.messenger_b, &silent_checks)?;
    records.hooks.lose_next_reply.store(true, Ordering::Release);
    // A replacement at a new instance claims every transfer completed, under
    // the original incarnation.
    let replacement = create_messenger_tcp().await?;
    replacement.register_peer(pair.messenger_a.peer_info())?;
    pair.messenger_a.register_peer(replacement.peer_info())?;
    let replacement_checks = Arc::new(AtomicUsize::new(0));
    register_claiming_check(&replacement, records.incarnation(), &replacement_checks)?;

    let outcome = transfer(&client(&pair))?.await_drain().await;

    let recheck = recheck_of(&outcome);
    assert_names_the_worker(&recheck, &pair, &records);
    assert_eq!(
        silent_checks.load(Ordering::Acquire),
        1,
        "one check, to the original"
    );
    assert_eq!(
        replacement_checks.load(Ordering::Acquire),
        0,
        "the replacement must not be asked"
    );
    Ok(())
}

fn describe(outcome: &TransferDrainOutcome) -> String {
    match outcome {
        TransferDrainOutcome::Completed => "Completed".to_owned(),
        TransferDrainOutcome::DrainedWithError(error) => format!("DrainedWithError({error:#})"),
        TransferDrainOutcome::Unproven(error) => format!("Unproven({error:#})"),
    }
}

async fn wait_for_record(records: &LocalTransferRecords, transfer_id: uuid::Uuid, record: Record) {
    for _ in 0..200 {
        if records.check(transfer_id).record == record {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("transfer {transfer_id} never reached {record:?}");
}

/// Wait until the worker records a transfer as active, and return its id.
async fn wait_for_active(records: &LocalTransferRecords) -> uuid::Uuid {
    for _ in 0..200 {
        if let Some(transfer_id) = records.active_ids().into_iter().next() {
            return transfer_id;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("no transfer became active");
}

/// A worker that takes the transfer, loses every reply, and never answers a
/// check.
fn serve_silent_worker(
    messenger: &Messenger,
    incarnation: uuid::Uuid,
    checks: &Arc<AtomicUsize>,
) -> Result<()> {
    let transfer = ::velo::Handler::unary_handler_async(
        handler_names::scoped(handler_names::LOCAL_TRANSFER, Some(WORKER)),
        |_ctx| async { Err::<Option<Bytes>, _>(anyhow::anyhow!("injected lost reply")) },
    )
    .build();
    messenger.register_handler(transfer)?;
    let reply = Bytes::from(serde_json::to_vec(&IncarnationReply { incarnation })?);
    let identity = ::velo::Handler::unary_handler_async(
        handler_names::scoped(handler_names::WORKER_INCARNATION, Some(WORKER)),
        move |_ctx| {
            let reply = reply.clone();
            async move { Ok(Some(reply)) }
        },
    )
    .build();
    messenger.register_handler(identity)?;
    register_silent_check(messenger, checks)
}

fn register_silent_check(messenger: &Messenger, checks: &Arc<AtomicUsize>) -> Result<()> {
    let checks = Arc::clone(checks);
    let check = ::velo::Handler::unary_handler_async(
        handler_names::scoped(handler_names::LOCAL_TRANSFER_CHECK, Some(WORKER)),
        move |_ctx| {
            checks.fetch_add(1, Ordering::AcqRel);
            async {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok(Some(Bytes::new()))
            }
        },
    )
    .build();
    messenger.register_handler(check)
}

/// A check handler that claims every transfer completed under `incarnation`.
fn register_claiming_check(
    messenger: &Messenger,
    incarnation: uuid::Uuid,
    checks: &Arc<AtomicUsize>,
) -> Result<()> {
    let checks = Arc::clone(checks);
    let reply = Bytes::from(serde_json::to_vec(&CheckReply {
        incarnation,
        record: Record::Completed,
    })?);
    let check = ::velo::Handler::unary_handler_async(
        handler_names::scoped(handler_names::LOCAL_TRANSFER_CHECK, Some(WORKER)),
        move |_ctx| {
            checks.fetch_add(1, Ordering::AcqRel);
            let reply = reply.clone();
            async move { Ok(Some(reply)) }
        },
    )
    .build();
    messenger.register_handler(check)
}

/// One process serves two workers on one messenger. A check reaches the
/// records of the worker that took the copy, and no other.
#[tokio::test]
async fn a_check_reaches_only_the_worker_that_took_the_copy() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let taker = LocalTransferRecords::new();
    let neighbour = LocalTransferRecords::new();
    register_local_transfer_handlers(
        &pair.messenger_b,
        Some(WORKER),
        Arc::clone(&taker),
        copy_ending(|| TransferDrainOutcome::Completed),
    )?;
    register_local_transfer_handlers(
        &pair.messenger_b,
        Some(WORKER + 1),
        Arc::clone(&neighbour),
        copy_ending(|| TransferDrainOutcome::Completed),
    )?;
    taker.hooks.lose_next_reply.store(true, Ordering::Release);

    let outcome = transfer(&client(&pair))?.await_drain().await;

    assert!(
        matches!(outcome, TransferDrainOutcome::Completed),
        "got {}",
        describe(&outcome)
    );
    assert_eq!(checks(&taker), 1);
    assert_eq!(checks(&neighbour), 0);
    Ok(())
}

#[tokio::test]
async fn a_worker_record_without_drain_proof_stays_unproven() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let records = LocalTransferRecords::new();
    serve(
        &pair.messenger_b,
        &records,
        copy_ending(|| TransferDrainOutcome::Unproven(anyhow::anyhow!("injected unproven copy"))),
    )?;
    records.hooks.lose_next_reply.store(true, Ordering::Release);

    let outcome = transfer(&client(&pair))?.await_drain().await;

    let recheck = recheck_of(&outcome);
    assert_names_the_worker(&recheck, &pair, &records);
    assert!(matches!(
        records.check(recheck.transfer_id).record,
        Record::Unproven { .. }
    ));
    assert_eq!(checks(&records), 1, "exactly one check");
    Ok(())
}

/// The worker has no local-transfer handler, so velo fails the send before
/// the request leaves. No copy ran: the outcome is a drained failure, at
/// once, with no wait and no check.
#[tokio::test]
async fn a_transfer_that_never_reached_the_wire_is_a_drained_failure_at_once() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let incarnation = uuid::Uuid::new_v4();
    register_incarnation(
        &pair.messenger_b,
        incarnation,
        &Arc::new(AtomicUsize::new(0)),
    )?;
    let unknown_checks = Arc::new(AtomicUsize::new(0));
    register_unknown_check(&pair.messenger_b, incarnation, &unknown_checks)?;

    let started = Instant::now();
    let outcome = transfer(&client(&pair))?.await_drain().await;
    let elapsed = started.elapsed();

    assert!(
        matches!(outcome, TransferDrainOutcome::DrainedWithError(_)),
        "a send that never reached the wire launched no copy, got {}",
        describe(&outcome)
    );
    assert_eq!(unknown_checks.load(Ordering::Acquire), 0, "no check");
    assert!(
        elapsed < TIMING.wait,
        "a send that never left must not wait, but resolved after {elapsed:?}"
    );
    Ok(())
}

/// The worker has no incarnation handler (an older worker). The transfer runs
/// unbound. A check could not decide, so a lost reply is unknown at once,
/// with no check sent.
#[tokio::test]
async fn an_unbound_lost_reply_is_unproven_without_a_check() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    register_lost_reply_transfer(&pair.messenger_b)?;
    let unknown_checks = Arc::new(AtomicUsize::new(0));
    register_unknown_check(&pair.messenger_b, uuid::Uuid::new_v4(), &unknown_checks)?;

    let started = Instant::now();
    let outcome = transfer(&client(&pair))?.await_drain().await;
    let elapsed = started.elapsed();

    let recheck = recheck_of(&outcome);
    assert_eq!(recheck.incarnation, None);
    assert_eq!(unknown_checks.load(Ordering::Acquire), 0, "no check");
    assert!(
        elapsed < TIMING.wait,
        "an unbound lost reply must not wait for a check, but resolved after {elapsed:?}"
    );
    Ok(())
}

/// The first incarnation request times out. The client must ask again on the
/// next transfer, not treat the worker as an older one.
#[tokio::test]
async fn an_incarnation_request_that_timed_out_is_asked_again_on_the_next_transfer() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let records = LocalTransferRecords::new();
    serve(
        &pair.messenger_b,
        &records,
        copy_ending(|| TransferDrainOutcome::Completed),
    )?;
    let asks = Arc::new(AtomicUsize::new(0));
    register_slow_first_incarnation(&pair.messenger_b, records.incarnation(), &asks)?;
    let client = client(&pair);

    let first = transfer(&client)?.await_drain().await;
    assert!(
        matches!(first, TransferDrainOutcome::Completed),
        "got {}",
        describe(&first)
    );
    records.hooks.lose_next_reply.store(true, Ordering::Release);
    let second = transfer(&client)?.await_drain().await;

    assert_eq!(asks.load(Ordering::Acquire), 2, "the client asked again");
    assert!(
        matches!(second, TransferDrainOutcome::Completed),
        "the second transfer is bound, so its check decides, got {}",
        describe(&second)
    );
    Ok(())
}

/// An older worker has no incarnation handler. velo rejects the request at
/// once, and the client does not ask again on the next transfer.
#[tokio::test]
async fn an_older_worker_is_not_asked_for_its_incarnation_again_at_once() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let transfer_handler = ::velo::Handler::unary_handler_async(
        handler_names::scoped(handler_names::LOCAL_TRANSFER, Some(WORKER)),
        |_ctx| async { Ok(Some(Bytes::new())) },
    )
    .build();
    pair.messenger_b.register_handler(transfer_handler)?;
    let client = client(&pair);

    let first = transfer(&client)?.await_drain().await;
    assert!(
        matches!(first, TransferDrainOutcome::Completed),
        "got {}",
        describe(&first)
    );
    let asks = Arc::new(AtomicUsize::new(0));
    register_incarnation(&pair.messenger_b, uuid::Uuid::new_v4(), &asks)?;
    let second = transfer(&client)?.await_drain().await;

    assert!(
        matches!(second, TransferDrainOutcome::Completed),
        "got {}",
        describe(&second)
    );
    assert_eq!(
        asks.load(Ordering::Acquire),
        0,
        "not asked again within the retry window"
    );
    Ok(())
}

/// A legacy request has no transfer id. The worker runs it and replies, and
/// keeps no record.
#[tokio::test]
async fn a_legacy_request_without_a_transfer_id_runs_and_keeps_no_record() -> Result<()> {
    let pair = create_messenger_pair_tcp().await?;
    let records = LocalTransferRecords::new();
    serve(
        &pair.messenger_b,
        &records,
        copy_ending(|| TransferDrainOutcome::Completed),
    )?;
    let legacy = serde_json::json!({
        "src": "G2",
        "dst": "G1",
        "src_block_ids": [0],
        "dst_block_ids": [0],
        "options": {
            "layer_range": null,
            "nixl_write_notification": null,
            "bounce_buffer_handle": null,
            "bounce_buffer_block_ids": null,
            "metric_route": null
        }
    });
    let reply = pair
        .messenger_a
        .unary(&handler_names::scoped(
            handler_names::LOCAL_TRANSFER,
            Some(WORKER),
        ))?
        .raw_payload(Bytes::from(serde_json::to_vec(&legacy)?))
        .instance(pair.messenger_b.instance_id())
        .send()
        .await?;

    assert!(matches!(
        LocalTransferReply::decode(&reply),
        TransferDrainOutcome::Completed
    ));
    assert!(records.active_ids().is_empty());
    assert_eq!(records.record_count(), 0, "no record for a legacy request");
    Ok(())
}

fn register_incarnation(
    messenger: &Messenger,
    incarnation: uuid::Uuid,
    asks: &Arc<AtomicUsize>,
) -> Result<()> {
    let asks = Arc::clone(asks);
    let reply = Bytes::from(serde_json::to_vec(&IncarnationReply { incarnation })?);
    let handler = ::velo::Handler::unary_handler_async(
        handler_names::scoped(handler_names::WORKER_INCARNATION, Some(WORKER)),
        move |_ctx| {
            asks.fetch_add(1, Ordering::AcqRel);
            let reply = reply.clone();
            async move { Ok(Some(reply)) }
        },
    )
    .build();
    messenger.register_handler(handler)
}

/// An incarnation handler whose first answer comes after the check deadline.
fn register_slow_first_incarnation(
    messenger: &Messenger,
    incarnation: uuid::Uuid,
    asks: &Arc<AtomicUsize>,
) -> Result<()> {
    let asks = Arc::clone(asks);
    let reply = Bytes::from(serde_json::to_vec(&IncarnationReply { incarnation })?);
    let handler = ::velo::Handler::unary_handler_async(
        handler_names::scoped(handler_names::WORKER_INCARNATION, Some(WORKER)),
        move |_ctx| {
            let first = asks.fetch_add(1, Ordering::AcqRel) == 0;
            let reply = reply.clone();
            async move {
                if first {
                    tokio::time::sleep(TIMING.deadline * 2).await;
                }
                Ok(Some(reply))
            }
        },
    )
    .build();
    messenger.register_handler(handler)
}

fn register_lost_reply_transfer(messenger: &Messenger) -> Result<()> {
    let handler = ::velo::Handler::unary_handler_async(
        handler_names::scoped(handler_names::LOCAL_TRANSFER, Some(WORKER)),
        |_ctx| async { Err::<Option<Bytes>, _>(anyhow::anyhow!("injected lost reply")) },
    )
    .build();
    messenger.register_handler(handler)
}

/// A check handler that answers "no record" at once.
fn register_unknown_check(
    messenger: &Messenger,
    incarnation: uuid::Uuid,
    checks: &Arc<AtomicUsize>,
) -> Result<()> {
    let checks = Arc::clone(checks);
    let reply = Bytes::from(serde_json::to_vec(&CheckReply {
        incarnation,
        record: Record::Unknown,
    })?);
    let handler = ::velo::Handler::unary_handler_async(
        handler_names::scoped(handler_names::LOCAL_TRANSFER_CHECK, Some(WORKER)),
        move |_ctx| {
            checks.fetch_add(1, Ordering::AcqRel);
            let reply = reply.clone();
            async move { Ok(Some(reply)) }
        },
    )
    .build();
    messenger.register_handler(handler)
}
