// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The local G1←G2 onboard driver.
//!
//! The driver owns the fenced action, local worker dispatch, source holds,
//! and both terminals. Each entry point prepares its copies and projects the
//! outcome for its own request lifecycle.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use futures::StreamExt;
use futures::future::{BoxFuture, Either, Ready};
use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_physical::TransferOptions;
use kvbm_physical::transfer::{TransferCompleteNotification, TransferDrainOutcome};
use kvbm_protocols::connector::{
    ActionFailure, ActionId, ActionStatus, BlockId, FenceHandle, LeaderEngine, OnboardHandle,
    RequestId, ResourceOnboard, SequenceHash,
};
use tokio_util::sync::CancellationToken;

use super::driver::ActionRecord;
use super::inflight::InflightKey;
use super::local::LocalConnectorEngine;
use super::paused::{CdDecodeContinuation, LoadContinuation, PauseCell, PausedLoad};
use super::reconcile::OnboardingState;
use crate::G2;
use crate::g2_capacity::reserve_required_staging;
use crate::leader::InstanceLeader;
use crate::p2p::session::{AvailabilityDelta, CommitDelta, Session};
use crate::remote::cd::prefill::PrefillKick;
use crate::remote::cd::state::CdRequestState;

/// The per-shard staging future type — exactly what
/// [`crate::leader::FindMatchesResult::wait_for_completion`] yields. A `Ready`
/// left arm for already-terminal `Ready` shards, a boxed right arm for the
/// async-session path.
pub(super) type StagingCompletion = Either<Ready<Result<()>>, BoxFuture<'static, Result<()>>>;

/// Select the G1 block IDs that correspond to externally-matched (onboard)
/// blocks from the request's full allocation.
///
/// `dest` is `[computed_blocks… | external_blocks… | new_blocks…]`; the external
/// (matched) blocks begin right after the computed prefix and span
/// `num_external_tokens / block_size` entries. Bounds are clamped so a `dest`
/// shorter than expected degrades to an empty/partial selection rather than
/// panicking.
pub(super) fn select_onboard_block_ids(
    dest: &[BlockId],
    num_computed_tokens: usize,
    num_external_tokens: usize,
    block_size: usize,
) -> Vec<BlockId> {
    let num_computed_blocks = num_computed_tokens / block_size;
    let num_external_blocks = num_external_tokens / block_size;
    let end = (num_computed_blocks + num_external_blocks).min(dest.len());
    let start = num_computed_blocks.min(end);
    dest[start..end].to_vec()
}

/// Collect the accepted prefix of the contiguous matched G2 span.
/// A hybrid checkpoint can end before the available block history ends.
fn collect_g2_blocks(
    state: &mut OnboardingState,
    block_size: usize,
    desired_blocks: usize,
) -> Result<Vec<ImmutableBlock<G2>>> {
    debug_assert!(!state.shards.is_empty());
    debug_assert!(state.all_shards_terminal());

    let (effective_start, final_end) = state.matched_span(block_size);
    debug_assert!(effective_start <= final_end);
    if desired_blocks > final_end - effective_start {
        bail!("accepted onboard span exceeds the matched G2 prefix");
    }
    let leading_skip = effective_start - state.shards[0].start_block;

    let mut collected: Vec<ImmutableBlock<G2>> = Vec::new();
    for shard in state.shards.iter_mut() {
        if collected.len() >= leading_skip + desired_blocks {
            break;
        }
        let blocks = shard
            .find_session
            .take_g2_blocks()
            .ok_or_else(|| anyhow!("No G2 blocks found for shard at {}", shard.start_block))?;
        collected.extend(blocks);
    }

    if leading_skip > collected.len() {
        bail!(
            "effective_start mask ({}) exceeds collected g2 blocks ({})",
            leading_skip,
            collected.len()
        );
    }
    collected.drain(..leading_skip);

    if collected.len() < desired_blocks {
        bail!(
            "collected {} g2 blocks but span requested {}",
            collected.len(),
            desired_blocks
        );
    }
    collected.truncate(desired_blocks);

    Ok(collected)
}

pub(super) enum OnboardDrain<H, T = ActionStatus> {
    Proven(T),
    Unproven { error: anyhow::Error, holds: H },
}

/// Keep the source holds until the receipt proves drain or the caller takes ownership.
async fn await_onboard_drain<H>(
    notification: TransferCompleteNotification,
    holds: H,
) -> OnboardDrain<H, Result<()>> {
    match notification.await_drain().await {
        TransferDrainOutcome::Completed => OnboardDrain::Proven(Ok(())),
        TransferDrainOutcome::DrainedWithError(error) => OnboardDrain::Proven(Err(error)),
        TransferDrainOutcome::Unproven(error) => OnboardDrain::Unproven { error, holds },
    }
}

/// A copy whose sources and route-specific settlement must survive the driver task.
pub(super) struct PendingOnboard {
    holds: Box<dyn Send>,
    continuation: Option<LoadContinuation>,
}

pub(super) type OnboardWork = OnboardDrain<PendingOnboard>;

impl OnboardWork {
    pub(super) fn unproven(
        error: anyhow::Error,
        holds: impl Send + 'static,
        continuation: Option<LoadContinuation>,
    ) -> Self {
        Self::Unproven {
            error,
            holds: PendingOnboard {
                holds: Box::new(holds),
                continuation,
            },
        }
    }
}

pub(super) struct OnboardWatchdog {
    pub(super) cancel: CancellationToken,
    pub(super) duration: Duration,
}

impl LocalConnectorEngine {
    /// Register before the handle escapes so eviction and request finish can
    /// fence the whole onboard, including staging and any parked prefill kick.
    pub(super) fn mint_onboard(
        self: &Arc<Self>,
        request: RequestId,
        destinations: Vec<BlockId>,
        cancel: Option<CancellationToken>,
        inflight_hashes: Option<Vec<SequenceHash>>,
    ) -> (OnboardHandle, PausedLoad) {
        let action = ActionId::new();
        let cell = Arc::new(Mutex::new(ActionStatus::Pending));
        let physical_drain = Arc::new(AtomicBool::new(false));
        let pause = PauseCell::default();
        let fenced = cancel.is_some();
        let mut record = ActionRecord::new(request.clone(), Arc::downgrade(&cell));
        if let Some(cancel) = cancel {
            record = record.with_physical_drain(cancel);
        }
        let key = InflightKey::Action(action);
        if inflight_hashes.is_some() {
            record = record.with_inflight(key);
        }
        self.actions.insert(action, record);
        if let Some(hashes) = inflight_hashes {
            self.inflight
                .lock()
                .expect("inflight-guard mutex poisoned")
                .record(key, hashes);
        }
        self.by_request
            .entry(request.clone())
            .or_default()
            .push(action);

        let owner: Arc<dyn LeaderEngine> = self.clone();
        let handle = if fenced {
            OnboardHandle::new_with_physical_drain(
                action,
                Arc::downgrade(&owner),
                cell,
                destinations.clone(),
                FenceHandle::new(physical_drain.clone()),
            )
            .with_pause_cell(pause.clone())
        } else {
            // CD rejections before dispatch have always returned an unfenced handle.
            OnboardHandle::new(action, Arc::downgrade(&owner), cell, destinations.clone())
        };
        let load = PausedLoad {
            action,
            request,
            destinations,
            physical_drain,
            pause,
            holds: Box::new(()),
            continuation: None,
        };
        (handle, load)
    }

    pub(super) fn reject_cd_onboard(
        self: &Arc<Self>,
        request: RequestId,
        destinations: Vec<BlockId>,
        state: Arc<CdRequestState>,
    ) -> OnboardHandle {
        let (handle, load) = self.mint_onboard(request, destinations, None, None);
        let status = ActionStatus::Failed(ActionFailure::Partial {
            block_ids: load.destinations.clone(),
        });
        self.settle_onboard(
            load.action,
            &load.request,
            Some((status, load.destinations)),
            None,
            Some(LoadContinuation::CdDecode(CdDecodeContinuation::new(state))),
        );
        handle
    }

    pub(super) fn spawn_onboard(
        self: &Arc<Self>,
        load: PausedLoad,
        work: impl Future<Output = OnboardWork> + Send + 'static,
        watchdog: Option<OnboardWatchdog>,
    ) {
        let driver = self.clone();
        self.leader.runtime().spawn(async move {
            driver.drive_onboard(load, work, watchdog).await;
        });
    }

    /// Run route-specific preparation and copies under a single terminal owner.
    /// The bundle watchdog or cancel can settle the logical result early. The
    /// copy task continues to proof or a pause in either case. A bundle pause
    /// never dispatches its tail: a Completed proof preserves landed blocks
    /// and fails the unfilled ids; a drained error keeps the resource failure.
    pub(super) async fn drive_onboard(
        self: &Arc<Self>,
        mut load: PausedLoad,
        work: impl Future<Output = OnboardWork> + Send + 'static,
        watchdog: Option<OnboardWatchdog>,
    ) {
        #[cfg(test)]
        let action = load.action;
        let mut logical_done = false;
        let outcome = if let Some(OnboardWatchdog { cancel, duration }) = watchdog {
            let mut physical = self.leader.runtime().spawn(work);
            let ready = tokio::select! {
                biased;
                result = &mut physical => Some(result),
                () = cancel.cancelled() => None,
                () = tokio::time::sleep(duration) => {
                    tracing::warn!(request_id = %load.request, watchdog = ?duration,
                        "bundle onboard exceeded its logical watchdog");
                    None
                }
            };
            let result = match ready {
                Some(result) => result,
                None => {
                    self.fail_onboard_logically(&mut load);
                    logical_done = true;
                    physical.await
                }
            };
            match result {
                Ok(outcome) => outcome,
                Err(error) => {
                    tracing::error!(%error, request_id = %load.request,
                        "bundle onboard physical task failed; destination quarantine remains fail-closed");
                    if !logical_done {
                        self.fail_onboard_logically(&mut load);
                    }
                    return;
                }
            }
        } else {
            work.await
        };

        match outcome {
            OnboardDrain::Proven(status) => load.finish(self, (!logical_done).then_some(status)),
            OnboardDrain::Unproven { error, holds } => {
                load.holds = holds.holds;
                if let Some(mut continuation) = holds.continuation {
                    if let LoadContinuation::Bundle(bundle) = &mut continuation {
                        bundle.logical_done = logical_done;
                    }
                    load.continuation = Some(continuation);
                }
                self.pause_load_action(load, error);
            }
        }
        #[cfg(test)]
        super::test_support::note_onboard_driver_finished(action);
    }

    fn fail_onboard_logically(&self, load: &mut PausedLoad) {
        self.settle_onboard(
            load.action,
            &load.request,
            Some((
                ActionStatus::Failed(ActionFailure::AllBlocks),
                std::mem::take(&mut load.destinations),
            )),
            None,
            None,
        );
    }

    fn settle_onboard(
        &self,
        action: ActionId,
        request: &RequestId,
        logical: Option<(ActionStatus, Vec<BlockId>)>,
        physical: Option<&Arc<AtomicBool>>,
        continuation: Option<LoadContinuation>,
    ) {
        // Stash a failed prefill kick before the logical terminal. Cleanup
        // follows physical settlement so release can see the stash at either.
        let prefill_reason = "prefill onboard kick failed";
        let failed_prefill = match (&continuation, &logical) {
            (Some(LoadContinuation::CdPrefill(state)), Some((ActionStatus::Failed(_), _))) => {
                state.stash_failure(prefill_reason.to_owned());
                Some(state)
            }
            _ => None,
        };

        if let Some((status, destinations)) = logical {
            let cd_status = matches!(&continuation, Some(LoadContinuation::CdDecode(_)))
                .then(|| status.clone());
            self.finish_load_action(action, request, status, destinations);
            if let (Some(LoadContinuation::CdDecode(cd)), Some(status), Some(runtime)) =
                (&continuation, cd_status, &self.cd)
            {
                runtime.complete_load(request, &cd.state, &status);
            }
        }
        if let Some(fence) = physical {
            self.finish_physical_load_action(action, fence);
        }
        if let Some(state) = failed_prefill {
            self.cleanup_failed_prefill(request, state, prefill_reason.to_owned());
        }
    }
}

impl PausedLoad {
    pub(super) fn finish(self, engine: &LocalConnectorEngine, status: Option<ActionStatus>) {
        drop(self.holds);
        engine.settle_onboard(
            self.action,
            &self.request,
            status.map(|status| (status, self.destinations)),
            Some(&self.physical_drain),
            self.continuation,
        );
    }
}

impl PrefillKick {
    pub(super) fn into_load(self, request: RequestId) -> PausedLoad {
        PausedLoad {
            action: self.action_id,
            request,
            destinations: self.external_g1,
            physical_drain: self.physical_drain,
            pause: self.pause,
            holds: Box::new(()),
            continuation: None,
        }
    }
}

/// The only G2 to G1 dispatch site. A dispatch error means no copy launched.
/// Once a receipt exists, its typed drain outcome controls the source hold.
pub(super) async fn dispatch_onboard<H>(
    leader: &InstanceLeader,
    transfer: ResourceOnboard,
    holds: H,
) -> Result<OnboardDrain<H, Result<()>>> {
    let notification = leader.execute_local_transfer_for_resource(
        transfer.resource,
        LogicalLayoutHandle::G2,
        LogicalLayoutHandle::G1,
        transfer.source_block_ids,
        transfer.destination_block_ids,
        TransferOptions::default(),
    )?;
    Ok(await_onboard_drain(notification, holds).await)
}

pub(super) async fn dispatch_pinned_onboard(
    leader: &InstanceLeader,
    sources: Vec<ImmutableBlock<G2>>,
    destinations: Vec<BlockId>,
) -> Result<OnboardDrain<Vec<ImmutableBlock<G2>>, Result<()>>> {
    let transfer = ResourceOnboard {
        resource: leader.primary_g2_resource(),
        source_block_ids: sources.iter().map(|block| block.block_id()).collect(),
        destination_block_ids: destinations,
    };
    dispatch_onboard(leader, transfer, sources).await
}

/// Wait for staging and drive the G2 to G1 copy on the leader's runtime.
pub(super) async fn run_onboard(
    leader: &Arc<InstanceLeader>,
    onboarding: &mut OnboardingState,
    g1_block_ids: Vec<BlockId>,
    staging_futs: Vec<StagingCompletion>,
    block_size: usize,
    accepted_blocks: usize,
) -> OnboardDrain<Vec<ImmutableBlock<G2>>> {
    if g1_block_ids.len() != accepted_blocks {
        tracing::error!(
            accepted = accepted_blocks,
            destinations = g1_block_ids.len(),
            "onboard allocation does not cover the accepted prefix"
        );
        return OnboardDrain::Proven(ActionStatus::Failed(ActionFailure::AllBlocks));
    }
    // Nothing external to move (e.g. a `Matched { hit_blocks: 0 }` onboard):
    // immediately terminal, no transfer issued.
    if g1_block_ids.is_empty() {
        return OnboardDrain::Proven(ActionStatus::Complete);
    }

    // Wait for every shard's find_session to reach a terminal state.
    for fut in staging_futs {
        if let Err(e) = fut.await {
            tracing::error!(error = %e, "onboard staging future failed");
            return OnboardDrain::Proven(ActionStatus::Failed(ActionFailure::AllBlocks));
        }
    }

    // Pull the matched G2 sources (held alive until the transfer completes).
    let g2_blocks = match collect_g2_blocks(onboarding, block_size, accepted_blocks) {
        Ok(blocks) => blocks,
        Err(e) => {
            tracing::error!(error = %e, "onboard G2 collect failed");
            return OnboardDrain::Proven(ActionStatus::Failed(ActionFailure::AllBlocks));
        }
    };
    match dispatch_pinned_onboard(leader, g2_blocks, g1_block_ids).await {
        Ok(drain) => match drain {
            OnboardDrain::Proven(Ok(())) => OnboardDrain::Proven(ActionStatus::Complete),
            OnboardDrain::Proven(Err(error)) => {
                tracing::error!(%error, "onboard transfer drained with an error");
                OnboardDrain::Proven(ActionStatus::Failed(ActionFailure::AllBlocks))
            }
            OnboardDrain::Unproven { error, holds } => OnboardDrain::Unproven { error, holds },
        },
        Err(e) => {
            tracing::error!(error = %e, "onboard G2->G1 transfer dispatch failed");
            OnboardDrain::Proven(ActionStatus::Failed(ActionFailure::AllBlocks))
        }
    }
}

/// Pull the remote-prefill slice over the parked conditional-disagg session and
/// onboard each run G2→G1. Ported from the legacy decode `run_remote_pipeline`,
/// adapted to the connector engine:
///
/// * No token sequence — the engine deals in hashes, so each pulled block is
///   staged by its committed hash via the same `MutableBlock::stage(hash,
///   block_size)` path `p2p::control` uses (it never needs a `TokenBlock`).
/// * No cancel token — eviction/decline closes the session, which surfaces here
///   as a `Closed`/short commit stream or a drained availability stream (or a
///   stream end). The engine's eviction fences gate the worker copies; this task
///   simply bails when its drains run dry, and the load terminal resolves the
///   unfilled remote dest ids.
///
/// This loop's liveness depends on the session terminating BOTH streams on
/// teardown — a `CommitDelta::Closed` and an `AvailabilityDelta::Drained` (or a
/// stream end). The `MockSession` used in the engine tests (`u5`, mid-pull)
/// satisfies this by dropping its stream senders. The production `VeloSession`
/// now satisfies it BY CONSTRUCTION: a LOCAL `close()` (evict / decline) and the
/// monitor's terminal exit (peer finalize / detach / stream error) each push
/// `CommitDelta::Closed` + `AvailabilityDelta::Drained` into the local
/// subscriber streams — idempotently shared with the inbound peer-terminator
/// dispatch — so a driver parked on `commits()`/`availability()` is released on
/// either local close or peer crash rather than stranding forever. See
/// `p2p::session::velo::VeloSession::close` and `spawn_monitor`'s post-loop
/// `close_local_commit_stream` / `drain_local_avail_stream` calls.
///
/// `remote_pairs` is the expected remote slice in absolute-position order:
/// `(expected_hash, g1_dst_block_id)`. `filled` accumulates the hashes whose
/// blocks have landed in G1. The caller uses it to project failures. An
/// unproven run returns its pins and destinations and stops the pipeline.
///
/// Superset tolerance: the peer may legitimately publish MORE than this
/// onboard's slice — a conditional-disagg prefill commits its whole computed
/// suffix, which can include blocks the decode excluded (e.g. the
/// recompute-tail block), since the prefill cannot know the decode's unified
/// bound. Unexpected hashes are warned and ignored on BOTH drains (they never
/// count toward the commit barrier and never bail the availability phase),
/// and a hash duplicated within one availability delta (a peer
/// double-publish, or the wire replay coalescing deltas) is pulled once;
/// the real liveness guards stay: a `Closed` before the full expected commit
/// set and a `Drained` with unfilled expected hashes both fail the onboard.
pub(super) async fn run_remote_onboard(
    leader: &Arc<InstanceLeader>,
    session: &Arc<dyn Session>,
    remote_pairs: &[(SequenceHash, BlockId)],
    block_size: usize,
    filled: &mut HashSet<SequenceHash>,
) -> Result<OnboardDrain<PendingRemoteCopy, ()>> {
    let expected_count = remote_pairs.len();
    if expected_count == 0 {
        return Ok(OnboardDrain::Proven(()));
    }

    // Gate the pull pipeline on attach completion: the peer's worker metadata
    // must be imported before we drain commits and pull, otherwise the RDMA
    // read races the metadata exchange and fails with "invalid source handle".
    // The holder-side `Frame::Attach` handler imports the peer metadata on a
    // spawned task and only opens this gate once it resolves, so waiting here
    // restores the awaited-attach sequencing the legacy disagg coordinator
    // owned. (Defence-in-depth: `rdma_pull_with_opts` also single-flights the
    // import, so the pull stays correct even if this gate is a no-op.)
    session
        .wait_attached()
        .await
        .map_err(|e| anyhow!("cd remote onboard: attach gate failed: {e}"))?;

    // Position (slot) index of each expected remote hash.
    let slot_of: HashMap<SequenceHash, usize> = remote_pairs
        .iter()
        .enumerate()
        .map(|(i, (hash, _))| (*hash, i))
        .collect();

    // 1. Drain commits until every expected remote hash has been committed by the
    //    peer. Only expected hashes count toward the barrier — a superset batch
    //    must not satisfy it while expected hashes are still missing. A `Closed`
    //    terminator before the full expected set is protocol under-delivery —
    //    the peer promised more than it committed, or the session was torn down.
    let mut commit_seen: HashSet<SequenceHash> = HashSet::new();
    let mut commits = session.commits();
    while let Some(delta) = commits.next().await {
        match delta {
            CommitDelta::Added(hashes) => {
                for h in hashes {
                    if slot_of.contains_key(&h) {
                        commit_seen.insert(h);
                    } else {
                        tracing::warn!(
                            hash = ?h,
                            "cd remote onboard: ignoring committed hash outside the \
                             expected remote slice"
                        );
                    }
                }
                if commit_seen.len() >= expected_count {
                    break;
                }
            }
            CommitDelta::Closed => {
                if commit_seen.len() < expected_count {
                    bail!(
                        "cd remote onboard: commits closed before all expected hashes arrived \
                         (got {} of {})",
                        commit_seen.len(),
                        expected_count
                    );
                }
                break;
            }
        }
    }
    drop(commits);

    // 2. Drain availability and pull each chunk. Availability deltas may be sparse
    //    or coalesced, so each delta is filtered to expected-and-unfilled, sorted
    //    by slot index, and split into maximal contiguous runs — one onboard
    //    transaction per run.
    let mut avail = session.availability();
    while let Some(delta) = avail.next().await {
        let blocks = match delta {
            AvailabilityDelta::Available(blocks) => blocks,
            AvailabilityDelta::Verified(records) => {
                records.into_iter().map(|record| record.block).collect()
            }
            AvailabilityDelta::Drained => break,
        };
        let mut indexed: Vec<(usize, SequenceHash)> = blocks
            .into_iter()
            .filter_map(|b| match slot_of.get(&b.hash) {
                Some(slot) if !filled.contains(&b.hash) => Some((*slot, b.hash)),
                Some(_) => None,
                None => {
                    tracing::warn!(
                        hash = ?b.hash,
                        "cd remote onboard: ignoring available hash outside the \
                         expected remote slice"
                    );
                    None
                }
            })
            .collect();
        if indexed.is_empty() {
            continue;
        }
        indexed.sort_by_key(|(slot, _)| *slot);
        // The `filled` filter only dedups ACROSS deltas; a hash
        // duplicated WITHIN one delta (a peer double-publish, or the
        // replay coalescer merging pre-subscribe deltas) would land
        // the same slot twice — violating the strictly-increasing
        // run contract and double-pulling a hash whose holder pin
        // the first pull's ack already dropped.
        indexed.dedup_by_key(|(slot, _)| *slot);

        for run in group_contiguous_runs(indexed) {
            match pull_register_onboard_run(leader, session, remote_pairs, &run, block_size).await?
            {
                OnboardDrain::Proven(result) => result?,
                OnboardDrain::Unproven { error, holds } => {
                    return Ok(OnboardDrain::Unproven {
                        error,
                        holds: PendingRemoteCopy {
                            sources: holds,
                            destinations: run
                                .iter()
                                .map(|(slot, _)| remote_pairs[*slot].1)
                                .collect(),
                        },
                    });
                }
            }
            for (_, hash) in &run {
                filled.insert(*hash);
            }
        }
        if filled.len() == expected_count {
            break;
        }
    }
    drop(avail);

    if filled.len() != expected_count {
        bail!(
            "cd remote onboard: availability drained with {} of {} remote hashes filled",
            filled.len(),
            expected_count
        );
    }
    Ok(OnboardDrain::Proven(()))
}

pub(super) struct PendingRemoteCopy {
    pub(super) sources: Vec<ImmutableBlock<G2>>,
    pub(super) destinations: Vec<BlockId>,
}

/// Split position-indexed entries (pre-sorted ascending) into maximal
/// contiguous runs — one pull/register transaction per run. Sparse or
/// coalesced availability deltas are valid session behaviour, so both the
/// decode and prefill pull pipelines regroup before pulling.
pub(super) fn group_contiguous_runs(
    indexed: Vec<(usize, SequenceHash)>,
) -> Vec<Vec<(usize, SequenceHash)>> {
    debug_assert!(indexed.windows(2).all(|w| w[0].0 < w[1].0));
    let mut runs: Vec<Vec<(usize, SequenceHash)>> = Vec::new();
    for entry in indexed {
        match runs.last_mut() {
            Some(run) if run.last().unwrap().0 + 1 == entry.0 => run.push(entry),
            _ => runs.push(vec![entry]),
        }
    }
    runs
}

/// Pull ONE contiguous run of committed hashes into freshly-allocated G2
/// mutables over the session, stage each by its committed hash (the engine
/// has no token sequence — mirror the `p2p::control` pull-staging path, which
/// stages by `(hash, block_size)`), and register them in the leader's G2
/// cache. Returns the registered pins in run order; the caller owns where
/// they park (held across the G2→G1 copy on the decode path; parked in the
/// prefill request state until its USAA kick).
pub(super) async fn pull_run_into_g2(
    leader: &Arc<InstanceLeader>,
    session: &Arc<dyn Session>,
    hashes: Vec<SequenceHash>,
    block_size: usize,
) -> Result<Vec<ImmutableBlock<G2>>> {
    let run_len = hashes.len();
    let capacity = Arc::clone(leader.g2_capacity());
    let dst = reserve_required_staging(Arc::clone(&capacity), run_len).map_err(|error| {
        anyhow!("cd session pull: failed to reserve {run_len} G2 mutables: {error}")
    })?;
    let pulled = dst
        .transfer_with(|mutables| session.pull(hashes.clone(), mutables))
        .await?;
    if pulled.len() != run_len {
        bail!(
            "cd session pull: session.pull returned {} blocks, expected {}",
            pulled.len(),
            run_len
        );
    }

    let registered = pulled
        .stage_all(&hashes, block_size)
        .map_err(|e| anyhow!("cd session pull: stage pulled block: {e:#}"))?
        .publish()?;
    if registered.len() != run_len {
        bail!(
            "cd session pull: registration returned {} blocks, expected {}",
            registered.len(),
            run_len
        );
    }
    Ok(registered)
}

/// Pull ONE contiguous run of remote slots into G2 (via [`pull_run_into_g2`]),
/// hold the registered pins across the G2→G1 copy, and execute it.
async fn pull_register_onboard_run(
    leader: &Arc<InstanceLeader>,
    session: &Arc<dyn Session>,
    remote_pairs: &[(SequenceHash, BlockId)],
    run: &[(usize, SequenceHash)],
    block_size: usize,
) -> Result<OnboardDrain<Vec<ImmutableBlock<G2>>, Result<()>>> {
    let ordered_hashes: Vec<SequenceHash> = run.iter().map(|(_, hash)| *hash).collect();
    let g1_block_ids: Vec<BlockId> = run.iter().map(|(slot, _)| remote_pairs[*slot].1).collect();

    let registered = pull_run_into_g2(leader, session, ordered_hashes, block_size).await?;
    dispatch_pinned_onboard(leader, registered, g1_block_ids).await
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::Mutex;

    use super::*;
    use crate::g2_capacity::test_support::RecordingG2Capacity;
    use crate::g2_capacity::{
        G2AllocationKind, G2Capacity, G2CapacityDecision, G2CapacityError, G2CapacityRequest,
        G2CapacityRequirement, G2ExactAllocation, G2ExactRegistrationOwner, G2StagedAllocation,
    };
    use crate::p2p::session::{MockSessionFactory, SessionFactory};
    use crate::testing::{create_messenger_tcp, managers::TestManagerBuilder};
    use kvbm_logical::blocks::{BlockRegistry, CompleteBlock};

    struct ExactOnboardCapacity {
        manager: Arc<kvbm_logical::BlockManager<G2>>,
        requests: Mutex<Vec<G2CapacityRequest>>,
    }

    struct ExactOnboardRegistration {
        manager: Arc<kvbm_logical::BlockManager<G2>>,
    }

    impl G2ExactRegistrationOwner for ExactOnboardRegistration {
        fn register_blocks(
            &mut self,
            blocks: Vec<CompleteBlock<G2>>,
        ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
            Ok(self.manager.register_blocks(blocks))
        }

        fn rollback_required_staging(&mut self, blocks: Vec<ImmutableBlock<G2>>) {
            self.manager.release_blocks(blocks, Some(true));
        }

        fn retain_cache_extension(self: Box<Self>) {}
    }

    impl G2Capacity for ExactOnboardCapacity {
        fn reserve(
            &self,
            request: G2CapacityRequest,
        ) -> Result<G2CapacityDecision, G2CapacityError> {
            self.requests.lock().expect("requests lock").push(request);
            if request.requirement() != G2CapacityRequirement::ExactReclaim {
                return Err(G2CapacityError::RequirementMismatch {
                    expected: G2CapacityRequirement::ExactReclaim,
                    actual: request.requirement(),
                });
            }
            let blocks = self
                .manager
                .allocate_blocks_from_reset(request.count())
                .ok_or(G2CapacityError::Unavailable(request))?;
            G2ExactAllocation::new(
                request,
                blocks,
                Box::new(ExactOnboardRegistration {
                    manager: Arc::clone(&self.manager),
                }),
            )
            .map(G2CapacityDecision::ExactGranted)
        }

        fn block_size(&self) -> usize {
            self.manager.block_size()
        }

        fn manager_id(&self) -> kvbm_logical::ManagerId {
            self.manager.id()
        }

        fn register_compatibility(
            &self,
            _allocation: G2StagedAllocation,
        ) -> Result<Vec<ImmutableBlock<G2>>, G2CapacityError> {
            Err(G2CapacityError::RequirementMismatch {
                expected: G2CapacityRequirement::ExactReclaim,
                actual: G2CapacityRequirement::Compatibility,
            })
        }

        fn match_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
            self.manager.match_blocks(hashes)
        }

        fn match_inactive_blocks(&self, hashes: &[SequenceHash]) -> Vec<ImmutableBlock<G2>> {
            self.manager.match_inactive_blocks(hashes)
        }

        fn has_any_registered_hashes(&self, hashes: &[SequenceHash]) -> bool {
            self.manager.has_any_registered_hashes(hashes)
        }

        fn scan_matches(
            &self,
            hashes: &[SequenceHash],
            touch: bool,
        ) -> HashMap<SequenceHash, ImmutableBlock<G2>> {
            self.manager.scan_matches(hashes, touch)
        }
    }

    #[tokio::test]
    async fn conditional_disagg_pull_uses_the_leader_capacity() {
        let registry = BlockRegistry::builder().build();
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(1)
                .block_size(4)
                .registry(registry.clone())
                .build(),
        );
        let capacity = Arc::new(RecordingG2Capacity::new(Arc::clone(&manager)));
        let leader = Arc::new(
            InstanceLeader::builder()
                .messenger(create_messenger_tcp().await.expect("test messenger"))
                .registry(registry)
                .g2_manager(manager)
                .g2_capacity(capacity.clone())
                .build()
                .expect("test leader"),
        );
        let hash = SequenceHash::new(11, None, 0);
        let factory = MockSessionFactory::new();
        factory
            .open(uuid::Uuid::new_v4())
            .expect("test session open");
        let session = factory.last_opened().expect("opened test session");
        session.inject_peer_available(vec![crate::p2p::session::CommittedBlock {
            hash,
            peer_block_id: 0,
        }]);

        let task = tokio::spawn({
            let leader = Arc::clone(&leader);
            let session: Arc<dyn Session> = session.clone();
            async move { pull_run_into_g2(&leader, &session, vec![hash], 4).await }
        });
        session.wait_pull_count(1).await;
        session.resolve_pull(0, Ok(()));

        let registered = task
            .await
            .expect("conditional-disagg task")
            .expect("conditional-disagg pull");

        assert_eq!(registered.len(), 1);
        assert_eq!(
            capacity.allocation_kinds(),
            vec![G2AllocationKind::RequiredStaging]
        );
        assert_eq!(capacity.registration_count(), 1);
        assert_eq!(capacity.registrations_with_live_lease(), 1);
        assert_eq!(capacity.lease_drop_count(), 1);
    }

    #[tokio::test]
    async fn conditional_disagg_pull_accepts_exact_required_staging_capacity() {
        let registry = BlockRegistry::builder().build();
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(1)
                .block_size(4)
                .registry(registry.clone())
                .build(),
        );
        let capacity = Arc::new(ExactOnboardCapacity {
            manager: Arc::clone(&manager),
            requests: Mutex::new(Vec::new()),
        });
        let leader = Arc::new(
            InstanceLeader::builder()
                .messenger(create_messenger_tcp().await.expect("test messenger"))
                .registry(registry)
                .g2_manager(manager)
                .g2_capacity(capacity.clone())
                .build()
                .expect("test leader"),
        );
        let hash = SequenceHash::new(12, None, 0);
        let factory = MockSessionFactory::new();
        factory
            .open(uuid::Uuid::new_v4())
            .expect("test session open");
        let session = factory.last_opened().expect("opened test session");
        session.inject_peer_available(vec![crate::p2p::session::CommittedBlock {
            hash,
            peer_block_id: 0,
        }]);

        let task = tokio::spawn({
            let leader = Arc::clone(&leader);
            let session: Arc<dyn Session> = session.clone();
            async move { pull_run_into_g2(&leader, &session, vec![hash], 4).await }
        });
        session.wait_pull_count(1).await;
        session.resolve_pull(0, Ok(()));

        let registered = task
            .await
            .expect("conditional-disagg task")
            .expect("conditional-disagg exact pull");

        assert_eq!(registered.len(), 1);
        assert_eq!(
            *capacity.requests.lock().expect("requests lock"),
            vec![G2CapacityRequest::exact_reclaim(
                G2AllocationKind::RequiredStaging,
                1,
            )]
        );
    }
}
