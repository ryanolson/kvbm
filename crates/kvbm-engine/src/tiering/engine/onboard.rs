// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The local G1←G2 onboard driver.
//!
//! This is the async tail of [`super::local::LocalConnectorEngine::onboard`]:
//! it runs **off** the forward-pass thread (spawned on the leader's tokio
//! runtime), awaits every shard's staging future, folds the reconciled
//! [`OnboardingState`] into a concrete G2 source list via the matched span,
//! and issues the G2→G1 transfer through the leader's parallel worker. It
//! returns a terminal [`ActionStatus`]; the caller records it and fires the
//! worker sink with no engine lock held (see [`super::driver`]).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use futures::StreamExt;
use futures::future::{BoxFuture, Either, Ready};
use kvbm_common::LogicalLayoutHandle;
use kvbm_logical::blocks::ImmutableBlock;
use kvbm_physical::TransferOptions;
use kvbm_protocols::connector::{ActionFailure, ActionStatus};
use kvbm_protocols::connector::{BlockId, SequenceHash};

use super::reconcile::OnboardingState;
use crate::G2;
use crate::leader::InstanceLeader;
use crate::p2p::session::{AvailabilityDelta, CommitDelta, Session};

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

/// Collect the G2 blocks destined for onboarding from every shard, honoring the
/// `[effective_start .. final_end)` span (first-hole contiguous match).
///
/// Mirrors the legacy `collect_g2_blocks_from_shards`: walk shards in order
/// taking their G2 blocks, drop the leading `effective_start -
/// shards[0].start_block` mask, then truncate to `final_end - effective_start`.
fn collect_g2_blocks(
    state: &mut OnboardingState,
    block_size: usize,
) -> Result<Vec<ImmutableBlock<G2>>> {
    debug_assert!(!state.shards.is_empty());
    debug_assert!(state.all_shards_terminal());

    let (effective_start, final_end) = state.matched_span(block_size);
    debug_assert!(effective_start <= final_end);
    let desired_blocks = final_end - effective_start;
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

/// Drive one onboard to a terminal [`ActionStatus`].
///
/// Runs on the leader's runtime, off the forward-pass thread. Awaits every
/// shard's staging future, collects the matched G2 sources (held resident
/// across the copy via `_g2_hold`), and issues the G2→G1 transfer. Any failure
/// is folded into `Failed(AllBlocks)`; the onboard driver in `engine::onboard`
/// then RESOLVES that to `Failed(Partial { dest ids })` — vLLM invalidates
/// failed loads by block_id, so the concrete dest set must reach the worker
/// (an empty failed set finishes the recv with nothing invalidated). Per-block
/// granularity within a transfer is left for when the transfer layer reports
/// per-block status.
pub(super) async fn run_onboard(
    leader: &Arc<InstanceLeader>,
    onboarding: &mut OnboardingState,
    g1_block_ids: Vec<BlockId>,
    staging_futs: Vec<StagingCompletion>,
    block_size: usize,
) -> ActionStatus {
    // Nothing external to move (e.g. a `Matched { hit_blocks: 0 }` onboard):
    // immediately terminal, no transfer issued.
    if g1_block_ids.is_empty() {
        return ActionStatus::Complete;
    }

    // Wait for every shard's find_session to reach a terminal state.
    for fut in staging_futs {
        if let Err(e) = fut.await {
            tracing::error!(error = %e, "onboard staging future failed");
            return ActionStatus::Failed(ActionFailure::AllBlocks);
        }
    }

    // Pull the matched G2 sources (held alive until the transfer completes).
    let g2_blocks = match collect_g2_blocks(onboarding, block_size) {
        Ok(blocks) => blocks,
        Err(e) => {
            tracing::error!(error = %e, "onboard G2 collect failed");
            return ActionStatus::Failed(ActionFailure::AllBlocks);
        }
    };
    let src_block_ids: Vec<BlockId> = g2_blocks.iter().map(|b| b.block_id()).collect();
    if src_block_ids.len() != g1_block_ids.len() {
        tracing::error!(
            src = src_block_ids.len(),
            dst = g1_block_ids.len(),
            "onboard G2/G1 block count mismatch"
        );
        return ActionStatus::Failed(ActionFailure::AllBlocks);
    }

    // The ImmutableBlock<G2> Drop returns the block to the pool, so the source
    // Vec must outlive the transfer await.
    let _g2_hold = g2_blocks;
    match leader.execute_local_transfer(
        LogicalLayoutHandle::G2,
        LogicalLayoutHandle::G1,
        src_block_ids,
        g1_block_ids,
        TransferOptions::default(),
    ) {
        Ok(notification) => match notification.await {
            Ok(()) => ActionStatus::Complete,
            Err(e) => {
                tracing::error!(error = %e, "onboard G2->G1 transfer failed");
                ActionStatus::Failed(ActionFailure::AllBlocks)
            }
        },
        Err(e) => {
            tracing::error!(error = %e, "onboard G2->G1 transfer dispatch failed");
            ActionStatus::Failed(ActionFailure::AllBlocks)
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
/// blocks have landed in G1; on `Err` the caller reads it to report only the
/// still-unfilled remote dest ids.
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
) -> Result<()> {
    let expected_count = remote_pairs.len();
    if expected_count == 0 {
        return Ok(());
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
        match delta {
            AvailabilityDelta::Available(blocks) => {
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
                    pull_register_onboard_run(leader, session, remote_pairs, &run, block_size)
                        .await?;
                    for (_, hash) in &run {
                        filled.insert(*hash);
                    }
                }
                if filled.len() == expected_count {
                    break;
                }
            }
            AvailabilityDelta::Drained => break,
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
    Ok(())
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
    let dst = leader
        .g2_manager()
        .allocate_blocks(run_len)
        .ok_or_else(|| anyhow!("cd session pull: failed to allocate {run_len} G2 mutables"))?;
    let pulled = session.pull(hashes.clone(), dst).await?;
    if pulled.len() != run_len {
        bail!(
            "cd session pull: session.pull returned {} blocks, expected {}",
            pulled.len(),
            run_len
        );
    }

    let mut completes = Vec::with_capacity(run_len);
    for (mutable, hash) in pulled.into_iter().zip(hashes.iter()) {
        completes.push(
            mutable
                .stage(*hash, block_size)
                .map_err(|e| anyhow!("cd session pull: stage pulled block: {e:#}"))?,
        );
    }
    let registered = leader.g2_manager().register_blocks(completes);
    if registered.len() != run_len {
        bail!(
            "cd session pull: register_blocks returned {} blocks, expected {}",
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
) -> Result<()> {
    let ordered_hashes: Vec<SequenceHash> = run.iter().map(|(_, hash)| *hash).collect();
    let g1_block_ids: Vec<BlockId> = run.iter().map(|(slot, _)| remote_pairs[*slot].1).collect();

    let registered = pull_run_into_g2(leader, session, ordered_hashes, block_size).await?;
    let g2_block_ids: Vec<BlockId> = registered.iter().map(|b| b.block_id()).collect();

    // Hold the registered pins for the lifetime of the G2→G1 copy (the
    // ImmutableBlock Drop would return them to the pool otherwise).
    let _hold = registered;
    leader
        .execute_local_transfer(
            LogicalLayoutHandle::G2,
            LogicalLayoutHandle::G1,
            g2_block_ids,
            g1_block_ids,
            TransferOptions::default(),
        )?
        .await?;
    Ok(())
}
