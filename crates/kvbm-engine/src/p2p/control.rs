// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../../docs/control-transfer.md")]
//!
//! ## Module implementation
//!
//! The handlers in this file are thin shims: deserialize the request,
//! call an `InstanceLeader` method, wrap the result in a [`ControlReply`],
//! and return. The substantive logic lives below as free functions
//! invoked by both the `InstanceLeader` methods and the `search_prefix`
//! / `search_scatter` shims for the hub's HTTP query routes. Putting the
//! work on `InstanceLeader` keeps the public surface discoverable for
//! in-process callers and avoids forcing every caller through the velo
//! wire.

use std::sync::Arc;

use anyhow::Result;
use kvbm_common::LogicalResourceId;
use velo::{Handler, Messenger};

use kvbm_logical::BlockManager;
use kvbm_logical::blocks::ImmutableBlock;

use crate::G3;
use crate::leader::BlockHolder;
use crate::leader::stage_g3_to_g2;
use kvbm_protocols::control::modules::transfer::{
    CLOSE_SESSION_HANDLER, CloseTransferSessionRequest, CloseTransferSessionResponse, FindMode,
    MatchBreakdown, OPEN_SESSION_HANDLER, OpenTransferSessionRequest, OpenTransferSessionResponse,
    PULL_FROM_SESSION_HANDLER, PullFromSessionRequest, PullFromSessionResponse,
    SEARCH_PREFIX_HANDLER, SEARCH_SCATTER_HANDLER, SearchMode, SearchRequest, SearchResponse,
    TierSelection, TransferSessionCapability,
};
use kvbm_protocols::control::{ControlError, ControlReply, ModuleId};

use crate::leader::InstanceLeader;
use crate::leader::control::ControlModule;
use crate::p2p::PayloadBlock;
use crate::p2p::pull_transaction::resolve_g2_manager;
use crate::p2p::session::{Session, VerifiedPayload};
use crate::{G2, SequenceHash};

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// The `transfer` control module — always enabled.
///
/// Carries only an `Arc<InstanceLeader>`; everything the handlers need
/// (g2_manager, g3_manager, session_factory_cell, session_manager,
/// runtime, messenger.instance_id) is reached through it.
pub struct TransferModule {
    leader: Arc<InstanceLeader>,
}

impl TransferModule {
    pub fn new(leader: Arc<InstanceLeader>) -> Self {
        Self { leader }
    }
}

impl ControlModule for TransferModule {
    fn id(&self) -> ModuleId {
        ModuleId::Transfer
    }

    fn register(&self, messenger: &Arc<Messenger>) -> Result<()> {
        register_open_session(messenger, &self.leader)?;
        register_pull_from_session(messenger, &self.leader)?;
        register_close_session(messenger, &self.leader)?;
        // Handlers for the hub HTTP query routes, retained as shims over open_session.
        register_search_prefix(messenger, &self.leader)?;
        register_search_scatter(messenger, &self.leader)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Handler registration
// ---------------------------------------------------------------------------

fn register_open_session(messenger: &Arc<Messenger>, leader: &Arc<InstanceLeader>) -> Result<()> {
    let leader = Arc::clone(leader);
    let handler = Handler::typed_unary_async(OPEN_SESSION_HANDLER, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let req: OpenTransferSessionRequest = ctx.input;
            let reply: ControlReply<OpenTransferSessionResponse> =
                leader.open_transfer_session(req).await.into();
            Ok::<ControlReply<OpenTransferSessionResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({OPEN_SESSION_HANDLER}): {e}"))?;
    Ok(())
}

fn register_close_session(messenger: &Arc<Messenger>, leader: &Arc<InstanceLeader>) -> Result<()> {
    let leader = Arc::clone(leader);
    let handler = Handler::typed_unary_async(CLOSE_SESSION_HANDLER, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let req: CloseTransferSessionRequest = ctx.input;
            let reply: ControlReply<CloseTransferSessionResponse> =
                leader.close_transfer_session(req).await.into();
            Ok::<ControlReply<CloseTransferSessionResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({CLOSE_SESSION_HANDLER}): {e}"))?;
    Ok(())
}

fn register_pull_from_session(
    messenger: &Arc<Messenger>,
    leader: &Arc<InstanceLeader>,
) -> Result<()> {
    let leader = Arc::clone(leader);
    let handler = Handler::typed_unary_async(PULL_FROM_SESSION_HANDLER, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let req: PullFromSessionRequest = ctx.input;
            let reply: ControlReply<PullFromSessionResponse> =
                leader.pull_from_session(req).await.into();
            Ok::<ControlReply<PullFromSessionResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({PULL_FROM_SESSION_HANDLER}): {e}"))?;
    Ok(())
}

fn register_search_prefix(messenger: &Arc<Messenger>, leader: &Arc<InstanceLeader>) -> Result<()> {
    register_search_shim(messenger, leader, SEARCH_PREFIX_HANDLER, SearchMode::Prefix)
}

fn register_search_scatter(messenger: &Arc<Messenger>, leader: &Arc<InstanceLeader>) -> Result<()> {
    register_search_shim(
        messenger,
        leader,
        SEARCH_SCATTER_HANDLER,
        SearchMode::Scatter,
    )
}

/// Adapter: `SearchRequest`/`SearchResponse` for the hub's HTTP query
/// routes, over the `open_session` path with `find_mode = Sync` and
/// `tiers = default`.
fn register_search_shim(
    messenger: &Arc<Messenger>,
    leader: &Arc<InstanceLeader>,
    handler_name: &'static str,
    mode: SearchMode,
) -> Result<()> {
    let leader = Arc::clone(leader);
    let handler = Handler::typed_unary_async(handler_name, move |ctx| {
        let leader = Arc::clone(&leader);
        async move {
            let req: SearchRequest = ctx.input;
            let open_req = OpenTransferSessionRequest {
                sequence_hashes: req.sequence_hashes,
                search_mode: mode,
                find_mode: FindMode::Sync,
                // Query-only route: the caller reads the matched set and
                // never pulls. Selecting a tier that must copy before it
                // can serve spends a DMA on nothing.
                tiers: TierSelection::default(),
                resource: None,
                watchdog_ms: None,
                registration_epoch: None,
                require_payload_integrity: false,
            };
            let reply: ControlReply<SearchResponse> = leader
                .open_transfer_session(open_req)
                .await
                .map(|resp| match resp {
                    OpenTransferSessionResponse::NoBlocksFound => SearchResponse::NoBlocksFound,
                    OpenTransferSessionResponse::Sync { capability, .. } => {
                        SearchResponse::Session {
                            session_id: capability.session_id,
                        }
                    }
                    // Sync request must produce Sync or NoBlocksFound; treat
                    // anything else as internal error.
                    OpenTransferSessionResponse::Async { .. } => SearchResponse::Session {
                        session_id: uuid::Uuid::nil(),
                    },
                })
                .into();
            Ok::<ControlReply<SearchResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({handler_name}): {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// open_transfer_session — substantive logic
// ---------------------------------------------------------------------------

/// Result of the populator's `find_phase`, with pinned source blocks.
/// Sync responses contain the selected hashes and the tier breakdown.
/// The background `stage_phase` consumes the blocks.
struct FindOutcome {
    /// Selected hashes in request order. Each tier vector below holds a
    /// disjoint subset of these hashes, also in request order.
    committed: Vec<SequenceHash>,
    g2_blocks: Vec<ImmutableBlock<G2>>,
    g3_blocks: Vec<ImmutableBlock<G3>>,
    g1_blocks: Option<Box<dyn super::g1_source::PinnedG1Source>>,
    breakdown: MatchBreakdown,
}

/// Blocks one tier walk selected, before the G1 pins join them.
struct TierBlocks {
    committed: Vec<SequenceHash>,
    g2_blocks: Vec<ImmutableBlock<G2>>,
    g3_blocks: Vec<ImmutableBlock<G3>>,
}

/// Search the holder's tiers for `hashes` and pin what it can serve.
///
/// Lookup order is G2, then G1 (`tiers.g1`), then G3 (`tiers.g3`,
/// `Scatter` only), and a hash goes to the first tier that holds it.
/// G2 needs no copy at all. A G1 hit costs one local DMA, a G3 hit
/// costs a disk read *and* the same DMA, so G1 wins wherever both hold a
/// hash.
///
/// `Prefix` walks the request left to right and stops at the first hash
/// that no selected tier holds. G2 and G1 extend one shared cursor in
/// turn, so a run G2 serves and a run G1 serves join into one contiguous
/// prefix. The walk ends when neither tier advances the cursor. G3 stays
/// out of the prefix walk: a disk gap needs handling that does not pay
/// for itself yet.
///
/// Synchronous body — every tier read is an in-memory lookup. `async fn`
/// for forward-compat with G4 (object-store) scans in v1.1.
async fn find_phase(
    leader: &Arc<InstanceLeader>,
    g2_manager: &Arc<BlockManager<G2>>,
    resource: LogicalResourceId,
    hashes: &[SequenceHash],
    search_mode: SearchMode,
    tiers: TierSelection,
) -> Result<FindOutcome, ControlError> {
    let mut pins = if tiers.g1 {
        leader.g1_source(resource).and_then(|source| source.pins())
    } else {
        None
    };
    let TierBlocks {
        committed,
        g2_blocks,
        g3_blocks,
    } = match search_mode {
        SearchMode::Prefix => find_prefix(g2_manager, hashes, &mut pins),
        SearchMode::Scatter => {
            find_scatter(leader, g2_manager, resource, hashes, tiers, &mut pins)?
        }
    };
    // Only served blocks are pinned, so a set that stayed empty carries
    // no copy and no capacity reservation.
    let g1_blocks = pins.filter(|pins| pins.len() > 0);
    let breakdown = MatchBreakdown {
        device_blocks: g1_blocks.as_ref().map_or(0, |pins| pins.len()),
        host_blocks: g2_blocks.len(),
        disk_blocks: g3_blocks.len(),
        object_blocks: 0,
    };
    Ok(FindOutcome {
        committed,
        g2_blocks,
        g3_blocks,
        g1_blocks,
        breakdown,
    })
}

/// Contiguous prefix across G2 and G1, from one cursor.
///
/// Each tier resolves its run under a single store lock, so a request of
/// N hashes costs one lock per run rather than one per hash. G2 keeps its
/// LRU touch: a prefix it serves is a prefix the puller reads.
fn find_prefix(
    g2_manager: &Arc<BlockManager<G2>>,
    hashes: &[SequenceHash],
    pins: &mut Option<Box<dyn super::g1_source::PinnedG1Source>>,
) -> TierBlocks {
    let mut g2_blocks: Vec<ImmutableBlock<G2>> = Vec::new();
    let mut cursor = 0;
    // touch = false: the holder answers a remote peer here, the same as
    // find_scatter and the G1 source. A block another node wants must not
    // outrank a block this node still reads. The `touch` flag gates only
    // the frequency-sketch write. The store-side inactive-pool
    // resurrection happens regardless, because every backend's lookup
    // ignores its `_touch` parameter.
    let run = g2_manager.match_prefix(&hashes[cursor..], false);
    cursor += run.len();
    g2_blocks.extend(run);
    loop {
        // G2 stopped at this cursor, so a G1 run of zero ends the walk
        // here, and G1 stopped at the cursor the next G2 call receives,
        // so a G2 run of zero ends it too. Asking a tier again at a
        // cursor where it already declared a miss takes a store lock
        // only to re-learn that same miss.
        let g1_run = pins
            .as_mut()
            .map_or(0, |pins| pins.pin_prefix(&hashes[cursor..]));
        if g1_run == 0 {
            break;
        }
        cursor += g1_run;
        let run = g2_manager.match_prefix(&hashes[cursor..], false);
        if run.is_empty() {
            break;
        }
        cursor += run.len();
        g2_blocks.extend(run);
    }
    TierBlocks {
        committed: hashes[..cursor].to_vec(),
        g2_blocks,
        g3_blocks: Vec::new(),
    }
}

/// Every hash any selected tier holds, gaps included.
fn find_scatter(
    leader: &Arc<InstanceLeader>,
    g2_manager: &Arc<BlockManager<G2>>,
    resource: LogicalResourceId,
    hashes: &[SequenceHash],
    tiers: TierSelection,
    pins: &mut Option<Box<dyn super::g1_source::PinnedG1Source>>,
) -> Result<TierBlocks, ControlError> {
    if tiers.g3 && resource != leader.primary_g2_resource() {
        return Err(ControlError::Internal(format!(
            "resource_g3_unsupported: logical resource {resource:?} requested G3, but G3 is only configured for primary resource {:?}",
            leader.primary_g2_resource()
        )));
    }

    // touch = false: an RPC search must not perturb the local G2 LRU.
    // The `touch` flag gates only the frequency-sketch write. The
    // store-side inactive-pool resurrection happens regardless, because
    // every backend's lookup ignores its `_touch` parameter.
    let mut g2_map = g2_manager.scan_matches(hashes, /* touch */ false);
    let missing: Vec<SequenceHash> = hashes
        .iter()
        .copied()
        .filter(|hash| !g2_map.contains_key(hash))
        .collect();
    if let Some(pins) = pins.as_mut() {
        pins.pin(&missing);
    }
    let mut g1_hashes: std::collections::HashSet<SequenceHash> = pins
        .as_ref()
        .map(|pins| pins.hashes().into_iter().collect())
        .unwrap_or_default();

    let mut g3_map = std::collections::HashMap::new();
    if tiers.g3
        && let Some(g3_manager) = leader.g3_manager()
    {
        // A device copy beats a disk read, so the disk tier only sees the
        // hashes G1 does not hold.
        let remaining: Vec<SequenceHash> = missing
            .iter()
            .copied()
            .filter(|hash| !g1_hashes.contains(hash))
            .collect();
        if !remaining.is_empty() {
            g3_map = g3_manager.scan_matches(&remaining, false);
        }
    }

    let mut committed = Vec::new();
    let mut g2_blocks = Vec::new();
    let mut g3_blocks = Vec::new();
    for hash in hashes {
        let served = if let Some(block) = g2_map.remove(hash) {
            g2_blocks.push(block);
            true
        } else if g1_hashes.remove(hash) {
            true
        } else if let Some(block) = g3_map.remove(hash) {
            g3_blocks.push(block);
            true
        } else {
            false
        };
        if served {
            committed.push(*hash);
        }
    }
    Ok(TierBlocks {
        committed,
        g2_blocks,
        g3_blocks,
    })
}

/// Drive the session's commit, availability, and terminator calls.
/// Each tier publishes its own availability batch as it lands: the
/// resident G2 hits, then the G1 sources that staged into temporary G2,
/// then the G3 sources that staged through the local path.
/// Each checksum uses its position in the full committed set.
///
/// Errors propagate as `ControlError::Internal`. On error the caller
/// must call `session.close(reason)`. That call pushes
/// `LifecycleEvent::Detached { reason: Some(reason) }` on the lifecycle
/// stream of the holder. The `SessionManager` watcher of the holder
/// consumes it and evicts the entry. The same call also enqueues
/// `Frame::CommitsClosed` and `Frame::Drained` on the wire. The puller
/// observes these as `CommitDelta::Closed` and `AvailabilityDelta::Drained`.
/// The puller sees `LifecycleEvent::Detached { reason: None }` only when
/// the Finalized sentinel lands. Finalization waits for the last inbound
/// `PullAck` while pulls are in flight. The reason string never reaches
/// the puller. `LifecycleEvent::Failed` does not come from this path.
/// It comes from an inbound `Frame::Error`, an attach failure, or a
/// velo stream error other than `SenderDropped`.
async fn stage_phase(
    leader: Arc<InstanceLeader>,
    session: Arc<dyn Session>,
    resource: LogicalResourceId,
    require_payload_integrity: bool,
    find: FindOutcome,
) -> Result<(), ControlError> {
    let FindOutcome {
        committed,
        g2_blocks,
        g3_blocks,
        g1_blocks,
        ..
    } = find;
    let mut ordinals = std::collections::HashMap::with_capacity(committed.len());
    for (index, hash) in committed.iter().enumerate() {
        let ordinal = u32::try_from(index)
            .map_err(|_| ControlError::Internal("payload ordinal exceeds u32".to_owned()))?;
        ordinals.insert(*hash, ordinal);
    }
    // Commit every selected hash up front, whichever tier holds it. Then
    // seal the commit stream before any tier publishes. find_phase already
    // computes the full committed set, so no later commit exists to wait
    // for. An attached puller passes `drain_committed` here, before the
    // G1 and G3 copies run.
    if !committed.is_empty() {
        session
            .commit(committed)
            .map_err(|error| ControlError::Internal(format!("commit blocks: {error:#}")))?;
    }
    session
        .finish_commits()
        .map_err(|e| ControlError::Internal(format!("finish_commits: {e:#}")))?;
    // Publish each tier as it lands. The resident G2 hits need no copy, so
    // holding them back charges every hit the latency of the slowest
    // tier the search touched. Each batch already arrives in request order.
    if !g2_blocks.is_empty() {
        publish_available(
            &leader,
            &session,
            resource,
            g2_blocks,
            &ordinals,
            require_payload_integrity,
        )
        .await
        .map_err(|e| ControlError::Internal(format!("make_available g2: {e:#}")))?;
    }

    if let Some(source) = g1_blocks {
        let staged = source
            .stage()
            .await
            .map_err(|error| ControlError::Internal(format!("stage G1 blocks: {error:#}")))?;
        if !staged.is_empty() {
            publish_available(
                &leader,
                &session,
                resource,
                staged,
                &ordinals,
                require_payload_integrity,
            )
            .await
            .map_err(|e| ControlError::Internal(format!("make_available staged g1: {e:#}")))?;
        }
    }

    if !g3_blocks.is_empty() {
        let parallel_worker = leader.parallel_worker().ok_or_else(|| {
            ControlError::Internal(
                "G3 staging requires a parallel_worker; leader was built without workers".into(),
            )
        })?;

        let holder = BlockHolder::<G3>::new(g3_blocks);
        let g2_capacity = leader.g2_capacity_for(resource).ok_or_else(|| {
            ControlError::Internal(format!(
                "logical_resource_not_found: no G2 capacity for resource {resource:?}"
            ))
        })?;
        let staged = stage_g3_to_g2(&holder, g2_capacity, &*parallel_worker)
            .await
            .map_err(|e| ControlError::Internal(format!("stage_g3_to_g2: {e:#}")))?;

        publish_available(
            &leader,
            &session,
            resource,
            staged.new_g2_blocks,
            &ordinals,
            require_payload_integrity,
        )
        .await
        .map_err(|e| ControlError::Internal(format!("make_available staged: {e:#}")))?;
    }

    session
        .finish_availability()
        .map_err(|e| ControlError::Internal(format!("finish_availability: {e:#}")))?;
    Ok(())
}

async fn publish_available(
    leader: &Arc<InstanceLeader>,
    session: &Arc<dyn Session>,
    resource: LogicalResourceId,
    blocks: Vec<ImmutableBlock<G2>>,
    ordinals: &std::collections::HashMap<SequenceHash, u32>,
    require_payload_integrity: bool,
) -> Result<()> {
    if !require_payload_integrity {
        return session.make_available(blocks);
    }
    let payload_blocks = blocks
        .iter()
        .map(|block| {
            let ordinal = *ordinals
                .get(&block.sequence_hash())
                .ok_or_else(|| anyhow::anyhow!("available block has no committed ordinal"))?;
            Ok(PayloadBlock {
                hash: block.sequence_hash(),
                block_id: block.block_id(),
                ordinal,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let checksums = leader
        .payload_checksums(resource, &payload_blocks)
        .await
        .map_err(|error| anyhow::anyhow!("payload_checksum_unavailable: {error:#}"))?;
    let payloads = payload_blocks
        .into_iter()
        .zip(checksums)
        .map(|(block, checksum)| VerifiedPayload {
            ordinal: block.ordinal,
            checksum,
        })
        .collect();
    session
        .make_available_verified(blocks, payloads)
        .map_err(|error| anyhow::anyhow!("payload_checksum_publish_failed: {error:#}"))
}

/// Engine-side implementation behind [`InstanceLeader::open_transfer_session`].
///
/// G1 sources stage through G2. G3 remains an optional scatter tier.
pub(crate) async fn open_transfer_session(
    leader: &Arc<InstanceLeader>,
    req: OpenTransferSessionRequest,
) -> Result<OpenTransferSessionResponse, ControlError> {
    // Complete-bundle callers validate manifest/lineages before RPC. This
    // holder-side check prevents that validated hit from crossing into a
    // replacement lifecycle before any raw resource search or session open.
    match req.registration_epoch {
        Some(epoch) if leader.current_registration_epoch() != Some(epoch) => {
            return Err(ControlError::RegistrationEpochMismatch);
        }
        None if req.require_payload_integrity => {
            return Err(ControlError::RegistrationEpochMismatch);
        }
        _ => {}
    }
    let (resource, g2_manager) = resolve_g2_manager(leader, req.resource)?;
    let find = find_phase(
        leader,
        &g2_manager,
        resource,
        &req.sequence_hashes,
        req.search_mode,
        req.tiers,
    )
    .await?;
    let committed = find.committed.clone();
    let breakdown = find.breakdown;

    // Pre-flight: if find_phase produced G3 matches we cannot stage, fail
    // *before* opening a session. `stage_g3_to_g2` requires a
    // `parallel_worker`; without it the background populator would close
    // the session moments after `open_transfer_session` returned, leaving
    // the caller with a capability that points to a teardown-in-progress
    // session — a "usable-looking session that cannot serve blocks". The
    // honest answer is to reject the open and tell the operator what to
    // fix.
    if !find.g3_blocks.is_empty() && leader.parallel_worker().is_none() {
        return Err(ControlError::Internal(
            "g3_requires_parallel_worker: leader has no parallel_worker but G3 matches \
             were found; configure a parallel_worker on this leader or set \
             tiers.g3=false in the request"
                .into(),
        ));
    }

    // Sync mode + zero matches across all selected tiers: short-circuit,
    // do not open a session.
    if matches!(req.find_mode, FindMode::Sync) && committed.is_empty() {
        crate::engine_audit!(
            "transfer_session_no_matches",
            requested = req.sequence_hashes.len(),
            search_mode = ?req.search_mode
        );
        return Ok(OpenTransferSessionResponse::NoBlocksFound);
    }

    let factory = leader
        .session_factory_cell()
        .get()
        .ok_or(ControlError::NotInitialized)?
        .clone();

    let session_id = uuid::Uuid::new_v4();
    let session = factory
        .open(session_id)
        .map_err(|e| ControlError::Internal(format!("open session: {e:#}")))?;
    let endpoint = session
        .endpoint()
        .ok_or_else(|| ControlError::Internal("opened session has no endpoint".into()))?;
    let instance_id = leader.messenger().instance_id();

    let capability = TransferSessionCapability {
        session_id,
        instance_id,
        endpoint,
        resource,
    };

    // Park before launching the populator. The requester may shorten the
    // holder-side watchdog; the manager caps it at its configured maximum.
    if let Some(watchdog_ms) = req.watchdog_ms {
        leader.session_manager().register_with_watchdog(
            Arc::clone(&session),
            std::time::Duration::from_millis(watchdog_ms),
        );
    } else {
        leader.session_manager().register(Arc::clone(&session));
    }

    crate::engine_audit!(
        "transfer_session_opened",
        %session_id,
        find_mode = ?req.find_mode,
        search_mode = ?req.search_mode,
        committed = committed.len(),
        g1_hits = breakdown.device_blocks,
        g2_hits = breakdown.host_blocks,
        g3_hits = breakdown.disk_blocks
    );

    // Always spawn stage_phase in the background. Sync mode awaits the
    // (synchronous-body) find_phase before returning so its response
    // can include `committed` + `breakdown`; stage_phase runs after
    // regardless.
    let runtime = leader.runtime();
    let leader_for_task = Arc::clone(leader);
    let session_for_task = Arc::clone(&session);
    let require_payload_integrity = req.require_payload_integrity;
    let find_mode = req.find_mode;
    runtime.spawn(async move {
        match stage_phase(
            leader_for_task,
            Arc::clone(&session_for_task),
            resource,
            require_payload_integrity,
            find,
        )
        .await
        {
            Ok(()) => {
                crate::engine_audit!(
                    "transfer_populator_complete",
                    %session_id
                );
            }
            Err(err) => {
                tracing::error!(error = %err, %session_id, "transfer populator failed");
                crate::engine_audit!(
                    "transfer_populator_failed",
                    %session_id,
                    error = %err
                );
                session_for_task.close(Some(format!("populator: {err}")));
            }
        }
    });

    match find_mode {
        FindMode::Sync => Ok(OpenTransferSessionResponse::Sync {
            capability,
            committed,
            breakdown,
        }),
        FindMode::Async => Ok(OpenTransferSessionResponse::Async { capability }),
    }
}

// ---------------------------------------------------------------------------
// close_transfer_session — substantive logic
// ---------------------------------------------------------------------------

/// Engine-side implementation behind [`InstanceLeader::close_transfer_session`].
///
/// Idempotent: a missing session returns `Ok(was_present: false)`.
pub(crate) async fn close_transfer_session(
    leader: &Arc<InstanceLeader>,
    req: CloseTransferSessionRequest,
) -> Result<CloseTransferSessionResponse, ControlError> {
    let removed = leader.session_manager().remove(&req.session_id);
    let was_present = removed.is_some();
    let session_id = req.session_id;
    let reason_for_audit = req.reason.clone();
    if let Some(session) = removed {
        session.close(req.reason);
    }
    crate::engine_audit!(
        "transfer_session_closed",
        %session_id,
        was_present,
        reason = ?reason_for_audit
    );
    Ok(CloseTransferSessionResponse { was_present })
}

// ---------------------------------------------------------------------------
// pull_from_session — substantive logic
// ---------------------------------------------------------------------------

/// Engine-side implementation behind [`InstanceLeader::pull_from_session`].
///
/// The one-resource endpoint publishes immediately after the shared
/// staging transaction succeeds. Complete-bundle callers use
/// [`crate::p2p::stage_from_session`] directly and defer publication until all
/// resources have staged.
pub(crate) async fn pull_from_session(
    leader: &Arc<InstanceLeader>,
    req: PullFromSessionRequest,
) -> Result<PullFromSessionResponse, ControlError> {
    let staged = crate::p2p::stage_from_session(leader, req).await?;
    let response = staged.response();
    let _published = staged
        .publish()
        .map_err(|error| ControlError::Internal(format!("pull: publish G2 blocks: {error}")))?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvbm_logical::manager::FrequencyTrackingCapacity;

    /// `InstanceLeader` around `g2_manager`, with no G1 source, no G3
    /// manager, and no workers. `find_phase` never reaches any of those
    /// when `tiers` selects G2 alone, so the search under test needs none.
    async fn leader_for_search(g2_manager: Arc<BlockManager<G2>>) -> Arc<InstanceLeader> {
        let messenger = crate::testing::messenger::create_messenger_tcp()
            .await
            .expect("test messenger");
        Arc::new(
            InstanceLeader::builder()
                .messenger(messenger)
                .registry(kvbm_logical::blocks::BlockRegistry::new())
                .g2_manager(g2_manager)
                .workers(vec![])
                .build()
                .expect("test leader"),
        )
    }

    /// Three-block prefix in a frequency-tracked G2 manager, with the
    /// registry count of every hash after population.
    fn tracked_prefix() -> (Arc<BlockManager<G2>>, Vec<SequenceHash>, Vec<u32>) {
        let g2_manager = Arc::new(
            crate::testing::managers::TestManagerBuilder::<G2>::new()
                .block_count(8)
                .block_size(4)
                .frequency_tracking(FrequencyTrackingCapacity::Small)
                .build(),
        );
        let token_sequence = crate::testing::token_blocks::create_token_sequence(3, 4, 0);
        let hashes = crate::testing::managers::populate_manager_with_blocks(
            &g2_manager,
            token_sequence.blocks(),
        )
        .expect("populate three-block prefix");
        assert_eq!(hashes.len(), 3);
        let registry = g2_manager.block_registry();
        let before = hashes.iter().map(|hash| registry.count(*hash)).collect();
        (g2_manager, hashes, before)
    }

    /// Registry counts before and after one `find_phase` in `mode` over
    /// the whole prefix, with G2 as the only selected tier.
    async fn counts_around_search(mode: SearchMode) -> (Vec<u32>, Vec<u32>) {
        let (g2_manager, hashes, before) = tracked_prefix();
        let leader = leader_for_search(g2_manager.clone()).await;
        let found = find_phase(
            &leader,
            &g2_manager,
            LogicalResourceId::default(),
            &hashes,
            mode,
            TierSelection::default(),
        )
        .await
        .expect("search");
        assert_eq!(found.committed, hashes);
        let registry = g2_manager.block_registry();
        let after = hashes.iter().map(|hash| registry.count(*hash)).collect();
        (before, after)
    }

    /// A remote search must not touch the frequency tracker: a block
    /// another node wants must not outrank a block this node still reads.
    /// `find_prefix` must pass `touch = false`, the same as `find_scatter`
    /// and the G1 source.
    #[tokio::test]
    async fn remote_prefix_search_does_not_touch_frequency() {
        let (before, after) = counts_around_search(SearchMode::Prefix).await;
        assert_eq!(
            before, after,
            "SearchMode::Prefix must not touch the frequency tracker"
        );
    }

    /// Control: `find_scatter` passes `touch = false`, so the same prefix
    /// through `SearchMode::Scatter` leaves every count unchanged.
    #[tokio::test]
    async fn remote_scatter_search_does_not_touch_frequency() {
        let (before, after) = counts_around_search(SearchMode::Scatter).await;
        assert_eq!(
            before, after,
            "SearchMode::Scatter must not touch the frequency tracker"
        );
    }
}
