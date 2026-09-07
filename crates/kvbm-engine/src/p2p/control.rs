// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../../docs/control-transfer.md")]
//!
//! ## Module implementation
//!
//! The handlers in this file are thin shims: deserialize the request,
//! call an `InstanceLeader` method, wrap the result in a [`ControlReply`],
//! and return. The substantive logic lives below as free functions
//! invoked by both the `InstanceLeader` methods and the legacy
//! `search_prefix` / `search_scatter` back-compat shims. Putting the
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
        // Legacy back-compat handlers, retained as shims over open_session.
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

/// Adapter: legacy `SearchRequest`/`SearchResponse` over the new
/// `open_session` path with `find_mode = Sync` and `tiers = default`.
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
    g2_committed: Vec<SequenceHash>,
    g2_blocks: Vec<ImmutableBlock<G2>>,
    g3_committed: Vec<SequenceHash>,
    g3_blocks: Vec<ImmutableBlock<G3>>,
    g1_blocks: Option<Box<dyn super::g1_source::PinnedG1Source>>,
    breakdown: MatchBreakdown,
}

impl FindOutcome {
    fn committed(&self, requested: &[SequenceHash]) -> Vec<SequenceHash> {
        let mut found = self
            .g2_committed
            .iter()
            .chain(&self.g3_committed)
            .copied()
            .collect::<std::collections::HashSet<_>>();
        if let Some(source) = &self.g1_blocks {
            found.extend(source.hashes());
        }
        requested
            .iter()
            .copied()
            .filter(|hash| found.remove(hash))
            .collect()
    }
}

async fn find_phase(
    leader: &Arc<InstanceLeader>,
    g2_manager: &Arc<BlockManager<G2>>,
    resource: LogicalResourceId,
    hashes: &[SequenceHash],
    search_mode: SearchMode,
    tiers: TierSelection,
) -> Result<FindOutcome, ControlError> {
    let Some(source) = leader.g1_source(resource) else {
        return find_lower_tiers(leader, g2_manager, resource, hashes, search_mode, tiers).await;
    };
    let lower_tiers = if search_mode == SearchMode::Prefix {
        TierSelection::default()
    } else {
        tiers
    };
    let mut found = find_lower_tiers(
        leader,
        g2_manager,
        resource,
        hashes,
        SearchMode::Scatter,
        lower_tiers,
    )
    .await?;
    let mut selected = found
        .g2_committed
        .iter()
        .chain(&found.g3_committed)
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let missing = hashes
        .iter()
        .copied()
        .filter(|hash| !selected.contains(hash))
        .collect::<Vec<_>>();
    found.g1_blocks = source.pin(&missing);
    if let Some(pins) = &found.g1_blocks {
        selected.extend(pins.hashes());
    }
    if search_mode == SearchMode::Prefix {
        selected = hashes
            .iter()
            .copied()
            .take_while(|hash| selected.contains(hash))
            .collect();
    }
    found.g2_committed.retain(|hash| selected.contains(hash));
    found.g3_committed.retain(|hash| selected.contains(hash));
    found
        .g2_blocks
        .retain(|block| selected.contains(&block.sequence_hash()));
    found
        .g3_blocks
        .retain(|block| selected.contains(&block.sequence_hash()));
    if let Some(pins) = &mut found.g1_blocks {
        pins.retain(&selected);
        found.breakdown.device_blocks = pins.hashes().len();
    }
    found.breakdown.host_blocks = found.g2_blocks.len();
    found.breakdown.disk_blocks = found.g3_blocks.len();
    Ok(found)
}

/// Scan local tiers per `search_mode` / `tiers`. Synchronous body — both
/// G2 and G3 scans are in-memory hashmap lookups. `async fn` for
/// forward-compat with G4 (object-store) scans in v1.1.
///
/// In v1, **G3 is only consulted in `SearchMode::Scatter`**. The Prefix
/// mode preserves the existing semantic (contiguous G2 prefix); extending
/// the prefix walk into G3 requires careful gap handling that doesn't
/// pay for itself yet.
async fn find_lower_tiers(
    leader: &Arc<InstanceLeader>,
    g2_manager: &Arc<BlockManager<G2>>,
    resource: LogicalResourceId,
    hashes: &[SequenceHash],
    search_mode: SearchMode,
    tiers: TierSelection,
) -> Result<FindOutcome, ControlError> {
    if search_mode == SearchMode::Scatter && tiers.g3 && resource != leader.primary_g2_resource() {
        return Err(ControlError::Internal(format!(
            "resource_g3_unsupported: logical resource {resource:?} requested G3, but G3 is only configured for primary resource {:?}",
            leader.primary_g2_resource()
        )));
    }

    match search_mode {
        SearchMode::Prefix => {
            // Contiguous prefix in G2 only. G3/G4 not searched.
            let g2_blocks = g2_manager.match_blocks(hashes);
            let g2_committed: Vec<SequenceHash> =
                g2_blocks.iter().map(|b| b.sequence_hash()).collect();
            let breakdown = MatchBreakdown {
                device_blocks: 0,
                host_blocks: g2_blocks.len(),
                disk_blocks: 0,
                object_blocks: 0,
            };
            Ok(FindOutcome {
                g2_committed,
                g2_blocks,
                g3_committed: Vec::new(),
                g3_blocks: Vec::new(),
                g1_blocks: None,
                breakdown,
            })
        }
        SearchMode::Scatter => {
            let g2_map = g2_manager.scan_matches(hashes, /* touch */ false);
            let g2_committed: Vec<SequenceHash> = g2_map.keys().copied().collect();
            let g2_blocks: Vec<ImmutableBlock<G2>> = g2_map.into_values().collect();

            let mut g3_committed: Vec<SequenceHash> = Vec::new();
            let mut g3_blocks: Vec<ImmutableBlock<G3>> = Vec::new();

            if tiers.g3
                && let Some(g3_manager) = leader.g3_manager()
            {
                let g2_set: std::collections::HashSet<SequenceHash> =
                    g2_committed.iter().copied().collect();
                let remaining: Vec<SequenceHash> = hashes
                    .iter()
                    .filter(|h| !g2_set.contains(h))
                    .copied()
                    .collect();
                if !remaining.is_empty() {
                    let g3_map = g3_manager.scan_matches(&remaining, false);
                    for (h, b) in g3_map {
                        g3_committed.push(h);
                        g3_blocks.push(b);
                    }
                }
            }

            let breakdown = MatchBreakdown {
                device_blocks: 0,
                host_blocks: g2_blocks.len(),
                disk_blocks: g3_blocks.len(),
                object_blocks: 0,
            };
            Ok(FindOutcome {
                g2_committed,
                g2_blocks,
                g3_committed,
                g3_blocks,
                g1_blocks: None,
                breakdown,
            })
        }
    }
}

/// Drive the session's commit, availability, and terminator calls.
/// G1 sources reach temporary G2 before the first availability batch.
/// G3 sources use the existing local staging path.
/// Each checksum uses its position in the full committed set.
///
/// Errors propagate as `ControlError::Internal`; on error the caller
/// is expected to call `session.close(...)` to surface
/// `LifecycleEvent::Failed` to the puller.
async fn stage_phase(
    leader: Arc<InstanceLeader>,
    session: Arc<dyn Session>,
    resource: LogicalResourceId,
    require_payload_integrity: bool,
    committed: Vec<SequenceHash>,
    find: FindOutcome,
) -> Result<(), ControlError> {
    let FindOutcome {
        mut g2_blocks,
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
    // Commit G3 hashes up front so the puller sees the full
    // committed set via `commits()` before staging completes.
    if !committed.is_empty() {
        session
            .commit(committed)
            .map_err(|error| ControlError::Internal(format!("commit blocks: {error:#}")))?;
    }
    if let Some(source) = g1_blocks {
        g2_blocks.extend(
            source
                .stage()
                .await
                .map_err(|error| ControlError::Internal(format!("stage G1 blocks: {error:#}")))?,
        );
    }
    g2_blocks.sort_by_key(|block| ordinals.get(&block.sequence_hash()).copied());
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
        .finish_commits()
        .map_err(|e| ControlError::Internal(format!("finish_commits: {e:#}")))?;
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
    let committed = find.committed(&req.sequence_hashes);
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
    let stage_committed = committed.clone();
    runtime.spawn(async move {
        match stage_phase(
            leader_for_task,
            Arc::clone(&session_for_task),
            resource,
            require_payload_integrity,
            stage_committed,
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
/// The legacy one-resource endpoint publishes immediately after the shared
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
