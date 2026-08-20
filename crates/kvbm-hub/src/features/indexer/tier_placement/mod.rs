// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side consumer for the tier-placement advisory stream (R7b §4).
//!
//! # What this owns
//!
//! One projection per `(cache, instance)` of "which keys this instance holds
//! Ready, at which depth". Deltas arrive lossily over ZMQ; snapshots arrive
//! reliably and authenticated over the HTTP control plane. The projection
//! tracks the continuity evidence that reaches it. If it detects a
//! failure, it answers **empty** until an authorized snapshot installs.
//!
//! # The invalid-state invariant
//!
//! *An invalid projection is a temporary miss.* Every read path filters on
//! `valid` before it consults the ready map. Thus, a detected continuity failure
//! cannot produce a holder.
//!
//! This guarantee starts after detection. ZMQ can drop a terminal delta with no
//! later sequence evidence. The projection then remains valid by its local
//! evidence. A lost terminal `Remove` can leave a stale holder until a
//! successful snapshot installs. Every holder is advisory. The caller must
//! acquire the exact owner `BundleLease` before it trusts that holder.
//!
//! # Why this is a sub-module of `indexer`, not its own feature
//!
//! A top-level `FeatureManager` would cost a [`FeatureKey`] variant, a
//! [`Feature`] payload variant, CLI/registration/dependency-closure wiring —
//! and would still need its own copy of the owner-credential and
//! [`RegistrationEpoch`] machinery. Everything the projection needs,
//! [`IndexerManager`] already owns: owner credentials and epochs (through
//! [`BundleDirectory`]), `on_unregister` parity, the registered-instance set,
//! and the `/v1/features/indexer` namespace.
//!
//! [`FeatureKey`]: crate::protocol::FeatureKey
//! [`Feature`]: crate::protocol::Feature
//! [`IndexerManager`]: super::manager::IndexerManager
//! [`BundleDirectory`]: super::bundle::BundleDirectory
//!
//! # Naming
//!
//! `kvbm-hub` already uses "tier" for the conditional-disagg circuit breaker
//! (`TIER_SIGNAL_HANDLER`, `handlers::TierSignal`,
//! `kvbm_protocols::disagg::BreakerTier`). Nothing here is ever named a bare
//! `Tier*`: it is `TierPlacement*` / `tier_placement` throughout, so no reader
//! mistakes a placement record for breaker state.
//!
//! # The periodic-push window, and why the publisher must gate on the ack
//!
//! Every snapshot bumps the generation, so the deltas that follow one carry a
//! generation the hub has not installed until the snapshot's HTTP call lands.
//! Deltas travel the fast lossy plane and snapshots the reliable slow one, so an
//! ungated publisher's deltas routinely win the race: the projection sees
//! `GenerationAhead`, invalidates, requests, and answers empty until the
//! snapshot installs.
//!
//! It does **not** stop there, and an earlier version of this note wrongly said
//! the window was bounded by the publisher's push latency. The delta that
//! outran the snapshot is dropped, and ZMQ pub/sub has no retransmission, so
//! after the install resumes the projection at `seq_floor + 1` the next real
//! delta is `seq_floor + 2` and gaps. The publisher then pushes another
//! snapshot, whose own successor delta outruns it in turn. Under sustained
//! emission the projection is valid only for the instant between an install and
//! the next delta — an availability collapse, not a bounded window.
//!
//! The fix is publisher-side and lives in
//! [`TierPlacementSequencer`](kvbm_protocols::tier_protocol::TierPlacementSequencer):
//! sealing a snapshot arms an emission gate that `seal` honours until
//! `note_snapshot_installed` confirms the install landed, so the first
//! post-snapshot delta cannot precede the state it describes. Nothing on the
//! consumer side changes, and nothing on the consumer side *can* — buffering
//! future-generation deltas here would contradict replace-all recovery, and
//! accepting a delta past the resume point would be exactly the silent hole the
//! sequence rule exists to catch.
//!
//! The gate is not the sequencer's only publisher-side lever. A publisher that
//! loses an op before transport handoff leaves the sequence contiguous. The
//! projection cannot detect that loss. `TierPlacementSequencer::mark_divergent`
//! reports it. The method burns one sequence number, which arrives as an ordinary
//! `InvalidationReason::SequenceGap` and recovers through the path already
//! documented above.
//!
//! The transport gives the publisher no result after handoff. Thus, a final
//! dropped batch remains invisible to both sides. A later batch exposes the
//! gap. A successful snapshot also replaces the stale state. Until one event
//! occurs, a terminal lost `Remove` can remain visible.
//!
//! If a publisher ignores the gate, the generation race is visible:
//! `invalidated_generation_ahead` and `invalidated_sequence_gap` climb together
//! with `snapshot_requests` while `snapshots_installed` lags. The projection
//! answers empty after either invalidation.
//!
//! # Trust
//!
//! The delta plane is unauthenticated (the same trust level as the legacy index
//! stream it rides alongside), but it carries epochs and generations, which are
//! levers the legacy stream lacks. The mitigations are structural rather than
//! advisory:
//!
//! - `installed_epoch` is written **only** by a credential-authorized snapshot
//!   install, so a forged batch cannot install an epoch and cannot make itself
//!   authoritative.
//! - A delta never creates a map entry. Both halves of the key would otherwise
//!   need bounding, and only one of them is authenticated: the registered set
//!   bounds `instance_id`, but `cache` is publisher-chosen, so entry creation on
//!   the delta path would let one registered id plus N forged cache ids fill the
//!   map — and starve the snapshot install that is the only route back to valid.
//!   The map grows through authorized installs alone.
//! - Snapshot requests are rate-limited, so a persistently lossy or hostile link
//!   cannot amplify each dropped batch into an active message.
//! - Every rejection path ends in *empty*. There is no rejection path that ends
//!   in "serve what we had".
//!
//! What these do **not** buy, stated plainly: a forged batch carrying a wrong
//! [`RegistrationEpoch`] clears an installed projection, and one such packet per
//! rate-limit window holds it empty for as long as the attacker keeps sending.
//! That is not a bug to fix here. [`RegistrationEpoch`] is a UUID, so epochs are
//! unordered and "old" is not a thing the consumer can recognise; and the only
//! alternative — ignoring mismatched-epoch deltas without clearing — would keep
//! serving a dead process lifetime's Ready set as *valid* after a genuine
//! publisher restart, which is the stale success this whole module exists to
//! prevent. Fail-safe is the correct trade at this trust level, and the tier
//! stream is strictly better off than the legacy index stream it rides beside,
//! where a forged frame inserts a false holder outright. Authenticating the
//! delta plane is the real answer and is a separate design change.
//!
//! [`RegistrationEpoch`]: kvbm_protocols::cache_manifest::RegistrationEpoch

use std::sync::Arc;

use kvbm_protocols::cache_manifest::CacheManifestId;
use kvbm_protocols::tier_protocol::InstanceId;

mod projection;

#[cfg(test)]
mod tests;

pub use projection::{
    DeltaOutcome, DiscardReason, InvalidationReason, ProjectionLimits, SnapshotInstall,
    TierPlacementCounters, TierPlacementHolder, TierPlacementProjection,
    TierPlacementProjectionError,
};

/// Velo active-message handler the *publisher* installs, and the hub calls, to
/// ask for a fresh snapshot after it loses continuity.
///
/// Named without a leading underscore to match the handler convention actually
/// in the source (`kvbm_hub_indexer_query`, `kvbm_hub_bundle_publish`); R7b §3's
/// `_kvbm_tier_snapshot_request` and the hub CLAUDE.md's `_kvbm_hub_heartbeat`
/// both disagree with the code, and the code wins.
pub const TIER_PLACEMENT_SNAPSHOT_REQUEST_HANDLER: &str =
    "kvbm_hub_tier_placement_snapshot_request";

/// Hub → publisher: "I lost continuity for this cache; push me a snapshot."
///
/// Carries no sequence or generation on purpose. The hub is asking for *current*
/// state, and any number it supplied would be state the publisher would then
/// have to reason about — the whole point of replace-all recovery is that the
/// consumer's prior belief is irrelevant.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TierPlacementSnapshotRequestMsg {
    /// Cache whose projection the hub could not keep continuous.
    pub cache: CacheManifestId,
}

/// Publisher → hub acknowledgement of a snapshot request.
///
/// `accepted: false` means the publisher will not push (it is shutting down, or
/// does not publish this cache). The hub does not treat that as an error: the
/// projection simply stays invalid and keeps answering empty.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TierPlacementSnapshotRequestAck {
    /// Whether the publisher intends to push a snapshot.
    pub accepted: bool,
}

/// Fire-and-forget snapshot request, behind a seam.
///
/// The hub side of the recovery loop is "ask, then wait": it must not block a
/// ZMQ ingest tick on an active message, and it must not care whether the
/// request is delivered — an undelivered request costs one more interval of
/// empty answers, and the publisher's periodic push is the backstop.
pub trait SnapshotRequester: Send + Sync {
    /// Ask `instance` to push a full snapshot for `cache`. Must not block.
    fn request(&self, cache: CacheManifestId, instance: InstanceId);
}

/// Velo-backed requester installed by
/// [`IndexerManager::attach`](super::manager::IndexerManager) when the hub runs
/// with a transport.
pub struct VeloSnapshotRequester {
    messenger: Arc<velo::Messenger>,
}

impl VeloSnapshotRequester {
    /// Wrap the hub's messenger.
    #[must_use]
    pub fn new(messenger: Arc<velo::Messenger>) -> Self {
        Self { messenger }
    }
}

impl SnapshotRequester for VeloSnapshotRequester {
    fn request(&self, cache: CacheManifestId, instance: InstanceId) {
        let messenger = Arc::clone(&self.messenger);
        // Detached: the caller is holding no lock we want to extend across an
        // await, and the reply is advisory. Delivery failure is logged at debug
        // because a publisher that has already died is the common case, and the
        // projection is correct either way.
        tokio::spawn(async move {
            let call = messenger
                .typed_unary::<TierPlacementSnapshotRequestAck>(
                    TIER_PLACEMENT_SNAPSHOT_REQUEST_HANDLER,
                )
                .and_then(|builder| builder.payload(&TierPlacementSnapshotRequestMsg { cache }));
            let call = match call {
                Ok(call) => call.instance(instance).send(),
                Err(error) => {
                    tracing::debug!(%instance, %error, "tier placement snapshot request not encodable");
                    return;
                }
            };
            match call.await {
                Ok(ack) if ack.accepted => {
                    tracing::debug!(%instance, %cache, "tier placement snapshot requested");
                }
                Ok(_) => {
                    tracing::debug!(%instance, %cache, "publisher declined a tier placement snapshot");
                }
                Err(error) => {
                    tracing::debug!(%instance, %cache, %error, "tier placement snapshot request failed");
                }
            }
        });
    }
}

/// Requester for a hub with no transport (discovery-only).
///
/// Deliberately *not* a fallback that serves stale state: with no way to ask for
/// a snapshot, a projection that loses continuity stays invalid and keeps
/// answering empty, which is the correct degradation. The counter makes the
/// condition visible rather than silent.
pub struct UnavailableSnapshotRequester;

impl SnapshotRequester for UnavailableSnapshotRequester {
    fn request(&self, cache: CacheManifestId, instance: InstanceId) {
        tracing::debug!(
            %instance,
            %cache,
            "tier placement snapshot needed but this hub has no transport to request one"
        );
    }
}
