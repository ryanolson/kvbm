// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo active-message handlers that KVBM clients install.
//!
//! These are the control-plane messages the hub sends to its clients over
//! velo (not HTTP). HTTP is reserved for discovery + bootstrap registration;
//! velo is used once both sides know about each other.

use kvbm_protocols::disagg::BreakerTier;
use serde::{Deserialize, Serialize};

/// Velo handler name for the hub → decode CD-circuit-breaker tier push.
///
/// The hub's prefill-router breaker-tick task fans this message out to the
/// DECODE participants of the ConditionalDisagg feature (never to prefill) on
/// every tier transition and on a new decode registration. The decode side
/// installs a handler under this name that caches the latest tier (idempotent,
/// applied only when `epoch >= cached_epoch`) and reads it synchronously inside
/// GNMT. See `kvbm-connector`'s decode leader for the consumer.
pub const TIER_SIGNAL_HANDLER: &str = "kvbm_hub_tier_signal";

/// Payload pushed by the hub to a decode worker on a CD-breaker tier change.
///
/// `epoch` is a hub-boot-monotonic, strictly-increasing value bumped on every
/// tier transition; the decode side keeps only the highest-epoch tier it has
/// seen (latest-by-recency, absolute state — NOT a toggle). The `epoch` lets a
/// RESTARTED hub's pushes win over a stale value a decode cached from a prior
/// hub instance (the hub seeds its epoch above any prior value).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TierSignal {
    /// The router-owned tier the decode should adopt.
    pub tier: BreakerTier,
    /// Hub-boot-monotonic epoch; higher == more recent.
    pub epoch: u64,
}

/// Response returned by the decode worker to acknowledge a tier signal.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TierSignalAck {
    /// Echoed `epoch` from the request.
    pub epoch: u64,
    /// `true` once the decode has applied (or already had a `>=`-epoch) tier.
    pub ok: bool,
}
