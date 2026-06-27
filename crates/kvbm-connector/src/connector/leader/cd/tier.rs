// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo tier-signal handler writing the engine-owned [`TierCell`].
//!
//! The hub's prefill-router breaker PUSHES `(tier, epoch)` to each decode;
//! the engine reads the cached cell synchronously inside its search path.
//! The handler must be installed BEFORE the hub registration: the hub seeds
//! the current tier on `ConditionalDisaggManager::on_register`, and a push
//! racing ahead of the handler would be dropped — defeating the
//! seed-on-join (the exact case the seeding exists for: a hub already HOT
//! when a decode joins).

use std::sync::Arc;

use anyhow::{Result, anyhow};

use kvbm_engine::cd::TierCell;
use kvbm_hub::{TIER_SIGNAL_HANDLER, TierSignal, TierSignalAck};

/// Apply one inbound tier push to the cell and build its ack. Pure handler
/// body (no velo), factored out so the epoch-gating contract is unit-testable.
///
/// The ack's `ok` mirrors [`TierCell::apply`]: `true` for a winning or
/// equal-epoch idempotent push, `false` for a recency-rejected (strictly
/// older) one — which stores nothing.
pub(in super::super) fn apply_tier_signal(signal: TierSignal, cell: &TierCell) -> TierSignalAck {
    let applied = cell.apply(signal.tier, signal.epoch);
    if applied {
        tracing::info!(
            tier = ?signal.tier,
            epoch = signal.epoch,
            "decode: applied CD-breaker tier push"
        );
    } else {
        tracing::debug!(
            tier = ?signal.tier,
            epoch = signal.epoch,
            cached_epoch = cell.epoch(),
            "decode: ignored stale CD-breaker tier push (lower epoch)"
        );
    }
    TierSignalAck {
        epoch: signal.epoch,
        ok: applied,
    }
}

/// Install the velo `TIER_SIGNAL` handler targeting the engine's [`TierCell`].
/// Connector twin of the legacy `install_tier_signal_handler` (which is
/// typed to the legacy `DecodeTierCache`); same handler key, same payload,
/// same ack.
pub(in super::super) fn install_tier_signal_handler(
    messenger: &Arc<velo::Messenger>,
    cell: Arc<TierCell>,
) -> Result<()> {
    let handler = velo::Handler::typed_unary_async::<TierSignal, TierSignalAck, _, _>(
        TIER_SIGNAL_HANDLER,
        move |ctx| {
            let cell = Arc::clone(&cell);
            async move { Ok(apply_tier_signal(ctx.input, &cell)) }
        },
    )
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow!("registering CD tier-signal handler: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use kvbm_protocols::disagg::BreakerTier;

    use super::*;

    fn signal(tier: BreakerTier, epoch: u64) -> TierSignal {
        TierSignal { tier, epoch }
    }

    #[test]
    fn default_cell_is_calm_at_epoch_zero() {
        let cell = TierCell::default();
        assert_eq!(cell.tier(), BreakerTier::Calm);
        assert_eq!(cell.epoch(), 0);
    }

    #[test]
    fn newer_epoch_applies_and_acks_ok() {
        let cell = TierCell::default();
        let ack = apply_tier_signal(signal(BreakerTier::Hot, 5), &cell);
        assert!(ack.ok);
        assert_eq!(ack.epoch, 5);
        assert_eq!(cell.tier(), BreakerTier::Hot);
        assert_eq!(cell.epoch(), 5);
    }

    #[test]
    fn stale_epoch_is_rejected_and_state_unchanged() {
        let cell = TierCell::default();
        assert!(apply_tier_signal(signal(BreakerTier::Hot, 50), &cell).ok);

        let ack = apply_tier_signal(signal(BreakerTier::Calm, 10), &cell);
        assert!(!ack.ok, "strictly older push must ack ok=false");
        assert_eq!(ack.epoch, 10, "ack echoes the push's epoch");
        assert_eq!(cell.tier(), BreakerTier::Hot, "tier must not regress");
        assert_eq!(cell.epoch(), 50, "epoch must not regress");
    }

    #[test]
    fn equal_epoch_reapply_is_idempotent_ok() {
        let cell = TierCell::default();
        assert!(apply_tier_signal(signal(BreakerTier::Warm, 7), &cell).ok);

        let ack = apply_tier_signal(signal(BreakerTier::Warm, 7), &cell);
        assert!(ack.ok, "equal-epoch re-apply is accepted");
        assert_eq!(cell.tier(), BreakerTier::Warm);
        assert_eq!(cell.epoch(), 7);
    }
}
