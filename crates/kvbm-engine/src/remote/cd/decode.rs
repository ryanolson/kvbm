// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side conditional-disaggregation decision core.
//!
//! [`plan`] is the resource-free derivation of the legacy decode GNMT decision
//! order (policy → local/remote → breaker → zero-block → budget → commit),
//! reduced to a pure function whose ONLY side effect is the inflight-budget
//! reservation on the [`PlanOutcome::Remote`] outcome.
//!
//! Deliberately NOT here (these belong to the calling/wiring layer):
//! * The idempotent same-request retry — vLLM may invoke GNMT multiple times for
//!   one request without an intervening allocation; the caller checks its own
//!   per-request state and short-circuits BEFORE [`plan`] is ever reached.
//! * The pending passthrough — when the inner connector's `get_num_new_matched`
//!   returns `None` there is no match to onboard and the caller returns that
//!   passthrough directly, again BEFORE [`plan`]. [`plan`] is only consulted once
//!   a concrete `matched_tokens` count exists.
//! * Audit/metric emission — the wiring layer emits the kvbm_audit events and the
//!   `cd_metrics` Prometheus counters using the [`LocalReason`] → label mapping
//!   documented on that enum, so the strings stay identical to the legacy path.

use kvbm_protocols::disagg::BreakerTier;

use super::DisaggConfig;
use super::budget::{InflightBudget, TierCell};
use super::policy::{PolicyInputs, PrefillSelection};

/// Token-denominated inputs for a single decode GNMT decision. Callers quantize
/// from block counts before constructing this (every field is a token count).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlanInputs {
    /// Full sequence length of the request (prompt tokens).
    pub(crate) total_tokens: usize,
    /// Tokens vLLM already has cached in G1 (the prefix-cache hit length).
    pub(crate) num_computed_tokens: usize,
    /// Tokens matched by the inner connector's local G2 cache.
    pub(crate) matched_tokens: usize,
    /// Block granularity used to floor the external prefill window.
    pub(crate) block_size: usize,
}

/// Final placement decision for one decode GNMT call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanOutcome {
    /// Prefill stays local on the decode worker. See [`LocalReason`] for which of
    /// the four collapse paths produced it; the budget is untouched.
    Local { reason: LocalReason },
    /// Remote prefill is declined AND no external match is onboarded: the caller
    /// answers `(None, false)` so vLLM recomputes the whole uncached prompt
    /// itself. Reached only when the inflight budget is exhausted and
    /// `local_fallback_on_overload` is `false`. Nothing was reserved.
    Reject,
    /// Remote prefill is committed. The reservation of `full_block_external_tokens`
    /// tokens against the inflight budget is now HELD — [`plan`] performed the
    /// `try_reserve` and it succeeded. The CALLER owns releasing exactly this many
    /// tokens back to the budget on any downstream outcome: transfer/onboard
    /// failure, action terminal, or block eviction. Dropping the outcome without
    /// releasing leaks the reservation.
    Remote { full_block_external_tokens: usize },
}

/// Why a request collapsed to a local prefill (or was rejected). Each variant
/// maps 1:1 to the legacy metric labels so the wiring layer can emit identical
/// strings; the table below is authoritative for that wiring:
///
/// | outcome / variant            | `record_decision`                  | `record_remote_declined` |
/// |------------------------------|------------------------------------|--------------------------|
/// | `Local { Policy }`           | `"local"`                          | (none)                   |
/// | `Local { BreakerHot }`       | `"remote_downgraded_breaker_hot"`  | `"breaker_hot"`          |
/// | `Local { ZeroBlock }`        | `"remote_downgraded_zero_block"`   | `"zero_block"`           |
/// | `Local { OverloadFallback }` | `"remote_downgraded_overload"`     | `"budget_exhausted"`     |
/// | `Reject`                     | `"remote_rejected_budget"`         | `"budget_exhausted"`     |
/// | `Remote { .. }`              | `"remote"`                         | (none)                   |
///
/// Token-accounting nuance for the local-prefill metric
/// (`kvbm_cd_local_prefill_tokens_total`):
/// * Every `Local { .. }` variant returns the inner passthrough `(Some(matched),
///   …)`, so vLLM onboards the G2 local match and the recorded local quantity is
///   [`PolicyInputs::num_prefill_tokens`] = `total − computed − matched`.
/// * [`PlanOutcome::Reject`] returns `(None, false)`: the G2 match is NOT
///   onboarded, so vLLM recomputes the whole uncached prompt and the recorded
///   local quantity is `total − computed` (NOT `num_prefill_tokens()`, which
///   would undercount by exactly `matched`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocalReason {
    /// The selection policy chose `Local` outright.
    Policy,
    /// The CD circuit breaker is `Hot`; remote prefill is coarsely downgraded.
    BreakerHot,
    /// The block-floored external window is empty — no full block to send.
    ZeroBlock,
    /// The inflight budget is exhausted and `local_fallback_on_overload` downgraded
    /// the request rather than rejecting it.
    OverloadFallback,
}

/// Derive the decode placement decision for one GNMT call.
///
/// Deterministic and I/O-free. The single side effect is the inflight-budget
/// reservation performed on the [`PlanOutcome::Remote`] path (see that variant's
/// contract). The decision order is a 1:1 port of the legacy `decode_gnmt`:
///
/// 1. Policy evaluation over the request's token counts.
/// 2. `Local` ⇒ passthrough.
/// 3. `Remote` ⇒ size the window: `prefill_window = total − computed` (saturating),
///    `full_block_external_tokens = (prefill_window / block_size) * block_size`.
/// 4. Breaker tier: `Hot` downgrades to local; `Warm` and `Calm` both continue.
///    `Warm` is explicitly NOT a downgrade — it can only ever match or NARROW
///    disaggregation versus the existing flow (more `Local`, never more `Remote`),
///    so it falls through to the same path as `Calm`.
/// 5. `full_block_external_tokens == 0` ⇒ downgrade to local (zero-block).
/// 6. `try_reserve` fails ⇒ if `local_fallback_on_overload` downgrade to local
///    (nothing was reserved — there is zero CD state to unwind, so this is
///    behaviorally identical to a policy-`Local` decision and only ever narrows
///    disaggregation); else `Reject` (no external match onboarded).
/// 7. Otherwise the reservation is HELD ⇒ `Remote`.
pub(crate) fn plan(
    cfg: &DisaggConfig,
    tier: &TierCell,
    budget: &InflightBudget,
    inputs: &PlanInputs,
) -> PlanOutcome {
    let policy_inputs = PolicyInputs {
        total_tokens: inputs.total_tokens,
        num_computed_tokens: inputs.num_computed_tokens,
        num_connector_tokens: inputs.matched_tokens,
    };

    match cfg.selection.select(&policy_inputs) {
        PrefillSelection::Local => PlanOutcome::Local {
            reason: LocalReason::Policy,
        },
        PrefillSelection::Remote => {
            let prefill_window = inputs
                .total_tokens
                .saturating_sub(inputs.num_computed_tokens);
            let full_block_external_tokens =
                (prefill_window / inputs.block_size) * inputs.block_size;

            match tier.tier() {
                BreakerTier::Hot => {
                    return PlanOutcome::Local {
                        reason: LocalReason::BreakerHot,
                    };
                }
                // Warm is NOT a downgrade — it can only narrow versus the Calm
                // flow, so both continue to the existing path below.
                BreakerTier::Warm | BreakerTier::Calm => {}
            }

            if full_block_external_tokens == 0 {
                return PlanOutcome::Local {
                    reason: LocalReason::ZeroBlock,
                };
            }

            if !budget.try_reserve(full_block_external_tokens) {
                // try_reserve FAILED, so nothing was reserved and no per-request
                // CD state exists — the fallback downgrade has zero state to
                // unwind and is identical to a policy-Local decision.
                if cfg.local_fallback_on_overload {
                    return PlanOutcome::Local {
                        reason: LocalReason::OverloadFallback,
                    };
                }
                return PlanOutcome::Reject;
            }

            // Reservation of `full_block_external_tokens` is now held by the caller.
            PlanOutcome::Remote {
                full_block_external_tokens,
            }
        }
    }
}

/// Block-aligned uncached remote-COMPUTE remainder for the
/// `kvbm_cd_remote_prefill_tokens_total` metric: the external window minus the
/// block-floored local match (which the remote PULLS rather than computes).
///
/// `fbet` already excludes the prefix-cache hit (it is `(total − computed)`
/// block-floored). Subtracting only the block-floored match — NOT the raw
/// `matched_tokens` — keeps the result block-aligned so it reconciles 1:1 with
/// the prefill-side computed tokens; using `fbet` alone would over-count by the
/// local match.
pub(crate) fn remote_compute_tokens(
    fbet: usize,
    matched_tokens: usize,
    block_size: usize,
) -> usize {
    let local_match_tokens = (matched_tokens / block_size) * block_size;
    fbet.saturating_sub(local_match_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::cd::policy::SelectionPolicy;

    const BS: usize = 16;

    fn cfg(selection: SelectionPolicy, local_fallback_on_overload: bool) -> DisaggConfig {
        DisaggConfig {
            selection,
            max_inflight_remote_prefill_tokens: usize::MAX,
            local_fallback_on_overload,
            ..DisaggConfig::default()
        }
    }

    fn always(local_fallback_on_overload: bool) -> DisaggConfig {
        cfg(SelectionPolicy::Always, local_fallback_on_overload)
    }

    fn tier_at(t: BreakerTier) -> TierCell {
        let cell = TierCell::default();
        if t != BreakerTier::Calm {
            assert!(cell.apply(t, 1));
        }
        cell
    }

    fn inputs(total: usize, computed: usize, matched: usize) -> PlanInputs {
        PlanInputs {
            total_tokens: total,
            num_computed_tokens: computed,
            matched_tokens: matched,
            block_size: BS,
        }
    }

    #[test]
    fn policy_local_passthrough() {
        let cfg = cfg(SelectionPolicy::Never, true);
        let tier = TierCell::default();
        let budget = InflightBudget::new(256);
        let out = plan(&cfg, &tier, &budget, &inputs(1000, 0, 0));
        assert_eq!(
            out,
            PlanOutcome::Local {
                reason: LocalReason::Policy
            }
        );
        // Never even reaches the budget.
        assert_eq!(budget.available(), 256);
    }

    #[test]
    fn remote_breaker_hot_downgrades_local_budget_untouched() {
        let cfg = always(true);
        let tier = tier_at(BreakerTier::Hot);
        let budget = InflightBudget::new(256);
        let out = plan(&cfg, &tier, &budget, &inputs(10 * BS, 0, 0));
        assert_eq!(
            out,
            PlanOutcome::Local {
                reason: LocalReason::BreakerHot
            }
        );
        // Hot returns before try_reserve — budget never consulted.
        assert_eq!(budget.available(), 256);
    }

    #[test]
    fn remote_calm_zero_window_downgrades_zero_block() {
        let cfg = always(true);
        let tier = tier_at(BreakerTier::Calm);
        let budget = InflightBudget::new(256);
        // window = 10 < block_size 16 ⇒ fbet = 0.
        let out = plan(&cfg, &tier, &budget, &inputs(10, 0, 0));
        assert_eq!(
            out,
            PlanOutcome::Local {
                reason: LocalReason::ZeroBlock
            }
        );
        assert_eq!(budget.available(), 256);
    }

    #[test]
    fn remote_total_below_computed_saturates_to_zero_block() {
        let cfg = always(true);
        let tier = TierCell::default();
        let budget = InflightBudget::new(256);
        // total < computed ⇒ saturating window 0 ⇒ fbet 0 ⇒ ZeroBlock.
        let out = plan(&cfg, &tier, &budget, &inputs(10, 20, 0));
        assert_eq!(
            out,
            PlanOutcome::Local {
                reason: LocalReason::ZeroBlock
            }
        );
        assert_eq!(budget.available(), 256);
    }

    #[test]
    fn remote_budget_exhausted_with_fallback_downgrades_local_budget_unchanged() {
        let cfg = always(true);
        let tier = TierCell::default();
        // fbet = 128, capacity 100 ⇒ try_reserve fails.
        let budget = InflightBudget::new(100);
        let out = plan(&cfg, &tier, &budget, &inputs(8 * BS, 0, 0));
        assert_eq!(
            out,
            PlanOutcome::Local {
                reason: LocalReason::OverloadFallback
            }
        );
        // Failed try_reserve must leave available untouched.
        assert_eq!(budget.available(), 100);
    }

    #[test]
    fn remote_budget_exhausted_without_fallback_rejects_budget_unchanged() {
        let cfg = always(false);
        let tier = TierCell::default();
        let budget = InflightBudget::new(100);
        let out = plan(&cfg, &tier, &budget, &inputs(8 * BS, 0, 0));
        assert_eq!(out, PlanOutcome::Reject);
        // Failed try_reserve must leave available untouched on the reject path too.
        assert_eq!(budget.available(), 100);
    }

    #[test]
    fn remote_with_computed_prefix_reserves_block_floored_suffix_only() {
        let cfg = always(true);
        let tier = TierCell::default();
        let budget = InflightBudget::new(256);
        // computed prefix of 2 blocks; the ragged 4*BS + 3 window floors to
        // 4 blocks. The reservation covers the SUFFIX window only — the
        // prefix never reserves.
        let out = plan(&cfg, &tier, &budget, &inputs(6 * BS + 3, 2 * BS, BS));
        assert_eq!(
            out,
            PlanOutcome::Remote {
                full_block_external_tokens: 4 * BS
            }
        );
        assert_eq!(budget.available(), 256 - 4 * BS);
    }

    #[test]
    fn remote_success_reserves_exactly_fbet() {
        let cfg = always(true);
        let tier = TierCell::default();
        let budget = InflightBudget::new(256);
        // window = 4*BS = 64 ⇒ fbet = 64.
        let out = plan(&cfg, &tier, &budget, &inputs(4 * BS, 0, 0));
        assert_eq!(
            out,
            PlanOutcome::Remote {
                full_block_external_tokens: 4 * BS
            }
        );
        assert_eq!(budget.available(), 256 - 4 * BS);
    }

    #[test]
    fn warm_behaves_exactly_as_calm() {
        let cfg = always(true);
        let budget_warm = InflightBudget::new(256);
        let budget_calm = InflightBudget::new(256);
        let in_ = inputs(4 * BS, 0, 0);

        let warm = plan(&cfg, &tier_at(BreakerTier::Warm), &budget_warm, &in_);
        let calm = plan(&cfg, &tier_at(BreakerTier::Calm), &budget_calm, &in_);

        // Same outcome.
        assert_eq!(warm, calm);
        assert_eq!(
            warm,
            PlanOutcome::Remote {
                full_block_external_tokens: 4 * BS
            }
        );
        // Same reservation.
        assert_eq!(budget_warm.available(), budget_calm.available());
        assert_eq!(budget_warm.available(), 256 - 4 * BS);
    }

    #[test]
    fn fbet_quantization_window_eq_block_size() {
        let cfg = always(true);
        let tier = TierCell::default();
        let budget = InflightBudget::new(256);
        // window == block_size ⇒ fbet == block_size.
        let out = plan(&cfg, &tier, &budget, &inputs(BS, 0, 0));
        assert_eq!(
            out,
            PlanOutcome::Remote {
                full_block_external_tokens: BS
            }
        );
        assert_eq!(budget.available(), 256 - BS);
    }

    #[test]
    fn fbet_quantization_window_eq_two_blocks_minus_one_floors_to_one_block() {
        let cfg = always(true);
        let tier = TierCell::default();
        let budget = InflightBudget::new(256);
        // window == 2*BS - 1 ⇒ fbet == BS (floored down).
        let out = plan(&cfg, &tier, &budget, &inputs(2 * BS - 1, 0, 0));
        assert_eq!(
            out,
            PlanOutcome::Remote {
                full_block_external_tokens: BS
            }
        );
        assert_eq!(budget.available(), 256 - BS);
    }

    #[test]
    fn remote_compute_tokens_matched_zero_is_full_fbet() {
        assert_eq!(remote_compute_tokens(4 * BS, 0, BS), 4 * BS);
    }

    #[test]
    fn remote_compute_tokens_matched_mid_block_floors_down() {
        // matched = 2*BS + 5 ⇒ block-floored match = 2*BS ⇒ remainder = 4*BS - 2*BS.
        assert_eq!(remote_compute_tokens(4 * BS, 2 * BS + 5, BS), 2 * BS);
    }

    #[test]
    fn remote_compute_tokens_matched_ge_fbet_saturates_to_zero() {
        assert_eq!(remote_compute_tokens(2 * BS, 4 * BS, BS), 0);
    }
}
