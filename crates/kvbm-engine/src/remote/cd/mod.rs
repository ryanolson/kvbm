// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional-disaggregation pure logic: the prefill-selection policy, the
//! inflight-token admission budget + breaker tier cell, and the per-request
//! decode-side bookkeeping. Ported from the legacy connector's disagg leaders,
//! reduced to the resource-free derivation core.
//!
//! Wired through `RemoteOps::with_disagg_transports`: the connector assembles
//! the transports (session factory, prefill plane, tier cell, peer resolver)
//! and the engine builds its CD runtime from them. The connector-facing types
//! re-export at [`crate::cd`].

pub(crate) mod budget;
pub(crate) mod commit;
pub(crate) mod decode;
pub(crate) mod output;
pub(crate) mod policy;
pub(crate) mod prefill;
pub(crate) mod state;
pub(crate) mod wire;

use std::time::Duration;

use policy::SelectionPolicy;

/// Engine-local conditional-disaggregation configuration consumed by
/// [`decode::plan`]. Distinct from the on-the-wire `kvbm_config::DisaggConfig`:
/// this is the resource-free decision-core view (the selection policy plus the
/// two decode-side admission knobs), built by the wiring layer from the parsed
/// config via [`Self::from_connector_config`].
///
/// The struct is `pub` (re-exported at [`crate::cd`]) but its fields stay
/// `pub(crate)`: the connector constructs it ONLY through
/// `from_connector_config`, which covers every translation arm — so
/// [`SelectionPolicy`] itself never needs to cross the crate boundary.
///
/// The manual [`Default`] is the inert local-only base — `Never` /
/// `usize::MAX` / `true` — so a defaulted config preserves a purely-local
/// connector (the policy never selects `Remote`, the budget never throttles, and
/// the overload path would downgrade rather than reject if it were ever reached).
#[derive(Debug, Clone, Copy)]
pub struct DisaggConfig {
    /// Per-request prefill-selection policy.
    pub(crate) selection: SelectionPolicy,
    /// Capacity (in tokens) of the decode-side inflight remote-prefill budget.
    /// The wiring layer constructs the [`budget::InflightBudget`] from this;
    /// [`decode::plan`] reads the budget directly, not this field.
    pub(crate) max_inflight_remote_prefill_tokens: usize,
    /// On inflight-budget exhaustion: `true` downgrades the Remote decision to a
    /// local prefill; `false` rejects it (no external match onboarded).
    pub(crate) local_fallback_on_overload: bool,
    /// Poll interval of the prefill release's deferred-finalize drain — how
    /// often the drain task re-checks the output observer's `has_pending`.
    pub(crate) output_drain_poll: Duration,
    /// Watchdog bound on that drain: a residual that never empties forces the
    /// session finalize after this long (with a warning), so wedged output can
    /// never park the session forever.
    pub(crate) output_drain_watchdog: Duration,
}

impl Default for DisaggConfig {
    fn default() -> Self {
        Self {
            selection: SelectionPolicy::Never,
            max_inflight_remote_prefill_tokens: usize::MAX,
            local_fallback_on_overload: true,
            output_drain_poll: Duration::from_millis(2),
            output_drain_watchdog: Duration::from_secs(10),
        }
    }
}

impl DisaggConfig {
    /// Translate the connector's parsed `kvbm_config::DisaggConfig` into the
    /// engine's decision-core view:
    ///
    /// * `role == Prefill` ⇒ [`SelectionPolicy::Never`] — a prefill-role worker
    ///   must never CD-dispatch its own traffic (the legacy prefill leader had
    ///   no decode policy at all, and the hub client's role guard would reject
    ///   its push anyway).
    /// * `role == Decode` ⇒ `min_remote_prefill_tokens == 0` selects
    ///   [`SelectionPolicy::Always`]; a positive threshold selects
    ///   [`SelectionPolicy::Threshold`].
    /// * `max_inflight_remote_prefill_tokens` / `cd_local_fallback_on_overload`
    ///   copy through directly.
    /// * The output-drain knobs keep the engine defaults (no connector knob).
    pub fn from_connector_config(cfg: &kvbm_config::DisaggConfig) -> Self {
        let selection = match cfg.role {
            kvbm_config::DisaggregationRole::Prefill => SelectionPolicy::Never,
            kvbm_config::DisaggregationRole::Decode => {
                if cfg.min_remote_prefill_tokens == 0 {
                    SelectionPolicy::Always
                } else {
                    SelectionPolicy::Threshold {
                        min_remote_prefill_tokens: cfg.min_remote_prefill_tokens,
                    }
                }
            }
        };
        Self {
            selection,
            max_inflight_remote_prefill_tokens: cfg.max_inflight_remote_prefill_tokens,
            local_fallback_on_overload: cfg.cd_local_fallback_on_overload,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvbm_config::DisaggregationRole;

    fn connector_cfg(role: DisaggregationRole) -> kvbm_config::DisaggConfig {
        kvbm_config::DisaggConfig {
            role,
            ..Default::default()
        }
    }

    #[test]
    fn prefill_role_translates_to_never() {
        // Even with a decode-only threshold set, the prefill role must never
        // select Remote for its own traffic.
        let cfg = kvbm_config::DisaggConfig {
            min_remote_prefill_tokens: 256,
            ..connector_cfg(DisaggregationRole::Prefill)
        };
        let translated = DisaggConfig::from_connector_config(&cfg);
        assert!(matches!(translated.selection, SelectionPolicy::Never));
    }

    #[test]
    fn decode_zero_threshold_translates_to_always() {
        let cfg = connector_cfg(DisaggregationRole::Decode);
        assert_eq!(cfg.min_remote_prefill_tokens, 0, "serde default");
        let translated = DisaggConfig::from_connector_config(&cfg);
        assert!(matches!(translated.selection, SelectionPolicy::Always));
    }

    #[test]
    fn decode_positive_threshold_translates_to_threshold() {
        let cfg = kvbm_config::DisaggConfig {
            min_remote_prefill_tokens: 512,
            ..connector_cfg(DisaggregationRole::Decode)
        };
        let translated = DisaggConfig::from_connector_config(&cfg);
        assert!(matches!(
            translated.selection,
            SelectionPolicy::Threshold {
                min_remote_prefill_tokens: 512
            }
        ));
    }

    #[test]
    fn admission_knobs_copy_through() {
        let cfg = kvbm_config::DisaggConfig {
            max_inflight_remote_prefill_tokens: 4096,
            cd_local_fallback_on_overload: false,
            ..connector_cfg(DisaggregationRole::Decode)
        };
        let translated = DisaggConfig::from_connector_config(&cfg);
        assert_eq!(translated.max_inflight_remote_prefill_tokens, 4096);
        assert!(!translated.local_fallback_on_overload);
    }

    #[test]
    fn drain_knobs_keep_engine_defaults() {
        let translated =
            DisaggConfig::from_connector_config(&connector_cfg(DisaggregationRole::Decode));
        let defaults = DisaggConfig::default();
        assert_eq!(translated.output_drain_poll, defaults.output_drain_poll);
        assert_eq!(
            translated.output_drain_watchdog,
            defaults.output_drain_watchdog
        );
    }
}
