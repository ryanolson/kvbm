// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional-disaggregation prefill-selection policy.
//!
//! The policy decides, per request, whether prefill runs locally on the decode
//! worker or is dispatched to a remote prefill peer. The choice is engine
//! *configuration* data (not an extension point), so the legacy
//! `ConditionalDisaggPolicy` trait + unit-struct impls collapse into a single
//! [`SelectionPolicy`] enum.

/// Outcome of a per-request disaggregation policy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefillSelection {
    Local,
    Remote,
}

/// Inputs available when deciding whether a request should prefill locally or
/// remotely.
#[derive(Debug, Clone)]
pub(crate) struct PolicyInputs {
    pub total_tokens: usize,
    pub num_computed_tokens: usize,
    pub num_connector_tokens: usize,
}

impl PolicyInputs {
    pub fn num_prefill_tokens(&self) -> usize {
        self.total_tokens
            .saturating_sub(self.num_computed_tokens)
            .saturating_sub(self.num_connector_tokens)
    }
}

/// Per-request prefill-selection policy. The default-equivalent variant is
/// [`SelectionPolicy::Never`], which preserves a local-only connector.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SelectionPolicy {
    /// Always prefill locally on the decode worker.
    Never,
    /// Always dispatch prefill to a remote peer.
    Always,
    /// Disaggregate (`Remote`) only when the *uncached* prefill work meets
    /// `min_remote_prefill_tokens`, otherwise prefill locally on the decode
    /// worker. The comparison is against [`PolicyInputs::num_prefill_tokens`]
    /// (`total − num_computed − local connector match`) — i.e. the tokens that
    /// still need a prefill forward pass after local cache — not the raw prompt
    /// length.
    ///
    /// A threshold of `0` makes every request `Remote` (equivalent to
    /// [`SelectionPolicy::Always`]); larger values keep short prompts local.
    Threshold { min_remote_prefill_tokens: usize },
}

impl SelectionPolicy {
    pub fn select(&self, inputs: &PolicyInputs) -> PrefillSelection {
        match self {
            SelectionPolicy::Never => PrefillSelection::Local,
            SelectionPolicy::Always => PrefillSelection::Remote,
            SelectionPolicy::Threshold {
                min_remote_prefill_tokens,
            } => {
                if inputs.num_prefill_tokens() >= *min_remote_prefill_tokens {
                    PrefillSelection::Remote
                } else {
                    PrefillSelection::Local
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_remote_preserves_local_default() {
        let inputs = PolicyInputs {
            total_tokens: 128,
            num_computed_tokens: 16,
            num_connector_tokens: 32,
        };

        assert_eq!(
            SelectionPolicy::Never.select(&inputs),
            PrefillSelection::Local
        );
        assert_eq!(inputs.num_prefill_tokens(), 80);
    }

    #[test]
    fn decode_local_token_quantities_per_final_local_path() {
        // The decode-side `kvbm_cd_local_prefill_tokens_total` is recorded with a
        // DIFFERENT quantity depending on the final-local path, and the two must
        // not be conflated (the difference is exactly the G2 local-match):
        //   * passthrough paths — policy-Local, breaker-HOT, B-GNMT overload,
        //     zero-block — return (Some(matched), …), so vLLM onboards the G2
        //     match and computes num_prefill_tokens() = total − computed − match.
        //   * budget-reject — returns (None, false): NO external match onboarded,
        //     so vLLM recomputes the whole uncached prompt = total − computed.
        let inputs = PolicyInputs {
            total_tokens: 1000,
            num_computed_tokens: 100,  // vLLM G1 prefix-cache hit
            num_connector_tokens: 200, // kvbm G2 local match
        };
        // passthrough (G2 onboarded): total − computed − match
        assert_eq!(inputs.num_prefill_tokens(), 700);
        // budget-reject (None ⇒ G2 NOT onboarded): total − computed
        let reject_local_tokens = inputs
            .total_tokens
            .saturating_sub(inputs.num_computed_tokens);
        assert_eq!(reject_local_tokens, 900);
        // The reject path computes the local-match too — exactly num_connector more.
        assert_eq!(
            reject_local_tokens - inputs.num_prefill_tokens(),
            inputs.num_connector_tokens
        );
    }

    #[test]
    fn threshold_remote_gates_on_uncached_prefill_tokens() {
        let policy = SelectionPolicy::Threshold {
            min_remote_prefill_tokens: 256,
        };
        let cold = |total: usize| PolicyInputs {
            total_tokens: total,
            num_computed_tokens: 0,
            num_connector_tokens: 0,
        };
        // Cold prompt of 300 uncached tokens >= 256 -> disaggregate.
        assert_eq!(policy.select(&cold(300)), PrefillSelection::Remote);
        // Cold prompt of 200 uncached tokens < 256 -> local prefill.
        assert_eq!(policy.select(&cold(200)), PrefillSelection::Local);
        // Boundary: exactly 256 uncached -> Remote (>=).
        assert_eq!(policy.select(&cold(256)), PrefillSelection::Remote);
        // 300 total but mostly cached (100 computed + 60 matched) => 140
        // uncached < 256 -> local prefill, even though the prompt is long.
        let mostly_cached = PolicyInputs {
            total_tokens: 300,
            num_computed_tokens: 100,
            num_connector_tokens: 60,
        };
        assert_eq!(mostly_cached.num_prefill_tokens(), 140);
        assert_eq!(policy.select(&mostly_cached), PrefillSelection::Local);
        // Threshold 0 is equivalent to Always.
        let always = SelectionPolicy::Threshold {
            min_remote_prefill_tokens: 0,
        };
        assert_eq!(always.select(&cold(1)), PrefillSelection::Remote);
    }
}
