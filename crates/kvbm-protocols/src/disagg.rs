// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared protocol types for KVBM conditional disaggregation.
//!
//! This module intentionally contains only serializable control-plane data. It
//! is shared by the connector, hub, and future admission code without making
//! any one of those crates depend on another.

use std::collections::BTreeSet;

use crate::cache_manifest::{BundleKey, CacheManifestId, CacheScope};
use dynamo_tokens::{TokenBlockMmInfo, compute_hash_v2};
use kvbm_common::{LogicalResourceId, SequenceHash};
use serde::{Deserialize, Serialize};
use velo_ext::InstanceId;

/// xxh3_64 seed for `expected_hash_digest`. Distinct from
/// `CHAIN_XXH3_SEED` so an accidental collision with a PLH chain hash
/// can't sneak past the verifier.
pub const CD_DIGEST_SEED: u64 = 0xCD_CD_CD_CD_CD_CD_CD_CD;

/// Compute the defense-in-depth digest of a contiguous PLH slice.
///
/// Decode calls this over the `[0, DNPT/block_size)` slice of its slot's
/// PLH chain (everything decode commits to serve); prefill calls it over
/// the same slice of its own slot's locally-recomputed PLH chain. The
/// decode-side digest rides on the wire as
/// `RemotePrefillRequest.expected_hash_digest`; the prefill-side verifier
/// asserts equality before accepting the dispatch. Any mismatch indicates
/// hashing-input divergence (missed salt/LoRA propagation, hasher version
/// skew, etc.) — fail loud here instead of letting prefill publish
/// unrelated hashes and decode hang at the RDMA pull.
///
/// Stable across builds: xxh3_64 over the big-endian byte representation
/// of each PLH's inner `u128`, prefixed by the slice length so a longer
/// vs. shorter slice with a common prefix produces distinct digests.
pub fn digest_provided_hashes(hashes: &[SequenceHash]) -> u64 {
    let mut buf = Vec::with_capacity(8 + hashes.len() * 16);
    buf.extend_from_slice(&(hashes.len() as u64).to_be_bytes());
    for h in hashes {
        buf.extend_from_slice(&h.as_u128().to_be_bytes());
    }
    compute_hash_v2(&buf, CD_DIGEST_SEED)
}

/// Current disaggregation protocol version.
pub const DISAGG_PROTOCOL_VERSION: u16 = 1;

/// Unique identifier for a disaggregation session.
pub type SessionId = uuid::Uuid;

/// JSON-safe representation of a KVBM sequence hash.
///
/// Native KVBM hashes are currently backed by `u128`, which `serde_json` does
/// not support directly. The wire protocol carries the decimal representation.
pub type DisaggSequenceHash = String;

mod serde_uuid_string {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};
    use uuid::Uuid;

    pub fn serialize<S>(id: &Uuid, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&id.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Uuid, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Uuid::parse_str(&value).map_err(D::Error::custom)
    }
}

mod serde_instance_id_string {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};
    use uuid::Uuid;
    use velo_ext::InstanceId;

    pub fn serialize<S>(id: &InstanceId, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&id.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<InstanceId, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Uuid::parse_str(&value)
            .map(InstanceId::from)
            .map_err(D::Error::custom)
    }
}

/// Role advertised by a worker or instance to the hub.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerRole {
    Decode,
    Prefill,
    Hybrid,
}

/// Opaque endpoint descriptor for a session/control channel.
///
/// Later PRs can standardize concrete endpoint kinds such as velo-streaming
/// anchors without changing where protocol ownership lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionEndpoint {
    pub kind: String,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub payload: serde_json::Value,
}

/// Manifest-scoped input and target carried by a bundle prefill dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BundlePrefillContext {
    manifest: CacheManifestId,
    resources: BTreeSet<LogicalResourceId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    initial_bundle: Option<BundleKey>,
    target_bundle: BundleKey,
    estimated_bundle_bytes: u64,
}

#[derive(Deserialize)]
struct BundlePrefillContextWire {
    manifest: CacheManifestId,
    resources: BTreeSet<LogicalResourceId>,
    #[serde(default)]
    initial_bundle: Option<BundleKey>,
    target_bundle: BundleKey,
    estimated_bundle_bytes: u64,
}

impl<'de> Deserialize<'de> for BundlePrefillContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = BundlePrefillContextWire::deserialize(deserializer)?;
        Self::new(
            wire.manifest,
            wire.resources,
            wire.initial_bundle,
            wire.target_bundle,
            wire.estimated_bundle_bytes,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl BundlePrefillContext {
    pub fn new<I>(
        manifest: CacheManifestId,
        resources: I,
        initial_bundle: Option<BundleKey>,
        target_bundle: BundleKey,
        estimated_bundle_bytes: u64,
    ) -> Result<Self, BundlePrefillContextError>
    where
        I: IntoIterator<Item = LogicalResourceId>,
    {
        if target_bundle.manifest() != manifest
            || initial_bundle.is_some_and(|key| key.manifest() != manifest)
        {
            return Err(BundlePrefillContextError::ManifestMismatch);
        }
        if initial_bundle
            .is_some_and(|key| key.boundary_tokens() >= target_bundle.boundary_tokens())
        {
            return Err(BundlePrefillContextError::InvalidBoundaryOrder);
        }
        let resources = resources.into_iter().collect::<BTreeSet<_>>();
        if resources.is_empty() {
            return Err(BundlePrefillContextError::EmptyResources);
        }
        Ok(Self {
            manifest,
            resources,
            initial_bundle,
            target_bundle,
            estimated_bundle_bytes,
        })
    }

    pub const fn manifest(&self) -> CacheManifestId {
        self.manifest
    }

    pub fn resources(&self) -> impl Iterator<Item = LogicalResourceId> + '_ {
        self.resources.iter().copied()
    }

    pub const fn initial_bundle(&self) -> Option<BundleKey> {
        self.initial_bundle
    }

    pub const fn target_bundle(&self) -> BundleKey {
        self.target_bundle
    }

    pub const fn estimated_bundle_bytes(&self) -> u64 {
        self.estimated_bundle_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BundlePrefillContextError {
    #[error("bundle prefill keys do not match the declared manifest")]
    ManifestMismatch,
    #[error("bundle prefill target must be later than its initial bundle")]
    InvalidBoundaryOrder,
    #[error("bundle prefill context requires at least one logical resource")]
    EmptyResources,
}

/// Typed transfer parameters carried in request metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_prefill: Option<RemotePrefillParams>,
}

impl TransferParams {
    pub fn remote_prefill(params: RemotePrefillParams) -> Self {
        Self {
            remote_prefill: Some(params),
        }
    }
}

/// Wire-shape mirror of `kv_hashing::Request`'s hashing inputs.
///
/// `tokens` is carried separately on [`RemotePrefillRequest::token_ids`] (it is
/// also the vLLM dispatch payload). This envelope carries the remaining inputs
/// — `lora_name`, `salt`, `mm_info` — so the prefill side can rebuild the
/// canonical [`kv_hashing::Request`] (field-for-field) and run the same Rust
/// hasher locally instead of trusting decode-shipped `PositionalLineageHash`
/// values. The result: a single source of truth for hashing, no PLH bytes on
/// the wire, and forward-compatible coverage for salt / LoRA / multimodal.
///
/// **Current scope:** decode-side construction asserts each field is empty
/// (no salt, no LoRA, no multimodal yet wired through the CD path). The
/// fields exist so future enablement is a value change, not a wire-format
/// change.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHashingRequestEnvelope {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub salt: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mm_info: Vec<TokenBlockMmInfo>,
}

impl KvHashingRequestEnvelope {
    /// True iff no hashing-affecting auxiliary input is set.
    pub fn is_empty(&self) -> bool {
        self.lora_name.is_none() && self.salt.is_none() && self.mm_info.is_empty()
    }
}

/// Parameters identifying a remote prefill session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemotePrefillParams {
    pub protocol_version: u16,
    #[serde(with = "serde_uuid_string")]
    pub session_id: SessionId,
    #[serde(with = "serde_instance_id_string")]
    pub initiator_instance_id: InstanceId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_endpoint: Option<SessionEndpoint>,
    /// **DNPT — Decode's Num Provided Tokens.** Contiguous tokens decode
    /// commits to serve via the session, counted from absolute position 0.
    /// Folds in both the vLLM-decode G1 prefix AND the G2 local-match
    /// window decode resolved during this request. By construction
    /// `DNPT >= D_vllm_decode`; decode may over-commit by promoting
    /// G1→G2 or staging G3→G2 but never under-commits relative to what
    /// vLLM-decode declared computed.
    ///
    /// **Prefill's contract:** prefill provides all full-token blocks
    /// for `[DNPT/block_size, ..)` (everything decode doesn't have,
    /// prefill produces). Prefill independently computes PNCT (its
    /// own local cache view) and pulls hashes `[PNCT, DNPT)` from
    /// decode when `PNCT < DNPT`. If `PNCT >= DNPT`, prefill skips
    /// the prefix pull; a future enhancement is for prefill to
    /// publish back `[DNPT, PNCT)` to the same session so decode
    /// can pull and warm its own cache.
    #[serde(default)]
    pub num_provided_tokens: usize,
    /// Canonical hashing inputs decode used to build its `kv_hashing::Request`.
    /// Prefill rebuilds the same request from these fields + `token_ids` so
    /// the local PLH chain is bit-identical by construction.
    #[serde(default)]
    pub request: KvHashingRequestEnvelope,
    /// The salt hash of the request's reuse scope. Prefill hashes and
    /// registers its blocks with it. `None` is the global scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_salt: Option<u64>,
    /// xxh3_64 digest of decode's `[0, DNPT/block_size)` PLH slice.
    /// Prefill recomputes the digest from its own slot's PLH slice and
    /// asserts equality before accepting the dispatch. Defense-in-depth
    /// against silent hash divergence (a missed salt/LoRA propagation,
    /// hasher version skew, etc.); cost is one u64 on the wire. `None`
    /// skips the assertion (legacy tests and paths that don't compute it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_hash_digest: Option<u64>,
    /// Complete-bundle compute context. When present, the prefill connector
    /// uses normal manifest-scoped search/onboard and publishes the target
    /// through the atomic bundle lifecycle instead of the legacy unitary
    /// session pipeline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle: Option<BundlePrefillContext>,
}

impl RemotePrefillParams {
    pub fn new(session_id: SessionId, initiator_instance_id: InstanceId) -> Self {
        Self {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            session_id,
            initiator_instance_id,
            decode_endpoint: None,
            num_provided_tokens: 0,
            request: KvHashingRequestEnvelope::default(),
            scope_salt: None,
            expected_hash_digest: None,
            bundle: None,
        }
    }

    /// Validate request-independent wire metadata against the prompt window.
    ///
    /// Model-specific manifest, resource, alignment, and hash-chain checks
    /// remain the receiving worker's responsibility. This gate rejects wire
    /// shapes that are invalid for every worker before they enter scheduling.
    pub fn validate_for_prompt_len(
        &self,
        prompt_len: usize,
    ) -> Result<(), RemotePrefillValidationError> {
        if self.protocol_version != DISAGG_PROTOCOL_VERSION {
            return Err(RemotePrefillValidationError::UnsupportedProtocolVersion {
                actual: self.protocol_version,
                expected: DISAGG_PROTOCOL_VERSION,
            });
        }
        if prompt_len == 0 {
            return Err(RemotePrefillValidationError::EmptyPrompt);
        }
        if self.num_provided_tokens > prompt_len {
            return Err(RemotePrefillValidationError::ProvidedWindowOutOfBounds {
                provided_tokens: self.num_provided_tokens,
                prompt_tokens: prompt_len,
            });
        }
        if !self.request.is_empty() {
            return Err(RemotePrefillValidationError::UnsupportedHashingInputs);
        }
        let Some(bundle) = &self.bundle else {
            return Ok(());
        };
        bundle.validate_wire_shape()?;
        let target_tokens = bundle.target_bundle().boundary_tokens();
        let prompt_tokens = u64::try_from(prompt_len).unwrap_or(u64::MAX);
        if target_tokens >= prompt_tokens {
            return Err(RemotePrefillValidationError::BundleTargetOutOfBounds {
                target_tokens,
                prompt_tokens: prompt_len,
            });
        }
        if u64::try_from(self.num_provided_tokens).unwrap_or(u64::MAX) > target_tokens {
            return Err(
                RemotePrefillValidationError::ProvidedWindowPastBundleTarget {
                    provided_tokens: self.num_provided_tokens,
                    target_tokens,
                },
            );
        }
        Ok(())
    }

    /// Validate the complete side-effect-free worker contract.
    ///
    /// Callers must run this before creating a search, prefill lifecycle, or
    /// destination allocation. It binds wire metadata to the receiving cache
    /// scope and the worker-recomputed sequence hash chain.
    pub fn validate_for_worker(
        &self,
        request_id: &str,
        cache: &CacheScope,
        sequence_hashes: &[SequenceHash],
        total_tokens: usize,
        block_size: usize,
    ) -> Result<(), RemotePrefillValidationError> {
        if request_id.trim().is_empty() {
            return Err(RemotePrefillValidationError::EmptyRequestId);
        }
        self.validate_for_prompt_len(total_tokens)?;
        if self.num_provided_tokens >= total_tokens {
            return Err(RemotePrefillValidationError::ProvidedWindowConsumesPrompt {
                provided_tokens: self.num_provided_tokens,
                prompt_tokens: total_tokens,
            });
        }
        if block_size == 0 {
            return Err(RemotePrefillValidationError::ZeroWorkerBlockSize);
        }
        if !self.num_provided_tokens.is_multiple_of(block_size) {
            return Err(RemotePrefillValidationError::UnalignedProvidedWindow {
                provided_tokens: self.num_provided_tokens,
                block_tokens: block_size,
            });
        }
        let provided_blocks = self.num_provided_tokens / block_size;
        if provided_blocks > sequence_hashes.len() {
            return Err(
                RemotePrefillValidationError::ProvidedHashWindowOutOfBounds {
                    provided_blocks,
                    available_hashes: sequence_hashes.len(),
                },
            );
        }
        if let Some(expected) = self.expected_hash_digest {
            let actual = digest_provided_hashes(&sequence_hashes[..provided_blocks]);
            if actual != expected {
                return Err(RemotePrefillValidationError::HashDigestMismatch { expected, actual });
            }
        }

        match (cache, self.bundle.as_ref()) {
            (CacheScope::Manifest(_), None) => {
                Err(RemotePrefillValidationError::ManifestRequestMissingBundle)
            }
            (CacheScope::LegacyPrimary, Some(_)) => {
                Err(RemotePrefillValidationError::BundleRequestRequiresManifest)
            }
            (CacheScope::LegacyPrimary, None) => {
                if self.decode_endpoint.is_none() {
                    return Err(RemotePrefillValidationError::MissingDecodeEndpoint);
                }
                Ok(())
            }
            (CacheScope::Manifest(identity), Some(bundle)) => {
                if self.num_provided_tokens != 0 {
                    return Err(RemotePrefillValidationError::BundleProvidedWindowNotEmpty {
                        provided_tokens: self.num_provided_tokens,
                    });
                }
                bundle.validate_for_worker(identity, sequence_hashes, total_tokens, block_size)
            }
        }
    }
}

impl BundlePrefillContext {
    fn validate_wire_shape(&self) -> Result<(), RemotePrefillValidationError> {
        if self.resources.is_empty() {
            return Err(RemotePrefillValidationError::EmptyBundleResources);
        }
        if self.target_bundle.manifest() != self.manifest
            || self
                .initial_bundle
                .is_some_and(|initial| initial.manifest() != self.manifest)
        {
            return Err(RemotePrefillValidationError::BundleManifestMismatch);
        }
        if self.initial_bundle.is_some_and(|initial| {
            initial.boundary_tokens() >= self.target_bundle.boundary_tokens()
        }) {
            return Err(RemotePrefillValidationError::InvalidBundleBoundaryOrder);
        }
        Ok(())
    }

    fn validate_for_worker(
        &self,
        identity: &crate::cache_manifest::CacheIdentity,
        sequence_hashes: &[SequenceHash],
        total_tokens: usize,
        block_size: usize,
    ) -> Result<(), RemotePrefillValidationError> {
        if self.manifest != identity.manifest() {
            return Err(RemotePrefillValidationError::BundleManifestMismatch);
        }
        let expected_resources = identity
            .resources()
            .iter()
            .map(|requirement| requirement.resource())
            .collect::<BTreeSet<_>>();
        if self.resources != expected_resources {
            return Err(RemotePrefillValidationError::BundleResourceSetMismatch);
        }
        if let Some(initial) = self.initial_bundle {
            validate_bundle_key(
                "initial",
                initial,
                identity,
                sequence_hashes,
                total_tokens,
                block_size,
            )?;
        }
        validate_bundle_key(
            "target",
            self.target_bundle,
            identity,
            sequence_hashes,
            total_tokens,
            block_size,
        )
    }
}

fn validate_bundle_key(
    role: &'static str,
    key: BundleKey,
    identity: &crate::cache_manifest::CacheIdentity,
    sequence_hashes: &[SequenceHash],
    total_tokens: usize,
    block_size: usize,
) -> Result<(), RemotePrefillValidationError> {
    if !key.is_compatible_with(identity) {
        return Err(RemotePrefillValidationError::BundleKeyIncompatible { role });
    }
    let boundary_tokens = usize::try_from(key.boundary_tokens()).map_err(|_| {
        RemotePrefillValidationError::BundleBoundaryOutOfBounds {
            role,
            boundary_tokens: key.boundary_tokens(),
        }
    })?;
    let eligible_blocks = total_tokens.saturating_sub(1) / block_size;
    let index = boundary_tokens
        .checked_div(block_size)
        .filter(|_| boundary_tokens.is_multiple_of(block_size))
        .and_then(|blocks| blocks.checked_sub(1))
        .filter(|index| *index < eligible_blocks.min(sequence_hashes.len()))
        .ok_or(RemotePrefillValidationError::BundleBoundaryOutOfBounds {
            role,
            boundary_tokens: key.boundary_tokens(),
        })?;
    if sequence_hashes[index] != key.boundary_hash() {
        return Err(RemotePrefillValidationError::BundleBoundaryHashMismatch {
            role,
            boundary_tokens: key.boundary_tokens(),
        });
    }
    Ok(())
}

/// Worker-independent validation failures for remote-prefill wire metadata.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RemotePrefillValidationError {
    #[error("remote-prefill request id is empty")]
    EmptyRequestId,
    #[error("remote-prefill prompt is empty")]
    EmptyPrompt,
    #[error(
        "remote-prefill protocol version {actual} is unsupported; this worker requires {expected}"
    )]
    UnsupportedProtocolVersion { actual: u16, expected: u16 },
    #[error(
        "remote-prefill provided window {provided_tokens} exceeds prompt length {prompt_tokens}"
    )]
    ProvidedWindowOutOfBounds {
        provided_tokens: usize,
        prompt_tokens: usize,
    },
    #[error("remote-prefill hashing inputs are not supported by this protocol revision")]
    UnsupportedHashingInputs,
    #[error("remote-prefill bundle context requires at least one logical resource")]
    EmptyBundleResources,
    #[error("remote-prefill bundle keys do not match the declared manifest")]
    BundleManifestMismatch,
    #[error("remote-prefill initial bundle must precede its target")]
    InvalidBundleBoundaryOrder,
    #[error(
        "remote-prefill bundle target {target_tokens} must leave a token inside prompt length {prompt_tokens}"
    )]
    BundleTargetOutOfBounds {
        target_tokens: u64,
        prompt_tokens: usize,
    },
    #[error(
        "remote-prefill provided window {provided_tokens} extends past bundle target {target_tokens}"
    )]
    ProvidedWindowPastBundleTarget {
        provided_tokens: usize,
        target_tokens: u64,
    },
    #[error(
        "remote-prefill provided window {provided_tokens} must leave one prompt token out of {prompt_tokens}"
    )]
    ProvidedWindowConsumesPrompt {
        provided_tokens: usize,
        prompt_tokens: usize,
    },
    #[error("remote-prefill worker block size is zero")]
    ZeroWorkerBlockSize,
    #[error(
        "remote-prefill provided window {provided_tokens} is not aligned to worker block size {block_tokens}"
    )]
    UnalignedProvidedWindow {
        provided_tokens: usize,
        block_tokens: usize,
    },
    #[error(
        "remote-prefill provided window requires {provided_blocks} hashes but only {available_hashes} are available"
    )]
    ProvidedHashWindowOutOfBounds {
        provided_blocks: usize,
        available_hashes: usize,
    },
    #[error("remote-prefill hash digest mismatch: expected 0x{expected:016x}, got 0x{actual:016x}")]
    HashDigestMismatch { expected: u64, actual: u64 },
    #[error("manifest-scoped remote-prefill request requires bundle metadata")]
    ManifestRequestMissingBundle,
    #[error("bundle remote-prefill metadata requires a manifest-scoped cache")]
    BundleRequestRequiresManifest,
    #[error("unitary remote-prefill request is missing its decode endpoint")]
    MissingDecodeEndpoint,
    #[error("bundle remote-prefill provided window must be zero, got {provided_tokens}")]
    BundleProvidedWindowNotEmpty { provided_tokens: usize },
    #[error("remote-prefill bundle resources do not match the worker cache manifest")]
    BundleResourceSetMismatch,
    #[error("remote-prefill bundle {role} key is incompatible with the worker cache manifest")]
    BundleKeyIncompatible { role: &'static str },
    #[error("remote-prefill bundle {role} boundary {boundary_tokens} is outside the request chain")]
    BundleBoundaryOutOfBounds {
        role: &'static str,
        boundary_tokens: u64,
    },
    #[error(
        "remote-prefill bundle {role} boundary {boundary_tokens} hash does not match the request chain"
    )]
    BundleBoundaryHashMismatch {
        role: &'static str,
        boundary_tokens: u64,
    },
}

/// Router-owned circuit-breaker tier for CD prefill-overload control.
///
/// The hub's prefill router computes this tier (a hysteresis state machine
/// driven by a dedicated breaker-tick task; see `kvbm-hub`) and — in P2 —
/// pushes it to decode workers, which cache it and read it synchronously
/// inside GNMT. It only ever NARROWS disaggregation (more Local, never more
/// Remote):
/// - `Calm` (closed) — existing `min_remote_prefill_tokens` threshold policy.
/// - `Warm` (half-open) — proportional middle (per-request admission; P3).
/// - `Hot` (open) — coarsely downgrade all would-be-Remote to Local.
///
/// Default `Calm` ⇒ identical to pre-breaker behavior, so a decode that has
/// never received a tier push (or a build with the breaker disabled) behaves
/// exactly as today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum BreakerTier {
    /// Closed: no overload; existing threshold policy applies.
    #[default]
    Calm,
    /// Half-open: elevated pressure; proportional admission (P3 WARM tier).
    Warm,
    /// Open: saturated; downgrade all would-be-Remote to Local.
    Hot,
}

/// Payload enqueued by a decode worker and consumed by a prefill worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemotePrefillRequest {
    pub protocol_version: u16,
    pub request_id: String,
    #[serde(with = "serde_uuid_string")]
    pub session_id: SessionId,
    #[serde(with = "serde_instance_id_string")]
    pub initiator_instance_id: InstanceId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_endpoint: Option<SessionEndpoint>,
    /// Full prompt token IDs (NOT a windowed suffix). Prefill builds its
    /// slot from these so its PLH chain matches decode's at absolute
    /// positions — required for cross-instance pull keys to align.
    pub token_ids: Vec<u32>,
    /// See [`RemotePrefillParams::num_provided_tokens`].
    #[serde(default)]
    pub num_provided_tokens: usize,
    #[serde(default)]
    pub request: KvHashingRequestEnvelope,
    /// The salt hash of the request's reuse scope. Prefill hashes and
    /// registers its blocks with it. `None` is the global scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_salt: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_hash_digest: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle: Option<BundlePrefillContext>,
}

impl RemotePrefillRequest {
    /// Validate worker-independent wire invariants before scheduler admission.
    pub fn validate(&self) -> Result<(), RemotePrefillValidationError> {
        if self.request_id.trim().is_empty() {
            return Err(RemotePrefillValidationError::EmptyRequestId);
        }
        self.remote_prefill_params()
            .validate_for_prompt_len(self.token_ids.len())
    }

    pub fn remote_prefill_params(&self) -> RemotePrefillParams {
        RemotePrefillParams {
            protocol_version: self.protocol_version,
            session_id: self.session_id,
            initiator_instance_id: self.initiator_instance_id,
            decode_endpoint: self.decode_endpoint.clone(),
            num_provided_tokens: self.num_provided_tokens,
            request: self.request.clone(),
            scope_salt: self.scope_salt,
            expected_hash_digest: self.expected_hash_digest,
            bundle: self.bundle.clone(),
        }
    }

    pub fn transfer_params(&self) -> TransferParams {
        TransferParams::remote_prefill(self.remote_prefill_params())
    }
}

/// Hub-issued lifecycle/control signal for decode or prefill workers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlSignal {
    Pause {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    Resume,
    Drain,
    ShutdownGracefully,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_manifest::{
        BundleKey, CacheManifest, CacheManifestId, ModelIdentity, ResourceRequirement, ResourceRole,
    };
    use crate::connector::CacheScope;

    fn instance_id() -> InstanceId {
        uuid::Uuid::new_v4().into()
    }

    #[test]
    fn bundle_prefill_context_round_trips_with_manifest_scoped_keys() {
        let manifest = CacheManifestId::from_bytes([7; 32]);
        let initial = BundleKey::from_parts(manifest, fake_plh(1, 1), 8).unwrap();
        let target = BundleKey::from_parts(manifest, fake_plh(2, 3), 16).unwrap();
        let resources = [
            kvbm_common::LogicalResourceId(3),
            kvbm_common::LogicalResourceId(5),
        ];
        let context =
            BundlePrefillContext::new(manifest, resources, Some(initial), target, 65_536).unwrap();

        let encoded = serde_json::to_vec(&context).unwrap();
        let decoded: BundlePrefillContext = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded, context);
        assert_eq!(decoded.initial_bundle(), Some(initial));
        assert_eq!(decoded.target_bundle(), target);
        assert_eq!(decoded.resources().collect::<Vec<_>>(), resources);
        assert_eq!(decoded.estimated_bundle_bytes(), 65_536);
    }

    #[test]
    fn bundle_prefill_context_deserialization_reapplies_constructor_invariants() {
        let manifest = CacheManifestId::from_bytes([7; 32]);
        let initial = BundleKey::from_parts(manifest, fake_plh(1, 1), 8).unwrap();
        let target = BundleKey::from_parts(manifest, fake_plh(2, 3), 16).unwrap();
        let context = BundlePrefillContext::new(
            manifest,
            [kvbm_common::LogicalResourceId(3)],
            Some(initial),
            target,
            1,
        )
        .unwrap();
        let mut empty_resources = serde_json::to_value(&context).unwrap();
        empty_resources["resources"] = serde_json::json!([]);
        assert!(serde_json::from_value::<BundlePrefillContext>(empty_resources).is_err());

        let mut invalid_order = serde_json::to_value(&context).unwrap();
        invalid_order["initial_bundle"] = invalid_order["target_bundle"].clone();
        assert!(serde_json::from_value::<BundlePrefillContext>(invalid_order).is_err());
    }

    #[test]
    fn remote_prefill_request_builds_transfer_params() {
        let request = RemotePrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: "req-1".to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: instance_id(),
            decode_endpoint: Some(SessionEndpoint {
                kind: "test".to_string(),
                payload: serde_json::json!({"anchor": "a"}),
            }),
            token_ids: vec![1, 2, 3],
            num_provided_tokens: 48,
            request: KvHashingRequestEnvelope::default(),
            scope_salt: Some(7),
            expected_hash_digest: Some(0xDEADBEEF),
            bundle: None,
        };

        let params = request
            .transfer_params()
            .remote_prefill
            .expect("remote params populated");
        assert_eq!(params.protocol_version, request.protocol_version);
        assert_eq!(params.session_id, request.session_id);
        assert_eq!(params.initiator_instance_id, request.initiator_instance_id);
        assert_eq!(params.decode_endpoint, request.decode_endpoint);
        assert_eq!(params.num_provided_tokens, request.num_provided_tokens);
        assert_eq!(params.request, request.request);
        assert_eq!(params.scope_salt, request.scope_salt);
        assert_eq!(params.expected_hash_digest, request.expected_hash_digest);
    }

    fn valid_remote_prefill_request() -> RemotePrefillRequest {
        RemotePrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: "req-valid".to_owned(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: instance_id(),
            decode_endpoint: None,
            token_ids: vec![1, 2, 3, 4, 5],
            num_provided_tokens: 4,
            request: KvHashingRequestEnvelope::default(),
            scope_salt: None,
            expected_hash_digest: None,
            bundle: None,
        }
    }

    fn worker_identity() -> crate::cache_manifest::CacheIdentity {
        CacheManifest::new(
            ModelIdentity::new("dsv4", "revision", [3; 32]).unwrap(),
            "worker-validation",
            vec![
                ResourceRequirement::new(
                    kvbm_common::LogicalResourceId(3),
                    ResourceRole::PrefixHistory,
                    4,
                )
                .unwrap(),
            ],
            std::collections::BTreeMap::new(),
        )
        .unwrap()
        .identity()
    }

    fn unitary_endpoint() -> SessionEndpoint {
        SessionEndpoint {
            kind: "test".to_owned(),
            payload: serde_json::Value::Null,
        }
    }

    #[test]
    fn worker_validation_requires_bundle_metadata_for_manifest_scope() {
        let mut params = RemotePrefillParams::new(uuid::Uuid::new_v4(), instance_id());
        params.decode_endpoint = Some(unitary_endpoint());
        let hashes = [fake_plh(1, 0), fake_plh(2, 1)];

        assert!(matches!(
            params.validate_for_worker(
                "manifest-without-bundle",
                &CacheScope::Manifest(worker_identity()),
                &hashes,
                9,
                4,
            ),
            Err(RemotePrefillValidationError::ManifestRequestMissingBundle)
        ));
    }

    #[test]
    fn worker_validation_rejects_a_unitary_window_covering_the_full_prompt() {
        let mut params = RemotePrefillParams::new(uuid::Uuid::new_v4(), instance_id());
        params.decode_endpoint = Some(unitary_endpoint());
        params.num_provided_tokens = 8;
        let hashes = [fake_plh(1, 0), fake_plh(2, 1)];

        assert!(matches!(
            params.validate_for_worker("full-window", &CacheScope::LegacyPrimary, &hashes, 8, 4,),
            Err(RemotePrefillValidationError::ProvidedWindowConsumesPrompt { .. })
        ));
    }

    #[test]
    fn worker_validation_rejects_a_stale_bundle_hash() {
        let identity = worker_identity();
        let hashes = [fake_plh(1, 0), fake_plh(2, 1)];
        let target = BundleKey::new(&identity, fake_plh(99, 1), 8).unwrap();
        let bundle = BundlePrefillContext::new(
            identity.manifest(),
            [kvbm_common::LogicalResourceId(3)],
            None,
            target,
            1,
        )
        .unwrap();
        let mut params = RemotePrefillParams::new(uuid::Uuid::new_v4(), instance_id());
        params.bundle = Some(bundle);

        assert!(matches!(
            params.validate_for_worker(
                "stale-target",
                &CacheScope::Manifest(identity),
                &hashes,
                9,
                4,
            ),
            Err(RemotePrefillValidationError::BundleBoundaryHashMismatch { .. })
        ));
    }

    #[test]
    fn remote_prefill_wire_validation_rejects_unknown_protocol_version() {
        let mut request = valid_remote_prefill_request();
        request.protocol_version = DISAGG_PROTOCOL_VERSION + 1;

        assert!(matches!(
            request.validate(),
            Err(RemotePrefillValidationError::UnsupportedProtocolVersion { .. })
        ));
    }

    #[test]
    fn remote_prefill_wire_validation_rejects_empty_identity_and_prompt() {
        let mut request = valid_remote_prefill_request();
        request.request_id.clear();
        assert!(matches!(
            request.validate(),
            Err(RemotePrefillValidationError::EmptyRequestId)
        ));

        request.request_id = "req-empty-prompt".to_owned();
        request.token_ids.clear();
        assert!(matches!(
            request.validate(),
            Err(RemotePrefillValidationError::EmptyPrompt)
        ));
    }

    #[test]
    fn remote_prefill_wire_validation_rejects_provided_window_beyond_prompt() {
        let mut request = valid_remote_prefill_request();
        request.num_provided_tokens = request.token_ids.len() + 1;

        assert!(matches!(
            request.validate(),
            Err(RemotePrefillValidationError::ProvidedWindowOutOfBounds { .. })
        ));
    }

    #[test]
    fn remote_prefill_wire_validation_requires_a_token_after_bundle_target() {
        let manifest = CacheManifestId::from_bytes([9; 32]);
        let target = BundleKey::from_parts(manifest, fake_plh(4, 3), 5).unwrap();
        let mut request = valid_remote_prefill_request();
        request.bundle = Some(
            BundlePrefillContext::new(
                manifest,
                [kvbm_common::LogicalResourceId(3)],
                None,
                target,
                1,
            )
            .unwrap(),
        );

        assert!(matches!(
            request.validate(),
            Err(RemotePrefillValidationError::BundleTargetOutOfBounds { .. })
        ));
    }

    #[test]
    fn transfer_params_round_trips_json() {
        let params = TransferParams::remote_prefill(RemotePrefillParams {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: instance_id(),
            decode_endpoint: None,
            num_provided_tokens: 48,
            request: KvHashingRequestEnvelope::default(),
            scope_salt: None,
            expected_hash_digest: Some(0x1234_5678_9ABC_DEF0),
            bundle: None,
        });

        let encoded = serde_json::to_vec(&params).unwrap();
        let decoded: TransferParams = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded, params);
    }

    // Defense-in-depth digest reproducer-first tests. The digest helper
    // did not exist before this refactor — these would fail to compile
    // against pre-fix HEAD, which is the strongest form of "reproducer
    // catches the regression" since the verifier itself is what closes
    // the silent-hash-divergence loop.
    fn fake_plh(seed: u64, pos: u64) -> SequenceHash {
        SequenceHash::new(seed, None, pos)
    }

    #[test]
    fn digest_is_deterministic() {
        let hashes = vec![fake_plh(1, 0), fake_plh(2, 1), fake_plh(3, 2)];
        let a = digest_provided_hashes(&hashes);
        let b = digest_provided_hashes(&hashes);
        assert_eq!(a, b, "digest must be deterministic for identical input");
    }

    #[test]
    fn digest_distinguishes_different_slices() {
        let a = digest_provided_hashes(&[fake_plh(1, 0), fake_plh(2, 1)]);
        let b = digest_provided_hashes(&[fake_plh(1, 0), fake_plh(99, 1)]);
        assert_ne!(a, b, "differing PLHs must produce distinct digests");
    }

    #[test]
    fn digest_distinguishes_length_prefix() {
        // The length prefix in the digest input guarantees that a
        // 2-element slice and a 3-element slice sharing the same prefix
        // produce distinct digests — guards against an off-by-one slicing
        // bug on either side silently passing the verifier.
        let two = vec![fake_plh(1, 0), fake_plh(2, 1)];
        let three = vec![fake_plh(1, 0), fake_plh(2, 1), fake_plh(3, 2)];
        assert_ne!(digest_provided_hashes(&two), digest_provided_hashes(&three));
    }

    #[test]
    fn digest_empty_slice_is_stable() {
        let a = digest_provided_hashes(&[]);
        let b = digest_provided_hashes(&[]);
        assert_eq!(a, b);
        // And distinct from any single-element digest.
        assert_ne!(a, digest_provided_hashes(&[fake_plh(0, 0)]));
    }

    #[test]
    fn envelope_round_trips_with_mm_info() {
        let env = KvHashingRequestEnvelope {
            lora_name: Some("adapter-x".into()),
            salt: Some("model-tag".into()),
            mm_info: vec![TokenBlockMmInfo {
                mm_hash: 0xCAFE,
                offset: 4,
                length: 8,
            }],
        };
        let encoded = serde_json::to_vec(&env).unwrap();
        let decoded: KvHashingRequestEnvelope = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, env);
        assert!(!env.is_empty());
        assert!(KvHashingRequestEnvelope::default().is_empty());
    }

    #[test]
    fn breaker_tier_serde_and_default() {
        // Default is Calm (== prior behavior for a decode that never got a push).
        assert_eq!(BreakerTier::default(), BreakerTier::Calm);
        // snake_case wire form.
        assert_eq!(
            serde_json::to_string(&BreakerTier::Calm).unwrap(),
            r#""calm""#
        );
        assert_eq!(
            serde_json::to_string(&BreakerTier::Warm).unwrap(),
            r#""warm""#
        );
        assert_eq!(
            serde_json::to_string(&BreakerTier::Hot).unwrap(),
            r#""hot""#
        );
        for t in [BreakerTier::Calm, BreakerTier::Warm, BreakerTier::Hot] {
            let s = serde_json::to_string(&t).unwrap();
            let back: BreakerTier = serde_json::from_str(&s).unwrap();
            assert_eq!(back, t);
        }
        // Ordering reflects increasing pressure: Calm < Warm < Hot.
        assert!(BreakerTier::Calm < BreakerTier::Warm);
        assert!(BreakerTier::Warm < BreakerTier::Hot);
    }

    #[test]
    fn control_signal_round_trips_json() {
        let signal = ControlSignal::Pause {
            reason: Some("operator".to_string()),
        };

        let encoded = serde_json::to_string(&signal).unwrap();
        let decoded: ControlSignal = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, signal);
    }
}
