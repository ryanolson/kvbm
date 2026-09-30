// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::num::NonZeroU64;
use std::str::FromStr;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::{ManifestError, ResourceRequirement, ResourceRole};

/// Stable digest that namespaces otherwise-identical token lineage hashes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheManifestId([u8; 32]);

impl CacheManifestId {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for CacheManifestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "CacheManifestId({self})")
    }
}

impl fmt::Display for CacheManifestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for CacheManifestId {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != 64 {
            return Err("cache manifest id must contain exactly 64 hexadecimal characters".into());
        }
        if !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("cache manifest id must contain only hexadecimal characters".into());
        }
        let mut bytes = [0; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
                .map_err(|_| "cache manifest id must contain only hexadecimal characters")?;
        }
        Ok(Self(bytes))
    }
}

/// Model facts that prevent cache reuse across incompatible weights.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    architecture: String,
    revision: String,
    weights_digest: [u8; 32],
}

impl ModelIdentity {
    pub fn new(
        architecture: impl Into<String>,
        revision: impl Into<String>,
        weights_digest: [u8; 32],
    ) -> Result<Self, ManifestError> {
        let identity = Self {
            architecture: architecture.into(),
            revision: revision.into(),
            weights_digest,
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub const fn weights_digest(&self) -> &[u8; 32] {
        &self.weights_digest
    }

    pub(super) fn validate(&self) -> Result<(), ManifestError> {
        if self.architecture.is_empty() {
            return Err(ManifestError::EmptyModelField {
                field: "architecture",
            });
        }
        if self.revision.is_empty() {
            return Err(ManifestError::EmptyModelField { field: "revision" });
        }
        Ok(())
    }
}

/// Compact request-time identity derived from a validated cache manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheIdentity {
    manifest: CacheManifestId,
    resources: Arc<[ResourceRequirement]>,
    alignment_tokens: NonZeroU64,
}

impl CacheIdentity {
    pub(super) fn new(
        manifest: CacheManifestId,
        resources: Vec<ResourceRequirement>,
        alignment_tokens: NonZeroU64,
    ) -> Self {
        Self {
            manifest,
            resources: Arc::from(resources),
            alignment_tokens,
        }
    }

    pub const fn manifest(&self) -> CacheManifestId {
        self.manifest
    }

    pub fn resources(&self) -> &[ResourceRequirement] {
        &self.resources
    }

    pub const fn alignment_tokens(&self) -> NonZeroU64 {
        self.alignment_tokens
    }

    /// Deterministic fine-grained lineage anchor for bundle keys.
    ///
    /// Mixed-native histories are projected from the smallest native block
    /// size. Resource id breaks equal-size ties so physical registration order
    /// cannot change the canonical anchor.
    pub fn canonical_history(&self) -> Option<&ResourceRequirement> {
        self.resources
            .iter()
            .filter(|requirement| requirement.role() == ResourceRole::PrefixHistory)
            .min_by_key(|requirement| (requirement.native_block_tokens(), requirement.resource()))
    }
}

/// Explicit compatibility mode for one find request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CacheScope {
    /// Use the configured primary resource exactly as pre-manifest clients did.
    #[default]
    LegacyPrimary,
    /// Require the complete manifest-scoped resource bundle.
    Manifest(CacheIdentity),
}

impl CacheScope {
    pub const fn identity(&self) -> Option<&CacheIdentity> {
        match self {
            Self::LegacyPrimary => None,
            Self::Manifest(identity) => Some(identity),
        }
    }
}
