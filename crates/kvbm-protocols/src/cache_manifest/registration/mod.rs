// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registration-lifecycle identity shared by directory and transfer protocols.

use serde::{Deserialize, Serialize};

/// Opaque identity for one successful owner registration lifecycle.
///
/// The hub mints a fresh random value for every registration, including a
/// same-`InstanceId` replacement. It is a public binding, not an authority:
/// the hub registration credential still authenticates directory writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RegistrationEpoch(uuid::Uuid);

impl RegistrationEpoch {
    /// Rebuild an epoch from its UUID, e.g. the hub lease's epoch.
    #[must_use]
    pub const fn from_uuid(value: uuid::Uuid) -> Self {
        Self(value)
    }

    /// Mint a fresh registration identity with UUID-v4 entropy.
    #[must_use]
    #[allow(clippy::new_without_default)] // random identity; a `Default` would hide a lifecycle mint.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

#[cfg(test)]
mod tests {
    use super::RegistrationEpoch;

    #[test]
    fn minted_epochs_are_distinct_and_serde_round_trip() {
        let first = RegistrationEpoch::new();
        let second = RegistrationEpoch::new();
        assert_ne!(first, second);

        let json = serde_json::to_string(&first).unwrap();
        assert_eq!(
            serde_json::from_str::<RegistrationEpoch>(&json).unwrap(),
            first
        );
    }
}
