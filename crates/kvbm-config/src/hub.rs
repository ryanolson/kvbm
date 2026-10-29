// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector→hub configuration.
//!
//! [`LeaderHubConfig`] is the **sole** way the connector reaches a `kvbm-hub`.
//! When absent ([`KvbmConfig::hub`](crate::KvbmConfig::hub) is `None`), the
//! connector does normal hub-less work and **none** of the hub features
//! (`indexer`, `p2p`, `disagg`) are available.

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Connector-side hub configuration block (`leader.hub` in
/// `kv_connector_extra_config`).
///
/// # JSON example
/// ```json
/// { "url": "http://127.0.0.1:1337", "features": ["indexer"] }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LeaderHubConfig {
    /// Hub discovery base URL, e.g. `http://hub-host:1337`. The connector pulls
    /// `GET {url}/v1/ext/kvbm/config` at startup to learn the hub's enabled features and
    /// shared config.
    pub url: String,

    /// Subset of hub features this connector participates in. Values:
    /// `indexer`, `p2p`, `disagg`.
    ///
    /// - **Empty / omitted** → discover the hub's enabled set, intersect with
    ///   the connector's capabilities, **best-effort** (a feature absent on the
    ///   hub is simply skipped).
    /// - **Non-empty** → validated as a subset of the hub's enabled features
    ///   (with dependency closure); any unmet feature/dependency is a
    ///   **hard-fail** at startup.
    #[serde(default)]
    pub features: Vec<String>,

    /// The hub control port, where the connector registers. It must match
    /// the hub's `control_port`; the `url` port is the discovery port only.
    #[serde(default = "default_control_port")]
    pub control_port: u16,
}

/// The kvbm hub's default control port.
pub const DEFAULT_HUB_CONTROL_PORT: u16 = 8337;

fn default_control_port() -> u16 {
    DEFAULT_HUB_CONTROL_PORT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_with_features() {
        let json = r#"{"url": "http://127.0.0.1:1337", "features": ["indexer", "p2p"]}"#;
        let cfg: LeaderHubConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.url, "http://127.0.0.1:1337");
        assert_eq!(cfg.features, vec!["indexer", "p2p"]);
    }

    #[test]
    fn deserialize_url_only_defaults_to_empty_features() {
        let json = r#"{"url": "http://hub:1337"}"#;
        let cfg: LeaderHubConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.url, "http://hub:1337");
        assert!(cfg.features.is_empty());
        assert_eq!(cfg.control_port, DEFAULT_HUB_CONTROL_PORT);
    }

    #[test]
    fn deserialize_control_port() {
        let json = r#"{"url": "http://hub:18337", "control_port": 18338}"#;
        let cfg: LeaderHubConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.control_port, 18338);
    }
}
