// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-client builder shared by the leader bring-up paths.
//!
//! Constructs a [`kvbm_hub::HubClient`] from a hub base URL, applying the
//! discovery/control port defaults. Used both by the leader's hub-discovery
//! seed (peer-discovery backend) and the CD/P2P registration wiring.

use std::sync::Arc;

use anyhow::Result;
use kvbm_hub::{HubClient, HubClientBuilder};

pub fn build_hub_client(hub_url: &str) -> Result<Arc<HubClient>> {
    HubClientBuilder::from_url(hub_url)?.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_hub_client_accepts_explicit_port() {
        let client = build_hub_client("http://127.0.0.1:1337").unwrap();
        assert_eq!(client.config().discovery_url.port(), Some(1337));
    }

    #[test]
    fn build_hub_client_rejects_malformed_url() {
        assert!(build_hub_client("not a url").is_err());
    }

    #[test]
    fn build_hub_client_defaults_control_port() {
        let client = build_hub_client("http://127.0.0.1:1337").unwrap();
        assert_eq!(client.config().control_url.port(), Some(8337));
    }
}
