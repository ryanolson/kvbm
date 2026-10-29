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

/// `hub_url` names the discovery port; `control_port` is where the client
/// registers.
pub fn build_hub_client(hub_url: &str, control_port: u16) -> Result<Arc<HubClient>> {
    HubClientBuilder::from_url(hub_url)?
        .control_port(control_port)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_hub_client_accepts_explicit_port() {
        let client = build_hub_client("http://127.0.0.1:1337", 8337).unwrap();
        assert_eq!(client.public_url().port(), Some(1337));
    }

    #[test]
    fn build_hub_client_rejects_malformed_url() {
        assert!(build_hub_client("not a url", 8337).is_err());
    }

    #[test]
    fn build_hub_client_uses_the_configured_control_port() {
        let client = build_hub_client("http://127.0.0.1:18337", 18338).unwrap();
        assert_eq!(client.public_url().port(), Some(18337));
        assert_eq!(client.control_url().port(), Some(18338));
    }
}
