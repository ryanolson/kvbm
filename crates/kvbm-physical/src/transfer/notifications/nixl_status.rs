// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL status polling-based completion checker.

use anyhow::{Result, anyhow};
use kvbm_memory::nixl::{Agent as NixlAgent, XferRequest};

use super::CompletionChecker;

/// Completion checker that polls NIXL transfer status.
pub struct NixlStatusChecker {
    agent: NixlAgent,
    xfer_req: XferRequest,
    _registrations: crate::transfer::executor::registration::RegistrationGuards,
}

impl NixlStatusChecker {
    pub fn new(
        agent: NixlAgent,
        xfer_req: XferRequest,
        registrations: crate::transfer::executor::registration::RegistrationGuards,
    ) -> Self {
        Self {
            agent,
            xfer_req,
            _registrations: registrations,
        }
    }
}

impl CompletionChecker for NixlStatusChecker {
    fn is_complete(&self) -> Result<bool> {
        // get_xfer_status returns XferStatus enum:
        // - XferStatus::Success means transfer is complete
        // - XferStatus::InProgress means still pending
        match self.agent.get_xfer_status(&self.xfer_req) {
            Ok(status) => Ok(status.is_success()),
            Err(e) => Err(anyhow!("NIXL transfer status check failed: {}", e)),
        }
    }
}

impl Drop for NixlStatusChecker {
    fn drop(&mut self) {
        if !self._registrations.is_empty() && !matches!(self.is_complete(), Ok(true)) {
            // A failed status query or stopped worker does not prove DMA completion.
            std::mem::forget(std::mem::take(&mut self._registrations));
        }
    }
}
