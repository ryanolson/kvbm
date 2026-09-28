// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL agent wrapper and configuration.
//!
//! This module provides:
//! - `NixlAgent`: Wrapper around nixl_sys::Agent that tracks initialized backends
//! - `NixlBackendConfig`: Configuration for NIXL backends from environment variables

use anyhow::{Context, Result, ensure};
use nixl_sys::{Agent, MemType, RegDescList, is_stub};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

use crate::nixl::{HostAgentMetadata, NixlBackendConfig, NixlDescriptor};

/// Environment variable naming an explicit `libnixl_capi.so` to preload
/// before the first `nixl_sys::Agent::new` call. See [`NixlAgent::new`].
pub const NIXL_CAPI_LIB_ENV: &str = "KVBM_NIXL_CAPI_LIB";

/// Holds the dlopen'd NIXL C API library for the process lifetime.
///
/// `nixl_sys` caches its first `dlopen("libnixl_capi.so")` result --
/// including failure -- so whichever library the dynamic loader finds first
/// wins for the process lifetime. Preloading a specific path before the
/// first `nixl_sys` call pins which runtime this process uses.
static NIXL_CAPI: OnceLock<libloading::os::unix::Library> = OnceLock::new();

/// A NIXL agent wrapper that tracks which backends were successfully initialized.
///
/// This wrapper provides:
/// - Runtime validation of backend availability
/// - Clear error messages when operations need unavailable backends
/// - Single source of truth for backend state in tests and production
///
/// # Backend Tracking
///
/// Since `nixl_sys::Agent` doesn't provide a method to query active backends,
/// we track them during initialization. The `available_backends` set is populated
/// based on successful `create_backend()` calls.
#[derive(Clone, Debug)]
pub struct NixlAgent {
    agent: Agent,
    available_backends: HashSet<String>,
}

impl NixlAgent {
    /// dlopen the given `libnixl_capi.so` (`RTLD_NOW | RTLD_GLOBAL`) before
    /// the first `nixl_sys` call, pinning which NIXL runtime this process
    /// uses.
    ///
    /// Idempotent: only the first successful call has any effect; later calls
    /// (with any path) are no-ops that return `Ok(())`. This must run before
    /// any `nixl_sys::Agent` is constructed -- once `nixl_sys` has resolved
    /// its symbols, preloading a different library has no effect.
    pub fn preload_runtime(path: &Path) -> Result<()> {
        if NIXL_CAPI.get().is_some() {
            return Ok(());
        }
        // SAFETY: this loads a NIXL C API shared object and retains the
        // handle for the process lifetime, matching the packaged-runtime guard
        // that a host application uses for the same purpose.
        let library = unsafe {
            libloading::os::unix::Library::open(Some(path), libc::RTLD_NOW | libc::RTLD_GLOBAL)
        }
        .with_context(|| format!("load NIXL runtime {}", path.display()))?;
        // Another thread may have won the race; either way the slot is now
        // occupied and a library is loaded, so ignore a losing `set`.
        let _ = NIXL_CAPI.set(library);
        Ok(())
    }

    /// Opt-in preload hook driven by [`NIXL_CAPI_LIB_ENV`]. No-op (returns
    /// `Ok(())`) when the variable is unset, so existing callers of
    /// [`NixlAgent::new`] see zero behavior change.
    fn preload_runtime_from_env() -> Result<()> {
        match std::env::var_os(NIXL_CAPI_LIB_ENV) {
            Some(path) => Self::preload_runtime(Path::new(&path)),
            None => Ok(()),
        }
    }

    /// Create a NIXL agent without any backends.
    ///
    /// Honors [`NIXL_CAPI_LIB_ENV`] (`KVBM_NIXL_CAPI_LIB`): if set, its path
    /// is dlopen'd (`RTLD_NOW | RTLD_GLOBAL`) before the first `nixl_sys`
    /// call, pinning which NIXL runtime this process resolves to. No-op when
    /// unset -- source-compatible with existing callers.
    ///
    /// To pin a packaged NIXL runtime explicitly rather than through the
    /// `KVBM_NIXL_CAPI_LIB` environment variable, call
    /// [`NixlAgent::preload_runtime`] before this constructor.
    pub fn new(name: &str) -> Result<Self> {
        Self::preload_runtime_from_env()?;
        Self::new_uncached(name)
    }

    fn new_uncached(name: &str) -> Result<Self> {
        if is_stub() {
            return Err(anyhow::anyhow!("NIXL is not supported in stub mode"));
        }
        let agent = Agent::new(name)?;

        Ok(Self {
            agent,
            available_backends: HashSet::new(),
        })
    }

    /// Creates a new agent configured with backends from the given config.
    ///
    /// This method iterates over all backends in the config and initializes them
    /// with their associated parameters. If a backend has custom parameters defined
    /// in the config, those are used; otherwise, default plugin parameters are used.
    pub fn from_nixl_backend_config(name: &str, config: NixlBackendConfig) -> Result<Self> {
        let mut agent = Self::new(name)?;
        for (backend, params) in config.iter() {
            agent.add_backend_with_params(backend, params)?;
        }
        Ok(agent)
    }

    /// Add a backend to the agent with default parameters.
    pub fn add_backend(&mut self, backend: &str) -> Result<()> {
        self.add_backend_with_params(backend, &HashMap::new())
    }

    /// Add a backend to the agent with optional custom parameters.
    ///
    /// If `custom_params` is non-empty, each entry overrides the plugin's
    /// default parameters (fetched first via `get_plugin_params`) before the
    /// backend is created. If empty, the plugin defaults are used as-is.
    ///
    /// # Errors
    /// Returns an error if the plugin is unavailable, a parameter is rejected
    /// by NIXL, or backend creation fails.
    pub fn add_backend_with_params(
        &mut self,
        backend: &str,
        custom_params: &HashMap<String, String>,
    ) -> Result<()> {
        let backend_upper = backend.to_uppercase();
        if self.available_backends.contains(&backend_upper) {
            return Ok(());
        }

        // Get default params from plugin, then layer custom params on top.
        let (_, mut params) = self
            .agent
            .get_plugin_params(&backend_upper)
            .map_err(|e| anyhow::anyhow!("no {backend_upper} plugin found: {e}"))?;
        for (key, value) in custom_params {
            params
                .set(key, value)
                .with_context(|| format!("set NIXL {backend_upper} backend param {key}={value}"))?;
        }

        self.agent
            .create_backend(&backend_upper, &params)
            .with_context(|| format!("create NIXL backend {backend_upper}"))?;
        self.available_backends.insert(backend_upper);
        Ok(())
    }

    /// Create a NIXL agent requiring ALL specified backends to be available.
    ///
    /// Unlike `new_with_backends()` which continues if some backends fail, this method
    /// will return an error if ANY backend fails to initialize. Use this in production
    /// when specific backends are mandatory.
    ///
    /// # Arguments
    /// * `name` - Agent name
    /// * `backends` - List of backend names that MUST be available
    ///
    /// # Returns
    /// A `NixlAgent` with all requested backends initialized.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Agent creation fails
    /// - Any backend fails to initialize
    pub fn with_backends(name: &str, backends: &[&str]) -> Result<Self> {
        let mut agent = Self::new(name)?;
        let mut failed_backends = Vec::new();

        for backend in backends {
            let backend_upper = backend.to_uppercase();
            match agent.add_backend(&backend_upper) {
                Ok(_) => {
                    tracing::debug!("Initialized NIXL backend: {}", backend_upper);
                }
                Err(e) => {
                    tracing::error!("Failed to initialize {} backend: {}", backend_upper, e);
                    failed_backends.push((backend_upper, e.to_string()));
                }
            }
        }

        if !failed_backends.is_empty() {
            let error_details: Vec<String> = failed_backends
                .iter()
                .map(|(name, reason)| format!("{}: {}", name, reason))
                .collect();

            anyhow::bail!(
                "Failed to initialize required backends: [{}]",
                error_details.join(", ")
            );
        }

        Ok(agent)
    }

    /// Get a reference to the underlying raw NIXL agent.
    pub fn raw_agent(&self) -> &Agent {
        &self.agent
    }

    /// Consume and return the underlying raw NIXL agent.
    ///
    /// **Warning**: Once consumed, backend tracking is lost. Use this only when
    /// interfacing with code that requires `nixl_sys::Agent` directly.
    pub fn into_raw_agent(self) -> Agent {
        self.agent
    }

    /// Check if a specific backend is available.
    pub fn has_backend(&self, backend: &str) -> bool {
        self.available_backends.contains(&backend.to_uppercase())
    }

    /// Get all available backends.
    pub fn backends(&self) -> &HashSet<String> {
        &self.available_backends
    }

    /// Export agent metadata that names the given host registrations only.
    ///
    /// The peer receives the backend connection info and the memory section
    /// entries for `hosts`. It receives no entry for any other registration on
    /// this agent, so it cannot address a GPU mapping through this metadata.
    ///
    /// # Errors
    /// Returns an error if a descriptor is not host memory, or if NIXL refuses
    /// either export.
    pub fn host_metadata(&self, hosts: &[NixlDescriptor]) -> Result<HostAgentMetadata> {
        for host in hosts {
            ensure!(
                host.mem_type == MemType::Dram,
                "host metadata rejects the {:?} registration at {:#x}",
                host.mem_type,
                host.addr
            );
        }

        // An empty descriptor list selects every backend and emits the
        // connection info with an empty memory section. See
        // `nixlAgent::getLocalPartialMD`.
        let connections_only = RegDescList::new(MemType::Dram)
            .map_err(|error| anyhow::anyhow!("build NIXL descriptor list: {error:?}"))?;
        let connections = self
            .agent
            .get_local_partial_md(&connections_only, None)
            .map_err(|error| anyhow::anyhow!("export NIXL connection info: {error:?}"))?;

        if hosts.is_empty() {
            return Ok(HostAgentMetadata {
                connections,
                host_registrations: Vec::new(),
            });
        }

        let mut host_list = RegDescList::new(MemType::Dram)
            .map_err(|error| anyhow::anyhow!("build NIXL descriptor list: {error:?}"))?;
        for host in hosts {
            let addr = usize::try_from(host.addr).context("host registration address")?;
            host_list.add_desc(addr, host.size, host.device_id);
        }
        let host_registrations = self
            .agent
            .get_local_partial_md(&host_list, None)
            .map_err(|error| anyhow::anyhow!("export NIXL host registrations: {error:?}"))?;

        Ok(HostAgentMetadata {
            connections,
            host_registrations,
        })
    }

    /// Load host-only metadata from a peer and return the peer agent name.
    ///
    /// The connection info loads first. NIXL refuses a memory section for an
    /// agent whose connection info it does not hold.
    pub fn load_host_metadata(&self, metadata: &HostAgentMetadata) -> Result<String> {
        let name = self
            .agent
            .load_remote_md(&metadata.connections)
            .map_err(|error| anyhow::anyhow!("load NIXL connection info: {error:?}"))?;
        if metadata.host_registrations.is_empty() {
            return Ok(name);
        }
        let named = self
            .agent
            .load_remote_md(&metadata.host_registrations)
            .map_err(|error| anyhow::anyhow!("load NIXL host registrations: {error:?}"))?;
        ensure!(
            named == name,
            "NIXL host metadata names agent '{named}' after connection info named '{name}'"
        );
        Ok(name)
    }

    /// Require a specific backend, returning an error if unavailable.
    ///
    /// Use this at the start of operations that need specific backends.
    ///
    /// Note: In general, you want to instantiate all your backends before you start registering memory.
    /// We may change this to a builder pattern in the future to enforce all backends are instantiated
    /// before you start registering memory.
    pub fn require_backend(&self, backend: &str) -> Result<()> {
        let backend_upper = backend.to_uppercase();
        if self.has_backend(&backend_upper) {
            Ok(())
        } else {
            anyhow::bail!(
                "Operation requires {} backend, but it was not initialized. Available backends: {:?}",
                backend_upper,
                self.available_backends
            )
        }
    }
}

// Delegate common methods to the underlying agent.
//
// The workspace pin in `crates/Cargo.toml` is nixl-sys `=1.4.1`. Two defects
// that made `Agent::fetch_remote_md` and `Agent::invalidate_remote_md` unusable
// at `=1.0.1` are fixed at that version. The former self-deadlocked, and the
// latter passed a string without a NUL terminator to the C API.
//
// `Agent::get_local_md` exports every registration on the agent. Use
// [`NixlAgent::host_metadata`] for a peer export, so a shared agent does not
// hand its GPU mappings to that peer.
impl std::ops::Deref for NixlAgent {
    type Target = Agent;

    fn deref(&self) -> &Self::Target {
        &self.agent
    }
}

#[cfg(all(test, feature = "testing-nixl"))]
mod tests {
    use super::*;

    #[test]
    fn test_agent_backend_tracking() {
        // Try to create agent with UCX
        let agent = NixlAgent::with_backends("test", &["UCX"]).expect("Need UCX for test");

        // Should succeed if UCX is available
        assert!(agent.has_backend("UCX"));
        assert!(agent.has_backend("ucx")); // Case insensitive
    }

    #[test]
    fn test_require_backend() {
        let agent = NixlAgent::with_backends("test", &["UCX"]).expect("Need UCX for test");

        // Should succeed for available backend
        assert!(agent.require_backend("UCX").is_ok());

        // Should fail for unavailable backend
        assert!(agent.require_backend("GDS_MT").is_err());
    }

    #[test]
    fn test_require_backends_strict() {
        // Should succeed if UCX is available
        let agent =
            NixlAgent::with_backends("test_strict", &["UCX"]).expect("Failed to require backends");
        assert!(agent.has_backend("UCX"));

        // Should fail if any backend is missing (GDS likely not available)
        let result = NixlAgent::with_backends("test_strict_fail", &["UCX", "DUDE"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_backend_with_empty_params() {
        let mut agent = NixlAgent::new("test_empty_params").expect("Failed to create agent");

        // Empty params should work (uses plugin defaults)
        let result = agent.add_backend_with_params("UCX", &HashMap::new());
        assert!(result.is_ok());
        assert!(agent.has_backend("UCX"));
    }

    #[test]
    fn test_add_backend_with_custom_params_applies_them() {
        let mut agent = NixlAgent::new("test_custom_params").expect("Failed to create agent");

        // Custom params are layered onto the plugin defaults and passed through
        // to `create_backend`; UCX ignores keys it doesn't recognize.
        let mut params = HashMap::new();
        params.insert("thread_count".to_string(), "4".to_string());

        let result = agent.add_backend_with_params("UCX", &params);
        assert!(
            result.is_ok(),
            "custom params should plumb through: {result:?}"
        );
        assert!(agent.has_backend("UCX"));
    }

    #[test]
    fn test_add_backend_with_unknown_plugin_fails() {
        let mut agent = NixlAgent::new("test_unknown_plugin").expect("Failed to create agent");

        let result = agent.add_backend_with_params("NOT_A_REAL_BACKEND", &HashMap::new());
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("no NOT_A_REAL_BACKEND plugin found"));
    }

    #[test]
    fn test_from_nixl_backend_config_with_custom_params_applies_them() {
        // Config with custom params should plumb through to the created backend.
        let mut params = HashMap::new();
        params.insert("thread_count".to_string(), "4".to_string());

        let config = NixlBackendConfig::default().with_backend_params("UCX", params);

        let result = NixlAgent::from_nixl_backend_config("test_config_params", config);
        assert!(
            result.is_ok(),
            "custom params should plumb through: {result:?}"
        );
        assert!(result.unwrap().has_backend("UCX"));
    }

    #[test]
    fn test_from_nixl_backend_config_with_empty_params() {
        // Config with empty params should work
        let config = NixlBackendConfig::default().with_backend("UCX");

        let result = NixlAgent::from_nixl_backend_config("test_config_empty", config);
        assert!(result.is_ok());

        let agent = result.unwrap();
        assert!(agent.has_backend("UCX"));
    }
}
