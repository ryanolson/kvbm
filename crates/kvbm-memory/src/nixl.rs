// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL registration wrapper for storage types.

mod agent;
mod config;

use super::{MemoryDescriptor, StorageKind};
use std::any::Any;
use std::fmt;
use std::sync::Arc;

pub use agent::{NIXL_CAPI_LIB_ENV, NixlAgent};
pub use config::NixlBackendConfig;

pub use nixl_sys::{
    Agent, MemType, NotificationMap, OptArgs, RegDescList, RegistrationHandle, XferDescList,
    XferOp, XferRequest, is_stub,
};
pub use serde::{Deserialize, Serialize};

/// Agent metadata that names host registrations only.
///
/// NIXL keeps one memory section for each agent, so a full-agent export names
/// every registration that any owner made on that agent. A worker that shares
/// one agent between its GPU pool and its host mirror therefore hands a peer
/// the GPU mappings as well. This type carries the host half only.
///
/// The pair exists because NIXL splits the two facts across two calls. A
/// partial export with a non-empty descriptor list carries the memory section
/// and no backend connection info, and NIXL rejects such a blob with
/// `NIXL_ERR_NOT_FOUND`. A partial export with an empty descriptor list
/// carries the connection info and an empty memory section. The peer loads
/// both, in order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostAgentMetadata {
    /// Backend connection info for the agent. This blob names no memory.
    pub connections: Vec<u8>,
    /// Memory section that names the host registrations only.
    ///
    /// Empty when the agent exports no host registration.
    pub host_registrations: Vec<u8>,
}

/// Owns registration and mapped memory until a transfer completes.
pub trait MappedRegistrationGuard: Send + Sync + fmt::Debug {}
impl<T: Send + Sync + fmt::Debug> MappedRegistrationGuard for T {}

/// Registered intervals and their lifetime owner for one requested range.
#[derive(Debug)]
pub struct MappedRegistrationLease {
    /// Ordered, contiguous registered intervals that exactly cover the request.
    pub ranges: Vec<std::ops::Range<usize>>,
    /// Keeps every covered registration and physical allocation alive.
    pub guard: Arc<dyn MappedRegistrationGuard>,
}

/// Acquires only published mapped ranges. Withdrawal rejects new acquisitions.
pub trait MappedRegistrationProvider: Send + Sync + fmt::Debug {
    /// Acquire published intervals and retain them through actual completion.
    fn acquire(&self, address: usize, bytes: usize) -> anyhow::Result<MappedRegistrationLease>;
}

/// Trait for storage types that can be registered with NIXL.
pub trait NixlCompatible {
    /// Get parameters needed for NIXL registration.
    ///
    /// Returns (ptr, size, mem_type, device_id)
    fn nixl_params(&self) -> (*const u8, usize, MemType, u64);
}

/// Combined trait for memory that can be registered with NIXL.
///
/// This supertrait enables type erasure via `Arc<dyn NixlMemory>`.
/// Any type implementing both `MemoryDescriptor` and `NixlCompatible`
/// automatically implements this trait via the blanket implementation.
pub trait NixlMemory: MemoryDescriptor + NixlCompatible {}

// Blanket impl - any type with both traits automatically implements NixlMemory
impl<T: MemoryDescriptor + NixlCompatible + ?Sized> NixlMemory for T {}

/// NIXL descriptor containing registration information.
///
/// This struct holds the information needed to describe a memory region
/// to NIXL for transfer operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NixlDescriptor {
    /// Base address of the memory region.
    pub addr: u64,
    /// Size of the memory region in bytes.
    pub size: usize,
    /// Type of memory (host, device, etc.).
    pub mem_type: MemType,
    /// Device identifier (GPU index for device memory, 0 for host memory).
    pub device_id: u64,
}

impl nixl_sys::MemoryRegion for NixlDescriptor {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl nixl_sys::NixlDescriptor for NixlDescriptor {
    fn mem_type(&self) -> MemType {
        self.mem_type
    }

    fn device_id(&self) -> u64 {
        self.device_id
    }
}

/// View trait for accessing registration information without unwrapping.
pub trait RegisteredView {
    /// Get the name of the NIXL agent that registered this memory.
    fn agent_name(&self) -> &str;

    /// Get the NIXL descriptor for this registered memory.
    fn descriptor(&self) -> NixlDescriptor;
}

/// The state backing a [`NixlRegistered`] wrapper.
enum Registration {
    /// We registered this memory ourselves; the handle deregisters it on
    /// drop (before the wrapped storage drops). The field is never read
    /// directly -- it exists solely for its `Drop` side effect.
    Owned(#[allow(dead_code)] RegistrationHandle),
    /// Storage arrived already carrying its own [`NixlDescriptor`] (see
    /// [`MemoryDescriptor::nixl_descriptor`]); its registration lifetime is
    /// managed elsewhere and there is nothing for this wrapper to
    /// deregister.
    PreRegistered,
}

/// Wrapper for storage that has been registered with NIXL.
///
/// This wrapper ensures proper drop order: the registration handle is
/// dropped before the storage, ensuring deregistration happens before
/// the memory is freed.
pub struct NixlRegistered<S: NixlCompatible> {
    storage: S,
    // `Option` only so `Drop`/`into_storage` can `take()` it.
    registration: Option<Registration>,
    agent_name: String,
}

impl<S: NixlCompatible> Drop for NixlRegistered<S> {
    fn drop(&mut self) {
        // Explicitly drop the registration handle first
        drop(self.registration.take());
        // Storage drops naturally after
    }
}

impl<S: NixlCompatible> NixlRegistered<S> {
    /// True when NIXL can address this memory, whether this wrapper owns
    /// the registration (`Owned`) or the storage arrived pre-registered
    /// (`PreRegistered`).
    pub fn is_registered(&self) -> bool {
        self.registration.is_some()
    }

    /// True only when this wrapper owns the registration handle (i.e. it
    /// will deregister the memory on drop). `false` for pre-registered
    /// storage, whose registration lifetime is managed elsewhere.
    pub fn owns_registration(&self) -> bool {
        matches!(self.registration, Some(Registration::Owned(_)))
    }
}

impl<S: NixlCompatible + fmt::Debug> fmt::Debug for NixlRegistered<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NixlRegistered")
            .field("storage", &self.storage)
            .field("agent_name", &self.agent_name)
            .field("is_registered", &self.is_registered())
            .field("owns_registration", &self.owns_registration())
            .finish()
    }
}

impl<S: MemoryDescriptor + NixlCompatible + 'static> MemoryDescriptor for NixlRegistered<S> {
    fn addr(&self) -> usize {
        self.storage.addr()
    }

    fn size(&self) -> usize {
        self.storage.size()
    }

    fn storage_kind(&self) -> StorageKind {
        self.storage.storage_kind()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        Some(self.descriptor())
    }
}

impl<S: MemoryDescriptor + NixlCompatible> RegisteredView for NixlRegistered<S> {
    fn agent_name(&self) -> &str {
        &self.agent_name
    }

    fn descriptor(&self) -> NixlDescriptor {
        let (ptr, size, mem_type, device_id) = self.storage.nixl_params();
        NixlDescriptor {
            addr: ptr as u64,
            size,
            mem_type,
            device_id,
        }
    }
}

impl<S: MemoryDescriptor + NixlCompatible> NixlRegistered<S> {
    /// Get a reference to the underlying storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Get a mutable reference to the underlying storage.
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    /// Consume this wrapper and return the underlying storage.
    ///
    /// This will deregister the storage from NIXL (a no-op for
    /// pre-registered storage, whose registration is managed elsewhere).
    pub fn into_storage(mut self) -> S {
        drop(self.registration.take());
        let mut this = std::mem::ManuallyDrop::new(self);
        unsafe {
            let storage = std::ptr::read(&this.storage);
            std::ptr::drop_in_place(&mut this.agent_name);
            storage
        }
    }
}

/// Registration with a NIXL agent failed.
///
/// The storage is returned so callers can retry or fall back to another
/// path; `source` carries the underlying `nixl_sys` error that caused the
/// failure.
pub struct RegisterError<S> {
    /// The storage that failed to register. Ownership is handed back to the
    /// caller (registration failure does not destroy the memory).
    pub storage: S,
    /// The underlying NIXL error that caused registration to fail.
    pub source: nixl_sys::NixlError,
}

impl<S> fmt::Display for RegisterError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to register memory with NIXL agent: {}",
            self.source
        )
    }
}

impl<S> fmt::Debug for RegisterError<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisterError")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<S> std::error::Error for RegisterError<S> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Register storage with a NIXL agent.
///
/// This consumes the storage and returns a `NixlRegistered` wrapper that
/// manages the registration lifetime. The registration handle will be
/// automatically dropped when the wrapper is dropped, ensuring proper
/// cleanup order.
///
/// # Arguments
/// * `storage` - The storage to register (consumed)
/// * `agent` - The NIXL agent to register with
/// * `opt` - Optional arguments for registration
///
/// # Returns
/// A `NixlRegistered` wrapper containing the storage and registration handle
/// on success. On failure, a [`RegisterError`] carrying both the storage
/// (for retry/fallback) and the underlying `nixl_sys` error.
pub fn register_with_nixl<S>(
    storage: S,
    agent: &Agent,
    opt: Option<&OptArgs>,
) -> std::result::Result<NixlRegistered<S>, RegisterError<S>>
where
    S: MemoryDescriptor + NixlCompatible,
{
    // let storage_kind = storage.storage_kind();

    // // Determine if registration is needed based on storage type and available backends
    // let should_register = match storage_kind {
    //     StorageKind::System | StorageKind::Pinned => {
    //         // System/Pinned memory needs UCX for remote transfers
    //         agent.has_backend("UCX") || agent.has_backend("POSIX")
    //     }
    //     StorageKind::Device(_) => {
    //         // Device memory needs UCX for remote transfers OR GDS for direct disk transfers
    //         agent.has_backend("UCX") || agent.has_backend("GDS_MT")
    //     }
    //     StorageKind::Disk(_) => {
    //         // Disk storage needs POSIX for regular I/O OR GDS for GPU direct I/O
    //         agent.has_backend("POSIX") || agent.has_backend("GDS_MT")
    //     } // StorageKind::Object(_) => {
    //       //     // Object storage is always registered via NIXL's OBJ plugin
    //       //     agent.has_backend("OBJ")
    //       // }
    // };

    // this is not true for our future object storage. so let's rethink this.
    // for object, if there is no device_id or device_id is 0, then we need to register
    // alternatively, the object storage holds it's own internal metadata but does not
    // expose as a nixl descriptor, thus ObjectStorag will by default like all other storage
    // types have a None for nixl_descriptor(), and we will use the internal
    if storage.nixl_descriptor().is_some() {
        return Ok(NixlRegistered {
            storage,
            registration: Some(Registration::PreRegistered),
            agent_name: agent.name().to_string(),
        });
    }

    // Get NIXL parameters
    let (ptr, size, mem_type, device_id) = storage.nixl_params();

    // Create a NIXL descriptor for registration
    let descriptor = NixlDescriptor {
        addr: ptr as u64,
        size,
        mem_type,
        device_id,
    };

    match agent.register_memory(&descriptor, opt) {
        Ok(handle) => Ok(NixlRegistered {
            storage,
            registration: Some(Registration::Owned(handle)),
            agent_name: agent.name().to_string(),
        }),
        Err(e) => Err(RegisterError { storage, source: e }),
    }
}

// =============================================================================
// Arc<dyn NixlMemory> support
// =============================================================================

impl NixlCompatible for Arc<dyn NixlMemory + Send + Sync> {
    fn nixl_params(&self) -> (*const u8, usize, MemType, u64) {
        (**self).nixl_params()
    }
}

impl MemoryDescriptor for Arc<dyn NixlMemory + Send + Sync> {
    fn addr(&self) -> usize {
        (**self).addr()
    }

    fn size(&self) -> usize {
        (**self).size()
    }

    fn storage_kind(&self) -> StorageKind {
        (**self).storage_kind()
    }

    fn as_any(&self) -> &dyn Any {
        (**self).as_any()
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        (**self).nixl_descriptor()
    }
}

// =============================================================================
// Extension trait for ergonomic API
// =============================================================================

/// Extension trait providing ergonomic `.register()` method for NIXL registration.
///
/// This trait is automatically implemented for all types that implement both
/// `MemoryDescriptor` and `NixlCompatible`. Import this trait to use the
/// method syntax:
///
///
pub trait NixlRegisterExt: MemoryDescriptor + NixlCompatible + Sized {
    /// Get this memory as NIXL-registered.
    ///
    /// This operation is idempotent - it's a no-op if the memory is already registered.
    ///
    /// # Arguments
    /// * `agent` - The NIXL agent to register with
    /// * `opt` - Optional arguments for registration
    ///
    /// # Returns
    /// A `NixlRegistered` wrapper on success, or a [`RegisterError`] carrying
    /// the original storage and the underlying `nixl_sys` error on failure.
    fn register(
        self,
        agent: &NixlAgent,
        opt: Option<&OptArgs>,
    ) -> std::result::Result<NixlRegistered<Self>, RegisterError<Self>> {
        register_with_nixl(self, agent, opt)
    }
}

// Blanket impl for all compatible types
impl<T: MemoryDescriptor + NixlCompatible + Sized> NixlRegisterExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SystemStorage;

    #[cfg(feature = "testing-nixl")]
    #[test]
    fn registration_without_a_backend_returns_storage() -> anyhow::Result<()> {
        let agent = Agent::new("registration-without-backend")?;
        let storage = SystemStorage::new(1024)?;
        let address = storage.addr();
        let error = register_with_nixl(storage, &agent, None)
            .expect_err("an agent without a backend cannot register memory");
        assert_eq!(error.storage.addr(), address);
        assert_eq!(error.storage.size(), 1024);
        Ok(())
    }

    /// §5.4 regression test: storage that arrives pre-registered (i.e.
    /// `nixl_descriptor()` already returns `Some`) must report
    /// `is_registered() == true`, since NIXL can address it -- even though
    /// this wrapper does not own the registration and has nothing to
    /// deregister on drop.
    #[test]
    fn pre_registered_is_registered_but_not_owned() {
        let storage = SystemStorage::new(1024).expect("allocation should succeed");
        let registered = NixlRegistered {
            storage,
            registration: Some(Registration::PreRegistered),
            agent_name: "test-agent".to_string(),
        };

        assert!(registered.is_registered());
        assert!(!registered.owns_registration());
    }

    /// §5.3 regression test: a failed registration must not swallow the
    /// underlying `nixl_sys::NixlError` -- it travels with the returned
    /// storage inside `RegisterError`, reachable via `Display` and
    /// `std::error::Error::source()`.
    #[test]
    fn register_error_carries_source_and_storage() {
        let storage = SystemStorage::new(1024).expect("allocation should succeed");
        let err = RegisterError {
            storage,
            source: nixl_sys::NixlError::InvalidParam,
        };

        let msg = err.to_string();
        assert!(
            msg.contains("failed to register memory with NIXL agent"),
            "unexpected message: {msg}"
        );
        assert!(
            msg.contains("Invalid parameter"),
            "message should include the source error: {msg}"
        );

        let source = std::error::Error::source(&err).expect("source() should delegate");
        assert!(source.to_string().contains("Invalid parameter"));

        // Storage is handed back so callers can retry or fall back.
        assert_eq!(err.storage.size(), 1024);
    }
}
