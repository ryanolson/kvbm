// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA device memory storage.

use super::{MemoryDescriptor, Result, StorageError, StorageKind, nixl::NixlDescriptor};
use cudarc::driver::CudaContext;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// A caller-installed source of CUDA contexts, keyed by device ID.
///
/// Install one with [`set_cuda_context_provider`] to have this crate's CUDA
/// allocations (`DeviceStorage::new`, NUMA-aware `PinnedStorage`) bind to
/// contexts owned by an embedder (e.g. rhino-core's context registry)
/// instead of the default process-global cache.
pub type CudaContextProvider = dyn Fn(u32) -> Result<Arc<CudaContext>> + Send + Sync;

/// The single process-wide source of CUDA contexts: either a caller-installed
/// [`CudaContextProvider`] or the unchanged default (create-and-cache primary
/// contexts on demand).
enum ContextSource {
    /// Caller-installed provider (e.g. rhino-core's context registry).
    Provider(Box<CudaContextProvider>),
    /// Unchanged default: create-and-cache primary contexts on demand.
    Default(Mutex<HashMap<u32, Arc<CudaContext>>>),
}

/// Backs both [`set_cuda_context_provider`] and the default per-device
/// context cache. A single `OnceLock` (rather than a separate flag) is what
/// makes "install after a prior default-path use" and "install twice" the
/// same rejection: `cuda_context` occupies this slot with the default cache
/// the first time it runs, exactly as a provider install would.
static CONTEXT_SOURCE: OnceLock<ContextSource> = OnceLock::new();

/// Install a process-wide CUDA context provider.
///
/// Must be called before the first device/pinned allocation in this
/// process. Returns an error if a context source is already initialized --
/// either a provider was installed earlier, or [`cuda_context`] already ran
/// once and fell back to the default cache. Embedders (e.g. rhino) point
/// this at their own context registry at startup.
pub fn set_cuda_context_provider(provider: Box<CudaContextProvider>) -> Result<()> {
    install_context_provider(&CONTEXT_SOURCE, provider)
}

fn install_context_provider(
    source: &OnceLock<ContextSource>,
    provider: Box<CudaContextProvider>,
) -> Result<()> {
    source.set(ContextSource::Provider(provider)).map_err(|_| {
        StorageError::OperationFailed(
            "CUDA context source already initialized (a provider was installed earlier, \
                 or a device/pinned allocation already ran and fell back to the default cache)"
                .into(),
        )
    })
}

fn context_source(source: &OnceLock<ContextSource>) -> &ContextSource {
    source.get_or_init(|| ContextSource::Default(Mutex::new(HashMap::new())))
}

/// Get or create a CUDA context for the given device.
///
/// Delegates to the installed [`CudaContextProvider`] if one was set via
/// [`set_cuda_context_provider`]; otherwise falls back to the original
/// create-and-cache-on-demand behavior, unchanged.
pub(crate) fn cuda_context(device_id: u32) -> Result<Arc<CudaContext>> {
    match context_source(&CONTEXT_SOURCE) {
        ContextSource::Provider(provider) => provider(device_id),
        ContextSource::Default(cache) => {
            let mut map = cache.lock().unwrap();

            if let Some(existing) = map.get(&device_id) {
                return Ok(existing.clone());
            }

            let ctx = CudaContext::new(device_id as usize)?;
            map.insert(device_id, ctx.clone());
            Ok(ctx)
        }
    }
}

/// CUDA device memory allocated via cudaMalloc.
#[derive(Debug)]
pub struct DeviceStorage {
    /// CUDA context used for allocation and deallocation.
    ctx: Arc<CudaContext>,
    /// Device pointer to the allocated memory.
    ptr: u64,
    /// CUDA device ID where memory is allocated.
    device_id: u32,
    /// Size of the allocation in bytes.
    len: usize,
}

unsafe impl Send for DeviceStorage {}
unsafe impl Sync for DeviceStorage {}

impl DeviceStorage {
    /// Allocate new device memory of the given size.
    ///
    /// # Arguments
    /// * `len` - Size in bytes to allocate
    /// * `device_id` - CUDA device on which to allocate
    pub fn new(len: usize, device_id: u32) -> Result<Self> {
        let ctx = cuda_context(device_id)?;
        Self::new_with_context(len, ctx)
    }

    /// Allocate new device memory of the given size, using an explicitly
    /// supplied CUDA context rather than the process-wide
    /// [`cuda_context`]/[`set_cuda_context_provider`] seam.
    ///
    /// This bypasses the process-wide context source entirely -- useful when
    /// the caller already owns the right context and wants no dependency on
    /// this crate's global context lookup at all. The device ID recorded
    /// against this allocation (for [`StorageKind::Device`] and the NIXL
    /// `devId`) is derived from `ctx` itself, so it can never disagree with
    /// the context actually used to allocate.
    ///
    /// # Arguments
    /// * `len` - Size in bytes to allocate
    /// * `ctx` - CUDA context to bind and allocate from
    pub fn new_with_context(len: usize, ctx: Arc<CudaContext>) -> Result<Self> {
        if len == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }

        ctx.bind_to_thread().map_err(StorageError::Cuda)?;
        let ptr = unsafe { cudarc::driver::result::malloc_sync(len).map_err(StorageError::Cuda)? };
        let device_id = ctx.ordinal() as u32;

        Ok(Self {
            ctx,
            ptr,
            device_id,
            len,
        })
    }

    /// Get the device pointer value.
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    /// Get the CUDA device ID this memory is allocated on.
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

impl Drop for DeviceStorage {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.bind_to_thread() {
            tracing::debug!("failed to bind CUDA context for free: {e}");
        }
        unsafe {
            if let Err(e) = cudarc::driver::result::free_sync(self.ptr) {
                tracing::debug!("failed to free device memory: {e}");
            }
        };
    }
}

impl MemoryDescriptor for DeviceStorage {
    fn addr(&self) -> usize {
        self.device_ptr() as usize
    }

    fn size(&self) -> usize {
        self.len
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Device(self.device_id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

// Support for NIXL registration
impl super::nixl::NixlCompatible for DeviceStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (
            self.ptr as *const u8,
            self.len,
            nixl_sys::MemType::Vram,
            self.device_id as u64,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_source_blocks_late_provider() {
        let source = OnceLock::new();
        let _ = context_source(&source);

        let provider: Box<CudaContextProvider> = Box::new(|_| unreachable!());
        assert!(install_context_provider(&source, provider).is_err());
    }
}
