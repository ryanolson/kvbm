// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `mmap` + `cuMemHostRegister` host pinned storage with optional NUMA
//! `mbind` and best-effort hugepages.
//!
//! Companion to [`crate::pinned::PinnedStorage`] (which uses `cuMemHostAlloc`,
//! no hugepage / mbind control). This path exists because hugepage-backed
//! pinned memory is unreachable through `cuMemHostAlloc` — the kernel only
//! honors `MAP_HUGETLB` on caller-owned `mmap` regions. Use this for the
//! kvbm-service host-memory pool; use [`crate::PinnedStorage`] for legacy
//! per-GPU allocations.
//!
//! Pipeline per allocation. For an allocation with a target node, the
//! caller's thread should already be pinned to a CPU in the cpulist of that
//! node for correct first-touch:
//!
//! 1. `mmap(NULL, len, RW, MAP_PRIVATE | MAP_ANON [| MAP_HUGETLB|MAP_HUGE_2MB])`
//!    — try explicit hugetlb, fall back per [`HugepageMode`].
//! 2. `mbind(addr, len, MPOL_BIND, nodemask({target}), …, MPOL_MF_STRICT)`
//!    — belt-and-suspenders against the kernel migrating away from the
//!    target node, even after first-touch lands the page locally. An
//!    allocation without a target node skips this step, and the OS places
//!    the pages ([`NumaPlacementPolicy::OsDefault`]).
//! 3. First-touch one byte per (huge)page from the calling thread.
//! 4. `cuMemHostRegister(ptr, len, DEVICEMAP)` to pin + map to CUDA.
//!
//! Drop ordering inside [`MmappedPinnedStorage`] guarantees
//! `cuMemHostUnregister` runs before `munmap`, and the [`Arc<CudaContext>`]
//! outlives both so the unregister has a context to bind.

use std::any::Any;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use nix::libc;
use serde::{Deserialize, Serialize};

use crate::numa::placement::PlacementDecision;
use crate::numa::{NumaNode, PagePlacement};
use crate::{MemoryDescriptor, Result, StorageError, StorageKind, actions, nixl::NixlDescriptor};

/// Hugepage allocation policy for one allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum HugepageMode {
    /// Don't ask for hugepages. Plain anon `mmap`, no `MADV_HUGEPAGE`.
    #[default]
    Disabled,
    /// Try `MAP_HUGETLB` first; fall back to anon + `MADV_HUGEPAGE`; fall
    /// back to plain anon. Records the actual tier landed on the slab.
    BestEffort,
    /// Only `MAP_HUGETLB`. Allocation fails if hugepages aren't reserved.
    Required,
}

/// Which page-backing tier this slab actually landed on. Surfaced to the
/// pool so a fallback can be reported in metrics + `/v1/pool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HugepageTier {
    /// `mmap(MAP_HUGETLB | MAP_HUGE_<size>)` succeeded.
    Explicit {
        /// Page size requested via the `MAP_HUGE_*` flag, in bytes.
        page_size: usize,
    },
    /// Plain anon `mmap` + `madvise(MADV_HUGEPAGE)`. Kernel may or may not
    /// have actually promoted the pages — observable via `/proc/<pid>/smaps`.
    Thp,
    /// Plain anon `mmap`, no hugepage hints.
    None,
}

/// The NUMA placement policy that one slab used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumaPlacementPolicy {
    /// `mbind(MPOL_BIND)` bound the pages to this node.
    Bound(NumaNode),
    /// No `mbind`. The OS placed the pages with its default policy.
    OsDefault,
}

/// Why pinned host memory was placed by the OS instead of bound to the
/// device's NUMA node. Recorded in [`HostPlacement::fallback`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PlacementFallback {
    /// The device's NUMA node could not be determined.
    NodeUnknown,
    /// NUMA placement is disabled (`DYN_MEMORY_DISABLE_NUMA`).
    NumaDisabled,
    /// The CPU topology under `/sys/devices/system/node` could not be read.
    TopologyUnavailable,
    /// The device's node has no CPUs (a memory-only node), so no thread can
    /// first-touch the pages there.
    NodeHasNoCpus(NumaNode),
    /// `mbind(MPOL_BIND)` to the node was refused, for example by a cpuset
    /// or a seccomp profile. The pages stay on the same mapping and are
    /// touched from the same thread; the OS places them.
    BindRefused {
        /// The node the bind asked for.
        node: NumaNode,
        /// The error `mbind` returned.
        error: String,
    },
}

/// Where one slab of pinned host memory was placed, and why.
///
/// [`allocate_pinned_host`] fills every field. The strict
/// [`MmappedPinnedStorage::allocate`] records the node it was given in
/// `requested`, has no fallback (it fails instead), and does not observe the
/// pages, so `observed` is `None`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostPlacement {
    /// The device's NUMA node, as [`crate::numa::get_device_numa_node`]
    /// reported it. `None` when it is unknown.
    pub requested: Option<NumaNode>,
    /// The policy the pages got: bound to a node, or placed by the OS.
    pub policy: NumaPlacementPolicy,
    /// Why the pages were not bound. `None` when they were bound, or when the
    /// caller asked for OS placement.
    pub fallback: Option<PlacementFallback>,
    /// The node the kernel reports for the first page after first touch.
    /// `None` when the query failed (for example, denied by seccomp) or was
    /// not made.
    pub observed: Option<NumaNode>,
}

/// Options for [`MmappedPinnedStorage::allocate`].
#[derive(Debug, Clone)]
pub struct MmappedPinnedOptions {
    /// Requested user-visible size in bytes. The actual mapping may be
    /// rounded up to the effective page size and reported via
    /// [`MmappedPinnedStorage::mapped_len`].
    pub size: usize,
    /// Target NUMA node for the pages. **Must be a host-CPU node** —
    /// memory-only nodes are out of scope (see
    /// `crate::resources::NumaNodeRole::HostCpu`, gated behind the
    /// `host-pools` feature). `mbind(MPOL_BIND)` is
    /// applied against this node.
    ///
    /// `None` skips the `mbind`. The OS then places the pages with its
    /// default policy. Use `None` when the node is unknown or when NUMA
    /// placement is disabled.
    pub numa_node: Option<NumaNode>,
    /// Hugepage strategy.
    pub hugepage_mode: HugepageMode,
    /// Page size to request from the explicit hugetlb pool. `None` => use
    /// the system default (`/proc/meminfo Hugepagesize:`).
    pub hugepage_size: Option<usize>,
    /// CUDA device ordinal whose context is used for `cuMemHostRegister`.
    /// Any visible GPU works — the registration is device-portable.
    pub ctx_device_id: u32,
}

/// `mmap`-backed host pinned memory registered with CUDA via
/// `cuMemHostRegister`. `mbind` enforces the NUMA placement when the
/// allocation has a target node.
#[derive(Debug)]
pub struct MmappedPinnedStorage {
    // RAII guard — only used for its Drop impl (cuMemHostUnregister).
    // Field order is load-bearing: `register_guard` is declared before
    // `mmap_guard` so the unregister runs before munmap.
    #[allow(dead_code)]
    register_guard: HostRegisterGuard,
    mmap_guard: MmapGuard,
    hugepage_tier: HugepageTier,
    placement: HostPlacement,
    ctx: Arc<CudaContext>,
}

unsafe impl Send for MmappedPinnedStorage {}
unsafe impl Sync for MmappedPinnedStorage {}

impl MmappedPinnedStorage {
    /// Allocate one slab according to `opt`. If `opt.numa_node` is a node,
    /// **the caller must already be on a CPU in the cpulist of that node** so
    /// the first-touch policy lands pages on the right node. The
    /// [`crate::numa::worker_pool::NumaWorkerPool`] is the production caller
    /// and handles thread pinning. If `opt.numa_node` is `None`, the function
    /// does no `mbind` and the OS places the pages.
    pub fn allocate(opt: MmappedPinnedOptions) -> Result<Self> {
        if opt.size == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }
        let MmappedPinnedOptions {
            size,
            numa_node,
            hugepage_mode,
            hugepage_size,
            ctx_device_id,
        } = opt;

        let page_size = hugepage_size.unwrap_or_else(default_hugepage_size);
        let (mmap_guard, hugepage_tier) = mmap_with_tier(size, page_size, hugepage_mode)?;

        let numa_policy = match numa_node {
            Some(node) => {
                mbind_to_node(mmap_guard.ptr, mmap_guard.len, node).map_err(|e| {
                    StorageError::AllocationFailed(format!(
                        "mbind(MPOL_BIND) on node {}: {}",
                        node.0, e
                    ))
                })?;
                NumaPlacementPolicy::Bound(node)
            }
            None => NumaPlacementPolicy::OsDefault,
        };

        first_touch(mmap_guard.ptr, mmap_guard.len, hugepage_tier);

        let ctx = crate::device::cuda_context(ctx_device_id)?;
        ctx.bind_to_thread().map_err(StorageError::Cuda)?;

        let register_guard = HostRegisterGuard::new(mmap_guard.ptr, mmap_guard.len, ctx.clone())?;

        Ok(Self {
            register_guard,
            mmap_guard,
            hugepage_tier,
            placement: HostPlacement {
                requested: numa_node,
                policy: numa_policy,
                fallback: None,
                observed: None,
            },
            ctx,
        })
    }

    /// User-requested size in bytes. May be less than [`Self::mapped_len`]
    /// when rounded up for hugepages.
    pub fn size(&self) -> usize {
        // Size and mapped_len are identical when no rounding occurred; we
        // track the request through mmap_guard.len which equals the rounded
        // mapping length — there's no separate user-size field because the
        // pool treats the rounded length as the actual capacity.
        self.mmap_guard.len
    }

    /// Length actually mapped (size rounded up to the effective page size).
    pub fn mapped_len(&self) -> usize {
        self.mmap_guard.len
    }

    /// Which page-backing tier this slab landed on.
    pub fn hugepage_tier(&self) -> HugepageTier {
        self.hugepage_tier
    }

    /// NUMA placement policy that this slab used.
    pub fn numa_policy(&self) -> NumaPlacementPolicy {
        self.placement.policy
    }

    /// Where this slab was placed, why, and where the kernel put its first
    /// page. [`Self::numa_policy`] and [`Self::numa_node`] agree with it.
    pub fn placement(&self) -> &HostPlacement {
        &self.placement
    }

    /// NUMA node the pages were bound to via `mbind(MPOL_BIND)`. `None` when
    /// the OS placed the pages.
    pub fn numa_node(&self) -> Option<NumaNode> {
        match self.placement.policy {
            NumaPlacementPolicy::Bound(node) => Some(node),
            NumaPlacementPolicy::OsDefault => None,
        }
    }

    /// CUDA context used for registration.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Pointer to the start of the mapping.
    ///
    /// # Safety
    /// Caller must not retain the pointer past this storage's drop.
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.mmap_guard.ptr as *const u8
    }
}

impl MemoryDescriptor for MmappedPinnedStorage {
    fn addr(&self) -> usize {
        self.mmap_guard.ptr
    }

    fn size(&self) -> usize {
        self.mmap_guard.len
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Pinned
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

impl crate::nixl::NixlCompatible for MmappedPinnedStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (
            self.mmap_guard.ptr as *const u8,
            self.mmap_guard.len,
            nixl_sys::MemType::Dram,
            0,
        )
    }
}

impl actions::Memset for MmappedPinnedStorage {
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<()> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| StorageError::OperationFailed("memset: offset overflow".into()))?;
        if end > self.mmap_guard.len {
            return Err(StorageError::OperationFailed(
                "memset: offset + size > storage size".into(),
            ));
        }
        unsafe {
            let ptr = (self.mmap_guard.ptr as *mut u8).add(offset);
            std::ptr::write_bytes(ptr, value, size);
        }
        Ok(())
    }
}

// =============================================================================
// the pinned host entry: decide, map, bind, touch, observe, register
// =============================================================================

/// Allocate one region of pinned host memory for `device_id`.
///
/// Registered `mmap` on base pages ([`HugepageTier::None`]). The pages are
/// bound with `mbind(MPOL_BIND)` to the device's NUMA node only when that
/// node is known and has CPUs; the mapping, bind, first touch and
/// `cuMemHostRegister` then run on a thread pinned to that node. Otherwise
/// the OS places the pages, and [`HostPlacement::fallback`] says why.
///
/// A refused `mbind` is not an error: the same mapping is touched from the
/// same pinned thread and the OS places it, recorded as
/// [`PlacementFallback::BindRefused`]. Any other failure (`mmap`, CUDA
/// context, `cuMemHostRegister`) fails the call. Logs one `info` line with
/// the placement.
pub fn allocate_pinned_host(size: usize, device_id: u32) -> Result<MmappedPinnedStorage> {
    let _ = (size, device_id);
    todo!("allocate_pinned_host")
}

/// Allocate one region of pinned host memory per entry of `sizes`, in order,
/// each as [`allocate_pinned_host`] does. Stops at the first error; regions
/// already allocated are freed.
pub fn allocate_pinned_host_regions(
    sizes: &[usize],
    device_id: u32,
) -> Result<Vec<MmappedPinnedStorage>> {
    sizes
        .iter()
        .map(|&size| allocate_pinned_host(size, device_id))
        .collect()
}

/// A mapping that went through [`map_place_register`]. `registered` is
/// declared first so it drops (unregisters) before `mmap` unmaps.
pub(crate) struct Placed<T> {
    pub(crate) registered: T,
    pub(crate) mmap: MmapGuard,
    pub(crate) placement: HostPlacement,
}

/// Map `size` bytes on base pages, apply `decision`, first-touch from the
/// calling thread, observe the first page, and register.
///
/// Only `bind` can change the placement: its error becomes
/// [`PlacementFallback::BindRefused`]. An `observe` error is logged and
/// recorded as `observed = None`. A `register` error fails the call as is.
/// The steps are parameters so tests can inject failures.
pub(crate) fn map_place_register<T>(
    size: usize,
    requested: Option<NumaNode>,
    decision: PlacementDecision,
    bind: impl FnOnce(usize, usize, NumaNode) -> std::result::Result<(), String>,
    observe: impl FnOnce(usize) -> std::io::Result<PagePlacement>,
    register: impl FnOnce(usize, usize) -> Result<T>,
) -> Result<Placed<T>> {
    let _ = (size, requested, decision, bind, observe, register);
    todo!("map_place_register")
}

// =============================================================================
// internal: tier ladder, mmap, mbind, first-touch, register guard
// =============================================================================

/// `MPOL_MF_STRICT` from `<numaif.h>` — kernel returns `EIO` if a page can't
/// be placed on the requested node. Not exposed by `libc`, so define inline.
const MPOL_MF_STRICT: libc::c_uint = 1;

#[derive(Debug)]
struct MmapGuard {
    ptr: usize,
    len: usize,
}

impl Drop for MmapGuard {
    fn drop(&mut self) {
        if self.ptr == 0 || self.len == 0 {
            return;
        }
        // SAFETY: we own the mapping; no outstanding pointers because Drop
        // runs only when the owning MmappedPinnedStorage is being destroyed.
        let r = unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
        if r != 0 {
            tracing::warn!(
                "munmap({:#x}, {}) failed: {}",
                self.ptr,
                self.len,
                std::io::Error::last_os_error()
            );
        }
    }
}

#[derive(Debug)]
struct HostRegisterGuard {
    ptr: usize,
    ctx: Arc<CudaContext>,
}

impl HostRegisterGuard {
    fn new(ptr: usize, len: usize, ctx: Arc<CudaContext>) -> Result<Self> {
        // SAFETY: the mapping is live for the duration of this Self
        // (declared after MmapGuard in MmappedPinnedStorage, drops first).
        let cu = unsafe {
            cudarc::driver::sys::cuMemHostRegister_v2(
                ptr as *mut std::ffi::c_void,
                len,
                cudarc::driver::sys::CU_MEMHOSTREGISTER_DEVICEMAP,
            )
        };
        cu.result().map_err(StorageError::Cuda)?;
        Ok(Self { ptr, ctx })
    }
}

impl Drop for HostRegisterGuard {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.bind_to_thread() {
            tracing::warn!("bind CUDA context for cuMemHostUnregister: {:?}", e);
        }
        let cu =
            unsafe { cudarc::driver::sys::cuMemHostUnregister(self.ptr as *mut std::ffi::c_void) };
        if let Err(e) = cu.result() {
            tracing::warn!("cuMemHostUnregister({:#x}): {:?}", self.ptr, e);
        }
    }
}

fn mmap_with_tier(
    size: usize,
    page_size: usize,
    mode: HugepageMode,
) -> Result<(MmapGuard, HugepageTier)> {
    match mode {
        HugepageMode::Disabled => mmap_anon(size).map(|g| (g, HugepageTier::None)),
        HugepageMode::Required => mmap_explicit(size, page_size)
            .map(|g| (g, HugepageTier::Explicit { page_size }))
            .map_err(|e| {
                StorageError::AllocationFailed(format!(
                    "explicit hugetlb required but unavailable ({} bytes, {} byte pages): {}",
                    size, page_size, e
                ))
            }),
        HugepageMode::BestEffort => match mmap_explicit(size, page_size) {
            Ok(g) => Ok((g, HugepageTier::Explicit { page_size })),
            Err(huge_err) => {
                tracing::info!(
                    "explicit MAP_HUGETLB unavailable for {} bytes ({}); trying THP",
                    size,
                    huge_err
                );
                match mmap_anon_thp(size) {
                    Ok(g) => Ok((g, HugepageTier::Thp)),
                    Err(thp_err) => {
                        tracing::info!(
                            "THP path failed for {} bytes ({}); falling back to plain anon",
                            size,
                            thp_err
                        );
                        mmap_anon(size).map(|g| (g, HugepageTier::None))
                    }
                }
            }
        },
    }
}

fn mmap_anon(size: usize) -> Result<MmapGuard> {
    let len = round_up(size, system_page_size());
    do_mmap(len, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, 0)
}

fn mmap_anon_thp(size: usize) -> Result<MmapGuard> {
    let len = round_up(size, system_page_size());
    let g = do_mmap(len, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, 0)?;
    // Best-effort hint; ignore failures (e.g. THP disabled at boot).
    let _ = unsafe { libc::madvise(g.ptr as *mut libc::c_void, g.len, libc::MADV_HUGEPAGE) };
    Ok(g)
}

fn mmap_explicit(size: usize, page_size: usize) -> Result<MmapGuard> {
    let len = round_up(size, page_size);
    let huge_flag = map_huge_flag_for(page_size).ok_or_else(|| {
        StorageError::AllocationFailed(format!(
            "no MAP_HUGE_* flag known for page size {} bytes",
            page_size
        ))
    })?;
    do_mmap(
        len,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
        huge_flag,
    )
}

fn do_mmap(len: usize, base_flags: libc::c_int, extra_flags: libc::c_int) -> Result<MmapGuard> {
    let flags = base_flags | extra_flags;
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            flags,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(StorageError::AllocationFailed(format!(
            "mmap len={} flags={:#x}: {}",
            len,
            flags,
            std::io::Error::last_os_error()
        )));
    }
    Ok(MmapGuard {
        ptr: ptr as usize,
        len,
    })
}

/// Bind `[addr, addr+len)` to a single NUMA node via `mbind(MPOL_BIND)`.
fn mbind_to_node(addr: usize, len: usize, node: NumaNode) -> std::result::Result<(), String> {
    if node.0 >= 64 {
        return Err(format!(
            "mbind nodemask supports up to 64 nodes; got node id {}",
            node.0
        ));
    }
    let nodemask: u64 = 1u64 << node.0;
    // maxnode is the bit count the kernel reads from nodemask. 64 covers a
    // single u64 mask.
    let r = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            addr,
            len,
            libc::MPOL_BIND as usize,
            &nodemask as *const u64 as usize,
            64usize,
            MPOL_MF_STRICT as usize,
        )
    };
    if r != 0 {
        return Err(format!("mbind: {}", std::io::Error::last_os_error()));
    }
    Ok(())
}

/// Touch one byte per page so first-touch lands pages on the calling
/// thread's CPU's NUMA node. For a target node, the caller pins the thread
/// before invoking.
fn first_touch(addr: usize, len: usize, tier: HugepageTier) {
    let step = match tier {
        HugepageTier::Explicit { page_size } => page_size,
        HugepageTier::Thp | HugepageTier::None => system_page_size(),
    };
    let mut off = 0usize;
    while off < len {
        unsafe {
            std::ptr::write_volatile((addr as *mut u8).add(off), 0);
        }
        off = off.saturating_add(step);
    }
    // Ensure the last byte is touched if size isn't an exact page multiple.
    if len > 0 {
        unsafe {
            std::ptr::write_volatile((addr as *mut u8).add(len - 1), 0);
        }
    }
}

fn system_page_size() -> usize {
    match unsafe { libc::sysconf(libc::_SC_PAGESIZE) } {
        n if n > 0 => n as usize,
        _ => 4096,
    }
}

fn default_hugepage_size() -> usize {
    crate::hugepage::HugepageInfo::discover()
        .default_size_bytes
        .max(2 * 1024 * 1024)
}

fn round_up(size: usize, page: usize) -> usize {
    if page == 0 {
        return size;
    }
    size.div_ceil(page).saturating_mul(page)
}

/// Map a desired page size onto the `MAP_HUGE_*` flag the kernel expects.
/// Encoded as `log2(page_size) << MAP_HUGE_SHIFT` — same scheme the kernel
/// uses for `MAP_HUGE_2MB`, `MAP_HUGE_1GB`, and the Grace 64K-page kernel's
/// 512 MiB hugepages.
fn map_huge_flag_for(page_size: usize) -> Option<libc::c_int> {
    if page_size == 0 || !page_size.is_power_of_two() {
        return None;
    }
    let log2 = page_size.trailing_zeros() as libc::c_int;
    // Sanity range: 64 KiB (2^16) up to 16 GiB (2^34). MAP_HUGE_MASK is
    // 6 bits on Linux, so encoding values above 63 would overflow.
    if !(16..=34).contains(&log2) {
        return None;
    }
    Some(log2 << libc::MAP_HUGE_SHIFT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_up_basic() {
        assert_eq!(round_up(1, 4096), 4096);
        assert_eq!(round_up(4096, 4096), 4096);
        assert_eq!(round_up(4097, 4096), 8192);
        assert_eq!(round_up(0, 4096), 0);
        assert_eq!(round_up(123, 0), 123);
    }

    #[test]
    fn map_huge_flag_covers_common_sizes() {
        assert_eq!(map_huge_flag_for(2 * 1024 * 1024), Some(libc::MAP_HUGE_2MB));
        assert_eq!(
            map_huge_flag_for(1024 * 1024 * 1024),
            Some(libc::MAP_HUGE_1GB)
        );
        // 4 KiB is below the hugepage range.
        assert!(map_huge_flag_for(4096).is_none());
        // Grace 64K-page kernel default `Hugepagesize:` is 512 MiB.
        assert_eq!(
            map_huge_flag_for(512 * 1024 * 1024),
            Some(29 << libc::MAP_HUGE_SHIFT)
        );
        // 32 MiB hugepages exist on some kernels — should round-trip.
        assert_eq!(
            map_huge_flag_for(32 * 1024 * 1024),
            Some(25 << libc::MAP_HUGE_SHIFT)
        );
        // Non-power-of-two is rejected.
        assert!(map_huge_flag_for(3 * 1024 * 1024).is_none());
    }

    #[test]
    fn hugepage_tier_serializes() {
        let json = serde_json::to_string(&HugepageTier::Explicit {
            page_size: 2 * 1024 * 1024,
        })
        .unwrap();
        assert!(json.contains("Explicit"));
        let parsed: HugepageTier = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed,
            HugepageTier::Explicit {
                page_size: 2 * 1024 * 1024
            }
        );
    }

    /// `mmap` an anon region (no hugetlb, no CUDA register) and confirm we
    /// can read + write it. Exercises the mmap path without touching CUDA
    /// or NUMA so it runs on any Linux host.
    #[test]
    fn mmap_anon_basic_rw() {
        let g = mmap_anon(8192).unwrap();
        unsafe {
            for i in 0..8192 {
                std::ptr::write_volatile((g.ptr as *mut u8).add(i), (i % 256) as u8);
            }
            for i in 0..8192 {
                let v = std::ptr::read_volatile((g.ptr as *const u8).add(i));
                assert_eq!(v, (i % 256) as u8);
            }
        }
    }

    /// An allocation without a NUMA node does no `mbind`. The first-touch
    /// and the CUDA host registration still occur.
    #[cfg(feature = "testing-cuda")]
    #[test]
    fn allocate_without_numa_node_uses_os_default_policy() {
        let storage = MmappedPinnedStorage::allocate(MmappedPinnedOptions {
            size: 8192,
            numa_node: None,
            hugepage_mode: HugepageMode::Disabled,
            hugepage_size: None,
            ctx_device_id: 0,
        })
        .unwrap();

        assert_eq!(storage.numa_policy(), NumaPlacementPolicy::OsDefault);
        assert_eq!(storage.numa_node(), None);
        assert_eq!(storage.hugepage_tier(), HugepageTier::None);
        assert!(storage.mapped_len() >= 8192);
        unsafe {
            let ptr = storage.as_ptr() as *mut u8;
            std::ptr::write_volatile(ptr, 0xA5);
            assert_eq!(std::ptr::read_volatile(ptr), 0xA5);
        }
    }

    /// The first node id below 64 that this host does not have.
    fn absent_node() -> NumaNode {
        (0..64u32)
            .find(|n| !std::path::Path::new(&format!("/sys/devices/system/node/node{n}")).exists())
            .map(NumaNode)
            .expect("a node id below 64 that the host lacks")
    }

    fn no_register(_: usize, _: usize) -> Result<()> {
        Ok(())
    }

    /// A real `mbind` to a node the host lacks is refused. The entry keeps the
    /// mapping, records `BindRefused`, and the kernel reports no policy on it.
    #[test]
    fn a_refused_bind_falls_back_to_os_placement() {
        let node = absent_node();
        let placed = map_place_register(
            1 << 16,
            Some(node),
            PlacementDecision::Bind(node),
            mbind_to_node,
            crate::numa::page_placement,
            no_register,
        )
        .expect("a refused bind is not an error");

        match &placed.placement.fallback {
            Some(PlacementFallback::BindRefused { node: n, error }) => {
                assert_eq!(*n, node);
                assert!(error.contains("mbind"), "error: {error}");
            }
            other => panic!("expected BindRefused, got {other:?}"),
        }
        assert_eq!(placed.placement.policy, NumaPlacementPolicy::OsDefault);
        assert_eq!(placed.placement.requested, Some(node));

        let page = crate::numa::page_placement(placed.mmap.ptr).unwrap();
        assert_eq!(page.mode, crate::numa::MemPolicyMode::Default);
        assert!(page.nodes.is_empty(), "nodes: {:?}", page.nodes);
        // The pages were touched, so the kernel reports where they went.
        assert_eq!(placed.placement.observed, Some(page.node));
    }

    /// A registration error comes back as is. It is not relabeled as a
    /// placement fallback, and nothing is retried.
    #[test]
    fn a_registration_error_is_not_a_placement_fallback() {
        let binds = std::cell::Cell::new(0);
        let registers = std::cell::Cell::new(0);
        let result = map_place_register(
            1 << 16,
            Some(NumaNode(0)),
            PlacementDecision::Bind(NumaNode(0)),
            |_, _, _| {
                binds.set(binds.get() + 1);
                Ok(())
            },
            crate::numa::page_placement,
            |_, _| -> Result<()> {
                registers.set(registers.get() + 1);
                Err(StorageError::RegistrationFailed("injected".into()))
            },
        );
        match result {
            Err(StorageError::RegistrationFailed(msg)) => assert_eq!(msg, "injected"),
            Err(other) => panic!("expected the injected error, got {other:?}"),
            Ok(_) => panic!("expected the injected error, got Ok"),
        }
        assert_eq!(binds.get(), 1);
        assert_eq!(registers.get(), 1);
    }

    /// A denied bind and a denied `get_mempolicy` (as Docker's default
    /// seccomp profile can do) still register the mapping.
    #[test]
    fn a_denied_bind_and_denied_query_still_register() {
        let registered = std::cell::Cell::new(false);
        let placed = map_place_register(
            1 << 16,
            Some(NumaNode(0)),
            PlacementDecision::Bind(NumaNode(0)),
            |_, _, _| Err(format!("mbind: {}", std::io::Error::from_raw_os_error(libc::EPERM))),
            |_| Err(std::io::Error::from_raw_os_error(libc::EPERM)),
            |_, _| -> Result<()> {
                registered.set(true);
                Ok(())
            },
        )
        .expect("a denied bind and query still register");

        assert!(registered.get());
        assert!(matches!(
            placed.placement.fallback,
            Some(PlacementFallback::BindRefused { node: NumaNode(0), .. })
        ));
        assert_eq!(placed.placement.policy, NumaPlacementPolicy::OsDefault);
        assert_eq!(placed.placement.observed, None);
    }

    /// An OS-placement decision does no bind and keeps its reason.
    #[test]
    fn an_os_default_decision_skips_the_bind() {
        let placed = map_place_register(
            1 << 16,
            None,
            PlacementDecision::OsDefault(PlacementFallback::NodeUnknown),
            |_, _, _| panic!("no bind for an OS-placement decision"),
            crate::numa::page_placement,
            no_register,
        )
        .unwrap();
        assert_eq!(
            placed.placement,
            HostPlacement {
                requested: None,
                policy: NumaPlacementPolicy::OsDefault,
                fallback: Some(PlacementFallback::NodeUnknown),
                observed: Some(crate::numa::page_placement(placed.mmap.ptr).unwrap().node),
            }
        );
    }

    #[test]
    fn host_placement_serializes() {
        let p = HostPlacement {
            requested: Some(NumaNode(1)),
            policy: NumaPlacementPolicy::OsDefault,
            fallback: Some(PlacementFallback::NodeHasNoCpus(NumaNode(1))),
            observed: Some(NumaNode(0)),
        };
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<HostPlacement>(&json).unwrap(), p);
    }

    /// The memory type the driver reports for a host address.
    #[cfg(feature = "testing-cuda")]
    fn driver_memory_type(addr: usize) -> cudarc::driver::sys::CUmemorytype {
        use cudarc::driver::sys;
        let mut ty: u32 = 0;
        let r = unsafe {
            sys::cuPointerGetAttribute(
                &mut ty as *mut u32 as *mut std::ffi::c_void,
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                addr as sys::CUdeviceptr,
            )
        };
        r.result().expect("cuPointerGetAttribute(MEMORY_TYPE)");
        match ty {
            1 => sys::CUmemorytype::CU_MEMORYTYPE_HOST,
            2 => sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            3 => sys::CUmemorytype::CU_MEMORYTYPE_ARRAY,
            4 => sys::CUmemorytype::CU_MEMORYTYPE_UNIFIED,
            other => panic!("unknown CUmemorytype {other}"),
        }
    }

    /// Each region is registered (the driver says so) and bound to the
    /// device's node (the kernel says so). Reads the facts from the driver
    /// and the kernel, not from the returned object. A missing GPU fails.
    #[cfg(feature = "testing-cuda")]
    #[test]
    fn pinned_host_regions_are_registered_and_bound() {
        use crate::numa::MemPolicyMode;

        let n = crate::numa::get_device_numa_node(0)
            .expect("GPU 0 and its NUMA node must be visible for this test");
        let sizes: Vec<usize> = [3usize << 16, 1 << 16].repeat(5);
        let regions = allocate_pinned_host_regions(&sizes, 0).expect("allocate regions");
        assert_eq!(regions.len(), sizes.len());

        let ctx = crate::device::cuda_context(0).unwrap();
        ctx.bind_to_thread().unwrap();
        let page = system_page_size();

        for (region, &size) in regions.iter().zip(&sizes) {
            let addr = MemoryDescriptor::addr(region);
            let len = region.mapped_len();
            assert!(len >= size);

            for byte in [addr, addr + len - 1] {
                assert_eq!(
                    driver_memory_type(byte),
                    cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_HOST,
                    "byte {byte:#x} of a {len}-byte region"
                );
            }

            let pages = len / page;
            for index in [0, pages / 2, pages - 1] {
                let p = crate::numa::page_placement(addr + index * page).unwrap();
                assert_eq!(p.mode, MemPolicyMode::Bind, "page {index}");
                assert_eq!(p.nodes, vec![n], "page {index}");
                assert_eq!(p.node, n, "page {index}");
            }

            assert_eq!(region.hugepage_tier(), HugepageTier::None);
            assert_eq!(
                region.placement(),
                &HostPlacement {
                    requested: Some(n),
                    policy: NumaPlacementPolicy::Bound(n),
                    fallback: None,
                    observed: Some(n),
                }
            );
            assert_eq!(region.numa_policy(), NumaPlacementPolicy::Bound(n));
            assert_eq!(region.numa_node(), Some(n));
        }
    }

    /// Wall time of one 8 GiB pinned host allocation. Information only.
    #[cfg(feature = "testing-cuda")]
    #[test]
    #[ignore = "allocates 8 GiB; run by hand"]
    fn time_an_8_gib_pinned_host_allocation() {
        let start = std::time::Instant::now();
        let region = allocate_pinned_host(8 << 30, 0).expect("allocate 8 GiB");
        let elapsed = start.elapsed();
        println!(
            "8 GiB pinned host allocation: {:.3} s, placement {:?}",
            elapsed.as_secs_f64(),
            region.placement()
        );
    }
}
