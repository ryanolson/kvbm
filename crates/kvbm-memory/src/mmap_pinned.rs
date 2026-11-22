// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `mmap` + `cuMemHostRegister` host pinned storage with optional NUMA
//! `mbind` and best-effort hugepages.
//!
//! Companion to [`crate::pinned::PinnedStorage`] (which uses `cuMemHostAlloc`,
//! no hugepage / mbind control). Two entries build this storage:
//!
//! - [`allocate_pinned_host`] and [`allocate_pinned_host_regions`]: pinned
//!   host memory for one GPU, such as a G2 pool. Registered `mmap` on base
//!   pages, bound to the GPU's NUMA node when it is safe to, and placed by
//!   the OS otherwise. Each storage records why in
//!   [`MmappedPinnedStorage::placement`], and each region logs one `info`
//!   line. Use this instead of `cuMemHostAlloc` for large host pools: on
//!   GB300, a `cuMemHostAlloc` pool slowed decode steps (p99 97 ms against
//!   19 ms) though no kernel read it.
//! - [`MmappedPinnedStorage::allocate`]: the strict, caller-configured path
//!   for the kvbm-service host-memory pool. It takes a node and a hugepage
//!   mode, and fails if the bind fails. Hugepage-backed pinned memory needs
//!   this path, because the kernel honors `MAP_HUGETLB` only on caller-owned
//!   `mmap` regions.
//!
//! # Placement in `allocate_pinned_host`
//!
//! The device's node comes from [`crate::numa::get_device_numa_node`]. The
//! pages are bound only when NUMA placement is enabled, the node is known,
//! and the node has CPUs. Otherwise the OS places them, and
//! [`HostPlacement::fallback`] names the reason ([`PlacementFallback`]). A
//! node without CPUs is refused because no thread can first-touch there.
//!
//! When the node qualifies, a one-shot thread pinned to it maps, binds
//! (`mbind(MPOL_BIND)`), touches every page, and registers. If the kernel
//! refuses the bind (a cpuset or seccomp profile can), the call does not
//! fail. The same mapping is touched from the same pinned thread, so the
//! OS's default local policy normally still puts the pages on that node.
//! The storage records [`PlacementFallback::BindRefused`]. If the thread
//! could not be pinned either, first touch runs unpinned, and the
//! `BindRefused` error names both failures.
//! [`HostPlacement::observed`] is the node the kernel reports for the first
//! page after the touch, so an `OsDefault` policy never hides where the
//! pages went. The queries are best-effort: if `get_mempolicy` is denied,
//! `observed` is `None` and the allocation goes on.
//!
//! Only a refused bind turns into a fallback. An `mmap`, CUDA context or
//! `cuMemHostRegister` error fails the call and is not retried. A bound
//! node with too little free memory is not checked: `MPOL_BIND` has no
//! fallback once pages fault in, so first touch can end in an OOM kill.
//!
//! [`HostPlacement::policy`] reflects the kernel before registration. After
//! first touch the entry reads the policy of the first, middle and last page.
//! Any reading that is not `MPOL_BIND` on the node turns the bind into
//! [`PlacementFallback::BindRefused`]. A bind with no reading against it
//! stays `Bound(n)`. [`HostPlacement::observed`] is `Some` on a bound region
//! only if both first-page queries answered: the policy showed `MPOL_BIND`
//! on the node, and the node query gave the page's node. Then the bind is
//! kernel-confirmed on the first page. If either query failed, `observed` is
//! `None`, and the record does not claim a confirmation: `mbind` returned
//! success under `MPOL_MF_STRICT`, and no policy reading contradicted it. On GB10 with driver 610.57.04, `cuMemHostRegister` itself
//! calls `mbind(MPOL_PREFERRED_MANY)` on the range with the device's node,
//! whether or not kvbm bound it, and `cuMemHostUnregister` resets the range
//! to `MPOL_DEFAULT`. So after registration the kernel's policy no longer
//! shows kvbm's bind. The pages were already faulted on the bound node and
//! are pinned, so they stay there; [`crate::numa::page_placement`] shows the
//! kernel's current view. The driver checks `get_mempolicy` when it
//! starts. If that call works but `mbind` is denied, `cuMemHostRegister`
//! fails with `CUDA_ERROR_NOT_PERMITTED`, and the entry returns that error.
//! When both are denied (as Docker's default seccomp profile does), the
//! driver skips its own call and registration succeeds.
//!
//! # Pipeline
//!
//! For an allocation with a target node, the caller's thread should already
//! be pinned to a CPU in the cpulist of that node for correct first-touch:
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

use crate::numa::placement::{PageReading, PlacementDecision, decide_placement};
use crate::numa::{MemPolicyMode, NumaNode};
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
    ///
    /// From [`allocate_pinned_host`], `Bound(n)` means `mbind(MPOL_BIND,
    /// MPOL_MF_STRICT)` to `n` returned success and no policy reading
    /// contradicted it. [`Self::observed`] then says whether the kernel's
    /// view of the first page was read in full. `Some` requires both a
    /// first-page policy reading that showed `MPOL_BIND` on `n` and a
    /// successful first-page node query. `None` means at least one of those
    /// two queries failed. The bind may then rest on `mbind`'s answer alone
    /// (no policy reading at all), or on policy readings of the middle and
    /// last page; the record does not say which.
    pub policy: NumaPlacementPolicy,
    /// Why the pages were not bound. `None` when they were bound, or when the
    /// caller asked for OS placement.
    pub fallback: Option<PlacementFallback>,
    /// The node the kernel reports for the first page after first touch.
    /// `None` when the query failed (for example, denied by seccomp) or was
    /// not made. For a bound region, also `None` when the first page's policy
    /// query failed, so `Some` on a bound region always means the first
    /// page's policy showed the bind and its node was read.
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
    if size == 0 {
        return Err(StorageError::AllocationFailed(
            "zero-sized allocations are not supported".into(),
        ));
    }
    // Create the context first: the device's node is read through the CUDA
    // driver, which answers only after it is initialized. This also fails
    // fast when the device is missing.
    let ctx = crate::device::cuda_context(device_id)?;
    let requested = crate::numa::get_device_numa_node(device_id);
    let decision = decide_placement(
        requested,
        crate::numa::is_numa_disabled(),
        crate::numa::topology::get_numa_topology(),
    );

    let storage = match decision {
        PlacementDecision::Bind(node) => crate::numa::worker_pool::run_on_node_thread(
            node,
            format!("pinned-host-node-{}", node.0),
            move |pinned| place_pinned_host(size, ctx, requested, decision, pinned),
        )?,
        PlacementDecision::OsDefault(_) => {
            place_pinned_host(size, ctx, requested, decision, Ok(()))?
        }
    };

    let placement = storage.placement();
    tracing::info!(
        size = storage.mapped_len(),
        device_id,
        requested = ?placement.requested,
        policy = ?placement.policy,
        fallback = ?placement.fallback,
        observed = ?placement.observed,
        "pinned host region"
    );
    Ok(storage)
}

/// Run [`map_place_register`] with the real steps on the calling thread.
fn place_pinned_host(
    size: usize,
    ctx: Arc<CudaContext>,
    requested: Option<NumaNode>,
    decision: PlacementDecision,
    pinned: std::result::Result<(), String>,
) -> Result<MmappedPinnedStorage> {
    let Placed {
        registered,
        mmap,
        placement,
    } = map_place_register(
        size,
        requested,
        decision,
        pinned,
        mbind_to_node,
        crate::numa::placement::read_page,
        |ptr, len| {
            ctx.bind_to_thread().map_err(StorageError::Cuda)?;
            HostRegisterGuard::new(ptr, len, ctx.clone())
        },
    )?;
    Ok(MmappedPinnedStorage {
        register_guard: registered,
        mmap_guard: mmap,
        hugepage_tier: HugepageTier::None,
        placement,
        ctx,
    })
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
/// `pinned` is whether the calling thread was pinned to the node; a pin
/// failure is named in a refused bind's error. Only `bind` can change the
/// placement: its error, or any policy answer (read with `observe` on the
/// first, middle and last page) that does not show `MPOL_BIND` on the node,
/// becomes [`PlacementFallback::BindRefused`]. Each query's answer is kept on
/// its own. A failed query is logged; `observed` is the first page's node if
/// that query answered, and with no policy answer the bind's own result
/// stands. A
/// `register` error fails the call as is. The steps are parameters so tests
/// can inject failures.
pub(crate) fn map_place_register<T, R: Into<PageReading>>(
    size: usize,
    requested: Option<NumaNode>,
    decision: PlacementDecision,
    pinned: std::result::Result<(), String>,
    bind: impl FnOnce(usize, usize, NumaNode) -> std::result::Result<(), String>,
    observe: impl FnMut(usize) -> R,
    register: impl FnOnce(usize, usize) -> Result<T>,
) -> Result<Placed<T>> {
    if size == 0 {
        return Err(StorageError::AllocationFailed(
            "zero-sized allocations are not supported".into(),
        ));
    }
    let mmap = mmap_anon(size)?;

    let (mut policy, mut fallback) = match decision {
        PlacementDecision::Bind(node) => match bind(mmap.ptr, mmap.len, node) {
            Ok(()) => (NumaPlacementPolicy::Bound(node), None),
            Err(error) => {
                let error = match &pinned {
                    Ok(()) => error,
                    Err(pin) => format!("{error}; thread not pinned to node {}: {pin}", node.0),
                };
                tracing::warn!(
                    "mbind(MPOL_BIND) to node {} refused ({}); the OS places {} bytes{}",
                    node.0,
                    error,
                    mmap.len,
                    if pinned.is_err() {
                        ", and first touch runs unpinned"
                    } else {
                        ""
                    }
                );
                (
                    NumaPlacementPolicy::OsDefault,
                    Some(PlacementFallback::BindRefused { node, error }),
                )
            }
        },
        PlacementDecision::OsDefault(reason) => (NumaPlacementPolicy::OsDefault, Some(reason)),
    };

    first_touch(mmap.ptr, mmap.len, HugepageTier::None);

    // Read the kernel's view of the first, middle and last page before
    // registration (the driver may change the policy when it registers).
    // Each answer is kept on its own: a failed query must not hide another
    // query's answer.
    let page = system_page_size();
    let last = (mmap.len - 1) / page;
    let mut observe = observe;
    let readings: Vec<PageReading> = [0, last / 2, last]
        .into_iter()
        .map(|index| observe(mmap.ptr + index * page).into())
        .collect();
    let failed = readings
        .iter()
        .flat_map(|r| [r.policy.as_ref().err(), r.node.as_ref().err()])
        .flatten()
        .next();
    if let Some(e) = failed {
        tracing::warn!(
            "get_mempolicy on the slab at {:#x} failed ({}); the placement is only partly observed",
            mmap.ptr,
            e
        );
    }
    // A bind counts only if the kernel shows it: any policy answer that is
    // not MPOL_BIND on the node refuses it. With no policy answer at all,
    // the bind's own result stands.
    if let NumaPlacementPolicy::Bound(node) = policy
        && let Some((mode, nodes)) = readings
            .iter()
            .filter_map(|r| r.policy.as_ref().ok())
            .find(|(mode, nodes)| *mode != MemPolicyMode::Bind || nodes.as_slice() != [node])
    {
        let error = format!(
            "mbind returned success, but the kernel reports {:?} on nodes {:?}",
            mode, nodes
        );
        tracing::warn!("bind to node {} did not take: {}", node.0, error);
        policy = NumaPlacementPolicy::OsDefault;
        fallback = Some(PlacementFallback::BindRefused { node, error });
    }

    // `observed` is the first page's node. For a bound region it is reported
    // only if the first page's policy was read back (and so showed the bind);
    // otherwise the bind rests on mbind's answer alone, and `None` says so.
    let observed = match (policy, &readings[0].policy) {
        (NumaPlacementPolicy::Bound(_), Err(_)) => None,
        _ => readings[0].node.as_ref().ok().copied(),
    };

    let registered = register(mmap.ptr, mmap.len)?;
    Ok(Placed {
        registered,
        mmap,
        placement: HostPlacement {
            requested,
            policy,
            fallback,
            observed,
        },
    })
}

// =============================================================================
// internal: tier ladder, mmap, mbind, first-touch, register guard
// =============================================================================

/// `MPOL_MF_STRICT` from `<numaif.h>` — kernel returns `EIO` if a page can't
/// be placed on the requested node. Not exposed by `libc`, so define inline.
const MPOL_MF_STRICT: libc::c_uint = 1;

#[derive(Debug)]
pub(crate) struct MmapGuard {
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
    set_range_policy(addr, len, libc::MPOL_BIND, node, MPOL_MF_STRICT)
}

/// `mbind(addr, len, mode, {node}, …, flags)`. The mask is as long as the
/// node id needs, so the kernel, not this helper, decides which nodes exist.
fn set_range_policy(
    addr: usize,
    len: usize,
    mode: libc::c_int,
    node: NumaNode,
    flags: libc::c_uint,
) -> std::result::Result<(), String> {
    if node.0 >= NODE_ID_LIMIT {
        return Err(format!(
            "node id {} is not below {NODE_ID_LIMIT}, beyond any kernel's node count",
            node.0
        ));
    }
    let (mask, maxnode) = node_mask(node);
    // SAFETY: `mask` outlives the call and holds `maxnode - 1` bits.
    let r = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            addr,
            len,
            mode as usize,
            mask.as_ptr() as usize,
            maxnode,
            flags as usize,
        )
    };
    if r != 0 {
        return Err(format!("mbind: {}", std::io::Error::last_os_error()));
    }
    Ok(())
}

/// Node ids at or above this are refused before `mbind`, so a stray id
/// (such as [`NumaNode::UNKNOWN`]) cannot size a huge mask. Linux allows at
/// most 1024 nodes (`CONFIG_NODES_SHIFT` <= 10).
const NODE_ID_LIMIT: u32 = 4096;

/// The node mask for one node, and the `maxnode` to pass with it. The kernel
/// reads `maxnode - 1` bits, so `maxnode` is the mask's bit count plus one
/// (as libnuma passes); one less would drop the top bit.
fn node_mask(node: NumaNode) -> (Vec<u64>, usize) {
    let word = node.0 as usize / 64;
    let mut mask = vec![0u64; word + 1];
    mask[word] = 1u64 << (node.0 % 64);
    let maxnode = mask.len() * 64 + 1;
    (mask, maxnode)
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
            Ok(()),
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

    /// A real `mbind` to a node with CPUs holds through first touch: before
    /// any CUDA registration, the kernel reports `MPOL_BIND` on that node.
    #[test]
    fn a_bind_holds_until_registration() {
        let topology = crate::numa::topology::get_numa_topology().expect("NUMA topology");
        let node = (0..64u32)
            .find(|&n| topology.cpus_for_node(n).is_some_and(|c| !c.is_empty()))
            .map(NumaNode)
            .expect("a node with CPUs");
        let placed = map_place_register(
            1 << 16,
            Some(node),
            PlacementDecision::Bind(node),
            Ok(()),
            mbind_to_node,
            crate::numa::page_placement,
            no_register,
        )
        .unwrap();
        assert_eq!(placed.placement.policy, NumaPlacementPolicy::Bound(node));
        assert_eq!(placed.placement.fallback, None);

        let page_size = system_page_size();
        let pages = placed.mmap.len / page_size;
        for index in [0, pages / 2, pages - 1] {
            let p = crate::numa::page_placement(placed.mmap.ptr + index * page_size).unwrap();
            assert_eq!(p.mode, crate::numa::MemPolicyMode::Bind, "page {index}");
            assert_eq!(p.nodes, vec![node], "page {index}");
            assert_eq!(p.node, node, "page {index}");
        }
        assert_eq!(placed.placement.observed, Some(node));
    }

    /// Node ids of 64 and above reach the kernel: the helper does not refuse
    /// them itself. This host lacks such nodes, so the kernel refuses them
    /// with `EINVAL`.
    #[test]
    fn a_node_id_above_63_reaches_the_kernel() {
        let g = mmap_anon(system_page_size()).unwrap();
        for id in [64u32, 127, 1023] {
            if std::path::Path::new(&format!("/sys/devices/system/node/node{id}")).exists() {
                continue;
            }
            match set_range_policy(g.ptr, g.len, libc::MPOL_PREFERRED, NumaNode(id), 0) {
                Err(e) => assert!(e.contains("Invalid argument"), "node {id}: {e}"),
                Ok(()) => panic!("node {id} is absent, yet mbind accepted it"),
            }
        }
    }

    /// A stray node id is refused before any mask is built.
    #[test]
    fn a_stray_node_id_is_refused_before_mbind() {
        let g = mmap_anon(system_page_size()).unwrap();
        let e = set_range_policy(g.ptr, g.len, libc::MPOL_BIND, NumaNode::UNKNOWN, 0).unwrap_err();
        assert!(e.contains("beyond any kernel's node count"), "error: {e}");
    }

    /// The node mask has exactly the requested bit set, in the word that
    /// holds it, and `maxnode` covers every bit of the mask plus one.
    #[test]
    fn the_node_mask_holds_exactly_the_requested_bit() {
        let (mask, maxnode) = node_mask(NumaNode(0));
        assert_eq!((mask, maxnode), (vec![1], 65));
        let (mask, maxnode) = node_mask(NumaNode(63));
        assert_eq!((mask, maxnode), (vec![1 << 63], 65));
        let (mask, maxnode) = node_mask(NumaNode(64));
        assert_eq!((mask, maxnode), (vec![0, 1], 129));
        let (mask, maxnode) = node_mask(NumaNode(130));
        assert_eq!((mask, maxnode), (vec![0, 0, 1 << 2], 193));
    }

    /// The policy helper hands the kernel every bit of the mask.
    /// `MPOL_PREFERRED` with an empty mask becomes `MPOL_LOCAL`, so a dropped
    /// bit 63 shows as `MPOL_LOCAL`. Node 63 is absent on most hosts; then
    /// the kernel sees the bit and refuses it.
    #[test]
    fn a_policy_mask_reaches_the_highest_node_bit() {
        let g = mmap_anon(system_page_size()).unwrap();
        match set_range_policy(g.ptr, g.len, libc::MPOL_PREFERRED, NumaNode(63), 0) {
            Err(e) => assert!(e.contains("Invalid argument"), "mbind error: {e}"),
            Ok(()) => {
                let p = crate::numa::page_placement(g.ptr).unwrap();
                assert_eq!(p.mode, crate::numa::MemPolicyMode::Preferred);
                assert_eq!(p.nodes, vec![NumaNode(63)]);
            }
        }
    }

    /// A bind step that reports success but leaves no policy in the kernel
    /// is not recorded as bound. The record follows the kernel's view before
    /// registration.
    #[test]
    fn a_bind_the_kernel_does_not_show_is_not_recorded_as_bound() {
        let placed = map_place_register(
            1 << 16,
            Some(NumaNode(0)),
            PlacementDecision::Bind(NumaNode(0)),
            Ok(()),
            |_, _, _| Ok(()),
            crate::numa::page_placement,
            no_register,
        )
        .unwrap();
        assert_eq!(placed.placement.policy, NumaPlacementPolicy::OsDefault);
        match &placed.placement.fallback {
            Some(PlacementFallback::BindRefused { node, error }) => {
                assert_eq!(*node, NumaNode(0));
                assert!(error.contains("kernel reports"), "error: {error}");
            }
            other => panic!("expected BindRefused, got {other:?}"),
        }
    }

    #[test]
    fn a_later_query_error_does_not_erase_the_first_page() {
        let node = NumaNode(0);
        let page = system_page_size();
        let mut first_reading = None;
        let placed = map_place_register(
            4 * page,
            Some(node),
            PlacementDecision::Bind(node),
            Ok(()),
            |_, _, _| Ok(()),
            |addr| {
                if first_reading.is_some() {
                    return Err(std::io::Error::from_raw_os_error(libc::EPERM));
                }
                let reading = crate::numa::page_placement(addr)?;
                assert_eq!(reading.mode, MemPolicyMode::Default);
                first_reading = Some(reading.node);
                Ok(reading)
            },
            no_register,
        )
        .unwrap();

        assert!(first_reading.is_some());
        assert_eq!(
            (placed.placement.policy, placed.placement.observed),
            (NumaPlacementPolicy::OsDefault, first_reading)
        );
        assert!(matches!(
            placed.placement.fallback,
            Some(PlacementFallback::BindRefused { .. })
        ));
    }

    /// The two queries for a page are kept apart: a policy answer that shows
    /// no bind refuses the bind even when the node query fails.
    #[test]
    fn a_failed_node_query_does_not_hide_a_policy_answer() {
        let placed = map_place_register(
            1 << 16,
            Some(NumaNode(0)),
            PlacementDecision::Bind(NumaNode(0)),
            Ok(()),
            |_, _, _| Ok(()),
            |_| PageReading {
                policy: Ok((MemPolicyMode::Default, vec![])),
                node: Err(std::io::Error::from_raw_os_error(libc::EPERM)),
            },
            no_register,
        )
        .unwrap();
        assert_eq!(placed.placement.policy, NumaPlacementPolicy::OsDefault);
        assert!(matches!(
            placed.placement.fallback,
            Some(PlacementFallback::BindRefused { .. })
        ));
        assert_eq!(placed.placement.observed, None);
    }

    /// A bind whose policy was not read back stays `Bound` (mbind's own
    /// answer), but is not reported as confirmed: `observed` is `None` even
    /// though the node query answered.
    #[test]
    fn a_bind_without_a_policy_reading_is_not_reported_as_confirmed() {
        let placed = map_place_register(
            1 << 16,
            Some(NumaNode(0)),
            PlacementDecision::Bind(NumaNode(0)),
            Ok(()),
            |_, _, _| Ok(()),
            |_| PageReading {
                policy: Err(std::io::Error::from_raw_os_error(libc::EPERM)),
                node: Ok(NumaNode(0)),
            },
            no_register,
        )
        .unwrap();
        assert_eq!(
            placed.placement.policy,
            NumaPlacementPolicy::Bound(NumaNode(0))
        );
        assert_eq!(placed.placement.fallback, None);
        assert_eq!(placed.placement.observed, None);
    }

    /// A bound region whose first-page node query failed reports no
    /// observation, even when every policy reading showed the bind.
    #[test]
    fn a_bound_region_without_the_first_node_answer_reports_no_observation() {
        let mut calls = 0;
        let placed = map_place_register(
            1 << 16,
            Some(NumaNode(0)),
            PlacementDecision::Bind(NumaNode(0)),
            Ok(()),
            |_, _, _| Ok(()),
            |_| {
                calls += 1;
                PageReading {
                    policy: Ok((MemPolicyMode::Bind, vec![NumaNode(0)])),
                    node: if calls == 1 {
                        Err(std::io::Error::from_raw_os_error(libc::EPERM))
                    } else {
                        Ok(NumaNode(0))
                    },
                }
            },
            no_register,
        )
        .unwrap();
        assert_eq!(calls, 3);
        assert_eq!(
            placed.placement.policy,
            NumaPlacementPolicy::Bound(NumaNode(0))
        );
        assert_eq!(placed.placement.fallback, None);
        assert_eq!(placed.placement.observed, None);
    }

    /// For a region the OS places, a failed policy query does not hide the
    /// page's node.
    #[test]
    fn a_failed_policy_query_does_not_hide_the_page_node() {
        let placed = map_place_register(
            1 << 16,
            None,
            PlacementDecision::OsDefault(PlacementFallback::NodeUnknown),
            Ok(()),
            |_, _, _| panic!("no bind for an OS-placement decision"),
            |_| PageReading {
                policy: Err(std::io::Error::from_raw_os_error(libc::EPERM)),
                node: Ok(NumaNode(0)),
            },
            no_register,
        )
        .unwrap();
        assert_eq!(placed.placement.policy, NumaPlacementPolicy::OsDefault);
        assert_eq!(placed.placement.observed, Some(NumaNode(0)));
    }

    /// When the thread could not be pinned and the bind is refused, the
    /// record names both failures: first touch ran unpinned.
    #[test]
    fn a_refused_bind_on_an_unpinned_thread_names_both_failures() {
        let placed = map_place_register(
            1 << 16,
            Some(NumaNode(0)),
            PlacementDecision::Bind(NumaNode(0)),
            Err("Failed to set CPU affinity: Operation not permitted".into()),
            |_, _, _| Err("mbind: Operation not permitted (os error 1)".into()),
            crate::numa::page_placement,
            no_register,
        )
        .unwrap();
        assert_eq!(placed.placement.policy, NumaPlacementPolicy::OsDefault);
        match &placed.placement.fallback {
            Some(PlacementFallback::BindRefused { node, error }) => {
                assert_eq!(*node, NumaNode(0));
                assert!(
                    error.contains("mbind: Operation not permitted"),
                    "error: {error}"
                );
                assert!(error.contains("not pinned"), "error: {error}");
                assert!(
                    error.contains("Failed to set CPU affinity"),
                    "error: {error}"
                );
            }
            other => panic!("expected BindRefused, got {other:?}"),
        }
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
            Ok(()),
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
            Ok(()),
            |_, _, _| {
                Err(format!(
                    "mbind: {}",
                    std::io::Error::from_raw_os_error(libc::EPERM)
                ))
            },
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
            Some(PlacementFallback::BindRefused {
                node: NumaNode(0),
                ..
            })
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
            Ok(()),
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

        // The driver answers the device's PCI address only once initialized.
        let ctx = crate::device::cuda_context(0).expect("GPU 0 must be visible for this test");
        ctx.bind_to_thread().unwrap();
        let n = crate::numa::get_device_numa_node(0)
            .expect("GPU 0's NUMA node must be known for this test");
        let sizes: Vec<usize> = [3usize << 16, 1 << 16].repeat(5);
        let regions = allocate_pinned_host_regions(&sizes, 0).expect("allocate regions");
        assert_eq!(regions.len(), sizes.len());
        // Allocation ran the steps on another thread; make the context
        // current here for cuPointerGetAttribute.
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
                // The driver may turn MPOL_BIND into MPOL_PREFERRED_MANY at
                // registration; see the module doc. MPOL_DEFAULT fails.
                println!("region {addr:#x} page {index}: kernel mode {:?}", p.mode);
                assert!(
                    matches!(p.mode, MemPolicyMode::Bind | MemPolicyMode::PreferredMany),
                    "page {index}: mode {:?}",
                    p.mode
                );
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

    /// The first allocation in a process with no CUDA context yet still
    /// finds the device's node and binds to it. Other tests in this binary
    /// create contexts, so the test reruns itself alone in a child process,
    /// where the allocation is the first CUDA call.
    #[cfg(feature = "testing-cuda")]
    #[test]
    fn the_first_allocation_in_a_fresh_process_binds() {
        if !rerun_alone_in_a_child(
            "mmap_pinned::tests::the_first_allocation_in_a_fresh_process_binds",
        ) {
            return;
        }

        // Child: this allocation is the first CUDA call in the process.
        let region = allocate_pinned_host(1 << 16, 0).expect("allocate");
        let n = crate::numa::get_device_numa_node(0)
            .expect("GPU 0 and its NUMA node must be visible for this test");
        assert_eq!(region.placement().requested, Some(n));
        assert_eq!(region.placement().policy, NumaPlacementPolicy::Bound(n));
        let page = crate::numa::page_placement(MemoryDescriptor::addr(&region)).unwrap();
        assert!(
            matches!(
                page.mode,
                crate::numa::MemPolicyMode::Bind | crate::numa::MemPolicyMode::PreferredMany
            ),
            "mode {:?}",
            page.mode
        );
        assert_eq!(page.nodes, vec![n]);
        assert_eq!(page.node, n);
    }

    /// Rerun the test `name` alone in a child process. Returns `true` in the
    /// child, which then runs the test body. In the parent, asserts that the
    /// child passed and returns `false`.
    fn rerun_alone_in_a_child(name: &str) -> bool {
        const CHILD: &str = "KVBM_MEMORY_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            return true;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([name, "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .output()
            .expect("run the child test process");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "child failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        false
    }

    fn bpf_stmt(code: u32, k: u32) -> libc::sock_filter {
        libc::sock_filter {
            code: code as u16,
            jt: 0,
            jf: 0,
            k,
        }
    }

    fn bpf_jeq(k: u32, jt: u8, jf: u8) -> libc::sock_filter {
        libc::sock_filter {
            code: (libc::BPF_JMP | libc::BPF_JEQ | libc::BPF_K) as u16,
            jt,
            jf,
            k,
        }
    }

    const BPF_LOAD_WORD: u32 = libc::BPF_LD | libc::BPF_W | libc::BPF_ABS;
    const BPF_RETURN: u32 = libc::BPF_RET | libc::BPF_K;
    const RET_EPERM: u32 = libc::SECCOMP_RET_ERRNO | libc::EPERM as u32;
    /// Offsets in `struct seccomp_data`: the syscall number, and the low
    /// word of the fifth argument (little-endian).
    const SECCOMP_NR: u32 = 0;
    const SECCOMP_ARG4_LO: u32 = 16 + 4 * 8;

    /// Install a seccomp filter program on this thread; the threads it starts
    /// inherit it, and filters stack. Only for a child test process.
    fn install_seccomp(mut prog: Vec<libc::sock_filter>) {
        let fprog = libc::sock_fprog {
            len: prog.len() as u16,
            filter: prog.as_mut_ptr(),
        };
        // SAFETY: `fprog` and `prog` outlive the calls.
        unsafe {
            assert_eq!(libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
            assert_eq!(
                libc::prctl(
                    libc::PR_SET_SECCOMP,
                    libc::SECCOMP_MODE_FILTER,
                    &fprog as *const libc::sock_fprog
                ),
                0,
                "install seccomp filter: {}",
                std::io::Error::last_os_error()
            );
        }
    }

    /// Fail each syscall in `denied` with `EPERM`, as a container's seccomp
    /// profile can.
    #[cfg(feature = "testing-cuda")]
    fn deny_syscalls(denied: &[libc::c_long]) {
        let mut prog = vec![bpf_stmt(BPF_LOAD_WORD, SECCOMP_NR)];
        for &nr in denied {
            prog.push(bpf_jeq(nr as u32, 0, 1));
            prog.push(bpf_stmt(BPF_RETURN, RET_EPERM));
        }
        prog.push(bpf_stmt(BPF_RETURN, libc::SECCOMP_RET_ALLOW));
        install_seccomp(prog);
    }

    /// The public diagnostic stops at a failed policy query: the node query,
    /// which faults the page in, does not run. Only the policy query
    /// (`get_mempolicy` with flags `MPOL_F_ADDR`) is denied, in a child
    /// process, so the result does not depend on the order of the tests.
    #[test]
    fn a_denied_policy_query_does_not_fault_the_page() {
        if !rerun_alone_in_a_child(
            "mmap_pinned::tests::a_denied_policy_query_does_not_fault_the_page",
        ) {
            return;
        }
        install_seccomp(vec![
            bpf_stmt(BPF_LOAD_WORD, SECCOMP_NR),
            bpf_jeq(libc::SYS_get_mempolicy as u32, 0, 3),
            bpf_stmt(BPF_LOAD_WORD, SECCOMP_ARG4_LO),
            bpf_jeq(2, 0, 1), // MPOL_F_ADDR alone: the policy query
            bpf_stmt(BPF_RETURN, RET_EPERM),
            bpf_stmt(BPF_RETURN, libc::SECCOMP_RET_ALLOW),
        ]);

        let page = system_page_size();
        let g = mmap_anon(page).unwrap();
        let resident = || {
            let mut vec = 0u8;
            // SAFETY: the mapping is live and one page long; `vec` holds one
            // byte per page.
            assert_eq!(
                unsafe { libc::mincore(g.ptr as *mut libc::c_void, page, &mut vec) },
                0
            );
            vec & 1 == 1
        };
        assert!(!resident(), "a fresh anonymous page is not resident");
        let err = crate::numa::page_placement(g.ptr).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(libc::EPERM));
        assert!(!resident(), "the failed diagnostic faulted the page in");
        // The node query itself is still allowed, and does fault the page in.
        crate::numa::placement::read_page(g.ptr).node.unwrap();
        assert!(resident());
    }

    /// Through the public entry: with `mbind` and `get_mempolicy` denied (as
    /// Docker's default seccomp profile can), the region still registers,
    /// and the record says the bind was refused and nothing was observed.
    #[cfg(feature = "testing-cuda")]
    #[test]
    fn a_denied_bind_and_query_through_the_entry_still_register() {
        if !rerun_alone_in_a_child(
            "mmap_pinned::tests::a_denied_bind_and_query_through_the_entry_still_register",
        ) {
            return;
        }
        deny_syscalls(&[libc::SYS_mbind, libc::SYS_get_mempolicy]);
        assert!(
            crate::numa::page_placement(0).is_err_and(|e| e.raw_os_error() == Some(libc::EPERM))
        );

        let region = allocate_pinned_host(1 << 16, 0).expect("allocate");
        let n = crate::numa::get_device_numa_node(0).expect("GPU 0's NUMA node");
        let placement = region.placement();
        assert_eq!(placement.requested, Some(n));
        assert_eq!(placement.policy, NumaPlacementPolicy::OsDefault);
        match &placement.fallback {
            Some(PlacementFallback::BindRefused { node, error }) => {
                assert_eq!(*node, n);
                assert!(error.contains("Operation not permitted"), "error: {error}");
            }
            other => panic!("expected BindRefused, got {other:?}"),
        }
        assert_eq!(placement.observed, None);

        region.ctx().bind_to_thread().unwrap();
        let addr = MemoryDescriptor::addr(&region);
        for byte in [addr, addr + region.mapped_len() - 1] {
            assert_eq!(
                driver_memory_type(byte),
                cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_HOST
            );
        }
    }

    /// Through the public entry: with the thread pin and the bind both
    /// refused, the region still registers, and the record names both
    /// failures.
    #[cfg(feature = "testing-cuda")]
    #[test]
    fn a_denied_pin_and_bind_through_the_entry_record_both() {
        if !rerun_alone_in_a_child(
            "mmap_pinned::tests::a_denied_pin_and_bind_through_the_entry_record_both",
        ) {
            return;
        }
        // The filters go in a fixed order, so the run is deterministic.
        // `mbind` and `get_mempolicy` are denied from the start, together, as
        // Docker's default profile does; the driver then skips its own `mbind`
        // at registration. The driver sets thread affinity while it creates a
        // context, so `sched_setaffinity` is denied after that, standing in
        // for a cpuset that excludes the node.
        deny_syscalls(&[libc::SYS_mbind, libc::SYS_get_mempolicy]);
        crate::device::cuda_context(0).expect("GPU 0 must be visible for this test");
        let n = crate::numa::get_device_numa_node(0).expect("GPU 0's NUMA node");
        deny_syscalls(&[libc::SYS_sched_setaffinity]);

        let region = allocate_pinned_host(1 << 16, 0).expect("allocate");
        let placement = region.placement();
        assert_eq!(placement.policy, NumaPlacementPolicy::OsDefault);
        match &placement.fallback {
            Some(PlacementFallback::BindRefused { node, error }) => {
                assert_eq!(*node, n);
                assert!(
                    error.contains("mbind: Operation not permitted"),
                    "error: {error}"
                );
                assert!(error.contains("not pinned"), "error: {error}");
            }
            other => panic!("expected BindRefused, got {other:?}"),
        }
        assert_eq!(placement.observed, None);

        region.ctx().bind_to_thread().unwrap();
        assert_eq!(
            driver_memory_type(MemoryDescriptor::addr(&region)),
            cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_HOST
        );
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
