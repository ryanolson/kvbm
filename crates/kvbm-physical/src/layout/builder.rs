// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed builder for constructing [`PhysicalLayout`](crate::layout::PhysicalLayout)
//! instances with strongly-typed configuration, layout selection, and memory provisioning.
//!
//! The builder enforces the three steps required to materialize a physical layout:
//! 1. Provide a [`LayoutConfig`]
//! 2. Select a concrete layout (fully contiguous or layer separate)
//! 3. Specify memory backing (either by allocating or by supplying existing regions)
//!
//! The public builder requires NIXL registration. It accepts an agent and either
//! registers memory that implements [`NixlCompatible`] or accepts regions that
//! already carry a descriptor.

use crate::layout::physical::PhysicalLayout;

use super::{
    BlockDimension, FullyContiguousLayout, KvBlockLayout, LayerSeparateLayout, Layout,
    LayoutConfig, MemoryDescriptor, RaggedLayerSeparateLayout,
    physical::{LayoutRegistration, NixlMetadata},
};
use crate::layout::serialize::BlockFormat;

use anyhow::{Result, anyhow, bail};
use kvbm_memory::{
    Buffer, DeviceStorage, DiskStorage, OffsetBuffer, PinnedStorage, StorageKind, SystemStorage,
    allocate_pinned_host, create_buffer,
    nixl::{MemType, NixlAgent, NixlDescriptor, RegisterError, register_with_nixl},
    prelude::{NixlCompatible, RegisteredView},
};
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;

const REGION_ALIGNMENT: usize = 512;

/// Layout selection exposed by the builder.
#[derive(Debug, Clone)]
pub enum LayoutKind {
    FullyContiguous,
    LayerSeparate { block_dim: BlockDimension },
    RaggedLayerSeparate { bytes_per_layer_block: Vec<usize> },
}

/// How the builder allocates pinned host memory.
///
/// The builder defaults to [`Self::CuMemHostAlloc`]. Changing the default
/// requires transfer measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PinnedAllocator {
    /// Allocate page-locked host memory with CUDA's `cuMemHostAlloc`.
    #[default]
    CuMemHostAlloc,
    /// Map host memory with `mmap` and pin it with CUDA's `cuMemHostRegister`.
    RegisteredMmap,
}

/// Allocation strategies for builder-managed memory.
#[derive(Debug, Clone)]
enum AllocationKind {
    System,
    /// Page-locked host memory.
    Pinned {
        device_id: Option<u32>,
        allocator: PinnedAllocator,
    },
    Device {
        device_id: u32,
    },
    Disk {
        path: Option<PathBuf>,
    },
}

/// Memory provisioning plan (either provided regions or an allocation request).
#[derive(Debug, Clone)]
enum MemoryPlan {
    Provided(Vec<MemoryEntry>),
    Allocate(AllocationKind),
}

#[derive(Clone, Copy)]
enum RegistrationPolicy {
    Required,
    // The policy compiles in every build. Only test fixtures can select it.
    #[cfg_attr(not(all(test, feature = "testing-kvbm")), allow(dead_code))]
    LocalOnly,
}

struct RegistrationConfig {
    agent: Option<NixlAgent>,
    agent_name: String,
    policy: RegistrationPolicy,
}

/// Memory tenancy captured during the build process.
#[derive(Debug, Clone)]
struct MemoryEntry {
    region: Buffer,
    descriptor: Option<NixlDescriptor>,
    /// The NIXL registration that covers `region`.
    ///
    /// This differs from `descriptor` when one allocation is registered once
    /// and then sliced into several regions. NIXL matches a partial metadata
    /// export against the registration, not against a slice of it, so the
    /// export needs this descriptor rather than `descriptor`.
    registration: Option<NixlDescriptor>,
}

impl MemoryEntry {
    fn new(region: Buffer, descriptor: Option<NixlDescriptor>) -> Self {
        let registration = descriptor.clone();
        Self {
            region,
            descriptor,
            registration,
        }
    }

    fn with_registration(
        region: Buffer,
        descriptor: Option<NixlDescriptor>,
        registration: Option<NixlDescriptor>,
    ) -> Self {
        Self {
            region,
            descriptor,
            registration,
        }
    }

    fn validate_registration(mut self, policy: RegistrationPolicy) -> Result<Self> {
        if self.descriptor.is_none() {
            self.descriptor = self.region.nixl_descriptor();
            self.registration = self.descriptor.clone();
        }

        match (policy, self.descriptor.is_some()) {
            (RegistrationPolicy::Required, false) => bail!(
                "memory region {} is not registered with NIXL",
                self.region.addr()
            ),
            (RegistrationPolicy::LocalOnly, true) => {
                bail!("local-only layout cannot use a NIXL-registered memory region")
            }
            _ => Ok(self),
        }
    }
}

/// Marker types for the builder state machine.
pub struct NoConfig;
pub struct HasConfig;

pub struct NoLayout;
pub struct HasLayout;

pub struct NoMemory;
pub struct HasMemory;

/// Default builder state type alias.
pub type PhysicalLayoutBuilderDefault = PhysicalLayoutBuilder<NoConfig, NoLayout, NoMemory>;

/// Typed builder enforcing configuration, layout selection, and memory provisioning phases.
pub struct PhysicalLayoutBuilder<C, L, M> {
    registration: RegistrationConfig,
    config: Option<LayoutConfig>,
    layout_kind: Option<LayoutKind>,
    memory_plan: Option<MemoryPlan>,
    /// Optional block-layout override. When `None`, layouts default to
    /// `KvBlockLayout::Unknown`.
    block_layout: Option<KvBlockLayout>,
    _config: PhantomData<C>,
    _layout: PhantomData<L>,
    _memory: PhantomData<M>,
}

impl PhysicalLayoutBuilder<NoConfig, NoLayout, NoMemory> {
    /// Create a new builder in its initial state.
    pub fn new(agent: NixlAgent) -> Self {
        let agent_name = agent.name().to_string();
        Self {
            registration: RegistrationConfig {
                agent: Some(agent),
                agent_name,
                policy: RegistrationPolicy::Required,
            },
            config: None,
            layout_kind: None,
            memory_plan: None,
            block_layout: None,
            _config: PhantomData,
            _layout: PhantomData,
            _memory: PhantomData,
        }
    }

    /// Create a local-only builder for layout tests that run without NIXL.
    #[cfg(all(test, feature = "testing-kvbm"))]
    pub(crate) fn new_local(agent_name: impl Into<String>) -> Self {
        Self {
            registration: RegistrationConfig {
                agent: None,
                agent_name: agent_name.into(),
                policy: RegistrationPolicy::LocalOnly,
            },
            config: None,
            layout_kind: None,
            memory_plan: None,
            block_layout: None,
            _config: PhantomData,
            _layout: PhantomData,
            _memory: PhantomData,
        }
    }

    #[cfg(all(test, feature = "testing-kvbm"))]
    pub(crate) fn local_only(mut self) -> Self {
        self.registration.policy = RegistrationPolicy::LocalOnly;
        self
    }
}

impl<C, L, M> PhysicalLayoutBuilder<C, L, M> {
    fn into_parts(
        self,
    ) -> (
        RegistrationConfig,
        Option<LayoutConfig>,
        Option<LayoutKind>,
        Option<MemoryPlan>,
        Option<KvBlockLayout>,
    ) {
        (
            self.registration,
            self.config,
            self.layout_kind,
            self.memory_plan,
            self.block_layout,
        )
    }

    fn from_parts<C2, L2, M2>(
        registration: RegistrationConfig,
        config: Option<LayoutConfig>,
        layout_kind: Option<LayoutKind>,
        memory_plan: Option<MemoryPlan>,
        block_layout: Option<KvBlockLayout>,
    ) -> PhysicalLayoutBuilder<C2, L2, M2> {
        PhysicalLayoutBuilder {
            registration,
            config,
            layout_kind,
            memory_plan,
            block_layout,
            _config: PhantomData,
            _layout: PhantomData,
            _memory: PhantomData,
        }
    }

    /// Set the `KvBlockLayout` describing per-block dimension ordering.
    ///
    /// Optional: layouts default to `KvBlockLayout::Unknown` when not
    /// specified. Callable in any state since this is purely additional
    /// information the layout will use at construction time.
    ///
    /// For `LayerSeparate`, only operational layouts (NHD/HND) plus
    /// `Unknown` are accepted; other variants will be rejected at
    /// `build()`.
    pub fn with_block_layout(mut self, kv_block_layout: KvBlockLayout) -> Self {
        self.block_layout = Some(kv_block_layout);
        self
    }
}

impl<L, M> PhysicalLayoutBuilder<NoConfig, L, M> {
    /// Attach the [`LayoutConfig`] required to size the layout and allocations.
    pub fn with_config(self, config: LayoutConfig) -> PhysicalLayoutBuilder<HasConfig, L, M> {
        let (registration, _config, layout_kind, memory_plan, block_layout) = self.into_parts();
        PhysicalLayoutBuilder::<HasConfig, L, M>::from_parts(
            registration,
            Some(config),
            layout_kind,
            memory_plan,
            block_layout,
        )
    }
}

impl<M> PhysicalLayoutBuilder<HasConfig, NoLayout, M> {
    /// Select the fully contiguous layout variant.
    pub fn fully_contiguous(self) -> PhysicalLayoutBuilder<HasConfig, HasLayout, M> {
        let (registration, config, _layout, memory_plan, block_layout) = self.into_parts();
        PhysicalLayoutBuilder::<HasConfig, HasLayout, M>::from_parts(
            registration,
            config,
            Some(LayoutKind::FullyContiguous),
            memory_plan,
            block_layout,
        )
    }

    /// Select the layer-separate layout variant with the provided block dimension ordering.
    pub fn layer_separate(
        self,
        block_dim: BlockDimension,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, M> {
        let (registration, config, _layout, memory_plan, block_layout) = self.into_parts();
        PhysicalLayoutBuilder::<HasConfig, HasLayout, M>::from_parts(
            registration,
            config,
            Some(LayoutKind::LayerSeparate { block_dim }),
            memory_plan,
            block_layout,
        )
    }

    /// Select independently sized contiguous segments for each logical block.
    pub fn ragged_layer_separate(
        self,
        bytes_per_layer_block: Vec<usize>,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, M> {
        let (registration, config, _layout, memory_plan, block_layout) = self.into_parts();
        PhysicalLayoutBuilder::<HasConfig, HasLayout, M>::from_parts(
            registration,
            config,
            Some(LayoutKind::RaggedLayerSeparate {
                bytes_per_layer_block,
            }),
            memory_plan,
            block_layout,
        )
    }
}

impl PhysicalLayoutBuilder<HasConfig, HasLayout, NoMemory> {
    fn set_memory_plan(
        self,
        plan: MemoryPlan,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
        let (registration, config, layout_kind, _memory, block_layout) = self.into_parts();
        PhysicalLayoutBuilder::<HasConfig, HasLayout, HasMemory>::from_parts(
            registration,
            config,
            layout_kind,
            Some(plan),
            block_layout,
        )
    }

    pub fn allocate_system(self) -> PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
        self.set_memory_plan(MemoryPlan::Allocate(AllocationKind::System))
    }

    /// Allocate pinned (page-locked) host memory.
    ///
    /// # Arguments
    /// * `device_id` - If `Some(id)`, enables NUMA-aware allocation on the GPU's NUMA node
    ///   (disable with `DYN_MEMORY_DISABLE_NUMA=1`). If `None`, uses direct allocation.
    pub fn allocate_pinned(
        self,
        device_id: Option<u32>,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
        self.allocate_pinned_with(device_id, PinnedAllocator::CuMemHostAlloc)
    }

    /// Allocate pinned host memory with the selected [`PinnedAllocator`].
    ///
    /// `CuMemHostAlloc` calls `cuMemHostAlloc` and stays the default until
    /// transfer measurements support a change. `RegisteredMmap` uses `mmap`
    /// and `cuMemHostRegister` with the device's CUDA context. It tries to
    /// place pages on the device's NUMA node.
    ///
    /// # Errors
    /// [`PhysicalLayoutBuilder::build`] fails if `RegisteredMmap` has no `device_id`.
    pub fn allocate_pinned_with(
        self,
        device_id: Option<u32>,
        allocator: PinnedAllocator,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
        self.set_memory_plan(MemoryPlan::Allocate(AllocationKind::Pinned {
            device_id,
            allocator,
        }))
    }

    /// Allocate device memory on the specified CUDA device (or the context device if `None`).
    pub fn allocate_device(
        self,
        device_id: u32,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
        self.set_memory_plan(MemoryPlan::Allocate(AllocationKind::Device { device_id }))
    }

    /// Allocate disk-backed storage. When `path` is `None`, a temporary file is used.
    pub fn allocate_disk(
        self,
        path: Option<PathBuf>,
    ) -> PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
        self.set_memory_plan(MemoryPlan::Allocate(AllocationKind::Disk { path }))
    }

    /// Use existing NIXL-compatible memory regions supplied by the caller.
    pub fn with_memory_regions<S>(
        self,
        regions: Vec<S>,
    ) -> Result<PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory>>
    where
        S: MemoryDescriptor + NixlCompatible + 'static,
    {
        let (registration, config, layout_kind, _memory, block_layout) = self.into_parts();
        let entries = register_existing_regions(&registration, regions)?;
        Ok(
            PhysicalLayoutBuilder::<HasConfig, HasLayout, HasMemory>::from_parts(
                registration,
                config,
                layout_kind,
                Some(MemoryPlan::Provided(entries)),
                block_layout,
            ),
        )
    }

    /// Use pre-registered memory regions (already wrapped in `Arc<dyn MemoryDescriptor>`).
    ///
    /// All regions must already expose a NIXL descriptor.
    pub fn with_registered_regions(
        self,
        regions: Vec<Buffer>,
    ) -> Result<PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory>> {
        let entries = regions
            .into_iter()
            .enumerate()
            .map(|(index, region)| {
                let descriptor = region.nixl_descriptor().ok_or_else(|| {
                    anyhow!(
                        "provided memory region at index {} is not NIXL registered",
                        index
                    )
                })?;
                Ok(MemoryEntry::new(region, Some(descriptor)))
            })
            .collect::<Result<Vec<_>>>()?;

        let (registration, config, layout_kind, _memory, block_layout) = self.into_parts();
        Ok(
            PhysicalLayoutBuilder::<HasConfig, HasLayout, HasMemory>::from_parts(
                registration,
                config,
                layout_kind,
                Some(MemoryPlan::Provided(entries)),
                block_layout,
            ),
        )
    }

    /// Register external KV cache tensors with NIXL for RDMA access.
    ///
    /// This is the **CRITICAL** step that enables remote GPU-to-GPU transfers.
    /// Each tensor's memory is wrapped in `ExternalDeviceMemory` and registered
    /// with NIXL.
    ///
    /// # Arguments
    /// * `tensors` - KV cache tensors from vLLM (one per layer). All tensors must:
    ///   - Be on the same CUDA device
    ///   - Be contiguous in memory
    ///   - Match the selected layout's required region count and sizes. Standard
    ///     layer-separated layouts use one same-shaped tensor per layer;
    ///     ragged layouts intentionally accept different shapes.
    ///
    /// # Requirements
    /// - The NIXL agent must be registered with an RDMA-capable backend
    /// - The external framework (vLLM) must keep the tensors valid while registered
    ///
    /// # Example
    /// ```ignore
    /// let physical_layout = PhysicalLayoutBuilder::new(nixl_agent)
    ///     .with_config(layout_config)
    ///     .layer_separate(block_dim)
    ///     .with_external_device_regions(kv_tensors)?  // NIXL registration here
    ///     .build()?;
    /// ```
    pub fn with_external_device_regions(
        self,
        tensors: Vec<Arc<dyn kvbm_memory::TensorDescriptor>>,
    ) -> Result<PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory>> {
        use kvbm_memory::TensorDescriptorExt;

        if tensors.is_empty() {
            bail!("with_external_device_regions requires at least one tensor");
        }

        let (registration, config, layout_kind, _memory, block_layout) = self.into_parts();

        let mut entries = Vec::with_capacity(tensors.len());

        for (index, tensor) in tensors.into_iter().enumerate() {
            // Verify the tensor is on a CUDA device
            if tensor.cuda_device_id().is_none() {
                bail!("tensor at index {} is not on a CUDA device", index);
            }

            // Register tensor with NIXL for RDMA
            // Arc<dyn TensorDescriptor> implements both MemoryDescriptor and NixlCompatible,
            // so we can register it directly. This is the critical step that enables
            // remote GPU-to-GPU transfers via UCX backend.
            let entry = register_storage(tensor, &registration).map_err(|e| {
                anyhow!(
                    "failed to register tensor {} with NIXL (UCX backend required for VRAM): {}",
                    index,
                    e
                )
            })?;

            entries.push(entry);
        }

        Ok(
            PhysicalLayoutBuilder::<HasConfig, HasLayout, HasMemory>::from_parts(
                registration,
                config,
                layout_kind,
                Some(MemoryPlan::Provided(entries)),
                block_layout,
            ),
        )
    }
}

impl PhysicalLayoutBuilder<HasConfig, HasLayout, HasMemory> {
    /// Finalize the builder, constructing the [`PhysicalLayout`].
    pub fn build(self) -> Result<PhysicalLayout> {
        let (registration, config, layout_kind, memory_plan, block_layout) = self.into_parts();

        let config = config.ok_or_else(|| anyhow!("layout config missing despite type state"))?;
        let layout_kind =
            layout_kind.ok_or_else(|| anyhow!("layout kind missing despite type state"))?;
        let memory_plan =
            memory_plan.ok_or_else(|| anyhow!("memory plan missing despite type state"))?;

        let required_sizes = compute_allocation_sizes(&config, &layout_kind)?;
        let entries = resolve_memory_plan(&registration, memory_plan, &required_sizes)?;
        validate_memory_sizes(&entries, &required_sizes)?;
        let kind = derive_storage_kind(&entries)?;
        let metadata = derive_layout_registration(&registration, &entries)?;
        let registrations = if matches!(metadata, LayoutRegistration::Registered(_)) {
            collect_registrations(&entries)
        } else {
            Vec::new()
        };

        let kv_block_layout = block_layout.unwrap_or(KvBlockLayout::Unknown);

        let layout: Arc<dyn Layout> = match layout_kind {
            LayoutKind::FullyContiguous => {
                let entry = entries.first().ok_or_else(|| {
                    anyhow!("fully contiguous layout requires a single memory region")
                })?;
                let layout = FullyContiguousLayout::new_with_format(
                    config.clone(),
                    entry.region.clone(),
                    BlockFormat::default(),
                    kv_block_layout,
                )?;
                Arc::new(layout)
            }
            LayoutKind::LayerSeparate { block_dim } => {
                let regions: Vec<Buffer> =
                    entries.iter().map(|entry| entry.region.clone()).collect();
                let layout = LayerSeparateLayout::new_with_block_layout(
                    config.clone(),
                    regions,
                    block_dim,
                    kv_block_layout,
                )?;
                Arc::new(layout)
            }
            LayoutKind::RaggedLayerSeparate {
                bytes_per_layer_block,
            } => {
                let regions = entries.iter().map(|entry| entry.region.clone()).collect();
                Arc::new(RaggedLayerSeparateLayout::new(
                    config.clone(),
                    regions,
                    bytes_per_layer_block,
                    kv_block_layout,
                )?)
            }
        };

        Ok(PhysicalLayout::new_local(layout, kind, metadata).with_registrations(registrations))
    }
}

/// The distinct NIXL registrations that back these regions, in address order.
fn collect_registrations(entries: &[MemoryEntry]) -> Vec<NixlDescriptor> {
    let mut registrations: Vec<NixlDescriptor> = Vec::new();
    for entry in entries {
        let Some(registration) = entry.registration.as_ref() else {
            continue;
        };
        if registrations.iter().any(|held| {
            held.addr == registration.addr
                && held.size == registration.size
                && held.mem_type == registration.mem_type
                && held.device_id == registration.device_id
        }) {
            continue;
        }
        registrations.push(registration.clone());
    }
    registrations.sort_by_key(|registration| registration.addr);
    registrations
}

fn register_existing_regions<S>(
    registration: &RegistrationConfig,
    regions: Vec<S>,
) -> Result<Vec<MemoryEntry>>
where
    S: MemoryDescriptor + NixlCompatible + 'static,
{
    regions
        .into_iter()
        .map(|region| register_storage(region, registration))
        .collect()
}

fn resolve_memory_plan(
    registration: &RegistrationConfig,
    plan: MemoryPlan,
    sizes: &[usize],
) -> Result<Vec<MemoryEntry>> {
    match plan {
        MemoryPlan::Provided(entries) => {
            if entries.len() != sizes.len() {
                bail!(
                    "provided memory count ({}) does not match required allocations ({})",
                    entries.len(),
                    sizes.len()
                );
            }
            entries
                .into_iter()
                .map(|entry| entry.validate_registration(registration.policy))
                .collect()
        }
        MemoryPlan::Allocate(strategy) => allocate_regions(registration, strategy, sizes),
    }
}

fn allocate_regions(
    registration: &RegistrationConfig,
    strategy: AllocationKind,
    sizes: &[usize],
) -> Result<Vec<MemoryEntry>> {
    if sizes.is_empty() {
        return Ok(Vec::new());
    }

    let reserve_size = total_allocation_size(sizes, REGION_ALIGNMENT)?;

    let base_entry = match strategy {
        AllocationKind::System => allocate_system_entry(reserve_size, registration)?,
        AllocationKind::Pinned {
            device_id,
            allocator,
        } => allocate_pinned_entry(reserve_size, registration, device_id, allocator)?,
        AllocationKind::Device { device_id } => {
            allocate_device_entry(reserve_size, registration, device_id)?
        }
        AllocationKind::Disk { path } => allocate_disk_entry(reserve_size, registration, path)?,
    };

    create_offset_entries(base_entry, sizes, REGION_ALIGNMENT)
}

fn allocate_system_entry(size: usize, registration: &RegistrationConfig) -> Result<MemoryEntry> {
    let storage = SystemStorage::new(size)
        .map_err(|e| anyhow!("failed to allocate system memory ({size} bytes): {e}"))?;
    register_storage(storage, registration)
}

fn allocate_pinned_entry(
    size: usize,
    registration: &RegistrationConfig,
    device_id: Option<u32>,
    allocator: PinnedAllocator,
) -> Result<MemoryEntry> {
    match allocator {
        PinnedAllocator::CuMemHostAlloc => {
            let storage = PinnedStorage::new_for_device(size, device_id)
                .map_err(|e| anyhow!("failed to allocate pinned memory ({size} bytes): {e}"))?;
            register_storage(storage, registration)
        }
        PinnedAllocator::RegisteredMmap => {
            let device_id = device_id
                .ok_or_else(|| anyhow!("the registered-mmap allocator needs a CUDA device"))?;
            let storage = allocate_pinned_host(size, device_id).map_err(|e| {
                anyhow!(
                    "failed to allocate registered-mmap pinned memory ({size} bytes) on device {device_id}: {e}"
                )
            })?;
            register_storage(storage, registration)
        }
    }
}

fn allocate_device_entry(
    size: usize,
    registration: &RegistrationConfig,
    device_id: u32,
) -> Result<MemoryEntry> {
    let storage = DeviceStorage::new(size, device_id).map_err(|e| {
        anyhow!("failed to allocate device memory ({size} bytes) on device {device_id}: {e}")
    })?;
    register_storage(storage, registration)
}

fn allocate_disk_entry(
    size: usize,
    registration: &RegistrationConfig,
    path: Option<PathBuf>,
) -> Result<MemoryEntry> {
    let storage = if let Some(path) = path {
        DiskStorage::new_at(&path, size)
            .map_err(|e| anyhow!("failed to allocate disk storage at {}: {e}", path.display()))?
    } else {
        DiskStorage::new(size).map_err(|e| anyhow!("failed to allocate disk storage: {e}"))?
    };
    register_storage(storage, registration)
}

fn register_storage<S>(storage: S, registration: &RegistrationConfig) -> Result<MemoryEntry>
where
    S: MemoryDescriptor + NixlCompatible + 'static,
{
    if matches!(registration.policy, RegistrationPolicy::LocalOnly) {
        return Ok(MemoryEntry::new(create_buffer(storage), None));
    }

    let agent = registration
        .agent
        .as_ref()
        .ok_or_else(|| anyhow!("NIXL registration policy requires an agent"))?;
    match register_with_nixl(storage, agent, None) {
        Ok(registered) => {
            let descriptor = registered.descriptor();
            let region: Buffer = create_buffer(registered);
            Ok(MemoryEntry::new(region, Some(descriptor)))
        }
        Err(RegisterError { source, .. }) => {
            bail!(
                "failed to register memory with NIXL agent {}: {source}",
                registration.agent_name
            )
        }
    }
}

fn create_offset_entries(
    base_entry: MemoryEntry,
    sizes: &[usize],
    alignment: usize,
) -> Result<Vec<MemoryEntry>> {
    if sizes.is_empty() {
        return Ok(Vec::new());
    }

    let base_region = base_entry.region;
    let base_descriptor = base_entry.descriptor;
    let base_addr = base_region.addr();
    let base_len = base_region.size();

    let mut entries = Vec::with_capacity(sizes.len());
    let mut offset = 0usize;

    for (index, &size) in sizes.iter().enumerate() {
        let region = if index == 0 && offset == 0 && size == base_len && sizes.len() == 1 {
            base_region.clone()
        } else {
            let view = OffsetBuffer::new(base_region.clone(), offset, size)
                .map_err(|e| anyhow!("failed to create offset region: {e}"))?;
            create_buffer(view)
        };

        let descriptor = base_descriptor
            .as_ref()
            .map(|descriptor| derive_descriptor(descriptor, offset, size))
            .transpose()?;

        entries.push(MemoryEntry::with_registration(
            region,
            descriptor,
            base_descriptor.clone(),
        ));

        offset = offset
            .checked_add(size)
            .ok_or_else(|| anyhow!("offset computation overflow"))?;

        if index + 1 < sizes.len() && alignment > 1 {
            let current_addr = base_addr
                .checked_add(offset)
                .ok_or_else(|| anyhow!("address computation overflow"))?;
            let aligned_addr = align_up(current_addr, alignment)?;
            offset = aligned_addr
                .checked_sub(base_addr)
                .ok_or_else(|| anyhow!("alignment subtraction overflow"))?;
        }
    }

    if offset > base_len {
        bail!(
            "allocated base region ({base_len} bytes) is insufficient for {offset} bytes with padding"
        );
    }

    Ok(entries)
}

fn derive_descriptor(base: &NixlDescriptor, offset: usize, size: usize) -> Result<NixlDescriptor> {
    let mut descriptor = base.clone();
    descriptor.size = size;
    if descriptor.mem_type != MemType::File {
        descriptor.addr = descriptor
            .addr
            .checked_add(offset as u64)
            .ok_or_else(|| anyhow!("descriptor address overflow"))?;
    }
    Ok(descriptor)
}

fn compute_allocation_sizes(config: &LayoutConfig, kind: &LayoutKind) -> Result<Vec<usize>> {
    match kind {
        LayoutKind::FullyContiguous => {
            let factors = [
                config.num_blocks,
                config.num_layers,
                config.outer_dim,
                config.page_size,
                config.inner_dim,
                config.dtype_width_bytes,
            ];
            let total = mul_chain(&factors)?;
            Ok(vec![total])
        }
        LayoutKind::LayerSeparate { .. } => {
            let factors = [
                config.num_blocks,
                config.outer_dim,
                config.page_size,
                config.inner_dim,
                config.dtype_width_bytes,
            ];
            let per_layer = mul_chain(&factors)?;
            Ok(vec![per_layer; config.num_layers])
        }
        LayoutKind::RaggedLayerSeparate {
            bytes_per_layer_block,
        } => bytes_per_layer_block
            .iter()
            .map(|bytes| {
                config.num_blocks.checked_mul(*bytes).ok_or_else(|| {
                    anyhow!("allocation size overflow during ragged layout computation")
                })
            })
            .collect(),
    }
}

fn mul_chain(factors: &[usize]) -> Result<usize> {
    factors.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value)
            .ok_or_else(|| anyhow!("allocation size overflow during layout computation"))
    })
}

fn total_allocation_size(sizes: &[usize], alignment: usize) -> Result<usize> {
    if sizes.is_empty() {
        return Ok(0);
    }

    let mut total = *sizes
        .first()
        .ok_or_else(|| anyhow!("allocation requires at least one region"))?;

    for size in sizes.iter().skip(1) {
        total = total
            .checked_add(*size)
            .ok_or_else(|| anyhow!("allocation size overflow during aggregation"))?;
        if alignment > 1 {
            total = total
                .checked_add(alignment - 1)
                .ok_or_else(|| anyhow!("allocation alignment padding overflow"))?;
        }
    }

    Ok(total)
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    if alignment <= 1 {
        return Ok(value);
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| anyhow!("alignment overflow"))
    }
}

fn validate_memory_sizes(entries: &[MemoryEntry], required: &[usize]) -> Result<()> {
    for (entry, &required_size) in entries.iter().zip(required.iter()) {
        if entry.region.size() < required_size {
            bail!(
                "memory region too small: required {} bytes, available {} bytes",
                required_size,
                entry.region.size()
            );
        }
    }
    Ok(())
}

fn derive_storage_kind(entries: &[MemoryEntry]) -> Result<StorageKind> {
    let first = entries
        .first()
        .ok_or_else(|| anyhow!("no memory regions available to determine storage location"))?;
    let first_kind = first.region.storage_kind();

    for entry in entries.iter().skip(1) {
        let kind = entry.region.storage_kind();
        if kind != first_kind {
            bail!(
                "all memory regions must share the same storage location (found {:?} and {:?})",
                first_kind,
                kind
            );
        }
    }

    Ok(first_kind)
}

fn derive_layout_registration(
    registration: &RegistrationConfig,
    entries: &[MemoryEntry],
) -> Result<LayoutRegistration> {
    let agent_name = registration.agent_name.clone();
    match registration.policy {
        RegistrationPolicy::Required => {
            let descriptor = entries
                .iter()
                .find_map(|entry| entry.descriptor.as_ref())
                .ok_or_else(|| anyhow!("memory entries missing NIXL registration metadata"))?;
            Ok(LayoutRegistration::Registered(NixlMetadata::new(
                agent_name,
                descriptor.mem_type,
                descriptor.device_id,
            )))
        }
        RegistrationPolicy::LocalOnly => Ok(LayoutRegistration::LocalOnly { agent_name }),
    }
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::super::{BlockDimension, LayoutConfig};
    use super::*;

    use kvbm_memory::{Buffer, MemoryDescriptor, StorageKind};
    use std::any::Any;

    #[derive(Debug)]
    struct TestRegisteredRegion {
        data: Vec<u8>,
        kind: StorageKind,
        descriptor: NixlDescriptor,
    }

    impl TestRegisteredRegion {
        fn new(size: usize, kind: StorageKind, mem_type: MemType, device_id: u64) -> Self {
            let data = vec![0u8; size];
            let addr = data.as_ptr() as u64;
            let descriptor = NixlDescriptor {
                addr,
                size,
                mem_type,
                device_id,
            };
            Self {
                data,
                kind,
                descriptor,
            }
        }
    }

    impl MemoryDescriptor for TestRegisteredRegion {
        fn addr(&self) -> usize {
            self.data.as_ptr() as usize
        }

        fn size(&self) -> usize {
            self.data.len()
        }

        fn storage_kind(&self) -> StorageKind {
            self.kind
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
            Some(self.descriptor.clone())
        }
    }

    impl NixlCompatible for TestRegisteredRegion {
        fn nixl_params(&self) -> (*const u8, usize, MemType, u64) {
            (
                self.data.as_ptr(),
                self.data.len(),
                self.descriptor.mem_type,
                self.descriptor.device_id,
            )
        }
    }

    fn make_layout_config() -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(2)
            .num_layers(3)
            .outer_dim(2)
            .page_size(4)
            .inner_dim(8)
            .dtype_width_bytes(2)
            .build()
            .unwrap()
    }

    fn fully_contiguous_size(cfg: &LayoutConfig) -> usize {
        cfg.num_blocks
            * cfg.num_layers
            * cfg.outer_dim
            * cfg.page_size
            * cfg.inner_dim
            * cfg.dtype_width_bytes
    }

    fn per_layer_size(cfg: &LayoutConfig) -> usize {
        cfg.num_blocks * cfg.outer_dim * cfg.page_size * cfg.inner_dim * cfg.dtype_width_bytes
    }

    #[test]
    fn registration_policy_default_requires_native_registration() -> Result<()> {
        let agent = NixlAgent::new("registration-policy-default")?;
        let cfg = make_layout_config();
        let storage = SystemStorage::new(fully_contiguous_size(&cfg))?;
        let result = PhysicalLayout::builder(agent)
            .with_config(cfg)
            .fully_contiguous()
            .with_memory_regions(vec![storage])
            .and_then(PhysicalLayoutBuilder::build);

        let error = result.expect_err("default builder requires NIXL registration");
        assert_eq!(
            error.to_string(),
            "failed to register memory with NIXL agent registration-policy-default: Backend error occurred"
        );
        Ok(())
    }

    #[test]
    fn registration_policy_local_only_has_no_peer_descriptor_or_route() -> Result<()> {
        use crate::transfer::TransferCapabilities;
        use crate::transfer::strategy::{TransferPlan, TransferStrategy, select_strategy};

        let agent = NixlAgent::new("registration-policy-local")?;
        let cfg = make_layout_config();
        let storage = SystemStorage::new(fully_contiguous_size(&cfg))?;
        let local = PhysicalLayout::builder(agent.clone())
            .local_only()
            .with_config(cfg.clone())
            .fully_contiguous()
            .with_memory_regions(vec![storage])?
            .build()?;
        assert!(local.registrations().is_empty());
        assert!(
            local.layout().memory_regions()[0]
                .nixl_descriptor()
                .is_none()
        );
        assert_eq!(
            local.to_descriptor().unwrap_err().to_string(),
            "local-only layout has no NIXL registration"
        );

        let caps = TransferCapabilities::default();
        assert_eq!(
            select_strategy(&local, &local, &agent.name(), &caps)?,
            TransferPlan::Direct(TransferStrategy::Memcpy)
        );

        let peer_region = create_buffer(TestRegisteredRegion::new(
            fully_contiguous_size(&cfg),
            StorageKind::System,
            MemType::Dram,
            0,
        ));
        let peer = PhysicalLayout::builder(NixlAgent::new("registration-policy-peer")?)
            .with_config(cfg)
            .fully_contiguous()
            .with_registered_regions(vec![peer_region])?
            .build()?;
        for (src, dst) in [(&local, &peer), (&peer, &local)] {
            assert_eq!(
                select_strategy(src, dst, &agent.name(), &caps)
                    .unwrap_err()
                    .to_string(),
                "local-only layout has no NIXL registration"
            );
        }
        Ok(())
    }

    #[test]
    fn registration_policy_local_only_rejects_registered_regions() -> Result<()> {
        let agent = NixlAgent::new("registration-policy-local-registered")?;
        let cfg = make_layout_config();
        let registered_region = || {
            TestRegisteredRegion::new(
                fully_contiguous_size(&cfg),
                StorageKind::System,
                MemType::Dram,
                0,
            )
        };
        let builder = || {
            PhysicalLayout::builder(agent.clone())
                .local_only()
                .with_config(cfg.clone())
                .fully_contiguous()
        };

        let supplied = builder()
            .with_registered_regions(vec![create_buffer(registered_region())])
            .and_then(PhysicalLayoutBuilder::build);
        let compatible = builder()
            .with_memory_regions(vec![registered_region()])
            .and_then(PhysicalLayoutBuilder::build);
        for result in [supplied, compatible] {
            assert_eq!(
                result
                    .expect_err("local-only layouts must not expose registered regions")
                    .to_string(),
                "local-only layout cannot use a NIXL-registered memory region"
            );
        }
        Ok(())
    }

    #[test]
    fn registration_policy_local_only_rejects_same_agent_disk_routes() -> Result<()> {
        use crate::transfer::TransferCapabilities;
        use crate::transfer::strategy::{TransferPlan, TransferStrategy, select_strategy};

        let agent = NixlAgent::new("registration-policy-local-disk")?;
        let cfg = make_layout_config();
        let region = create_buffer(TestRegisteredRegion::new(
            fully_contiguous_size(&cfg),
            StorageKind::Disk(0),
            MemType::File,
            0,
        ));
        let disk = PhysicalLayout::builder(agent.clone())
            .with_config(cfg.clone())
            .fully_contiguous()
            .with_registered_regions(vec![region])?
            .build()?;
        let local_host = PhysicalLayout::builder(agent.clone())
            .local_only()
            .with_config(cfg.clone())
            .fully_contiguous()
            .allocate_system()
            .build()?;
        let local_disk = PhysicalLayout::builder(agent.clone())
            .local_only()
            .with_config(cfg)
            .fully_contiguous()
            .allocate_disk(None)
            .build()?;

        let caps = TransferCapabilities::default();
        assert!(matches!(
            select_strategy(&disk, &disk, &agent.name(), &caps)?,
            TransferPlan::TwoHop {
                first: TransferStrategy::NixlReadFlipped,
                ..
            }
        ));
        for (src, dst) in [
            (&local_host, &disk),
            (&disk, &local_host),
            (&disk, &local_disk),
        ] {
            assert_eq!(
                select_strategy(src, dst, &agent.name(), &caps)
                    .expect_err("local-only layouts must not select NIXL")
                    .to_string(),
                "local-only layout has no NIXL registration"
            );
        }
        Ok(())
    }

    #[test]
    fn registration_policy_required_preserves_a_supplied_descriptor() -> Result<()> {
        let agent = NixlAgent::new("registration-policy-pre-registered")?;
        let cfg = make_layout_config();
        let region = TestRegisteredRegion::new(
            fully_contiguous_size(&cfg),
            StorageKind::System,
            MemType::Dram,
            0,
        );
        let expected = region.descriptor.clone();
        let layout = PhysicalLayout::builder(agent)
            .with_config(cfg)
            .fully_contiguous()
            .with_memory_regions(vec![region])?
            .build()?;
        let [registration] = layout.registrations() else {
            bail!("expected the original registration");
        };
        assert_eq!(registration.addr, expected.addr);
        assert_eq!(registration.size, expected.size);
        assert_eq!(registration.mem_type, expected.mem_type);
        assert_eq!(registration.device_id, expected.device_id);
        assert_eq!(layout.to_descriptor()?.memory_descriptors().len(), 1);
        Ok(())
    }

    #[test]
    fn registered_mmap_requires_a_device() {
        let agent = NixlAgent::new("builder-test-mmap-device").expect("failed to create agent");
        let error = PhysicalLayoutBuilder::new(agent)
            .with_config(make_layout_config())
            .fully_contiguous()
            .allocate_pinned_with(None, PinnedAllocator::RegisteredMmap)
            .build()
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "the registered-mmap allocator needs a CUDA device"
        );
    }

    #[test]
    fn builds_fully_contiguous_from_registered_regions() {
        let agent = NixlAgent::new("builder-test-fully").expect("failed to create agent");
        let cfg = make_layout_config();

        let required = fully_contiguous_size(&cfg);
        let region = create_buffer(TestRegisteredRegion::new(
            required,
            StorageKind::System,
            MemType::Dram,
            0,
        ));

        let physical = PhysicalLayoutBuilder::new(agent.clone())
            .with_config(cfg.clone())
            .fully_contiguous()
            .with_registered_regions(vec![region])
            .expect("registered regions accepted")
            .build()
            .expect("builder should succeed");

        assert_eq!(physical.location(), StorageKind::System);
        assert!(physical.layout().as_ref().is_fully_contiguous());
        assert_eq!(physical.layout().config().num_blocks, cfg.num_blocks);
        assert_eq!(physical.layout().config().num_layers, cfg.num_layers);

        let metadata = physical.nixl_metadata().unwrap();
        assert_eq!(metadata.agent_name(), agent.name());
        assert_eq!(metadata.mem_type(), MemType::Dram);
    }

    #[test]
    fn builds_layer_separate_from_registered_regions() {
        let agent = NixlAgent::new("builder-test-layer").expect("failed to create agent");
        let cfg = make_layout_config();

        let per_layer = per_layer_size(&cfg);
        let regions: Vec<Buffer> = (0..cfg.num_layers)
            .map(|_| {
                create_buffer(TestRegisteredRegion::new(
                    per_layer,
                    StorageKind::System,
                    MemType::Dram,
                    0,
                ))
            })
            .collect();

        let physical = PhysicalLayoutBuilder::new(agent.clone())
            .with_config(cfg.clone())
            .layer_separate(BlockDimension::BlockIsFirstDim)
            .with_registered_regions(regions)
            .expect("registered layer regions accepted")
            .build()
            .expect("builder should succeed");

        assert_eq!(physical.location(), StorageKind::System);
        assert!(!physical.layout().as_ref().is_fully_contiguous());
        assert_eq!(physical.layout().config().num_layers, cfg.num_layers);

        let metadata = physical.nixl_metadata().unwrap();
        assert_eq!(metadata.agent_name(), agent.name());
        assert_eq!(metadata.mem_type(), MemType::Dram);
    }

    /// `with_block_layout(NHD)` propagates to the resulting fully-contiguous
    /// layout's `block_layout()`, replacing the default `Unknown`.
    #[test]
    fn with_block_layout_threads_kv_block_layout_to_fully_contiguous() {
        let agent = NixlAgent::new("builder-block-layout-fc").expect("agent");
        let cfg = make_layout_config();
        let region = create_buffer(TestRegisteredRegion::new(
            fully_contiguous_size(&cfg),
            StorageKind::System,
            MemType::Dram,
            0,
        ));

        let physical = PhysicalLayoutBuilder::new(agent)
            .with_config(cfg)
            .fully_contiguous()
            .with_block_layout(KvBlockLayout::OperationalNHD)
            .with_registered_regions(vec![region])
            .expect("registered region")
            .build()
            .expect("build");

        assert_eq!(
            physical.layout().block_layout(),
            KvBlockLayout::OperationalNHD
        );
    }

    /// Default (no `with_block_layout` call) leaves the layout as `Unknown`,
    /// matching pre-refactor behaviour.
    #[test]
    fn without_with_block_layout_defaults_to_unknown() {
        let agent = NixlAgent::new("builder-block-layout-default").expect("agent");
        let cfg = make_layout_config();
        let region = create_buffer(TestRegisteredRegion::new(
            fully_contiguous_size(&cfg),
            StorageKind::System,
            MemType::Dram,
            0,
        ));

        let physical = PhysicalLayoutBuilder::new(agent)
            .with_config(cfg)
            .fully_contiguous()
            .with_registered_regions(vec![region])
            .expect("registered region")
            .build()
            .expect("build");

        assert_eq!(physical.layout().block_layout(), KvBlockLayout::Unknown);
    }

    /// Layer-separate accepts only operational/unknown layouts; passing a
    /// universal layout is rejected at `build()`.
    #[test]
    fn layer_separate_rejects_universal_block_layout() {
        let agent = NixlAgent::new("builder-block-layout-ls-reject").expect("agent");
        let cfg = make_layout_config();
        let regions: Vec<Buffer> = (0..cfg.num_layers)
            .map(|_| {
                create_buffer(TestRegisteredRegion::new(
                    per_layer_size(&cfg),
                    StorageKind::System,
                    MemType::Dram,
                    0,
                ))
            })
            .collect();

        let result = PhysicalLayoutBuilder::new(agent)
            .with_config(cfg)
            .layer_separate(BlockDimension::BlockIsFirstDim)
            .with_block_layout(KvBlockLayout::Universal)
            .with_registered_regions(regions)
            .expect("registered regions")
            .build();

        let err = result.unwrap_err().to_string();
        assert!(err.contains("operational"), "got: {err}");
    }
}

// fn context_device_id(ctx: &TransferContext) -> u32 {
//     ctx.stream().context().ordinal() as u32
// }

#[cfg(test)]
mod registration_tests {
    use super::*;

    /// One allocation registered once and then sliced must export as one
    /// registration. NIXL matches a partial metadata export against the
    /// registration, and it refuses a slice of one with `NIXL_ERR_NOT_FOUND`.
    #[test]
    fn sliced_regions_report_the_one_registration_that_backs_them() -> Result<()> {
        let storage = SystemStorage::new(4096)?;
        let base_addr = storage.addr() as u64;
        let base = MemoryEntry::new(
            create_buffer(storage),
            Some(NixlDescriptor {
                addr: base_addr,
                size: 4096,
                mem_type: MemType::Dram,
                device_id: 0,
            }),
        );

        let entries = create_offset_entries(base, &[256, 256], 1)?;
        assert_eq!(entries.len(), 2);
        assert_ne!(
            entries[0].descriptor.as_ref().map(|d| d.size),
            Some(4096),
            "each region must still describe its own slice"
        );

        let registrations = collect_registrations(&entries);
        assert_eq!(registrations.len(), 1);
        assert_eq!(registrations[0].addr, base_addr);
        assert_eq!(registrations[0].size, 4096);
        assert_eq!(registrations[0].mem_type, MemType::Dram);
        Ok(())
    }

    /// An unregistered region contributes nothing, so the export never names
    /// memory that NIXL does not hold.
    #[test]
    fn an_unregistered_region_contributes_no_registration() -> Result<()> {
        let storage = SystemStorage::new(1024)?;
        let entry = MemoryEntry::new(create_buffer(storage), None);
        assert!(collect_registrations(&[entry]).is_empty());
        Ok(())
    }
}
