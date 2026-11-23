// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing-kvbm")]

use std::any::Any;
use std::sync::Arc;

use anyhow::{Context, Result};
use kvbm_memory::nixl::{MemType, NixlAgent, NixlDescriptor, NixlRegistered, register_with_nixl};
use kvbm_memory::{MemoryDescriptor, StorageKind, SystemStorage, TensorDescriptor};
use kvbm_physical::layout::{BlockDimension, LayoutConfig, PhysicalLayout};

const DEVICE: u32 = 3;
const TENSOR_BYTES: usize = 16;

#[derive(Debug)]
struct TestTensor {
    backing: SystemStorage,
    kind: StorageKind,
    registration: Option<NixlDescriptor>,
}

impl TestTensor {
    fn device(registered: bool) -> Result<Arc<Self>> {
        // Host allocation stands in for device memory. The tests never copy
        // to it, and the device cases use an agent with no backends.
        let backing = SystemStorage::new(4096)?;
        let registration = registered.then(|| NixlDescriptor {
            addr: backing.addr() as u64,
            // A registration can cover more bytes than this tensor uses.
            size: backing.size(),
            mem_type: MemType::Vram,
            device_id: u64::from(DEVICE),
        });
        Ok(Arc::new(Self {
            backing,
            kind: StorageKind::Device(DEVICE),
            registration,
        }))
    }
}

impl MemoryDescriptor for TestTensor {
    fn addr(&self) -> usize {
        self.backing.addr()
    }

    fn size(&self) -> usize {
        TENSOR_BYTES
    }

    fn storage_kind(&self) -> StorageKind {
        self.kind
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        self.registration.clone()
    }
}

impl TensorDescriptor for TestTensor {
    fn shape(&self) -> &[usize] {
        &[2, 1, 4]
    }

    fn stride(&self) -> &[usize] {
        &[4, 4, 1]
    }

    fn element_size(&self) -> usize {
        2
    }
}

fn build_device_layout(name: &str, tensor: Arc<dyn TensorDescriptor>) -> Result<PhysicalLayout> {
    let agent = NixlAgent::new(name)?;
    let config = LayoutConfig::builder()
        .num_blocks(2)
        .num_layers(1)
        .outer_dim(1)
        .page_size(2)
        .inner_dim(2)
        .dtype_width_bytes(2)
        .build()?;
    PhysicalLayout::builder(agent)
        .with_config(config)
        .layer_separate(BlockDimension::BlockIsFirstDim)
        .with_external_device_regions(vec![tensor])?
        .build()
}

fn assert_descriptor(actual: NixlDescriptor, expected: &NixlDescriptor) {
    assert_eq!(actual.addr, expected.addr);
    assert_eq!(actual.size, expected.size);
    assert_eq!(actual.mem_type, expected.mem_type);
    assert_eq!(actual.device_id, expected.device_id);
}

fn assert_forwards_tensor(wrapped: &impl MemoryDescriptor, owner: &TestTensor) -> Result<()> {
    let concrete = wrapped
        .as_any()
        .downcast_ref::<TestTensor>()
        .context("the wrapper must expose the concrete tensor")?;
    assert!(std::ptr::eq(concrete, owner));
    assert_descriptor(
        wrapped.nixl_descriptor().context("forwarded descriptor")?,
        owner.registration.as_ref().context("test descriptor")?,
    );
    Ok(())
}

#[test]
fn builder_keeps_external_registration_and_tensor_owner() -> Result<()> {
    let tensor = TestTensor::device(true)?;
    let owner = Arc::downgrade(&tensor);
    let layout = build_device_layout("borrowed-device-tensor", tensor)?;
    let [region] = layout.layout().memory_regions() else {
        anyhow::bail!("expected one tensor region");
    };
    let wrapper = region
        .as_any()
        .downcast_ref::<NixlRegistered<Arc<dyn TensorDescriptor>>>()
        .context("the production builder must retain the tensor wrapper")?;
    assert!(wrapper.is_registered());
    assert!(!wrapper.owns_registration());

    let tensor = owner.upgrade().context("the layout must own the tensor")?;
    assert_forwards_tensor(wrapper.storage(), &tensor)?;
    let expected = tensor.registration.as_ref().context("test descriptor")?;
    assert_eq!(region.size(), TENSOR_BYTES);
    assert!(expected.size > region.size());
    assert_descriptor(
        region.nixl_descriptor().context("layout descriptor")?,
        expected,
    );
    drop(tensor);
    drop(layout);
    assert!(
        owner.upgrade().is_none(),
        "the layout must release its tensor"
    );
    Ok(())
}

#[test]
fn builder_requires_registration_for_an_ordinary_device_tensor() -> Result<()> {
    let tensor = TestTensor::device(false)?;
    let error = build_device_layout("ordinary-device-tensor", tensor)
        .expect_err("registration requires a backend when the tensor supplies no descriptor");
    assert!(
        error
            .to_string()
            .contains("failed to register memory with NIXL agent"),
        "{error:#}"
    );
    Ok(())
}

#[test]
fn arc_tensor_forwards_concrete_identity_and_descriptor() -> Result<()> {
    let owner = TestTensor::device(true)?;
    let wrapped: Arc<dyn TensorDescriptor> = owner.clone();
    assert_forwards_tensor(&wrapped, &owner)
}

#[test]
fn arc_send_sync_tensor_forwards_concrete_identity_and_descriptor() -> Result<()> {
    let owner = TestTensor::device(true)?;
    let wrapped: Arc<dyn TensorDescriptor + Send + Sync> = owner.clone();
    assert_forwards_tensor(&wrapped, &owner)
}

#[test]
fn ordinary_dram_tensor_gets_an_owned_native_registration() -> Result<()> {
    let agent = NixlAgent::with_backends("ordinary-dram-tensor", &["UCX"])?;
    let tensor: Arc<dyn TensorDescriptor> = Arc::new(TestTensor {
        backing: SystemStorage::new(4096)?,
        kind: StorageKind::System,
        registration: None,
    });
    let addr = tensor.addr() as u64;
    let registered = register_with_nixl(tensor, &agent, None)?;
    assert!(registered.owns_registration());
    assert_descriptor(
        registered.nixl_descriptor().context("native descriptor")?,
        &NixlDescriptor {
            addr,
            size: TENSOR_BYTES,
            mem_type: MemType::Dram,
            device_id: 0,
        },
    );
    Ok(())
}
