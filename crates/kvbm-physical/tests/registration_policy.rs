// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing-kvbm")]

use anyhow::Result;
use kvbm_memory::nixl::{NixlAgent, register_with_nixl};
use kvbm_memory::{MemoryDescriptor, SystemStorage, create_buffer};
use kvbm_physical::layout::{LayoutConfig, PhysicalLayout};

fn layout_config() -> Result<LayoutConfig> {
    Ok(LayoutConfig::builder()
        .num_blocks(2)
        .num_layers(3)
        .outer_dim(2)
        .page_size(4)
        .inner_dim(8)
        .dtype_width_bytes(2)
        .build()?)
}

#[test]
fn default_builder_requires_native_registration() -> Result<()> {
    let agent = NixlAgent::new("registration-policy-default")?;
    let config = layout_config()?;
    let storage = SystemStorage::new(config.required_bytes())?;
    let result = PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous()
        .with_memory_regions(vec![storage])
        .and_then(|builder| builder.build());

    let error = result.expect_err("default builder requires NIXL registration");
    assert_eq!(
        error.to_string(),
        "failed to register memory with NIXL agent registration-policy-default: Backend error occurred"
    );
    Ok(())
}

#[test]
fn default_builder_preserves_a_native_pre_registered_region() -> Result<()> {
    let agent = NixlAgent::with_backends("registration-policy-registered", &["UCX"])?;
    let config = layout_config()?;
    let storage = SystemStorage::new(config.required_bytes())?;
    let registered = register_with_nixl(storage, &agent, None)?;
    assert!(registered.owns_registration());
    let descriptor = registered.nixl_descriptor().unwrap();
    let layout = PhysicalLayout::builder(agent.clone())
        .with_config(config)
        .fully_contiguous()
        .with_registered_regions(vec![create_buffer(registered)])?
        .build()?;
    let region = &layout.layout().memory_regions()[0];
    let actual = region.nixl_descriptor().unwrap();
    assert_eq!(actual.addr, descriptor.addr);
    assert_eq!(actual.size, descriptor.size);
    assert_eq!(actual.mem_type, descriptor.mem_type);
    assert_eq!(actual.device_id, descriptor.device_id);
    Ok(())
}
