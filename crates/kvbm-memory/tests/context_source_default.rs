// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing-cuda")]

//! This regression test verifies the default CUDA context source.
//!
//! A default device allocation reserves the context source.
//! A later provider installation must fail.
//!
//! The test uses CUDA and runs in its own integration-test process.

use kvbm_memory::{CudaContextProvider, DeviceStorage, set_cuda_context_provider};

#[test]
fn default_cache_use_occupies_the_slot_too() {
    // The allocation reserves the default source before it creates a CUDA context.
    // The test only checks that a later provider cannot replace that source.
    let _ = DeviceStorage::new(4096, 0);

    let unreachable_provider: Box<CudaContextProvider> = Box::new(|_| unreachable!());
    let result = set_cuda_context_provider(unreachable_provider);
    assert!(
        result.is_err(),
        "installing a provider after a default-path allocation attempt must be rejected"
    );
}
