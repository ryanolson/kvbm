// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test fixtures for the logical crate.
//!
//! Enable the `testing` feature to use the public fixtures and measure
//! block-store mutex acquisitions in integration tests.

pub mod config;

// when testing is enabled with out
mod blocks;
mod managers;
mod metered_tracker;
mod pools;
mod sequences;
mod token_blocks;

pub use metered_tracker::MeteredFrequencyTracker;

#[cfg(feature = "testing")]
pub use crate::pools::store::count_store_lock_acquisitions;

// ============================================================================
// Canonical test metadata types
// ============================================================================

/// Primary test metadata type used across logical tests.
/// Replaces: `TestMetadata` (registry), `TestData` (pools), `TestBlockData` (manager)
#[derive(Debug, Clone, PartialEq)]
pub struct TestMeta {
    pub value: u64,
}

/// Multi-type registry test metadata A
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataA(pub u32);

/// Multi-type registry test metadata B
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataB(pub String);

/// Multi-type registry test metadata C (unit struct)
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataC;

// ============================================================================
// Constants
// ============================================================================

/// Standard salt value for test token blocks.
/// Standardizes mixed salt values (42 in pools/manager, 1337 in registry).
pub const TEST_SALT: dynamo_tokens::SaltHash = 42;

// ============================================================================
// Re-exports from submodules
// ============================================================================

// pub items — usable by downstream crates via the testing feature
pub use managers::{
    create_test_manager, create_test_manager_metered, create_test_manager_with_backend,
    create_test_manager_with_block_size, create_test_manager_with_default_reset_on_release,
};
pub use sequences::BlockSequenceBuilder;
pub use token_blocks::{
    create_iota_token_block, create_test_token_block, sequential_tokens, tokens_for_id,
};

// pub(crate) items — internal helpers
#[cfg(test)]
pub(crate) use blocks::{block_id_and_hash, hash_for_tokens};
#[cfg(test)]
pub(crate) use pools::TestPoolSetupBuilder;
