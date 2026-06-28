// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable identities shared by logical and physical KV resources.

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Stable model-local identity for one logical KV resource.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Encode,
    Decode,
    Serialize,
    Deserialize,
)]
pub struct LogicalResourceId(pub u16);
