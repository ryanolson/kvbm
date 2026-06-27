// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector metadata carried from the leader to the workers each scheduler step.
//!
//! [`KvConnectorMetadata`] is the serde wire type holding the forward-pass /
//! intra-pass transfer plan; [`IntraPassLoad`] and [`IntraPassStore`] carry the
//! G2↔G1 block-id pairs for the load and store directions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::InstanceId;
use kvbm_common::BlockId;
use velo::EventHandle;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvConnectorMetadata {
    pub iteration: usize,

    /// Map of worker instance_id to event handle for forward pass completion.
    /// Workers trigger their corresponding event in clear_connector_metadata.
    pub foward_pass_completion_events: Option<HashMap<InstanceId, EventHandle>>,

    /// This will hold the G2 source and G1 destination block_ids
    pub intra_pass_load: Option<IntraPassLoad>,

    /// This will hold the G1 source and G2 destination block_ids
    pub intra_pass_store: Option<IntraPassStore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntraPassLoad {
    pub g2_src_block_ids: Vec<BlockId>,
    pub g1_dst_block_ids: Vec<BlockId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntraPassStore {
    pub g1_src_block_ids: Vec<BlockId>,
    pub g2_dst_block_ids: Vec<BlockId>,
}

impl KvConnectorMetadata {
    pub fn new(iteration: usize) -> Self {
        Self {
            iteration,
            foward_pass_completion_events: None,
            intra_pass_load: None,
            intra_pass_store: None,
        }
    }

    pub fn with_events(mut self, events: HashMap<InstanceId, EventHandle>) -> Self {
        self.foward_pass_completion_events = Some(events);
        self
    }

    pub fn summary(&self) -> String {
        let intra_pass_load_num_blocks = self
            .intra_pass_load
            .as_ref()
            .map(|l| l.g1_dst_block_ids.len())
            .unwrap_or(0);
        let will_signal_completion = self.foward_pass_completion_events.is_some();

        format!(
            "Iteration: {}, Intra pass load: {}, Forward pass completion events: {:?}",
            self.iteration, intra_pass_load_num_blocks, will_signal_completion
        )
    }

    pub fn should_bind(&self) -> bool {
        // self.foward_pass_completion_events.is_some() || self.intra_pass_load.is_some()
        true
    }
}
