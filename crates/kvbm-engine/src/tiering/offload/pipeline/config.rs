// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for block and object offload pipelines.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use anyhow::ensure;
use kvbm_common::LogicalResourceId;
use kvbm_logical::blocks::BlockMetadata;

use super::super::batch::BatchConfig;
use super::super::pending::PendingTracker;
use super::super::policy::OffloadPolicy;
use crate::object::ObjectLockManager;

/// Settings that all pipeline destinations use.
#[derive(Clone)]
pub(crate) struct PipelineBaseConfig<Src: BlockMetadata> {
    pub(crate) policies: Vec<Arc<dyn OffloadPolicy<Src>>>,
    pub(crate) batch_config: BatchConfig,
    pub(crate) policy_timeout: Duration,
    pub(crate) sweep_interval: Duration,
    pub(crate) skip_transfers: bool,
    pub(crate) max_concurrent_transfers: usize,
    pub(crate) pending_tracker: Option<Arc<PendingTracker>>,
    pub(crate) max_concurrent_precondition_awaits: usize,
    source: PhantomData<Src>,
}

impl<Src: BlockMetadata> Default for PipelineBaseConfig<Src> {
    fn default() -> Self {
        Self {
            policies: Vec::new(),
            batch_config: BatchConfig::default(),
            policy_timeout: Duration::from_millis(100),
            sweep_interval: Duration::from_millis(10),
            skip_transfers: false,
            max_concurrent_transfers: 1,
            pending_tracker: None,
            max_concurrent_precondition_awaits: 8,
            source: PhantomData,
        }
    }
}

/// Block destination settings for the shared pipeline configuration.
#[doc(hidden)]
#[derive(Clone)]
pub struct BlockPipelineOptions<Dst: BlockMetadata> {
    pub(crate) resource: Option<LogicalResourceId>,
    pub(crate) auto_chain: bool,
    destination: PhantomData<Dst>,
}

impl<Dst: BlockMetadata> Default for BlockPipelineOptions<Dst> {
    fn default() -> Self {
        Self {
            resource: None,
            auto_chain: false,
            destination: PhantomData,
        }
    }
}

/// Object destination settings for the shared pipeline configuration.
#[doc(hidden)]
#[derive(Clone, Default)]
pub struct ObjectPipelineOptions {
    pub(crate) lock_manager: Option<Arc<dyn ObjectLockManager>>,
}

/// One configuration owner for all pipeline destination types.
#[doc(hidden)]
#[derive(Clone)]
pub struct SharedPipelineConfig<Src: BlockMetadata, Options> {
    pub(crate) base: PipelineBaseConfig<Src>,
    pub(crate) options: Options,
}

impl<Src, Options> SharedPipelineConfig<Src, Options>
where
    Src: BlockMetadata,
    Options: Default,
{
    pub(crate) fn into_parts(self) -> (PipelineBaseConfig<Src>, Options) {
        (self.base, self.options)
    }

    /// Reject durations that cause Tokio interval construction to panic.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            !self.base.batch_config.flush_interval.is_zero(),
            "offload batch flush interval must be nonzero"
        );
        ensure!(
            !self.base.sweep_interval.is_zero(),
            "offload cancellation sweep interval must be nonzero"
        );
        Ok(())
    }
}

impl<Src, Options> Default for SharedPipelineConfig<Src, Options>
where
    Src: BlockMetadata,
    Options: Default,
{
    fn default() -> Self {
        Self {
            base: PipelineBaseConfig::default(),
            options: Options::default(),
        }
    }
}

/// Configuration for a block destination pipeline.
pub type PipelineConfig<Src, Dst> = SharedPipelineConfig<Src, BlockPipelineOptions<Dst>>;

/// Configuration for an object destination pipeline.
pub type ObjectPipelineConfig<Src> = SharedPipelineConfig<Src, ObjectPipelineOptions>;

/// One builder implementation for all pipeline destination types.
#[doc(hidden)]
pub struct SharedPipelineBuilder<Src: BlockMetadata, Options> {
    config: SharedPipelineConfig<Src, Options>,
}

impl<Src, Options> SharedPipelineBuilder<Src, Options>
where
    Src: BlockMetadata,
    Options: Default,
{
    /// Create a pipeline builder with default settings.
    pub fn new() -> Self {
        Self {
            config: SharedPipelineConfig::default(),
        }
    }

    /// Add one policy to the pipeline.
    pub fn policy(mut self, policy: Arc<dyn OffloadPolicy<Src>>) -> Self {
        self.config.base.policies.push(policy);
        self
    }

    /// Set the maximum batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.base.batch_config.max_batch_size = size;
        self
    }

    /// Set the minimum batch size.
    pub fn min_batch_size(mut self, size: usize) -> Self {
        self.config.base.batch_config.min_batch_size = size;
        self
    }

    /// Set the batch flush interval.
    pub fn flush_interval(mut self, interval: Duration) -> Self {
        self.config.base.batch_config.flush_interval = interval;
        self
    }

    /// Set the policy timeout.
    pub fn policy_timeout(mut self, timeout: Duration) -> Self {
        self.config.base.policy_timeout = timeout;
        self
    }

    /// Set the cancellation sweep interval.
    pub fn sweep_interval(mut self, interval: Duration) -> Self {
        self.config.base.sweep_interval = interval;
        self
    }

    /// Skip physical transfers.
    pub fn skip_transfers(mut self, skip: bool) -> Self {
        self.config.base.skip_transfers = skip;
        self
    }

    /// Set the maximum number of concurrent transfer batches.
    pub fn max_concurrent_transfers(mut self, count: usize) -> Self {
        self.config.base.max_concurrent_transfers = count.max(1);
        self
    }

    /// Set the maximum number of concurrent precondition waits.
    pub fn max_concurrent_precondition_awaits(mut self, count: usize) -> Self {
        self.config.base.max_concurrent_precondition_awaits = count.max(1);
        self
    }

    /// Set the tracker that prevents duplicate transfers.
    pub fn pending_tracker(mut self, tracker: Arc<PendingTracker>) -> Self {
        self.config.base.pending_tracker = Some(tracker);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> SharedPipelineConfig<Src, Options> {
        self.config
    }
}

impl<Src, Options> Default for SharedPipelineBuilder<Src, Options>
where
    Src: BlockMetadata,
    Options: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Src: BlockMetadata, Dst: BlockMetadata> SharedPipelineBuilder<Src, BlockPipelineOptions<Dst>> {
    /// Route physical transfers through one logical resource.
    pub fn resource(mut self, resource: LogicalResourceId) -> Self {
        self.config.options.resource = Some(resource);
        self
    }

    /// Enable automatic output to a downstream pipeline.
    pub fn auto_chain(mut self, enabled: bool) -> Self {
        self.config.options.auto_chain = enabled;
        self
    }
}

impl<Src: BlockMetadata> SharedPipelineBuilder<Src, ObjectPipelineOptions> {
    /// Set the lock manager for object storage.
    pub fn lock_manager(mut self, manager: Arc<dyn ObjectLockManager>) -> Self {
        self.config.options.lock_manager = Some(manager);
        self
    }
}

/// Builder for a block destination pipeline.
pub type PipelineBuilder<Src, Dst> = SharedPipelineBuilder<Src, BlockPipelineOptions<Dst>>;

/// Builder for an object destination pipeline.
pub type ObjectPipelineBuilder<Src> = SharedPipelineBuilder<Src, ObjectPipelineOptions>;
