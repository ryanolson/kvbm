// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The opt-in `dev` control module: `reset`.
//!
//! Operator/debug tooling. Off by default, enabled via
//! `control.dev = true`. Safe to run in production — no warning is logged
//! when enabled. Migrated from the connector's `ConnectorControlApi`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Result;
use velo::{Handler, Messenger};

use kvbm_protocols::control::{
    ControlError, ControlReply, ModuleId, RESET_HANDLER, ResetRequest, ResetResponse, Tier,
    TierError, plan_reset,
};

use super::ControlModule;
use crate::leader::InstanceLeader;
use crate::p2p::session::LocalTierReset;

/// The `dev` control module — opt-in.
pub struct DevModule {
    leader: Arc<InstanceLeader>,
}

impl DevModule {
    pub fn new(leader: Arc<InstanceLeader>) -> Self {
        Self { leader }
    }
}

impl ControlModule for DevModule {
    fn id(&self) -> ModuleId {
        ModuleId::Dev
    }

    fn register(&self, messenger: &Arc<Messenger>) -> Result<()> {
        let leader = self.leader.clone();
        let handler = Handler::typed_unary_async(RESET_HANDLER, move |ctx| {
            let leader = Arc::clone(&leader);
            async move {
                let req: ResetRequest = ctx.input;
                let reply: ControlReply<ResetResponse> = reset(&leader, req).await.into();
                Ok::<ControlReply<ResetResponse>, anyhow::Error>(reply)
            }
        })
        .build();
        messenger
            .register_handler(handler)
            .map_err(|e| anyhow::anyhow!("velo register_handler({RESET_HANDLER}): {e}"))?;
        Ok(())
    }
}

/// Reset the inactive pools of the requested (or all configured) tiers.
async fn reset(leader: &InstanceLeader, req: ResetRequest) -> Result<ResetResponse, ControlError> {
    let local_reset = leader.local_tier_reset();
    let mut available = HashSet::new();
    // G2 is always present once an InstanceLeader is up.
    available.insert(Tier::G2);
    if leader.g3_manager().is_some() {
        available.insert(Tier::G3);
    }
    if let Some(hook) = &local_reset {
        available.extend(
            hook.tiers()
                .into_iter()
                .filter(|tier| matches!(*tier, Tier::Carrier | Tier::G1)),
        );
    }

    let (to_reset, skipped) = plan_reset(&req, &available)?;
    let to_reset: HashSet<_> = to_reset.into_iter().collect();
    let hook_tiers: Vec<_> = Tier::ORDERED
        .iter()
        .copied()
        .filter(|tier| matches!(*tier, Tier::Carrier | Tier::G1) && to_reset.contains(tier))
        .collect();
    let mut hook_errors = if hook_tiers.is_empty() {
        HashMap::new()
    } else {
        reset_local_tiers(local_reset.as_deref(), hook_tiers)
            .await
            .into_iter()
            .map(|error| (error.tier, error))
            .collect::<HashMap<_, _>>()
    };

    let mut reset = Vec::with_capacity(to_reset.len());
    let mut failed = Vec::new();
    for tier in Tier::ORDERED
        .iter()
        .copied()
        .filter(|tier| to_reset.contains(tier))
    {
        match tier {
            Tier::Carrier | Tier::G1 => match hook_errors.remove(&tier) {
                Some(error) => {
                    tracing::warn!(?tier, message = %error.message, "tier reset failed");
                    failed.push(error);
                }
                None => {
                    tracing::info!(?tier, "tier reset succeeded");
                    reset.push(tier);
                }
            },
            Tier::G2 => {
                let drained = leader
                    .g2_managers()
                    .iter()
                    .map(|(_, manager)| manager.drain_inactive_pool())
                    .sum::<usize>();
                tracing::info!(?tier, drained, "tier reset drained inactive blocks");
                reset.push(tier);
            }
            Tier::G3 => {
                let Some(manager) = leader.g3_manager() else {
                    failed.push(TierError {
                        tier,
                        message: "G3 is not configured".into(),
                    });
                    continue;
                };
                let drained = manager.drain_inactive_pool();
                tracing::info!(?tier, drained, "tier reset drained inactive blocks");
                reset.push(tier);
            }
        }
    }

    Ok(ResetResponse {
        reset,
        failed,
        skipped_unconfigured: skipped,
    })
}

async fn reset_local_tiers(hook: Option<&dyn LocalTierReset>, tiers: Vec<Tier>) -> Vec<TierError> {
    let Some(hook) = hook else {
        return tiers
            .into_iter()
            .map(|tier| TierError {
                tier,
                message: "no local tier reset hook installed".into(),
            })
            .collect();
    };
    hook.reset(tiers).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leader::InstanceLeader;
    use crate::p2p::session::LocalTierReset;
    use crate::testing::{
        managers::TestManagerBuilder, messenger::create_messenger_tcp,
        token_blocks::create_sequential_block,
    };
    use crate::{G2, SequenceHash};
    use futures::future::BoxFuture;
    use kvbm_common::LogicalResourceId;
    use kvbm_logical::{
        BlockManagerSet, KvbmSequenceHashProvider,
        blocks::{BlockRegistry, ImmutableBlock},
        manager::BlockManager,
    };
    use std::sync::Mutex;

    struct RecordingReset {
        tiers: Vec<Tier>,
        calls: Mutex<Vec<Vec<Tier>>>,
        errors: Vec<TierError>,
    }

    impl LocalTierReset for RecordingReset {
        fn tiers(&self) -> Vec<Tier> {
            self.tiers.clone()
        }

        fn reset(&self, tiers: Vec<Tier>) -> BoxFuture<'static, Vec<TierError>> {
            self.calls.lock().unwrap().push(tiers);
            let errors = self.errors.clone();
            Box::pin(async move { errors })
        }
    }

    fn manager(block_count: usize) -> Arc<BlockManager<G2>> {
        Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(block_count)
                .block_size(4)
                .registry(BlockRegistry::new())
                .build(),
        )
    }

    async fn leader_with_managers(managers: Vec<Arc<BlockManager<G2>>>) -> InstanceLeader {
        let mut manager_set = BlockManagerSet::new();
        for (index, manager) in managers.into_iter().enumerate() {
            manager_set
                .insert(LogicalResourceId(index as u16 + 1), manager)
                .unwrap();
        }
        InstanceLeader::builder()
            .messenger(create_messenger_tcp().await.unwrap())
            .registry(BlockRegistry::new())
            .g2_manager_set(Arc::new(manager_set), LogicalResourceId(1))
            .build()
            .unwrap()
    }

    fn manager_with_live_and_inactive(
        start: u32,
    ) -> (
        Arc<BlockManager<G2>>,
        SequenceHash,
        SequenceHash,
        ImmutableBlock<G2>,
    ) {
        let manager = manager(2);
        let token_blocks = [
            create_sequential_block(start, 4),
            create_sequential_block(start + 4, 4),
        ];
        let hashes = token_blocks
            .iter()
            .map(|block| block.kvbm_sequence_hash())
            .collect::<Vec<_>>();
        let complete = manager
            .allocate_blocks(2)
            .unwrap()
            .into_iter()
            .zip(&token_blocks)
            .map(|(block, token_block)| block.complete(token_block).unwrap())
            .collect();
        let mut registered = manager.register_blocks(complete);
        let live = registered.pop().unwrap();
        drop(registered);
        (manager, hashes[0], hashes[1], live)
    }

    #[tokio::test]
    async fn reset_calls_local_hook_in_tier_order() {
        let hook = Arc::new(RecordingReset {
            tiers: vec![Tier::G1, Tier::Carrier, Tier::G2],
            calls: Mutex::new(Vec::new()),
            errors: Vec::new(),
        });
        let leader = leader_with_managers(vec![manager(2)]).await;
        assert!(leader.set_local_tier_reset(hook.clone()));

        let response = reset(
            &leader,
            ResetRequest {
                tiers: Some(vec![Tier::G1, Tier::Carrier]),
            },
        )
        .await
        .unwrap();

        assert_eq!(
            *hook.calls.lock().unwrap(),
            vec![vec![Tier::Carrier, Tier::G1]]
        );
        assert_eq!(response.reset, vec![Tier::Carrier, Tier::G1]);
    }

    #[tokio::test]
    async fn reset_rejects_local_tier_without_hook() {
        let leader = leader_with_managers(vec![manager(2)]).await;
        let error = reset(
            &leader,
            ResetRequest {
                tiers: Some(vec![Tier::G1]),
            },
        )
        .await
        .unwrap_err();

        assert_eq!(error, ControlError::TierNotConfigured(Tier::G1));
    }

    #[tokio::test]
    async fn reset_maps_local_hook_error_to_failed_tier() {
        let hook = Arc::new(RecordingReset {
            tiers: vec![Tier::Carrier, Tier::G1],
            calls: Mutex::new(Vec::new()),
            errors: vec![TierError {
                tier: Tier::Carrier,
                message: "carrier reset failed".into(),
            }],
        });
        let leader = leader_with_managers(vec![manager(2)]).await;
        assert!(leader.set_local_tier_reset(hook));

        let response = reset(
            &leader,
            ResetRequest {
                tiers: Some(vec![Tier::G1, Tier::Carrier]),
            },
        )
        .await
        .unwrap();

        assert_eq!(response.reset, vec![Tier::G1]);
        assert_eq!(response.failed.len(), 1);
        assert_eq!(response.failed[0].tier, Tier::Carrier);
        assert_eq!(response.failed[0].message, "carrier reset failed");
    }

    #[tokio::test]
    async fn reset_drains_all_g2_managers_without_touching_live_blocks() {
        let (first, first_inactive, first_live_hash, _first_live) =
            manager_with_live_and_inactive(100_000);
        let (second, second_inactive, second_live_hash, _second_live) =
            manager_with_live_and_inactive(200_000);
        let leader = leader_with_managers(vec![Arc::clone(&first), Arc::clone(&second)]).await;

        let response = reset(
            &leader,
            ResetRequest {
                tiers: Some(vec![Tier::G2]),
            },
        )
        .await
        .unwrap();

        assert_eq!(response.reset, vec![Tier::G2]);
        assert!(response.failed.is_empty());
        assert!(first.match_blocks(&[first_inactive]).is_empty());
        assert_eq!(first.match_blocks(&[first_live_hash]).len(), 1);
        assert!(second.match_blocks(&[second_inactive]).is_empty());
        assert_eq!(second.match_blocks(&[second_live_hash]).len(), 1);
    }

    #[tokio::test]
    async fn set_local_tier_reset_is_first_write_wins() {
        let leader = leader_with_managers(vec![manager(2)]).await;
        let hook = Arc::new(RecordingReset {
            tiers: vec![Tier::Carrier],
            calls: Mutex::new(Vec::new()),
            errors: Vec::new(),
        });

        assert!(leader.set_local_tier_reset(hook.clone()));
        assert!(!leader.set_local_tier_reset(hook));
    }
}
