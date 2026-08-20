// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::anyhow;
use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_logical::BlockManager;
use kvbm_logical::blocks::BlockRegistry;

use super::stage_attached;
use crate::G2;
use crate::g2_capacity::G2Capacity;
use crate::g2_capacity::test_support::RecordingG2Capacity;
use crate::leader::InstanceLeader;
use crate::p2p::session::{
    CommittedBlock, MockSession, MockSessionFactory, Session, SessionFactory,
};
use crate::testing::{create_messenger_tcp, managers::TestManagerBuilder};

struct CompletePullRig {
    leader: Arc<InstanceLeader>,
    manager: Arc<BlockManager<G2>>,
    capacity: Arc<RecordingG2Capacity>,
    session: Arc<MockSession>,
}

impl CompletePullRig {
    async fn new(block_count: usize) -> Self {
        let registry = BlockRegistry::builder().build();
        let manager = Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(block_count)
                .block_size(4)
                .registry(registry.clone())
                .build(),
        );
        let capacity = Arc::new(RecordingG2Capacity::new(Arc::clone(&manager)));
        let leader = Arc::new(
            InstanceLeader::builder()
                .messenger(create_messenger_tcp().await.expect("test messenger"))
                .registry(registry)
                .g2_manager(Arc::clone(&manager))
                .g2_capacity(capacity.clone())
                .build()
                .expect("test leader"),
        );
        let factory = MockSessionFactory::new();
        factory
            .open(uuid::Uuid::new_v4())
            .expect("test session open");
        let session = factory.last_opened().expect("opened test session");
        Self {
            leader,
            manager,
            capacity,
            session,
        }
    }

    fn advertise(&self, hashes: &[SequenceHash]) {
        self.session.inject_peer_commit(hashes.to_vec());
        self.session.inject_peer_finish_commits();
        self.session.inject_peer_available(
            hashes
                .iter()
                .copied()
                .enumerate()
                .map(|(peer_block_id, hash)| CommittedBlock {
                    hash,
                    peer_block_id,
                })
                .collect(),
        );
        self.session.inject_peer_drained();
    }

    fn spawn(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> tokio::task::JoinHandle<
        Result<super::super::StagedPull, kvbm_protocols::control::ControlError>,
    > {
        let leader = Arc::clone(&self.leader);
        let session: Arc<dyn Session> = self.session.clone();
        let capacity: Arc<dyn G2Capacity> = self.capacity.clone();
        tokio::spawn(async move {
            stage_attached(
                &leader,
                LogicalResourceId(0),
                capacity,
                session,
                Some(hashes),
                false,
            )
            .await
        })
    }
}

fn three_hash_lineage() -> Vec<SequenceHash> {
    vec![
        SequenceHash::new(21, None, 0),
        SequenceHash::new(22, Some(21), 1),
        SequenceHash::new(23, Some(22), 2),
    ]
}

#[tokio::test]
async fn reserves_and_pulls_the_whole_lineage_once() {
    let rig = CompletePullRig::new(3).await;
    let hashes = three_hash_lineage();
    rig.advertise(&hashes);

    let task = rig.spawn(hashes.clone());
    rig.session.wait_pull_count(1).await;

    assert_eq!(rig.capacity.allocation_count(), 1);
    assert_eq!(rig.session.pull_calls().len(), 1);
    assert_eq!(rig.session.pull_calls()[0].0, hashes);
    assert_eq!(rig.manager.available_blocks(), 0);

    rig.session.resolve_pull(0, Ok(()));
    let staged = task.await.expect("staged pull task").expect("staged pull");

    assert_eq!(staged.hashes(), hashes);
    assert_eq!(rig.capacity.registration_count(), 0);

    let registered = staged.publish().expect("publish staged pull");

    assert_eq!(registered.len(), hashes.len());
    assert_eq!(rig.capacity.registration_count(), 1);
    assert_eq!(rig.manager.match_blocks(&hashes).len(), hashes.len());
}

#[tokio::test]
async fn pull_failure_rolls_back_the_whole_reservation() {
    let rig = CompletePullRig::new(3).await;
    let hashes = three_hash_lineage();
    rig.advertise(&hashes);

    let task = rig.spawn(hashes.clone());
    rig.session.wait_pull_count(1).await;

    assert_eq!(rig.capacity.allocation_count(), 1);
    assert_eq!(rig.manager.available_blocks(), 0);

    rig.session
        .resolve_pull(0, Err(anyhow!("forced complete-pull failure")));
    let error = match task.await.expect("staged pull task") {
        Ok(_) => panic!("complete pull must fail"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("forced complete-pull failure"));
    assert_eq!(rig.manager.available_blocks(), hashes.len());
    assert_eq!(rig.capacity.registration_count(), 0);
    assert_eq!(rig.capacity.lease_drop_count(), 1);
    assert!(rig.manager.match_blocks(&hashes).is_empty());
}
