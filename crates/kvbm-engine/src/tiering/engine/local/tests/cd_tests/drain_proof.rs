// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::onboard_drain::reusable_g2_ids;
use super::super::onboard_pause::{eventually, tick};
use super::*;

use std::time::Duration;

use kvbm_common::LogicalResourceId;
use kvbm_physical::transfer::TransferDrainOutcome;
use tokio::sync::{mpsc, oneshot};

use crate::testing::messenger::create_messenger_pair_tcp;
use crate::tiering::engine::paused::PROBE_INTERVAL;
use crate::tiering::engine::test_support::build_leader_with_messenger;
use crate::worker::group::SpmdParallelWorkers;
use crate::worker::velo::{
    HeldLocalTransfers, LocalTransferCheckTiming, VeloWorkerClient, serve_held_local_transfers,
};

const CAPACITY: usize = 256;
const DESTINATIONS: [BlockId; 4] = [50, 51, 52, 53];

struct HeldCopy {
    sources: Vec<BlockId>,
    destinations: Vec<BlockId>,
    proof: oneshot::Sender<TransferDrainOutcome>,
}

impl HeldCopy {
    fn complete(self) {
        self.report(TransferDrainOutcome::Completed);
    }

    fn report(self, outcome: TransferDrainOutcome) {
        assert!(self.proof.send(outcome).is_ok());
    }
}

struct DecodeRestore {
    engine: Arc<LocalConnectorEngine>,
    sink: Arc<RecordingSink>,
    g2: Arc<BlockManager<G2>>,
    factory: Arc<MockSessionFactory>,
    session: Arc<MockSession>,
    hashes: Vec<SequenceHash>,
    handle: OnboardHandle,
    transfers: HeldLocalTransfers,
    copies: mpsc::UnboundedReceiver<HeldCopy>,
    _search: FindBlocksHandle,
    _server: Arc<velo::Messenger>,
}

impl DecodeRestore {
    async fn new(lose_local_reply: bool) -> Result<Self> {
        let pair = create_messenger_pair_tcp().await?;
        let (started, copies) = mpsc::unbounded_channel();
        let transfers = serve_held_local_transfers(
            &pair.messenger_b,
            0,
            lose_local_reply,
            move |resource, sources, destinations| {
                assert_eq!(resource, LogicalResourceId(0));
                let (proof, receipt) = oneshot::channel();
                started.send(HeldCopy {
                    sources,
                    destinations,
                    proof,
                })?;
                Ok(TransferCompleteNotification::from_drain_future(
                    async move {
                        receipt
                            .await
                            .unwrap_or_else(|error| TransferDrainOutcome::Unproven(error.into()))
                    },
                ))
            },
        )?;
        let client = VeloWorkerClient::new_for_worker(
            pair.messenger_a.clone(),
            pair.messenger_b.instance_id(),
            0,
        )
        .with_local_transfer_check(LocalTransferCheckTiming {
            wait: Duration::from_millis(10),
            ..LocalTransferCheckTiming::default()
        });
        let workers = Arc::new(SpmdParallelWorkers::new(
            vec![Arc::new(client)],
            Arc::new(velo::EventManager::local()),
            pair.messenger_a.runtime().clone(),
        ));
        let (leader, g2) =
            build_leader_with_messenger(pair.messenger_a, LogicalResourceId(0), 6, BS, workers)?;
        let factory = MockSessionFactory::new();
        let plane = RecordingPrefillPlane::ok();
        let cd = cd_runtime(
            SelectionPolicy::Always,
            CAPACITY,
            true,
            factory.clone(),
            plane,
        );
        let sink = RecordingSink::new();
        let engine = LocalConnectorEngine::with_offload_submit(
            Arc::new(leader),
            sink.clone(),
            BS,
            false,
            Arc::new(DisabledOffloadSubmit),
            Some(cd),
        );

        let peer_g2 = cd_g2_manager(8);
        let mut window = cd_immutables(&peer_g2, DESTINATIONS.len(), 100);
        let local = cd_immutables(&g2, 1, 100).pop().unwrap();
        assert_eq!(window[0].sequence_hash(), local.sequence_hash());
        window[0] = local;
        let hashes = cd_block_hashes(&window);
        let (session, search_id) = latch_and_prepare_onboard(&engine, &factory, &window, 1);
        drop(window);

        // Acknowledge the peer's pull of the published local block. The
        // onward copy must own its pin once the peer has removed its hold.
        let peer_factory = MockSessionFactory::new();
        peer_factory.open(session.session_id())?;
        let peer = peer_factory.last_opened().unwrap();
        session.pair_with(&peer);
        let peer_dst = peer_g2.allocate_blocks(1).unwrap();
        drop(
            tokio::time::timeout(Duration::from_secs(5), peer.pull(vec![hashes[0]], peer_dst))
                .await??,
        );
        drop(peer);
        drop(peer_factory);

        session.inject_peer_commit(hashes[1..].to_vec());
        let owner: Arc<dyn LeaderEngine> = engine.clone();
        let search = FindBlocksHandle::search("rq".into(), search_id, Arc::downgrade(&owner));
        let handle =
            engine
                .clone()
                .onboard_blocks(&search, &DESTINATIONS, DESTINATIONS.len() * BS)?;
        Ok(Self {
            engine,
            sink,
            g2,
            factory,
            session,
            hashes,
            handle,
            transfers,
            copies,
            _search: search,
            _server: pair.messenger_b,
        })
    }

    fn publish(&self, slots: &[usize]) {
        self.session.inject_peer_available(
            slots
                .iter()
                .map(|&slot| CommittedBlock {
                    hash: self.hashes[slot],
                    peer_block_id: 900 + slot,
                })
                .collect(),
        );
    }

    async fn copy(&mut self, destinations: &[BlockId]) -> HeldCopy {
        let mut copy = None;
        eventually("the worker dispatches the CD copy", || {
            copy = self.copies.try_recv().ok();
            copy.is_some()
        })
        .await;
        let copy = copy.unwrap();
        assert_eq!(copy.destinations, destinations);
        copy
    }

    async fn pull_and_copy(&mut self, pull: usize, slots: &[usize]) -> HeldCopy {
        eventually("the CD decode requests the next pull", || {
            self.session.pull_calls().len() > pull || self.handle.is_complete()
        })
        .await;
        assert!(
            !self.handle.is_complete(),
            "CD decode completed before pulling the remaining blocks"
        );
        let hashes: Vec<_> = slots.iter().map(|&slot| self.hashes[slot]).collect();
        assert_eq!(self.session.pull_calls()[pull].0, hashes);
        self.session.resolve_pull(pull, Ok(()));
        let destinations: Vec<_> = slots.iter().map(|&slot| DESTINATIONS[slot]).collect();
        self.copy(&destinations).await
    }

    async fn assert_paused(&mut self, copy: &HeldCopy) -> FenceHandle {
        eventually("the CD copy disposition", || {
            self.handle.paused().is_some() || self.handle.is_complete()
        })
        .await;
        assert!(
            self.transfers.active(),
            "the worker copy must still be running"
        );
        let reusable = reusable_g2_ids(&self.g2);
        let fence = self.handle.take_physical_drain_fence();
        let physical_pending = self
            .engine
            .actions
            .get(self.handle.id())
            .is_some_and(|record| record.physical_pending);
        let budget = self.engine.cd.as_ref().unwrap().budget.available();
        let loads = self.sink.loads();
        let outcome = self.handle.outcome();
        let sources_held = copy.sources.iter().all(|source| !reusable.contains(source));
        assert!(
            self.handle.paused().is_some()
                && outcome.is_none()
                && loads.is_empty()
                && sources_held
                && fence.as_ref().is_some_and(|fence| !fence.is_complete())
                && physical_pending
                && self.session.closed_reason().is_none()
                && self.session.finished_reason().is_none()
                && budget == CAPACITY - DESTINATIONS.len() * BS,
            "CD copy to {:?} lost drain protection: outcome {outcome:?}, sink {loads:?}, \
             G2 sources {:?}, reusable {reusable:?}, fence {}, physical_pending {physical_pending}, \
             closed {:?}, budget {budget}",
            copy.destinations,
            copy.sources,
            fence.is_some(),
            self.session.closed_reason()
        );
        fence.unwrap()
    }

    async fn prove(&self, copy: HeldCopy, outcome: TransferDrainOutcome) {
        copy.report(outcome);
        eventually("the worker records CD copy proof", || {
            !self.transfers.active()
        })
        .await;
        tokio::time::pause();
        tick(PROBE_INTERVAL).await;
        eventually("proof clears the CD pause", || {
            self.handle.paused().is_none()
        })
        .await;
        tokio::time::resume();
    }

    async fn assert_settled(&self, fence: &FenceHandle, failed: &[BlockId]) {
        let cd = self.engine.cd.as_ref().unwrap();
        eventually("the CD load and lifecycle settle", || {
            !self.sink.loads().is_empty() && cd.requests.is_empty()
        })
        .await;
        let outcome = if failed.is_empty() {
            LoadOutcome::Done
        } else {
            LoadOutcome::FailedPartial {
                block_ids: failed.to_vec(),
            }
        };
        assert_eq!(self.handle.outcome(), Some(outcome.clone()));
        assert_eq!(self.sink.loads(), vec![("rq".into(), outcome)]);
        assert!(fence.is_complete());
        assert_eq!(cd.budget.available(), CAPACITY);
        if failed.is_empty() {
            assert_eq!(self.session.finished_reason(), Some(None));
        } else {
            assert_eq!(
                self.session.closed_reason(),
                Some(Some("cd onboard failed".to_owned())),
            );
        }
    }
}

fn drained_error() -> TransferDrainOutcome {
    TransferDrainOutcome::DrainedWithError(anyhow::anyhow!("injected copy error"))
}

#[tokio::test]
async fn cd_decode_pauses_on_a_lost_local_span_reply() -> Result<()> {
    local_pause(TransferDrainOutcome::Completed, &[51, 52, 53]).await
}

#[tokio::test]
async fn cd_decode_reports_all_external_ids_after_a_local_drained_error() -> Result<()> {
    local_pause(drained_error(), &DESTINATIONS).await
}

async fn local_pause(outcome: TransferDrainOutcome, failed: &[BlockId]) -> Result<()> {
    let mut restore = DecodeRestore::new(true).await?;
    restore.publish(&[1, 2, 3]);
    let local = restore.copy(&[50]).await;
    let fence = restore.assert_paused(&local).await;
    assert!(restore.session.pull_calls().is_empty());
    restore.prove(local, outcome).await;
    restore.assert_settled(&fence, failed).await;
    assert!(restore.session.pull_calls().is_empty());
    assert!(restore.copies.try_recv().is_err());
    Ok(())
}

#[tokio::test]
async fn cd_decode_fails_undispatched_runs_after_a_remote_pause() -> Result<()> {
    remote_pause(TransferDrainOutcome::Completed, &[53]).await
}

#[tokio::test]
async fn cd_decode_reports_the_paused_run_after_a_remote_drained_error() -> Result<()> {
    remote_pause(drained_error(), &[51, 53]).await
}

async fn remote_pause(outcome: TransferDrainOutcome, failed: &[BlockId]) -> Result<()> {
    let mut restore = DecodeRestore::new(false).await?;
    restore.copy(&[50]).await.complete();
    restore.publish(&[2]);
    let prior = restore.pull_and_copy(0, &[2]).await;
    restore.transfers.lose_next_reply();
    prior.complete();
    // One batch contains two disjoint runs. The second must stay undispatched.
    restore.publish(&[1, 3]);
    let remote = restore.pull_and_copy(1, &[1]).await;
    let fence = restore.assert_paused(&remote).await;
    assert_eq!(restore.session.pull_calls().len(), 2);
    restore.prove(remote, outcome).await;
    restore.assert_settled(&fence, failed).await;
    assert_eq!(restore.session.pull_calls().len(), 2);
    assert!(restore.copies.try_recv().is_err());
    Ok(())
}

#[tokio::test]
async fn cd_decode_completes_when_the_paused_copy_was_the_last_run() -> Result<()> {
    let mut restore = DecodeRestore::new(false).await?;
    let local = restore.copy(&[50]).await;
    restore.transfers.lose_next_reply();
    local.complete();
    restore.publish(&[1, 2, 3]);
    let remote = restore.pull_and_copy(0, &[1, 2, 3]).await;
    let fence = restore.assert_paused(&remote).await;
    restore.prove(remote, TransferDrainOutcome::Completed).await;
    restore.assert_settled(&fence, &[]).await;
    assert_eq!(restore.session.pull_calls().len(), 1);
    Ok(())
}

#[tokio::test]
async fn cd_decode_evict_and_relatch_while_paused_settles_only_the_old_load() -> Result<()> {
    let mut restore = DecodeRestore::new(true).await?;
    restore.publish(&[1, 2, 3]);
    let local = restore.copy(&[50]).await;
    let fence = restore.assert_paused(&local).await;
    let request_id = RequestId::from("rq");
    let cd = restore.engine.cd.as_ref().unwrap();
    let old_state = cd.requests.get(&request_id).unwrap();
    let eviction = restore.engine.evict(&request_id);
    let eviction_fence = eviction.handle.unwrap();
    assert!(!eviction_fence.is_complete());
    assert!(restore.sink.fences().is_empty());
    assert_eq!(cd.budget.available(), CAPACITY);
    assert_eq!(
        restore.session.closed_reason(),
        Some(Some("evicted".to_owned())),
    );
    let reusable = reusable_g2_ids(&restore.g2);
    assert!(
        local
            .sources
            .iter()
            .all(|source| !reusable.contains(source))
    );

    let replacement_g2 = cd_g2_manager(4);
    let window = cd_immutables(&replacement_g2, 2, 200);
    let (replacement, search) =
        latch_and_prepare_onboard(&restore.engine, &restore.factory, &window, 0);
    let replacement_state = cd.requests.get(&request_id).unwrap();
    assert!(!Arc::ptr_eq(&old_state, &replacement_state));

    restore.prove(local, TransferDrainOutcome::Completed).await;
    assert!(fence.is_complete());
    assert!(eviction_fence.is_complete());
    assert_eq!(restore.sink.fences().len(), 1);
    assert!(restore.sink.loads().is_empty());
    assert_eq!(
        restore.handle.outcome(),
        Some(LoadOutcome::FailedPartial {
            block_ids: vec![51, 52, 53]
        }),
    );
    assert!(restore.session.pull_calls().is_empty());
    assert!(restore.copies.try_recv().is_err());

    let current = cd
        .requests
        .get(&request_id)
        .expect("the replacement lifecycle must stay tracked");
    assert!(Arc::ptr_eq(&current, &replacement_state));
    assert_eq!(cd.budget.available(), CAPACITY - 2 * BS);
    assert!(replacement.closed_reason().is_none());
    assert!(replacement.finished_reason().is_none());
    restore.engine.release_search(&search);
    Ok(())
}
