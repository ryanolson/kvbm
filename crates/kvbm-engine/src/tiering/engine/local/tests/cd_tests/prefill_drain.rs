// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::onboard_drain::reusable_g2_ids;
use super::super::onboard_pause::{eventually, tick};
use super::*;

use std::time::Duration;

use kvbm_common::LogicalResourceId;
use kvbm_physical::transfer::TransferDrainOutcome;
use tokio::sync::{mpsc, oneshot};

use crate::remote::cd::prefill::PrefillRequestState;
use crate::testing::messenger::create_messenger_pair_tcp;
use crate::tiering::engine::paused::PROBE_INTERVAL;
use crate::tiering::engine::test_support::build_leader_with_messenger;
use crate::worker::group::SpmdParallelWorkers;
use crate::worker::velo::{
    HeldLocalTransfers, LocalTransferCheckTiming, VeloWorkerClient, serve_held_local_transfers,
};

const EXTERNAL: &[BlockId] = &[50, 51];
const ALLOCATION: [BlockId; 3] = [40, EXTERNAL[0], EXTERNAL[1]];

struct HeldKick {
    sources: Vec<BlockId>,
    proof: oneshot::Sender<TransferDrainOutcome>,
}

impl HeldKick {
    fn report(self, outcome: TransferDrainOutcome) -> Vec<BlockId> {
        assert!(
            self.proof.send(outcome).is_ok(),
            "the worker received the proof"
        );
        self.sources
    }
}

struct PrefillDrain {
    engine: Arc<LocalConnectorEngine>,
    sink: Arc<RecordingSink>,
    g2: Arc<BlockManager<G2>>,
    session: Arc<MockSession>,
    search: FindBlocksHandle,
    transfers: HeldLocalTransfers,
    kicks: mpsc::UnboundedReceiver<HeldKick>,
    _server: Arc<velo::Messenger>,
}

impl PrefillDrain {
    async fn prepare(lose_reply: bool) -> Result<Self> {
        let pair = create_messenger_pair_tcp().await?;
        let (started, kicks) = mpsc::unbounded_channel();
        let transfers = serve_held_local_transfers(
            &pair.messenger_b,
            0,
            lose_reply,
            move |resource, sources, destinations| {
                assert_eq!(resource, LogicalResourceId(0));
                assert_eq!(destinations, EXTERNAL);
                let (proof, receipt) = oneshot::channel();
                started.send(HeldKick { sources, proof })?;
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
        let (holder_factory, puller_factory) = MockSessionFactory::make_paired();
        let cd = CdRuntime::new(
            prefill_cfg(),
            Arc::new(TierCell::default()),
            puller_factory.clone(),
            RecordingPrefillPlane::ok(),
            None,
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
        let window = cd_immutables(&peer_g2, ALLOCATION.len(), 100);
        let hashes = cd_block_hashes(&window);
        let session_id = Uuid::new_v4();
        holder_factory.open(session_id)?;
        let holder = holder_factory.last_opened().expect("decode holder opened");
        let request = fb_prefill_req(
            &hashes,
            session_id,
            Uuid::new_v4().into(),
            holder.endpoint(),
            ALLOCATION.len() * BS,
            BS,
            ALLOCATION.len() * BS + 1,
        );
        let (matched, search, _) = expect_resolved(engine.clone().find_blocks(&request, None)?);
        assert_eq!(matched, EXTERNAL.len() * BS);
        let search = search.expect("prefill lifecycle latched");
        holder.commit(hashes)?;
        holder.make_available(window)?;

        let state = engine.cd.as_ref().unwrap().prefill.get("rq").unwrap();
        eventually("the prefill pull finishes", || state.pulls_complete()).await;
        let session = puller_factory
            .last_attached()
            .expect("prefill puller attached");
        Ok(Self {
            engine,
            sink,
            g2,
            session,
            search,
            transfers,
            kicks,
            _server: pair.messenger_b,
        })
    }

    fn state(&self) -> Option<Arc<PrefillRequestState>> {
        self.engine.cd.as_ref().unwrap().prefill.get("rq")
    }

    fn pin_counts(&self) -> Option<Vec<(BlockId, usize)>> {
        let suffix = self.state()?.registered_suffix(EXTERNAL.len())?;
        Some(
            suffix
                .iter()
                .map(|pin| (pin.block_id(), pin.use_count()))
                .collect(),
        )
    }

    async fn onboard(&mut self) -> Result<(OnboardHandle, HeldKick)> {
        let handle =
            self.engine
                .clone()
                .onboard_blocks(&self.search, &ALLOCATION, EXTERNAL.len() * BS)?;
        let kick = tokio::time::timeout(Duration::from_secs(5), self.kicks.recv())
            .await?
            .expect("prefill kick dispatched through Velo");
        Ok((handle, kick))
    }

    async fn onboard_until_paused(&mut self) -> Result<(OnboardHandle, HeldKick, FenceHandle)> {
        let initial_pins = self
            .pin_counts()
            .expect("the lifecycle owns the pulled suffix");
        let (mut handle, kick) = self.onboard().await?;
        assert_eq!(
            kick.sources,
            initial_pins.iter().map(|(id, _)| *id).collect::<Vec<_>>()
        );
        eventually("the kick reports its disposition", || {
            handle.paused().is_some() || handle.is_complete()
        })
        .await;
        assert!(self.transfers.active(), "the worker is still copying");

        let reusable = reusable_g2_ids(&self.g2);
        let pins = self.pin_counts();
        // The paused kick must hold a separate clone of each lifecycle pin.
        let expected_pins: Vec<_> = initial_pins
            .iter()
            .map(|(id, count)| (*id, count + 1))
            .collect();
        let fence = handle.take_physical_drain_fence();
        let physical_pending = self
            .engine
            .actions
            .get(handle.id())
            .is_some_and(|record| record.physical_pending);
        let outcome = handle.outcome();
        let loads = self.sink.loads();
        let closed = self.session.closed_reason();
        let lifecycle_intact = self
            .state()
            .is_some_and(|state| state.has_session() && state.pending_failure().is_none());
        assert!(
            handle.paused().is_some()
                && outcome.is_none()
                && loads.is_empty()
                && fence.as_ref().is_some_and(|fence| !fence.is_complete())
                && physical_pending
                && kick.sources.iter().all(|source| !reusable.contains(source))
                && pins.as_ref() == Some(&expected_pins)
                && lifecycle_intact
                && closed.is_none()
                && self.session.finished_reason().is_none(),
            "prefill kick lost drain protection: outcome {outcome:?}, sink {loads:?}, \
             G2 sources {:?}, reusable {reusable:?}, pins {pins:?}, fence {}, \
             physical_pending {physical_pending}, closed {closed:?}, lifecycle_intact {lifecycle_intact}",
            kick.sources,
            fence.is_some(),
        );
        Ok((handle, kick, fence.unwrap()))
    }

    async fn prove(
        &self,
        handle: &OnboardHandle,
        kick: HeldKick,
        outcome: TransferDrainOutcome,
    ) -> Vec<BlockId> {
        let sources = kick.report(outcome);
        eventually("the worker records the kick proof", || {
            !self.transfers.active()
        })
        .await;
        tokio::time::pause();
        tick(PROBE_INTERVAL).await;
        eventually("the recheck clears the pause", || handle.paused().is_none()).await;
        tokio::time::resume();
        sources
    }

    async fn assert_failed(
        &self,
        handle: &OnboardHandle,
        fence: &FenceHandle,
        state: &PrefillRequestState,
    ) {
        eventually("the drained kick settles", || handle.is_complete()).await;
        let failure = LoadOutcome::FailedPartial {
            block_ids: EXTERNAL.to_vec(),
        };
        assert_eq!(handle.outcome(), Some(failure.clone()));
        assert!(
            fence.is_complete(),
            "the kick's physical fence must complete"
        );
        assert_eq!(self.sink.loads(), vec![("rq".into(), failure)]);
        assert_eq!(
            state.pending_failure().as_deref(),
            Some("prefill onboard kick failed")
        );
        assert!(self.state().is_none());
        assert_eq!(
            self.session.closed_reason(),
            Some(Some("prefill onboard kick failed".into()))
        );
        assert!(self.session.finished_reason().is_none());
    }
}

#[tokio::test]
async fn cd_prefill_kick_pauses_when_the_worker_reply_is_lost() -> Result<()> {
    let mut prefill = PrefillDrain::prepare(true).await?;
    let (handle, kick, fence) = prefill.onboard_until_paused().await?;
    let state = prefill.state().expect("prefill lifecycle latched");
    let sources = prefill.prove(&handle, kick, kick_error()).await;
    prefill.assert_failed(&handle, &fence, &state).await;
    drop(state);
    let reusable = reusable_g2_ids(&prefill.g2);
    assert!(sources.iter().all(|source| reusable.contains(source)));
    Ok(())
}

#[tokio::test]
async fn cd_prefill_kick_completed_recheck_keeps_output_session_parked() -> Result<()> {
    let mut prefill = PrefillDrain::prepare(true).await?;
    let (handle, kick, fence) = prefill.onboard_until_paused().await?;
    let state = prefill.state().expect("prefill lifecycle latched");
    let sources = prefill
        .prove(&handle, kick, TransferDrainOutcome::Completed)
        .await;
    assert_eq!(handle.outcome(), Some(LoadOutcome::Done));
    assert!(fence.is_complete());
    assert_eq!(prefill.sink.loads(), vec![("rq".into(), LoadOutcome::Done)]);
    let kept = prefill.state().expect("the output lifecycle stays latched");
    assert!(Arc::ptr_eq(&kept, &state));
    assert!(state.has_session() && state.pending_failure().is_none());
    assert!(prefill.session.closed_reason().is_none());
    assert!(prefill.session.finished_reason().is_none());
    let reusable = reusable_g2_ids(&prefill.g2);
    assert!(sources.iter().all(|source| !reusable.contains(source)));
    assert!(prefill.kicks.try_recv().is_err());
    Ok(())
}

#[tokio::test]
async fn cd_prefill_kick_eviction_keeps_pins_until_recheck() -> Result<()> {
    let mut prefill = PrefillDrain::prepare(true).await?;
    let (handle, kick, fence) = prefill.onboard_until_paused().await?;
    let eviction = prefill.engine.evict(&"rq".into());
    let eviction_fence = eviction.handle.expect("an in-flight kick arms eviction");
    assert!(!eviction_fence.is_complete());
    assert!(!fence.is_complete());
    assert!(prefill.sink.fences().is_empty());
    assert!(prefill.sink.loads().is_empty());
    assert!(
        prefill.state().is_none(),
        "eviction releases the tracked lifecycle"
    );
    let reusable = reusable_g2_ids(&prefill.g2);
    assert!(
        kick.sources.iter().all(|source| !reusable.contains(source)),
        "the paused kick keeps the G2 sources pinned after eviction"
    );
    let sources = prefill
        .prove(&handle, kick, TransferDrainOutcome::Completed)
        .await;
    assert!(fence.is_complete());
    assert!(eviction_fence.is_complete());
    assert_eq!(prefill.sink.fences(), eviction.fence.per_worker);
    assert!(prefill.sink.loads().is_empty());
    eventually(
        "both the kick and the evicted output lifecycle release the pins",
        || {
            let reusable = reusable_g2_ids(&prefill.g2);
            sources.iter().all(|source| reusable.contains(source))
        },
    )
    .await;
    Ok(())
}

#[tokio::test]
async fn cd_prefill_kick_drained_failure_keeps_failure_cleanup() -> Result<()> {
    let mut prefill = PrefillDrain::prepare(false).await?;
    let state = prefill.state().expect("prefill lifecycle latched");
    let (mut handle, kick) = prefill.onboard().await?;
    let fence = handle
        .take_physical_drain_fence()
        .expect("the kick carries a fence");
    assert!(!fence.is_complete());
    let sources = kick.report(kick_error());
    prefill.assert_failed(&handle, &fence, &state).await;
    drop(state);
    let reusable = reusable_g2_ids(&prefill.g2);
    assert!(sources.iter().all(|source| reusable.contains(source)));
    Ok(())
}

fn kick_error() -> TransferDrainOutcome {
    TransferDrainOutcome::DrainedWithError(anyhow::anyhow!("injected copy error"))
}
