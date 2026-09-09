use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn accepted_frontier_copies_only_the_requested_sources_and_destinations() -> Result<()> {
    for computed in [0, 1] {
        for accepted in [0, 1, 4 - computed] {
            let rig = fallthrough_rig(4).await?;
            let engine = LocalConnectorEngine::new(
                rig.engine.leader.clone(),
                NoopWorkerSink::new(),
                BS,
                false,
            );
            let mut request = fb("accepted-prefix", rig.plhs.clone(), 4 * BS + 1);
            request.num_computed_tokens = computed * BS;
            let (matched, handle, _) = expect_resolved(engine.clone().find_blocks(&request, None)?);
            assert_eq!(matched, (4 - computed) * BS);
            let handle = handle.expect("the resident prefix must have a search handle");
            let destinations = [40, 50, 51, 52];
            let onboard = engine
                .clone()
                .onboard_blocks(&handle, &destinations, accepted * BS)?;
            wait_complete(&onboard).await;
            assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
            let expected_destinations = if accepted == 0 {
                Vec::new()
            } else {
                vec![destinations[computed..computed + accepted].to_vec()]
            };
            let expected_sources = if accepted == 0 {
                Vec::new()
            } else {
                vec![
                    rig._held[computed..computed + accepted]
                        .iter()
                        .map(|block| block.block_id())
                        .collect::<Vec<_>>(),
                ]
            };
            assert_eq!(rig.workers.transfer_calls(), expected_destinations);
            assert_eq!(*rig.workers.sources.lock().unwrap(), expected_sources);
            let guard = engine.inflight.lock().unwrap();
            assert!(!guard.overlaps(&rig.plhs[..computed]));
            assert!(!guard.overlaps(&rig.plhs[computed + accepted..]));
            if accepted > 0 {
                assert!(guard.overlaps(&rig.plhs[computed..computed + accepted]));
            }
            drop(guard);
            drop(onboard);
            drop(handle);
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_short_destination_does_not_silently_reduce_the_accepted_frontier() -> Result<()> {
    let rig = fallthrough_rig(4).await?;
    let engine =
        LocalConnectorEngine::new(rig.engine.leader.clone(), NoopWorkerSink::new(), BS, false);
    let request = fb("short-destination", rig.plhs.clone(), 4 * BS + 1);
    let (_, handle, _) = expect_resolved(engine.clone().find_blocks(&request, None)?);
    let handle = handle.expect("the resident prefix must have a search handle");
    let onboard = engine.clone().onboard_blocks(&handle, &[50], 2 * BS)?;
    wait_complete(&onboard).await;
    assert_eq!(
        onboard.outcome(),
        Some(LoadOutcome::FailedPartial {
            block_ids: vec![50]
        })
    );
    assert!(rig.workers.transfer_calls().is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dispatched_remote_continuation_requires_its_complete_commitment() -> Result<()> {
    let factory = MockSessionFactory::new();
    let plane = RecordingPrefillPlane::ok();
    let workers = RecordingWorkers::new();
    let engine = cd_onboard_engine(256, factory.clone(), plane.clone(), workers.clone()).await?;
    let blocks = cd_immutables(&cd_g2_manager(8), 3, 100);
    let hashes = cd_block_hashes(&blocks);
    let (session, search_id) = latch_and_prepare_onboard(&engine, &factory, &blocks, 1);
    wait_for(|| plane.count() >= 1).await;
    let owner: Arc<dyn LeaderEngine> = engine.clone();
    let handle = FindBlocksHandle::search("rq".into(), search_id, Arc::downgrade(&owner));

    assert!(matches!(
        engine.clone().onboard_blocks(&handle, &[50, 51, 52], 2 * BS),
        Err(LeaderEngineError::ExternalTokensMismatch { expected, got })
            if expected == 3 * BS && got == 2 * BS
    ));
    assert!(engine.searches.contains_key(&search_id));
    assert!(workers.transfer_calls().is_empty());

    let onboard = engine
        .clone()
        .onboard_blocks(&handle, &[50, 51, 52], 3 * BS)?;
    session.inject_peer_commit(hashes[1..].to_vec());
    session.inject_peer_available(vec![
        CommittedBlock {
            hash: hashes[1],
            peer_block_id: 900,
        },
        CommittedBlock {
            hash: hashes[2],
            peer_block_id: 901,
        },
    ]);
    wait_for(|| !session.pull_calls().is_empty()).await;
    session.resolve_pull(0, Ok(()));
    wait_complete(&onboard).await;
    assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
    assert_eq!(workers.transfer_calls(), vec![vec![50], vec![51, 52]]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_accepted_prefix_crosses_shards_after_a_computed_block() -> Result<()> {
    let rig = fallthrough_rig(4).await?;
    let engine =
        LocalConnectorEngine::new(rig.engine.leader.clone(), NoopWorkerSink::new(), BS, false);
    let shard = |start| OnboardingShard {
        start_block: start,
        num_queried_blocks: 2,
        find_session: FindMatchesResult::Ready(ReadyResult::new(
            rig._held[start..start + 2].to_vec(),
            MatchBreakdown {
                host_blocks: 2,
                disk_blocks: 0,
                object_blocks: 0,
            },
        )),
    };
    let mut onboarding = OnboardingState::new(BS, 4 * BS + 1, shard(0));
    onboarding.shards.push(shard(2));
    let search_id = SearchId::new();
    engine.searches.insert(
        search_id,
        SearchState {
            request_id: "two-shards".into(),
            status: Arc::new(Mutex::new(MatchStatus::Matched { hit_blocks: 3 })),
            onboarding,
            buffer: rig.plhs.clone(),
        },
    );
    let owner: Arc<dyn LeaderEngine> = engine.clone();
    let handle = FindBlocksHandle::search("two-shards".into(), search_id, Arc::downgrade(&owner));
    let onboard = engine
        .clone()
        .onboard_blocks(&handle, &[40, 50, 51, 52], 2 * BS)?;
    wait_complete(&onboard).await;
    assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
    assert_eq!(rig.workers.transfer_calls(), vec![vec![50, 51]]);
    assert_eq!(
        *rig.workers.sources.lock().unwrap(),
        vec![vec![rig._held[1].block_id(), rig._held[2].block_id()]]
    );
    let guard = engine.inflight.lock().unwrap();
    assert!(guard.overlaps(&rig.plhs[1..3]));
    assert!(!guard.overlaps(&[rig.plhs[0], rig.plhs[3]]));
    Ok(())
}
