use super::*;

fn matched_handle(
    engine: &Arc<LocalConnectorEngine>,
    computed_blocks: usize,
    matched_blocks: u32,
) -> FindBlocksHandle {
    let search_id = SearchId::new();
    engine.searches.insert(
        search_id,
        SearchState {
            request_id: "bounded-prefix".into(),
            status: Arc::new(Mutex::new(MatchStatus::Matched {
                hit_blocks: matched_blocks,
            })),
            onboarding: OnboardingState::new(
                computed_blocks * BS,
                (computed_blocks + matched_blocks as usize) * BS + 1,
                OnboardingShard {
                    start_block: computed_blocks,
                    num_queried_blocks: matched_blocks as usize,
                    find_session: complete_async(matched_blocks as usize),
                },
            ),
            buffer: plhs_for(0, (computed_blocks + matched_blocks as usize) as u64),
        },
    );
    let owner: Arc<dyn LeaderEngine> = engine.clone();
    FindBlocksHandle::search("bounded-prefix".into(), search_id, Arc::downgrade(&owner))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn accepted_frontier_limits_onboard_destinations_and_failure_ids() -> Result<()> {
    let leader = Arc::new(build_test_leader().await?);
    let sink = RecordingSink::new();
    let engine = LocalConnectorEngine::new(leader, sink.clone(), BS, true);
    let handle = matched_handle(&engine, 1, 3);

    // No source payload exists. The failure must name only the accepted
    // destination, not the computed prefix or the suffix for recomputation.
    let onboard = engine
        .clone()
        .onboard_blocks(&handle, &[40, 50, 51, 52], BS)?;
    wait_complete(&onboard).await;
    let failed = LoadOutcome::FailedPartial {
        block_ids: vec![50],
    };
    assert_eq!(onboard.outcome(), Some(failed.clone()));
    assert_eq!(sink.loads(), vec![("bounded-prefix".into(), failed)]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn accepted_zero_frontier_completes_without_payload() -> Result<()> {
    let leader = Arc::new(build_test_leader().await?);
    let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, true);
    let handle = matched_handle(&engine, 0, 3);
    let onboard = engine.clone().onboard_blocks(&handle, &[50, 51, 52], 0)?;
    wait_complete(&onboard).await;
    assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn accepted_full_frontier_keeps_the_complete_destination_set() -> Result<()> {
    let leader = Arc::new(build_test_leader().await?);
    let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, true);
    let handle = matched_handle(&engine, 1, 3);
    let onboard = engine
        .clone()
        .onboard_blocks(&handle, &[40, 50, 51, 52], 3 * BS)?;
    wait_complete(&onboard).await;
    assert_eq!(
        onboard.outcome(),
        Some(LoadOutcome::FailedPartial {
            block_ids: vec![50, 51, 52]
        })
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_invalid_accepted_frontier_keeps_the_search_available_for_retry() -> Result<()> {
    for accepted in [BS + 1, 4 * BS] {
        let leader = Arc::new(build_test_leader().await?);
        let engine = LocalConnectorEngine::new(leader, NoopWorkerSink::new(), BS, true);
        let handle = matched_handle(&engine, 0, 3);
        assert!(matches!(
            engine.clone().onboard_blocks(&handle, &[50, 51, 52], accepted),
            Err(LeaderEngineError::ExternalTokensMismatch { expected, got })
                if expected == 3 * BS && got == accepted
        ));
        let onboard = engine.clone().onboard_blocks(&handle, &[50, 51, 52], 0)?;
        wait_complete(&onboard).await;
        assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
    }
    Ok(())
}
