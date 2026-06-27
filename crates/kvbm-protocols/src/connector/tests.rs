// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Surface-assertion tests for the connector seam.
//!
//! Compile-time properties (a handle is not `Serialize`; the drain is
//! consume-self) are asserted by `compile_fail` doctests on the types
//! themselves (see [`super::handles`] / [`super::protocol`]); a *passing* serde
//! doctest on [`FenceToken`] anchors those, proving `serde::Serialize` is
//! nameable so the `compile_fail` is attributable to the property, not to a
//! naming error. The tests below cover the runtime surface.
//!
//! `kvbm-protocols` carries **no tokio dependency** — the seam stays
//! wire-clean. The handles hold only `Weak<dyn LeaderEngine>` + `std`
//! sync/`Uuid`, never a tokio/`watch` type, which is what keeps that true.

use std::sync::{Arc, Mutex};

use crate::disagg::RemotePrefillParams;

use super::actions::{EngineWorkerSink, WorkerEngineDriver};
use super::actions::{LoadOutcome, SaveOutcome};
use super::engine::LeaderEngine;
use super::handles::{FenceHandle, FindBlocksHandle, OnboardHandle};
use super::noop::NoopBlockEngine;
use super::noop::NoopWorkerSink;
use super::protocol::RequestId;
use super::protocol::{
    AcceptId, ActionFailure, ActionId, ActionStatus, EvictionFence, EvictionOutcome, FenceToken,
    FindBlocksOutcome, FindBlocksRequest, LeaderEngineError, SearchId,
};

fn engine() -> Arc<dyn LeaderEngine> {
    NoopBlockEngine::new()
}

/// `LoadOutcome` has no id-less failure, so an onboard handle whose cell holds
/// an UNRESOLVED `Failed(AllBlocks)` (an engine that skipped dest-set
/// resolution) must project onto the handle's full dest set — never an empty
/// failed set, which would let vLLM finish the recv with nothing invalidated.
#[test]
fn onboard_outcome_projects_total_failure_onto_dest_set() {
    let engine = engine();
    let onboard = OnboardHandle::new(
        ActionId::new(),
        Arc::downgrade(&engine),
        Arc::new(Mutex::new(ActionStatus::Failed(ActionFailure::AllBlocks))),
        vec![4, 7],
    );
    assert!(onboard.is_complete());
    assert_eq!(
        onboard.outcome(),
        Some(LoadOutcome::FailedPartial {
            block_ids: vec![4, 7],
        })
    );
}

#[test]
fn noop_offload_handle_is_immediately_terminal() {
    let engine = engine();
    let offload = Arc::clone(&engine)
        .offload(&"r1".to_string(), vec![])
        .unwrap();
    assert!(offload.is_complete());
    assert_eq!(offload.outcome(), Some(SaveOutcome::Done));
}

#[test]
fn noop_evict_mints_an_empty_fence_and_no_handle() {
    let engine = engine();
    let outcome = engine.evict(&"r1".to_string());
    assert_eq!(outcome.fence.request_id, "r1");
    assert!(outcome.fence.per_worker.is_empty());
    assert!(
        outcome.handle.is_none(),
        "nothing armed — there is no barrier for the leader to observe"
    );
}

/// The fence handle is a pure observational cell: it reads the shared
/// `AtomicBool` the engine flips and owns nothing — flipping the cell is the
/// ONLY thing that completes it, and dropping a handle leaves the cell (and any
/// sibling handle) untouched.
#[test]
fn fence_handle_observes_shared_cell() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let cell = Arc::new(AtomicBool::new(false));
    let handle = FenceHandle::new(Arc::clone(&cell));
    let sibling = FenceHandle::new(Arc::clone(&cell));
    assert!(!handle.is_complete());

    // No RAII side effects: dropping one observer changes nothing.
    drop(sibling);
    assert!(!handle.is_complete());
    assert!(!cell.load(Ordering::Acquire));

    // The single Pending→Complete transition (the engine barrier's drop).
    cell.store(true, Ordering::Release);
    assert!(handle.is_complete());
    assert!(
        format!("{handle:?}").contains("true"),
        "Debug shows the cell"
    );
}

#[test]
fn noop_offload_drain_commits_once() {
    let engine = engine();
    let drain = engine
        .take_offload_drain(&"r1".to_string())
        .expect("noop yields a no-op drain");
    // Consume-self: `commit` takes the drain by value. A second `commit` is a
    // compile error (proven by the doctest on `RequestOffloadDrain`).
    drain.commit();
}

#[test]
fn fence_types_are_wire_serializable() {
    // Contrast with the leader-local handles: the fence wire types round-trip.
    let token = FenceToken::new(3);
    let restored: FenceToken =
        serde_json::from_str(&serde_json::to_string(&token).unwrap()).unwrap();
    assert_eq!(token, restored);

    let fence = EvictionFence {
        request_id: "r1".to_string(),
        per_worker: vec![token],
    };
    let restored: EvictionFence =
        serde_json::from_str(&serde_json::to_string(&fence).unwrap()).unwrap();
    assert_eq!(fence, restored);
}

#[test]
fn accept_ids_are_process_unique() {
    assert_ne!(AcceptId::new(), AcceptId::new());
}

#[test]
fn worker_delegates_are_object_safe() {
    let sink: Arc<dyn EngineWorkerSink> = NoopWorkerSink::new();
    sink.mark_load_finished(&"r1".to_string(), LoadOutcome::Done);
    sink.mark_save_finished(&"r1".to_string(), SaveOutcome::Done);
    sink.mark_fence_complete(FenceToken::new(0));

    struct Driver;
    impl WorkerEngineDriver for Driver {
        fn begin_forward_pass(&self, _iteration: usize) {}
        fn finish_forward_pass(&self, _iteration: usize) {}
        fn await_fence(&self, _token: FenceToken) {}
        fn shutdown(&self) {}
    }
    let driver: Arc<dyn WorkerEngineDriver> = Arc::new(Driver);
    driver.begin_forward_pass(0);
    driver.await_fence(FenceToken::new(1));
    driver.finish_forward_pass(0);
    driver.shutdown();
}

// ---------------------------------------------------------------------------
// Unified find / onboard seam
// ---------------------------------------------------------------------------

fn find_blocks_req(id: &str) -> FindBlocksRequest {
    FindBlocksRequest {
        request_id: id.to_string(),
        sequence_hashes: Arc::from([]),
        num_computed_tokens: 0,
        total_tokens: 0,
        transfer_params: None,
    }
}

#[test]
fn noop_find_blocks_resolves_zero_no_async() {
    let engine = engine();
    match Arc::clone(&engine)
        .find_blocks(&find_blocks_req("r1"), None)
        .unwrap()
    {
        FindBlocksOutcome::Resolved {
            matched_tokens,
            minted,
            release_parked,
        } => {
            assert_eq!(matched_tokens, 0);
            assert!(minted.is_none(), "noop caches nothing — parks no handle");
            assert!(!release_parked);
        }
        other => panic!("noop must resolve zero, got {other:?}"),
    }
}

#[test]
fn noop_onboard_blocks_is_immediately_terminal() {
    let engine = engine();
    // Noop's find_blocks mints nothing, so fabricate a search-kind handle to
    // satisfy onboard_blocks' typed precondition (engine-minting permitted here).
    let handle =
        FindBlocksHandle::search("r1".to_string(), SearchId::new(), Arc::downgrade(&engine));
    let onboard = Arc::clone(&engine)
        .onboard_blocks(&handle, &[3, 4], 2)
        .unwrap();
    assert!(onboard.is_complete());
    assert_eq!(onboard.outcome(), Some(LoadOutcome::Done));
}

/// `find_blocks` carries the shared hash chain + counts + the WHOLE parsed
/// transfer params (the engine extracts `remote_prefill` internally) — and
/// never tokens. The chain is an `Arc`: a request clone bumps a refcount, it
/// never copies hashes.
#[test]
fn find_blocks_request_carries_chain_counts_and_transfer_params() {
    let params = RemotePrefillParams::new(uuid::Uuid::new_v4(), uuid::Uuid::new_v4().into());
    let chain: Arc<[super::protocol::SequenceHash]> = Arc::from([
        super::protocol::SequenceHash::default(),
        super::protocol::SequenceHash::default(),
        super::protocol::SequenceHash::default(),
    ]);
    let req = FindBlocksRequest {
        request_id: "r1".to_string(),
        sequence_hashes: Arc::clone(&chain),
        num_computed_tokens: 16,
        total_tokens: 48,
        transfer_params: Some(crate::disagg::TransferParams::remote_prefill(params)),
    };
    assert_eq!(req.sequence_hashes.len(), 3);
    assert_eq!(req.num_computed_tokens, 16);
    assert_eq!(req.total_tokens, 48);
    assert!(
        req.transfer_params
            .as_ref()
            .is_some_and(|t| t.remote_prefill.is_some())
    );
    let cloned = req.clone();
    assert!(
        Arc::ptr_eq(&cloned.sequence_hashes, &chain),
        "a request clone shares the chain Arc"
    );
    // Plain-local construction carries no params.
    assert!(find_blocks_req("r2").transfer_params.is_none());
}

/// The three outcome variants construct and pattern-match with the fields the
/// connector reads.
#[test]
fn find_blocks_outcome_variants_construct() {
    let engine = engine();
    assert!(matches!(
        FindBlocksOutcome::Deferred,
        FindBlocksOutcome::Deferred
    ));

    let searching = FindBlocksOutcome::Searching {
        minted: Some(FindBlocksHandle::search(
            "r1".to_string(),
            SearchId::new(),
            Arc::downgrade(&engine),
        )),
    };
    assert!(matches!(
        searching,
        FindBlocksOutcome::Searching { minted: Some(_) }
    ));

    match (FindBlocksOutcome::Resolved {
        matched_tokens: 64,
        minted: None,
        release_parked: true,
    }) {
        FindBlocksOutcome::Resolved {
            matched_tokens,
            minted,
            release_parked,
        } => {
            assert_eq!(matched_tokens, 64);
            assert!(minted.is_none());
            assert!(release_parked);
        }
        _ => unreachable!(),
    }
}

/// Recording engine for the [`FindBlocksHandle`] RAII suite: every verb is inert
/// except the two RAII release targets, which record their generation ids — the
/// observables the drop tests below pin.
struct FindBlocksReleaseRecorder {
    searches: Mutex<Vec<SearchId>>,
    prefills: Mutex<Vec<(RequestId, AcceptId)>>,
}

impl FindBlocksReleaseRecorder {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            searches: Mutex::new(Vec::new()),
            prefills: Mutex::new(Vec::new()),
        })
    }
}

impl LeaderEngine for FindBlocksReleaseRecorder {
    // The unified match/onboard verbs are inert here — the RAII drop tests below
    // build handles directly and never drive them.
    fn find_blocks(
        self: Arc<Self>,
        _req: &FindBlocksRequest,
        _live: Option<&FindBlocksHandle>,
    ) -> Result<FindBlocksOutcome, LeaderEngineError> {
        Ok(FindBlocksOutcome::Resolved {
            matched_tokens: 0,
            minted: None,
            release_parked: false,
        })
    }
    fn onboard_blocks(
        self: Arc<Self>,
        _handle: &FindBlocksHandle,
        _dest: &[usize],
        _num_external_tokens: usize,
    ) -> Result<OnboardHandle, LeaderEngineError> {
        Err(LeaderEngineError::Shutdown)
    }
    fn offload(
        self: Arc<Self>,
        _req: &RequestId,
        _pairs: Vec<(super::protocol::SequenceHash, usize)>,
    ) -> Result<super::handles::OffloadHandle, LeaderEngineError> {
        Err(LeaderEngineError::Shutdown)
    }
    fn evict(&self, req: &RequestId) -> EvictionOutcome {
        EvictionOutcome {
            fence: EvictionFence {
                request_id: req.clone(),
                per_worker: Vec::new(),
            },
            handle: None,
        }
    }
    fn take_offload_drain(&self, _req: &RequestId) -> Option<super::handles::RequestOffloadDrain> {
        None
    }
    fn poll_action(&self, _id: &ActionId) -> ActionStatus {
        ActionStatus::Complete
    }
    fn release_search(&self, id: &SearchId) {
        self.searches.lock().unwrap().push(*id);
    }
    fn release_action(&self, _action: &ActionId) {}
    fn release_prefill_session(&self, request_id: &RequestId, accept_id: AcceptId) {
        self.prefills
            .lock()
            .unwrap()
            .push((request_id.clone(), accept_id));
    }
}

/// A Search-kind handle's RAII drop fires `release_search` exactly once with the
/// embedded `SearchId`, and never touches the prefill release.
#[test]
fn find_blocks_handle_search_kind_drop_releases_search_generation() {
    let recorder = FindBlocksReleaseRecorder::new();
    let engine: Arc<dyn LeaderEngine> = Arc::clone(&recorder) as Arc<dyn LeaderEngine>;
    let id = SearchId::new();
    let handle = FindBlocksHandle::search("r1".to_string(), id, Arc::downgrade(&engine));
    assert_eq!(handle.request_id(), "r1");
    assert_eq!(handle.search_id(), Some(id));
    assert_eq!(handle.prefill_accept_id(), None);

    drop(handle);
    assert_eq!(recorder.searches.lock().unwrap().as_slice(), &[id]);
    assert!(recorder.prefills.lock().unwrap().is_empty());
}

/// A Prefill-kind handle's RAII drop fires `release_prefill_session` exactly once
/// with the embedded `(request_id, accept_id)` generation, and never touches the
/// search release.
#[test]
fn find_blocks_handle_prefill_kind_drop_releases_prefill_generation() {
    let recorder = FindBlocksReleaseRecorder::new();
    let engine: Arc<dyn LeaderEngine> = Arc::clone(&recorder) as Arc<dyn LeaderEngine>;
    let accept = AcceptId::new();
    let handle = FindBlocksHandle::prefill("r1".to_string(), accept, Arc::downgrade(&engine));
    assert_eq!(handle.prefill_accept_id(), Some(accept));
    assert_eq!(handle.search_id(), None);

    drop(handle);
    assert_eq!(
        recorder.prefills.lock().unwrap().as_slice(),
        &[("r1".to_string(), accept)],
        "drop released exactly this generation"
    );
    assert!(recorder.searches.lock().unwrap().is_empty());
}

/// A drop after the engine is gone (dead `Weak`) skips the release silently, for
/// either kind.
#[test]
fn find_blocks_handle_drop_after_engine_teardown_is_safe() {
    let recorder = FindBlocksReleaseRecorder::new();
    let weak = {
        let engine: Arc<dyn LeaderEngine> = Arc::clone(&recorder) as Arc<dyn LeaderEngine>;
        Arc::downgrade(&engine)
    };
    drop(recorder); // the only strong ref chain is gone
    let search = FindBlocksHandle::search("r1".to_string(), SearchId::new(), weak.clone());
    let prefill = FindBlocksHandle::prefill("r2".to_string(), AcceptId::new(), weak);
    drop(search); // must not panic on a dead Weak
    drop(prefill); // must not panic on a dead Weak
}
