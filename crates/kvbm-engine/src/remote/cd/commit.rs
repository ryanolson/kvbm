// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side conditional-disaggregation session choreography.
//!
//! The decode worker is the session HOLDER: it commits the hashes it will
//! provide and makes the backing G2 blocks available for the prefill peer to
//! pull. [`open_and_commit`] performs the up-front publish in the exact order
//! the legacy decode coordinator used, and [`AvailabilityLedger`] tracks the
//! deferred tail — blocks that are committed up front but land later (e.g.
//! remote-search-inflight pulls still settling into G2). Every deferral source
//! feeds the ONE ledger: `finish_availability` may not fire until the last
//! outstanding block lands.
//!
//! Pure: no runtime, no spawn, no I/O. The only collaborators are the injected
//! [`SessionFactory`] / [`Session`] (mockable) and the [`ImmutableBlock<G2>`]
//! the caller has already pinned.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, anyhow};

use kvbm_logical::blocks::ImmutableBlock;
use kvbm_protocols::disagg::SessionId;

use crate::p2p::session::{Session, SessionFactory};
use crate::{G2, SequenceHash};

/// The full planned commit set for one remote-prefill session, split by source.
///
/// The three vecs stay separate because the REQUEST side needs them apart: the
/// `RemotePrefillRequest` places local-match blocks positionally from the decode
/// offset, so it must carry only the local-match range (the positional / DNPT
/// contract). Only the SESSION side concatenates them. Conceptual
/// absolute-position layout:
///
/// * `prefix_hashes` occupy `[0, P)` — the vLLM-computed prefix window the
///   decode side serves from its own G2.
/// * `pending_hashes` occupy the same conceptual prefix/window slots their
///   blocks will land into — any deferred source (e.g. remote-search-inflight
///   G2 blocks the policy already counted as local). They are committed UP
///   FRONT (so `finish_commits` can seal the full planned set) even though
///   their blocks arrive later via [`AvailabilityLedger::deliver`].
/// * `local_match_hashes` follow, at `[P + pending, ..)`.
///
/// Session-side concatenation order is therefore prefix ++ pending ++
/// local-match — ABSOLUTE-POSITION order across the committed set.
pub(crate) struct RemoteCommitPlan {
    pub(crate) prefix_hashes: Vec<SequenceHash>,
    pub(crate) pending_hashes: Vec<SequenceHash>,
    pub(crate) local_match_hashes: Vec<SequenceHash>,
}

impl RemoteCommitPlan {
    /// Concatenate the planned commit set in absolute-position order:
    /// prefix ++ pending ++ local-match. The SESSION-side view only — the
    /// request side keeps the three ranges separate.
    fn session_commit_set(&self) -> Vec<SequenceHash> {
        let mut set = Vec::with_capacity(
            self.prefix_hashes.len() + self.pending_hashes.len() + self.local_match_hashes.len(),
        );
        set.extend_from_slice(&self.prefix_hashes);
        set.extend_from_slice(&self.pending_hashes);
        set.extend_from_slice(&self.local_match_hashes);
        set
    }
}

/// Tracks the committed-but-not-yet-available tail of a session's availability
/// set and fires `finish_availability` exactly once, when the last outstanding
/// source lands.
///
/// One ledger serves every deferral source (e.g. remote-search-inflight pulls)
/// because they share the single deferred-close state machine. A ledger
/// constructed with no outstanding hashes starts already drained (and sealed);
/// the no-pending happy path seals inside [`open_and_commit`].
pub(crate) struct AvailabilityLedger {
    /// Committed hashes whose blocks have not yet been made available.
    outstanding: HashSet<SequenceHash>,
    /// Hashes already delivered — kept so a re-delivery is diagnosed as such
    /// (refused, but distinctly from a never-committed hash).
    landed: HashSet<SequenceHash>,
    /// Once `true`, `finish_availability` is no longer pending: it has either
    /// fired (the ledger drained) or the session was abandoned. No further
    /// [`Self::deliver`] may fire the terminator.
    sealed: bool,
}

impl AvailabilityLedger {
    /// Deliver a batch of landed blocks.
    ///
    /// Every block's sequence hash MUST be outstanding — the commit set is
    /// sealed, so a hash that was never committed is a contract violation and
    /// returns `Err` WITHOUT calling `make_available` (nothing un-committed may
    /// be published). On success the blocks are handed to
    /// [`Session::make_available`] and removed from the outstanding set; when the
    /// set empties, [`Session::finish_availability`] fires exactly once.
    ///
    /// Returns whether THIS call drained the ledger. Idempotent against an
    /// already-drained (or abandoned) ledger: an empty deliver is `Ok(false)`
    /// and never double-fires `finish_availability`.
    pub(crate) fn deliver(
        &mut self,
        session: &Arc<dyn Session>,
        blocks: Vec<ImmutableBlock<G2>>,
    ) -> Result<bool> {
        if blocks.is_empty() {
            // Empty deliver never mutates and never fires the terminator —
            // idempotent no-op whether sealed or still outstanding.
            return Ok(false);
        }
        if self.sealed {
            return Err(anyhow!(
                "deliver: availability ledger is sealed; {} block(s) cannot be delivered",
                blocks.len()
            ));
        }
        // Validate the whole batch BEFORE any state mutation or make_available,
        // distinguishing the three refusal shapes: already delivered (refused —
        // re-publishing a landed hash), duplicated within this batch, and never
        // committed at all (the commit set is sealed).
        let mut seen_in_batch: HashSet<SequenceHash> = HashSet::new();
        for block in &blocks {
            let hash = block.sequence_hash();
            if self.landed.contains(&hash) {
                return Err(anyhow!(
                    "deliver: block hash {hash:?} was already delivered to this session"
                ));
            }
            if !seen_in_batch.insert(hash) {
                return Err(anyhow!(
                    "deliver: block hash {hash:?} is duplicated within the batch"
                ));
            }
            if !self.outstanding.contains(&hash) {
                return Err(anyhow!(
                    "deliver: block hash {hash:?} was never committed to this session"
                ));
            }
        }
        let hashes: Vec<SequenceHash> = blocks.iter().map(|b| b.sequence_hash()).collect();
        // make_available before mutating the ledger: if it errors the
        // outstanding set is left intact for the caller to abandon.
        session.make_available(blocks)?;
        for hash in &hashes {
            self.outstanding.remove(hash);
            self.landed.insert(*hash);
        }
        if self.outstanding.is_empty() {
            session.finish_availability()?;
            self.sealed = true;
            return Ok(true);
        }
        Ok(false)
    }

    /// Abandon the deferred tail: a pending source will never land (its
    /// transfer failed). Closes the session with `reason` and seals the ledger
    /// so no later [`Self::deliver`] can fire.
    pub(crate) fn abandon(&mut self, session: &Arc<dyn Session>, reason: &str) {
        session.close(Some(reason.to_string()));
        self.sealed = true;
    }

    /// `true` once every outstanding source has landed and `finish_availability`
    /// has fired (the success terminal). `false` while sources are still
    /// outstanding, and `false` after [`Self::abandon`] (which seals without
    /// draining).
    pub(crate) fn is_drained(&self) -> bool {
        self.sealed && self.outstanding.is_empty()
    }
}

/// Open the session and perform the up-front publish for a remote prefill.
///
/// Ordering is a 1:1 port of the legacy decode coordinator:
///   open
///     -> commit(prefix ++ pending ++ local-match) as ONE call
///     -> make_available(initial_blocks)
///     -> finish_commits          (seals the commit set)
///     -> finish_availability     ONLY if nothing is pending
///
/// `initial_blocks` are the blocks available NOW (the prefix + local-match G2 the
/// decode side already holds); the `pending_hashes` blocks land later through the
/// returned [`AvailabilityLedger`]. When `pending_hashes` is empty the ledger
/// starts drained and `finish_availability` fires here; otherwise it is DEFERRED.
///
/// On ANY session-op error the half-published session is closed (it must not
/// survive) and the error is returned — nothing may be appended after the
/// `finish_commits` seal.
pub(crate) fn open_and_commit(
    factory: &Arc<dyn SessionFactory>,
    session_id: SessionId,
    plan: &RemoteCommitPlan,
    initial_blocks: Vec<ImmutableBlock<G2>>,
) -> Result<(Arc<dyn Session>, AvailabilityLedger)> {
    let session = factory.open(session_id)?;

    let commit_set = plan.session_commit_set();
    if let Err(e) = session.commit(commit_set) {
        return close_and_err(&session, "commit", e);
    }
    if let Err(e) = session.make_available(initial_blocks) {
        return close_and_err(&session, "make_available", e);
    }
    if let Err(e) = session.finish_commits() {
        return close_and_err(&session, "finish_commits", e);
    }

    let mut ledger = AvailabilityLedger {
        outstanding: plan.pending_hashes.iter().copied().collect(),
        landed: HashSet::new(),
        sealed: false,
    };
    if ledger.outstanding.is_empty() {
        // Nothing deferred: the availability set is complete now.
        if let Err(e) = session.finish_availability() {
            return close_and_err(&session, "finish_availability", e);
        }
        ledger.sealed = true;
    }

    Ok((session, ledger))
}

/// Close `session` (a failed op must not leave a half-published session alive)
/// and surface the originating error.
fn close_and_err<T>(session: &Arc<dyn Session>, op: &str, err: anyhow::Error) -> Result<T> {
    session.close(Some(format!("{op} failed: {err}")));
    Err(err)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use futures::future::BoxFuture;
    use parking_lot::Mutex;

    use kvbm_logical::blocks::MutableBlock;
    use kvbm_logical::manager::BlockManager;
    use kvbm_protocols::disagg::SessionEndpoint;

    use crate::InstanceId;
    use crate::p2p::session::{
        AvailabilityStream, CommitStream, LifecycleStream, MockSessionFactory, PeerAvailable,
        PeerCommitted,
    };
    use crate::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
    use crate::testing::token_blocks::create_token_sequence;

    // `Session`, `SessionFactory`, `ImmutableBlock<G2>`, `SequenceHash`, `G2`,
    // `SessionId`, `Arc`, `Result`, and the `anyhow!` macro arrive via the glob.
    use super::*;

    const BS: usize = 16;

    fn g2_manager(count: usize) -> Arc<BlockManager<G2>> {
        let registry = TestRegistryBuilder::new().build();
        Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(count)
                .block_size(BS)
                .registry(registry)
                .build(),
        )
    }

    /// Allocate `count` registered immutable G2 blocks whose hash chain starts at
    /// `start_token`. Distinct `start_token`s yield disjoint hash sets.
    fn immutables(
        manager: &Arc<BlockManager<G2>>,
        count: usize,
        start_token: u32,
    ) -> Vec<ImmutableBlock<G2>> {
        let seq = create_token_sequence(count, BS, start_token);
        let mutable = manager.allocate_blocks(count).expect("alloc failed");
        let complete: Vec<_> = mutable
            .into_iter()
            .zip(seq.blocks().iter())
            .map(|(b, tb)| b.complete(tb).expect("complete failed"))
            .collect();
        manager.register_blocks(complete)
    }

    fn hashes(blocks: &[ImmutableBlock<G2>]) -> Vec<SequenceHash> {
        blocks.iter().map(|b| b.sequence_hash()).collect()
    }

    fn dyn_factory(factory: &Arc<MockSessionFactory>) -> Arc<dyn SessionFactory> {
        factory.clone()
    }

    // ------------------------------------------------------------------------
    // t1: no-pending happy path
    // ------------------------------------------------------------------------

    #[test]
    fn no_pending_single_commit_and_starts_drained() {
        let mgr = g2_manager(4);
        let prefix = immutables(&mgr, 1, 0);
        let local = immutables(&mgr, 1, 1000);
        let prefix_h = hashes(&prefix);
        let local_h = hashes(&local);

        let plan = RemoteCommitPlan {
            prefix_hashes: prefix_h.clone(),
            pending_hashes: Vec::new(),
            local_match_hashes: local_h.clone(),
        };
        // commit set == initial make_available order == prefix ++ local.
        let expected: Vec<SequenceHash> = prefix_h.iter().chain(local_h.iter()).copied().collect();
        let initial: Vec<ImmutableBlock<G2>> = prefix.into_iter().chain(local).collect();

        let factory = MockSessionFactory::new();
        let (_session, ledger) =
            open_and_commit(&dyn_factory(&factory), uuid::Uuid::new_v4(), &plan, initial)
                .expect("open_and_commit ok");

        let mock = factory.last_opened().expect("session opened");
        // Exactly one commit call, hash order == prefix ++ pending(empty) ++ local.
        assert_eq!(mock.commit_calls(), vec![expected.clone()]);
        // Initial blocks made available (recorded before the seal ran).
        assert_eq!(mock.make_available_calls(), vec![expected]);
        assert!(mock.finish_commits_called());
        assert!(mock.finish_availability_called());
        assert!(ledger.is_drained(), "ledger starts drained with no pending");
    }

    // ------------------------------------------------------------------------
    // t2: pending defers the seal but commits up front
    // ------------------------------------------------------------------------

    #[test]
    fn pending_defers_finish_availability_but_commits_up_front() {
        let mgr = g2_manager(8);
        let prefix = immutables(&mgr, 1, 0);
        let local = immutables(&mgr, 1, 1000);
        let pending = immutables(&mgr, 2, 2000);
        let prefix_h = hashes(&prefix);
        let local_h = hashes(&local);
        let pending_h = hashes(&pending);

        let plan = RemoteCommitPlan {
            prefix_hashes: prefix_h.clone(),
            pending_hashes: pending_h.clone(),
            local_match_hashes: local_h.clone(),
        };
        let expected_commit: Vec<SequenceHash> = prefix_h
            .iter()
            .chain(pending_h.iter())
            .chain(local_h.iter())
            .copied()
            .collect();
        // Pending blocks are NOT in initial — they land later.
        let initial: Vec<ImmutableBlock<G2>> = prefix.into_iter().chain(local).collect();

        let factory = MockSessionFactory::new();
        let (_session, ledger) =
            open_and_commit(&dyn_factory(&factory), uuid::Uuid::new_v4(), &plan, initial)
                .expect("open_and_commit ok");

        let mock = factory.last_opened().expect("session opened");
        // Pending hashes committed up front, in absolute-position order.
        assert_eq!(mock.commit_calls(), vec![expected_commit]);
        assert!(mock.finish_commits_called());
        // Deferred: finish_availability NOT fired by open_and_commit.
        assert!(!mock.finish_availability_called());
        assert!(!ledger.is_drained());
    }

    // ------------------------------------------------------------------------
    // t3: incremental landing fires finish_availability exactly once
    // ------------------------------------------------------------------------

    #[test]
    fn incremental_landing_fires_finish_availability_once() {
        let mgr = g2_manager(8);
        let prefix = immutables(&mgr, 1, 0);
        let local = immutables(&mgr, 1, 1000);
        let pending_a = immutables(&mgr, 1, 2000);
        let pending_b = immutables(&mgr, 1, 3000);
        let pending_h: Vec<SequenceHash> = hashes(&pending_a)
            .into_iter()
            .chain(hashes(&pending_b))
            .collect();

        let plan = RemoteCommitPlan {
            prefix_hashes: hashes(&prefix),
            pending_hashes: pending_h,
            local_match_hashes: hashes(&local),
        };
        let initial: Vec<ImmutableBlock<G2>> = prefix.into_iter().chain(local).collect();

        let factory = MockSessionFactory::new();
        let (session, mut ledger) =
            open_and_commit(&dyn_factory(&factory), uuid::Uuid::new_v4(), &plan, initial)
                .expect("open_and_commit ok");
        let mock = factory.last_opened().expect("session opened");

        // batch1: pending_a lands -> not drained, finish_availability not fired.
        let drained1 = ledger.deliver(&session, pending_a).expect("deliver a");
        assert!(!drained1, "one pending block still outstanding");
        assert!(!mock.finish_availability_called());
        assert!(!ledger.is_drained());
        // initial make_available + batch1 == 2 recorded calls.
        assert_eq!(mock.make_available_calls().len(), 2);

        // batch2 = the rest: pending_b lands -> drained.
        let drained2 = ledger.deliver(&session, pending_b).expect("deliver b");
        assert!(drained2, "last source landed -> drained");
        assert!(mock.finish_availability_called());
        assert!(ledger.is_drained());
        assert_eq!(mock.make_available_calls().len(), 3);

        // A further empty deliver is a no-op: no second finish_availability and
        // no extra make_available.
        let drained3 = ledger.deliver(&session, Vec::new()).expect("empty deliver");
        assert!(!drained3);
        assert!(ledger.is_drained());
        assert_eq!(mock.make_available_calls().len(), 3);
    }

    // ------------------------------------------------------------------------
    // t4: a never-committed hash is a contract violation
    // ------------------------------------------------------------------------

    #[test]
    fn unexpected_hash_errors_without_make_available() {
        let mgr = g2_manager(8);
        let prefix = immutables(&mgr, 1, 0);
        let local = immutables(&mgr, 1, 1000);
        let pending = immutables(&mgr, 1, 2000);
        // A block whose hash was never committed to this session.
        let stray = immutables(&mgr, 1, 9000);

        let plan = RemoteCommitPlan {
            prefix_hashes: hashes(&prefix),
            pending_hashes: hashes(&pending),
            local_match_hashes: hashes(&local),
        };
        let initial: Vec<ImmutableBlock<G2>> = prefix.into_iter().chain(local).collect();

        let factory = MockSessionFactory::new();
        let (session, mut ledger) =
            open_and_commit(&dyn_factory(&factory), uuid::Uuid::new_v4(), &plan, initial)
                .expect("open_and_commit ok");
        let mock = factory.last_opened().expect("session opened");
        let before = mock.make_available_calls().len();

        let err = ledger
            .deliver(&session, stray)
            .expect_err("stray hash must error");
        assert!(
            err.to_string().contains("never committed"),
            "unexpected error: {err}"
        );
        // No make_available for the stray block, no seal.
        assert_eq!(mock.make_available_calls().len(), before);
        assert!(!mock.finish_availability_called());
        assert!(!ledger.is_drained());
    }

    // ------------------------------------------------------------------------
    // t5: abandon closes the session and seals the ledger
    // ------------------------------------------------------------------------

    #[test]
    fn abandon_closes_and_seals_so_later_deliver_errors() {
        let mgr = g2_manager(8);
        let prefix = immutables(&mgr, 1, 0);
        let local = immutables(&mgr, 1, 1000);
        let pending = immutables(&mgr, 1, 2000);

        let plan = RemoteCommitPlan {
            prefix_hashes: hashes(&prefix),
            pending_hashes: hashes(&pending),
            local_match_hashes: hashes(&local),
        };
        let initial: Vec<ImmutableBlock<G2>> = prefix.into_iter().chain(local).collect();

        let factory = MockSessionFactory::new();
        let (session, mut ledger) =
            open_and_commit(&dyn_factory(&factory), uuid::Uuid::new_v4(), &plan, initial)
                .expect("open_and_commit ok");
        let mock = factory.last_opened().expect("session opened");

        ledger.abandon(&session, "promotion failed");
        assert_eq!(
            mock.closed_reason(),
            Some(Some("promotion failed".to_string()))
        );
        assert!(!ledger.is_drained(), "abandon seals without draining");

        // A later deliver into the sealed ledger errors (the source is gone).
        let err = ledger
            .deliver(&session, pending)
            .expect_err("deliver after abandon must error");
        assert!(
            err.to_string().contains("sealed"),
            "unexpected error: {err}"
        );
    }

    // ------------------------------------------------------------------------
    // t6: open_and_commit closes the half-published session on any op error
    // ------------------------------------------------------------------------

    #[test]
    fn open_and_commit_closes_session_on_make_available_error() {
        // Realistic reproducer: an initial block whose hash is NOT in the commit
        // set trips the session's make_available committed-set validation.
        let mgr = g2_manager(8);
        let prefix = immutables(&mgr, 1, 0);
        let local = immutables(&mgr, 1, 1000);
        let stray = immutables(&mgr, 1, 9000);

        let plan = RemoteCommitPlan {
            prefix_hashes: hashes(&prefix),
            pending_hashes: Vec::new(),
            local_match_hashes: hashes(&local),
        };
        // initial carries a block whose hash was never committed.
        let initial: Vec<ImmutableBlock<G2>> =
            prefix.into_iter().chain(local).chain(stray).collect();

        let factory = MockSessionFactory::new();
        let result = open_and_commit(&dyn_factory(&factory), uuid::Uuid::new_v4(), &plan, initial);
        assert!(result.is_err(), "make_available failure must propagate");

        let mock = factory.last_opened().expect("session was opened");
        let reason = mock
            .closed_reason()
            .expect("half-published session must be closed")
            .expect("close carries a reason");
        assert!(
            reason.contains("make_available"),
            "unexpected close reason: {reason}"
        );
    }

    #[test]
    fn open_and_commit_closes_session_on_seal_op_errors() {
        // The MockSession can only force make_available to fail; cover the seal
        // arms (finish_commits / finish_availability) with a stub that injects an
        // error from one chosen op and records the close reason.
        for op in ["commit", "finish_commits", "finish_availability"] {
            let session = FailingSession::arc(op);
            let factory: Arc<dyn SessionFactory> = Arc::new(FailingFactory {
                session: session.clone(),
            });
            let plan = RemoteCommitPlan {
                prefix_hashes: Vec::new(),
                pending_hashes: Vec::new(),
                local_match_hashes: Vec::new(),
            };
            let result = open_and_commit(&factory, uuid::Uuid::new_v4(), &plan, Vec::new());
            assert!(result.is_err(), "{op}: expected Err");
            let closed = session
                .closed
                .lock()
                .clone()
                .expect("session must be closed on error");
            assert!(
                closed.contains(op),
                "{op}: close reason was {closed:?}, expected it to name the op"
            );
        }
    }

    /// The exact holder-op interleaving is the legacy decode coordinator's:
    /// commit -> make_available -> finish_commits -> finish_availability (the
    /// last only when nothing is pending). MockSession has no global op log, so
    /// this pins the order through the logging stub.
    #[test]
    fn open_and_commit_pins_legacy_op_order() {
        // No pending: all four ops, in order.
        let session = FailingSession::arc("");
        let factory: Arc<dyn SessionFactory> = Arc::new(FailingFactory {
            session: session.clone(),
        });
        let plan = RemoteCommitPlan {
            prefix_hashes: Vec::new(),
            pending_hashes: Vec::new(),
            local_match_hashes: Vec::new(),
        };
        open_and_commit(&factory, uuid::Uuid::new_v4(), &plan, Vec::new()).expect("ok");
        assert_eq!(
            *session.log.lock(),
            vec![
                "commit",
                "make_available",
                "finish_commits",
                "finish_availability"
            ]
        );

        // Pending: the availability terminator is deferred.
        let session = FailingSession::arc("");
        let factory: Arc<dyn SessionFactory> = Arc::new(FailingFactory {
            session: session.clone(),
        });
        let mgr = g2_manager(1);
        let plan = RemoteCommitPlan {
            prefix_hashes: Vec::new(),
            pending_hashes: hashes(&immutables(&mgr, 1, 7000)),
            local_match_hashes: Vec::new(),
        };
        open_and_commit(&factory, uuid::Uuid::new_v4(), &plan, Vec::new()).expect("ok");
        assert_eq!(
            *session.log.lock(),
            vec!["commit", "make_available", "finish_commits"]
        );
    }

    /// Re-delivering an already-landed hash mid-deferral is refused with the
    /// already-delivered diagnosis, not the never-committed one.
    #[test]
    fn redelivery_is_diagnosed_as_already_delivered() {
        let mgr = g2_manager(8);
        let pending_a = immutables(&mgr, 1, 2000);
        let pending_b = immutables(&mgr, 1, 3000);
        let dup = vec![pending_a[0].clone()];
        let pending_h: Vec<SequenceHash> = hashes(&pending_a)
            .into_iter()
            .chain(hashes(&pending_b))
            .collect();

        let plan = RemoteCommitPlan {
            prefix_hashes: Vec::new(),
            pending_hashes: pending_h,
            local_match_hashes: Vec::new(),
        };

        let factory = MockSessionFactory::new();
        let (session, mut ledger) = open_and_commit(
            &dyn_factory(&factory),
            uuid::Uuid::new_v4(),
            &plan,
            Vec::new(),
        )
        .expect("open_and_commit ok");

        ledger.deliver(&session, pending_a).expect("first deliver");
        let err = ledger
            .deliver(&session, dup)
            .expect_err("re-delivery must error");
        assert!(
            err.to_string().contains("already delivered"),
            "unexpected error: {err}"
        );
        // The ledger is still live for the remaining source.
        assert!(!ledger.is_drained());
        ledger.deliver(&session, pending_b).expect("final deliver");
        assert!(ledger.is_drained());
    }

    // ------------------------------------------------------------------------
    // Failing holder-side Session stub for the seal-arm error tests.
    // ------------------------------------------------------------------------

    /// Returns `Err` from one chosen holder op (or never, with `fail_op = ""`),
    /// records every holder-op invocation in order, and records its `close`
    /// reason. Puller-side surfaces are unreachable for [`open_and_commit`].
    struct FailingSession {
        fail_op: &'static str,
        log: Mutex<Vec<&'static str>>,
        closed: Mutex<Option<String>>,
    }

    impl FailingSession {
        fn arc(fail_op: &'static str) -> Arc<Self> {
            Arc::new(Self {
                fail_op,
                log: Mutex::new(Vec::new()),
                closed: Mutex::new(None),
            })
        }

        fn op(&self, op: &'static str) -> Result<()> {
            self.log.lock().push(op);
            if self.fail_op == op {
                Err(anyhow!("injected {op} failure"))
            } else {
                Ok(())
            }
        }
    }

    impl Session for FailingSession {
        fn session_id(&self) -> SessionId {
            uuid::Uuid::nil()
        }
        fn endpoint(&self) -> Option<SessionEndpoint> {
            None
        }
        fn commit(&self, _hashes: Vec<SequenceHash>) -> Result<()> {
            self.op("commit")
        }
        fn finish_commits(&self) -> Result<()> {
            self.op("finish_commits")
        }
        fn make_available(&self, _blocks: Vec<ImmutableBlock<G2>>) -> Result<()> {
            self.op("make_available")
        }
        fn finish_availability(&self) -> Result<()> {
            self.op("finish_availability")
        }
        fn commits(&self) -> CommitStream {
            unreachable!("puller-side surface unused by open_and_commit")
        }
        fn availability(&self) -> AvailabilityStream {
            unreachable!("puller-side surface unused by open_and_commit")
        }
        fn peer_committed(&self) -> PeerCommitted {
            unreachable!("puller-side surface unused by open_and_commit")
        }
        fn peer_available(&self) -> PeerAvailable {
            unreachable!("puller-side surface unused by open_and_commit")
        }
        fn pull(
            &self,
            _hashes: Vec<SequenceHash>,
            _dst: Vec<MutableBlock<G2>>,
        ) -> BoxFuture<'static, Result<Vec<MutableBlock<G2>>>> {
            unreachable!("puller-side surface unused by open_and_commit")
        }
        fn lifecycle(&self) -> LifecycleStream {
            unreachable!("puller-side surface unused by open_and_commit")
        }
        fn finalize(&self, _reason: Option<String>) {}
        fn close(&self, reason: Option<String>) {
            *self.closed.lock() = reason;
        }
    }

    struct FailingFactory {
        session: Arc<FailingSession>,
    }

    impl SessionFactory for FailingFactory {
        fn open(&self, _session_id: SessionId) -> Result<Arc<dyn Session>> {
            Ok(self.session.clone())
        }
        fn attach(
            &self,
            _session_id: SessionId,
            _peer_instance_id: InstanceId,
            _peer_endpoint: SessionEndpoint,
        ) -> BoxFuture<'static, Result<Arc<dyn Session>>> {
            unreachable!("attach unused by open_and_commit")
        }
        fn active_session_count(&self) -> usize {
            0
        }
    }
}
