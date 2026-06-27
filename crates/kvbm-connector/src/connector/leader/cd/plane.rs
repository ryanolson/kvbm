// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production [`PrefillPlane`]: assemble the on-wire
//! [`RemotePrefillRequest`] and enqueue it for a prefill worker.
//!
//! The engine's [`PrefillDispatch`] carries only what the engine owns at
//! dispatch time (request/session correlation, the attach-back endpoint, the
//! provided-prefix and committed-window token counts). The wire form's
//! `token_ids` and the provided-prefix hash digest never cross the engine
//! seam — this plane reads both from the leader's slot map at dispatch time.

use std::sync::{Arc, Weak};

use anyhow::{Result, anyhow, bail};
use futures::FutureExt;
use futures::future::BoxFuture;

use kvbm_common::SequenceHash;
use kvbm_engine::cd::{PrefillDispatch, PrefillPlane};
use kvbm_hub::ConditionalDisaggClient;
use kvbm_protocols::disagg::{
    DISAGG_PROTOCOL_VERSION, KvHashingRequestEnvelope, RemotePrefillRequest, digest_provided_hashes,
};

use crate::InstanceId;

use super::super::Leader;

/// Enqueue seam for the completed wire request. The production impl wraps a
/// hub-queue [`ConditionalDisaggClient`]; tests record.
pub(in super::super) trait CdEnqueue: Send + Sync {
    fn push(&self, req: RemotePrefillRequest) -> BoxFuture<'static, Result<()>>;
}

/// Production enqueue: the hub's CD prefill queue
/// (`ConditionalDisaggClient::push_prefill_request`).
///
/// Built with the worker's role as registered: on a Decode-role worker every
/// dispatch enqueues; on a Prefill-role worker the engine's translated policy
/// is `Never` so the engine never dispatches — and if it ever did, the
/// client's hard role guard rejects the push loudly rather than letting a
/// prefill worker feed its own queue.
pub(in super::super) struct HubCdEnqueue {
    client: Arc<ConditionalDisaggClient>,
}

impl HubCdEnqueue {
    pub(in super::super) fn new(client: Arc<ConditionalDisaggClient>) -> Arc<Self> {
        Arc::new(Self { client })
    }
}

impl CdEnqueue for HubCdEnqueue {
    fn push(&self, req: RemotePrefillRequest) -> BoxFuture<'static, Result<()>> {
        let client = Arc::clone(&self.client);
        async move { client.push_prefill_request(&req).await }.boxed()
    }
}

/// Production [`PrefillPlane`] over the leader's slot map.
///
/// Holds a `Weak<Leader>` — a strong reference would close an `Arc` cycle
/// (`Leader` → engine → CD runtime → plane → `Leader`). The dispatch future
/// upgrades it, takes the leader state lock just long enough to clone the
/// slot's token ids + absolute PLH chain, and DROPS the lock before the
/// enqueue await.
pub(in super::super) struct SlotPrefillPlane {
    /// This decode's velo instance id — the wire's `initiator_instance_id`,
    /// which the prefill side resolves through the hub to attach back.
    own_instance_id: InstanceId,
    /// Layout block size: converts the dispatch's `num_provided_tokens` into
    /// the digest's PLH-slice length.
    block_size: usize,
    enqueue: Arc<dyn CdEnqueue>,
    leader: Weak<Leader>,
}

impl SlotPrefillPlane {
    pub(in super::super) fn new(
        own_instance_id: InstanceId,
        block_size: usize,
        enqueue: Arc<dyn CdEnqueue>,
        leader: Weak<Leader>,
    ) -> Arc<Self> {
        Arc::new(Self {
            own_instance_id,
            block_size,
            enqueue,
            leader,
        })
    }

    /// Snapshot the slot data the wire request needs: the prompt token ids
    /// from absolute position 0 truncated at the committed window end
    /// (`num_window_tokens`), and the full absolute-position PLH chain.
    /// Starting at position 0 keeps prefill hashing the same
    /// `TokenBlockSequence` decode hashed, so its PLH chain matches at
    /// absolute positions; truncating at the window end keeps prefill from
    /// computing or offloading blocks decode's pull plan never covers — the
    /// partial tail block, if any, stays on decode. Holds the leader state
    /// lock only for the clone.
    fn snapshot_slot(
        &self,
        request_id: &str,
        num_window_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<SequenceHash>)> {
        let leader = self
            .leader
            .upgrade()
            .ok_or_else(|| anyhow!("cd plane: leader dropped before prefill dispatch"))?;
        let state = leader.state.lock();
        let slot = state
            .get(request_id)
            .ok_or_else(|| anyhow!("cd plane: no slot for request_id={request_id}"))?;
        // Loud-fail guards for not-yet-wired CD inputs (mirrors the legacy
        // decode enqueue site): the wire envelope mirrors `kv_hashing::Request`
        // so enabling LoRA / salt is a value change, not a wire-format change —
        // but until each is validated end-to-end on the prefill recompute
        // path, fail here instead of risking silent hash divergence.
        // Timing divergence from legacy: legacy bails BEFORE the session opens
        // or budget reserves; this guard runs after the engine's search-time
        // Remote commit, so a guarded request burns one session+budget cycle
        // before the failed-load recompute. Acceptable while LoRA/salt are
        // unrouted; a search-seam CD-ineligible hint removes the cost if these
        // become common.
        if slot.lora_name.is_some() {
            bail!(
                "CD wire: LoRA not yet wired end-to-end for conditional disagg \
                 (request_id={request_id}); the prefill-side recompute path is not validated"
            );
        }
        if slot.salt.is_some() {
            bail!(
                "CD wire: cache_salt not yet wired end-to-end for conditional disagg \
                 (request_id={request_id}); the prefill-side recompute path is not validated"
            );
        }
        let total_tokens = slot.sequence.total_tokens();
        if num_window_tokens > total_tokens {
            bail!(
                "cd plane: prefill window [0..{num_window_tokens}] out of bounds for \
                 {total_tokens} tokens (request_id={request_id})"
            );
        }
        let token_ids: Vec<u32> = slot.sequence.tokens_at(0..num_window_tokens).into();
        let chain = slot.all_sequence_hashes();
        Ok((token_ids, chain))
    }
}

impl PrefillPlane for SlotPrefillPlane {
    fn dispatch(&self, req: PrefillDispatch) -> BoxFuture<'static, Result<()>> {
        let PrefillDispatch {
            request_id,
            session_id,
            decode_endpoint,
            num_provided_tokens,
            num_window_tokens,
        } = req;

        let snapshot = self.snapshot_slot(&request_id, num_window_tokens);
        let own_instance_id = self.own_instance_id;
        let block_size = self.block_size;
        let enqueue = Arc::clone(&self.enqueue);

        async move {
            let (token_ids, chain) = snapshot?;

            // The digest pins exactly the `[0, DNPT/BS)` PLH slice decode
            // commits to serve; the prefill side recomputes the same slice
            // from its own absolute-coordinate chain and asserts equality
            // before accepting the dispatch.
            let dnpt_blocks = num_provided_tokens / block_size;
            if dnpt_blocks > chain.len() {
                bail!(
                    "cd plane: provided-token window covers {dnpt_blocks} blocks but the \
                     slot's hash chain has only {} (request_id={request_id})",
                    chain.len()
                );
            }
            let expected_hash_digest = Some(digest_provided_hashes(&chain[..dnpt_blocks]));

            let request = RemotePrefillRequest {
                protocol_version: DISAGG_PROTOCOL_VERSION,
                request_id,
                session_id,
                initiator_instance_id: own_instance_id,
                decode_endpoint,
                token_ids,
                num_provided_tokens,
                // Empty by the guards above: no LoRA / salt / multimodal
                // crosses the CD wire yet.
                request: KvHashingRequestEnvelope::default(),
                expected_hash_digest,
            };
            enqueue.push(request).await
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;

    use crate::common::Request;
    use crate::connector::engine::noop_leader_engine;

    use super::*;

    const BLOCK_SIZE: usize = 4;

    /// Records every pushed wire request.
    #[derive(Default)]
    struct RecordingEnqueue {
        pushed: Mutex<Vec<RemotePrefillRequest>>,
    }

    impl CdEnqueue for RecordingEnqueue {
        fn push(&self, req: RemotePrefillRequest) -> BoxFuture<'static, Result<()>> {
            self.pushed.lock().push(req);
            async { Ok(()) }.boxed()
        }
    }

    fn leader_with_slot(request: Request) -> Arc<Leader> {
        let leader = Arc::new(Leader::with_engine(noop_leader_engine(), BLOCK_SIZE));
        leader.create_slot(request).expect("create slot");
        leader
    }

    fn plain_request(request_id: &str, tokens: Vec<u32>) -> Request {
        Request::new(request_id, tokens, None, None, None)
    }

    fn plane_over(
        leader: &Arc<Leader>,
        own: InstanceId,
        enqueue: Arc<RecordingEnqueue>,
    ) -> Arc<SlotPrefillPlane> {
        SlotPrefillPlane::new(own, BLOCK_SIZE, enqueue, Arc::downgrade(leader))
    }

    fn dispatch_for(
        request_id: &str,
        num_provided_tokens: usize,
        num_window_tokens: usize,
    ) -> PrefillDispatch {
        PrefillDispatch {
            request_id: request_id.to_string(),
            session_id: uuid::Uuid::new_v4(),
            decode_endpoint: Some(kvbm_protocols::disagg::SessionEndpoint {
                kind: "velo-streaming".to_string(),
                payload: serde_json::json!({"anchor": "a-1"}),
            }),
            num_provided_tokens,
            num_window_tokens,
        }
    }

    #[tokio::test]
    async fn dispatch_assembles_wire_request_from_slot() {
        // 18 tokens at block size 4 ⇒ 4 complete blocks + a 2-token partial
        // tail. The committed window covers the 4 full blocks; the tail stays
        // on decode.
        let tokens: Vec<u32> = (0..18).collect();
        let leader = leader_with_slot(plain_request("req-1", tokens.clone()));
        let own: InstanceId = uuid::Uuid::new_v4().into();
        let enqueue = Arc::new(RecordingEnqueue::default());
        let plane = plane_over(&leader, own, enqueue.clone());

        // DNPT of 8 tokens ⇒ digest covers exactly chain[..2].
        let dispatch = dispatch_for("req-1", 8, 16);
        let session_id = dispatch.session_id;
        let endpoint = dispatch.decode_endpoint.clone();
        plane.dispatch(dispatch).await.expect("dispatch ok");

        // The slot's own absolute-position chain is the digest's source of
        // truth; recompute the prefill-side acceptance check over the same
        // slice and assert the round-trip matches.
        let chain = {
            let state = leader.state.lock();
            state.get("req-1").expect("slot").all_sequence_hashes()
        };
        assert_eq!(chain.len(), 4);

        let pushed = enqueue.pushed.lock();
        assert_eq!(pushed.len(), 1, "exactly one wire request");
        let req = &pushed[0];
        assert_eq!(req.protocol_version, DISAGG_PROTOCOL_VERSION);
        assert_eq!(req.request_id, "req-1");
        assert_eq!(req.session_id, session_id);
        assert_eq!(req.initiator_instance_id, own);
        assert_eq!(req.decode_endpoint, endpoint);
        assert_eq!(
            req.token_ids,
            tokens[..16],
            "wire carries tokens from position 0 truncated at the committed \
             window end — the partial tail stays on decode"
        );
        assert_eq!(req.num_provided_tokens, 8);
        assert!(req.request.is_empty(), "envelope must be empty");
        assert_eq!(
            req.expected_hash_digest,
            Some(digest_provided_hashes(&chain[..2])),
            "digest covers exactly the DNPT/BS prefix of the slot's chain"
        );
        // Sanity: a different slice length yields a different digest, so the
        // assertion above genuinely pins the slice.
        assert_ne!(
            req.expected_hash_digest,
            Some(digest_provided_hashes(&chain[..3]))
        );
    }

    #[tokio::test]
    async fn dispatch_digest_spans_computed_plus_local_window() {
        // 24 tokens ⇒ 6 complete blocks. DNPT = 12 (3 blocks): a decode-side
        // computed prefix of 2 blocks plus a 1-block local match. DNPT is a
        // LENGTH, not a set — the digest pins the ABSOLUTE [0, DNPT/BS)
        // slice (a hole at the computed prefix is not expressible on the
        // wire) and the token window spans [0, num_window) from position 0.
        let tokens: Vec<u32> = (0..24).collect();
        let leader = leader_with_slot(plain_request("req-1", tokens.clone()));
        let enqueue = Arc::new(RecordingEnqueue::default());
        let plane = plane_over(&leader, uuid::Uuid::new_v4().into(), enqueue.clone());

        plane
            .dispatch(dispatch_for("req-1", 12, 20))
            .await
            .expect("dispatch ok");

        let chain = {
            let state = leader.state.lock();
            state.get("req-1").expect("slot").all_sequence_hashes()
        };
        let pushed = enqueue.pushed.lock();
        assert_eq!(pushed.len(), 1);
        let req = &pushed[0];
        assert_eq!(req.num_provided_tokens, 12);
        assert_eq!(
            req.token_ids,
            tokens[..20],
            "wire tokens cover the committed window from position 0"
        );
        assert_eq!(
            req.expected_hash_digest,
            Some(digest_provided_hashes(&chain[..3])),
            "digest = absolute [0, DNPT/BS) — the computed prefix is inside \
             the digest, never a hole"
        );
        assert_ne!(
            req.expected_hash_digest,
            Some(digest_provided_hashes(&chain[2..3])),
            "a suffix-only slice diverges"
        );
    }

    #[tokio::test]
    async fn dispatch_unknown_request_id_errors() {
        let leader = leader_with_slot(plain_request("req-1", (0..16).collect()));
        let enqueue = Arc::new(RecordingEnqueue::default());
        let plane = plane_over(&leader, uuid::Uuid::new_v4().into(), enqueue.clone());

        let err = plane
            .dispatch(dispatch_for("req-unknown", 4, 16))
            .await
            .expect_err("unknown request must error");
        assert!(err.to_string().contains("no slot"), "got: {err}");
        assert!(enqueue.pushed.lock().is_empty(), "nothing enqueued");
    }

    #[tokio::test]
    async fn dispatch_window_exceeding_chain_errors() {
        // 16 tokens ⇒ 4 complete blocks; a 20-token DNPT claims 5.
        let leader = leader_with_slot(plain_request("req-1", (0..16).collect()));
        let enqueue = Arc::new(RecordingEnqueue::default());
        let plane = plane_over(&leader, uuid::Uuid::new_v4().into(), enqueue.clone());

        let err = plane
            .dispatch(dispatch_for("req-1", 20, 16))
            .await
            .expect_err("oversized window must error");
        assert!(err.to_string().contains("hash chain"), "got: {err}");
        assert!(enqueue.pushed.lock().is_empty(), "nothing enqueued");
    }

    #[tokio::test]
    async fn dispatch_window_exceeding_tokens_errors() {
        // 16 tokens, but a 20-token committed window end — out of bounds.
        let leader = leader_with_slot(plain_request("req-1", (0..16).collect()));
        let enqueue = Arc::new(RecordingEnqueue::default());
        let plane = plane_over(&leader, uuid::Uuid::new_v4().into(), enqueue.clone());

        let err = plane
            .dispatch(dispatch_for("req-1", 4, 20))
            .await
            .expect_err("out-of-bounds window must error");
        assert!(err.to_string().contains("out of bounds"), "got: {err}");
        assert!(enqueue.pushed.lock().is_empty(), "nothing enqueued");
    }

    #[tokio::test]
    async fn dispatch_after_leader_drop_errors() {
        let leader = leader_with_slot(plain_request("req-1", (0..16).collect()));
        let enqueue = Arc::new(RecordingEnqueue::default());
        let plane = plane_over(&leader, uuid::Uuid::new_v4().into(), enqueue.clone());
        drop(leader);

        let err = plane
            .dispatch(dispatch_for("req-1", 4, 16))
            .await
            .expect_err("dead provider must error");
        assert!(err.to_string().contains("leader dropped"), "got: {err}");
        assert!(enqueue.pushed.lock().is_empty(), "nothing enqueued");
    }

    #[tokio::test]
    async fn dispatch_with_lora_or_salt_errors() {
        // The CD wire guard: LoRA / salt are not validated end-to-end, so a
        // dispatch for such a slot must fail loudly.
        let lora_request = Request::new(
            "req-lora",
            (0..16).collect::<Vec<u32>>(),
            Some("adapter-a".to_string()),
            None,
            None,
        );
        let salt_request = Request::new(
            "req-salt",
            (0..16).collect::<Vec<u32>>(),
            None,
            Some("pepper".to_string()),
            None,
        );
        for (request, marker) in [(lora_request, "LoRA"), (salt_request, "cache_salt")] {
            let rid = request.request_id.clone();
            let leader = leader_with_slot(request);
            let enqueue = Arc::new(RecordingEnqueue::default());
            let plane = plane_over(&leader, uuid::Uuid::new_v4().into(), enqueue.clone());

            let err = plane
                .dispatch(dispatch_for(&rid, 4, 16))
                .await
                .expect_err("guarded input must error");
            assert!(err.to_string().contains(marker), "got: {err}");
            assert!(enqueue.pushed.lock().is_empty(), "nothing enqueued");
        }
    }
}
