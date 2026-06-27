// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side prefill-dispatch seam.
//!
//! Once the decode worker has opened + committed a remote-prefill session (see
//! [`super::commit::open_and_commit`]), the wiring layer hands a
//! [`PrefillDispatch`] to a [`PrefillPlane`] implementation, which completes the
//! on-wire `kvbm_protocols::disagg::RemotePrefillRequest` and enqueues it for a
//! prefill worker to consume.

use futures::future::BoxFuture;

use kvbm_protocols::disagg::{SessionEndpoint, SessionId};

/// Engine-side half of a `kvbm_protocols::disagg::RemotePrefillRequest`.
///
/// Carries only the fields the engine owns at dispatch time. The transport
/// [`PrefillPlane`] impl completes the wire form — notably the wire `token_ids`
/// from absolute position 0 truncated at `num_window_tokens`, which never cross
/// the engine seam (the engine deals in hashes, block ids, and token COUNTS,
/// not raw token streams) — before enqueueing.
///
/// No hash digest rides here: the cross-instance hash-chain guard
/// (`kvbm_protocols::disagg::digest_provided_hashes` over the `[0,
/// num_provided_tokens / block_size)` PLH range) derives from the request's
/// absolute hash chain, which never crosses the engine seam. The transport
/// computes it when assembling the `RemotePrefillRequest`; the prefill side
/// recomputes the same slice from its own slot's absolute-coordinate PLH chain.
pub struct PrefillDispatch {
    /// Correlates the dispatch with the decode-side request and session.
    pub request_id: String,
    /// The session decode opened in [`super::commit::open_and_commit`]; the
    /// prefill side attaches to it to pull the committed/available blocks.
    pub session_id: SessionId,
    /// Endpoint the prefill peer attaches back on. `None` until the session
    /// surfaces one.
    pub decode_endpoint: Option<SessionEndpoint>,
    /// vLLM prefix + block-floored local match, in tokens — the length of the
    /// provided-prefix slice the transport's digest covers.
    pub num_provided_tokens: usize,
    /// The committed remote-prefill window end, in tokens: the vLLM prefix plus
    /// the block-floored external window the budget reserved. The transport
    /// truncates the wire `token_ids` here — the partial tail block, if any,
    /// stays on decode, and the prefill side never computes or offloads blocks
    /// beyond the window decode's pull plan covers.
    pub num_window_tokens: usize,
}

/// Decode-side direction of the hub coupling: enqueue a completed remote-prefill
/// request for a prefill worker.
///
/// Dispatch-only by design — the consume direction never joins this trait.
/// A dispatched request reaches the prefill worker through vLLM itself: the
/// `kv_transfer_params` on the generated request carry the
/// `RemotePrefillParams`, the connector's `find_blocks` poll routes them into
/// the engine's prefill accept core, and the engine's prefill pipeline
/// takes over from there. A single dispatch verb returning a `'static` boxed
/// future so a slow queue never blocks the synchronous caller.
pub trait PrefillPlane: Send + Sync {
    fn dispatch(&self, req: PrefillDispatch) -> BoxFuture<'static, anyhow::Result<()>>;
}

#[cfg(test)]
mod tests {
    use futures::FutureExt;
    use parking_lot::Mutex;

    use super::*;

    /// Test double recording every dispatched [`PrefillDispatch`].
    #[derive(Default)]
    struct RecordingPrefillPlane {
        dispatches: Mutex<Vec<PrefillDispatch>>,
    }

    impl PrefillPlane for RecordingPrefillPlane {
        fn dispatch(&self, req: PrefillDispatch) -> BoxFuture<'static, anyhow::Result<()>> {
            self.dispatches.lock().push(req);
            async { Ok(()) }.boxed()
        }
    }

    #[tokio::test]
    async fn dispatch_carries_request_session_and_provided_tokens() {
        // The engine-owned half of the request: request/session correlation, the
        // attach-back endpoint, and the provided-prefix token count. No hash
        // vector or digest crosses the seam — the transport computes the digest.
        let session_id = uuid::Uuid::new_v4();

        let plane = RecordingPrefillPlane::default();
        plane
            .dispatch(PrefillDispatch {
                request_id: "req-1".to_string(),
                session_id,
                decode_endpoint: None,
                num_provided_tokens: 3 * 16,
                num_window_tokens: 5 * 16,
            })
            .await
            .expect("dispatch ok");

        let recorded = plane.dispatches.lock();
        assert_eq!(recorded.len(), 1, "plane saw exactly one dispatch");
        assert_eq!(recorded[0].request_id, "req-1");
        assert_eq!(recorded[0].session_id, session_id);
        assert_eq!(recorded[0].num_provided_tokens, 3 * 16);
        assert_eq!(recorded[0].num_window_tokens, 5 * 16);
    }
}
