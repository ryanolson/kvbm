// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request slot — a pure data bag.
//!
//! Lifecycle/transaction state lives engine-side behind the
//! [`kvbm_protocols::connector::LeaderEngine`] seam; the slot here only carries
//! what the leader needs to align hashes ↔ block ids and to hold the engine's
//! RAII handles ([`FindBlocksHandle`] keeps the latched match lifecycle alive;
//! [`OnboardHandle`]/[`OffloadHandle`] keep their actions tracked). No
//! `TransactionState`/`SlotState`, and no match bookkeeping — the engine owns
//! window derivation, fresh-vs-refresh, and hit state.

use std::sync::Arc;

use dynamo_tokens::{TokenBlockError, TokenBlockSequence};
use kvbm_common::{BlockId, SequenceHash};
use kvbm_logical::KvbmSequenceHashProvider as _;
use kvbm_protocols::connector::{FenceHandle, FindBlocksHandle, OffloadHandle, OnboardHandle};
use kvbm_protocols::disagg::{RemotePrefillParams, TransferParams};

use crate::common::Request;

/// vLLM-side data for one request.
#[derive(Debug)]
pub struct RequestSlot {
    pub request_id: String,
    /// Token sequence used to compute per-block sequence hashes. Populated by
    /// `from_request` — every slot is created from a vLLM Request before the
    /// first GNMT poll (the binding layer's `_create_slot` ordering).
    pub sequence: TokenBlockSequence,
    /// LoRA adapter name from the originating request. Read by the CD wire's
    /// loud-fail guard: LoRA is not yet validated end-to-end for conditional
    /// disagg, so a remote-prefill dispatch for a LoRA request must error
    /// rather than risk silent hash divergence on the prefill side.
    pub lora_name: Option<String>,
    /// Raw salt string from the originating request (the input to `salt_hash`).
    /// Same CD wire guard as `lora_name`.
    pub salt: Option<String>,
    /// THE one parked match lifecycle; `Some` while the engine holds a latched
    /// lifecycle for this request (a pinned local match OR an accepted remote
    /// prefill — the slot cannot tell which). Pure RAII: its kind-routed drop
    /// fires the engine's generation-guarded release. Parked when
    /// `find_blocks` mints, dropped when the outcome instructs
    /// `release_parked` (Issue-A gated) or a reap/recv terminal fires.
    pub proposal: Option<FindBlocksHandle>,
    /// In-flight onboard (external load) handle, issued by the USAA allocate
    /// path when vLLM commits external tokens; released on `finished_recving`
    /// once terminal.
    pub onboard: Option<OnboardHandle>,
    /// In-flight offload (save) handles; populated by the save path.
    pub offloads: Vec<OffloadHandle>,
    /// On each eviction the parked `proposal` (+ in-flight `onboard`, + the
    /// eviction's leader-side [`FenceHandle`]) are MOVED here so the lifecycle
    /// is *drained*, not dropped. One entry per eviction generation: a restored
    /// request can be evicted again while an older holder still drains, and
    /// overwriting it would release the old pin mid-read. Each entry drops
    /// (kind-routed RAII release) once its own drain completes: the engine
    /// fence when the eviction minted one (it covers the held onboard AND the
    /// view-detached offloads), else the held onboard's terminal.
    pub drain_holders: Vec<(FindBlocksHandle, Option<OnboardHandle>, Option<FenceHandle>)>,
    /// Leader-side fence handles from evictions that parked NO proposal (only
    /// an onboard/offload was in flight) — there is no drain-holder entry to
    /// carry them, but the leader keeps its observational view of the drain.
    /// Swept (completed entries dropped) by the same `update_connector_output`
    /// pass that polls the drain-holders.
    pub fence_holders: Vec<FenceHandle>,
    /// Disagg transfer params parsed from the request's `kv_transfer_params`
    /// JSON at slot creation. A `remote_prefill` marker carries the
    /// decode-minted params that ride `FindBlocksRequest.remote_prefill` —
    /// request construction only, never a routing predicate (the engine
    /// decides local-vs-remote).
    pub transfer_params: Option<TransferParams>,
    /// G1 block ids vLLM allocated for this request.
    pub block_ids: Vec<BlockId>,
    /// Token-granular save cursor. Counts tokens whose KV has been scheduled
    /// for computation and whose completed blocks have been handed to the
    /// engine's offload pipeline. The offload cursor can never pass
    /// `assigned_blocks() * block_size`.
    pub evaluated_tokens: usize,
    /// Once-flag: set on the first resolved GNMT hit and cleared when a NEW
    /// lifecycle is minted. A refresh that enlarges the hit count is still
    /// the same lifecycle — the flag is NOT re-armed for it. Mirrors legacy
    /// `mark_matched_tokens_reported` / `reset_matched_tokens_reported`:
    /// one increment per minted lifecycle, never for re-polls of the same one.
    matched_tokens_reported: bool,
    /// Cached full hash chain, shared with `FindBlocksRequest` (a GNMT poll
    /// clones the `Arc`, never the hashes). `None` = stale: invalidated only
    /// when the token sequence changes ([`Self::extend_tokens`] — both decode
    /// growth and the resumed-request resync go through it) and rebuilt
    /// lazily on the next [`Self::sequence_hashes`] read.
    chain: Option<Arc<[SequenceHash]>>,
}

impl RequestSlot {
    /// Slot minted from a vLLM `create_slot` Request (carries tokens).
    ///
    /// The request's `salt_hash` seeds every per-block hash in the sequence;
    /// `block_size` determines block boundaries. The request's
    /// `kv_transfer_params` JSON is parsed as disagg [`TransferParams`] here:
    /// malformed JSON logs a warning and yields `None` — a bad payload
    /// degrades the request to the plain local path, it never fails slot
    /// creation.
    pub fn from_request(request: Request, block_size: u32) -> Self {
        let transfer_params = match request.disagg_transfer_params() {
            Ok(params) => params,
            Err(error) => {
                tracing::warn!(
                    request_id = %request.request_id,
                    %error,
                    "malformed kv_transfer_params; treating request as non-disagg"
                );
                None
            }
        };
        let sequence = TokenBlockSequence::new(
            Vec::<u32>::from(request.tokens).into(),
            block_size,
            Some(request.salt_hash),
        );
        Self {
            request_id: request.request_id,
            sequence,
            lora_name: request.lora_name,
            salt: request.salt,
            proposal: None,
            onboard: None,
            offloads: Vec::new(),
            drain_holders: Vec::new(),
            fence_holders: Vec::new(),
            transfer_params,
            block_ids: Vec::new(),
            evaluated_tokens: 0,
            matched_tokens_reported: false,
            chain: None,
        }
    }

    pub fn set_block_ids(&mut self, block_ids: Vec<BlockId>) {
        self.block_ids = block_ids;
    }

    /// Total number of tokens tracked by the sequence (complete blocks + any
    /// partial tail).
    pub fn total_tokens(&self) -> usize {
        self.sequence.total_tokens()
    }

    /// Extend the sequence with additional tokens, completing new blocks as
    /// they fill. Returns an error if the sequence's internal state rejects the
    /// extension. The ONE chain-cache invalidation point: any accepted
    /// extension may complete new blocks, so the cached `Arc` goes stale here
    /// and rebuilds on the next [`Self::sequence_hashes`] read.
    pub fn extend_tokens(&mut self, tokens: Vec<u32>) -> Result<(), TokenBlockError> {
        self.sequence.extend(tokens.into())?;
        self.chain = None;
        Ok(())
    }

    /// Number of blocks that BOTH have a complete sequence hash (from the
    /// token sequence) and a matching allocated G1 id. The offload cursor can
    /// never advance past this boundary.
    pub fn assigned_blocks(&self) -> usize {
        self.sequence.blocks().len().min(self.block_ids.len())
    }

    /// Per-block sequence hash at `block_index` in the token sequence.
    pub fn sequence_hash(&self, block_index: usize) -> SequenceHash {
        self.sequence.blocks()[block_index].kvbm_sequence_hash()
    }

    /// FULL per-block hash chain in absolute-position order — every COMPLETE
    /// block of the token sequence, unsliced — as the slot's cached `Arc`.
    /// The GNMT hot path: a pure re-poll is a refcount bump, never a hash
    /// copy; the cache rebuilds only after [`Self::extend_tokens`] changed the
    /// sequence. Feeds `FindBlocksRequest.sequence_hashes` (the engine derives
    /// both the local search window and the prefill provided window by index
    /// range over it).
    pub fn sequence_hashes(&mut self) -> Arc<[SequenceHash]> {
        let chain = self.chain.get_or_insert_with(|| {
            self.sequence
                .blocks()
                .iter()
                .map(|b| b.kvbm_sequence_hash())
                .collect()
        });
        Arc::clone(chain)
    }

    /// Owned copy of the full chain for the COLD paths that hold the slot
    /// immutably (the CD plane's dispatch snapshot/digest — once per
    /// dispatch, never per poll).
    pub fn all_sequence_hashes(&self) -> Vec<SequenceHash> {
        self.sequence
            .blocks()
            .iter()
            .map(|b| b.kvbm_sequence_hash())
            .collect()
    }

    /// The decode-minted remote-prefill params, if this slot was created from
    /// a dispatched remote-prefill request. Request construction only — the
    /// connector clones them into `FindBlocksRequest.remote_prefill` and never
    /// branches on them.
    pub fn remote_prefill_params(&self) -> Option<&RemotePrefillParams> {
        self.transfer_params.as_ref()?.remote_prefill.as_ref()
    }

    /// Set the once-flag if it is unset; return `true` iff the flag
    /// transitioned from clear to set (i.e. this is the first call for this
    /// lifecycle). Subsequent calls for the same lifecycle return `false` so
    /// the counter is never double-incremented on re-polls.
    pub fn mark_matched_tokens_reported(&mut self) -> bool {
        if self.matched_tokens_reported {
            false
        } else {
            self.matched_tokens_reported = true;
            true
        }
    }

    /// Re-arm the once-flag so the NEXT resolved hit for a NEW lifecycle is
    /// reported. Called in GNMT whenever `find_blocks` mints a fresh lifecycle
    /// for this slot.
    pub fn reset_matched_tokens_reported(&mut self) {
        self.matched_tokens_reported = false;
    }

    /// True while ANY eviction drain-holder is still draining — that holder's
    /// held lifecycle is being READ by the draining work, so the slot must not
    /// drop (dropping the holder fires the RAII release, the A×E
    /// use-after-release). Holders whose drain completed are safe to drop.
    pub fn drain_holder_draining(&self) -> bool {
        self.drain_holders.iter().any(holder_draining)
    }
}

/// Per-holder release gate. A holder that carries the eviction's leader-side
/// fence handle drains until the ENGINE fence completes — the fence covers the
/// held onboard and the view-detached offloads, so it is strictly
/// later-or-equal to the onboard terminal. A fence-less holder (the eviction
/// armed nothing engine-side) falls back to the held onboard's terminal, the
/// pin's only remaining reader.
pub(crate) fn holder_draining(
    (_, onboard, fence): &(FindBlocksHandle, Option<OnboardHandle>, Option<FenceHandle>),
) -> bool {
    match fence {
        Some(fence) => !fence.is_complete(),
        None => onboard.as_ref().is_some_and(|h| !h.is_complete()),
    }
}
