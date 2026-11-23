// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dashmap::mapref::one::RefMut;
use kvbm_protocols::connector::{
    AcceptId, ActionId, FenceHandle, FindBlocksHandle, FindBlocksOutcome, LeaderEngineError,
    RequestId, SearchId,
};

use super::local::LocalConnectorEngine;

pub(super) type AdmissionGuard<'a> = RefMut<'a, RequestId, RequestActions>;

#[derive(Default)]
pub(super) struct RequestActions {
    pub(super) ids: Vec<ActionId>,
    pub(super) finishing: bool,
    pub(super) eviction_epoch: u64,
    pub(super) evictions: usize,
    pub(super) eviction_fences: Vec<FenceHandle>,
    pub(super) physical_loads: usize,
    find_polls: usize,
    offers: usize,
}

impl RequestActions {
    pub(super) fn idle(&self) -> bool {
        self.ids.is_empty() && self.evictions == 0 && self.find_polls == 0 && self.offers == 0
    }

    fn prior_generation_draining(&mut self) -> bool {
        self.eviction_fences.retain(|fence| !fence.is_complete());
        self.evictions != 0 || !self.eviction_fences.is_empty()
    }
}

#[cfg(test)]
impl From<Vec<ActionId>> for RequestActions {
    fn from(ids: Vec<ActionId>) -> Self {
        Self {
            ids,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub(super) enum AdmissionOffer {
    Search(SearchId),
    Prefill(AcceptId),
}

impl AdmissionOffer {
    fn for_handle(handle: &FindBlocksHandle) -> Self {
        if let Some(id) = handle.search_id() {
            Self::Search(id)
        } else {
            Self::Prefill(handle.prefill_accept_id().expect("prefill handle"))
        }
    }

    fn stale(self) -> LeaderEngineError {
        match self {
            Self::Search(_) => LeaderEngineError::SearchNotMatched,
            Self::Prefill(_) => LeaderEngineError::PrefillSessionStale,
        }
    }
}

impl LocalConnectorEngine {
    pub(super) fn begin_admitted_find(&self, request: &RequestId) -> Option<u64> {
        let mut actions = self.by_request.entry(request.clone()).or_default();
        if actions.finishing || actions.prior_generation_draining() {
            return None;
        }
        actions.find_polls += 1;
        Some(actions.eviction_epoch)
    }

    pub(super) fn end_admitted_find(
        &self,
        request: &RequestId,
        epoch: u64,
        outcome: &Result<FindBlocksOutcome, LeaderEngineError>,
    ) -> bool {
        let mut actions = self.by_request.get_mut(request).expect("find registered");
        let current = !actions.finishing && actions.eviction_epoch == epoch;
        if current
            && let Ok(
                FindBlocksOutcome::Searching {
                    minted: Some(handle),
                }
                | FindBlocksOutcome::Resolved {
                    minted: Some(handle),
                    ..
                },
            ) = outcome
            && self
                .admission_offers
                .insert(AdmissionOffer::for_handle(handle), (request.clone(), epoch))
                .is_none()
        {
            actions.offers += 1;
        }
        actions.find_polls -= 1;
        drop(actions);
        self.prune_request_actions(request);
        current
    }

    pub(super) fn release_admission_offer(&self, offer: AdmissionOffer) {
        let Some(request) = self
            .admission_offers
            .get(&offer)
            .map(|entry| entry.0.clone())
        else {
            return;
        };
        if let Some(mut actions) = self.by_request.get_mut(&request)
            && self.admission_offers.remove(&offer).is_some()
        {
            actions.offers -= 1;
        }
        self.prune_request_actions(&request);
    }

    pub(super) fn prune_request_actions(&self, request: &RequestId) {
        self.by_request
            .remove_if(request, |_, actions| actions.idle());
    }

    /// Keep the request guard through action registration. Callers perform
    /// dispatch and publish proofs only after dropping it.
    pub(super) fn admit_action(
        &self,
        request: &RequestId,
        offer: Option<AdmissionOffer>,
        saving: bool,
    ) -> Result<AdmissionGuard<'_>, LeaderEngineError> {
        let mut actions = self.by_request.entry(request.clone()).or_default();
        if actions.finishing || actions.prior_generation_draining() {
            return Err(LeaderEngineError::PriorGenerationDraining);
        }
        if let Some(offer) = offer
            && self
                .admission_offers
                .get(&offer)
                .is_some_and(|bound| &bound.0 != request || bound.1 != actions.eviction_epoch)
        {
            drop(actions);
            self.prune_request_actions(request);
            return Err(offer.stale());
        }
        if saving && actions.physical_loads != 0 {
            return Err(LeaderEngineError::OnboardAlreadyInFlight);
        }
        Ok(actions)
    }
}
