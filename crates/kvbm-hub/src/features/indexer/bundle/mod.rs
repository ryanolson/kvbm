// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifest-scoped complete-bundle directory.

mod advertisements;
mod expiry;
mod generation;

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::RwLock;

use kvbm_protocols::cache_manifest::{
    BundleLineageValidationError, RegistrationEpoch, validate_bundle_lineages,
};
use velo_ext::InstanceId;

use super::protocol::{
    BundleAdvertisementRecord, BundleInvalidateRequest, BundlePublishRequest, BundleQueryHit,
    BundleQueryMissReason, BundleQueryOutcome, BundleQueryRequest,
};
use crate::protocol::MutationCredential;
use crate::registry::RegistryIncarnation;
use advertisements::{AdvertisementCapacity, AdvertisementIndex};
use expiry::ExpiredAdvertisementHistory;
use generation::{RetiredGenerationCapacity, RetiredGenerations};

/// Matches the connector's publication horizon while preventing a publisher
/// from converting `u64::MAX` into permanent hub state.
const MAX_ADVERTISEMENT_TTL_MS: u64 = 300_000;
/// Narrow live and absent budgets bound their individual abuse surfaces. The
/// combined limits derived in `BundleDirectoryState::new` also bound repeated
/// publish-to-retirement cycling.
const MAX_ADVERTISEMENTS_PER_OWNER: usize = 4_096;
const MAX_ADVERTISEMENTS_GLOBAL: usize = 65_536;
const MAX_ABSENT_RETIREMENTS_PER_OWNER: usize = 4_096;
const MAX_ABSENT_RETIREMENTS_GLOBAL: usize = 65_536;

/// Live-owner directory for complete manifest-scoped bundles.
pub struct BundleDirectory {
    state: RwLock<BundleDirectoryState>,
    lease_ttl_ms: u64,
    advertisement_ttl_ms: u64,
    clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    #[cfg(test)]
    publish_after_owner_check: Option<Arc<dyn Fn() + Send + Sync>>,
}

struct BundleDirectoryState {
    advertisements: AdvertisementIndex,
    expired_advertisements: ExpiredAdvertisementHistory,
    owner_credentials: HashMap<InstanceId, OwnerAuthority>,
    pending_owner_registrations: HashMap<InstanceId, PendingOwnerRegistration>,
    retired_generations: RetiredGenerations,
    identity_capacity_per_owner: usize,
    identity_global_capacity: usize,
}

/// A registration transaction temporarily removes mutation and query
/// authority without moving the owner's bounded directory metadata. Keeping
/// the data in the canonical indexes preserves capacity accounting, expiry
/// schedules, and generation retirements exactly until commit or rollback.
struct PendingOwnerRegistration {
    previous_authority: Option<OwnerAuthority>,
    next_authority: NextAuthority,
    transaction_epoch: RegistrationEpoch,
    incarnation: Option<RegistryIncarnation>,
}

#[derive(Clone, PartialEq, Eq)]
struct OwnerAuthority {
    credential: MutationCredential,
    registration_epoch: RegistrationEpoch,
}

#[derive(Clone, PartialEq, Eq)]
enum NextAuthority {
    Active(OwnerAuthority),
    Removed,
}

enum DirectoryIdentityCapacity {
    Owner { owner: InstanceId, capacity: usize },
    Global { capacity: usize },
}

impl BundleDirectory {
    pub fn new(lease_ttl_ms: u64) -> Self {
        Self::with_clock_inner(lease_ttl_ms, Arc::new(unix_time_ms))
    }

    #[cfg(feature = "test-support")]
    pub(super) fn advertisement_count(&self) -> Result<usize, BundleDirectoryError> {
        self.state
            .read()
            .map(|state| state.advertisements.len())
            .map_err(|_| BundleDirectoryError::Unavailable)
    }

    fn with_clock_inner(lease_ttl_ms: u64, clock: Arc<dyn Fn() -> u64 + Send + Sync>) -> Self {
        Self::with_limits_inner(
            lease_ttl_ms,
            MAX_ADVERTISEMENT_TTL_MS,
            MAX_ADVERTISEMENTS_PER_OWNER,
            MAX_ADVERTISEMENTS_GLOBAL,
            MAX_ABSENT_RETIREMENTS_PER_OWNER,
            MAX_ABSENT_RETIREMENTS_GLOBAL,
            clock,
        )
    }

    fn with_limits_inner(
        lease_ttl_ms: u64,
        advertisement_ttl_ms: u64,
        advertisements_per_owner: usize,
        advertisements_global: usize,
        absent_retirements_per_owner: usize,
        absent_retirements_global: usize,
        clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Self {
        Self {
            state: RwLock::new(BundleDirectoryState::new(
                lease_ttl_ms,
                advertisements_per_owner,
                advertisements_global,
                absent_retirements_per_owner,
                absent_retirements_global,
            )),
            lease_ttl_ms,
            advertisement_ttl_ms,
            clock,
            #[cfg(test)]
            publish_after_owner_check: None,
        }
    }

    #[cfg(test)]
    fn with_clock(lease_ttl_ms: u64, clock: Arc<dyn Fn() -> u64 + Send + Sync>) -> Self {
        Self::with_clock_inner(lease_ttl_ms, clock)
    }

    #[cfg(test)]
    fn with_limits(
        lease_ttl_ms: u64,
        advertisement_ttl_ms: u64,
        advertisements_per_owner: usize,
        advertisements_global: usize,
        absent_retirements_per_owner: usize,
        absent_retirements_global: usize,
        clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Self {
        Self::with_limits_inner(
            lease_ttl_ms,
            advertisement_ttl_ms,
            advertisements_per_owner,
            advertisements_global,
            absent_retirements_per_owner,
            absent_retirements_global,
            clock,
        )
    }

    #[cfg(test)]
    fn with_publish_after_owner_check(
        lease_ttl_ms: u64,
        hook: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        Self {
            state: RwLock::new(BundleDirectoryState::new(
                lease_ttl_ms,
                MAX_ADVERTISEMENTS_PER_OWNER,
                MAX_ADVERTISEMENTS_GLOBAL,
                MAX_ABSENT_RETIREMENTS_PER_OWNER,
                MAX_ABSENT_RETIREMENTS_GLOBAL,
            )),
            lease_ttl_ms,
            advertisement_ttl_ms: MAX_ADVERTISEMENT_TTL_MS,
            clock: Arc::new(|| 0),
            publish_after_owner_check: Some(hook),
        }
    }

    #[cfg(test)]
    pub(crate) fn register_owner(
        &self,
        owner: InstanceId,
        credential: MutationCredential,
    ) -> Result<(), BundleDirectoryError> {
        let mut state = self
            .state
            .write()
            .map_err(|_| BundleDirectoryError::Unavailable)?;
        state.prune((self.clock)());
        state.pending_owner_registrations.remove(&owner);
        state.clear_owner_metadata(owner);
        state.owner_credentials.insert(
            owner,
            OwnerAuthority {
                credential,
                registration_epoch: test_registration_epoch(owner),
            },
        );
        Ok(())
    }

    /// Stage an owner replacement or feature omission while leaving its bounded
    /// metadata in place.
    ///
    /// The owner is fail-closed until [`Self::finalize_owner_registration`]. A
    /// call carrying the preserved prior authority is the registration rollback
    /// path and restores authority without rebuilding any indexes.
    pub(crate) fn stage_owner_transition(
        &self,
        owner: InstanceId,
        credential: Option<MutationCredential>,
        transaction_epoch: RegistrationEpoch,
    ) -> Result<(), BundleDirectoryError> {
        let mut state = self
            .state
            .write()
            .map_err(|_| BundleDirectoryError::Unavailable)?;
        let next_authority = match credential {
            Some(credential) => NextAuthority::Active(OwnerAuthority {
                credential,
                registration_epoch: transaction_epoch,
            }),
            None => NextAuthority::Removed,
        };

        if let Some(pending) = state.pending_owner_registrations.get(&owner) {
            if pending.next_authority == next_authority {
                if pending.transaction_epoch == transaction_epoch {
                    return Ok(());
                }
                // The hub credential transaction may already have committed
                // this staged authority while its post-commit callback is still
                // pending. A concurrent replacement that then rolls back carries
                // that credential with a restored registry incarnation. Commit
                // the already-authoritative lifecycle; never let the failed
                // replacement roll it back to the older generation domain.
                let pending = state
                    .pending_owner_registrations
                    .remove(&owner)
                    .expect("checked pending owner registration must remain present");
                state.clear_owner_metadata(owner);
                state.install_next_authority(owner, pending.next_authority);
                return Ok(());
            }
            if next_authority
                .active()
                .is_some_and(|next| pending.previous_authority.as_ref() == Some(next))
                || matches!(&next_authority, NextAuthority::Removed)
                    && pending.previous_authority.is_none()
            {
                state.pending_owner_registrations.remove(&owner);
                state.install_next_authority(owner, next_authority);
                return Ok(());
            }
            return Err(BundleDirectoryError::OwnerRegistrationInProgress { owner });
        }

        match &next_authority {
            NextAuthority::Active(next) if state.owner_credentials.get(&owner) == Some(next) => {
                return Ok(());
            }
            NextAuthority::Removed if !state.owner_credentials.contains_key(&owner) => {
                return Ok(());
            }
            _ => {}
        }
        let previous_authority = state.owner_credentials.remove(&owner);
        state.pending_owner_registrations.insert(
            owner,
            PendingOwnerRegistration {
                previous_authority,
                next_authority,
                transaction_epoch,
                incarnation: None,
            },
        );
        Ok(())
    }

    /// Bind the exact registry write to a previously staged transition.
    pub(crate) fn bind_owner_registration(
        &self,
        owner: InstanceId,
        transaction_epoch: RegistrationEpoch,
        incarnation: RegistryIncarnation,
    ) -> Result<(), BundleDirectoryError> {
        let mut state = self
            .state
            .write()
            .map_err(|_| BundleDirectoryError::Unavailable)?;
        let Some(pending) = state.pending_owner_registrations.get_mut(&owner) else {
            return Ok(());
        };
        if pending.transaction_epoch != transaction_epoch {
            return Err(BundleDirectoryError::StaleOwnerTransaction { owner });
        }
        match pending.incarnation {
            Some(current) if current != incarnation => {
                return Err(BundleDirectoryError::StaleOwnerRegistration {
                    owner,
                    expected: current,
                    attempted: incarnation,
                });
            }
            Some(_) => {}
            None => pending.incarnation = Some(incarnation),
        }
        Ok(())
    }

    /// Commit the exact staged registration. A delayed callback for another
    /// incarnation fails closed and cannot rotate the pending owner.
    pub(crate) fn finalize_owner_registration(
        &self,
        owner: InstanceId,
        incarnation: RegistryIncarnation,
    ) -> Result<(), BundleDirectoryError> {
        let mut state = self
            .state
            .write()
            .map_err(|_| BundleDirectoryError::Unavailable)?;
        let Some(pending) = state.pending_owner_registrations.get(&owner) else {
            return Ok(());
        };
        let Some(expected) = pending.incarnation else {
            return Err(BundleDirectoryError::UnboundOwnerRegistration { owner });
        };
        if expected != incarnation {
            return Err(BundleDirectoryError::StaleOwnerRegistration {
                owner,
                expected,
                attempted: incarnation,
            });
        }
        let pending = state
            .pending_owner_registrations
            .remove(&owner)
            .expect("checked pending owner registration must remain present");
        state.clear_owner_metadata(owner);
        state.prune((self.clock)());
        state.install_next_authority(owner, pending.next_authority);
        Ok(())
    }

    /// Authenticate `credential` against `owner` and return the registration
    /// epoch it authorizes.
    ///
    /// The tier-placement snapshot endpoint authorizes through this rather than
    /// keeping its own owner→credential map: forking that map would fork the
    /// staging and fail-closed semantics with it, and an owner mid-registration
    /// must be closed to a snapshot install for exactly the reason it is closed
    /// to a bundle publish.
    pub(super) fn authorize_owner_epoch(
        &self,
        owner: InstanceId,
        credential: &MutationCredential,
    ) -> Result<RegistrationEpoch, BundleDirectoryError> {
        let state = self
            .state
            .read()
            .map_err(|_| BundleDirectoryError::Unavailable)?;
        authorize_owner(&state, owner, credential)
    }

    pub fn remove_owner(&self, owner: InstanceId) {
        if let Ok(mut state) = self.state.write() {
            let now_unix_ms = (self.clock)();
            state.owner_credentials.remove(&owner);
            state.pending_owner_registrations.remove(&owner);
            for advertisement in state.advertisements.records_for_owner(owner) {
                let retired = state.retired_generations.retire_live(
                    advertisement.key,
                    owner,
                    advertisement.generation,
                    advertisement.expires_at_unix_ms,
                    advertisement.expires_at_unix_ms,
                    now_unix_ms,
                );
                if retired.is_err() {
                    tracing::error!(
                        owner = %owner,
                        key = ?advertisement.key,
                        "bundle retirement capacity invariant blocked owner removal"
                    );
                    continue;
                }
                let _ = state.advertisements.remove(advertisement.key, owner);
                state.expired_advertisements.forget(advertisement.key);
            }
            state.prune(now_unix_ms);
        }
    }

    pub fn publish(&self, request: BundlePublishRequest) -> Result<(), BundleDirectoryError> {
        let BundlePublishRequest {
            credential,
            mut advertisement,
        } = request;
        validate_bundle_lineages(
            advertisement.key,
            &advertisement.requirements,
            &advertisement.lineages,
        )?;
        // `requirements` defines the bundle's resource set, so a placement for a
        // resource outside it is a claim about something this advertisement does
        // not describe — and it feeds `ready_tier()`, which a CT-2a consumer
        // reads as a stage-cost hint. Rejecting keeps the cost signal derived
        // only from resources the record actually owns. Additive-safe: a
        // publisher that predates R7b sends no placements at all.
        if let Some(placement) = advertisement.placements.iter().find(|placement| {
            !advertisement
                .requirements
                .iter()
                .any(|requirement| requirement.resource() == placement.resource)
        }) {
            return Err(BundleDirectoryError::UnrequiredPlacement {
                owner: advertisement.owner,
                resource: placement.resource,
            });
        }
        let key = advertisement.key;
        let owner = advertisement.owner;
        let generation = advertisement.generation;
        let mut state = self
            .state
            .write()
            .map_err(|_| BundleDirectoryError::Unavailable)?;
        let registration_epoch = authorize_owner(&state, owner, &credential)?;
        if advertisement.registration_epoch != Some(registration_epoch) {
            return Err(BundleDirectoryError::RegistrationEpochMismatch { owner });
        }
        let now_unix_ms = (self.clock)();
        state.prune(now_unix_ms);
        advertisement.expires_at_unix_ms = advertisement
            .expires_at_unix_ms
            .min(now_unix_ms.saturating_add(self.advertisement_ttl_ms));
        // Hub-stamped, overwriting whatever the publisher sent: this is the
        // freshness signal query rows report, so a publisher-supplied value
        // would be both spoofable and clock-skewed. `expires_at_unix_ms` cannot
        // stand in — it is clamped above and says nothing about arrival.
        advertisement.advertised_at_unix_ms = Some(now_unix_ms);
        #[cfg(test)]
        if let Some(hook) = &self.publish_after_owner_check {
            hook();
        }
        let owner_key = (key, owner);
        if let Some(current) =
            state
                .retired_generations
                .rejected_generation(key, owner, generation, now_unix_ms)
        {
            return Err(BundleDirectoryError::StaleGeneration {
                current,
                attempted: generation,
            });
        }
        if let Some(current) = state.advertisements.get(owner_key.0, owner_key.1)
            && current.generation > generation
        {
            return Err(BundleDirectoryError::StaleGeneration {
                current: current.generation,
                attempted: generation,
            });
        }
        if state.advertisements.get(owner_key.0, owner_key.1).is_none() {
            state
                .ensure_new_identity(owner)
                .map_err(map_identity_capacity)?;
        }
        state
            .advertisements
            .insert(advertisement)
            .map_err(|capacity| match capacity {
                AdvertisementCapacity::Owner { owner, capacity } => {
                    BundleDirectoryError::OwnerCapacity { owner, capacity }
                }
                AdvertisementCapacity::Global { capacity } => {
                    BundleDirectoryError::GlobalCapacity { capacity }
                }
            })?;
        state.expired_advertisements.forget(key);
        Ok(())
    }

    pub fn query(&self, request: BundleQueryRequest) -> BundleQueryOutcome {
        let required = request
            .requirements
            .iter()
            .map(|requirement| requirement.resource())
            .collect::<BTreeSet<_>>();
        if required.len() != request.requirements.len() {
            return BundleQueryOutcome::Miss(BundleQueryMissReason::Incomplete);
        }
        {
            let Ok(mut state) = self.state.write() else {
                return BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound);
            };
            state.prune((self.clock)());
        }
        let Ok(state) = self.state.read() else {
            return BundleQueryOutcome::Miss(BundleQueryMissReason::NotFound);
        };
        let mut reason = BundleQueryMissReason::NotFound;
        for key in &request.candidates {
            if key.manifest() != request.manifest {
                if reason == BundleQueryMissReason::NotFound {
                    reason = BundleQueryMissReason::Incompatible;
                }
                continue;
            }
            if reason == BundleQueryMissReason::NotFound
                && state.expired_advertisements.contains(*key)
            {
                reason = BundleQueryMissReason::Expired;
            }
            let mut selected = None;
            for entry in state.advertisements.for_key(key) {
                let Some(authority) = state.owner_credentials.get(&entry.owner) else {
                    continue;
                };
                if entry.registration_epoch != Some(authority.registration_epoch) {
                    continue;
                }
                if state
                    .retired_generations
                    .rejected_generation(
                        entry.key,
                        entry.owner,
                        entry.generation,
                        request.now_unix_ms,
                    )
                    .is_some()
                {
                    continue;
                }
                if entry.expires_at_unix_ms <= request.now_unix_ms {
                    if reason == BundleQueryMissReason::NotFound {
                        reason = BundleQueryMissReason::Expired;
                    }
                    continue;
                }
                if entry.requirements != request.requirements {
                    if reason == BundleQueryMissReason::NotFound {
                        reason = BundleQueryMissReason::Incompatible;
                    }
                    continue;
                }
                if validate_bundle_lineages(*key, &request.requirements, &entry.lineages).is_err() {
                    if reason == BundleQueryMissReason::NotFound {
                        reason = BundleQueryMissReason::Incomplete;
                    }
                    continue;
                }
                let replace =
                    selected
                        .as_ref()
                        .is_none_or(|current: &BundleAdvertisementRecord| {
                            entry.owner.to_string() < current.owner.to_string()
                        });
                if replace {
                    selected = Some(entry.clone());
                }
            }
            let Some(entry) = selected else {
                continue;
            };
            return BundleQueryOutcome::Hit(BundleQueryHit {
                // Both R7b §5 row fields are echoed from the winning record, so
                // a row can never disagree with the advertisement it came from.
                ready_tier: entry.ready_tier(),
                advertised_at_unix_ms: entry.advertised_at_unix_ms,
                advertisement: entry.clone(),
                lease_id: uuid::Uuid::new_v4(),
                lease_expires_at_unix_ms: entry
                    .expires_at_unix_ms
                    .min(request.now_unix_ms.saturating_add(self.lease_ttl_ms)),
            });
        }
        BundleQueryOutcome::Miss(reason)
    }

    pub fn invalidate(
        &self,
        request: BundleInvalidateRequest,
    ) -> Result<bool, BundleDirectoryError> {
        let mut state = self
            .state
            .write()
            .map_err(|_| BundleDirectoryError::Unavailable)?;
        authorize_owner(&state, request.owner, &request.credential)?;
        let now_unix_ms = (self.clock)();
        state.prune(now_unix_ms);
        let owner_key = (request.key, request.owner);
        let live = state
            .advertisements
            .get(owner_key.0, owner_key.1)
            .filter(|entry| entry.generation <= request.generation)
            .map(|entry| {
                (
                    entry.generation == request.generation,
                    entry.expires_at_unix_ms,
                )
            });
        let exact = match live {
            Some((exact, expiration)) => {
                state
                    .retired_generations
                    .retire_live(
                        request.key,
                        request.owner,
                        request.generation,
                        expiration,
                        request.retain_until_unix_ms,
                        now_unix_ms,
                    )
                    .map_err(map_retirement_capacity)?;
                let _ = state.advertisements.remove(owner_key.0, owner_key.1);
                state.expired_advertisements.forget(request.key);
                exact
            }
            None => {
                if !state
                    .retired_generations
                    .contains(request.key, request.owner)
                {
                    state
                        .ensure_new_identity(request.owner)
                        .map_err(map_identity_capacity)?;
                }
                state
                    .retired_generations
                    .retire_absent(
                        request.key,
                        request.owner,
                        request.generation,
                        request.retain_until_unix_ms,
                        now_unix_ms,
                    )
                    .map_err(map_retirement_capacity)?;
                false
            }
        };
        Ok(exact)
    }
}

impl BundleDirectoryState {
    fn new(
        expired_retention_ms: u64,
        advertisements_per_owner: usize,
        advertisements_global: usize,
        absent_retirements_per_owner: usize,
        absent_retirements_global: usize,
    ) -> Self {
        let identity_capacity_per_owner =
            advertisements_per_owner.saturating_add(absent_retirements_per_owner);
        let identity_global_capacity =
            advertisements_global.saturating_add(absent_retirements_global);
        Self {
            advertisements: AdvertisementIndex::new(
                advertisements_per_owner,
                advertisements_global,
            ),
            expired_advertisements: ExpiredAdvertisementHistory::new(expired_retention_ms),
            owner_credentials: HashMap::new(),
            pending_owner_registrations: HashMap::new(),
            retired_generations: RetiredGenerations::new(
                expired_retention_ms,
                absent_retirements_per_owner,
                absent_retirements_global,
                identity_capacity_per_owner,
                identity_global_capacity,
            ),
            identity_capacity_per_owner,
            identity_global_capacity,
        }
    }

    fn prune(&mut self, observed_unix_ms: u64) {
        self.retired_generations.prune(observed_unix_ms);
        let expired = self.advertisements.prune_expired(observed_unix_ms);
        self.expired_advertisements
            .record(expired, observed_unix_ms);
    }

    fn ensure_new_identity(&self, owner: InstanceId) -> Result<(), DirectoryIdentityCapacity> {
        if self
            .advertisements
            .len()
            .saturating_add(self.retired_generations.len())
            >= self.identity_global_capacity
        {
            return Err(DirectoryIdentityCapacity::Global {
                capacity: self.identity_global_capacity,
            });
        }
        if self
            .advertisements
            .owner_len(owner)
            .saturating_add(self.retired_generations.owner_len(owner))
            >= self.identity_capacity_per_owner
        {
            return Err(DirectoryIdentityCapacity::Owner {
                owner,
                capacity: self.identity_capacity_per_owner,
            });
        }
        Ok(())
    }

    /// A registration credential identifies one owner lifecycle. Replacing it
    /// starts a fresh generation domain, so no advertisement or replay guard
    /// from the prior lifecycle may survive the rotation.
    fn clear_owner_metadata(&mut self, owner: InstanceId) {
        let advertisements = self.advertisements.remove_owner(owner);
        let retired_keys = self.retired_generations.remove_owner(owner);
        for key in advertisements
            .into_iter()
            .map(|advertisement| advertisement.key)
            .chain(retired_keys)
        {
            self.expired_advertisements.forget(key);
        }
    }

    fn install_next_authority(&mut self, owner: InstanceId, next: NextAuthority) {
        match next {
            NextAuthority::Active(authority) => {
                self.owner_credentials.insert(owner, authority);
            }
            NextAuthority::Removed => {
                self.owner_credentials.remove(&owner);
            }
        }
    }
}

impl NextAuthority {
    fn active(&self) -> Option<&OwnerAuthority> {
        match self {
            Self::Active(authority) => Some(authority),
            Self::Removed => None,
        }
    }
}

#[cfg(test)]
pub(super) fn test_registration_epoch(owner: InstanceId) -> RegistrationEpoch {
    serde_json::from_value(serde_json::Value::String(
        uuid::Uuid::from_u128(owner.as_u128()).to_string(),
    ))
    .expect("UUID test owner must decode as a registration epoch")
}

fn authorize_owner(
    state: &BundleDirectoryState,
    owner: InstanceId,
    credential: &MutationCredential,
) -> Result<RegistrationEpoch, BundleDirectoryError> {
    let Some(expected) = state.owner_credentials.get(&owner) else {
        return Err(BundleDirectoryError::UnknownOwner { owner });
    };
    if &expected.credential != credential {
        return Err(BundleDirectoryError::UnauthorizedOwner { owner });
    }
    Ok(expected.registration_epoch)
}

fn map_identity_capacity(capacity: DirectoryIdentityCapacity) -> BundleDirectoryError {
    match capacity {
        DirectoryIdentityCapacity::Owner { owner, capacity } => {
            BundleDirectoryError::OwnerCapacity { owner, capacity }
        }
        DirectoryIdentityCapacity::Global { capacity } => {
            BundleDirectoryError::GlobalCapacity { capacity }
        }
    }
}

fn map_retirement_capacity(capacity: RetiredGenerationCapacity) -> BundleDirectoryError {
    match capacity {
        RetiredGenerationCapacity::AbsentOwner { owner, capacity } => {
            BundleDirectoryError::OwnerCapacity { owner, capacity }
        }
        RetiredGenerationCapacity::AbsentGlobal { capacity } => {
            BundleDirectoryError::GlobalCapacity { capacity }
        }
        RetiredGenerationCapacity::TotalOwner { owner, capacity } => {
            BundleDirectoryError::OwnerCapacity { owner, capacity }
        }
        RetiredGenerationCapacity::TotalGlobal { capacity } => {
            BundleDirectoryError::GlobalCapacity { capacity }
        }
    }
}

/// Wall-clock milliseconds. Shared with the tier-placement projection so both
/// halves of the indexer feature stamp advisory freshness from one clock.
pub(crate) fn unix_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BundleDirectoryError {
    #[error("invalid bundle advertisement: {0}")]
    InvalidAdvertisement(#[from] BundleLineageValidationError),
    #[error("bundle owner {owner} is not registered")]
    UnknownOwner { owner: InstanceId },
    #[error("bundle mutation credential does not authorize owner {owner}")]
    UnauthorizedOwner { owner: InstanceId },
    #[error("bundle advertisement does not match owner {owner}'s registration epoch")]
    RegistrationEpochMismatch { owner: InstanceId },
    #[error(
        "bundle advertisement from owner {owner} claims a placement for unrequired resource {resource:?}"
    )]
    UnrequiredPlacement {
        owner: InstanceId,
        resource: kvbm_common::LogicalResourceId,
    },
    #[error("bundle owner {owner} registration transaction changed")]
    StaleOwnerTransaction { owner: InstanceId },
    #[error("bundle owner {owner} already has a registration transaction in progress")]
    OwnerRegistrationInProgress { owner: InstanceId },
    #[error("bundle owner {owner} registration has not been bound to a registry incarnation")]
    UnboundOwnerRegistration { owner: InstanceId },
    #[error(
        "bundle owner {owner} registration incarnation changed (expected {expected}, got {attempted})"
    )]
    StaleOwnerRegistration {
        owner: InstanceId,
        expected: RegistryIncarnation,
        attempted: RegistryIncarnation,
    },
    #[error("bundle generation {attempted} is stale relative to generation {current}")]
    StaleGeneration { current: u64, attempted: u64 },
    #[error("bundle owner {owner} reached a directory capacity of {capacity}")]
    OwnerCapacity { owner: InstanceId, capacity: usize },
    #[error("bundle directory reached a global capacity of {capacity}")]
    GlobalCapacity { capacity: usize },
    #[error("bundle directory state is unavailable")]
    Unavailable,
}

#[cfg(test)]
mod reclamation_tests;
#[cfg(test)]
mod tests;
