// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Construction-time configuration for the connector engine.
//!
//! [`ConnectorEngineConfig`] is the single config object the connector hands to
//! [`build_local_connector_engine`](super::build_local_connector_engine). It
//! carries the layout `block_size` (the engine has no other source for it —
//! `InstanceLeader` exposes no block-size accessor) and the engine's remote-ops
//! selection, [`RemoteOps`].
//!
//! Each remote capability in `RemoteOps` is an independent `Option` carrying
//! its own required dependency: a field present means the capability is enabled
//! with the transport it needs; absent means disabled. Misconfiguration (enabled
//! without the needed dependency) is therefore unrepresentable rather than
//! checked at runtime.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use kvbm_common::{LogicalResourceId, SequenceHash};

use crate::p2p::session::{PeerResolver, SessionFactory};
use crate::remote::cd::DisaggConfig;
use crate::remote::cd::budget::TierCell;
use crate::remote::cd::wire::PrefillPlane;
use crate::remote::search::discovery::RemoteDiscoveryHandle;
use crate::tiering::policy::{ResourceComponentBytes, ResourcePolicies};

/// Construction config for the in-process connector engine.
///
/// No `Default`: `block_size` has no sane default, so an engine configured by
/// omission would be a silent misconfiguration. Callers must name the layout,
/// remote capabilities, and resource policies explicitly.
#[derive(Clone, Debug)]
pub struct ConnectorEngineConfig {
    /// Layout block size, carried here because `InstanceLeader` exposes no
    /// block-size accessor.
    pub block_size: usize,
    /// Which remote-block operations the engine offers.
    pub remote: RemoteOps,
    /// Per-resource admission, retention, and inactive-backend policy.
    pub resource_policies: ResourcePolicies,
    /// Positive byte widths of each atomic physical component in one logical
    /// block, keyed by the same resource set as `resource_policies`.
    pub resource_component_bytes: ResourceComponentBytes,
}

/// Observer invoked after a remotely-pulled bundle is committed to the local
/// catalog, carrying the lineage that just became resident in G2.
///
/// The map is `resource → sequence hashes in position order`, i.e. exactly the
/// keys a tier-placement publisher advertises `Ready`. Remote pull is the one
/// route to a G2 copy that the offload pipeline's G1→G2 register observer
/// cannot see, so without this seam a puller under-reports its own residency.
///
/// **Hashes, not `ImmutableBlock`s, deliberately.** The register-observer
/// precedent hands out blocks and documents that cloning one retains a pin; an
/// observer that leaked a clone here would pin pulled G2 slots against eviction
/// and silently change cache behaviour. Nothing an advertiser needs is in the
/// block beyond its hash.
///
/// Contract: fires **after** the catalog commit succeeds, so it can never
/// announce a copy that failed to publish, and never on the idempotent
/// already-committed path. Must not block — it runs on the pull-completion
/// task, ahead of the directory advertisement.
pub type PulledBundleReadyObserver =
    Arc<dyn Fn(&BTreeMap<LogicalResourceId, Vec<SequenceHash>>) + Send + Sync + 'static>;

/// The remote-block operations the engine offers.
///
/// Each remote capability is an independent `Option` carrying its own required
/// dependency; the engine enables exactly the capabilities whose `Option` is
/// `Some`. The default (all `None`) is fully local and safe to use by omission.
/// Remote search-and-pull and conditional disagg are siblings — either, both,
/// or neither.
///
/// All fields are `pub(crate)` to keep the public surface curated; callers
/// construct via [`RemoteOps::default`] (all `None`), [`RemoteOps::with_search`],
/// [`RemoteOps::with_disagg_transports`], and/or
/// [`RemoteOps::observing_pulled_bundles`].
#[derive(Clone, Default)]
pub struct RemoteOps {
    /// Remote search-and-pull: install the discovery on the leader and request
    /// the remote path on shard searches.
    pub(crate) search: Option<RemoteSearchOps>,
    /// Conditional disaggregation: the decode-side remote-prefill plane.
    pub(crate) disagg: Option<DisaggOps>,
    /// Notified when a remote pull publishes into G2 (advisory; not a
    /// capability).
    pub(crate) pulled_bundle_ready: Option<PulledBundleReadyObserver>,
}

impl RemoteOps {
    /// Enable remote search-and-pull with the given hub-backed discovery
    /// (conditional disagg left disabled).
    pub fn with_search(discovery: RemoteDiscoveryHandle) -> Self {
        Self {
            search: Some(RemoteSearchOps { discovery }),
            disagg: None,
            pulled_bundle_ready: None,
        }
    }

    /// Observe remotely-pulled bundles as they become resident in G2.
    ///
    /// Composable with the capability builders and independent of them: an
    /// observer without [`Self::with_search`] simply never fires, because
    /// nothing pulls. Kept here rather than on
    /// [`ConnectorEngineConfig`] so adding it breaks no existing struct
    /// literal, and rather than as a post-construction `add_*_observer` because
    /// the built engine is returned only as trait objects — there is no
    /// concrete receiver to call.
    #[must_use]
    pub fn observing_pulled_bundles(mut self, observer: PulledBundleReadyObserver) -> Self {
        self.pulled_bundle_ready = Some(observer);
        self
    }

    /// Add the conditional-disagg sibling with the full production transport
    /// set, composable with [`Self::with_search`]. The connector's CD wiring
    /// assembles the parts: the hub-built session factory, the hub-queue
    /// prefill plane, the tier cell its velo tier-signal handler writes, the
    /// translated engine-local config, and the hub-backed peer resolver
    /// (`None` only when every peer is pre-registered — single process,
    /// tests).
    pub fn with_disagg_transports(
        mut self,
        sessions: Arc<dyn SessionFactory>,
        prefill_plane: Arc<dyn PrefillPlane>,
        tier: Arc<TierCell>,
        cfg: DisaggConfig,
        peer_resolver: Option<Arc<dyn PeerResolver>>,
    ) -> Self {
        self.disagg = Some(DisaggOps {
            sessions,
            prefill_plane,
            tier,
            cfg,
            peer_resolver,
        });
        self
    }

    /// Test convenience: [`Self::with_disagg_transports`] without a peer
    /// resolver (in-process tests pre-register their peers).
    #[cfg(test)]
    pub(crate) fn with_disagg(
        self,
        sessions: Arc<dyn SessionFactory>,
        prefill_plane: Arc<dyn PrefillPlane>,
        tier: Arc<TierCell>,
        cfg: DisaggConfig,
    ) -> Self {
        self.with_disagg_transports(sessions, prefill_plane, tier, cfg, None)
    }
}

/// Dependencies required to enable remote search-and-pull.
#[derive(Clone)]
pub(crate) struct RemoteSearchOps {
    /// Hub-backed resolver for which remote instances hold uncached blocks.
    pub discovery: RemoteDiscoveryHandle,
}

/// Dependencies required to enable conditional disaggregation (the decode-side
/// remote-prefill plane). Crate-internal: assembled via
/// [`RemoteOps::with_disagg_transports`], whose signature carries the five
/// parts as arguments so this struct never needs to cross the crate boundary.
#[derive(Clone)]
pub(crate) struct DisaggOps {
    /// Opens the holder-side CD session committed at search time.
    pub(crate) sessions: Arc<dyn SessionFactory>,
    /// Enqueues the completed remote-prefill request for a prefill worker.
    pub(crate) prefill_plane: Arc<dyn PrefillPlane>,
    /// The hub-pushed circuit-breaker tier the decode reads in its search path.
    pub(crate) tier: Arc<TierCell>,
    /// The resource-free decision-core config (selection + admission knobs).
    pub(crate) cfg: DisaggConfig,
    /// Resolves + registers the decode peer on the local velo before the
    /// prefill pipeline attaches to its session (the streaming-transport
    /// registry is populated lazily — see [`PeerResolver`]). `None` skips the
    /// resolve, which only works when the peer is already registered (single
    /// process, tests).
    pub(crate) peer_resolver: Option<Arc<dyn PeerResolver>>,
}

impl fmt::Debug for RemoteOps {
    // `RemoteDiscoveryHandle` / the session+plane handles are `dyn` trait
    // objects with no `Debug` bound. Print each field as present/absent.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteOps")
            .field("search", &self.search.as_ref().map(|_| "<RemoteSearchOps>"))
            .field("disagg", &self.disagg.as_ref().map(|_| "<DisaggOps>"))
            .field(
                "pulled_bundle_ready",
                &self
                    .pulled_bundle_ready
                    .as_ref()
                    .map(|_| "<PulledBundleReadyObserver>"),
            )
            .finish()
    }
}

impl fmt::Debug for RemoteSearchOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteSearchOps")
            .field("discovery", &"<dyn RemoteBlockDiscovery>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use futures::FutureExt;

    use super::*;
    use crate::p2p::session::MockSessionFactory;
    use crate::remote::cd::wire::PrefillDispatch;

    struct NoopPlane;

    impl PrefillPlane for NoopPlane {
        fn dispatch(
            &self,
            _req: PrefillDispatch,
        ) -> futures::future::BoxFuture<'static, anyhow::Result<()>> {
            async { Ok(()) }.boxed()
        }
    }

    struct NoopResolver;

    impl crate::p2p::session::PeerResolver for NoopResolver {
        fn resolve_and_register(
            &self,
            _instance_id: velo::InstanceId,
        ) -> futures::future::BoxFuture<'_, anyhow::Result<()>> {
            async { Ok(()) }.boxed()
        }
    }

    /// The observer composes with the capability builders instead of replacing
    /// them, and survives being set alongside either one — a pull observer is
    /// only useful on an engine that also has remote search.
    #[test]
    fn the_pulled_bundle_observer_composes_with_the_capability_builders() {
        let fired = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&fired);
        let observer: PulledBundleReadyObserver = Arc::new(move |_| {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });

        let ops = RemoteOps::default()
            .observing_pulled_bundles(Arc::clone(&observer))
            .with_disagg_transports(
                Arc::new(MockSessionFactory::default()),
                Arc::new(NoopPlane),
                Arc::new(TierCell::default()),
                DisaggConfig::default(),
                None,
            );
        let stored = ops
            .pulled_bundle_ready
            .as_ref()
            .expect("the handed observer must be stored");
        assert!(
            Arc::ptr_eq(stored, &observer),
            "the handed observer must be stored verbatim"
        );
        assert!(ops.disagg.is_some(), "and must not displace a capability");
        stored(&BTreeMap::new());
        assert_eq!(fired.load(std::sync::atomic::Ordering::SeqCst), 1);

        assert!(
            RemoteOps::default().pulled_bundle_ready.is_none(),
            "an engine observes nothing by default"
        );
    }

    /// The production constructor stores every transport it is handed —
    /// asserted directly on the `DisaggOps` payload (pointer identity for
    /// the Arcs, presence for the resolver), not by comparing against the
    /// delegating test convenience.
    #[test]
    fn with_disagg_transports_stores_the_handed_parts() {
        let sessions = Arc::new(MockSessionFactory::default());
        let plane: Arc<dyn PrefillPlane> = Arc::new(NoopPlane);
        let tier = Arc::new(TierCell::default());
        let resolver: Arc<dyn crate::p2p::session::PeerResolver> = Arc::new(NoopResolver);
        let cfg = DisaggConfig::default();

        let sessions_dyn: Arc<dyn crate::p2p::session::SessionFactory> = sessions.clone();
        let production = RemoteOps::default().with_disagg_transports(
            sessions.clone(),
            plane.clone(),
            tier.clone(),
            cfg,
            Some(resolver.clone()),
        );

        let ops = production.disagg.expect("production path enables disagg");
        assert!(
            Arc::ptr_eq(&ops.sessions, &sessions_dyn),
            "the handed session factory must be stored verbatim"
        );
        assert!(
            Arc::ptr_eq(&ops.prefill_plane, &plane),
            "the handed prefill plane must be stored verbatim"
        );
        assert!(
            Arc::ptr_eq(&ops.tier, &tier),
            "the handed tier cell must be stored verbatim"
        );
        let stored = ops
            .peer_resolver
            .as_ref()
            .expect("the handed resolver must be stored");
        assert!(
            Arc::ptr_eq(stored, &resolver),
            "the handed peer resolver must be stored verbatim"
        );
        assert_eq!(
            format!("{:?}", ops.cfg),
            format!("{cfg:?}"),
            "the handed config must be stored verbatim"
        );
    }
}
