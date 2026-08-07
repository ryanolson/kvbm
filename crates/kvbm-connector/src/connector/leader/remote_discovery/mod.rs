//! Hub-backed remote block discovery for the engine search path.

mod index;
mod wiring;

pub(super) use wiring::wire_remote_search;

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use kvbm_engine::leader::{RemoteBlockDiscovery, RemoteCandidates};
use kvbm_engine::p2p::session::PeerResolver;
use kvbm_engine::remote::search::bundle::{
    BundleAdvertisement, BundleDiscoveryOutcome, BundleDiscoveryQuery, BundleInvalidation,
};
use kvbm_hub::IndexerLookupClient;
use kvbm_logical::SequenceHash;

use index::{BlockIndex, HubBlockIndex};

/// Resolves indexed block holders and makes them reachable before returning.
pub struct HubRemoteDiscovery {
    index: Arc<dyn BlockIndex>,
    peers: Arc<dyn PeerResolver>,
}

impl HubRemoteDiscovery {
    /// Build discovery over the hub indexer and the local Velo peer resolver.
    pub fn new(index: Arc<IndexerLookupClient>, peers: Arc<dyn PeerResolver>) -> Arc<Self> {
        Arc::new(Self {
            index: Arc::new(HubBlockIndex(index)),
            peers,
        })
    }

    #[cfg(test)]
    fn with_backends(index: Arc<dyn BlockIndex>, peers: Arc<dyn PeerResolver>) -> Arc<Self> {
        Arc::new(Self { index, peers })
    }
}

impl RemoteBlockDiscovery for HubRemoteDiscovery {
    fn discover(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<RemoteCandidates>>> {
        let index = Arc::clone(&self.index);
        let peers = Arc::clone(&self.peers);
        Box::pin(async move {
            let Some(hit) = index.find_blocks(hashes).await? else {
                return Ok(None);
            };

            let mut reachable = Vec::with_capacity(hit.candidates.len());
            let mut last_error = None;
            for instance in hit.candidates {
                match peers.resolve_and_register(instance).await {
                    Ok(()) => reachable.push(instance),
                    Err(error) => {
                        tracing::debug!(
                            %instance,
                            error = %error,
                            "indexed KVBM peer is unreachable",
                        );
                        last_error = Some(error);
                    }
                }
            }

            if reachable.is_empty() {
                return match last_error {
                    Some(error) => Err(error.context("all indexed KVBM peers are unreachable")),
                    None => Ok(None),
                };
            }
            Ok(Some(RemoteCandidates {
                deepest: hit.matched,
                instances: reachable,
            }))
        })
    }

    fn discover_bundle(
        &self,
        query: BundleDiscoveryQuery,
    ) -> BoxFuture<'static, Result<BundleDiscoveryOutcome>> {
        let index = Arc::clone(&self.index);
        let peers = Arc::clone(&self.peers);
        Box::pin(async move {
            let candidate = match index.find_bundle(query).await? {
                BundleDiscoveryOutcome::Hit(candidate) => candidate,
                miss @ BundleDiscoveryOutcome::Miss(_) => return Ok(miss),
            };
            let owner = candidate.advertisement().owner();
            peers
                .resolve_and_register(owner)
                .await
                .with_context(|| format!("remote bundle owner {owner} is unreachable"))?;
            Ok(BundleDiscoveryOutcome::Hit(candidate))
        })
    }

    fn advertise_bundle(
        &self,
        advertisement: BundleAdvertisement,
    ) -> BoxFuture<'static, Result<()>> {
        self.index.publish_bundle(advertisement)
    }

    fn invalidate_bundle(
        &self,
        invalidation: BundleInvalidation,
    ) -> BoxFuture<'static, Result<()>> {
        self.index.invalidate_bundle(invalidation)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::collections::HashSet;
    use std::sync::Mutex;

    use anyhow::{Result, bail};
    use kvbm_engine::InstanceId;
    use kvbm_hub::{BundleAdvertisementRecord, BundleQueryHit, FindBlocksHit};
    use kvbm_protocols::cache_manifest::{
        BundleKey, BundleResourceLineage, CacheManifest, ModelIdentity, RegistrationEpoch,
        ResourceRequirement, ResourceRole,
    };

    use super::index::{advertisement_record, directory_hit};
    use super::*;

    struct StubIndex {
        hit: Option<FindBlocksHit>,
    }

    impl BlockIndex for StubIndex {
        fn find_blocks(
            &self,
            _hashes: Vec<SequenceHash>,
        ) -> BoxFuture<'static, Result<Option<FindBlocksHit>>> {
            let hit = self.hit.clone();
            Box::pin(async move { Ok(hit) })
        }
    }

    struct RecordingPeers {
        failed: HashSet<InstanceId>,
        seen: Mutex<Vec<InstanceId>>,
    }

    impl PeerResolver for RecordingPeers {
        fn resolve_and_register(&self, id: InstanceId) -> BoxFuture<'_, Result<()>> {
            self.seen.lock().unwrap().push(id);
            Box::pin(async move {
                if self.failed.contains(&id) {
                    bail!("peer {id} is unreachable");
                }
                Ok(())
            })
        }
    }

    fn hash(position: u64) -> SequenceHash {
        (1..=position).fold(SequenceHash::root(1), |parent, block| {
            parent.extend(block + 1)
        })
    }

    fn bundle_query() -> (BundleDiscoveryQuery, BundleKey) {
        let identity = CacheManifest::new(
            ModelIdentity::new("remote-discovery", "v1", [3; 32]).unwrap(),
            "remote-discovery-v1",
            vec![
                ResourceRequirement::new(
                    kvbm_common::LogicalResourceId(7),
                    ResourceRole::PrefixHistory,
                    4,
                )
                .unwrap(),
            ],
            BTreeMap::new(),
        )
        .unwrap()
        .identity();
        let key = BundleKey::new(&identity, hash(1), 8).unwrap();
        (BundleDiscoveryQuery::new(identity, vec![key], 1_000), key)
    }

    #[test]
    fn bundle_directory_hit_preserves_manifest_owner_and_lease() {
        let (query, key) = bundle_query();
        let requirements = query.identity().resources().to_vec();
        let owner = InstanceId::new_v4();
        let registration_epoch = RegistrationEpoch::new();
        let candidate = directory_hit(
            query,
            BundleQueryHit {
                advertisement: BundleAdvertisementRecord {
                    key,
                    generation: 4,
                    owner,
                    registration_epoch: Some(registration_epoch),
                    requirements,
                    lineages: vec![
                        BundleResourceLineage::new(
                            kvbm_common::LogicalResourceId(7),
                            vec![hash(0), hash(1)],
                        )
                        .unwrap(),
                    ],
                    expires_at_unix_ms: 10_000,
                    placements: Vec::new(),
                    stage_cost_hint_us: None,
                    advertised_at_unix_ms: None,
                },
                lease_id: uuid::Uuid::new_v4(),
                lease_expires_at_unix_ms: 2_000,
                ready_tier: None,
                advertised_at_unix_ms: None,
            },
        )
        .unwrap();

        assert_eq!(candidate.advertisement().key(), key);
        assert_eq!(candidate.advertisement().owner(), owner);
        assert_eq!(
            candidate.advertisement().registration_epoch(),
            registration_epoch
        );
        assert_eq!(candidate.lease_expires_at_unix_ms(), 2_000);
    }

    #[test]
    fn bundle_directory_hit_requires_an_owner_registration_epoch() {
        let (query, key) = bundle_query();
        let requirements = query.identity().resources().to_vec();
        let result = directory_hit(
            query,
            BundleQueryHit {
                advertisement: BundleAdvertisementRecord {
                    key,
                    generation: 4,
                    owner: InstanceId::new_v4(),
                    registration_epoch: None,
                    requirements,
                    lineages: vec![
                        BundleResourceLineage::new(
                            kvbm_common::LogicalResourceId(7),
                            vec![hash(0), hash(1)],
                        )
                        .unwrap(),
                    ],
                    expires_at_unix_ms: 10_000,
                    placements: Vec::new(),
                    stage_cost_hint_us: None,
                    advertised_at_unix_ms: None,
                },
                lease_id: uuid::Uuid::new_v4(),
                lease_expires_at_unix_ms: 2_000,
                ready_tier: None,
                advertised_at_unix_ms: None,
            },
        );

        assert!(result.is_err());
    }

    #[test]
    fn bundle_directory_hit_rejects_requirements_different_from_query() {
        let (query, key) = bundle_query();
        let result = directory_hit(
            query,
            BundleQueryHit {
                advertisement: BundleAdvertisementRecord {
                    key,
                    generation: 4,
                    owner: InstanceId::new_v4(),
                    registration_epoch: Some(RegistrationEpoch::new()),
                    requirements: vec![
                        ResourceRequirement::new(
                            kvbm_common::LogicalResourceId(7),
                            ResourceRole::BoundaryCapsule,
                            4,
                        )
                        .unwrap(),
                    ],
                    lineages: vec![
                        BundleResourceLineage::new(
                            kvbm_common::LogicalResourceId(7),
                            vec![hash(0), hash(1)],
                        )
                        .unwrap(),
                    ],
                    expires_at_unix_ms: 10_000,
                    placements: Vec::new(),
                    stage_cost_hint_us: None,
                    advertised_at_unix_ms: None,
                },
                lease_id: uuid::Uuid::new_v4(),
                lease_expires_at_unix_ms: 2_000,
                ready_tier: None,
                advertised_at_unix_ms: None,
            },
        );

        assert!(result.is_err());
    }

    #[test]
    fn hub_round_trip_preserves_distinct_mixed_native_resource_lineages() {
        let primary = kvbm_common::LogicalResourceId(17);
        let secondary = kvbm_common::LogicalResourceId(18);
        let capsule = kvbm_common::LogicalResourceId(19);
        let identity = CacheManifest::new(
            ModelIdentity::new("remote-discovery-mixed-native", "v1", [6; 32]).unwrap(),
            "remote-discovery-mixed-native-v1",
            vec![
                ResourceRequirement::new(primary, ResourceRole::PrefixHistory, 4).unwrap(),
                ResourceRequirement::new(secondary, ResourceRole::PrefixHistory, 8).unwrap(),
                ResourceRequirement::new(capsule, ResourceRole::BoundaryCapsule, 4).unwrap(),
            ],
            BTreeMap::new(),
        )
        .unwrap()
        .identity();
        let primary_hashes = vec![hash(0), hash(1)];
        let secondary_hashes =
            BundleResourceLineage::project_from_canonical(secondary, &primary_hashes, 2)
                .unwrap()
                .hashes()
                .to_vec();
        let key = BundleKey::new(&identity, primary_hashes[1], 8).unwrap();
        let owner = InstanceId::new_v4();
        let registration_epoch = RegistrationEpoch::new();
        let advertisement = BundleAdvertisement::new(
            identity.clone(),
            key,
            7,
            owner,
            registration_epoch,
            10_000,
            [
                BundleResourceLineage::new(primary, primary_hashes.clone()).unwrap(),
                BundleResourceLineage::new(secondary, secondary_hashes.clone()).unwrap(),
                BundleResourceLineage::new(capsule, vec![primary_hashes[1]]).unwrap(),
            ],
        )
        .unwrap();
        let record = advertisement_record(&advertisement);
        assert_eq!(record.requirements.as_slice(), identity.resources());
        let query = BundleDiscoveryQuery::new(identity.clone(), vec![key], 1_000);
        let hit = BundleQueryHit {
            advertisement: record,
            lease_id: uuid::Uuid::new_v4(),
            lease_expires_at_unix_ms: 1_500,
            ready_tier: None,
            advertised_at_unix_ms: None,
        };
        assert_eq!(
            hit.advertisement.requirements.as_slice(),
            identity.resources()
        );
        let candidate = directory_hit(query, hit).unwrap();
        let round_trip = candidate
            .advertisement()
            .lineages()
            .map(|lineage| (lineage.resource(), lineage.hashes().to_vec()))
            .collect::<BTreeMap<_, _>>();

        assert_eq!(round_trip[&primary], primary_hashes);
        assert_eq!(round_trip[&secondary], secondary_hashes);
        assert_eq!(round_trip[&capsule], vec![key.boundary_hash()]);
    }

    #[tokio::test]
    async fn returns_only_candidates_registered_with_velo() {
        let unreachable = InstanceId::new_v4();
        let reachable = InstanceId::new_v4();
        let peers = Arc::new(RecordingPeers {
            failed: HashSet::from([unreachable]),
            seen: Mutex::new(Vec::new()),
        });
        let discovery = HubRemoteDiscovery::with_backends(
            Arc::new(StubIndex {
                hit: Some(FindBlocksHit {
                    matched: hash(7),
                    candidates: vec![unreachable, reachable],
                }),
            }),
            Arc::clone(&peers) as Arc<dyn PeerResolver>,
        );

        let found = discovery
            .discover(vec![hash(3), hash(7)])
            .await
            .unwrap()
            .unwrap();

        assert_eq!(found.deepest, hash(7));
        assert_eq!(found.instances, vec![reachable]);
        assert_eq!(*peers.seen.lock().unwrap(), vec![unreachable, reachable]);
    }

    #[tokio::test]
    async fn full_index_miss_does_not_resolve_peers() {
        let peers = Arc::new(RecordingPeers {
            failed: HashSet::new(),
            seen: Mutex::new(Vec::new()),
        });
        let discovery = HubRemoteDiscovery::with_backends(
            Arc::new(StubIndex { hit: None }),
            Arc::clone(&peers) as Arc<dyn PeerResolver>,
        );

        assert!(discovery.discover(vec![hash(1)]).await.unwrap().is_none());
        assert!(peers.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn all_unreachable_candidates_surface_an_infrastructure_error() {
        let unreachable = InstanceId::new_v4();
        let discovery = HubRemoteDiscovery::with_backends(
            Arc::new(StubIndex {
                hit: Some(FindBlocksHit {
                    matched: hash(1),
                    candidates: vec![unreachable],
                }),
            }),
            Arc::new(RecordingPeers {
                failed: HashSet::from([unreachable]),
                seen: Mutex::new(Vec::new()),
            }),
        );

        let error = discovery.discover(vec![hash(1)]).await.unwrap_err();
        assert!(error.to_string().contains("unreachable"));
    }
}
