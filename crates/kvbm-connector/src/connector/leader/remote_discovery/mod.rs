//! Hub-backed remote block discovery for the engine search path.

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use kvbm_engine::leader::{RemoteBlockDiscovery, RemoteCandidates};
use kvbm_engine::p2p::session::PeerResolver;
use kvbm_hub::{FindBlocksHit, IndexerLookupClient};
use kvbm_logical::SequenceHash;

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
}

trait BlockIndex: Send + Sync {
    fn find_blocks(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<FindBlocksHit>>>;
}

struct HubBlockIndex(Arc<IndexerLookupClient>);

impl BlockIndex for HubBlockIndex {
    fn find_blocks(
        &self,
        hashes: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Result<Option<FindBlocksHit>>> {
        let index = Arc::clone(&self.0);
        Box::pin(async move {
            index
                .find_blocks(hashes)
                .await
                .context("query KVBM hub block index")
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Mutex;

    use anyhow::{Result, bail};
    use kvbm_engine::InstanceId;

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
        SequenceHash::new(position, None, position)
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
