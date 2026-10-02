//! Production hub and peer wiring for standalone remote search.

use std::sync::Arc;

use anyhow::{Context, Result};
use kvbm_engine::p2p::session::PeerResolver;

use super::super::cd;
use super::super::construct::{EngineStack, build_indexer_publisher};
use super::super::hub_handshake::HubHandshake;
use super::super::{Construction, Leader};
use super::HubRemoteDiscovery;

/// Register as an Indexer+P2P participant without enabling disaggregation.
pub(in crate::connector::leader) async fn wire_remote_search(
    leader: &Arc<Leader>,
    construction: &Construction,
    stack: &EngineStack,
    handshake: &HubHandshake,
) -> Result<kvbm_engine::RemoteOps> {
    let runtime = &construction.runtime;
    let manifest = construction
        .cache_manifest
        .lock()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("remote search requires a registered cache manifest"))?;
    let manifest_id = manifest.id();
    let velo = runtime
        .velo()
        .context("remote bundle search requires a Velo runtime")?
        .clone();
    let (worker_metadata, num_workers) = {
        let workers = construction.workers.lock();
        (workers.metadata.first().cloned(), workers.metadata.len())
    };
    let worker_metadata =
        worker_metadata.ok_or_else(|| anyhow::anyhow!("remote search worker metadata is empty"))?;
    let layout_compat = cd::wiring::build_layout_compat_payload(
        runtime,
        &stack.reference_config,
        &worker_metadata,
        num_workers,
    )?;
    let features = vec![
        kvbm_hub::Feature::P2P(kvbm_hub::P2pConfig { layout_compat }),
        kvbm_hub::Feature::Indexer(kvbm_hub::IndexerFeatureConfig {
            max_seq_len: runtime.config().max_seq_len,
            manifest,
            create_kind: kvbm_logical::events::CreateKind::Block,
        }),
    ];
    let foundation = cd::wiring::wire_hub(runtime, &stack.instance_leader, handshake, features)
        .await
        .context("remote search hub/P2P wiring failed")?;
    let index = foundation
        .hub
        .indexer_lookup_client(velo.messenger().clone())
        .await?
        .ok_or_else(|| anyhow::anyhow!("remote search requires the hub bundle index"))?;
    let discovery = HubRemoteDiscovery::new(
        index,
        Arc::clone(&foundation.peer_resolver) as Arc<dyn PeerResolver>,
        manifest_id,
    );

    if let (Some(transport), Some(events)) = (
        handshake.indexer_transport.as_ref(),
        stack.events_manager.as_ref(),
    ) && let Some(publisher) = build_indexer_publisher(runtime, transport, events).await
    {
        let _ = leader.indexer_publisher.set(publisher);
    }
    let _ = leader.cd_hub_client.set(Arc::clone(&foundation.hub));
    Ok(kvbm_engine::RemoteOps::with_search(discovery))
}
