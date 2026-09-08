use super::*;
use crate::worker::PhysicalWorker;
use kvbm_physical::layout::{LayoutConfig, PhysicalLayout};
use kvbm_physical::manager::{ResourceLayoutHandles, TierLayoutHandles, TransferManager};
use kvbm_physical::testing::{TestAgentBuilder, create_fc_layout_with_config};
use kvbm_physical::transfer::{FillPattern, StorageKind, fill_blocks};

fn check_bytes(layout: &PhysicalLayout, block: BlockId, expected: u8) -> Result<usize> {
    let config = layout.layout().config();
    let mut count = 0;
    for layer in 0..config.num_layers {
        for outer in 0..config.outer_dim {
            let region = layout.memory_region(block, layer, outer)?;
            let bytes =
                unsafe { std::slice::from_raw_parts(region.addr() as *const u8, region.size()) };
            ensure!(
                bytes.iter().all(|byte| *byte == expected),
                "byte mismatch at block {block} layer {layer} outer {outer}"
            );
            count += bytes.len();
        }
    }
    Ok(count)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_staging_device_exact_bytes_use_the_bound_resource() -> Result<()> {
    let resource = LogicalResourceId(7);
    let decoy = LogicalResourceId::default();
    let agent = TestAgentBuilder::new(format!("session-staging-{}", uuid::Uuid::new_v4()))
        .require_backend("UCX")
        .build()?
        .into_nixl_agent();
    let config = LayoutConfig::builder()
        .num_blocks(2)
        .num_layers(2)
        .outer_dim(2)
        .page_size(4)
        .inner_dim(32)
        .dtype_width_bytes(2)
        .build()?;
    let layout = |storage| create_fc_layout_with_config(agent.clone(), storage, config.clone());
    let device = layout(StorageKind::Device(0));
    let host = layout(StorageKind::Pinned);
    let decoy_device = layout(StorageKind::Device(0));
    let decoy_host = layout(StorageKind::Pinned);
    fill_blocks(&device, &[0], FillPattern::Constant(0x31))?;
    fill_blocks(&device, &[1], FillPattern::Constant(0xc7))?;
    fill_blocks(&host, &[0, 1], FillPattern::Constant(0))?;
    fill_blocks(&decoy_device, &[0, 1], FillPattern::Constant(0xd3))?;
    fill_blocks(&decoy_host, &[0, 1], FillPattern::Constant(0xee))?;
    let manager = TransferManager::builder()
        .event_system(Arc::new(velo::EventManager::local()))
        .nixl_agent(agent)
        .cuda_device_id(0)
        .build()?;
    let handles = ResourceLayoutHandles::new(
        decoy,
        vec![
            (
                resource,
                TierLayoutHandles::new(
                    Some(manager.register_layout(device)?),
                    Some(manager.register_layout(host.clone())?),
                    None,
                ),
            ),
            (
                decoy,
                TierLayoutHandles::new(
                    Some(manager.register_layout(decoy_device)?),
                    Some(manager.register_layout(decoy_host.clone())?),
                    None,
                ),
            ),
        ],
    )?;
    let worker = Arc::new(
        PhysicalWorker::builder()
            .manager(manager)
            .resource_handles(handles)
            .build()?,
    );
    let Source {
        manager: g1,
        mut pins,
    } = source(2)?;
    pins.reverse();
    let source_ids = pins
        .iter()
        .map(ImmutableBlock::block_id)
        .collect::<Vec<_>>();
    let hashes = pins
        .iter()
        .map(ImmutableBlock::sequence_hash)
        .collect::<Vec<_>>();
    let registry = BlockRegistry::new();
    let logical = || {
        Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(2)
                .block_size(4)
                .registry(registry.clone())
                .build(),
        )
    };
    let g2 = logical();
    let mut managers = BlockManagerSet::new();
    managers.insert(resource, g2.clone())?;
    managers.insert(decoy, logical())?;
    let transport = velo::transports::tcp::TcpTransportBuilder::new()
        .from_listener(std::net::TcpListener::bind("127.0.0.1:0")?)?
        .build()?;
    let velo = velo::Velo::builder()
        .add_transport(Arc::new(transport))
        .build()
        .await?;
    let leader = InstanceLeader::builder()
        .messenger(velo.messenger().clone())
        .registry(registry)
        .g2_manager_set(Arc::new(managers), decoy)
        .workers(vec![worker])
        .build()?;
    let installation = unsafe { PolicyG1G2Route::new(leader.clone(), resource) }?;
    let route = installation
        .validate(resource, &g1)
        .map_err(|(error, _)| error)?
        .bind();
    let factory = crate::p2p::session::VeloSessionFactory::new(
        velo,
        Arc::new(leader),
        tokio::runtime::Handle::current(),
    );
    let holder = factory.open_concrete(uuid::Uuid::new_v4())?;
    holder.commit(hashes.clone())?;
    holder.finish_commits()?;
    ensure!(
        g2.match_blocks(&hashes).is_empty(),
        "G1-only source must have no mirrored G2 blocks"
    );
    let staged = route.stage_to_g2(pins).await?;
    holder.make_available(staged)?;
    let blocks = g2.match_blocks(&hashes);
    ensure!(blocks.len() == 2);
    let mut checked_bytes = 0;
    for (block, source_id) in blocks.iter().zip(source_ids) {
        let expected = if source_id == 0 { 0x31 } else { 0xc7 };
        checked_bytes += check_bytes(&host, block.block_id(), expected)?;
        check_bytes(&decoy_host, block.block_id(), 0xee)?;
    }
    ensure!(checked_bytes == 2048);
    ensure!(g1.match_blocks(&hashes).len() == 2);
    drop(blocks);
    ensure!(g1.available_blocks() == 2);
    for pull_id in [1, 2] {
        holder.test_inject_inbound_frame(crate::p2p::session::Frame::Pull {
            pull_id,
            hashes: hashes.clone(),
        });
    }
    ensure!(holder.test_inbound_pulls_count() == 2);
    holder.close(None);
    holder.test_inject_inbound_frame(crate::p2p::session::Frame::PullAck { pull_id: 1 });
    ensure!(g2.available_blocks() == 0);
    holder.test_inject_inbound_frame(crate::p2p::session::Frame::PullAck { pull_id: 2 });
    ensure!(holder.test_available_pin_count() == 0);
    ensure!(g2.match_blocks(&hashes).is_empty());
    ensure!(g2.available_blocks() == 2);
    eprintln!(
        "checked {checked_bytes} transferred bytes and {checked_bytes} untouched bytes in another resource"
    );
    Ok(())
}
