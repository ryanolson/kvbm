// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Routing for committed G1-to-G2 chain output.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::G2;
use kvbm_logical::blocks::WeakBlock;

use super::handle::settle_transfer_unit;
use super::pipeline::{ChainOutput, ChainOutputRx, PipelineIngress};
use super::remote_g4::RemoteG4OffloadRequest;
use super::source::SourceBlocks;

/// Route committed G1-to-G2 output to each configured downstream tier.
///
/// This function creates the exact route-unit fan-out before it creates a
/// downstream owner. A remote request retains strong source guards. Local
/// routes receive weak blocks and remain best effort under G2 pressure.
pub(super) async fn run(
    mut chain_rx: ChainOutputRx<G2>,
    g2_to_g3_queue: Option<PipelineIngress<G2>>,
    g2_to_g4_queue: Option<PipelineIngress<G2>>,
    remote_g4_tx: Option<mpsc::Sender<RemoteG4OffloadRequest>>,
) {
    while let Some(output) = chain_rx.recv().await {
        let ChainOutput {
            transfer_id,
            blocks,
            state,
            cancellation,
        } = output;

        if blocks.is_empty() {
            drop(blocks);
            settle_transfer_unit(cancellation, state);
            continue;
        }

        let target_count = usize::from(g2_to_g3_queue.is_some())
            + usize::from(g2_to_g4_queue.is_some())
            + usize::from(remote_g4_tx.is_some());
        if target_count == 0 {
            drop(blocks);
            settle_transfer_unit(cancellation, state);
            continue;
        }
        let mut cancellation_units = cancellation.fan_out(target_count).into_iter();
        let weak_blocks: Vec<WeakBlock<G2>> =
            blocks.iter().map(|block| block.downgrade()).collect();

        let remote_request = if remote_g4_tx.is_some() {
            Some(RemoteG4OffloadRequest::from_blocks(
                transfer_id,
                blocks,
                Arc::clone(&state),
                cancellation_units
                    .next()
                    .expect("the remote G2 to G4 route owns one cancellation unit"),
            ))
        } else {
            drop(blocks);
            None
        };

        tracing::debug!(
            %transfer_id,
            num_blocks = weak_blocks.len(),
            "Routing chain output to downstream pipelines as WeakBlocks"
        );

        enqueue_local_route(
            g2_to_g3_queue.as_ref(),
            transfer_id,
            &weak_blocks,
            &state,
            &mut cancellation_units,
            "G2 to G3",
        );
        enqueue_local_route(
            g2_to_g4_queue.as_ref(),
            transfer_id,
            &weak_blocks,
            &state,
            &mut cancellation_units,
            "local G2 to G4",
        );

        if let (Some(tx), Some(request)) = (&remote_g4_tx, remote_request)
            && let Err(error) = tx.send(request).await
        {
            error
                .0
                .fail_before_dispatch("remote G4 offload channel closed".to_string());
            tracing::warn!(%transfer_id, "Remote G4 offload channel closed");
        }
        debug_assert!(cancellation_units.next().is_none());
    }

    tracing::debug!("Chain router task shutting down");
}

fn enqueue_local_route(
    ingress: Option<&PipelineIngress<G2>>,
    transfer_id: super::handle::TransferId,
    weak_blocks: &[WeakBlock<G2>],
    state: &Arc<std::sync::Mutex<super::handle::TransferState>>,
    cancellation_units: &mut std::vec::IntoIter<super::cancel::CancellationUnit>,
    route: &str,
) {
    let Some(ingress) = ingress else {
        return;
    };
    if !ingress.enqueue_chained(
        transfer_id,
        SourceBlocks::Weak(weak_blocks.to_vec()),
        Arc::clone(state),
        cancellation_units
            .next()
            .expect("each configured local route owns one cancellation unit"),
    ) {
        tracing::debug!(%transfer_id, %route, "Chained enqueue skipped after cancellation");
    }
}
