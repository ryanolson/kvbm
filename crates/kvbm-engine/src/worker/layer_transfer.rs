// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use futures::FutureExt;
use futures::future::{Either, select};
use kvbm_physical::transfer::{TransferCompleteNotification, TransferDrainOutcome};
use std::time::{Duration, Instant};

// Guarded and custom receipts can need completion polling. Bound that wait
// independently of the connector runtime so a stalled poller cannot hang decode.
const LAYER_TRANSFER_TIMEOUT: Duration = Duration::from_secs(30);

/// Check a caller-stream transfer from a synchronous decode hook.
/// CUDA success with a caller-owned stream returns a ready dispatch receipt.
pub(crate) fn finish_layer_transfer(
    receipt: TransferCompleteNotification,
    layer: usize,
) -> Result<()> {
    finish_receipt(receipt, layer, LAYER_TRANSFER_TIMEOUT)
}

fn finish_receipt(
    receipt: TransferCompleteNotification,
    layer: usize,
    timeout: Duration,
) -> Result<()> {
    if !receipt.could_yield() {
        return Ok(());
    }

    let mut drain = Box::pin(receipt.await_drain());
    let outcome = if let Some(ready) = drain.as_mut().now_or_never() {
        ready
    } else if tokio::runtime::Handle::try_current().is_ok() {
        std::mem::forget(drain);
        TransferDrainOutcome::Unproven(anyhow::anyhow!(
            "layer {layer} cannot wait for drain on a Tokio thread"
        ))
    } else {
        let started = Instant::now();
        match futures::executor::block_on(select(drain, futures_timer::Delay::new(timeout))) {
            Either::Left((outcome, _)) => outcome,
            Either::Right(((), pending)) => {
                // A timeout leaves the work unproven. Keep the receipt's holds.
                std::mem::forget(pending);
                let elapsed = started.elapsed();
                tracing::error!(
                    layer,
                    elapsed_ms = elapsed.as_millis(),
                    "Layer transfer timed out"
                );
                TransferDrainOutcome::Unproven(anyhow::anyhow!(
                    "layer {layer} transfer timed out after {elapsed:?}"
                ))
            }
        }
    };

    match outcome {
        TransferDrainOutcome::Completed => Ok(()),
        TransferDrainOutcome::DrainedWithError(error) | TransferDrainOutcome::Unproven(error) => {
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn layer_transfer_accepts_the_immediate_success_receipt() {
        let receipt = TransferCompleteNotification::completed();
        assert!(!receipt.could_yield());
        finish_layer_transfer(receipt, 7).unwrap();
    }

    #[test]
    fn layer_transfer_propagates_ready_failures() {
        for outcome in [
            TransferDrainOutcome::DrainedWithError(anyhow::anyhow!("copy failed")),
            TransferDrainOutcome::Unproven(anyhow::anyhow!("drain unproven")),
        ] {
            let receipt =
                TransferCompleteNotification::from_drain_future(std::future::ready(outcome));
            assert!(finish_receipt(receipt, 7, Duration::from_secs(1)).is_err());
        }
    }

    #[test]
    fn layer_transfer_bounds_pending_receipt_on_plain_thread() {
        let kept = Arc::new(());
        let ticket = kept.clone();
        let receipt = TransferCompleteNotification::from_drain_future(async move {
            let _keep = ticket;
            std::future::pending().await
        });
        let error = finish_receipt(receipt, 7, Duration::from_millis(15)).unwrap_err();
        assert!(error.to_string().contains("layer 7"));
        assert!(error.to_string().contains("timed out"));
        assert_eq!(
            Arc::strong_count(&kept),
            2,
            "timed-out receipt dropped its hold"
        );
    }
}
