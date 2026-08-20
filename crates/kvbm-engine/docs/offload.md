# Offload Module

The offload module manages the asynchronous transfer of KV cache blocks between storage tiers. It provides a pipeline-based architecture for evaluating, batching, and executing block transfers with full cancellation support.

## Overview

Offloading moves blocks from a source tier (e.g., GPU memory) to a destination tier (e.g., host memory, remote storage, or object storage). The pipeline ensures:

- **Policy-based filtering**: Only blocks meeting criteria are transferred
- **Batched execution**: Blocks are grouped for efficient transfer
- **Cancellation support**: Transfers can be cancelled at any point before commitment
- **Precondition synchronization**: Transfers wait for forward pass completion

## Pipeline Architecture

```text
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ PolicyEvaluator │────►│ PreconditionAwaiter │────►│       Batcher       │────►│ TransferExecutor │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘     └──────────────────┘
                                                             ▲                          ▲
                                                             │                          │
                                                    CancellableQueue          CancellableQueue
                                                             │                          │
                                                             └──────── CancelSweeper ───┘
```

### Stages

| Stage | Purpose |
|-------|---------|
| **PolicyEvaluator** | Filters blocks based on configured policies (frequency, presence, etc.) |
| **PreconditionAwaiter** | Waits for forward pass completion before proceeding |
| **Batcher** | Groups containers into batches based on total block count |
| **TransferExecutor** | Upgrades blocks and executes the actual transfer |

## Container Data Model

`OffloadContainer<T>` is a private pipeline value. It owns one optional payload and one optional `CancellationUnit`.
The payload owns transfer identity, source data, evaluated blocks, state, and an optional precondition event.
Evaluated blocks retain their pending guards until they fail or cross the commitment boundary.
The policy evaluator atomically claims each `SequenceHash` before it creates an evaluated block.
A failed claim does not create a second evaluated block.

```rust,ignore
struct OffloadContainer<T: BlockMetadata> {
    payload: Option<ContainerPayload<T>>,
    cancellation: Option<CancellationUnit>,
}

struct TransferBatch<T: BlockMetadata> {
    containers: Vec<OffloadContainer<T>>,
    timing: TimingTrace,
}
```

`TransferBatch` groups complete containers. It does not split a container before commitment.


### P1: Container is the Unit of Cancellation

One container owns one precommit cancellation unit. Its blocks do not cancel independently.
When cancellation wins before commitment, the complete container releases its source and pending guards.

### P2: Token Travels with Container

The handle and container share internal cancellation state. The container owns the sole `CancellationUnit`.
That unit moves through every stage. The pipeline does not clone or replace it before commitment.

### P3: Upgrade is the Commitment Boundary

The final claim and weak-to-strong upgrade form the physical commitment boundary.

- Before commitment, cancellation drops the complete container.
- After commitment, a request starts drain tracking. It does not cancel physical work.
- Confirmation waits for every committed route unit to settle.

### P4: Sweep Before Upgrade

The executor sweeps cancelled containers immediately before upgrade. Each remaining container then claims commitment against cancellation.

### P5: Flat Map After Upgrade

The executor flattens only containers that claim commitment. `ResolvedBatch<T>` retains `ResolvedBlock<T>` values.
Each resolved block retains transfer identity, source state, and any source or pending guard.
The container wrapper ends after upgrade. The physical ownership data remains until transfer completion.

### P6: PreconditionAwaiter Uses Select

The precondition awaiter selects the event and effective precommit cancellation. If cancellation wins, it releases the container immediately.

## Configuration

Pipeline behavior is controlled via `PipelineConfig`:

| Option | Default | Description |
|--------|---------|-------------|
| `resource` | `None` | Logical model resource whose physical tier layouts execute this pipeline; `None` uses the leader primary compatibility route |
| `batch_config.max_batch_size` | 1024 | Maximum blocks per batch |
| `batch_config.min_batch_size` | 8 | Minimum blocks before flush |
| `batch_config.flush_interval` | 10ms | Time before a partial batch flushes. It must not be zero. |
| `policy_timeout` | 100ms | Timeout for policy evaluation |
| `sweep_interval` | 10ms | Interval for the cancel sweeper. It must not be zero. |
| `max_concurrent_transfers` | 1 | Concurrent transfer batches |

## Usage

### Enqueue Blocks

```rust,ignore
let mut handle = engine.enqueue_g1_to_g2(blocks)?;

// Track progress
println!("Status: {:?}", handle.status());

// Wait for completion
let result = handle.wait().await?;
```

### Cancelling a Transfer

```rust,ignore
// Request cancellation and wait for confirmation
handle.cancel().wait().await;
// All offload-owned source and pending guards are now released.
```

## Related Documentation

- [offload-developer.md](offload-developer.md) - Implementation details and extension rules

