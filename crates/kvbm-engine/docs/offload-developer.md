# Offload Module Developer Guide

This document provides implementation details for developers working on the offload pipeline. For high-level concepts and policy statements, see [offload.md](offload.md).

## Container-Based Architecture

### OffloadContainer

The container is the fundamental unit that flows through the pipeline:

```rust,ignore
struct OffloadContainer<T: BlockMetadata> {
    payload: Option<ContainerPayload<T>>,
    cancellation: Option<CancellationUnit>,
}

impl<T: BlockMetadata> OffloadContainer<T> {
    fn is_cancelled(&self) -> bool {
        self.cancellation_token().is_precommit_cancelled()
    }

    fn upgrade(self) -> Option<UpgradedContainer<T>> {
        if !self.cancellation.claim_commitment() {
            return None;
        }
        // Move the same unit into the upgraded container.
    }
}
```

### TransferBatch

Batches group multiple containers for efficient transfer:

```rust,ignore
struct TransferBatch<T: BlockMetadata> {
    containers: Vec<OffloadContainer<T>>,
}

impl<T: BlockMetadata> TransferBatch<T> {
    fn len(&self) -> usize {
        self.containers.iter().map(|c| c.evaluated_len()).sum()
    }

    fn sweep_cancelled(&mut self) -> usize {
        // Drop each cancelled container before upgrade.
    }

    fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }
}
```

### Data Transformations Per Stage

| Stage | Input | Output | Transform |
|-------|-------|--------|-----------|
| Enqueue | `SourceBlocks<T>` | `OffloadContainer<T>` | Add ID, state, precondition, and root work unit |
| PolicyEvaluator | `OffloadContainer<T>` | `OffloadContainer<T>` | Move passing source blocks into `evaluated_blocks` |
| PreconditionAwaiter | `OffloadContainer<T>` | `OffloadContainer<T>` | Await event or effective cancellation |
| BatchCollector | `OffloadContainer<T>` | `TransferBatch<T>` | Group complete containers without splitting them |
| TransferExecutor | `TransferBatch<T>` | `ResolvedBatch<T>` | Sweep, claim, upgrade, and retain work units |

---

## Resource-Owned Pipelines

`PipelineBuilder::resource` binds a block pipeline to one logical model resource.
The transfer executor passes that ID to `InstanceLeader` for physical route selection.

For G1-to-G2, the builder selects the configured resource or the leader primary resource.
It resolves the manager and capacity for that same resource.
If supplied capacity belongs to another manager, build fails.
This capacity controls destination admission for the selected resource.

Multi-resource runtimes construct one `OffloadEngine` per resource.
They install these engines through `build_local_connector_engine_with_resources`.
The connector records the resource on each buffered action.
It selects the matching engine at the forward-pass flush boundary.

Resource routing does not change the cancellation contract. Each selected
pipeline receives its own `OffloadContainer` and carries its work unit through
the same stages. The wrapper ends after the commitment claim. Resolved blocks
retain transfer state and guards until physical completion.

---

## Shared Pipeline Configuration and Runtime

`PipelineBaseConfig` owns all settings that block and object pipelines share.
These settings include policies, batch limits, timeouts, concurrency limits, and the pending tracker.

### Pending Hash Claims

`PolicyEvaluator` evaluates configured policies before it claims a pending hash.
`PendingTracker::try_claim` uses the `DashSet` insertion result as the atomic ownership decision.
Only the winner creates an `EvaluatedBlock` and retains a `PendingGuard`.
The guard remains live until the pipeline releases the block after failure or physical completion.

Presence policies call `is_pending` as an early filter only.
That observation does not prevent a concurrent policy pass.
The later `try_claim` call prevents duplicate work for every policy configuration.

A losing external or strong input records its actual `BlockId` as filtered.
A losing weak input has no source `BlockId`, so it records no synthetic value.
This rule keeps `SequenceHash` values unique in each object transfer batch.

`SharedPipelineConfig<Src, Options>` adds only destination-specific options.
`PipelineConfig` and `ObjectPipelineConfig` are public type aliases for this shared configuration.
Their builders use one `SharedPipelineBuilder` implementation.

`PipelineRuntime` owns the common stage queues, cancellation watchers, registration gate, and task handles.
Block and object pipelines add only their destination executor state.
The runtime uses one internal executor channel with capacity eight.

Engine drop publishes shutdown under the commitment gate.
It closes and drains every stage queue while it holds that gate.
The drain returns every queued container without failure or destruction.
After the gate releases, engine drop fails each container with `PRECOMMIT_SHUTDOWN_ERROR`.
Late producers receive their container back and fail it with the same error.
No callback, destructor, await, or transfer runs under the admission or commitment gate.
Precondition tasks stop before commitment, and queued precommit containers release their resources.
The collector fails its current batch and each batch that cannot reserve the executor channel.
The executor closes its batch receiver and starts a precommit drainer after shutdown.
The drainer fails each noncommitted batch, including a batch sent through a permit reserved before shutdown.
A committed batch waits for its transfer permit and completes its physical drain.
The executor waits for committed children and the precommit drainer before it exits.
The runtime does not abort the executor task.

Pipeline shutdown and commitment use one synchronous gate.
Runtime drop calls `send_replace(true)` while it holds the gate.
It then closes and drains every stage queue.
Each executor locks the gate only for the final sweep and weak-to-strong upgrade.
No async wait or physical transfer holds the gate.
If shutdown wins, the executor fails the untouched precommit batch after gate release and starts the drainer.
If upgrade wins, the batch commits before shutdown and finishes its physical drain.

---

## Token-Based Cancellation

### Token Lifecycle

1. **Creation**: Create one token and one root work unit.
2. **Distribution**: Give the token to the handle. Give the work unit to the container.
3. **Propagation**: Move the same work unit through each pipeline stage.
4. **Commitment**: Use one locked core to select cancellation or commitment.
5. **Fan-out**: Replace one chain unit with one unit for each configured route.
6. **Settlement**: Publish one terminal result after the last route settles.
7. **Confirmation**: Confirm cancellation after the terminal result and final unit release.

Before commitment, confirmation waits for the container drop. The drop releases source and pending guards.
After commitment, confirmation requires proven physical drain and final work-unit release.
An `Unproven` outcome retains ownership and never confirms cancellation.

The core has an `Open`, `Committed`, or `Cancelled` phase. A request changes only `Open` to `Cancelled`.
After commitment, later claims succeed even after a request. This rule prevents a partial chain.

`is_requested()` reports the user request. Pipeline stages use `is_precommit_cancelled()` for effective cancellation.

```rust,ignore
// At enqueue
let cancel_token = CancellationToken::new();
let cancellation = cancel_token.root_unit()?;

// Give to handle
let handle = TransferHandle { cancel_token: cancel_token.clone(), ... };

// Give to container
let container = OffloadContainer::with_cancellation(
    transfer_id,
    blocks,
    state,
    Some(event),
    cancellation,
);
```

### Internal Cancellation Core

`CancellationToken` and `CancellationUnit` are private implementation types.
`TransferHandle::cancel()` is the public cancellation entry point.

```rust,ignore
impl CancellationToken {
    /// Request cancellation (called by handle)
    fn request(&self);

    /// Check if cancellation requested
    fn is_requested(&self) -> bool;

    /// Await effective precommit cancellation
    async fn wait_precommit_cancelled(&self);

    /// Await confirmation that all blocks released
    fn wait_confirmed(&self) -> CancelConfirmation;
}

impl CancellationUnit {
    /// Claim the upgrade boundary against cancellation
    fn claim_commitment(&self) -> bool;

    /// Replace this unit with one unit for each route
    fn fan_out(self, route_count: usize) -> Vec<CancellationUnit>;

    /// Run the terminal action only when this is the last route unit
    fn settle(self, on_last: impl FnOnce());
}
```

### Logical Terminal Fan-In

One cancellation unit represents one active physical route.
The chain router replaces the source unit with the exact route count.
Each route records its progress or error before it settles its unit.
Each container binds its transfer state to its work unit.

An unsettled committed unit records a route error when it drops.
This drop path joins the same fan-in and prevents false success after a task panic.

`CancellationUnit::settle` decrements nonfinal units without a terminal result.
The final unit publishes `Failed` when any route recorded an error.
The final unit publishes `Complete` when all routes succeeded.

The final unit remains active during terminal publication.
Thus, cancellation confirmation cannot precede the terminal result.

Release source blocks and pending guards before route settlement.
Use `OffloadContainer::fail` for precondition errors and rejected precommit batches.
This method releases the complete payload before it publishes failure.

### PreconditionAwaiter Select Pattern

The awaiter uses `select!` for event completion and effective cancellation.
It tracks all tasks in one `JoinSet`.
The configured limit bounds the active waits and their retained containers.
The `JoinSet` aborts all active waits when the stage stops.

```rust,ignore
async fn process(&self, container: OffloadContainer<T>) {
    if let Some(event_handle) = container.precondition() {
        let awaiter = self.leader.messenger().events().awaiter(event_handle)?;
        tokio::select! {
            result = tokio::time::timeout(Duration::from_secs(300), awaiter) => result?,
            _ = container.cancellation_token().wait_precommit_cancelled() => {
                return;
            }
        }
    }
    if !container.is_cancelled() {
        self.output_queue.push(transfer_id, container);
    }
}
```

Set `max_concurrent_precondition_awaits` through either pipeline builder.
The default limit is eight waits.
A configured zero becomes one wait.

### CancellableQueue Sweep Mechanics

The queue supports active cancellation via sweeping:

```rust,ignore
impl<T> CancellableQueue<T> {
    /// Push an item, and reject an ID that is already cancelled.
    fn push(&self, transfer_id: TransferId, item: T) -> bool {
        if self.cancelled.contains(&transfer_id) {
            return false;
        }
        self.inner.push(QueueItem::new(transfer_id, item));
        true
    }

    /// Pop an item, and drop it when its ID is cancelled.
    fn pop_valid(&self) -> Option<QueueItem<T>> {
        loop {
            match self.inner.pop() {
                Some(item) if self.cancelled.contains(&item.transfer_id) => continue,
                other => return other,
            }
        }
    }

    /// Remove all cancelled items
    fn sweep(&self) -> usize {
        let mut removed = 0;
        let mut kept = Vec::new();

        while let Some(item) = self.inner.pop() {
            if self.cancelled.contains(&item.transfer_id) {
                removed += 1;
            } else {
                kept.push(item);
            }
        }

        let restored_work = !kept.is_empty();
        for item in kept {
            self.inner.push(item);
        }
        if restored_work {
            self.notify.notify_one();
        }
        removed
    }
}
```

The retained-work wake stores one availability permit. It covers a consumer
that observed an empty queue before it called `notified()`.

Sweep and close-and-drain use the same admission gate.
A sweep holds the gate until it restores retained work.
Close-and-drain waits for an active push or sweep.
It marks the queue closed and moves every queued payload into a returned vector.
After it releases the admission gate, it wakes consumers.
Runtime shutdown fails the returned payloads after it releases the commitment gate.
Later producers receive their payload back.

### Transfer ID Ownership

One pipeline accepts only one active registration for each `TransferId`.
A shared registration lock protects map updates and cancellation marker updates.
The pipeline rejects a duplicate ID while the first registration remains active.

Terminal cleanup clears queue and batch markers before it removes the map entry.
The same lock prevents a new registration during this cleanup.
This rule prevents an old watcher from clearing markers for later work.

`OffloadEngine` keeps only weak references to transfer state.
It prunes terminal and dead entries before each insert and active-count query.
Thus, the engine registry cannot retain a completed state.

### Cancellation at Each Stage

| Stage | Mechanism | Behavior |
|-------|-----------|----------|
| PolicyEvaluator | Token check | Check `is_cancelled()` between block evaluations |
| PreconditionAwaiter | Bounded `JoinSet` and `select!` | Drop only after effective precommit cancellation |
| BatchCollector | CancellableQueue | Sweep drops cancelled containers before batching |
| TransferExecutor | Final sweep and claim | Claim once, retain the unit, and finish physical work |

### Cancellation Boundary at Upgrade

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        CANCELLABLE ZONE                                 │
│                                                                         │
│  Enqueue → PolicyEval → PrecondAwaiter → BatchCollector → Executor      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                                                 ┌───────────────────┐
                                                 │  sweep_cancelled  │
                                                 │  (last sweep)     │
                                                 └───────────────────┘
                                                            │
                                                            ▼
═══════════════════════════════════════════════════════════════════════════
                         ATOMIC COMMITMENT BOUNDARY
═══════════════════════════════════════════════════════════════════════════
                                                            │
                                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        COMMITTED ZONE                                   │
│                                                                         │
│  Claim → Upgrade → Flat Map → Transfer                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Auto-Chain Ownership

The source executor moves one work unit into each `ChainOutput`.
The router atomically replaces that unit with the exact route count.
Each local route receives one unit with its `OffloadContainer`.
Each remote G4 request receives one unit and the shared transfer state.
The remote intake task starts one child drain task and waits for that task.
The child holds strong G2 blocks and its work unit through receipt drain.

`TransferCompleteNotification::await_drain()` reports one of three outcomes.
`Completed` proves successful physical drain.
`DrainedWithError` proves all launched work drained but reports a dispatch failure.
`Unproven` reports a completion error without drain proof.

The child releases guards and settles on `Completed`.
The child releases guards and fails the route on `DrainedWithError`.
It retains guards and the work unit on `Unproven`.
A synchronous error before every worker returns a receipt releases guards and fails the route.

Multi-worker dispatch returns one receipt if any worker returns a receipt.
The receipt directly owns all worker notifications and does not start an aggregate task.
A later dispatch error reaches the caller only after all returned notifications resolve.

A notification error, panic, or runtime abort removes the only drain proof.
The drop owner then retains the guards and work unit for the process lifetime.
The route stays nonterminal because worker upload can continue.

### G4 Mode Selection

Configure either the local G2-to-G4 pipeline or remote G4 offload.
The engine builder rejects a configuration that enables both modes.

The local mode requires `ObjectBlockOps` and `ObjectPipelineConfig<G2>`.
The executor passes `LogicalLayoutHandle::G2` to `ObjectBlockOps`.
That implementation resolves the source physical layout.
The engine builder does not accept a separate G2 physical layout.

Use remote mode when workers own the object upload path.
The leader sends committed G2 blocks through its worker group.

The source executor records source progress before it sends `ChainOutput`.
A fast downstream route therefore cannot publish a result before source progress.
No source or downstream executor publishes a terminal result directly.

A post-commit request does not mark downstream queues as cancelled.
The request changes cancellation state to `Draining` until all route units release.
No API can create a new root unit after commitment or confirmation.

---

## TransferExecutor Design

### Batch Commitment

Both executor types receive a `TransferBatch<T>` from the batch channel.
Each executor calls `upgrade_batch()` before it starts physical work.
That helper performs the final sweep, claims commitment, and creates one `ResolvedBatch<T>`.

```rust,ignore
while let Some(batch) = self.input_rx.recv().await {
    let mut resolved = upgrade_batch(batch);
    if resolved.is_empty() {
        resolved.settle_cancellation_units();
        continue;
    }

    Self::execute_transfer(&shared, &mut resolved).await?;
}
```

The executor gathers completion data before it releases the source guards.
It records route progress after this release.
It then settles each retained route unit.
Only the last logical route publishes `Complete` or `Failed`.

### Local Physical Ownership

`LocalPhysicalOwnership` owns all resources after local physical dispatch starts.
It owns source guards, pending guards, cancellation units, and the destination allocation.

The block executor arms the guard before the synchronous dispatch call.
A synchronous dispatch error restores the batch and releases the destination allocation.

The executor passes the guard to `TransferCompleteNotification::await_drain()`.
`Completed` restores all resources and permits destination registration.
`DrainedWithError` restores all resources before normal terminal failure.
`Unproven` records a diagnostic and retains all resources for the process lifetime.

The guard drop path applies the same retention rule after a panic or task abort.
The route stays nonterminal because physical work can still access its resources.

The object executor arms the guard before the first `put_blocks()` poll.
It releases the guard immediately after `put_blocks()` returns.
An abort retains the source guards and cancellation units for the process lifetime.

If the batch output channel closes, the collector recovers the rejected batch.
The collector fails each container after it releases all payload guards.

### Block Destination (G2, G3)

`BlockDestination::allocate()` returns one `DestinationAllocation`.
The allocation owns destination blocks and any capacity lease.
Registration consumes the allocation after physical drain completes.

```rust,ignore
let allocation = shared
    .destination
    .allocate(batch.blocks.len())?
    .ok_or_else(|| anyhow!("destination capacity is unavailable"))?;
let dst_block_ids = allocation.block_ids();
let ownership = LocalPhysicalOwnership::new(batch, allocation);

let dispatch = shared.leader.execute_local_transfer(
    shared.src_layout,
    shared.dst_layout,
    src_block_ids,
    dst_block_ids,
    TransferOptions::default(),
);
let notification = match dispatch {
    Ok(notification) => notification,
    Err(error) => {
        drop(ownership.release(batch));
        return Err(error);
    }
};

set_transferring_status();
let allocation = await_local_drain(batch, ownership, notification).await?;
let registered_blocks = allocation.register(&sequence_hashes)?;
```

### Object Destination (G4)

The object executor uses `ObjectBlockOps::put_blocks()`.
It has no destination allocation or destination registration step.

```rust,ignore
let keys = batch.blocks.iter().map(|block| block.sequence_hash).collect();
let block_ids = batch.blocks.iter().map(|block| block.block_id).collect();

set_transferring_status();
let ownership = LocalPhysicalOwnership::new(batch, ());
let results = shared
    .object_ops
    .put_blocks(keys, shared.src_layout, block_ids)
    .await;
ownership.release(batch);

record_object_results(results);
```

---

## BatchCollector Design

### Grouping Containers

The batcher accumulates containers and flushes when:

- Total blocks reach `max_batch_size`
- Flush interval expires and `min_batch_size` is met
- All blocks for a transfer have been processed (sentinel flush)

The collector owns the current batch, the cancellation watcher, and the shutdown watcher.
On stage shutdown, it fails queued containers and its current precommit batch.

On flush, it selects an executor-channel reservation and shutdown.
It checks shutdown again after it gets the reservation.
It fails the batch when shutdown wins or the channel closes.
The executor drainer handles a batch that arrives through a reservation acquired before shutdown.

### Preserving Per-Container Cancellability

Each container retains its own cancellation unit until the commitment claim.

1. The final sweep drops containers that already carry cancellation.
2. Each later claim selects cancellation or commitment under one gate.
3. The executor flat maps only containers that claim commitment.
4. The resolved batch retains each unit through physical completion.

---

## Extension Rules

### Adding a New Policy

1. Implement the `OffloadPolicy` trait.
2. Return `sync_result` or `async_result` from `evaluate`.
3. Add the policy to the pipeline configuration.

```rust,ignore
struct MyPolicy;

impl<T: BlockMetadata> OffloadPolicy<T> for MyPolicy {
    fn name(&self) -> &str {
        "my-policy"
    }

    fn evaluate<'a>(&'a self, _ctx: &'a EvalContext<T>) -> PolicyFuture<'a> {
        sync_result(Ok(true))
    }
}
```

### Adding a New Destination Type

Destination traits are private engine seams.

1. Add a destination adapter in `destination.rs`.
2. Implement `BlockDestination` and `DestinationAllocation` inside the engine.
3. Add destination-specific registration and cleanup.
4. Add a pipeline variant only when the shared runtime cannot support the destination.

### Maintaining Cancellation Invariants

When modifying the pipeline:

1. Never skip the upgrade boundary. It is the commitment point.
2. Always sweep and claim before an upgrade. The claim closes the sweep race.
3. Move the same work unit through every stage. Do not create continuation units from the token.
4. Use `fan_out()` once at the chain router. Give one child unit to each route.
5. Use effective cancellation for pipeline drops. Keep raw requests for handle reports.
6. Settle each committed unit exactly once. Let an abnormal drop publish a route failure.
7. Release source guards before terminal status. Release work units after terminal status.
8. If physical drain proof disappears, retain all physical resources and the work unit.
9. Arm local ownership before dispatch or before the first object future poll.
10. Track precondition tasks in one bounded `JoinSet`.
11. Remove terminal transfer states with an identity check.

---

## Testing Guidance

### Unit Tests

- Test each stage in isolation
- Use a real `CancellationToken` and `OffloadContainer` for cancellation cases
- Verify the final sweep and the commitment claim

### Integration Tests

- Test full pipeline with cancel at each stage
- Verify no orphaned blocks after cancellation
- Test partial batch cancellation

### Performance Tests

- Measure overhead of cancellation checks
- Benchmark sweep operation at scale
- Profile upgrade → flat map → transfer path
