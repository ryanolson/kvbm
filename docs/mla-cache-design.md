# MLA cache ownership and placement

This document records the incremental design for making KVBM the logical and
physical owner of Rhino's MLA KV cache. Narwhal decides which logical blocks are
needed; KVBM owns their identities, lifetimes, tier placement, and transfers;
Rhino supplies model tensors and executes attention.

## Identity is separate from placement

`SequenceHash` is the stable content identity used for matching and deduplication.
It should not encode a worker rank or memory tier. A `BlockId` is an allocation
handle within a logical tier. Physical placement translates that logical ID into
a worker and a worker-local slot.

For replicated MLA with tensor-parallel size `W`, G1 is replicated and G2 is
striped:

```text
global G2 id = local slot * W + owner rank
owner rank   = global G2 id % W
local slot   = global G2 id / W
```

For TP=2, global G2 IDs `0, 1, 2, 3, 4` map to:

| Global G2 ID | Owner | Local G2 slot |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 0 | 1 |
| 3 | 1 | 1 |
| 4 | 0 | 2 |

Each process still allocates only its configured local G2 capacity. The logical
G2 `BlockManager` exposes the aggregate capacity, so TP=2 has twice the deduplicated
lower-tier capacity of one worker. G1 allocation must assign the same logical
destination block IDs on every replica.

## Transfer bookkeeping

Offload takes paired `(replicated G1 ID, global G2 ID)` values. Each rank keeps
only pairs whose G2 owner is that rank, translates the global G2 ID to its local
slot, and performs those copies. Other G1 replicas do no work for those blocks.

Onboard groups `(global G2 ID, replicated G1 ID)` pairs by owner. For each owner,
in rank order:

1. The owner copies its local G2 slots into its G1 destination IDs.
2. Every rank enters the same broadcast with that owner as the root.
3. The next owner batch starts only after the prior batch completes.

The rank ordering is deliberately conservative. It prevents mismatched collective
call order across processes. Pipelining can be added after the contract is covered
by multi-process tests.

`CollectiveOps` is the transport seam. NCCL is the first practical implementation.
NIXL can implement the same contract. Direct `cudaMemcpyAsync` across worker
processes requires CUDA IPC/peer access plus explicit handle and lifetime
management; it is not an ordinary same-process device-to-device copy.

The connector uses a KVBM-owned NCCL communicator rather than borrowing vLLM's
model-parallel communicator. This isolates collective ordering: attention
all-reduces cannot interleave differently with cache broadcasts across ranks.
The leader generates one NCCL ID and distributes it with worker initialization.

## Current implementation boundary

Implemented and tested:

- TP=1 DeepSeek MLA registration with `[Block, Page, HeadSize]` and no head axis.
- TP=1 G1↔G2 offload/onboard using `v2ray/DeepSeek-V3-1B-Test`.
- Global-to-striped lower-tier placement and aggregate logical capacity.
- Owner-only offload planning and dynamic-root onboard planning.
- A replicated transfer policy used by both RPC and intra-pass worker-engine transfers.
- Leader-distributed, KVBM-owned NCCL bootstrap with a dedicated communicator
  and CUDA stream on every worker.
- The replicated policy is a complete KVBM `Worker`, so in-process runtimes
  such as Rhino can install the same owner routing directly in
  `InstanceLeader`. KVBM can initialize an ordered in-process worker group
  concurrently because `ncclCommInitRank` is collective.
- Logical managers, physical layouts, placement metadata, transfer sessions,
  and proactive G1-to-G2 pipelines are keyed by logical resource. One worker
  group can therefore combine tensor-sharded attention with replicated MLA.
- Narwhal preserves model resource and group identity, submits proactive
  mirrors to the matching KVBM pipeline, and acquires resource-specific G2
  RAII pins for `Mirror` and `Move` blocks while allowing `Drop` blocks to be
  released without a lower-tier copy.
- Same-request restoration preserves each retained block's logical resource,
  logical position, and exact pinned G2 source ID. After Narwhal allocates new
  G1 pages, one request-scoped KVBM action dispatches the corresponding
  G2-to-G1 transfers through every resource's own physical placement policy.

Not yet production-wired:

- A TP=2 multi-process MLA end-to-end test exists, but has not yet been run on
  a two-GPU host.
- Replicated remote search has owner-routed G2→G2 planning and dispatch, but
  still needs a real two-instance, two-GPU NIXL validation.
- Replicated G2↔G3 placement.
- Multi-resource external-prefix discovery. Same-request restoration is
  resource-complete, while a new request still discovers and onboards only the
  selected primary resource's shared prefix.
- Pipelined per-layer replicated onboard; the first implementation completes
  replicated onboard before model execution for correctness.

Until the two-GPU tests run, `ReplicatedData` should be described as wired and
unit/integration tested, but not hardware-validated for TP>1 production use.

## Progression to hybrid MLA

The implementation order is:

1. Basic TP=1 MLA: detect MLA, register the latent-cache dimensions, and prove
   local tier movement.
2. Basic TP>1 MLA: replicated G1, striped/deduplicated G2, owner copies, and
   broadcast reload.
3. Hybrid models: classify cache regions by attention and retention behavior,
   then compose managers without erasing those semantics. Resource-owned local
   transfer, proactive offload, and same-request multi-resource restoration are
   implemented; external-prefix discovery across resources and live DSV4 TP/EP
   validation remain.

Hybrid support is expressed as typed resources rather than one generic pool
with flags. Model registration produces a cache schema whose regions retain
their layout and replication/sharding mode. Narwhal schedules model-ordered
resource groups while KVBM keeps RAII ownership of each resource's blocks.

## Sliding-window retention TODO

Eviction and offload are different decisions. A G1 block can be:

- **mirrored**: already present in G2, so G1 eviction requires no copy;
- **moved**: worth retaining, but a G2 copy must be created before releasing G1;
- **dropped**: outside the model's useful window and safe to recompute or discard.

Narwhal now emits explicit `Mirror`, `Move`, or `Drop` actions from each
attention policy. Full attention never drops; sliding-window attention drops
blocks fully outside its useful window. The proactive mirror target is
configurable and transfer execution remains separate. Remaining policy work is
to tune that target from live pressure/reuse signals and extend the same
resource semantics to lower tiers beyond G2.

## Remote-search direction

Remote search keeps requester-driven pull/RDMA GET semantics. The holder
session supplies each hash's global source block ID, so deterministic striping
resolves the physical owner without indexing hashes by worker rank. A future
push/PUT session remains useful for opaque or topology-selected placement, but
it requires destination leases and a larger completion protocol.

See [Remote transfers for replicated MLA caches](remote-mla-transfer-design.md)
for the owner-routing and pull-versus-push decision.
