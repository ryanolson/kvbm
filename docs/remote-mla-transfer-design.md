# Remote transfers for replicated MLA caches

## Decision

Use owner-routed requester pulls first. Keep holder-driven RDMA writes as a
future transport option, not as a prerequisite for correct replicated MLA.

The remote index remains keyed by `SequenceHash -> InstanceId`. It does not
need a second `SequenceHash -> worker rank` table. A holder session already
returns the global G2 block ID associated with each requested hash, and the
requester allocates a global destination G2 block ID. Given each worker-group
size, `StripedBlockPlacement` deterministically resolves both physical owners:

```text
source global G2 ID      destination global G2 ID
          |                         |
          v                         v
 remote owner/local ID       local owner/local ID
          |                         |
          +------ NIXL READ --------+
```

This keeps hash identity in the logical/session layer and block placement in
the physical layer.

## Pull protocol

Each exported worker layout advertises its data placement. The replicated MLA
value means:

- every worker has the same logical G1 data;
- every logical G2/G3 block has exactly one striped owner;
- remote transfer planning operates on whole latent blocks, not tensor-axis
  slices.

For each `(remote global source ID, local global destination ID)` pair, the
requester:

1. resolves the remote source owner and worker-local source ID;
2. resolves the local destination owner and worker-local destination ID;
3. groups pairs by `(local owner, remote owner)`;
4. dispatches one full-block NIXL read batch to each participating local
   owner;
5. waits for all batches before publishing the destination blocks.

Remote pulls land once in striped local G2. They do not broadcast directly
into G1. A later local onboard uses the existing owner G2-to-G1 copy followed
by a KVBM collective broadcast to the other G1 replicas.

The session RAII contract remains unchanged: the holder keeps immutable source
blocks pinned until acknowledgement, and the requester keeps mutable
destination blocks private until transfer completion and registration.

## Why not push first

A holder-driven WRITE/PUT/PUSH protocol is useful when placement is opaque,
topology policy must choose destinations dynamically, or NIXL write semantics
are materially better than reads. It also requires a larger authority and
lifetime protocol. The requester must issue destination leases containing at
least the resource, global block ID, owner rank, worker-local ID, and a
capability tied to the live memory registration. The holder then needs
idempotent completion, retry, and notification semantics.

Deterministic striped placement already solves owner discovery without those
costs, so push is deferred until measurements or a non-deterministic placement
strategy justify it.

## Hybrid extension

Placement is a property of a cache resource, not of an entire model. A hybrid
model may combine tensor-sharded attention resources with replicated MLA
resources. The current worker-group placement marker establishes the routing
contract; the next schema revision should attach the same contract to each
typed resource produced by model registration.

Retention is independent of transport. Sliding-window policy chooses
`Mirror`, `Move`, or `Drop`; owner-routed pull and local owner+broadcast execute
the movement selected for replicated resources.

## Validation boundary

Pure placement, wire compatibility, placement-consistency, and dispatch tests
cover the implementation. A real two-instance, two-GPU NIXL test remains
required before calling replicated remote search production-ready.
