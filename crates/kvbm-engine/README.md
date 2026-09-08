# kvbm-engine

Distributed coordination primitives for KV cache block management (KVBM).

This crate implements the leader/worker architecture for managing KV cache blocks across a tiered storage hierarchy:

**G1** (GPU HBM) → **G2** (Pinned DRAM) → **G3** (NVMe/SSD) → **G4** (S3/MinIO)

Leaders own block metadata and make placement decisions. Workers execute data transfers (RDMA, NVMe, object storage). Sessions coordinate multi-instance block transfers.

## G1 source staging for sessions

The crate-private `PolicyG1G2BoundRoute::stage_to_g2` method copies caller-owned G1 pins into temporary G2 blocks and returns their registered pins. A caller outside the crate reaches it through an installed `G1SessionSource`. The certified route fixes the resource, G1 manager, G2 capacity, and workers. The session publisher owns commitments, checksums, and availability.

The method checks source manager identity, unique hashes, and equal logical block sizes before dispatch. It reuses G2 matches and reserves only missing blocks through `G2Capacity`. An independent runtime task owns source pins and destination capacity until physical completion, so a copy never occupies the blocking pool. A dropped future cancels publication, not DMA. An uncertain completion or dispatch panic retains the affected memory and capacity. A runtime shutdown drops the task at its await point and retains the same memory and capacity. A copy that stopped mid-await has no drain proof.

The method marks new blocks as temporary before registration. A collision does not change the retention of an existing primary. The offload policy can adopt a temporary primary with `ImmutableBlock::set_evict_on_reset(false)`. Otherwise, the last session or external pin release returns it to the free pool. Transport does not evict G1.

A caller installs resource-bound `G1SessionSource` handles on the holder. The holder keeps weak references, and the logical manager owns the handles. A source exposes only registered blocks, which must represent completed source writes. The logical owner can disable lookup and does not cancel a copy that already holds pins. A physical resource keeps its original logical binding after source retirement.

Holder search combines G1 and G2 hits in request order. Prefix search stops at the first cross-tier gap. Scatter search can also include G3. The publisher sends one batch per tier as that tier lands. The resident G2 hits publish first, then the staged device blocks, then the staged disk blocks. The publisher assigns each checksum the ordinal of its hash in the complete committed set.

Rhino installs these sources independently of proactive mirroring. Hub discovery and GLM fixed-state transport remain separate tasks in Rhino's `agent-docs/kvbm-transfer-handoff.md`.

## Feature Flags


| Flag           | Purpose                                  |
| -------------- | ---------------------------------------- |
| `s3` (default) | S3/MinIO object storage (G4 tier)        |
| `testing`      | Test utilities and mock infrastructure   |
| `nats`         | NATS-based pub/sub transport             |
| `collectives`  | NIXL + NCCL multi-GPU collectives        |
| `nccl`         | NCCL via cudarc                          |
| `nvtx`         | NVIDIA Tools Extension profiling markers |


## Documentation

Detailed module documentation lives in `[docs/](docs/)`:

- [Architecture](docs/architecture.md) — Overall system design
- [Leader](docs/leader.md) — Block coordination and metadata management
- [Session](docs/session.md) — Distributed onboarding protocol
- [Worker](docs/worker.md) — Transfer execution
- [Worker Group](docs/worker-group.md) — SPMD parallel workers
- [Offload](docs/offload.md) — Async tier-demotion pipeline
- [Offload Developer Guide](docs/offload-developer.md) — Contributing to the offload module
- [Object Storage](docs/object.md) — S3/MinIO integration
- [Runtime](docs/runtime.md) — Runtime bundle (tokio, Velo, NIXL)
- [Testing](docs/testing.md) — Test utilities and fixtures

