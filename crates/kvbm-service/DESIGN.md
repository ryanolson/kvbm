# kvbm-service — design and status

## Project metadata

- **Linear**: <https://linear.app/nvidia/project/kvbm-service-25b4e5cd5c5f/overview>
- **Status**: Backlog
- **Lead**: unassigned
- **Start**: 2026-05-22

## Summary

A network-addressable wrapper around `kvbm-engine` that lets external
orchestrators drive the engine's leader/worker fan-out over RPC. The
service runs as its own process, owns the host-memory pool that will back
KV cache writes, and exposes a curated slice of the engine's leader as a
gRPC API. Tenants (vLLM/SGLang/etc.) connect over a Unix domain socket,
register their model topology, and — in the next milestone — attach
CUDA-IPC memory regions for the engine to write into.

## Why

The kvbm-engine leader + workers run in-process inside the inference
engine today. That couples lifecycle (a model crash takes the cache with
it), forces the cache topology to match the engine's TP/PP layout
exactly, and prevents multi-tenant sharing of host pinned memory.
Externalizing it behind a service gives us:

- **Lifecycle independence** — cache outlives a model restart; engine
  reconnect is a `Register` call.
- **Topology flexibility** — leader and workers live behind a
  parallelism abstraction; workers can be local PhysicalWorkers today or
  remote (over velo) tomorrow without changing the external API.
- **Multi-tenant ceiling** — slot accounting (`num_gpus / tp_size`)
  gates how many same-key registrations the host serves; first-key-wins
  per service.
- **Clean control/data split** — gRPC over UDS for control; axum
  sidecar over TCP for discovery, health, metrics; host pinned memory
  (managed by the service) for the data plane.

The proto API is shaped after pegaflow's `Engine` service — same
server-streaming liveness/attach pattern, same stream-drop-equals-
ungraceful-detach contract.

## Key design decisions

- **First-key-wins single-tenancy** — for MVP only. The state-machine
  slot for multiple concurrent unrelated tenants is
  `Empty | SingleKey | Draining`. Moving to multi-tenant is a state-
  machine widening, not an API change.
- **Stream events are protocol-ordered** — `Accepted` is always the
  first event; `ServerShutdownInitiated` (and any future event) is
  strictly after. The two-phase commit pattern enforces this — the
  lifecycle isn't reachable to the shutdown broadcaster until commit,
  which happens after `Accepted` is enqueued.
- **Grace period as data-correctness guarantee** — the policy "min 60s,
  second signal ignored" is load-bearing. Don't relax it without a
  corresponding change to the engine's write-cancellation guarantees.
- **`ServiceContainer` is the engine boundary** — the engine work lives
  entirely inside one container impl; the shell stays engine-agnostic.
  Other future containers (test harnesses, alternative cache backends)
  plug in the same way.

## Architecture

```
                     ┌──────────────────────────┐
                     │  KvbmService (shell)     │
                     │  ─ Registry              │   gRPC over UDS
                     │  ─ axum HTTP sidecar     │   HTTP over TCP
                     │  ─ Lifecycle / shutdown  │
                     └─────────┬────────────────┘
                               │ owns
                     ┌─────────▼───────────┐
                     │  HostMemoryPool     │   one per service process
                     │  (per-host alloc)   │
                     └─┬────┬────┬────┬────┘
                       │    │    │    │
              NodeSlab │    │    │    │   one per host-CPU NUMA node
              ─ MmappedPinnedStorage   (mmap + mbind + cuMemHostRegister)
              ─ NixlAgent              (one per slab — local or registered)
              ─ NixlRegistered handle  (RAII deregister, when registered)
              ─ HugepageTier metadata  (Explicit / Thp / None)
                               │ lease_all()
                     ┌─────────▼───────────┐
                     │  ServiceContainer   │
                     │  (Noop today;       │
                     │   kvbm-engine next) │
                     └─────────────────────┘
```

## Milestones

### M1 — Resource Discovery and Allocation

> At startup, allocate a configured amount of host pinned memory, split
> evenly across discovered NUMA nodes. Allocate from hugepages if
> possible. Register host memory with NIXL.

**Status: implemented, GB200 hardware-verified.**

Delivered:

- **Topology discovery (`dynamo-memory`)**
  - `NumaNodeRole` ∈ {`HostCpu`, `GpuMemory`, `Reserved`} on every
    `NumaNodeView`; `Resources::host_memory_nodes()` returns only
    `HostCpu` — the pool's allocation targets.
  - Per-node `total_bytes` parsed from
    `/sys/devices/system/node/node{N}/meminfo`.
  - cgroup v2 `cpuset.mems.effective` (kernel-enforced mask after
    ancestor + online-node intersection); pool prefers effective over
    configured.
  - v2 reader path order fixed — descended cgroup path wins over root,
    matching v1.
  - `inspect_resources` binary surfaces all of this with a `Host-memory
    pool view` summary and a collapsed `CPU-less nodes [N-M]` line for
    Grace/GB200's long-tail MIG slots.

- **Hugepage discovery (`dynamo-memory::hugepage`)**
  - System-wide and per-node pool stats from
    `/proc/meminfo`, `/sys/kernel/mm/hugepages/`, and per-node sysfs.
  - THP mode from `/sys/kernel/mm/transparent_hugepage/enabled`.
  - Folded into `Resources::Display`.

- **Allocator (`dynamo-memory::mmap_pinned`)**
  - `MmappedPinnedStorage`: `mmap(MAP_PRIVATE|MAP_ANON [|MAP_HUGETLB
    |MAP_HUGE_<size>])` → `mbind(MPOL_BIND, MPOL_MF_STRICT)` →
    parallel first-touch → `cuMemHostRegister(DEVICEMAP)`.
  - `HugepageMode::{Disabled, BestEffort, Required}` with tier ladder:
    explicit hugetlb → THP (`madvise(MADV_HUGEPAGE)`) → plain anon.
  - Generic `MAP_HUGE_*` encoding via `log2(page_size) <<
    MAP_HUGE_SHIFT` — covers x86 2 MiB / 1 GiB and Grace 64K-page-
    kernel's 512 MiB.
  - Drop order enforced by field declaration order: NIXL handle →
    `cuMemHostUnregister` → `munmap`; agent outlives the handle.
  - `NumaWorkerPool::allocate_mmap_pinned_on_node` — one-shot worker
    thread pinned to the target node performs the allocation; the
    existing per-GPU CUDA-path worker pool is untouched.

- **Pool (`kvbm-service::pool`)**
  - `HostMemoryPool` allocates one `NodeSlab` per host-CPU NUMA node.
  - `NodeSlab` carries `SlabStorage::Registered(NixlRegistered<…>)` (UCX
    path) or `SlabStorage::Local(MmappedPinnedStorage)` (no-NIXL path),
    plus optional `NixlAgent`.
  - `PoolConfig` + fluent `PoolConfigBuilder`:
    - `.sizing(...)` / `.per_node_bytes(...)` / `.ratio(...)` — sizing
      policy. `PoolSizing` ∈ {`Ratio(0.85 default)`, `Total`, `PerNode`,
      `Explicit`}; `Total` splits proportionally by node `MemTotal`.
    - `.hugepage_mode(...)` / `.hugepage_size_bytes(...)` —
      hugepage policy.
    - `.nixl_backend("UCX")` / `.with_ucx()` / `.nixl_backends(...)` —
      NIXL backend list (default `["UCX"]`).
    - `.local_only()` — clears backends *and* sets
      `allow_no_nixl_backends=true` atomically.
  - cgroup-aware: `nodes_allowed_by_cgroup` filters by effective mems,
    `cap_by_cgroup_memory` scales the pool proportionally to
    `memory.max`. Refuses to start if cpuset excludes every host node.
  - Warnings: `Ratio > 0.95`, per-node request > 90% of `MemTotal`,
    pool < 50% of host memory (explicit sizing only).
  - Backend resolution precedence (`resolve_backend_config`):
    1. Explicit list — wins unconditionally.
    2. `allow_no_nixl_backends=true` — empty config, env *not* consulted.
    3. Legacy env fallback — `DYN_KVBM_NIXL_BACKEND_*` via a closure
       so a malformed env var doesn't abort startup on paths that
       don't use it.
  - `PoolLease` is the single-tenant handle the container holds; `Drop`
    returns it to the pool.

- **Service wiring**
  - `KvbmService::start_with_pool(cfg, container)` constructs the pool,
    takes the lease, hands it to the container via
    `ServiceContainer::on_resources_attached` (default impl drops
    it — appropriate for `NoopContainer`).
  - `KvbmService::start` / `start_with_container` stay shell-only
    (pool = `None`) for tests.
  - `kvbm_service` binary always uses `start_with_pool`.

- **Surfacing**
  - `GET /v1/pool` → `PoolSnapshot` JSON: per-slab `{numa_node,
    size_bytes, hugepage_tier, agent_name, registered}` + hugepage
    defaults + lease state. Returns 503 when no pool was constructed.
  - Prometheus gauges:
    `kvbm_service_pool_bytes_total{node, tier}`,
    `kvbm_service_pool_slabs_total{tier}`.

- **Docs** — `docs/hugepages.md` (operator-facing): why hugepages, tier
  table, host + container reservation recipes, Grace/GB200 specifics
  (`init_on_alloc=0`, `kernel.numa_balancing=0`, mTHP requirements),
  observability cheatsheet, troubleshooting.

**Test coverage**
- 179 `dynamo-memory` lib tests (hugepage parsing, role
  classification, cgroup reader, `mmap_pinned` tier decisions, etc.).
- 86 `kvbm-service` lib tests (29 pool-specific: sizing, cgroup
  filtering/capping, backend resolution policy, lease semantics,
  builder).
- 4 `testing-cuda`-gated integration tests under
  `tests/pool_alloc.rs`: per-host-node slab count + size, `/v1/pool`
  HTTP shape, 503-without-pool, `.local_only()` no-NIXL path.

**Hardware verification (4-GPU GB200, Grace + Blackwell tray)**
- `inspect_resources` reports `host-memory nodes: 0, 1` and
  `total host memory: 882.8 GiB` (~441 GiB LPDDR5X per Grace).
- 4 GPUs PCI-attached to nodes 0 and 1 directly (this box's BIOS does
  not expose HBM as separate NUMA nodes — the NVL72 idealized layout
  doesn't apply here, but role classification is correct either way).
- Default `Hugepagesize` is 512 MiB (64K-page kernel) — generic
  `MAP_HUGE_*` encoding handles it.
- All 4 CUDA-gated integration tests pass; with UCX in the default
  backend list, slabs land as `SlabStorage::Registered` and the C++
  `registerMem: no available backends` log spam disappears.

### M2 — Create KVBM Engine the Service's Instance Container

> ServiceContainer is the engine boundary — the engine work lives
> entirely inside one container impl. The ResourceManager provides each
> ServiceContainer with a subset of the resources. For our MVP, there
> will be only a single ServiceContainer per service.

**Status: not started.** The trait + lease API are wired
(`ServiceContainer::on_resources_attached`); the kvbm-engine container
itself is the next milestone.

Sketch of what M2 brings (subject to revision when planning starts):

- An `EngineLeader` with G2/G3 block managers.
- `PhysicalLayouts` per G2 / G3:
  - **Operational** layout — slice each G2 memory by device count,
    produce `N × TP_SIZE` G2 layouts; refuse registration if TP isn't
    evenly divisible by GPU count.
  - **Universal** layout — treat host memory as if the TP dimension
    were `num_numa_nodes`; allocate `num_numa_nodes` G2/G3 layouts and
    permute via existing kernels for arbitrary attached TP.
- Each `ServiceContainer` can have multiple attached clients; when all
  clients detach, the container is destroyed and resources return to
  the pool.

## Module map

**Modified in `dynamo-memory` (`lib/memory/`):**
- `src/lib.rs` — register new modules, export new types
- `src/resources/mod.rs` — `NumaNodeRole`, `total_bytes`,
  `host_memory_nodes`, `total_host_memory_bytes`,
  `cpuset_mems_effective`, role-aware Display
- `src/numa/topology.rs` — `NumaTopology::from_node_cpus` (test-only)
- `src/numa/worker_pool.rs` — `allocate_mmap_pinned_on_node`
- `bin/inspect_resources.rs` — surfaces all of the above

**New in `dynamo-memory`:**
- `src/hugepage.rs` — read-only sysfs discovery
- `src/mmap_pinned.rs` — `MmappedPinnedStorage`, `HugepageMode`,
  `HugepageTier`, `MmappedPinnedOptions`

**New / modified in `kvbm-service` (`lib/kvbm-service/`):**
- `src/pool/mod.rs` — `HostMemoryPool`, `PoolLease`, `PoolSnapshot`,
  `resolve_backend_config`, cgroup helpers
- `src/pool/slab.rs` — `NodeSlab`, `SlabStorage`, `NodeSlabSnapshot`
- `src/pool/config.rs` — `PoolConfig`, `PoolSizing`,
  `PoolConfigBuilder`
- `src/server/{mod,http}.rs` — `start_with_pool`, `/v1/pool`,
  pool gauges
- `src/container/mod.rs` — `on_resources_attached` hook
- `src/config/mod.rs` — `pool: PoolConfig` field
- `src/metrics.rs` — pool gauges
- `src/bin/kvbm_service.rs` — uses `start_with_pool`
- `docs/hugepages.md` — operator docs
- `tests/pool_alloc.rs` — CUDA-gated end-to-end

## Configuration surface

```toml
[pool]
hugepage_mode = "BestEffort"      # Disabled | BestEffort | Required
hugepage_size_bytes = 2097152     # optional override; default = system Hugepagesize
ctx_device_id = 0                 # any CUDA-visible GPU
validate_placement = false        # move_pages(2) sample after alloc; debug only
allow_no_nixl_backends = false    # opt out of UCX requirement (local-only)
backends = ["UCX"]                # NIXL backends; default = ["UCX"]

[pool.sizing]
Ratio = 0.85                      # default — fraction of each host node's MemTotal
# OR:
# Total = { bytes = 137438953472 }              # split proportionally to per-node MemTotal
# PerNode = { bytes_per_node = 68719476736 }    # fixed per node
# Explicit = { "0" = 200_000_000_000, "1" = 100_000_000_000 }
```

Env-var equivalents follow `KVBM_SERVICE_POOL__*` (figment double-
underscore for nested). `DYN_KVBM_NIXL_BACKEND_<NAME>=true` is honored
only when `backends` is empty *and* `allow_no_nixl_backends = false`.

## Known follow-ups

- M2: kvbm-engine container — picks up the `PoolLease` and builds
  G2/G3 layouts.
- Per-node `MemFree` warning (currently we only warn against
  `MemTotal`).
- Multi-tenant `PoolState` (currently `Empty | Leased`; will widen to
  the same `Empty | SingleKey | Draining` shape the `Registry` uses).
- 1 GiB hugepage support (the generic `MAP_HUGE_*` encoding already
  handles arbitrary power-of-two sizes; only the operator docs need
  to surface the 1 GiB option once a workload needs it).
