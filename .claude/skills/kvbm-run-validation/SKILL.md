---
name: kvbm-run-validation
description: Run KVBM accuracy/determinism validation tests against the local .sandbox pytest venv
user-invocable: true
disable-model-invocation: true
---

# Run KVBM Validation Tests

Run KVBM integration tests to validate accuracy, determinism, and correctness.

This skill runs **local** mode: pytest against the `.sandbox/` venv. Fast iteration
loop. Requires `/kvbm-sandbox-venv` + `/kvbm-maturin-dev` first.

For the faster three-shell local iteration flow (deps / server / eval — skip this wrapper), see `/kvbm-decomposed-run`.

> Container mode (hermetic, CI-parity) is **not supported in the standalone `kvbm`
> repo**: it depended on a pre-built dynamo vllm image that this repo no longer
> produces. The recipe is preserved at the end under
> [Container mode (deferred)](#container-mode-deferred) for when an external image
> is available.

## Arguments

`/kvbm-run-validation [scope] [--spec SPEC_ID] [--fast] [--enable-mla]`

- **scope** (default: `quick`):
  - `quick` — Pre-merge marker (`-m "kvbm and pre_merge"`). 1 GPU. ~5 min.
  - `mla-smoke` — Small TP=1 MLA registration and G1↔G2 round trip. 1 GPU. ~1 min after download.
  - `mla-tp2` — Replicated MLA G1 with striped G2 and NCCL onboard broadcast. 2 GPUs. ~1 min after download.
  - `agg-intra` — connector intra-onboard determinism only. ~15 min.
  - `agg-inter` — connector inter-onboard determinism only. ~15 min.
  - `agg` — all connector (intra + inter) determinism specs. ~45 min.
  - `disagg` — `test_determinism_disagg.py`. 2 GPUs. ~15 min.
  - `full` — All KVBM tests (`-m "kvbm or kvbm_concurrency"`). ~30+ min.
  - `<filename>` — Run a specific file (e.g. `test_chunked_prefill.py`).
- **--spec SPEC_ID** — Run a single parametrization by id (e.g. `Qwen3-0.6B-intra`). Maps to `-k $SPEC_ID`. Overrides scope marker filters.
- **--fast** — `KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2`.
- **--enable-mla** — Set `KVBM_ENABLE_MLA=1` to unlock `DeepSeek-V2-Lite` specs.

## Step 0: Verify Local Prerequisites

```bash
# Local mode viable?
LOCAL_OK=no
if [ -x .sandbox/bin/python ] && .sandbox/bin/python -c "import kvbm" 2>/dev/null; then
    LOCAL_OK=yes
fi
```

If `LOCAL_OK=no`, **stop** and tell the user:
`run /kvbm-sandbox-venv + /kvbm-maturin-dev, then retry`.

## Step 1: Map Scope To Pytest Args

Spec ids are read from `_CACHE_RESET_SPECS` in `tests/kvbm_integration/test_determinism_agg.py`. Authoritative list via:

```bash
.sandbox/bin/python - <<'PY'
from tests.kvbm_integration.test_determinism_agg import _CACHE_RESET_SPECS
for s in _CACHE_RESET_SPECS: print(s.id)
PY
```

| Scope | Pytest args | GPUs | Est. time |
|---|---|---|---|
| `quick` | `tests/kvbm_integration/ --continue-on-collection-errors -m "kvbm and pre_merge"` | 1 | ~5 min |
| `mla-smoke` | `tests/kvbm_integration/test_mla_smoke.py` | 1 | ~1 min after download |
| `mla-tp2` | `tests/kvbm_integration/test_mla_tp2.py` | 2 | ~1 min after download |
| `agg-intra` | `tests/kvbm_integration/test_determinism_agg.py -k "intra"` | 1 | ~15 min / ~3 min |
| `agg-inter` | `tests/kvbm_integration/test_determinism_agg.py -k "inter"` | 1 | ~15 min / ~3 min |
| `agg` | `tests/kvbm_integration/test_determinism_agg.py` | 1 | ~45 min / ~10 min |
| `disagg` | `tests/kvbm_integration/test_determinism_disagg.py` | 2 | ~15 min |
| `full` | `tests/kvbm_integration/ --continue-on-collection-errors -m "kvbm or kvbm_concurrency"` | 1-2 | ~30+ min |
| `<file>` | `tests/kvbm_integration/<file>` | varies | varies |

If `--spec SPEC_ID` was provided, use `-k $SPEC_ID` instead of the scope filter.

**Gotcha**: spec ids containing `Qwen3-0.6B` only exist when `KVBM_MODEL_ID=Qwen/Qwen3-0.6B` is set. The default first model is `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.

## Step 2: Show Plan And Confirm

```
KVBM Validation Plan
────────────────────
Scope:     <scope>
Spec:      <spec-id or "scope filter">
Model:     <KVBM_MODEL_ID or "per-spec default">
MLA gate:  <enabled|disabled>
Fast mode: <yes|no>
GPUs:      <count>
Est. time: <estimate>

Command:
  <full command preview>
```

For `agg` without `--fast`, suggest `--fast` and offer the decomposed flow:

> Tip: `--fast` drops this to <5 min. For the three-shell iteration loop (iterate eval without re-spawning vllm), use `/kvbm-decomposed-run <spec-id> --fast` instead.

Confirm before proceeding.

## Step 3: Build Environment Variables

Always:
```
RUST_BACKTRACE=1
```

Conditional:
```
# --fast
KVBM_MAX_ITERATIONS=2
KVBM_NUM_ITERATIONS=2
KVBM_REQUEST_DELAY=2

# --enable-mla
KVBM_ENABLE_MLA=1

# KVBM_MODEL_ID — if the caller wants a Qwen spec id, this MUST be set
KVBM_MODEL_ID=Qwen/Qwen3-0.6B
```

On GB10, the reference knobs for Qwen3-0.6B:
```
KVBM_CPU_BLOCKS=2000
KVBM_GPU_BLOCKS=512
KVBM_GPU_MEMORY_UTILIZATION=0.5
KVBM_SERVER_START_TIMEOUT=600
```

## Step 4: Run Tests

Timeouts:
- `quick`: 600s
- `agg-*` fast: 600s
- `agg-*` full: 1800s (more for 8B specs — leave the per-test timeout at the computed value in `test_determinism_agg.py`)
- `disagg`: 1800s
- `full`: 3600s
- single file/spec: 900s

```bash
PATH="$(pwd)/.sandbox/bin:$PATH" RUST_BACKTRACE=1 <env_vars> \
    timeout <timeout> .sandbox/bin/python -m pytest <pytest_args> -v --tb=short -s
```

The `PATH` prefix is required because the integration fixture launches the
`vllm` console script as a subprocess.

Tests reuse host NATS/etcd on defaults (4222 / 2379) if reachable (aligned with `conftest.py:runtime_services`). Otherwise fixtures spawn them.

## Step 5: Report Results

```
KVBM Validation Results
───────────────────────
Passed:    X
Failed:    Y
Skipped:   Z
Errors:    W
Duration:  Nm Ns
```

If failures, show `--tb=short` output and point at `/kvbm-diagnose`:

```
For per-test log analysis and live /metrics inspection:
  /kvbm-diagnose
```

Stack health quick-check patterns (grep the newest per-test log under `/tmp/dynamo_tests/`):
- `KvConnectorWorker initialized` — worker bootstrapped
- `Auto-detected device layout` — tensor layout OK
- `ConnectorLeader initialized with onboard mode onboard_mode=Intra|Inter` — connector leader + mode
- `Application startup complete` — vllm ready
- `kvbm_offload_blocks_d2h > 0` in `/metrics` — offload active

## Reference: Test Files

| File | Tests | Markers | GPUs |
|---|---|---|---|
| `test_kvbm.py` | offload_and_onboard, gpu_cache_eviction, onboarding_determinism | kvbm, e2e, gpu_1, vllm, pre_merge | 1 |
| `test_chunked_prefill.py` | chunked prefill offload | kvbm, e2e, gpu_1, vllm, pre_merge | 1 |
| `test_kvbm_vllm_integration.py` | vLLM interface assumptions | kvbm, integration, gpu_0, vllm, nightly, pre_merge | 0 |
| `test_consolidator_router_e2e.py` | consolidator + router E2E | kvbm, e2e, slow, gpu_1, pre_merge | 1 |
| `test_determinism_agg.py` | cache_reset, concurrent load | e2e, slow, gpu_1, nightly | 1 |
| `test_mla_smoke.py` | TP=1 MLA registration and G1↔G2 round trip | kvbm, e2e, gpu_1, pre_merge | 1 |
| `test_mla_tp2.py` | Replicated G1, striped G2, and owner-root NCCL onboard | kvbm, e2e, gpu_2, pre_merge | 2 |
| `test_determinism_disagg.py` | disagg determinism | kvbm, vllm, trtllm, e2e, slow, gpu_2, nightly | 2 |
| `test_cuda_graph.py` | CUDA graph (TRT-LLM only) | kvbm, trtllm, nightly, gpu_1 | 1 |

> The `trtllm` markers on `test_determinism_disagg.py` and `test_cuda_graph.py` are framework labels. Whether the TRT-LLM specs/files transfer into the standalone `kvbm` repo is still open — see notes.

## Reference: Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KVBM_MODEL_ID` | DeepSeek-R1-Distill-Llama-8B (first `_MODEL_CONFIGS` entry) | Override the first model config in the connector spec set; required for Qwen spec ids |
| `KVBM_CPU_BLOCKS` | 10000 | CPU cache block count |
| `KVBM_GPU_BLOCKS` | 2048 | GPU cache block count |
| `KVBM_MAX_ITERATIONS` | 100 | Max iterations (cache-reset test) |
| `KVBM_NUM_ITERATIONS` | 15 | Number of iterations (concurrent test) |
| `KVBM_REQUEST_DELAY` | 30 | Delay between iterations (seconds) |
| `KVBM_ENABLE_MLA` | unset | Unlock DeepSeek-V2-Lite specs |
| `KVBM_MLA_MODEL_ID` | DeepSeek-V2-Lite | Override the larger determinism-suite MLA model |
| `KVBM_MLA_SMOKE_MODEL_ID` | v2ray/DeepSeek-V3-1B-Test | Override the small MLA smoke model |
| `KVBM_SERVER_START_TIMEOUT` | 600 | Server startup timeout |
| `KVBM_GPU_MEMORY_UTILIZATION` | per-spec | vllm memory fraction |
| `KVBM_EXTERNAL_BASE_URL` | unset | External-attach mode (set by `run_server.sh`) |
| `KVBM_EXTERNAL_METRICS_PORT` | unset | External-attach mode |
| `KVBM_SPEC_ID` | unset | Spec id handshake for decomposed flow |

## Container mode (deferred)

Container mode runs pytest inside a hermetic vllm image (mirrors CI). In the
dynamo monorepo this image was built by a now-dropped build step; the standalone
`kvbm` repo produces **no such image**, so this mode is not wired up here.

To run it against an **externally-supplied** image, that image must already
contain the installed `kvbm` wheel and the `tests/kvbm_integration/` tree (or
they must be mounted/installed at startup). `-v $(pwd):/workspace` below mounts
the `kvbm` repo root.

```bash
docker run --gpus all --rm \
    --shm-size=10G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    -e RUST_BACKTRACE=1 \
    -e HF_TOKEN \
    <extra_env_flags> \
    -v $(pwd):/workspace \
    -v /tmp:/tmp \
    -v /mnt/:/mnt \
    --cap-add CAP_SYS_PTRACE \
    --ipc host \
    -w /workspace \
    <image> \
    bash -c "pip install pytest-benchmark pytest-asyncio -q && timeout <timeout> pytest <pytest_args> -v --tb=short -s"
```

Note: do NOT use `--network host` (conflicts with host NATS/etcd), `--runtime nvidia` (not portable), or `-it` (hangs in non-interactive shells). Stock runtime images are missing `pytest-benchmark` / `pytest-asyncio`, so install them at container startup.

## Related Skills

- `/kvbm-sandbox-venv` — set up `.sandbox/` (local mode prerequisite)
- `/kvbm-maturin-dev` — rebuild kvbm-py3 (local mode prerequisite)
- `/kvbm-decomposed-run` — three-shell iteration loop (bypasses this wrapper)
- `/kvbm-diagnose` — read-only triage of failed runs
- `/kvbm-rebuild-lockfiles` — regenerate all Cargo.lock files
