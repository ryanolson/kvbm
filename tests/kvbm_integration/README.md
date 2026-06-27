# KVBM Determinism & Behavior Tests

## Overview

This suite validates that vLLM + KVBM produces deterministic outputs across
prefix-cache resets and under concurrent load. The aggregate path is decomposed
into three layers so local iteration can run them in separate shells and re-run
the eval loop without re-spawning vLLM.

| Layer | Module | Responsibility |
|-------|--------|----------------|
| A. Deps | `fixtures/deps.py` | Preserve fixture ordering; aggregate KVBM needs no fixture-managed NATS or etcd. |
| B. Server | `fixtures/server.py` | Launch `vllm serve` with `build_kv_transfer_config(model_config, ...)`. |
| C. Eval | `fixtures/eval.py` | Bind `AggDeterminismTester` to the running server and run the determinism loop. |

`test_determinism_agg.py` parametrizes one `kvbm_server_spec` axis that bundles
`(model_config, cpu_blocks, gpu_blocks, onboard_mode)`. Each model is crossed
with both onboard modes (`intra` and `inter`) so every run catches
mode-specific regressions. Spec IDs take the form `<model>-<mode>`, with
layout dimensions appended by matrix tests, for example
`Qwen3-0.6B-inter-g2uni-g1fc`.

## Running - Composed

```bash
# Quick smoke (2 iterations instead of 100)
KVBM_MAX_ITERATIONS=2 KVBM_NUM_ITERATIONS=2 KVBM_REQUEST_DELAY=2 \
    pytest tests/kvbm_integration/test_determinism_agg.py \
        -v -k "test_determinism_agg_with_cache_reset" --tb=short

# Full aggregate run
pytest tests/kvbm_integration/test_determinism_agg.py -v -s

# G1/G2 routing matrix
pytest tests/kvbm_integration/test_determinism_agg_matrix.py -v -s
```

## Running - Decomposed

The decomposed flow is keyed on a spec ID that matches `KvbmServerSpec.id` in
`test_determinism_agg.py`. `run_server.sh` reconstructs the exact spec from the
test module's parametrize list, so attention backend, block size, and
`batch_invariant` always come from the canonical source.

```bash
# Shell 1: no dependency process is needed for aggregate KVBM.
unset NATS_SERVER ETCD_ENDPOINTS

# Shell 2: launch one server spec.
KVBM_MODEL_ID=Qwen/Qwen3-0.6B \
    bash tests/kvbm_integration/scripts/run_server.sh Qwen3-0.6B-intra

# Shell 3: export the values printed by shell 2, then run eval.
export KVBM_EXTERNAL_BASE_URL=http://localhost:NNNN
export KVBM_EXTERNAL_METRICS_PORT=NNNN
export KVBM_SPEC_ID=Qwen3-0.6B-intra
bash tests/kvbm_integration/scripts/run_eval.sh
```

`run_eval.sh` defaults to running `test_determinism_agg_with_cache_reset`
filtered by `KVBM_SPEC_ID`. Positional args override the target entirely.

`run_server.sh` honors `KVBM_CPU_BLOCKS`, `KVBM_GPU_BLOCKS`, and
`KVBM_SERVER_START_TIMEOUT` env overrides by applying them to the canonical
spec. `KVBM_CPU_BLOCKS` drives `cache.host.num_blocks` directly; the Rust
leader bails at startup if neither a host nor disk cache tier is configured.

`DeepSeek-V2-Lite` is the suite's MLA model and is currently gated. Set
`KVBM_ENABLE_MLA=1` in both the server shell and eval shell to opt in. Pytest
reports it as skipped otherwise.

## External-Attach Mode

Setting `KVBM_EXTERNAL_BASE_URL` makes the `kvbm_server` and `kvbm_deps`
fixtures skip spawn and bind to a long-lived external server. The test loop
runs against the existing process.

`KVBM_EXTERNAL_METRICS_PORT` is required alongside the base URL. `KVBM_SPEC_ID`
is required by `run_eval.sh` to filter pytest to the launched parametrization.

## Markers

- `kvbm` - KV behavior and model determinism tests
- `kvbm_concurrency` - concurrent-load variant
- `e2e` - end-to-end tests
- `slow` - long-running tests
- `gpu_1` - needs one GPU
- `nightly` - preferred for nightly runs

## Configuration

Server and model settings:

- `KVBM_MODEL_ID` (default: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)
- `KVBM_SERVER_PORT` - pin the vLLM port; otherwise dynamically allocated
- `KVBM_SERVER_START_TIMEOUT` (default: `600`s)
- `KVBM_GPU_MEMORY_UTILIZATION` (default: `0.9`)
- `KVBM_MLA_BACKEND` (default: `TRITON_MLA`; set `FLASH_ATTN_MLA` on H100)

Cache size overrides:

- `KVBM_CPU_BLOCKS` (default: `10000` for cache-reset, `30000` for concurrent)
- `KVBM_GPU_BLOCKS` (default: `2048`)

Test duration:

- `KVBM_MAX_ITERATIONS` (default: `100`) - cache-reset test
- `KVBM_NUM_ITERATIONS` (default: `15`) - concurrent test
- `KVBM_REQUEST_DELAY` (default: `30`s)
- `KVBM_HTTP_TIMEOUT` (default: `30`s)

External attach:

- `KVBM_EXTERNAL_BASE_URL` - when set, fixture skips spawn
- `KVBM_EXTERNAL_METRICS_PORT` - required alongside `KVBM_EXTERNAL_BASE_URL`
- `KVBM_SPEC_ID` - spec ID launched by `run_server.sh`

MLA gate:

- `KVBM_ENABLE_MLA` - set to `1`/`true`/`yes`/`on` to run MLA specs

## Requirements

- `vllm` executable on `PATH`
- The `llm_server_kvbm` fixture in `common.py` wires `kvbm.vllm.connector`
- One GPU for aggregate cache-reset tests
- `vLLM bench` installed for the concurrent test

Some older tests in this directory still spawn NATS/etcd through local fixtures
and require `etcd` and `nats-server` on `PATH`; the primary aggregate
determinism tests do not.

## Notes

- Logs are written under per-test directories via
  `tests/utils/test_output.py:resolve_test_output_path`.
- Warmup is critical; disabling it hides initialization-related determinism bugs.
- `test_kvbm.py`, `test_chunked_prefill.py`, and SWA tests use the older
  `llm_server_kvbm` fixture. Consolidating them with `kvbm_server` is future
  cleanup.
