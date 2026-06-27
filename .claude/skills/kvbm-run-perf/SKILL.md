---
name: kvbm-run-perf
description: Run aiperf performance benchmarks against an externally-running KVBM-enabled serving stack (URL-only)
user-invocable: true
disable-model-invocation: true
---

# Run KVBM Performance Benchmarks

Run aiperf benchmarks against an already-running, KVBM-enabled Dynamo + vLLM serving
stack reachable at an OpenAI-compatible URL.

> This repo (`kvbm`) ships the KVBM crates and the `kvbm._core` wheel only — it does
> **not** contain a serving stack, launch scripts, or a container to run them in. KVBM
> only runs inside a vLLM/Dynamo deployment. Stand that stack up separately (e.g. from
> the `dynamo` repo) and point this skill at its endpoint with `--url`.

## Arguments

`/kvbm-run-perf <framework> --url URL [options...]`

- **framework** (required): Only `vllm` is supported. Reject others: "Only vllm is supported for KVBM benchmarks currently."
- **--url URL** (required): OpenAI-compatible base URL of the running stack (e.g. `http://localhost:8000`).
- **--model MODEL** (default: `Qwen/Qwen3-0.6B`)
- **--concurrency N** (default: `10`)
- **--isl N** (default: `1024`): Input sequence length mean
- **--osl N** (default: `256`): Output sequence length mean
- **--requests N** (default: `100`): Total request count
- **--topology TOPO** (default: `agg`): Descriptive label for the report only (`agg`, `disagg`, `agg-router`, `disagg-router`, `disagg-2p2d`). It records how the external stack was deployed; it does **not** launch anything.
- **--artifact-dir DIR** (default: `artifacts/kvbm-perf`): Where to save results

## Step 1: Validate Framework

If the first argument is not `vllm`:
> "Only `vllm` is currently supported for KVBM perf benchmarks. TensorRT-LLM and SGLang support is planned."

## Step 2: Parse Configuration and Show Plan

Collect all options, fill defaults, display:

```
KVBM Performance Benchmark Plan
────────────────────────────────
Framework:   vllm
Topology:    <topology>   (label only)
Model:       <model>
Concurrency: <N>
ISL/OSL:     <ISL>/<OSL>
Requests:    <N>
Target URL:  <url>
Artifact dir: <dir>
```

If `--url` was not provided, stop and tell the user it is required: this skill does not
launch a serving stack, so it needs the URL of an already-running endpoint.

Confirm with the user before proceeding.

## Step 3: Check Prerequisites

```bash
# aiperf must be installed on the host (or available on PATH)
which aiperf || echo "ERROR: aiperf not found on PATH"

# The serving stack must already be up and serving the model
curl -sf "<url>/v1/models" | python -m json.tool || echo "ERROR: stack not reachable at <url>"
```

Confirm the requested `--model` appears in the `/v1/models` listing. If the endpoint is
unreachable, stop and ask the user to bring up the stack (or fix `--url`) before
benchmarking.

## Step 4: Run aiperf Benchmark

Run aiperf against the endpoint:

```bash
aiperf profile \
    --model "<model>" \
    --url "<url>" \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --concurrency <concurrency> \
    --request-count <requests> \
    --synthetic-input-tokens-mean <isl> \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean <osl> \
    --output-tokens-stddev 0 \
    --extra-inputs ignore_eos:true \
    --extra-inputs temperature:0.0 \
    --artifact-dir "<artifact-dir>" \
    --random-seed 100
```

Stream output so the user can see progress.

## Step 5: Report Results

After aiperf completes, present key metrics from the summary table:

```
KVBM Performance Results
────────────────────────
Topology:          <topology>
Model:             <model>
Concurrency:       <N>
ISL/OSL:           <ISL>/<OSL>

Key Metrics:
  TTFT (avg/p50/p99):          X / Y / Z ms
  ITL (avg/p50/p99):           X / Y / Z ms
  Request Latency (avg/p99):   X / Y ms
  Output Throughput:            X tokens/s
  Request Throughput:           X req/s

Artifacts saved to: <artifact-dir>
```

Suggest visualization:
```bash
aiperf plot --artifact-dir <artifact-dir>
```

## Concurrency Sweep

If the user asks for a sweep or Pareto analysis, run multiple concurrency levels sequentially. For each level, use a separate artifact subdirectory:

```bash
for c in 1 2 5 10 25 50; do
    aiperf profile \
        --model "<model>" \
        --url "<url>" \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --concurrency $c \
        --request-count $(( c * 5 > 20 ? c * 5 : 20 )) \
        --synthetic-input-tokens-mean <isl> \
        --output-tokens-mean <osl> \
        --extra-inputs ignore_eos:true \
        --artifact-dir "<artifact-dir>/c${c}" \
        --random-seed 100
done
```

Then plot all results: `aiperf plot --artifact-dir <artifact-dir>`

## Reference: Key aiperf Flags

| Flag | Description |
|------|-------------|
| `--concurrency N` | Max concurrent requests |
| `--request-count N` | Total requests to send |
| `--request-rate N` | Target requests/second (alternative to concurrency) |
| `--benchmark-duration N` | Run for N seconds instead of request count |
| `--warmup-request-count N` | Warmup requests before measurement |
| `--gpu-telemetry` | Collect GPU metrics via DCGM |
| `--input-file FILE` | Use trace dataset instead of synthetic |
| `--custom-dataset-type mooncake_trace` | For Mooncake trace replay |
| `--fixed-schedule` | Replay at original timestamps |
