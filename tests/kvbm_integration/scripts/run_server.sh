#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Layer B — launch a vllm+KVBM server using the same KvbmServerManager the
# integration tests use. Spec attributes (model id, attention backend,
# block size, batch_invariant) are sourced from the test module's
# parametrize list — this script never hardcodes per-model attributes.
# Prints KVBM_EXTERNAL_BASE_URL / KVBM_EXTERNAL_METRICS_PORT /
# KVBM_SPEC_ID exports for shell 3 (run_eval.sh). Ctrl-C to stop.
#
# Usage: run_server.sh <spec-id>
#
# spec-id matches KvbmServerSpec.id in the test module, e.g.:
#   DeepSeek-R1-Distill-Llama-8B-intra
#   DeepSeek-V2-Lite-inter      (MLA; enabled by default, KVBM_ENABLE_MLA=0 to skip)
#   Qwen3-0.6B-intra            (only if KVBM_MODEL_ID=Qwen/Qwen3-0.6B)
#
# Env overrides honored:
#   KVBM_CPU_BLOCKS              (override spec.cpu_blocks)
#   KVBM_GPU_BLOCKS              (override spec.gpu_blocks)
#   KVBM_SERVER_START_TIMEOUT    (server bring-up timeout, default 600s)
#   KVBM_ENABLE_MLA              (opt-out escape hatch; set to 0 to skip MLA specs)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <spec-id>" >&2
    echo "       (e.g. DeepSeek-R1-Distill-Llama-8B-intra, Qwen3-0.6B-inter)" >&2
    exit 2
fi

SPEC_ID="$1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

START_TIMEOUT="${KVBM_SERVER_START_TIMEOUT:-600}"

export KVBM_SPEC_ID="$SPEC_ID"
export KVBM_SERVER_START_TIMEOUT="$START_TIMEOUT"

exec python - <<'PY'
import dataclasses
import os
import signal
import sys
from pathlib import Path

from tests.kvbm_integration.fixtures import KvbmServerManager
from tests.kvbm_integration.test_determinism_agg import _CACHE_RESET_SPECS

spec_id = os.environ["KVBM_SPEC_ID"]
specs_by_id = {s.id: s for s in _CACHE_RESET_SPECS}
if spec_id not in specs_by_id:
    sys.stderr.write(
        f"[server] unknown spec id: {spec_id!r}\n"
        f"[server] known: {sorted(specs_by_id)}\n"
    )
    sys.exit(6)

spec = specs_by_id[spec_id]

if spec.model_config.use_mla and os.environ.get("KVBM_ENABLE_MLA", "1").lower() not in (
    "1",
    "true",
    "yes",
    "on",
):
    sys.stderr.write(
        f"[server] {spec_id} is an MLA spec and MLA is disabled via KVBM_ENABLE_MLA=0; "
        f"unset or set to 1 to launch\n"
    )
    sys.exit(7)

cpu_override = os.environ.get("KVBM_CPU_BLOCKS")
gpu_override = os.environ.get("KVBM_GPU_BLOCKS")
overrides = {}
if cpu_override is not None:
    overrides["cpu_blocks"] = int(cpu_override)
if gpu_override is not None:
    overrides["gpu_blocks"] = int(gpu_override)
if overrides:
    spec = dataclasses.replace(spec, **overrides)

start_timeout = int(os.environ["KVBM_SERVER_START_TIMEOUT"])

mgr = KvbmServerManager(spec=spec, log_dir=Path("/tmp/kvbm-run-server-logs"))


def _stop(*_):
    print("[server] stopping ...", flush=True)
    try:
        mgr.stop_server()
    finally:
        sys.exit(0)


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

print(
    f"[server] starting spec={spec.id} "
    f"(cpu_blocks={spec.cpu_blocks}, gpu_blocks={spec.gpu_blocks}, "
    f"timeout={start_timeout}s) ...",
    flush=True,
)
ok = mgr.start_server(timeout=start_timeout)
if not ok:
    print("[server] failed to start", flush=True)
    sys.exit(5)

print("", flush=True)
print("=" * 64, flush=True)
print("[server] READY. Export these in shell 3 (run_eval.sh):", flush=True)
print(f"  export KVBM_EXTERNAL_BASE_URL={mgr.base_url}", flush=True)
print(f"  export KVBM_EXTERNAL_METRICS_PORT={mgr.metrics_port}", flush=True)
print(f"  export KVBM_SPEC_ID={spec.id}", flush=True)
print("=" * 64, flush=True)
print("[server] Ctrl-C to stop.", flush=True)
signal.pause()
PY
