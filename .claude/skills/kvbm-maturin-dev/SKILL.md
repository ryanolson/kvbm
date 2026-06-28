---
name: kvbm-maturin-dev
description: Rebuild kvbm-py3 Python bindings with correct env ordering (post-maturin nccl re-bump trap)
user-invocable: true
disable-model-invocation: true
---

# KVBM maturin develop

Rebuild the `kvbm-py3` PyO3 extension against the current `.sandbox/` torch, with the correct CUDA env vars exported and the post-maturin nccl re-bump that the bring-up learned about the hard way. The maturin manifest lives at the repo root (`Cargo.toml`, package `kvbm-py3`, lib `_core`), so every command in this skill runs **from the repo root** — there is no `lib/bindings/kvbm` subdir.

**The gotcha**: `maturin develop` runs a pip install step that respects vllm's `nvidia-nccl-cu13==2.28.9` pin. torch 2.11.0 calls `ncclDevCommDestroy` which only exists in nccl 2.29+. Every time you run `maturin develop`, nccl silently rolls back — you **must** re-bump it after.

## Arguments

`/kvbm-maturin-dev [--clean] [--debug] [--features FEATS]`

- **--clean** (default for ABI-change runs): `cargo clean -p kvbm-py3` before rebuilding. Required when torch ABI changed (e.g. after `/kvbm-sandbox-venv`).
- **--debug**: build the unoptimized debug profile for fast binding-dev iteration. **Off by default — the default build is `--release`.** A debug kvbm extension is materially slower at runtime and will drag the smokes/perf/matrix (e.g. the 4-cell × ~40-iter agg matrix). Only use `--debug` when actively iterating on the Rust binding code and you need short rebuilds. Any run that feeds a smoke/perf/matrix result must be a release build — that is the reproducibility contract.
- **--features FEATS** (default: none — the Cargo default `[KVBM connector, hub]` is used): extra Cargo feature flags. Options: `KVBM connector`, `hub`, `kernels`. The dev build needs no `--features` flag; pass one only to add an optional feature such as `kernels`.

## Step 1: Preflight

Confirm the venv is present and torch is sm_120+ capable:

```bash
test -x .sandbox/bin/python || { echo "no .sandbox venv — run /kvbm-sandbox-venv first"; exit 1; }
.sandbox/bin/python -c "import torch; archs = torch.cuda.get_arch_list(); print('archs:', archs); assert any('sm_10' in a or 'sm_11' in a or 'sm_12' in a for a in archs), 'torch has no sm_100+ kernels; run /kvbm-sandbox-venv'"
```

Also confirm CUDA is on disk at the expected location:

```bash
test -d /usr/local/cuda/bin || { echo "CUDA toolkit not at /usr/local/cuda"; exit 1; }
```

## Step 2: Export Env

All of this must be in the same shell as `maturin develop`:

```bash
source .sandbox/bin/activate
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export KVBM_REQUIRE_CUDA=1
```

`KVBM_REQUIRE_CUDA=1` makes the kernels build fail loud rather than silently producing a stub.

## Step 3: (Optional) Clean

If `--clean` or if the last rebuild was against a different torch (run from the repo root — `kvbm-py3` is the wheel package):

```bash
cargo clean -p kvbm-py3
```

Signs you need `--clean`: `undefined symbol` at `import kvbm`, PyO3 ABI mismatch errors, unfamiliar torch-libc symbols in ldd output.

## Step 4: maturin develop

Release is the default. Drop `--release` only if `--debug` was requested. Run from the repo root (manifest-path resolves to the root `Cargo.toml`, default features `[KVBM connector, hub]`):

```bash
maturin develop --release
```

Stream the output. The release build takes ~2-5 minutes cold, under 1 minute incremental. Watch for:
- `Finished \`release\` profile [optimized] target(s)` — rust build OK (with `--debug`: `Finished \`dev\` profile [unoptimized + debuginfo]`)
- `📦 Built wheel` — maturin packaging OK
- `Installed kvbm-1.2.0` — site-packages install OK (version tracks `pyproject.toml`)

> Release artifacts live in `target/release/`, separate from `target/debug/`. Switching debug↔release does **not** require `--clean` (different profile dirs, same torch ABI) — only a torch ABI change does.

## Step 5: Post-Maturin NCCL Re-Bump (CRITICAL)

maturin's install step just rolled nvidia-nccl-cu13 back to 2.28.9 to satisfy vllm's pin. Undo it:

```bash
uv pip install --force-reinstall --no-deps 'nvidia-nccl-cu13>=2.29'
```

Verify:

```bash
python - <<'PY'
import ctypes
import site
from pathlib import Path

path = Path(site.getsitepackages()[0]) / "nvidia/nccl/lib/libnccl.so.2"
library = ctypes.CDLL(str(path))
version = ctypes.c_int()
assert library.ncclGetVersion(ctypes.byref(version)) == 0
print("NCCL runtime:", version.value)
assert version.value >= 22900
PY
```

Expect `22900` or newer. Do not use `torch.cuda.nccl.version()` for this
check: it reports the NCCL version PyTorch was compiled against, which can
remain `(2, 28, 9)` even while the dynamically loaded runtime is newer.

## Step 6: Smoke Verification

```bash
python - <<'PY'
import kvbm
print('kvbm version:', kvbm.__version__)
from kvbm import KvbmRuntime  # noqa: F401
from kvbm.vllm.connector import KvbmConnector  # noqa: F401
assert kvbm._CORE_AVAILABLE
print('OK — all connector imports resolved')
PY

# Connector import regression test.
pytest python/tests/test_connector_lazy_import.py -q
```

Expected: prints `OK — all connector imports resolved`, then verifies that `kvbm.vllm.connector` does not import vLLM eagerly.

## Step 7: Next Steps

Tell the user:

```
kvbm-py3 rebuilt. Next:

  Run the determinism flow:
    /kvbm-decomposed-run Qwen3-0.6B-intra --fast
    /kvbm-decomposed-run Qwen3-0.6B-inter --fast

  If imports fail:
    /kvbm-diagnose
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `undefined symbol: _ZN3c10...` at `import kvbm` | PyO3 / torch ABI drift | Re-run with `--clean`; if still broken, re-run `/kvbm-sandbox-venv` first |
| `ncclDevCommDestroy` error at `import torch` / `import kvbm` | nccl rolled back to 2.28.9 (step 5 skipped or silently failed) | Re-run step 5 |
| `cudarc` link failures during build | `CUDA_PATH` or `CUDA_HOME` not exported | Re-run step 2 in the same shell |
| `nvcc: command not found` during build | CUDA bin not on PATH | `export PATH=/usr/local/cuda/bin:$PATH` before maturin |
| `AttributeError: RustScheduler` during smoke | The scheduler-output pyclass is not exported from the root bindings | The façade import (`kvbm.vllm.connector.KvbmConnector`) should still succeed; the exception is caught inside `kvbm.vllm.connector.base` |
| `ImportError: cannot import name 'nixl_connector'` from pd.py | vllm 0.19.1 renamed the module to `.nixl` | Already fixed via try/except fallback; if you see this, check `python/kvbm/vllm/connector/pd.py:17` |

## Reference: Features

| Feature | Purpose | Default |
|---|---|---|
| `hub` | Hub-facing pyclasses (`PrefillRouterHandler`, `CompletionEvent`); pulls kvbm-hub + velo | ✓ |
| `kernels` | Bundles the standalone kvbm-kernels CUDA kernels via cudarc (exposes the `kvbm.kernels` submodule) | no |

The KVBM connector bindings are always built. The usual dev build needs no `--features` flag. Add `kernels` only if you need the standalone kernels submodule.
