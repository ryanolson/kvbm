# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in the standalone **kvbm** repository.

## What this repo is

kvbm is the KV Block Manager, extracted from the `ai-dynamo/dynamo` monorepo into its
own repo for independent development. This repo is the **sole development home** for
kvbm — dynamo retains a frozen copy of `lib/kvbm-*` that diverges; we do **not** retarget
or modify dynamo from here.

## Layout

- **Repo root** = the `kvbm-py3` PyO3 wheel package (`Cargo.toml`). Builds the cdylib
  `_core` (Python module `kvbm._core`); default features `[hub]`. The KVBM
  connector/runtime bindings are mandatory. It is its own
  single-package workspace (`[workspace] exclude = ["crates"]`).
- **`crates/`** = the Rust workspace (`crates/Cargo.toml`) of the 14 `kvbm-*` library
  crates (client, common, config, connector, consolidator, engine, hub, kernels,
  logical, observability, physical, protocols, runtime, service).
- **`python/kvbm/`** = the Python package. Public API is flat: `kvbm.<bindings>`,
  the single `kvbm.vllm` module (`kvbm.vllm.connector.KvbmConnector`, `kvbm.vllm.config`,
  `kvbm.vllm.consolidator_config`, ...), and `kvbm.hub`.
- **`tests/kvbm_integration/`** = the integration/determinism suite (self-contained;
  provides its own etcd/nats fixtures).
- **`docs/`**, **`.claude/skills/`** (the `kvbm-*` skills), **`.github/workflows/`**.

## External dependencies

- Four **git** deps on dynamo, pinned to branch `ryan/kvbm-engine-service`:
  `dynamo-tokens`, `dynamo-memory`, `dynamo-kv-router`, `dynamo-kv-hashing` (the latter two
  via `kvbm-consolidator`). The dependency only runs kvbm → dynamo; dynamo does not depend
  on this repo. Keep that branch fetchable; prefer pinning to an immutable rev for CI
  reproducibility.
- `velo` / `velo-ext` are registry deps.

## Build

The binding builds from the **repo root** (not a subdirectory). Provision the dev venv
with `/kvbm-sandbox-venv` first (GB10/sm_121 needs the cu130 nightly torch + nccl ≥ 2.29).

```bash
source .sandbox/bin/activate
export CUDA_PATH=/usr/local/cuda CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH KVBM_REQUIRE_CUDA=1
maturin develop --release          # or `/kvbm-maturin-dev`
# maturin's install step rolls nccl back to vllm's pin — re-bump after:
uv pip install --force-reinstall --no-deps 'nvidia-nccl-cu13>=2.29'
```

Runtime import needs the NIXL + CUDA shared libs on the loader path:

```bash
export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python -c "import kvbm; from kvbm.vllm.connector import KvbmConnector"
```

Build profiles live in the **root** `Cargo.toml` (they govern the wheel; `crates/`
profiles do not apply to it): `dev` = `debug="line-tables-only"` (lean, keeps backtraces),
`release` = stripped + thin-LTO (~28 MiB cdylib).

## Test

```bash
# Rust gates (run from crates/):
cd crates && cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings && cargo machete

# Integration/determinism (GB10):
export KVBM_HARDWARE_PROFILE=spark-gb10 KVBM_GPU_MEMORY_UTILIZATION=0.7
timeout <N> python -m pytest tests/kvbm_integration/test_determinism_agg_matrix.py -v
```

Always wrap `cargo test`/`pytest` with `timeout`. Run `cargo fmt --all -- --check` and
`cargo clippy --all-targets -- -D warnings` before declaring a Rust task done.

## Documentation Alignment Policy

When modifying code, evaluate and update related documentation so it stays accurate:

- **README / docs / SKILL.md**: verify API descriptions, code examples, file trees, and
  build/run commands match the implementation.
- **Inline docs**: keep function signatures, parameters, and usage examples current.
- **Remove stale references**: if an API/module is removed, remove its doc references —
  don't leave orphans. Any symbol used in docs must be defined before first use.

## Active Plan

If an `ACTIVE_PLAN.md` exists, load it each session and advance it to completion; update it
before returning control. If it should be ignored, rename it to `_ACTIVE_PLAN_.md`.
