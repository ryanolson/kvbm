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

- All external git deps come from **public** branches in the ai-dynamo org that
  carry upstream plus the changes kvbm and Rhino need. These branches are
  add-only. Never force-push or rebase them, because kvbm and Rhino pin revs on
  them.
- kvbm must resolve to the same dependency graph as Rhino, which consumes it as a
  git dependency. Change a shared pin (dynamo, nixl-sys, velo, tokio) only
  together with Rhino.
- Three **git** deps on dynamo, pinned to a rev on `ai-dynamo/dynamo` branch
  `rhino-main` (upstream plus the carrier stack):
  `dynamo-tokens`, `dynamo-kv-router`, `dynamo-kv-hashing` (the latter two
  via `kvbm-consolidator`). The dependency only runs kvbm → dynamo; dynamo does not depend
  on this repo.
- `kvbm-memory` is a locally-owned crate (`crates/kvbm-memory`), forked from
  `dynamo-memory` and no longer resolved via git. The single graph-wide NIXL
  binding pin (`nixl-sys = "=1.4.1"`, a rev on `ai-dynamo/nixl` branch `rhino-1.4`:
  NIXL 1.4.1 plus the deregistration and compiler fixes) is
  declared once in `crates/Cargo.toml`'s `[workspace.dependencies]`; `kvbm-memory`
  inherits it via `nixl-sys = { workspace = true }`. It stays at 1.4.1 because
  Rhino runs the `nixl-cu13` 1.4.1 wheel.
- `tokio` is a minimum version (`1.54.1`), not an exact pin.
- `velo`, `velo-ext`, and `velo-hub` come from the public `ai-dynamo/velo` tag
  `v0.21.0`. velo-hub is not on crates.io, and mixing a registry velo with the
  git velo-hub would build two copies of the velo types.

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

## Checking a change against Rhino

Rhino consumes the kvbm crates as git dependencies pinned to one revision. Rhino
enables `static-kernels`, which selects how the compiled `kvbm_kernels` object
links: as a static archive, or as an nvcc-built shared library. Real-versus-stub
kernel selection depends only on nvcc availability in `kvbm-kernels/build.rs`.
`KVBM_REQUIRE_CUDA` makes a missing nvcc fatal.

To check a kvbm change against Rhino before moving Rhino's pin, add a `[patch]`
for kvbm's git URL in a Rhino worktree's root `Cargo.toml`, with one entry per
`kvbm-*` crate pointing at this checkout's `crates/kvbm-*`. Then build the Rhino
crates that use kvbm through Rhino's uv environment:

```bash
timeout 3000 uv run cargo check -p rhino-kvbm -p rhino-hub -p rhino-fast-load --all-targets -j 8
```

Confirm that Rhino's `Cargo.lock` holds one copy each of `dynamo-kv-router`,
`nixl-sys`, and `tokio`.
