# KVBM

KVBM (KV Block Manager) is the Dynamo subsystem for managing LLM KV-cache
blocks across the GPU (G1), host (G2), and remote/storage tiers, including
offload, prefix reuse, KV-event indexing, and conditional disaggregated
prefill/decode transfer.

This package, `kvbm`, is the standalone Python wheel (`maturin` /
[`pyo3`](https://pyo3.rs) backed) that exposes the KVBM Rust crates as a native
extension module, `kvbm._core`.

## Layout

- `src/` — the `_core` extension crate (`kvbm-py3`), its own single-package
  Cargo workspace.
- `crates/` — the KVBM Rust crates (`kvbm-connector`, `kvbm-common`,
  `kvbm-hub`, `kvbm-protocols`, `kvbm-kernels`, ...), consumed as path deps.
- `python/kvbm/` — the Python package source installed alongside `_core`.

## Features

The connector/runtime path is mandatory in the wheel (see `Cargo.toml`).
Only ancillary surfaces are feature-gated:

- `hub` *(default)* — hub-facing pyclasses (`PrefillRouterHandler`,
  `CompletionEvent`), pulling `kvbm-hub` + `velo`.
- `kernels` — standalone CUDA kernel bindings.

## Building

```bash
uv venv .sandbox --python 3.12
source .sandbox/bin/activate
uv pip install 'maturin>=1.0,<2.0' patchelf 'pydantic>=2.0'

# dev build (lean line-tables-only debug profile)
CUDA_HOME=/usr/local/cuda KVBM_REQUIRE_CUDA=1 maturin develop

# shippable wheel (stripped, optimized)
maturin build --release
```

CUDA is required (`KVBM_REQUIRE_CUDA=1`); the build links NIXL, so set an
`LD_LIBRARY_PATH` that includes the NIXL install (`/opt/nvidia/nvda_nixl/lib/<arch>`)
and `/usr/local/cuda/lib64` at runtime.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
