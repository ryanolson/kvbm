# kvbm Roadmap

Status: **extraction complete.** kvbm now lives standalone and is the sole development
home (dynamo keeps a frozen `lib/kvbm-*` copy that diverges; we do not retarget it). The
repo builds (`maturin develop`/`--release`), imports the flat API, passes the v2 aggregation
determinism smokes, and the `KvbmConnector` rename is verified live end-to-end. CI is
code-green on `crates/` (fmt / clippy `--all-targets` / machete / dead_code).

This is the backlog of features and follow-ups to pursue **in this repo**. None block
day-to-day development; they are the forward work.

## Features

### 1. Engine-internal G2 search dedup for concurrent overlapping inflight searches
- **What:** when multiple decode requests concurrently search the G2 (host/remote) tier
  for overlapping prefixes, the engine issues redundant searches. Deduplicate inflight
  searches so overlapping concurrent lookups share a single search + result.
- **Where:** `crates/kvbm-engine` (search / remote-search path).
- **Why:** eliminates redundant G2 search work and transfer churn under concurrency.

### 2. CD remote-selection: account for prefix over-pull cost
- **What:** conditional-disagg (CD) remote selection chooses a remote to pull prefix blocks
  from but does not weigh the cost of *over-pulling* — fetching more prefix than the local
  request actually needs. Fold over-pull cost into the selection policy.
- **Where:** `crates/kvbm-engine` CD / remote-search selection.
- **Why:** better remote choice → less wasted interconnect bandwidth.

### 3. CD cross-instance prefill-attach-failure fast signaling
- **What:** when a cross-instance prefill *attach* fails, the decode side currently waits
  the full 60s watchdog instead of receiving a prompt failure signal. Add a
  failure-signaling path so decode fails/retries fast instead of stalling.
- **Where:** `crates/kvbm-engine` CD cross-instance path (+ a signal in
  `crates/kvbm-protocols`).
- **Why:** removes 60s stalls on prefill-attach failure; matches the local-failure latency.

## Known issues

### nixl-sys test-link gap on GB10 (`-lstream`)
- **What:** on GB10, `kvbm-connector` **test** binaries fail to link against nixl's
  `libstream` (nixl-sys). `cargo clippy --all-targets` and `cargo test` on the connector
  are blocked; `cargo clippy --lib --bins` is clean.
- **Where:** `crates/kvbm-connector` test-target link config + the `nixl-sys` build.
- **Status:** environment/build issue, not a logic bug. Resolve so `--all-targets` and the
  connector test suite link on GB10.

## Infrastructure & reproducibility

### CI finalization
- 5 workflows are drafted under `.github/workflows/` (rust-checks, wheel-build,
  kvbm-kernels-cuda, gpu-tests, python-lint). Remaining: provision real GPU / CUDA-toolkit /
  arm runner labels (currently placeholders), and the `DYNAMO_GIT_TOKEN` secret if the
  `ai-dynamo/dynamo` branch the deps point at is private.

### Pin the dynamo git-deps to an immutable rev
- `crates/Cargo.toml` git-deps `dynamo-tokens` / `dynamo-memory` / `dynamo-kv-router` /
  `dynamo-kv-hashing` on the floating branch `ryan/kvbm-engine-service`. Pin to an immutable
  rev (or tag) for reproducible builds, and keep that branch alive until then.

### Durable vllm wheel for the dev/CI venv
- The GB10 venv needs `vllm 0.19.1rc1.dev232+cu130`, which has been GC'd from the nightly
  index. The 209 MiB wheel is vendored locally at `wheels/` (gitignored) and pinned in
  `.sandbox/requirements.release-pinned.txt`; the per-commit URL on `wheels.vllm.ai` is
  still live. For fresh-clone / CI reproducibility, mirror the wheel to a durable store
  (e.g. a GitHub Release asset) and URL-pin it in the freeze. See `/kvbm-sandbox-venv`.
