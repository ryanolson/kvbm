# CLAUDE.md

## Commands

```bash
# check
cargo check -p kvbm-hub

# All tests (always use timeout — async tests can deadlock)
timeout 60 cargo test -p kvbm-hub

# Single test
timeout 60 cargo test -p kvbm-hub -- <test_name>

# Lint
cargo clippy -p kvbm-hub
cargo fmt
cargo machete
```

## Architecture

`kvbm-hub` is a library crate targeting clients and a server executable (kvbm_hub). It runs a `velo-hub` hub and adds the KVBM extensions to it.

`velo-hub` owns discovery, registration, heartbeats, lease epochs, the registration store and both HTTP listeners. Do not add a second copy of any of these here.

**Dual-port design (from velo-hub):**
- **Discovery port** (default `1337`) — read-only peer lookups (`GET /v1/peers/...`).
- **Control port** (default `8337`) — register, unregister, heartbeat, plus the discovery routes.

**Key types and their roles:**

| Type | Module | Role |
|------|--------|------|
| `HubServerBuilder` / `HubServer` | `server` | Builds and runs a `velo_hub::Hub` with the KVBM extensions and the aggregate KVBM config |
| `HubClient` | `client` | Wraps `velo_hub::client::HubClient`; `register` sends the KVBM feature payloads as extensions |
| `KvbmFeature` | `features` | The KVBM config that each `velo_hub::Extension` advertises |
| `protocol` | `protocol` | Shared JSON types and URL path constants |
| `handlers` | `handlers` | Velo handler names for hub → client pushes |

**Routes:** each extension serves its routes under `/v1/ext/{NAME}`. The aggregate KVBM config is `GET /v1/ext/kvbm/config` (`protocol::paths::HUB_CONFIG`).

**Tests:** `test_hub` registers through a real `HubServer`. Use `test_hub::attempt` to check a rejected registration, and `test_hub::register_as` for a registration that must succeed.

**Integration test pattern:** tests bind both ports to `0` (OS-assigned) and extract the actual addresses from the returned `HubServer`. This avoids port conflicts across parallel test runs.

## Configuration (figment merge strategy)

Priority order (lowest → highest):

1. `HubConfig::default()` — compiled-in defaults (`bind_addr = 0.0.0.0`, `discovery_port = 1337`, `control_port = 8337`)
2. TOML or JSON file — path from `--config` CLI arg or `KVBM_HUB_CONFIG` env var; format auto-detected by `.toml`/`.json` extension
3. `KVBM_HUB_*` env vars — `KVBM_HUB_BIND_ADDR`, `KVBM_HUB_DISCOVERY_PORT`, `KVBM_HUB_CONTROL_PORT`
4. CLI args — `--bind-addr`, `--discovery-port`, `--control-port` (highest priority; only override when explicitly passed)

`KVBM_HUB_CONFIG` is consumed by the CLI's `--config` arg before `HubConfig::figment()` is called, and is explicitly ignored in the env layer to avoid double-processing.

**Why `Option<T>` CLI fields (not clap defaults):** CLI fields in the binary are `Option<T>` with no defaults. Only `Some` values are merged into Figment as a final layer. This ensures CLI args only win when explicitly passed — clap defaults would otherwise shadow env vars for fields the user never specified.
