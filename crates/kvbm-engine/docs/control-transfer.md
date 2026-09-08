# Control-plane Transfer module

The `transfer` control module exposes the public KVBM transfer surface
over velo. It is **a remote-control API**: each handler is dispatched
*at* a leader and tells that leader to do something on behalf of the
caller. Holder and puller are different leaders; the same RPC client
talks to each.

## Surface

| Velo handler | Side | Purpose |
|---|---|---|
| `kvbm.leader.control.open_session` | Holder | Search the leader's tiers, open a holder-side disagg session, populate it. Returns the attach triple `(session_id, instance_id, endpoint)`. |
| `kvbm.leader.control.pull_from_session` | Puller | Attach to a session living on `request.source_instance_id`, drain its commits/availability, pull blocks into the puller's local G2 pool. Long-poll. |
| `kvbm.leader.control.close_session` | Holder | Idempotent teardown of a parked session. |
| `kvbm.leader.control.search_prefix` | Holder | Hub HTTP query route: shim over `open_session(find_mode = Sync, search_mode = Prefix)`. |
| `kvbm.leader.control.search_scatter` | Holder | Hub HTTP query route: shim over `open_session(find_mode = Sync, search_mode = Scatter)`. |

Wire types live in
[`kvbm_protocols::control::modules::transfer`](../../kvbm-protocols/src/control/modules/transfer.rs);
all handlers go through `InstanceLeader` methods so in-process callers
can skip the velo round-trip:

| Method on `InstanceLeader` | Handler shim |
|---|---|
| `open_transfer_session` | `open_session`, `search_prefix`, `search_scatter` |
| `pull_from_session` | `pull_from_session` |
| `close_transfer_session` | `close_session` |

## Search modes

| Mode | G2 lookup | G1 lookup (only if `tiers.g1 = true`) | G3 lookup (only if `tiers.g3 = true`) | Intended use |
|---|---|---|---|---|
| `Prefix` | `BlockManager::match_blocks` (contiguous prefix; stop at first miss) | `match_prefix` with `touch=false`, from the cursor G2 reached | Not consulted in v1 | LLM prompt-prefix KV reuse |
| `Scatter` | `BlockManager::scan_matches` with `touch=false` | `scan_matches` for hashes that missed in G2 | `scan_matches` for hashes that missed in G2 **and** in G1 | Arbitrary-subset reuse, cross-session sharing |

Every tier beyond G2 is opt-in, and `TierSelection` defaults to all
tiers off. A holder that gains a G1 source therefore serves the same
results as before until a caller asks for the device tier. The two
`open_session` callers that pull set `tiers.g1 = true`. The query-only
`search_prefix` / `search_scatter` shims keep the default, because they
read the matched set and never pull.

Tier order is device before disk. A G1 hit costs one local DMA, a G3 hit
costs a disk read *and* the same DMA, so G3 sees only the hashes G1
cannot supply. G2 needs no copy and wins over both.

`Scatter` and the G1 walk use `touch=false` so an RPC search does not
perturb the local LRU: a block a peer wants must not outrank a block
this node still reads. `Prefix` deliberately does not extend into G3 in
v1. The contiguous-prefix walk needs gap handling that does not
pay for itself yet.

In `Prefix` mode G2 and G1 extend one shared cursor in turn, so a run G2
serves and a run G1 serves join into one contiguous prefix. The walk
ends when neither tier advances the cursor. Each run costs one store
lock, and the pinned G1 set never reaches past the committed prefix.

## Find modes

| Mode | Behavior | Use case |
|---|---|---|
| `Async` (default) | Returns as soon as the session is opened and the populator is spawned. Caller learns matched hashes via the disagg `commits()` stream after attach. | Lowest-latency open + immediate-attach. |
| `Sync` | Awaits the find phase across `tiers`. Response carries `committed` + `breakdown` inline. Staging still runs in background. | Orchestrators that fan out opens across several holders and compare matched sets before committing to a puller. |

Staging (G1→G2 and G3→G2) is always background. `Sync` awaits only
the local scan, not the stage. The response stays fast with
`tiers.g1 = true` or `tiers.g3 = true`.

## Populator: find_phase + stage_phase

```text
find_phase                        stage_phase  (always background)
─────────                         ───────────
G2: match_blocks/scan_matches     commit(committed) → finish_commits
  └─ ImmutableBlock<G2>             ├─ make_available(g2_blocks)
(if tiers.g1)                       │
G1: match_prefix/scan_matches       ├─ stage_to_g2 → temporary G2 blocks
  └─ pinned ImmutableBlock<G1>      ├─ make_available(staged)
(if tiers.g3, Scatter only)         │
G3: scan_matches                    ├─ stage_g3_to_g2 → new G2 blocks
  └─ ImmutableBlock<G3>             ├─ make_available(new_g2_blocks)
                                    └─ finish_availability
```

`find_phase` computes `committed` once, in request order, and every
per-tier block vector holds a disjoint subset of it in the same order.
The commit therefore names every selected hash before any copy starts.

`find_phase` is synchronous in body for G1/G2/G3 (in-memory lookups). It
is `async fn` for forward-compat with G4 scans in v1.1.

`stage_phase` is `async fn` because `stage_to_g2` and `stage_g3_to_g2`
await their transfer notifications.

Publication is tiered. The resident G2 batch goes out before the device
copy starts, so an attached puller reads the hits that need no copy
while the copy runs. Each tier publishes its own batch in request order.
The commit stream closes before the first batch publishes, so the
puller's `drain_committed` step finishes before any copy lands.

Failures in `stage_phase` call `Session::close(reason)`. That call
pushes `LifecycleEvent::Detached { reason: Some(reason) }` on the
lifecycle stream of the holder. The `SessionManager` watcher of the
holder consumes it and evicts the entry. The same call also enqueues
`Frame::CommitsClosed` and `Frame::Drained` on the wire. Any attached
puller observes these as `CommitDelta::Closed` and
`AvailabilityDelta::Drained`. The puller sees `LifecycleEvent::Detached
{ reason: None }` only when the Finalized sentinel lands. Finalization
waits for the last inbound `PullAck` while pulls are in flight. The
reason string never reaches the puller. `LifecycleEvent::Failed` does
not come from this path. It comes from an inbound `Frame::Error`, an
attach failure, or a velo stream error other than `SenderDropped`.

### Fail-fast: G3 without a parallel_worker

`stage_g3_to_g2` requires a configured `parallel_worker` on the leader.
If `find_phase` produces G3 matches but `leader.parallel_worker()` is
`None`, `open_transfer_session` returns
`ControlError::Internal("g3_requires_parallel_worker: …")` **before**
opening a session. Without this check, the response would promise G3
blocks that the background populator immediately fails to stage,
leaving the caller with a capability that points to a
teardown-in-progress session — a "usable-looking session that cannot
serve blocks".

The check fires only when G3 matches were actually produced. A
`tiers.g3 = true` request that finds nothing in G3 (or whose leader has
no `g3_manager`) is fine on a no-worker leader — there is nothing to
stage.

## Puller pump loop

`pull_from_session` wraps:

1. `factory.attach(session_id, source_instance_id, endpoint)`.
2. Drain `commits()` to build the committed set; stop on `CommitDelta::Closed`.
3. Validate `selector ⊆ committed` (if a selector was supplied). Any
   hash in the selector that is not committed surfaces as a
   `hashes_not_committed` error.
4. Drain `availability()` per batch:
   - Filter to selected, not-yet-pulled hashes.
   - Allocate G2 mutables from the puller's local `g2_manager`.
   - `session.pull(hashes, dst)` → RDMA into the mutables.
   - `mutable.stage(hash, block_size)` → `complete` → `register_blocks`.
5. Break when every targeted hash is pulled or availability `Drained`.
6. `session.finalize(None)` for cooperative shutdown.

Pulled blocks become `ImmutableBlock<G2>` in the puller's registry —
discoverable by subsequent `match_blocks` / `scan_matches` calls on
the puller's `g2_manager`.

## Lifecycle

A session opened by `open_session` is registered in the leader's
`SessionManager`. It is evicted when:

- The session's lifecycle stream emits `Detached` (peer detach or
  populator failure) or `Failed` (protocol, attach, or stream error).
- The `SessionManager` watchdog timeout elapses (default 30s) without
  the peer attaching or any lifecycle event.
- An explicit `close_session` removes it.

## Audit events

All emitted on the `kvbm_audit` tracing target.

| Event | Fields |
|---|---|
| `transfer_session_no_matches` | `requested`, `search_mode` |
| `transfer_session_opened` | `session_id`, `find_mode`, `search_mode`, `committed`, `g1_hits`, `g2_hits`, `g3_hits` |
| `transfer_populator_complete` | `session_id` |
| `transfer_populator_failed` | `session_id`, `error` |
| `transfer_pull_started` | `session_id`, `source`, `selector_present` |
| `transfer_pull_completed` | `session_id`, `pulled` |
| `transfer_session_closed` | `session_id`, `was_present`, `reason` |

## Limitations (v1)

- **Single attach.** `disagg::session::velo` allows only one attached
  puller per holder-side session today (a second `Frame::Attach`
  overwrites `peer_instance_id` at `disagg/session/velo.rs:416`). Two
  concurrent `pull_from_session` calls against the same `session_id`
  will race.
- **No `extend_session` / `session_status`.** `SessionManager` has a
  fixed watchdog at construction time; `disagg::Session` doesn't expose
  state snapshots. Both are v1.1.
- **No hub-registry endpoint resolution.** `pull_from_session` with
  `endpoint: None` errors with `endpoint_required`. Hub-orchestrated
  workflows pass `Some(endpoint)` from the open response; resolving via
  the hub peer registry is v1.1.
- **G4 not yet wired** through the populator; `tiers.g4 = true` is
  currently a no-op.
- **`Prefix` mode reaches G2 and G1 only.** G3 is consulted only in
  `Scatter` mode in v1.

See `~/.claude/plans/control-transfer-v1.md` for the full design and
phased build trace.
