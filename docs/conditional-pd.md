# Conditional Prefill/Decode Disaggregation in Dynamo

## TL;DR

Standard disaggregated serving pins every request to a fixed prefill→decode pipeline, even when the cache state would make a same-worker prefill+decode cheaper. Dynamo's conditional P/D evaluates each request individually and uses a remote prefill worker only when the policy decides it's worth it.

Routing is policy-driven rather than role-fixed. Any worker can serve as prefill or decode on demand, so the planner rebalances to changing traffic shape **without a model restart** — by adjusting which worker IDs sit in the decode pool vs the prefill pool.

## Why this is different from existing P/D

Current P/D solutions (vLLM, TRT-LLM, our prior R-A pipeline) integrate at the **request-orchestration layer**:
- Run prefill to completion on worker P
- Hold P's KV memory
- Schedule decode on worker D
- D pulls KV from P's held memory

Conditional P/D integrates one layer down — **inside the connector** — by extending the vLLM / TRT-LLM connector API rather than wrapping the engine. That buys us:

1. **Per-request routing decisions** instead of fixed role assignments
2. **Cache-aware policy** with visibility into all KV tiers (defined below)
3. **No KV memory pinning on the prefill side** beyond the duration of a chunk
4. **Streaming KV transfer** as prefill chunks complete — not a batch transfer at end

## KV cache tiers (terminology)

- **G1** — engine block table (HBM, on-device)
- **G2** — host-memory offload tier
- **G3** — NIXL fabric (peer-accessible KV across workers)
- **G4** — remote shared store

The conditional policy sees matches across all four tiers when deciding local vs remote prefill.

## Request lifecycle

Every request enters the **decode router** first. Routing to decode is **unchanged** from aggregated mode.

### On the decode worker

1. Scheduler evaluates G1 match (local engine cache)
2. Connector evaluates G2/G3/G4 match (offload / fabric / remote tiers)
3. **Conditional policy** returns either `::PrefillLocal` or `::PrefillRemote`
   - Policy is behind a trait — trivial impls use cache-hit thresholds; a sophisticated impl could tap planner state for queue-depth / utilization-aware decisions
4. If `::PrefillLocal` → request runs as a normal aggregated request on the decode worker
5. If `::PrefillRemote`:
   - Decode worker creates a **session** (the central new primitive — see below)
   - Decode worker registers its G2 blocks into the session
   - Decode worker builds a prefill request containing:
     - token IDs
     - matched blocks already available on decode
     - decode worker ID
     - session ID
   - Decode worker sends the prefill request to the **prefill router**
   - Decode worker monitors the session; as each block becomes available, it issues an RDMA pull from prefill → decode

### On the prefill worker

1. Prefill router selects an instance; the extra metadata rides in **TransferParameters** (existing P/D mechanism — no new RPC surface)
2. Prefill runs the request like any normal request: scheduler G1 match, connector G2/G3/G4 match
3. **New:** connector detects TransferParameters
4. Connector evaluates local matches vs matches on decode; optionally pulls missing blocks from decode → prefill
5. Chunked prefill forward passes execute in a loop. As each chunk completes:
   - Connector offloads G1 → G2 for that chunk
   - G2 completion fires a callback that pushes those blocks into the session
   - The session emits events that the decode worker consumes to trigger its pull

The result: KV streams from prefill to decode chunk-by-chunk, overlapped with the prefill compute, with no end-of-prefill memory hold.

## Key architectural concepts

**Session.** A bidirectional event stream shared between the decode and prefill worker for the lifetime of one request. Coordinates KV availability and triggers NIXL ops on both sides. This is the only new cross-worker primitive.

**TransferParameters.** Existing P/D plumbing. We encode the new fields (decode worker ID, session ID, decode-side matched blocks) into it — no new request schema.

**Workers are just aggregated workers.** There is no "prefill worker binary" or "decode worker binary." Both run the same image, same engine, same connector. A worker's role is determined entirely by which router sends it traffic:
- Request **without** TransferParameters → decode request
- Request **with** TransferParameters → prefill request

**Routing pools.** Decode and prefill are pools of worker IDs maintained by the routers. The planner can move a worker between pools instantly — no restart, no reload, no warmup. The next request after the move sees the new role.

## Implementation surface

| Component | Change |
|---|---|
| Connector (vLLM + TRT-LLM) | G2/G3/G4 match reporting, conditional-policy hook, session create/consume, chunk-completion callbacks |
| Prefill router | Minor changes to request construction and dispatch |
| Decode router | **Unchanged** from aggregated mode |
| Engine code | **Untouched** |
| Request schema | **Unchanged** — piggybacks on TransferParameters |

## Operational benefits

1. **Per-request economics.** Cheap requests stay local; expensive ones (long context, low cache hit) take a remote prefill.
2. **Instant rebalancing.** Planner adjusts pool membership; effect is immediate, no restart.
3. **One binary, one engine.** No fragmented prefill/decode codepaths.
4. **Composable with planner intelligence.** The policy trait lets a smarter planner inject hints without changing the connector.
