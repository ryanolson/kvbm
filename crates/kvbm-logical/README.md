# kvbm-logical

Logical block lifecycle management for KVBM (KV Block Manager). Manages KV cache blocks for LLM inference through a type-safe state machine, registry, and pool system.

## Block Lifecycle

Blocks follow a compile-time enforced state machine via the type-state pattern:

```text
MutableBlock<T> → CompleteBlock<T> → ImmutableBlock<T> ⇄ WeakBlock<T>
   (Reset)           (Staged)          (Registered)       (Non-owning)
```

- **MutableBlock** — Allocated from the reset pool, writable. Drop returns to the reset pool.
- **CompleteBlock** — Staged with a `SequenceHash` but not yet registered. Drop returns to the reset pool.
- **ImmutableBlock** — A strong reference prevents eviction. The last release caches a primary in the inactive pool by default. A duplicate or reset-on-release primary returns to the free pool.
- **WeakBlock** — Non-owning reference that does not prevent eviction. Upgradeable back to `ImmutableBlock` via two-phase lookup.

The type parameter `T: BlockMetadata` is a marker for the storage tier (e.g. GPU, CPU, disk).

## Temporary blocks for transfer

`CompleteBlock::set_evict_on_reset(true)` selects the free pool for the staged slot after its last registered owner releases it. The setter does not register the block or change an existing primary with the same hash. The caller must wait for physical writes before registration. The next mutable allocation restores the manager default. `CompleteBlock::reset()` also restores that default.

With `BlockDuplicationPolicy::Reject`, registration can return an existing primary instead of the staged destination. A temporary registration preserves that primary's retention flag. In contrast, `ImmutableBlock::set_evict_on_reset` changes the flag that all owners of the returned primary share.

A retention policy can adopt a temporary primary through `ImmutableBlock::set_evict_on_reset(false)`. Registration alone does not adopt it. The session retains its pins until each authorized physical transfer settles. The final pin release returns an unadopted temporary primary to the free pool, not the inactive pool.

This API supports G1-to-G2 copies for remote sessions. `PolicyG1G2BoundRoute::stage_to_g2` in kvbm-engine is the live consumer. The holder-side session source in kvbm-engine/src/p2p/g1_source.rs reaches it through stage_to_g2. Temporary registration does not change the source G1 retention policy or provide isolated session visibility. A registered temporary primary remains matchable while a pin holds it.

## Usage

```rust,no_run
use kvbm_logical::{
    BlockManager, BlockRegistry, MutableBlock, CompleteBlock, ImmutableBlock, WeakBlock,
    SequenceHash,
    manager::FrequencyTrackingCapacity,
};

# fn main() {
// Any Clone + Send + Sync + 'static type satisfies BlockMetadata.
#[derive(Clone)]
struct G2; // CPU tier marker

// Build a registry with TinyLFU frequency tracking.
let tracker = FrequencyTrackingCapacity::Medium.create_tracker();
let registry = BlockRegistry::builder()
    .frequency_tracker(tracker)
    .build();

// Build the block manager with an LRU eviction backend.
let manager = BlockManager::<G2>::builder()
    .block_count(1024)
    .block_size(16)
    .registry(registry)
    .with_lru_backend()
    .build()
    .expect("failed to build block manager");

// Allocate mutable blocks from the reset pool.
let mut blocks: Vec<MutableBlock<G2>> = manager
    .allocate_blocks(2)
    .expect("not enough blocks available");

// Stage a block with a pre-computed sequence hash, producing a CompleteBlock.
// SequenceHash wraps a positional lineage hash computed from token data.
let seq_hash_0 = SequenceHash::new(42, None, 0);
let complete: CompleteBlock<G2> = blocks
    .remove(0)
    .stage(seq_hash_0, manager.block_size())
    .expect("block size should match");

// Register the staged block, producing an ImmutableBlock.
let immutable: ImmutableBlock<G2> = manager.register_block(complete);

// Prefix-match registered blocks by sequence hash.
let matched: Vec<ImmutableBlock<G2>> = manager.match_blocks(&[seq_hash_0]);
assert_eq!(matched.len(), 1);

// Downgrade to a WeakBlock (does not prevent eviction).
let weak: WeakBlock<G2> = immutable.downgrade();

// Upgrade back to ImmutableBlock if the block hasn't been evicted.
if let Some(restored) = weak.upgrade() {
    assert_eq!(restored.sequence_hash(), seq_hash_0);
}

// RAII: dropping an ImmutableBlock moves it to the inactive pool for caching.
{
    let temporary = manager.match_blocks(&[seq_hash_0]);
    // `temporary` dropped here → block returns to inactive pool
}

// Introspect pool state.
let available = manager.available_blocks();
let total = manager.total_blocks();
# }
```

## Prometheus Metrics

All metrics carry a `pool` label identifying the storage tier.

### Counters

| Name | Description |
|------|-------------|
| `kvbm_allocations_total` | Total blocks allocated from pools |
| `kvbm_allocations_from_reset_total` | Total blocks allocated from the reset pool |
| `kvbm_evictions_total` | Total blocks evicted from inactive pool |
| `kvbm_registrations_total` | Total blocks registered (CompleteBlock → ImmutableBlock) |
| `kvbm_duplicate_blocks_total` | Total duplicate blocks created (Allow policy) |
| `kvbm_registration_dedup_total` | Total block registrations deduplicated (Reject policy) |
| `kvbm_stagings_total` | Total MutableBlock → CompleteBlock transitions |
| `kvbm_match_hashes_requested_total` | Total hashes requested in match_blocks calls |
| `kvbm_match_blocks_returned_total` | Total blocks returned from match_blocks calls |
| `kvbm_scan_hashes_requested_total` | Total hashes requested in scan_matches calls |
| `kvbm_scan_blocks_returned_total` | Total blocks returned from scan_matches calls |

### Gauges

| Name | Description |
|------|-------------|
| `kvbm_inflight_mutable` | Current MutableBlocks held outside pool |
| `kvbm_inflight_immutable` | Current ImmutableBlocks held outside pool |
| `kvbm_held_residency` | Current registered slots held by a pressure action |
| `kvbm_reset_pool_size` | Current reset pool size |
| `kvbm_inactive_pool_size` | Current inactive pool size |
