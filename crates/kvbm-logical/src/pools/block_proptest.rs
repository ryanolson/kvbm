// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Property-based tests for guard state transitions through the unified store.

use super::tests::*;
use crate::blocks::*;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::testing::config::{
        COMMON_TEST_BLOCK_SIZES, generate_test_tokens, validate_test_block_size,
    };
    use crate::testing::{TestPoolSetupBuilder, create_test_token_block};

    use proptest::prelude::*;

    proptest! {
        /// Property: complete() on a token block of mismatched size returns
        /// `BlockSizeMismatch` with the original MutableBlock recoverable.
        #[test]
        fn prop_complete_mismatch_returns_block(
            block_size in prop::sample::select(COMMON_TEST_BLOCK_SIZES),
            wrong_size in prop::sample::select(&[1usize, 2, 8, 16, 32]),
        ) {
            prop_assume!(validate_test_block_size(block_size));
            prop_assume!(wrong_size != block_size);
            prop_assume!(validate_test_block_size(wrong_size));

            let store = TestPoolSetupBuilder::default()
                .block_count(2)
                .block_size(block_size)
                .build()
                .unwrap()
                .build_store::<TestData>();

            let mut blocks = store.allocate_reset_blocks(1);
            let mutable = blocks.pop().unwrap();
            let original_id = mutable.block_id();

            let tokens = generate_test_tokens(100, wrong_size);
            let tb = create_test_token_block(&tokens, wrong_size as u32);
            let result = mutable.complete(&tb);

            // Exhaustive match: if `BlockError` grows a new variant, this
            // fails to compile rather than silently passing the property.
            match result {
                Ok(_) => prop_assert!(false, "expected BlockSizeMismatch, got Ok"),
                Err(BlockError::BlockSizeMismatch { expected, actual, block: recovered }) => {
                    prop_assert_eq!(expected, block_size);
                    prop_assert_eq!(actual, wrong_size);
                    let recovered: MutableBlock<TestData> = recovered;
                    prop_assert_eq!(recovered.block_id(), original_id);
                    prop_assert_eq!(recovered.block_size(), block_size);
                }
            }
        }

        /// Property: block_id and block_size are preserved through
        /// Reset → Staged → Reset round-trips.
        #[test]
        fn prop_state_transitions_preserve_properties(
            block_size in prop::sample::select(&[1usize, 4, 16, 64]),
            base_token in 0u32..1000u32,
        ) {
            prop_assume!(validate_test_block_size(block_size));

            let store = TestPoolSetupBuilder::default()
                .block_count(2)
                .block_size(block_size)
                .build()
                .unwrap()
                .build_store::<TestData>();

            let mut blocks = store.allocate_reset_blocks(1);
            let mutable = blocks.pop().unwrap();
            let id = mutable.block_id();
            prop_assert_eq!(mutable.block_size(), block_size);

            let tokens = generate_test_tokens(base_token, block_size);
            let tb = create_test_token_block(&tokens, block_size as u32);
            let complete = mutable.complete(&tb).expect("complete should succeed");
            prop_assert_eq!(complete.block_id(), id);
            prop_assert_eq!(complete.block_size(), block_size);
            let seq_hash = complete.sequence_hash();

            let reset_again = complete.reset();
            prop_assert_eq!(reset_again.block_id(), id);
            prop_assert_eq!(reset_again.block_size(), block_size);

            // Re-stage and ensure the same hash recomputes deterministically.
            let tb2 = create_test_token_block(
                &generate_test_tokens(base_token, block_size),
                block_size as u32,
            );
            let staged = reset_again.complete(&tb2).expect("complete should succeed");
            prop_assert_eq!(staged.sequence_hash(), seq_hash);
        }

        /// Property: every common block size yields a usable store and a
        /// complete-able guard.
        #[test]
        fn prop_valid_block_sizes_work(
            block_size in prop::sample::select(&[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
        ) {
            prop_assert!(validate_test_block_size(block_size));

            let store = TestPoolSetupBuilder::default()
                .block_count(1)
                .block_size(block_size)
                .build()
                .unwrap()
                .build_store::<TestData>();

            let mut blocks = store.allocate_reset_blocks(1);
            let mutable = blocks.pop().unwrap();
            prop_assert_eq!(mutable.block_size(), block_size);

            let tokens = generate_test_tokens(0, block_size);
            let tb = create_test_token_block(&tokens, block_size as u32);
            prop_assert!(mutable.complete(&tb).is_ok());
        }
    }

    mod focused_properties {
        use super::*;

        proptest! {
            /// Property: identical token sequences hash identically regardless
            /// of which slot they're staged through.
            #[test]
            fn prop_sequence_hash_deterministic(
                tokens in prop::collection::vec(any::<u32>(), 4..=4),
            ) {
                let store = TestPoolSetupBuilder::default()
                    .block_count(2)
                    .block_size(4)
                    .build()
                    .unwrap()
                    .build_store::<TestData>();

                let mut allocated = store.allocate_reset_blocks(2);
                let m2 = allocated.pop().unwrap();
                let m1 = allocated.pop().unwrap();

                let tb1 = create_test_token_block(&tokens, 4);
                let tb2 = create_test_token_block(&tokens, 4);

                let c1 = m1.complete(&tb1).expect("complete");
                let c2 = m2.complete(&tb2).expect("complete");

                prop_assert_eq!(c1.sequence_hash(), c2.sequence_hash());
            }

            /// Property: distinct token sequences produce distinct hashes.
            #[test]
            fn prop_different_tokens_different_hashes(
                tokens1 in prop::collection::vec(0u32..100u32, 4..=4),
                tokens2 in prop::collection::vec(100u32..200u32, 4..=4),
            ) {
                prop_assume!(tokens1 != tokens2);

                let store = TestPoolSetupBuilder::default()
                    .block_count(2)
                    .block_size(4)
                    .build()
                    .unwrap()
                    .build_store::<TestData>();

                let mut allocated = store.allocate_reset_blocks(2);
                let m2 = allocated.pop().unwrap();
                let m1 = allocated.pop().unwrap();

                let tb1 = create_test_token_block(&tokens1, 4);
                let tb2 = create_test_token_block(&tokens2, 4);

                let c1 = m1.complete(&tb1).expect("complete");
                let c2 = m2.complete(&tb2).expect("complete");

                prop_assert_ne!(c1.sequence_hash(), c2.sequence_hash());
            }
        }
    }

    /// Tests for `BlockStore::release_blocks` (batched release) — the
    /// state-equivalence proptest is the load-bearing correctness
    /// contract; the rest are targeted unit tests for specific
    /// behaviours called out in the design (duplicate fallback, deferred
    /// entries, report counts).
    mod release_batch {
        use super::*;

        use crate::pools::BlockDuplicationPolicy;
        use crate::pools::store::{BlockStore, DebugStoreSnapshot, ReleaseOpts, SlotKind};
        use crate::registry::BlockRegistry;
        use std::sync::Arc;

        /// Register a fresh block for `seq_hash` against `store`/`registry`.
        /// Uses `MutableBlock::stage` with a directly-supplied
        /// `SequenceHash` (no real token block needed) — mirrors the
        /// `SequenceHash::new(..)` synthetic-hash pattern used by
        /// `store.rs`'s own unit tests. `policy` only matters if
        /// `seq_hash` already has a live primary registered in `store`
        /// (i.e. this call produces a duplicate); for a fresh hash it is
        /// ignored.
        fn register_new(
            store: &Arc<BlockStore<TestData>>,
            registry: &BlockRegistry,
            seq_hash: SequenceHash,
            policy: BlockDuplicationPolicy,
        ) -> ImmutableBlock<TestData> {
            let mutable = store
                .allocate_reset_blocks(1)
                .pop()
                .expect("reset block available");
            let block_size = mutable.block_size();
            let complete = mutable.stage(seq_hash, block_size).expect("stage");
            let handle = registry.register_sequence_hash(seq_hash);
            let inner = handle.register_block(complete, policy, store);
            ImmutableBlock::from_inner(inner)
        }

        fn synth_hash(slot: usize) -> SequenceHash {
            SequenceHash::new(0x1000 + slot as u64, None, slot as u64)
        }

        /// Requirement #3 (duplicate-fallback), corrected: a batch
        /// containing both a primary and its live duplicate. Earlier,
        /// this function deferred the primary (its `Arc::try_unwrap`
        /// failed — the duplicate's `_primary_keepalive` held a second
        /// strong reference) and only released it later, via ordinary
        /// `Drop`'s field-glue cascade in phase 2. A proptest caught
        /// that this reordered the reset pool relative to one-at-a-time
        /// drops (see `release_blocks`'s docs), so `release_entry_at`
        /// now explicitly replays the keepalive cascade *inline*: both
        /// the duplicate and the primary release within the same lock
        /// acquisition, with the primary's release landing at the
        /// duplicate's position (the *later* of the two, matching
        /// one-at-a-time order) — no double-release, block count
        /// conserved.
        #[test]
        fn duplicate_and_its_primary_cascade_resolve_inline() {
            let store = TestPoolSetupBuilder::default()
                .block_count(2)
                .block_size(4)
                .build()
                .unwrap()
                .build_store::<TestData>();
            let registry = BlockRegistry::new();
            let hash = synth_hash(0);

            let primary = register_new(&store, &registry, hash, BlockDuplicationPolicy::Allow);
            let duplicate = register_new(&store, &registry, hash, BlockDuplicationPolicy::Allow);
            primary.set_evict_on_reset(true);

            let report = store.release_blocks(vec![primary, duplicate], ReleaseOpts::default());

            assert_eq!(report.deferred_to_drop, 0, "nothing left unresolved");
            assert_eq!(report.duplicate_reset, 1, "duplicate released inline");
            assert_eq!(
                report.primary_reset, 1,
                "primary's cascade also resolved inline (reset_on_release = true)"
            );
            assert_eq!(store.reset_len(), 2, "both slots back in the reset pool");
            assert_eq!(store.inactive_len(), 0);
            assert!(!store.has_inactive(hash));
        }

        /// The other half of the cascade-position rule: when the
        /// duplicate comes *before* the primary in the input `Vec`, the
        /// primary's own entry is what resolves the cascade (the
        /// duplicate merely decrements the keepalive it holds) — the
        /// release still lands at the *later* position (the primary's),
        /// matching one-at-a-time drop order (`drop(primary)` first is a
        /// no-op decrement; `drop(duplicate)`... here reversed: the
        /// duplicate drops first, decrementing the keepalive, and the
        /// primary's own drop is what then hits zero).
        #[test]
        fn duplicate_before_primary_in_input_order_still_resolves_inline() {
            let store = TestPoolSetupBuilder::default()
                .block_count(2)
                .block_size(4)
                .build()
                .unwrap()
                .build_store::<TestData>();
            let registry = BlockRegistry::new();
            let hash = synth_hash(0);

            let primary = register_new(&store, &registry, hash, BlockDuplicationPolicy::Allow);
            let duplicate = register_new(&store, &registry, hash, BlockDuplicationPolicy::Allow);
            primary.set_evict_on_reset(false);

            let report = store.release_blocks(vec![duplicate, primary], ReleaseOpts::default());

            assert_eq!(report.deferred_to_drop, 0);
            assert_eq!(report.duplicate_reset, 1);
            assert_eq!(
                report.primary_inactive, 1,
                "primary's cascade resolved inline (reset_on_release = false)"
            );
            assert_eq!(store.reset_len(), 1, "only the duplicate's slot");
            assert_eq!(store.inactive_len(), 1);
            assert!(store.has_inactive(hash));
        }

        /// Pins down the exact free-list *position* the "not yet
        /// visited" cascade branch (`j > idx` in `release_entry_at`) is
        /// responsible for: an independent, unrelated block released
        /// *between* a duplicate and its own (later, co-batched) primary
        /// in input order. One-at-a-time drop order is
        /// [duplicate, independent, primary] → the primary's cascade
        /// only fires when the *primary's own* drop runs (last), so it
        /// must land in the free list *after* the independent block, not
        /// before. (An earlier draft of the fix retried the primary
        /// immediately upon seeing the duplicate release, landing it
        /// *before* the independent block instead — this test's
        /// `assert_eq!` on the exact `free` sequence, not just its
        /// contents, is what would catch that regression; a set/sorted
        /// comparison would not.)
        #[test]
        fn duplicate_before_independent_before_primary_preserves_fifo_order() {
            let store = TestPoolSetupBuilder::default()
                .block_count(3)
                .block_size(4)
                .build()
                .unwrap()
                .build_store::<TestData>();
            let registry = BlockRegistry::new();
            let dup_hash = synth_hash(0);

            let primary = register_new(&store, &registry, dup_hash, BlockDuplicationPolicy::Allow);
            let independent = register_new(
                &store,
                &registry,
                synth_hash(1),
                BlockDuplicationPolicy::Allow,
            );
            let duplicate =
                register_new(&store, &registry, dup_hash, BlockDuplicationPolicy::Allow);
            primary.set_evict_on_reset(true);
            independent.set_evict_on_reset(true);

            let duplicate_id = duplicate.block_id();
            let independent_id = independent.block_id();
            let primary_id = primary.block_id();

            let report = store.release_blocks(
                vec![duplicate, independent, primary],
                ReleaseOpts::default(),
            );

            assert_eq!(report.deferred_to_drop, 0);
            assert_eq!(report.duplicate_reset, 1);
            assert_eq!(report.primary_reset, 2, "independent + cascaded primary");

            let free = &store.debug_snapshot().free;
            assert_eq!(
                free,
                &vec![duplicate_id, independent_id, primary_id],
                "primary's cascade must land after the independent block, matching \
                 one-at-a-time drop order — not immediately after the duplicate"
            );
        }

        /// The keepalive-cascade replay must not fire — and must not
        /// drop anything — for a duplicate whose primary is *not* part
        /// of this batch (kept alive by a separate, external guard).
        /// `release_entry_at` cannot prove that dropping the extracted
        /// keepalive wouldn't be the primary's last reference (which
        /// would reacquire the store lock while this call still holds
        /// it), so it must defer that drop to after the lock is
        /// released, same as a genuinely `Arc`-shared entry.
        #[test]
        fn duplicate_targeting_external_primary_defers_the_keepalive() {
            let store = TestPoolSetupBuilder::default()
                .block_count(2)
                .block_size(4)
                .build()
                .unwrap()
                .build_store::<TestData>();
            let registry = BlockRegistry::new();
            let hash = synth_hash(0);

            let primary = register_new(&store, &registry, hash, BlockDuplicationPolicy::Allow);
            let duplicate = register_new(&store, &registry, hash, BlockDuplicationPolicy::Allow);
            primary.set_evict_on_reset(true);

            // Only the duplicate goes through the batch; `primary` is
            // held externally (as if by some other part of the system).
            let report = store.release_blocks(vec![duplicate], ReleaseOpts::default());
            assert_eq!(report.duplicate_reset, 1);
            assert_eq!(report.deferred_to_drop, 0);
            assert_eq!(store.reset_len(), 1, "only the duplicate's own slot");

            // Primary still alive (its own guard hasn't dropped yet).
            assert!(!store.has_inactive(hash));

            drop(primary);
            assert_eq!(store.reset_len(), 2, "primary's ordinary Drop completes it");
        }

        /// `register_blocks` (batched register) must validate every
        /// block/handle hash pair *before* mutating or disarming any
        /// guard — matching the singular `register_block`'s
        /// validate-before-mutate ordering (registration.rs). A batch
        /// with a mismatched pair anywhere must panic before touching
        /// slot state, so every `CompleteBlock` in the batch is still
        /// armed when the panic unwinds and drops them — each one
        /// releasing `Staged → Reset` normally, with no half-applied
        /// batch and no stranded `Staged` slots.
        #[test]
        fn register_blocks_validates_before_mutating_any_guard() {
            let store = TestPoolSetupBuilder::default()
                .block_count(2)
                .block_size(4)
                .build()
                .unwrap()
                .build_store::<TestData>();
            let registry = BlockRegistry::new();

            let hash_a = synth_hash(0);
            let hash_b = synth_hash(1);
            let mismatched_hash = synth_hash(2);

            let mutable_a = store.allocate_reset_blocks(1).pop().expect("block a");
            let mutable_b = store.allocate_reset_blocks(1).pop().expect("block b");
            let block_size = mutable_a.block_size();
            let complete_a = mutable_a.stage(hash_a, block_size).expect("stage a");
            let complete_b = mutable_b.stage(hash_b, block_size).expect("stage b");

            // `complete_a`'s handle deliberately does not match its own
            // sequence hash.
            let handle_a = registry.register_sequence_hash(mismatched_hash);
            let handle_b = registry.register_sequence_hash(hash_b);

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                store.register_blocks(
                    vec![complete_a, complete_b],
                    vec![handle_a, handle_b],
                    BlockDuplicationPolicy::Allow,
                )
            }));
            assert!(result.is_err(), "mismatched hash pair must panic");

            // Both guards were still armed (never disarmed, since the
            // mismatch is caught before any mutation) so unwinding
            // `register_blocks`'s stack dropped them normally, returning
            // both slots to Reset — no stranded `Staged` slots, no
            // partially-applied batch.
            assert_eq!(store.reset_len(), 2, "both blocks safely reset on panic");
            assert_eq!(store.inactive_len(), 0);
            assert!(!registry.check_presence::<TestData>(&[hash_a])[0].1);
            assert!(!registry.check_presence::<TestData>(&[hash_b])[0].1);
        }

        /// Requirement #2, adjusted: the task text frames this as
        /// "identity-check under a racing eager Primary → Inactive
        /// transition", but `release_blocks` holds each `Arc` alive from
        /// extraction through the under-lock identity check, so
        /// `Weak::upgrade` can never fail for a block in flight through
        /// this function — the eager-transition race is structurally
        /// unreachable here (see the design-note comment on
        /// `release_blocks`). The reachable analogue: a block that is
        /// still `Arc`-shared (an outstanding clone *outside* the batch)
        /// is deferred rather than released inline, and its eventual
        /// ordinary `Drop` — once the outstanding clone also drops —
        /// completes the transition with no double-release.
        #[test]
        fn still_shared_block_is_deferred_then_released_by_ordinary_drop() {
            let store = TestPoolSetupBuilder::default()
                .block_count(1)
                .block_size(4)
                .build()
                .unwrap()
                .build_store::<TestData>();
            let registry = BlockRegistry::new();
            let hash = synth_hash(0);

            let primary = register_new(&store, &registry, hash, BlockDuplicationPolicy::Allow);
            primary.set_evict_on_reset(false);
            let outstanding = primary.clone();

            let report = store.release_blocks(vec![primary], ReleaseOpts::default());
            assert_eq!(report.deferred_to_drop, 1);
            assert_eq!(report.released(), 0);
            // Still alive: outstanding clone holds the slot as Primary.
            assert_eq!(store.reset_len(), 0);
            assert_eq!(store.inactive_len(), 0);
            assert!(!store.has_inactive(hash));

            drop(outstanding);
            // Now the only reference is gone; ordinary Drop must have
            // routed it to Inactive (reset_on_release was false).
            assert_eq!(store.reset_len(), 0);
            assert_eq!(store.inactive_len(), 1);
            assert!(store.has_inactive(hash));
        }

        /// Requirement #5: `ReleaseReport` counts for a mixed batch of
        /// *independent* (non-conflicting) blocks — the duplicate/primary
        /// pair here is deliberately kept independent of the batch's
        /// other entries (the primary is held externally, not co-batched)
        /// so the counts stay simple to state; the co-batched
        /// cascade-resolution case is covered by the
        /// `duplicate_and_its_primary_cascade_resolve_inline` and
        /// `duplicate_before_primary_in_input_order_still_resolves_inline`
        /// tests above.
        #[test]
        fn release_report_counts_for_mixed_independent_batch() {
            let store = TestPoolSetupBuilder::default()
                .block_count(6)
                .block_size(4)
                .build()
                .unwrap()
                .build_store::<TestData>();
            let registry = BlockRegistry::new();

            // Two independent primaries routed to Reset.
            let reset_a = register_new(
                &store,
                &registry,
                synth_hash(0),
                BlockDuplicationPolicy::Allow,
            );
            reset_a.set_evict_on_reset(true);
            let reset_b = register_new(
                &store,
                &registry,
                synth_hash(1),
                BlockDuplicationPolicy::Allow,
            );
            reset_b.set_evict_on_reset(true);

            // One independent primary routed to Inactive.
            let inactive_a = register_new(
                &store,
                &registry,
                synth_hash(2),
                BlockDuplicationPolicy::Allow,
            );
            inactive_a.set_evict_on_reset(false);

            // A primary (kept alive for the whole test) plus its
            // duplicate, so the duplicate can be released independently
            // (no cascade complicates the count: the primary is never
            // handed to `release_blocks`).
            let dup_hash = synth_hash(3);
            let dup_primary =
                register_new(&store, &registry, dup_hash, BlockDuplicationPolicy::Allow);
            let duplicate =
                register_new(&store, &registry, dup_hash, BlockDuplicationPolicy::Allow);

            // One still-shared entry: an outstanding clone kept alive
            // outside the batch.
            let shared = register_new(
                &store,
                &registry,
                synth_hash(4),
                BlockDuplicationPolicy::Allow,
            );
            let _shared_keepalive = shared.clone();

            let report = store.release_blocks(
                vec![reset_a, reset_b, inactive_a, duplicate, shared],
                ReleaseOpts::default(),
            );

            assert_eq!(report.primary_reset, 2);
            assert_eq!(report.primary_inactive, 1);
            assert_eq!(report.duplicate_reset, 1);
            assert_eq!(report.deferred_to_drop, 1);
            assert_eq!(report.released(), 4);

            drop(dup_primary);
        }

        // Thin alias so the proptest body reads a little less noisily.
        fn snapshot(store: &Arc<BlockStore<TestData>>) -> DebugStoreSnapshot {
            store.debug_snapshot()
        }

        proptest! {
            /// **Load-bearing**: for an arbitrary mix of blocks — fresh
            /// primaries, duplicates of an earlier primary in the same
            /// batch, a per-block `reset_on_release` override, and
            /// optionally an outstanding clone kept alive outside the
            /// release call — `release_blocks` must leave the store in a
            /// state indistinguishable from dropping the same guards, in
            /// the same order, one at a time via the pre-existing
            /// per-block path.
            ///
            /// Two comparison points, on two independently-constructed
            /// stores (identical `block_count`/`block_size`, so `BlockId`
            /// allocation lines up call-for-call): immediately after the
            /// release/drop pass (captures still-shared entries remaining
            /// alive identically on both sides), and after also dropping
            /// the outstanding "kept alive" clones in the same order on
            /// both sides (captures the final, fully-drained state,
            /// including any primary/duplicate keepalive cascades).
            #[test]
            fn prop_release_blocks_state_equivalent_to_per_block_drop(
                entries in prop::collection::vec(
                    (0usize..5, any::<bool>(), any::<bool>()),
                    2usize..=12,
                ),
            ) {
                let n = entries.len();
                let store_batch = TestPoolSetupBuilder::default()
                    .block_count(n)
                    .block_size(4)
                    .build()
                    .unwrap()
                    .build_store::<TestData>();
                let store_base = TestPoolSetupBuilder::default()
                    .block_count(n)
                    .block_size(4)
                    .build()
                    .unwrap()
                    .build_store::<TestData>();
                let registry_batch = BlockRegistry::new();
                let registry_base = BlockRegistry::new();

                let mut seen_slots = std::collections::HashSet::new();
                let mut batch_release = Vec::with_capacity(n);
                let mut base_release = Vec::with_capacity(n);
                let mut batch_kept = Vec::new();
                let mut base_kept = Vec::new();

                for &(slot, reset_flag, still_shared) in &entries {
                    // First occurrence of a slot registers a primary;
                    // later occurrences register a duplicate against the
                    // primary already live in *that store* — each store
                    // resolves its own duplicate independently, so the
                    // two stores never share any state, only the same
                    // sequence of operations.
                    seen_slots.insert(slot);
                    let hash = synth_hash(slot);

                    let b = register_new(&store_batch, &registry_batch, hash, BlockDuplicationPolicy::Allow);
                    let g = register_new(&store_base, &registry_base, hash, BlockDuplicationPolicy::Allow);
                    b.set_evict_on_reset(reset_flag);
                    g.set_evict_on_reset(reset_flag);

                    if still_shared {
                        batch_kept.push(b.clone());
                        base_kept.push(g.clone());
                    }
                    batch_release.push(b);
                    base_release.push(g);
                }

                let _report = store_batch.release_blocks(batch_release, ReleaseOpts::default());
                for block in base_release {
                    drop(block);
                }

                prop_assert_eq!(
                    snapshot(&store_batch),
                    snapshot(&store_base),
                    "mid-state (still-shared entries left alive) must match"
                );

                // Drop the outstanding "kept alive" clones, in the same
                // relative order, on both sides.
                for block in batch_kept {
                    drop(block);
                }
                for block in base_kept {
                    drop(block);
                }

                prop_assert_eq!(
                    snapshot(&store_batch),
                    snapshot(&store_base),
                    "fully-drained state must match"
                );

                // Sanity: every slot ended up Reset or Inactive — nothing
                // left dangling in Mutable/Staged/Primary/Duplicate.
                let final_snapshot = snapshot(&store_batch);
                for kind in &final_snapshot.slots {
                    prop_assert!(
                        matches!(kind, SlotKind::Reset) || matches!(kind, SlotKind::Inactive(_)),
                        "unexpected leftover slot kind: {kind:?}"
                    );
                }
            }
        }
    }
}
