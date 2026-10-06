use proptest::prelude::*;

use crate::KvbmSequenceHashProvider;
use crate::blocks::ImmutableBlock;
use crate::manager::BlockManager;
use crate::testing::{TestMeta, create_iota_token_block, create_test_manager};

use super::{RebindPlan, SlotKind};

struct ModelPlan {
    plan: RebindPlan<TestMeta>,
    copied: bool,
    seq_hash: crate::SequenceHash,
    src: usize,
    dst: usize,
}

fn assert_model_invariants(
    manager: &BlockManager<TestMeta>,
    capacity: usize,
    content: &[Option<crate::SequenceHash>],
    plans: &[Option<ModelPlan>],
) {
    let snapshot = manager.store.debug_snapshot();
    let ceiling = manager.allocation_ceiling();
    assert_eq!(snapshot.slots.len(), capacity);
    assert_eq!(content.len(), capacity);

    for id in 0..capacity {
        let is_free = snapshot.free.contains(&id);
        let is_fenced = snapshot.fenced.contains(&id);
        if matches!(&snapshot.slots[id], SlotKind::Reset) {
            assert_ne!(is_free, is_fenced);
            if is_free {
                assert!(id < ceiling);
            } else {
                assert!(id >= ceiling);
            }
        } else {
            assert!(!is_free && !is_fenced);
        }

        match &snapshot.slots[id] {
            SlotKind::Primary(hash)
            | SlotKind::Duplicate(hash)
            | SlotKind::Inactive(hash)
            | SlotKind::Held(hash) => assert_eq!(content[id], Some(*hash)),
            SlotKind::Reset | SlotKind::Mutable | SlotKind::Staged(_) => {}
        }
    }

    for (&seq_hash, &block_id) in &snapshot.active_by_hash {
        assert_eq!(&snapshot.slots[block_id], &SlotKind::Primary(seq_hash));
    }
    assert_eq!(
        snapshot
            .slots
            .iter()
            .filter(|slot| matches!(slot, SlotKind::Inactive(_)))
            .count(),
        manager.inactive_len()
    );
    assert_eq!(
        snapshot.free.len() + snapshot.fenced.len() + manager.occupied_blocks(),
        capacity
    );

    let mut destinations = std::collections::HashSet::new();
    for record in plans.iter().flatten() {
        assert!(destinations.insert(record.plan.dst()));
        assert_eq!(record.src, record.plan.src());
        assert_eq!(record.dst, record.plan.dst());
        assert_eq!(record.seq_hash, record.plan.sequence_hash());
        assert!(matches!(
            &snapshot.slots[record.dst],
            SlotKind::Mutable
        ));
        assert!(!snapshot.free.contains(&record.dst));
        assert!(!snapshot.fenced.contains(&record.dst));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    #[test]
    fn random_store_operations_preserve_rebind_and_content_invariants(
        operations in prop::collection::vec((0u8..10, any::<u8>()), 0..80),
    ) {
        let capacity = 8;
        let manager = create_test_manager::<TestMeta>(capacity);
        let mut content = vec![None; capacity];
        let mut hashes = Vec::new();
        let mut guards: Vec<Option<ImmutableBlock<TestMeta>>> = Vec::new();
        let mut plans: Vec<Option<ModelPlan>> = Vec::new();
        let mut next_token = 0u32;

        for (operation, argument) in operations {
            match operation {
                0 => {
                    let token = create_iota_token_block(next_token * 4, 4);
                    next_token += 1;
                    let seq_hash = token.kvbm_sequence_hash();
                    if let Some((mutables, _)) = manager.allocate_blocks_with_evictions(1) {
                        let mutable = mutables.into_iter().next().unwrap();
                        let block_id = mutable.block_id();
                        assert!(block_id < manager.allocation_ceiling());
                        content[block_id] = None;
                        let immutable = manager.register_block(mutable.complete(&token).unwrap());
                        content[block_id] = Some(seq_hash);
                        hashes.push(seq_hash);
                        guards.push(Some(immutable));
                    }
                }
                1 => {
                    if !guards.is_empty() {
                        let index = argument as usize % guards.len();
                        drop(guards[index].take());
                    }
                }
                2 => {
                    if !hashes.is_empty() {
                        let seq_hash = hashes[argument as usize % hashes.len()];
                        for matched in manager.match_blocks(&[seq_hash]) {
                            assert_eq!(content[matched.block_id()], Some(seq_hash));
                        }
                    }
                }
                3 => {
                    let count = argument as usize % 2 + 1;
                    if let Some((mutables, _)) = manager.allocate_blocks_with_evictions(count) {
                        for mutable in mutables {
                            assert!(mutable.block_id() < manager.allocation_ceiling());
                            content[mutable.block_id()] = None;
                            drop(mutable);
                        }
                    }
                }
                4 => {
                    let (mutables, _) = manager.store.drain_inactive_to_mutable();
                    for mutable in mutables {
                        content[mutable.block_id()] = None;
                        drop(mutable);
                    }
                }
                5 => {
                    let fence = if argument % 3 == 0 {
                        None
                    } else {
                        Some(argument as usize % (capacity + 1))
                    };
                    manager.set_allocation_ceiling(fence).unwrap();
                }
                6 => {
                    if let Ok(plan) = manager.prepare_rebind(argument as usize % capacity) {
                        let src = plan.src();
                        let dst = plan.dst();
                        let seq_hash = plan.sequence_hash();
                        assert!(dst < manager.allocation_ceiling());
                        content[dst] = None;
                        plans.push(Some(ModelPlan {
                            plan,
                            copied: false,
                            seq_hash,
                            src,
                            dst,
                        }));
                    }
                }
                7 => {
                    if !plans.is_empty() {
                        if let Some(record) = plans[argument as usize % plans.len()].as_mut() {
                            if !record.copied {
                                content[record.dst] = content[record.src];
                                record.copied = true;
                            }
                        }
                    }
                }
                8 => {
                    if !plans.is_empty() {
                        let index = argument as usize % plans.len();
                        if plans[index].as_ref().is_some_and(|record| record.copied) {
                            let ModelPlan { plan, copied: _, seq_hash, src, dst } =
                                plans[index].take().unwrap();
                            match plan.commit() {
                                super::RebindOutcome::Moved {
                                    seq_hash: moved_hash,
                                    src: moved_src,
                                    dst: moved_dst,
                                } => {
                                    assert_eq!((moved_hash, moved_src, moved_dst), (seq_hash, src, dst));
                                    assert_eq!(content[dst], Some(seq_hash));
                                }
                                super::RebindOutcome::Busy(plan) => {
                                    plans[index] = Some(ModelPlan {
                                        plan,
                                        copied: true,
                                        seq_hash,
                                        src,
                                        dst,
                                    });
                                }
                                super::RebindOutcome::Stale | super::RebindOutcome::Fenced => {}
                            }
                        }
                    }
                }
                _ => {
                    if !plans.is_empty() {
                        let index = argument as usize % plans.len();
                        drop(plans[index].take());
                    }
                }
            }

            assert_model_invariants(&manager, capacity, &content, &plans);
        }
    }
}
