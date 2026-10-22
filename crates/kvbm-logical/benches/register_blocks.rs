// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registration cost for each block: one `register_block` call for each
//! block versus one `register_blocks` batch, on the LRU and lineage
//! backends, with fresh hashes.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use kvbm_logical::BlockManager;
use kvbm_logical::blocks::CompleteBlock;
use kvbm_logical::registry::BlockRegistry;
use kvbm_logical::testing::TestMeta;

const BLOCK_SIZE: u32 = 4;

fn manager(block_count: usize, lineage: bool) -> BlockManager<TestMeta> {
    let builder = BlockManager::<TestMeta>::builder()
        .block_count(block_count)
        .block_size(BLOCK_SIZE as usize)
        .registry(BlockRegistry::new());
    let builder = if lineage {
        builder.with_lineage_backend()
    } else {
        builder.with_lru_backend()
    };
    builder.build().expect("build manager")
}

fn stage(
    manager: &BlockManager<TestMeta>,
    count: usize,
    salt: u64,
) -> Vec<CompleteBlock<TestMeta>> {
    let tokens: Vec<u32> = (0..(count as u32 * BLOCK_SIZE)).collect();
    let sequence = dynamo_tokens::TokenBlockSequence::from_slice(&tokens, BLOCK_SIZE, Some(salt));
    manager
        .allocate_blocks(count)
        .expect("allocate")
        .into_iter()
        .zip(sequence.blocks())
        .map(|(block, token_block)| block.complete(token_block).ok().expect("stage"))
        .collect()
}

fn bench_register(c: &mut Criterion) {
    let mut group = c.benchmark_group("register");
    for count in [64usize, 1024, 16384] {
        group.throughput(Throughput::Elements(count as u64));
        for lineage in [false, true] {
            let backend = if lineage { "lineage" } else { "lru" };
            for batch in [false, true] {
                let mode = if batch { "batch" } else { "per_block" };
                let manager = manager(2 * count, lineage);
                let mut salt = 0u64;
                group.bench_with_input(
                    BenchmarkId::new(format!("{backend}/{mode}"), count),
                    &count,
                    |b, &count| {
                        b.iter_batched(
                            || {
                                salt += 1;
                                stage(&manager, count, salt)
                            },
                            |staged| {
                                if batch {
                                    manager.register_blocks(staged)
                                } else {
                                    staged
                                        .into_iter()
                                        .map(|block| manager.register_block(block))
                                        .collect()
                                }
                            },
                            criterion::BatchSize::PerIteration,
                        );
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_register);
criterion_main!(benches);
