// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Base page admission for a resource with more than one backing pool.

use super::*;

const BLOCKS: usize = 3;

#[test]
fn base_capacity_refuses_one_more_page_without_dropping_pins() {
    let manager = TestManagerBuilder::<G2>::new()
        .block_count(BLOCKS)
        .block_size(4)
        .build();
    let pins = manager.allocate_blocks(BLOCKS).expect("N pages fit");

    assert_eq!(pins.len(), BLOCKS);
    assert!(
        manager.allocate_blocks(1).is_none(),
        "admission must refuse N+1 pages"
    );
    assert_eq!(
        manager.occupied_blocks(),
        BLOCKS,
        "the refused page must not invalidate existing pins"
    );
    assert_eq!(
        pins.iter()
            .map(|block| block.block_id())
            .collect::<Vec<_>>(),
        (0..BLOCKS).collect::<Vec<_>>()
    );
}
