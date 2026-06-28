// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use kvbm_common::StripedBlockPlacement;

#[test]
fn tp2_stripes_global_blocks_without_duplicates() {
    let placement = StripedBlockPlacement::new(2).expect("valid TP size");

    assert_eq!(placement.resolve(0), (0, 0));
    assert_eq!(placement.resolve(1), (1, 0));
    assert_eq!(placement.resolve(2), (0, 1));
    assert_eq!(placement.resolve(3), (1, 1));
    assert_eq!(placement.resolve(4), (0, 2));

    assert_eq!(placement.local_capacity(5, 0).unwrap(), 3);
    assert_eq!(placement.local_capacity(5, 1).unwrap(), 2);
}

#[test]
fn inverse_mapping_preserves_global_identity() {
    let placement = StripedBlockPlacement::new(4).expect("valid TP size");

    for global in 0..37 {
        let (owner, local) = placement.resolve(global);
        assert_eq!(placement.global(owner, local).unwrap(), global);
    }
}

#[test]
fn rejects_invalid_world_or_rank() {
    assert!(StripedBlockPlacement::new(0).is_err());

    let placement = StripedBlockPlacement::new(2).expect("valid TP size");
    assert!(placement.local_capacity(8, 2).is_err());
    assert!(placement.global(2, 0).is_err());
}

#[test]
fn aggregates_equal_per_rank_capacity() {
    let placement = StripedBlockPlacement::new(2).unwrap();

    assert_eq!(placement.global_capacity(512).unwrap(), 1024);
    assert!(placement.global_capacity(usize::MAX).is_err());
}
