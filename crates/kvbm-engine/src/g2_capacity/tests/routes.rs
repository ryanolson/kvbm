// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[test]
fn production_g2_routes_do_not_allocate_from_block_manager() {
    let routes = [
        ("leader/staging.rs", include_str!("../../leader/staging.rs")),
        (
            "p2p/pull_transaction/mod.rs",
            include_str!("../../p2p/pull_transaction/mod.rs"),
        ),
        (
            "tiering/engine/onboard.rs",
            include_str!("../../tiering/engine/onboard.rs"),
        ),
        (
            "remote/search/g4.rs",
            include_str!("../../remote/search/g4.rs"),
        ),
        (
            "tiering/offload/pipeline.rs",
            include_str!("../../tiering/offload/pipeline.rs"),
        ),
    ];

    for (path, source) in routes {
        assert!(
            !source.contains(".allocate_blocks("),
            "{path} must route G2 destination allocation through G2Capacity"
        );
    }
}
