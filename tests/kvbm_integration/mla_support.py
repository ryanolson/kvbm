#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared assertions for basic MLA cache integration tests."""

import re
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from .common import fetch_kvbm_metrics


def assert_mla_round_trip(
    test_case: Any,
    kvbm_tester: Any,
    kvbm_server: Any,
    *,
    iterations: int,
    tensor_parallel_size: int,
) -> None:
    """Run determinism phases and verify compressed MLA G1/G2 traffic."""
    kvbm_tester.max_iterations = iterations
    metrics_port = kvbm_server.metrics_port

    time.sleep(2)
    m_start = fetch_kvbm_metrics(port=metrics_port)
    boundary: Dict[str, dict] = {}
    original_reset = kvbm_tester.reset_prefix_cache

    def reset_with_snapshot() -> None:
        boundary["phase1"] = fetch_kvbm_metrics(port=metrics_port)
        original_reset()

    kvbm_tester.reset_prefix_cache = reset_with_snapshot
    try:
        test_case.base_test_determinism_with_cache_reset(
            kvbm_tester, kvbm_server, None
        )
    finally:
        kvbm_tester.reset_prefix_cache = original_reset

    phase1 = boundary.get("phase1")
    if phase1 is None:
        pytest.fail("MLA phase boundary metrics were not captured")
    m_end = fetch_kvbm_metrics(port=metrics_port)

    offloaded = _metric_delta(phase1, m_start, "kvbm_offload_blocks_d2h")
    onboarded = _metric_delta(m_end, phase1, "kvbm_onboard_blocks_h2d")
    assert offloaded > 0, "MLA did not offload any G1 blocks to G2"
    assert onboarded > 0, "MLA did not onboard any G2 blocks to G1"

    log_text = _read_server_log(Path(kvbm_server.log_dir))
    assert re.search(r"use_mla=True", log_text)
    assert re.search(r"dims=\['Block', 'Page', '(?:HeadSize|Payload)'\]", log_text)
    registrations = re.findall(
        r"\[KVBM\] KV caches registered \(deferred mode\):.*", log_text
    )
    assert registrations, "KVBM MLA registration log entry was not found"
    assert all("HeadCount" not in registration for registration in registrations)

    if tensor_parallel_size > 1:
        assert "selecting replicated-data placement" in log_text
        assert len(re.findall(r"NCCL communicator initialized", log_text)) >= 2

    print(
        f"[mla-smoke] tp={tensor_parallel_size} offloaded={offloaded} "
        f"onboarded={onboarded} model={kvbm_server.model_config.model_id}"
    )


def _metric_delta(after: dict, before: dict, name: str) -> int:
    return int(after.get(name, 0)) - int(before.get(name, 0))


def _read_server_log(log_dir: Path) -> str:
    parts = []
    for path in sorted(log_dir.glob("*")):
        if path.is_file() and path.suffix in (".log", ".txt"):
            parts.append(path.read_text(errors="replace"))
    return "\n".join(parts)
