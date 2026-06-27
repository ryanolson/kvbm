# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer A: dependency bring-up.

Aggregated KVBM returns an empty handle — discovery defaults to None per
`crates/kvbm-config/src/messenger.rs:43` so single-process agg mode needs
no external services.

External-attach mode: when `KVBM_EXTERNAL_BASE_URL` is set the fixture
short-circuits to the same empty handle — the long-lived external server
already brought up its own deps.
"""

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class DepsHandle:
    """Layer-A handle.

    Aggregated KVBM does not require a fixture-managed NATS or etcd service.
    The optional fields remain so callers can log a uniform handle shape.
    """

    nats_url: Optional[str] = None
    etcd_endpoints: Optional[str] = None


@pytest.fixture(scope="function")
def kvbm_deps() -> DepsHandle:
    """Dependency bring-up for KVBM.

    The fixture exists to preserve the three-layer test structure used by
    external attach and server-spawn workflows.
    """
    return DepsHandle()
