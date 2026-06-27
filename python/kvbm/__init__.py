# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kvbm._core import (
    ConnectorLeader,
    ConnectorWorker,
    KvbmRequest,
    KvbmRuntime,
    KvbmVllmConfig,
    SchedulerOutput,
    Tensor,
    __version__ as __version__,
    is_available,
)
from kvbm._feature_stubs import _make_module_stub

_CORE_AVAILABLE = True

# kernels feature: kernels submodule (optional)
try:
    from kvbm._core import kernels as kernels

    _KERNELS_AVAILABLE = True
except ImportError:
    kernels = _make_module_stub("kvbm.kernels", "kernels")
    _KERNELS_AVAILABLE = False

__all__ = [
    "__version__",
    "is_available",
    "KvbmRuntime",
    "KvbmVllmConfig",
    "ConnectorLeader",
    "ConnectorWorker",
    "KvbmRequest",
    "SchedulerOutput",
    "Tensor",
    "kernels",
    "_CORE_AVAILABLE",
    "_KERNELS_AVAILABLE",
]
