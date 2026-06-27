# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KVBM-aware launch wrappers and vLLM integration for KVBM.

Sub-modules:

- :mod:`kvbm.vllm.prefill` — `python -m kvbm.vllm.prefill` standalone
  OpenAI-API server that auto-attaches a `PrefillRouterHandler` to its
  engine when the rendered kv-transfer-config carries a hub URL and a
  prefill role.
- :mod:`kvbm.vllm.connector` — the vLLM KV-connector façade resolved by
  vLLM's ``kv_connector_module_path``.
"""

# version_check() is intentionally NOT called here: doing so at package
# init would force `import vllm` every time something merely imports
# `kvbm.vllm.connector` to scan the module path. The check runs lazily
# from `kvbm.vllm.config` instead (the gateway every vllm-touching path
# transits). KvbmVllmConfig is _core-only, so this import never forces vllm.
from kvbm._core import KvbmVllmConfig
from .version_check import version_check

__all__ = ["KvbmVllmConfig", "version_check"]
