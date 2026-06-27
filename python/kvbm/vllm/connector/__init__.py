# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical KVBM vLLM connector façade — lazy re-export of the impl.

vLLM resolves ``kv_connector_module_path`` via
``importlib.import_module('kvbm.vllm.connector')`` + ``getattr(mod, 'KvbmConnector')``.
The lazy ``__getattr__`` shim defers the redirect to attribute access so simply
importing this module does NOT force ``import vllm`` (regression-critical: a
fresh-interp import of this package must not pull vllm into sys.modules).
"""

_EXPORTS = {
    "KvbmConnector": (".base", "KvbmConnector"),
    "KvbmConnectorMetadata": (".base", "KvbmConnectorMetadata"),
    "PdConnector": (".pd", "PdConnector"),
    "PdConnectorMetadata": (".pd", "PdConnectorMetadata"),
    "PdHandshakeMetadata": (".pd", "PdHandshakeMetadata"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        mod_path, attr = _EXPORTS[name]
    except KeyError as e:
        raise AttributeError(
            f"module 'kvbm.vllm.connector' has no attribute {name!r}"
        ) from e
    import importlib

    return getattr(importlib.import_module(mod_path, __name__), attr)


def __dir__():
    return sorted(set(list(globals().keys()) + list(_EXPORTS)))
