# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test: importing kvbm.vllm.connector must NOT eagerly import vllm.

This test is DISCRIMINATING: it runs the assertion in a fresh interpreter
subprocess so sibling tests cannot pre-populate sys.modules.  An eager façade
would fail because base.py imports vllm at module level (line 19:
``from vllm.distributed.kv_transfer.kv_connector.v1.base import ...``); if
the __getattr__ indirection were removed and the import fell through to
base.py at import-time, "vllm" would appear in sys.modules and the assertion
would trigger.

The test would PASS (falsely) if run in-process after any test that already
imported vllm — hence the subprocess boundary.
"""

import subprocess
import sys


def test_connector_facade_does_not_force_vllm_import() -> None:
    """Importing kvbm.vllm.connector must leave 'vllm' out of sys.modules."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import kvbm.vllm.connector, sys; "
                # If the lazy __getattr__ is removed or the facade eagerly imports
                # base.py, 'vllm' will appear here because base.py has a top-level
                # `from vllm... import ...`.
                "assert 'vllm' not in sys.modules, "
                "'vllm was imported eagerly by kvbm.vllm.connector'; "
                # The impl module must NOT be touched either — only the facade module
                # itself should be loaded.
                "assert 'kvbm.vllm.connector.base' not in sys.modules, "
                "'kvbm.vllm.connector.base was imported eagerly (facade broke)'"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Lazy-import contract violated.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
