#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TP=2 replicated MLA cache validation for multi-GPU hosts."""

import os

import pytest
import torch

from .common import TestDeterminism as BaseTestDeterminism
from .fixtures import KvbmModelConfig, KvbmServerSpec
from .mla_support import assert_mla_round_trip


_MLA_MODEL = KvbmModelConfig(
    model_id=os.environ.get(
        "KVBM_MLA_SMOKE_MODEL_ID", "v2ray/DeepSeek-V3-1B-Test"
    ),
    tensor_parallel_size=2,
    attention_backend=os.environ.get("KVBM_MLA_BACKEND", "TRITON_MLA"),
    max_model_len=int(os.environ.get("KVBM_MLA_SMOKE_MAX_MODEL_LEN", "512")),
    batch_invariant=False,
)
_MLA_SPEC = KvbmServerSpec(
    model_config=_MLA_MODEL,
    cpu_blocks=int(os.environ.get("KVBM_MLA_SMOKE_CPU_BLOCKS", "512")),
    gpu_blocks=int(os.environ.get("KVBM_MLA_SMOKE_GPU_BLOCKS", "128")),
    onboard_mode="intra",
    prefer_fc=False,
)
_ITERATIONS = int(os.environ.get("KVBM_MLA_SMOKE_ITERATIONS", "2"))
_STARTUP_TIMEOUT = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "300"))
_TEST_TIMEOUT = _STARTUP_TIMEOUT + 2 * (_ITERATIONS * 4 + 60)


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_2,
    pytest.mark.kvbm,
    pytest.mark.pre_merge,
    pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="TP=2 MLA validation requires two visible CUDA devices",
    ),
]


class TestMlaTp2(BaseTestDeterminism):
    """Validate replicated G1 with striped G2 and owner-root broadcasts."""

    @pytest.mark.parametrize(
        "kvbm_server_spec",
        [pytest.param(_MLA_SPEC, id=_MLA_SPEC.id)],
        indirect=True,
    )
    @pytest.mark.timeout(_TEST_TIMEOUT)
    def test_tp2_mla_round_trip(self, kvbm_tester, kvbm_server):
        assert_mla_round_trip(
            self,
            kvbm_tester,
            kvbm_server,
            iterations=_ITERATIONS,
            tensor_parallel_size=2,
        )
