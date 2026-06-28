# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `build_kv_transfer_config`."""

import pytest

from .server import (
    KvbmModelConfig,
    KvbmServerManager,
    KvbmServerSpec,
    build_kv_transfer_config,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


@pytest.fixture
def model_config() -> KvbmModelConfig:
    return KvbmModelConfig(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        block_size=16,
        attention_backend="FLASH_ATTN",
    )


@pytest.mark.parametrize("onboard_mode", ["intra", "inter"])
def test_payload_uses_canonical_connector_facade(
    model_config: KvbmModelConfig, onboard_mode: str
) -> None:
    cfg = build_kv_transfer_config(model_config, onboard_mode=onboard_mode)
    assert cfg["kv_connector_module_path"] == "kvbm.vllm.connector"
    assert cfg["kv_connector"] == "KvbmConnector"
    assert cfg["kv_role"] == "kv_both"


def test_payload_omits_leader_discovery(model_config: KvbmModelConfig) -> None:
    """Aggregated mode omits the discovery block."""
    cfg = build_kv_transfer_config(model_config)
    leader = cfg["kv_connector_extra_config"]["leader"]
    assert "nova" not in leader
    assert "velo" not in leader  # rename caveat: serde key is still 'nova'


@pytest.mark.parametrize("onboard_mode", ["intra", "inter"])
def test_payload_has_required_leader_blocks(
    model_config: KvbmModelConfig, onboard_mode: str
) -> None:
    cfg = build_kv_transfer_config(
        model_config, onboard_mode=onboard_mode, cpu_blocks=2000
    )
    leader = cfg["kv_connector_extra_config"]["leader"]
    assert leader["cache"]["host"] == {"num_blocks": 2000}
    assert leader["tokio"]["worker_threads"] == 2
    assert leader["onboard"] == {"mode": onboard_mode}


def test_payload_omits_cache_host_when_cpu_blocks_none(
    model_config: KvbmModelConfig,
) -> None:
    """When cpu_blocks is None, the leader config must NOT contain a
    cache.host block — the Rust leader will then fail hard on startup per
    the mandatory-tier contract."""
    cfg = build_kv_transfer_config(model_config, cpu_blocks=None)
    leader = cfg["kv_connector_extra_config"]["leader"]
    assert "cache" not in leader


def test_payload_default_onboard_mode_is_intra(
    model_config: KvbmModelConfig,
) -> None:
    cfg = build_kv_transfer_config(model_config)
    assert cfg["kv_connector_extra_config"]["leader"]["onboard"]["mode"] == "intra"


def test_payload_has_required_worker_blocks(model_config: KvbmModelConfig) -> None:
    cfg = build_kv_transfer_config(model_config)
    worker = cfg["kv_connector_extra_config"]["worker"]
    assert "UCX" in worker["nixl"]["backends"]
    assert "POSIX" in worker["nixl"]["backends"]
    assert worker["tokio"]["worker_threads"] == 2


def test_unknown_onboard_mode_raises(model_config: KvbmModelConfig) -> None:
    with pytest.raises(ValueError, match="unknown onboard_mode"):
        build_kv_transfer_config(model_config, onboard_mode="bogus")


def test_tp2_spec_id_names_parallelism() -> None:
    spec = KvbmServerSpec(
        model_config=KvbmModelConfig(model_id="example/mla", tensor_parallel_size=2)
    )

    assert spec.id == "mla-intra-tp2"


def test_tp2_server_command_enables_tensor_parallelism(tmp_path) -> None:
    spec = KvbmServerSpec(
        model_config=KvbmModelConfig(model_id="example/mla", tensor_parallel_size=2)
    )
    manager = KvbmServerManager(spec=spec, log_dir=tmp_path)

    try:
        flag = manager.server_cmd.index("--tensor-parallel-size")
        assert manager.server_cmd[flag + 1] == "2"
    finally:
        manager.stop_server()
