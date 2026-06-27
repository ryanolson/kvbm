# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KVBM-specific conftest: reuse pre-existing NATS/etcd or spawn with dynamic ports.

Avoids conflicts with services already running on the host and removes the
requirement for nats-server/etcd binaries on PATH when services are available.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
import requests

from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import (
    allocate_port,
    allocate_ports,
    deallocate_port,
    deallocate_ports,
)

# Register the layered fixtures (deps / server / eval) for test discovery.
# See tests/kvbm_integration/fixtures/ and the README for the layered architecture.
from .fixtures import (  # noqa: F401
    kvbm_deps,
    kvbm_server,
    kvbm_server_spec,
    kvbm_tester,
)

_logger = logging.getLogger(__name__)


def _parse_port(url: str, default: int) -> int:
    """Extract port from a URL like nats://localhost:4222 or http://localhost:2379."""
    parsed = urlparse(url)
    return parsed.port or default


def _nats_available(url: str) -> bool:
    """Probe NATS using async connect + JetStream check (same as NatsServer._nats_ready)."""
    import nats

    async def _check():
        try:
            nc = await nats.connect(url, connect_timeout=2)
            try:
                js = nc.jetstream()
                await js.account_info()
                return True
            finally:
                await nc.close()
        except Exception:
            return False

    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_check())
        finally:
            loop.close()
    except Exception:
        return False


def _etcd_available(endpoint: str) -> bool:
    """Probe etcd via its health endpoint."""
    try:
        resp = requests.get(f"{endpoint}/health", timeout=2)
        return resp.ok
    except Exception:
        return False


def _save_and_set_env(nats_url, etcd_url):
    """Save original env vars and set new ones. Returns originals for restore."""
    orig_nats = os.environ.get("NATS_SERVER")
    orig_etcd = os.environ.get("ETCD_ENDPOINTS")
    os.environ["NATS_SERVER"] = nats_url
    os.environ["ETCD_ENDPOINTS"] = etcd_url
    return orig_nats, orig_etcd


def _restore_env(orig_nats, orig_etcd):
    """Restore original env vars."""
    for key, orig in [("NATS_SERVER", orig_nats), ("ETCD_ENDPOINTS", orig_etcd)]:
        if orig is not None:
            os.environ[key] = orig
        else:
            os.environ.pop(key, None)


@pytest.fixture()
def runtime_services(request, discovery_backend, request_plane, durable_kv_events):
    """Use pre-existing NATS/etcd if reachable, otherwise spawn with dynamic ports."""
    nats_url = os.environ.get("NATS_SERVER", "nats://localhost:4222")
    etcd_url = os.environ.get("ETCD_ENDPOINTS", "http://localhost:2379")

    if _nats_available(nats_url) and _etcd_available(etcd_url):
        nats_port = _parse_port(nats_url, 4222)
        etcd_port = _parse_port(etcd_url, 2379)

        orig_nats, orig_etcd = _save_and_set_env(nats_url, etcd_url)
        yield SimpleNamespace(port=nats_port), SimpleNamespace(port=etcd_port)
        _restore_env(orig_nats, orig_etcd)
    else:
        # Fall back: spawn fresh instances with dynamic ports.
        # EtcdServer / NatsServer are defined locally in this module (below),
        # so the spawn path resolves without any dynamo dependency.
        with NatsServer(
            request, port=0, disable_jetstream=not durable_kv_events
        ) as nats_proc:
            with EtcdServer(request, port=0) as etcd_proc:
                orig_nats, orig_etcd = _save_and_set_env(
                    f"nats://localhost:{nats_proc.port}",
                    f"http://localhost:{etcd_proc.port}",
                )
                yield nats_proc, etcd_proc
                _restore_env(orig_nats, orig_etcd)


# ---------------------------------------------------------------------------
# Local NATS / etcd process wrappers + the dynamo-compatible runtime-service
# fixtures the KVBM suite depends on.
#
# These are kvbm-local copies (not imports from dynamo's repo-root conftest) so
# the test tree resolves its full fixture graph with zero dynamo dependency.
# They subclass the staged tests.utils.managed_process.ManagedProcess and use
# the staged tests.utils.port_utils allocator; the etcd / nats-server *binaries*
# remain an external system requirement for any test that actually spawns them
# (see README "External system dependencies").
# ---------------------------------------------------------------------------


class EtcdServer(ManagedProcess):
    def __init__(self, request, port=2379, timeout=300):
        # Allocate free ports if port is 0
        use_random_port = port == 0
        if use_random_port:
            # Need two ports: client port and peer port for parallel execution
            # Start from 2380 (etcd default 2379 + 1)
            port, peer_port = allocate_ports(2, 2380)
        else:
            peer_port = None

        self.port = port
        self.peer_port = peer_port  # Store for cleanup
        self.use_random_port = use_random_port  # Track if we allocated the port
        port_string = str(port)
        etcd_env = os.environ.copy()
        etcd_env["ALLOW_NONE_AUTHENTICATION"] = "yes"
        data_dir = tempfile.mkdtemp(prefix="etcd_")

        command = [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{port_string}",
            "--advertise-client-urls",
            f"http://0.0.0.0:{port_string}",
        ]

        # Add peer port configuration only for random ports (parallel execution)
        if peer_port is not None:
            peer_port_string = str(peer_port)
            command.extend(
                [
                    "--listen-peer-urls",
                    f"http://0.0.0.0:{peer_port_string}",
                    "--initial-advertise-peer-urls",
                    f"http://localhost:{peer_port_string}",
                    "--initial-cluster",
                    f"default=http://localhost:{peer_port_string}",
                ]
            )

        command.extend(
            [
                "--data-dir",
                data_dir,
            ]
        )
        super().__init__(
            env=etcd_env,
            command=command,
            timeout=timeout,
            display_output=False,
            terminate_all_matching_process_names=not use_random_port,  # For distributed tests, do not terminate all matching processes
            health_check_ports=[port],
            data_dir=data_dir,
            log_dir=request.node.name,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release allocated ports when server exits."""
        try:
            # Only deallocate ports that were dynamically allocated (not default ports)
            if self.use_random_port:
                ports_to_release = [self.port]
                if self.peer_port is not None:
                    ports_to_release.append(self.peer_port)
                deallocate_ports(ports_to_release)
        except Exception as e:
            logging.warning(f"Failed to release EtcdServer port: {e}")

        return super().__exit__(exc_type, exc_val, exc_tb)


class NatsServer(ManagedProcess):
    def __init__(self, request, port=4222, timeout=300, disable_jetstream=False):
        # Allocate a free port if port is 0
        use_random_port = port == 0
        if use_random_port:
            # Start from 4223 (nats-server default 4222 + 1)
            port = allocate_port(4223)

        self.port = port
        self.use_random_port = use_random_port  # Track if we allocated the port
        self._request = request  # Store for restart
        self._timeout = timeout
        self._disable_jetstream = disable_jetstream
        data_dir = tempfile.mkdtemp(prefix="nats_") if not disable_jetstream else None
        command = [
            "nats-server",
            "--trace",
            "-p",
            str(port),
        ]
        if not disable_jetstream and data_dir:
            command.extend(["-js", "--store_dir", data_dir])
        super().__init__(
            command=command,
            timeout=timeout,
            display_output=False,
            terminate_all_matching_process_names=not use_random_port,  # For distributed tests, do not terminate all matching processes
            data_dir=data_dir,
            health_check_ports=[port],
            health_check_funcs=[self._nats_ready],
            log_dir=request.node.name,
        )

    def _nats_ready(self, timeout: float = 5) -> bool:
        """Verify NATS server is ready by connecting and optionally checking JetStream."""
        import asyncio

        import nats

        async def check():
            try:
                nc = await nats.connect(
                    f"nats://localhost:{self.port}",
                    connect_timeout=min(timeout, 2),
                )
                try:
                    if not self._disable_jetstream:
                        # Verify JetStream is initialized
                        js = nc.jetstream()
                        await js.account_info()
                    return True
                finally:
                    await nc.close()
            except Exception:
                return False

        # Handle both sync and async contexts
        try:
            asyncio.get_running_loop()  # Check if we're in async context
            # Already in async context - run in a thread to avoid blocking
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, check()).result(timeout=timeout)
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            return asyncio.run(check())

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release allocated port when server exits."""
        try:
            # Only deallocate ports that were dynamically allocated (not default ports)
            if self.use_random_port:
                deallocate_port(self.port)
        except Exception as e:
            logging.warning(f"Failed to release NatsServer port: {e}")

        return super().__exit__(exc_type, exc_val, exc_tb)

    def stop(self):
        """Stop the NATS server for restart. Does not release port or clean up fully."""
        _logger.info(f"Stopping NATS server on port {self.port}")
        self._stop_started_processes()

    def start(self):
        """Restart a stopped NATS server with fresh state."""
        _logger.info(f"Starting NATS server on port {self.port} with fresh state")
        # Clean up old data directory and create fresh one (only if JetStream enabled)
        if not self._disable_jetstream:
            old_data_dir = self.data_dir  # type: ignore[has-type]
            if old_data_dir is not None:
                shutil.rmtree(old_data_dir, ignore_errors=True)
            self.data_dir = tempfile.mkdtemp(prefix="nats_")

        # Rebuild command
        self.command = [
            "nats-server",
            "--trace",
            "-p",
            str(self.port),
        ]
        if not self._disable_jetstream and self.data_dir:
            self.command.extend(["-js", "--store_dir", self.data_dir])

        self._start_process()
        elapsed = self._check_ports(self._timeout)
        self._check_funcs(self._timeout - elapsed)


@pytest.fixture
def discovery_backend(request):
    """
    Discovery backend for runtime. Defaults to "etcd".

    To iterate over multiple backends in a test:
        @pytest.mark.parametrize("discovery_backend", ["file", "etcd"], indirect=True)
        def test_example(runtime_services):
            ...
    """
    return getattr(request, "param", "etcd")


@pytest.fixture
def request_plane(request):
    """
    Request plane for runtime. Defaults to "nats".

    To iterate over multiple transports in a test:
        @pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True)
        def test_example(runtime_services):
            ...
    """
    return getattr(request, "param", "nats")


@pytest.fixture
def durable_kv_events(request):
    """
    Whether to use durable KV events via JetStream. Defaults to False (NATS Core mode).

    When False (default):
    - NATS server starts without JetStream (-js flag omitted) for faster startup
    - Workers use local indexer mode (NATS Core / fire-and-forget events)

    When True:
    - NATS server starts with JetStream for durable KV event distribution
    - Workers use --durable-kv-events flag to publish to JetStream

    To use JetStream mode:
        @pytest.mark.parametrize("durable_kv_events", [True], indirect=True)
        def test_example(runtime_services_dynamic_ports):
            ...
    """
    return getattr(request, "param", False)


@pytest.fixture()
def runtime_services_dynamic_ports(
    request, discovery_backend, request_plane, durable_kv_events
):
    """Provide NATS and Etcd servers with truly dynamic ports per test.

    This fixture actually allocates dynamic ports by passing port=0 to the servers.
    It also sets the NATS_SERVER and ETCD_ENDPOINTS environment variables so that
    Dynamo processes can find the services on the dynamic ports.

    xdist/parallel safety:
    - Function-scoped: each test gets its own NATS/etcd instances and ports.
    - Each pytest-xdist worker runs tests in a separate process, so env vars do not
      leak across workers.

    - If discovery_backend != "etcd", etcd is not started (returns None)
    - NATS is always started when etcd is used, because KV events require NATS
      regardless of the request_plane (tcp/nats only affects request transport)
    - NATS Core mode (no JetStream) is the default; JetStream is enabled when durable_kv_events=True

    Returns a tuple of (nats_process, etcd_process) where each has a .port attribute.
    """
    # Port cleanup is now handled in NatsServer and EtcdServer __exit__ methods
    # Always start NATS when etcd is used - KV events require NATS regardless of request_plane
    # When durable_kv_events=False (default), disable JetStream for faster startup
    if discovery_backend == "etcd":
        with NatsServer(
            request, port=0, disable_jetstream=not durable_kv_events
        ) as nats_process:
            with EtcdServer(request, port=0) as etcd_process:
                # Save original env vars (may be set by session-scoped fixture)
                orig_nats = os.environ.get("NATS_SERVER")
                orig_etcd = os.environ.get("ETCD_ENDPOINTS")

                # Set environment variables for this test's dynamic ports
                os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
                os.environ["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"

                yield nats_process, etcd_process

                # Restore original env vars (or remove if they weren't set)
                if orig_nats is not None:
                    os.environ["NATS_SERVER"] = orig_nats
                else:
                    os.environ.pop("NATS_SERVER", None)
                if orig_etcd is not None:
                    os.environ["ETCD_ENDPOINTS"] = orig_etcd
                else:
                    os.environ.pop("ETCD_ENDPOINTS", None)
    elif request_plane == "nats":
        with NatsServer(
            request, port=0, disable_jetstream=not durable_kv_events
        ) as nats_process:
            orig_nats = os.environ.get("NATS_SERVER")
            os.environ["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
            yield nats_process, None
            if orig_nats is not None:
                os.environ["NATS_SERVER"] = orig_nats
            else:
                os.environ.pop("NATS_SERVER", None)
    else:
        yield None, None
