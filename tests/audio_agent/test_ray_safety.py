# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ray bootstrap safety regressions; all Ray/process interactions are mocked."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_curator.audio_agent import _ray


@pytest.fixture(autouse=True)
def _isolated_bootstrap_state(monkeypatch: pytest.MonkeyPatch):
    saved = dict(_ray._STARTED)
    _ray._STARTED.clear()
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_MAX_LIMIT_FROM_API_SERVER", raising=False)
    yield
    _ray._STARTED.clear()
    _ray._STARTED.update(saved)


@pytest.mark.parametrize(
    "address",
    ["auto", "ray://ray.example.test:10001", "10.20.30.40:6379"],
)
def test_every_external_ray_address_is_validated_and_never_overwritten(
    monkeypatch: pytest.MonkeyPatch,
    address: str,
) -> None:
    probes: list[str] = []
    monkeypatch.setenv("RAY_ADDRESS", address)
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda candidate: probes.append(candidate) or {"CPU": 4.0},
    )
    monkeypatch.setattr(
        _ray,
        "_free_port",
        lambda: pytest.fail("external addresses must never reach local bootstrap"),
    )

    assert _ray.ensure_cluster() == address
    assert probes == [address]
    assert os.environ["RAY_ADDRESS"] == address
    assert _ray._STARTED == {}


def test_failed_external_connection_fails_closed_without_local_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    address = "ray://unreachable.example.test:10001"
    monkeypatch.setenv("RAY_ADDRESS", address)

    def fail_probe(_address: str) -> dict[str, float]:
        raise ConnectionError("unreachable")

    monkeypatch.setattr(_ray, "cluster_resources", fail_probe)
    monkeypatch.setattr(
        _ray,
        "_free_port",
        lambda: pytest.fail("failed external connection must not start a local head"),
    )

    with pytest.raises(RuntimeError, match="refusing to replace it"):
        _ray.ensure_cluster()

    assert os.environ["RAY_ADDRESS"] == address
    assert _ray._STARTED == {}


def test_preinitialized_ray_without_environment_is_reused_not_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    address = "127.0.0.1:6379"
    fake_ray = SimpleNamespace(is_initialized=lambda: True)
    probes: list[str] = []
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(_ray, "_connected_address", lambda _module: address)
    monkeypatch.setattr(
        _ray,
        "cluster_resources",
        lambda candidate: probes.append(candidate) or {"CPU": 4.0},
    )
    monkeypatch.setattr(
        _ray,
        "_free_port",
        lambda: pytest.fail("an initialized driver must not trigger local bootstrap"),
    )

    assert _ray.ensure_cluster() == address
    assert probes == [address]
    assert "RAY_ADDRESS" not in os.environ
    assert _ray._STARTED == {}


def test_cluster_probe_refuses_an_already_initialized_address_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queried = False

    def resources() -> dict[str, float]:
        nonlocal queried
        queried = True
        return {"CPU": 8.0}

    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        cluster_resources=resources,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        _ray,
        "_connected_address",
        lambda _module: "127.0.0.1:6379",
    )

    with pytest.raises(RuntimeError, match="mismatched address"):
        _ray.cluster_resources("127.0.0.1:6380")

    assert queried is False


def test_cluster_probe_reads_matching_initialized_cluster_without_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        cluster_resources=lambda: {"CPU": 8, "GPU": 1.0, "label": "ignored"},
        shutdown=lambda: pytest.fail("pre-existing Ray connection must stay initialized"),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        _ray,
        "_connected_address",
        lambda _module: "127.0.0.1:6379",
    )

    assert _ray.cluster_resources("localhost:6379") == {"CPU": 8.0, "GPU": 1.0}


def test_cluster_probe_disconnects_a_new_connection_that_resolves_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdowns = 0
    queried = False

    def shutdown() -> None:
        nonlocal shutdowns
        shutdowns += 1

    def resources() -> dict[str, float]:
        nonlocal queried
        queried = True
        return {"CPU": 8.0}

    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **_kwargs: None,
        cluster_resources=resources,
        shutdown=shutdown,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        _ray,
        "_connected_address",
        lambda _module: "127.0.0.1:9999",
    )

    with pytest.raises(RuntimeError, match="mismatched address"):
        _ray.cluster_resources("127.0.0.1:6379")

    assert queried is False
    assert shutdowns == 1


def _mock_local_start(
    monkeypatch: pytest.MonkeyPatch,
    temp_dir: Path,
    run: Any,  # noqa: ANN401
) -> None:
    ports = iter((6381, 8266))
    temp_dir.mkdir()
    monkeypatch.setattr(_ray, "_free_port", lambda: next(ports))
    monkeypatch.setattr(_ray, "_detect_gpus", lambda: 0)
    monkeypatch.setattr(_ray, "_ray_binary", lambda: "/mock/ray")
    monkeypatch.setattr(
        _ray,
        "_adopt_started_address",
        lambda port: f"127.0.0.1:{port}",
    )
    monkeypatch.setattr(_ray.tempfile, "mkdtemp", lambda **_kwargs: str(temp_dir))
    monkeypatch.setattr(_ray.subprocess, "run", run)


def test_owned_local_cluster_is_cleaned_and_environment_is_restored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)

    temp_dir = tmp_path / "ray_audio_agent_owned"
    _mock_local_start(monkeypatch, temp_dir, run)

    address = _ray.ensure_cluster()

    assert address == "127.0.0.1:6381"
    assert _ray.owns_cluster(address)
    assert _ray._STARTED["owner_pid"] == os.getpid()
    assert os.environ["RAY_ADDRESS"] == address
    assert temp_dir.exists()

    assert _ray.shutdown_cluster(address) is True
    assert calls[0][1] == "start"
    assert calls[1] == ["/mock/ray", "stop", "--force"]
    assert not temp_dir.exists()
    assert "RAY_ADDRESS" not in os.environ
    assert "RAY_MAX_LIMIT_FROM_API_SERVER" not in os.environ
    assert _ray._STARTED == {}


def test_failed_local_start_removes_temp_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        raise subprocess.CalledProcessError(1, command, stderr="boom")

    temp_dir = tmp_path / "ray_audio_agent_failed"
    _mock_local_start(monkeypatch, temp_dir, run)

    with pytest.raises(RuntimeError, match="failed to start"):
        _ray.ensure_cluster()

    assert not temp_dir.exists()
    assert "RAY_ADDRESS" not in os.environ
    assert _ray._STARTED == {}


def test_shutdown_refuses_address_mismatch_before_node_wide_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temp_dir = tmp_path / "ray_audio_agent_mismatch"
    temp_dir.mkdir()
    _ray._STARTED.update(
        {
            "address": "127.0.0.1:6381",
            "temp_dir": str(temp_dir),
            "owned": True,
            "owner_pid": os.getpid(),
        }
    )
    monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:9999")
    monkeypatch.setattr(
        _ray.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("mismatch must not invoke node-wide ray stop"),
    )

    assert _ray.shutdown_cluster("127.0.0.1:6381") is False
    assert temp_dir.exists()
    assert _ray.owns_cluster("127.0.0.1:6381")


def test_shutdown_refuses_inherited_ownership_from_another_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temp_dir = tmp_path / "ray_audio_agent_inherited"
    temp_dir.mkdir()
    address = "127.0.0.1:6381"
    _ray._STARTED.update(
        {
            "address": address,
            "temp_dir": str(temp_dir),
            "owned": True,
            "owner_pid": os.getpid() + 1,
        }
    )
    monkeypatch.setenv("RAY_ADDRESS", address)
    monkeypatch.setattr(
        _ray.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("a child process must not stop its parent's Ray head"),
    )

    assert _ray.shutdown_cluster(address) is False
    assert temp_dir.exists()


def test_failed_stop_preserves_state_and_temp_directory_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temp_dir = tmp_path / "ray_audio_agent_retry"
    temp_dir.mkdir()
    address = "127.0.0.1:6381"
    _ray._STARTED.update(
        {
            "address": address,
            "temp_dir": str(temp_dir),
            "owned": True,
            "owner_pid": os.getpid(),
        }
    )
    monkeypatch.setenv("RAY_ADDRESS", address)
    monkeypatch.setattr(_ray, "_ray_binary", lambda: "/mock/ray")
    monkeypatch.setattr(
        _ray.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )

    assert _ray.shutdown_cluster(address) is False
    assert _ray.owns_cluster(address)
    assert temp_dir.exists()
    assert os.environ["RAY_ADDRESS"] == address


def test_shutdown_refuses_a_driver_connected_to_a_different_cluster(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temp_dir = tmp_path / "ray_audio_agent_driver_mismatch"
    temp_dir.mkdir()
    address = "127.0.0.1:6381"
    _ray._STARTED.update(
        {
            "address": address,
            "temp_dir": str(temp_dir),
            "owned": True,
            "owner_pid": os.getpid(),
        }
    )
    monkeypatch.setenv("RAY_ADDRESS", address)
    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        shutdown=lambda: pytest.fail("mismatched driver must not be disconnected"),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        _ray,
        "_connected_address",
        lambda _module: "127.0.0.1:9999",
    )
    monkeypatch.setattr(
        _ray.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("mismatched driver must not trigger ray stop"),
    )

    assert _ray.shutdown_cluster(address) is False
    assert temp_dir.exists()
    assert _ray.owns_cluster(address)
