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

"""Opt-in Ray bootstrap so ``run``/``smoke`` are hands-free.

By default the agent core is infra-agnostic (it assumes a healthy Ray cluster or
an externally set ``RAY_ADDRESS``). ``ensure_cluster()`` is the opt-in path that
makes the agent self-sufficient on nodes where Ray does not come up by default:

  * respects an externally provided ``RAY_ADDRESS`` (never clobbers it),
  * reuses a head this process already started,
  * otherwise starts a correctly-configured local head on a FREE port with the
    plasma store on a writable dir (avoids the ``/dev/shm`` permission trap) and
    ``RAY_MAX_LIMIT_FROM_API_SERVER`` set (cosmos_xenna state-API cap).

It is deliberately NOT called unless the caller opts in (``bootstrap_ray=True``),
so normal Curator users with their own cluster are unaffected.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
from typing import Any

# Info about a head THIS process started (for reuse within the process).
_STARTED: dict[str, Any] = {}

_API_LIMIT = "40000"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reachable(address: str, timeout: float = 1.0) -> bool:
    """True if something is listening at ``host:port`` (cheap liveness probe)."""
    try:
        host, _, port = address.rpartition(":")
        with socket.create_connection((host or "127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _detect_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        return 0


def _ray_binary() -> str:
    """Locate the ``ray`` CLI next to the running interpreter, else on PATH."""
    candidate = os.path.join(os.path.dirname(sys.executable), "ray")
    if os.path.exists(candidate):
        return candidate
    from shutil import which

    found = which("ray")
    if found:
        return found
    msg = "the 'ray' CLI was not found next to the interpreter or on PATH"
    raise RuntimeError(msg)


def ensure_cluster(
    *,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    object_store_memory: int = 2_000_000_000,
    reuse: bool = True,
) -> str:
    """Ensure a usable Ray cluster and return its ``host:port`` address.

    Order of preference: an externally set ``RAY_ADDRESS`` that is reachable ->
    a head this process already started -> a freshly started local head. Sets
    ``RAY_ADDRESS`` and ``RAY_MAX_LIMIT_FROM_API_SERVER`` in the environment so
    the executor's ``ray.init()`` connects to it.
    """
    # 1. Respect an externally provided, reachable cluster.
    external = os.environ.get("RAY_ADDRESS")
    if external and external != "auto" and _reachable(external):
        os.environ.setdefault("RAY_MAX_LIMIT_FROM_API_SERVER", _API_LIMIT)
        return external

    # 2. Reuse a head we started earlier in this process.
    if reuse and _STARTED.get("address") and _reachable(_STARTED["address"]):
        os.environ["RAY_ADDRESS"] = _STARTED["address"]
        os.environ.setdefault("RAY_MAX_LIMIT_FROM_API_SERVER", _API_LIMIT)
        return _STARTED["address"]

    # 3. Start a fresh, correctly-configured local head.
    port = _free_port()
    dashboard_port = _free_port()
    temp_dir = tempfile.mkdtemp(prefix="ray_audio_agent_", dir="/tmp")  # noqa: S108 - short writable path required for Ray sockets/plasma
    gpus = _detect_gpus() if num_gpus is None else num_gpus
    cpus = num_cpus if num_cpus is not None else min(os.cpu_count() or 4, 8)

    cmd = [
        _ray_binary(), "start", "--head",
        f"--port={port}",
        f"--dashboard-port={dashboard_port}",
        f"--temp-dir={temp_dir}",
        f"--plasma-directory={temp_dir}",   # avoid /dev/shm (may be unwritable)
        f"--object-store-memory={object_store_memory}",
        f"--num-cpus={cpus}",
        f"--num-gpus={gpus}",
        "--disable-usage-stats",
    ]
    env = {**os.environ, "RAY_MAX_LIMIT_FROM_API_SERVER": _API_LIMIT, "RAY_TMPDIR": "/tmp"}
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=180)  # noqa: S603
    except subprocess.CalledProcessError as e:
        msg = (
            "failed to start a local Ray head. This node may have a port/GCS or shared-memory "
            f"restriction. stderr:\n{(e.stderr or '')[-1500:]}"
        )
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        msg = "timed out starting a local Ray head (raylet/GCS did not become responsive)"
        raise RuntimeError(msg) from e

    address = f"127.0.0.1:{port}"
    os.environ["RAY_ADDRESS"] = address
    os.environ["RAY_MAX_LIMIT_FROM_API_SERVER"] = _API_LIMIT
    _STARTED.update({"address": address, "temp_dir": temp_dir, "port": port})
    return address


def shutdown_cluster() -> bool:
    """Stop a head this process started (no-op if we didn't start one).

    Uses ``ray stop`` (node-wide); only call when you know this process owns the
    only Ray on the node. Returns True if a stop was attempted.
    """
    if not _STARTED.get("address"):
        return False
    try:
        subprocess.run([_ray_binary(), "stop", "--force"], check=False, capture_output=True, timeout=60)  # noqa: S603
    except Exception:  # noqa: BLE001
        return False
    finally:
        _STARTED.clear()
    return True
