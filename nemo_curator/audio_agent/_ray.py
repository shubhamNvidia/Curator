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
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import urlsplit

# Info about a head THIS process started (for reuse and ownership-safe cleanup).
_STARTED: dict[str, Any] = {}

_API_LIMIT = "40000"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reachable(address: str, timeout: float = 1.0) -> bool:
    """True if something is listening at ``host:port`` (cheap liveness probe)."""
    try:
        host, sep, port = address.rpartition(":")
        if not sep or not port.isdigit():  # no "host:port" -> not a probeable address
            return False
        with socket.create_connection((host or "127.0.0.1", int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def _address_identity(address: str) -> tuple[str, str, int | None] | None:
    """Return a conservative comparison key for a concrete Ray address."""
    value = str(address or "").strip()
    if not value or value == "auto":
        return None
    parsed = urlsplit(value if "://" in value else f"//{value}")
    try:
        port = parsed.port
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if host == "localhost":
        host = "127.0.0.1"
    if not host:
        return None
    # Distinguish Ray Client from a direct GCS address: the ports have different
    # meanings even when both happen to point at the same host.
    scheme = parsed.scheme.lower() if parsed.scheme else "gcs"
    return scheme, host, port


def _connected_address(ray_module: Any) -> str | None:  # noqa: ANN401
    """Best-effort address of Ray's existing connection.

    A caller-supplied address must never be silently ignored merely because Ray
    was already initialized. Ray Client exposes its connection string through
    its client context; regular Ray exposes the GCS address on RuntimeContext.
    """
    try:
        from ray.util.client import ray as client_ray

        if client_ray.is_connected():
            context = client_ray.get_context()
            worker = getattr(context, "client_worker", None)
            connection = getattr(worker, "_conn_str", None)
            if connection:
                return f"ray://{connection}"
    except Exception:  # noqa: BLE001 - regular Ray need not expose Client internals
        pass
    try:
        address = ray_module.get_runtime_context().gcs_address
    except Exception:  # noqa: BLE001 - absence is handled by the fail-closed caller
        return None
    return str(address) if address else None


def _addresses_match(requested: str, connected: str | None) -> bool:
    """Whether an initialized Ray connection proves the requested target."""
    if requested.strip() == "auto":
        return bool(connected)
    requested_identity = _address_identity(requested)
    connected_identity = _address_identity(connected or "")
    return requested_identity is not None and requested_identity == connected_identity


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


def _adopt_started_address(port: int) -> str:
    """The address Ray actually advertises for a head we just started on ``port``.

    Ray's GCS commonly reports the node's LAN IP even when reached via loopback, so a
    hardcoded ``127.0.0.1:port`` disagrees with it and ``cluster_resources`` then refuses
    to read the very cluster we started (the multi-NIC/cloud failure). Probe the real
    ``gcs_address`` once (via loopback, which does reach the local head) and adopt it, so
    ``RAY_ADDRESS`` is self-consistent with what Ray reports -- exactly how RayClient
    behaves. Falls back to loopback if the probe is unreadable.
    """
    fallback = f"127.0.0.1:{port}"
    try:
        import ray
    except Exception:  # noqa: BLE001 - the CLI-only bootstrap path keeps the loopback default
        return fallback
    opened = not ray.is_initialized()
    gcs: str | None = None
    try:
        if opened:
            ray.init(address=fallback, ignore_reinit_error=True, logging_level="ERROR")
        gcs = _connected_address(ray)
    except Exception:  # noqa: BLE001 - a probe failure keeps the loopback default
        gcs = None
    finally:
        if opened and ray.is_initialized():
            ray.shutdown()
    return gcs if (gcs and _address_identity(gcs)) else fallback


def ensure_cluster(
    *,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    object_store_memory: int = 2_000_000_000,
    reuse: bool = True,
) -> str:
    """Ensure a usable Ray cluster and return its ``host:port`` address.

    An externally supplied ``RAY_ADDRESS`` is authoritative, including ``auto``
    and ``ray://`` addresses. It is validated through Ray and returned unchanged;
    failure raises instead of silently replacing it with a local cluster. With no
    external address, reuse a head owned by this process or start a fresh local
    head. The local path sets ``RAY_ADDRESS`` for the executor's ``ray.init()``.
    """
    owned_address = _STARTED.get("address")
    external = os.environ.get("RAY_ADDRESS")
    owned_by_this_process = (
        bool(owned_address)
        and _STARTED.get("owned") is True
        and _STARTED.get("owner_pid") == os.getpid()
    )

    # 1. Reuse a healthy head this process started. Its exported environment
    # address is not an independently supplied external target.
    if external and owned_by_this_process and external == owned_address:
        if reuse and _reachable(owned_address):
            os.environ.setdefault("RAY_MAX_LIMIT_FROM_API_SERVER", _API_LIMIT)
            return str(owned_address)
        msg = (
            f"the process-owned Ray cluster at {owned_address!r} is no longer "
            "reachable; refusing to start a second local head"
        )
        raise RuntimeError(msg)

    # 2. Every non-empty externally supplied value is authoritative. Ray itself
    # understands address forms (notably auto and ray://) that a TCP probe does
    # not. A failed connection must not turn an external request into local work.
    if external and external.strip():
        try:
            cluster_resources(external)
        except Exception as exc:  # noqa: BLE001 - reframe all connection failures
            msg = (
                f"externally supplied RAY_ADDRESS={external!r} could not be "
                "validated; refusing to replace it with a local Ray cluster"
            )
            raise RuntimeError(msg) from exc
        os.environ.setdefault("RAY_MAX_LIMIT_FROM_API_SERVER", _API_LIMIT)
        return external

    # An owned head without its exact exported address is ambiguous: ray stop is
    # node-wide, so neither overwrite the environment nor start another head.
    if owned_address:
        msg = (
            f"Ray bootstrap ownership exists for {owned_address!r}, but "
            "RAY_ADDRESS no longer matches it; refusing an ambiguous bootstrap"
        )
        raise RuntimeError(msg)

    # 3. A Ray driver may already be connected even when no environment address
    # was exported. Reuse that connection rather than starting a competing local
    # head. If its target cannot be established, fail closed.
    try:
        import ray
    except Exception:  # noqa: BLE001 - the CLI-only bootstrap path remains valid
        ray = None
    if ray is not None and ray.is_initialized():
        connected = _connected_address(ray)
        if not connected:
            msg = (
                "Ray is already initialized but its cluster address cannot be "
                "verified; refusing to start a competing local head"
            )
            raise RuntimeError(msg)
        cluster_resources(connected)
        os.environ.setdefault("RAY_MAX_LIMIT_FROM_API_SERVER", _API_LIMIT)
        return connected

    # 4. No external, owned, or initialized cluster: start a fresh local head.
    port = _free_port()
    dashboard_port = _free_port()
    # Ray sockets/plasma need the short, writable /tmp path.
    temp_dir = tempfile.mkdtemp(prefix="ray_audio_agent_", dir="/tmp")  # noqa: S108
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
    previous_address = os.environ.get("RAY_ADDRESS")
    previous_api_limit = os.environ.get("RAY_MAX_LIMIT_FROM_API_SERVER")
    env = {
        **os.environ,
        "RAY_MAX_LIMIT_FROM_API_SERVER": previous_api_limit or _API_LIMIT,
        "RAY_TMPDIR": "/tmp",
    }
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=180)  # noqa: S603
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        msg = (
            "failed to start a local Ray head. This node may have a port/GCS or shared-memory "
            f"restriction. stderr:\n{(e.stderr or '')[-1500:]}"
        )
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        msg = "timed out starting a local Ray head (raylet/GCS did not become responsive)"
        raise RuntimeError(msg) from e
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    # Adopt the address Ray actually advertises (its GCS commonly reports the node's LAN
    # IP even when reached via loopback). Being self-consistent -- like RayClient -- avoids
    # the mismatch that made cluster_resources() refuse to read the cluster we just started.
    address = _adopt_started_address(port)
    os.environ["RAY_ADDRESS"] = address
    os.environ.setdefault("RAY_MAX_LIMIT_FROM_API_SERVER", _API_LIMIT)
    _STARTED.update(
        {
            "address": address,
            "temp_dir": temp_dir,
            "port": port,
            "owned": True,
            "owner_pid": os.getpid(),
            "previous_address": previous_address,
            "previous_api_limit": previous_api_limit,
        }
    )
    return address


def cluster_resources(address: str) -> dict[str, float]:
    """Read total schedulable Ray resources without leaving Ray initialized.

    Planning against driver hardware is wrong for both capped local heads and
    remote clusters. The caller supplies a known cluster address (normally from
    ``ensure_cluster`` or ``RAY_ADDRESS``); this helper never starts a cluster
    implicitly and disconnects again when it opened the connection.
    """
    if not address:
        msg = "a Ray address is required to probe cluster resources"
        raise ValueError(msg)
    import ray

    already_initialized = ray.is_initialized()
    opened_connection = False
    if already_initialized:
        connected = _connected_address(ray)
        if not _addresses_match(address, connected):
            msg = (
                f"Ray is already initialized at {connected!r}; refusing to read "
                f"resources for mismatched address {address!r}"
            )
            raise RuntimeError(msg)
    else:
        try:
            ray.init(
                address=address,
                ignore_reinit_error=True,
                logging_level="ERROR",
            )
            opened_connection = True
        except Exception as exc:  # noqa: BLE001 - expose a stable fail-closed error
            msg = f"failed to connect to Ray cluster at {address!r}"
            raise RuntimeError(msg) from exc
        connected = _connected_address(ray)
        if not _addresses_match(address, connected):
            ray.shutdown()
            opened_connection = False
            msg = (
                f"Ray connected at {connected!r}; refusing to read resources for "
                f"mismatched address {address!r}"
            )
            raise RuntimeError(msg)
    try:
        resources = ray.cluster_resources()
        return {
            str(key): float(value)
            for key, value in resources.items()
            if isinstance(value, (int, float))
        }
    finally:
        if opened_connection:
            ray.shutdown()


def owns_cluster(address: str | None = None) -> bool:
    """Whether this exact process owns the bootstrapped local Ray head."""
    owned_address = _STARTED.get("address")
    return bool(
        owned_address
        and _STARTED.get("owned") is True
        and _STARTED.get("owner_pid") == os.getpid()
        and (address is None or address == owned_address)
    )


def _restore_owned_environment(state: dict[str, Any]) -> None:
    """Undo only environment values that still equal values we installed."""
    address = state["address"]
    if os.environ.get("RAY_ADDRESS") == address:
        previous_address = state.get("previous_address")
        if previous_address is None:
            os.environ.pop("RAY_ADDRESS", None)
        else:
            os.environ["RAY_ADDRESS"] = previous_address
    if (
        state.get("previous_api_limit") is None
        and os.environ.get("RAY_MAX_LIMIT_FROM_API_SERVER") == _API_LIMIT
    ):
        os.environ.pop("RAY_MAX_LIMIT_FROM_API_SERVER", None)


def _safe_temp_dir(path: Any) -> str | None:  # noqa: ANN401
    """Return an owned Ray temp path only when it is safe to remove recursively."""
    if not isinstance(path, str):
        return None
    absolute_path = os.path.abspath(path)
    real_path = os.path.realpath(path)
    if os.path.islink(absolute_path) or real_path != absolute_path:
        return None
    if os.path.commonpath(("/tmp", real_path)) != "/tmp":
        return None
    if not os.path.basename(real_path).startswith("ray_audio_agent_"):
        return None
    return real_path


def _disconnect_owned_driver(address: str) -> bool:
    """Disconnect an initialized driver only when it targets our owned head."""
    try:
        import ray
    except Exception:  # noqa: BLE001 - the CLI may exist without an importable SDK
        return True
    try:
        if not ray.is_initialized():
            return True
        connected = _connected_address(ray)
        if not _addresses_match(address, connected):
            return False
        ray.shutdown()
    except Exception:  # noqa: BLE001 - preserve ownership state for a safe retry
        return False
    return True


def shutdown_cluster(address: str | None = None) -> bool:
    """Stop and clean a local head owned by this exact process.

    ``ray stop`` is node-wide, so ownership, process identity, the optional
    expected address, and the current ``RAY_ADDRESS`` must all agree before it is
    invoked. Failed or ambiguous stops preserve ownership state for a safe retry.
    On success, the bootstrap temp directory and environment changes are cleaned.
    """
    if not owns_cluster(address):
        return False
    state = dict(_STARTED)
    owned_address = state["address"]
    if os.environ.get("RAY_ADDRESS") != owned_address:
        return False
    if not _disconnect_owned_driver(owned_address):
        return False
    try:
        completed = subprocess.run(  # noqa: S603
            [_ray_binary(), "stop", "--force"],
            check=False,
            capture_output=True,
            timeout=60,
        )
    except Exception:  # noqa: BLE001
        return False
    if completed.returncode != 0:
        return False

    temp_dir = _safe_temp_dir(state.get("temp_dir"))
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)
    _restore_owned_environment(state)
    _STARTED.clear()
    return True
