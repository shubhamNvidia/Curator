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

import contextlib
import os
import shutil
import socket
import sys
import tempfile
from typing import Any
from urllib.parse import urlsplit

# Info about a head THIS process started (for reuse and ownership-safe cleanup).
_STARTED: dict[str, Any] = {}

_API_LIMIT = "40000"


# Ray binds its plasma/raylet sockets at
#   <root>/ray_audio_agent_XXXXXXXX/session_<26-char timestamp>_<pid>/sockets/plasma_store
# and validates the result itself (ray/_private/utils.py::validate_socket_filepath), so an
# over-long root does not fail at bind time but aborts ``ray start`` with an AF_UNIX length
# error. That fixed suffix is 25 (our mkdtemp component) + 64 (Ray's session dir and socket)
# = 89 bytes, worst-cased on a 7-digit pid. Budgeting the ROOT by what is actually left is
# the point of the check: a flat 60 was over three times too permissive, so precisely the
# long per-job scratch paths this indirection exists to honour passed it and then failed
# inside ray start. Too long now means "prefer a shorter root", not "fail late".
_AF_UNIX_MAX = 103 if sys.platform.startswith("darwin") else 107
_RAY_SESSION_SUFFIX = 89
_AF_UNIX_ROOT_BUDGET = _AF_UNIX_MAX - _RAY_SESSION_SUFFIX


def _temp_root_candidates() -> list[str]:
    """Resolved, de-duplicated temp roots this module may use, most-preferred first.

    Honors an explicitly configured ``RAY_TMPDIR``/``TMPDIR`` (per-job scratch on HPC, a
    sized volume in a container) before falling back to the platform temp dir.
    """
    roots: list[str] = []
    for candidate in (os.environ.get("RAY_TMPDIR"), os.environ.get("TMPDIR"), tempfile.gettempdir(), "/tmp"):
        if not candidate:
            continue
        resolved = os.path.realpath(os.path.expanduser(candidate))
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _ray_temp_root() -> str:
    """A writable, short-enough directory to host Ray's session temp and plasma store.

    Previously hardcoded to ``/tmp``, which ignored ``$TMPDIR`` (so HPC jobs filled a tiny
    node-local ``/tmp``) and put the object store on a size-capped tmpfs in containers.
    Prefers the first configured root that is writable AND short enough for the socket
    limit, so an over-long scratch path degrades to a usable one instead of failing.
    """
    usable = [r for r in _temp_root_candidates() if os.path.isdir(r) and os.access(r, os.W_OK)]
    if not usable:
        return tempfile.gettempdir()
    short_enough = [r for r in usable if len(r) <= _AF_UNIX_ROOT_BUDGET]
    return (short_enough or usable)[0]


def _reachable(address: str, timeout: float = 1.0) -> bool:
    """True if something is listening at ``host:port`` (cheap liveness probe).

    Parsed with :func:`_address_identity` so this agrees with every other address decision
    in the module. Splitting on the last colon instead disagreed on IPv6: Ray brackets the
    literals it advertises (``[::1]:6379``), and the bracketed host it produced was rejected
    by ``getaddrinfo``, so a live head always probed as unreachable. That silently turned the
    teardown's ground-truth gate into a rubber stamp on IPv6-only nodes -- failing open in
    precisely the direction it exists to prevent -- and made cluster reuse always refuse.
    """
    identity = _address_identity(address)
    if identity is None:
        return False
    _scheme, host, port = identity
    if port is None:  # no port -> not a probeable address
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
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
        except Exception as exc:
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

    # 4. No external, owned, or initialized cluster: start a fresh local head -- through the
    # SHARED RayClient, not a private bootstrap. Curator already solves this for the text and
    # video pipelines, and the two properties that matter here are both consequences of how
    # it starts Ray: ``ray start --block`` under ``Popen(start_new_session=True)`` keeps the
    # CLI alive as a supervised child in its own process group, and ``--block`` is what makes
    # Ray enable its OWN cleanup (``shutdown_at_exit`` and ``spawn_reaper``; ray/scripts/
    # scripts.py:970). Teardown is then one ``killpg`` on a known pid.
    #
    # The private bootstrap this replaces used a non-blocking ``ray start``, which turns both
    # of those off: the CLI exits, the daemons detach, and stopping them again required
    # re-discovering processes by scanning /proc and matching command lines. That search was
    # where the real defects lived -- renamed workers it could not see, bystanders it could,
    # and a re-scan that went blind once the parents died. None of it is reachable from a
    # process group.
    temp_dir = tempfile.mkdtemp(prefix="ray_audio_agent_", dir=_ray_temp_root())
    previous_address = os.environ.get("RAY_ADDRESS")
    previous_api_limit = os.environ.get("RAY_MAX_LIMIT_FROM_API_SERVER")
    os.environ.setdefault("RAY_MAX_LIMIT_FROM_API_SERVER", _API_LIMIT)

    from nemo_curator.core.client import RayClient

    client = RayClient(
        ray_temp_dir=temp_dir,
        num_cpus=num_cpus if num_cpus is not None else min(os.cpu_count() or 4, 8),
        num_gpus=_detect_gpus() if num_gpus is None else num_gpus,
        object_store_memory=object_store_memory,
        # The agent is a library caller, not an operator session: registering Ray with a
        # shared Prometheus/Grafana install is a side effect on state we do not own.
        include_dashboard=False,
    )
    try:
        # Verifies the head is responsive and stops what it started if it is not, so there is
        # no window in which a live head exists with nothing recorded to reach it.
        client.start()
        address = os.environ["RAY_ADDRESS"]  # set by RayClient to the node IP it bound
    except BaseException:  # KeyboardInterrupt too: Ctrl-C must not orphan a live head
        with contextlib.suppress(Exception):
            client.stop()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    _STARTED.update(
        {
            "address": address,
            "temp_dir": temp_dir,
            "client": client,
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
        except Exception as exc:
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


def _within_any_temp_root(real_path: str) -> bool:
    """Whether a RESOLVED path sits inside one of the plausible temp roots.

    Compares resolved path against resolved root, so a site whose temp dir is a symlink
    (a scratch mount) is recognized instead of silently failing containment -- which
    previously meant the bootstrap directory was never cleaned up. ``commonpath`` raises
    when two paths share no root, which is a non-match, never a crash.
    """
    for root in _temp_root_candidates():
        try:
            if os.path.commonpath((root, real_path)) == root:
                return True
        except ValueError:  # no shared root (e.g. different drives) -> simply not a match
            continue
    return False


def _safe_temp_dir(path: Any) -> str | None:  # noqa: ANN401
    """Return an owned Ray temp path only when it is safe to remove recursively.

    Three independent gates, all of which must hold: the entry itself is not a symlink
    (so a swapped link is never followed), the RESOLVED target is inside a temp root,
    and it carries this module's own prefix.
    """
    if not isinstance(path, str):
        return None
    if os.path.islink(os.path.abspath(path)):
        return None
    real_path = os.path.realpath(path)
    if not _within_any_temp_root(real_path):
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


def shutdown_cluster(address: str | None = None) -> bool:  # noqa: PLR0911 - one guard clause per refusal reason; flattening them would hide which check failed
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

    # Resolved BEFORE the driver is disconnected. ``ray.shutdown()`` is irreversible and
    # closes the caller's own session, so every gate that can refuse without side effects
    # belongs above it -- otherwise a refusal that promises "nothing changed, retry safely"
    # has already dropped the caller's actor handles and object refs on the floor.
    temp_dir = _safe_temp_dir(state.get("temp_dir"))
    if not temp_dir:
        return False
    if not _disconnect_owned_driver(owned_address):
        return False

    # One signal to the process group RayClient started, which is the whole point of going
    # through it: ``ray start --block`` keeps that group's supervisor alive, so killpg reaches
    # every component and nothing else. ``ray stop`` is never used -- it has no scoping
    # options and stops every Ray process on the machine (ray-project/ray#54989), which on a
    # shared box or CI runner would take a co-tenant's cluster down with ours.
    client = state.get("client")
    if client is None:
        return False
    try:
        client.stop()
    except Exception:  # noqa: BLE001 - preserve ownership state for a safe retry
        return False

    shutil.rmtree(temp_dir, ignore_errors=True)
    _restore_owned_environment(state)
    _STARTED.clear()
    return True
