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

"""Local run-record store — per-run provenance for tracing + incremental continuation.

**Local history only, NOT shared memory / cross-user learning** (a permanent non-goal).
Records support *deterministic memoization* — content-addressed "has this exact computation
already been done?" (see ``REUSE_ARCHITECTURE.md``) — and provenance; they are never fed back
as learned priors to influence *what* the agent plans.

One JSON per run under a runs directory: ``AUDIO_AGENT_RUNS_DIR`` if set, else
``<AUDIO_AGENT_WORKSPACE>/.audio_agent_runs``, else ``<cwd>/.audio_agent_runs``. Records are
written by ``run`` and read by the ``runs`` / ``continue`` / ``reuse-scan`` verbs.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from typing import Any

from nemo_curator.audio_agent.contracts import RunRecord

# A run id becomes a filename, and ``load`` is reached with a caller-supplied one (the ``runs``
# verb and ``continue --parent-run-id``, both exposed over the CLI and MCP). Unvalidated, a
# ``../`` in it reads any ``.json`` on the box back through the record fields.
#
# Deliberately NOT the sibling pattern in ``calibration_store`` (``[A-Za-z0-9_-]``): a config
# hash is hex, but a run id carries the microsecond timestamp ``new_run_id`` builds
# (``run-20260816T073201.869815Z-04188532-e01c``), so that pattern would reject every real id
# and make the store unreadable. A dot is admitted; a separator is what must not be. ``.`` and
# ``..`` alone therefore pass and are harmless -- they name a file inside the runs directory.
_SAFE_RUN_ID = re.compile(r"\A[A-Za-z0-9_.-]{1,128}\Z")


def runs_dir() -> str:
    """The directory run records live in (env > workspace > cwd)."""
    explicit = os.environ.get("AUDIO_AGENT_RUNS_DIR")
    if explicit:
        return os.path.expanduser(explicit)
    from nemo_curator.audio_agent._safety import workspace_root

    root = workspace_root() or os.getcwd()
    return os.path.join(root, ".audio_agent_runs")


def _ensure_private_dir(path: str) -> None:
    """Create a state directory readable only by its owner.

    Run records carry dataset paths, dataset keys, goals and output locations. Created
    under a normal umask (002/022) they are group- and world-readable, which on a shared
    build agent, HPC project space or team NFS share exposes one user's curation history
    to every other. Only a directory this call actually CREATES is tightened -- one that
    already exists was configured deliberately and is left as the deployment set it.

    Underscored but NOT private: imported by ``artifacts`` and ``calibration_store``, which
    write their own state alongside the run records and need the same permissions. It lives
    here rather than in a utility module because the reasoning above is about run records.
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        return
    except OSError:
        raise
    with contextlib.suppress(OSError):  # best-effort: a filesystem may not honour chmod
        os.chmod(path, 0o700)


def _write_private_json(path: str, payload: dict) -> None:
    """Write JSON to a file created owner-only (0600); pre-existing files keep their mode.

    Underscored but NOT private: imported by ``artifacts`` and ``calibration_store``, always
    paired with :func:`_ensure_private_dir`.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def scratch_dir() -> str:
    """Where to put a recipe written for one job, created on demand.

    A recipe assembled for a single request is working material, not a contribution, but it has
    to live in a file because every verb takes a path. With nowhere designated, it lands in
    whatever directory the caller happened to start in -- which for anyone working inside a
    checkout is the repository root. Five such files have accumulated there, each looking like
    an untracked change someone forgot to commit.

    Sits under :func:`runs_dir`, which is already git-ignored and already moves with
    ``AUDIO_AGENT_RUNS_DIR``, so a scratch recipe is discardable by the same gesture that
    discards the run records describing what it did.
    """
    path = os.path.join(runs_dir(), "recipes")
    _ensure_private_dir(runs_dir())
    _ensure_private_dir(path)
    return path


def exact_recipe_path(run_id: str) -> str | None:
    """Where a run's verbatim recipe copy lives, or ``None`` for an id that cannot name a file."""
    if not run_id or not _SAFE_RUN_ID.match(str(run_id)):
        return None
    return os.path.join(runs_dir(), "exact_recipes", f"{run_id}.json")


def save_exact_recipe(run_id: str, recipe: dict[str, Any]) -> str | None:
    """Keep a verbatim copy of a recipe the run record cannot reproduce. Best-effort.

    The record's own copy has secret-valued params masked, which is right for a payload that
    reaches a host LLM and wrong for the one thing history is asked to do besides tracing:
    re-run this pipeline over what changed. A masked param is part of reuse identity, so a recipe
    rebuilt from the record hashes differently and matches none of that run's own prior work --
    the request "do the same thing again on the new files" fails on a pipeline that needs a
    credential.

    Written only when redaction actually changed something, so the ordinary run adds no second
    file, and only inside the owner-only state directory (0700/0600) the run records already use.
    Returns the path written, or ``None`` when there was nothing to keep.
    """
    from nemo_curator.audio_agent._safety import redact

    if redact(recipe, redact_transcripts=False) == recipe:
        return None  # the record reproduces it exactly; a second copy would be one more place to leak
    path = exact_recipe_path(run_id)
    if path is None:
        return None
    _ensure_private_dir(runs_dir())
    _ensure_private_dir(os.path.dirname(path))
    _write_private_json(path, dict(recipe))
    return path


def load_exact_recipe(run_id: str) -> dict[str, Any] | None:
    """The verbatim recipe for a run, or ``None`` when the record's own copy is already exact."""
    path = exact_recipe_path(run_id)
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:  # noqa: BLE001 - a corrupt copy falls back to the record, it does not raise
        return None
    return loaded if isinstance(loaded, dict) else None


def new_run_id(config_hash: str | None = None) -> str:
    """A sortable, collision-resistant run id: ``run-<UTC ts.microseconds>Z-<hash8>-<rand4>``.

    Microsecond precision plus a short random tiebreak keeps ids unique even when two
    identical-config runs start in the same wall-clock second -- a plain second-resolution
    id (with the same config_hash) would collide and silently overwrite the earlier record.
    The zero-padded fixed-width timestamp keeps ids lexicographically time-sortable.
    """
    now = time.time()
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now)) + f".{int((now % 1) * 1_000_000):06d}Z"
    rand = os.urandom(2).hex()  # 4 hex chars: tiebreak within the same microsecond
    suffix = (config_hash or "")[:8]
    return f"run-{ts}-{suffix}-{rand}" if suffix else f"run-{ts}-{rand}"


def record_path(run_id: str) -> str | None:
    """The file a run record occupies, or ``None`` when the id cannot safely name one.

    One place builds this path, so ``load`` and ``save`` cannot disagree about which ids are
    allowed to steer it.
    """
    return os.path.join(runs_dir(), f"{run_id}.json") if run_id and _SAFE_RUN_ID.match(str(run_id)) else None


def save(record: RunRecord) -> str:
    """Persist a run record as JSON and index it; returns the path.

    The JSON is the source of truth; the SQLite index is a rebuildable cache, so a failure
    to index is swallowed rather than losing the record.

    Raises rather than silently relocating a record whose id could steer the path. Nothing
    reaches here with a caller-chosen id today -- ``verbs._record_run`` passes what
    ``new_run_id`` produced -- so this guards the invariant rather than a live route.
    """
    directory = runs_dir()
    path = record_path(record.run_id)
    if path is None:
        msg = (
            f"run_id {record.run_id!r} cannot name a record file; expected the shape "
            f"new_run_id() produces (letters, digits, '.', '_', '-')"
        )
        raise ValueError(msg)
    _ensure_private_dir(directory)
    _write_private_json(path, record.to_dict())
    with contextlib.suppress(Exception):  # the index is a cache; never fail a save over it
        from nemo_curator.audio_agent import run_index

        run_index.index_run(record)
    return path


def load(run_id: str) -> RunRecord | None:
    """Load a run record by id, or None if the id is unusable / it doesn't exist / can't parse.

    An id that could steer the path out of the runs directory reads as "no such record", which
    is what it is: the store holds records under ids it issued, and nothing else is one.
    """
    path = record_path(run_id)
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return RunRecord.from_dict(json.load(f))
    except Exception:  # noqa: BLE001 - a corrupt record must not break the caller
        return None


def list_runs() -> list[dict[str, Any]]:
    """Summaries of all stored run records (most recent first)."""
    directory = runs_dir()
    if not os.path.isdir(directory):
        return []
    out: list[dict[str, Any]] = []
    for fn in sorted(os.listdir(directory), reverse=True):
        if not fn.endswith(".json"):
            continue
        rec = load(fn[:-len(".json")])
        if rec is None:
            continue
        out.append(
            {
                "run_id": rec.run_id,
                "config_hash": rec.config_hash,
                "semantic_hash": rec.semantic_hash,
                "dataset_key": rec.dataset_key,
                "parent_run_id": rec.parent_run_id,
                "status": rec.status,
                "accepted": rec.accepted,
                "input_count": rec.input_count,
                "data_source": rec.data_source,
                "elapsed_sec": rec.elapsed_sec,
                "created_at": rec.created_at,
            }
        )
    return out
