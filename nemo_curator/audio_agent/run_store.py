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

**Local history only, NOT shared memory / cross-user learning** (a permanent
non-goal). One JSON per run under a runs directory: ``AUDIO_AGENT_RUNS_DIR`` if set,
else ``<AUDIO_AGENT_WORKSPACE>/.audio_agent_runs``, else ``<cwd>/.audio_agent_runs``.
Records are written by ``run`` and read by the ``runs`` / ``plan_continuation`` verbs;
they are never fed back to influence planning across sessions.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from nemo_curator.audio_agent.contracts import RunRecord


def runs_dir() -> str:
    """The directory run records live in (env > workspace > cwd)."""
    explicit = os.environ.get("AUDIO_AGENT_RUNS_DIR")
    if explicit:
        return os.path.expanduser(explicit)
    from nemo_curator.audio_agent._safety import workspace_root

    root = workspace_root() or os.getcwd()
    return os.path.join(root, ".audio_agent_runs")


def new_run_id(config_hash: str | None = None) -> str:
    """A sortable run id: ``run-<UTC timestamp>-<hash8>``."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = (config_hash or "")[:8]
    return f"run-{ts}-{suffix}" if suffix else f"run-{ts}"


def save(record: RunRecord) -> str:
    """Persist a run record as JSON; returns the path (best-effort, never raises)."""
    directory = runs_dir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{record.run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    return path


def load(run_id: str) -> RunRecord | None:
    """Load a run record by id, or None if it doesn't exist / can't parse."""
    path = os.path.join(runs_dir(), f"{run_id}.json")
    if not os.path.isfile(path):
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
                "parent_run_id": rec.parent_run_id,
                "status": rec.status,
                "accepted": rec.accepted,
                "input_count": rec.input_count,
                "data_source": rec.data_source,
                "created_at": rec.created_at,
            }
        )
    return out
