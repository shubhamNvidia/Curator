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

"""Incremental continuation planner — reuse prior work for a follow-up request.

Deterministic recipe diff against a parent :class:`RunRecord`. Reuse is only
claimed where it is provably safe:

* **already_done** — the new recipe is identical to the parent's: reuse the parent
  output as-is.
* **incremental** — the parent's stages are an exact PREFIX of the new recipe (the
  request only *appends*, e.g. "also add transcripts"): reuse the parent's final
  output and run only the appended suffix.
* **full_rerun** — any divergence *within* the shared range (a changed param, a
  removed/reordered stage) or a different source dataset: there is no persisted
  intermediate to reuse from, so honestly rerun. The divergence point is reported.

Reuse additionally requires the SAME source data (a matching ``data_fingerprint``),
so "add transcripts to this dataset" never silently reuses another dataset's output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.audio_agent.contracts import RunRecord
    from nemo_curator.audio_agent.recipe import Recipe


def _stage_dicts(recipe: Recipe) -> list[dict[str, Any]]:
    return [s.to_dict() for s in recipe.stages]


def _common_prefix_len(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    """Number of leading stages identical (ref + params) in both recipes."""
    n = 0
    for sa, sb in zip(a, b):
        if sa != sb:
            break
        n += 1
    return n


def plan_continuation(
    new_recipe: Recipe,
    parent: RunRecord,
    *,
    data_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Compute the incremental execution plan for ``new_recipe`` given a ``parent`` run."""
    parent_stages = list(parent.recipe.get("stages") or [])
    new_stages = _stage_dicts(new_recipe)
    parent_refs = [s.get("ref") for s in parent_stages]
    new_refs = [s.get("ref") for s in new_stages]

    # Same-data guard: reuse is invalid if the source dataset changed.
    if data_fingerprint is not None and parent.data_fingerprint and data_fingerprint != parent.data_fingerprint:
        return {
            "mode": "full_rerun",
            "parent_run_id": parent.run_id,
            "reason": "source data changed since the parent run (data-fingerprint mismatch); nothing can be reused",
            "run_stages": new_refs,
        }

    if new_stages == parent_stages:
        return {
            "mode": "already_done",
            "parent_run_id": parent.run_id,
            "reuse_from": list(parent.output_paths),
            "reuse_stages": parent_refs,
            "run_stages": [],
            "rationale": "identical recipe on the same data; reuse the parent output as-is (nothing to run)",
        }

    prefix = _common_prefix_len(parent_stages, new_stages)
    if prefix == len(parent_stages) and len(new_stages) > len(parent_stages):
        suffix = new_stages[prefix:]
        return {
            "mode": "incremental",
            "parent_run_id": parent.run_id,
            "reuse_stages": parent_refs,
            "run_stages": [s.get("ref") for s in suffix],
            "reuse_from": list(parent.output_paths),
            "rationale": (
                f"the new recipe extends the parent by {len(suffix)} stage(s); reuse the parent's output "
                f"as input and run only the appended stage(s)"
            ),
        }

    return {
        "mode": "full_rerun",
        "parent_run_id": parent.run_id,
        "diverged_at": prefix,
        "reason": (
            f"recipes diverge at stage index {prefix} "
            f"(changed/removed/reordered stage); no persisted intermediate exists to reuse from"
        ),
        "run_stages": new_refs,
    }
