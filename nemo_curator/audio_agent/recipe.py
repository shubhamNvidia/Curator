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

"""Recipe IR — the single artifact the host LLM emits and the core consumes.

A ``Recipe`` is a typed, hashable, serializable description of an ordered audio
pipeline: a list of ``{ref, params}`` stages plus inputs and an optional preset.
It round-trips to the ``stages:`` YAML that ``nemo_curator.config.run`` already
understands, and freezes to a ``recipe_id`` + ``config_hash`` for reproducibility
and plan-execution integrity ("what was approved is what runs").

The Recipe IR is the anti-hallucination boundary: the host proposes a Recipe,
never raw Python, and the core validates it structurally (here) and semantically
(``verbs.validate``) before anything runs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


# Constructor keys that configure the framework, not stage semantics; peeled out
# and re-applied via .with_() rather than passed to the dataclass constructor.
_WITH_KEYS = frozenset({"resources", "batch_size", "runtime_env", "num_workers"})


@dataclass
class StageRef:
    """One stage in a recipe: a registered stage class name + its params."""

    ref: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageRef:
        if not isinstance(d, dict) or "ref" not in d:
            msg = f"stage entry must be a dict with a 'ref' key, got {d!r}"
            raise ValueError(msg)
        return cls(ref=str(d["ref"]), params=dict(d.get("params") or {}))


@dataclass
class Recipe:
    """An ordered, configured audio pipeline the agent builds and validates."""

    stages: list[StageRef] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    preset: str | None = None
    rationale: str = ""
    name: str = "audio_agent_recipe"
    recipe_id: str | None = None
    config_hash: str | None = None
    # Layered save: recomputable annotations kept OUT of the hash so the recipe
    # stays portable. Re-run on a different machine/dataset recomputes these rather
    # than reusing stale, machine-/data-specific numbers.
    machine_plan: dict[str, Any] | None = None  # mode + per-stage resources (per machine)
    data_derived: dict[str, Any] | None = None  # data-derived values, e.g. relative thresholds (per dataset)
    knowledge_version: str | None = None  # knowledge/cards version the plan was approved against
    parent_run_id: str | None = None  # provenance chain for incremental continuation

    # ------------------------------------------------------------------ #
    # (de)serialization
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Recipe:
        if not isinstance(d, dict):
            msg = f"recipe must be a dict, got {type(d).__name__}"
            raise ValueError(msg)
        stages = [StageRef.from_dict(s) for s in (d.get("stages") or [])]
        return cls(
            stages=stages,
            inputs=dict(d.get("inputs") or {}),
            preset=d.get("preset"),
            rationale=str(d.get("rationale") or ""),
            name=str(d.get("name") or "audio_agent_recipe"),
            recipe_id=d.get("recipe_id"),
            config_hash=d.get("config_hash"),
            machine_plan=d.get("machine_plan"),
            data_derived=d.get("data_derived"),
            knowledge_version=d.get("knowledge_version"),
            parent_run_id=d.get("parent_run_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["stages"] = [s.to_dict() for s in self.stages]
        return out

    def _canonical(self) -> str:
        """Stable JSON of the PORTABLE semantic content only (stages + inputs + preset).

        Deliberately excludes id/hash/rationale AND the recomputable layered-save
        annotations (``machine_plan`` / ``data_derived`` / ``knowledge_version`` /
        ``parent_run_id``), so ``config_hash`` stays portable: the same intent on a
        different machine or dataset hashes identically.
        """
        payload = {
            "stages": [s.to_dict() for s in self.stages],
            "inputs": self.inputs,
            "preset": self.preset,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def compute_hash(self) -> str:
        return hashlib.sha256(self._canonical().encode("utf-8")).hexdigest()[:16]

    def freeze(self) -> Recipe:
        """Stamp a ``config_hash`` and a stable ``recipe_id`` (integrity anchor).

        Only the portable layer (see :meth:`_canonical`) is hashed; the layered-save
        annotations are attached separately and never change the hash.
        """
        self.config_hash = self.compute_hash()
        if not self.recipe_id:
            self.recipe_id = f"{self.name}-{self.config_hash[:8]}"
        return self

    # ------------------------------------------------------------------ #
    # layered save: recomputable annotations (never affect config_hash)
    # ------------------------------------------------------------------ #
    def with_machine_plan(self, plan: dict[str, Any], *, machine_fingerprint: str) -> Recipe:
        """Attach the resolved machine plan (mode + per-stage resources), stamped with
        the machine it was computed for. Does not change ``config_hash``."""
        self.machine_plan = {**plan, "machine_fingerprint": machine_fingerprint}
        return self

    def with_data_derived(self, values: dict[str, Any], *, data_fingerprint: str) -> Recipe:
        """Attach data-derived values (e.g. relative thresholds), stamped with the
        dataset they were computed from. Does not change ``config_hash``."""
        self.data_derived = {**values, "data_fingerprint": data_fingerprint}
        return self

    def stale_layers(self, *, machine_fingerprint: str | None = None, data_fingerprint: str | None = None) -> list[str]:
        """Recomputable layers that must be (re)built for the given machine/data.

        A layer is stale when it is absent or was stamped for a different
        fingerprint, so a re-run recomputes it instead of reusing stale numbers.
        """
        stale: list[str] = []
        if machine_fingerprint is not None and (self.machine_plan or {}).get("machine_fingerprint") != machine_fingerprint:
            stale.append("machine_plan")
        if data_fingerprint is not None and (self.data_derived or {}).get("data_fingerprint") != data_fingerprint:
            stale.append("data_derived")
        return stale

    # ------------------------------------------------------------------ #
    # pipeline-config bridge (round-trips to config.run's `stages:` format)
    # ------------------------------------------------------------------ #
    def to_pipeline_config(self) -> dict[str, Any]:
        """Return the ``{"stages": [...]}`` dict ``create_pipeline_from_yaml`` reads."""
        from nemo_curator.audio_agent._resolve import resolve_target

        stages_cfg: list[dict[str, Any]] = []
        for s in self.stages:
            entry: dict[str, Any] = {"_target_": resolve_target(s.ref)}
            entry.update(s.params)
            stages_cfg.append(entry)
        return {"stages": stages_cfg}


def _accepted_params(cls: type) -> list[str]:
    """Constructor param names a stage accepts, for a helpful ``bad_params`` error."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    return [
        n
        for n, p in sig.parameters.items()
        if n != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def build_stages(recipe: Recipe) -> tuple[list[ProcessingStage] | None, list[dict[str, Any]]]:
    """Instantiate a recipe's stages, returning ``(stages, issues)``.

    Instantiation doubles as a pre-flight check: a ``ref`` whose module failed to
    import (missing optional dep) or whose required params are absent yields an
    actionable issue instead of a stage. ``resources`` is applied via ``.with_()``.
    Returns ``(None, issues)`` if any stage could not be built.
    """
    from nemo_curator.audio_agent._resolve import resolve_stage_class

    issues: list[dict[str, Any]] = []
    stages: list[ProcessingStage] = []

    for idx, s in enumerate(recipe.stages):
        try:
            cls = resolve_stage_class(s.ref)
        except KeyError:
            issues.append(
                {
                    "code": "unknown_stage",
                    "severity": "error",
                    "stage_index": idx,
                    "stage": s.ref,
                    "message": f"{s.ref!r} is not a registered agent-ready audio stage in this environment",
                    "fix": "check the name via discover(), or install the extra that provides it (audio_cpu/audio_cuda12)",
                }
            )
            continue
        except Exception as e:  # noqa: BLE001 - import-time failure of an optional dep
            issues.append(
                {
                    "code": "stage_import_error",
                    "severity": "error",
                    "stage_index": idx,
                    "stage": s.ref,
                    "message": f"could not load {s.ref!r}: {type(e).__name__}: {e}",
                    "fix": "install the audio extra that provides this stage's dependency",
                }
            )
            continue

        params = dict(s.params)
        with_kwargs = {k: params.pop(k) for k in list(params) if k in _WITH_KEYS}
        try:
            inst = cls(**params)
            if with_kwargs:
                inst = _apply_with(inst, with_kwargs)
        except TypeError as e:
            accepted = _accepted_params(cls)
            fix = f"accepted params for {s.ref}: {accepted}" if accepted else "check required/allowed params via describe() or cards()"
            issues.append(
                {
                    "code": "bad_params",
                    "severity": "error",
                    "stage_index": idx,
                    "stage": s.ref,
                    "message": f"could not construct {s.ref!r} with params {sorted(params)}: {e}",
                    "fix": fix,
                }
            )
            continue
        except Exception as e:  # noqa: BLE001 - stage __post_init__ validation, etc.
            issues.append(
                {
                    "code": "construct_error",
                    "severity": "error",
                    "stage_index": idx,
                    "stage": s.ref,
                    "message": f"{s.ref!r} rejected its configuration: {type(e).__name__}: {e}",
                    "fix": "see the stage's card for valid parameter ranges",
                }
            )
            continue
        stages.append(inst)

    if any(i["severity"] == "error" for i in issues):
        return None, issues
    return stages, issues


def _apply_with(stage: ProcessingStage, with_kwargs: dict[str, Any]) -> ProcessingStage:
    """Apply framework knobs via ``.with_()``, coercing a resources dict."""
    resources = with_kwargs.get("resources")
    if isinstance(resources, dict):
        from nemo_curator.stages.resources import Resources

        with_kwargs = dict(with_kwargs)
        with_kwargs["resources"] = Resources(**resources)
    return stage.with_(**with_kwargs)
