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

"""Typed data objects of the planning arc.

These are the JSON-safe contracts the deterministic core emits and consumes so
the host LLM (and the eval harness) get structured grounding. The host-produced
``GoalSpec`` / ``Critique`` are defined by the skill, not here — the objects in
this module are the ones our verbs return.

    PlanningContext   what the router/planner is handed (category tree + facts)
    Verdict           the output of ``validate`` (roles / keys / cards / gates)
    SmokeReport       the output of ``smoke`` (bounded evidence)
    PlanResult        a frozen, validated plan (or an escalate/refuse decision)
    DataProfile       the profiler's read of the input data
    EnvProfile        the env probe's read of the machine
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Severity = Literal["error", "warning", "info"]


def _clean(value: Any) -> Any:  # noqa: ANN401
    """Coerce dataclasses / sets / tuples to JSON-serializable primitives."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, (set, frozenset)):
        return sorted(_clean(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    return value


@dataclass
class DataProfile:
    """The profiler's structured read of an input dataset (no learning/memory)."""

    source: str = ""
    kind: Literal["manifest", "folder", "unknown"] = "unknown"
    num_files: int = 0
    sample_rates: dict[int, int] = field(default_factory=dict)  # sr -> count
    channels: dict[int, int] = field(default_factory=dict)  # nchan -> count
    total_duration_sec: float = 0.0
    mean_duration_sec: float = 0.0
    codecs: dict[str, int] = field(default_factory=dict)
    has_transcripts: bool = False
    manifest_keys: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class EnvProfile:
    """The env probe's structured read of the machine (deps / GPU / secrets)."""

    has_gpu: bool = False
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory=list)
    has_ffmpeg: bool = False
    installed_extras: list[str] = field(default_factory=list)
    missing_packages: list[str] = field(default_factory=list)
    available_secrets: list[str] = field(default_factory=list)
    curator_version: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class PlanningContext:
    """The compact, high-signal bundle handed to the host router/planner.

    Built deterministically by the Knowledge Index + Context Assembler; it never
    dumps source. ``category_tree`` (L0) lets the host prune before drilling into
    ``selected_stages`` (L2 full cards for finalists it picked).
    """

    goal: dict[str, Any] = field(default_factory=dict)
    category_tree: list[dict[str, Any]] = field(default_factory=list)
    selected_stages: list[dict[str, Any]] = field(default_factory=list)
    presets: dict[str, Any] = field(default_factory=dict)
    matched_blueprints: list[dict[str, Any]] = field(default_factory=list)
    matched_recipes: list[dict[str, Any]] = field(default_factory=list)
    patterns: list[dict[str, Any]] = field(default_factory=list)
    role_graph_slice: dict[str, Any] = field(default_factory=dict)
    data_profile: dict[str, Any] | None = None
    env_profile: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class Issue:
    """A single problem surfaced by ``validate`` (mirrors the planner Verdict)."""

    code: str
    severity: Severity
    message: str
    stage_index: int | None = None
    stage: str | None = None
    fix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class Verdict:
    """The output of ``validate``: is this recipe composable and runnable here?

    ``ok`` is the role-level necessary condition (rename-tolerant); ``keys_ok``
    adds the literal-key-identity check; ``card_violations`` and ``gate_flags``
    surface model-constraint and environment problems.
    """

    ok: bool = False
    keys_ok: bool = False
    issues: list[Issue] = field(default_factory=list)
    card_violations: list[Issue] = field(default_factory=list)
    gate_flags: list[Issue] = field(default_factory=list)
    unproducible_roles: list[str] = field(default_factory=list)
    produced_roles: list[str] = field(default_factory=list)
    produced_keys: list[str] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        """True when there are no error-severity problems anywhere."""
        pools = [self.issues, self.card_violations, self.gate_flags]
        return not any(i.severity == "error" for pool in pools for i in pool)

    def summary(self) -> str:
        all_issues = [*self.issues, *self.card_violations, *self.gate_flags]
        errs = [i for i in all_issues if i.severity == "error"]
        warns = [i for i in all_issues if i.severity == "warning"]
        if not all_issues:
            return "recipe OK (roles satisfied, keys connect, no card/gate problems)"
        lines = [f"{len(errs)} error(s), {len(warns)} warning(s):"]
        for i in all_issues:
            loc = f" [{i.stage}]" if i.stage else ""
            fix = f" -> {i.fix}" if i.fix else ""
            lines.append(f"  [{i.severity}] {i.code}{loc}: {i.message}{fix}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "keys_ok": self.keys_ok,
            "runnable": self.runnable,
            "issues": [i.to_dict() for i in self.issues],
            "card_violations": [i.to_dict() for i in self.card_violations],
            "gate_flags": [i.to_dict() for i in self.gate_flags],
            "unproducible_roles": sorted(self.unproducible_roles),
            "produced_roles": sorted(self.produced_roles),
            "produced_keys": sorted(self.produced_keys),
            "summary": self.summary(),
        }


@dataclass
class SmokeReport:
    """The output of ``smoke``: bounded execution on a small sample."""

    ran: bool = False
    sample: int = 0
    input_count: int = 0
    retained: int = 0
    rejected: int = 0
    errors: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    per_stage_metrics: dict[str, Any] = field(default_factory=dict)
    goals_met: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class PlanResult:
    """A frozen, validated plan handed to the confirm-gate + full run.

    Emitted by the (host-driven) Finalizer/Controller once the loops converge,
    or an ``escalate`` / ``refused`` decision.
    """

    status: Literal["finalized", "escalate", "refused"] = "escalate"
    recipe: dict[str, Any] | None = None
    rationale: str = ""
    confidence: float | None = None
    blocking_issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))
