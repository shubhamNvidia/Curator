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

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

Severity = Literal["error", "warning", "info"]


def _fingerprint(payload: dict[str, Any]) -> str:
    """Stable short hash of a payload (for machine/data fingerprints, layered save)."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


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

    def fingerprint(self) -> str:
        """Stable id of the dataset's identifying shape (not its contents).

        Used to stamp data-derived recipe annotations (layered save): a change to
        this fingerprint means data-derived values (e.g. relative thresholds) must
        be recomputed rather than silently reused.
        """
        return _fingerprint(
            {
                "source": self.source,
                "kind": self.kind,
                "num_files": self.num_files,
                "sample_rates": {str(k): v for k, v in self.sample_rates.items()},
                "channels": {str(k): v for k, v in self.channels.items()},
                "mean_duration_sec": round(self.mean_duration_sec, 3),
                "has_transcripts": self.has_transcripts,
                "manifest_keys": sorted(self.manifest_keys),
            }
        )


@dataclass
class EnvProfile:
    """The env probe's structured read of the machine (deps / GPU / secrets)."""

    has_gpu: bool = False
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory=list)
    gpu_mem_gb: float = 0.0  # VRAM per GPU (GB); for the resource planner's GPU-fit math
    total_cpus: int = 0
    total_ram_gb: float = 0.0
    free_disk_gb: float = 0.0
    has_ffmpeg: bool = False
    installed_extras: list[str] = field(default_factory=list)
    missing_packages: list[str] = field(default_factory=list)
    available_secrets: list[str] = field(default_factory=list)
    curator_version: str = ""
    python_version: str = ""  # running interpreter, e.g. "3.13.1"
    python_supported: bool = True  # satisfies the project's requires-python (else a note is added)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))

    def fingerprint(self) -> str:
        """Stable id of the machine's resource shape (layered save).

        A change means the machine plan (mode + per-stage resources) must be
        recomputed for the new hardware rather than reused from another machine.
        """
        return _fingerprint(
            {
                "gpu_count": self.gpu_count,
                "gpu_names": sorted(self.gpu_names),
                "gpu_mem_gb": round(self.gpu_mem_gb, 1),
                "total_cpus": self.total_cpus,
                "has_gpu": self.has_gpu,
                "has_ffmpeg": self.has_ffmpeg,
                "installed_extras": sorted(self.installed_extras),
                "curator_version": self.curator_version,
            }
        )


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
    # Set by a check that cannot decide (missing fact / ambiguous case) instead of
    # guessing pass or false-failing: hand off to "smoke" / "reviewer" / "user".
    escalate_to: Literal["smoke", "reviewer", "user"] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class Verdict:
    """The output of ``validate``: is this recipe composable and runnable here?

    IMPORTANT — pick the right field to gate on:

    * ``ok`` / ``keys_ok`` are the *data-flow* necessary conditions ONLY (roles connect,
      then literal keys connect). They are **not** "safe to run": a recipe can be
      ``ok=True`` while a card constraint (e.g. ``task_type_mismatch``, batch > model max)
      or an environment gate makes it unrunnable.
    * To decide whether to run, gate on ``runnable`` (no error-severity problem anywhere)
      or ``status == "pass"``. Use ``ok``/``keys_ok`` only for data-flow diagnostics.

    ``card_violations`` and ``gate_flags`` carry the model-constraint and environment
    problems that ``ok`` deliberately ignores.
    """

    ok: bool = False  # data-flow only (roles connect); NOT safe-to-run -- gate on runnable/status
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

    @property
    def status(self) -> Literal["pass", "fail", "uncertain"]:
        """Tri-state: ``fail`` if any error; else ``uncertain`` if any check
        escalated (couldn't decide); else ``pass``. ``runnable`` remains the
        no-error boolean; ``status`` distinguishes "clean pass" from "needs a
        human/smoke/reviewer to resolve an unknown."
        """
        pools = [self.issues, self.card_violations, self.gate_flags]
        if any(i.severity == "error" for pool in pools for i in pool):
            return "fail"
        if any(i.escalate_to for pool in pools for i in pool):
            return "uncertain"
        return "pass"

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
            "status": self.status,
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


# --------------------------------------------------------------------------- #
# acceptance layer (1A.1): the "did we solve the user's problem?" contract
# --------------------------------------------------------------------------- #
CriterionType = Literal["quality_standard", "output_completeness", "yield", "distribution", "honesty", "semantic_fit"]
CriterionStatus = Literal["met", "not_met", "unverifiable", "unachievable"]


@dataclass
class AcceptanceCriterion:
    """One checkable condition of success (1A §5.3).

    Host-derived from intent, confirmed at the gate, then verified against
    evidence. Generic and metric-agnostic: ``check.field`` is any metric/output
    key; the verifier contains no metric names. ``type`` routes it to the cheapest
    sufficient owner (deterministic vs reviewer).
    """

    id: str
    type: str  # CriterionType
    description: str = ""
    kind: str | None = None  # absolute | relative | operational
    check: dict[str, Any] = field(default_factory=dict)  # scope/field/op/value/tolerance/method
    compiles_to: str | None = None  # e.g. a producible-role name (output_completeness)
    source: dict[str, Any] = field(default_factory=dict)
    severity: Literal["must", "nice"] = "must"
    on_unachievable: Literal["escalate", "relax_with_confirmation"] = "escalate"

    @classmethod
    def from_dict(cls, d: Any) -> AcceptanceCriterion:  # noqa: ANN401
        if isinstance(d, AcceptanceCriterion):
            return d
        d = dict(d or {})
        return cls(
            id=str(d.get("id", "")),
            type=str(d.get("type", "")),
            description=str(d.get("description", "")),
            kind=d.get("kind"),
            check=dict(d.get("check") or {}),
            compiles_to=d.get("compiles_to"),
            source=dict(d.get("source") or {}),
            severity=d.get("severity", "must"),
            on_unachievable=d.get("on_unachievable", "escalate"),
        )

    @property
    def field_name(self) -> str | None:
        f = self.check.get("field")
        return str(f) if f else None

    @property
    def is_deterministic(self) -> bool:
        """True unless this criterion needs the reviewer (semantic / reviewer_judgment)."""
        return self.type != "semantic_fit" and self.check.get("method") != "reviewer_judgment"

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class CriterionResult:
    """One criterion's verification outcome (four honest states)."""

    id: str
    status: str  # CriterionStatus
    severity: str = "must"
    evidence: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class AcceptanceReport:
    """Per-criterion verification of the success contract (1A §8).

    ``overall`` is ``met`` iff every ``must`` criterion is ``met`` — the
    anti-goalpost-moving gate. ``not_met``/``unverifiable``/``unachievable`` are
    reported honestly, never silently relaxed.
    """

    overall: Literal["met", "not_met"] = "not_met"
    criteria: list[CriterionResult] = field(default_factory=list)
    verdict: str = ""
    # Honesty meta-check (1A.3): goalpost-moving violations found by comparing the
    # criteria actually verified against the confirmed (frozen) contract. Non-empty
    # forces overall=not_met (a relaxed 'must' bar cannot be declared met).
    honesty: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "criteria": [c.to_dict() for c in self.criteria],
            "verdict": self.verdict,
            "honesty": list(self.honesty),
        }


@dataclass
class ConfigStrategyEntry:
    """How one parameter's value was chosen (1A §6.4) — the auditable record.

    ``recompute_on`` drives layered save: ``none`` (absolute/explicit) is portable
    and travels with the recipe; ``data_change`` (relative/data-derived) and
    ``machine_change`` (operational) are recomputed on re-run rather than reused.
    """

    param: str
    value: Any = None
    metric: str | None = None
    kind: str = "absolute"  # absolute | relative | operational
    mode: str = "knowledge_driven"  # knowledge_driven | data_informed
    source: dict[str, Any] = field(default_factory=dict)  # {from: user_explicit|card_anchor|card_preset|..., ref: ...}
    recompute_on: str = "none"  # none | data_change | machine_change
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class RunRecord:
    """Local, per-run provenance for tracing + incremental continuation.

    A durable trace of one run — the frozen recipe + ``config_hash``, the success
    contract, evidence counts, output paths, the data fingerprint, and the parent
    link — so a follow-up request can reuse prior work and every result is
    traceable. This is **local** history, **not** shared cross-user memory or
    learning (a permanent non-goal): records are never read back to "teach" the
    agent across sessions; they only support provenance and continuation.
    """

    run_id: str
    recipe: dict[str, Any] = field(default_factory=dict)  # frozen Recipe.to_dict()
    config_hash: str | None = None
    parent_run_id: str | None = None
    goal: dict[str, Any] = field(default_factory=dict)
    data_source: str | None = None
    data_fingerprint: str | None = None
    acceptance_criteria: list[dict[str, Any]] = field(default_factory=list)
    status: str = ""
    accepted: int = 0
    input_count: int = 0
    output_paths: list[str] = field(default_factory=list)
    created_at: str = ""
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))
