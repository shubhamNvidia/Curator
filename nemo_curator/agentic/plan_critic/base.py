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
"""Shared contract for all build-time critics.

Every critic — deterministic (:class:`SanityCritic`) or LLM-driven
(:class:`PlanCritic`) — emits the same :class:`CriticFinding` shape so
the orchestrator can:

1. Render a uniform "why" panel in the UI.
2. Apply structured ``intent_patch`` suggestions and re-plan in a loop.
3. Decide whether to block the build (any ``error`` finding) or proceed
   with warnings.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from nemo_curator.agentic.intent import IntentCategories
from nemo_curator.agentic.ir import PipelineIR


class CriticSeverity(str, Enum):
    """Severity of a single finding."""

    INFO = "info"     # surfaced in UI, never blocks
    WARN = "warn"     # surfaced + may trigger one re-plan attempt
    ERROR = "error"   # blocks the build; user must change intent


class CriticFinding(BaseModel):
    """One observation from a critic.

    ``intent_patch`` is the proposed mutation in *dotted-path* form, the
    same shape the web layer's :func:`apply_answers` already understands
    (e.g. ``{"quality.mos_threshold": 3.5}``). Leaving it ``None`` means
    the critic spotted a problem but cannot fix it automatically — the
    user has to decide.
    """

    model_config = ConfigDict(extra="forbid")

    severity: CriticSeverity = Field(..., description="info | warn | error")
    code: str = Field(
        ...,
        description="Stable machine-readable code (e.g. ``missing_output_writer``).",
    )
    detail: str = Field(..., description="Short human-readable explanation.")
    stage: str | None = Field(
        default=None,
        description="Stage class name this finding targets, if any.",
    )
    field: str | None = Field(
        default=None,
        description="Dotted intent path (e.g. ``quality.mos_threshold``) if applicable.",
    )
    suggested_change: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional intent patch in {dotted_path: value} form. When set the "
            "orchestrator may re-plan with this applied."
        ),
    )
    rationale: str | None = Field(
        default=None,
        description="Longer explanation surfaced in the UI 'why' panel.",
    )
    source: str = Field(
        ...,
        description="Critic that produced this finding (``sanity`` | ``plan``).",
    )


class CriticReport(BaseModel):
    """Aggregated findings from one or more critics on one IR snapshot."""

    model_config = ConfigDict(extra="forbid")

    findings: list[CriticFinding] = Field(default_factory=list)

    @property
    def errors(self) -> list[CriticFinding]:
        return [f for f in self.findings if f.severity == CriticSeverity.ERROR]

    @property
    def warnings(self) -> list[CriticFinding]:
        return [f for f in self.findings if f.severity == CriticSeverity.WARN]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def has_actionable_patches(self) -> bool:
        return any(
            f.suggested_change
            for f in self.findings
            if f.severity != CriticSeverity.INFO
        )

    def extend(self, other: "CriticReport") -> None:
        self.findings.extend(other.findings)

    def to_payload(self) -> list[dict[str, Any]]:
        """Render findings as plain dicts for JSON serialization / UI."""

        return [f.model_dump(mode="json", exclude_none=True) for f in self.findings]


class BaseCritic(Protocol):
    """All critics implement this single method."""

    name: str

    def review(
        self,
        *,
        intent: IntentCategories,
        ir: PipelineIR,
        profile: Any = None,
    ) -> CriticReport: ...


# ---------------------------------------------------------------------------
# Intent patching
# ---------------------------------------------------------------------------


_CLARIFIER_NOTE_PREFIX = "clarifier_answer:"


def user_locked_paths(intent: IntentCategories) -> set[str]:
    """Return the dotted intent paths the user explicitly answered.

    The clarifier records every form pick as a ``clarifier_answer:<path>=<value>``
    note in ``intent.notes`` (see :func:`nemo_curator.agentic.clarifier.apply_answers`).
    That trail is the source of truth for "what the user said with
    their own hands", regardless of what the extractor / heuristics
    inferred earlier.

    The Plan Critic must never silently override these — see regression
    run 72eb5552bd2e8772 where ``duration_max_sec=120`` (user pick)
    was patched to ``15`` because the prompt mentioned "short".
    """

    locked: set[str] = set()
    for note in intent.notes or []:
        if not isinstance(note, str) or not note.startswith(_CLARIFIER_NOTE_PREFIX):
            continue
        body = note[len(_CLARIFIER_NOTE_PREFIX):]
        path, _, _ = body.partition("=")
        path = path.strip()
        if path:
            locked.add(path)
    return locked


def apply_findings(
    intent: IntentCategories,
    report: CriticReport,
    *,
    only_severities: tuple[CriticSeverity, ...] = (CriticSeverity.WARN, CriticSeverity.ERROR),
) -> tuple[IntentCategories, list[CriticFinding]]:
    """Apply each finding's ``suggested_change`` to *intent* in order.

    Each patch is applied independently so one bogus value (e.g. an LLM
    that proposed an unknown enum like ``quality.mos = "preserve"``)
    doesn't drop the rest of the well-formed patches. Findings whose
    patch fails to apply are mutated in-place to record the validation
    error on ``rationale`` and are downgraded to ``INFO`` severity so
    they don't trigger another re-plan loop — but they still surface in
    the UI so the user can see what the LLM intended.

    Patches that target a path the user explicitly answered in the
    clarifier form are *never* applied: the user's choice wins. The
    finding is demoted to INFO and annotated so the UI can show the
    critic's concern without overriding the user.

    Returns the patched intent and the subset of findings whose patches
    actually contributed a change.
    """

    from nemo_curator.agentic.clarifier import apply_answers  # noqa: PLC0415

    locked = user_locked_paths(intent)
    patched_findings: list[CriticFinding] = []
    current = intent
    for finding in report.findings:
        if finding.severity not in only_severities:
            continue
        patch = finding.suggested_change
        if not patch:
            continue

        # Respect explicit user answers — never override them with an
        # LLM-proposed patch. Demote to INFO so it still surfaces in
        # the UI as "we noticed something but the user already chose".
        overridden = sorted(p for p in patch if p in locked)
        if overridden:
            note = (
                "Skipped because the user explicitly answered "
                + ", ".join(f"{p}={getattr(_resolve_value(intent, p), 'value', _resolve_value(intent, p))}" for p in overridden)
                + " in the clarifier form."
            )
            finding.rationale = (
                f"{finding.rationale}\n{note}".strip()
                if finding.rationale
                else note
            )
            finding.code = f"{finding.code}__user_locked"
            finding.severity = CriticSeverity.INFO
            finding.suggested_change = None
            continue

        try:
            updated = apply_answers(current, patch)
        except Exception as exc:  # noqa: BLE001
            invalid_note = (
                f"Critic proposed {patch!r} but the intent schema "
                f"rejected it: {type(exc).__name__}: {exc}. Patch skipped."
            )
            finding.rationale = (
                f"{finding.rationale}\n{invalid_note}".strip()
                if finding.rationale
                else invalid_note
            )
            finding.suggested_change = None
            finding.code = f"{finding.code}__invalid_patch"
            finding.severity = CriticSeverity.INFO
            continue

        # The intent model validator may have coerced the patched value
        # to keep the intent internally consistent (see
        # ``IntentCategories._consistency`` — e.g.
        # ``output_unit=single_speaker_clips`` together with a
        # non-SPLIT ``speakers.mode`` gets reverted because the selector
        # cannot produce SpeakerSeparation in that combination). When
        # that happens, the patch *was* applied but the validator
        # immediately undid (part of) it. Surface this so the critic
        # panel doesn't lie about the actual final state.
        coerced: list[tuple[str, Any, Any]] = []
        for dotted, want in patch.items():
            actual = _resolve_value(updated, dotted)
            actual_val = getattr(actual, "value", actual)
            want_val = getattr(want, "value", want)
            if actual_val != want_val and actual_val is not None:
                coerced.append((dotted, want_val, actual_val))
        if coerced:
            revert_note = "Validator coerced the patch for consistency: " + ", ".join(
                f"{p}={w!r} → {a!r}" for p, w, a in coerced
            ) + ". See intent notes for the reason."
            finding.rationale = (
                f"{finding.rationale}\n{revert_note}".strip()
                if finding.rationale
                else revert_note
            )
            finding.code = f"{finding.code}__validator_coerced"
            # Keep the finding visible but stop it from claiming "applied"
            # in the UI; demote to INFO since the effective change is no
            # longer what the critic asked for.
            finding.severity = CriticSeverity.INFO
            finding.suggested_change = None
            current = updated
            continue

        current = updated
        patched_findings.append(finding)
    return current, patched_findings


def _resolve_value(intent: IntentCategories, dotted_path: str) -> Any:
    """Resolve a dotted intent path against the model for display only.

    Returns ``None`` if any segment is missing — this function exists
    purely so the "you picked X" rationale text in
    :func:`apply_findings` can show the current value back to the user
    without exploding on nested attributes that don't exist on the
    schema we know.
    """

    obj: Any = intent
    for part in dotted_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
        if obj is None:
            return None
    return obj
