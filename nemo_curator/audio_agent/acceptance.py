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

"""Acceptance layer (1A.1): turn a success contract into deterministic checks.

Three metric-agnostic pieces the deterministic core owns:

* :func:`expected_roles_from_criteria` — compile output/quality criteria into
  producible-role requirements so ``validate``'s output-completeness check
  catches "asked for X, nothing produces X" *before* a run.
* :func:`missing_implied` — request-type sanity: a request implying an output
  must carry the matching criterion (a *filtering* request needs a yield
  criterion; a *transcription* request needs its transcript output). A missing
  implied criterion is surfaced (so success can't be declared while ignoring the
  point of the request).
* :func:`verify` — evaluate each criterion against gathered evidence into an
  :class:`AcceptanceReport` with four honest states (met / not_met /
  unverifiable / unachievable). Never marks ``met`` without evidence, never
  silently relaxes a ``must``.

Everything is generic: metric/output keys are variables (``check.field``); there
are no metric names in the logic. A new metric ships a card block and works here
unchanged.
"""

from __future__ import annotations

import operator
from typing import Any

from nemo_curator.audio_agent.contracts import AcceptanceCriterion, AcceptanceReport, CriterionResult

# Request-type -> the criterion TYPE it implies (type-level, not metric-level, so
# it stays generic). Substring match on the goal's request_type/task keeps it
# robust to phrasing. Unknown request types imply nothing (no false flags).
_REQUEST_TYPE_IMPLIES: tuple[tuple[str, str], ...] = (
    ("filter", "yield"),  # "keep the clean ones" must define how much is kept
    ("transcri", "output_completeness"),  # transcription must declare its transcript output
    ("caption", "output_completeness"),
    ("diariz", "output_completeness"),  # diarization must declare its speaker-label output
    ("align", "output_completeness"),
)
_IMPLIED_HINT = {
    "yield": "a filtering/curation request should carry a 'yield' criterion (define how much to keep, e.g. retained > 0 or ~=N%)",
    "output_completeness": "a request that produces an output should carry an 'output_completeness' criterion naming that output",
}

_OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">": operator.gt, "<": operator.lt}


def parse_criteria(raw: Any) -> list[AcceptanceCriterion]:  # noqa: ANN401
    """Coerce a list of dicts (or criteria) into ``AcceptanceCriterion`` objects."""
    return [AcceptanceCriterion.from_dict(c) for c in (raw or [])]


def missing_implied(request_type: str | None, criteria: list[AcceptanceCriterion]) -> list[tuple[str, str]]:
    """Implied-but-absent criterion types for the request type, as ``(type, hint)``."""
    if not request_type:
        return []
    rt = request_type.lower()
    present = {c.type for c in criteria}
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for token, implied in _REQUEST_TYPE_IMPLIES:
        if token in rt and implied not in present and implied not in seen:
            seen.add(implied)
            out.append((implied, _IMPLIED_HINT.get(implied, f"request implies a {implied!r} criterion")))
    return out


def expected_roles_from_criteria(criteria: list[AcceptanceCriterion]) -> list[str]:
    """Producible-role requirements compiled from the criteria (for output-completeness).

    A criterion's ``compiles_to`` (explicit role) or ``check.field`` becomes a
    required output role — but only when it resolves to a *known* role, so an
    unrecognized metric key is skipped rather than flagged as a false gap.
    """
    from nemo_curator.stages.audio._roles import role_for_value

    out: list[str] = []
    for c in criteria:
        if not c.is_deterministic:
            continue
        target: str | None = None
        if c.compiles_to and c.compiles_to != "producible_role":
            target = c.compiles_to
        elif c.type in ("output_completeness", "quality_standard", "yield"):
            target = c.field_name
        if target and role_for_value(target) != "unknown":
            out.append(target)
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify(criteria: list[AcceptanceCriterion], evidence: dict[str, Any]) -> AcceptanceReport:
    """Evaluate each criterion against evidence into an :class:`AcceptanceReport`.

    ``evidence`` (all optional) may carry: ``produced_roles`` / ``produced_keys``
    (from ``validate``), ``metrics`` (aggregate ``{field: value}``), ``per_item``
    (``[{field: value}, ...]``), ``retained`` / ``input_count`` (from smoke/run),
    and ``unachievable_fields`` (metrics the data provably cannot meet).
    """
    ev = evidence or {}
    produced = set(ev.get("produced_roles") or []) | set(ev.get("produced_keys") or [])
    metrics = ev.get("metrics") or {}
    per_item = [d for d in (ev.get("per_item") or []) if isinstance(d, dict)]
    retained = ev.get("retained")
    input_count = ev.get("input_count")
    unachievable = set(ev.get("unachievable_fields") or [])
    has_producer_evidence = bool(produced)

    results = [
        _verify_one(c, produced, has_producer_evidence, metrics, per_item, retained, input_count, unachievable)
        for c in criteria
    ]
    musts = [r for r in results if r.severity == "must"]
    overall = "met" if all(r.status == "met" for r in musts) else "not_met"
    return AcceptanceReport(overall=overall, criteria=results, verdict=_summary(results, overall))


def _verify_one(  # noqa: PLR0911, PLR0913 - one honest branch per criterion type
    c: AcceptanceCriterion,
    produced: set[str],
    has_producer_evidence: bool,
    metrics: dict[str, Any],
    per_item: list[dict[str, Any]],
    retained: Any,  # noqa: ANN401
    input_count: Any,  # noqa: ANN401
    unachievable: set[str],
) -> CriterionResult:
    def result(status: str, evidence: str = "", note: str = "") -> CriterionResult:
        return CriterionResult(id=c.id, status=status, severity=c.severity, evidence=evidence, note=note)

    if not c.is_deterministic:
        return result("unverifiable", note="semantic criterion — routed to the reviewer (LLM)")

    field = c.field_name
    chk = c.check or {}

    if c.type == "output_completeness":
        target = c.compiles_to if (c.compiles_to and c.compiles_to != "producible_role") else field
        if not target:
            return result("unverifiable", note="no output field/role named")
        if not has_producer_evidence:
            return result("unverifiable", note="no producer evidence (pass produced_roles/keys)")
        ok = target in produced
        return result("met" if ok else "not_met", evidence=f"{target!r} {'in' if ok else 'not in'} produced roles/keys")

    if field in unachievable:
        return result("unachievable", evidence=f"data cannot meet the target for {field!r}")

    if c.type == "yield":
        if retained is None:
            return result("unverifiable", note="no retained count in evidence")
        op = chk.get("op", ">")
        val = chk.get("value", 0)
        tol = chk.get("tolerance", 0)
        if c.kind == "relative" and input_count:
            frac = 100.0 * float(retained) / float(input_count)
            ok = _cmp(frac, op or "~=", _num(val), tol)
            return result("met" if ok else "not_met", evidence=f"retained {retained}/{input_count} = {frac:.1f}% vs {op} {val}")
        ok = _cmp(_num(retained), op, _num(val), tol)
        return result("met" if ok else "not_met", evidence=f"retained={retained} vs {op} {val}")

    if c.type in ("quality_standard", "distribution"):
        if not field:
            return result("unverifiable", note="no metric field named")
        op = chk.get("op", ">=")
        val = chk.get("value")
        tol = chk.get("tolerance", 0)
        if chk.get("scope") == "per_retained_item":
            vals = [d[field] for d in per_item if field in d]
            if not vals:
                return result("unverifiable", note=f"no per-item values for {field!r}")
            ok = all(_cmp(_num(x), op, _num(val), tol) for x in vals)
            return result("met" if ok else "not_met", evidence=f"{len(vals)} items vs {op} {val}")
        if field in metrics:
            ok = _cmp(_num(metrics[field]), op, _num(val), tol)
            return result("met" if ok else "not_met", evidence=f"{field}={metrics[field]} vs {op} {val}")
        return result("unverifiable", note=f"no aggregate metric for {field!r}")

    # honesty and any future types: no deterministic evidence path yet -> reviewer.
    return result("unverifiable", note=f"criterion type {c.type!r} is not deterministically checkable (reviewer)")


def _cmp(lhs: Any, op: str, rhs: Any, tol: Any = 0) -> bool:  # noqa: ANN401
    try:
        if op == "non_empty":
            return bool(lhs)
        if op == "~=":
            return abs(float(lhs) - float(rhs)) <= float(tol or 0)
        return bool(_OPS[op](lhs, rhs))
    except (KeyError, TypeError, ValueError):
        return False


def _num(v: Any) -> float:  # noqa: ANN401
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _summary(results: list[CriterionResult], overall: str) -> str:
    n_met = sum(1 for r in results if r.status == "met")
    lines = [f"acceptance: {overall.upper()} ({n_met}/{len(results)} criteria met)"]
    for r in results:
        if r.status != "met":
            detail = r.note or r.evidence
            lines.append(f"  - {r.id} [{r.severity}]: {r.status}" + (f" ({detail})" if detail else ""))
    if overall == "not_met":
        lines.append(
            "options: adjust the recipe/thresholds and re-run, provide missing references, or relax a 'nice' "
            "criterion — never silently relax a 'must'."
        )
    return "\n".join(lines)
