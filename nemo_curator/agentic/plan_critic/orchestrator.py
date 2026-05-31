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
"""Build-time critic orchestration.

:func:`review_and_replan` runs the available critics on a freshly
compiled IR, applies any actionable patches, and re-plans up to
``max_iterations`` times. The final ``CriticReport`` collects findings
from every iteration so the UI can show the user the full reasoning
trace, not just the last pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from nemo_curator.agentic.intent import IntentCategories
from nemo_curator.agentic.plan_critic.base import (
    BaseCritic,
    CriticFinding,
    CriticReport,
    CriticSeverity,
    apply_findings,
)
from nemo_curator.agentic.plan_critic.plan import PlanCritic
from nemo_curator.agentic.plan_critic.sanity import SanityCritic

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Outcome of one or more review+re-plan iterations.

    ``ir`` is whatever shape the caller's ``replan`` callable returns —
    typically a :class:`~nemo_curator.agentic.ir.PipelineIR`. The
    orchestrator stays agnostic to that type so callers can keep richer
    result objects in their own closures.
    """

    intent: IntentCategories
    ir: Any
    report: CriticReport
    iterations: int
    re_planned: bool
    patches_applied: list[CriticFinding] = field(default_factory=list)


def review_and_replan(
    *,
    intent: IntentCategories,
    ir: Any,
    replan: Callable[[IntentCategories], Any],
    critics: list[BaseCritic],
    profile: Any = None,
    max_iterations: int = 2,
) -> ReviewResult:
    """Run *critics* on *ir*, apply patches, re-plan, repeat.

    Parameters
    ----------
    intent
        Current intent that produced ``ir``.
    ir
        The compiled :class:`~nemo_curator.agentic.ir.PipelineIR`.
    replan
        Callable that takes a (possibly patched) intent and returns a
        new IR. Provided by the caller so this module does not depend
        on the planner directly. Should raise on failure.
    critics
        Critic implementations to run in order. Use ``[SanityCritic()]``
        when no LLM is available; ``[SanityCritic(), PlanCritic(llm)]``
        otherwise.
    profile
        Optional dataset profile passed through to each critic.
    max_iterations
        Upper bound on review → re-plan iterations. Defaults to ``2``:
        one initial compile + one critique-driven re-plan. Use ``1`` to
        disable re-planning entirely.
    """

    aggregated = CriticReport()
    patches_applied: list[CriticFinding] = []
    current_intent = intent
    current_ir = ir
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        round_report = CriticReport()
        for critic in critics:
            try:
                round_report.extend(critic.review(
                    intent=current_intent,
                    ir=current_ir,
                    profile=profile,
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "critic %s failed; skipping: %s", critic.name, exc,
                )
        aggregated.extend(round_report)

        # If nothing actionable came back, we're done.
        if not round_report.has_actionable_patches:
            break

        # Apply patches and re-plan.
        patched_intent, contributing = apply_findings(current_intent, round_report)
        if not contributing or patched_intent.model_dump() == current_intent.model_dump():
            break
        patches_applied.extend(contributing)

        try:
            current_ir = replan(patched_intent)
        except Exception as exc:  # noqa: BLE001
            # Re-plan failed — back out, surface the original findings.
            logger.warning("critic-driven re-plan failed: %s", exc)
            aggregated.findings.append(CriticFinding(
                severity=CriticSeverity.WARN,
                code="replan_failed",
                detail=(
                    "Critic suggested changes but re-compiling with them "
                    f"failed: {exc}. Original pipeline kept."
                ),
                source="orchestrator",
            ))
            break
        current_intent = patched_intent

    return ReviewResult(
        intent=current_intent,
        ir=current_ir,
        report=aggregated,
        iterations=iterations,
        re_planned=bool(patches_applied),
        patches_applied=patches_applied,
    )


def default_critics(llm: Any = None, *, tier: str = "synth") -> list[BaseCritic]:
    """Standard critic stack: sanity first, then LLM plan critic if available."""

    critics: list[BaseCritic] = [SanityCritic()]
    if llm is not None:
        critics.append(PlanCritic(llm, tier=tier))
    return critics


__all__ = ["ReviewResult", "default_critics", "review_and_replan"]
