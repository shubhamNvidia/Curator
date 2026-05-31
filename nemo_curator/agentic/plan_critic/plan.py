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
"""LLM-driven build-time critic.

PlanCritic asks the model: *given the user's original prompt + dataset
profile, does this compiled pipeline match the user's intent?* The
model can emit zero or more findings with optional intent patches; the
orchestrator then applies the patches and re-plans.

The LLM does **not** modify the IR directly — it only proposes intent
mutations, which the deterministic planner then translates into stage
changes. This keeps the LLM in its "language → schema" lane.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nemo_curator.agentic.intent import IntentCategories
from nemo_curator.agentic.ir import PipelineIR
from nemo_curator.agentic.llm import LLMClient, Message
from nemo_curator.agentic.plan_critic.base import (
    CriticFinding,
    CriticReport,
    CriticSeverity,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "nat" / "prompts" / "plan_critic.md"


def _load_prompt() -> str:
    if not _PROMPT_PATH.exists():
        msg = (
            f"plan_critic prompt not found at {_PROMPT_PATH}. "
            "Install seems incomplete."
        )
        raise FileNotFoundError(msg)
    return _PROMPT_PATH.read_text(encoding="utf-8")


class PlanCritic:
    """Reviews a compiled IR against the original prompt + intent + profile."""

    name = "plan"

    def __init__(self, llm: LLMClient | None, *, tier: str = "synth") -> None:
        self.llm = llm
        self.tier = tier

    def review(
        self,
        *,
        intent: IntentCategories,
        ir: PipelineIR,
        profile: Any = None,
    ) -> CriticReport:
        if self.llm is None:
            return CriticReport()  # graceful no-op when LLM unavailable

        payload = self._build_payload(intent, ir, profile)
        try:
            sys_prompt = _load_prompt()
        except FileNotFoundError as exc:
            logger.warning("plan_critic prompt missing, skipping: %s", exc)
            return CriticReport()

        messages = [
            Message("system", sys_prompt),
            Message("user", json.dumps(payload, indent=2, default=str)),
        ]
        try:
            raw = self.llm.chat_json(
                messages, tier=self.tier, temperature=0.0, purpose="plan_critic",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("plan_critic LLM call failed; skipping: %s", exc)
            return CriticReport()

        return self._parse_response(raw)

    # ------------------------------------------------------------------ #
    # Payload construction
    # ------------------------------------------------------------------ #

    def _build_payload(
        self,
        intent: IntentCategories,
        ir: PipelineIR,
        profile: Any,
    ) -> dict[str, Any]:
        from nemo_curator.agentic.plan_critic.base import user_locked_paths  # noqa: PLC0415

        # Trim stage refs to the essentials the critic needs to reason
        # about — full IR is too noisy for the LLM context.
        stages_summary: list[dict[str, Any]] = []
        for ref in ir.stages:
            stages_summary.append({
                "stage": ref.stage,
                "params": ref.params,
                "insert_reason": ref.insert_reason,
                "tuner_reasons": ref.tuner_reasons,
                "auto_inserted": ref.auto_inserted,
            })

        profile_dict: dict[str, Any] | None = None
        if profile is not None:
            try:
                profile_dict = profile.model_dump(mode="json")
            except AttributeError:
                profile_dict = None

        return {
            "user_prompt": intent.raw_prompt or "",
            "intent": intent.model_dump(mode="json", exclude_none=True),
            "compiled_pipeline": stages_summary,
            "dataset_profile": profile_dict,
            "user_locked_paths": sorted(user_locked_paths(intent)),
            "instruction": (
                "Review the pipeline. Emit findings for anything that "
                "doesn't match the user's intent or seems wasteful. "
                "Never propose a suggested_change for any path listed "
                "in user_locked_paths. Output JSON only."
            ),
        }

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #

    def _parse_response(self, raw: Any) -> CriticReport:
        if not isinstance(raw, dict):
            logger.warning("plan_critic: expected dict, got %s", type(raw).__name__)
            return CriticReport()

        findings_data = raw.get("findings") or []
        if not isinstance(findings_data, list):
            return CriticReport()

        report = CriticReport()
        for entry in findings_data:
            if not isinstance(entry, dict):
                continue
            try:
                severity = CriticSeverity(str(entry.get("severity", "info")).lower())
            except ValueError:
                severity = CriticSeverity.INFO

            patch = entry.get("suggested_change")
            if patch is not None and not isinstance(patch, dict):
                patch = None

            code = str(entry.get("code") or "plan_critic_finding").strip()
            detail = str(entry.get("detail") or "").strip()
            if not detail:
                continue  # skip empty findings rather than surface noise

            report.findings.append(CriticFinding(
                severity=severity,
                code=code,
                detail=detail,
                stage=entry.get("stage") or None,
                field=entry.get("field") or None,
                suggested_change=patch,
                rationale=entry.get("rationale") or None,
                source=self.name,
            ))
        return report
