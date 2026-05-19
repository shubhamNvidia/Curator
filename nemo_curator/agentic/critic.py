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
"""Layer 5 output critic — deterministic-only stub.

For Phase 2 the critic does only the cheap, deterministic checks:

- Per-stage drop rates by comparing ``tasks_in`` / ``tasks_out`` in the
  :class:`RunCard`.
- Output-vs-input distribution shifts (sample-rate / channels / duration)
  surfaced as warnings.
- A sample re-evaluation hook is exposed but no-op (returns count zero) until
  Phase 5 brings the relevant analytical stages.

The LLM-driven intent-alignment scoring lands in Phase 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nemo_curator.agentic.cards import (
    CriticReport,
    DatasetCard,
    DatasetProfile,
    RunCard,
)
from nemo_curator.agentic.intent import IntentCategories

if TYPE_CHECKING:  # pragma: no cover
    from nemo_curator.agentic.llm import LLMClient


@dataclass
class CriticOptions:
    """Tunable thresholds for the deterministic critic."""

    max_drop_rate_warning: float = 0.5
    max_drop_rate_error: float = 0.9
    min_keep_count: int = 1
    drift_p50_ratio: float = 0.5  # warn if median duration changes by > this factor
    use_llm: bool = False  # Phase 3: enables the single LLM call in this layer
    llm_min_score: float = 0.7


def critique(
    *,
    run_card: RunCard,
    intent: IntentCategories | None = None,
    input_card: DatasetCard | None = None,
    output_card: DatasetCard | None = None,
    options: CriticOptions | None = None,
    llm_client: "LLMClient | None" = None,
) -> CriticReport:
    """Build a :class:`CriticReport` from a completed run."""

    opts = options or CriticOptions()
    findings: list[str] = []
    drop_rates: dict[str, float] = {}

    # Per-stage drop rates -------------------------------------------------
    for record in run_card.stage_records:
        if record.tasks_in <= 0:
            continue
        drop = max(0, record.tasks_in - record.tasks_out)
        rate = drop / record.tasks_in
        drop_rates[record.stage_name] = round(rate, 4)
        if rate >= opts.max_drop_rate_error:
            findings.append(
                f"{record.stage_name}: dropped {drop}/{record.tasks_in} ({rate*100:.0f}%) — "
                f"likely too strict; relax thresholds or move filter earlier."
            )
        elif rate >= opts.max_drop_rate_warning:
            findings.append(
                f"{record.stage_name}: dropped {drop}/{record.tasks_in} ({rate*100:.0f}%) — "
                "review threshold."
            )

    # Output profile -------------------------------------------------------
    out_profile = output_card.profile if output_card else DatasetProfile()
    if output_card and output_card.profile.total_files < opts.min_keep_count:
        findings.append(
            f"Output dataset has only {output_card.profile.total_files} files — below min_keep_count={opts.min_keep_count}."
        )

    # Duration drift -------------------------------------------------------
    if input_card and output_card:
        in_p50 = input_card.profile.duration_p50_sec
        out_p50 = output_card.profile.duration_p50_sec
        if in_p50 and out_p50 and not math.isclose(in_p50, 0.0):
            ratio = abs(out_p50 - in_p50) / in_p50
            if ratio > opts.drift_p50_ratio:
                findings.append(
                    f"Median duration shifted {ratio*100:.0f}% (in={in_p50:.1f}s → out={out_p50:.1f}s); "
                    "verify segmentation parameters."
                )

    # Intent-alignment ----------------------------------------------------
    if intent and output_card:
        findings.extend(_check_intent_alignment(intent, output_card.profile))

    llm_used = False
    llm_score: float | None = None
    llm_findings: list[str] = []
    if opts.use_llm and llm_client is not None and intent is not None:
        llm_used = True
        llm_score, llm_findings = _llm_intent_alignment(
            llm_client,
            intent=intent,
            run_card=run_card,
            output_profile=out_profile,
        )

    return CriticReport(
        deterministic_pass=not findings,
        deterministic_findings=findings,
        drop_rates_by_stage=drop_rates,
        output_profile=out_profile,
        llm_used=llm_used,
        llm_intent_alignment_score=llm_score,
        llm_findings=llm_findings,
    )


def _llm_intent_alignment(
    client: "LLMClient",
    *,
    intent: IntentCategories,
    run_card: RunCard,
    output_profile: DatasetProfile,
) -> tuple[float | None, list[str]]:
    """Single, scoped LLM call that grades the output against intent.

    The model receives only summary statistics — never raw audio bytes or
    transcripts — so latency, cost, and privacy stay bounded.
    """

    from nemo_curator.agentic.llm import sys as _sys  # noqa: PLC0415
    from nemo_curator.agentic.llm import user as _user  # noqa: PLC0415

    payload = {
        "intent": intent.model_dump(mode="json"),
        "run_card_summary": {
            "stages": [r.stage_name for r in run_card.stage_records],
            "tasks_in_first": run_card.stage_records[0].tasks_in if run_card.stage_records else None,
            "tasks_out_last": run_card.stage_records[-1].tasks_out if run_card.stage_records else None,
            "errors": run_card.failure_reason,
        },
        "output_profile": output_profile.model_dump(mode="json"),
    }

    messages = [
        _sys(
            "You are the ADV output critic. Compare the structured user intent "
            "to the structured run + output statistics and return JSON exactly "
            "matching this schema: "
            '{"score": <float 0-1>, "findings": [<short string>, ...] }. '
            "Do not invent numbers. Do not request more data. Maximum 8 findings."
        ),
        _user(
            "Score this run against the intent and explain any mismatches.\n\n"
            f"INPUT:\n{payload!r}"
        ),
    ]
    try:
        data = client.chat_json(messages, tier="synth")
    except Exception as exc:  # noqa: BLE001
        return None, [f"LLM critic call failed: {exc}"]

    score = float(data.get("score", 0.0))
    findings = [str(s) for s in data.get("findings", []) if s]
    return score, findings


def _check_intent_alignment(intent: IntentCategories, profile: DatasetProfile) -> list[str]:
    out: list[str] = []
    if intent.duration_max_sec is not None and profile.duration_p95_sec is not None:
        if profile.duration_p95_sec > intent.duration_max_sec * 1.05:
            out.append(
                f"95th-percentile output duration ({profile.duration_p95_sec:.1f}s) exceeds "
                f"intent.duration_max_sec ({intent.duration_max_sec}s)."
            )
    if intent.duration_min_sec is not None and profile.duration_p05_sec is not None:
        if profile.duration_p05_sec < intent.duration_min_sec * 0.95:
            out.append(
                f"5th-percentile output duration ({profile.duration_p05_sec:.1f}s) is below "
                f"intent.duration_min_sec ({intent.duration_min_sec}s)."
            )
    if isinstance(intent.sample_rate, int) and profile.sample_rates_hz:
        majority = max(profile.sample_rates_hz.items(), key=lambda kv: kv[1])[0]
        if int(majority) != intent.sample_rate:
            out.append(
                f"Output majority sample rate ({majority} Hz) does not match intent ({intent.sample_rate} Hz)."
            )
    return out


__all__ = ["CriticOptions", "critique"]
