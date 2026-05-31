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
"""Deterministic prompt-to-pipeline compiler.

The new flow (matches ``INTENT_V2.md`` §6):

    user prompt + source URI
        │
        ▼
    profile_source              (deterministic — Layer 1)
        │
        ▼
    extract_intent              (single LLM call; returns IntentCategories V2)
        │
        ▼
    infer_intent_from_prompt    (heuristic prefills + assumptions)
    apply_profile_prefills      (dataset-profile prefills)
        │
        ▼
    apply_answers               (whatever the user picked in the form)
        │
        ▼
    select_stages               (deterministic, §4 stage-application matrix)
        │
        ▼
    validate + dry-run          (deterministic, with retunes/auto-inserts)

There is no clarification loop and no goal/profile inference. The
clarifier's form is now a one-shot collection of ingredient questions
on top of the already-pre-filled intent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_curator.agentic.cards import DatasetCard
from nemo_curator.agentic.clarifier import (
    apply_answers,
    apply_profile_prefills,
    infer_intent_from_prompt,
)
from nemo_curator.agentic.dryrun import DryRunReport, dry_run_pipeline
from nemo_curator.agentic.intent import IntentCategories
from nemo_curator.agentic.ir import (
    ClusterProfile,
    PipelineIR,
    SinkSpec,
    SourceSpec,
    StageRef,
)
from nemo_curator.agentic.llm import LLMClient
from nemo_curator.agentic.planner_dag import extract_intent
from nemo_curator.agentic.profiler import profile_source
from nemo_curator.agentic.registry import CapabilityRegistry
from nemo_curator.agentic.stage_selector import select_stages
from nemo_curator.agentic.tuner import TunerInputs, tune
from nemo_curator.agentic.validator import ValidationReport, validate

logger = logging.getLogger(__name__)


@dataclass
class DeterministicPlannerResult:
    """Final artifacts from the deterministic compiler path."""

    ir: PipelineIR
    report: ValidationReport
    intent: IntentCategories
    profile: DatasetCard | None
    dry_run: DryRunReport


class DeterministicPlanningError(RuntimeError):
    """Raised when no structurally safe IR can be produced."""

    def __init__(
        self,
        message: str,
        *,
        report: ValidationReport | None = None,
        dry_run: DryRunReport | None = None,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.dry_run = dry_run


def plan_from_prompt(
    prompt: str,
    source_uri: str,
    source_kind: str,
    target_dir: str,
    *,
    llm: LLMClient,
    registry: CapabilityRegistry,
    sample_limit: int = 64,
    tier: str = "synth",
    answers: dict[str, Any] | None = None,
    cluster: ClusterProfile | None = None,
) -> DeterministicPlannerResult:
    """Extract intent with one LLM call, apply prompt+profile prefills, then compile."""

    source = SourceSpec(kind=source_kind, uri=source_uri)  # type: ignore[arg-type]
    try:
        profile = profile_source(source, sample_limit=sample_limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("deterministic planner: profile_source failed: %s", exc)
        profile = None

    intent = extract_intent(prompt, profile, llm, tier=tier)
    if not intent.raw_prompt:
        intent = intent.model_copy(update={"raw_prompt": prompt})

    intent, _prompt_assumptions = infer_intent_from_prompt(prompt, intent)
    intent, _profile_assumptions = apply_profile_prefills(intent, profile)
    if answers:
        intent = apply_answers(intent, answers)

    return plan_from_intent(
        intent,
        source_uri=source_uri,
        source_kind=source_kind,
        target_dir=target_dir,
        registry=registry,
        profile=profile,
        cluster=cluster,
    )


def plan_from_intent(
    intent: IntentCategories,
    *,
    source_uri: str,
    source_kind: str,
    target_dir: str,
    registry: CapabilityRegistry,
    profile: DatasetCard | None = None,
    require_dry_run_clean: bool = True,
    cluster: ClusterProfile | None = None,
) -> DeterministicPlannerResult:
    """Compile a fully-filled intent into a validated, dry-run-clean IR."""

    # Re-validate the intent in case the caller built it with
    # ``model_copy(update=...)`` (which skips validators in Pydantic v2)
    # — that would silently miss our consistency coercions, like
    # speakers=SPLIT + speech_segments → single_speaker_clips. Doing a
    # round-trip through ``model_validate`` is cheap and idempotent.
    intent = IntentCategories.model_validate(intent.model_dump())

    sink = SinkSpec(
        target_dir=target_dir,
        manifest_filename=intent.policy.target_manifest_filename,
        output_format=intent.output.audio_format or "wav",
    )
    source = _source_for_planning(source_kind, source_uri, sink)
    stage_refs = select_stages(intent, registry=registry, sink=sink)

    cluster_profile = cluster or ClusterProfile()
    tuner_result = tune(TunerInputs(
        stages=stage_refs,
        registry=registry,
        cluster=cluster_profile,
    ))

    ir = PipelineIR(
        source=source,
        sink=sink,
        stages=tuner_result.stages,
        intent=intent,
        cluster=cluster_profile,
        executor_config=tuner_result.executor_config,
        executor=tuner_result.executor_config.backend,
    )

    report, dry = _validate_and_dry_run(ir, registry)
    if report.is_ok():
        # The validator may have auto-inserted stages (writers, mono,
        # concat). Re-run the tuner so the new stages also get sized
        # resources + backend hints. The tuner is idempotent on stages
        # it has already tagged, so re-running on the full list is safe.
        post_tuner = tune(TunerInputs(
            stages=list(report.ir.stages),
            registry=registry,
            cluster=cluster_profile,
            # Pin the auto-decided execution mode so the second tuner pass
            # doesn't flip it after the validator inserted extra CPU
            # stages (which don't change the streaming feasibility floor
            # anyway).
            forced_execution_mode=tuner_result.executor_config.execution_mode,
        ))
        report.ir.stages = post_tuner.stages
        report.ir.executor_config = post_tuner.executor_config
    if not report.is_ok():
        details = "; ".join(f"{f.code}:{f.detail}" for f in report.errors())
        raise DeterministicPlanningError(
            f"deterministic planner produced validation errors: {details}",
            report=report,
            dry_run=dry,
        )
    if require_dry_run_clean and dry.has_errors:
        details = "; ".join(
            f"{i['stage_name']}@{i['stage_index']}: {i['issue']}"
            for i in dry.all_issues
        )
        raise DeterministicPlanningError(
            f"deterministic planner produced AudioTask-flow errors: {details}",
            report=report,
            dry_run=dry,
        )

    return DeterministicPlannerResult(
        ir=report.ir,
        report=report,
        intent=intent,
        profile=profile,
        dry_run=dry,
    )


def _source_for_planning(source_kind: str, source_uri: str, sink: SinkSpec) -> SourceSpec:
    """Return a SourceSpec that can be compiled by the existing adapters."""

    if source_kind != "directory":
        return SourceSpec(kind=source_kind, uri=source_uri)  # type: ignore[arg-type]

    from nemo_curator.agentic.adapters import build_manifest_from_directory  # noqa: PLC0415

    manifest_path = (
        Path(sink.target_dir).expanduser().resolve()
        / ".adv"
        / "source_manifest.jsonl"
    )
    built = build_manifest_from_directory(source_uri, output_path=manifest_path)
    return SourceSpec(
        kind="directory",
        uri=source_uri,
        options={"synthesized_manifest_path": str(built)},
    )


# ----------------------------------------------------------------------------
# Validation gate (unchanged from V1; the validator + dry-run own auto-insert)
# ----------------------------------------------------------------------------


def _validate_and_dry_run(
    ir: PipelineIR,
    registry: CapabilityRegistry,
) -> tuple[ValidationReport, DryRunReport]:
    report = validate(ir, registry, intent=ir.intent, mutate=True)
    dry = dry_run_pipeline(report.ir, registry)
    if not dry.has_errors:
        return report, dry

    repaired = _repair_nested_vad_for_concat(report.ir)
    if not repaired:
        return report, dry

    report = validate(report.ir, registry, intent=report.ir.intent, mutate=True)
    dry = dry_run_pipeline(report.ir, registry)
    return report, dry


def _repair_nested_vad_for_concat(ir: PipelineIR) -> bool:
    """Repair legacy IRs by making upstream VAD produce nested segments."""

    changed = False
    for i, ref in enumerate(ir.stages):
        if ref.stage != "SegmentConcatenationStage":
            continue
        for upstream in reversed(ir.stages[:i]):
            if upstream.stage == "VADSegmentationStage" and not upstream.params.get("nested"):
                upstream.params = {**upstream.params, "nested": True}
                changed = True
                break
    return changed


__all__ = [
    "DeterministicPlannerResult",
    "DeterministicPlanningError",
    "plan_from_intent",
    "plan_from_prompt",
]
