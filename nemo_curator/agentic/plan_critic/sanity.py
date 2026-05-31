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
"""Deterministic build-time critic.

These checks run after every compile and never call an LLM. They catch
structural bugs the planner's :mod:`validator` could not flag because
they require cross-referencing the *intent* against the produced IR:

- intent fields that should have produced a stage but didn't,
- intent fields that *did* produce a stage but with the wrong knob,
- contradictory intent that the selector silently degraded.

Each finding carries an actionable ``suggested_change`` whenever the fix
is unambiguous, so the orchestrator can re-plan automatically. Findings
without a patch are surfaced for the user to review.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.agentic.intent import FilterMode, IntentCategories
from nemo_curator.agentic.ir import PipelineIR, StageRef
from nemo_curator.agentic.plan_critic.base import (
    CriticFinding,
    CriticReport,
    CriticSeverity,
)


class SanityCritic:
    """Cross-checks intent ↔ IR for structural mismatches."""

    name = "sanity"

    def review(
        self,
        *,
        intent: IntentCategories,
        ir: PipelineIR,
        profile: Any = None,
    ) -> CriticReport:
        report = CriticReport()
        stage_names = [s.stage for s in ir.stages]
        stage_set = set(stage_names)

        self._check_writer(stage_set, report)
        self._check_speaker_intent_vs_stages(intent, stage_names, ir.stages, report)
        self._check_quality_intent_vs_stages(intent, stage_set, report)
        self._check_segmentation_consistency(intent, stage_set, report)
        self._check_threshold_bounds(intent, report)
        self._check_orphan_filters(ir.stages, report)
        self._check_long_window_requires_alm(intent, stage_set, report)
        return report

    # ------------------------------------------------------------------ #
    # 1. A writer must exist or the pipeline produces nothing.
    # ------------------------------------------------------------------ #
    def _check_writer(self, stage_set: set[str], report: CriticReport) -> None:
        writer_stages = {"ManifestWriterStage", "AudioToDocumentStage"}
        if not writer_stages & stage_set:
            report.findings.append(CriticFinding(
                severity=CriticSeverity.ERROR,
                code="missing_output_writer",
                detail="Pipeline has no manifest/document writer; output will be lost.",
                source=self.name,
                rationale=(
                    "Every pipeline must end with a writer stage. The "
                    "validator usually auto-inserts ManifestWriterStage — "
                    "if it didn't, the IR was hand-edited or the registry "
                    "is missing the writer card."
                ),
            ))

    # ------------------------------------------------------------------ #
    # 2. Speakers intent ↔ presence of diarization / separation stages.
    # ------------------------------------------------------------------ #
    def _check_speaker_intent_vs_stages(
        self,
        intent: IntentCategories,
        stage_names: list[str],
        stages: list[StageRef],
        report: CriticReport,
    ) -> None:
        spk = intent.speakers
        diar_stages = {"InferenceSortformerStage", "PyAnnoteDiarizationStage"}
        has_diar = bool(set(stage_names) & diar_stages)
        has_sep = "SpeakerSeparationStage" in stage_names

        if spk.mode == FilterMode.SPLIT and not has_sep:
            unit = intent.segmentation.output_unit
            if unit == "original_files":
                report.findings.append(CriticFinding(
                    severity=CriticSeverity.WARN,
                    code="speaker_split_coerced",
                    detail=(
                        "speakers.mode=SPLIT was degraded to diarize-only "
                        "because output_unit=original_files keeps one row "
                        "per file. Switch output_unit to "
                        "'single_speaker_clips' to actually fan out."
                    ),
                    field="segmentation.output_unit",
                    suggested_change={"segmentation.output_unit": "single_speaker_clips"},
                    source=self.name,
                ))
            else:
                report.findings.append(CriticFinding(
                    severity=CriticSeverity.ERROR,
                    code="speaker_split_missing_stage",
                    detail=(
                        "speakers.mode=SPLIT but no SpeakerSeparationStage "
                        "was emitted. The selector should always produce it "
                        "for this mode."
                    ),
                    stage="SpeakerSeparationStage",
                    source=self.name,
                ))

        if spk.mode == FilterMode.FILTER:
            if not has_diar:
                report.findings.append(CriticFinding(
                    severity=CriticSeverity.ERROR,
                    code="speaker_filter_missing_diarization",
                    detail=(
                        "speakers.mode=FILTER but no diarization stage was "
                        "emitted; speaker count is unknown and the filter "
                        "would drop everything."
                    ),
                    source=self.name,
                ))
            bounds = (spk.target_count, spk.min_count, spk.max_count)
            if all(b is None for b in bounds):
                report.findings.append(CriticFinding(
                    severity=CriticSeverity.WARN,
                    code="speaker_filter_no_bounds",
                    detail=(
                        "speakers.mode=FILTER but no target_count / "
                        "min_count / max_count given — every row will pass. "
                        "Did you mean ANNOTATE?"
                    ),
                    field="speakers.mode",
                    suggested_change={"speakers.mode": "annotate"},
                    source=self.name,
                ))

        # When the user asks for single-speaker clips we expect both
        # diarization + separation; flag if either is missing.
        if intent.segmentation.output_unit == "single_speaker_clips" and not has_sep:
            report.findings.append(CriticFinding(
                severity=CriticSeverity.ERROR,
                code="single_speaker_clips_missing_separation",
                detail=(
                    "output_unit=single_speaker_clips needs "
                    "SpeakerSeparationStage but it isn't in the pipeline."
                ),
                stage="SpeakerSeparationStage",
                source=self.name,
            ))

    # ------------------------------------------------------------------ #
    # 3. Quality intent ↔ scoring stages present.
    # ------------------------------------------------------------------ #
    def _check_quality_intent_vs_stages(
        self,
        intent: IntentCategories,
        stage_set: set[str],
        report: CriticReport,
    ) -> None:
        q = intent.quality
        if q.mos in (FilterMode.ANNOTATE, FilterMode.FILTER) and "UTMOSFilterStage" not in stage_set:
            report.findings.append(CriticFinding(
                severity=CriticSeverity.ERROR,
                code="utmos_intent_no_stage",
                detail="quality.mos requires UTMOSFilterStage which is not in the pipeline.",
                stage="UTMOSFilterStage",
                source=self.name,
            ))
        if q.sigmos in (FilterMode.ANNOTATE, FilterMode.FILTER) and "SIGMOSFilterStage" not in stage_set:
            report.findings.append(CriticFinding(
                severity=CriticSeverity.ERROR,
                code="sigmos_intent_no_stage",
                detail="quality.sigmos requires SIGMOSFilterStage which is not in the pipeline.",
                stage="SIGMOSFilterStage",
                source=self.name,
            ))

        # Gates referencing a key whose scoring stage isn't present.
        gate_required: dict[str, str] = {
            "utmos_mos": "UTMOSFilterStage",
            "band_prediction": "BandFilterStage",
        }
        for axis in ("ovrl", "noise", "sig", "col", "disc", "loud", "reverb"):
            gate_required[f"sigmos_{axis}"] = "SIGMOSFilterStage"
        for gate in q.gates:
            need = gate_required.get(gate.key)
            if need and need not in stage_set:
                report.findings.append(CriticFinding(
                    severity=CriticSeverity.ERROR,
                    code="gate_missing_scoring_stage",
                    detail=(
                        f"quality.gates references '{gate.key}' but "
                        f"{need} is not in the pipeline; the gate would "
                        "always pass."
                    ),
                    stage=need,
                    source=self.name,
                ))

    # ------------------------------------------------------------------ #
    # 4. Segmentation knob consistency.
    # ------------------------------------------------------------------ #
    def _check_segmentation_consistency(
        self,
        intent: IntentCategories,
        stage_set: set[str],
        report: CriticReport,
    ) -> None:
        seg = intent.segmentation
        unit = seg.output_unit

        if seg.duration_min_sec is not None and seg.duration_max_sec is not None:
            if seg.duration_min_sec > seg.duration_max_sec:
                report.findings.append(CriticFinding(
                    severity=CriticSeverity.ERROR,
                    code="duration_min_gt_max",
                    detail=(
                        f"segmentation.duration_min_sec ({seg.duration_min_sec}) "
                        f"> duration_max_sec ({seg.duration_max_sec}); no row can pass."
                    ),
                    field="segmentation.duration_min_sec",
                    source=self.name,
                ))

        if unit == "long_windows" and seg.long_window_sec is None:
            report.findings.append(CriticFinding(
                severity=CriticSeverity.ERROR,
                code="long_windows_no_window_sec",
                detail=(
                    "output_unit=long_windows but segmentation.long_window_sec "
                    "is unset; ALM data builder cannot run."
                ),
                field="segmentation.long_window_sec",
                source=self.name,
            ))

        # speech_policy is only meaningful on whole-file outputs; flag if set
        # alongside a fan-out unit (the selector ignores it but the user
        # likely meant something).
        if unit != "original_files" and seg.speech_policy != FilterMode.OFF:
            report.findings.append(CriticFinding(
                severity=CriticSeverity.WARN,
                code="speech_policy_ignored",
                detail=(
                    f"segmentation.speech_policy={seg.speech_policy.value} is "
                    f"only meaningful when output_unit='original_files' "
                    f"(current: {unit}); the setting will be ignored."
                ),
                field="segmentation.speech_policy",
                suggested_change={"segmentation.speech_policy": "off"},
                source=self.name,
            ))

    # ------------------------------------------------------------------ #
    # 5. Threshold values should be inside the natural MOS range.
    # ------------------------------------------------------------------ #
    def _check_threshold_bounds(
        self,
        intent: IntentCategories,
        report: CriticReport,
    ) -> None:
        q = intent.quality
        if q.mos_threshold is not None and not (0.0 <= q.mos_threshold <= 5.0):
            report.findings.append(CriticFinding(
                severity=CriticSeverity.ERROR,
                code="utmos_threshold_out_of_range",
                detail=(
                    f"quality.mos_threshold={q.mos_threshold} is outside the "
                    "valid MOS range [0, 5]. Did the extractor misread the prompt?"
                ),
                field="quality.mos_threshold",
                source=self.name,
            ))

        # SIGMOS thresholds: same [0, 5] range across all axes.
        for axis, value in (q.sigmos_thresholds or {}).items():
            if value is None:
                continue
            if not (0.0 <= float(value) <= 5.0):
                report.findings.append(CriticFinding(
                    severity=CriticSeverity.ERROR,
                    code="sigmos_threshold_out_of_range",
                    detail=(
                        f"quality.sigmos_thresholds[{axis}]={value} is "
                        "outside the valid MOS range [0, 5]."
                    ),
                    field=f"quality.sigmos_thresholds.{axis}",
                    source=self.name,
                ))

    # ------------------------------------------------------------------ #
    # 6. Orphan PreserveByValueStage (no producing stage upstream).
    # ------------------------------------------------------------------ #
    def _check_orphan_filters(
        self,
        stages: list[StageRef],
        report: CriticReport,
    ) -> None:
        # Map filter input keys → upstream producer stage(s).
        producers: dict[str, set[str]] = {
            "utmos_mos": {"UTMOSFilterStage"},
            "band_prediction": {"BandFilterStage"},
            "num_speakers": {"InferenceSortformerStage", "PyAnnoteDiarizationStage"},
            "duration": {"GetAudioDurationStage", "VADSegmentationStage", "SegmentExtractionStage"},
        }
        for axis in ("ovrl", "noise", "sig", "col", "disc", "loud", "reverb"):
            producers[f"sigmos_{axis}"] = {"SIGMOSFilterStage"}

        seen: set[str] = set()
        for ref in stages:
            seen.add(ref.stage)
            if ref.stage != "PreserveByValueStage":
                continue
            key = ref.params.get("input_value_key")
            if not key:
                continue
            req = producers.get(str(key))
            if req and not (req & seen):
                report.findings.append(CriticFinding(
                    severity=CriticSeverity.ERROR,
                    code="orphan_preserve_by_value",
                    detail=(
                        f"PreserveByValueStage on '{key}' has no upstream "
                        f"producer ({' / '.join(sorted(req))} expected before it)."
                    ),
                    stage="PreserveByValueStage",
                    source=self.name,
                ))

    # ------------------------------------------------------------------ #
    # 7. long_windows → ALMDataBuilderStage must be present.
    # ------------------------------------------------------------------ #
    def _check_long_window_requires_alm(
        self,
        intent: IntentCategories,
        stage_set: set[str],
        report: CriticReport,
    ) -> None:
        if intent.segmentation.output_unit == "long_windows" and "ALMDataBuilderStage" not in stage_set:
            report.findings.append(CriticFinding(
                severity=CriticSeverity.ERROR,
                code="long_windows_missing_alm",
                detail=(
                    "output_unit=long_windows requires ALMDataBuilderStage "
                    "but it isn't in the pipeline."
                ),
                stage="ALMDataBuilderStage",
                source=self.name,
            ))
