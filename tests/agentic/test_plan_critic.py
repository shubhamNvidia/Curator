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
"""Tests for the build-time critic framework.

Three layers covered here:

1. ``SanityCritic`` — deterministic checks against intent ↔ IR.
2. ``PlanCritic`` — LLM-driven critic with ``MockLLM``.
3. ``review_and_replan`` — orchestrator applies patches + re-plans.
"""

from __future__ import annotations

import json

import pytest

from nemo_curator.agentic.deterministic_planner import plan_from_intent
from nemo_curator.agentic.intent import (
    FilterMode,
    IntentCategories,
    OutputFormat,
    Quality,
    QualityGate,
    Segmentation,
    Speakers,
)
from nemo_curator.agentic.ir import ClusterProfile, PipelineIR, SinkSpec, SourceSpec, StageRef
from nemo_curator.agentic.llm import Message, MockLLM
from nemo_curator.agentic.plan_critic import (
    CriticFinding,
    CriticReport,
    CriticSeverity,
    PlanCritic,
    SanityCritic,
    review_and_replan,
)
from nemo_curator.agentic.registry import build_registry


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def registry():
    return build_registry(cross_check_runtime=False, eager=False)


def _plan(intent: IntentCategories, registry, tmp_dir):
    return plan_from_intent(
        intent,
        source_uri=str(tmp_dir / "audio"),
        source_kind="directory",
        target_dir=str(tmp_dir / "out"),
        registry=registry,
        cluster=ClusterProfile(),
    )


def _hand_ir(stages: list[StageRef], intent: IntentCategories) -> PipelineIR:
    """Build a minimal IR without running the planner (for negative tests)."""

    return PipelineIR(
        source=SourceSpec(kind="directory", uri="/tmp/in"),
        sink=SinkSpec(target_dir="/tmp/out", manifest_filename="out.json", output_format="wav"),
        stages=stages,
        intent=intent,
    )


# --------------------------------------------------------------------------- #
# SanityCritic
# --------------------------------------------------------------------------- #


class TestSanityCritic:
    def test_clean_pipeline_no_findings(self, registry, tmp_path):
        """A well-formed intent → IR triggers nothing from the sanity critic."""

        (tmp_path / "audio").mkdir()
        (tmp_path / "audio" / "sample.wav").write_bytes(b"")
        intent = IntentCategories(
            output=OutputFormat(sample_rate=16000, channels="mono", resample_input=True),
            segmentation=Segmentation(
                output_unit="single_speaker_clips",
                duration_min_sec=2.0,
                duration_max_sec=20.0,
            ),
            speakers=Speakers(mode=FilterMode.SPLIT),
        )
        result = _plan(intent, registry, tmp_path)
        report = SanityCritic().review(intent=intent, ir=result.ir)
        assert report.findings == [], report.findings

    def test_missing_writer_is_error(self):
        """A hand-built IR without a writer trips the missing_output_writer code."""

        intent = IntentCategories()
        ir = _hand_ir(
            stages=[StageRef(stage="MonoConversionStage")],
            intent=intent,
        )
        report = SanityCritic().review(intent=intent, ir=ir)
        codes = {f.code for f in report.findings}
        assert "missing_output_writer" in codes
        assert any(f.severity == CriticSeverity.ERROR for f in report.findings)

    def test_speaker_filter_without_bounds_warns_and_proposes_annotate(self, registry, tmp_path):
        """speakers.mode=FILTER with no counts → warn + patch to ANNOTATE."""

        intent = IntentCategories(
            speakers=Speakers(mode=FilterMode.FILTER),  # no counts
            segmentation=Segmentation(output_unit="original_files"),
        )
        # Build the IR via the planner so we exercise the realistic case.
        # Some intents are rejected at plan time; if so, hand-build an IR
        # with diarization so we isolate this check.
        try:
            result = _plan(intent, registry, tmp_path)
            ir = result.ir
        except Exception:
            ir = _hand_ir(
                stages=[
                    StageRef(stage="ManifestReaderStage"),
                    StageRef(stage="InferenceSortformerStage"),
                    StageRef(stage="ManifestWriterStage"),
                ],
                intent=intent,
            )

        report = SanityCritic().review(intent=intent, ir=ir)
        codes = {f.code for f in report.findings}
        assert "speaker_filter_no_bounds" in codes
        finding = next(f for f in report.findings if f.code == "speaker_filter_no_bounds")
        assert finding.severity == CriticSeverity.WARN
        assert finding.suggested_change == {"speakers.mode": "annotate"}

    def test_single_speaker_clips_without_separation_errors(self):
        """output_unit=single_speaker_clips but no SpeakerSeparationStage."""

        intent = IntentCategories(
            segmentation=Segmentation(output_unit="single_speaker_clips"),
        )
        ir = _hand_ir(
            stages=[
                StageRef(stage="ManifestReaderStage"),
                StageRef(stage="ManifestWriterStage"),
            ],
            intent=intent,
        )
        report = SanityCritic().review(intent=intent, ir=ir)
        assert any(f.code == "single_speaker_clips_missing_separation" for f in report.findings)
        assert any(f.severity == CriticSeverity.ERROR for f in report.findings)

    def test_quality_intent_without_scoring_stage_errors(self):
        """quality.mos=FILTER but no UTMOSFilterStage emitted."""

        intent = IntentCategories(
            quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5),
        )
        ir = _hand_ir(
            stages=[
                StageRef(stage="ManifestReaderStage"),
                StageRef(stage="ManifestWriterStage"),
            ],
            intent=intent,
        )
        report = SanityCritic().review(intent=intent, ir=ir)
        assert any(f.code == "utmos_intent_no_stage" for f in report.findings)

    def test_gate_without_scoring_stage_errors(self):
        """A QualityGate on sigmos_noise without SIGMOSFilterStage upstream."""

        intent = IntentCategories(
            quality=Quality(gates=[QualityGate(key="sigmos_noise", operator="ge", value=4.0)]),
        )
        ir = _hand_ir(
            stages=[
                StageRef(stage="ManifestReaderStage"),
                StageRef(
                    stage="PreserveByValueStage",
                    params={"input_value_key": "sigmos_noise", "target_value": 4.0, "operator": "ge"},
                ),
                StageRef(stage="ManifestWriterStage"),
            ],
            intent=intent,
        )
        report = SanityCritic().review(intent=intent, ir=ir)
        assert any(f.code == "gate_missing_scoring_stage" for f in report.findings)

    def test_orphan_preserve_by_value_errors(self):
        """PreserveByValueStage on utmos_mos with no UTMOSFilterStage upstream."""

        intent = IntentCategories()
        ir = _hand_ir(
            stages=[
                StageRef(stage="ManifestReaderStage"),
                StageRef(
                    stage="PreserveByValueStage",
                    params={"input_value_key": "utmos_mos", "target_value": 3.5, "operator": "ge"},
                ),
                StageRef(stage="ManifestWriterStage"),
            ],
            intent=intent,
        )
        report = SanityCritic().review(intent=intent, ir=ir)
        assert any(f.code == "orphan_preserve_by_value" for f in report.findings)

    def test_duration_min_gt_max_errors(self):
        """duration_min_sec > duration_max_sec → error.

        Pydantic catches this at IntentCategories construction, but the
        critic still defends against hand-built intents / model_construct
        callers, so we bypass validation here to exercise the check.
        """

        # Bypass validation at every layer (Segmentation, IntentCategories,
        # PipelineIR all re-check the bound) using model_construct.
        seg = Segmentation.model_construct(duration_min_sec=30.0, duration_max_sec=5.0)
        intent = IntentCategories.model_construct(segmentation=seg)
        ir = PipelineIR.model_construct(
            source=SourceSpec(kind="directory", uri="/tmp/in"),
            sink=SinkSpec(target_dir="/tmp/out", manifest_filename="out.json", output_format="wav"),
            stages=[
                StageRef(stage="ManifestReaderStage"),
                StageRef(stage="ManifestWriterStage"),
            ],
            intent=intent,
        )
        report = SanityCritic().review(intent=intent, ir=ir)
        assert any(f.code == "duration_min_gt_max" for f in report.findings)

    def test_speech_policy_ignored_on_fan_out_unit(self):
        """speech_policy set on single_speaker_clips → warn + patch to off."""

        intent = IntentCategories(
            segmentation=Segmentation(
                output_unit="single_speaker_clips",
                speech_policy=FilterMode.ANNOTATE,
            ),
            speakers=Speakers(mode=FilterMode.SPLIT),
        )
        ir = _hand_ir(
            stages=[
                StageRef(stage="ManifestReaderStage"),
                StageRef(stage="SpeakerSeparationStage"),
                StageRef(stage="ManifestWriterStage"),
            ],
            intent=intent,
        )
        report = SanityCritic().review(intent=intent, ir=ir)
        finding = next((f for f in report.findings if f.code == "speech_policy_ignored"), None)
        assert finding is not None
        assert finding.suggested_change == {"segmentation.speech_policy": "off"}

    def test_threshold_out_of_range_errors(self):
        """quality.mos_threshold outside [0, 5] → error.

        Pydantic clamps this at Quality construction; bypass via
        ``model_construct`` to test the critic's defense-in-depth path.
        """

        q = Quality.model_construct(mos=FilterMode.FILTER, mos_threshold=7.5)
        intent = IntentCategories.model_construct(quality=q)
        ir = _hand_ir(
            stages=[
                StageRef(stage="ManifestReaderStage"),
                StageRef(stage="UTMOSFilterStage", params={"mos_threshold": 0.0}),
                StageRef(stage="ManifestWriterStage"),
            ],
            intent=intent,
        )
        report = SanityCritic().review(intent=intent, ir=ir)
        assert any(f.code == "utmos_threshold_out_of_range" for f in report.findings)


# --------------------------------------------------------------------------- #
# PlanCritic (LLM-driven, MockLLM responder)
# --------------------------------------------------------------------------- #


class TestPlanCritic:
    def test_returns_empty_report_when_llm_none(self):
        """No LLM available → critic is a no-op, never raises."""

        critic = PlanCritic(llm=None)
        ir = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=IntentCategories())
        report = critic.review(intent=IntentCategories(), ir=ir)
        assert report.findings == []

    def test_parses_well_formed_response(self):
        """A clean JSON response is turned into typed findings."""

        canned = json.dumps({
            "findings": [
                {
                    "severity": "warn",
                    "code": "studio_threshold_too_low",
                    "detail": "Prompt says studio but mos_threshold is 3.2.",
                    "field": "quality.mos_threshold",
                    "suggested_change": {"quality.mos_threshold": 4.3},
                    "rationale": "Studio implies UTMOS >= 4.0.",
                },
                {
                    "severity": "info",
                    "code": "vad_inserted_post_speaker",
                    "detail": "VAD was added after SpeakerSeparation.",
                }
            ]
        })
        llm = MockLLM(responder=lambda _msgs, _tier: canned)
        critic = PlanCritic(llm=llm)
        ir = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=IntentCategories(raw_prompt="studio quality"))
        report = critic.review(intent=IntentCategories(raw_prompt="studio quality"), ir=ir)
        assert len(report.findings) == 2
        warn = report.findings[0]
        assert warn.severity == CriticSeverity.WARN
        assert warn.suggested_change == {"quality.mos_threshold": 4.3}
        assert warn.source == "plan"

    def test_malformed_response_does_not_crash(self):
        """If the LLM returns garbage, we swallow it and report nothing."""

        llm = MockLLM(responder=lambda _msgs, _tier: "this is not json at all")
        critic = PlanCritic(llm=llm)
        ir = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=IntentCategories())
        report = critic.review(intent=IntentCategories(), ir=ir)
        assert report.findings == []

    def test_empty_findings_array_accepted(self):
        """``{"findings": []}`` returns an empty report — that's success."""

        llm = MockLLM(responder=lambda _msgs, _tier: '{"findings": []}')
        critic = PlanCritic(llm=llm)
        ir = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=IntentCategories())
        report = critic.review(intent=IntentCategories(), ir=ir)
        assert report.findings == []

    def test_finding_without_detail_is_dropped(self):
        """Findings missing required text are silently dropped."""

        canned = json.dumps({"findings": [
            {"severity": "warn", "code": "x", "detail": ""},
            {"severity": "warn", "code": "y", "detail": "real"},
        ]})
        llm = MockLLM(responder=lambda _msgs, _tier: canned)
        critic = PlanCritic(llm=llm)
        ir = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=IntentCategories())
        report = critic.review(intent=IntentCategories(), ir=ir)
        codes = [f.code for f in report.findings]
        assert codes == ["y"]

    def test_payload_includes_insert_reasons(self):
        """The user message must carry insert_reason strings so the LLM can reason."""

        captured: dict = {}

        def capture(messages: list[Message], _tier: str) -> str:
            captured["user"] = messages[-1].content
            return '{"findings": []}'

        llm = MockLLM(responder=capture)
        critic = PlanCritic(llm=llm)
        ir = _hand_ir(
            stages=[
                StageRef(
                    stage="SpeakerSeparationStage",
                    insert_reason="user wants per-speaker clips",
                ),
                StageRef(stage="ManifestWriterStage"),
            ],
            intent=IntentCategories(raw_prompt="per-speaker"),
        )
        critic.review(intent=IntentCategories(raw_prompt="per-speaker"), ir=ir)
        assert "user wants per-speaker clips" in captured["user"]


# --------------------------------------------------------------------------- #
# review_and_replan orchestrator
# --------------------------------------------------------------------------- #


class TestOrchestrator:
    def test_no_critics_returns_original(self):
        """Empty critic list → single iteration, no re-plan."""

        intent = IntentCategories()
        ir = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=intent)
        review = review_and_replan(
            intent=intent,
            ir=ir,
            replan=lambda _: ir,
            critics=[],
            max_iterations=2,
        )
        assert review.iterations == 1
        assert review.re_planned is False
        assert review.report.findings == []

    def test_replan_applies_patch_and_stops_on_clean_second_pass(self):
        """One critic emits a patch → orchestrator re-plans → critic clears."""

        intent = IntentCategories(
            quality=Quality(mos=FilterMode.FILTER, mos_threshold=2.0),
            raw_prompt="studio",
        )
        ir1 = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=intent)
        ir2 = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=intent)

        class _FakeCritic:
            name = "fake"

            def __init__(self):
                self.calls = 0

            def review(self, *, intent, ir, profile=None):
                self.calls += 1
                report = CriticReport()
                if self.calls == 1:
                    report.findings.append(CriticFinding(
                        severity=CriticSeverity.WARN,
                        code="raise_threshold",
                        detail="studio implies higher MOS",
                        field="quality.mos_threshold",
                        suggested_change={"quality.mos_threshold": 4.3},
                        source=self.name,
                    ))
                return report

        critic = _FakeCritic()

        def replan(it: IntentCategories):
            assert it.quality.mos_threshold == 4.3  # patch was applied
            return ir2

        review = review_and_replan(
            intent=intent,
            ir=ir1,
            replan=replan,
            critics=[critic],
            max_iterations=3,
        )
        assert review.iterations == 2
        assert review.re_planned is True
        assert len(review.patches_applied) == 1
        assert critic.calls == 2  # initial + post-replan

    def test_info_only_findings_do_not_trigger_replan(self):
        """``info`` severity never re-plans even if a patch is supplied."""

        intent = IntentCategories()
        ir = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=intent)

        class _InfoCritic:
            name = "info"

            def review(self, *, intent, ir, profile=None):
                r = CriticReport()
                r.findings.append(CriticFinding(
                    severity=CriticSeverity.INFO,
                    code="x", detail="just fyi",
                    suggested_change={"quality.mos_threshold": 4.0},
                    source=self.name,
                ))
                return r

        review = review_and_replan(
            intent=intent,
            ir=ir,
            replan=lambda _: ir,
            critics=[_InfoCritic()],
            max_iterations=3,
        )
        assert review.iterations == 1
        assert review.re_planned is False

    def test_invalid_patch_is_demoted_and_does_not_crash(self):
        """LLM proposes an invalid enum value → patch dropped, finding kept.

        Reproduces the production failure where the plan critic
        suggested ``quality.mos="preserve"`` and the whole batch crashed
        because the schema rejected the enum.
        """

        intent = IntentCategories(
            quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.4),
        )
        ir = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=intent)

        class _BadEnumCritic:
            name = "fake"

            def review(self, *, intent, ir, profile=None):
                r = CriticReport()
                r.findings.append(CriticFinding(
                    severity=CriticSeverity.WARN,
                    code="bad_enum",
                    detail="propose impossible value",
                    field="quality.mos",
                    suggested_change={"quality.mos": "preserve"},
                    source=self.name,
                ))
                return r

        review = review_and_replan(
            intent=intent,
            ir=ir,
            replan=lambda _: ir,
            critics=[_BadEnumCritic()],
            max_iterations=3,
        )
        # No re-plan: the invalid patch was demoted, not applied.
        assert review.re_planned is False
        # Finding still surfaces (it's now INFO) so the user sees what happened.
        codes = [f.code for f in review.report.findings]
        assert any(c.endswith("__invalid_patch") for c in codes), codes
        demoted = next(f for f in review.report.findings if f.code.endswith("__invalid_patch"))
        assert demoted.severity == CriticSeverity.INFO
        assert demoted.suggested_change is None
        assert "schema rejected" in (demoted.rationale or "").lower()

    def test_user_locked_paths_are_never_overridden(self):
        """When the user explicitly answered ``duration_max_sec`` in the
        clarifier form, the critic must NOT silently rewrite it.

        Reproduces production run 72eb5552bd2e8772 where the user
        picked 120 s for the max but the LLM critic patched it to 15 s
        because the original prompt said "short". The user's pick wins.
        """

        intent = IntentCategories(
            segmentation=Segmentation(
                output_unit="speech_segments",
                duration_min_sec=2.0,
                duration_max_sec=120.0,
            ),
            notes=[
                "clarifier_answer:segmentation.duration_max_sec=120",
                "clarifier_answer:segmentation.duration_min_sec=2",
            ],
        )
        ir = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=intent)

        class _NaggingCritic:
            name = "fake"

            def __init__(self):
                self.calls = 0

            def review(self, *, intent, ir, profile=None):
                self.calls += 1
                r = CriticReport()
                r.findings.append(CriticFinding(
                    severity=CriticSeverity.WARN,
                    code="duration_max_too_long",
                    detail="prompt said short, max is 120",
                    field="segmentation.duration_max_sec",
                    suggested_change={"segmentation.duration_max_sec": 15},
                    source=self.name,
                ))
                return r

        critic = _NaggingCritic()
        review = review_and_replan(
            intent=intent,
            ir=ir,
            replan=lambda _: ir,
            critics=[critic],
            max_iterations=3,
        )
        # The locked path must be untouched: 120, not 15.
        assert review.intent.segmentation.duration_max_sec == 120.0
        # No re-plan happened — the patch was demoted, not applied.
        assert review.re_planned is False
        # The finding still surfaces in the UI, but as INFO + suffixed
        # code so the user sees why it was skipped.
        codes = [f.code for f in review.report.findings]
        assert any(c.endswith("__user_locked") for c in codes), codes
        locked = next(f for f in review.report.findings if f.code.endswith("__user_locked"))
        assert locked.severity == CriticSeverity.INFO
        assert locked.suggested_change is None
        assert (
            "user explicitly answered" in (locked.rationale or "").lower()
            or "user's pick wins" in (locked.rationale or "").lower()
            or "form pick" in (locked.rationale or "").lower()
        )

    def test_patch_coerced_by_validator_is_surfaced_as_info(self):
        """When the intent validator reverts (part of) a patch to keep
        the model self-consistent, ``apply_findings`` must surface the
        revert so the critic UI doesn't claim "Applied: …" while the
        value actually settled elsewhere.

        Reproduces run ``c27f2786eaad1d75``: the Plan Critic patched
        ``output_unit=single_speaker_clips`` even though the user had
        locked ``speakers.mode=filter`` in the clarifier; the validator
        immediately reverted the output_unit to ``speech_segments``
        because FILTER mode cannot produce per-speaker clips. Without
        this surfacing the UI showed "Applied: output_unit =
        single_speaker_clips" but the compiled pipeline disagreed,
        which then tripped the SanityCritic with
        ``single_speaker_clips_missing_separation``.
        """

        from nemo_curator.agentic.plan_critic.base import apply_findings

        intent = IntentCategories(
            segmentation=Segmentation(output_unit="original_files"),
            speakers=Speakers(mode=FilterMode.FILTER, exclude_overlaps=True),
            notes=["clarifier_answer:speakers.mode=filter"],
        )
        report = CriticReport()
        report.findings.append(CriticFinding(
            severity=CriticSeverity.WARN,
            code="tts_missing_segmentation",
            detail="TTS wants single_speaker_clips",
            field="segmentation.output_unit",
            suggested_change={"segmentation.output_unit": "single_speaker_clips"},
            source="plan",
        ))
        patched, applied = apply_findings(intent, report)
        # Validator reverted the patch.
        assert patched.segmentation.output_unit == "speech_segments"
        assert patched.speakers.mode == FilterMode.FILTER
        # Nothing got recorded as "really applied".
        assert applied == []
        # Finding is now INFO with the coerced suffix and a rationale
        # so the user sees what actually happened.
        codes = [f.code for f in report.findings]
        assert any(c.endswith("__validator_coerced") for c in codes), codes
        f = next(x for x in report.findings if x.code.endswith("__validator_coerced"))
        assert f.severity == CriticSeverity.INFO
        assert f.suggested_change is None
        assert "validator coerced" in (f.rationale or "").lower()

    def test_user_locked_paths_helper_parses_clarifier_notes(self):
        intent = IntentCategories(notes=[
            "prompt_inference: irrelevant",
            "clarifier_answer:segmentation.duration_max_sec=120",
            "clarifier_answer:quality.mos=filter",
        ])
        from nemo_curator.agentic.plan_critic.base import user_locked_paths
        assert user_locked_paths(intent) == {
            "segmentation.duration_max_sec",
            "quality.mos",
        }

    def test_unlocked_paths_in_same_patch_are_still_applied(self):
        """Mixed patches: ``user_locked_paths`` only blocks LOCKED paths.

        If a finding proposes an unlocked path it still applies.
        """

        intent = IntentCategories(
            segmentation=Segmentation(
                output_unit="speech_segments",
                duration_max_sec=120.0,
            ),
            notes=["clarifier_answer:segmentation.duration_max_sec=120"],
        )
        ir = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=intent)

        class _MixedCritic:
            name = "fake"

            def review(self, *, intent, ir, profile=None):
                r = CriticReport()
                r.findings.append(CriticFinding(
                    severity=CriticSeverity.WARN,
                    code="touches_locked_field",
                    detail="proposes overriding the user's max",
                    suggested_change={"segmentation.duration_max_sec": 15},
                    source=self.name,
                ))
                r.findings.append(CriticFinding(
                    severity=CriticSeverity.WARN,
                    code="touches_free_field",
                    detail="proposes setting min — user didn't lock this",
                    suggested_change={"segmentation.duration_min_sec": 2.0},
                    source=self.name,
                ))
                return r

        captured: list[IntentCategories] = []

        def replan(new_intent):
            captured.append(new_intent)
            return ir

        review = review_and_replan(
            intent=intent,
            ir=ir,
            replan=replan,
            critics=[_MixedCritic()],
            max_iterations=3,
        )
        assert review.intent.segmentation.duration_max_sec == 120.0
        assert review.intent.segmentation.duration_min_sec == 2.0
        assert review.re_planned is True
        # The locked finding was demoted; the unlocked one was applied.
        codes = [f.code for f in review.report.findings]
        assert any(c == "touches_free_field" for c in codes)
        assert any(c.endswith("__user_locked") for c in codes)

    def test_replan_failure_does_not_crash(self):
        """If the second compile raises, we record a warning and stop."""

        intent = IntentCategories()
        ir = _hand_ir(stages=[StageRef(stage="ManifestWriterStage")], intent=intent)

        class _Critic:
            name = "fake"

            def review(self, *, intent, ir, profile=None):
                r = CriticReport()
                r.findings.append(CriticFinding(
                    severity=CriticSeverity.WARN,
                    code="x", detail="x",
                    suggested_change={"output.sample_rate": 22050},
                    source=self.name,
                ))
                return r

        def replan(_):
            raise RuntimeError("compile failed")

        review = review_and_replan(
            intent=intent, ir=ir, replan=replan, critics=[_Critic()], max_iterations=3,
        )
        assert any(f.code == "replan_failed" for f in review.report.findings)
        # Even on failure we report the iteration count truthfully.
        assert review.iterations == 1
