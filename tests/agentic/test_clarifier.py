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
"""Tests for the V2 ingredient-picker clarification form.

The V2 clarifier exposes four pure helpers:

- ``infer_intent_from_prompt`` (prompt heuristics → namespaced intent),
- ``apply_profile_prefills``   (DatasetCard heuristics → intent),
- ``build_clarification_form`` (one-shot ingredient form),
- ``apply_answers``            (flat ``{intent_path: value}`` merge).

These tests pin the behaviors documented in ``INTENT_V2.md`` §5–§7.
"""

from __future__ import annotations

import pytest

from nemo_curator.agentic.cards import DatasetCard, DatasetProfile
from nemo_curator.agentic.clarifier import (
    Section,
    apply_answers,
    apply_profile_prefills,
    build_clarification_form,
    infer_intent_from_prompt,
)
from nemo_curator.agentic.deterministic_planner import plan_from_intent
from nemo_curator.agentic.intent import (
    FilterMode,
    IntentCategories,
    Quality,
    Segmentation,
)
from nemo_curator.agentic.registry import build_registry


@pytest.fixture(scope="module")
def registry():
    return build_registry(cross_check_runtime=False, eager=False)


# ----------------------------------------------------------------------------
# Prompt → intent heuristics
# ----------------------------------------------------------------------------


class TestInferIntentFromPrompt:
    def test_explicit_sample_rate_sets_output_sr(self) -> None:
        intent = IntentCategories()
        intent, assumptions = infer_intent_from_prompt("Resample to 24 kHz mono", intent)
        assert intent.output.sample_rate == 24000
        assert intent.output.channels == "mono"
        assert intent.output.resample_input is True
        assert any("24000" in a for a in assumptions)

    def test_clean_word_enables_balanced_mos_and_sigmos(self) -> None:
        intent, _ = infer_intent_from_prompt("Give me a clean dataset", IntentCategories())
        assert intent.quality.mos == FilterMode.FILTER
        assert intent.quality.mos_threshold == 3.4
        assert intent.quality.sigmos == FilterMode.FILTER
        assert "ovrl" in intent.quality.sigmos_axes
        assert "noise" in intent.quality.sigmos_axes

    def test_studio_word_enables_strict_mos_and_sigmos(self) -> None:
        intent, _ = infer_intent_from_prompt("Studio quality audio", IntentCategories())
        assert intent.quality.mos == FilterMode.FILTER
        assert intent.quality.mos_threshold == 4.0
        assert intent.quality.sigmos == FilterMode.FILTER

    def test_phone_word_enables_band_filter_narrowband(self) -> None:
        intent, _ = infer_intent_from_prompt("These are phone calls", IntentCategories())
        assert intent.quality.band == FilterMode.FILTER
        assert intent.quality.band_value == "narrow_band"
        assert intent.output.sample_rate == 16000

    def test_diarize_word_enables_speaker_annotate(self) -> None:
        intent, _ = infer_intent_from_prompt(
            "I have meetings; give me speaker labels.", IntentCategories(),
        )
        assert intent.speakers.mode == FilterMode.ANNOTATE

    def test_tts_hint_only_sets_sample_rate_and_leaves_speakers_off(self) -> None:
        """The 'TTS' word is a hint for sample-rate only.

        Speakers and quality are *not* auto-enabled — those decisions
        belong to the user (or the smart clarifier's LLM follow-up
        question). The old behavior silently flipped speakers→SPLIT and
        mos→FILTER, which produced wrong pipelines for single-speaker
        TTS datasets — see regression run 332c0ecf06a981f6.

        ``"clean"`` is still picked up by the high-quality keyword path
        (separate branch), so MOS=FILTER here comes from that word —
        not from "TTS".
        """
        intent, assumptions = infer_intent_from_prompt(
            "Clean TTS dataset at 24 kHz", IntentCategories(),
        )
        # TTS still sets the sample-rate default.
        assert intent.output.sample_rate == 24000
        # TTS no longer touches speakers — left at the default.
        assert intent.speakers.mode == FilterMode.OFF
        # MOS=FILTER here is driven by the "clean" keyword, not the TTS
        # quick-pack; the assumption mentions both branches.
        assert intent.quality.mos == FilterMode.FILTER
        assert any("TTS" in a for a in assumptions)
        # An explicit hint that we deferred speakers/quality to the user.
        assert any(
            "ask the user" in a.lower() or "leave" in a.lower() or "left at off" in a.lower()
            for a in assumptions
        ), assumptions

    def test_tts_alone_does_not_enable_quality_filter(self) -> None:
        """Without any quality word in the prompt, "TTS" alone leaves
        MOS / SIGMOS at OFF. Pre-fix, the TTS quick-pack would have
        set ``mos=FILTER`` with a 3.4 threshold and a SpeakerSeparation
        stage — silently, with no question to the user."""
        intent, _ = infer_intent_from_prompt(
            "build a tts dataset", IntentCategories(),
        )
        assert intent.quality.mos == FilterMode.OFF
        assert intent.quality.sigmos == FilterMode.OFF
        assert intent.speakers.mode == FilterMode.OFF
        assert intent.output.sample_rate == 24000

    def test_duration_range_switches_to_speech_segments(self) -> None:
        intent, _ = infer_intent_from_prompt(
            "Make 2-30 second clips", IntentCategories(),
        )
        assert intent.segmentation.output_unit == "speech_segments"
        assert intent.segmentation.duration_min_sec == 2.0
        assert intent.segmentation.duration_max_sec == 30.0

    def test_speech_presence_phrase_enables_filter_when_unit_is_files(self) -> None:
        intent, _ = infer_intent_from_prompt(
            "no i want speaker labels and quality doesnt matter make sure atleast speech is there in audio",
            IntentCategories(),
        )
        assert intent.segmentation.output_unit == "original_files"
        assert intent.segmentation.speech_policy == FilterMode.FILTER

    def test_alm_phrase_picks_long_windows_and_word_timing(self) -> None:
        intent, _ = infer_intent_from_prompt(
            "I have long podcasts; prepare them for an audio-language model.",
            IntentCategories(),
        )
        assert intent.segmentation.output_unit == "long_windows"
        assert intent.segmentation.long_window_sec is not None
        assert intent.text.transcript_source == "generate"
        assert intent.text.word_timing is True

    def test_assumptions_are_appended_to_notes(self) -> None:
        intent = IntentCategories(notes=["llm_extracted=keep_me"])
        intent, _ = infer_intent_from_prompt("Make a TTS dataset", intent)
        assert "llm_extracted=keep_me" in intent.notes
        assert any(n.startswith("prompt_inference:") for n in intent.notes)


# ----------------------------------------------------------------------------
# Profile → intent prefills
# ----------------------------------------------------------------------------


class TestProfilePrefills:
    def test_single_sample_rate_in_profile_fills_output_sr(self) -> None:
        card = DatasetCard(
            name="x", uri="/x",
            profile=DatasetProfile(sample_rates_hz={"48000": 100}),
        )
        intent, assumptions = apply_profile_prefills(IntentCategories(), card)
        assert intent.output.sample_rate == 48000
        assert intent.output.resample_input is False
        assert any("48000" in a for a in assumptions)

    def test_mixed_sample_rates_in_profile_leaves_output_sr_unset(self) -> None:
        card = DatasetCard(
            name="x", uri="/x",
            profile=DatasetProfile(sample_rates_hz={"16000": 50, "48000": 50}),
        )
        intent, _ = apply_profile_prefills(IntentCategories(), card)
        assert intent.output.sample_rate is None

    def test_profile_does_not_override_explicit_prompt_value(self) -> None:
        card = DatasetCard(
            name="x", uri="/x",
            profile=DatasetProfile(sample_rates_hz={"48000": 100}),
        )
        intent_with_prompt, _ = infer_intent_from_prompt("Resample to 24 kHz", IntentCategories())
        intent, _ = apply_profile_prefills(intent_with_prompt, card)
        assert intent.output.sample_rate == 24000


# ----------------------------------------------------------------------------
# Form construction
# ----------------------------------------------------------------------------


class TestBuildClarificationForm:
    def test_form_has_all_top_sections(self) -> None:
        form = build_clarification_form("Make a dataset", IntentCategories())
        ids = [s.id for s in form.sections]
        assert ids == ["output", "segmentation", "quality", "annotations", "policy"]

    def test_form_contains_prompt_assumptions(self) -> None:
        form = build_clarification_form("Clean TTS dataset at 24 kHz", IntentCategories())
        assert any("24000" in a for a in form.assumptions)
        assert any("TTS" in a for a in form.assumptions)

    def test_resample_input_question_hidden_when_sr_unset(self) -> None:
        form = build_clarification_form("Make a dataset", IntentCategories())
        out = next(s for s in form.sections if s.id == "output")
        rq = next(q for q in out.questions if q.id == "output.resample_input")
        assert rq.visible is False

    def test_resample_input_question_visible_when_sr_concrete(self) -> None:
        form = build_clarification_form("Resample to 16 kHz", IntentCategories())
        out = next(s for s in form.sections if s.id == "output")
        rq = next(q for q in out.questions if q.id == "output.resample_input")
        assert rq.visible is True

    def test_long_window_question_only_when_unit_is_long_windows(self) -> None:
        form = build_clarification_form(
            "Long-audio chunks for an audio-language model",
            IntentCategories(),
        )
        seg = next(s for s in form.sections if s.id == "segmentation")
        lw = next(q for q in seg.questions if q.id == "segmentation.long_window_sec")
        assert lw.visible is True

    def test_speech_policy_question_hidden_when_unit_not_files(self) -> None:
        form = build_clarification_form("Make 2-30 second clips", IntentCategories())
        seg = next(s for s in form.sections if s.id == "segmentation")
        sp = next(q for q in seg.questions if q.id == "segmentation.speech_policy")
        assert sp.visible is False

    def test_speaker_count_question_only_when_mode_is_filter(self) -> None:
        intent = IntentCategories(speakers={"mode": FilterMode.FILTER, "target_count": 1})
        form = build_clarification_form("Make a dataset", intent)
        ann = next(s for s in form.sections if s.id == "annotations")
        sc = next(q for q in ann.questions if q.id == "speakers.count")
        assert sc.visible is True

    def test_form_intent_carries_prefilled_values(self) -> None:
        """Only safe, reversible prefills (sample rate) are carried through.

        Pre-fix, the form also carried speakers=SPLIT silently because
        the prompt mentioned "TTS". We now leave that to the smart
        clarifier's follow-up question instead.
        """

        form = build_clarification_form("Clean TTS dataset at 24 kHz", IntentCategories())
        assert form.intent.output.sample_rate == 24000
        # "clean" still triggers MOS=FILTER (high-quality keyword branch).
        assert form.intent.quality.mos == FilterMode.FILTER
        # TTS no longer flips speakers.
        assert form.intent.speakers.mode == FilterMode.OFF

    def test_form_includes_profile_summary_when_card_present(self) -> None:
        card = DatasetCard(
            name="x", uri="/x",
            profile=DatasetProfile(total_files=10, decodable_files=8),
        )
        form = build_clarification_form("Make a dataset", IntentCategories(), profile=card)
        assert form.profile_summary is not None
        assert form.profile_summary["total_files"] == 10


# ----------------------------------------------------------------------------
# Apply answers
# ----------------------------------------------------------------------------


class TestApplyAnswers:
    def test_simple_flat_answer_writes_namespaced_field(self) -> None:
        intent = IntentCategories()
        out = apply_answers(intent, {"output.sample_rate": 24000})
        assert out.output.sample_rate == 24000

    def test_apply_side_effects_run_before_explicit_paths(self) -> None:
        intent = IntentCategories()
        out = apply_answers(intent, {
            "__apply__": {"segmentation.duration_min_sec": 2.0, "segmentation.duration_max_sec": 60.0},
        })
        assert out.segmentation.duration_min_sec == 2.0
        assert out.segmentation.duration_max_sec == 60.0

    def test_filter_mode_string_round_trips(self) -> None:
        intent = IntentCategories()
        out = apply_answers(intent, {"quality.mos": "filter", "quality.mos_threshold": 3.7})
        assert out.quality.mos == FilterMode.FILTER
        assert out.quality.mos_threshold == 3.7

    def test_existing_notes_preserved_alongside_clarifier_notes(self) -> None:
        intent = IntentCategories(notes=["llm_extracted=keep"])
        out = apply_answers(intent, {"output.sample_rate": 48000})
        assert "llm_extracted=keep" in out.notes
        assert any(n.startswith("clarifier_answer:") for n in out.notes)


# ----------------------------------------------------------------------------
# End-to-end: form → planner round-trip
# ----------------------------------------------------------------------------


class TestEndToEnd:
    def test_clean_tts_prompt_compiles(self, registry) -> None:
        """An explicit prompt should still produce a working pipeline.

        We pass ``speakers=split`` and ``mos=filter`` explicitly via the
        intent (mimicking the user answering the new clarifier
        questions) and verify the pipeline shape, since the heuristic
        prefills no longer flip those for us.

        Once :class:`IntentCategories` coerces ``speech_segments`` +
        ``speakers=SPLIT`` to ``single_speaker_clips``, the compiled
        pipeline runs SpeakerSeparation **before** VAD — fixing the
        regression in run 06578e4b178f11e3 where the duration window
        applied at the wrong stage.
        """

        prompt = "I have voice recordings. Make a clean TTS dataset at 24 kHz, 2-60 second clips, split by speaker."
        form = build_clarification_form(prompt, IntentCategories())
        intent = form.intent.model_copy(update={
            "speakers": form.intent.speakers.model_copy(update={"mode": FilterMode.SPLIT}),
        })
        result = plan_from_intent(
            intent,
            source_uri="/data/audio",
            source_kind="manifest",
            target_dir="/out",
            registry=registry,
        )
        names = [s.stage for s in result.ir.stages]
        assert names[0] == "ManifestReader"
        assert names[-1] == "ManifestWriterStage"
        assert "VADSegmentationStage" in names
        assert "SpeakerSeparationStage" in names
        assert "UTMOSFilterStage" in names
        assert not result.dry_run.has_errors, result.dry_run.all_issues
        # SpeakerSeparation must run BEFORE VAD so the per-clip duration
        # window applies to per-speaker stems (not to file-level VAD
        # segments that get concatenated away).
        idx_speaker = names.index("SpeakerSeparationStage")
        idx_vad = names.index("VADSegmentationStage")
        assert idx_speaker < idx_vad, (
            f"Expected SpeakerSeparationStage before VADSegmentationStage; "
            f"got order {names}"
        )
        # Quality scoring + PBV gating must run AFTER VAD so the score
        # values refer to the final per-clip rows the user receives.
        idx_utmos = names.index("UTMOSFilterStage")
        assert idx_vad < idx_utmos
        # No SegmentConcat should be auto-inserted — it would discard
        # the duration window the user picked.
        assert "SegmentConcatenationStage" not in names

    def test_speech_presence_filter_adds_nested_vad(self, registry) -> None:
        prompt = "Drop empty audio; keep files with at least some speech."
        form = build_clarification_form(prompt, IntentCategories())
        intent = form.intent
        assert intent.segmentation.output_unit == "original_files"
        assert intent.segmentation.speech_policy == FilterMode.FILTER

        result = plan_from_intent(
            intent,
            source_uri="/data/audio",
            source_kind="manifest",
            target_dir="/out",
            registry=registry,
        )
        names = [s.stage for s in result.ir.stages]
        assert "VADSegmentationStage" in names
        vad = next(s for s in result.ir.stages if s.stage == "VADSegmentationStage")
        assert vad.params["nested"] is True

    def test_annotate_only_does_not_drop_rows(self, registry) -> None:
        """ANNOTATE mode writes the metric without filtering — threshold collapses to 0."""

        intent = IntentCategories(
            quality=Quality(mos=FilterMode.ANNOTATE),
            raw_prompt="annotate UTMOS",
        )
        result = plan_from_intent(
            intent,
            source_uri="/data/audio",
            source_kind="manifest",
            target_dir="/out",
            registry=registry,
        )
        utmos = next(s for s in result.ir.stages if s.stage == "UTMOSFilterStage")
        assert utmos.params["mos_threshold"] == 0.0

    def test_filter_mode_uses_user_threshold(self, registry) -> None:
        """``mos_threshold`` now lives on a downstream PreserveByValueStage
        instead of the UTMOSFilterStage itself — see the module-level
        "score → gate" contract in ``stage_selector._quality_stages``."""

        intent = IntentCategories(
            quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.7),
            raw_prompt="MOS ≥ 3.7",
        )
        result = plan_from_intent(
            intent,
            source_uri="/data/audio",
            source_kind="manifest",
            target_dir="/out",
            registry=registry,
        )
        utmos = next(s for s in result.ir.stages if s.stage == "UTMOSFilterStage")
        assert utmos.params["mos_threshold"] == 0.0
        pbv = next(
            s for s in result.ir.stages
            if s.stage == "PreserveByValueStage"
            and s.params.get("input_value_key") == "utmos_mos"
        )
        assert pbv.params["operator"] == "ge"
        assert pbv.params["target_value"] == 3.7

    def test_freeform_duration_range_via_apply(self, registry) -> None:
        """The form's freeform 'min,max' freeform option ships an
        ``__apply__`` dict; ``apply_answers`` must write both legs."""

        form = build_clarification_form(
            "Make speech clips between 3 and 45 seconds", IntentCategories(),
        )
        # The prompt heuristic already filled in 3..45 — verify the apply
        # path works on top.
        intent = apply_answers(form.intent, {
            "__apply__": {
                "segmentation.duration_min_sec": 5.0,
                "segmentation.duration_max_sec": 25.0,
            },
        })
        result = plan_from_intent(
            intent,
            source_uri="/data/audio",
            source_kind="manifest",
            target_dir="/out",
            registry=registry,
        )
        vad = next(s for s in result.ir.stages if s.stage == "VADSegmentationStage")
        assert vad.params["min_duration_sec"] == 5.0
        assert vad.params["max_duration_sec"] == 25.0

    def test_profile_suppression_keeps_single_sr_pass_through(self, registry) -> None:
        """When the dataset profile shows a single SR, the form's prefill is
        the profile source and ``resample_input`` is set to ``False`` so the
        compiler doesn't insert a redundant ResampleAudioStage."""

        card = DatasetCard(
            name="x", uri="/x",
            profile=DatasetProfile(sample_rates_hz={"48000": 100}),
        )
        form = build_clarification_form("Make a dataset", IntentCategories(), profile=card)
        assert form.intent.output.sample_rate == 48000
        assert form.intent.output.resample_input is False
        result = plan_from_intent(
            form.intent,
            source_uri="/data/audio",
            source_kind="manifest",
            target_dir="/out",
            registry=registry,
        )
        names = [s.stage for s in result.ir.stages]
        # With strict_sample_rate, the resample stage is *not* added by the
        # selector; only mono conversion runs.
        assert names.count("ResampleAudioStage") == 0 or names.count("ResampleAudioStage") == 1
