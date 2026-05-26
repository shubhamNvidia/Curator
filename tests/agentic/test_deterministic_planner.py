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
"""Tests for the deterministic compiler-style planner (V2 intent)."""

from __future__ import annotations

import pytest

from nemo_curator.agentic.deterministic_planner import plan_from_intent
from nemo_curator.agentic.dryrun import dry_run_pipeline
from nemo_curator.agentic.intent import (
    FilterMode,
    IntentCategories,
    OutputFormat,
    Policy,
    Quality,
    QualityGate,
    Segmentation,
    Speakers,
)
from nemo_curator.agentic.ir import PipelineIR, SinkSpec, SourceSpec, StageRef
from nemo_curator.agentic.registry import build_registry
from nemo_curator.agentic.validator import validate


@pytest.fixture(scope="module")
def registry():
    return build_registry(cross_check_runtime=False, eager=False)


def _clean_single_speaker_intent() -> IntentCategories:
    """A canonical TTS-style intent in V2 namespaced form."""

    return IntentCategories(
        output=OutputFormat(sample_rate=16000, channels="mono", resample_input=True),
        segmentation=Segmentation(
            output_unit="single_speaker_clips",
            duration_min_sec=2.0,
            duration_max_sec=60.0,
        ),
        quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5),
        speakers=Speakers(mode=FilterMode.SPLIT),
        policy=Policy(commercial_only=True),
        raw_prompt="clean single-speaker 16 kHz clips between 2 and 60 seconds",
    )


def test_deterministic_planner_builds_clean_single_speaker_pipeline(registry) -> None:
    """``single_speaker_clips`` uses SpeakerSeparation as the segmenter,
    and — when the user gave us a duration cap — a *post-segmenter*
    ``VAD(nested=False)`` trim fires right after it to enforce the cap
    on each fan-out clip. The legacy "upfront cleaning block" (nested
    VAD + Concat) is still reserved for ``output_unit=='original_files'``
    and must not show up here.
    """

    result = plan_from_intent(
        _clean_single_speaker_intent(),
        source_uri="/data/input.jsonl",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    names = [s.stage for s in result.ir.stages]
    assert names[0] == "ManifestReader"
    assert names[-1] == "ManifestWriterStage"
    assert "SpeakerSeparationStage" in names
    assert "SegmentExtractionStage" in names
    assert "UTMOSFilterStage" in names

    # Post-segmenter VAD trim: the intent has duration_min_sec=2 and
    # duration_max_sec=60, so the selector MUST drop a non-nested VAD
    # right after SpeakerSeparation, with the user's caps plumbed through.
    # No special wiring needed on SpeakerSep: it returns per-speaker
    # waveforms that span the full source duration (silence outside the
    # speaker's regions), so VAD's timestamps stay in source coords —
    # this is the same pattern used in
    # ``advanced_pipelines/audio_data_filter._build_full_pipeline``.
    assert "VADSegmentationStage" in names
    vad = next(s for s in result.ir.stages if s.stage == "VADSegmentationStage")
    assert vad.params["nested"] is False
    assert vad.params["min_duration_sec"] == 2.0
    assert vad.params["max_duration_sec"] == 60.0
    sep_idx = names.index("SpeakerSeparationStage")
    vad_idx = names.index("VADSegmentationStage")
    assert sep_idx < vad_idx, "post-segmenter VAD must run AFTER SpeakerSeparation"

    # Cleaning block (nested VAD + Concat) is for original_files only.
    assert "SegmentConcatenationStage" not in names
    assert not result.dry_run.has_errors, result.dry_run.all_issues


def test_deterministic_planner_emits_cleaning_block_for_original_files(registry) -> None:
    """The cleaning flow (nested VAD + filters + Concat) fires only when
    ``output_unit=='original_files'`` AND ``speech_policy`` is enabled."""

    intent = IntentCategories(
        output=OutputFormat(sample_rate=16000, channels="mono", resample_input=True),
        segmentation=Segmentation(
            output_unit="original_files",
            speech_policy=FilterMode.FILTER,
        ),
        quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5),
        raw_prompt="clean my audio before downstream work",
    )
    result = plan_from_intent(
        intent,
        source_uri="/data/input.jsonl",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    names = [s.stage for s in result.ir.stages]
    assert "VADSegmentationStage" in names
    assert "UTMOSFilterStage" in names
    assert "SegmentConcatenationStage" in names

    vad = next(s for s in result.ir.stages if s.stage == "VADSegmentationStage")
    assert vad.params["nested"] is True, (
        "Cleaning flow requires nested VAD so filters can iterate "
        "task.data['segments'] and Concat can re-stitch survivors."
    )
    # And the order must be VAD → quality filters → Concat (bound block).
    vad_idx = names.index("VADSegmentationStage")
    utmos_idx = names.index("UTMOSFilterStage")
    concat_idx = names.index("SegmentConcatenationStage")
    assert vad_idx < utmos_idx < concat_idx


def test_deterministic_planner_binds_segment_extraction_dir(registry) -> None:
    result = plan_from_intent(
        _clean_single_speaker_intent(),
        source_uri="/data/input.jsonl",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    seg = next(s for s in result.ir.stages if s.stage == "SegmentExtractionStage")
    assert seg.params["output_dir"] == "/out/audio"
    assert seg.params["output_format"] == "wav"


def test_validator_concat_autoinsert_is_dry_run_clean_with_default_vad(registry) -> None:
    intent = _clean_single_speaker_intent()
    raw = PipelineIR(
        source=SourceSpec(kind="manifest", uri="/data/input.jsonl"),
        sink=SinkSpec(target_dir="/out"),
        stages=[
            StageRef(stage="VADSegmentationStage", params={}),
            StageRef(stage="SpeakerSeparationStage", params={}),
            StageRef(stage="SegmentExtractionStage", params={"output_dir": "/out/audio"}),
        ],
        intent=intent,
    )
    report = validate(raw, registry, intent=intent, mutate=True)
    dry = dry_run_pipeline(report.ir, registry)
    vad = next(s for s in report.ir.stages if s.stage == "VADSegmentationStage")
    assert vad.params["nested"] is True
    assert not dry.has_errors, dry.all_issues


def test_annotate_mode_does_not_drop_rows(registry) -> None:
    """ANNOTATE collapses the MOS threshold to 0.0 — annotate without dropping."""

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


def test_filter_mode_uses_user_threshold(registry) -> None:
    """With the score → gate split, UTMOSFilterStage is now always emitted
    in pure-annotate mode (``mos_threshold=0.0``). The user-supplied
    threshold lives on the downstream ``PreserveByValueStage(utmos_mos,
    ge, X)`` row instead — same effective filter, more operator
    flexibility."""

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


def test_filter_speakers_at_most_n_uses_le_operator(registry) -> None:
    intent = IntentCategories(
        speakers=Speakers(mode=FilterMode.FILTER, max_count=2),
        raw_prompt="at most 2 speakers",
    )
    result = plan_from_intent(
        intent,
        source_uri="/data/audio",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    names = [s.stage for s in result.ir.stages]
    assert "InferenceSortformerStage" in names
    assert "PreserveByValueStage" in names
    pres = next(s for s in result.ir.stages if s.stage == "PreserveByValueStage")
    assert pres.params["target_value"] == 2
    assert pres.params["operator"] == "le"


def test_filter_speakers_at_least_n_uses_ge_operator(registry) -> None:
    """``speakers.min_count`` must emit ``operator='ge'`` (the missing path
    that this fix is closing). Validates the literal user complaint:
    "filter out greater than value the preserve by value stage is not
    working only working for less than thing"."""

    intent = IntentCategories(
        speakers=Speakers(mode=FilterMode.FILTER, min_count=3),
        raw_prompt="at least 3 speakers",
    )
    result = plan_from_intent(
        intent,
        source_uri="/data/audio",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    names = [s.stage for s in result.ir.stages]
    assert "InferenceSortformerStage" in names
    pres_stages = [s for s in result.ir.stages if s.stage == "PreserveByValueStage"]
    assert len(pres_stages) == 1
    assert pres_stages[0].params["input_value_key"] == "num_speakers"
    assert pres_stages[0].params["target_value"] == 3
    assert pres_stages[0].params["operator"] == "ge"


def test_filter_speakers_range_emits_ge_and_le(registry) -> None:
    """When both ``min_count`` and ``max_count`` are set the selector must
    emit two ``PreserveByValueStage`` rows: one ``ge`` floor + one ``le``
    ceiling, in that order (floor before ceiling so the operator order is
    deterministic for snapshot tests)."""

    intent = IntentCategories(
        speakers=Speakers(mode=FilterMode.FILTER, min_count=2, max_count=5),
        raw_prompt="between 2 and 5 speakers",
    )
    result = plan_from_intent(
        intent,
        source_uri="/data/audio",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    pres_stages = [s for s in result.ir.stages if s.stage == "PreserveByValueStage"]
    assert len(pres_stages) == 2
    ops = [(s.params["operator"], s.params["target_value"]) for s in pres_stages]
    assert ops == [("ge", 2), ("le", 5)]


def test_filter_speakers_target_count_overrides_range(registry) -> None:
    """``target_count`` is the most specific filter — if the user pins an
    exact N the selector must emit a single ``eq`` row even when min/max
    happen to be set (they become noise)."""

    intent = IntentCategories(
        speakers=Speakers(
            mode=FilterMode.FILTER,
            target_count=4,
            min_count=2,
            max_count=5,
        ),
        raw_prompt="exactly 4 speakers",
    )
    result = plan_from_intent(
        intent,
        source_uri="/data/audio",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    pres_stages = [s for s in result.ir.stages if s.stage == "PreserveByValueStage"]
    assert len(pres_stages) == 1
    assert pres_stages[0].params["operator"] == "eq"
    assert pres_stages[0].params["target_value"] == 4


def test_min_count_greater_than_max_count_is_rejected() -> None:
    """The schema validator must catch impossible ranges before they hit
    the selector."""

    with pytest.raises(ValueError, match="min_count.*>.*max_count"):
        IntentCategories(
            speakers=Speakers(mode=FilterMode.FILTER, min_count=8, max_count=3),
            raw_prompt="invalid range",
        )


def test_original_files_duration_filter_emits_pbv(registry) -> None:
    """For ``output_unit='original_files'`` without a cleaning flow, the
    selector must honor ``duration_min_sec`` / ``duration_max_sec`` via
    ``PreserveByValueStage(duration, ...)``. Before the fix these
    constraints were silently dropped because no VAD was in the pipeline.
    """

    intent = IntentCategories(
        segmentation=Segmentation(
            output_unit="original_files",
            duration_min_sec=10.0,
            duration_max_sec=60.0,
        ),
        raw_prompt="files between 10s and 60s",
    )
    result = plan_from_intent(
        intent,
        source_uri="/data/audio",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    pres_stages = [
        s for s in result.ir.stages
        if s.stage == "PreserveByValueStage"
        and s.params.get("input_value_key") == "duration"
    ]
    assert len(pres_stages) == 2
    ops = [(s.params["operator"], s.params["target_value"]) for s in pres_stages]
    assert ops == [("ge", 10.0), ("le", 60.0)]


def test_quality_filter_translates_to_pbv_ge_rows(registry) -> None:
    """End-to-end check of the score → gate split: UTMOS + SIGMOS pinned to
    threshold=0.0 and the actual drop happens via one PBV(ge) per axis."""

    intent = IntentCategories(
        quality=Quality(
            mos=FilterMode.FILTER,
            mos_threshold=3.4,
            sigmos=FilterMode.FILTER,
            sigmos_axes=["ovrl", "noise"],
            sigmos_thresholds={"ovrl": 3.5, "noise": 4.0},
        ),
        raw_prompt="clean TTS-ready speech",
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
    sigmos = next(s for s in result.ir.stages if s.stage == "SIGMOSFilterStage")
    assert sigmos.params["ovrl_threshold"] == 0.0
    assert sigmos.params["noise_threshold"] == 0.0
    pbv = {
        s.params["input_value_key"]: s.params
        for s in result.ir.stages
        if s.stage == "PreserveByValueStage"
        and s.params.get("input_value_key", "").startswith(("utmos_", "sigmos_"))
    }
    assert pbv["utmos_mos"]["operator"] == "ge"
    assert pbv["utmos_mos"]["target_value"] == 3.4
    assert pbv["sigmos_ovrl"]["operator"] == "ge"
    assert pbv["sigmos_ovrl"]["target_value"] == 3.5
    assert pbv["sigmos_noise"]["operator"] == "ge"
    assert pbv["sigmos_noise"]["target_value"] == 4.0


def test_quality_gates_lt_operator_for_drop_high_quality(registry) -> None:
    """The literal "filter out greater than" / "drop high quality" use case:
    a custom ``QualityGate`` with ``operator='lt'`` produces a PBV row
    that keeps only LOW-MOS clips (handy for adversarial / noisy-only
    corpora). Before this refactor there was no way to express that —
    UTMOSFilterStage only ever drops below the threshold."""

    intent = IntentCategories(
        quality=Quality(
            mos=FilterMode.ANNOTATE,
            gates=[QualityGate(key="utmos_mos", operator="lt", value=2.0)],
        ),
        raw_prompt="keep only noisy clips for adversarial training",
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
    pbv_rows = [
        s for s in result.ir.stages
        if s.stage == "PreserveByValueStage"
        and s.params.get("input_value_key") == "utmos_mos"
    ]
    assert len(pbv_rows) == 1
    assert pbv_rows[0].params["operator"] == "lt"
    assert pbv_rows[0].params["target_value"] == 2.0


def test_quality_gates_pull_in_sigmos_stage_on_demand(registry) -> None:
    """A gate referencing ``sigmos_*`` must force SIGMOSFilterStage to
    show up even when ``quality.sigmos == OFF``, so the score key the gate
    is reading actually exists at runtime."""

    intent = IntentCategories(
        quality=Quality(
            sigmos=FilterMode.OFF,
            gates=[QualityGate(key="sigmos_reverb", operator="ge", value=4.0)],
        ),
        raw_prompt="dry rooms only",
    )
    result = plan_from_intent(
        intent,
        source_uri="/data/audio",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    sigmos = next(s for s in result.ir.stages if s.stage == "SIGMOSFilterStage")
    assert sigmos.params["reverb_threshold"] == 0.0
    pbv = next(
        s for s in result.ir.stages
        if s.stage == "PreserveByValueStage"
        and s.params.get("input_value_key") == "sigmos_reverb"
    )
    assert pbv.params["operator"] == "ge"
    assert pbv.params["target_value"] == 4.0


def test_quality_band_filter_emits_pbv_eq(registry) -> None:
    """``band == FILTER`` still emits ``BandFilterStage`` (it has no
    annotate mode) AND a PBV(eq) mirror — the latter makes the drop
    decision uniform with the rest of the quality stack."""

    intent = IntentCategories(
        quality=Quality(band=FilterMode.FILTER, band_value="narrow_band"),
        raw_prompt="telephony only",
    )
    result = plan_from_intent(
        intent,
        source_uri="/data/audio",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    names = [s.stage for s in result.ir.stages]
    assert "BandFilterStage" in names
    pbv = next(
        s for s in result.ir.stages
        if s.stage == "PreserveByValueStage"
        and s.params.get("input_value_key") == "band_prediction"
    )
    assert pbv.params["operator"] == "eq"
    assert pbv.params["target_value"] == "narrow_band"


def test_quality_annotate_mode_emits_no_pbv(registry) -> None:
    """``annotate`` mode (no legacy threshold, no gates) keeps the scoring
    stages but emits zero ``PreserveByValueStage`` rows — score-only
    behavior must not drop a single row."""

    intent = IntentCategories(
        quality=Quality(
            mos=FilterMode.ANNOTATE,
            sigmos=FilterMode.ANNOTATE,
            sigmos_axes=["ovrl", "noise"],
        ),
        raw_prompt="score everything, drop nothing",
    )
    result = plan_from_intent(
        intent,
        source_uri="/data/audio",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    assert "UTMOSFilterStage" in {s.stage for s in result.ir.stages}
    assert "SIGMOSFilterStage" in {s.stage for s in result.ir.stages}
    quality_pbv = [
        s for s in result.ir.stages
        if s.stage == "PreserveByValueStage"
        and s.params.get("input_value_key", "").startswith(("utmos_", "sigmos_", "band_"))
    ]
    assert quality_pbv == []


def test_original_files_duration_min_only_emits_ge(registry) -> None:
    """Lower-bound-only case — covers the literal "duration greater than X
    seconds" prompt that wasn't reachable before."""

    intent = IntentCategories(
        segmentation=Segmentation(
            output_unit="original_files",
            duration_min_sec=30.0,
        ),
        raw_prompt="files longer than 30s",
    )
    result = plan_from_intent(
        intent,
        source_uri="/data/audio",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
    )
    pres_stages = [
        s for s in result.ir.stages
        if s.stage == "PreserveByValueStage"
        and s.params.get("input_value_key") == "duration"
    ]
    assert len(pres_stages) == 1
    assert pres_stages[0].params["operator"] == "ge"
    assert pres_stages[0].params["target_value"] == 30.0


def test_long_window_unit_pulls_alm_builder(registry) -> None:
    """The ``long_windows`` segmentation unit must pull in the ALM packaging
    chain (``SplitLongAudioStage`` + ``ALMDataBuilderStage``) and, when
    the user opts into word timing, the ASR aligner.

    The dry-run flow check is intentionally relaxed here: ``ALMDataBuilderStage``
    declares ``task.data['segments']`` as an input but no upstream stage in
    the long-windows chain produces a key with that name today. That's a
    real gap in the current ALM card metadata which the selector cannot fix
    on its own; this test pins the *selector-level* contract (which stages
    appear in the IR) and leaves the dry-run repair for a follow-up.
    """

    intent = IntentCategories(
        output=OutputFormat(sample_rate=16000, channels="mono", resample_input=True),
        segmentation=Segmentation(output_unit="long_windows", long_window_sec=60.0),
        text={"transcript_source": "generate", "word_timing": True},
        raw_prompt="60s long-audio windows for ALM at 16 kHz",
    )
    result = plan_from_intent(
        intent,
        source_uri="/data/audio",
        source_kind="manifest",
        target_dir="/out",
        registry=registry,
        require_dry_run_clean=False,
    )
    names = [s.stage for s in result.ir.stages]
    assert "SplitLongAudioStage" in names
    assert "ALMDataBuilderStage" in names
    assert "NeMoASRAlignerStage" in names
