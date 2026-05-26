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
"""Tests for the symbolic dry-run engine (:mod:`nemo_curator.agentic.dryrun`).

These exercise the smart staging analysis: feeding an IR through
``dry_run_pipeline`` and asserting that the right concrete violations
surface for known-bad orderings (and no violations for good orderings).
"""

from __future__ import annotations

import pytest

from nemo_curator.agentic.dryrun import dry_run_pipeline, initial_state
from nemo_curator.agentic.intent import (
    FilterMode,
    IntentCategories,
    Quality,
    Segmentation,
    Speakers,
)
from nemo_curator.agentic.ir import PipelineIR, SinkSpec, SourceSpec, StageRef
from nemo_curator.agentic.registry import build_registry
from nemo_curator.agentic.validator import validate


@pytest.fixture(scope="module")
def registry():
    return build_registry(cross_check_runtime=False, eager=False)


def _default_intent() -> IntentCategories:
    return IntentCategories(quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5))


def _ir(stages: list[str], *, intent: IntentCategories | None = None) -> PipelineIR:
    return PipelineIR(
        source=SourceSpec(kind="manifest", uri="/data"),
        sink=SinkSpec(target_dir="/out"),
        stages=[StageRef(stage=name, params={}) for name in stages],
        intent=intent or _default_intent(),
    )


# ----------------------------------------------------------------------------
# Initial state
# ----------------------------------------------------------------------------


def test_initial_state_is_whole_file_on_disk() -> None:
    s = initial_state()
    assert s.shape == "whole_file"
    assert s.audio_on_disk is True
    assert s.audio_in_memory is False
    assert s.sample_rate is None
    assert s.frozen is False
    assert "audio_filepath" in s.data_keys
    # Top-level mirror of the REAL AudioTask schema
    # (see nemo_curator/tasks/audio_task.py).
    assert "task_id" in s.top_level_keys
    assert "dataset_name" in s.top_level_keys
    assert "data" in s.top_level_keys
    # These fictional fields must NEVER appear in the initial state.
    assert "_audio_id" not in s.top_level_keys
    assert "source_id" not in s.top_level_keys


# ----------------------------------------------------------------------------
# Happy-path: a canonical, validator-clean pipeline produces no issues
# ----------------------------------------------------------------------------


def test_clean_canonical_pipeline_passes(registry) -> None:
    """The full TTS-style pipeline (post-validator) should dry-run cleanly."""

    intent = IntentCategories(
        segmentation=Segmentation(
            output_unit="single_speaker_clips",
            duration_min_sec=2.0, duration_max_sec=60.0,
        ),
        quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5),
        speakers=Speakers(mode=FilterMode.SPLIT),
    )
    raw = _ir(
        ["VADSegmentationStage", "UTMOSFilterStage", "SegmentExtractionStage"],
        intent=intent,
    )
    canonical = validate(raw, registry, intent=intent, mutate=True).ir
    report = dry_run_pipeline(canonical, registry)
    assert not report.has_errors, report.all_issues


# ----------------------------------------------------------------------------
# Specific violations
# ----------------------------------------------------------------------------


def test_unknown_stage_is_flagged(registry) -> None:
    ir = _ir(["NoSuchStage"])
    report = dry_run_pipeline(ir, registry)
    assert report.has_errors
    msg = report.all_issues[0]["issue"]
    assert "unknown stage" in msg.lower()


def test_fan_out_without_concat_is_flagged(registry) -> None:
    """VAD fans out, then a whole-file-requiring stage runs without concat."""

    # Find a whole-file-requiring stage that exists in the registry.
    whole_file_stages = [
        n for n, e in registry.by_name.items()
        if e.card.input_shape.value == "whole_file"
        and e.card.phase.value in {"analyze", "concat"}
    ]
    if not whole_file_stages:
        pytest.skip("No whole_file-requiring stage in registry to test against")
    # Build an INTENTIONALLY broken IR: VAD then a whole-file stage with
    # no SegmentConcatenationStage between them. Skip the validator so the
    # broken order survives to the dry-run.
    ir = _ir(["VADSegmentationStage", whole_file_stages[0]])
    report = dry_run_pipeline(ir, registry)
    # The whole_file stage should complain about the fan_out_segments
    # upstream.
    flagged = [i for i in report.all_issues if "fan_out" in i["issue"] or "input_shape" in i["issue"]]
    assert flagged, f"expected a fan_out / input_shape complaint; got {report.all_issues}"


def test_terminal_then_analytical_is_flagged(registry) -> None:
    """A non-sink stage after a ``terminal: true`` stage must be flagged.

    The dry-run can only catch this when at least one stage in the
    registry actually has ``terminal: true`` declared on its card. If
    the registry doesn't yet have any terminal stages, we skip — the
    engine logic is exercised separately below via the constructed
    fake card.
    """

    terminal_stages = [
        n for n, e in registry.by_name.items()
        if e.card.terminal and e.card.phase.value not in {"sink", "materialize"}
    ]
    analytical_candidate = next(
        (n for n, e in registry.by_name.items()
         if e.card.category.value == "filter"
         and e.card.phase.value not in {"sink", "materialize"}),
        None,
    )
    if not terminal_stages or analytical_candidate is None:
        pytest.skip(
            "No terminal-flagged stage in registry yet — engine logic for the "
            "frozen check is also covered in test_terminal_freezes_state_synthetic"
        )
    ir = _ir([terminal_stages[0], analytical_candidate])
    report = dry_run_pipeline(ir, registry)
    flagged = [i for i in report.all_issues if "terminal" in i["issue"].lower() or "frozen" in i["issue"].lower()]
    assert flagged, f"expected a terminal-then-analytical complaint; got {report.all_issues}"


def test_terminal_freezes_state_synthetic() -> None:
    """Direct unit test of the frozen-state logic, independent of card YAML.

    Builds a registry with a single synthetic ``terminal=True`` card and
    a downstream filter card, then runs the dry-run and asserts the
    frozen complaint surfaces. This proves the engine logic regardless
    of whether the production cards currently declare ``terminal``.
    """

    from pathlib import Path

    from nemo_curator.agentic.cards import (
        CapabilityTag,
        IOSpec,
        OutputShape,
        Phase,
        StageCard,
        StageCategory,
    )
    from nemo_curator.agentic.registry import CapabilityRegistry, RegistryEntry

    term = StageCard(
        name="FakeTerminalStage",
        target="x.FakeTerminalStage",
        description="d",
        summary="s",
        category=StageCategory.IO,
        phase=Phase.POLISH,
        output_shape=OutputShape.REBUILD_TASK,
        terminal=True,
    )
    flt = StageCard(
        name="FakeAnalyzeStage",
        target="x.FakeAnalyzeStage",
        description="d",
        summary="s",
        category=StageCategory.FILTER,
        phase=Phase.ANALYZE,
        capabilities=[CapabilityTag.QUALITY_FILTER_MOS],
        inputs=IOSpec(top_level=[], data=[]),
    )
    fake_path = Path("/tmp/_synthetic")
    reg = CapabilityRegistry(
        by_name={
            "FakeTerminalStage": RegistryEntry(card=term, source_path=fake_path, origin="test"),
            "FakeAnalyzeStage": RegistryEntry(card=flt, source_path=fake_path, origin="test"),
        },
    )
    ir = _ir(["FakeTerminalStage", "FakeAnalyzeStage"])
    report = dry_run_pipeline(ir, reg)
    flagged = [i for i in report.all_issues if "terminal" in i["issue"].lower()]
    assert flagged, f"expected a frozen-state complaint; got {report.all_issues}"


def test_missing_data_key_is_flagged(registry) -> None:
    """A stage reading task.data[X] when X was never produced must be flagged."""

    # SegmentExtractionStage reads task.data["segments"] (or similar) that
    # VAD produces. Without VAD, it should complain.
    if "SegmentExtractionStage" not in registry.by_name:
        pytest.skip("SegmentExtractionStage not in registry")
    card = registry.by_name["SegmentExtractionStage"].card
    if not card.inputs.data:
        pytest.skip("SegmentExtractionStage has no declared data inputs")
    ir = _ir(["SegmentExtractionStage"])
    report = dry_run_pipeline(ir, registry)
    flagged = [
        i for i in report.all_issues
        if any(k in i["issue"] for k in card.inputs.data)
    ]
    assert flagged, (
        f"expected missing-data-key complaint for one of {card.inputs.data}; "
        f"got {report.all_issues}"
    )


# ----------------------------------------------------------------------------
# State propagation: each stage's effects are visible to the next
# ----------------------------------------------------------------------------


def test_data_keys_accumulate_across_stages(registry) -> None:
    """task.data keys produced by upstream stages must appear in the next stage's input state."""

    if "VADSegmentationStage" not in registry.by_name:
        pytest.skip("VADSegmentationStage not in registry")
    vad_card = registry.by_name["VADSegmentationStage"].card
    produced = set(vad_card.outputs.data) | set(vad_card.produces_keys_after_run)
    if not produced:
        pytest.skip("VAD card does not declare produced keys")

    intent = IntentCategories(
        quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5),
        speakers=Speakers(mode=FilterMode.SPLIT),
    )
    raw = _ir(["VADSegmentationStage", "UTMOSFilterStage"], intent=intent)
    canonical = validate(raw, registry, intent=intent, mutate=True).ir
    report = dry_run_pipeline(canonical, registry)
    # Find VAD's trace entry; the next entry's input_state should include
    # VAD's produced keys.
    for i, t in enumerate(report.stages[:-1]):
        if t.stage_name == "VADSegmentationStage":
            next_input = report.stages[i + 1].input_state
            assert produced & next_input.data_keys, (
                f"VAD produced {produced} but next stage's input_state has "
                f"data_keys={sorted(next_input.data_keys)}"
            )
            return
    pytest.fail("VADSegmentationStage not found in trace")


def test_terminal_freezes_state(registry) -> None:
    """If a registry card actually declares terminal=True, the trace must freeze."""

    terminal_stages = [
        n for n, e in registry.by_name.items() if e.card.terminal
    ]
    if not terminal_stages:
        pytest.skip(
            "No production card declares terminal=True yet — "
            "engine logic for the frozen output state is covered "
            "in test_terminal_freezes_state_synthetic"
        )
    ir = _ir([terminal_stages[0]])
    report = dry_run_pipeline(ir, registry)
    assert report.stages[0].output_state.frozen is True


# ----------------------------------------------------------------------------
# Payload shape — used by the staging critic + CLI artifact
# ----------------------------------------------------------------------------


def test_to_payload_is_json_serializable(registry) -> None:
    import json

    ir = _ir(["VADSegmentationStage", "UTMOSFilterStage"])
    report = dry_run_pipeline(ir, registry)
    payload = report.to_payload()
    # Should round-trip through JSON without errors.
    raw = json.dumps(payload)
    parsed = json.loads(raw)
    assert "trace" in parsed
    assert "initial_state" in parsed
    assert isinstance(parsed["trace"], list)
