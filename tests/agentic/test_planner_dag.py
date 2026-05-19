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
"""Tests for the multi-agent planner DAG.

Each test stubs the LLM with :class:`MockLLM` and dispatches per-step
responses based on the system-prompt content. This keeps the tests
deterministic (no network, no NIM key) while still covering the actual
plumbing in :mod:`nemo_curator.agentic.planner_dag`.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any
from unittest.mock import patch

import pytest

from nemo_curator.agentic.cards import (
    CapabilityTag,
    DatasetCard,
    DatasetProfile,
)
from nemo_curator.agentic.intent import IntentCategories, required_capabilities
from nemo_curator.agentic.llm import Message, MockLLM
from nemo_curator.agentic.planner_dag import (
    CriticVerdict,
    critic_review,
    derive_capabilities,
    extract_intent,
    pick_stages,
    plan,
    tune_params,
)
from nemo_curator.agentic.registry import build_registry


@pytest.fixture(scope="module")
def registry():
    return build_registry()


def _detect_role(messages: list[Message]) -> str:
    """Identify which role prompt is in the system message of an LLM call."""

    sys_msg = next((m.content for m in messages if m.role == "system"), "")
    if "Intent Extractor" in sys_msg:
        return "extractor"
    if "Stage Picker" in sys_msg:
        return "picker"
    if "Param Tuner" in sys_msg:
        return "tuner"
    if "Critic" in sys_msg:
        return "critic"
    return "unknown"


def _make_mock(
    responses_by_role: dict[str, list[str] | dict[str, str]],
) -> tuple[MockLLM, Counter[str]]:
    """Build a MockLLM that returns canned per-role responses.

    For each role:

    - A ``list[str]`` means consume one entry per call in order (looping
      the last one forever).
    - A ``dict[str, str]`` is a per-capability or per-stage dispatch: the
      mock peeks at the user JSON for a ``capability`` or ``stage`` key
      and returns the matching entry. Missing key → ``""`` empty JSON.
    """

    counter: Counter[str] = Counter()

    def responder(messages: list[Message], _tier: str) -> str:
        role = _detect_role(messages)
        counter[role] += 1
        bucket = responses_by_role.get(role)
        if not bucket:
            return "{}"
        if isinstance(bucket, dict):
            user_msg = next((m.content for m in messages if m.role == "user"), "{}")
            try:
                payload = json.loads(user_msg)
            except Exception:
                payload = {}
            key = payload.get("capability") or payload.get("stage") or ""
            return bucket.get(key, "{}")
        idx = min(counter[role] - 1, len(bucket) - 1)
        return bucket[idx]

    return MockLLM(responder=responder), counter


# ----------------------------------------------------------------------------
# Step 1: Intent Extractor
# ----------------------------------------------------------------------------


class TestIntentExtractor:
    def test_clean_translates_to_quality_mos_min(self) -> None:
        llm, counter = _make_mock({
            "extractor": [json.dumps({
                "sample_rate": 48000,
                "channels": "mono",
                "duration_min_sec": 2.0,
                "duration_max_sec": 60.0,
                "output_extract_clips": True,
                "quality_mos_min": 3.5,
                "speakers": 1,
                "commercial_only": True,
            })],
        })
        intent = extract_intent(
            "Build a clean single-speaker dataset at 48 kHz mono, 2-60 s, commercial-safe.",
            profile=None,
            llm=llm,
        )
        assert counter["extractor"] == 1
        assert intent.quality_mos_min == 3.5
        assert intent.speakers == 1
        assert intent.sample_rate == 48000
        assert intent.channels == "mono"


# ----------------------------------------------------------------------------
# Step 3: Stage Picker
# ----------------------------------------------------------------------------


class TestStagePicker:
    def test_picker_dispatches_per_capability(self, registry) -> None:
        """The picker is invoked once per CapabilityRequirement that has
        multiple candidates. Single-candidate caps short-circuit (no LLM)."""
        intent = IntentCategories(
            quality_mos_min=3.5,
            speakers=1,
            duration_min_sec=2.0,
            duration_max_sec=60.0,
            output_extract_clips=True,
        )
        caps = required_capabilities(intent)

        # Capability → picker's choice. AudioDataFilterStage is a composite
        # that claims many capabilities via also_handles; tests choose the
        # dedicated specialist for each cap so the picker's behavior is
        # explicit.
        llm, counter = _make_mock({
            "picker": {
                "quality_filter_mos": json.dumps({"capability": "quality_filter_mos", "chosen_stages": ["UTMOSFilterStage", "SIGMOSFilterStage"]}),
                "speaker_separation": json.dumps({"capability": "speaker_separation", "chosen_stages": ["SpeakerSeparationStage"]}),
                "vad": json.dumps({"capability": "vad", "chosen_stages": ["VADSegmentationStage"]}),
            },
        })
        names = pick_stages(caps, registry, llm)

        assert "SpeakerSeparationStage" in names
        assert "VADSegmentationStage" in names
        assert "SegmentExtractionStage" in names  # single candidate, no LLM call
        assert "UTMOSFilterStage" in names
        assert "SIGMOSFilterStage" in names  # multi-stage pick from picker

        # 3 multi-candidate capabilities → 3 picker calls.
        assert counter["picker"] == 3

    def test_picker_accepts_legacy_single_stage_shape(self, registry) -> None:
        """The shim should accept the older ``chosen_stage: str`` schema too."""
        intent = IntentCategories(quality_mos_min=3.5)
        caps = required_capabilities(intent)
        llm, _counter = _make_mock({
            "picker": {
                "quality_filter_mos": json.dumps({"capability": "quality_filter_mos", "chosen_stage": "UTMOSFilterStage"}),
            },
        })
        names = pick_stages(caps, registry, llm)
        assert "UTMOSFilterStage" in names

    def test_picker_falls_back_when_llm_returns_unknown_stage(self, registry) -> None:
        intent = IntentCategories(quality_mos_min=3.5)
        caps = required_capabilities(intent)
        llm, _counter = _make_mock({
            "picker": {
                "quality_filter_mos": json.dumps({"capability": "quality_filter_mos", "chosen_stages": ["NotARealStage"]}),
            },
        })
        names = pick_stages(caps, registry, llm)
        # Fallback to the first registry candidate; either UTMOS, SIGMOS, or
        # AudioDataFilterStage is acceptable as long as it's in the registry.
        candidates = {e.card.name for e in registry.search_by_capability(CapabilityTag.QUALITY_FILTER_MOS)}
        assert set(names) & candidates


# ----------------------------------------------------------------------------
# Step 4: Param Tuner
# ----------------------------------------------------------------------------


class TestParamTuner:
    def test_uses_threshold_bands(self, registry) -> None:
        intent = IntentCategories(quality_mos_min=3.5)
        llm, counter = _make_mock({
            "tuner": [json.dumps({
                "stage": "UTMOSFilterStage",
                "params": {"mos_threshold": 3.5},
            })],
        })
        stages = tune_params(["UTMOSFilterStage"], intent, registry, llm)
        assert len(stages) == 1
        assert stages[0].stage == "UTMOSFilterStage"
        assert stages[0].params["mos_threshold"] == 3.5
        assert counter["tuner"] == 1

    def test_tuner_clamps_out_of_range_values(self, registry) -> None:
        """The tuner shim drops unknown params and clamps numeric values to min/max."""
        intent = IntentCategories(quality_mos_min=3.5)
        llm, _ = _make_mock({
            "tuner": [json.dumps({
                "stage": "UTMOSFilterStage",
                "params": {
                    "mos_threshold": 99.0,            # clamped to 5.0 (card max)
                    "not_a_real_param": "boom",       # dropped
                },
            })],
        })
        stages = tune_params(["UTMOSFilterStage"], intent, registry, llm)
        assert stages[0].params["mos_threshold"] == 5.0
        assert "not_a_real_param" not in stages[0].params


# ----------------------------------------------------------------------------
# Step 6: Critic loop
# ----------------------------------------------------------------------------


class TestCriticLoop:
    def test_refinement_terminates_at_max_iters(self, registry, monkeypatch) -> None:
        """If the critic keeps complaining, the planner must give up after
        ``max_refine_iters`` and return whatever it has."""

        # Profile mock — bypass real I/O.
        monkeypatch.setattr(
            "nemo_curator.agentic.planner_dag.profile_source",
            lambda *a, **kw: DatasetCard(
                name="mock", uri="/tmp/m.jsonl", profile=DatasetProfile()
            ),
        )

        responses = {
            "extractor": [json.dumps({
                "channels": "mono",
                "speakers": 1,
                "duration_min_sec": 2.0,
                "duration_max_sec": 60.0,
                "output_extract_clips": True,
                "quality_mos_min": 3.5,
            })],
            "picker": [json.dumps({"capability": "quality_filter_mos", "chosen_stages": ["UTMOSFilterStage"]})],
            "tuner": [json.dumps({"stage": "X", "params": {}})],
            "critic": [json.dumps({"approved": False, "score": 0.4, "complaints": ["nope"], "patch": {}})],
        }
        llm, counter = _make_mock(responses)
        result = plan(
            prompt="clean single-speaker 48k mono 2-60s",
            source_uri="/tmp/m.jsonl",
            source_kind="manifest",
            target_dir="/tmp/out",
            llm=llm,
            registry=registry,
            max_refine_iters=2,
        )
        # 2 refinements + the initial pass = 3 critic calls total.
        assert counter["critic"] == 3
        assert len(result.critic_history) == 3
        assert result.critic_history[-1].approved is False


# ----------------------------------------------------------------------------
# End-to-end DAG
# ----------------------------------------------------------------------------


class TestEndToEndDAG:
    def test_tts_prompt_includes_speaker_and_mos(self, registry, monkeypatch) -> None:
        """The big de-bias regression test: 'clean single-speaker' must
        produce a pipeline containing SpeakerSeparationStage and at least
        one MOS filter, and must NOT include BandFilterStage."""

        monkeypatch.setattr(
            "nemo_curator.agentic.planner_dag.profile_source",
            lambda *a, **kw: DatasetCard(
                name="mock", uri="/tmp/m.jsonl", profile=DatasetProfile()
            ),
        )

        responses = {
            "extractor": [json.dumps({
                "sample_rate": 48000,
                "channels": "mono",
                "duration_min_sec": 2.0,
                "duration_max_sec": 60.0,
                "output_extract_clips": True,
                "quality_mos_min": 3.5,
                "speakers": 1,
                "commercial_only": True,
            })],
            "picker": {
                "quality_filter_mos": json.dumps({"capability": "quality_filter_mos", "chosen_stages": ["UTMOSFilterStage", "SIGMOSFilterStage"]}),
                "speaker_separation": json.dumps({"capability": "speaker_separation", "chosen_stages": ["SpeakerSeparationStage"]}),
                "vad": json.dumps({"capability": "vad", "chosen_stages": ["VADSegmentationStage"]}),
                "segment_extract": json.dumps({"capability": "segment_extract", "chosen_stages": ["SegmentExtractionStage"]}),
            },
            # Tuner answer — used for every stage that has tunable params.
            # The shim drops unrecognized params per-stage, so a single
            # generic reply is safe.
            "tuner": [json.dumps({
                "stage": "any",
                "params": {
                    "mos_threshold": 3.5,
                    "min_duration_sec": 2.0,
                    "max_duration_sec": 60.0,
                    "exclude_overlaps": True,
                    "output_dir": "/tmp/out",
                    "output_format": "wav",
                    "output_path": "/tmp/out/manifest.jsonl",
                },
            })],
            "critic": [json.dumps({"approved": True, "score": 0.95, "complaints": [], "patch": {}})],
        }
        llm, counter = _make_mock(responses)
        result = plan(
            prompt="Build a clean single-speaker dataset at 48 kHz mono, 2-60 s, commercial-safe.",
            source_uri="/tmp/m.jsonl",
            source_kind="manifest",
            target_dir="/tmp/out",
            llm=llm,
            registry=registry,
            max_refine_iters=2,
        )
        names = {s.stage for s in result.ir.stages}
        assert "SpeakerSeparationStage" in names, f"expected SpeakerSeparation in {names}"
        # Both MOS filters (the picker returns the complementary pair).
        assert {"UTMOSFilterStage", "SIGMOSFilterStage"} <= names, (
            f"expected both MOS filters in {names}"
        )
        # BandFilter should not be triggered by 'clean' alone.
        assert "BandFilterStage" not in names, f"BandFilter sneaked in: {names}"
        # The critic approved on the first turn.
        assert result.critic_history[-1].approved
        assert counter["critic"] == 1
