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
"""Tests for the multi-agent planner DAG (V2 intent JSON).

Each test stubs the LLM with :class:`MockLLM` and dispatches per-step
responses based on the system-prompt content. This keeps the tests
deterministic (no network, no NIM key) while still covering the actual
plumbing in :mod:`nemo_curator.agentic.planner_dag`.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from nemo_curator.agentic.cards import (
    CapabilityTag,
    DatasetCard,
    DatasetProfile,
)
from nemo_curator.agentic.intent import (
    FilterMode,
    IntentCategories,
    Quality,
    Segmentation,
    Speakers,
    required_capabilities,
)
from nemo_curator.agentic.llm import Message, MockLLM
from nemo_curator.agentic.planner_dag import (
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
    """Build a MockLLM that returns canned per-role responses."""

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


def _v2_clean_tts_json() -> str:
    """Standard V2 IntentCategories payload for a clean single-speaker TTS intent."""

    return json.dumps({
        "output": {"sample_rate": 48000, "channels": "mono"},
        "segmentation": {
            "output_unit": "single_speaker_clips",
            "duration_min_sec": 2.0,
            "duration_max_sec": 60.0,
        },
        "quality": {"mos": "filter", "mos_threshold": 3.5},
        "speakers": {"mode": "split"},
        "policy": {"commercial_only": True},
    })


# ----------------------------------------------------------------------------
# Step 1: Intent Extractor
# ----------------------------------------------------------------------------


class TestIntentExtractor:
    def test_v2_json_round_trips_into_namespaced_intent(self) -> None:
        llm, counter = _make_mock({"extractor": [_v2_clean_tts_json()]})
        intent = extract_intent(
            "Build a clean single-speaker dataset at 48 kHz mono, 2-60 s, commercial-safe.",
            profile=None,
            llm=llm,
        )
        assert counter["extractor"] == 1
        assert intent.output.sample_rate == 48000
        assert intent.output.channels == "mono"
        assert intent.quality.mos == FilterMode.FILTER
        assert intent.quality.mos_threshold == 3.5
        assert intent.speakers.mode == FilterMode.SPLIT
        assert intent.policy.commercial_only is True


# ----------------------------------------------------------------------------
# Step 3: Stage Picker
# ----------------------------------------------------------------------------


class TestStagePicker:
    def _tts_intent(self) -> IntentCategories:
        return IntentCategories(
            segmentation=Segmentation(
                output_unit="single_speaker_clips",
                duration_min_sec=2.0, duration_max_sec=60.0,
            ),
            quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5),
            speakers=Speakers(mode=FilterMode.SPLIT),
        )

    def test_picker_dispatches_per_capability(self, registry) -> None:
        """``single_speaker_clips`` pulls speaker-separation + extract +
        the user-requested quality filter; with a duration cap it ALSO
        pulls VAD as the post-segmenter trim (see
        :func:`required_capabilities`)."""

        intent = self._tts_intent()
        caps = required_capabilities(intent)
        llm, counter = _make_mock({
            "picker": {
                "quality_filter_mos": json.dumps({"capability": "quality_filter_mos", "chosen_stages": ["UTMOSFilterStage", "SIGMOSFilterStage"]}),
                "speaker_separation": json.dumps({"capability": "speaker_separation", "chosen_stages": ["SpeakerSeparationStage"]}),
                "vad": json.dumps({"capability": "vad", "chosen_stages": ["VADSegmentationStage"]}),
            },
        })
        names = pick_stages(caps, registry, llm)
        assert "SpeakerSeparationStage" in names
        assert "SegmentExtractionStage" in names  # single candidate, no LLM call
        assert "UTMOSFilterStage" in names
        assert "SIGMOSFilterStage" in names
        assert "VADSegmentationStage" in names  # post-segmenter trim
        # 3 multi-candidate capabilities → 3 picker calls
        # (quality_filter_mos, speaker_separation, vad).
        assert counter["picker"] == 3

    def test_picker_accepts_legacy_single_stage_shape(self, registry) -> None:
        intent = IntentCategories(quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5))
        caps = required_capabilities(intent)
        llm, _ = _make_mock({
            "picker": {
                "quality_filter_mos": json.dumps({"capability": "quality_filter_mos", "chosen_stage": "UTMOSFilterStage"}),
            },
        })
        names = pick_stages(caps, registry, llm)
        assert "UTMOSFilterStage" in names

    def test_picker_falls_back_when_llm_returns_unknown_stage(self, registry) -> None:
        intent = IntentCategories(quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5))
        caps = required_capabilities(intent)
        llm, _ = _make_mock({
            "picker": {
                "quality_filter_mos": json.dumps({"capability": "quality_filter_mos", "chosen_stages": ["NotARealStage"]}),
            },
        })
        names = pick_stages(caps, registry, llm)
        candidates = {e.card.name for e in registry.search_by_capability(CapabilityTag.QUALITY_FILTER_MOS)}
        assert set(names) & candidates


# ----------------------------------------------------------------------------
# Step 4: Param Tuner
# ----------------------------------------------------------------------------


class TestParamTuner:
    def test_uses_threshold_bands(self, registry) -> None:
        intent = IntentCategories(quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5))
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
        intent = IntentCategories(quality=Quality(mos=FilterMode.FILTER, mos_threshold=3.5))
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
        monkeypatch.setattr(
            "nemo_curator.agentic.planner_dag.profile_source",
            lambda *a, **kw: DatasetCard(
                name="mock", uri="/tmp/m.jsonl", profile=DatasetProfile()
            ),
        )
        responses = {
            "extractor": [_v2_clean_tts_json()],
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
        """The big de-bias regression test."""

        monkeypatch.setattr(
            "nemo_curator.agentic.planner_dag.profile_source",
            lambda *a, **kw: DatasetCard(
                name="mock", uri="/tmp/m.jsonl", profile=DatasetProfile()
            ),
        )
        responses = {
            "extractor": [_v2_clean_tts_json()],
            "picker": {
                "quality_filter_mos": json.dumps({"capability": "quality_filter_mos", "chosen_stages": ["UTMOSFilterStage", "SIGMOSFilterStage"]}),
                "speaker_separation": json.dumps({"capability": "speaker_separation", "chosen_stages": ["SpeakerSeparationStage"]}),
                "vad": json.dumps({"capability": "vad", "chosen_stages": ["VADSegmentationStage"]}),
                "segment_extract": json.dumps({"capability": "segment_extract", "chosen_stages": ["SegmentExtractionStage"]}),
            },
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
        assert {"UTMOSFilterStage", "SIGMOSFilterStage"} <= names, (
            f"expected both MOS filters in {names}"
        )
        assert "BandFilterStage" not in names, f"BandFilter sneaked in: {names}"
        assert result.critic_history[-1].approved
        assert counter["critic"] == 1
