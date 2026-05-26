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
"""Tests for the LLM client and the LLM-augmented critic path.

These tests run entirely offline; the :class:`MockLLM` stub stands in for the
NIM endpoint, so the production code path is exercised end-to-end without ever
opening a socket.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from nemo_curator.agentic.cards import (
    Cardinality,
    CriticReport,
    DatasetCard,
    DatasetProfile,
    RunCard,
    RunStageRecord,
)
from nemo_curator.agentic.critic import CriticOptions, critique
from nemo_curator.agentic.intent import IntentCategories
from nemo_curator.agentic.llm import LLMTier, Message, MockLLM, _parse_json_lenient


def _build_run_card() -> RunCard:
    return RunCard(
        run_id="run-test",
        started_at=datetime.now(timezone.utc),
        pipeline_ir_path="/tmp/ir.json",
        target_dir="/tmp/out",
        success=True,
        stage_records=[
            RunStageRecord(
                stage_name="ManifestReader",
                target="nemo_curator.stages.audio.io.ManifestReader",
                cardinality=Cardinality.ONE_TO_ONE,
                tasks_in=100,
                tasks_out=100,
            ),
            RunStageRecord(
                stage_name="VADSegmentationStage",
                target="nemo_curator.stages.audio.io.VADSegmentationStage",
                cardinality=Cardinality.ONE_TO_MANY,
                tasks_in=100,
                tasks_out=320,
            ),
            RunStageRecord(
                stage_name="ManifestWriterStage",
                target="nemo_curator.stages.audio.io.ManifestWriterStage",
                cardinality=Cardinality.ONE_TO_ONE,
                tasks_in=320,
                tasks_out=320,
            ),
        ],
    )


def _build_intent() -> IntentCategories:
    """A V2 namespaced TTS-style intent used by the critic tests."""

    return IntentCategories(
        output={"sample_rate": 48000, "channels": "mono", "audio_format": "wav"},
        segmentation={
            "output_unit": "single_speaker_clips",
            "duration_min_sec": 2.0,
            "duration_max_sec": 60.0,
        },
        speakers={"mode": "split"},
        policy={"commercial_only": True},
        raw_prompt="test",
    )


def test_mock_llm_returns_canned_response() -> None:
    client = MockLLM(responder=lambda _msgs, _tier: '{"score": 0.95, "findings": []}')
    text = client.chat([Message("user", "hi")], tier="planner")
    parsed = _parse_json_lenient(text)
    assert parsed["score"] == 0.95


def test_parse_json_lenient_extracts_first_object() -> None:
    text = 'noise before {"a": 1, "b": [2, 3]} more noise after'
    assert _parse_json_lenient(text) == {"a": 1, "b": [2, 3]}


def test_default_tiers_picks_up_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CURATOR_ADV_PLANNER_MODEL", "planner-x")
    monkeypatch.setenv("CURATOR_ADV_SYNTH_MODEL", "synth-x")
    from nemo_curator.agentic.llm import default_tiers

    tiers = default_tiers()
    assert tiers.planner == "planner-x"
    assert tiers.synth == "synth-x"


def test_critic_deterministic_only_when_llm_disabled() -> None:
    report = critique(
        run_card=_build_run_card(),
        intent=_build_intent(),
        input_card=None,
        output_card=None,
        options=CriticOptions(use_llm=False),
        llm_client=None,
    )
    assert isinstance(report, CriticReport)
    assert report.llm_used is False
    assert report.llm_intent_alignment_score is None


def test_critic_invokes_llm_when_enabled() -> None:
    captured: dict[str, object] = {}

    def responder(messages: list[Message], tier: str) -> str:
        captured["tier"] = tier
        captured["last_user_message"] = messages[-1].content
        return json.dumps({"score": 0.82, "findings": ["duration p95 within bounds"]})

    client = MockLLM(responder=responder)
    out_profile = DatasetProfile(total_files=320, duration_p50_sec=12.0, duration_p95_sec=55.0)
    out_card = DatasetCard(name="output", uri="/tmp/out", profile=out_profile)

    report = critique(
        run_card=_build_run_card(),
        intent=_build_intent(),
        input_card=None,
        output_card=out_card,
        options=CriticOptions(use_llm=True),
        llm_client=client,
    )
    assert report.llm_used is True
    assert report.llm_intent_alignment_score == pytest.approx(0.82, rel=0.01)
    assert "duration p95 within bounds" in report.llm_findings
    assert captured["tier"] == "synth"
    assert "intent" in str(captured["last_user_message"]).lower()


def test_critic_handles_llm_failure_gracefully() -> None:
    def responder(_msgs: list[Message], _tier: str) -> str:
        raise RuntimeError("simulated NIM 500")

    client = MockLLM(responder=responder)
    out_card = DatasetCard(name="output", uri="/tmp/out", profile=DatasetProfile())
    report = critique(
        run_card=_build_run_card(),
        intent=_build_intent(),
        output_card=out_card,
        options=CriticOptions(use_llm=True),
        llm_client=client,
    )
    assert report.llm_used is True
    assert report.llm_intent_alignment_score is None
    assert any("LLM critic call failed" in m for m in report.llm_findings)


def test_llm_tier_static_defaults_documented() -> None:
    tier = LLMTier()
    # Defaults must point at real, currently-supported models on a
    # NIM-style endpoint. Path layout is "<provider>/<model>" — kept
    # provider-agnostic so we can swap Qwen / Llama / DeepSeek without
    # touching this assertion.
    for name in (tier.planner, tier.synth):
        assert "/" in name
        assert len(name.split("/")[1]) > 0
