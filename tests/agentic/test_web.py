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
"""Tests for the smart-clarifier web app.

The contract is a two-call flow:

1. ``POST /api/plan`` → returns
   ``{status: 'form_ready', smart_form: SmartForm, ...}``.
   The smart form carries (a) inferred chips, (b) 1-5 adaptive questions,
   and (c) the legacy ``advanced_form`` for the "Show all options"
   expander.

2. ``POST /api/build`` → applies answers and returns
   ``{status: 'ready', yaml: ...}``.

The LLM is deliberately stubbed in these tests so we exercise the
deterministic fallback path. The smart-form composer falls back to the
template phrasing when the LLM is unavailable or errors, so the same
test prompts cover both wire-shape and gap-analysis correctness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datetime import datetime, timezone

from nemo_curator.agentic.intent import IntentCategories
from nemo_curator.agentic.web import AgenticWebApp, UserSession, WebDefaults

_FAKE_API_KEY = "nvapi-testkey1234567890abcdef"


class _StubIntent:
    """Replaces :func:`extract_intent` to skip the LLM round-trip."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        prompt: str,
        profile,
        llm,
        *,
        tier: str = "synth",
    ) -> IntentCategories:
        self.calls += 1
        return IntentCategories(raw_prompt=prompt)


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    """A tmp directory containing one placeholder ``.wav`` file."""

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "sample.wav").write_bytes(b"")
    return audio_dir


@pytest.fixture()
def app(monkeypatch, tmp_path: Path) -> AgenticWebApp:
    monkeypatch.setattr(
        "nemo_curator.agentic.web.extract_intent",
        _StubIntent(),
    )
    monkeypatch.setattr(
        "nemo_curator.agentic.web.profile_source",
        lambda *_a, **_k: None,
    )
    instance = AgenticWebApp(WebDefaults(dataset=str(tmp_path), out_root=str(tmp_path)))
    # Force the smart clarifier into its template-fallback path so tests
    # never make a real network call to NVIDIA NIM.
    instance.llm = None

    # Inject a pre-authenticated test user so plan/build calls don't
    # hit the "Please sign in first" guard added by the auth layer.
    session_dir = tmp_path / "sessions" / "test_user__deadbeef"
    (session_dir / "runs").mkdir(parents=True, exist_ok=True)
    test_user = UserSession(
        user_id="test-user-id",
        email="test@nvidia.com",
        api_key=_FAKE_API_KEY,
        created_at=datetime.now(timezone.utc),
        session_dir=session_dir,
    )
    instance.user_sessions["test-user-id"] = test_user
    instance._test_user_id = "test-user-id"  # convenience for tests
    return instance


def _plan(app: AgenticWebApp, payload: dict) -> dict:
    """Inject the test user_id into every plan call."""
    return app.plan({"user_id": app._test_user_id, **payload})


def _build(app: AgenticWebApp, payload: dict) -> dict:
    """Inject the test user_id into every build call."""
    return app.build({"user_id": app._test_user_id, **payload})


def test_plan_returns_smart_form_with_advanced_fallback(
    app: AgenticWebApp, dataset_dir: Path, tmp_path: Path
) -> None:
    """A vague prompt produces (a) a short question list and
    (b) the legacy full ingredient form for the advanced expander."""

    response = _plan(app, {
        "prompt": "Clean my audio and make it useful.",
        "dataset": str(dataset_dir),
        "kind": "directory",
        "out": str(tmp_path / "run1"),
    })
    assert response["status"] == "form_ready"
    assert response["session_id"]
    smart = response["smart_form"]
    assert isinstance(smart["questions"], list), "smart form must always carry a questions list"
    # 'clean' is a quality word → the regex heuristic enriches mos=filter
    # behind the scenes, so it shows up as an inferred chip instead of a
    # question. The sample-rate gap, however, is essential and still
    # missing, so it must surface in the question list.
    paths = [q["intent_path"] for q in smart["questions"]]
    assert "output.sample_rate" in paths
    # MOS chip must be in the inferred panel since the heuristic set it.
    chip_paths = [c["intent_path"] for c in smart["inferred"]]
    assert "quality.mos" in chip_paths
    # The advanced form is the legacy ClarificationForm — backwards compat
    # for power users via the "Show all options" expander.
    section_ids = [s["id"] for s in smart["advanced_form"]["sections"]]
    assert section_ids == ["output", "segmentation", "quality", "annotations", "policy"]
    # Heuristics fire even on a vague "clean" prompt → MOS gate pre-filled.
    intent = smart["intent"]
    assert intent["quality"]["mos"] == "filter"


def test_build_with_no_answers_uses_prefills(
    app: AgenticWebApp, dataset_dir: Path, tmp_path: Path
) -> None:
    """If the user submits the form without changing anything, the build
    succeeds using whatever ingredients the prompt + profile produced."""

    out_dir = tmp_path / "run2"
    plan_resp = _plan(app, {
        "prompt": "Clean TTS dataset at 24 kHz, 2-60 second clips.",
        "dataset": str(dataset_dir),
        "kind": "directory",
        "out": str(out_dir),
    })
    session_id = plan_resp["session_id"]

    build_resp = _build(app, {
        "session_id": session_id,
        "answers": {},
    })
    assert build_resp["status"] == "ready", build_resp
    assert build_resp["yaml"], "expected compiled YAML"
    assert (out_dir / "compiled.yaml").exists()
    assert (out_dir / "ir.validated.json").exists()
    assert (out_dir / "prompt.txt").read_text().strip() == (
        "Clean TTS dataset at 24 kHz, 2-60 second clips."
    )
    # No answer changes ⇒ saved answers is the empty dict.
    saved = json.loads((out_dir / "clarification_answers.json").read_text())
    assert saved == {}


def test_build_with_user_answers_overrides_prefills(
    app: AgenticWebApp, dataset_dir: Path, tmp_path: Path
) -> None:
    """The user's namespaced answers win over prompt-driven prefills."""

    out_dir = tmp_path / "run3"
    plan_resp = _plan(app, {
        "prompt": "Clean TTS dataset at 24 kHz.",
        "dataset": str(dataset_dir),
        "kind": "directory",
        "out": str(out_dir),
    })
    session_id = plan_resp["session_id"]

    answers = {
        "output.sample_rate": 48000,
        "output.resample_input": True,
    }
    build_resp = _build(app, {"session_id": session_id, "answers": answers})
    assert build_resp["status"] == "ready", build_resp
    # The intent in the response reflects the user's override.
    assert build_resp["intent"]["output"]["sample_rate"] == 48000
    saved = json.loads((out_dir / "clarification_answers.json").read_text())
    assert saved["output.sample_rate"] == 48000


def test_build_expired_session_returns_error(
    app: AgenticWebApp, dataset_dir: Path, tmp_path: Path
) -> None:
    """Building against a never-planned session id surfaces a friendly error."""

    response = _build(app, {"session_id": "not-a-real-session", "answers": {}})
    assert response["status"] == "expired"
    assert "error" in response


def test_freeform_apply_side_effect_writes_both_legs(
    app: AgenticWebApp, dataset_dir: Path, tmp_path: Path
) -> None:
    """The freeform 'min,max' duration option ships an ``__apply__`` dict."""

    out_dir = tmp_path / "run4"
    plan_resp = _plan(app, {
        "prompt": "Make speech clips with a custom duration range.",
        "dataset": str(dataset_dir),
        "kind": "directory",
        "out": str(out_dir),
    })
    session_id = plan_resp["session_id"]

    build_resp = _build(app, {
        "session_id": session_id,
        "answers": {
            "segmentation.duration_min_sec": 5.0,
            "segmentation.duration_max_sec": 25.0,
            "segmentation.output_unit": "speech_segments",
        },
    })
    assert build_resp["status"] == "ready", build_resp
    vad_stages = [s for s in build_resp.get("stages", []) if s == "VADSegmentationStage"]
    assert vad_stages, build_resp
