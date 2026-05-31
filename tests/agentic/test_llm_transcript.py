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
"""Tests for the per-request LLM call transcript.

We exercise the three things callers depend on:

1. ``LLMClient.chat[_json]`` writes one JSONL row per call when a
   transcript is bound.
2. ``bind_transcript(None)`` silences recording even inside an outer
   binding (so tests can opt out cleanly).
3. The recorder truncates long messages but preserves the original
   character count for forensics.
"""

from __future__ import annotations

import json
from pathlib import Path

from nemo_curator.agentic.llm import Message, MockLLM
from nemo_curator.agentic.llm_transcript import (
    LLMTranscript,
    bind_transcript,
    current_transcript,
)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_records_one_row_per_chat_call(tmp_path: Path) -> None:
    log = tmp_path / "llm_calls.jsonl"
    transcript = LLMTranscript(path=log, run_id="r1", user_id="u1")
    llm = MockLLM(responder=lambda _msgs, _tier: '{"ok": true}')

    with bind_transcript(transcript):
        out = llm.chat_json(
            [Message("system", "be helpful"), Message("user", "hi")],
            tier="synth",
            purpose="unit_test",
        )

    assert out == {"ok": True}
    rows = _read_rows(log)
    assert len(rows) == 1
    row = rows[0]
    assert row["purpose"] in ("unit_test", "chat_json")  # purpose passes through
    assert row["tier"] == "synth"
    assert row["status"] == "ok"
    assert row["run_id"] == "r1"
    assert row["user_id"] == "u1"
    assert row["n_messages"] == 2
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    assert "latency_s" in row and row["latency_s"] >= 0
    assert row["error"] is None


def test_no_transcript_means_no_log(tmp_path: Path) -> None:
    """Without a binding, no file should be written and the call still works."""

    llm = MockLLM(responder=lambda _msgs, _tier: "hi")
    # Sanity: no current binding.
    assert current_transcript() is None
    out = llm.chat([Message("user", "say hi")], tier="planner")
    assert out == "hi"
    # No stray files in tmp_path.
    assert list(tmp_path.iterdir()) == []


def test_bind_none_silences_outer_binding(tmp_path: Path) -> None:
    """Nested ``bind_transcript(None)`` blocks while keeping outer state."""

    log = tmp_path / "calls.jsonl"
    transcript = LLMTranscript(path=log)
    llm = MockLLM(responder=lambda _msgs, _tier: "y")

    with bind_transcript(transcript):
        llm.chat([Message("user", "first")], tier="planner")
        with bind_transcript(None):
            llm.chat([Message("user", "silenced")], tier="planner")
        llm.chat([Message("user", "third")], tier="planner")

    rows = _read_rows(log)
    assert len(rows) == 2, rows
    bodies = [r["messages"][0]["content"] for r in rows]
    assert bodies == ["first", "third"]


def test_records_errors_with_status_error(tmp_path: Path) -> None:
    """A raising responder still produces a row with ``status=error``."""

    log = tmp_path / "err.jsonl"
    transcript = LLMTranscript(path=log)

    def _raise(_msgs, _tier):  # noqa: ANN001
        raise RuntimeError("boom")

    llm = MockLLM(responder=_raise)
    with bind_transcript(transcript):
        try:
            llm.chat([Message("user", "x")], tier="planner")
        except RuntimeError:
            pass

    rows = _read_rows(log)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "RuntimeError: boom" in rows[0]["error"]


def test_long_message_truncation_keeps_chars_count(tmp_path: Path) -> None:
    """Messages longer than ``max_chars`` are truncated but the original
    length is preserved on the row so forensics can spot oversize prompts."""

    log = tmp_path / "long.jsonl"
    transcript = LLMTranscript(path=log, max_chars=64)
    body = "x" * 1024
    llm = MockLLM(responder=lambda _msgs, _tier: body)
    with bind_transcript(transcript):
        llm.chat([Message("user", body)], tier="synth")

    rows = _read_rows(log)
    assert len(rows) == 1
    msg = rows[0]["messages"][0]
    assert msg["chars"] == 1024
    assert msg["truncated"] is True
    assert "[truncated" in msg["content"]
    # Response is also bounded.
    assert "[truncated" in rows[0]["response_text"]
    assert rows[0]["response_chars"] == 1024
