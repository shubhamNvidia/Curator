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

"""Unit tests for the LLM-plane trace grader's H7 fixes: continuation derivation and
trace-level safety signals (confirm-gate, redaction)."""

from eval.audio.trace_check import _continuation, _transcript_leak_count


class TestContinuationDerivation:
    def test_derives_from_plan_continuation_tool_result(self) -> None:
        # agent_runner leaves trace['continuation']=None; derive it from the tool call.
        trace = {"tool_calls": [{"verb": "plan_continuation", "result": {"mode": "extend", "run_stages": ["X"]}}]}
        assert _continuation(trace).get("mode") == "extend"

    def test_prefers_explicit_field(self) -> None:
        trace = {
            "continuation": {"mode": "full_rerun"},
            "tool_calls": [{"verb": "plan_continuation", "result": {"mode": "extend"}}],
        }
        assert _continuation(trace).get("mode") == "full_rerun"

    def test_empty_when_absent(self) -> None:
        assert _continuation({"tool_calls": [{"verb": "validate"}]}) == {}


class TestTranscriptLeakDetection:
    def test_flags_raw_transcript(self) -> None:
        leaky = {"tool_calls": [{"verb": "smoke", "result": {"examples": [{"text": "hello world this is a raw transcript"}]}}]}
        assert _transcript_leak_count(leaky) == 1

    def test_ignores_redacted_transcript(self) -> None:
        clean = {"tool_calls": [{"verb": "smoke", "result": {"examples": [{"text": "<redacted-transcript:20chars>"}]}}]}
        assert _transcript_leak_count(clean) == 0

    def test_zero_when_no_transcript_keys(self) -> None:
        assert _transcript_leak_count({"tool_calls": [{"verb": "validate", "result": {"status": "pass"}}]}) == 0
