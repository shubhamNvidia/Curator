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

"""Unit tests for the deterministic grounding layer (validate / checks)."""

from nemo_curator import audio_agent as aa

_READER = {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/m.jsonl"}}
_WRITER = {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/out.jsonl"}}
_DURATION = {"ref": "GetAudioDurationStage", "params": {}}


def _validate(stages: list[dict], **kw: object) -> dict:
    return aa.validate({"stages": stages}, **kw)


def _codes(verdict: dict) -> set[str]:
    return {
        issue["code"]
        for pool in ("issues", "card_violations", "gate_flags")
        for issue in (verdict.get(pool) or [])
    }


class TestValidateContract:
    def test_returns_json_dict(self) -> None:
        r = _validate([_READER, _DURATION, _WRITER])
        assert isinstance(r, dict)
        assert "status" in r

    def test_valid_recipe_is_runnable(self) -> None:
        r = _validate([_READER, _DURATION, _WRITER])
        assert r.get("runnable") is not False
        assert r.get("status") != "fail"


class TestDataFlowChecks:
    def test_tensor_into_sink_flagged(self) -> None:
        # A resident-waveform producer feeding a JSON sink without AudioToDocument.
        r = _validate([_READER, {"ref": "SpeakerSeparationStage", "params": {}}, _WRITER])
        assert "tensor_into_sink" in _codes(r)
        assert r.get("status") == "fail"

    def test_sanitized_flow_clears_tensor_into_sink(self) -> None:
        r = _validate(
            [
                _READER,
                {"ref": "SpeakerSeparationStage", "params": {}},
                {"ref": "AudioToDocumentStage", "params": {}},
                _WRITER,
            ]
        )
        assert "tensor_into_sink" not in _codes(r)


class TestCheckIsolation:
    def test_malformed_num_speakers_does_not_crash(self) -> None:
        # H3: a non-numeric card-constrained param must yield a JSON verdict, not a traceback.
        r = _validate([_READER, {"ref": "InferenceSortformerStage", "params": {"num_speakers": "two"}}, _WRITER])
        assert isinstance(r, dict)
        assert "status" in r


class TestOutputCompleteness:
    def test_missing_producer_flagged(self) -> None:
        r = _validate([_READER, _DURATION, _WRITER], expected_outputs=["nonexistent_metric_xyz"])
        assert "missing_output_producer" in _codes(r)

    def test_satisfied_by_produced_key(self) -> None:
        # H5: 'duration' is produced by GetAudioDurationStage -> not flagged.
        r = _validate([_READER, _DURATION, _WRITER], expected_outputs=["duration"])
        assert "missing_output_producer" not in _codes(r)


class TestDiarizationContinuity:
    def test_vad_before_diarizer_without_rejoin_flagged(self) -> None:
        r = _validate(
            [_READER, {"ref": "VADSegmentationStage", "params": {}}, {"ref": "InferenceSortformerStage", "params": {}}, _WRITER]
        )
        assert "diarization_needs_continuous_audio" in _codes(r)

    def test_diarizer_on_continuous_audio_ok(self) -> None:
        # Diarizing before any VAD is fine (continuous waveform).
        r = _validate([_READER, {"ref": "InferenceSortformerStage", "params": {}}, _WRITER])
        assert "diarization_needs_continuous_audio" not in _codes(r)
