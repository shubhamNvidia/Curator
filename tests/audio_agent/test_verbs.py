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

"""Unit tests for verb-level guardrails: the confirm gate, workspace lock, require-smoke,
resolve, and row-accurate evidence counting (no GPU / Ray execution needed)."""

from nemo_curator import audio_agent as aa
from nemo_curator.audio_agent.report import _row_count

_READER = {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/m.jsonl"}}
_WRITER = {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/out.jsonl"}}
_RECIPE = {"stages": [_READER, {"ref": "GetAudioDurationStage", "params": {}}, _WRITER]}


class TestRunConfirmGate:
    def test_refuses_without_confirmation(self) -> None:
        r = aa.run(_RECIPE, confirm=False)
        assert r["status"] == "refused"
        assert "confirmation" in r["reason"].lower()
        assert "config_hash" in r

    def test_refuses_on_hash_mismatch(self) -> None:
        r = aa.run(_RECIPE, confirm="not-the-real-hash")
        assert r["status"] == "refused"
        assert "integrity" in r["reason"].lower()


class TestRunWorkspaceLock:
    def test_refuses_path_outside_workspace(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        r = aa.run(_RECIPE, confirm=True, data="/etc/passwd")
        assert r["status"] == "refused"
        assert "workspace" in r["reason"].lower()


class TestRunRequireSmoke:
    def test_refuses_without_smoke_token(self, monkeypatch) -> None:
        monkeypatch.setenv("AUDIO_AGENT_REQUIRE_SMOKE", "1")
        monkeypatch.delenv("AUDIO_AGENT_WORKSPACE", raising=False)
        r = aa.run(_RECIPE, confirm=True)
        assert r["status"] == "refused"
        assert "smoke" in r["reason"].lower()


class TestResolve:
    def test_resolves_label_to_concrete_params(self) -> None:
        r = aa.resolve("UTMOSFilterStage", label="studio")
        assert isinstance(r, dict)
        assert "mos_threshold" in str(r)  # a concrete threshold was resolved


class TestRowCount:
    def test_counts_rows_not_tasks(self) -> None:
        class _T:
            def __init__(self, n: int) -> None:
                self.num_items = n

        assert _row_count([_T(1), _T(1)]) == 2  # AudioTask-like: 1 row each
        assert _row_count([_T(500)]) == 500  # DocumentBatch-like: one task, 500 rows
        assert _row_count(None) == 0
