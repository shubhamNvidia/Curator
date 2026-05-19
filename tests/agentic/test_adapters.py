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
"""Tests for :mod:`nemo_curator.agentic.adapters`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemo_curator.agentic.adapters import (
    DEFAULT_AUDIO_EXTS,
    build_manifest_from_directory,
    normalize_source,
    probe_source,
    reader_stage,
)
from nemo_curator.agentic.ir import SourceSpec


class TestNormalizeSource:
    def test_manifest_default_passes_through(self) -> None:
        src = SourceSpec(kind="manifest", uri="/tmp/m.jsonl")
        normed = normalize_source(src)
        assert normed.kind == "manifest"

    def test_directory_detection(self, tmp_path: Path) -> None:
        src = SourceSpec(kind="manifest", uri=str(tmp_path))
        normed = normalize_source(src)
        assert normed.kind == "directory"

    def test_fleurs_scheme(self) -> None:
        src = SourceSpec(kind="manifest", uri="fleurs://en_us")
        normed = normalize_source(src)
        assert normed.kind == "fleurs"

    def test_readspeech_scheme(self) -> None:
        src = SourceSpec(kind="manifest", uri="readspeech://main")
        normed = normalize_source(src)
        assert normed.kind == "readspeech"


class TestProbe:
    def test_empty_directory(self, tmp_path: Path) -> None:
        src = SourceSpec(kind="directory", uri=str(tmp_path))
        probe = probe_source(src)
        assert probe.discovered_files == []
        assert probe.total_estimated == 0

    def test_directory_finds_wavs(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"f{i}.wav").write_bytes(b"fake")
        src = SourceSpec(kind="directory", uri=str(tmp_path))
        probe = probe_source(src, sample_n=2)
        assert probe.total_estimated == 3
        assert len(probe.discovered_files) == 2

    def test_manifest_streams_jsonl(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.jsonl"
        with manifest.open("w") as f:
            for i in range(5):
                f.write(json.dumps({"audio_filepath": f"/no/such/f{i}.wav"}) + "\n")
        src = SourceSpec(kind="manifest", uri=str(manifest))
        probe = probe_source(src, sample_n=3)
        assert probe.total_estimated == 5
        assert len(probe.manifest_lines or []) == 3

    def test_missing_manifest_returns_empty(self, tmp_path: Path) -> None:
        src = SourceSpec(kind="manifest", uri=str(tmp_path / "missing.jsonl"))
        probe = probe_source(src)
        assert probe.discovered_files == []


class TestReaderStage:
    def test_manifest_uses_manifest_reader(self) -> None:
        ref = reader_stage(SourceSpec(kind="manifest", uri="/tmp/m.jsonl"))
        assert ref.stage == "ManifestReader"
        assert ref.params["manifest_path"] == "/tmp/m.jsonl"
        assert ref.auto_inserted is True

    def test_fleurs(self) -> None:
        ref = reader_stage(SourceSpec(kind="fleurs", uri="fleurs://en_us", options={"lang": "en_us", "split": "test"}))
        assert ref.stage == "CreateInitialManifestFleursStage"
        assert ref.params["lang"] == "en_us"

    def test_readspeech(self) -> None:
        ref = reader_stage(SourceSpec(kind="readspeech", uri="readspeech://main", options={"max_samples": 100}))
        assert ref.stage == "CreateInitialManifestReadSpeechStage"
        assert ref.params["max_samples"] == 100

    def test_directory_requires_preprocess(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="preprocessed into a JSONL manifest"):
            reader_stage(SourceSpec(kind="directory", uri=str(tmp_path)))


class TestBuildManifestFromDirectory:
    def test_walks_and_filters(self, tmp_path: Path) -> None:
        (tmp_path / "a.wav").write_bytes(b"x")
        (tmp_path / "b.txt").write_text("ignore")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.flac").write_bytes(b"x")
        out = build_manifest_from_directory(tmp_path)
        rows = [json.loads(l) for l in out.read_text().splitlines() if l]
        paths = {Path(r["audio_filepath"]).name for r in rows}
        assert paths == {"a.wav", "c.flac"}

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_manifest_from_directory(tmp_path)


def test_default_extensions_include_common_formats() -> None:
    assert ".wav" in DEFAULT_AUDIO_EXTS
    assert ".flac" in DEFAULT_AUDIO_EXTS
