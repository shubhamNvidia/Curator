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
"""Tests for the Layer-1 profiler.

These exercise the deterministic logic. ``profile_files`` against real audio
needs ``soundfile``; we generate tiny WAV fixtures via the stdlib ``wave``
module so the test stays self-contained.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from nemo_curator.agentic.ir import SourceSpec
from nemo_curator.agentic.profiler import profile_files, profile_source


def _write_silent_wav(path: Path, *, seconds: float, sample_rate: int, channels: int = 1) -> None:
    n_samples = int(seconds * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # 16-bit silence
        wf.writeframes(b"\x00\x00" * n_samples * channels)


class TestProfileFiles:
    def test_empty_input(self) -> None:
        profile = profile_files([])
        assert profile.total_files == 0
        assert profile.decode_failure_rate == 0.0
        assert profile.notes  # at least one note about emptiness

    def test_single_uniform_dataset(self, tmp_path: Path) -> None:
        paths = []
        for i in range(4):
            p = tmp_path / f"clip_{i}.wav"
            _write_silent_wav(p, seconds=2.0, sample_rate=16000)
            paths.append(p)
        profile = profile_files(paths)
        assert profile.total_files == 4
        assert profile.decodable_files == 4
        assert profile.decode_failure_rate == 0.0
        assert profile.sample_rates_hz == {"16000": 4}
        assert profile.channel_distribution == {"mono": 4}
        assert pytest.approx(profile.duration_p50_sec, rel=0.05) == 2.0

    def test_mixed_sample_rates_emits_note(self, tmp_path: Path) -> None:
        a = tmp_path / "a.wav"; _write_silent_wav(a, seconds=1.0, sample_rate=16000)
        b = tmp_path / "b.wav"; _write_silent_wav(b, seconds=1.0, sample_rate=48000)
        profile = profile_files([a, b])
        joined = " ".join(profile.notes)
        assert "Mixed sample rates" in joined

    def test_undecodable_file_increases_failure_rate(self, tmp_path: Path) -> None:
        good = tmp_path / "g.wav"; _write_silent_wav(good, seconds=1.0, sample_rate=16000)
        bad = tmp_path / "b.wav"; bad.write_bytes(b"not a wav file")
        profile = profile_files([good, bad])
        assert profile.total_files == 2
        assert profile.decodable_files == 1
        assert pytest.approx(profile.decode_failure_rate) == 0.5

    def test_sample_limit_caps_walk(self, tmp_path: Path) -> None:
        paths = []
        for i in range(20):
            p = tmp_path / f"f{i}.wav"
            _write_silent_wav(p, seconds=0.1, sample_rate=16000)
            paths.append(p)
        profile = profile_files(paths, sample_limit=5)
        assert profile.total_files == 5


class TestProfileSource:
    def test_returns_dataset_card_with_profile(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_silent_wav(tmp_path / f"x{i}.wav", seconds=1.5, sample_rate=16000)
        card = profile_source(SourceSpec(kind="directory", uri=str(tmp_path)))
        assert card.profile.total_files == 3
        assert card.profile.decodable_files == 3
        assert card.uri == str(tmp_path)
