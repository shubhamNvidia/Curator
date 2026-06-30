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

"""Characterization tests pinning the exact behavior of the audio-resolution
helpers BEFORE consolidating them onto ``_residency.resolve_audio``.

These lock the current return shapes/types/values so the refactor of
``utmos._load_waveform_tensor`` and ``sigmos._get_audio_numpy_sr`` into thin
wrappers around ``resolve_audio`` is provably byte-identical (no behavior change).
``common.resolve_waveform_from_item`` is included to document its unique
sample-rate-from-header behavior (kept as-is for now).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from nemo_curator.stages.audio._residency import resolve_audio
from nemo_curator.stages.audio.common import resolve_waveform_from_item
from nemo_curator.stages.audio.filtering.sigmos import _get_audio_numpy_sr
from nemo_curator.stages.audio.filtering.utmos import _load_waveform_tensor

_SR = 16000


def _wav(path: Path, *, channels: int = 1, n: int = 1600) -> str:
    data = np.linspace(-0.5, 0.5, n, dtype="float32")
    arr = data if channels == 1 else np.stack([data, data * 0.5], axis=1)  # (n, ch) for soundfile
    sf.write(str(path), arr, _SR)
    return str(path)


# --------------------------------------------------------------------------- #
# utmos._load_waveform_tensor  ->  torch (1, N) mono
# --------------------------------------------------------------------------- #
def test_utmos_waveform_stereo_tensor_to_mono():
    wf = torch.ones(2, 1600)
    out = _load_waveform_tensor({"waveform": wf, "sample_rate": _SR}, "t")
    assert out is not None
    t, sr = out
    assert torch.is_tensor(t) and t.shape == (1, 1600) and sr == _SR


def test_utmos_waveform_numpy_1d_to_mono():
    out = _load_waveform_tensor({"waveform": np.ones(1600, dtype="float32"), "sample_rate": _SR}, "t")
    t, sr = out
    assert torch.is_tensor(t) and t.shape == (1, 1600)


def test_utmos_waveform_present_no_sr_returns_none():
    assert _load_waveform_tensor({"waveform": torch.ones(1, 1600)}, "t") is None


def test_utmos_file_path_loads_mono(tmp_path: Path):
    out = _load_waveform_tensor({"audio_filepath": _wav(tmp_path / "u.wav", channels=2)}, "t")
    t, sr = out
    assert t.shape[0] == 1 and sr == _SR


def test_utmos_residency_waveform_no_data_returns_none(tmp_path: Path):
    out = _load_waveform_tensor({"audio_filepath": _wav(tmp_path / "u2.wav")}, "t", input_residency="waveform")
    assert out is None


def test_utmos_missing_returns_none():
    assert _load_waveform_tensor({}, "t") is None


# --------------------------------------------------------------------------- #
# sigmos._get_audio_numpy_sr  ->  np.float32 1D mono
# --------------------------------------------------------------------------- #
def test_sigmos_waveform_stereo_tensor_to_mono_numpy():
    out = _get_audio_numpy_sr({"waveform": torch.ones(2, 1600), "sample_rate": _SR}, "t")
    a, sr = out
    assert isinstance(a, np.ndarray) and a.dtype == np.float32 and a.ndim == 1 and a.shape == (1600,) and sr == _SR


def test_sigmos_waveform_numpy_to_mono():
    out = _get_audio_numpy_sr({"waveform": np.ones((2, 1600), dtype="float32"), "sample_rate": _SR}, "t")
    a, _ = out
    assert a.ndim == 1 and a.shape == (1600,)


def test_sigmos_file_path_loads_1d_numpy(tmp_path: Path):
    out = _get_audio_numpy_sr({"audio_filepath": _wav(tmp_path / "s.wav", channels=2)}, "t")
    a, sr = out
    assert isinstance(a, np.ndarray) and a.ndim == 1 and sr == _SR


def test_sigmos_waveform_present_no_sr_falls_back_to_file(tmp_path: Path):
    # sigmos requires BOTH wf+sr for the in-memory branch; otherwise tries the file.
    out = _get_audio_numpy_sr(
        {"waveform": torch.ones(1, 1600), "audio_filepath": _wav(tmp_path / "s2.wav")}, "t"
    )
    assert out is not None and out[0].ndim == 1


def test_sigmos_residency_waveform_no_data_returns_none(tmp_path: Path):
    out = _get_audio_numpy_sr({"audio_filepath": _wav(tmp_path / "s3.wav")}, "t", input_residency="waveform")
    assert out is None


def test_sigmos_missing_returns_none():
    assert _get_audio_numpy_sr({}, "t") is None


# --------------------------------------------------------------------------- #
# resolve_audio (unchanged) — pin its current contract for reference
# --------------------------------------------------------------------------- #
def test_resolve_audio_waveform_branch_does_not_force_mono():
    # resolve_audio keeps channels on the in-memory branch (mono only applies on file load).
    out = resolve_audio({"waveform": torch.ones(2, 1600), "sample_rate": _SR})
    t, sr = out
    assert t.shape == (2, 1600) and sr == _SR


def test_resolve_audio_file_branch_applies_mono(tmp_path: Path):
    out = resolve_audio({"audio_filepath": _wav(tmp_path / "r.wav", channels=2)}, mono=True)
    t, _ = out
    assert t.shape[0] == 1


# --------------------------------------------------------------------------- #
# common.resolve_waveform_from_item — unique sr-from-header behavior
# --------------------------------------------------------------------------- #
def test_common_reads_sr_from_header_without_reloading_waveform(tmp_path: Path):
    wav = _wav(tmp_path / "c.wav")
    provided = torch.ones(1, 1600)
    item = {"waveform": provided, "audio_filepath": wav}  # waveform present, sample_rate MISSING
    out = resolve_waveform_from_item(item, "t")
    t, sr = out
    assert sr == _SR  # read from the file header
    assert t.shape == (1, 1600)  # kept the provided waveform (not reloaded from file)
    assert item["sample_rate"] == _SR  # written back into the item
