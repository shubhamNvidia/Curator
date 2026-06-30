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

from __future__ import annotations

import contextlib
import os
import tempfile
from typing import Any, Literal

import soundfile as sf

from nemo_curator.stages.audio.common import ensure_waveform_2d, load_audio_file

InputResidency = Literal["file", "waveform", "auto"]


def resolve_audio(
    item: dict[str, Any],
    *,
    residency: InputResidency = "auto",
    audio_filepath_key: str = "audio_filepath",
    waveform_key: str = "waveform",
    sample_rate_key: str = "sample_rate",
    mono: bool = True,
) -> tuple[Any, int] | None:
    """Return ``(waveform_2d, sample_rate)`` from tensor keys or a file path.

    ``auto`` prefers an existing waveform, then falls back to file loading.
    ``waveform`` never falls back to disk. ``file`` always loads from the
    configured path key.
    """
    waveform = item.get(waveform_key)
    sample_rate = item.get(sample_rate_key)
    if residency != "file" and waveform is not None and sample_rate is not None:
        return ensure_waveform_2d(waveform), int(sample_rate)

    if residency == "waveform":
        return None

    path = item.get(audio_filepath_key)
    if path and os.path.exists(path):
        return load_audio_file(path, mono=mono)
    return None


def _as_soundfile_array(waveform: Any) -> Any:  # noqa: ANN401
    waveform = ensure_waveform_2d(waveform)
    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()
    if getattr(waveform, "ndim", 0) == 2:
        channels, samples = waveform.shape
        if channels == 1:
            return waveform[0]
        if channels < samples:
            return waveform.T
    return waveform


def resolve_audio_path(
    item: dict[str, Any],
    *,
    residency: InputResidency = "auto",
    audio_filepath_key: str = "audio_filepath",
    waveform_key: str = "waveform",
    sample_rate_key: str = "sample_rate",
    temp_dir: str | None = None,
    register_temp: list[str] | None = None,
) -> str | None:
    """Return an audio path, writing a temp WAV when only a waveform exists.

    When a temp WAV is materialized from an in-memory waveform and
    ``register_temp`` is provided, the temp path is appended to that list so the
    caller can delete it after use (see :func:`cleanup_temp_files`). Without
    ``register_temp`` the caller is responsible for cleanup itself.
    """
    path = item.get(audio_filepath_key)
    if residency != "waveform" and path and os.path.exists(path):
        return path

    if residency == "file":
        return None

    waveform = item.get(waveform_key)
    sample_rate = item.get(sample_rate_key)
    if waveform is None or sample_rate is None:
        return None

    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=temp_dir)
    os.close(fd)
    sf.write(tmp, _as_soundfile_array(waveform), int(sample_rate))
    if register_temp is not None:
        register_temp.append(tmp)
    return tmp


def cleanup_temp_files(paths: list[str] | None) -> None:
    """Best-effort removal of temp files created by :func:`resolve_audio_path`."""
    for path in paths or ():
        with contextlib.suppress(OSError):
            os.remove(path)


def produce_audio_filepath(
    item: dict[str, Any],
    new_path: str,
    *,
    key: str = "audio_filepath",
    original_key: str = "original_audio_filepath",
) -> None:
    """Update a canonical audio path while preserving the first prior value."""
    if key in item and original_key not in item:
        item[original_key] = item[key]
    item[key] = new_path
