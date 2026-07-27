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

"""The agent's eyes: DataProfiler (input data) and EnvProbe (the machine).

Both are deterministic, cheap, and read-only. They feed the confirm gate (scale),
the pre-flight in ``validate`` (deps/GPU/paths/SR), and the PlanningContext. They
introduce no memory/learning — every call re-reads current reality.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from typing import Any

from nemo_curator.audio_agent.contracts import DataProfile, EnvProfile

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac")
_TRANSCRIPT_KEYS = ("text", "text_ref", "reference_text", "transcript", "pred_text")
# Cap how many files we probe so profiling a huge corpus stays fast (sampling).
_MAX_PROBE = 256


# --------------------------------------------------------------------------- #
# Data profiling
# --------------------------------------------------------------------------- #
def profile_data(source: str, *, audio_filepath_key: str = "audio_filepath", max_probe: int = _MAX_PROBE) -> DataProfile:
    """Profile a manifest (JSONL) or a folder of audio files.

    Reads sample rates / channels / durations / codecs from up to ``max_probe``
    files (via ``soundfile``), detects transcript presence, and records any
    unreadable files. Never raises on bad input — problems become ``notes`` /
    ``unreadable`` entries so the agent can triage.
    """
    prof = DataProfile(source=source)
    expanded = os.path.expanduser(str(source))

    if os.path.isdir(expanded):
        prof.kind = "folder"
        paths = _list_audio_files(expanded)
        prof.num_files = len(paths)
        _probe_files(paths[:max_probe], prof)
    elif expanded.endswith((".jsonl", ".json")) and os.path.isfile(expanded):
        prof.kind = "manifest"
        _profile_manifest(expanded, prof, audio_filepath_key=audio_filepath_key, max_probe=max_probe)
    elif os.path.isfile(expanded) and expanded.endswith(_AUDIO_EXTS):
        prof.kind = "folder"
        prof.num_files = 1
        _probe_files([expanded], prof)
    else:
        prof.notes.append(f"could not interpret source {source!r} as a manifest or an audio folder")

    if prof.num_files:
        prof.mean_duration_sec = round(prof.total_duration_sec / max(1, _probed_count(prof)), 3)
    return prof


def _list_audio_files(folder: str) -> list[str]:
    out: list[str] = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(_AUDIO_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def _profile_manifest(path: str, prof: DataProfile, *, audio_filepath_key: str, max_probe: int) -> None:
    audio_paths: list[str] = []
    keys: set[str] = set()
    has_transcript_value = False
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                prof.unreadable.append(f"{path}:line{count}")
                continue
            if isinstance(row, dict):
                keys.update(row.keys())
                # A transcript COLUMN existing isn't enough -- an all-empty "text" field
                # would falsely imply transcripts (and e.g. that WER is computable). Require
                # at least one row to carry a non-empty transcript value.
                if not has_transcript_value:
                    has_transcript_value = any(str(row.get(k) or "").strip() for k in _TRANSCRIPT_KEYS)
                ap = row.get(audio_filepath_key)
                if ap and len(audio_paths) < max_probe:
                    audio_paths.append(os.path.expanduser(str(ap)))
    prof.num_files = count
    prof.manifest_keys = sorted(keys)
    prof.has_transcripts = has_transcript_value
    _probe_files(audio_paths, prof)


def _probe_files(paths: list[str], prof: DataProfile) -> None:
    try:
        import soundfile as sf
    except Exception:  # noqa: BLE001 - soundfile is an audio-extra dep
        prof.notes.append("soundfile not installed; cannot read audio headers (install an audio extra)")
        return
    srs: Counter[int] = Counter()
    chans: Counter[int] = Counter()
    codecs: Counter[str] = Counter()
    for p in paths:
        if not os.path.exists(p):
            prof.unreadable.append(p)
            continue
        try:
            info = sf.info(p)
            srs[int(info.samplerate)] += 1
            chans[int(info.channels)] += 1
            codecs[str(info.format)] += 1
            prof.total_duration_sec += float(info.frames) / float(info.samplerate or 1)
        except Exception:  # noqa: BLE001 - corrupt/unsupported file
            prof.unreadable.append(p)
    prof.sample_rates = dict(srs)
    prof.channels = dict(chans)
    prof.codecs = dict(codecs)


def _probed_count(prof: DataProfile) -> int:
    return sum(prof.sample_rates.values()) or 1


# --------------------------------------------------------------------------- #
# Environment probing
# --------------------------------------------------------------------------- #
_AUDIO_PACKAGES = {
    "soundfile": "soundfile",
    "torchaudio": "torchaudio",
    "silero_vad": "silero-vad",
    "librosa": "librosa",
    "onnxruntime": "onnxruntime",
    "nemo": "nemo_toolkit[asr]",
    "whisperx": "whisperx",
    "pyannote.audio": "pyannote-audio",
    "nemo_text_processing": "nemo_text_processing",
}
_KNOWN_SECRET_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NVIDIA_API_KEY", "AWS_ACCESS_KEY_ID")


def probe_env() -> EnvProfile:
    """Probe GPU / ffmpeg / installed audio deps / secrets / Curator version."""
    env = EnvProfile()
    env.has_ffmpeg = shutil.which("ffmpeg") is not None
    if not env.has_ffmpeg:
        env.notes.append("ffmpeg not on PATH; resample/convert stages will fail")

    _probe_gpu(env)
    _probe_packages(env)
    _probe_resources(env)

    env.available_secrets = [k for k in _KNOWN_SECRET_ENVS if os.environ.get(k)]
    try:
        import nemo_curator

        env.curator_version = getattr(nemo_curator, "__version__", "") or ""
    except Exception:  # noqa: BLE001
        env.curator_version = ""
    _probe_python(env)
    return env


def _probe_python(env: EnvProfile) -> None:
    """Record the interpreter version and whether it satisfies the project's requires-python.

    A version mismatch is a common, hard-to-spot cause of import / CUDA / model failures, so
    surface it up front. Just as important: on a *supported* version this confirms the
    interpreter is fine, so it isn't wrongly blamed when a failure is really elsewhere.
    """
    env.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        from importlib.metadata import metadata

        req = metadata("nemo-curator").get("Requires-Python") or ""
    except Exception:  # noqa: BLE001 - dist metadata may be unavailable (odd installs)
        return
    if not req:
        return
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        env.python_supported = SpecifierSet(req).contains(Version(env.python_version), prereleases=True)
    except Exception:  # noqa: BLE001 - packaging missing / unparseable specifier -> don't guess
        return
    if not env.python_supported:
        env.notes.append(
            f"Python {env.python_version} is OUTSIDE the project's requires-python {req!r} -- "
            "imports or GPU/model stages may fail; use a supported interpreter."
        )


def _probe_gpu(env: EnvProfile) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            env.has_gpu = True
            env.gpu_count = torch.cuda.device_count()
            env.gpu_names = [torch.cuda.get_device_name(i) for i in range(env.gpu_count)]
            try:
                env.gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            except Exception:  # noqa: BLE001 - property lookup can fail on odd drivers
                pass
    except Exception:  # noqa: BLE001 - torch missing or driver issue -> treat as no GPU
        env.notes.append("torch/CUDA not usable; treating as CPU-only")


def _probe_packages(env: EnvProfile) -> None:
    import importlib.util

    installed: list[str] = []
    missing: list[str] = []
    for module, pkg in _AUDIO_PACKAGES.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except Exception:  # noqa: BLE001 - namespace-package edge cases
            found = False
        (installed if found else missing).append(pkg)
    env.installed_extras = sorted(set(installed))
    env.missing_packages = sorted(set(missing))


def _probe_resources(env: EnvProfile) -> None:
    """CPU count, host RAM, and free disk for the resource planner (best-effort)."""
    env.total_cpus = os.cpu_count() or 0
    try:
        import psutil

        env.total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:  # noqa: BLE001 - psutil optional; fall back to /proc/meminfo
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        env.total_ram_gb = round(int(line.split()[1]) / (1024**2), 1)  # kB -> GB
                        break
        except Exception:  # noqa: BLE001 - /proc unavailable (non-Linux)
            pass
    try:
        env.free_disk_gb = round(shutil.disk_usage(os.getcwd()).free / (1024**3), 1)
    except Exception:  # noqa: BLE001 - disk_usage can fail on odd mounts
        pass
