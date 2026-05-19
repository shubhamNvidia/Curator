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
"""Layer 1 profiler — deterministic, LLM-free dataset characterization.

The profiler does **only** the cheap, deterministic checks the agent needs to
decide whether the input dataset is reasonable to run a pipeline against. It
deliberately omits language-ID and any acoustic ML inference; that surface
lands in a later phase together with the corresponding catalog stages.

Outputs a :class:`DatasetCard` populated with a :class:`DatasetProfile` and a
list of human-friendly findings. The validator and runner read this card; the
critic reads the post-run version of it for drift detection.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from loguru import logger

from nemo_curator.agentic.adapters import SourceProbe
from nemo_curator.agentic.cards import DatasetCard, DatasetProfile, LicenseKind
from nemo_curator.agentic.ir import SourceSpec

# Public default — caller can override.
DEFAULT_SAMPLE_LIMIT = 64


def profile_files(
    files: Iterable[str | Path],
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> DatasetProfile:
    """Walk up to ``sample_limit`` files and return a :class:`DatasetProfile`."""

    try:
        import soundfile  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - soundfile is in audio_common
        msg = "soundfile is required for the profiler; install nemo_curator[audio_cpu]."
        raise RuntimeError(msg) from exc

    profile = DatasetProfile()
    durations: list[float] = []
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    formats: Counter[str] = Counter()
    total = 0
    decodable = 0

    for f in files:
        if total >= sample_limit:
            break
        total += 1
        path = Path(f)
        suffix = path.suffix.lower().lstrip(".") or "unknown"
        formats[suffix] += 1
        try:
            info = soundfile.info(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=False).debug(f"profile: failed to decode {path}: {exc}")
            continue
        decodable += 1
        if info.samplerate:
            sample_rates[int(info.samplerate)] += 1
        if info.channels:
            channels[int(info.channels)] += 1
        if info.samplerate and info.frames:
            durations.append(info.frames / info.samplerate)

    profile.total_files = total
    profile.decodable_files = decodable
    profile.decode_failure_rate = (total - decodable) / total if total else 0.0
    profile.formats = {k: v for k, v in formats.most_common()}
    profile.sample_rates_hz = {str(k): v for k, v in sample_rates.most_common()}
    profile.channel_distribution = {("mono" if k == 1 else "stereo" if k == 2 else f"{k}ch"): v for k, v in channels.most_common()}

    if durations:
        durations.sort()
        profile.duration_p05_sec = _percentile(durations, 0.05)
        profile.duration_p50_sec = _percentile(durations, 0.50)
        profile.duration_p95_sec = _percentile(durations, 0.95)
        profile.total_duration_hours = round(sum(durations) / 3600.0, 4)

    profile.notes = list(_emit_notes(profile, sample_rates, channels))
    return profile


def profile_source(
    source: SourceSpec,
    *,
    probe: SourceProbe | None = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> DatasetCard:
    """Profile a :class:`SourceSpec` and return a fully-populated :class:`DatasetCard`."""

    from nemo_curator.agentic.adapters import probe_source  # noqa: PLC0415

    if probe is None:
        probe = probe_source(source, sample_n=sample_limit)

    profile = profile_files(probe.discovered_files, sample_limit=sample_limit)

    return DatasetCard(
        name=Path(source.uri).name or source.uri,
        uri=source.uri,
        uri_scheme=_scheme_for(source),
        description=f"Auto-profile of {source.uri} (kind={source.kind}).",
        license=LicenseKind.UNKNOWN,
        profile=profile,
        created_at=datetime.now(timezone.utc),
        sample_count_for_profile=probe.total_estimated or profile.total_files,
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = q * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo))


def _scheme_for(source: SourceSpec) -> str:
    if source.kind in ("manifest", "directory"):
        return "manifest" if source.kind == "manifest" else "file"
    if source.kind == "hf":
        return "hf"
    if source.kind in ("fleurs", "readspeech"):
        return "file"
    return "file"


def _emit_notes(
    profile: DatasetProfile,
    sample_rates: Counter[int],
    channels: Counter[int],
) -> Iterable[str]:
    if profile.total_files == 0:
        yield "No files discovered in the sample window — the source may be empty or inaccessible."
        return
    if profile.decode_failure_rate > 0.0:
        pct = round(profile.decode_failure_rate * 100, 1)
        yield f"{pct}% of sampled files failed to decode."
    if len(sample_rates) > 1:
        yield (
            "Mixed sample rates detected; the validator will auto-insert MonoConversionStage / "
            "ResampleAudioStage as needed for downstream stages."
        )
    if any(c > 1 for c in channels):
        yield "Multi-channel files detected; downstream stages expect mono — MonoConversionStage will be auto-inserted."
    if profile.duration_p95_sec and profile.duration_p95_sec > 600.0:
        yield "Some files exceed 10 minutes; consider SplitLongAudioStage if downstream stages are GPU-bound."
    if profile.duration_p05_sec is not None and profile.duration_p05_sec < 0.5:
        yield "Some files are under 0.5s; they will likely be dropped by VAD."


__all__ = ["DEFAULT_SAMPLE_LIMIT", "profile_files", "profile_source"]
