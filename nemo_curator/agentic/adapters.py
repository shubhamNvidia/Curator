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
"""URI / source adapters for the agentic core.

Resolves a :class:`SourceSpec` into:

1. A concrete reader stage (``StageRef``) that the compiler can prepend to the
   pipeline. Today the agentic core supports the four catalog sources:

   - ``ManifestReader``                              (``manifest://`` / local file)
   - ``CreateInitialManifestFleursStage``            (``fleurs://`` / kind=fleurs)
   - ``CreateInitialManifestReadSpeechStage``        (``readspeech://`` / kind=readspeech)
   - ``ManifestReader`` over an auto-built JSONL     (``directory://`` / kind=directory)

2. A profile fixture (list of file paths or manifest entries) suitable for
   the Layer 1 profiler and the dry-run sampler.

The adapter never touches the network at *plan* time. ``probe_source`` is a
fast, file-system / metadata-only operation that returns enough to feed
:class:`DatasetProfile` later.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from nemo_curator.agentic.ir import SourceSpec, StageRef

# Audio extensions we recognize for directory probes
DEFAULT_AUDIO_EXTS: tuple[str, ...] = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".m4a")


@dataclass
class SourceProbe:
    """Light-weight characterization of a source for the profiler & dry-run."""

    kind: str
    resolved_uri: str
    discovered_files: list[str]
    manifest_lines: list[dict] | None = None
    total_estimated: int | None = None


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def normalize_source(source: SourceSpec) -> SourceSpec:
    """Apply URI-scheme detection and fill in ``kind`` if the user only gave a URI."""

    if source.kind != "manifest":
        return source

    uri = source.uri
    parsed = urlparse(uri)
    scheme = (parsed.scheme or "file").lower()

    if scheme == "hf":
        # hf://<org>/<dataset> — naïve mapping; the agent prefers FLEURS / RS today.
        target_kind = "hf"
    elif scheme in {"file", ""} and Path(uri).is_dir():
        target_kind = "directory"
    elif scheme == "fleurs":
        target_kind = "fleurs"
    elif scheme == "readspeech":
        target_kind = "readspeech"
    elif uri.endswith(".jsonl") or uri.endswith(".json"):
        target_kind = "manifest"
    elif scheme in {"file", "", "s3", "gs"}:
        target_kind = "manifest"
    else:
        target_kind = source.kind

    if target_kind == source.kind:
        return source

    return source.model_copy(update={"kind": target_kind})


def probe_source(source: SourceSpec, *, sample_n: int = 4) -> SourceProbe:
    """Quickly probe ``source`` enough to feed the profiler / dry-run."""

    src = normalize_source(source)

    if src.kind == "directory":
        return _probe_directory(src, sample_n=sample_n)
    if src.kind == "manifest":
        return _probe_manifest(src, sample_n=sample_n)
    if src.kind in {"fleurs", "readspeech", "hf"}:
        return SourceProbe(
            kind=src.kind,
            resolved_uri=src.uri,
            discovered_files=[],
            total_estimated=None,
        )

    msg = f"Unsupported source kind: {src.kind!r}"
    raise ValueError(msg)


def reader_stage(source: SourceSpec) -> StageRef:
    """Return the :class:`StageRef` the compiler prepends to satisfy this source."""

    src = normalize_source(source)
    opts = dict(src.options or {})

    if src.kind == "manifest":
        return StageRef(
            stage="ManifestReader",
            params={
                "manifest_path": src.uri,
                "files_per_partition": opts.get("files_per_partition", 1),
                "file_extensions": src.file_extensions or [".jsonl", ".json"],
                "storage_options": src.storage_options,
            },
            auto_inserted=True,
            insert_reason="source=manifest",
        )

    if src.kind == "directory":
        manifest = opts.get("synthesized_manifest_path")
        if manifest is None:
            msg = (
                "directory source must be preprocessed into a JSONL manifest before compiling; "
                "call build_manifest_from_directory() first."
            )
            raise ValueError(msg)
        return StageRef(
            stage="ManifestReader",
            params={"manifest_path": str(manifest), "files_per_partition": 1},
            auto_inserted=True,
            insert_reason="source=directory (synthesized manifest)",
        )

    if src.kind == "fleurs":
        return StageRef(
            stage="CreateInitialManifestFleursStage",
            params={
                "lang": opts.get("lang", ""),
                "split": opts.get("split", ""),
                "raw_data_dir": opts.get("raw_data_dir", ""),
            },
            auto_inserted=True,
            insert_reason="source=fleurs",
        )

    if src.kind == "readspeech":
        return StageRef(
            stage="CreateInitialManifestReadSpeechStage",
            params={
                "max_samples": opts.get("max_samples", 5000),
                "auto_download": opts.get("auto_download", True),
            },
            auto_inserted=True,
            insert_reason="source=readspeech",
        )

    msg = f"Cannot create reader stage for source kind={src.kind!r}"
    raise ValueError(msg)


def build_manifest_from_directory(
    directory: str | Path,
    *,
    extensions: Iterable[str] = DEFAULT_AUDIO_EXTS,
    output_path: str | Path | None = None,
) -> Path:
    """Walk a directory, find audio files, emit a JSONL manifest and return its path.

    Used by the runner when the user supplies ``directory://`` as their source.
    """

    d = Path(directory).expanduser().resolve()
    if not d.is_dir():
        msg = f"Not a directory: {d}"
        raise NotADirectoryError(msg)
    exts = {e.lower() for e in extensions}

    if output_path is None:
        output_path = d / ".curator-adv-source.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for root, _, files in os.walk(d):
            for fname in sorted(files):
                if Path(fname).suffix.lower() in exts:
                    entry = {"audio_filepath": str(Path(root) / fname)}
                    f.write(json.dumps(entry) + "\n")
                    count += 1
    if count == 0:
        msg = f"No audio files (extensions={sorted(exts)}) found under {d}"
        raise FileNotFoundError(msg)
    return output_path


# ----------------------------------------------------------------------------
# Internal probe helpers
# ----------------------------------------------------------------------------


def _probe_directory(source: SourceSpec, *, sample_n: int) -> SourceProbe:
    exts = tuple(e.lower() for e in (source.file_extensions or DEFAULT_AUDIO_EXTS))
    found: list[str] = []
    for root, _, files in os.walk(source.uri):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in exts:
                found.append(str(Path(root) / fname))
                if len(found) >= 10_000:
                    break
        if len(found) >= 10_000:
            break
    return SourceProbe(
        kind="directory",
        resolved_uri=source.uri,
        discovered_files=found[:sample_n] if sample_n else found,
        total_estimated=len(found),
    )


def _probe_manifest(source: SourceSpec, *, sample_n: int) -> SourceProbe:
    """Stream the first ``sample_n`` JSONL lines and discover audio_filepath entries."""

    lines: list[dict] = []
    files: list[str] = []
    p = Path(source.uri)
    if not p.exists():
        return SourceProbe(kind="manifest", resolved_uri=source.uri, discovered_files=[], total_estimated=None)
    total = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            if len(lines) < sample_n:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lines.append(obj)
                ap = obj.get("audio_filepath") or obj.get("filepath") or obj.get("audio_path")
                if ap and Path(ap).exists():
                    files.append(ap)
    return SourceProbe(
        kind="manifest",
        resolved_uri=source.uri,
        discovered_files=files,
        manifest_lines=lines,
        total_estimated=total,
    )


__all__ = [
    "DEFAULT_AUDIO_EXTS",
    "SourceProbe",
    "build_manifest_from_directory",
    "normalize_source",
    "probe_source",
    "reader_stage",
]
