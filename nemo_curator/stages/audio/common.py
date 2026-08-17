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

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from operator import eq, ge, gt, le, lt, ne
from typing import Any, ClassVar, Literal

import soundfile
import torch
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio._agent_ready import AgentReady, Gates, IOSpec, Role, StageContract, StaticHints
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask


def get_audio_duration(audio_filepath: str) -> float:
    """Get the duration of the audio file in seconds."""
    try:
        info = soundfile.info(audio_filepath)
        return info.frames / info.samplerate
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to get duration for audio file {audio_filepath}: {e}")
        return -1.0


@dataclass
class GetAudioDurationStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Compute audio duration from the file at *audio_filepath_key* and
    store the result under *duration_key*.

    Args:
        audio_filepath_key: Key to get path to wav file.
        duration_key: Key to put audio duration.
        waveform_key: Key for an in-memory waveform tensor.
        sample_rate_key: Key for the in-memory waveform sample rate.
        input_residency: Which input to use — "file" (audio_filepath only; default,
            unchanged), "waveform" (in-memory only), or "auto" (waveform first, file fallback).
    """

    name: str = "GetAudioDurationStage"
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    input_residency: Literal["file", "waveform", "auto"] = "file"

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        import soundfile

        self._soundfile = soundfile

    def inputs(self) -> tuple[list[str], list[str]]:
        if self.input_residency == "waveform":
            return [], [self.waveform_key, self.sample_rate_key]
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.duration_key]

    def describe(self) -> StageContract:
        # Lazy import avoids a module-level cycle (_residency imports from common).
        from nemo_curator.stages.audio._residency import residency_read_specs

        return StageContract(
            reads_one_of=residency_read_specs(
                self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            ),
            writes=IOSpec(data_keys=[self.duration_key]),
            gates=Gates(per_row_independent=True),
        )

    def validate_input(self, task: AudioTask) -> bool:
        """Require the audio source implied by ``input_residency`` (default: file)."""
        data = task.data
        has_waveform = data.get(self.waveform_key) is not None and data.get(self.sample_rate_key) is not None
        has_file = self.audio_filepath_key in data
        if self.input_residency == "waveform":
            return has_waveform
        if self.input_residency == "file":
            return has_file
        return has_waveform or has_file  # auto

    def _resolve_duration(self, data: dict[str, Any]) -> float:
        """Duration from an in-memory waveform (samples / sample_rate) or the file.

        Default (``input_residency="file"``) reads the file exactly as before.
        """
        if self.input_residency != "file":
            waveform = data.get(self.waveform_key)
            sr = data.get(self.sample_rate_key)
            if waveform is not None and sr is not None and int(sr) > 0:
                return ensure_waveform_2d(waveform).shape[-1] / float(sr)
            if self.input_residency == "waveform":
                logger.warning(f"Missing '{self.waveform_key}'+'{self.sample_rate_key}' (input_residency='waveform')")
                return -1.0
        return get_audio_duration(data[self.audio_filepath_key])

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        duration = self._resolve_duration(task.data)
        task.data[self.duration_key] = duration
        self._log_metrics({"process_time": time.perf_counter() - t0, "duration": max(duration, 0.0)})
        return task


class PreserveByValueStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Filter entries by comparing *input_value_key* against *target_value*.

    Returns ``None`` from ``process()`` to drop entries that fail the
    comparison, matching the text-modality filter convention.

    Args:
        input_value_key: The field in the dataset entries to evaluate.
        target_value: The value to compare with.
        operator: Comparison operator (lt, le, eq, ne, ge, gt).
    """

    name: str = "PreserveByValueStage"
    BATCH_ONLY = True  # process() raises; only process_batch is implemented (agent-discovery hint)

    def __init__(
        self,
        input_value_key: str,
        target_value: int | str,
        operator: str = "eq",
    ):
        self.input_value_key = input_value_key
        self.target_value = target_value
        ops = {"lt": lt, "le": le, "eq": eq, "ne": ne, "ge": ge, "gt": gt}
        if operator not in ops:
            msg = f"Operator must be one of: {', '.join(ops)}"
            raise ValueError(msg)
        self.operator = ops[operator]

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.input_value_key]),
            writes=IOSpec(data_keys=[self.input_value_key]),
            cardinality="filter",
            # Compares one row against a fixed target value, so batching changes throughput
            # rather than the verdict -- no row's fate depends on the rows beside it.
            gates=Gates(per_row_independent=True),
        )

    def process(self, task: AudioTask) -> AudioTask | None:
        msg = "PreserveByValueStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        t0 = time.perf_counter()
        results = []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)
            if self.operator(task.data[self.input_value_key], self.target_value):
                results.append(task)
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "input_count": len(tasks),
                "output_count": len(results),
                "filtered_count": len(tasks) - len(results),
            }
        )
        return results


def _row_names_file(row: dict[str, Any], key: str, wanted: set[str]) -> bool:
    """Whether a manifest row's audio path is one of ``wanted`` (absolute-path comparison)."""
    value = row.get(key)
    return isinstance(value, str) and os.path.abspath(os.path.expanduser(value)) in wanted


@dataclass
class ManifestReaderStage(AgentReady, ProcessingStage[FileGroupTask, AudioTask]):
    """Read JSONL manifest files from a FileGroupTask and emit one AudioTask per line.

    Uses line-by-line streaming via fsspec (no Pandas) to keep memory at ~1x file size.
    Supports local and cloud paths (S3, GCS).

    Args:
        include_files: Emit only rows whose audio path (under ``include_files_key``) is one of
            these files, comparing absolute paths. ``None`` -- the default -- reads every row.
            It restricts the same reader over the same manifest rather than pointing a delta
            run at a filtered copy, so the rows a partial run emits are the rows a full run
            would have emitted.
        include_files_key: Which row column holds that path.
    """

    name: str = "manifest_reader_stage"
    include_files: list[str] | None = None
    include_files_key: str = "audio_filepath"
    # Declared statically as well as in describe(): a narrowable source has to be readable as
    # safe-to-narrow without constructing it, which is how the instance-free conformance sweep
    # sees it.
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(lifecycle_side_effects=True, per_row_independent=True)
    )
    # It points at the column holding a row's audio path, which is an existing role rather
    # than a new one -- a filter comparing something else would not be filtering by file.
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {"include_files_key": "audio_filepath"}

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        paths = task.data
        results: list[AudioTask] = []
        count = 0
        wanted = (
            None
            if self.include_files is None
            else {os.path.abspath(os.path.expanduser(p)) for p in self.include_files}
        )
        for manifest in paths:
            fs, resolved = url_to_fs(manifest)
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line.strip())
                        if wanted is not None and not _row_names_file(row, self.include_files_key, wanted):
                            continue
                        results.append(
                            AudioTask(
                                dataset_name=task.dataset_name,
                                data=row,
                                _metadata=task._metadata,
                                _stage_perf=list(task._stage_perf),
                            )
                        )
                        count += 1
            logger.info(f"ManifestReaderStage: loaded {count} entries from {manifest}")
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "manifests_read": len(paths),
                "entries_read": len(results),
            }
        )
        return results

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}

    def num_workers(self) -> int | None:
        return 1

    def describe(self) -> StageContract:
        return StageContract(
            writes=IOSpec(data_keys=["audio_filepath"]),
            cardinality="1:N fan-out",
            gates=Gates(lifecycle_side_effects=True, per_row_independent=True),
        )


@dataclass
class ManifestReader(AgentReady, CompositeStage[EmptyTask, AudioTask]):
    """Composite stage for reading JSONL manifests.

    Decomposes into:
    1. FilePartitioningStage — discovers and partitions manifest files
    2. ManifestReaderStage — reads each partition line-by-line (no Pandas)

    Args:
        manifest_path: Path or list of paths to JSONL manifests (local or cloud).
        files_per_partition: Number of manifest files per partition. Defaults to 1.
        blocksize: Target size per partition (e.g., "100MB"). Ignored if files_per_partition is set.
        file_extensions: File extensions to filter. Defaults to [".jsonl", ".json"].
        storage_options: Storage options for cloud paths (S3, GCS credentials, endpoints).
        include_files: Read only the rows naming these audio files (see ``ManifestReaderStage``).
        include_files_key: Which row column holds the audio path.
    """

    manifest_path: str | list[str]
    name: str = "manifest_reader"
    files_per_partition: int | None = 1
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".jsonl", ".json"])
    storage_options: dict[str, Any] | None = None
    include_files: list[str] | None = None
    include_files_key: str = "audio_filepath"
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(gates=Gates(per_row_independent=True))
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {"include_files_key": "audio_filepath"}

    def __post_init__(self) -> None:
        super().__init__()
        if not self.manifest_path:
            msg = "manifest_path is required for ManifestReader"
            raise ValueError(msg)

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.manifest_path,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.storage_options,
            ),
            ManifestReaderStage(
                include_files=self.include_files,
                include_files_key=self.include_files_key,
            ),
        ]

    def get_description(self) -> str:
        parts = [f"Read JSONL manifests from {self.manifest_path}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        return ", ".join(parts)

    def describe(self) -> StageContract:
        return StageContract(
            cardinality="1:N fan-out",
            wrappable=False,
            gates=Gates(per_row_independent=True),
        )


@dataclass
class CreateInitialManifestAudioFolderStage(AgentReady, ProcessingStage[EmptyTask, AudioTask]):
    """Create an initial manifest from any local folder of audio files.

    Recursively scans ``data_dir`` for audio files and emits one AudioTask per file with its
    path under ``audio_filepath_key`` (plus a filename-derived ``audio_item_id``). A generic,
    dataset-agnostic source: no download, no transcripts, and no dataset-specific filename
    parsing -- unlike ``CreateInitialManifest{ReadSpeech,Fleurs}Stage``. Use it to start a
    pipeline from a plain folder of WAV/FLAC/MP3/... when there is no JSONL manifest (use
    ``ManifestReader`` when a manifest already exists).

    Args:
        data_dir: Local folder to scan for audio files.
        extensions: Audio file extensions to include (case-insensitive).
        recursive: Recurse into subfolders (default True).
        max_samples: Maximum number of files to include (-1 for all).
        include_files: Process only these files (absolute paths), skipping the rest of the
            folder. ``None`` -- the default -- means the whole folder, exactly as before.
            Restricting the file list rather than swapping in a different source stage is what
            lets a delta run over new files produce rows identical to a full run's.
    """

    data_dir: str
    extensions: list[str] = field(default_factory=lambda: [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"])
    recursive: bool = True
    max_samples: int = -1
    include_files: list[str] | None = None
    audio_filepath_key: str = "audio_filepath"
    audio_item_id_key: str = "audio_item_id"
    name: str = "CreateInitialManifestAudioFolder"
    batch_size: int = 1
    # See ManifestReaderStage: the narrowing claim has to survive being read off the class.
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(gates=Gates(per_row_independent=True))

    def __post_init__(self) -> None:
        super().__init__()
        if not self.data_dir:
            msg = "data_dir is required for CreateInitialManifestAudioFolderStage"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.audio_item_id_key]

    def describe(self) -> StageContract:
        return StageContract(
            # No ``produces``: the audio already exists on disk, this stage only points at it
            # (unlike the dataset CreateInitialManifest*Stage sources, which download and write).
            writes=IOSpec(data_keys=[self.audio_filepath_key, self.audio_item_id_key]),
            cardinality="1:N fan-out",
            # scans existing files -- no download, no disk writes; one task per file, and a
            # file's row says nothing about the other files in the folder. Except under
            # ``max_samples``, which truncates the SORTED list, so which files survive is a fact
            # about the whole folder rather than about any one of them.
            gates=Gates(per_row_independent=self.max_samples is None or self.max_samples < 0),
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}

    def num_workers(self) -> int | None:
        return 1

    def _collect_audio_files(self) -> list[str]:
        exts = tuple((e if e.startswith(".") else f".{e}").lower() for e in self.extensions)
        if not os.path.isdir(self.data_dir):
            logger.error(f"[{self.name}] data_dir not found: {self.data_dir}")
            return []
        found: list[str] = []
        if self.recursive:
            for root, _dirs, files in os.walk(self.data_dir):
                found.extend(os.path.join(root, f) for f in files if f.lower().endswith(exts))
        else:
            found = [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.lower().endswith(exts) and os.path.isfile(os.path.join(self.data_dir, f))
            ]
        if self.include_files is not None:
            wanted = {os.path.abspath(os.path.expanduser(p)) for p in self.include_files}
            found = [p for p in found if os.path.abspath(p) in wanted]
            missing = wanted - {os.path.abspath(p) for p in found}
            if missing:
                # Named rather than silently skipped: a caller that asked for specific files and
                # got fewer would otherwise read the short result as "those files held nothing".
                logger.warning(f"[{self.name}] include_files named {len(missing)} file(s) not found under {self.data_dir}")
        return sorted(found)

    def process(self, _: EmptyTask) -> list[AudioTask]:
        """Emit one AudioTask per audio file found under ``data_dir``."""
        paths = self._collect_audio_files()
        if self.max_samples is not None and self.max_samples >= 0:
            paths = paths[: self.max_samples]
        if not paths:
            logger.warning(f"[{self.name}] no audio files {self.extensions} under {self.data_dir}")
            return []
        tasks: list[AudioTask] = []
        for path in paths:
            abspath = os.path.abspath(path)
            # Relpath, not basename: ``recursive`` defaults True and speaker-per-folder is the
            # standard layout, so a basename id gives spk1/utt1.wav and spk2/utt1.wav the same
            # id -- and downstream that id becomes an output filename. A flat corpus is
            # unaffected. Not injective: a flat ``spk1__utt1.wav`` still aliases spk1/utt1.wav.
            rel = os.path.relpath(abspath, os.path.abspath(self.data_dir))
            item_id = os.path.splitext(rel)[0].replace(os.sep, "__")
            tasks.append(
                AudioTask(
                    dataset_name="local-audio-folder",
                    data={self.audio_filepath_key: abspath, self.audio_item_id_key: item_id},
                    filepath_key=self.audio_filepath_key,
                )
            )
        logger.info(f"[{self.name}] created {len(tasks)} AudioTask(s) from {self.data_dir}")
        return tasks


@dataclass
class ManifestWriterStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Append a single AudioTask to a JSONL manifest file.

    The output file is truncated once in ``setup()`` (called on the driver)
    so repeated pipeline runs produce a clean output.  ``setup_on_node()``
    only creates the parent directory -- it never truncates, so multi-node
    deployments do not erase each other's data.

    .. note::
       Because all nodes append to the same path, callers in multi-node
       setups should either use a shared filesystem or provide a
       node-unique ``output_path``.

    Supports local and cloud paths via fsspec.

    Args:
        output_path: Destination JSONL path (local or cloud).
    """

    output_path: str
    name: str = "manifest_writer"

    def __post_init__(self) -> None:
        if not self.output_path:
            msg = "output_path is required for ManifestWriterStage"
            raise ValueError(msg)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Truncate the output file once on the driver before processing starts."""
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)
        with self._fs.open(self._path, "w", encoding="utf-8"):
            pass
        logger.info(f"ManifestWriterStage: writing to {self.output_path}")

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Ensure parent directory exists on each node (no truncation)."""
        self._fs, self._path = url_to_fs(self.output_path)
        parent_dir = "/".join(self._path.split("/")[:-1])
        if parent_dir:
            self._fs.makedirs(parent_dir, exist_ok=True)

    def process(self, task: AudioTask) -> AudioTask:
        with self._fs.open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(task.data, ensure_ascii=False) + "\n")
        return AudioTask(
            dataset_name=task.dataset_name,
            data=task.data,
            _metadata=task._metadata,
            _stage_perf=list(task._stage_perf),
        )

    def num_workers(self) -> int | None:
        return 1

    def describe(self) -> StageContract:
        return StageContract(
            gates=Gates(
                writes_to_disk=True,
                output_path_params=["output_path"],
                lifecycle_side_effects=True,
                # Serializes task.data as-is via json.dumps; a resident tensor
                # (e.g. a waveform) will crash it. Stop carrying the tensor
                # before this AudioTask sink, or convert to a DocumentBatch and
                # use DocumentBatchJsonlWriterStage instead.
                requires_serializable_input=True,
                # Appends each row as it arrives; a row's line is its own contents.
                per_row_independent=True,
            ),
        )


def load_audio_file(audio_path: str, mono: bool = True) -> tuple[torch.Tensor, int]:
    """Load audio file and return waveform tensor (channels, samples) and sample rate."""
    data, sample_rate = soundfile.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data)
    waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform.T
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def ensure_waveform_2d(waveform: Any) -> torch.Tensor:  # noqa: ANN401
    """Ensure waveform is a torch.Tensor in 2D (channels, samples) format."""
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert multi-channel waveform to mono. Assumes 2D (channels, samples) input."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def resolve_waveform_from_item(
    item: dict[str, Any], task_id: str, mono: bool = True
) -> tuple[torch.Tensor, int] | None:
    """
    Resolve (waveform, sample_rate) from an item dict, loading from file if needed.

    Checks item['waveform'] + item['sample_rate'], falls back to loading from
    item['audio_filepath'], resolves missing sample_rate from file header.
    Updates item in-place when loading from file.
    Returns None if resolution fails.

    .. note::
       The canonical resolver is :func:`nemo_curator.stages.audio._residency.resolve_audio`.
       This helper is retained for its unique behavior — reading ``sample_rate`` from the
       file header *without* reloading an already-present waveform, and writing the loaded
       waveform/sample_rate back into ``item`` — which ``resolve_audio`` does not replicate.
       Prefer ``resolve_audio`` in new code.
    """
    waveform = item.get("waveform")
    sample_rate = item.get("sample_rate")

    if waveform is None:
        audio_filepath = item.get("audio_filepath")
        if audio_filepath and os.path.exists(audio_filepath):
            try:
                waveform, sample_rate = load_audio_file(audio_filepath, mono=mono)
                item["waveform"] = waveform
                item["sample_rate"] = sample_rate
            except (OSError, RuntimeError, soundfile.SoundFileError) as e:
                logger.error(f"[{task_id}] Failed to load audio file: {e}")
                return None
        else:
            logger.warning(f"[{task_id}] No waveform or valid audio_filepath found")
            return None
    elif sample_rate is None:
        audio_filepath = item.get("audio_filepath")
        if audio_filepath and os.path.exists(audio_filepath):
            try:
                info = soundfile.info(audio_filepath)
                sample_rate = info.samplerate
                item["sample_rate"] = sample_rate
            except (OSError, RuntimeError, soundfile.SoundFileError) as e:
                logger.error(
                    f"[{task_id}] Waveform present but sample_rate missing "
                    f"and could not read from '{audio_filepath}': {e}"
                )
                return None
        else:
            logger.error(f"[{task_id}] Waveform present but 'sample_rate' missing and no audio_filepath available.")
            return None

    waveform = ensure_waveform_2d(waveform)
    if mono:
        waveform = ensure_mono(waveform)

    return waveform, sample_rate


def resolve_model_path(model_path: str, reference_file: str, module_subdir: str) -> str:
    """Resolve a relative model path using the reference file's directory and module subdirectory."""
    if os.path.isabs(model_path):
        return model_path
    current_dir = os.path.dirname(os.path.abspath(reference_file))
    module_dir = os.path.join(current_dir, module_subdir)
    for base in (module_dir, current_dir):
        resolved = os.path.join(base, model_path)
        if os.path.exists(resolved):
            return resolved
    return os.path.join(module_dir, model_path)
