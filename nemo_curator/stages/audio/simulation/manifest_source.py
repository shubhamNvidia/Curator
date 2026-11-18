# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Finite meeting requests over the existing local-manifest source protocol."""

import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, Role, StageContract, StaticHints
from nemo_curator.stages.audio.simulation._runner import validate_config
from nemo_curator.stages.audio.simulation.fastmss import FastMSSManifestReaderStage, FastMSSSimulationStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            msg = f"Request dependencies must be local files without symlinks: {path}"
            raise ValueError(msg)
        if path.is_file():
            digest.update(json.dumps([str(path.relative_to(root)), _file_digest(path)]).encode())
    return digest.hexdigest()


def _inside(value: object, root: Path, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        msg = f"FastMSS request inputs must use absolute local paths: {value!r}"
        raise ValueError(msg)
    path = Path(value).resolve()
    if not path.is_relative_to(root) or not (path.is_dir() if directory else path.is_file()):
        msg = f"FastMSS request dependency must exist inside {root}: {value}"
        raise ValueError(msg)
    return path


def _validate_cut(cut: dict, root: Path) -> None:
    recording = cut.get("recording", {})
    if cut.get("type") != "MonoCut" or recording.get("transforms"):
        msg = "Meeting requests support file-backed MonoCuts without lazy recording transforms"
        raise ValueError(msg)
    sources = recording.get("sources", [])
    if not sources or any(source.get("type") != "file" for source in sources):
        msg = "Meeting requests require local file-backed recordings"
        raise ValueError(msg)
    for source in sources:
        _inside(source.get("source"), root)
    for supervision in cut.get("supervisions", []):
        speaker = supervision.get("speaker")
        if not isinstance(speaker, str) or not re.fullmatch(r"[\w.-]+", speaker) or speaker in {".", ".."}:
            msg = "Meeting request speaker IDs must be single safe filename tokens"
            raise ValueError(msg)


def _validate_meeting_bounds(config: dict) -> None:
    validate_config(config)
    count, duration = config.get("n_meetings"), config.get("duration")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        msg = "Native n_meetings must be a positive integer"
        raise ValueError(msg)
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        msg = "Native duration must be finite positive seconds"
        raise ValueError(msg)


def _validate_native_inputs(config: dict, root: Path) -> None:
    _validate_meeting_bounds(config)
    prefix = config.get("manifest_prefix")
    if not isinstance(prefix, str) or not re.fullmatch(r"[\w.-]+", prefix) or prefix in {".", ".."}:
        msg = "manifest_prefix must be a single filename component"
        raise ValueError(msg)
    manifests = _inside(config.get("manifest_dir"), root, directory=True)
    splits = config.get("dset_splits")
    if not isinstance(splits, list) or not splits or any(not isinstance(split, str) for split in splits):
        msg = "dset_splits must name nonempty local CutSet splits"
        raise ValueError(msg)
    for split in splits:
        path = _inside(str(manifests / f"{prefix}_{split}.jsonl.gz"), root)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                _validate_cut(json.loads(line), root)
    if config.get("add_noise") and config.get("noise_folders") is not None:
        folders = config.get("noise_folders")
        if not isinstance(folders, list) or not folders:
            msg = "File-noise meeting requests require nonempty local noise_folders"
            raise ValueError(msg)
        for folder in folders:
            _inside(folder, root, directory=True)
    if config.get("hmm_fit_transitions_to"):
        _inside(config["hmm_fit_transitions_to"], root)


def prepare_fastmss_requests(
    config_path: str,
    input_root: str,
    fastmss_root: str,
    manifest_path: str,
    *,
    overrides: list[str] | None = None,
) -> str:
    """Freeze one native YAML into one JSON object per requested meeting.

    This explicit preparation writes planning metadata only, without running
    FastMSS or inspecting waveforms. All source recordings, CutSets, optional
    noise and HMM-fit inputs must live beneath ``input_root``. The trusted
    FastMSS checkout and all inputs must be shared with executor workers.
    The original YAML API remains available for other native layouts.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_file, inputs, engine, destination = (
        Path(value).expanduser().resolve() for value in (config_path, input_root, fastmss_root, manifest_path)
    )
    if config_file.suffix != ".yaml" or not config_file.is_file():
        msg = "Expected one existing native .yaml configuration"
        raise ValueError(msg)
    if not inputs.is_dir() or not (engine / "recipes" / "sim.py").is_file() or not (engine / "fastmss").is_dir():
        msg = "Expected an input directory and a trusted FastMSS checkout with recipes/sim.py"
        raise ValueError(msg)
    if destination.is_relative_to(inputs) or destination.is_relative_to(engine) or destination == config_file:
        msg = "Write the request manifest outside input and engine dependency directories"
        raise ValueError(msg)
    previous_root = os.environ.get("FASTMSS_ROOT")
    try:
        os.environ["FASTMSS_ROOT"] = str(engine)
        with initialize_config_dir(config_dir=str(config_file.parent), version_base=None):
            config = OmegaConf.to_container(
                compose(config_name=config_file.stem, overrides=overrides or []), resolve=True
            )
    finally:
        if previous_root is None:
            os.environ.pop("FASTMSS_ROOT", None)
        else:
            os.environ["FASTMSS_ROOT"] = previous_root
    _validate_native_inputs(config, inputs)
    request = {
        "schema_version": 1,
        "config_sha256": _file_digest(config_file),
        "input_sha256": _tree_digest(inputs),
        "engine_sha256": _tree_digest(engine),
        "native_config": config,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        for ordinal in range(config["n_meetings"]):
            stream.write(json.dumps({**request, "meeting_request": ordinal}, allow_nan=False) + "\n")
    return str(destination)


@dataclass
class FastMSSRequestSource(AgentReady, ProcessingStage[EmptyTask, FileGroupTask]):
    """Bind one local request manifest and emit its path as a file-group task.

    Args:
        manifest_path: One local JSONL meeting-request catalog.
    """

    manifest_path: str
    name: str = "fastmss_request_source"
    is_resumable: bool = False
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(gates=Gates(per_row_independent=False))

    def __post_init__(self) -> None:
        path = Path(self.manifest_path).expanduser().resolve()
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            msg = "Expected one local JSONL meeting-request catalog"
            raise ValueError(msg)
        self.manifest_path = str(path)

    def describe(self) -> StageContract:
        return StageContract(
            cardinality="1:1",
            preserves_upstream_keys=False,
            gates=Gates(per_row_independent=False),
        )

    def num_workers(self) -> int:
        return 1

    def process(self, task: EmptyTask) -> list[FileGroupTask]:
        return [
            FileGroupTask(
                dataset_name=task.dataset_name,
                data=[self.manifest_path],
                _metadata=task._metadata.copy(),
                _stage_perf=list(task._stage_perf),
            )
        ]


@dataclass
class FastMSSManifestGenerationStage(AgentReady, ProcessingStage[FileGroupTask, AudioTask]):
    """Generate one whole meeting per frozen local request-manifest record.

    The manifest represents a prefix of the native YAML's meeting requests,
    allowing the existing smoke sampler to cap actual generation. Native
    configuration is resolved before capping, preserving dependent settings.
    Generation remains one bulk native call; requests are not independent jobs.

    Args:
        manifest_path: Local JSONL created by prepare_fastmss_requests.
        config_path: Original local .yaml file bound by the requests.
        input_root: Directory enclosing every native data dependency.
        fastmss_root: Trusted local FastMSS checkout, including recipes and package.
        output_dir: New run directories, disjoint from all input dependencies.
        python_executable: Existing worker Python with audio_simulation installed.
        jobs_per_config: Native child workers and reserved CPUs.
        timeout_seconds: Maximum wall time for the native generation call.
        audio_filepath_key: Output field containing the mixed meeting waveform.
        duration_key: Output field containing whole-meeting seconds.
        text_key: Output field containing joined reference transcripts.
        rttm_filepath_key: Output field containing the diarization reference.
    """

    manifest_path: str
    config_path: str
    input_root: str
    fastmss_root: str
    output_dir: str
    python_executable: str = field(default_factory=lambda: sys.executable)
    jobs_per_config: int = 1
    timeout_seconds: float = 3600.0
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    text_key: str = "text"
    rttm_filepath_key: str = "rttm_filepath"
    name: str = "fastmss_manifest_source"
    is_resumable: bool = False
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {
        "text_key": "reference_text",
        "rttm_filepath_key": "output_path",
    }
    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
        error_policy="fail",
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            msg = "timeout_seconds must be finite positive seconds"
            raise ValueError(msg)
        for key in ("manifest_path", "config_path", "input_root", "fastmss_root", "output_dir"):
            setattr(self, key, str(Path(getattr(self, key)).expanduser().resolve()))
        output = Path(self.output_dir)
        for dependency in (self.manifest_path, self.config_path, self.input_root, self.fastmss_root):
            path = Path(dependency)
            if output.is_relative_to(path) or path.is_relative_to(output):
                msg = "output_dir must be disjoint from every request dependency"
                raise ValueError(msg)
        keys = (self.audio_filepath_key, self.duration_key, self.text_key, self.rttm_filepath_key)
        if any(not isinstance(key, str) or not key.strip() for key in keys) or len(set(keys)) != len(keys):
            msg = "FastMSS output keys must be four distinct nonempty strings"
            raise ValueError(msg)
        self.resources = self._worker().resources
        self._requests()

    def _worker(self, config: dict | None = None, count: int = 1) -> FastMSSSimulationStage:
        return FastMSSSimulationStage(
            output_dir=self.output_dir,
            fastmss_root=self.fastmss_root,
            python_executable=self.python_executable,
            jobs_per_config=self.jobs_per_config,
            timeout_seconds=self.timeout_seconds,
            isolate_work_dirs=True,
            max_meetings=count,
            planned_config=config,
            expected_meetings=count,
            expected_duration=config["duration"] if config else None,
        )

    def _requests(self, manifest_path: str | None = None) -> list[dict]:
        with Path(manifest_path or self.manifest_path).open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not rows:
            return []
        expected = {
            "schema_version": 1,
            "config_sha256": _file_digest(Path(self.config_path)),
            "input_sha256": _tree_digest(Path(self.input_root)),
            "engine_sha256": _tree_digest(Path(self.fastmss_root)),
            "native_config": rows[0].get("native_config") if isinstance(rows[0], dict) else None,
        }
        for ordinal, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or type(row.get("schema_version")) is not int
                or type(row.get("meeting_request")) is not int
                or row != {**expected, "meeting_request": ordinal}
            ):
                msg = "Expected current, contiguous one-meeting requests; prepare the request manifest again"
                raise ValueError(msg)
        config = expected["native_config"]
        if not isinstance(config, dict):
            msg = "Meeting requests must contain a resolved native configuration"
            raise TypeError(msg)
        _validate_native_inputs(config, Path(self.input_root))
        if len(rows) > config["n_meetings"]:
            msg = "Request count exceeds native n_meetings"
            raise ValueError(msg)
        return rows

    def num_workers(self) -> int:
        return 1

    def describe(self) -> StageContract:
        return StageContract(
            writes=IOSpec(
                data_keys=[self.audio_filepath_key, self.duration_key, self.text_key, self.rttm_filepath_key]
            ),
            cardinality="1:N fan-out",
            preserves_upstream_keys=False,
            preserves_upstream_segment_keys=False,
            gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
            error_policy="fail",
        )

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        if len(task.data) != 1:
            msg = "Generation requires exactly one selected request manifest"
            raise ValueError(msg)
        catalog = self._requests()
        requests = self._requests(task.data[0])
        if requests != catalog[: len(requests)]:
            msg = "Selected requests must be a prefix of the configured request catalog"
            raise ValueError(msg)
        if not requests:
            return []
        config = requests[0]["native_config"]
        worker = self._worker(config, len(requests))
        reader = FastMSSManifestReaderStage(
            audio_filepath_key=self.audio_filepath_key,
            duration_key=self.duration_key,
            text_key=self.text_key,
            rttm_filepath_key=self.rttm_filepath_key,
        )
        source = FileGroupTask(
            dataset_name=task.dataset_name,
            data=[self.config_path],
            _metadata=task._metadata.copy(),
            _stage_perf=list(task._stage_perf),
        )
        rows = [row for group in worker.process(source) for row in reader.process(group)]
        if len(rows) != len(requests):
            msg = f"FastMSS generated {len(rows)} meetings for {len(requests)} requests"
            raise RuntimeError(msg)
        return rows
