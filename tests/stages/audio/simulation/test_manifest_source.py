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

"""Test real FastMSS requests, dependency admission and downstream composition."""

import gzip
import json
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.stages.audio.common import GetAudioDurationStage, ManifestWriterStage, PreserveByValueStage
from nemo_curator.stages.audio.simulation import (
    FastMSSManifestGenerationStage,
    FastMSSManifestReaderStage,
    FastMSSRequestSource,
    FastMSSSimulationStage,
    prepare_fastmss_requests,
)
from nemo_curator.tasks import EmptyTask, FileGroupTask


def build_request_fixture(root: Path) -> dict[str, str]:
    """Create aligned synthetic signals and use the actual installed engine files."""
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet, MonoCut, Recording
    from lhotse.supervision import AlignmentItem, SupervisionSegment
    from omegaconf import OmegaConf

    package = distribution("fastmss")
    engine = root / "engine"
    for item in package.files or []:
        if item.parts[0] not in {"fastmss", "recipes"} or "__pycache__" in item.parts:
            continue
        target = engine / str(item)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package.locate_file(item), target)
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    cuts = []
    for index, frequency in enumerate((220, 330)):
        audio = inputs / f"speaker-{index}.wav"
        samples = 0.05 * np.sin(2 * np.pi * frequency * np.arange(9600) / 16000)
        sf.write(audio, samples, 16000)
        recording = Recording.from_file(audio)
        supervision = SupervisionSegment(
            id=f"utterance-{index}",
            recording_id=recording.id,
            start=0,
            duration=0.6,
            channel=0,
            text="hello world",
            speaker=f"speaker-{index}",
            alignment={"word": [AlignmentItem("hello", 0, 0.3), AlignmentItem("world", 0.3, 0.3)]},
        )
        cuts.append(
            MonoCut(
                id=f"cut-{index}", start=0, duration=0.6, channel=0, recording=recording, supervisions=[supervision]
            )
        )
    CutSet.from_cuts(cuts).to_file(inputs / "source_train.jsonl.gz")
    config = root / "meeting.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "defaults": ["default", "_self_"],
                "hydra": {"searchpath": ["file://${oc.env:FASTMSS_ROOT}/recipes"]},
                "manifest_dir": str(inputs),
                "manifest_prefix": "source",
                "dset_splits": ["train"],
                "min_spk_utt": 1,
                "split_fa_factor": None,
                "seed": 42,
                "n_meetings": 3,
                "duration": 2,
                "min_max_spk": [2, 2],
                "max_concurrent_speakers": 2,
                "hmm_params": {"p_ind": [0, 1, 0, 0]},
            }
        ),
        config,
    )
    manifest = prepare_fastmss_requests(str(config), str(inputs), str(engine), str(root / "requests.jsonl"))
    return {
        "manifest_path": manifest,
        "config_path": str(config),
        "input_root": str(inputs),
        "fastmss_root": str(engine),
        "output_dir": str(root / "generated"),
    }


@pytest.fixture
def request_params(tmp_path: Path) -> dict[str, str]:
    try:
        distribution("fastmss")
    except PackageNotFoundError:
        pytest.skip("requires the pinned audio_simulation extra")
    return build_request_fixture(tmp_path)


@pytest.mark.parametrize("remap", [False, True])
def test_real_requests_flow_through_audio_consumer_filter_and_writer(
    request_params: dict[str, str], tmp_path: Path, remap: bool
) -> None:
    pytest.importorskip("pyroomacoustics")
    keys = (
        {
            "audio_filepath_key": "mixture",
            "duration_key": "seconds",
            "text_key": "reference",
            "rttm_filepath_key": "rttm",
        }
        if remap
        else {}
    )
    source = FastMSSManifestGenerationStage(**request_params, **keys)
    task = EmptyTask(_metadata={"test_lineage": "retained"})
    groups = BaseStageAdapter(FastMSSRequestSource(source.manifest_path)).process_batch([task])
    rows = BaseStageAdapter(source).process_batch(groups)
    assert len(rows) == 3
    assert len({row.task_id for row in rows}) == 3
    duration = GetAudioDurationStage(audio_filepath_key=source.audio_filepath_key, duration_key="measured_duration")
    duration.setup()
    for row in rows:
        assert row._metadata["test_lineage"] == "retained"
        assert duration.validate_input(row)
        duration.process(row)
        assert row.data["measured_duration"] == pytest.approx(row.data[source.duration_key])
        assert row.data[source.text_key].strip()
        assert Path(row.data[source.rttm_filepath_key]).read_text().startswith("SPEAKER")
    kept = PreserveByValueStage("measured_duration", 0, "gt").process_batch(rows)
    assert len(kept) == 3
    assert not PreserveByValueStage("measured_duration", 0, "lt").process_batch(rows)
    writer = ManifestWriterStage(output_path=str(tmp_path / "terminal.jsonl"))
    writer.setup()
    writer.process_batch(kept)
    writer.teardown()
    saved = [json.loads(line) for line in (tmp_path / "terminal.jsonl").read_text().splitlines()]
    assert len(saved) == 3
    assert all(row[source.audio_filepath_key] and row[source.text_key] for row in saved)


@pytest.mark.parametrize("gaussian_noise", [False, True])
def test_prefix_bounds_generation_and_preserves_native_effective_settings(
    request_params: dict[str, str], gaussian_noise: bool
) -> None:
    pytest.importorskip("pyroomacoustics")
    if gaussian_noise:
        from omegaconf import OmegaConf

        config = OmegaConf.load(request_params["config_path"])
        config.update(add_noise=True, noise_folders=None, noise_probability_global=1)
        OmegaConf.save(config, request_params["config_path"])
        request_params["manifest_path"] = prepare_fastmss_requests(
            request_params["config_path"],
            request_params["input_root"],
            request_params["fastmss_root"],
            str(Path(request_params["manifest_path"]).with_name("gaussian-requests.jsonl")),
        )
    path = Path(request_params["manifest_path"])
    selected = path.with_name("sampled-requests.jsonl")
    selected.write_text(path.read_text().splitlines()[0] + "\n")
    source = FastMSSManifestGenerationStage(**request_params)
    group = FastMSSRequestSource(str(selected)).process(EmptyTask())[0]
    rows = source.process(group)
    assert len(rows) == 1
    assert len(path.read_text().splitlines()) == 3
    effective = json.loads(next(Path(source.output_dir).glob("*/effective_config.json")).read_text())
    worker = FastMSSSimulationStage(
        fastmss_root=source.fastmss_root,
        output_dir=str(Path(source.output_dir).parent / "native-control"),
        isolate_work_dirs=True,
        max_meetings=1,
    )
    control = worker.process(FileGroupTask(dataset_name="native-control", data=[source.config_path]))
    control_rows = FastMSSManifestReaderStage().process(control[0])
    assert len(control_rows) == 1
    native = json.loads((Path(control[0].data[0]).parent / "effective_config.json").read_text())
    for key in ("output_dir", "data_dir", "rir_dir"):
        effective.pop(key)
        native.pop(key)
    assert effective == native


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf"), True, 0])
def test_nonfinite_or_nonpositive_timeout_refuses_before_writes(
    request_params: dict[str, str], timeout: float
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        FastMSSManifestGenerationStage(**request_params, timeout_seconds=timeout)
    assert not Path(request_params["output_dir"]).exists()


def test_real_native_worker_timeout_terminates_without_emitting_meetings(request_params: dict[str, str]) -> None:
    source = FastMSSManifestGenerationStage(**request_params, timeout_seconds=0.01)
    with pytest.raises(subprocess.TimeoutExpired):
        source.process(FileGroupTask(dataset_name="requests", data=[source.manifest_path]))
    assert list(Path(source.output_dir).glob("*/fastmss.log"))
    assert not list(Path(source.output_dir).glob("*/nemo_manifest.json"))


@pytest.mark.parametrize("dependency", ["config", "audio", "engine"])
def test_changed_dependencies_refuse_before_output_is_created(request_params: dict[str, str], dependency: str) -> None:
    path = {
        "config": Path(request_params["config_path"]),
        "audio": Path(request_params["input_root"]) / "speaker-0.wav",
        "engine": Path(request_params["fastmss_root"]) / "recipes" / "default.yaml",
    }[dependency]
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="prepare the request manifest again"):
        FastMSSManifestGenerationStage(**request_params)
    assert not Path(request_params["output_dir"]).exists()


@pytest.mark.parametrize("change", ["duplicate", "gap", "over_count"])
def test_invalid_request_cardinality_refuses_before_generation(request_params: dict[str, str], change: str) -> None:
    path = Path(request_params["manifest_path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "gap":
        rows[1]["meeting_request"] = 12
    else:
        rows.append({**rows[0], "meeting_request": 3})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=r"contiguous|exceeds native"):
        FastMSSManifestGenerationStage(**request_params)
    assert not Path(request_params["output_dir"]).exists()


def test_external_recording_is_not_hidden_by_a_declared_input_directory(
    request_params: dict[str, str], tmp_path: Path
) -> None:
    path = Path(request_params["input_root"]) / "source_train.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    outside = tmp_path / "outside.wav"
    outside.touch()
    rows[0]["recording"]["sources"][0]["source"] = str(outside)
    with gzip.open(path, "wt") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="dependency must exist inside"):
        prepare_fastmss_requests(
            request_params["config_path"],
            request_params["input_root"],
            request_params["fastmss_root"],
            str(tmp_path / "new.jsonl"),
        )
    assert not (tmp_path / "new.jsonl").exists()


def test_output_paths_and_key_aliases_are_admitted_before_writes(request_params: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        FastMSSManifestGenerationStage(**{**request_params, "output_dir": request_params["input_root"]})
    with pytest.raises(ValueError, match="distinct"):
        FastMSSManifestGenerationStage(**request_params, duration_key="text")
    assert not Path(request_params["output_dir"]).exists()


def test_separate_nonprefix_selection_refuses_before_generation(request_params: dict[str, str]) -> None:
    path = Path(request_params["manifest_path"])
    selected = path.with_name("nonprefix.jsonl")
    selected.write_text(path.read_text().splitlines()[1] + "\n")
    source = FastMSSManifestGenerationStage(**request_params)
    group = FastMSSRequestSource(str(selected)).process(EmptyTask())[0]
    with pytest.raises(ValueError, match=r"contiguous|prefix"):
        source.process(group)
    assert not Path(request_params["output_dir"]).exists()


@pytest.mark.parametrize("remap", [False, True])
def test_source_dependencies_and_request_field_replacement_use_existing_core(
    request_params: dict[str, str], remap: bool
) -> None:
    from nemo_curator.stages.audio.audio_agent._capabilities import io_for
    from nemo_curator.stages.audio.audio_agent.artifacts import model_version
    from nemo_curator.stages.audio.audio_agent.card_conformance import check_card
    from nemo_curator.stages.audio.audio_agent.index import get_index
    from nemo_curator.stages.audio.audio_agent.input_identity import resolve_dataset_binding
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef
    from nemo_curator.stages.audio.audio_agent.verbs import validate

    index = get_index()
    assert {"FastMSSRequestSource", "FastMSSManifestGenerationStage"} <= set(index.stage_names())
    for name in ("FastMSSRequestSource", "FastMSSManifestGenerationStage"):
        assert not check_card(name, index.card(name))
    spec = io_for("FastMSSRequestSource")
    assert spec["source"] == {"kind": "manifest", "path_param": "manifest_path"}
    if remap:
        request_params = {
            **request_params,
            "audio_filepath_key": "mixture",
            "duration_key": "seconds",
            "text_key": "reference",
            "rttm_filepath_key": "rttm_ref",
        }
    source = StageRef("FastMSSRequestSource", {"manifest_path": request_params["manifest_path"]})
    generate = StageRef("FastMSSManifestGenerationStage", request_params)
    binding = resolve_dataset_binding(Recipe(stages=[source, generate]))
    assert binding.status == "resolved"
    recipe = Recipe(
        stages=[
            source,
            generate,
            StageRef("PreserveByValueStage", {"input_value_key": "meeting_request", "target_value": 0}),
        ]
    )
    verdict = validate(recipe)
    assert not verdict["runnable"]
    positive = Recipe(
        stages=[
            source,
            generate,
            StageRef(
                "GetAudioDurationStage",
                {"audio_filepath_key": request_params.get("audio_filepath_key", "audio_filepath")},
            ),
        ]
    )
    assert validate(positive)["runnable"]
    before = model_version(request_params, stage_ref="FastMSSManifestGenerationStage")
    audio = Path(request_params["input_root"]) / "speaker-0.wav"
    with audio.open("ab") as stream:
        stream.write(b"\n")
    assert model_version(request_params, stage_ref="FastMSSManifestGenerationStage") != before
