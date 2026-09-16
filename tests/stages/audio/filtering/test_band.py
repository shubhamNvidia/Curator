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

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio._agent._agent_registry import static_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready, assert_residency_consumption
from nemo_curator.stages.audio.filtering.band import BandFilterStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


def _make_task(waveform: torch.Tensor | None = None, sample_rate: int = 48000) -> AudioTask:
    if waveform is None:
        waveform = torch.randn(1, sample_rate)
    return AudioTask(
        data={"waveform": waveform, "sample_rate": sample_rate},
        dataset_name="test",
    )


def _write_wav(path: Path, *, samples: int = 32, sample_rate: int = 16000) -> str:
    sf.write(path, np.linspace(-0.25, 0.25, samples, dtype=np.float32), sample_rate)
    return str(path)


def _stage(*, mode: str = "task", scorable: bool = True, input_residency: str = "auto") -> BandFilterStage:
    stage = BandFilterStage(mode=mode, action="annotate", input_residency=input_residency)
    if scorable:
        stage._predictor = MagicMock()
        stage._predictor.predict_audio.return_value = "full_band"
    return stage


class TestBandFilterStage:
    def test_legacy_positional_constructor_order_is_preserved(self) -> None:
        resources = Resources(cpus=2)

        stage = BandFilterStage("model.joblib", "/cache", "narrow_band", "legacy", 8, resources)

        assert stage.model_path == "model.joblib"
        assert stage.cache_dir == "/cache"
        assert stage.band_value == "narrow_band"
        assert stage.name == "legacy"
        assert stage.batch_size == 8
        assert stage.resources is resources

    @pytest.mark.parametrize("residency", ["wavefrom", "", "FILE"])
    def test_invalid_input_residency_is_rejected(self, residency: str) -> None:
        with pytest.raises(ValueError, match="input_residency"):
            BandFilterStage(input_residency=residency)  # type: ignore[arg-type]

    def test_prediction_key_cannot_overwrite_audio_input(self) -> None:
        with pytest.raises(ValueError, match="must not collide"):
            BandFilterStage(prediction_key="waveform")

    def test_explicit_missing_model_path_does_not_download(self, tmp_path: Path) -> None:
        stage = BandFilterStage(model_path=str(tmp_path / "missing.joblib"))

        with (
            patch("nemo_curator.stages.audio.filtering.band.hf_hub_download") as download,
            pytest.raises(FileNotFoundError, match="model_path"),
        ):
            stage._resolve_model_path()

        download.assert_not_called()

    def test_static_contract_conservatively_reports_download_and_row_independence(self) -> None:
        contract = static_contract(BandFilterStage)

        assert contract.gates.requires_internet_first_run is True
        assert contract.gates.per_row_independent is True

    def test_resident_waveform_uses_only_file_header_for_missing_sample_rate(self, tmp_path: Path) -> None:
        stage = _stage()
        resident = torch.full((1, 13), 0.75)
        path = _write_wav(tmp_path / "different.wav", samples=31)
        task = AudioTask(dataset_name="test", data={"waveform": resident, "audio_filepath": path})

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        predicted_waveform, predicted_sample_rate = stage._predictor.predict_audio.call_args.args
        assert predicted_waveform.shape == (1, 13)
        assert torch.allclose(predicted_waveform, resident)
        assert predicted_sample_rate == 16000
        assert task.data["sample_rate"] == 16000

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_full_band_passes(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "full_band"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["band_prediction"] == "full_band"

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_narrow_band_filtered_out(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_annotate_keeps_non_target_band(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="full_band", action="annotate")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["band_prediction"] == "narrow_band"
        assert stage.describe().cardinality == "1:1"

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_narrow_band_passes_when_configured(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="narrow_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["band_prediction"] == "narrow_band"

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_error_prediction_skipped(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "Error: model failed"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_no_waveform_no_filepath_skipped(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="full_band")
        stage._predictor = MagicMock()

        task = AudioTask(data={"some_key": "value"}, dataset_name="test")
        result = stage.process(task)

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_nested_segments_filters(self, mock_init: MagicMock) -> None:
        """Nested segments: only segments passing the band filter survive."""
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        call_count = {"n": 0}

        def predict_side_effect(_waveform: object, _sample_rate: int) -> str:
            call_count["n"] += 1
            return "full_band" if call_count["n"] % 2 == 1 else "narrow_band"

        predictor.predict_audio = predict_side_effect
        stage._predictor = predictor

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(4)]
        task = AudioTask(
            data={"segments": segments, "original_file": "test.wav"},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 2

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_nested_all_filtered_returns_empty(self, mock_init: MagicMock) -> None:
        """Nested segments: when all segments are filtered, return []."""
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_annotate_nested_keeps_all_segments(self, mock_init: MagicMock) -> None:
        """Nested annotate mode predicts every segment without target-band dropping."""
        stage = BandFilterStage(band_value="full_band", action="annotate")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 3
        assert all(seg["band_prediction"] == "narrow_band" for seg in result.data["segments"])


@pytest.mark.parametrize("mode", ["task", "segments", "auto"])
@pytest.mark.parametrize("scorable", [True, False], ids=["success", "unscorable"])
def test_band_agent_conformance_covers_all_scopes(mode: str, scorable: bool) -> None:
    stage = _stage(mode=mode, scorable=scorable, input_residency="waveform")

    def fixture() -> AudioTask:
        audio = {"waveform": torch.zeros(1, 32), "sample_rate": 16000}
        data = {"segments": [audio]} if mode == "segments" else audio
        return AudioTask(dataset_name="test", data=data)

    assert_agent_ready(stage, fixture, expected_cardinality="1:1", segments_key="segments")


def test_band_consumes_file_and_waveform_residencies(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "band.wav")

    assert_residency_consumption(
        lambda residency: _stage(input_residency=residency),
        file_fixture=lambda: AudioTask(dataset_name="test", data={"audio_filepath": path}),
        waveform_fixture=lambda: AudioTask(
            dataset_name="test",
            data={"waveform": torch.zeros(1, 32), "sample_rate": 16000},
        ),
    )
