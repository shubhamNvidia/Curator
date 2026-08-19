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

"""Unit tests for UTMOSFilterStage."""

from unittest.mock import MagicMock, patch

import torch

from nemo_curator.stages.audio.common import PreserveByValueConditionsStage, PreserveByValueStage
from nemo_curator.stages.audio.filtering.utmos import UTMOSFilterStage
from nemo_curator.tasks import AudioTask


def _make_task(duration_s: float = 1.0, sample_rate: int = 16000) -> AudioTask:
    num_samples = int(duration_s * sample_rate)
    return AudioTask(
        data={"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate},
        dataset_name="test",
    )


def _mock_model(score: float) -> MagicMock:
    model = MagicMock()
    model.return_value = torch.tensor([score])
    model.parameters = lambda: iter([torch.tensor([0.0])])
    return model


class TestUTMOSFilterStage:
    def test_native_defaults_remain_filter_and_auto(self) -> None:
        stage = UTMOSFilterStage()

        assert stage.action == "filter"
        assert stage.mode == "auto"
        assert stage.mos_threshold == 3.5

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_passes_above_threshold(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = _mock_model(4.5)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert abs(result.data["utmos_mos"] - 4.5) < 1e-3

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_filters_below_threshold(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=4.0)
        stage._model = _mock_model(2.5)

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_annotate_keeps_below_threshold(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=4.0, action="annotate")
        stage._model = _mock_model(2.5)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert abs(result.data["utmos_mos"] - 2.5) < 1e-3
        assert stage.describe().cardinality == "1:1"

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_none_threshold_passes_all(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=None)
        stage._model = _mock_model(1.0)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert abs(result.data["utmos_mos"] - 1.0) < 1e-3

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_prediction_error_skips(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        model = MagicMock(side_effect=RuntimeError("CUDA error"))
        model.parameters = lambda: iter([torch.tensor([0.0])])
        stage._model = model

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_no_waveform_no_filepath_skipped(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = _mock_model(4.0)

        task = AudioTask(data={"some_key": "value"}, dataset_name="test")
        result = stage.process(task)

        assert result == []

    def test_model_not_loaded(self) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = None

        with patch.object(stage, "_ensure_model"):
            result = stage.process(_make_task())

        assert result == []

    def test_teardown_clears_model(self) -> None:
        stage = UTMOSFilterStage()
        stage._model = MagicMock()
        stage.teardown()
        assert stage._model is None

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_nested_segments_filters(self, mock_ensure: MagicMock) -> None:
        """Nested segments: only segments above threshold survive."""
        stage = UTMOSFilterStage(mos_threshold=3.0)
        call_count = {"n": 0}

        def model_side_effect(_waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:  # noqa: ARG001
            call_count["n"] += 1
            return torch.tensor([4.0 if call_count["n"] % 2 == 1 else 2.0])

        model = MagicMock(side_effect=model_side_effect)
        model.parameters = lambda: iter([torch.tensor([0.0])])
        stage._model = model

        sr = 16000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(4)]
        task = AudioTask(
            data={"segments": segments, "original_file": "test.wav"},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 2
        for seg in result.data["segments"]:
            assert "utmos_mos" in seg

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_nested_all_filtered_returns_empty(self, mock_ensure: MagicMock) -> None:
        """Nested segments: when all fail threshold, return []."""
        stage = UTMOSFilterStage(mos_threshold=4.0)
        stage._model = _mock_model(2.0)

        sr = 16000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_annotate_nested_keeps_all_segments(self, mock_ensure: MagicMock) -> None:
        """Nested annotate mode scores every segment without threshold dropping."""
        stage = UTMOSFilterStage(mos_threshold=4.0, action="annotate")
        stage._model = _mock_model(2.0)

        sr = 16000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 3
        assert all(seg["utmos_mos"] == 2.0 for seg in result.data["segments"])

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_task_annotate_then_drop_selector_matches_native_filter(
        self,
        mock_ensure: MagicMock,
    ) -> None:
        threshold = 3.5
        cases = [
            ("passing", _make_task(), 4.2),
            ("failing", _make_task(), 2.8),
            ("nan", _make_task(), float("nan")),
            ("positive_inf", _make_task(), float("inf")),
            ("negative_inf", _make_task(), float("-inf")),
            ("unscorable", AudioTask(data={"id": "missing"}), None),
        ]

        for case_id, source, score in cases:
            source.data["id"] = case_id
            if case_id in {"nan", "positive_inf", "negative_inf"}:
                source.data["custom_utmos"] = 99.0
            native = UTMOSFilterStage(
                action="filter",
                mode="task",
                mos_threshold=threshold,
                score_key="custom_utmos",
            )
            annotate = UTMOSFilterStage(
                action="annotate",
                mode="task",
                mos_threshold=threshold,
                score_key="custom_utmos",
            )
            if score is not None:
                native._model = _mock_model(score)
                annotate._model = _mock_model(score)

            native_result = native.process(
                AudioTask(data=dict(source.data), dataset_name=source.dataset_name)
            )
            annotated = annotate.process(
                AudioTask(data=dict(source.data), dataset_name=source.dataset_name)
            )
            assert isinstance(annotated, AudioTask)
            if case_id in {"nan", "positive_inf", "negative_inf"}:
                assert "custom_utmos" not in annotated.data
            selected = PreserveByValueStage(
                input_value_key="custom_utmos",
                target_value=threshold,
                operator="ge",
                missing_value_policy="drop",
            ).process_batch([annotated])

            assert bool(selected) is isinstance(native_result, AudioTask)

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_segment_annotate_then_generic_selector_matches_native_filter(
        self,
        mock_ensure: MagicMock,
    ) -> None:
        threshold = 3.5
        sample_rate = 16000

        def make_parent() -> AudioTask:
            return AudioTask(
                data={
                    "recording": "r1",
                    "clips": [
                        {
                            "id": "pass",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                        },
                        {
                            "id": "fail",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                        },
                        {
                            "id": "nan",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                            "quality": 99.0,
                        },
                        {
                            "id": "inf",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                            "quality": 99.0,
                        },
                        {"id": "unscorable"},
                    ],
                },
                dataset_name="test",
            )

        def sequence_model() -> MagicMock:
            scores = iter([4.2, 2.8, float("nan"), float("inf")])
            model = MagicMock(side_effect=lambda *_args, **_kwargs: torch.tensor([next(scores)]))
            model.parameters = lambda: iter([torch.tensor([0.0])])
            return model

        native = UTMOSFilterStage(
            action="filter",
            mode="segments",
            segments_key="clips",
            score_key="quality",
            mos_threshold=threshold,
        )
        annotate = UTMOSFilterStage(
            action="annotate",
            mode="segments",
            segments_key="clips",
            score_key="quality",
            mos_threshold=threshold,
        )
        native._model = sequence_model()
        annotate._model = sequence_model()

        native_result = native.process(make_parent())
        annotated = annotate.process(make_parent())
        assert isinstance(native_result, AudioTask)
        assert isinstance(annotated, AudioTask)
        invalid = {
            item["id"]: item
            for item in annotated.data["clips"]
            if item["id"] in {"nan", "inf"}
        }
        assert all("quality" not in item for item in invalid.values())
        selected = PreserveByValueConditionsStage(
            [{"input_value_key": "quality", "target_value": threshold, "operator": "ge"}],
            missing_value_policy="drop",
            items_key="clips",
            drop_parent_if_empty=True,
        ).process_batch([annotated])

        assert [item["id"] for item in native_result.data["clips"]] == ["pass"]
        assert [item["id"] for item in selected[0].data["clips"]] == ["pass"]

        native_all_fail = UTMOSFilterStage(
            action="filter",
            mode="segments",
            segments_key="clips",
            mos_threshold=threshold,
        )
        annotate_all_fail = UTMOSFilterStage(
            action="annotate",
            mode="segments",
            segments_key="clips",
            mos_threshold=threshold,
        )
        native_all_fail._model = _mock_model(2.0)
        annotate_all_fail._model = _mock_model(2.0)
        native_empty = native_all_fail.process(
            AudioTask(
                data={"clips": [{"waveform": torch.randn(1, sample_rate), "sample_rate": sample_rate}]}
            )
        )
        annotated_empty = annotate_all_fail.process(
            AudioTask(
                data={"clips": [{"waveform": torch.randn(1, sample_rate), "sample_rate": sample_rate}]}
            )
        )
        selected_empty = PreserveByValueConditionsStage(
            [{"input_value_key": "utmos_mos", "target_value": threshold, "operator": "ge"}],
            missing_value_policy="drop",
            items_key="clips",
            drop_parent_if_empty=True,
        ).process_batch([annotated_empty])
        assert native_empty == []
        assert selected_empty == []
