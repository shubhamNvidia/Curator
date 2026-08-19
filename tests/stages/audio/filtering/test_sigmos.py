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

from unittest.mock import MagicMock, patch

import torch

from nemo_curator.stages.audio.common import PreserveByValueConditionsStage
from nemo_curator.stages.audio.filtering.sigmos import SIGMOSFilterStage
from nemo_curator.tasks import AudioTask

_GOOD_SCORES = {
    "MOS_NOISE": 4.5,
    "MOS_OVRL": 4.0,
    "MOS_SIG": 4.2,
    "MOS_COL": 4.1,
    "MOS_DISC": 4.3,
    "MOS_LOUD": 3.8,
    "MOS_REVERB": 4.0,
}

_BAD_SCORES = {
    "MOS_NOISE": 2.0,
    "MOS_OVRL": 2.0,
    "MOS_SIG": 2.0,
    "MOS_COL": 2.0,
    "MOS_DISC": 2.0,
    "MOS_LOUD": 2.0,
    "MOS_REVERB": 2.0,
}


def _make_task(duration_s: float = 1.0, sample_rate: int = 48000) -> AudioTask:
    num_samples = int(duration_s * sample_rate)
    return AudioTask(
        data={"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate},
        dataset_name="test",
    )


def _make_mock_model(scores: dict) -> MagicMock:
    model = MagicMock()
    model.run.return_value = scores
    return model


class TestSIGMOSFilterStage:
    def test_native_defaults_remain_filter_auto_and_two_thresholds(self) -> None:
        stage = SIGMOSFilterStage()

        assert stage.action == "filter"
        assert stage.mode == "auto"
        assert stage.noise_threshold == 4.0
        assert stage.ovrl_threshold == 3.5

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_process_passes_good_scores(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        stage._model = _make_mock_model(_GOOD_SCORES)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["sigmos_noise"] == 4.5
        assert result.data["sigmos_ovrl"] == 4.0

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_process_rejects_bad_scores(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        stage._model = _make_mock_model(_BAD_SCORES)

        result = stage.process(_make_task())

        assert result == []

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_annotate_keeps_bad_scores(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5, action="annotate")
        stage._model = _make_mock_model(_BAD_SCORES)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["sigmos_noise"] == 2.0
        assert result.data["sigmos_ovrl"] == 2.0
        assert stage.describe().cardinality == "1:1"

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_none_thresholds_disable_checks(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(
            noise_threshold=None,
            ovrl_threshold=None,
            sig_threshold=None,
            col_threshold=None,
            disc_threshold=None,
            loud_threshold=None,
            reverb_threshold=None,
        )
        stage._model = _make_mock_model(_BAD_SCORES)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_partial_threshold_fail(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=None)
        stage._model = _make_mock_model(
            {
                "MOS_NOISE": 3.0,
                "MOS_OVRL": 5.0,
                "MOS_SIG": 5.0,
                "MOS_COL": 5.0,
                "MOS_DISC": 5.0,
                "MOS_LOUD": 5.0,
                "MOS_REVERB": 5.0,
            }
        )

        result = stage.process(_make_task())

        assert result == []

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_sigmos_output_keys(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(noise_threshold=1.0, ovrl_threshold=1.0)
        stage._model = _make_mock_model(_GOOD_SCORES)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        for key in [
            "sigmos_noise",
            "sigmos_ovrl",
            "sigmos_sig",
            "sigmos_col",
            "sigmos_disc",
            "sigmos_loud",
            "sigmos_reverb",
        ]:
            assert key in result.data

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_no_audio_no_filepath_skipped(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage()
        stage._model = _make_mock_model(_GOOD_SCORES)

        task = AudioTask(data={"some_key": "value"}, dataset_name="test")
        result = stage.process(task)

        assert result == []

    def test_model_not_available(self) -> None:
        stage = SIGMOSFilterStage()
        stage._model = None

        with patch.object(stage, "_initialize_model"):
            result = stage.process(_make_task())

        assert result == []

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_process_nested_segments_filters(self, mock_init: MagicMock) -> None:
        """Nested segments: only segments passing thresholds survive."""
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        call_count = {"n": 0}

        def fake_run(audio: object, sr: int) -> dict:  # noqa: ARG001
            call_count["n"] += 1
            if call_count["n"] % 2 == 1:
                return _GOOD_SCORES
            return _BAD_SCORES

        model = MagicMock()
        model.run.side_effect = fake_run
        stage._model = model

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(4)]
        task = AudioTask(
            data={"segments": segments, "original_file": "test.wav"},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 2
        for seg in result.data["segments"]:
            assert "sigmos_noise" in seg

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_process_nested_all_filtered_returns_empty(self, mock_init: MagicMock) -> None:
        """Nested segments: when all fail thresholds, return []."""
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        stage._model = _make_mock_model(_BAD_SCORES)

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert result == []

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_annotate_nested_keeps_all_segments(self, mock_init: MagicMock) -> None:
        """Nested annotate mode scores every segment without threshold dropping."""
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5, action="annotate")
        stage._model = _make_mock_model(_BAD_SCORES)

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 3
        assert all(seg["sigmos_noise"] == 2.0 for seg in result.data["segments"])

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_task_annotate_then_compound_selector_matches_all_native_thresholds(
        self,
        mock_init: MagicMock,
    ) -> None:
        keys = {
            "noise_key": "quality_noise",
            "ovrl_key": "quality_ovrl",
            "reverb_key": "quality_reverb",
        }
        thresholds = {
            "noise_threshold": 4.0,
            "ovrl_threshold": 3.5,
            "reverb_threshold": 3.8,
            "sig_threshold": None,
            "col_threshold": None,
            "disc_threshold": None,
            "loud_threshold": None,
        }
        cases = [
            ("passing", _make_task(), _GOOD_SCORES),
            (
                "noise_fails",
                _make_task(),
                {**_GOOD_SCORES, "MOS_NOISE": 3.9},
            ),
            (
                "ovrl_fails",
                _make_task(),
                {**_GOOD_SCORES, "MOS_OVRL": 3.4},
            ),
            (
                "reverb_fails",
                _make_task(),
                {**_GOOD_SCORES, "MOS_REVERB": 3.7},
            ),
            (
                "nan_dimension",
                _make_task(),
                {**_GOOD_SCORES, "MOS_SIG": float("nan")},
            ),
            (
                "positive_inf_dimension",
                _make_task(),
                {**_GOOD_SCORES, "MOS_COL": float("inf")},
            ),
            (
                "negative_inf_dimension",
                _make_task(),
                {**_GOOD_SCORES, "MOS_DISC": float("-inf")},
            ),
            ("unscorable", AudioTask(data={"id": "missing"}), None),
        ]

        for case_id, source, scores in cases:
            source.data["id"] = case_id
            if "dimension" in case_id:
                source.data.update(
                    {
                        "quality_noise": 99.0,
                        "quality_ovrl": 99.0,
                        "quality_reverb": 99.0,
                        "sigmos_sig": 99.0,
                        "sigmos_col": 99.0,
                        "sigmos_disc": 99.0,
                        "sigmos_loud": 99.0,
                    }
                )
            native = SIGMOSFilterStage(
                action="filter",
                mode="task",
                **thresholds,
                **keys,
            )
            annotate = SIGMOSFilterStage(
                action="annotate",
                mode="task",
                **thresholds,
                **keys,
            )
            if scores is not None:
                native._model = _make_mock_model(scores)
                annotate._model = _make_mock_model(scores)

            native_result = native.process(
                AudioTask(data=dict(source.data), dataset_name=source.dataset_name)
            )
            annotated = annotate.process(
                AudioTask(data=dict(source.data), dataset_name=source.dataset_name)
            )
            assert isinstance(annotated, AudioTask)
            if "dimension" in case_id:
                assert not any(
                    key.startswith(("quality_", "sigmos_"))
                    for key in annotated.data
                )
            selected = PreserveByValueConditionsStage(
                conditions=[
                    {
                        "input_value_key": "quality_noise",
                        "target_value": 4.0,
                        "operator": "ge",
                    },
                    {
                        "input_value_key": "quality_ovrl",
                        "target_value": 3.5,
                        "operator": "ge",
                    },
                    {
                        "input_value_key": "quality_reverb",
                        "target_value": 3.8,
                        "operator": "ge",
                    },
                ],
                missing_value_policy="drop",
            ).process_batch([annotated])

            assert bool(selected) is isinstance(native_result, AudioTask)

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_segment_annotate_then_compound_selector_matches_native_filter(
        self,
        mock_init: MagicMock,
    ) -> None:
        sample_rate = 48000
        thresholds = {
            "noise_threshold": 4.0,
            "ovrl_threshold": 3.5,
            "sig_threshold": None,
            "col_threshold": None,
            "disc_threshold": None,
            "loud_threshold": None,
            "reverb_threshold": 3.8,
        }
        keys = {
            "noise_key": "quality_noise",
            "ovrl_key": "quality_ovrl",
            "reverb_key": "quality_reverb",
        }

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
                            "quality_noise": 99.0,
                            "sigmos_sig": 99.0,
                        },
                        {
                            "id": "inf",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                            "quality_noise": 99.0,
                            "sigmos_sig": 99.0,
                        },
                        {"id": "unscorable"},
                    ],
                },
                dataset_name="test",
            )

        def sequence_model() -> MagicMock:
            model = MagicMock()
            model.run.side_effect = [
                _GOOD_SCORES,
                {**_GOOD_SCORES, "MOS_REVERB": 3.7},
                {**_GOOD_SCORES, "MOS_SIG": float("nan")},
                {**_GOOD_SCORES, "MOS_COL": float("inf")},
            ]
            return model

        native = SIGMOSFilterStage(
            action="filter",
            mode="segments",
            segments_key="clips",
            **thresholds,
            **keys,
        )
        annotate = SIGMOSFilterStage(
            action="annotate",
            mode="segments",
            segments_key="clips",
            **thresholds,
            **keys,
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
        assert all(
            not any(key.startswith(("quality_", "sigmos_")) for key in item)
            for item in invalid.values()
        )
        selected = PreserveByValueConditionsStage(
            conditions=[
                {"input_value_key": "quality_noise", "target_value": 4.0, "operator": "ge"},
                {"input_value_key": "quality_ovrl", "target_value": 3.5, "operator": "ge"},
                {"input_value_key": "quality_reverb", "target_value": 3.8, "operator": "ge"},
            ],
            missing_value_policy="drop",
            items_key="clips",
            drop_parent_if_empty=True,
        ).process_batch([annotated])

        assert [item["id"] for item in native_result.data["clips"]] == ["pass"]
        assert [item["id"] for item in selected[0].data["clips"]] == ["pass"]
