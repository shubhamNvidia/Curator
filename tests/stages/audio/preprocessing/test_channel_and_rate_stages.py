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

"""Channel conversion and sample-rate selection as two independent stages.

Each does one job and leaves the other alone, so a pipeline sets a channel policy and a
rate policy separately -- or uses only the one it needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.preprocessing import ChannelConversionStage, SampleRateFilterStage
from nemo_curator.stages.audio.preprocessing import sample_rate_filter as srf
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


def _wav(tmp_path: Path, channels: int = 1, rate: int = 16000, name: str = "a.wav") -> str:
    path = tmp_path / name
    data = (
        np.zeros(rate, dtype="float32")
        if channels == 1
        else np.zeros((rate, channels), dtype="float32")
    )
    sf.write(str(path), data, rate)
    return str(path)


def _task(path: str) -> AudioTask:
    return AudioTask(task_id="t", dataset_name="d", data={"audio_filepath": path})


class TestChannelConversion:
    @pytest.mark.parametrize(
        ("source", "target"),
        [(1, 1), (2, 1), (6, 1), (2, 2), (6, 6)],
    )
    def test_downmix_and_passthrough_produce_the_requested_count(self, tmp_path: Path, source: int, target: int) -> None:
        result = ChannelConversionStage(target_channels=target).process(
            _task(_wav(tmp_path, channels=source, name=f"{source}to{target}.wav"))
        )
        assert result != []
        assert result.data["num_channels"] == target

    def test_mono_downmix_averages_rather_than_selecting_a_channel(self, tmp_path: Path) -> None:
        """Averaging is what mono means; taking channel 0 would discard half the signal."""
        path = str(tmp_path / "stereo.wav")
        left = np.ones(16000, dtype="float32")
        right = -np.ones(16000, dtype="float32")
        sf.write(path, np.stack([left, right], axis=1), 16000)

        result = ChannelConversionStage(target_channels=1).process(_task(path))

        assert result != []
        # +1 and -1 average to ~0; selecting either channel would give ~1. The residual is
        # 16-bit quantization (sf.write stores PCM_16, so +/-1.0 becomes 32767/-32768),
        # which is ~3e-5 -- three orders of magnitude below what a channel-select yields.
        assert float(result.data["waveform"].abs().max()) < 0.01

    def test_mono_upmixes_by_duplication(self, tmp_path: Path) -> None:
        """Matches ``ffmpeg -ac``. It adds no information -- the channels are identical."""
        result = ChannelConversionStage(target_channels=2).process(_task(_wav(tmp_path, channels=1)))

        assert result != []
        waveform = result.data["waveform"]
        assert waveform.shape[0] == 2
        assert waveform[0].equal(waveform[1]), "upmix duplicates, so both channels are the same signal"

    @pytest.mark.parametrize(("source", "target"), [(6, 2), (4, 3), (6, 5)])
    def test_a_surround_downmix_is_refused_rather_than_approximated(self, tmp_path: Path, source: int, target: int) -> None:
        """Correct downmix to >1 channel needs BS.775 coefficients AND the file's channel
        order, and a (channels, samples) tensor carries neither -- the WAV channel mask is
        gone by then. Averaging 5.1 into two channels sounds plausible and is wrong, so the
        row is dropped and ResampleAudioStage (ffmpeg, layout-aware) is named instead."""
        result = ChannelConversionStage(target_channels=target).process(
            _task(_wav(tmp_path, channels=source, name=f"{source}to{target}.wav"))
        )
        assert result == []

    def test_the_sample_rate_is_never_changed(self, tmp_path: Path) -> None:
        """This stage converts channels only; rate policy belongs to another stage."""
        result = ChannelConversionStage(target_channels=1).process(
            _task(_wav(tmp_path, channels=2, rate=44100))
        )
        assert result != []
        assert result.data["sample_rate"] == 44100

    def test_a_nonsensical_target_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="target_channels"):
            ChannelConversionStage(target_channels=0)


class TestSampleRateFilter:
    @pytest.mark.parametrize(
        ("kwargs", "keeps"),
        [
            ({"allowed_sample_rates": [16000]}, True),
            ({"allowed_sample_rates": [22050, 44100]}, False),
            ({"min_sample_rate": 16000}, True),
            ({"min_sample_rate": 22050}, False),
            ({"max_sample_rate": 16000}, True),
            ({"max_sample_rate": 8000}, False),
            ({"min_sample_rate": 8000, "max_sample_rate": 48000}, True),
            ({"allowed_sample_rates": [16000], "min_sample_rate": 22050}, False),
            ({}, True),
        ],
    )
    def test_a_list_and_a_range_are_separate_constraints(self, tmp_path: Path, kwargs: dict[str, object], keeps: bool) -> None:
        """Separate parameters on purpose: ``[16000, 48000]`` as a single knob is ambiguous
        between "these two rates" and "this range", and the readings filter very different
        corpora. Every constraint that IS set must be satisfied."""
        result = SampleRateFilterStage(**kwargs).process(_task(_wav(tmp_path, rate=16000)))
        assert (result != []) is keeps

    def test_the_rate_is_recorded_on_rows_that_pass(self, tmp_path: Path) -> None:
        result = SampleRateFilterStage().process(_task(_wav(tmp_path, rate=44100)))
        assert result != []
        assert result.data["sample_rate"] == 44100

    def test_it_reads_the_header_and_never_decodes(self, tmp_path: Path) -> None:
        """The rate is metadata sitting in the first bytes. Decoding to read it costs ~186x
        more per file, and placing this stage before any decoding is what keeps rejected
        rows from ever being decoded -- a decode here would forfeit exactly that."""
        stage = SampleRateFilterStage(allowed_sample_rates=[16000])
        path = _wav(tmp_path, rate=16000)

        def explode(*_args: object, **_kwargs: object) -> None:
            msg = "decoded the audio to read a header value"
            raise AssertionError(msg)

        with patch.object(srf.sf, "read", explode):
            result = stage.process(_task(path))

        assert result != []
        assert result.data["sample_rate"] == 16000

    def test_a_resident_rate_avoids_touching_disk_entirely(self) -> None:
        """An upstream stage that already resolved the rate makes this free."""
        stage = SampleRateFilterStage(allowed_sample_rates=[16000])
        task = AudioTask(
            task_id="t", dataset_name="d",
            data={"audio_filepath": "/nonexistent/never-opened.wav", "sample_rate": 16000},
        )

        result = stage.process(task)

        assert result != []
        assert result.data["sample_rate"] == 16000

    def test_an_unreadable_row_is_dropped_not_crashed(self, tmp_path: Path) -> None:
        result = SampleRateFilterStage().process(_task(str(tmp_path / "missing.wav")))
        assert result == []

    def test_an_empty_allow_list_is_rejected_at_construction(self) -> None:
        """``[]`` would silently discard the entire corpus; None means "no constraint"."""
        with pytest.raises(ValueError, match="at least one rate"):
            SampleRateFilterStage(allowed_sample_rates=[])

    def test_an_inverted_range_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="nothing can pass"):
            SampleRateFilterStage(min_sample_rate=48000, max_sample_rate=16000)


class TestTheyCompose:
    def test_rate_selection_then_channel_conversion(self, tmp_path: Path) -> None:
        """Independent policies: accept a range of rates, and separately require mono.

        Neither stage constrains the other, so a 22.05 kHz corpus can be taken to mono
        without also having to declare 22050 the only acceptable rate.
        """
        path = _wav(tmp_path, channels=2, rate=22050)

        selected = SampleRateFilterStage(min_sample_rate=16000).process(_task(path))
        assert selected != []

        converted = ChannelConversionStage(target_channels=1).process(selected)
        assert converted != []
        assert converted.data["num_channels"] == 1
        assert converted.data["sample_rate"] == 22050, "selection does not resample"
