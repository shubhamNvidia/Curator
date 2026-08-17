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

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch

from nemo_curator.stages.audio.tagging.resample_audio import ResampleAudioStage
from nemo_curator.tasks import AudioTask


class TestResampleAudioStage:
    """Tests for ResampleAudioStage."""

    def test_process(self, audio_task: Callable[..., AudioTask], audio_filepath: Path) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stage = ResampleAudioStage(resampled_audio_dir=tmpdir)
            stage.setup()
            task = audio_task(
                audio_filepath=str(audio_filepath),
                audio_item_id="id_1",
            )
            result = stage.process(task)
            out = result.data
            assert out.get("audio_filepath") == str(audio_filepath)
            assert out.get("resampled_audio_filepath") == f"{tmpdir}/id_1.wav"
            assert out.get("duration") == 60.0

    def test_a_file_input_keeps_the_name_it_has_always_had(self, audio_filepath: Path) -> None:
        """Every tutorial reads real files off disk; their output names must not move."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage = ResampleAudioStage(resampled_audio_dir=tmpdir)
            stage.setup()
            stage.process(AudioTask(task_id="t", dataset_name="d", data={"audio_filepath": str(audio_filepath)}))

            path_hash = hashlib.sha256(str(audio_filepath).encode()).hexdigest()[:8]
            assert os.listdir(tmpdir) == [f"{audio_filepath.stem}_{path_hash}.wav"]

    def test_a_waveform_input_writes_one_file_however_often_it_is_rerun(self) -> None:
        waveform = torch.sin(torch.arange(0, 16000 * 2) * 0.01).unsqueeze(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(3):
                stage = ResampleAudioStage(resampled_audio_dir=tmpdir, input_residency="waveform")
                stage.setup()
                stage.process(
                    AudioTask(
                        task_id="t",
                        dataset_name="d",
                        data={"waveform": waveform.clone(), "sample_rate": 16000},
                    )
                )

            assert len(os.listdir(tmpdir)) == 1, "the same audio must not pile up a file per run"

    def test_changing_the_target_rate_does_not_reuse_the_old_conversion(self) -> None:
        """The name carries the settings, so 'it already exists, skip it' cannot serve 48 kHz for 16."""
        waveform = torch.sin(torch.arange(0, 16000 * 2) * 0.01).unsqueeze(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            for rate in (16000, 8000):
                stage = ResampleAudioStage(
                    resampled_audio_dir=tmpdir, input_residency="waveform", target_sample_rate=rate
                )
                stage.setup()
                stage.process(
                    AudioTask(
                        task_id="t",
                        dataset_name="d",
                        data={"waveform": waveform.clone(), "sample_rate": 16000},
                    )
                )

            assert len(os.listdir(tmpdir)) == 2, "a different target rate must not answer from the old file"

    def test_a_second_run_at_a_new_rate_does_not_serve_the_old_file(self) -> None:
        import numpy as np
        import soundfile

        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as out:
            src = os.path.join(srcdir, "a.wav")
            soundfile.write(src, np.sin(np.arange(48000) * 0.01).astype("float32"), 48000)

            for rate in (16000, 8000):
                stage = ResampleAudioStage(resampled_audio_dir=out, write_to_disk=True, target_sample_rate=rate)
                stage.setup()
                stage.process(AudioTask(task_id="t", dataset_name="d", data={"audio_filepath": src}))

                written = os.path.join(out, os.listdir(out)[0])
                assert soundfile.info(written).samplerate == rate, "served audio at the previous run's rate"

    def test_segments_sharing_a_parent_id_each_get_their_own_file(self) -> None:
        waveform = torch.sin(torch.arange(0, 16000) * 0.01).unsqueeze(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            stage = ResampleAudioStage(resampled_audio_dir=tmpdir, input_residency="waveform")
            stage.setup()
            for segment in range(3):
                stage.process(
                    AudioTask(
                        task_id="t",
                        dataset_name="d",
                        data={
                            # What VAD hands every child: the parent's id, identical across siblings.
                            "audio_item_id": "utt1",
                            "waveform": (waveform * (segment + 1)).clone(),
                            "sample_rate": 16000,
                        },
                    )
                )

            assert len(os.listdir(tmpdir)) == 3, "sibling segments collapsed onto one filename"
