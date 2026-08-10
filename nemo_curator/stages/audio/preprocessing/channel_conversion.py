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

"""
Audio channel-count conversion stage.

Brings audio to a requested channel count, in memory, and does nothing else. Sample rate,
format and file layout are left exactly as they were, so a pipeline pairs this with
whatever rate policy it wants -- or with none at all.

It never resamples. When a rate must actually change, ``ResampleAudioStage`` converts it
via ffmpeg; that stage also knows channel layouts, which is what a layout-aware surround
downmix requires and a waveform tensor cannot provide.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.preprocessing import ChannelConversionStage

    pipeline = Pipeline(name="audio_pipeline")
    pipeline.add_stage(ChannelConversionStage(target_channels=1))
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import ClassVar

import soundfile as sf
import torch
from loguru import logger

from nemo_curator.stages.audio._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._residency import (
    InputResidency,
    produce_audio_filepath,
    residency_read_specs,
    resolve_audio,
)
from nemo_curator.stages.audio.common import ensure_waveform_2d, load_audio_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class ChannelConversionStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Bring audio to ``target_channels``, in memory, without touching the sample rate.

    What happens depends on how many channels the input actually has:

    ============  ==================  ==================================================
    input         target              behaviour
    ============  ==================  ==================================================
    ``N``         ``N``               passed through unchanged
    ``N > 1``     ``1``               averaged into one channel (standard mono downmix)
    ``1``         ``T > 1``           duplicated into ``T`` identical channels
    ``N > T > 1`` ``T``               REFUSED -- the row is dropped
    ============  ==================  ==================================================

    The refusal is deliberate. A correct downmix to more than one channel needs ITU-R
    BS.775 coefficients *and* the file's channel order, and a bare ``(channels, samples)``
    tensor carries neither -- WAV channel order comes from the file's channel mask, which
    is gone by the time the audio is a tensor. Averaging 5.1 into two channels does not
    produce stereo, it produces a phase-smeared mix that sounds plausible and is wrong.
    ``ResampleAudioStage`` drives ffmpeg, which does know layouts, so that is the honest
    tool for those conversions. Downmix to ``1`` is not the same problem: averaging every
    channel together IS what mono means, so it needs no layout knowledge.

    Args:
        target_channels: Channel count to produce (default: 1).
        audio_filepath_key: Key in data dict for the audio file path.
        waveform_key: Key in data dict for the in-memory waveform tensor.
        sample_rate_key: Key in data dict for the waveform sample rate.
        num_channels_key: Key where the resulting channel count is written.
        duration_key: Key where the audio duration in seconds is written.
        output_audio_filepath_key: Key where the written WAV path is stored
            (write_to_disk=True only).
        original_audio_filepath_key: Key preserving the pre-conversion path when
            update_audio_filepath=True.
        input_residency: Which input to use -- "waveform" (in-memory only), "file"
            (audio_filepath only), or "auto" (waveform first, file fallback; default).
        keep_waveform_in_task: If True (default), store the converted waveform and sample
            rate in task.data for downstream in-memory consumers.
        write_to_disk: If True, write the converted audio to a WAV file. Without
            output_dir this writes to the system temp dir and nothing cleans it up; in
            multi-node runs point output_dir at shared storage.
        update_audio_filepath: If True, repoint audio_filepath_key at the written file and
            preserve the original under original_audio_filepath_key.
        output_dir: Directory for written audio (write_to_disk=True only).
    """

    target_channels: int = 1

    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    num_channels_key: str = "num_channels"
    duration_key: str = "duration"
    output_audio_filepath_key: str = "converted_audio_filepath"
    original_audio_filepath_key: str = "original_audio_filepath"

    input_residency: InputResidency = "auto"
    keep_waveform_in_task: bool = True
    write_to_disk: bool = False
    update_audio_filepath: bool = False
    output_dir: str | None = None

    # Own bookkeeping: the resulting channel count, recorded for readers and reports.
    # Nothing routes on it, so it needs no shared role -- and declaring it here means
    # adding this stage did not require touching the central role table.
    INTERNAL_KEY_FIELDS: ClassVar[frozenset[str]] = frozenset({"num_channels_key"})

    name: str = "ChannelConversion"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        super().__init__()
        # Type as well as range. YAML reads ``target_channels: 2.0`` as a float, which used to
        # construct fine and then die inside a worker at ``waveform.repeat(2.0, 1)`` with a
        # TypeError -- not one of the (OSError, RuntimeError) this stage drops rows for, so it
        # propagated and took the run down mid-corpus instead of being caught at the recipe.
        if isinstance(self.target_channels, bool) or not isinstance(self.target_channels, int):
            msg = (
                f"target_channels must be a whole number of channels, got "
                f"{self.target_channels!r} ({type(self.target_channels).__name__})"
            )
            raise ValueError(msg)
        if self.target_channels < 1:
            msg = f"target_channels must be at least 1, got {self.target_channels}"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], self._written_keys()

    def _written_keys(self) -> list[str]:
        keys = [
            self.num_channels_key,
            self.duration_key,
        ]
        if self.keep_waveform_in_task:
            keys.extend([self.waveform_key, self.sample_rate_key])
        if self.write_to_disk:
            keys.append(self.output_audio_filepath_key)
            if self.update_audio_filepath:
                keys.append(self.audio_filepath_key)
        return keys

    def describe(self) -> StageContract:
        produces = []
        if self.keep_waveform_in_task:
            produces.append("tensor")
        if self.write_to_disk:
            produces.append("disk")
        return StageContract(
            reads_one_of=residency_read_specs(
                self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            ),
            writes=IOSpec(data_keys=self._written_keys(), produces=produces),
            # Downmixing to mono always succeeds, but any other target refuses the conversions
            # it cannot do correctly (N > target > 1) and drops those rows. That makes the stage
            # a filter for those configurations, and saying so is what puts a seam in the
            # semantic review packet for a reviewer to ask about.
            cardinality="filter" if self.target_channels > 1 else "1:1",
            # Declared here, by the stage that owns the parameter, so a caller running
            # this in a sandbox knows what to redirect without a central table entry.
            gates=Gates(
                writes_to_disk=self.write_to_disk,
                output_path_params=["output_dir"],
            ),
        )

    def _convert(self, waveform: torch.Tensor, source: str) -> torch.Tensor | None:
        """Bring ``waveform`` to ``target_channels``, or None when that cannot be done right."""
        num_channels = waveform.shape[0]
        if num_channels == self.target_channels:
            return waveform
        if self.target_channels == 1:
            logger.debug(f"Averaging {num_channels} channels to mono")
            return torch.mean(waveform, dim=0, keepdim=True)
        if num_channels == 1:
            # Duplicate the single channel. This adds no information -- the result is the
            # same signal N times -- but it is what `ffmpeg -ac` does and what a consumer
            # expecting a fixed channel count needs.
            logger.debug(f"Duplicating mono into {self.target_channels} channels")
            return waveform.repeat(self.target_channels, 1)
        logger.warning(
            f"Cannot downmix {num_channels} channels to {self.target_channels} without the "
            f"file's channel layout, which a waveform tensor does not carry: {source}. "
            "Use ResampleAudioStage (ffmpeg) for a layout-aware downmix, or "
            "target_channels=1."
        )
        return None

    def _write_audio(self, waveform: torch.Tensor, sample_rate: int, task: AudioTask) -> str:
        output_dir = self.output_dir or tempfile.gettempdir()
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(str(task.data.get(self.audio_filepath_key, "audio"))))[0]
        fd, path = tempfile.mkstemp(prefix=f"{stem}_ch{self.target_channels}_", suffix=".wav", dir=output_dir)
        os.close(fd)
        audio = waveform.detach().cpu()
        arr = audio[0].numpy() if audio.shape[0] == 1 else audio.T.numpy()
        sf.write(path, arr, sample_rate)
        return path

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Convert the audio's channel count. Returns [] for a row that cannot be converted."""
        try:
            resolved = resolve_audio(
                task.data,
                residency=self.input_residency,  # type: ignore[arg-type]
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
                mono=False,
                loader=load_audio_file,  # module-level symbol: patchable at this module
            )
        except (OSError, RuntimeError) as e:  # corrupt/unreadable audio -> skip the row
            logger.error(f"Failed to load audio for {task.data.get(self.audio_filepath_key)!r}: {e}")
            return []
        if resolved is None:
            logger.error(f"Audio input not found for key {self.audio_filepath_key!r}")
            return []

        try:
            waveform, sample_rate = resolved
            waveform = ensure_waveform_2d(waveform)

            if sample_rate <= 0:
                logger.error(f"Invalid sample rate ({sample_rate}) in audio input")
                return []

            source = str(task.data.get(self.audio_filepath_key, self.waveform_key))
            converted = self._convert(waveform, source)
            if converted is None:
                return []

            if self.keep_waveform_in_task:
                task.data[self.waveform_key] = converted
                task.data[self.sample_rate_key] = sample_rate
            task.data[self.num_channels_key] = converted.shape[0]
            task.data[self.duration_key] = converted.shape[1] / sample_rate

            if self.write_to_disk:
                path = self._write_audio(converted, sample_rate, task)
                task.data[self.output_audio_filepath_key] = path
                if self.update_audio_filepath:
                    produce_audio_filepath(
                        task.data,
                        path,
                        key=self.audio_filepath_key,
                        original_key=self.original_audio_filepath_key,
                    )

        except (OSError, RuntimeError) as e:
            logger.error(f"Error processing audio input: {e}")
            return []
        else:
            return task
