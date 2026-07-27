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

"""FIXTURE: a brand-new agent-ready stage authored AFTER the agent's development.

Used by test-plan level 14 (generalization to unseen modules). It is deliberately
NOT part of the shipped catalog (it lives under eval/audio/fixtures), so it does
not affect the 44/44 card gate. Importing this module registers the class via the
stage metaclass, at which point `resolve_stage_class`, `build_contract`,
`find_producers`, and `card_conformance.check_card` all work on it unchanged —
which is exactly the generalization the test asserts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from nemo_curator.stages.audio._agent_ready import AgentReady, IOSpec, StageContract
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class NewLoudnessNormStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """(FIXTURE) Normalize perceived loudness toward a target LUFS.

    Args:
        audio_filepath_key: Key holding the path to the audio file.
        loudness_key: Key to write the (normalized) integrated loudness in LUFS.
        target_lufs: Target integrated loudness in LUFS.
    """

    name: str = "NewLoudnessNormStage"
    audio_filepath_key: str = "audio_filepath"
    loudness_key: str = "loudness_lufs"
    target_lufs: float = -23.0

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.loudness_key]

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]),
            writes=IOSpec(data_keys=[self.loudness_key]),
        )

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        # Fixture: no real DSP — record the target so the output row is well-formed.
        task.data[self.loudness_key] = float(self.target_lufs)
        self._log_metrics({"process_time": time.perf_counter() - t0})
        return task
