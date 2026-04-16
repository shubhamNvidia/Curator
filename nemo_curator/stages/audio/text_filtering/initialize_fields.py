# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class InitializeFieldsStage(ProcessingStage[AudioTask, AudioTask]):
    """Copy pred_text into cleaned_text and initialize skip_me=0.

    This stage sets up the two fields that all downstream text-filtering
    stages depend on, leaving the original pred_text field intact.
    """

    pred_text_key: str = "pred_text"
    cleaned_text_key: str = "cleaned_text"
    skip_me_key: str = "skip_me"
    name: str = "InitializeFields"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pred_text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.cleaned_text_key, self.skip_me_key]

    def process(self, task: AudioTask) -> AudioTask:
        task.data[self.cleaned_text_key] = task.data[self.pred_text_key]
        task.data[self.skip_me_key] = 0
        return task
