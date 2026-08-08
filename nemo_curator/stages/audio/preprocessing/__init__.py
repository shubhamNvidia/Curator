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
Audio preprocessing stages.

These stages prepare audio for further processing:
- ChannelConversionStage: Bring audio to a requested channel count (never resamples)
- SampleRateFilterStage: Keep only acceptable sample rates, recording each (header-only read)
- MonoConversionStage: Convert to mono and verify sample rate in one step
- SegmentConcatenationStage: Concatenate multiple audio segments

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.preprocessing import (
        ChannelConversionStage,
        SampleRateFilterStage,
    )

    pipeline = Pipeline(name="preprocessing_pipeline")
    pipeline.add_stage(SampleRateFilterStage(allowed_sample_rates=[16000]))
    pipeline.add_stage(ChannelConversionStage(target_channels=1))
"""

from .channel_conversion import ChannelConversionStage
from .concatenation import SegmentConcatenationStage
from .mono_conversion import MonoConversionStage
from .sample_rate_filter import SampleRateFilterStage

__all__ = [
    "ChannelConversionStage",
    "MonoConversionStage",
    "SampleRateFilterStage",
    "SegmentConcatenationStage",
]
