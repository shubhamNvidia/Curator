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
NeMo Curator Audio Processing Stages.

This module provides stages for processing and curating audio data,
including ASR inference, quality assessment, and ALM data preparation.

Preprocessing:
    - MonoConversionStage: Convert multi-channel audio to mono
    - SegmentConcatenationStage: Concatenate audio segments

Postprocessing:
    - TimestampMapperStage: Map timestamps back to original files

Segmentation:
    - VADSegmentationStage: Voice Activity Detection segmentation
    - SpeakerSeparationStage: Speaker diarization and separation

Filtering:
    - NISQAFilterStage: NISQA speech quality filtering
    - SIGMOSFilterStage: SIGMOS quality filtering
    - UTMOSFilterStage: UTMOS MOS prediction filtering
    - BandFilterStage: Bandwidth classification filtering

ALM:
    - ALMDataBuilderStage: Build ALM data
    - ALMDataOverlapStage: ALM data overlap processing

Advanced Pipelines:
    - AudioDataFilterStage: Complete audio curation pipeline (VAD + Quality + Speaker Sep)

Common:
    - GetAudioDurationStage: Extract audio duration
    - LegacySpeechStage: Legacy speech processing
    - PreserveByValueStage: Filter by field value

Example::

    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio import (
        MonoConversionStage,
        VADSegmentationStage,
        NISQAFilterStage,
    )

    pipeline = Pipeline(name="audio_curation")
    pipeline.add_stage(MonoConversionStage(output_sample_rate=48000))
    pipeline.add_stage(VADSegmentationStage(min_duration_sec=2.0))
    pipeline.add_stage(NISQAFilterStage(mos_threshold=4.5))

    # Or use the unified AudioDataFilterStage:
    from nemo_curator.stages.audio import AudioDataFilterStage

    pipeline.add_stage(AudioDataFilterStage())
"""

from nemo_curator.stages.audio.advance_pipelines import (
    AudioDataFilterStage,
)
from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    LegacySpeechStage,
    PreserveByValueStage,
)
from nemo_curator.stages.audio.filtering import (
    BandFilterStage,
    NISQAFilterStage,
    SIGMOSFilterStage,
    UTMOSFilterStage,
)
from nemo_curator.stages.audio.postprocessing import (
    TimestampMapperStage,
)
from nemo_curator.stages.audio.preprocessing import (
    MonoConversionStage,
    SegmentConcatenationStage,
)
from nemo_curator.stages.audio.segmentation import (
    SpeakerSeparationStage,
    VADSegmentationStage,
)

__all__ = [
    # Preprocessing
    "MonoConversionStage",
    "SegmentConcatenationStage",
    # Postprocessing
    "TimestampMapperStage",
    # Segmentation
    "VADSegmentationStage",
    "SpeakerSeparationStage",
    # Filtering
    "BandFilterStage",
    "NISQAFilterStage",
    "SIGMOSFilterStage",
    "UTMOSFilterStage",
    # ALM
    "ALMDataBuilderStage",
    "ALMDataOverlapStage",
    # Advanced Pipelines
    "AudioDataFilterStage",
    # Common
    "GetAudioDurationStage",
    "LegacySpeechStage",
    "PreserveByValueStage",
]
