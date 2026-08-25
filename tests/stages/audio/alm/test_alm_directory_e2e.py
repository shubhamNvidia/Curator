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

"""End-to-end ALM pipeline over a directory of manifests.

Lives here rather than in ``test_common.py`` because it drives the ALM stages: the
ManifestReader directory discovery it exercises is only the source of the pipeline, and
the assertions are about ALM's window output.
"""

from pathlib import Path

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
from nemo_curator.stages.audio.common import ManifestReader

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"
ALM_FIXTURES_DIR = FIXTURES_DIR / "audio" / "alm"


class TestALMDirectoryEndToEnd:
    """ManifestReader directory discovery feeding the full ALM pipeline."""

    def test_composite_end_to_end_with_directory(self) -> None:
        """End-to-end: ManifestReader composite with directory input through full pipeline."""
        nested = ALM_FIXTURES_DIR / "nested_manifests"

        pipeline = Pipeline(name="test_dir_e2e", description="Directory discovery end-to-end test")
        pipeline.add_stage(ManifestReader(manifest_path=str(nested)))
        pipeline.add_stage(
            ALMDataBuilderStage(
                target_window_duration=120.0,
                tolerance=0.1,
                min_sample_rate=16000,
                min_bandwidth=8000,
                min_speakers=2,
                max_speakers=5,
            )
        )
        pipeline.add_stage(ALMDataOverlapStage(overlap_percentage=50, target_duration=120.0))

        executor = XennaExecutor()
        results = pipeline.run(executor)

        output_entries = []
        for task in results or []:
            output_entries.append(task.data)

        assert len(output_entries) == 20  # 4 files x 5 entries
        total_windows = sum(len(e.get("filtered_windows", [])) for e in output_entries)
        assert total_windows == 100  # 25 per file x 4 files
        total_dur = sum(e.get("filtered_dur", 0) for e in output_entries)
        assert abs(total_dur - 12142.0) < 1.0
