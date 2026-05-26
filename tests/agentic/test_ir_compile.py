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
"""End-to-end IR → validator → compiler test for the canonical prompt.

`Clean, single-speaker, 2-60 second clips for TTS` is the headline Phase 3
demo. This test reaches all the way to compiled YAML without executing any
real audio.
"""

from __future__ import annotations

import pytest
import yaml

from nemo_curator.agentic.cards import ResourceSpec
from nemo_curator.agentic.compiler import compile_ir_to_yaml
from nemo_curator.agentic.intent import IntentCategories
from nemo_curator.agentic.ir import PipelineIR, SinkSpec, SourceSpec, stage
from nemo_curator.agentic.registry import build_registry
from nemo_curator.agentic.validator import validate


@pytest.fixture(scope="module")
def registry():
    return build_registry(cross_check_runtime=False)


def _tts_clean_ir(manifest_uri: str, target: str) -> PipelineIR:
    return PipelineIR(
        name="tts_clean_2_to_60_single_speaker",
        description="Clean, single-speaker 2-60s TTS clips.",
        intent=IntentCategories(
            output={"sample_rate": 48000, "channels": "mono"},
            segmentation={
                "output_unit": "single_speaker_clips",
                "duration_min_sec": 2.0,
                "duration_max_sec": 60.0,
            },
            quality={"mos": "filter", "mos_threshold": 3.5},
            speakers={"mode": "split"},
            policy={"commercial_only": True},
            raw_prompt="clean, single-speaker, 2-60s, English TTS clips",
        ),
        source=SourceSpec(kind="manifest", uri=manifest_uri),
        sink=SinkSpec(target_dir=target),
        stages=[
            stage("ManifestReader", manifest_path=manifest_uri),
            stage(
                "VADSegmentationStage",
                min_duration_sec=2.0,
                max_duration_sec=60.0,
                nested=False,
            ),
            stage("UTMOSFilterStage", mos_threshold=3.5),
            stage("SIGMOSFilterStage", ovrl_threshold=3.5, noise_threshold=4.0),
            stage("InferenceSortformerStage"),
            stage("PreserveByValueStage", input_value_key="num_speakers", target_value=1, operator="eq"),
            stage("SegmentExtractionStage", output_dir=f"{target}/audio", output_format="wav"),
            stage("ManifestWriterStage", output_path=f"{target}/manifest.jsonl"),
        ],
    )


class TestCanonicalTTSPipeline:
    def test_validates_and_auto_inserts_mono(self, registry, tmp_path) -> None:
        ir = _tts_clean_ir("/tmp/manifest.jsonl", str(tmp_path))
        report = validate(ir, registry, mutate=True)
        assert report.is_ok(), report.findings
        # Auto-insert should have prepended MonoConversionStage before VAD
        # (VAD requires the in-memory waveform).
        stage_names = [s.stage for s in report.ir.stages]
        # ManifestReader is the first, then potentially MonoConversion via card-driven insertion.
        assert "VADSegmentationStage" in stage_names
        # No errors implies fingerprint deterministically populated.
        assert report.fingerprint
        # Idempotent fingerprint across two validate calls.
        report2 = validate(ir, registry, mutate=True)
        assert report.fingerprint == report2.fingerprint

    def test_compiles_to_canonical_stages_yaml(self, registry, tmp_path) -> None:
        ir = _tts_clean_ir("/tmp/manifest.jsonl", str(tmp_path))
        report = validate(ir, registry, mutate=True)
        text = compile_ir_to_yaml(report.ir, registry)
        compiled = yaml.safe_load(text)
        assert "stages" in compiled
        assert isinstance(compiled["stages"], list)
        # Every stage must have a _target_ that imports as expected.
        for s in compiled["stages"]:
            assert "_target_" in s
            assert s["_target_"].startswith("nemo_curator.stages.audio.")

    def test_resource_xor_blocked(self, registry, tmp_path) -> None:
        from pydantic import ValidationError

        # ResourceSpec is constructed once at object-creation time; the
        # validator never sees the IR because ResourceSpec raises first.
        with pytest.raises(ValidationError):
            PipelineIR(
                source=SourceSpec(kind="manifest", uri="/tmp/m.jsonl"),
                sink=SinkSpec(target_dir=str(tmp_path)),
                stages=[
                    stage("ManifestReader", manifest_path="/tmp/m.jsonl").model_copy(
                        update={"resources": ResourceSpec(cpus=1.0, gpus=1.0, gpu_memory_gb=8.0)}
                    ),
                ],
            )


class TestPlanCLI:
    def test_plan_round_trip_via_path(self, registry, tmp_path) -> None:
        ir_path = tmp_path / "ir.json"
        ir = _tts_clean_ir(str(tmp_path / "manifest.jsonl"), str(tmp_path / "out"))
        ir.write(ir_path)
        loaded = PipelineIR.from_path(ir_path)
        assert loaded.name == ir.name
        assert len(loaded.stages) == len(ir.stages)
