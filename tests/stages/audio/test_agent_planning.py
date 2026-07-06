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

"""Tests for validate_pipeline — the agent's pipeline composition safety net."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_curator.stages.audio._agent_ready import AgentReady, StageContract
from nemo_curator.stages.audio._planning import validate_pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage
from nemo_curator.stages.audio.filtering.utmos import UTMOSFilterStage
from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage
from nemo_curator.stages.resources import Resources

if TYPE_CHECKING:
    from pathlib import Path


class _CompositeStub(AgentReady):
    """Stands in for a composite (e.g. AudioDataFilterStage) without the heavy import."""

    def describe(self) -> StageContract:
        return StageContract(wrappable=False)


def test_valid_chain_passes_and_accumulates_roles() -> None:
    report = validate_pipeline([MonoConversionStage(), GetAudioDurationStage(), UTMOSFilterStage()])
    assert report.ok, report.summary()
    assert not report.errors
    # roles produced along the way are available to downstream stages
    assert {"waveform", "sample_rate", "duration"} <= report.produced_roles


def test_missing_input_role_is_an_error() -> None:
    # GetAudioDuration needs an audio_filepath; with no initial roles nothing produces it.
    report = validate_pipeline([GetAudioDurationStage()], initial_roles=set())
    assert not report.ok
    assert len(report.errors) == 1
    err = report.errors[0]
    assert err.code == "unsatisfied_reads"
    assert err.stage_index == 0
    assert "audio_filepath" in err.message


def test_reads_one_of_is_satisfied_by_any_alternative() -> None:
    # UTMOS reads_one_of {waveform+sr | audio_filepath | segments}; audio_filepath suffices.
    report = validate_pipeline([UTMOSFilterStage()], initial_roles={"audio_filepath"})
    assert report.ok, report.summary()


def test_gpu_gate_warns_when_no_gpu_available() -> None:
    gpu_stage = UTMOSFilterStage(resources=Resources(cpus=1.0, gpus=1.0))
    report = validate_pipeline([gpu_stage], initial_roles={"audio_filepath"}, available_gpus=0)
    assert report.ok  # gate issues are warnings, not errors
    assert any(i.code == "gpu_unavailable" for i in report.warnings)


def test_gpu_gate_silent_when_gpus_unspecified() -> None:
    gpu_stage = UTMOSFilterStage(resources=Resources(cpus=1.0, gpus=1.0))
    report = validate_pipeline([gpu_stage], initial_roles={"audio_filepath"})  # available_gpus=None
    assert not any(i.code == "gpu_unavailable" for i in report.warnings)


def test_composite_is_flagged_not_errored() -> None:
    # A composite (wrappable=False, e.g. AudioDataFilterStage) is a warning, not an error,
    # and the validator does not try to reason about its hidden internal data flow.
    report = validate_pipeline([_CompositeStub()])
    assert report.ok
    assert any(i.code == "composite" for i in report.warnings)


def test_reads_after_composite_warn_instead_of_error() -> None:
    # A composite hides its writes, so a downstream read that is not visibly
    # satisfied must not hard-fail a possibly-runnable pipeline.
    report = validate_pipeline([_CompositeStub(), GetAudioDurationStage()], initial_roles=set())
    assert report.ok, report.summary()
    assert any(i.code == "unsatisfied_reads_after_composite" for i in report.warnings)


def test_renamed_producer_key_dangles_but_role_still_ok() -> None:
    # Rename the producer's output key and leave the consumer on its default:
    # the ROLE still matches (ok=True) but the literal key never connects, so
    # keys_ok=False with a dangling_key warning names the dangling read.
    mono = MonoConversionStage(waveform_key="pcm_data")
    consumer = GetAudioDurationStage(audio_filepath_key="somewhere_else")
    report = validate_pipeline([mono, consumer], initial_roles={"audio_filepath"}, initial_keys={"audio_filepath"})
    assert report.ok, report.summary()
    assert not report.keys_ok
    assert any(i.code == "dangling_key" and "somewhere_else" in i.message for i in report.warnings)


def test_seeded_initial_keys_satisfy_the_literal_check() -> None:
    # The same consumer read is fine when the input manifest actually carries the key.
    consumer = GetAudioDurationStage(audio_filepath_key="somewhere_else")
    report = validate_pipeline(
        [consumer], initial_roles={"audio_filepath"}, initial_keys={"somewhere_else"}
    )
    assert report.ok, report.summary()
    assert report.keys_ok, report.summary()


def test_tensor_into_raw_json_sink_warns_and_sanitizer_clears_it(tmp_path: Path) -> None:
    from nemo_curator.stages.audio.common import ManifestWriterStage
    from nemo_curator.stages.audio.io.convert import AudioToDocumentStage

    # Mono keeps the waveform tensor resident; the raw json.dumps sink must be warned about.
    mono = MonoConversionStage(keep_waveform_in_task=True)
    writer = ManifestWriterStage(output_path=str(tmp_path / "agent_planning_test.jsonl"))
    report = validate_pipeline([mono, writer])
    assert report.ok  # advisory, not an error
    assert any(i.code == "tensor_into_sink" for i in report.warnings)

    # The sanitizing document converter clears the resident-tensor hazard.
    report2 = validate_pipeline([mono, AudioToDocumentStage(), writer])
    assert not any(i.code == "tensor_into_sink" for i in report2.warnings), report2.summary()
