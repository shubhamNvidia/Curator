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
"""Tests for the eight Layer-2 validator checks."""

from __future__ import annotations

import pytest

from nemo_curator.agentic.intent import IntentCategories, Policy
from nemo_curator.agentic.ir import PipelineIR, SinkSpec, SourceSpec, stage
from nemo_curator.agentic.registry import build_registry
from nemo_curator.agentic.validator import Severity, validate


@pytest.fixture(scope="module")
def registry():
    return build_registry()


def _ir(*stages) -> PipelineIR:
    return PipelineIR(
        source=SourceSpec(kind="manifest", uri="/tmp/manifest.jsonl"),
        sink=SinkSpec(target_dir="/tmp/out"),
        stages=list(stages),
    )


class TestStageResolution:
    def test_unknown_stage_is_error(self, registry) -> None:
        report = validate(_ir(stage("NotAStage")), registry)
        codes = {f.code for f in report.errors()}
        assert "stage_not_registered" in codes
        assert not report.is_ok()


class TestParams:
    def test_unknown_param_is_warning(self, registry) -> None:
        report = validate(
            _ir(
                stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
                stage("VADSegmentationStage", threshold=0.5, not_a_real_param=True),
                stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
            ),
            registry,
        )
        codes = {f.code for f in report.findings}
        assert "unknown_param" in codes
        # Warnings alone do not block.
        assert report.is_ok()

    def test_out_of_range_is_error(self, registry) -> None:
        report = validate(
            _ir(
                stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
                stage("VADSegmentationStage", threshold=1.5),  # > max=1.0
                stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
            ),
            registry,
        )
        codes = {f.code for f in report.errors()}
        assert "param_over_max" in codes

    def test_choice_violation_is_error(self, registry) -> None:
        report = validate(
            _ir(
                stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
                stage("BandFilterStage", band_value="wide_band"),  # not in choices
                stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
            ),
            registry,
        )
        codes = {f.code for f in report.errors()}
        assert "param_choice_violated" in codes


class TestAutoInsert:
    def test_inserts_resample_then_mono_before_vad(self, registry) -> None:
        """VAD needs an in-memory waveform — that requires Resample (actual SR convert)
        chained with Mono (channel collapse + verify SR), in that order, so heterogeneous
        source corpora don't silently lose files to Mono's strict_sample_rate filter."""
        ir = _ir(
            stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
            stage("VADSegmentationStage"),
            stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
        )
        report = validate(ir, registry, mutate=True)
        names = [s.stage for s in report.ir.stages]
        assert "ResampleAudioStage" in names and "MonoConversionStage" in names
        idx_resample = names.index("ResampleAudioStage")
        idx_mono = names.index("MonoConversionStage")
        idx_vad = names.index("VADSegmentationStage")
        assert idx_resample < idx_mono < idx_vad

        resample = report.ir.stages[idx_resample]
        # Resample MUST overwrite the audio_filepath key so downstream Mono can pick up the resampled file.
        assert resample.params.get("resampled_audio_filepath_key") == "audio_filepath"
        assert resample.auto_inserted is True

        mono = report.ir.stages[idx_mono]
        assert mono.auto_inserted is True

    def test_mutate_false_surfaces_precondition_error(self, registry) -> None:
        ir = _ir(
            stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
            stage("WhisperXVADStage"),
            stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
        )
        report = validate(ir, registry, mutate=False)
        assert any(f.code == "precondition_unsatisfied" for f in report.errors())

    def test_mutate_true_inserts_resample_for_16k(self, registry) -> None:
        """WhisperXVADStage requires SR=16000 — chain must target that, not 48000."""
        ir = _ir(
            stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
            stage("WhisperXVADStage"),
            stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
        )
        report = validate(ir, registry, mutate=True)
        names = [s.stage for s in report.ir.stages]
        assert "ResampleAudioStage" in names
        idx_resample = names.index("ResampleAudioStage")
        idx_vad = names.index("WhisperXVADStage")
        assert idx_resample < idx_vad
        resample = report.ir.stages[idx_resample]
        assert resample.params["target_sample_rate"] == 16000

    def test_existing_normalizer_pair_is_retuned_not_duplicated(self, registry) -> None:
        """When the IR already carries a Resample+Mono pair (e.g. from the
        deterministic planner because the user asked for 48 kHz output) and a
        downstream stage requires a DIFFERENT SR, the validator must retune
        the existing pair instead of emitting a second Resample+Mono chain."""

        ir = _ir(
            stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
            stage(
                "ResampleAudioStage",
                resampled_audio_dir="/tmp/_resampled",
                target_sample_rate=48000,
                target_nchannels=1,
                target_format="wav",
                resampled_audio_filepath_key="audio_filepath",
            ),
            stage(
                "MonoConversionStage",
                output_sample_rate=48000,
                strict_sample_rate=True,
            ),
            stage("WhisperXVADStage"),  # requires_sample_rate=16000
            stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
        )
        report = validate(ir, registry, mutate=True)
        names = [s.stage for s in report.ir.stages]
        assert names.count("ResampleAudioStage") == 1, names
        assert names.count("MonoConversionStage") == 1, names
        resample = report.ir.stages[names.index("ResampleAudioStage")]
        mono = report.ir.stages[names.index("MonoConversionStage")]
        assert resample.params["target_sample_rate"] == 16000
        assert mono.params["output_sample_rate"] == 16000


class TestLicenseGate:
    def test_commercial_only_allows_apache(self, registry) -> None:
        report = validate(
            _ir(
                stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
                stage("VADSegmentationStage"),
                stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
            ),
            registry,
            intent=IntentCategories(policy=Policy(commercial_only=True)),
        )
        codes = {f.code for f in report.errors()}
        assert "commercial_only_violation" not in codes
        assert "commercial_only_model_block" not in codes


class TestBoundary:
    def test_missing_source_autoinserts_reader_when_mutate(self, registry) -> None:
        """When mutate=True, a non-source first stage triggers reader auto-insert."""
        report = validate(
            _ir(
                stage("VADSegmentationStage"),
                stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
            ),
            registry,
            mutate=True,
        )
        warnings = {f.code for f in report.warnings()}
        assert "missing_source" not in warnings
        info_codes = [f.code for f in report.findings if f.severity.value == "info"]
        assert "auto_inserted" in info_codes
        assert report.ir.stages[0].stage == "ManifestReader"
        assert report.ir.stages[0].auto_inserted is True

    def test_missing_source_warns_when_no_mutate(self, registry) -> None:
        """When mutate=False, the validator only reports the warning and never rewrites."""
        report = validate(
            _ir(
                stage("VADSegmentationStage"),
                stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
            ),
            registry,
            mutate=False,
        )
        warnings = {f.code for f in report.warnings()}
        assert "missing_source" in warnings
        assert report.ir.stages[0].stage == "VADSegmentationStage"

    def test_missing_sink_autoinserts_writer_when_mutate(self, registry) -> None:
        """A pipeline missing ManifestWriterStage gets one appended automatically."""
        report = validate(
            _ir(
                stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
                stage("VADSegmentationStage"),
            ),
            registry,
            mutate=True,
        )
        warnings = {f.code for f in report.warnings()}
        assert "missing_sink" not in warnings
        assert report.ir.stages[-1].stage == "ManifestWriterStage"
        assert report.ir.stages[-1].auto_inserted is True

    def test_missing_timestamp_mapper_autoinserts_before_extract(self, registry) -> None:
        """SegmentExtractionStage requires original_*_ms; if no upstream stage produces
        them, TimestampMapperStage must be auto-prepended."""
        report = validate(
            _ir(
                stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
                stage("VADSegmentationStage"),
                stage("SegmentExtractionStage", output_dir="/tmp/out", output_format="wav"),
                stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
            ),
            registry,
            mutate=True,
        )
        names = [s.stage for s in report.ir.stages]
        assert "TimestampMapperStage" in names
        assert names.index("TimestampMapperStage") < names.index("SegmentExtractionStage")
        tm = report.ir.stages[names.index("TimestampMapperStage")]
        assert tm.auto_inserted is True

    def test_missing_sink_is_warning_when_no_mutate(self, registry) -> None:
        """With mutate=False the validator only warns; it never rewrites the IR."""
        report = validate(
            _ir(
                stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
                stage("VADSegmentationStage"),
            ),
            registry,
            mutate=False,
        )
        warnings = {f.code for f in report.warnings()}
        assert "missing_sink" in warnings


class TestCardinality:
    def test_fan_in_not_at_end_is_error(self, registry) -> None:
        # AudioToDocumentStage is N:1 — placing it before the writer should fail.
        report = validate(
            _ir(
                stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
                stage("AudioToDocumentStage"),
                stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
            ),
            registry,
        )
        codes = {f.code for f in report.errors()}
        assert "fan_in_not_at_end" in codes


class TestFingerprintIdempotence:
    def test_same_ir_same_fingerprint(self, registry) -> None:
        ir = _ir(
            stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
            stage("VADSegmentationStage", threshold=0.5),
            stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
        )
        r1 = validate(ir, registry, mutate=True)
        r2 = validate(ir, registry, mutate=True)
        assert r1.fingerprint == r2.fingerprint
        assert r1.fingerprint  # not empty


class TestTopoSort:
    def test_already_valid_order_is_left_alone(self, registry) -> None:
        """A topologically correct IR must not be reordered."""
        ir = _ir(
            stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
            stage("VADSegmentationStage"),
            stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
        )
        report = validate(ir, registry, mutate=True)
        codes = {f.code for f in report.findings}
        assert "stage_reordered" not in codes

    def test_scrambled_order_gets_repaired(self, registry) -> None:
        """SegmentExtractionStage requires keys from TimestampMapperStage. When the
        user emits them in the wrong order (or omits the mapper), the validator
        either reorders or auto-inserts to make TimestampMapper precede the
        extraction. Either path is fine — we just assert the final IR is valid."""
        ir = _ir(
            stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
            stage("VADSegmentationStage"),
            stage("SegmentExtractionStage", output_dir="/tmp/segs", output_format="wav"),
            stage("TimestampMapperStage"),
            stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
        )
        report = validate(ir, registry, mutate=True)
        names = [s.stage for s in report.ir.stages]
        # TimestampMapper must precede SegmentExtraction in the repaired order.
        tm_pos = names.index("TimestampMapperStage")
        ex_pos = names.index("SegmentExtractionStage")
        assert tm_pos < ex_pos
        # SOME repair must have happened — either a topo reorder or an
        # auto-insert. Validate at least one of those is in findings.
        repair_codes = {f.code for f in report.findings}
        assert repair_codes & {"stage_reordered", "auto_inserted"}


class TestOrderingHints:
    def test_violation_emits_info_finding(self, registry, tmp_path) -> None:
        """A stage whose ordering_hints.prefer_after is not satisfied should
        get an INFO finding. We patch a card in-memory for the test."""
        from nemo_curator.agentic.cards import OrderingHints, SelectionHints

        # Make VAD prefer to run AFTER SpeakerSeparationStage. (Soft pref —
        # not implied by I/O contracts.)
        vad_entry = registry.get("VADSegmentationStage")
        vad_entry.card.selection_hints = SelectionHints(
            ordering_hints=OrderingHints(
                prefer_after=["SpeakerSeparationStage"],
                rationale="test hint",
            )
        )
        try:
            report = validate(
                _ir(
                    stage("ManifestReader", manifest_path="/tmp/m.jsonl"),
                    stage("VADSegmentationStage"),
                    stage("SpeakerSeparationStage", exclude_overlaps=True),
                    stage("ManifestWriterStage", output_path="/tmp/out.jsonl"),
                ),
                registry,
                mutate=True,
            )
            infos = [f for f in report.findings if f.code == "ordering_preference"]
            assert infos, f"expected an ordering_preference INFO, got {report.findings}"
            assert "prefers to run AFTER" in infos[0].detail
        finally:
            # Reset so we don't pollute other tests in this module.
            vad_entry.card.selection_hints = SelectionHints()
