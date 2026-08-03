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

"""Compatibility lock for the stage surface consumed by audio tutorials.

Agent hardening must not silently change the registered audio stages, their
constructor defaults, or their static data contracts.  This snapshot is based
on the reviewed working-tree state, not an older Git revision.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

from nemo_curator.stages.audio._agent_registry import static_contract
from nemo_curator.stages.audio._catalog import (
    get_agent_ready_stage_class,
    list_agent_ready_stages,
)

LEGACY_STAGE_NAMES = (
    "ALMDataBuilderStage",
    "ALMDataOverlapStage",
    "AudioDataFilterStage",
    "AudioToDocumentStage",
    "BandFilterStage",
    "BandwidthEstimationStage",
    "ChineseConversionStage",
    "ComputeWERStage",
    "CreateInitialManifestAudioFolderStage",
    "CreateInitialManifestFleursStage",
    "CreateInitialManifestReadSpeechStage",
    "GetAudioDurationStage",
    "GetPairwiseWerStage",
    "InferenceAsrNemoStage",
    "InferenceSortformerStage",
    "InverseTextNormalizationStage",
    "JoinSplitAudioMetadataStage",
    "ManifestGroupExportStage",
    "ManifestReader",
    "ManifestReaderStage",
    "ManifestWriterStage",
    "MergeAlignmentDiarizationStage",
    "MonoConversionStage",
    "NeMoASRAlignerStage",
    "OverlapFilterStage",
    "PrepareModuleSegmentsStage",
    "PreserveByValueStage",
    "PretrainMetricsAggregatorStage",
    "PyAnnoteDiarizationStage",
    "ReadLongFormManifestStage",
    "ResampleAudioStage",
    "SIGMOSFilterStage",
    "SegmentConcatenationStage",
    "SegmentExtractionStage",
    "SnippetCutPlannerStage",
    "SnippetExtractionStage",
    "SnippetManifestWriterStage",
    "SnippetRepetitionFilterStage",
    "SpeakerSeparationStage",
    "SplitASRAlignJoinStage",
    "SplitLongAudioStage",
    "TimestampMapperStage",
    "TorchSquimQualityMetricsStage",
    "UTMOSFilterStage",
    "VADSegmentationStage",
    "WhisperXVADStage",
)

ADDITIVE_STAGE_NAMES = ("DocumentBatchJsonlWriterStage",)

# Refreshed 2026-08-03 for the agentification stage extensions, which are ADDITIVE and
# backward-compatible (the 46-name set is unchanged -- see the names test above):
#   * diarizers now also write a derived `num_speakers` scalar (Sortformer/PyAnnote),
#   * residency: several stages accept a resident waveform in addition to a file path
#     (input_residency / waveform_key / sample_rate_key) -- a SUPERSET of the old file read,
#   * opt-in on-disk output knobs (write_to_disk / keep_waveform_in_task / *_dir).
# No stage lost a param, a read/write key, or changed an existing default. Regenerate this
# value ONLY after confirming (git diff of describe()/defaults) that a change is additive.
EXPECTED_LEGACY_COMPATIBILITY_SHA256 = (
    "0453b6a641a302b417e6a6e44b162e64246f4cac3685a5c700e52b59d0659078"
)


def _normalize(value: Any) -> Any:  # noqa: ANN401
    """Remove machine-specific roots while preserving semantic defaults."""
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_normalize(item) for item in value)
    if isinstance(value, str):
        repo_root = str(Path(__file__).resolve().parents[2])
        home_root = str(Path.home().resolve())
        return value.replace(repo_root, "<REPO>").replace(home_root, "<HOME>")
    return value


def _constructor_defaults(cls: type) -> dict[str, Any]:
    """Capture public constructor defaults, including dataclass factories.

    The agent's static parameter surface intentionally hides executor knobs such
    as ``resources`` and ``batch_size``. Those values still affect runtime
    behavior, so they need a separate compatibility lock.
    """
    if is_dataclass(cls):
        defaults: dict[str, Any] = {}
        for item in fields(cls):
            if not item.init or item.name.startswith("_"):
                continue
            if item.default_factory is not MISSING:
                value = item.default_factory()
            elif item.default is not MISSING:
                value = item.default
            else:
                value = "<required>"
            defaults[item.name] = _normalize(value)
        return defaults

    defaults = {}
    for name, parameter in inspect.signature(cls).parameters.items():
        if name.startswith("_"):
            continue
        value = (
            "<required>"
            if parameter.default is inspect.Parameter.empty
            else parameter.default
        )
        defaults[name] = _normalize(value)
    return defaults


def _compatibility_payload() -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    # Additive opt-in stages must not force the legacy compatibility hash to
    # move. The payload intentionally remains scoped to the original 46.
    for name in LEGACY_STAGE_NAMES:
        cls = get_agent_ready_stage_class(name)
        contract = static_contract(cls).to_dict()
        params = [
            {
                key: parameter.get(key)
                for key in ("name", "type", "default", "required", "choices", "role")
            }
            for parameter in contract["params"]
        ]
        payload.append(
            {
                "name": name,
                "target": f"{cls.__module__}.{cls.__qualname__}",
                "constructor_defaults": _constructor_defaults(cls),
                "params": _normalize(params),
                "reads": _normalize(contract["reads"]),
                "writes": _normalize(contract["writes"]),
                "reads_one_of": _normalize(contract["reads_one_of"]),
                "cardinality": contract["cardinality"],
                "cardinality_options": contract["cardinality_options"],
                "preserves_upstream_keys": contract["preserves_upstream_keys"],
                "gates": _normalize(contract["gates"]),
                "dispatch": contract["dispatch"],
                "batch_only": contract["batch_only"],
                "accepts_task_type": contract["accepts_task_type"],
                "produces_task_type": contract["produces_task_type"],
                "removes_keys": contract["removes_keys"],
                "key_defaults": {
                    parameter["name"]: _normalize(parameter.get("default"))
                    for parameter in contract["params"]
                    if parameter["name"].endswith("_key")
                },
            }
        )
    return payload


def test_registered_stage_names_are_backward_compatible() -> None:
    actual = set(list_agent_ready_stages())
    legacy = set(LEGACY_STAGE_NAMES)
    additive = set(ADDITIVE_STAGE_NAMES)

    assert len(LEGACY_STAGE_NAMES) == 46
    assert legacy <= actual
    assert actual - legacy == additive
    assert len(actual) == len(legacy) + len(additive)


def test_stage_defaults_and_static_contracts_are_backward_compatible() -> None:
    blob = json.dumps(
        _compatibility_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    actual = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    assert actual == EXPECTED_LEGACY_COMPATIBILITY_SHA256, (
        "The legacy 46-stage compatibility surface changed. If intentional, inspect "
        "constructor defaults and static contracts before updating this baseline. "
        f"Expected {EXPECTED_LEGACY_COMPATIBILITY_SHA256}, got {actual}."
    )


def test_document_batch_writer_is_explicitly_additive() -> None:
    cls = get_agent_ready_stage_class("DocumentBatchJsonlWriterStage")
    contract = static_contract(cls)

    assert contract.accepts_task_type == "DocumentBatch"
    assert contract.produces_task_type == "DocumentBatch"
    assert [parameter.name for parameter in contract.params] == ["output_path"]


def test_intentional_legacy_contract_corrections_are_explicit() -> None:
    audio_filter = static_contract(get_agent_ready_stage_class("AudioDataFilterStage"))
    snippet_writer = static_contract(get_agent_ready_stage_class("SnippetManifestWriterStage"))

    assert audio_filter.accepts_task_type == "AudioTask"
    assert audio_filter.produces_task_type == "AudioTask"
    assert snippet_writer.gates.requires_serializable_input is True
