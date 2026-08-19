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

ADDITIVE_STAGE_NAMES = (
    "ChannelCountStage",
    "DocumentBatchJsonlWriterStage",
    "ManifestCheckpointStage",
    "PreserveByValueConditionsStage",
    "SampleRateFilterStage",
)

# Refreshed 2026-08-10 for delta reuse. Verified additive by dumping the whole payload from
# this tree and from HEAD and diffing them: every line is an addition except two empty
# collections that gained entries, plus one correction noted below.
#   * gates.per_row_independent appears on all 46 (null where undeclared, which is what a
#     delta refuses on), and is true/false on the stages that now declare it,
#   * ManifestReader / ManifestReaderStage / CreateInitialManifestAudioFolderStage gained the
#     optional include_files (and include_files_key), whose default of None reads the whole
#     corpus exactly as before,
#   * ManifestReaderStage's STATIC gates now say lifecycle_side_effects=True. Not a behaviour
#     change: its describe() has always said so, and only the instance-free view disagreed.
# No stage lost a param, a key, a default, or a read/write.
#
# Refreshed 2026-08-07: disk-writing stages now DECLARE their output path params via the
# additive Gates.output_path_params, replacing a central table in verbs.py that every new
# writer had to be added to. Verified additive by diffing the full payload before/after:
# the only field that moved is gates.output_path_params, and only from absent to a declared
# list. No stage lost a param, a key, a default, or a read/write.
#
# Refreshed 2026-08-03 for the agentification stage extensions, which are ADDITIVE and
# backward-compatible (the 46-name set is unchanged -- see the names test above):
#   * diarizers now also write a derived `num_speakers` scalar (Sortformer/PyAnnote),
#   * residency: several stages accept a resident waveform in addition to a file path
#     (input_residency / waveform_key / sample_rate_key) -- a SUPERSET of the old file read,
#   * opt-in on-disk output knobs (write_to_disk / keep_waveform_in_task / *_dir).
# No stage lost a param, a read/write key, or changed an existing default. Regenerate this
# value ONLY after confirming (git diff of describe()/defaults) that a change is additive.
#
# Refreshed 2026-08-19 for exact model-filter separation. PreserveByValueStage gained the
# optional missing_value_policy="error" parameter and float target typing; both preserve its
# runtime default. The compound selector is additive and excluded from this legacy payload,
# so its additive condition_logic="and" constructor default does not change this hash.
#
# Refreshed 2026-08-19 for resume task-identity safety. Gates gained the additive
# requires_stable_task_id=False default; SnippetExtractionStage opts in from its configured
# contract because its task.task_id fallback enters durable snippet/member names. Constructor
# defaults, reads/writes, and runtime behavior are otherwise unchanged.
EXPECTED_LEGACY_COMPATIBILITY_SHA256 = (
    "0fdca099dd04c6a5b8358e90efdc4aa711076ff606446633f99b8e26b7e3edae"
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


def _shipped_agent_ready_stages() -> set[str]:
    """Agent-ready stages that NeMo Curator itself ships.

    The registry is deliberately open: any imported subclass of an agent-ready base
    registers itself, which is what lets a user extend the catalog. It also means the
    set is not a closed world -- ``tests/stages/audio`` defines ``ConcreteASRProcessor``
    as a test double, so whether it appears here depends on which suites were imported
    first. Compatibility is a claim about what Curator ships, so foreign classes are
    excluded rather than asserted about.
    """
    return {
        name
        for name in list_agent_ready_stages()
        if getattr(get_agent_ready_stage_class(name), "__module__", "").startswith("nemo_curator.")
    }


def test_registered_stage_names_are_backward_compatible() -> None:
    actual = _shipped_agent_ready_stages()
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


def test_manifest_checkpoint_is_explicitly_additive() -> None:
    cls = get_agent_ready_stage_class("ManifestCheckpointStage")
    contract = static_contract(cls)

    assert contract.accepts_task_type == "AudioTask"
    assert contract.produces_task_type == "AudioTask"
    assert [parameter.name for parameter in contract.params] == [
        "output_path",
        "retention_sec",
        "owner",
        "planning_provenance",
    ]
    assert contract.gates.writes_to_disk is True
    assert contract.gates.output_path_params == ["output_path"]
    assert contract.gates.requires_serializable_input is True
    assert contract.gates.per_row_independent is True
    assert contract.gates.lifecycle_side_effects is True


def test_compound_value_selector_is_explicitly_additive() -> None:
    cls = get_agent_ready_stage_class("PreserveByValueConditionsStage")
    contract = static_contract(cls)

    assert contract.accepts_task_type == "AudioTask"
    assert contract.produces_task_type == "AudioTask"
    assert [parameter.name for parameter in contract.params] == [
        "conditions",
        "missing_value_policy",
        "items_key",
        "drop_parent_if_empty",
        "condition_logic",
    ]
    assert contract.params[-1].default == "and"
    assert contract.params[-1].choices == ["and", "or"]
    assert contract.batch_only is True


def test_compound_value_selector_catalog_exposes_condition_logic() -> None:
    from nemo_curator.stages.audio._catalog import audio_stage_catalog

    entry = next(
        item
        for item in audio_stage_catalog()
        if item["name"] == "PreserveByValueConditionsStage"
    )
    parameter = next(
        item
        for item in entry["contract"]["params"]
        if item["name"] == "condition_logic"
    )
    schema = entry["params_schema"]["properties"]["condition_logic"]

    assert parameter["default"] == "and"
    assert parameter["choices"] == ["and", "or"]
    assert schema["default"] == "and"
    assert schema["enum"] == ["and", "or"]


def test_intentional_legacy_contract_corrections_are_explicit() -> None:
    audio_filter = static_contract(get_agent_ready_stage_class("AudioDataFilterStage"))
    snippet_writer = static_contract(get_agent_ready_stage_class("SnippetManifestWriterStage"))

    assert audio_filter.accepts_task_type == "AudioTask"
    assert audio_filter.produces_task_type == "AudioTask"
    assert snippet_writer.gates.requires_serializable_input is True
