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

"""Tests for the agent-ready foundation: param derivation, semantic roles,
contract serialization, the discovery catalog, and the conformance gate.

These exercise the shared machinery (``_agent_ready``, ``_agent_registry``,
``_roles``, ``_catalog``, ``_conformance``) without GPUs. The catalog sweep
imports the audio stage modules; optional heavy deps that are absent are skipped
by the catalog rather than failing the test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import pytest

from nemo_curator.stages.audio._agent_ready import (
    AgentReady,
    Gates,
    IOSpec,
    ParamSpec,
    StageContract,
    StaticHints,
    to_json_schema,
)
from nemo_curator.stages.audio._agent_registry import (
    _parse_args_section,
    build_contract,
    stage_params,
    static_contract,
)
from nemo_curator.stages.audio._conformance import (
    assert_agent_ready,
    assert_contract_wellformed,
    produced_roles,
    reads_satisfied_by_role,
)


# --------------------------------------------------------------------------- #
# Toy stages (deterministic; no heavy imports)
# --------------------------------------------------------------------------- #
@dataclass
class _ToyDataclassStage(AgentReady):
    """A toy dataclass stage.

    Args:
        audio_filepath_key: Key for the audio path.
        score_key: Where the score is written.
        mode: Operating mode.
        threshold: Minimum score; None disables.
    """

    audio_filepath_key: str = "audio_filepath"
    score_key: str = "utmos_mos"
    mode: Literal["task", "segments", "auto"] = "auto"
    threshold: float | None = 3.5
    name: str = "ToyDataclassStage"  # framework field -> excluded from params

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]),
            writes=IOSpec(data_keys=[self.score_key]),
            cardinality="1:1",
        )


class _ToyInitStage(AgentReady):
    """A toy stage that uses ``__init__`` (not a dataclass).

    Args:
        input_value_key: Field to evaluate.
        target_value: Value to compare against.
        operator: Comparison operator.
    """

    def __init__(self, input_value_key: str, target_value: int, operator: str = "eq"):
        self.input_value_key = input_value_key
        self.target_value = target_value
        self.operator = operator

    def describe(self) -> StageContract:
        return StageContract(reads=IOSpec(data_keys=[self.input_value_key]), cardinality="filter")


# --------------------------------------------------------------------------- #
# Param derivation
# --------------------------------------------------------------------------- #
def test_stage_params_dataclass_choices_required_roles_descriptions():
    params = {p.name: p for p in stage_params(_ToyDataclassStage)}
    assert "name" not in params  # framework field excluded
    assert params["mode"].choices == ["task", "segments", "auto"]
    assert params["mode"].type == "str"
    assert params["threshold"].type == "float | None"
    assert params["threshold"].default == 3.5
    assert all(not p.required for p in params.values())
    assert params["audio_filepath_key"].role == "audio_filepath"
    assert params["score_key"].role == "score"
    assert params["audio_filepath_key"].description == "Key for the audio path."
    assert params["threshold"].description == "Minimum score; None disables."


def test_stage_params_init_signature_required_and_defaults():
    params = {p.name: p for p in stage_params(_ToyInitStage)}
    assert params["input_value_key"].required is True
    assert params["target_value"].required is True
    assert params["operator"].required is False
    assert params["operator"].default == "eq"
    assert params["input_value_key"].description == "Field to evaluate."


def test_docstring_args_parser_handles_continuations_and_sections():
    doc = """Summary.

    Args:
        a: first.
        b: spans
            two lines.

    Returns:
        not a param.
    """
    parsed = _parse_args_section(doc)
    assert parsed["a"] == "first."
    assert parsed["b"] == "spans two lines."
    assert "not a param" not in str(parsed)


# --------------------------------------------------------------------------- #
# Contract assembly + serialization
# --------------------------------------------------------------------------- #
def test_build_contract_fills_params_and_key_roles():
    contract = build_contract(_ToyDataclassStage(score_key="renamed_score"))
    assert contract.contract_resolution == "configured"
    assert contract.params, "params should be auto-derived"
    assert contract.key_roles.get("renamed_score") == "score"
    assert contract.key_roles.get("audio_filepath") == "audio_filepath"
    assert contract.stage_id == "_ToyDataclassStage"


def test_static_contract_is_instance_free_for_required_arg_stage():
    contract = static_contract(_ToyInitStage)  # no instantiation needed
    assert contract.contract_resolution == "static_params_and_hints"
    names = {p.name for p in contract.params}
    assert {"input_value_key", "target_value", "operator"} <= names
    assert contract.stage_id == "_ToyInitStage"


def test_to_dict_is_json_safe_even_with_nonserializable_default():
    class _Weird:
        pass

    contract = StageContract(params=[ParamSpec(name="x", default=_Weird(), choices=["a"], role="score")])
    payload = contract.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["params"][0]["default"].startswith("<non-serializable")


def test_to_json_schema_maps_types_enums_and_required():
    schema = to_json_schema(
        [
            ParamSpec(name="mode", type="str", choices=["a", "b"], default="a"),
            ParamSpec(name="path", type="str", required=True),
            ParamSpec(name="n", type="int", default=1),
        ]
    )
    assert schema["type"] == "object"
    assert schema["properties"]["mode"]["enum"] == ["a", "b"]
    assert schema["properties"]["n"]["type"] == "integer"
    assert schema["required"] == ["path"]


# --------------------------------------------------------------------------- #
# By-role matching
# --------------------------------------------------------------------------- #
def test_reads_satisfied_by_role_survives_key_rename():
    # Producer writes a score under a renamed key value.
    producer = build_contract(_ToyDataclassStage(score_key="model_b_mos"))
    # Consumer reads a score under a *different* key value.
    consumer = StageContract(
        reads=IOSpec(data_keys=["some_other_name"]),
        key_roles={"some_other_name": "score"},
    )
    assert "score" in produced_roles(producer)
    assert reads_satisfied_by_role(consumer, produced_roles(producer)) is True
    # A consumer needing 'waveform' is not satisfied by a 'score'-only producer.
    needs_waveform = StageContract(reads=IOSpec(data_keys=["wf"]), key_roles={"wf": "waveform"})
    assert reads_satisfied_by_role(needs_waveform, produced_roles(producer)) is False


def test_assert_agent_ready_static_and_dynamic_on_toy_stage():
    contract = assert_agent_ready(
        _ToyDataclassStage(),
        fixture_factory=None,  # static-only (no execution)
        expected_cardinality="1:1",
        available_keys={"audio_filepath"},
    )
    assert contract.cardinality == "1:1"


# --------------------------------------------------------------------------- #
# Catalog + full-stack static conformance sweep
# --------------------------------------------------------------------------- #
def test_catalog_discovers_stages_and_round_trips_json():
    from nemo_curator.stages.audio._catalog import audio_stage_catalog, catalog_as_json, list_agent_ready_stages

    names = list_agent_ready_stages()
    assert len(names) >= 30, f"expected the audio catalog to discover many stages, got {len(names)}"
    payload = catalog_as_json()
    parsed = json.loads(payload)  # must round-trip
    assert len(parsed) == len(names)
    # representative stages are present
    assert "MonoConversionStage" in names
    assert "UTMOSFilterStage" in names
    # every entry carries a contract dict
    assert all("contract" in e and isinstance(e["contract"], dict) for e in audio_stage_catalog())


def test_all_agent_ready_stages_pass_static_conformance():
    from nemo_curator.stages.audio._catalog import get_agent_ready_stage_class, list_agent_ready_stages

    failures = []
    names = list_agent_ready_stages()
    for name in names:
        cls = get_agent_ready_stage_class(name)
        try:
            assert_contract_wellformed(cls)  # shape + roles + serialization, instance-free
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: {e}")
    assert not failures, "static conformance failures:\n" + "\n".join(failures)


def test_static_hints_are_optional_and_additive():
    # A stage with no AGENT_STATIC still yields a valid static contract.
    assert static_contract(_ToyDataclassStage).error_policy == "unknown"

    @dataclass
    class _Hinted(_ToyDataclassStage):
        AGENT_STATIC = StaticHints(error_policy="skip", gates=Gates(requires_gpu=True))

    contract = static_contract(_Hinted)
    assert contract.error_policy == "skip"
    assert contract.gates.requires_gpu is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
