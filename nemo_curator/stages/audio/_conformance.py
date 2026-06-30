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

"""Conformance harness for agent-ready audio stages.

``assert_agent_ready(stage, fixture_factory, ...)`` is the gate each stage's CL
must pass. It runs a set of *static* checks (contract well-formedness, semantic
roles, JSON serialization, by-role read satisfiability) that need no execution,
plus optional *dynamic* checks (run ``process``/``process_batch`` on a fixture
and verify declared writes appear, no undeclared top-level keys leak, cardinality
matches runtime, and ``accepts``/``produces`` hold).

The static checks alone catch the contract↔reality drift the prototype lacked
and can sweep every stage with no fixtures (see ``assert_contract_wellformed``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, get_args

from nemo_curator.stages.audio._agent_ready import (
    AudioForm,
    Cardinality,
    ProducedForm,
    Role,
    StageContract,
    to_json_schema,
)
from nemo_curator.stages.audio._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio._roles import field_has_declared_role

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_VALID_CARDINALITY: frozenset[str] = frozenset(get_args(Cardinality))
_VALID_ROLES: frozenset[str] = frozenset(get_args(Role))
_VALID_ACCEPTS: frozenset[str] = frozenset(get_args(AudioForm))
_VALID_PRODUCES: frozenset[str] = frozenset(get_args(ProducedForm))


# --------------------------------------------------------------------------- #
# By-role matching (the planner primitive)
# --------------------------------------------------------------------------- #
def _spec_roles(contract: StageContract, spec_keys: Iterable[str]) -> set[str]:
    return {contract.key_roles.get(k, "unknown") for k in spec_keys}


def produced_roles(producer: StageContract) -> set[str]:
    """Roles a producer emits (from its ``writes`` keys); excludes ``unknown``."""
    keys = [*producer.writes.data_keys, *producer.writes.segment_data_keys]
    return _spec_roles(producer, keys) - {"unknown"}


def reads_satisfied_by_role(consumer: StageContract, available_roles: set[str]) -> bool:
    """Can ``consumer`` run given upstream-produced ``available_roles``?

    Matches by semantic role, not key-string equality, so a producer that
    renamed its output key still satisfies a consumer that needs that role.
    ``unknown`` is permissive (never blocks).
    """
    avail = set(available_roles) | {"unknown"}
    reads_keys = [*consumer.reads.data_keys, *consumer.reads.segment_data_keys]
    if reads_keys and not _spec_roles(consumer, reads_keys).issubset(avail):
        return False
    if consumer.reads_one_of:
        return any(
            _spec_roles(consumer, [*opt.data_keys, *opt.segment_data_keys]).issubset(avail)
            for opt in consumer.reads_one_of
        )
    return True


# --------------------------------------------------------------------------- #
# Static checks (no execution)
# --------------------------------------------------------------------------- #
def _check_shape(c: StageContract, name: str) -> None:
    assert c.cardinality in _VALID_CARDINALITY, f"{name}: invalid cardinality {c.cardinality!r}"
    for opt in c.cardinality_options:
        # cardinality_options are short flag names (e.g. "fan_out","nested") OR full cardinalities
        assert isinstance(opt, str) and opt, f"{name}: bad cardinality option {opt!r}"
    if c.iteration_key is not None:
        assert c.cardinality in {"1:1 nested-list", "1:N fan-out", "N:1"}, (
            f"{name}: iteration_key set but cardinality is {c.cardinality!r}"
        )
    for spec, label in [(c.reads, "reads"), (c.writes, "writes"), *[(s, "reads_one_of") for s in c.reads_one_of]]:
        for a in spec.accepts:
            assert a in _VALID_ACCEPTS, f"{name}: {label}.accepts has invalid form {a!r}"
        for p in spec.produces:
            assert p in _VALID_PRODUCES, f"{name}: {label}.produces has invalid form {p!r}"
    # no duplicate keys within a single spec list
    for spec, label in [(c.reads, "reads"), (c.writes, "writes")]:
        assert len(spec.data_keys) == len(set(spec.data_keys)), f"{name}: duplicate {label}.data_keys"


def _check_roles(stage_or_cls: Any, c: StageContract, name: str) -> None:  # noqa: ANN401
    for value, role in c.key_roles.items():
        assert role in _VALID_ROLES, f"{name}: key_roles[{value!r}] has invalid role {role!r}"
    for p in c.params:
        if p.role is not None:
            assert p.role in _VALID_ROLES, f"{name}: param {p.name!r} has invalid role {p.role!r}"
        # check #8: a *_key constructor field must have a KEY_ROLES entry or be
        # explicitly allowlisted as internal (catches a forgotten role mapping).
        if p.name.endswith("_key"):
            assert field_has_declared_role(p.name), (
                f"{name}: param {p.name!r} ends in '_key' but has no role in KEY_ROLES "
                f"and is not allowlisted in INTERNAL_KEY_FIELDS (add one in _roles.py)"
            )


def _check_serialization(c: StageContract, name: str) -> None:
    try:
        json.dumps(c.to_dict())
    except (TypeError, ValueError) as e:  # pragma: no cover - defensive
        msg = f"{name}: contract.to_dict() is not JSON-serializable: {e}"
        raise AssertionError(msg) from e
    schema = to_json_schema(c.params)
    assert schema.get("type") == "object" and "properties" in schema, f"{name}: bad json schema"


def assert_contract_wellformed(stage_or_cls: Any) -> StageContract:  # noqa: ANN401
    """Static-only conformance: shape, roles, serialization. No execution.

    Accepts an instance (dynamic contract via ``build_contract``) or a class
    (instance-free ``static_contract``). Returns the contract so callers can
    reuse it. Safe to run across every stage with no fixtures.
    """
    if isinstance(stage_or_cls, type):
        c = static_contract(stage_or_cls)
        name = stage_or_cls.__name__
    else:
        c = build_contract(stage_or_cls)
        name = type(stage_or_cls).__name__
    _check_shape(c, name)
    _check_roles(stage_or_cls, c, name)
    _check_serialization(c, name)
    return c


# --------------------------------------------------------------------------- #
# Dynamic checks (execute the stage on a fixture)
# --------------------------------------------------------------------------- #
def _supports_batch(stage: Any) -> bool:  # noqa: ANN401
    fn = getattr(stage, "supports_batch_processing", None)
    try:
        return bool(fn()) if callable(fn) else False
    except Exception:  # noqa: BLE001
        return False


def _normalize_results(out: Any) -> list[Any]:  # noqa: ANN401
    if out is None:
        return []
    if isinstance(out, list):
        flat: list[Any] = []
        for item in out:
            if item is None:
                continue
            if isinstance(item, list):
                flat.extend(x for x in item if x is not None)
            else:
                flat.append(item)
        return flat
    return [out]


def _data_of(task: Any) -> dict[str, Any]:  # noqa: ANN401
    data = getattr(task, "data", None)
    return data if isinstance(data, dict) else {}


def assert_agent_ready(  # noqa: PLR0913
    stage: Any,  # noqa: ANN401
    fixture_factory: Callable[[], Any] | None = None,
    *,
    expected_cardinality: str | None = None,
    available_keys: Iterable[str] | None = None,
    segments_key: str | None = None,
    ignore_new_keys: Iterable[str] = (),
    run: bool = True,
    setup: bool = False,
) -> StageContract:
    """Assert a stage is agent-ready. Returns its (dynamic) contract.

    Always runs the static checks. When ``run`` and a ``fixture_factory`` are
    given, also executes the stage and verifies declared writes appear, no
    undeclared top-level keys leak, and cardinality matches the runtime shape.

    Args:
        stage: A constructed stage instance.
        fixture_factory: Returns a fresh input task (or batch) each call.
        expected_cardinality: If given, assert the contract declares it.
        available_keys: Upstream-available key values; asserts reads are
            satisfiable by role.
        segments_key: Resolved segments key, for checking segment-level writes.
        ignore_new_keys: Extra top-level keys allowed in output (framework
            bookkeeping) beyond declared writes.
        run: Execute the stage (default True).
        setup: Call ``stage.setup()`` before processing (default False; most
            lightweight stages need no setup, heavy ones are pre-set-up/stubbed
            by the caller).
    """
    c = build_contract(stage)
    name = type(stage).__name__
    _check_shape(c, name)
    _check_roles(stage, c, name)
    _check_serialization(c, name)

    if expected_cardinality is not None:
        assert c.cardinality == expected_cardinality, (
            f"{name}: cardinality {c.cardinality!r} != expected {expected_cardinality!r}"
        )
    if available_keys is not None:
        avail_roles = {c.key_roles.get(k, "unknown") for k in available_keys}
        # also resolve via literal table for keys not in this stage's key_roles
        from nemo_curator.stages.audio._roles import role_for_value

        avail_roles |= {role_for_value(k) for k in available_keys}
        assert reads_satisfied_by_role(c, avail_roles), (
            f"{name}: reads {c.reads.data_keys}/{[s.data_keys for s in c.reads_one_of]} "
            f"not satisfied by available roles {avail_roles}"
        )

    if not run or fixture_factory is None:
        return c

    if setup and hasattr(stage, "setup"):
        stage.setup()

    task = fixture_factory()
    batch_input = isinstance(task, list)
    input_keys = set(_data_of(task[0] if batch_input else task))

    if c.batch_only or _supports_batch(stage):
        out = stage.process_batch(task if batch_input else [task])
    else:
        out = stage.process(task)
    results = _normalize_results(out)

    # (6) cardinality vs runtime shape
    if c.cardinality == "1:N fan-out":
        assert isinstance(out, list), f"{name}: fan-out must return a list"
    elif c.cardinality in {"1:1", "1:1 nested-list"} and results:
        assert len(results) == 1, f"{name}: {c.cardinality} produced {len(results)} tasks"
    elif c.cardinality == "filter":
        assert len(results) <= (len(task) if batch_input else 1), f"{name}: filter increased task count"

    # (3) declared writes appear; (4) no undeclared top-level keys (non-fanout)
    if c.cardinality in {"1:1", "1:1 nested-list", "filter"} and results:
        out_data = _data_of(results[0])
        for key in c.writes.data_keys:
            assert key in out_data, f"{name}: declared write {key!r} missing from task.data"
        declared = set(c.writes.data_keys) | set(ignore_new_keys) | input_keys
        undeclared = set(out_data) - declared
        assert not undeclared, f"{name}: undeclared new top-level keys {sorted(undeclared)} (add to writes.data_keys)"
        # segment-level writes
        seg_key = segments_key or c.iteration_key
        if c.writes.segment_data_keys and seg_key and isinstance(out_data.get(seg_key), list) and out_data[seg_key]:
            seg0 = out_data[seg_key][0]
            if isinstance(seg0, dict):
                for key in c.writes.segment_data_keys:
                    assert key in seg0, f"{name}: declared segment write {key!r} missing from segment dict"
    return c
