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

"""Pipeline-level validation for agent-composed audio pipelines.

``validate_pipeline([stageA, stageB, ...])`` walks an ordered list of configured
stages and checks they actually compose — each stage's required inputs (matched
by semantic *role*, not key string) must be produced by an upstream stage or be
present in the initial task. It also surfaces resource-gate problems (GPU needed
but none available) and composite stages that must be decomposed first.

Two levels of confidence, deliberately separated:

* ``report.ok`` certifies a *role-level necessary condition* — every required
  input role is available. This is rename-tolerant by design and is the gate.
* ``report.keys_ok`` adds the stronger *literal-key-identity* check: each
  role-satisfied read's actual key *value* is produced upstream (or seeded). A
  ``True`` ``ok`` with ``False`` ``keys_ok`` means the roles line up but a
  producer key was renamed away from what the consumer reads — the pipeline
  would validate yet yield zero rows at runtime. It is surfaced as a WARNING
  (not an error) so that legitimate reads of source-manifest columns are not
  false-rejected.

This is advisory and read-only — it never executes a stage. ``ok`` is a
necessary, not sufficient, condition for a pipeline to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from nemo_curator.stages.audio._agent_registry import build_contract
from nemo_curator.stages.audio._conformance import produced_roles, reads_satisfied_by_role

if TYPE_CHECKING:
    from nemo_curator.stages.audio._agent_ready import StageContract

Severity = Literal["error", "warning"]

# Roles a typical audio task carries at the start (a manifest row with a file path).
_DEFAULT_INITIAL_ROLES: frozenset[str] = frozenset({"audio_filepath"})
# Key VALUES a typical audio task carries at the start (the conventional path key).
_DEFAULT_INITIAL_KEYS: frozenset[str] = frozenset({"audio_filepath"})


@dataclass(frozen=True)
class PipelineIssue:
    """A single problem found while validating a pipeline."""

    stage_index: int
    stage_name: str
    severity: Severity
    code: str
    message: str


@dataclass(frozen=True)
class PipelineReport:
    """Result of :func:`validate_pipeline`."""

    issues: list[PipelineIssue] = field(default_factory=list)
    produced_roles: set[str] = field(default_factory=set)  # roles available after the last stage
    produced_keys: set[str] = field(default_factory=set)  # key VALUES available after the last stage

    @property
    def ok(self) -> bool:
        """True when there are no error-severity issues (role-level composability).

        This is a *necessary* condition, not a guarantee the pipeline runs — see
        :attr:`keys_ok` for the stronger literal-key check.
        """
        return not any(i.severity == "error" for i in self.issues)

    @property
    def keys_ok(self) -> bool:
        """True when no ``dangling_key`` warnings — every role-satisfied read's
        actual key *value* is produced upstream or seeded. ``ok and keys_ok`` is
        the strong signal that the pipeline will actually flow data end-to-end.
        """
        return not any(i.code == "dangling_key" for i in self.issues)

    @property
    def errors(self) -> list[PipelineIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[PipelineIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        if not self.issues:
            return "pipeline OK (all reads satisfied by role)"
        lines = [f"{len(self.errors)} error(s), {len(self.warnings)} warning(s):"]
        for i in self.issues:
            lines.append(f"  [{i.severity}] stage {i.stage_index} {i.stage_name}: {i.message}")
        return "\n".join(lines)


def _required_roles(contract: StageContract) -> set[str]:
    keys = [*contract.reads.data_keys, *contract.reads.segment_data_keys]
    return {contract.key_roles.get(k, "unknown") for k in keys}


def _write_key_values(contract: StageContract) -> set[str]:
    """The literal key VALUES a stage writes (top-level + segment-level)."""
    return {*contract.writes.data_keys, *contract.writes.segment_data_keys}


def _dangling_read_keys(contract: StageContract, available_keys: set[str]) -> set[str]:
    """Primary-read key VALUES whose role is known but whose exact value was not
    produced upstream nor seeded — the renamed-producer dangle the role check misses.

    Scoped to primary ``reads`` (not ``reads_one_of`` alternatives) and to
    role-bearing keys (``unknown``/internal bookkeeping keys are excluded — a
    separate value-identity check for those is tracked in the backlog).
    """
    dangling: set[str] = set()
    for k in [*contract.reads.data_keys, *contract.reads.segment_data_keys]:
        role = contract.key_roles.get(k, "unknown")
        if role == "unknown":
            continue
        if k not in available_keys:
            dangling.add(k)
    return dangling


def validate_pipeline(  # noqa: C901 (complexity accepted: sequential per-stage validation checklist)
    stages: list[Any],
    *,
    initial_roles: set[str] | None = None,
    initial_keys: set[str] | None = None,
    available_gpus: float | None = None,
) -> PipelineReport:
    """Validate that an ordered list of configured stages composes.

    Args:
        stages: Configured stage instances in execution order.
        initial_roles: Semantic roles present in the input task. Defaults to
            ``{"audio_filepath"}`` (a manifest row). Pass an explicit set when the
            first stage is a source/reader or the input already carries waveforms.
        initial_keys: Literal key VALUES present in the input task (e.g. the
            columns of the source manifest: ``{"audio_filepath", "text"}``).
            Defaults to ``{"audio_filepath"}``. Seeding this lets the
            literal-key check (``keys_ok``) recognize reads satisfied by the
            input rather than by an upstream producer.
        available_gpus: If given, stages whose contract declares ``requires_gpu``
            while this is ``<= 0`` raise a warning.

    Returns:
        A :class:`PipelineReport`. ``report.ok`` is True when no errors were
        found (role-level); ``report.keys_ok`` additionally confirms literal-key
        identity (see the class docstring).
    """
    available: set[str] = set(initial_roles) if initial_roles is not None else set(_DEFAULT_INITIAL_ROLES)
    available_keys: set[str] = set(initial_keys) if initial_keys is not None else set(_DEFAULT_INITIAL_KEYS)
    tensor_resident = False  # an upstream stage left a non-serializable tensor in task.data
    past_composite = False  # a composite hides its true writes; downstream reads can't be judged
    issues: list[PipelineIssue] = []

    for index, stage in enumerate(stages):
        try:
            contract = build_contract(stage)
        except Exception as e:  # noqa: BLE001 - a stage that can't describe itself is an error
            issues.append(
                PipelineIssue(index, type(stage).__name__, "error", "contract_error", f"describe() failed: {e}")
            )
            continue
        name = contract.stage_id or type(stage).__name__

        if not contract.wrappable:
            issues.append(
                PipelineIssue(
                    index, name, "warning", "composite",
                    "composite stage — decompose before validating its data flow",
                )
            )
            # A composite hides its true I/O; don't reason about roles past it.
            past_composite = True
            continue

        if past_composite:
            # The composite's (hidden) writes may satisfy anything downstream:
            # unknown-availability must not produce false HARD errors on
            # runnable pipelines. Downgrade to an advisory warning.
            if not reads_satisfied_by_role(contract, available):
                issues.append(
                    PipelineIssue(
                        index, name, "warning", "unsatisfied_reads_after_composite",
                        f"requires role(s) {sorted(_required_roles(contract) - (available | {'unknown'}))} "
                        f"not visibly produced — but an upstream composite hides its writes; "
                        f"decompose it to validate this read",
                    )
                )
            available |= produced_roles(contract)
            available_keys |= _write_key_values(contract)
            continue

        if not reads_satisfied_by_role(contract, available):
            missing = _required_roles(contract) - (available | {"unknown"})
            alt = (
                f" (or one of: {[sorted(_roles_of(o, contract)) for o in contract.reads_one_of]})"
                if contract.reads_one_of
                else ""
            )
            issues.append(
                PipelineIssue(
                    index, name, "error", "unsatisfied_reads",
                    f"requires role(s) {sorted(missing) or sorted(_required_roles(contract))} "
                    f"not produced upstream; available so far: {sorted(available)}{alt}",
                )
            )
        else:
            # Role-satisfied: check literal-key identity. A read whose role is
            # available but whose exact key VALUE was not produced/seeded means a
            # producer key was renamed away from what this stage reads -> the
            # pipeline validates by role but yields no rows at runtime. WARNING
            # (not error) so source-manifest reads aren't false-rejected.
            dangling = _dangling_read_keys(contract, available_keys)
            if dangling:
                issues.append(
                    PipelineIssue(
                        index, name, "warning", "dangling_key",
                        f"reads key(s) {sorted(dangling)} satisfied by role but not produced "
                        f"upstream under that key value nor seeded (renamed producer key?); "
                        f"available keys: {sorted(available_keys)}",
                    )
                )

        if available_gpus is not None and contract.gates.requires_gpu and available_gpus <= 0:
            issues.append(
                PipelineIssue(
                    index, name, "warning", "gpu_unavailable",
                    "declares requires_gpu but available_gpus <= 0",
                )
            )

        # Serializability: a resident tensor (e.g. a waveform) reaching a
        # serialize-as-is sink (raw json.dumps) crashes at runtime. Warn before
        # the sink; a sanitizing stage (AudioToDocumentStage) clears the flag.
        if contract.gates.requires_serializable_input and tensor_resident:
            issues.append(
                PipelineIssue(
                    index, name, "warning", "tensor_into_sink",
                    "a resident tensor/audio blob from an upstream stage reaches this "
                    "serialize-as-JSON sink; route through AudioToDocumentStage first "
                    "(or drop the tensor) or it will fail at json.dumps",
                )
            )

        available |= produced_roles(contract)
        available_keys |= _write_key_values(contract)
        if "tensor" in contract.writes.produces:
            tensor_resident = True
        if contract.gates.sanitizes_output:
            tensor_resident = False

    return PipelineReport(issues=issues, produced_roles=available, produced_keys=available_keys)


def _roles_of(spec: Any, contract: StageContract) -> set[str]:  # noqa: ANN401
    keys = [*spec.data_keys, *spec.segment_data_keys]
    return {contract.key_roles.get(k, "unknown") for k in keys}
