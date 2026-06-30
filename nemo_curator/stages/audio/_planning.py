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

This is the safety net that turns per-stage contracts into a pipeline an agent
can trust: it catches "stage B reads a waveform nobody produced" *before* the
pipeline runs. It is advisory and read-only — it never executes a stage.
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

    @property
    def ok(self) -> bool:
        """True when there are no error-severity issues."""
        return not any(i.severity == "error" for i in self.issues)

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


def validate_pipeline(
    stages: list[Any],  # noqa: ANN401
    *,
    initial_roles: set[str] | None = None,
    available_gpus: float | None = None,
) -> PipelineReport:
    """Validate that an ordered list of configured stages composes.

    Args:
        stages: Configured stage instances in execution order.
        initial_roles: Semantic roles present in the input task. Defaults to
            ``{"audio_filepath"}`` (a manifest row). Pass an explicit set when the
            first stage is a source/reader or the input already carries waveforms.
        available_gpus: If given, stages whose contract declares ``requires_gpu``
            while this is ``<= 0`` raise a warning.

    Returns:
        A :class:`PipelineReport`. ``report.ok`` is True when no errors were found.
    """
    available: set[str] = set(initial_roles) if initial_roles is not None else set(_DEFAULT_INITIAL_ROLES)
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

        if available_gpus is not None and contract.gates.requires_gpu and available_gpus <= 0:
            issues.append(
                PipelineIssue(
                    index, name, "warning", "gpu_unavailable",
                    "declares requires_gpu but available_gpus <= 0",
                )
            )

        available |= produced_roles(contract)

    return PipelineReport(issues=issues, produced_roles=available)


def _roles_of(spec: Any, contract: StageContract) -> set[str]:  # noqa: ANN401
    keys = [*spec.data_keys, *spec.segment_data_keys]
    return {contract.key_roles.get(k, "unknown") for k in keys}
