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
from nemo_curator.stages.audio._roles import role_for_value

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
            return (
                "pipeline mechanically composable (all reads satisfied by role); "
                "this does not certify intent or field meaning"
            )
        lines = [f"{len(self.errors)} error(s), {len(self.warnings)} warning(s):"]
        for i in self.issues:
            lines.append(f"  [{i.severity}] stage {i.stage_index} {i.stage_name}: {i.message}")
        return "\n".join(lines)


def _required_roles(contract: StageContract) -> set[str]:
    keys = [*contract.reads.data_keys, *contract.reads.segment_data_keys]
    return {contract.key_roles.get(k, "unknown") for k in keys}


def _requirement_str(contract: StageContract, available: set[str]) -> str:
    """Human-readable "what this stage needs" for an unsatisfied-reads message.

    Renders top-level ``reads`` (all required) and ``reads_one_of`` (any one), so a
    stage whose reads live entirely in ``reads_one_of`` (e.g. a residency-derived
    contract) no longer renders a misleading empty ``role(s) []``.
    """
    missing = _required_roles(contract) - (available | {"unknown"})
    reqs: list[str] = []
    if missing:
        reqs.append(f"role(s) {sorted(missing)}")
    if contract.reads_one_of:
        reqs.append(f"one of {[sorted(_roles_of(o, contract)) for o in contract.reads_one_of]}")
    return "; ".join(reqs) or f"role(s) {sorted(_required_roles(contract))}"


def _write_key_values(contract: StageContract) -> set[str]:
    """The literal key VALUES a stage writes (top-level + segment-level)."""
    return {*contract.writes.data_keys, *contract.writes.segment_data_keys}


def _key_family(key: str) -> str:
    """The trailing token of a key name -- ``diar_segments`` and ``segments`` share ``segments``.

    A crude but load-bearing notion of "these two keys hold the same KIND of thing". Producers
    qualify a shared noun with a prefix (``diar_``, ``vad_``, ``pred_``), so the bare noun and
    its qualified siblings are exactly the set a consumer might have meant.
    """
    return key.rsplit("_", 1)[-1]


def _ambiguous_default_reads(
    stage: Any,  # noqa: ANN401 - any built stage
    contract: StageContract,
    available_keys: set[str],
    key_producer: dict[str, str],
) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """``(key, attribute, rivals)`` for read keys left at a default while a sibling key exists.

    The failure this catches is silence. ``MergeAlignmentDiarizationStage`` documents itself as
    merging into DIARIZATION segments, yet its ``segments_key`` defaults to ``"segments"`` --
    the key VAD writes. In a VAD+diarization pipeline both keys exist, so the read is satisfied
    and every other check passes: transcripts get merged into the wrong segments and the output
    is plausible, complete, and wrong.

    Deliberately narrow, because a warning nobody trusts is worse than none. It fires only when
    the key is still at its CLASS DEFAULT (an explicit setting is a decision, not an accident),
    the key IS available (an unavailable one is already reported as dangling), and some other
    available key of the same family was written by a DIFFERENT upstream stage -- so a real
    choice existed and was made by a default rather than by anyone.
    """
    fields = getattr(type(stage), "__dataclass_fields__", {})
    reads = {*contract.reads.data_keys, *contract.reads.segment_data_keys}
    found: list[tuple[str, str, list[tuple[str, str]]]] = []
    for attr, spec in fields.items():
        value = getattr(stage, attr, None)
        if not (isinstance(value, str) and value in reads and value in available_keys and value == spec.default):
            continue
        rivals = sorted(
            (k, key_producer[k])
            for k in available_keys
            if k != value
            # If the consumer reads both siblings (for example reference
            # ``text`` and ASR ``pred_text`` for WER), they are independent
            # operands rather than competing choices.
            and k not in reads
            and k in key_producer
            and _key_family(k) == _key_family(value)
            and key_producer[k] != key_producer.get(value)
        )
        if rivals:
            found.append((value, attr, rivals))
    return found


def _gate_issues(
    index: int,
    name: str,
    contract: StageContract,
    available_gpus: float | None,
    *,
    tensor_resident: bool,
) -> list[PipelineIssue]:
    """Environment/serialization gate problems for one stage.

    These reason about GPUs and serialization rather than roles, so they apply to every concrete
    stage even downstream of a composite that hides its writes.
    """
    out = []
    if available_gpus is not None and contract.gates.requires_gpu and available_gpus <= 0:
        out.append(
            PipelineIssue(index, name, "warning", "gpu_unavailable", "declares requires_gpu but available_gpus <= 0")
        )
    # A resident tensor (e.g. a waveform) reaching a serialize-as-is sink (raw json.dumps)
    # crashes at runtime. A sanitizing stage upstream clears the flag before we get here.
    if contract.gates.requires_serializable_input and tensor_resident:
        out.append(
            PipelineIssue(
                index, name, "error", "tensor_into_sink",
                "a resident tensor/audio blob from an upstream stage reaches this "
                "serialize-as-JSON sink; it WILL fail at json.dumps — drop the tensor "
                "upstream (e.g. keep_segment_waveform_in_task=False) or route through "
                "a sanitizing stage before the sink",
            )
        )
    return out


def _ambiguity_issues(  # noqa: PLR0913 - an ambiguity message must name the stage, key and producers
    index: int,
    name: str,
    stage: Any,  # noqa: ANN401 - any built stage
    contract: StageContract,
    available_keys: set[str],
    key_producer: dict[str, str],
) -> list[PipelineIssue]:
    """``ambiguous_default_key`` warnings for this stage, naming who wrote each candidate."""
    out = []
    for key, attr, rivals in _ambiguous_default_reads(stage, contract, available_keys, key_producer):
        others = ", ".join(f"{k!r} from {p}" for k, p in rivals)
        out.append(
            PipelineIssue(
                index, name, "warning", "ambiguous_default_key",
                f"reads {key!r} (the default for {attr}), but upstream also produced {others}. "
                f"The default silently picks {key!r} "
                f"({key_producer.get(key, 'the source manifest')}); if you meant the other, "
                f"set {attr} explicitly.",
            )
        )
    return out


def _dangling_read_keys(contract: StageContract, available_keys: set[str]) -> set[str]:
    """Read key VALUES whose role is known but whose exact value was not
    produced upstream nor seeded — the renamed-producer dangle the role check misses.

    Covers primary ``reads`` plus a ``reads_one_of`` that offers a *single*
    alternative: one option is not a choice, so its keys are as mandatory as a
    primary read (this is how a residency-derived contract expresses
    ``input_residency="file"``). A genuine multi-way ``reads_one_of`` is skipped —
    the stage may legitimately take the other branch. Role-bearing keys only
    (``unknown``/internal bookkeeping keys are excluded — a separate
    value-identity check for those is tracked in the backlog).
    """
    reads = [*contract.reads.data_keys, *contract.reads.segment_data_keys]
    if len(contract.reads_one_of) == 1:
        only = contract.reads_one_of[0]
        reads += [*only.data_keys, *only.segment_data_keys]
    dangling: set[str] = set()
    for k in reads:
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
    removed_roles: set[str] = set()  # roles whose carrier key an upstream stage deleted (removes_keys)
    key_producer: dict[str, str] = {}  # key value -> the stage that wrote it (for ambiguity messages)
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

        # Reads. A composite upstream hides its writes, so an unsatisfied read is
        # downgraded to an advisory (never a false HARD error); otherwise it is a
        # hard error. The literal-key (dangling) check only runs when upstream keys
        # are trustworthy (no composite hiding them).
        if not reads_satisfied_by_role(contract, available):
            if past_composite:
                issues.append(
                    PipelineIssue(
                        index, name, "warning", "unsatisfied_reads_after_composite",
                        f"requires {_requirement_str(contract, available)} "
                        f"not visibly produced — but an upstream composite hides its writes; "
                        f"decompose it to validate this read",
                    )
                )
            else:
                needed = _required_roles(contract) | {r for o in contract.reads_one_of for r in _roles_of(o, contract)}
                removed_hit = (needed & removed_roles) - available
                if removed_hit:
                    issues.append(
                        PipelineIssue(
                            index, name, "error", "key_removed_upstream",
                            f"reads role(s) {sorted(removed_hit)} that an upstream stage removed "
                            f"(removes_keys) and no stage re-produced; available so far: {sorted(available)}",
                        )
                    )
                else:
                    issues.append(
                        PipelineIssue(
                            index, name, "error", "unsatisfied_reads",
                            f"requires {_requirement_str(contract, available)} "
                            f"not produced upstream; available so far: {sorted(available)}",
                        )
                    )
        elif not past_composite:
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
            issues.extend(_ambiguity_issues(index, name, stage, contract, available_keys, key_producer))

        # The checks below reason about serialization / GPU / key-flow, NOT the
        # composite's hidden roles, so they run for every concrete stage even after
        # a composite (fixes the tensor_into_sink blind spot: ManifestReader is a
        # composite, so a start-with-reader pipeline used to skip these entirely).
        issues.extend(_gate_issues(index, name, contract, available_gpus, tensor_resident=tensor_resident))

        produced = produced_roles(contract)
        available |= produced
        removed_roles -= produced  # a re-produced role is no longer "removed"
        written = _write_key_values(contract)
        # Most recent writer wins -- that is who a downstream reader would actually get.
        key_producer.update(dict.fromkeys(written, name))
        available_keys |= written
        for rk in contract.removes_keys:
            available_keys.discard(rk)
            role = role_for_value(rk)
            if role != "unknown" and role not in produced and not any(role_for_value(k) == role for k in available_keys):
                available.discard(role)
                removed_roles.add(role)
        if "tensor" in contract.writes.produces:
            tensor_resident = True
        if contract.gates.sanitizes_output:
            tensor_resident = False

    return PipelineReport(issues=issues, produced_roles=available, produced_keys=available_keys)


def _roles_of(spec: Any, contract: StageContract) -> set[str]:  # noqa: ANN401
    keys = [*spec.data_keys, *spec.segment_data_keys]
    return {contract.key_roles.get(k, "unknown") for k in keys}
